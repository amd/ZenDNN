# *******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# *******************************************************************************

"""Tiered feature engineering for the DT pipeline.

Four tiers of derived features, each cumulative:
  none     — raw features only (whatever ``detect_columns`` put in ``feature_cols``)
  cheap    — pairwise add / abs-diff / compare, plus full sum when >= 3 bases
  moderate — pairwise integer mul / floored div
  expensive — per-feature log2, sqrt of product of all bases

Derived features are built from the **numeric** columns in ``config.feature_cols``
at the time ``apply_feature_engineering`` runs (after ``detect_columns``), so
matmul workloads keep M, K, N-style interactions while embbag-style CSVs use
BS, Heads, Seq, Dim, etc.

Usage:
    apply_feature_engineering(df, config)          # adds columns + updates config
    get_derived_feature_cpp(config)                # {name: cpp_expr}
    get_derived_feature_py(config)                 # {name: py_expr}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import PipelineConfig


TIER_ORDER = ["cheap", "moderate", "expensive"]


def _tiers_for_level(level: str) -> set[str]:
    """Return the set of tier names active at a given level."""
    if level == "none":
        return set()
    if level not in TIER_ORDER:
        print(f"WARNING: Unknown feature engineering level '{level}'. "
              f"Valid options: 'none', {TIER_ORDER}. Defaulting to 'none'.")
        return set()
    idx = TIER_ORDER.index(level)
    return set(TIER_ORDER[: idx + 1])


def _numeric_base_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Keep feature names that exist in df and are numeric (int/float/bool).

    Note: exported code (C++/Excel) casts thresholds to int, so float-typed
    features may cause boundary mismatches. A warning is printed if any
    float columns are included.
    """
    out: list[str] = []
    float_cols: list[str] = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
            if pd.api.types.is_float_dtype(df[c]):
                float_cols.append(c)
    if float_cols:
        print(f"WARNING: Float-typed feature columns {float_cols} detected. "
              f"Exported C++/Excel code casts thresholds to int, which may "
              f"shift decision boundaries for non-integer values.")
    return out


def _cpp_abs_diff(a: str, b: str) -> str:
    return f"(({a} >= {b}) ? ({a} - {b}) : ({b} - {a}))"


def _excel_abs_diff(a: str, b: str) -> str:
    return f"ABS({a}-{b})"


_MAX_FE_BASE_COLS = 10


def _build_dynamic_defs(
    cols: list[str],
    tiers: set[str],
) -> list[tuple[str, str, object, str, str, str]]:
    """Return rows: (name, tier, pandas_fn, cpp_expr, py_expr, excel_expr)."""
    defs: list[tuple[str, str, object, str, str, str]] = []
    n = len(cols)
    if n > _MAX_FE_BASE_COLS:
        print(f"WARNING: {n} base columns → O(n²) derived features. "
              f"Set config.feature_engineering_cols to a smaller subset "
              f"(max recommended: {_MAX_FE_BASE_COLS}).")

    def py_feat(c: str) -> str:
        return f"features[{c!r}]"

    # ── cheap ───────────────────────────────────────────────────────────
    if "cheap" in tiers:
        for i in range(n):
            for j in range(i + 1, n):
                a, b = cols[i], cols[j]
                name = f"{a}_plus_{b}"

                def _sum_pair(df: pd.DataFrame, ca=a, cb=b) -> pd.Series:
                    return df[ca] + df[cb]

                defs.append(
                    (name, "cheap", _sum_pair, f"{a} + {b}",
                     f"{py_feat(a)} + {py_feat(b)}", f"{a}+{b}"),
                )

                name_abs = f"abs_{a}_minus_{b}"

                def _abs_pair(df: pd.DataFrame, ca=a, cb=b) -> pd.Series:
                    return (df[ca] - df[cb]).abs()

                defs.append(
                    (name_abs, "cheap", _abs_pair, _cpp_abs_diff(a, b),
                     f"abs({py_feat(a)} - {py_feat(b)})", _excel_abs_diff(a, b)),
                )

                name_lt = f"{a}_lt_{b}"

                def _lt_pair(df: pd.DataFrame, ca=a, cb=b) -> pd.Series:
                    return (df[ca] < df[cb]).astype(int)

                defs.append(
                    (name_lt, "cheap", _lt_pair, f"({a} < {b})",
                     f"int({py_feat(a)} < {py_feat(b)})", f"IF({a}<{b},1,0)"),
                )

                name_eq = f"{a}_eq_{b}"

                def _eq_pair(df: pd.DataFrame, ca=a, cb=b) -> pd.Series:
                    return (df[ca] == df[cb]).astype(int)

                defs.append(
                    (name_eq, "cheap", _eq_pair, f"({a} == {b})",
                     f"int({py_feat(a)} == {py_feat(b)})", f"IF({a}={b},1,0)"),
                )

        if n >= 3:
            name = "_plus_".join(cols)

            def _sum_all(df: pd.DataFrame, cc: tuple[str, ...] = tuple(cols)) -> pd.Series:
                s = df[cc[0]]
                for c in cc[1:]:
                    s = s + df[c]
                return s

            cpp_sum = " + ".join(cols)
            py_sum = " + ".join(py_feat(c) for c in cols)
            xl_sum = "+".join(cols)
            defs.append((name, "cheap", _sum_all, cpp_sum, py_sum, xl_sum))

    # ── moderate ────────────────────────────────────────────────────────
    if "moderate" in tiers:
        for i in range(n):
            for j in range(i + 1, n):
                a, b = cols[i], cols[j]
                name = f"{a}_mul_{b}"

                def _mul_pair(df: pd.DataFrame, ca=a, cb=b) -> pd.Series:
                    return df[ca] * df[cb]

                defs.append(
                    (name, "moderate", _mul_pair, f"{a} * {b}",
                     f"{py_feat(a)} * {py_feat(b)}", f"{a}*{b}"),
                )

                name_div = f"{a}_div_{b}"

                def _div_pair(df: pd.DataFrame, ca=a, cb=b) -> pd.Series:
                    return np.where(df[cb] != 0, df[ca] // df[cb], 0)

                defs.append(
                    (name_div, "moderate", _div_pair, f"({b} != 0 ? {a} / {b} : 0)",
                     f"({py_feat(a)} // {py_feat(b)} if {py_feat(b)} != 0 else 0)",
                     f"IF({b}<>0,INT({a}/{b}),0)"),
                )

    # ── expensive ───────────────────────────────────────────────────────
    if "expensive" in tiers:
        for c in cols:
            name = f"log2_{c}"

            def _log2_one(df: pd.DataFrame, cc=c) -> pd.Series:
                return np.log2(df[cc].clip(lower=1)).astype(int)

            defs.append(
                (name, "expensive", _log2_one,
                 f"static_cast<int>(log2({c} > 1 ? {c} : 1))",
                 f"int(math.log2(max({py_feat(c)}, 1)))",
                 f"INT(LOG(IF({c}>1,{c},1),2))"),
            )

        if n >= 1:
            prod_cpp = " * ".join(
                [f"static_cast<float>({cols[0]})"] + cols[1:]
            ) if n > 1 else f"static_cast<float>({cols[0]})"
            name = "sqrt_" + "_".join(cols)

            def _sqrt_prod(df: pd.DataFrame, cc: tuple[str, ...] = tuple(cols)) -> pd.Series:
                p = df[cc[0]].astype(np.float32)
                for c in cc[1:]:
                    p = p * df[c].astype(np.float32)
                return np.sqrt(p).astype(int)

            py_terms = " * ".join(f"float({py_feat(c)})" for c in cols)
            defs.append(
                (name, "expensive", _sqrt_prod,
                 f"static_cast<int>(sqrtf({prod_cpp}))",
                 f"int(math.sqrt({py_terms}))",
                 f"INT(SQRT({'*'.join(cols)}))"),
            )

    return defs


def _clear_derived_caches(config: PipelineConfig) -> None:
    config.derived_feature_cpp = {}
    config.derived_feature_py = {}
    config.derived_feature_excel = {}


def apply_feature_engineering(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Add derived feature columns to the DataFrame and update config.

    Call after detect_columns() and before split_data().

    Both ``df`` and ``config`` are modified in-place:
    - ``df`` gets new columns for each derived feature.
    - ``config.feature_cols`` is extended with the derived feature names.
    - ``config.derived_feature_{cpp,py,excel}`` are filled for codegen.

    Args:
        df: Training DataFrame (modified in-place).
        config: PipelineConfig (feature_cols updated in-place).

    Returns:
        The modified DataFrame (same object, returned for convenience).
    """
    _clear_derived_caches(config)

    raw_set = set(config.all_feature_cols) if config.all_feature_cols else set()
    if raw_set:
        base_cols = [c for c in config.feature_cols if c in raw_set]
    else:
        base_cols = list(config.feature_cols)
    config.feature_cols = list(base_cols)

    if config.feature_engineering == "none":
        print("Feature engineering: none (raw features only)")
        return df

    base = list(base_cols)
    if config.feature_engineering_cols is not None:
        allowed = set(config.feature_engineering_cols)
        extra = allowed - set(base)
        if extra:
            print(f"WARNING: feature_engineering_cols contains names not in feature_cols {extra}; "
                  f"ignoring those.")
        base = [c for c in config.feature_engineering_cols if c in base]
        if not base:
            print("WARNING: feature_engineering_cols does not overlap feature_cols. Skipping FE.")
            return df
    numeric = _numeric_base_columns(df, base)
    if len(numeric) < 1:
        print("WARNING: Feature engineering needs at least one numeric column in "
              f"feature_cols; got {base}. Skipping.")
        return df

    tiers = _tiers_for_level(config.feature_engineering)
    defs = _build_dynamic_defs(numeric, tiers)

    added = []
    skipped_collisions: list[str] = []
    base_set = set(base_cols)
    cpp_map: dict[str, str] = {}
    py_map: dict[str, str] = {}
    xl_map: dict[str, str] = {}

    for name, _tier, compute_fn, cpp_expr, py_expr, xl_expr in defs:
        if name in base_set:
            skipped_collisions.append(name)
            continue
        df[name] = compute_fn(df)  # type: ignore[operator]
        if name not in config.feature_cols:
            config.feature_cols.append(name)
        added.append(name)
        cpp_map[name] = cpp_expr
        py_map[name] = py_expr
        xl_map[name] = xl_expr

    if skipped_collisions:
        print(f"WARNING: Skipped {len(skipped_collisions)} derived feature(s) "
              f"that collide with base columns: {skipped_collisions[:5]}")

    config.derived_feature_cpp = cpp_map
    config.derived_feature_py = py_map
    config.derived_feature_excel = xl_map

    print(f"Feature engineering ({config.feature_engineering}) on {numeric}: "
          f"added {len(added)} derived features")
    return df


def get_derived_feature_cpp(config: PipelineConfig) -> dict[str, str]:
    """Return {name: cpp_expression} for derived features from the last apply."""
    return dict(config.derived_feature_cpp)


def get_derived_feature_py(config: PipelineConfig) -> dict[str, str]:
    """Return {name: python_expression} for derived features from the last apply."""
    return dict(config.derived_feature_py)


def get_derived_feature_excel(config: PipelineConfig) -> dict[str, str]:
    """Return {name: excel_expression} with raw feature names (cell-subst later)."""
    return dict(config.derived_feature_excel)


def needs_cmath(used_feature_names: dict[str, str]) -> bool:
    """Check if any of the used features require <cmath> in C++."""
    for cpp in used_feature_names.values():
        if "log2" in cpp or "sqrt" in cpp.lower():
            return True
    return False


def needs_math_import(used_feature_names: dict[str, str]) -> bool:
    """Check if any of the used features require 'import math' in Python."""
    for py in used_feature_names.values():
        if "math." in py:
            return True
    return False
