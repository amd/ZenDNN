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
  none     — raw M, K, N only
  cheap    — add, sub, compare  (~1-3 CPU cycles)
  moderate — integer mul, div   (~3-30 cycles)
  expensive — log, sqrt         (~50-100+ cycles)

Usage:
    apply_feature_engineering(df, config)          # adds columns + updates config
    get_derived_feature_cpp(config)                # {name: cpp_expr}
    get_derived_feature_py(config)                 # {name: py_expr}
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from .config import PipelineConfig


TIER_ORDER = ["cheap", "moderate", "expensive"]
_M_PATTERN = re.compile(r'\bM\b')

# Each entry: (name, tier, pandas_fn, cpp_expr, py_expr)
# pandas_fn: callable(df) -> Series
# cpp_expr:  C++ expression using raw variable names (M, K, N)
# py_expr:   Python expression using features['M'] etc.

_FEATURE_DEFS = [
    # ── Cheap tier: add, sub, compare ──────────────────────────────────
    ("M_plus_K", "cheap",
     lambda df: df["M"] + df["K"],
     "M + K",
     "features['M'] + features['K']"),

    ("M_plus_N", "cheap",
     lambda df: df["M"] + df["N"],
     "M + N",
     "features['M'] + features['N']"),

    ("K_plus_N", "cheap",
     lambda df: df["K"] + df["N"],
     "K + N",
     "features['K'] + features['N']"),

    ("M_plus_K_plus_N", "cheap",
     lambda df: df["M"] + df["K"] + df["N"],
     "M + K + N",
     "features['M'] + features['K'] + features['N']"),

    ("abs_M_minus_K", "cheap",
     lambda df: (df["M"] - df["K"]).abs(),
     "((M >= K) ? (M - K) : (K - M))",
     "abs(features['M'] - features['K'])"),

    ("abs_M_minus_N", "cheap",
     lambda df: (df["M"] - df["N"]).abs(),
     "((M >= N) ? (M - N) : (N - M))",
     "abs(features['M'] - features['N'])"),

    ("abs_K_minus_N", "cheap",
     lambda df: (df["K"] - df["N"]).abs(),
     "((K >= N) ? (K - N) : (N - K))",
     "abs(features['K'] - features['N'])"),

    ("max_MKN", "cheap",
     lambda df: df[["M", "K", "N"]].max(axis=1),
     "((M >= K && M >= N) ? M : ((K >= N) ? K : N))",
     "max(features['M'], features['K'], features['N'])"),

    ("min_MKN", "cheap",
     lambda df: df[["M", "K", "N"]].min(axis=1),
     "((M <= K && M <= N) ? M : ((K <= N) ? K : N))",
     "min(features['M'], features['K'], features['N'])"),

    ("M_lt_K", "cheap",
     lambda df: (df["M"] < df["K"]).astype(int),
     "(M < K)",
     "int(features['M'] < features['K'])"),

    ("M_lt_N", "cheap",
     lambda df: (df["M"] < df["N"]).astype(int),
     "(M < N)",
     "int(features['M'] < features['N'])"),

    ("N_lt_K", "cheap",
     lambda df: (df["N"] < df["K"]).astype(int),
     "(N < K)",
     "int(features['N'] < features['K'])"),

    ("M_eq_K", "cheap",
     lambda df: (df["M"] == df["K"]).astype(int),
     "(M == K)",
     "int(features['M'] == features['K'])"),

    ("N_eq_K", "cheap",
     lambda df: (df["N"] == df["K"]).astype(int),
     "(N == K)",
     "int(features['N'] == features['K'])"),

    ("M_eq_N", "cheap",
     lambda df: (df["M"] == df["N"]).astype(int),
     "(M == N)",
     "int(features['M'] == features['N'])"),

    # ── Moderate tier: integer multiply, divide ────────────────────────
    ("MK", "moderate",
     lambda df: df["M"] * df["K"],
     "M * K",
     "features['M'] * features['K']"),

    ("MN", "moderate",
     lambda df: df["M"] * df["N"],
     "M * N",
     "features['M'] * features['N']"),

    ("KN", "moderate",
     lambda df: df["K"] * df["N"],
     "K * N",
     "features['K'] * features['N']"),

    ("MKN", "moderate",
     lambda df: df["M"] * df["K"] * df["N"],
     "M * K * N",
     "features['M'] * features['K'] * features['N']"),

    # M, K, N are matmul dimensions and will never be 0 in practice;
    # zero-guards are included purely as defensive coding.
    ("M_div_K", "moderate",
     lambda df: np.where(df["K"] != 0, df["M"] // df["K"], 0),
     "(K != 0 ? M / K : 0)",
     "(features['M'] // features['K'] if features['K'] != 0 else 0)"),

    ("M_div_N", "moderate",
     lambda df: np.where(df["N"] != 0, df["M"] // df["N"], 0),
     "(N != 0 ? M / N : 0)",
     "(features['M'] // features['N'] if features['N'] != 0 else 0)"),

    ("K_div_N", "moderate",
     lambda df: np.where(df["N"] != 0, df["K"] // df["N"], 0),
     "(N != 0 ? K / N : 0)",
     "(features['K'] // features['N'] if features['N'] != 0 else 0)"),

    # ── Expensive tier: log, sqrt ──────────────────────────────────────
    # These features use costlier math ops. The generated C++ code runs at
    # inference time on every matmul call, so we intentionally use float
    # (sqrtf) over double (sqrt) to minimise latency. All inputs are
    # integers and the result is truncated to int, so float32 precision
    # (~7 digits) is more than sufficient for practical M, K, N ranges.
    ("log2_M", "expensive",
     lambda df: np.log2(df["M"].clip(lower=1)).astype(int),
     "static_cast<int>(log2(M > 1 ? M : 1))",
     "int(math.log2(max(features['M'], 1)))"),

    ("log2_K", "expensive",
     lambda df: np.log2(df["K"].clip(lower=1)).astype(int),
     "static_cast<int>(log2(K > 1 ? K : 1))",
     "int(math.log2(max(features['K'], 1)))"),

    ("log2_N", "expensive",
     lambda df: np.log2(df["N"].clip(lower=1)).astype(int),
     "static_cast<int>(log2(N > 1 ? N : 1))",
     "int(math.log2(max(features['N'], 1)))"),

    ("sqrt_MKN", "expensive",
     lambda df: np.sqrt(df["M"].astype(np.float32) * df["K"] * df["N"]).astype(int),
     "static_cast<int>(sqrtf(static_cast<float>(M) * K * N))",
     "int(math.sqrt(float(features['M']) * features['K'] * features['N']))"),
]

_EXPENSIVE_NAMES = {name for name, tier, *_ in _FEATURE_DEFS if tier == "expensive"}


def _tiers_for_level(level: str) -> set[str]:
    """Return the set of tier names active at a given level."""
    if level == "none":
        return set()
    if level not in TIER_ORDER:
        print(f"WARNING: Unknown feature engineering level '{level}'. "
              f"Valid options: 'none', {TIER_ORDER}. Defaulting to 'none'.")
        return set()
    idx = TIER_ORDER.index(level)
    return set(TIER_ORDER[:idx + 1])


def apply_feature_engineering(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Add derived feature columns to the DataFrame and update config.

    Call after detect_columns() and before split_data().

    Both ``df`` and ``config`` are modified in-place:
    - ``df`` gets new columns for each derived feature.
    - ``config.feature_cols`` is extended with the derived feature names.

    Args:
        df: Training DataFrame (modified in-place).
        config: PipelineConfig (feature_cols updated in-place).

    Returns:
        The modified DataFrame (same object, returned for convenience).
    """
    if config.feature_engineering == "none":
        print("Feature engineering: none (raw features only)")
        return df

    required = {"M", "K", "N"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        print(f"WARNING: Feature engineering requires {required}, "
              f"but {missing} are missing. Skipping.")
        return df

    tiers = _tiers_for_level(config.feature_engineering)
    added = []
    skipped_m = []
    for name, tier, compute_fn, cpp_expr, _ in _FEATURE_DEFS:
        if tier in tiers:
            if config.exclude_m and _M_PATTERN.search(cpp_expr):
                skipped_m.append(name)
                continue
            df[name] = compute_fn(df)
            if name not in config.feature_cols:
                config.feature_cols.append(name)
            added.append(name)

    print(f"Feature engineering ({config.feature_engineering}): "
          f"added {len(added)} derived features")
    if skipped_m:
        print(f"  Skipped {len(skipped_m)} M-dependent features (exclude_m=True)")
    return df


def _should_include(cpp_expr: str, config: PipelineConfig) -> bool:
    """Check if a feature should be included given exclude_m setting."""
    return not (config.exclude_m and _M_PATTERN.search(cpp_expr))


def get_derived_feature_cpp(config: PipelineConfig) -> dict[str, str]:
    """Return {name: cpp_expression} for all derived features in active tier."""
    if config.feature_engineering == "none":
        return {}
    tiers = _tiers_for_level(config.feature_engineering)
    return {name: cpp for name, tier, _, cpp, _ in _FEATURE_DEFS
            if tier in tiers and _should_include(cpp, config)}


def get_derived_feature_py(config: PipelineConfig) -> dict[str, str]:
    """Return {name: python_expression} for all derived features in active tier."""
    if config.feature_engineering == "none":
        return {}
    tiers = _tiers_for_level(config.feature_engineering)
    return {name: py for name, tier, _, cpp, py in _FEATURE_DEFS
            if tier in tiers and _should_include(cpp, config)}


def needs_cmath(used_feature_names: dict[str, str]) -> bool:
    """Check if any of the used features require <cmath> in C++."""
    return bool(set(used_feature_names) & _EXPENSIVE_NAMES)


def needs_math_import(used_feature_names: dict[str, str]) -> bool:
    """Check if any of the used features require 'import math' in Python."""
    return bool(set(used_feature_names) & _EXPENSIVE_NAMES)
