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

"""Export fitted DecisionTree models to Python, C++, and Excel formula code.

When derived features are used (via feature engineering), the codegen emits
local variable declarations at the top of the function body.  Only features
the tree actually references are emitted.

Note: sklearn stores thresholds as float64 internally, but our input features
(M, K, N and derived dimensions) are always integers.  We cast thresholds to
int so the generated code uses clean integer comparisons, which is both
required by the C++ function signature and avoids floating-point literals in
the exported predictor.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from sklearn.tree import _tree
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeClassifier

from .feature_engineering import needs_cmath, needs_math_import


def _used_feature_names(tree: DecisionTreeClassifier, feature_names: list[str]) -> set[str]:
    """Return the set of feature names the tree actually splits on."""
    tree_ = tree.tree_
    return {
        feature_names[tree_.feature[n]]
        for n in range(tree_.node_count)
        if tree_.feature[n] != _tree.TREE_UNDEFINED
    }


def tree_to_python_code(tree: DecisionTreeClassifier, feature_names: list[str], derived_features: dict[str, str] | None = None) -> str:
    """Convert a trained DecisionTreeClassifier into a Python function string.

    Args:
        tree: Trained DecisionTreeClassifier model.
        feature_names: List of feature names used during training.
        derived_features: Optional dict {name: python_expression} for
            features computed from raw inputs.  When provided, derived
            features are declared as local variables and referenced by name.

    Returns:
        str: Python function source code.
    """
    check_is_fitted(tree)
    if derived_features is None:
        derived_features = {}

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    used_names = _used_feature_names(tree, feature_names)
    used_derived = {k: v for k, v in derived_features.items() if k in used_names}

    lines = []
    if needs_math_import(used_derived):
        lines.extend(["import math", ""])

    lines.append("def decision_tree_predict(features):")

    if used_derived:
        for name, expr in used_derived.items():
            lines.append(f"    {name} = {expr}")
        lines.append("")

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            # Cast float64 threshold to int — our features are always integers
            threshold = int(tree_.threshold[node])
            if name in derived_features:
                lines.append(f"{indent}if {name} <= {threshold}:")
            else:
                lines.append(f"{indent}if features['{name}'] <= {threshold}:")
            recurse(tree_.children_left[node], depth + 1)
            lines.append(f"{indent}else:")
            recurse(tree_.children_right[node], depth + 1)
        else:
            class_index = tree_.value[node].argmax()
            class_label = tree.classes_[class_index]
            lines.append(f"{indent}return {class_label}")

    recurse(0, 1)
    return "\n".join(lines)


def tree_to_c_code(tree: DecisionTreeClassifier, feature_names: list[str], function_name: str = "Decision_tree_path_BF16", base_features: list[str] | None = None, derived_features: dict[str, str] | None = None) -> str:
    """Convert a trained DecisionTreeClassifier into a C++ function string.

    Args:
        tree: Trained DecisionTreeClassifier model.
        feature_names: List of feature names used during training.
        function_name: Name for the generated C++ function.
        base_features: List of raw input features for the function signature.
            If None, falls back to feature_names (backward compatible).
        derived_features: Optional dict {name: cpp_expression} for features
            computed from the base inputs.  Emitted as local int variables.

    Returns:
        str: C++ function source code.
    """
    check_is_fitted(tree)
    if derived_features is None:
        derived_features = {}
    if base_features is None:
        base_features = [f for f in feature_names
                         if f not in derived_features]

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    used_names = _used_feature_names(tree, feature_names)
    used_derived = {k: v for k, v in derived_features.items() if k in used_names}

    lines = []

    if needs_cmath(used_derived):
        lines.extend(["#include <cmath>", ""])

    # Function signature — only base (raw) features as parameters
    params = ",\n    ".join(f"int {name}" for name in base_features)
    lines.append(f"int {function_name}(")
    lines.append(f"    {params}")
    lines.append(")")
    lines.append("{")

    # Derived feature declarations
    if used_derived:
        for name, expr in used_derived.items():
            lines.append(f"    int {name} = {expr};")
        lines.append("")

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            # Cast float64 threshold to int — our features are always integers
            threshold = int(tree_.threshold[node])
            lines.append(f"{indent}if ({name} <= {threshold})")
            lines.append(f"{indent}{{")
            recurse(tree_.children_left[node], depth + 1)
            lines.append(f"{indent}}}")
            lines.append(f"{indent}else")
            lines.append(f"{indent}{{")
            recurse(tree_.children_right[node], depth + 1)
            lines.append(f"{indent}}}")
        else:
            class_index = int(tree_.value[node].argmax())
            algo_path = int(tree.classes_[class_index])
            lines.append(f"{indent}return {algo_path};")

    recurse(0, 1)
    lines.append("}")

    return "\n".join(lines)


# ── Excel formula generation ───────────────────────────────────────────

def _col_letter(index: int) -> str:
    """Convert a 0-based column index to an Excel column letter (A, B, ..., Z, AA, ...)."""
    result = ""
    n = index
    while True:
        result = chr(ord('A') + n % 26) + result
        n = n // 26 - 1
        if n < 0:
            break
    return result


def _default_cell_map(feature_names: list[str], row: int = 2) -> dict[str, str]:
    """Build a default feature → cell mapping (A{row}, B{row}, ...)."""
    return {name: f"{_col_letter(i)}{row}" for i, name in enumerate(feature_names)}


def _substitute_cell_refs(excel_expr: str, cell_map: dict[str, str]) -> str:
    """Replace raw variable names in an Excel expression with cell references.

    Substitutes longest names first to avoid partial matches
    (e.g. "MN" before "M", or "K_N" before "K").
    """
    expr = excel_expr
    for name in sorted(cell_map, key=len, reverse=True):
        expr = re.sub(rf'\b{re.escape(name)}\b', cell_map[name], expr)
    return expr


def tree_to_excel_formula(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    cell_map: dict[str, str] | None = None,
    base_features: list[str] | None = None,
    derived_features: dict[str, str] | None = None,
) -> str:
    """Convert a trained DecisionTreeClassifier into a nested Excel IF() formula.

    Args:
        tree: Trained DecisionTreeClassifier model.
        feature_names: List of feature names used during training.
        cell_map: Optional dict mapping feature names to Excel cell references.
            If None, defaults to {first_feature: "A2", second: "B2", ...}
            based on ``base_features`` ordering.
        base_features: Raw input feature names (for default cell_map ordering).
            Falls back to feature_names if not provided.
        derived_features: Optional dict {name: excel_expression} for features
            computed from base inputs.  Pass the output of
            ``get_derived_feature_excel(config)``; these use native Excel
            syntax (IF, ABS, MAX, etc.) with raw variable names that are
            substituted with cell references automatically.

    Returns:
        str: A single Excel formula string (e.g. ``=IF(A2<=10,1,IF(B2<=5,2,1))``).

    Note:
        Thresholds are cast to int, which is correct for integer-valued features
        (sklearn uses x.5 midpoints). If float features are used, this truncation
        may shift decision boundaries.
    """
    check_is_fitted(tree)
    if derived_features is None:
        derived_features = {}
    if base_features is None:
        base_features = [f for f in feature_names
                         if f not in derived_features]

    if cell_map is None:
        cell_map = _default_cell_map(base_features)

    # Substitute cell references into native Excel expressions.
    derived_excel = {}
    for name, xl_expr in derived_features.items():
        derived_excel[name] = _substitute_cell_refs(xl_expr, cell_map)

    used = _used_feature_names(tree, feature_names)
    used_derived = {k: v for k, v in derived_excel.items() if k in used}
    if used_derived:
        print("NOTE: Derived features inlined as Excel sub-expressions:")
        for name, xl_expr in used_derived.items():
            print(f"  {name} → {xl_expr}")

    tree_ = tree.tree_

    def _cell_ref(node: int) -> str:
        """Return the Excel cell reference or inlined expression for a node's feature."""
        name = feature_names[tree_.feature[node]]
        if name in derived_excel:
            return derived_excel[name]
        return cell_map.get(name, name)

    def recurse(node: int) -> str:
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            class_index = int(tree_.value[node].argmax())
            return str(int(tree.classes_[class_index]))
        ref = _cell_ref(node)
        raw = tree_.threshold[node]
        threshold = str(int(raw))
        left = recurse(tree_.children_left[node])
        right = recurse(tree_.children_right[node])
        return f"IF({ref}<={threshold},{left},{right})"

    formula = "=" + recurse(0)

    print(f"\nCell mapping (feature → cell):")
    for name in base_features:
        marker = " *" if name in used else ""
        print(f"  {name:<20} → {cell_map.get(name, '?')}{marker}")
    if used_derived:
        print("  (* = used by tree)")
    print(f"\nFormula length: {len(formula)} characters")

    return formula
