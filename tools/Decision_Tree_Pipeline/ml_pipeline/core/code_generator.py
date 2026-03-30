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

"""Export fitted DecisionTree models to Python and C++ source code.

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
        base_features = feature_names

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
