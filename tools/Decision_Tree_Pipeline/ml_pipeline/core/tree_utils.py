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

"""Tree structure operations: fingerprinting, pruning, and node counting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.tree import _tree
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeClassifier

# Decimal places used when rounding split thresholds for tree fingerprinting.
# Two trees whose thresholds agree to this many digits are treated as identical.
_FINGERPRINT_PRECISION = 6


def get_tree_fingerprint(model: DecisionTreeClassifier) -> tuple:
    """Create a hashable fingerprint of a fitted DecisionTree's reachable structure.

    Two trees with the same fingerprint are functionally identical.

    Args:
        model: A fitted DecisionTreeClassifier.

    Returns:
        tuple: Hashable fingerprint.
    """
    check_is_fitted(model)
    tree = model.tree_
    parts = []

    def _traverse(node_id):
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            parts.append(('L', int(np.argmax(tree.value[node_id, 0]))))
        else:
            parts.append(('N', int(tree.feature[node_id]),
                          round(float(tree.threshold[node_id]), _FINGERPRINT_PRECISION)))
            _traverse(tree.children_left[node_id])
            _traverse(tree.children_right[node_id])

    _traverse(0)
    return tuple(parts)


def simplify_redundant_branches(model: DecisionTreeClassifier) -> int:
    """Collapse subtrees where all leaves predict the same class.

    Walks bottom-up. Any node whose entire subtree maps to a single
    class is converted to a leaf. Does NOT change predictions.

    Args:
        model: A fitted DecisionTreeClassifier (modified in-place).

    Returns:
        int: Number of nodes removed.
    """
    tree = model.tree_
    nodes_removed = 0

    def _dominant_class(node_id):
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            return int(np.argmax(tree.value[node_id, 0]))
        left_cls = _dominant_class(tree.children_left[node_id])
        right_cls = _dominant_class(tree.children_right[node_id])
        return left_cls if (left_cls == right_cls and left_cls != -1) else -1

    def _count_descendants(node_id):
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            return 0
        return (2 + _count_descendants(tree.children_left[node_id])
                  + _count_descendants(tree.children_right[node_id]))

    def _simplify(node_id):
        nonlocal nodes_removed
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            return
        _simplify(tree.children_left[node_id])
        _simplify(tree.children_right[node_id])
        if _dominant_class(node_id) != -1:
            nodes_removed += _count_descendants(node_id)
            tree.feature[node_id] = _tree.TREE_UNDEFINED
            tree.children_left[node_id] = _tree.TREE_LEAF
            tree.children_right[node_id] = _tree.TREE_LEAF

    _simplify(0)
    return nodes_removed


def count_reachable_nodes(model: DecisionTreeClassifier) -> int:
    """Count nodes reachable from root (excludes pruned unreachable nodes).

    Args:
        model: A fitted DecisionTreeClassifier.

    Returns:
        int: Number of reachable nodes.
    """
    tree = model.tree_
    count = [0]

    def _traverse(node_id):
        count[0] += 1
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            _traverse(tree.children_left[node_id])
            _traverse(tree.children_right[node_id])

    _traverse(0)
    return count[0]
