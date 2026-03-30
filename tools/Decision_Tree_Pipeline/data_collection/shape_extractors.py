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

"""Shape extraction functions for various PyTorch operator types."""

from __future__ import annotations

import re


def _has_add_buffer(inner_lists: list[str]) -> bool:
    """Detect an add_buffer operand by counting comma-separated sub-lists.

    If there are more than 2 sub-lists containing commas (i.e. multi-element
    tensors), the extra one is the add_buffer bias term.
    """
    return sum(1 for x in inner_lists if "," in x) > 2


def extract_bmkn_from_aten_bmm(shape_str: str) -> str:
    """Extract M, K, N from aten::bmm input shape string.

    Expected format: [[B, M, K], [B, K, N]]
    Returns: comma-separated string "M,K,N,B" (consistent with other extractors).
    """
    try:
        inner_lists = re.findall(r'\[([^\[\]]*)\]', shape_str)
        b_m_k = inner_lists[0].split(',')
        b_k_n = inner_lists[1].split(',')
        b = b_m_k[0].strip()
        m = b_m_k[1].strip()
        k = b_m_k[2].strip()
        n = b_k_n[2].strip()
        return f"{m},{k},{n},{b}"
    except (IndexError, AttributeError):
        return '-1,-1,-1,-1'


def extract_mkn_from_zendnn_linear(shape_str: str) -> str:
    """Extract M, K, N and add_buffer flag from zentorch linear shapes.

    Handles both 2D ([M, K]) and 3D ([M1, M2, N]) input tensors.
    """
    try:
        inner_lists = re.findall(r'\[([^\[\]]*)\]', shape_str)
        inner_lists = [x for x in inner_lists if x != ""]

        add_buffer = "True" if _has_add_buffer(inner_lists) else "False"

        m_dims = inner_lists[0].split(',')
        if len(m_dims) > 2:
            m0 = m_dims[0].strip()
            m1 = m_dims[1].strip()
            if m0.isdigit() and m1.isdigit():
                m = str(int(m0) * int(m1))
            else:
                return '-1,-1,-1,-1'
        else:
            m = m_dims[0].strip()

        n_k = inner_lists[1].split(',')
        k = n_k[1].strip()
        n = n_k[0].strip()
        return f"{m},{k},{n},{add_buffer}"
    except (IndexError, ValueError, AttributeError):
        return '-1,-1,-1,-1'


def extract_mkn_from_mkldnn_linear_pointwise(shape_str: str) -> str:
    """Extract M, K, N and add_buffer flag from mkldnn linear pointwise shapes."""
    try:
        inner_lists = re.findall(r'\[([^\[\]]*)\]', shape_str)
        inner_lists = [x for x in inner_lists if x != ""]

        add_buffer = "True" if _has_add_buffer(inner_lists) else "False"

        m_k = inner_lists[0].split(',')
        last_parts = inner_lists[-1].split(",")
        n = last_parts[0].strip() if len(last_parts) >= 1 else inner_lists[-1]

        m = m_k[0].strip()
        k = m_k[1].strip()
        return f"{m},{k},{n},{add_buffer}"
    except (IndexError, ValueError, AttributeError):
        return '-1,-1,-1,-1'


OP_EXTRACTORS = {
    'zentorch::zentorch_linear':    (extract_mkn_from_zendnn_linear, "matmul"),
    'mkldnn::_linear_pointwise':    (extract_mkn_from_mkldnn_linear_pointwise, "matmul"),
    'aten::bmm':                    (extract_bmkn_from_aten_bmm, "bmm"),
    'zendnn::zendnn_baddbmm':       (extract_bmkn_from_aten_bmm, "bmm"),
}

DEFAULT_TARGET_OPS = ["zentorch::zentorch_linear"]
