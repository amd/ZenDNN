# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""Shared test fixtures: the synthetic machine's ceilings."""

from __future__ import annotations

import copy
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from op_perf_utility.common.ceilings import Ceilings  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
EXAMPLE = os.path.join(FIXTURES, "example_ceilings.json")


def example_doc() -> Dict[str, Any]:
    with open(EXAMPLE, encoding="utf-8") as f:
        return json.load(f)


def example(**changes: Any) -> Ceilings:
    """The synthetic machine, with top-level blocks replaced by changes."""
    doc = copy.deepcopy(example_doc())
    doc.update(changes)
    return Ceilings(doc, "example")
