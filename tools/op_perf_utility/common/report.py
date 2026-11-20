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
"""Rows of a report (one dict per call) as an aligned table, CSV or JSON."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List, Optional, Sequence

FORMATS = ("table", "csv", "json")


def _cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "%.4g" % v
    return str(v)


def render(rows: List[Dict[str, Any]], fmt: str = "table", columns: Optional[Sequence[str]] = None) -> str:
    """rows in fmt; columns picks and orders the keys (default: the first row's)."""
    if fmt not in FORMATS:
        raise ValueError("format %r is not one of %s" % (fmt, ", ".join(FORMATS)))
    if fmt == "json":
        return json.dumps(rows, indent=1)
    cols = list(columns) if columns else (list(rows[0]) if rows else [])
    if fmt == "csv":
        out = io.StringIO()
        w = csv.writer(out, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])
        return out.getvalue()
    cells = [[_cell(r.get(c)) for c in cols] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]
    lines = ["  ".join(c.rjust(w) for c, w in zip(cols, widths))]
    lines += ["  ".join(v.rjust(w) for v, w in zip(row, widths)) for row in cells]
    return "\n".join(lines) + "\n"
