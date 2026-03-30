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

"""Convert a Markdown file with fenced code blocks into a Jupyter notebook.

Fenced code blocks (```python ... ```) become code cells.
Everything else becomes markdown cells.

Usage:
    python ipynb_generator.py DT_Pipeline.md DT_Pipeline.ipynb
"""

from __future__ import annotations

import json
import sys


def md_to_notebook(md_path: str, ipynb_path: str | None = None) -> bool:
    if ipynb_path is None:
        ipynb_path = md_path.rsplit(".", 1)[0] + ".ipynb"

    try:
        with open(md_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: File not found: {md_path}")
        return False
    except OSError as e:
        print(f"ERROR: Cannot read {md_path}: {e}")
        return False

    cells = []
    buf = []
    in_code = False

    for line in lines:
        stripped = line.strip()

        if not in_code and stripped.startswith("```") and len(stripped) > 3:
            if buf:
                text = "".join(buf).strip()
                if text:
                    cells.append(_markdown_cell(buf))
                buf = []
            in_code = True
            continue

        if in_code and stripped == "```":
            cells.append(_code_cell(buf))
            buf = []
            in_code = False
            continue

        buf.append(line)

    if buf:
        text = "".join(buf).strip()
        if text:
            if in_code:
                cells.append(_code_cell(buf))
            else:
                cells.append(_markdown_cell(buf))

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "cells": cells,
    }

    try:
        with open(ipynb_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1)
    except OSError as e:
        print(f"ERROR: Cannot write {ipynb_path}: {e}")
        return False

    print(f"Converted {md_path} -> {ipynb_path} ({len(cells)} cells)")
    return True


def _clean_source(lines: list[str]) -> list[str]:
    """Strip trailing blank lines, keep internal structure, ensure newlines."""
    while lines and lines[-1].strip() == "":
        lines.pop()
    while lines and lines[0].strip() == "":
        lines.pop(0)
    if not lines:
        return []
    source = []
    for line in lines:
        if not line.endswith("\n"):
            line += "\n"
        source.append(line)
    if source:
        source[-1] = source[-1].rstrip("\n")
    return source


def _code_cell(lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _clean_source(lines),
    }


def _markdown_cell(lines: list[str]) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _clean_source(lines),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ipynb_generator.py <input.md> [output.ipynb]")
        sys.exit(1)
    success = md_to_notebook(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    if not success:
        sys.exit(1)
