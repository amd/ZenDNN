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

"""Parse PyTorch profiler logs and extract target operator metrics."""

from __future__ import annotations

import re
from pathlib import Path

from .utils import time_converter
from .shape_extractors import DEFAULT_TARGET_OPS, OP_EXTRACTORS

PROFILER_COLUMNS = [
    "Name", "Self CPU %", "Self CPU", "CPU total %", "CPU total",
    "CPU time avg", "CPU Mem", "Self CPU Mem", "# of Calls", "Input Shapes",
]


def parse_profiler_log(log_file_path: str) -> tuple[dict[str, list[str]], float]:
    """Parse a PyTorch profiler log file into column-wise data.

    Returns:
        tuple: (column_dict, total_time_ms)
    """
    try:
        with open(log_file_path, "r") as f:
            lines = f.readlines()
    except (OSError, IOError) as e:
        print(f"WARNING: Cannot read '{log_file_path}' — {e}")
        return {col: [] for col in PROFILER_COLUMNS}, 0

    if len(lines) < 6:
        print(f"WARNING: Skipping '{log_file_path}' — too few lines ({len(lines)})")
        return {col: [] for col in PROFILER_COLUMNS}, 0

    total_time_raw = ""
    for line in reversed(lines):
        if "Self CPU time total:" in line:
            total_time_raw = line.split("Self CPU time total:")[1].strip()
            break
    converted = time_converter([total_time_raw])
    total_time = converted[0] if converted else 0.0

    columns = {col: [] for col in PROFILER_COLUMNS}

    for line in lines[3:-2]:
        if re.match(r"^-+$", line.strip()):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < len(PROFILER_COLUMNS):
            continue
        for i, col in enumerate(PROFILER_COLUMNS):
            columns[col].append(parts[i])

    return columns, total_time


def normalize_model_name(file_path: str) -> str:
    """Extract a clean model name from a profiler log file path.

    Preserves rank suffix to distinguish multiple files for the same model.
    E.g. 'zendnn_resnet50_rank_0.log' -> 'resnet50_rank_0'
    """
    base = Path(file_path).stem
    if "zendnn_" in base and "_rank_" in base:
        return base.split("zendnn_")[1]
    if "inductor_" in base and "_rank_" in base:
        return base.split("inductor_")[1]
    return base


def process_txt(input_file_path: str, target_ops: list[str] | None = None) -> tuple[dict[str, dict[str, list]], list[float]]:
    """Load profiler log(s) and extract target op metrics.

    Accepts either a single file path or a directory.

    Args:
        input_file_path: Path to a single log file or a directory of logs.
        target_ops: List of op name substrings to filter for.
                    Defaults to DEFAULT_TARGET_OPS if not provided.

    Returns:
        tuple: (model_data_dict, time_list)
    """
    if target_ops is None:
        target_ops = DEFAULT_TARGET_OPS
    src = Path(input_file_path)
    if src.is_file():
        paths = [str(src)]
    elif src.is_dir():
        try:
            paths = sorted(
                str(p) for p in src.iterdir()
                if p.is_file() and p.suffix in (".log", ".txt")
            )
        except PermissionError as exc:
            raise PermissionError(f"Cannot read directory '{src}': {exc}") from exc
    else:
        raise FileNotFoundError(f"Invalid path: {input_file_path}")

    model_data = {}
    time_list = []

    for file_path in paths:
        model_name = normalize_model_name(file_path)
        parsed_data, total_time = parse_profiler_log(file_path)
        time_list.append(total_time)

        if not parsed_data.get("Name"):
            print(f"WARNING: No data parsed from {file_path}")
            continue

        filtered = {
            "Name": [], "CPU%": [], "Self_CPU%": [], "Self_CPU(ms)": [],
            "CPU(ms)": [], "Count": [], "core_op": [], "extracted_shape": [],
        }

        for i, op_name in enumerate(parsed_data["Name"]):
            for target_op in target_ops:
                if target_op not in op_name:
                    continue

                raw_shape = parsed_data["Input Shapes"][i]

                # Parse M,K,N dimensions from the raw profiler shape string
                extracted_shape = "-1,-1,-1,-1"
                core_op = op_name
                for op_key, (extractor, label) in OP_EXTRACTORS.items():
                    if op_key in op_name:
                        extracted_shape = extractor(raw_shape)
                        core_op = label
                        break

                filtered["Name"].append(op_name)
                filtered["Self_CPU%"].append(parsed_data["Self CPU %"][i])
                filtered["Self_CPU(ms)"].append(parsed_data["Self CPU"][i])
                filtered["CPU%"].append(parsed_data["CPU total %"][i])
                filtered["CPU(ms)"].append(parsed_data["CPU total"][i])
                filtered["Count"].append(parsed_data["# of Calls"][i])
                filtered["extracted_shape"].append(extracted_shape)
                filtered["core_op"].append(core_op)
                break

        model_data[model_name] = filtered

    return model_data, time_list
