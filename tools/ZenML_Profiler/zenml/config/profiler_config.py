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

import json
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, List

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_log_info(path=None):
    """Load backend log format definitions from JSON."""
    if path is None:
        path = os.path.join(_CONFIG_DIR, "log_info.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load log_info.json from {path}: {e}") from e


log_info = load_log_info()

SAFE_MODE_OPS = ["matmul", "batch_matmul", "inner_product", "softmax", "convolution"]

EMBEDDING_CREATE_OPS = {"embedding_context_create", "embedding_op_create"}


@dataclass
class ProfilerConfig:
    """Central configuration for the ZenML Profiler pipeline."""

    # Per-log inputs (parallel lists, indexed by log number)
    inputs: List[str] = field(default_factory=list)
    backends: List[str] = field(default_factory=list)
    iterations: List[Optional[int]] = field(default_factory=list)
    numbers: List[Optional[int]] = field(default_factory=list)
    log_names: List[str] = field(default_factory=list)
    t_times: List[str] = field(default_factory=list)

    # Global settings
    log_count: int = 1
    compare_mm: bool = False
    disable_excel: bool = False
    disable_pattern: bool = False
    benchdnn: bool = False
    benchdnn_disp: bool = False
    flops: bool = False
    verbose: bool = False
    info: bool = False
    safe_mode: bool = False
    csv: bool = False
    no_flags: bool = False
    export: bool = False
    file_name: str = "sliced_output.txt"
    result_file: str = "ZenMLProfiler_report.txt"
    threshold: int = 2
    machine: str = ""
    roofline_path: str = ""
    inference_script: str = "custom"

    # PyTorch profiler inputs (parallel lists)
    pytorch_profiler_jsons: List[Optional[str]] = field(default_factory=list)
    pytorch_profiler_logs: List[Optional[str]] = field(default_factory=list)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ProfilerConfig":
        """Build ProfilerConfig from a parsed argparse.Namespace."""
        cfg = cls()
        cfg.log_count = args.log_count
        cfg.compare_mm = args.compare_mm
        cfg.disable_excel = args.disable_excel
        cfg.disable_pattern = args.disable_pattern
        cfg.benchdnn = args.benchdnn
        cfg.benchdnn_disp = args.benchdnn_disp
        cfg.flops = getattr(args, "flops", False)
        cfg.verbose = args.verbose
        cfg.info = args.info
        cfg.csv = args.csv
        cfg.no_flags = args.no_flags
        cfg.export = bool(args.export)
        cfg.file_name = args.file_name
        cfg.result_file = args.result_file
        cfg.threshold = args.threshold
        cfg.machine = args.machine
        cfg.roofline_path = getattr(args, "roofline_path", "")
        cfg.inference_script = args.inference_script

        for i in range(1, args.log_count + 1):
            cfg.inputs.append(getattr(args, f"input_{i}", False) or "")
            cfg.backends.append(getattr(args, f"backend_{i}", "ZenDNN_5.2"))

            iter_val = getattr(args, f"iter_{i}", False)
            cfg.iterations.append(int(iter_val) if iter_val and iter_val is not False else None)

            num_val = getattr(args, f"number_{i}", False)
            cfg.numbers.append(int(num_val) if num_val and num_val is not False else None)

            ln = getattr(args, f"log_name_{i}", False)
            cfg.log_names.append(ln if isinstance(ln, str) else f"Log {i}")

            t = getattr(args, f"time_{i}", False)
            cfg.t_times.append(t if isinstance(t, str) else "")

            ppj = getattr(args, f"pytorch_profiler_json_{i}", False)
            cfg.pytorch_profiler_jsons.append(ppj if isinstance(ppj, str) else None)

            ppl = getattr(args, f"pytorch_profiler_log_{i}", False)
            cfg.pytorch_profiler_logs.append(ppl if isinstance(ppl, str) else None)

        cfg.flops = cfg.roofline_path != "" and cfg.machine != ""

        return cfg

    @staticmethod
    def args_from_json(json_path: str) -> list:
        """Extract CLI-style arguments from a JSON file (args.json format).

        Returns a list of strings suitable for passing to argparse.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        arguments = []
        for key, val in data.get("ZenML", {}).items():
            if key == "_helper":
                continue
            val_str = str(val)
            if val_str == "" or val_str.upper() == "FALSE":
                continue
            if val_str.upper() == "TRUE":
                arguments.append(str(key))
            else:
                arguments.append(str(key))
                arguments.append(val_str)
        return arguments
