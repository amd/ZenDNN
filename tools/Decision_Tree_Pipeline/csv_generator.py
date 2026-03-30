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

"""Generate training CSV from PyTorch profiler logs.

Takes a parent directory containing algo subfolders and produces a CSV
ready for the ML pipeline.

Usage:
    python csv_generator.py /path/to/parent_dir
    python csv_generator.py /path/to/parent_dir -o output.csv -scale 100

Expected parent directory layout:
    parent_dir/
    ├── aocl/       (required)
    ├── brgemm/     (required)
    ├── libxsmm/    (optional)
    └── native/     (optional)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_collection.csv_builder import build_csv, DEFAULT_RATIO_SCALE

SUBFOLDER_MAP = {
    "aocl": 1,
    "brgemm": 2,
    "libxsmm": 3,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training CSV from PyTorch profiler logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Expected parent directory layout:\n"
            "  parent_dir/\n"
            "  ├── aocl/       (required)\n"
            "  ├── brgemm/     (required)\n"
            "  ├── libxsmm/    (optional)\n"
            "  └── native/     (optional)\n"
        ),
    )
    parser.add_argument("parent_dir", help="Directory containing algo subfolders")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path (default: <parent_dir>/output.csv)")
    parser.add_argument("-scale", type=float, default=DEFAULT_RATIO_SCALE,
                        help=f"Ratio scale factor (default: {DEFAULT_RATIO_SCALE})")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print detailed progress information")
    args = parser.parse_args()

    parent = Path(args.parent_dir)
    if not parent.is_dir():
        print(f"ERROR: '{parent}' is not a directory.")
        sys.exit(1)

    algo_paths = {}
    for name, algo_id in SUBFOLDER_MAP.items():
        sub = parent / name
        if sub.is_dir():
            algo_paths[algo_id] = str(sub)

    native_path = None
    native_dir = parent / "native"
    if native_dir.is_dir():
        native_path = str(native_dir)

    required = {"aocl": 1, "brgemm": 2}
    missing = [name for name, aid in required.items() if aid not in algo_paths]
    if missing:
        print(f"ERROR: Required subfolder(s) missing in {parent}: {', '.join(missing)}/")
        print(f"  Found: {[n for n in SUBFOLDER_MAP if SUBFOLDER_MAP[n] in algo_paths]}")
        sys.exit(1)

    output_path = args.output or str(parent / "output.csv")

    if args.verbose:
        print(f"Parent directory: {parent}")
        print(f"  aocl/    -> algo 1 (AOCL)")
        print(f"  brgemm/  -> algo 2 (BRGEMM)")
        if 3 in algo_paths:
            print(f"  libxsmm/ -> algo 3 (LIBXSMM)")
        if native_path:
            print(f"  native/  -> baseline (Native)")
        print()

    try:
        count = build_csv(algo_paths, native_path, output_path, args.scale,
                          verbose=args.verbose)
    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
