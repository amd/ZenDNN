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

"""Automation CLI — runs the full ML pipeline end-to-end.

Mirrors the notebook workflow: load data, configure, train, select best model,
and export C++ code.

Usage:
    python run_pipeline.py Sample.csv -o output_dir/
    python run_pipeline.py Sample.csv --exclude-m --weight-transform log+1
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from ml_pipeline.core.config import PipelineConfig
from ml_pipeline.core.data_loader import load_data, detect_columns, split_data
from ml_pipeline.core.feature_engineering import apply_feature_engineering, get_derived_feature_cpp
from ml_pipeline.core.trainer import run_grid_search
from ml_pipeline.core.results import sort_results, display_results, display_impact_groups
from ml_pipeline.core.utils import harmonic_mean
from ml_pipeline.core.code_generator import tree_to_c_code

# Allowed range for the --max-depth CLI argument.
_MIN_DEPTH = 1
_MAX_DEPTH = 50


def export_results_csv(sorted_results: list["ModelResult"], config: "PipelineConfig", csv_path: str | Path) -> None:
    """Write the sorted results table to a CSV file."""
    headers = ["Rank", "Index", "W_Acc_Whole", "Depth", "Nodes",
               "Mispred_Whole"]
    if config.has_baseline:
        headers.append("GeoMean")
    if not config.train_on_whole:
        headers += ["W_Acc_Train", "W_Acc_Test", "HMean",
                     "Mispred_Train", "Mispred_Test"]
    if config.run_cv:
        headers += ["CV_W_Acc", "CV_Std"]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for rank, entry in enumerate(sorted_results, 1):
            row = [rank, entry.index_key, entry.w_acc_whole,
                   entry.max_depth, entry.total_nodes, entry.mispred_whole]
            if config.has_baseline:
                row.append(entry.geo_mean if entry.geo_mean is not None else "")
            if not config.train_on_whole:
                hmean = round(harmonic_mean(entry.w_acc_train, entry.w_acc_test), 2)
                row += [entry.w_acc_train, entry.w_acc_test, hmean,
                        entry.mispred_train, entry.mispred_test]
            if config.run_cv:
                row += [entry.cv_avg if entry.cv_avg is not None else "",
                        entry.cv_std if entry.cv_std is not None else ""]
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DT Generation Pipeline — automated training and export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_pipeline.py data.csv -o results/\n"
            "  python run_pipeline.py data.csv --exclude-m --weight-transform log+1\n"
        ),
    )
    parser.add_argument("csv", help="Path to training CSV")
    parser.add_argument("-o", "--output-dir", default="output",
                        help="Directory for output files (default: output)")
    parser.add_argument("--exclude-m", action="store_true",
                        help="Exclude 'M' from input features")
    parser.add_argument("--train-on-whole", action="store_true",
                        help="Train on whole dataset instead of train split only")
    parser.add_argument("--weight-transform", default="minmax",
                        choices=["raw", "log+1", "sqrt", "minmax", "rank", "percentile_clip"],
                        help="Weight transformation method (default: minmax)")
    parser.add_argument("--minmax-low", type=float, default=1,
                        help="MinMax lower bound (default: 1)")
    parser.add_argument("--minmax-high", type=float, default=100,
                        help="MinMax upper bound (default: 100)")
    parser.add_argument("--no-prune-redundant", action="store_true",
                        help="Disable redundant branch pruning")
    parser.add_argument("--print-params", action="store_true",
                        help="Print hyperparameters during training")
    parser.add_argument("--run-cv", action="store_true",
                        help="Enable cross-validation (off by default, recommended for large datasets)")
    parser.add_argument("--feature-engineering", default="none",
                        choices=["none", "cheap", "moderate", "expensive"],
                        help="Derived feature tier (default: none)")
    parser.add_argument("--feature-engineering-cols", default=None,
                        metavar="COLS",
                        help="Comma-separated subset of feature columns used to build derived "
                             "features (default: all numeric columns in feature_cols, "
                             "e.g. M,K,N for matmul-only interactions while keeping Core)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Display only top N results (default: all)")
    parser.add_argument("--max-depth", type=int, nargs="+", default=None,
                        help="Override max_depth grid values")
    parser.add_argument("--function-name", default="Decision_tree_path_BF16",
                        help="Name for the generated C++ function")
    parser.add_argument("--resample", default="none",
                        choices=["none", "undersample", "oversample", "hybrid"],
                        help="Resampling strategy for class imbalance (default: none). "
                             "Normally applied to training data only, after the split; "
                             "with --train-on-whole, applied to the full dataset before "
                             "the diagnostic split.")
    parser.add_argument("--undersample-ceil", type=float, default=1.05,
                        help="Drop majority records with Ratio <= this (default: 1.05)")
    parser.add_argument("--undersample-max-factor", type=float, default=None,
                        help="Cap majority at N * minority count (default: no cap)")
    parser.add_argument("--oversample-target", type=float, default=1.0,
                        help="Target minority/majority ratio after oversampling (default: 1.0)")
    parser.add_argument("--minority-split", action="store_true",
                        help="Use 25%% train / 75%% test split for rigorous evaluation "
                             "on imbalanced datasets")
    parser.add_argument("--run-name", default="",
                        help="Label for this run in the history table (e.g. 'matmul', 'embbag')")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', args.function_name):
        print(f"ERROR: --function-name '{args.function_name}' is not a valid "
              f"C/C++ identifier")
        sys.exit(1)

    config = PipelineConfig()
    config.exclude_m = args.exclude_m
    config.train_on_whole = args.train_on_whole
    config.weight_transform = args.weight_transform
    if args.minmax_low >= args.minmax_high:
        print(f"ERROR: --minmax-low ({args.minmax_low}) must be less than "
              f"--minmax-high ({args.minmax_high})")
        sys.exit(1)
    config.weight_minmax_low = args.minmax_low
    config.weight_minmax_high = args.minmax_high
    config.post_prune_redundant = not args.no_prune_redundant
    config.print_params = args.print_params
    config.run_cv = args.run_cv
    config.feature_engineering = args.feature_engineering
    if args.feature_engineering_cols:
        config.feature_engineering_cols = [
            c.strip() for c in args.feature_engineering_cols.split(",") if c.strip()]
    config.resample_strategy = args.resample
    if args.undersample_ceil < 0:
        print(f"ERROR: --undersample-ceil must be >= 0 (got {args.undersample_ceil})")
        sys.exit(1)
    config.undersample_ratio_ceil = args.undersample_ceil
    if args.undersample_max_factor is not None and args.undersample_max_factor <= 0:
        print(f"ERROR: --undersample-max-factor must be > 0 (got {args.undersample_max_factor})")
        sys.exit(1)
    config.undersample_max_factor = args.undersample_max_factor
    if args.oversample_target <= 0:
        print(f"ERROR: --oversample-target must be > 0 (got {args.oversample_target})")
        sys.exit(1)
    config.oversample_target_ratio = args.oversample_target
    config.minority_split = args.minority_split
    config.run_name = args.run_name

    if args.max_depth:
        valid_depth_set = set(
            d for d in args.max_depth if _MIN_DEPTH <= d <= _MAX_DEPTH)
        invalid = [d for d in args.max_depth if d not in valid_depth_set]
        valid_depths = sorted(valid_depth_set)
        if invalid:
            print(f"WARNING: Ignoring invalid max_depth values {invalid} "
                  f"(must be {_MIN_DEPTH}–{_MAX_DEPTH})")
        if valid_depths:
            config.param_grid['max_depth'] = valid_depths
        else:
            print("WARNING: No valid max_depth values provided, using defaults.")

    try:
        df = load_data(args.csv)
        detect_columns(df, config)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    apply_feature_engineering(df, config)
    print(config.summary())
    print()

    try:
        train_df, test_df = split_data(df, config)
    except ValueError as e:
        print(f"ERROR: Cannot split data (too few samples per class?): {e}")
        sys.exit(1)
    df_whole = config.full_df_after_split if config.full_df_after_split is not None else df
    try:
        results_list, models_dict = run_grid_search(df_whole, train_df, test_df, config)
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        sys.exit(1)

    if not results_list:
        print("ERROR: No models were trained.")
        sys.exit(1)

    sorted_results = sort_results(results_list, config)
    display_results(sorted_results, config, top_n=args.top_n)

    selected_key = sorted_results[0].index_key

    selected_model = models_dict[selected_key]
    print(f"\nSelected model: {selected_key}")

    display_impact_groups(selected_model, df_whole, train_df, test_df, config,
                          selected_key=selected_key)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv_path = output_dir / "results.csv"
    try:
        export_results_csv(sorted_results, config, results_csv_path)
    except OSError as e:
        print(f"ERROR: Cannot write {results_csv_path}: {e}")
        sys.exit(1)
    print(f"Results CSV written to: {results_csv_path}")

    feature_names = list(config.feature_cols)
    derived_cpp = get_derived_feature_cpp(config)

    base_features = [c for c in config.all_feature_cols if c != 'M'] if config.exclude_m else list(config.all_feature_cols)
    cpp_code = tree_to_c_code(selected_model, feature_names,
                              function_name=args.function_name,
                              base_features=base_features,
                              derived_features=derived_cpp)
    cpp_path = output_dir / "decision_tree.cpp"
    try:
        with open(cpp_path, "w", encoding="utf-8") as f:
            f.write(cpp_code + "\n")
    except OSError as e:
        print(f"ERROR: Cannot write {cpp_path}: {e}")
        sys.exit(1)
    print(f"C++ code written to: {cpp_path}")

    print(f"\nPipeline complete. Output files in: {args.output_dir}/")
    print(f"  - results.csv       : Sorted results table ({len(sorted_results)} models)")
    print(f"  - decision_tree.cpp : C++ code for best model ({selected_key})")


if __name__ == "__main__":
    main()
