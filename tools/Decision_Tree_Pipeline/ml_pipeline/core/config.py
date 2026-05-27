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

"""Pipeline configuration — all modifiable parameters in one place."""

import re


class PipelineConfig:
    """Centralized configuration for the DT generation pipeline.

    Instantiate, tweak members as needed, and pass to all pipeline functions.
    No global state — everything flows through this object.
    """

    def __init__(self):
        # Run identification — shown in the history table to distinguish workloads.
        self.run_name = ""

        # Column roles (auto-detected after data load via detect_columns)
        self.target_col = "Algo"
        self.weight_col = "Ratio"
        self.baseline_col = "Native_time"
        self.feature_cols = []
        self.all_feature_cols = []
        self.timing_cols = []
        self.algo_to_col = {}
        self.has_baseline = False
        self.timing_col_pattern = re.compile(r'^[A-Za-z]\w*_time$')

        # Feature toggles
        self.exclude_m = False
        self.train_on_whole = False
        # Set by split_data() when train_on_whole + resampling: full resampled frame for fit + whole metrics.
        self.full_df_after_split = None
        self.post_prune_redundant = True
        self.print_params = False
        self.run_cv = False
        self.feature_engineering = "none"
        # If set, restrict derived features to this subset of feature_cols (order preserved).
        # None means: use every numeric column in feature_cols (e.g. BS, Heads, Seq, Dim).
        self.feature_engineering_cols = None
        # Filled by apply_feature_engineering() when tier != "none" (for codegen / export).
        self.derived_feature_cpp = {}
        self.derived_feature_py = {}
        self.derived_feature_excel = {}

        # Weight transformation
        self.weight_transform = "minmax"
        self.weight_minmax_low = 1
        self.weight_minmax_high = 100
        self.weight_clip_percentile = 95

        # Impact group thresholds
        self.impact_threshold_low = 5
        self.impact_threshold_high = 50

        # Resampling (train split only; full dataset when train_on_whole is enabled)
        self.resample_strategy = "none"        # "none", "undersample", "oversample", "hybrid"
        self.imbalance_warn_threshold = 0.33   # warn if minority class < this fraction
        self.undersample_ratio_ceil = 1.05     # drop majority records with Ratio <= this
        self.undersample_max_factor = None     # cap majority at N × minority count (None = no cap)
        self.oversample_target_ratio = 1.0     # target minority/majority ratio after oversampling

        # Minority-aware split: use 25% train / 75% test for more rigorous
        # evaluation when the minority class is small.
        self.minority_split = False

        # Grid search hyperparameters
        self.param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [1, 2, 3, 4],
            'min_samples_split': [2, 3, 4, 5, 7],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'max_features': [None, 'sqrt', 'log2'],
            'splitter': ['best', 'random'],
            'class_weight': [None, 'balanced'],
        }

        # Cross-validation and split
        self.cv_folds = 5
        self.random_state = 42
        self.test_size = 0.3

    def summary(self):
        """Return a human-readable summary of the current configuration."""
        if self.algo_to_col:
            algo_mapping_detail = ", ".join(
                f"{label} → {col}" for label, col in self.algo_to_col.items()
            )
        else:
            algo_mapping_detail = "(not detected)"

        lines = [
            f"All detected features : {self.all_feature_cols}",
            f"Active features (exclude_m={self.exclude_m}): {self.feature_cols}",
            f"Timing columns : {self.timing_cols}",
            f"Baseline column : {self.baseline_col if self.has_baseline else 'NOT FOUND — GeoMean disabled'}",
            f"Target column : {self.target_col}",
            f"Algo mapping   : {algo_mapping_detail}",
            f"Weight column : {self.weight_col}",
        ]

        wt_extra = ""
        if self.weight_transform == 'percentile_clip':
            wt_extra = f" (clip at {self.weight_clip_percentile}th pctl)"
        elif self.weight_transform == 'minmax':
            wt_extra = f" (range [{self.weight_minmax_low}, {self.weight_minmax_high}])"
        lines.append(f"Weight transform : {self.weight_transform}{wt_extra}")

        resample_detail = self.resample_strategy
        if self.resample_strategy != "none":
            parts = []
            if self.resample_strategy in ("undersample", "hybrid"):
                parts.append(f"ceil={self.undersample_ratio_ceil}")
                if self.undersample_max_factor is not None:
                    parts.append(f"max_factor={self.undersample_max_factor}")
            if self.resample_strategy in ("oversample", "hybrid"):
                parts.append(f"target_ratio={self.oversample_target_ratio}")
            if parts:
                resample_detail += f" ({', '.join(parts)})"

        if self.resample_strategy == "none":
            rs_scope = "off"
        elif self.train_on_whole:
            rs_scope = "full dataset (train_on_whole)"
        else:
            rs_scope = "train split only"

        lines.extend([
            f"Resampling          : {resample_detail} ({rs_scope})",
            f"Feature engineering  : {self.feature_engineering}"
            + (f" (bases: {self.feature_engineering_cols})"
               if self.feature_engineering != "none" and self.feature_engineering_cols else ""),
            f"Minority split      : {self.minority_split}"
            + (f" (25% train / 75% test)" if self.minority_split else ""),
            f"Train on whole data  : {self.train_on_whole}"
            + (f"" if self.train_on_whole else
               (f" (train 25% / test 75%)" if self.minority_split else
                f" (train {round((1 - self.test_size) * 100):.0f}% / test {round(self.test_size * 100):.0f}%)")),
            f"Cross-validation     : {self.run_cv} ({self.cv_folds}-fold)" if self.run_cv else f"Cross-validation     : {self.run_cv}",
            f"Redundant branch pruning: {self.post_prune_redundant}",
            f"Print params         : {self.print_params}",
            f"Impact groups        : Minimal (Ratio < {self.impact_threshold_low}), "
            f"Moderate ({self.impact_threshold_low}–{self.impact_threshold_high}), "
            f"Large (>= {self.impact_threshold_high})",
        ])
        return "\n".join(lines)

    def show_all_params(self):
        """Print every configurable parameter with its current value."""
        sep = "─" * 65
        print(sep)
        print("  PipelineConfig — All Parameters")
        print(sep)

        print("\n  Column Roles (auto-detected after load_data + detect_columns):")
        print(f"    target_col          = {self.target_col!r}")
        print(f"    weight_col          = {self.weight_col!r}")
        print(f"    baseline_col        = {self.baseline_col!r}")
        print(f"    has_baseline        = {self.has_baseline}")
        print(f"    timing_col_pattern  = {self.timing_col_pattern.pattern!r}")
        print(f"    feature_cols        = {self.feature_cols}")
        print(f"    all_feature_cols    = {self.all_feature_cols}")
        print(f"    timing_cols         = {self.timing_cols}")

        print("\n  Feature Toggles:")
        print(f"    exclude_m           = {self.exclude_m}")
        print(f"    feature_engineering = {self.feature_engineering!r}")
        print(f"    feature_engineering_cols = {self.feature_engineering_cols!r}")
        print(f"    train_on_whole      = {self.train_on_whole}")
        print(f"    post_prune_redundant= {self.post_prune_redundant}")
        print(f"    print_params        = {self.print_params}")
        print(f"    run_cv              = {self.run_cv}")
        print(f"    run_name            = {self.run_name!r}")

        print("\n  Weight Transformation:")
        print(f"    weight_transform    = {self.weight_transform!r}")
        print(f"    weight_minmax_low   = {self.weight_minmax_low}")
        print(f"    weight_minmax_high  = {self.weight_minmax_high}")
        print(f"    weight_clip_percentile = {self.weight_clip_percentile}")

        print("\n  Impact Group Thresholds:")
        print(f"    impact_threshold_low  = {self.impact_threshold_low}")
        print(f"    impact_threshold_high = {self.impact_threshold_high}")

        print("\n  Train/Test Split:")
        print(f"    test_size           = {self.test_size}")
        print(f"    minority_split      = {self.minority_split}")

        if self.resample_strategy == "none":
            rs_scope = "not applied"
        elif self.train_on_whole:
            rs_scope = "applied to full dataset"
        else:
            rs_scope = "applied to training data only, after split"
        print(f"\n  Resampling ({rs_scope}):")
        print(f"    resample_strategy   = {self.resample_strategy!r}")
        print(f"    imbalance_warn_threshold = {self.imbalance_warn_threshold}")
        print(f"    undersample_ratio_ceil   = {self.undersample_ratio_ceil}")
        print(f"    undersample_max_factor   = {self.undersample_max_factor}")
        print(f"    oversample_target_ratio  = {self.oversample_target_ratio}")

        print("\n  Grid Search Hyperparameters:")
        for key, vals in self.param_grid.items():
            print(f"    {key:<22} = {vals}")
        total = 1
        for vals in self.param_grid.values():
            total *= len(vals)
        print(f"    {'(total combinations)':<22} = {total}")

        print("\n  Cross-Validation & Random State:")
        print(f"    cv_folds            = {self.cv_folds}")
        print(f"    random_state        = {self.random_state}")
        print(sep)
