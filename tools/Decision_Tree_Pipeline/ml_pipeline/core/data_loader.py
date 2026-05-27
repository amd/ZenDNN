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

"""Data loading, column detection, splitting, and feature preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .resampling import detect_imbalance, resample

if TYPE_CHECKING:
    from .config import PipelineConfig

# Ratio values below this boundary are considered "near-equal performance"
# for stratified splitting, ensuring both train/test sets contain a mix of
# low-impact and high-impact samples.
_RATIO_STRATIFY_BOUNDARY = 1.04


def load_data(csv_path: str) -> pd.DataFrame:
    """Read CSV and return a DataFrame.

    Args:
        csv_path: Path to the training CSV.

    Returns:
        pandas DataFrame.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV file not found: {csv_path}") from exc
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read CSV file '{csv_path}': {exc}") from exc
    if df.empty:
        raise ValueError(f"CSV file '{csv_path}' has no data rows.")
    print(f"Loaded {len(df)} records from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    return df


def detect_columns(df: pd.DataFrame, config: PipelineConfig) -> None:
    """Auto-detect timing, feature, and baseline columns; populate config.

    Timing columns are those whose names match ``config.timing_col_pattern``
    (by default, a name that starts with a letter, followed by word chars,
    and ending with ``_time``; e.g., ``AOCL_time``, ``Native_time``).
    All columns before the first timing column are treated as input features.

    Raises ValueError if the CSV structure is invalid (missing required
    columns, no timing columns, label/column count mismatch, etc.).

    Args:
        df: Training DataFrame.
        config: PipelineConfig to populate in-place.
    """
    if config.target_col not in df.columns:
        raise ValueError(
            f"Target column '{config.target_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}")
    if config.weight_col not in df.columns:
        raise ValueError(
            f"Weight column '{config.weight_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}")
    algo_idx = list(df.columns).index(config.target_col)

    # Detect timing columns by pattern, not positional slicing.
    timing_matches = [
        col for col in df.columns[:algo_idx]
        if config.timing_col_pattern.match(col)
    ]

    if timing_matches:
        first_timing_idx = list(df.columns).index(timing_matches[0])
        config.all_feature_cols = list(df.columns[:first_timing_idx])
        config.timing_cols = timing_matches

        # Warn about unrecognized columns sitting in the timing region.
        region_cols = set(df.columns[first_timing_idx:algo_idx])
        unrecognized = region_cols - set(timing_matches)
        if unrecognized:
            print(f"WARNING: Column(s) {sorted(unrecognized)} sit between timing "
                  f"columns and '{config.target_col}' but do not match the "
                  f"timing pattern (*_time). They will be ignored.")
    else:
        config.all_feature_cols = [
            c for c in df.columns[:algo_idx] if c != config.weight_col
        ]
        config.timing_cols = []

    config.has_baseline = config.baseline_col in config.timing_cols

    non_baseline = [c for c in config.timing_cols if c != config.baseline_col]

    if len(non_baseline) < 2:
        raise ValueError(
            f"Need at least 2 non-baseline timing columns for classification, "
            f"found {len(non_baseline)}: {non_baseline}. "
            f"Timing columns must end with '_time' (e.g., 'AOCL_time'). "
            f"Detected timing columns: {config.timing_cols}")

    # Build mapping from timing columns (label 1..N in column order).
    # This matches how csv_builder assigns integer algo IDs.
    config.algo_to_col = {
        algo_idx: col for algo_idx, col in zip(
            range(1, len(non_baseline) + 1), non_baseline
        )
    }

    # Validate that observed Algo labels are within the expected range.
    algo_labels = sorted(df[config.target_col].unique())
    invalid_labels = [label for label in algo_labels if label not in config.algo_to_col]
    if invalid_labels:
        raise ValueError(
            f"Algo column contains labels {invalid_labels} that do not map to "
            f"any timing column. Expected labels in "
            f"{sorted(config.algo_to_col.keys())} based on non-baseline "
            f"timing columns: {non_baseline}.")

    if not config.has_baseline:
        print("NOTE: No 'Native_time' baseline column found.")
        print("      GeoMean will NOT be calculated; models ranked by "
              "W_Acc (whole) or HMean (train/test).\n")

    if config.exclude_m:
        config.feature_cols = [c for c in config.all_feature_cols if c != 'M']
    else:
        config.feature_cols = config.all_feature_cols.copy()

    detect_imbalance(df, config)


def split_data(df: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split with optional resampling.

    When ``config.minority_split`` is True the split flips to 25% train /
    75% test, giving more test data for rigorous evaluation on imbalanced
    datasets.  Otherwise the standard ``config.test_size`` is used.

    Resampling rules:

    - **Default** (``train_on_whole`` is False): after splitting, if
      ``resample_strategy`` is not ``"none"``, resampling runs **only on the
      training** split so the test set stays a clean holdout.

    - **Train on whole** (``train_on_whole`` is True) **and** resampling is
      enabled: the **entire** dataframe is resampled first, then a stratified
      train/test split is applied to that resampled data for diagnostics only.
      ``config.full_df_after_split`` is set to the resampled full frame so
      ``run_grid_search`` fits and scores whole-data metrics on the same rows.

    Args:
        df: Full DataFrame.
        config: PipelineConfig.

    Returns:
        tuple: (train_df, test_df) with reset indices.
    """
    config.full_df_after_split = None

    effective_test_size = 0.75 if config.minority_split else config.test_size

    if config.minority_split:
        print(f"\nMinority split enabled: 25% train / 75% test")

    bins = [0, _RATIO_STRATIFY_BOUNDARY, float('inf')]
    labels = [1, 2]

    if config.train_on_whole and config.resample_strategy != "none":
        df_base = df.copy().reset_index(drop=True)
        print("\ntrain_on_whole + resampling: applying strategy to the **full** "
              "dataset, then diagnostic train/test split.\n")
        df_resampled = resample(df_base, config)
        config.full_df_after_split = df_resampled

        df_s = df_resampled.copy()
        df_s['ratio_group'] = pd.cut(
            df_s[config.weight_col], bins=bins, labels=labels, right=True)
        df_s['stratify_col'] = (
            df_s[config.target_col].astype(str) + "_" + df_s['ratio_group'].astype(str))

        train, test = train_test_split(
            df_s, test_size=effective_test_size,
            stratify=df_s['stratify_col'],
            random_state=config.random_state,
        )

        train = train.drop(columns=['stratify_col', 'ratio_group']).reset_index(drop=True)
        test = test.drop(columns=['stratify_col', 'ratio_group']).reset_index(drop=True)

        _print_split_summary(df_resampled, train, test, config)
        return train, test

    df = df.copy()
    df['ratio_group'] = pd.cut(df[config.weight_col], bins=bins, labels=labels, right=True)
    df['stratify_col'] = df[config.target_col].astype(str) + "_" + df['ratio_group'].astype(str)

    train, test = train_test_split(
        df, test_size=effective_test_size,
        stratify=df['stratify_col'],
        random_state=config.random_state,
    )

    train = train.drop(columns=['stratify_col', 'ratio_group']).reset_index(drop=True)
    test = test.drop(columns=['stratify_col', 'ratio_group']).reset_index(drop=True)

    df_clean = df.drop(columns=['stratify_col', 'ratio_group'])
    if config.resample_strategy != "none":
        print("\nPre-resample split:")
    _print_split_summary(df_clean, train, test, config)

    train = resample(train, config)

    if config.resample_strategy != "none":
        print("\nPost-resample split:")
        _print_split_summary(df_clean, train, test, config)

    return train, test


_MIN_GROUP_COUNT = 10


def _print_split_summary(df: pd.DataFrame, train: pd.DataFrame,
                         test: pd.DataFrame, config: PipelineConfig) -> None:
    """Print impact group distribution across the whole, train, and test sets."""
    wc = config.weight_col
    groups = ["Unknown", "Minimal Impact", "Moderate Impact", "Large Impact"]

    lo, hi = config.impact_threshold_low, config.impact_threshold_high

    def _vectorized_labels(series: pd.Series) -> pd.Series:
        conditions = [series.isna(), series < lo, series < hi]
        choices = ["Unknown", "Minimal Impact", "Moderate Impact"]
        return pd.Series(
            np.select(conditions, choices, default="Large Impact"),
            index=series.index,
        )

    whole_labels = _vectorized_labels(df[wc])
    train_labels = _vectorized_labels(train[wc])
    test_labels = _vectorized_labels(test[wc])

    print(f"\nTrain: {len(train)} records, Test: {len(test)} records")
    print(f"\nImpact group distribution "
          f"(thresholds: {config.impact_threshold_low}, "
          f"{config.impact_threshold_high}):")
    header = f"  {'Group':<20} {'Whole':>8} {'Train':>8} {'Test':>8}"
    print(header)
    print(f"  {'─' * len(header.strip())}")

    empty_groups = []
    sparse_groups = []
    split_warnings: list[str] = []

    for group in groups:
        w = int((whole_labels == group).sum())
        tr = int((train_labels == group).sum())
        te = int((test_labels == group).sum())
        w_pct = w / len(df) * 100 if len(df) else 0
        tr_pct = tr / len(train) * 100 if len(train) else 0
        te_pct = te / len(test) * 100 if len(test) else 0
        print(f"  {group:<20} {w:>5} ({w_pct:4.1f}%) {tr:>5} ({tr_pct:4.1f}%) "
              f"{te:>5} ({te_pct:4.1f}%)")
        if group != "Unknown":
            if w == 0:
                empty_groups.append(group)
            elif w < _MIN_GROUP_COUNT:
                sparse_groups.append((group, w))
        if group != "Unknown" and w > 0 and tr == 0:
            split_warnings.append(f"{group} is empty in train split")
        if group != "Unknown" and w > 0 and te == 0:
            split_warnings.append(f"{group} is empty in test split")

    if empty_groups:
        print(f"\n  WARNING: No records in group(s): {', '.join(empty_groups)}.")
        print(f"  The impact thresholds ({config.impact_threshold_low}, "
              f"{config.impact_threshold_high}) may not suit this dataset's "
              f"{wc} range ({df[wc].min():.2f} – {df[wc].max():.2f}).")
    if sparse_groups:
        detail = ", ".join(f"{g} ({n})" for g, n in sparse_groups)
        print(f"\n  WARNING: Very few records in group(s): {detail}. "
              f"Per-group accuracy may be unreliable.")
    if split_warnings:
        for sw in split_warnings:
            print(f"\n  WARNING: {sw} — per-group metrics will be unreliable.")


def prepare_features(df: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y from a DataFrame.

    Args:
        df: DataFrame (train, test, or whole).
        config: PipelineConfig with feature_cols and target_col set.

    Returns:
        tuple: (X, y) as pandas DataFrame/Series.
    """
    X = df[config.feature_cols]
    y = df[config.target_col]
    return X, y
