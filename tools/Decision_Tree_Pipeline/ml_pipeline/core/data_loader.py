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

import pandas as pd
from sklearn.model_selection import train_test_split

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


def split_data(df: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 70/30 train/test split.

    Stratification is performed on a composite key of both the Algo label
    and a binned Ratio group, ensuring both splits have similar distributions.

    Args:
        df: Full DataFrame.
        config: PipelineConfig.

    Returns:
        tuple: (train_df, test_df) with reset indices.
    """
    bins = [0, _RATIO_STRATIFY_BOUNDARY, float('inf')]
    labels = [1, 2]
    df = df.copy()
    df['ratio_group'] = pd.cut(df[config.weight_col], bins=bins, labels=labels, right=True)
    df['stratify_col'] = df[config.target_col].astype(str) + "_" + df['ratio_group'].astype(str)

    train, test = train_test_split(
        df, test_size=config.test_size,
        stratify=df['stratify_col'],
        random_state=config.random_state,
    )

    train = train.drop(columns=['stratify_col', 'ratio_group']).reset_index(drop=True)
    test = test.drop(columns=['stratify_col', 'ratio_group']).reset_index(drop=True)

    return train, test


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
