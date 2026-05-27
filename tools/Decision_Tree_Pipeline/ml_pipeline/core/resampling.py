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

"""Class-imbalance detection and resampling strategies.

Normally resampling runs **after** the train/test split and **only on the
training split**, so the test set stays untouched. If ``train_on_whole`` is
True and resampling is enabled, ``split_data`` applies resampling to the
**full** dataset first (see ``data_loader.split_data``).
Three strategies are available:

- **undersample**: Remove majority-class records with low Ratio (near-equal
  performance), optionally capping the majority count.
- **oversample**: Duplicate minority-class records proportional to their Ratio
  so high-impact records get more copies.
- **hybrid**: Apply undersample first, then oversample the minority to close
  the remaining gap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import PipelineConfig

_VALID_STRATEGIES = {"none", "undersample", "oversample", "hybrid"}


def detect_imbalance(df: pd.DataFrame, config: PipelineConfig) -> dict:
    """Check class distribution and warn if imbalanced.

    Returns:
        dict with keys: counts (dict), minority_class, majority_class,
        minority_frac, is_imbalanced.
    """
    counts = df[config.target_col].value_counts().to_dict()
    total_valid = sum(counts.values())

    if not counts:
        print("\nWARNING: Target column has no valid classes — skipping imbalance check.")
        return {"counts": {}, "minority_class": None, "majority_class": None,
                "minority_frac": 0, "is_imbalanced": False}

    n_nan = len(df) - total_valid
    if n_nan > 0:
        print(f"\nWARNING: {n_nan} NaN value(s) in target column '{config.target_col}' "
              f"excluded from imbalance check.")

    minority_class = min(counts, key=counts.get)
    majority_class = max(counts, key=counts.get)
    minority_frac = counts[minority_class] / total_valid if total_valid > 0 else 0

    info = {
        "counts": counts,
        "minority_class": minority_class,
        "majority_class": majority_class,
        "minority_frac": minority_frac,
        "is_imbalanced": minority_frac < config.imbalance_warn_threshold,
    }

    if info["is_imbalanced"]:
        dist = ", ".join(
            f"{config.target_col} {k}: {v} ({v/total_valid*100:.1f}%)" for k, v in sorted(counts.items())
        )
        print(f"\nWARNING: Class imbalance detected — {dist}")
        print(f"  Minority class ({config.target_col} {minority_class}) is "
              f"{minority_frac*100:.1f}% of total "
              f"(threshold: {config.imbalance_warn_threshold*100:.0f}%).")
        if config.resample_strategy == "none":
            print("  Consider setting config.resample_strategy to "
                  "'undersample', 'oversample', or 'hybrid'.")

    return info


def undersample(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Remove low-Ratio majority-class records to reduce imbalance.

    Note: designed for binary or "one minority vs rest" scenarios.
    Only the global majority class is undersampled; intermediate classes
    in a multi-class setting are left unchanged.

    Steps:
        1. Identify the majority class.
        2. Drop majority records where Ratio <= undersample_ratio_ceil.
        3. If undersample_max_factor is set and the majority is still too
           large, randomly sample down to max_factor * minority_count.
    """
    counts = df[config.target_col].value_counts()
    majority_cls = counts.idxmax()
    minority_cls = counts.idxmin()
    minority_count = int(counts[minority_cls])

    majority_mask = df[config.target_col] == majority_cls
    majority_df = df[majority_mask]
    minority_df = df[~majority_mask]

    ceil = config.undersample_ratio_ceil
    keep_mask = majority_df[config.weight_col] > ceil
    kept = majority_df[keep_mask]
    dropped = len(majority_df) - len(kept)

    if len(kept) == 0:
        print(f"\nWARNING: undersample would remove ALL majority-class records "
              f"(every record has {config.weight_col} <= {ceil}). "
              f"Skipping undersampling to preserve at least two classes.")
        _print_distribution(df, config, "Undersample skipped")
        return df

    print(f"\n[Undersample] Majority class = {config.target_col} {majority_cls}")
    print(f"  Removed {dropped} records with {config.weight_col} <= {ceil}")

    if config.undersample_max_factor is not None:
        max_count = max(1, int(config.undersample_max_factor * minority_count))
        if len(kept) > max_count:
            before = len(kept)
            kept = kept.sample(n=max_count, random_state=config.random_state)
            print(f"  Capped majority from {before} to {max_count} "
                  f"({config.undersample_max_factor}x minority)")

    result = pd.concat([minority_df, kept], ignore_index=True)
    _print_distribution(result, config, "After undersample")
    return result


def oversample(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Duplicate minority-class records weighted by Ratio to reduce imbalance.

    Note: designed for binary or "one minority vs rest" scenarios.
    Only the global minority class is oversampled; intermediate classes
    in a multi-class setting are left unchanged.

    Higher-Ratio minority records receive more copies, ensuring the tree
    sees the most impactful cases more often.
    """
    counts = df[config.target_col].value_counts()
    majority_cls = counts.idxmax()
    minority_cls = counts.idxmin()
    majority_count = int(counts[majority_cls])
    minority_count = int(counts[minority_cls])

    target_count = int(majority_count * config.oversample_target_ratio)
    need = target_count - minority_count

    if need <= 0:
        print(f"\n[Oversample] Minority already meets target ratio — no action.")
        return df

    minority_df = df[df[config.target_col] == minority_cls]

    # Weight each minority record by its Ratio for proportional duplication.
    # Normalize so weights sum to 1 (probability distribution for sampling).
    ratios = minority_df[config.weight_col].values.astype(float)
    ratios = np.where(np.isfinite(ratios), ratios, 1.0)
    ratios = np.clip(ratios, 1.0, None)
    total = ratios.sum()
    if not np.isfinite(total) or total == 0:
        probs = np.ones(len(ratios)) / len(ratios)
    else:
        probs = ratios / total

    rng = np.random.default_rng(config.random_state)
    indices = rng.choice(len(minority_df), size=need, replace=True, p=probs)
    duplicates = minority_df.iloc[indices]

    result = pd.concat([df, duplicates], ignore_index=True)

    print(f"\n[Oversample] Minority class = {config.target_col} {minority_cls}")
    print(f"  Added {need} duplicated records (target_ratio={config.oversample_target_ratio})")
    _print_distribution(result, config, "After oversample")
    return result


def resample(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Apply the configured resampling strategy.

    Typically called on the training split only (after train/test split)
    to prevent data leakage.  Exception: when ``config.train_on_whole``
    is True, ``split_data()`` calls this on the full dataset since there
    is no separate test set.

    Args:
        df: DataFrame to resample (training split, or full dataset when
            train_on_whole is enabled).
        config: PipelineConfig with resample_strategy set.

    Returns:
        Resampled DataFrame (or original if strategy is "none").
    """
    strategy = config.resample_strategy
    if strategy not in _VALID_STRATEGIES:
        print(f"WARNING: Unknown resample_strategy '{strategy}', skipping. "
              f"Valid: {sorted(_VALID_STRATEGIES)}")
        return df

    if strategy == "none":
        return df

    print(f"\nResampling strategy: {strategy}")
    _print_distribution(df, config, "Before resampling")

    if strategy == "undersample":
        return undersample(df, config)
    elif strategy == "oversample":
        return oversample(df, config)
    elif strategy == "hybrid":
        df = undersample(df, config)
        return oversample(df, config)

    return df


def _print_distribution(df: pd.DataFrame, config: PipelineConfig,
                        label: str) -> None:
    """Print class distribution with a label."""
    counts = df[config.target_col].value_counts().sort_index()
    total = len(df)
    if total == 0:
        print(f"  {label}: 0 records")
        return
    parts = [f"{config.target_col} {k}: {v} ({v/total*100:.1f}%)" for k, v in counts.items()]
    print(f"  {label}: {total} records — {', '.join(parts)}")
