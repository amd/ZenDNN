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

"""Evaluation metrics for the DT pipeline."""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from .config import PipelineConfig


def calculate_mismatch_metric(y_pred: np.ndarray, df: pd.DataFrame, config: PipelineConfig) -> tuple[int, float, float]:
    """Calculate mismatch penalty, misprediction count, and weighted accuracy.

    For each wrong prediction: penalty = (Ratio - 1) * 100.
    Weighted accuracy = sum(Ratio[correct]) / sum(Ratio[all]) * 100.

    Always uses RAW Ratio values regardless of weight transform.

    Args:
        y_pred: Predicted labels.
        df: DataFrame with target and weight columns.
        config: PipelineConfig.

    Returns:
        tuple: (num_mispredictions, total_penalty, weighted_accuracy_pct)
    """
    y_true = df[config.target_col].values
    ratios = df[config.weight_col].values

    correct = np.asarray(y_pred) == y_true
    num_mispredictions = int((~correct).sum())

    penalties = np.where(correct, 0.0, (ratios - 1) * 100)
    total_penalty = round(float(penalties.sum()), 2)

    total_weight = ratios.sum()
    correct_weight = ratios[correct].sum()
    weighted_acc = round(correct_weight / total_weight * 100, 2) if total_weight > 0 else 100.0

    return num_mispredictions, total_penalty, weighted_acc


def calculate_geo_mean(y_pred: np.ndarray, df: pd.DataFrame, config: PipelineConfig) -> float:
    """Geometric mean of performance ratios based on predictions.

    Only meaningful when has_baseline is True (Native_time exists).

    Args:
        y_pred: Predicted labels.
        df: DataFrame with timing columns.
        config: PipelineConfig with timing_cols set.

    Returns:
        float: Geometric mean of ratios. Higher = better algo selection.
    """
    algo_to_col = config.algo_to_col
    baseline = df[config.baseline_col].values
    preds = np.asarray(y_pred, dtype=int)

    denoms = baseline.copy()
    valid_labels = set(algo_to_col.keys())
    for label, col_name in algo_to_col.items():
        mask = preds == label
        denoms[mask] = df[col_name].values[mask]

    invalid = np.array([p not in valid_labels for p in preds])
    if invalid.any():
        print(f"WARNING: {invalid.sum()} invalid prediction indices "
              f"(expected one of {sorted(valid_labels)})")

    denoms = np.where(denoms <= 0, baseline, denoms)
    denoms = np.where(denoms <= 0, 1.0, denoms)
    ratios = baseline / denoms
    ratios = np.where(np.isfinite(ratios) & (ratios > 0), ratios, 1.0)

    ratios_list = ratios.tolist()
    if not ratios_list:
        return 1.0
    return round(statistics.geometric_mean(ratios_list), 3)
