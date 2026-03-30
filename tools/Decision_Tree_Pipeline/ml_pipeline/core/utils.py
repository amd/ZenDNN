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

"""Small shared helpers for the ML pipeline."""

from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import PipelineConfig


# Each trained model's evaluation is stored as a ModelResult rather than a plain
# tuple.  A NamedTuple gives every field a descriptive name (e.g. entry.geo_mean
# instead of entry[9]), which makes consumer code in results.py, history.py, and
# run_pipeline.py self-documenting and less prone to index-mismatch bugs when
# fields are added or reordered.  It is fully backward-compatible with tuple
# unpacking, so existing positional access still works where convenient.
ModelResult = namedtuple("ModelResult", [
    "index_key",      # str  — unique key, e.g. "506_0.012"
    "params",         # dict — hyperparameters used
    "score_whole",    # float — mismatch penalty (whole dataset)
    "score_train",    # float|None — mismatch penalty (train split)
    "score_test",     # float|None — mismatch penalty (test split)
    "max_depth",      # int  — tree depth
    "total_nodes",    # int  — reachable node count
    "cv_avg",         # float|None — CV weighted accuracy mean
    "cv_std",         # float|None — CV weighted accuracy std
    "geo_mean",       # float|None — geometric mean (baseline only)
    "mispred_whole",  # int  — misprediction count (whole)
    "mispred_train",  # int|None — misprediction count (train)
    "mispred_test",   # int|None — misprediction count (test)
    "w_acc_whole",    # float — weighted accuracy % (whole)
    "w_acc_train",    # float|None — weighted accuracy % (train)
    "w_acc_test",     # float|None — weighted accuracy % (test)
])


def transform_weights(weights: pd.Series, config: PipelineConfig) -> pd.Series:
    """Apply the chosen transformation to raw Ratio values for sample_weight.

    Always uses raw Ratio from the DataFrame; the transform only affects
    training sample_weight, not evaluation metrics.

    Args:
        weights: pandas Series of raw Ratio values.
        config: PipelineConfig instance.

    Returns:
        pandas Series of transformed weights (always non-negative).
    """
    method = config.weight_transform

    if weights.empty:
        return weights.copy()

    valid_methods = {'raw', 'log+1', 'sqrt', 'minmax', 'rank', 'percentile_clip'}
    if method not in valid_methods:
        print(f"WARNING: Unknown weight_transform '{method}', using raw weights.")
        return weights.copy()

    if method == 'log+1':
        return np.log1p(weights.clip(lower=0))
    elif method == 'sqrt':
        return np.sqrt(weights.clip(lower=0))
    elif method == 'minmax':
        w_min, w_max = weights.min(), weights.max()
        if w_max == w_min:
            return pd.Series(np.full(len(weights), config.weight_minmax_low),
                             index=weights.index)
        normalized = (weights - w_min) / (w_max - w_min)
        return config.weight_minmax_low + normalized * (config.weight_minmax_high - config.weight_minmax_low)
    elif method == 'rank':
        return weights.rank(method='average')
    elif method == 'percentile_clip':
        cap = np.percentile(weights, config.weight_clip_percentile)
        return weights.clip(upper=cap)
    else:
        return weights.copy()


def harmonic_mean(a: float | None, b: float | None) -> float:
    """Harmonic mean of two values. Returns 0 if either is zero or None.

    Defined here (rather than in results.py) so that results, history, and
    run_pipeline can all import a single shared implementation.
    """
    if a is None or b is None or a + b == 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def impact_group_label(ratio: float, config: PipelineConfig) -> str:
    """Assign an impact group string based on Ratio thresholds."""
    if pd.isna(ratio):
        return 'Unknown'
    if ratio < config.impact_threshold_low:
        return 'Minimal Impact'
    elif ratio < config.impact_threshold_high:
        return 'Moderate Impact'
    else:
        return 'Large Impact'
