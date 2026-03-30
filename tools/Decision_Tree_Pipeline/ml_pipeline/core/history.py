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

"""In-memory run history for tracking experiments across notebook re-runs."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from prettytable import PrettyTable

# _sort_key is internal to ml_pipeline.core — imported here because history
# needs identical ranking logic to results for global-best selection.
from .results import _sort_key, get_sort_description
from .utils import harmonic_mean

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeClassifier
    from .config import PipelineConfig
    from .utils import ModelResult


def _config_snapshot(config: PipelineConfig) -> dict[str, Any]:
    """Capture the key config values that affect training."""
    depths = config.param_grid.get('max_depth', [])
    depth_str = f"d={min(depths)}-{max(depths)}" if depths else "d=?"
    raw_features = "+".join(config.all_feature_cols) if config.all_feature_cols else "auto"
    return {
        "weight_transform": config.weight_transform,
        "raw_features": raw_features,
        "num_features": len(config.feature_cols),
        "exclude_m": config.exclude_m,
        "train_on_whole": config.train_on_whole,
        "has_baseline": config.has_baseline,
        "post_prune_redundant": config.post_prune_redundant,
        "run_cv": config.run_cv,
        "feature_engineering": config.feature_engineering,
        "depth_range": depth_str,
        "param_grid_size": _grid_size(config.param_grid),
    }


def _grid_size(param_grid: dict[str, list]) -> int:
    total = 1
    for vals in param_grid.values():
        total *= len(vals)
    return total


def _config_summary_str(snap: dict[str, Any]) -> str:
    """One-line summary of a config snapshot for the table."""
    fe = snap.get("feature_engineering", "none")
    if fe != "none":
        feat_str = f"fe={fe} ({snap['num_features']})"
    else:
        feat_str = snap["raw_features"]
    parts = [snap["weight_transform"], feat_str, snap["depth_range"]]
    if snap["train_on_whole"]:
        parts.append("whole")
    if snap["exclude_m"]:
        parts.append("no-M")
    return ", ".join(parts)


def _run_metric_type(snap: dict[str, Any]) -> str:
    """Classify which sort metric a run used based on its config snapshot."""
    if snap["has_baseline"]:
        return "geomean"
    if snap["train_on_whole"]:
        return "w_acc"
    return "hmean"


def _sort_metric_label(config: PipelineConfig) -> str:
    """Short label for the active sort metric, used in the table column."""
    if config.has_baseline:
        return "GeoMean"
    if config.train_on_whole:
        return "W_Acc(W)"
    return "HMean"


def _sort_metric_value(entry: ModelResult, config: PipelineConfig) -> str:
    """Formatted string of the active sort metric for a result entry."""
    if config.has_baseline:
        return f"{entry.geo_mean:.4f}" if entry.geo_mean is not None else "—"
    if config.train_on_whole:
        return f"{entry.w_acc_whole}%"
    return f"{harmonic_mean(entry.w_acc_train, entry.w_acc_test):.2f}%"


class RunHistory:
    """Accumulates grid search results across multiple runs within a session.

    Usage:
        history = RunHistory()
        # ... after each grid search + sort ...
        history.add_run(config, sorted_results, models_dict)
        history.show_all_runs(config)
        run_id, key, model = history.get_global_best(config)
    """

    def __init__(self) -> None:
        self._runs = []

    def add_run(self, config: PipelineConfig, sorted_results: list[ModelResult], models_dict: dict[str, DecisionTreeClassifier]) -> int:
        """Store results from a completed grid search run.

        Args:
            config: PipelineConfig used for this run.
            sorted_results: Output of sort_results().
            models_dict: Dict {index_key: fitted DecisionTreeClassifier}.

        Returns:
            int: The run_id assigned to this run.
        """
        run_id = len(self._runs) + 1
        best_entry = sorted_results[0] if sorted_results else None

        self._runs.append({
            "run_id": run_id,
            "timestamp": datetime.now(),
            "config_snapshot": _config_snapshot(config),
            "has_baseline": config.has_baseline,
            "train_on_whole": config.train_on_whole,
            "sorted_results": sorted_results,
            "models_dict": models_dict,
            "best_entry": best_entry,
            "num_models": len(sorted_results),
        })
        metric = _sort_metric_value(best_entry, config) if best_entry else "—"
        label = _sort_metric_label(config)
        print(f"Run {run_id} saved ({len(sorted_results)} models, "
              f"best {label}: {metric})")
        return run_id

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        """Retrieve a stored run by its ID (1-indexed).

        Returns:
            dict with keys: run_id, timestamp, config_snapshot, sorted_results,
            models_dict, best_entry, num_models. Or None if not found.
        """
        if 1 <= run_id <= len(self._runs):
            return self._runs[run_id - 1]
        return None

    def get_model(self, run_id: int, model_key: str) -> DecisionTreeClassifier | None:
        """Retrieve a specific model from a past run.

        Args:
            run_id: 1-indexed run ID.
            model_key: The index key string (e.g. '506_0.012...').

        Returns:
            The fitted DecisionTreeClassifier, or None if not found.
        """
        run = self.get_run(run_id)
        if run is None:
            print(f"Run {run_id} not found (have {len(self._runs)} runs).")
            return None
        model = run["models_dict"].get(model_key)
        if model is None:
            print(f"Model key '{model_key}' not found in Run {run_id}.")
        return model

    def get_global_best(self, config: PipelineConfig) -> tuple[int | None, str | None, DecisionTreeClassifier | None]:
        """Find the best model across all stored runs.

        Uses the same ranking logic as sort_results (GeoMean / W_Acc / HMean)
        based on the provided config.

        Args:
            config: PipelineConfig (used for sort key selection).

        Returns:
            tuple: (run_id, model_key, model) or (None, None, None) if empty.
        """
        if not self._runs:
            return None, None, None

        best_run = None
        best_score = -1

        for run in self._runs:
            entry = run["best_entry"]
            if entry is None:
                continue
            score = _sort_key(entry, config)
            if score > best_score:
                best_score = score
                best_run = run

        if best_run is None:
            return None, None, None

        best_key = best_run["best_entry"].index_key
        best_model = best_run["models_dict"].get(best_key)
        return best_run["run_id"], best_key, best_model

    def show_all_runs(self, config: PipelineConfig) -> None:
        """Print a summary table comparing the best model from each run.

        Shows all metric columns that appear across stored runs.  Runs that
        used a different sort metric get "—" in the inapplicable columns.

        Args:
            config: PipelineConfig — determines which metric is used to
                    identify and label the global best.
        """
        if not self._runs:
            print("No runs recorded yet.")
            return

        metric_types = set()
        for run in self._runs:
            metric_types.add(_run_metric_type(run["config_snapshot"]))

        metric_cols = []
        if "geomean" in metric_types:
            metric_cols.append(("GeoMean", "geomean"))
        if "w_acc" in metric_types:
            metric_cols.append(("W_Acc(W)", "w_acc"))
        if "hmean" in metric_types:
            metric_cols.append(("HMean", "hmean"))

        headers = ["Run", "Time", "Config"]
        headers += [mc[0] for mc in metric_cols]
        headers += ["Depth", "Nodes", "Mispred", "Models"]

        table = PrettyTable()
        table.field_names = headers
        table.align = "r"
        table.align["Config"] = "l"

        scores = []
        for run in self._runs:
            snap = run["config_snapshot"]
            entry = run["best_entry"]

            if entry is None:
                row = [run["run_id"],
                       run["timestamp"].strftime("%H:%M:%S"),
                       _config_summary_str(snap)]
                row += ["—"] * len(metric_cols)
                row += ["—", "—", "—", run["num_models"]]
                table.add_row(row)
                scores.append(-1)
                continue

            row = [run["run_id"],
                   run["timestamp"].strftime("%H:%M:%S"),
                   _config_summary_str(snap)]

            for _, mt in metric_cols:
                if mt == "geomean":
                    row.append(f"{entry.geo_mean:.4f}" if entry.geo_mean is not None else "—")
                elif mt == "w_acc":
                    row.append(f"{entry.w_acc_whole}%")
                else:
                    if snap["train_on_whole"]:
                        row.append("—")
                    else:
                        row.append(f"{harmonic_mean(entry.w_acc_train, entry.w_acc_test):.2f}%")

            row += [entry.max_depth, entry.total_nodes, entry.mispred_whole,
                    run["num_models"]]
            table.add_row(row)
            scores.append(_sort_key(entry, config))

        print(f"Ranked by: {get_sort_description(config)}\n")
        print(table)

        if scores:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            if scores[best_idx] >= 0:
                best_run = self._runs[best_idx]
                sort_label = _sort_metric_label(config)
                best_val = _sort_metric_value(best_run["best_entry"], config)
                print(f"\n  * Global best: Run {best_run['run_id']} "
                      f"({sort_label}: {best_val})")

    def clear(self) -> None:
        """Clear all stored runs."""
        count = len(self._runs)
        self._runs.clear()
        print(f"Cleared {count} run(s) from history.")

    def __len__(self) -> int:
        return len(self._runs)

    def __repr__(self) -> str:
        return f"RunHistory({len(self._runs)} runs)"
