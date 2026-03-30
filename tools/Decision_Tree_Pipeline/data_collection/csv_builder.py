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

"""Build training CSV from profiler data across multiple algo runs."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict

from .profiler_parser import process_txt
from .utils import time_converter, MIN_SELF_CPU_RATIO

ALGO_LABELS = {1: "AOCL", 2: "BRGEMM", 3: "LIBXSMM"}
DEFAULT_RATIO_SCALE = 1


def extract_shapes_with_timing(input_path: str) -> dict[tuple[int, int, int], list[tuple[float, float]]]:
    """Run the profiler extractor and return per-shape (time, impact) data.

    Returns:
        dict: {(M,K,N): [(avg_time_ms, self_cpu_pct), ...]}
    """
    model_data, _ = process_txt(input_path)
    shape_data = defaultdict(list)

    for model_name, data in model_data.items():
        data["CPU(ms)"] = time_converter(data["CPU(ms)"])
        data["Self_CPU(ms)"] = time_converter(data["Self_CPU(ms)"])

        num_rows = min(len(data[col]) for col in data)
        if num_rows < len(data["Name"]):
            print(f"  WARNING: Column length mismatch in {model_name} "
                  f"({num_rows} vs {len(data['Name'])} entries) — using shortest")

        for j in range(num_rows):
            try:
                self_cpu_ms = float(data["Self_CPU(ms)"][j])
                total_cpu_ms = float(data["CPU(ms)"][j])
            except (ValueError, TypeError):
                continue

            # Filter out wrapper ops where most compute happens in child calls.
            # A low Self_CPU / CPU ratio indicates the op is just a wrapper.
            if total_cpu_ms <= 0:
                continue
            if self_cpu_ms / total_cpu_ms < MIN_SELF_CPU_RATIO:
                continue

            parts = data["extracted_shape"][j].split(",")[:3]
            if len(parts) < 3 or "-1" in parts:
                continue

            try:
                m, k, n = int(parts[0]), int(parts[1]), int(parts[2])
                count = int(data["Count"][j])
                self_cpu_pct = float(data["Self_CPU%"][j].rstrip("%"))
            except (ValueError, TypeError, IndexError):
                continue

            if count <= 0:
                continue
            avg_time = round(self_cpu_ms / count, 4)
            shape_data[(m, k, n)].append((avg_time, self_cpu_pct))

    return shape_data


def aggregate_shapes(shape_data: dict[tuple[int, int, int], list[tuple[float, float]]]) -> dict[tuple[int, int, int], tuple[float, float]]:
    """Aggregate per-shape data: geometric mean of time, max impact %.

    Returns:
        dict: {(M,K,N): (geo_mean_time, max_impact_pct)}
    """
    aggregated = {}
    for shape, entries in shape_data.items():
        times = [t for t, _ in entries]
        impacts = [p for _, p in entries]

        valid_times = [t for t in times if t > 0]
        if not valid_times or not impacts:
            continue

        geo_time = round(statistics.geometric_mean(valid_times), 4)
        max_impact = max(impacts)
        aggregated[shape] = (geo_time, max_impact)

    return aggregated


def compute_ratio(impact_pct: float, perf_ratio: float, scale: float) -> float:
    """Compute composite training weight.

    Ratio = max(1, impact_pct * (perf_ratio - 1) * scale)
    """
    return round(max(1.0, impact_pct * (perf_ratio - 1) * scale), 7)


def build_csv(algo_paths: dict[int, str], native_path: str | None, output_path: str, scale: float = DEFAULT_RATIO_SCALE,
              verbose: bool = False) -> int:
    """Full CSV generation pipeline.

    Args:
        algo_paths: dict {algo_id: path} (1-indexed, at least 2 required)
        native_path: path to native baseline logs, or None
        output_path: where to write the output CSV
        scale: Ratio formula scale factor
        verbose: if True, print progress details; warnings/errors always print

    Returns:
        int: number of shapes written, or 0 on failure
    """
    if len(algo_paths) < 2:
        raise ValueError("At least 2 algo paths are required.")

    invalid_ids = [algo_id for algo_id in algo_paths if algo_id not in ALGO_LABELS]
    if invalid_ids:
        print(f"WARNING: Unrecognized algo IDs {invalid_ids} — "
              f"expected {list(ALGO_LABELS.keys())}. Will use generic labels.")

    algo_agg = {}
    for algo_id in sorted(algo_paths):
        path = algo_paths[algo_id]
        label = ALGO_LABELS.get(algo_id, f"Algo_{algo_id}")
        if verbose:
            print(f"[{label}]  Extracting from: {path}")
        raw = extract_shapes_with_timing(path)
        algo_agg[algo_id] = aggregate_shapes(raw)
        if verbose:
            print(f"  -> {len(algo_agg[algo_id])} unique shapes\n")

    native_agg = None
    if native_path:
        if verbose:
            print(f"[Native]  Extracting from: {native_path}")
        raw = extract_shapes_with_timing(native_path)
        native_agg = aggregate_shapes(raw)
        if verbose:
            print(f"  -> {len(native_agg)} unique shapes\n")

    all_shape_sets = [set(agg.keys()) for agg in algo_agg.values()]
    common_shapes = set.intersection(*all_shape_sets)

    for algo_id, agg in algo_agg.items():
        label = ALGO_LABELS.get(algo_id, f"Algo_{algo_id}")
        algo_only = set(agg.keys()) - common_shapes
        if algo_only:
            print(f"WARNING: {len(algo_only)} shape(s) in {label} missing from other algos — dropped.")
            for shape in sorted(algo_only):
                print(f"  dropped: M={shape[0]}, K={shape[1]}, N={shape[2]}")

    if native_agg is not None:
        native_only = set(native_agg.keys()) - common_shapes
        if native_only:
            print(f"WARNING: {len(native_only)} shape(s) in Native missing from algos — dropped.")
            for shape in sorted(native_only):
                print(f"  dropped: M={shape[0]}, K={shape[1]}, N={shape[2]}")
        common_shapes = common_shapes & set(native_agg.keys())

    # Only shapes present in ALL algorithms can be compared, so we intersect.
    # If no shapes are common across all algos, there's nothing to compare.
    if verbose:
        print(f"\nTotal common shapes: {len(common_shapes)}")
    if not common_shapes:
        raise ValueError("No common shapes found across algorithms. Cannot generate CSV.")

    algo_ids_sorted = sorted(algo_paths.keys())
    rows = []

    for shape in common_shapes:
        m, k, n = shape
        row = {"M": m, "K": k, "N": n}

        if native_agg is not None:
            row["Native_time"] = native_agg[shape][0]

        algo_times = {}
        max_impact = 0.0

        for algo_id in algo_ids_sorted:
            geo_time, impact = algo_agg[algo_id][shape]
            label = ALGO_LABELS.get(algo_id, f"Algo_{algo_id}")
            row[f"{label}_time"] = geo_time
            algo_times[algo_id] = geo_time
            max_impact = max(max_impact, impact)

        if native_agg is not None:
            max_impact = max(max_impact, native_agg[shape][1])

        best_algo = min(algo_times, key=algo_times.get)
        row["Algo"] = best_algo

        min_time = min(algo_times.values())
        max_time = max(algo_times.values())
        perf_ratio = max_time / min_time if min_time > 0 else 1.0
        row["Ratio"] = compute_ratio(max_impact, perf_ratio, scale)

        rows.append(row)

    col_order = ["M", "K", "N"]
    if native_agg is not None:
        col_order.append("Native_time")
    for algo_id in algo_ids_sorted:
        label = ALGO_LABELS.get(algo_id, f"Algo_{algo_id}")
        col_order.append(f"{label}_time")
    col_order.extend(["Algo", "Ratio"])

    rows.sort(key=lambda row: row["Ratio"], reverse=True)

    try:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(col_order)
            for row in rows:
                writer.writerow([row[col] for col in col_order])
    except OSError as e:
        raise OSError(f"Cannot write {output_path}: {e}") from e

    print(f"CSV written to: {output_path} ({len(rows)} shapes)")

    if verbose:
        ratios = [r["Ratio"] for r in rows]
        algo_counts = {}
        for r in rows:
            algo_counts[r["Algo"]] = algo_counts.get(r["Algo"], 0) + 1

        print(f"  Ratio range  : {min(ratios):.4f} – {max(ratios):.4f}")
        print(f"  Algo distribution:")
        for algo_id in algo_ids_sorted:
            label = ALGO_LABELS.get(algo_id, f"Algo_{algo_id}")
            count = algo_counts.get(algo_id, 0)
            pct = count / len(rows) * 100 if rows else 0
            print(f"    {algo_id} ({label}): {count} shapes ({pct:.1f}%)")

    return len(rows)
