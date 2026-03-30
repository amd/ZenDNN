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

"""Results sorting, display, and per-impact-group accuracy analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prettytable import PrettyTable

from .utils import impact_group_label, harmonic_mean
from .data_loader import prepare_features

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd
    from .config import PipelineConfig
    from .utils import ModelResult


def _set_min_widths_for_groups(table: PrettyTable, column_groups: list[str]) -> None:
    """Ensure columns are wide enough for their super-header group names.

    PrettyTable's min_width sets the minimum content width; padding (1 space
    each side) is added on top.  For a group spanning N columns, the available
    super-header space is sum(content_widths) + 3*(N-1) for separators.

    Uses PrettyTable's internal ``_min_width`` dict.  If the attribute is
    absent in a future library version the adjustment is silently skipped
    and the table still renders — just with potentially misaligned
    super-headers.
    """
    try:
        min_width = table._min_width
    except AttributeError:
        return

    fields = table.field_names
    padding = table.padding_width * 2
    i = 0
    while i < len(column_groups):
        grp = column_groups[i]
        if not grp:
            i += 1
            continue
        span_start = i
        while i < len(column_groups) and column_groups[i] == grp:
            i += 1
        span_count = i - span_start
        sep_width = 3 * (span_count - 1)
        needed = len(grp) - sep_width - padding * (span_count - 1)
        if span_count == 1:
            needed = len(grp)
        per_col = max(0, (needed + span_count - 1) // span_count)
        for j in range(span_start, i):
            cur = min_width.get(fields[j], 0)
            if per_col > cur:
                min_width[fields[j]] = per_col


def _table_with_super_headers(table: PrettyTable, column_groups: list[str]) -> str:
    """Render a PrettyTable with a super-header row above the column headers.

    Merges top-border segments for columns that share a group, then centers
    the group name over the merged span.

    Args:
        table: Populated PrettyTable instance.
        column_groups: List of group names, one per column.
            Empty string means no group for that column.

    Returns:
        str: Formatted table string with super-headers inserted.
    """
    _set_min_widths_for_groups(table, column_groups)

    table_str = table.get_string()
    lines = table_str.split('\n')
    border = lines[0]
    plus_pos = [i for i, c in enumerate(border) if c == '+']
    num_cols = len(column_groups)

    merged_border = list(border)
    i = 0
    while i < num_cols:
        grp = column_groups[i]
        if not grp:
            i += 1
            continue
        span_start = i
        while i < num_cols and column_groups[i] == grp:
            i += 1
        for j in range(span_start + 1, i):
            merged_border[plus_pos[j]] = '-'

    super_chars = [' '] * len(border)
    super_chars[0] = '|'
    super_chars[-1] = '|'

    i = 0
    while i < num_cols:
        grp = column_groups[i]
        super_chars[plus_pos[i]] = '|'
        if not grp:
            i += 1
        else:
            span_start = i
            while i < num_cols and column_groups[i] == grp:
                i += 1
            content_start = plus_pos[span_start] + 1
            content_end = plus_pos[i]
            centered = grp.center(content_end - content_start)
            for j, c in enumerate(centered):
                if content_start + j < len(super_chars):
                    super_chars[content_start + j] = c

    return (''.join(merged_border) + '\n'
            + ''.join(super_chars) + '\n'
            + table_str)


def _sort_key(entry: ModelResult, config: PipelineConfig) -> float:
    """Return the numeric sort key for a single result entry."""
    if config.has_baseline:
        return entry.geo_mean if entry.geo_mean is not None else 0
    if config.train_on_whole:
        return entry.w_acc_whole
    return harmonic_mean(entry.w_acc_train, entry.w_acc_test)


def get_sort_description(config: PipelineConfig) -> str:
    """Return a human-readable description of the active sort criterion."""
    if config.has_baseline:
        return "GeoMean (descending) — Native_time baseline available"
    if config.train_on_whole:
        return "Whole-dataset Weighted Accuracy (descending) — training on whole dataset"
    return "Harmonic Mean of Train/Test W_Acc (descending) — penalizes overfitting"


def sort_results(results_list: list[ModelResult], config: PipelineConfig) -> list[ModelResult]:
    """Sort results using the appropriate metric for the current configuration.

    - has_baseline=True: GeoMean descending (higher = better algo selection).
    - train_on_whole=True (no baseline): Whole W_Acc descending.
    - train_on_whole=False (no baseline): Harmonic mean of train and test
      W_Acc descending — naturally penalizes large train/test gaps.

    Args:
        results_list: list of result tuples from trainer.
        config: PipelineConfig.

    Returns:
        list: Sorted results.
    """
    return sorted(results_list, key=lambda entry: _sort_key(entry, config), reverse=True)


def display_results(results_list: list[ModelResult], config: PipelineConfig, top_n: int | None = None) -> None:
    """Print results as a formatted table using PrettyTable.

    Args:
        results_list: sorted list of result tuples.
        config: PipelineConfig.
        top_n: If set, only show top N results.
    """
    sort_desc = get_sort_description(config)
    print(f"Sorted by: {sort_desc}\n")

    display = results_list[:top_n] if top_n else results_list
    if not display:
        print("  (no results)")
        return

    show_splits = not config.train_on_whole
    show_hmean = not config.train_on_whole
    show_geo = config.has_baseline
    show_cv = config.run_cv
    show_params = config.print_params

    headers = ["Rank", "Index"]
    groups = ["", ""]

    if show_splits:
        headers += ["W Train", "W Test"]
        groups += ["Weighted Accuracy", "Weighted Accuracy"]
    headers.append("W Whole" if show_splits else "Whole")
    groups.append("Weighted Accuracy")
    if show_hmean:
        headers.append("W HMean")
        groups.append("Weighted Accuracy")
    if show_geo:
        headers.append("GeoMean")
        groups.append("")
    headers += ["Depth", "Nodes"]
    groups += ["", ""]
    if show_splits:
        headers += ["M Whole", "M Train", "M Test"]
        groups += ["Mispredictions", "Mispredictions", "Mispredictions"]
    else:
        headers.append("Total")
        groups.append("Mispredictions")
    if show_cv:
        headers += ["CV Avg", "CV Std"]
        groups += ["Cross Validation", "Cross Validation"]
    if show_params:
        headers.append("Parameters")
        groups.append("")

    table = PrettyTable()
    table.field_names = headers
    table.align = "r"
    if show_params:
        table.align["Parameters"] = "l"

    for rank, entry in enumerate(display, 1):
        row = [rank, entry.index_key]
        if show_splits:
            row += [f"{entry.w_acc_train}%", f"{entry.w_acc_test}%"]
        row.append(f"{entry.w_acc_whole}%")
        if show_hmean:
            row.append(f"{harmonic_mean(entry.w_acc_train, entry.w_acc_test):.2f}%")
        if show_geo:
            row.append(f"{entry.geo_mean}" if entry.geo_mean is not None else "—")
        row += [entry.max_depth, entry.total_nodes]
        if show_splits:
            row += [entry.mispred_whole, entry.mispred_train, entry.mispred_test]
        else:
            row.append(entry.mispred_whole)
        if show_cv and entry.cv_avg is not None:
            row += [f"{entry.cv_avg:.2f}%", f"{entry.cv_std:.2f}%"]
        elif show_cv:
            row += ["—", "—"]
        if show_params:
            row.append(entry.params)
        table.add_row(row)

    has_groups = any(groups)
    if has_groups:
        print(_table_with_super_headers(table, groups))
    else:
        print(table)


def display_impact_groups(model: DecisionTreeClassifier, df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, config: PipelineConfig, selected_key: str | None = None) -> None:
    """Print per-impact-group accuracy breakdown for train, test, and whole splits.

    Args:
        model: Fitted DecisionTreeClassifier to evaluate.
        df: Full DataFrame.
        train_df: Training split.
        test_df: Test split.
        config: PipelineConfig.
        selected_key: Optional model key string for display header.
    """
    if selected_key:
        print(f"Selected model: {selected_key}")
    print(
        f"Impact thresholds: Minimal (Ratio < {config.impact_threshold_low}), "
        f"Moderate ({config.impact_threshold_low}–{config.impact_threshold_high}), "
        f"Large (>= {config.impact_threshold_high})"
    )

    X_train, y_train = prepare_features(train_df, config)
    X_test, y_test = prepare_features(test_df, config)
    X_whole, y_whole = prepare_features(df, config)

    _print_group_accuracy("Train", train_df, X_train, model, config)
    _print_group_accuracy("Test", test_df, X_test, model, config)
    _print_group_accuracy("Whole", df, X_whole, model, config)


def _print_group_accuracy(split_name: str, split_df: pd.DataFrame, X_split: pd.DataFrame, model: DecisionTreeClassifier, config: PipelineConfig) -> None:
    """Print per-group accuracy for a single data split."""
    y_pred = model.predict(X_split)
    split_df = split_df.copy()
    split_df['Predict'] = y_pred
    split_df['Correct'] = split_df['Predict'] == split_df[config.target_col]
    split_df['Impact_Group'] = split_df[config.weight_col].apply(
        lambda r: impact_group_label(r, config)
    )

    group_order = ['Minimal Impact', 'Moderate Impact', 'Large Impact']

    print(f"\n  {split_name} Split  (total records: {len(split_df)})")

    table = PrettyTable()
    table.field_names = ["Group", "Count", "Correct", "Wrong", "Accuracy", "W_Acc"]
    table.align = "r"
    table.align["Group"] = "l"

    for group in group_order:
        grp = split_df[split_df['Impact_Group'] == group]
        if len(grp) == 0:
            table.add_row([group, "—", "—", "—", "—", "—"])
            continue
        n = len(grp)
        correct = int(grp['Correct'].sum())
        wrong = n - correct
        acc = round(correct / n * 100, 1)
        total_w = grp[config.weight_col].sum()
        correct_w = grp.loc[grp['Correct'], config.weight_col].sum()
        w_acc = round(correct_w / total_w * 100, 1) if total_w > 0 else 100.0
        table.add_row([group, n, correct, wrong, f"{acc}%", f"{w_acc}%"])

    total = len(split_df)
    total_correct = int(split_df['Correct'].sum())
    total_wrong = total - total_correct
    overall_acc = round(total_correct / total * 100, 1) if total > 0 else 0.0
    total_w = split_df[config.weight_col].sum()
    correct_w = split_df.loc[split_df['Correct'], config.weight_col].sum()
    overall_w_acc = round(correct_w / total_w * 100, 1) if total_w > 0 else 100.0
    table.add_row(["OVERALL", total, total_correct, total_wrong,
                   f"{overall_acc}%", f"{overall_w_acc}%"])

    print(table)
