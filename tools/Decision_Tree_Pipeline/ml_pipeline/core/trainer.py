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

"""Grid search training loop with deduplication and pruning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from .utils import transform_weights, ModelResult

from .metrics import calculate_mismatch_metric, calculate_geo_mean

if TYPE_CHECKING:
    import pandas as pd
    from .config import PipelineConfig
from .tree_utils import get_tree_fingerprint, simplify_redundant_branches, count_reachable_nodes


def run_grid_search(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, config: PipelineConfig) -> tuple[list[ModelResult], dict[str, DecisionTreeClassifier]]:
    """Run the full grid search training loop.

    Iterates over all parameter combinations, performs cross-validation,
    cost-complexity pruning, redundant branch simplification, and
    tree deduplication.

    Args:
        df: Full DataFrame (used for whole-dataset evaluation).
        train_df: Training split DataFrame.
        test_df: Test split DataFrame.
        config: PipelineConfig instance.

    Returns:
        tuple: (results_list, models_dict)
            results_list: list of tuples with scores for each unique tree
            models_dict: dict {index_key: fitted DecisionTreeClassifier}
    """
    from .data_loader import prepare_features

    X, y = prepare_features(train_df, config)
    X_test, y_test = prepare_features(test_df, config)
    X_whole, y_whole = prepare_features(df, config)

    weights_train = transform_weights(train_df[config.weight_col], config)
    weights_whole = transform_weights(df[config.weight_col], config)

    if config.train_on_whole:
        X_fit, y_fit, weights_fit, fit_df = X_whole, y_whole, weights_whole, df
        print("Training mode: WHOLE dataset (train + test combined)")
    else:
        X_fit, y_fit, weights_fit, fit_df = X, y, weights_train, train_df
        print("Training mode: TRAIN split only (test is fully unseen)")
    print()

    skf = None
    if config.run_cv:
        skf = StratifiedKFold(
            n_splits=config.cv_folds, shuffle=True, random_state=config.random_state
        )

    all_scores = []
    models_dict = {}

    seen_trees = set()
    total_fitted = 0
    total_skipped = 0
    total_nodes_simplified = 0

    for idx, params in enumerate(ParameterGrid(config.param_grid)):
        avg_score = None
        std_score = None

        if config.run_cv:
            cv_scores = []
            for train_idx, val_idx in skf.split(X_fit, y_fit):
                X_train_cv, X_val_cv = X_fit.iloc[train_idx], X_fit.iloc[val_idx]
                y_train_cv, y_val_cv = y_fit.iloc[train_idx], y_fit.iloc[val_idx]
                val_df = fit_df.iloc[val_idx].copy().reset_index(drop=True)
                weights_cv = weights_fit.iloc[train_idx]

                model = DecisionTreeClassifier(**params, random_state=config.random_state)
                model.fit(X_train_cv, y_train_cv, sample_weight=weights_cv)

                y_pred = model.predict(X_val_cv)
                _, _, w_acc_fold = calculate_mismatch_metric(y_pred, val_df, config)
                cv_scores.append(w_acc_fold)

            avg_score = round(np.mean(cv_scores), 3)
            std_score = round(np.std(cv_scores), 3)
            if config.print_params:
                print(f"CV W_Acc: {avg_score}%, CV Std: {std_score}%")

        if config.print_params:
            print(f"Pruned models for params: {params}")

        model = DecisionTreeClassifier(**params, random_state=config.random_state)
        model.fit(X_fit, y_fit, sample_weight=weights_fit)
        path = model.cost_complexity_pruning_path(X_fit, y_fit, sample_weight=weights_fit)
        ccp_alphas = path.ccp_alphas

        pruned_count = 0
        for alpha in ccp_alphas:
            dt_pruned = DecisionTreeClassifier(**params, ccp_alpha=alpha, random_state=config.random_state)
            dt_pruned.fit(X_fit, y_fit, sample_weight=weights_fit)
            total_fitted += 1

            if config.post_prune_redundant:
                total_nodes_simplified += simplify_redundant_branches(dt_pruned)

            fingerprint = get_tree_fingerprint(dt_pruned)
            if fingerprint in seen_trees:
                total_skipped += 1
                continue
            seen_trees.add(fingerprint)
            pruned_count += 1

            y_whole_pred = dt_pruned.predict(X_whole)

            mispred_whole, score, w_acc = calculate_mismatch_metric(
                y_whole_pred, df, config)

            mispred_train = score_train = w_acc_train = None
            mispred_test = score_test = w_acc_test = None

            if not config.train_on_whole:
                y_train_pred = dt_pruned.predict(X)
                y_test_pred = dt_pruned.predict(X_test)

                mispred_test, score_test, w_acc_test = calculate_mismatch_metric(
                    y_test_pred, test_df, config)
                mispred_train, score_train, w_acc_train = calculate_mismatch_metric(
                    y_train_pred, train_df, config)

            max_depth = dt_pruned.get_depth()
            total_nodes = count_reachable_nodes(dt_pruned)
            indx = f"{idx}_{alpha}"

            geo_mean = None
            if config.has_baseline:
                geo_mean = calculate_geo_mean(y_whole_pred, df, config)

            all_scores.append(ModelResult(
                index_key=indx, params=params,
                score_whole=score, score_train=score_train, score_test=score_test,
                max_depth=max_depth, total_nodes=total_nodes,
                cv_avg=avg_score, cv_std=std_score, geo_mean=geo_mean,
                mispred_whole=mispred_whole, mispred_train=mispred_train,
                mispred_test=mispred_test,
                w_acc_whole=w_acc, w_acc_train=w_acc_train, w_acc_test=w_acc_test,
            ))

            geo_str = f", GeoMean: {geo_mean}" if config.has_baseline else ""
            params_str = f", Params: {params}" if config.print_params else ""
            if config.train_on_whole:
                print(
                    f"  {indx} | "
                    f"W_Acc: {w_acc}%{geo_str}, "
                    f"Depth: {max_depth}, Nodes: {total_nodes}, "
                    f"Mispred: {mispred_whole}"
                    f"{params_str}"
                )
            else:
                print(
                    f"  {indx} | "
                    f"W_Acc(Tr/Te/W): {w_acc_train}%/{w_acc_test}%/{w_acc}%{geo_str}, "
                    f"Depth: {max_depth}, Nodes: {total_nodes}, "
                    f"Mispred: {mispred_whole}/{mispred_train}/{mispred_test}"
                    f"{params_str}"
                )

            models_dict[indx] = dt_pruned

        if pruned_count == 0:
            print("  (all pruned variants were duplicates of previously seen trees)")
        print()

    dup_pct = 100 * total_skipped / max(total_fitted, 1)

    print("=" * 70)
    print("Deduplication summary:")
    print(f"  Total trees fitted     : {total_fitted}")
    print(f"  Duplicate trees skipped: {total_skipped} ({dup_pct:.1f}%)")
    print(f"  Unique trees evaluated : {total_fitted - total_skipped}")
    print(f"  Unique results stored  : {len(all_scores)}")  
    if config.post_prune_redundant:
        print(f"  Redundant nodes removed: {total_nodes_simplified} "
              f"(across all {total_fitted} trees)")
    print("=" * 70)
    print()

    return all_scores, models_dict
