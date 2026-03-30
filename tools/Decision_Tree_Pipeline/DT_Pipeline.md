# Decision Tree Generation Pipeline

Interactive experimentation notebook. All heavy logic lives in the Python modules;
this notebook is a thin wrapper for data exploration and model selection.

**Workflow:**
1. Configure `PipelineConfig` (tweak params below)
2. Load data, auto-detect columns
3. Split data (stratified 70/30)
4. Run grid search (one function call)
5. Review sorted results
6. Select model, inspect impact group accuracy
7. Export Python and C++ code


```python
from ml_pipeline.core.config import PipelineConfig
from ml_pipeline.core.data_loader import load_data, detect_columns, split_data, prepare_features
from ml_pipeline.core.feature_engineering import apply_feature_engineering, get_derived_feature_cpp, get_derived_feature_py
from ml_pipeline.core.trainer import run_grid_search
from ml_pipeline.core.results import sort_results, display_results, display_impact_groups
from ml_pipeline.core.code_generator import tree_to_python_code, tree_to_c_code
from ml_pipeline.core.history import RunHistory
```


```python
# ── Configuration ──────────────────────────────────────────────────────
# Tweak any of these before running the pipeline.

CSV_PATH = 'output.csv'

config = PipelineConfig()
config.exclude_m = False
config.train_on_whole = False
config.weight_transform = 'raw'        # 'raw', 'log+1', 'sqrt', 'minmax', 'rank', 'percentile_clip'
config.weight_minmax_low = 1
config.weight_minmax_high = 100
config.post_prune_redundant = True
config.print_params = False
config.run_cv = False                    # Enable for cross-validation (recommended for large datasets)
config.feature_engineering = "none"      # "none", "cheap", "moderate", "expensive"
config.impact_threshold_low = 5
config.impact_threshold_high = 50

# Override grid search params if needed:
# config.param_grid['max_depth'] = [1, 2, 3, 4]

# Uncomment to see every configurable parameter and its current value:
# config.show_all_params()

# Run history — persists across re-runs as long as the kernel is alive.
# Re-running this cell does NOT reset history; only kernel restart or history.clear() does.
if 'history' not in globals():
    history = RunHistory()
```


```python
# ── Load Data & Detect Columns ─────────────────────────────────────────
df = load_data(CSV_PATH)
detect_columns(df, config)
apply_feature_engineering(df, config)
print()
print(config.summary())
```


```python
# ── Split Data ─────────────────────────────────────────────────────────
train_df, test_df = split_data(df, config)

print(f'Train: {len(train_df)} records')
print(f'Test : {len(test_df)} records')
print(f'\nTrain target distribution:\n{train_df[config.target_col].value_counts(normalize=True)}')
print(f'\nTest target distribution:\n{test_df[config.target_col].value_counts(normalize=True)}')
```


```python
# ── Grid Search ────────────────────────────────────────────────────────
results_list, models_dict = run_grid_search(df, train_df, test_df, config)
```


```python
# ── Display Sorted Results ─────────────────────────────────────────────
sorted_results = sort_results(results_list, config)
display_results(sorted_results, config)

# Save this run to history
history.add_run(config, sorted_results, models_dict)
```


```python
# ── Run History ────────────────────────────────────────────────────────
# Compare best models across all runs in this session.
# Tweak config (cell 2), re-run cells 2-6, and this cell updates automatically.
history.show_all_runs(config)

# Retrieve the global best model across all runs:
best_run_id, best_key, best_model = history.get_global_best(config)
if best_run_id:
    print(f"\nGlobal best model key: {best_key}  (from Run {best_run_id})")
```


```python
# ── Select Model & Inspect ─────────────────────────────────────────────
# By default, uses the global best model across all runs.
# Uncomment a different option to pick from a specific run or the current run.

# Option 1: Global best from all runs (default)
_, SELECTED_MODEL_KEY, selected_model = history.get_global_best(config)

# Option 2: Best from current run only
# SELECTED_MODEL_KEY = sorted_results[0].index_key
# selected_model = models_dict[SELECTED_MODEL_KEY]

# Option 3: Pick a specific model from a specific past run
# SELECTED_MODEL_KEY = "506_0.012..."    # model key from that run's results table
# selected_model = history.get_model(run_id=1, model_key=SELECTED_MODEL_KEY)

if selected_model is None:
    print("No model selected. Run the grid search (cells 5-6) first.")
else:
    from sklearn.tree import export_text
    X_train, _ = prepare_features(train_df, config)
    print(export_text(selected_model, feature_names=list(X_train.columns)))

    # Impact group accuracy
    display_impact_groups(selected_model, df, train_df, test_df, config,
                          selected_key=SELECTED_MODEL_KEY)
```


```python
# ── Export Python and C++ Code ─────────────────────────────────────────
if selected_model is None:
    print("No model selected. Run cells 5-6 and 8 first.")
else:
    feature_names = list(config.feature_cols)
    base_features = list(config.all_feature_cols)
    derived_cpp = get_derived_feature_cpp(config)
    derived_py = get_derived_feature_py(config)

    # Uncomment to generate Python code for experimentation:
    # py_code = tree_to_python_code(selected_model, feature_names, derived_features=derived_py)
    # print('=== Python Code ===')
    # print(py_code)

    print('=== C++ Code ===')
    cpp_code = tree_to_c_code(selected_model, feature_names,
                              base_features=base_features,
                              derived_features=derived_cpp)
    print(cpp_code)
```
