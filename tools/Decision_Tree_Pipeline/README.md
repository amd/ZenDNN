# Decision Tree Pipeline

Generates optimal Decision Trees for matmul algorithm-path selection, exports them as C++ functions for integration into ZenDNN.

The pipeline has two phases:
1. **Data Collection** — Parse PyTorch profiler logs, extract matmul shapes and timings, generate a training CSV.
2. **ML Pipeline** — Train Decision Trees via grid search with instance-weighted training, evaluate models, and export the best tree as a C++ function.

## Project Structure

```
Decision_Tree_Pipeline/
├── csv_generator.py              # CLI: generate training CSV from profiler logs
├── run_pipeline.py               # CLI: train DT and export C++ code
├── ipynb_generator.py            # Convert DT_Pipeline.md into a Jupyter notebook
├── DT_Pipeline.md                # Notebook source (markdown with fenced code blocks)
│
├── data_collection/              # Data collection modules
│   ├── utils.py                  #   Shared helpers (time conversion, constants)
│   ├── shape_extractors.py       #   M/K/N extraction from operator shapes
│   ├── profiler_parser.py        #   PyTorch profiler log parsing
│   ├── csv_builder.py            #   Merge algo data, compute Ratio, write CSV
│   └── __init__.py
│
├── ml_pipeline/                  # ML pipeline modules
│   └── core/
│       ├── config.py             #   PipelineConfig class (all tuneable parameters)
│       ├── utils.py              #   Weight transforms, impact group labels
│       ├── data_loader.py        #   CSV loading, column detection, train/test split
│       ├── metrics.py            #   Mismatch metric, weighted accuracy, geo mean
│       ├── tree_utils.py         #   Tree fingerprinting, redundant branch pruning
│       ├── trainer.py            #   Grid search loop with dedup & CCP pruning
│       ├── results.py            #   Sort, display, per-impact-group analysis
│       ├── history.py            #   RunHistory for tracking experiments in notebook
│       ├── feature_engineering.py#   Derived feature tiers (cheap/moderate/expensive)
│       └── code_generator.py     #   Export DT as C++ / Python function
│
└── README.md
```

---

## Step 1: Generate Training CSV — `csv_generator.py`

Parses PyTorch profiler logs from multiple algo runs, extracts matmul shapes (M, K, N),
aggregates timings, and produces a single training CSV with computed weights.

```bash
# Basic usage — point at a parent directory containing algo subfolders
python csv_generator.py /path/to/profiler_logs

# Specify output path and ratio scale factor
python csv_generator.py /path/to/profiler_logs -o output.csv -scale 100
```

**Expected parent directory layout:**
```
profiler_logs/
├── aocl/       # Required — profiler logs for Algo 1 (AOCL)
├── brgemm/     # Required — profiler logs for Algo 2 (BRGEMM)
├── libxsmm/    # Optional — profiler logs for Algo 3 (LIBXSMM)
└── native/     # Optional — profiler logs for Native baseline
```

Each subfolder should contain `.txt` PyTorch profiler output files.
The script auto-detects which subfolders exist and builds the CSV accordingly.

**Output:** A CSV file with columns like `M,K,N,AOCL_time,BRGEMM_time,Algo,Ratio`.

---

## Step 2a: Train & Export (Automated) — `run_pipeline.py`

Runs the full ML pipeline end-to-end: loads data, trains Decision Trees via grid search,
selects the best model, and exports it as a C++ function.

```bash
# Basic usage
python run_pipeline.py output.csv -o results/

# With options
python run_pipeline.py output.csv \
    --weight-transform log+1 \
    --exclude-m \
    --train-on-whole \
    --max-depth 3 4 \
    --function-name Decision_tree_path_BF16 \
    -o results/
```

**Available options:**

| Flag                    | Default                      | Description                                                        |
|-------------------------|------------------------------|--------------------------------------------------------------------|
| `-o, --output-dir`      | `output/`                    | Directory for output files                                         |
| `--exclude-m`           | off                          | Exclude M from input features                                      |
| `--train-on-whole`      | off                          | Train on whole dataset instead of 70/30 split                      |
| `--weight-transform`    | `minmax`                     | Weight transform: raw, log+1, sqrt, minmax, rank, percentile_clip  |
| `--minmax-low`          | 1                            | MinMax normalization lower bound                                   |
| `--minmax-high`         | 100                          | MinMax normalization upper bound                                   |
| `--no-prune-redundant`  | off                          | Disable redundant branch pruning                                   |
| `--run-cv`              | off                          | Enable cross-validation (disabled by default for small datasets)   |
| `--feature-engineering` | `none`                       | Derived feature tier: none, cheap, moderate, expensive             |
| `--print-params`        | off                          | Print hyperparameters during training                              |
| `--top-n`               | all                          | Only display top N results                                         |
| `--max-depth`           | 1 2 3 4                      | Override max_depth grid values                                     |
| `--function-name`       | `Decision_tree_path_BF16`    | Name for the generated C++ function                                |

**Output:** `decision_tree.cpp` in the output directory, ready for ZenDNN integration.

---

## Step 2b: Train & Export (Interactive) — Jupyter Notebook

The notebook source is stored as `DT_Pipeline.md` for clean Git diffs.
To create the Jupyter notebook, run:

```bash
python ipynb_generator.py DT_Pipeline.md
```

This generates `DT_Pipeline.ipynb` in the same directory. Then open it:

```bash
jupyter notebook DT_Pipeline.ipynb
```

Same logic as `run_pipeline.py`, but lets you tweak parameters, inspect results,
and select models interactively. The notebook also includes **Run History** to
compare results across multiple experiments within a session.

**Cells:**
1. **Imports** — loads all modules from `ml_pipeline.core`
2. **Configuration** — create and tweak `PipelineConfig` (uncomment `config.show_all_params()` to see everything)
3. **Load Data** — reads CSV, auto-detects timing columns and features
4. **Split Data** — stratified 70/30 train/test split
5. **Grid Search** — trains all hyperparameter combinations with deduplication
6. **Results** — sorted table with weighted accuracy, harmonic mean, CV metrics
7. **Run History** — compare best models across all runs in the session
8. **Inspect** — selected model's tree structure and per-impact-group accuracy
9. **Export** — generates C++ code for the selected model

---

## Configuration Reference

All tuneable parameters live in `PipelineConfig` (`ml_pipeline/core/config.py`).
Use `config.show_all_params()` to print every parameter with its current value.

| Parameter                | Default          | Description                                  |
|--------------------------|------------------|----------------------------------------------|
| `exclude_m`              | `False`          | Drop M from input features                   |
| `train_on_whole`         | `False`          | Train on whole dataset vs train split only    |
| `weight_transform`       | `minmax`         | Weight transformation method                  |
| `weight_minmax_low`      | `1`              | MinMax lower bound                            |
| `weight_minmax_high`     | `100`            | MinMax upper bound                            |
| `post_prune_redundant`   | `True`           | Collapse redundant branches after training    |
| `print_params`           | `False`          | Show hyperparams in training output           |
| `run_cv`                 | `False`          | Enable cross-validation                       |
| `feature_engineering`    | `"none"`         | Derived feature tier                          |
| `impact_threshold_low`   | `5`              | Ratio below this = "Minimal Impact"           |
| `impact_threshold_high`  | `50`             | Ratio above this = "Large Impact"             |
| `param_grid`             | (see config.py)  | Hyperparameter grid search space              |
| `cv_folds`               | `5`              | Number of cross-validation folds              |
| `test_size`              | `0.3`            | Train/test split ratio                        |

## Dependencies

- Python 3.9+
- scikit-learn
- pandas
- numpy
- prettytable
