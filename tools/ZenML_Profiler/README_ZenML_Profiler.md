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

# ZenML Profiler

## Overview

ZenML Profiler is a log analysis tool for extracting performance insights from
ZenDNN/OneDNN primitive-level logs. It automatically detects iteration
boundaries, slices the log to the requested iteration, and produces grouped,
detailed, and FLOPs summaries. When multiple logs are provided it generates a
comparative analysis as well.

Key capabilities:

- **Multi-backend support** -- ZenDNN_5.2 (default), ZenDNN_5.1 (legacy
  positional logs), ZenDNN_4.2, OneDNN.
- **Automatic iteration detection** via pattern analysis, or user-supplied
  iteration count.
- **% Impact column** in every table, calculated against end-to-end time (if
  provided via `-t`) or against total library execution time.
- **Matmul/BMM comparison** (`--compare_mm`) with unified B, M, N, K columns.
- **Excel report** with pie charts, line charts, and time-ratio graphs.
- **PyTorch Profiler integration** from JSON traces or log files.
- **BenchDNN integration** for micro-benchmark comparison.

The analysis report is saved to the file specified by `-rf` (default:
`ZenMLProfiler_report.txt`). The command used to invoke the profiler is printed
at the top of the report for reproducibility.

Note: For getting analysis on the entire log, give total iteration (`-it`) as 1.

---

## Project Structure

```
ZenML_Profiler/
├── profiler.py                  # Main entry point
├── ZenML_Profiler_wrapper.py    # Backward-compatible wrapper (interactive mode)
├── requirements.txt
├── README_ZenML_Profiler.md
└── zenml/                       # Core package
    ├── config/
    │   ├── profiler_config.py   # ProfilerConfig dataclass, log_info loader
    │   ├── log_info.json        # Backend log format definitions
    │   └── args.json            # JSON argument template
    ├── parsers/
    │   ├── dispatcher.py        # Unified parse_log() dispatcher
    │   ├── positional_parser.py # Positional (CSV) log parser
    │   └── kv_parser.py         # Key-value log parser
    ├── analysis/
    │   ├── pattern_analyzer.py  # Repeating-pattern iteration detection
    │   ├── iteration_splitter.py# Iteration boundary finder
    │   └── log_slicer.py        # Extract a single iteration from a log
    ├── reporting/
    │   ├── orchestrator.py      # Top-level report generation
    │   ├── summary.py           # Group and detailed summary builders
    │   ├── flops.py             # FLOPs / efficiency table builder
    │   ├── comparison.py        # Multi-log comparison + matmul compare
    │   ├── excel.py             # Excel workbook + chart generation
    │   └── utils.py             # Shared helpers (table_maker, etc.)
    ├── integrations/
    │   ├── pytorch_profiler.py  # PyTorch Profiler trace analysis
    │   └── benchdnn.py          # BenchDNN input generation + algo extraction
    └── utils/
        ├── helpers.py           # block_print / enable_print
        └── logging_fmt.py       # Colored log formatter
```

---

## Quick Start

### Direct invocation (recommended)

    python profiler.py -in1 model.log -it1 10

The default backend is `ZenDNN_5.2`. To use a different backend, pass `-bk1`:

    python profiler.py -in1 model.log -bk1 ZenDNN_5.1 -it1 10

### Via wrapper (interactive JSON/CLI choice)

    python ZenML_Profiler_wrapper.py -in1 model.log -it1 10

---

## Usage

### Single Log Analysis

By default provides summary of all ops present in the second last iteration.

    python profiler.py -in1 <file path>

Optional: `-it1 <total iterations>` (recommended if known).

Example:

    python profiler.py -in1 GPT_J.log -it1 15

### Specific Iteration Analysis

    python profiler.py -in1 <file path> -it1 <total iterations> -n1 <iteration>

Example -- analyse the 13th iteration of a 15-iteration run:

    python profiler.py -in1 GPT_J.log -it1 15 -n1 13

### Using a Different Backend

For older ZenDNN 5.1 logs (positional/CSV format), specify the backend explicitly:

    python profiler.py -in1 model_old.log -bk1 ZenDNN_5.1 -it1 10

Supported backends: `ZenDNN_5.2` (default), `ZenDNN_5.1`, `ZenDNN_4.2`, `OneDNN`.

### Cross-Version Comparison

    python profiler.py \
        -in1 model_51.log -bk1 ZenDNN_5.1 -it1 10 -ln1 ZenDNN_5.1 \
        -in2 model_52.log -it2 10 -ln2 ZenDNN_5.2 \
        -lc 2 --compare_mm

The `--compare_mm` flag adds an additional Matmul/BMM Comparison Table with
unified B, M, N, K columns for easy cross-format comparison.

### Analysis with Execution Time (% Impact)

When `-t` is provided, the `% Impact` column uses end-to-end time as the
denominator. Without `-t`, it uses total library execution time.

    python profiler.py -in1 GPT_J.log -it1 15 -t1 0.512

### Analysis with FLOPs and Efficiency

    python profiler.py -in1 GPT_J.log -it1 15 -mc genoa --roofline_path /path/to/Roofline

Kernel efficiency requires `--roofline_path` pointing to the directory containing
`mmRoofline.py` and `machines_config.json`. Without it, FLOPs (GOPs/GFlops)
are still computed but efficiency is skipped.

### PyTorch Profiler Integration

From JSON trace:

    python profiler.py -in1 GPT_J.log -ppj1 pytorch_export.json

From profiler log output:

    python profiler.py -in1 GPT_J.log -ppl1 pytorch_output.log

### Comparative Analysis (n logs)

    python profiler.py \
        -in1 log_algo1.log -it1 15 -ln1 Algo_1 \
        -in2 log_algo2.log -it2 15 -ln2 Algo_2 \
        -in3 log_algo3.log -it3 15 -ln3 Algo_3 \
        -lc 3

### BenchDNN Analysis

    python profiler.py -in1 GPT_J.log -bk1 ZenDNN_5.1 -it1 15 --benchdnn

BenchDNN arguments are read from `zenml/config/args.json`.

### JSON Argument Passing

    python ZenML_Profiler_wrapper.py args.json

Update the JSON template (`zenml/config/args.json`) with required parameters.
Set a flag to `TRUE` to enable it; leave empty or `FALSE` to disable.

---

## CLI Reference

### Required (at least one)

| Flag                                       | Description                               |
|--------------------------------------------|-------------------------------------------|
| `-in<n>` / `--input_<n>`                   | Path to log file n                        |
| `-ppj<n>` / `--pytorch_profiler_json_<n>`  | Path to PyTorch Profiler JSON for log n   |
| `-ppl<n>` / `--pytorch_profiler_log_<n>`   | Path to PyTorch Profiler log for log n    |

### Per-Log Options

| Flag                          | Description                                                                                 |
|-------------------------------|---------------------------------------------------------------------------------------------|
| `-it<n>` / `--iter_<n>`       | Total iterations for log n                                                                  |
| `-n<n>` / `--number_<n>`      | Specific iteration to analyse for log n                                                     |
| `-ln<n>` / `--log_name_<n>`   | Display name for log n                                                                      |
| `-t<n>` / `--time_<n>`        | End-to-end time per iteration (seconds) for log n                                           |
| `-bk<n>` / `--backend_<n>`    | Backend (default: `ZenDNN_5.2`). Options:`ZenDNN_5.2`, `ZenDNN_5.1`, `ZenDNN_4.2`, `OneDNN` |

### Global Options

| Flag                         | Description                                                          |
|------------------------------|----------------------------------------------------------------------|
| `-lc` / `--log_count`        | Number of logs (default: 1)                                          |
| `-v` / `--verbose`           | Enable high verbosity (avg, median, min, max, std dev)               |
| `-mc` / `--machine`          | Machine name for efficiency calculation                              |
| `--roofline_path`            | Path to Roofline dir (mmRoofline.py + machines_config.json)          |
| `--compare_mm`               | Additional matmul/BMM comparison table with B, M, N, K columns       |
| `--benchdnn`                 | Enable BenchDNN analysis                                             |
| `--benchdnn_disp`            | Display BenchDNN comparison output                                   |
| `-is` / `--inference_script` | Inference script type: `custom` or `hugging_face`                    |
| `-e` / `--export`            | Export the sliced logs to a file                                     |
| `-f` / `--file_name`         | Name of the exported sliced log file                                 |
| `-rf` / `--result_file`      | Report output file (default: `ZenMLProfiler_report.txt`)             |
| `-dp` / `--disable_pattern`  | Disable automatic pattern analysis                                   |
| `-dx` / `--disable_excel`    | Disable Excel report generation                                      |
| `--csv`                      | Export analysis to CSV                                               |
| `--info`                     | Show explanatory text for each table                                 |
| `--threshold`                | CPU time % threshold for PyTorch Profiler filtering (default: 2)     |

---

## Adding New Backends

Backend log formats are defined in `zenml/config/log_info.json`. Each backend
entry specifies `"format": "positional"` or `"format": "kv"`, along with
per-log-type configurations that describe how to parse each line. Add a new
top-level key to support a new backend.

---

## Dependencies

See `requirements.txt`:

    pytimedinput
    prettytable
    pandas
    numpy
    openpyxl
    pytz
