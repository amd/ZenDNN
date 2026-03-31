# ZenML Profiler — Complete Feature Guide


---

## 1. Multi-Backend Log Parsing

ZenML Profiler supports multiple log backends, each with a different log format. The backend is selected via `-bk<n>`.

| Backend       | Format     | Description                            |
|---------------|------------|----------------------------------------|
| `ZenDNN_5.2`  | Key-Value  | Default. ZenDNN 5.2 KV-format logs     |
| `ZenDNN_5.1`  | Positional | Older ZenDNN 5.1 CSV-format logs       |
| `ZenDNN_4.2`  | Positional | ZenDNN 4.2 format                      |
| `OneDNN`      | Positional | OneDNN log format                      |

**Supported Operations:**
- `matmul` / `batch_matmul` / `inner_product` / `gemm_api`
- `softmax` / `softmax_v2`
- `convolution` / `conv_direct`
- `reorder`
- `embedding` (`embedding_context_create`, `embedding_op_create`, `embedding execute`)
- `layer_normalization`

**Example — ZenDNN 5.2 (default):**
```bash
python profiler.py -in1 model.log -it1 10
```

**Example — ZenDNN 5.1:**
```bash
python profiler.py -in1 model.log -bk1 ZenDNN_5.1 -it1 10
```


---

## 2. Automatic Iteration Detection

The profiler can automatically detect the number of iterations in a log by analyzing repeating operation patterns. When `-it<n>` is not provided, the pattern analyzer scans ~75% of the log to identify the repeating unit.

**How it works:**
1. A custom representation of the log is created (op + dimension per line)
2. The pattern analyzer finds the smallest repeating segment
3. Iteration count = ceil(total_ops / pattern_length)

**Command — Auto-detect iterations:**
```bash
python profiler.py -in1 model.log
```

**Command — Manually specify iterations:**
```bash
python profiler.py -in1 model.log -it1 10
```

**Command — Disable pattern analysis (treat entire log as one iteration):**
```bash
python profiler.py -in1 model.log --disable_pattern
```


---

## 3. Iteration Splitting & Log Slicing

Once iterations are detected, the profiler splits the log at iteration boundaries and analyzes a specific iteration (defaults to the last one).

**Features:**
- Identifies anomalies (malformed log lines) and reports their line numbers
- Safe Mode: Falls back to major ops only if splitting fails
- Auto-Tuner handling: Adjusts splitting when ZenDNN auto_tuner is detected

**Analyze a specific iteration:**
```bash
python profiler.py -in1 model.log -it1 10 -n1 5
```

**Export the sliced log to a file:**
```bash
python profiler.py -in1 model.log -it1 10 -e -f sliced_iter.txt
```


---

## 4. Grouped Execute Summary

A high-level summary showing total time and count per operation type.

**Command:**
```bash
python profiler.py -in1 model.log -bk1 ZenDNN_5.2 -it1 3
```

**Example Output:**
```
Grouped Execute Summary
+------------+--------------+----------------+-------+-------------------------+
|  Backend   |   Op Type    | Op Time (ms)   | Count | % Primitive Exec Impact |
+------------+--------------+----------------+-------+-------------------------+
| ZenDNN_5.2 |    matmul    |    19655.313   |  1460 |          78.78          |
+------------+--------------+----------------+-------+-------------------------+
| ZenDNN_5.2 | batch_matmul |     2888.915   |  480  |          11.58          |
+------------+--------------+----------------+-------+-------------------------+
| ZenDNN_5.2 |   softmax    |     2406.508   |  240  |           9.65          |
+------------+--------------+----------------+-------+-------------------------+
```


---

## 5. Detailed Op Summary

A per-unique-operation breakdown showing kernel, dimension, data format, post-ops, timing, and impact.

**Example Output:**
```
Detailed Op Summary
+------------+--------------+----------------+---------------+--------------------------------------+----------+-----------------+-------+----------+
|  Backend   |   Op Type    |     Kernel     |   Dimension   |             Data Format              | Post ops | Total Time (ms) | Count | % Impact |
+------------+--------------+----------------+---------------+--------------------------------------+----------+-----------------+-------+----------+
| ZenDNN_5.2 |    matmul    | onednn_blocked |  49152x1024:  | src_f32::wei_f32::dst_f32::bias_f32  |    NA    |     6522.51     |  960  |  26.14   |
|            |              |                |   1024x1024:  |                                      |          |                 |       |          |
|            |              |                |   49152x1024  |                                      |          |                 |       |          |
+------------+--------------+----------------+---------------+--------------------------------------+----------+-----------------+-------+----------+
| ZenDNN_5.2 |    matmul    | onednn_blocked |  49152x1024:  | src_f32::wei_f32::dst_f32::bias_f32  | gelu_erf |     5662.651    |  240  |   22.7   |
|            |              |                |   1024x4096:  |                                      |          |                 |       |          |
|            |              |                |   49152x4096  |                                      |          |                 |       |          |
+------------+--------------+----------------+---------------+--------------------------------------+----------+-----------------+-------+----------+
| ZenDNN_5.2 |   softmax    |       NA       |    128x384    |            src_1::dst_1              |    NA    |     2406.508    |  240  |   9.65   |
+------------+--------------+----------------+---------------+--------------------------------------+----------+-----------------+-------+----------+
```


---

## 6. Verbose Mode (Statistical Details)

Adds Avg, Median, Min, Max, and Std. Dev columns to the detailed summary.

**Command:**
```bash
python profiler.py -in1 model.log -it1 3 -v
```

**Example Output:**
```
+------------+-----------+----------------+---------------+--------------------------------------+----------+-----------------+--------------+------------------+--------------+--------------+----------+-------+----------+
|  Backend   |  Op Type  |     Kernel     |   Dimension   |             Data Format              | Post ops | Total Time (ms) | Avg Time (ms)| Median Time (ms) | Min Time (ms)| Max Time (ms)| Std. Dev | Count | % Impact |
+------------+-----------+----------------+---------------+--------------------------------------+----------+-----------------+--------------+------------------+--------------+--------------+----------+-------+----------+
| ZenDNN_5.2 |  matmul   | onednn_blocked |  49152x1024:  | src_f32::wei_f32::dst_f32::bias_f32  |    NA    |     6522.51     |    6.794     |      6.323       |    5.555     |    45.877    |  1.759   |  960  |  26.14   |
|            |           |                |   1024x1024:  |                                      |          |                 |              |                  |              |              |          |       |          |
|            |           |                |   49152x1024  |                                      |          |                 |              |                  |              |              |          |       |          |
+------------+-----------+----------------+---------------+--------------------------------------+----------+-----------------+--------------+------------------+--------------+--------------+----------+-------+----------+
| ZenDNN_5.2 |  softmax  |       NA       |    128x384    |            src_1::dst_1              |    NA    |     2406.508    |   10.027     |      9.798       |    9.49      |    57.506    |  3.083   |  240  |   9.65   |
+------------+-----------+----------------+---------------+--------------------------------------+----------+-----------------+--------------+------------------+--------------+--------------+----------+-------+----------+
```


---

## 7. % Impact Column (Dynamic Denominator)

Every table includes a `% Impact` column showing each operation's contribution to overall time.

**Without e2e time:** Denominator = total library execution time (sum of all ops)
```
* % Impact is w.r.t total library execution time (sum of all primitive ops)
```

**With e2e time (`-t1 30`):** Denominator = user-provided end-to-end time
```
* % Impact is w.r.t user-provided end-to-end time
```

**Group summary with e2e time — shows both columns:**
```
+------------+--------------+--------------+-------+-------------------------+--------------+
|  Backend   |   Op Type    | Op Time (ms) | Count | % Primitive Exec Impact | % E2E Impact |
+------------+--------------+--------------+-------+-------------------------+--------------+
| ZenDNN_5.2 |    matmul    |  19655.313   |  1460 |          78.78          |    65.52     |
+------------+--------------+--------------+-------+-------------------------+--------------+
| ZenDNN_5.2 | batch_matmul |   2888.915   |  480  |          11.58          |     9.63     |
+------------+--------------+--------------+-------+-------------------------+--------------+
| ZenDNN_5.2 |   softmax    |   2406.508   |  240  |           9.65          |     8.02     |
+------------+--------------+--------------+-------+-------------------------+--------------+
|   Other    |      -       |   5049.264   |   -   |            -            |    16.83     |
+------------+--------------+--------------+-------+-------------------------+--------------+
```

**Command:**
```bash
python profiler.py -in1 model.log -it1 3 -t1 30
```


---

## 8. FLOPs & Kernel Efficiency

Calculates GOPs (Giga Operations), GFlops, and kernel efficiency for matmul and batch_matmul operations.

**Command (without machine config):**
```bash
python profiler.py -in1 model.log -it1 3
```

**Example Output:**
```
FLOPs and Efficiency for matmul and batch_matmul ops:
+------------+--------------+----------------+----------+--------------------------------------+-------+---------+-------------+----------+-------------------+-------------------+
|  Backend   |   Op Type    |     Kernel     | Post ops |             Data Format              | Count | Batches |  Dimension  | % Impact | Kernel Efficiency |   FLOPS details   |
+------------+--------------+----------------+----------+--------------------------------------+-------+---------+-------------+----------+-------------------+-------------------+
| ZenDNN_5.2 |    matmul    | onednn_blocked |    NA    | src_f32::wei_f32::dst_f32::bias_f32  |  960  |    1    | 49152x1024: |  26.14   |         -         |  GOPS: 103.07922  |
|            |              |                |          |                                      |       |         |  1024x1024: |          |                   |    Time: 6.794    |
|            |              |                |          |                                      |       |         |  49152x1024 |          |                   | GFlops: 15172.095 |
+------------+--------------+----------------+----------+--------------------------------------+-------+---------+-------------+----------+-------------------+-------------------+
| ZenDNN_5.2 | batch_matmul |    aocl_dlp    |    NA    | src_f32::wei_f32::dst_f32::bias_f32  |  240  |   2048  |   384x64:   |   6.8    |         -         |   GOPS: 38.65471  |
|            |              |                |          |                                      |       |         |   64x384:   |          |                   |    Time: 7.073    |
|            |              |                |          |                                      |       |         |   384x384   |          |                   |  GFlops: 5465.108 |
+------------+--------------+----------------+----------+--------------------------------------+-------+---------+-------------+----------+-------------------+-------------------+
```

**With machine config for roofline efficiency:**
```bash
python profiler.py -in1 model.log -it1 3 -mc genoa --roofline_path /path/to/Roofline
```

**Note:** Kernel efficiency requires `--roofline_path` pointing to the directory containing
`mmRoofline.py` and `machines_config.json`. Without it, GOPs and GFlops are still computed
but the Kernel Efficiency column shows `"-"`.

**Formulae:**
- GOPs = 2 * M * N * K * Batches * 0.000000001
- GFlops = GOPs * 1000 / Avg Time (ms)
- Efficiency = Achieved GFlops / Theoretical GFlops * 100


---

## 9. Multi-Log Comparison

Compare operations across 2 or more logs side-by-side with automatic operation matching.

**Features:**
- Maps equivalent ops across backends (e.g., `matmul` ↔ `inner_product` ↔ `gemm_api`, `softmax` ↔ `softmax_v2`)
- Normalizes batch_matmul dimensions for matching across versions
- Time ratio columns (Log1/Log2)
- Sorted by % Impact of first log (descending), with unmatched ops at the bottom

**Command:**
```bash
python profiler.py \
  -in1 model_v1.log -bk1 ZenDNN_5.1 -it1 3 -ln1 ZenDNN_5.1 \
  -in2 model_v2.log -bk2 ZenDNN_5.2 -it2 3 -ln2 ZenDNN_5.2 \
  -lc 2
```

**Example Output:**
```
Comparison Table:
+--------------+--------------+-------------+----------------+-----------------+---------------+----------------+----------------+------------+------------+---------------------+
|  ZenDNN_5.1  |  ZenDNN_5.2  |  ZenDNN_5.1 |   ZenDNN_5.2   |    ZenDNN_5.1   |   ZenDNN_5.2  |   ZenDNN_5.1   |   ZenDNN_5.2   | ZenDNN_5.1 | ZenDNN_5.2 |     ZenDNN_5.1/     |
|      Op      |      Op      |    Kernel   |     Kernel     |    Dimension    |   Dimension   |   Time (ms)    |   Time (ms)    |  % Impact  |  % Impact  | ZenDNN_5.2 Time (ms)|
+--------------+--------------+-------------+----------------+-----------------+---------------+----------------+----------------+------------+------------+---------------------+
|    matmul    |    matmul    |    zendnn   | onednn_blocked |   49152x1024:   |  49152x1024:  |    5919.92     |    6606.68     |   24.64    |   26.39    |         0.9         |
|              |              |             |                |    1024x1024:   |   1024x1024:  |                |                |            |            |                     |
|              |              |             |                |    49152x1024   |   49152x1024  |                |                |            |            |                     |
+--------------+--------------+-------------+----------------+-----------------+---------------+----------------+----------------+------------+------------+---------------------+
| batch_matmul | batch_matmul |     brg:    |    aocl_dlp    |  128x16x384x64: |  2048x384x64: |    1822.06     |    1697.54     |    7.59    |    6.78    |        1.07         |
|              |              | avx512_core |                |  128x16x64x384: |  2048x64x384: |                |                |            |            |                     |
|              |              |             |                |  128x16x384x384 |  2048x384x384 |                |                |            |            |                     |
+--------------+--------------+-------------+----------------+-----------------+---------------+----------------+----------------+------------+------------+---------------------+
|  softmax_v2  |      -       |     jit:    |       -        |  128x16x384x384 |       -       |    2179.24     |       -        |    9.07    |     -      |          -          |
|              |              | avx512_core |                |                 |               |                |                |            |            |                     |
+--------------+--------------+-------------+----------------+-----------------+---------------+----------------+----------------+------------+------------+---------------------+
|      -       |   softmax    |      -      |       NA       |        -        |    128x384    |       -        |    2406.51     |     -      |    9.61    |          -          |
+--------------+--------------+-------------+----------------+-----------------+---------------+----------------+----------------+------------+------------+---------------------+
```


---

## 10. Matmul/BMM Comparison Table (`--compare_mm`)

A dedicated comparison table for matmul and batch_matmul operations using unified `B`, `M`, `N`, `K` columns.

**Why:** Raw dimension strings differ across backends. `B/M/N/K` makes comparison intuitive.

**Command:**
```bash
python profiler.py \
  -in1 model_v1.log -bk1 ZenDNN_5.1 -it1 3 -ln1 ZenDNN_5.1 \
  -in2 model_v2.log -bk2 ZenDNN_5.2 -it2 3 -ln2 ZenDNN_5.2 \
  -lc 2 --compare_mm
```

**Example Output (columns grouped by type — Kernels together, Times together, etc.):**
```
Matmul/BMM Comparison Table:
+------+-------+-------+------+-------------------+-------------------+---------------------+---------------------+------------------+------------------+---------------------+---------------------+--------------------------------+
|  B   |   M   |   N   |  K   | ZenDNN_5.1 Kernel | ZenDNN_5.2 Kernel | ZenDNN_5.1 Time (ms)| ZenDNN_5.2 Time (ms)| ZenDNN_5.1 Count | ZenDNN_5.2 Count | ZenDNN_5.1 % Impact | ZenDNN_5.2 % Impact | ZenDNN_5.1/ZenDNN_5.2 Time (ms)|
+------+-------+-------+------+-------------------+-------------------+---------------------+---------------------+------------------+------------------+---------------------+---------------------+--------------------------------+
|  1   | 49152 |  1024 | 1024 |       zendnn      |   onednn_blocked  |       5919.92       |       6606.68       |       970        |       970        |        24.64        |        26.39        |              0.9               |
+------+-------+-------+------+-------------------+-------------------+---------------------+---------------------+------------------+------------------+---------------------+---------------------+--------------------------------+
|  1   | 49152 |  1024 | 4096 |       zendnn      |   onednn_blocked  |       5693.15       |       5627.79       |       240        |       240        |         23.7        |        22.48        |              1.01              |
+------+-------+-------+------+-------------------+-------------------+---------------------+---------------------+------------------+------------------+---------------------+---------------------+--------------------------------+
| 2048 |  384  |  384  |  64  |  brg:avx512_core  |      aocl_dlp     |       1822.06       |       1697.54       |       240        |       240        |         7.59        |         6.78        |              1.07              |
+------+-------+-------+------+-------------------+-------------------+---------------------+---------------------+------------------+------------------+---------------------+---------------------+--------------------------------+
| 2048 |  384  |   64  | 384  |  brg:avx512_core  |      aocl_dlp     |       1193.94       |       1191.38       |       240        |       240        |         4.97        |         4.76        |              1.0               |
+------+-------+-------+------+-------------------+-------------------+---------------------+---------------------+------------------+------------------+---------------------+---------------------+--------------------------------+
```


---

## 11. Excel Report Generation

Automatically generates an Excel workbook with multiple sheets and charts.

**Sheets:**
- Grouped Summary (with pie charts per log)
- Detailed Op Summary
- FLOPs & Efficiency
- Op Time Comparison (when comparing logs) — with bar chart, line chart, and time ratio chart

**Command:**
```bash
python profiler.py -in1 model.log -it1 3
# Output: ./Excel_Analysis/ZenProfiler_Report_<timestamp>.xlsx
```

**Disable Excel:**
```bash
python profiler.py -in1 model.log -it1 3 --disable_excel
```


---

## 12. PyTorch Profiler Integration

Analyze PyTorch profiler outputs — either JSON traces or text logs.

**Features:**
- Self CPU time calculation (total duration - child durations)
- Grouped analysis by operation class
- Sub-group expansion for groups with >5% CPU
- Threshold filtering for minor ops
- Standalone mode (no ZenDNN log required)

**Command — JSON trace:**
```bash
python profiler.py -ppj1 trace.json
```

**Command — Text log:**
```bash
python profiler.py -ppl1 profiler_output.log
```

**Command — Combined with ZenDNN log:**
```bash
python profiler.py -in1 model.log -it1 10 -ppj1 trace.json
```

**Output includes:**
```
PyTorch Profiler Analysis:
+---------+--------------+--------------------+--------------+--------------------+------------------+-------+
| Op name | Self CPU %   | Self CPU Time(ms)  | CPU Total %  | CPU Total Time(ms) | CPU Avg Time(ms) | Count |
+---------+--------------+--------------------+--------------+--------------------+------------------+-------+
|  aten:: |    45.2      |      123.45        |    52.1      |      142.30        |      0.89        |  160  |
|  matmul |              |                    |              |                    |                  |       |
+---------+--------------+--------------------+--------------+--------------------+------------------+-------+

Grouped PyTorch Profiler Analysis:
+------------+--------------+--------------------+--------------+--------------------+------------------+-------+
| Class name | Self CPU %   | Self CPU Time(ms)  | CPU Total %  | CPU Total Time(ms) | CPU Avg Time(ms) | Count |
+------------+--------------+--------------------+--------------+--------------------+------------------+-------+
|    aten    |    78.3      |      214.56        |    85.7      |      234.89        |      1.23        |  190  |
+------------+--------------+--------------------+--------------+--------------------+------------------+-------+
```


---

## 13. BenchDNN Integration

Run BenchDNN micro-benchmarks for matmul dimensions found in the log and compare against model execution times.

**Command:**
```bash
python profiler.py -in1 model.log -bk1 ZenDNN_5.1 -it1 10 --benchdnn
```

**What it does:**
1. Extracts all matmul dimensions (M, N, K) from the log
2. Generates `input.txt` with BenchDNN-formatted dimensions
3. Runs BenchDNN driver script
4. Compares BenchDNN times against model log times

**Note:** BenchDNN integration is available only for the ZenDNN_5.1 backend.


---

## 14. Report File Output

The wrapper script saves all output to a text report file.

**Command:**
```bash
python ZenML_Profiler_wrapper.py -in1 model.log -it1 3 -rf my_report.txt
```

**Features:**
- The command used is printed at the top of the report for reproducibility
- Warning messages go to stderr (not included in the report file)
- Default filename: `ZenMLProfiler_report.txt`


---

## 15. CSV Export

Export the analysis tables to CSV format.

**Command:**
```bash
python profiler.py -in1 model.log -it1 3 --csv
```

The tool will interactively prompt for the output filename.


---

## 16. Info Mode

Displays explanatory descriptions for each table and metric.

**Command:**
```bash
python profiler.py -in1 model.log -it1 3 --info
```

**Example Output (before group summary):**
```
% Primitive Exec Impact  --- Percentage contribution of each op within
                             total op execution time in the run.
% E2E Impact             --- Percentage contribution of a particular op
                             to the end-to-end execution time of the model.
```

**Example Output (before FLOPs table):**
```
GOPs (Giga Operations) - the total number of arithmetic operations
GFlops - floating-point operations per second, in billions
Efficiency - Achieved GFLOPS / Theoretical GFLOPS * 100

Formulae:
  GFlops = GOPs * 1000 / Actual time in ms
  GOPs = 2 * M * N * K * 0.000000001
  Efficiency = Achieved GFLOPS / Theoretical GFLOPS * 100
```


---

## 17. Inference Script Support

Supports parsing logs from different inference frameworks.

| Script         | Description                              |
|----------------|------------------------------------------|
| `custom`       | Default. Standard ZenDNN log format      |
| `hugging_face` | Extracts e2e time from Hugging Face output |

**Command:**
```bash
python profiler.py -in1 model.log -it1 10 -is hugging_face
```


---

## CLI Quick Reference

| Flag                              | Short    | Description                                 | Default                  |
|-----------------------------------|----------|---------------------------------------------|--------------------------|
| `--input_<n>`                     | `-in<n>` | Input log file for log N                    | Required                 |
| `--backend_<n>`                   | `-bk<n>` | Backend for log N                           | `ZenDNN_5.2`             |
| `--iter_<n>`                      | `-it<n>` | Total iterations in log N                   | Auto-detected            |
| `--number_<n>`                    | `-n<n>`  | Iteration to analyze for log N              | Last iteration           |
| `--time_<n>`                      | `-t<n>`  | E2E time per iteration (seconds) for log N  | None                     |
| `--log_name_<n>`                  | `-ln<n>` | Display name for log N                      | "Log n"                  |
| `--log_count`                     | `-lc`    | Number of logs to compare                   | 1                        |
| `--verbose`                       | `-v`     | Add statistical columns to detailed table   | False                    |
| `--info`                          |          | Display explanatory text for tables         | False                    |
| `--machine`                       | `-mc`    | Machine config for roofline                 | ""                       |
| `--roofline_path`                 |          | Path to Roofline dir for efficiency calc    | ""                       |
| `--compare_mm`                    |          | Generate matmul/BMM comparison table        | False                    |
| `--csv`                           |          | Export tables to CSV                        | False                    |
| `--benchdnn`                      |          | Run BenchDNN micro-benchmarks               | False                    |
| `--benchdnn_disp`                 |          | Display BenchDNN results                    | False                    |
| `--export`                        | `-e`     | Export sliced log to file                   | False                    |
| `--file_name`                     | `-f`     | Filename for exported sliced log            | sliced_output.txt        |
| `--result_file`                   | `-rf`    | Output report filename                      | ZenMLProfiler_report.txt |
| `--inference_script`              | `-is`    | Inference framework (custom / hugging_face) | custom                   |
| `--threshold`                     |          | PyTorch profiler display threshold (%)      | 2                        |
| `--disable_excel`                 | `-dx`    | Skip Excel report generation                | False                    |
| `--disable_pattern`               | `-dp`    | Skip automatic pattern detection            | False                    |
| `--pytorch_profiler_json_<n>`     | `-ppj<n>`| PyTorch profiler JSON trace for log N       | None                     |
| `--pytorch_profiler_log_<n>`      | `-ppl<n>`| PyTorch profiler text log for log N         | None                     |
