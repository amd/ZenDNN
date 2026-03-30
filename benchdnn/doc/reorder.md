(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# Reorder Operator

## Overview
This describes all ways to provide input for the reorder benchmark in BenchDNN, including required parameters and example usage for each method.

---

## Usage
Run the reorder benchmark using the following input methods:

```sh
./install/benchdnn/bin/benchdnn --op=reorder --input_file=reorder_inputs.txt [--lowoha=true/false]
```

> **Note:**
> - The `--lowoha` option controls benchmarking for the Low Overhead API (LOWOHA). User can pass either `--lowoha=true` or `--lowoha=false`. If not specified, it is enabled by default.
> - When `--lowoha=false`, the benchmark runs the regular AOCL-based layout reorder.
> - When `--lowoha=true` (default), the benchmark runs the LOWOHA-based quantization/type-conversion reorder.

---

### Input File (`--input_file`)

The input file format differs based on whether LOWOHA mode is enabled or not.

#### Regular Reorder (`--lowoha=false`)

Provide a file with one configuration per line. Each line should contain:
- `rows` (Number of rows in the tensor)
- `cols` (Number of columns in the tensor)
- `iters` (Number of benchmark iterations)
- `dt` (Data type, e.g., f32, bf16)
- `kernel` (Kernel backend, e.g., aocl)
- `isInplace` (true/false, whether to perform in-place reorder)
- `warmup_iters` (optional)

**Example usage:**
```sh
./install/benchdnn/bin/benchdnn --op=reorder --input_file=reorder_inputs.txt --lowoha=false
```

**Example input file (`reorder_inputs.txt`):**
```
1024, 1024, 100, f32, aocl, true, 20
2048, 512, 50, bf16, aocl, false, 10
3072, 768, 100, f32, aocl, true, 30
```

#### LOWOHA Reorder (`--lowoha=true`, default)

Provide a file with one configuration per line. Each line should contain:
- `batch_size` (Batch size for the reorder operation)
- `rows` (Number of rows in the tensor)
- `cols` (Number of columns in the tensor)
- `iters` (Number of benchmark iterations)
- `src_dtype` (Source data type, e.g., f32, bf16)
- `dst_dtype` (Destination data type, e.g., s8, u8)
- `algo` (Reorder algorithm: `DT`, `native`, or `reference`)
- `scale_granularity` (Scale granularity: `per_tensor`, `per_channel_row`, `per_channel_col`, `per_group_row`, `per_group_col`)
- `group_size` (Group size for per-group granularities; 0 otherwise)
- `dynamic_quant` (Dynamic quantization flag: true/false or 1/0)
- `num_threads` (Number of threads; optional, defaults to 0 which uses `omp_get_max_threads`)
- `warmup_iters` (Number of warmup iterations; optional, defaults to 20% of iters)

**Example usage:**
```sh
./install/benchdnn/bin/benchdnn --op=reorder --input_file=reorder_lowoha_inputs.txt --lowoha=true
```

**Example input file (`reorder_lowoha_inputs.txt`):**
```
1, 1024, 1024, 100, f32, s8, reference, per_tensor, 0, false, 0, 20
1, 2048, 512, 50, bf16, s8, reference, per_channel_row, 0, false, 0, 10
4, 1024, 768, 100, f32, u8, reference, per_group_col, 128, false, 8, 20
1, 4096, 4096, 50, f32, u8, reference, per_channel_col, 0, true, 0, 10
2, 2048, 1024, 100, f32, u8, reference, per_group_row, 64, false, 4, 20
```
---

> **Note:**
> - The `isInplace` option controls whether the reorder is performed in-place or out-of-place.
> - The `warmup_iters` parameter is optional and can be used to specify the number of warmup iterations before benchmarking.

## Output

The benchmark prints the following for each input:
- Total execution time
- Average time per iteration
- For regular mode (when `MEASURE_INDIVIDUAL_TIMINGS` is enabled): detailed timing statistics, including:
  - Context creation time
  - Operator creation time
  - Operator execution time
  - Others time

Output is printed to the console and also saved to a CSV file named `timings_<current timestamp>.csv` in the `build` directory.

### Example (regular reorder with `--lowoha=false`, console/CSV)
```
Rows  Cols  Iterations  Data_type  Kernel_Name  In-place  Warmup_iters  Total_time(ms)  Avg_time(ms)  Ctx_Creation(ms_%)  Op_Creation(ms_%)  Op_Execution(ms_%)  Others(ms_%)
3072  768   100         f32        aocl         1         30            1318.81         13.188100     0.16 (0.01 %)       0.21 (0.02 %)      1318.12 (99.95 %)   0.31 (0.02 %)
768   3072  100         f32        aocl         1         30            1288.75         12.887500     0.16 (0.01 %)       0.20 (0.02 %)      1288.11 (99.95 %)   0.29 (0.02 %)
```

### Example (LOWOHA reorder with `--lowoha=true`, console/CSV)
```
Batch_size  Rows  Cols  Iterations  Src_dtype  Dst_dtype  Algo       Scale_granularity  Group_size  Dynamic_quant  Num_threads  Warmup_iters  Total_time(ms)  Avg_time(ms)  
1           1024  1024  100         f32        s8         reference  per_tensor         0           0              0            20            11.11           0.111148      
1           2048  512   50          bf16       s8         reference  per_channel_row    0           0              0            10            18.20           0.364014      
4           1024  768   100         f32        u8         reference  per_group_col      128         0              8            20            820.94          8.209417      
2           2048  1024  100         f32        u8         reference  per_group_row      64          0              4            20            2299.52         22.995202     
```