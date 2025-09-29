(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# Reorder Operator

## Overview
This describes all ways to provide input for the reorder benchmark in BenchDNN, including required parameters and example usage for each method.

---

## Usage
Run the reorder benchmark using one of the following input methods:

```sh
./benchdnn/benchdnn --op=reorder --input_file=reorder_inputs.txt
```

---

### Input File (`--input_file`)
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
./benchdnn/benchdnn --op=reorder --input_file=reorder_inputs.txt
```

**Example input file (`reorder_inputs.txt`):**
```
1024, 1024, 100, f32, aocl, true, 20
2048, 512, 50, bf16, aocl, false, 10
```
---

> **Note:**
> - The `isInplace` option controls whether the reorder is performed in-place or out-of-place.
> - The `warmup_iters` parameter is optional and can be used to specify the number of warmup iterations before benchmarking.

## Output

The benchmark prints the following for each input:
- Total execution time
- Achieved GFLOPS for matmul operator
- Detailed timing statistics for all iterations of each input, including:
  - Context creation time
  - Operator creation time
  - Operator execution time

Output is printed to the console and also saved to a CSV file named `timings_<current timestamp>.csv` in the `build` directory.

### Example (reorder, console/CSV)
```
Rows  Cols  Iterations  Data_type  Kernel_Name  In-place  Warmup_iters  Total_time(ms)  Ctx_Creation(ms_%)  Op_Creation(ms_%)  Op_Execution(ms_%)  Others(ms_%)
3072  768   100         f32        aocl         1         30            1318.81         0.16 (0.01 %)       0.21 (0.02 %)      1318.12 (99.95 %)   0.31 (0.02 %)
768   3072  100         f32        aocl         1         30            1288.75         0.16 (0.01 %)       0.20 (0.02 %)      1288.11 (99.95 %)   0.29 (0.02 %)
```