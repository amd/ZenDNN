(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# Embag Operator

## Overview
This describes all ways to provide input for the embag benchmark in BenchDNN, including required parameters and example usage for each method.

---

## Usage
Run the embag benchmark using the following input methods:

```sh
./install/benchdnn/bin/benchdnn --op=embag --input_file=embag_inputs.txt
```

---

### Input File (`--input_file`)
Provide a file with one configuration per line. Each line should contain:
- `num_embeddings` (Size of the dictionary of embeddings)
- `embedding_dims` (Size of each embedding vector)
- `num_bags` (Number of bags used in embagbag operation)
- `num_indices` (Number of indices across all bags)
- `algo` (Algorithm used for embag computation (e.g., "sum", "mean", "max"))
- `iters` (Number of iterations)
- `dt` (Data types: src:dst)
- `padding_index` (Index used for padding)
- `include_last_offset` (Flag indicating whether to include the last offset in the offsets array)
- `is_weights` (Flag indicating if weights are used for each index in the embag)
- `scatter_stride` (Scatter Stride used when scattering embeddings in memory )
- `warmup_iters` (optional)

**Example usage:**
```sh
./install/benchdnn/bin/benchdnn --op=embag --input_file=embag_inputs.txt
```

**Example input file (`embag_inputs.txt`):**
```
20, 6, 5, 15, sum, 100, f32:f32, -1, true, true, -1, 50
40, 8, 6, 20, mean, 100, f32:bf16, -1, true, true, -1, 50
```
---

> **Note:**
> - The `warmup_iters` parameter is optional and can be used to specify the number of warmup iterations before benchmarking.

## Output

The benchmark prints the following for each input:
- Total execution time
- Detailed timing statistics for all iterations of each input, including:
  - Context creation time
  - Operator creation time
  - Operator execution time

Output is printed to the console and also saved to a CSV file named `timings_<current timestamp>.csv` in the `build` directory.

### Example (embag, console/CSV)
```
Num_Embeddings  Embedding_Dims  Num_Bags  Num_Indices  Algo  Iterations  Data_type  Padding_Index  Include_Last_Offset  Is_Weights  Scatter_Stride  Warmup_iters  Total_time(ms) (all iters)  Ctx_Creation(ms_%)  Op_Creation(ms_%)  Op_Execution(ms_%)
20              6               5         15           sum   100         f32:f32    -1             1                    1           -1              50            1094.51                     0.20 (0.02 %)       0.09 (0.01 %)      1094.22 (99.97 %)
40              8               6         20           mean  100         f32:bf16   -1             1                    1           -1              50            1060.42                     0.24 (0.02 %)       0.10 (0.01 %)      1060.08 (99.97 %)
```