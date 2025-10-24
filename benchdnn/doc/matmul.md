(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# Matmul Operator

## Overview
This describes all ways to provide input for the matmul benchmark in BenchDNN, including required parameters and example usage for each method.

---

## Usage
Run the matmul benchmark using one of the following input methods:

```sh
./benchdnn/benchdnn --op=matmul [--input_file=inputs.txt] [--input_model_file=model_file] [command-line options] [--ndims=3] [--lowoha]
```

> **Note:**
> - When `--ndims=3` is specified, the benchmark runs in batched matmul (BMM) mode. In this case, `bs` (batch size) must be provided.
> - The `--lowoha` option enables benchmarking for low overhead API. Use `--lowoha` in combination with other matmul options to benchmark low overhead scenarios.

### 1. Input File (`--input_file`)
Provide a file with one configuration per line. Each line should contain:
- `bs` (Batch Size) *(required for BMM, i.e., when `--ndims=3`)*
- `m` (Rows in src)
- `k` (Columns in src / rows in weights)
- `n` (Columns in weights; colon-separated for multi-layer)
- `iters` (Number of iterations)
- `dt` (Data types: src:weights:dst)
- `bias` (true/false)
- `bias_dt` (Data type for bias)
- `post_ops` (Post-operations, e.g., relu, gelu, binary_mul)
- `kernel` (Kernel backend)
- `isTransA` (Transpose flag for src)
- `isTransB` (Transpose flag for weights)
- `warmup_iters` (optional)

**Example usage:**
```sh
./benchdnn/benchdnn --op=matmul --input_file=inputs.txt
```

**Example input file (`inputs.txt`):**

- Single-layer matmul:
  ```
  128, 9216, 4096, 1, f32:f32:f32, true, f32, relu, aocl_blis_blocked, false, false, 30
  128, 9216, 4096, 100, f32:f32:f32, true, f32, relu, aocl_blis_blocked, false, false, 30
  ```
- Multi-layer (pipeline) matmul:
  ```
  768, 3072, 512:256, 100, f32:f32:f32, true, f32, gelu_erf, aocl_blis_blocked, false, false, 30
  4096, 768, 256:3072:512, 100, f32:f32:f32, true, f32, gelu_erf, aocl_blis_blocked, false, false, 30
  ```
- Batched matmul (BMM):
  ```
  100, 100, 3456, 512, 100, f32:f32:f32, true, f32, , aocl_blis, false, false, 20
  ```
  > **Note:** For BMM, ensure you specify `--ndims=3` on the command line and provide `bs` in the input file.

---

### 2. Model File (`--input_model_file`)
Provide a model file with the following parameters:
- `modelname` (Name of the model)
- `bs` (Batch Size) *(required for BMM, i.e., when `--ndims=3`)*
- `m` (Rows in src) *(optional; can be specified in the file or via the `--m` command-line option)*
- `k` (Columns in src / rows in weights)
- `n` (Columns in weights)
- `bias` (true/false)
- `postops` (Post-operations)

Other options (e.g., `iters`, `dt`, etc.) can be provided via command-line arguments.

**Example usage:**
```sh
./benchdnn/benchdnn --op=matmul --input_model_file=../benchdnn/input/matmul/pytorch_hugging_face_bmm.txt --iters=100 --sdt=f32 --ddt=f32 --wdt=f32 --bias_dt=f32 --kernel_name=aocl_blis --isTransA=false --isTransB=false --warmup_iters=100
```

```sh
./benchdnn/benchdnn --op=matmul --input_model_file=../benchdnn/input/matmul/recsys.txt --m=256 --iters=100 --sdt=f32 --ddt=f32 --wdt=f32 --bias_dt=f32 --kernel_name=aocl_blis --isTransA=false --isTransB=false --warmup_iters=100
```

---

### 3. Command-Line Arguments
All configuration parameters can be provided directly via command-line options.

**Example usage:**
```sh
./benchdnn/benchdnn --op=matmul --bs=128 --m=9216 --k=4096 --n=512 --iters=100 --sdt=f32 --ddt=f32 --wdt=f32 --bias=true --bias_dt=f32 --post_ops=relu --kernel_name=aocl_blis --isTransA=false --isTransB=false --warmup_iters=30 --ndims=3
```
> **Note:** For BMM benchmarking, always specify `--ndims=3` and provide `bs`.

---


## Output

The benchmark prints the following for each input:
- Total execution time
- Achieved GFLOPS for matmul operator
- Detailed timing statistics for all iterations of each input, including:
  - Context creation time
  - Operator creation time
  - Operator execution time

Output is printed to the console and also saved to a CSV file named `timings_<current timestamp>.csv` in the `build` directory.

### Example (batched matmul, console/CSV)
```
BS  M     K    N    Iters  Data_type    Bias_Enabled  Bias_dt  PostOp  Kernel_Name  isTransA  isTransB  Warmup_iters  Total_time(ms, all iters)  GFLOPS  Ctx_Creation(ms_%)  Op_Creation(ms_%)  Op_Execution(ms_%)  
2   1024  256  512  10     f32:f32:f32  0                              aocl_blis    0         0         10            223.42                     24.03   0.19 (0.09 %)       0.04 (0.02 %)      223.18 (99.89 %) 
```