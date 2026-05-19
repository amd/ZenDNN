(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# Matmul Operator

## Overview
This describes all ways to provide input for the matmul benchmark in BenchDNN, including required parameters and example usage for each method.

---

## Usage
Run the matmul benchmark using one of the following input methods:

```sh
./install/benchdnn/bin/benchdnn --op=matmul [--input_file=inputs.txt] [--input_model_file=model_file] [command-line options] [--ndims=3] [--lowoha=true/false] [--cache_mode=cold|warm|hot] [--num_weight_buffers=<n>]
```

> **Note:**
> - When `--ndims=3` is specified, the benchmark runs in batched matmul (BMM) mode. In this case, `bs` (batch size) must be provided.
> - The `--lowoha` option controls benchmarking for low overhead API. User can pass either `--lowoha=true` or `--lowoha=false`. If not specified, it is enabled by default.
> - Algorithm selection can be overridden via environment variables `ZENDNNL_MATMUL_ALGO` (matmul) and `ZENDNNL_BMM_ALGO` (BMM).
> - Cache behavior for matmul is selected at runtime with `--cache_mode` (see [Cache mode](#5-cache-mode) below). Default is `hot`.
> - `--num_weight_buffers` is only used when `--cache_mode=warm` (see [Cache mode](#5-cache-mode)).

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
- `post_op_dt` (Data type for binary post-operations, e.g. `f32`, `bf16`; used when `post_ops` include binary ops; defaults to `f32`. Can be empty when no binary post-ops.)
- `kernel` (Kernel backend)
- `is_weights_const` (Whether weights are constant: `true`/`false` or `1`/`0`. If empty: default is `true` for 2D-Matmul, `false` for BMM.)
- `isTransA` (Transpose flag for src)
- `isTransB` (Transpose flag for weights)
- `alpha` (Alpha parameter for the matmul operation)
- `beta` (Beta parameter for the matmul operation)
- `weight_scale_granularity` (Weight scale granularity: `per-channel`/`channel`, `per-group`/`group`, `per-tensor`/`tensor`; defaults to per-channel)
- `weight_group_size` (Group size for per-group weight scaling; ignored otherwise. If empty or 0, defaults to K. Must be even; odd values fall back to per-channel.)
- `weight_scale_dt` (Data type for the weight scale tensor, e.g., `f32` or `bf16`; defaults to `f32`)
- `warmup_iters` (optional)
- `src_dynamic_quant` (optional; `true`/`false` or `1`/`0`. Enables dynamic source quantization on the LOWOHA path. Defaults to `false`.)
- `src_scale_granularity` (optional; `per-tensor` / `per-token` / `per-group`. Defaults to `per-tensor`.)
- `src_group_size` (optional; K-direction group size for `per-group`. Falls back to per-token if `K % src_group_size != 0`. Defaults to `0`.)
- `src_scale_dt` (optional; data type of the source scale tensor, `f32` or `bf16`. Defaults to `f32`.)

**Example usage:**
```sh
./install/benchdnn/bin/benchdnn --op=matmul --input_file=inputs.txt
```

**Example input file (`inputs.txt`):**

- Single-layer matmul:
  ```
  128, 9216, 4096, 1, f32:f32:f32, true, f32, relu, , aocl_dlp_blocked, false, false, false, 1.0, 0.0, per-channel, , f32, 30
  128, 9216, 4096, 100, bf16:s4:bf16, true, f32, binary_add:relu, f32, aocl_dlp_blocked, true, false, false, 2.0, 1.0, per-group, 256, bf16, 30
  ```
- Multi-layer (pipeline) matmul:
  ```
  768, 3072, 512:256, 100, f32:f32:f32, true, f32, gelu_erf, , aocl_dlp_blocked, true, false, false, 1.0, 0.0, per-channel, , f32, 30
  4096, 768, 256:3072:512, 100, bf16:s4:f32, true, f32, gelu_erf, , aocl_dlp_blocked, true, false, false, 1.5, 0.0, per-group, 128, f32, 30
  ```
- INT8 dynamic source quantization (W8A8, symmetric):
  ```
  # Per-token activation scales, bf16 src, s8 weights
  4096, 4096, 4096, 100, bf16:s8:bf16, false, f32, , , aocl_dlp, true, false, false, 1.0, 0.0, per-channel, , f32, 30, true, per-token, 0, f32

  # Per-group activation scales (group of 128 along K), bf16 scale dtype
  4096, 4096, 4096, 100, bf16:s8:bf16, false, f32, , , aocl_dlp, true, false, false, 1.0, 0.0, per-channel, , f32, 30, true, per-group, 128, bf16
  ```
- Batched matmul (BMM):
  ```
  100, 100, 3456, 512, 100, f32:f32:f32, true, f32, , , aocl_dlp, true, false, false, 1.0, 0.0, , , f32, 20
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
- `postops` (Post-operations)
- `bias` (true/false)

Other options (e.g., `iters`, `dt`, etc.) can be provided via command-line arguments.

**Example usage:**
```sh
./install/benchdnn/bin/benchdnn --op=matmul --input_model_file=../benchdnn/input/matmul/pytorch_hugging_face_bmm.txt --iters=100 --sdt=f32 --ddt=f32 --wdt=f32 --bias_dt=f32 --kernel_name=aocl_dlp --isTransA=false --isTransB=false --warmup_iters=100
```

```sh
./install/benchdnn/bin/benchdnn --op=matmul --input_model_file=../benchdnn/input/matmul/recsys.txt --m=256 --iters=100 --sdt=f32 --ddt=f32 --wdt=f32 --bias_dt=f32 --kernel_name=aocl_dlp --isTransA=false --isTransB=false --warmup_iters=100
```

---

### 3. Command-Line Arguments
All configuration parameters can be provided directly via command-line options.

**Example usage:**
```sh
./install/benchdnn/bin/benchdnn --op=matmul --bs=128 --m=9216 --k=4096 --n=512 --iters=100 --sdt=f32 --ddt=f32 --wdt=f32 --bias=true --bias_dt=f32 --post_ops=relu --post_op_dt=f32 --kernel_name=aocl_dlp --is_weights_const=true --isTransA=false --isTransB=false --alpha=1.0 --beta=0.0 --weight_scale_granularity=per-group --weight_group_size=256 --weight_scale_dt=bf16 --warmup_iters=30 --ndims=3
```
> **Note:** For BMM benchmarking, always specify `--ndims=3` and provide `bs`.

**Dynamic source quantization (W8A8, symmetric)** can be enabled with these additional flags (LOWOHA, 2D only):

- `--dynamic_quant=true|false`
- `--src_scale_granularity=per-tensor|per-token|per-group`
- `--src_group_size=<int>` (only used with `per-group`; falls back to per-token if K is not divisible).
  The source and weight group sizes are always kept in sync: setting `--src_group_size` also sets
  `--weight_group_size` (and vice versa), so you only need to specify one. If both are given on the
  CLI, the last one wins; if both appear in an input-file row and differ, a warning is printed and
  both are forced to the weight value.
- `--src_scale_dt=f32|bf16`

```sh
# Per-token W8A8
./install/benchdnn/bin/benchdnn --op=matmul --m=4096 --k=4096 --n=4096 --iters=100 \
  --sdt=bf16 --wdt=s8 --ddt=bf16 --kernel_name=aocl_dlp \
  --dynamic_quant=true --src_scale_granularity=per-token --src_scale_dt=f32

# Per-group W8A8 (K=4096, group=128 -> 32 groups along K)
./install/benchdnn/bin/benchdnn --op=matmul --m=4096 --k=4096 --n=4096 --iters=100 \
  --sdt=bf16 --wdt=s8 --ddt=bf16 --kernel_name=aocl_dlp \
  --dynamic_quant=true --src_scale_granularity=per-group --src_group_size=128 --src_scale_dt=bf16
```

Constraints:
- LOWOHA path only (`--lowoha=true`, the default).
- 2D only (`--ndims=2`); BMM and grp_matmul are not supported in this mode.
- `src` must be `bf16` or `f32`; `wei` must be `s8`. Compute target is fixed to `s8` (symmetric).
- Source zero-points are not used.

---

### 4. Algorithm selection via environment variables

The matmul benchmark reads algorithm selection from environment variables. When set, these override the `kernel` (input file) or `--kernel_name` (command line) for the backend used at runtime.

If the variable is unset, algorithm selection falls back to the `kernel` / `--kernel_name` from the input file or command line.

**Example usage:**
```sh
export ZENDNNL_MATMUL_ALGO=4
./install/benchdnn/bin/benchdnn --op=matmul --m=9216 --k=4096 --n=512 --iters=100 --sdt=f32 --ddt=f32 --wdt=f32
export ZENDNNL_BMM_ALGO=auto
./install/benchdnn/bin/benchdnn --op=matmul --ndims=3 --input_file=bmm_inputs.txt
```

---

### 5. Cache mode

Use `--cache_mode=<value>` on the command line. The value is case-insensitive and must be one of `cold`, `warm`, or `hot`. If omitted, the default is `hot`.

- **`hot`** (default): Single weight buffer; no extra cache flush between measured iterations. Typical “steady state” timing.
- **`cold`**: Flushes the CPU cache before each measured iteration so each timed run starts from a cold-cache state.
- **`warm`**: Allocates a pool of weight tensors sized from the detected cache and rotates which buffer is used across iterations (so the same weight bytes are not reused every iteration). The cache is flushed once before the warmup loop; measured iterations use the rotating buffers without a per-iteration flush.

#### `--num_weight_buffers` (warm mode only)

Use `--num_weight_buffers=<n>` with `--cache_mode=warm` to control how many distinct weight tensor copies are allocated and rotated. In `cold` and `hot` modes this option has no effect.

**Example usage:**
```sh
./install/benchdnn/bin/benchdnn --op=matmul --m=9216 --k=4096 --n=512 --iters=100 --cache_mode=cold --sdt=f32 --ddt=f32 --wdt=f32
./install/benchdnn/bin/benchdnn --op=matmul --ndims=3 --input_file=bmm_inputs.txt --cache_mode=warm
./install/benchdnn/bin/benchdnn --op=matmul --m=9216 --k=4096 --n=512 --iters=100 --cache_mode=warm --num_weight_buffers=8 --sdt=f32 --ddt=f32 --wdt=f32
```

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
BS  M     K    N    Iters  Data_type    Bias_Enabled  Bias_dt  PostOp  PostOp_dt  Kernel_Name  isWeightsConst  isTransA  isTransB  Alpha     Beta      Weight_Scale_Granularity  Weight_Group_Size  Weight_Scale_dt  Warmup_iters  Total_time(ms, all iters)  Avg_time(ms)  GFLOPS  Ctx_Creation(ms_%)  Op_Creation(ms_%)  Op_Execution(ms_%)  
2   1024  256  512  10     f32:f32:f32  1             f32      relu             aocl_dlp     1               0         0         1.000000  0.000000  group                       256                 bf16            30            220.46                     22.05         24.35   0.05 (0.02 %)       0.02 (0.01 %)      220.39 (99.97 %)
```