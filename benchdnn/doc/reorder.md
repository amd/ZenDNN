(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# Reorder Operator

## Overview
This describes all ways to provide input for the reorder benchmark in BenchDNN, including required parameters and example usage for each method.

---

## Usage
Run the reorder benchmark using the following input methods:

```sh
./install/benchdnn/bin/benchdnn --op=reorder --input_file=reorder_inputs.txt [--lowoha=true/false] [--cache_mode=cold|hot]
```

> **Note:**
> - The `--lowoha` option controls benchmarking for the Low Overhead API (LOWOHA). User can pass either `--lowoha=true` or `--lowoha=false`. If not specified, it is enabled by default.
> - When `--lowoha=false`, the benchmark runs the regular AOCL-based layout reorder.
> - When `--lowoha=true` (default), the benchmark runs the LOWOHA-based quantization/type-conversion reorder.
> - Cache behavior is selected with `--cache_mode` (see [Cache mode](#cache-mode) below). Default is `hot`.

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
- `src_dtype` (Source data type: `f32`, `bf16`, `f16`)
- `dst_dtype` (Destination data type: `s8`, `u8` for quantize cases; for static dequantize cases also `bf16`, `f16`, or `f32` — pick the destination dtype that matches the static reorder direction the library supports. Dynamic quantization only writes `s8` / `u8`.)
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
# FP16 dynamic-quant examples (requires AVX-512 host; FP16-FMA backend
# is auto-selected when AVX512-FP16 ISA is present, F32-FMA otherwise):
1, 2048, 1024, 100, f16, s8, native, per_channel_row, 0, true, 0, 20
1, 4096, 4096, 50,  f16, u8, native, per_tensor,      0, true, 0, 10
4, 1024, 768,  100, f16, u8, native, per_group_col,   128, true, 8, 20
# FP16 static dequant examples (s8/u8 -> f16, dynamic_quant=false). Same
# FMA backend selection rules as the dynamic-quant FP16 path above:
1, 2048, 1024, 100, s8, f16, native, per_tensor,      0, false, 0, 20
1, 4096, 4096, 50,  u8, f16, native, per_channel_row, 0, false, 0, 10
```
---

> **Note (FP16 on either side):** When `src_dtype=f16` (with `dynamic_quant=true` or `dynamic_quant=false`) or `dst_dtype=f16` (static dequant from `s8`/`u8`), the reorder kernel picks between two backends at dispatch time:
> - **F32-FMA** (AVX-512F + F16C): F16C convert on load/store, math in `__m512`.
> - **FP16-FMA** (`__m512h`-native, AVX512-FP16 ISA): full quant / dequant chain in FP16.
>
> Selection is automatic — FP16-FMA when the host has AVX512-FP16 and the library was built with GCC 12+, else F32-FMA. Decided by `can_use_f16_fma_kernel()` in `lowoha_reorder_common.hpp`; no runtime env var. To pin reorder to the F32-accumulating AVX-512 path for reproducibility studies, rebuild with `-DZENDNNL_NATIVE_F32_ACCUM=ON`. The static-reorder per-tensor path (used when the dynamic-quant fast-path doesn't match, and the only path used for the `s8/u8 → f16` dequant direction) follows the same policy. For per-tensor F16 source or F16 destination, this is the only granularity that has dedicated SIMD kernels at the static layer; other granularities fall through to the scalar element-wise loop (same as the BF16/F32 conventions).

> **Note:**
> - The `isInplace` option controls whether the reorder is performed in-place or out-of-place.
> - The `warmup_iters` parameter is optional and can be used to specify the number of warmup iterations before benchmarking.

### Cache mode

Use `--cache_mode=<value>` on the command line. The value is case-insensitive and must be one of `cold`, or `hot`. If omitted, the default is `hot`.

- **`hot`** (default): No CPU cache flush before each measured iteration. Typical steady-state timing.
- **`cold`**: Flushes the CPU cache before each measured iteration so each timed run starts from a cold-cache state.

**Example usage:**

```sh
./install/benchdnn/bin/benchdnn --op=reorder --input_file=reorder_inputs.txt --lowoha=false --cache_mode=cold
./install/benchdnn/bin/benchdnn --op=reorder --input_file=reorder_lowoha_inputs.txt --cache_mode=cold
```

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