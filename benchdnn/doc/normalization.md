(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# Normalization Operator

## Overview
This describes all ways to provide input for the normalization benchmark in BenchDNN, including required parameters and example usage.

---

## Usage
Run the normalization benchmark using the following input method:

```sh
./install/benchdnn/bin/benchdnn --op=normalization --input_file=norm_inputs.txt --lowoha=true
```

> **Note:**
> - Normalization benchmarking is only supported via the LOWOHA (Low Overhead API) path. The `--lowoha=true` flag must be used (it is also the default).
> - Supported normalization types: `layer_norm`, `batch_norm`, `rms_norm`, `fused_add_rms_norm`.

---

### Input File (`--input_file`)
Provide a file with one configuration per line. Lines starting with `#` are comments. Each line should contain:
- `norm_type` (Normalization variant: `layer_norm`, `batch_norm`, `rms_norm`, `fused_add_rms_norm`)
- `shape` (Tensor dimensions separated by `x`, e.g., `2x4096`, `32x64x56x56`)
- `norm_ndims` (Number of trailing dimensions to normalize; must be `0` for `batch_norm`, `1` to `ndims` for others)
- `src_dt:dst_dt` (Source and destination data types, e.g., `f32:f32`, `bf16:bf16`, `bf16:f32`, `f32:bf16`, `f16:f16`, `f16:f32`, `f32:f16`. `f16` cannot be cross-mixed with `bf16` between src and dst.)
- `epsilon` (Numerical stability constant, e.g., `1e-5`)
- `use_scale` (Whether to apply gamma/scale: `true`/`false`)
- `use_shift` (Whether to apply beta/shift: `true`/`false`; ignored for `rms_norm` and `fused_add_rms_norm`)
- `iters` (Number of benchmark iterations)
- `warmup_iters` (optional; defaults to 20% of `iters`)
- `gamma_dt` (optional; data type for gamma: `f32`, `bf16`, or `f16`; defaults to `f32`. May be chosen independently of `src_dt`/`dst_dt`.)
- `beta_dt` (optional; data type for beta: `f32`, `bf16`, or `f16`; defaults to `f32`. May be chosen independently of `src_dt`/`dst_dt`.)
- `algorithm` (optional; backend selection: `none`, `dynamic_dispatch`, `reference`; defaults to `none`)
- `num_threads` (optional; number of threads, `0` = auto/all available; defaults to `0`)
- `isInplace` (optional; `true`/`1` to perform in-place normalization where output overwrites the input buffer, `false`/`0` for out-of-place; defaults to `true`. Requires `src_dt == dst_dt` when in-place.)

**Example usage:**

- LOWOHA mode (default):
```sh
./install/benchdnn/bin/benchdnn --op=normalization --input_file=norm_inputs.txt
```

**Example input file (`norm_inputs.txt`):**
```
layer_norm, 2x4096, 1, f32:f32, 1e-5, true, true, 100, 50, bf16, bf16, none
rms_norm, 8x2048, 1, bf16:f32, 1e-6, true, false, 50, 10, bf16
batch_norm, 32x64x56x56, 0, f32:f32, 1e-5, true, true, 200
fused_add_rms_norm, 4x4096, 1, f32:f32, 1e-6, true, false, 100, 20, f32, f32, dynamic_dispatch, 8
layer_norm, 4x8x512, 2, f32:f32, 1e-5, true, true, 100, 20, bf16, bf16
layer_norm, 2x4096, 1, f32:f32, 1e-5, true, true, 100, 20, f32, f32, none, 0, true
rms_norm, 4x4096, 1, f16:f16, 1e-6, true, false, 100, 20, f16
layer_norm, 2x4096, 1, f16:f32, 1e-5, true, true, 100, 20, f16, f16
# F16-FMA fast path with f32 gamma/beta (require AVX512-FP16):
rms_norm,   8x4096, 1, f16:f16, 1e-6, true, false, 100, 20, f32
layer_norm, 2x4096, 1, f16:f16, 1e-5, true, true,  100, 20, f32, f32
layer_norm, 2x4096, 1, f32:f16, 1e-5, true, true,  100, 20, f16, f32
```
---

> **Note:**
> - For `batch_norm`, `shape` must have at least 2 dimensions (N, C, ...) and `norm_ndims` must be `0`.
> - For `rms_norm` and `fused_add_rms_norm`, the `use_shift` flag is ignored (beta is never applied).
> - `norm_ndims` controls how many trailing dimensions are normalized together. For example, with shape `4x8x512` and `norm_ndims=2`, normalization is performed over the last 2 dimensions (8x512). Setting `norm_ndims` equal to the total number of dimensions normalizes the entire tensor as a single group.
> - The `warmup_iters` parameter is optional and can be used to specify the number of warmup iterations before benchmarking.
> - In-place normalization (`isInplace=true`) requires that `src_dt` and `dst_dt` are the same data type. The output overwrites the input buffer, reducing memory usage. If `isInplace` is `true` (whether explicitly set or defaulted) and `src_dt != dst_dt`, it automatically falls back to out-of-place with a warning.
> - `f16` (input, output, gamma, or beta — whenever the corresponding buffer is actually used) requires **either** AVX512-FP16 ISA on the host **or** a library built with `-DZENDNNL_NATIVE_F32_ACCUM=ON`. With the macro OFF, hosts without AVX512-FP16 return `status_t::isa_unsupported` before any kernel dispatch; the benchmark driver currently logs these as failed runs (the driver does not yet treat `isa_unsupported` as a clean skip), so use an input file matched to your hardware. With the macro ON, the FP32-accumulating AVX-512 kernel handles f16 storage via F16C convert (`_mm512_cvtph_ps` / `_mm512_cvtps_ph`), which any AVX-512 host has — f16 inputs then run end-to-end on non-AVX512-FP16 hardware. The benchmark itself still runs; only AVX-512F (very widely available) is required.
> - **F16-FMA fast path:** On hosts with AVX512-FP16, the native FP16-FMA kernel performs the inner loop in `__m512h` registers (~2x throughput vs. the FP32-accumulating AVX-512 kernel). The eligibility predicate depends on the norm type:
>     - `rms_norm` — `src_dt`/`dst_dt` ∈ `{f16, f32}` with at least one `f16`, and `gamma_dt` ∈ `{f16, f32}`. `beta_dt` is irrelevant (RMSNorm never reads beta).
>     - `layer_norm` — same as `rms_norm`, plus `beta_dt` ∈ `{f16, f32}` only when `use_shift=true` (if `use_shift=false`, `beta_dt` is irrelevant).
>     - `fused_add_rms_norm` — strict `src_dt = dst_dt = gamma_dt = f16` (residual aliases src in place and must share the f16 storage layout). Mixed `(src, dst)` falls through to the FP32 path. `beta_dt` is irrelevant.
>     - `batch_norm` — always uses the reference kernel; the F16-FMA path does not apply.
>
>   `bf16` in a *checked* operand always disqualifies the FP16-FMA path for that call. To force the FP32 path library-wide for A/B comparisons on an AVX512-FP16 host, build with `-DZENDNNL_NATIVE_F32_ACCUM=ON`.

## Output

The benchmark prints the following for each input:
- Total execution time
- Average time per iteration

Output is printed to the console and also saved to a CSV file named `timings_<current timestamp>.csv` in the `build` directory.

### Example (console/CSV)
```
Norm_Type           Shape         Norm_Ndims  Data_Type  Epsilon  Use_Scale  Use_Shift  Iterations  Warmup_Iters  Gamma_DT  Beta_DT  Algorithm         Num_Threads  Inplace  Total_time(ms) (all iters)  Avg_time(ms)
layer_norm          2x4096        1           f32:f32    1e-05    true       true       100         50            bf16      bf16     none              auto         true     5.42                        0.054200
rms_norm            8x2048        1           bf16:f32   1e-06    true       false      50          10            bf16      f32      none              auto         false    3.18                        0.063600
batch_norm          32x64x56x56   0           f32:f32    1e-05    true       true       200         40            f32       f32      none              auto         true     1204.56                     6.022800
fused_add_rms_norm  4x4096        1           f32:f32    1e-06    true       false      100         20            f32       f32      dynamic_dispatch  8            true     8.91                        0.089100
layer_norm          4x8x512       2           f32:f32    1e-05    true       true       100         20            bf16      bf16     none              auto         true     2.74                        0.027400
layer_norm          2x4096        1           f32:f32    1e-05    true       true       100         20            f32       f32      none              auto         true     5.10                        0.051000
```
