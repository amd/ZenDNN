
(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# LOWOHA Normalization Operator

## Overview

The **LOWOHA Normalization Operator** is a high-performance, low-overhead normalization operator designed for **latency-sensitive inference workloads**. It provides a unified API for four normalization variants through a single entry point with minimal execution overhead.

LOWOHA Normalization provides a **function-based interface** optimized for:
- Minimal execution overhead
- FP32, BF16, and F16 data types (input, output, gamma, beta)
- Multiple normalization variants via a single API
- AVX-512 and AVX-512-FP16 vectorized kernels for LayerNorm, RMSNorm, and FusedAddRMSNorm
- Direct control over execution parameters

## Supported Normalization Types

### LayerNorm

Normalizes each sample across the normalized dimensions. Mean and variance are computed on-the-fly from the input.

$$
y_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
$$

Where:
- $\mu = \frac{1}{N}\sum_i x_i$ is the mean over the normalized dimensions
- $\sigma^2 = \frac{1}{N}\sum_i (x_i - \mu)^2$ is the variance
- $\gamma$ is the learned scale, $\beta$ is the learned shift

**Use case:** Transformer encoder models (BERT, GPT pre-norm layers).

### RMSNorm

Root Mean Square normalization. A simplified variant of LayerNorm that omits mean subtraction and the shift parameter.

$$
y_i = \gamma_i \cdot \frac{x_i}{\sqrt{\frac{1}{N}\sum_j x_j^2 + \epsilon}}
$$

**Use case:** LLaMA, Mistral, and other modern LLM architectures.

### FusedAddRMSNorm

Fuses a residual addition with RMSNorm in a single kernel call, reducing memory traffic. The residual buffer is updated in-place.

$$
\text{residual}_i \mathrel{+}= \text{input}_i
$$

$$
y_i = \gamma_i \cdot \frac{\text{residual}_i}{\sqrt{\frac{1}{N}\sum_j \text{residual}_j^2 + \epsilon}}
$$

**Use case:** Transformer decoder blocks (LLaMA, Mistral) where each sub-layer adds to a running residual stream and then normalizes.

### BatchNorm (Inference)

Normalizes per-channel using pre-computed running statistics from training.

$$
y_{n,c,s} = \gamma_c \cdot \frac{x_{n,c,s} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c
$$

Where $\mu_c$ and $\sigma_c^2$ are the pre-computed running mean and variance for channel $c$.

**Use case:** CNN inference (ResNet, VGG, etc.).

## Core API: `normalization_direct`

The primary interface for all normalization variants:

```cpp
status_t normalization_direct(
  const void *input,         // Input tensor (read-only)
  void       *output,        // Output tensor (same shape as input)
  const void *gamma,         // Scale parameters (read-only, may be nullptr)
  const void *beta,          // Shift parameters (read-only, may be nullptr)
  const void *running_mean,  // Pre-computed mean (BatchNorm only, may be nullptr)
  const void *running_var,   // Pre-computed variance (BatchNorm only, may be nullptr)
  void       *residual,      // Residual buffer (FusedAddRMSNorm only, may be nullptr)
  norm_params &params        // Normalization parameters
);
```
Current dispatch logic (see the [F16 FMA Mode](#f16-fma-mode-native-avx512-fp16) section for the full per-norm-type matrix):
- `LAYER_NORM` / `RMS_NORM` / `FUSED_ADD_RMS_NORM` → AVX-512-FP16 (`__m512h`) when eligible, otherwise the FP32-accumulating AVX-512 kernel.
- `BATCH_NORM` → Reference (scalar) kernel (`normalization_reference_wrapper`).
- For non-F16 configurations, any path falls back to the reference kernel if no AVX-512 hardware is available. F16 configurations (where any actually-used buffer is `f16`) require AVX512-FP16 ISA up-front: `normalization_direct` returns `status_t::isa_unsupported` and does **not** fall back to the reference kernel — see the [Error Handling](#error-handling) section.

## Parameters Structure

### `norm_params`

The main configuration structure. The caller must set `batch`, `norm_size`, and (for BatchNorm) `num_channels` directly. The kernel treats the input as a logically 2-D `[batch, norm_size]` matrix and normalizes each row independently.

```cpp
struct norm_params {
  // --- Normalization variant ---
  norm_type_t norm_type;          // Which normalization to apply (default: NONE)

  // --- Flattened dimensions (set by caller) ---
  uint64_t batch;                 // Product of all outer (non-normalized) dims (default: 0)
  uint64_t norm_size;             // Product of all normalized (trailing) dims (default: 0)
  uint64_t num_channels;          // Channel count C (BatchNorm only, default: 0)

  // --- Normalization parameters ---
  float epsilon;                  // Numerical stability constant (default: 1e-5)
  bool use_scale;                 // Whether to apply gamma (default: false)
  bool use_shift;                 // Whether to apply beta; ignored by RMSNorm (default: false)

  // --- Data types ---
  data_type_t src_dt;             // Source data type (default: none)
  data_type_t dst_dt;             // Destination data type (default: none)
  data_type_t gamma_dt;           // Gamma (scale) parameter data type (default: f32)
  data_type_t beta_dt;            // Beta (shift) parameter data type (default: f32)

  // --- Backend selection ---
  norm_algo_t algorithm;          // Selected algorithm / backend (default: none)

  int32_t num_threads;            // Number of threads, 0 = auto (default: 0)

  // --- Internal dispatch breadcrumb (do not set from caller code) ---
  data_type_t accum_type;
};
```

**Dimension flattening rules:**

The caller is responsible for flattening tensor dimensions before calling `normalization_direct`. The total element count must equal `batch * norm_size` (or `batch * num_channels * norm_size` for BatchNorm).

| Tensor Shape | Norm Dims | `batch` | `norm_size` | `num_channels` |
|---|---|---|---|---|
| `[B, D]` | last 1 | `B` | `D` | — |
| `[B, S, D]` | last 1 | `B * S` | `D` | — |
| `[B, S, H, D]` | last 2 | `B * S` | `H * D` | — |
| `[N, C, H, W]` (BatchNorm) | per-channel | `N` | `H * W` | `C` |

### Normalization Types

```cpp
enum class norm_type_t : int {
  NONE               = -1,  // Not specified
  LAYER_NORM         =  0,  // Layer Normalization
  BATCH_NORM         =  1,  // Batch Normalization
  RMS_NORM           =  2,  // Root Mean Square Normalization
  FUSED_ADD_RMS_NORM =  3   // Fused Add + RMS Normalization
};
```

### Data Type Support

**Input / Output (`src_dt`, `dst_dt`):**

| Input Type | Output Type |
|------------|-------------|
| FP32       | FP32        |
| BF16       | BF16        |
| BF16       | FP32        |
| FP32       | BF16        |
| F16        | F16         |
| F16        | FP32        |
| FP32       | F16         |

> **Note:** F16 cannot be cross-mixed with BF16 between `src_dt` and `dst_dt`
> (e.g. `F16 → BF16` or `BF16 → F16` is rejected at validation). BF16 and
> non-native-F16 paths perform normalization compute in FP32 internally, while
> native AVX512-FP16 F16 kernels accumulate and perform elementwise arithmetic
> in `__m512h`; type conversion still occurs at the load/store boundary.

**Gamma / Beta (`gamma_dt`, `beta_dt`):**

| Supported Types  | Default |
|------------------|---------|
| FP32, BF16, F16  | FP32    |

`gamma_dt` and `beta_dt` are independent of `src_dt` / `dst_dt` and may be
chosen freely from the supported list.

**BatchNorm running mean / variance:** FP32 only (cast to `const float *` internally).

### F16 FMA Mode (Native AVX512-FP16)

When the full eligibility matrix below is satisfied — i.e., the norm type is `LAYER_NORM`, `RMS_NORM`, or `FUSED_ADD_RMS_NORM`; at least one of `src_dt`/`dst_dt` is F16; the actually-used gamma/beta dtypes are F16/F32 (BF16 gamma/beta force the FP32 path); the host CPU exposes the AVX512-FP16 ISA; and the build was not configured with `-DZENDNNL_NATIVE_F32_ACCUM=ON` — the dispatch routes the call to a native FP16-FMA kernel that performs the inner loop in `__m512h` registers (32 lanes per register, ~2x throughput over the FP32-accumulating AVX-512 path). Storage-side conversions (F16↔F32) happen only at the load/store boundary; the FMA chain itself stays in FP16. `BATCH_NORM` is never eligible (it always uses the reference kernel).

**Dispatch matrix** for `LAYER_NORM` and `RMS_NORM`:

| `(src_dt, dst_dt)` | `gamma_dt` | `beta_dt` (LayerNorm + `use_shift`) | Kernel chosen |
|---|---|---|---|
| `(f16, f16)` / `(f16, f32)` / `(f32, f16)` | `f16` or `f32` | `f16` or `f32` | **AVX512-FP16** |
| as above, but `gamma_dt = bf16` or `beta_dt = bf16` | bf16 | bf16 | AVX-512 (FP32-accumulating) |
| `(f32, f32)` (no F16 anywhere) | any | any | AVX-512 (FP32-accumulating) |
| `(bf16, bf16)` / `(bf16, f32)` / `(f32, bf16)` | any | any | AVX-512 (FP32-accumulating) |
| `(f16, bf16)` / `(bf16, f16)` | any | any | **Rejected** — `normalization_direct` returns `status_t::failure` (F16↔BF16 cross-mixing between src and dst is not supported) |

`FUSED_ADD_RMS_NORM` keeps a stricter gate: `src_dt == dst_dt == gamma_dt == f16` (the residual buffer aliases src and is read-modify-written in place, so it must share the f16 storage layout). Mixed-dtype fused-add falls through to the AVX-512 (FP32) kernel.

`BATCH_NORM` always uses the reference kernel (no AVX-512 BatchNorm fast path).

**Build-time opt-out:** Pass `-DZENDNNL_NATIVE_F32_ACCUM=ON` to CMake to disable the native AVX512-FP16 fast path for the LayerNorm, RMSNorm, and FusedAddRMSNorm dispatches in the rows of the table above marked **AVX512-FP16**. With this flag set, those calls take the FP32-accumulating AVX-512 kernel instead — useful for A/B precision comparisons and numerical reproducibility studies. Two side effects of the flag:

- It also **enables f16 inputs on hosts without AVX-512-FP16**: the FP32-accumulating kernel handles f16 storage via F16C convert (`_mm512_cvtph_ps` / `_mm512_cvtps_ph`), which any AVX-512 host has. Without the flag, `normalization_direct` returns `status_t::isa_unsupported` early for any call that uses an f16 buffer on a non-AVX-512-FP16 host.
- It does **not** affect BatchNorm (always reference kernel) or `bf16`-gamma/beta combos (already on the FP32 path).

The same switch also covers the F16 path in the embedding-bag operator.

### Parameter Requirements by Norm Type

| Parameter | LayerNorm | RMSNorm | FusedAddRMSNorm | BatchNorm |
|-----------|-----------|---------|-----------------|-----------|
| `norm_type` | `LAYER_NORM` | `RMS_NORM` | `FUSED_ADD_RMS_NORM` | `BATCH_NORM` |
| `batch` | required | required | required | required (`N`) |
| `norm_size` | required | required | required | required (`H*W`) |
| `num_channels` | not used | not used | not used | required (`C`) |
| `gamma` | optional | optional | optional | optional |
| `beta` | optional | nullptr | nullptr | optional |
| `running_mean` | nullptr | nullptr | nullptr | required (FP32) |
| `running_var` | nullptr | nullptr | nullptr | required (FP32) |
| `residual` | nullptr | nullptr | required | nullptr |
| `use_scale` | true/false | true/false | true/false | true/false |
| `use_shift` | true/false | not used | not used | true/false |

## Usage Examples

### Example 1: RMSNorm (FP32)

```cpp
int rms_norm_example() {
  using namespace zendnnl::lowoha::normalization;

  const uint64_t batch      = 4;
  const uint64_t hidden_dim = 4096;
  const uint64_t total_size = batch * hidden_dim;

  std::vector<float> input(total_size);
  std::vector<float> gamma(hidden_dim, 1.0f);
  std::vector<float> output(total_size, 0.0f);

  // Fill input with data ...

  norm_params params;
  params.batch      = batch;
  params.norm_size  = hidden_dim;
  params.norm_type  = norm_type_t::RMS_NORM;
  params.src_dt     = data_type_t::f32;
  params.dst_dt     = data_type_t::f32;
  params.epsilon    = 1e-6f;
  params.use_scale  = true;

  status_t status = normalization_direct(
      input.data(), output.data(),
      gamma.data(), /*beta=*/nullptr,
      /*running_mean=*/nullptr, /*running_var=*/nullptr,
      /*residual=*/nullptr, params);

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 2: RMSNorm on 3D Tensor (Transformer)

For a 3D tensor `[batch, seq_len, hidden_dim]` normalized over the last dimension, flatten the outer dimensions into `params.batch`:

```cpp
int rms_norm_3d_example() {
  using namespace zendnnl::lowoha::normalization;

  const uint64_t batch      = 2;
  const uint64_t seq_len    = 512;
  const uint64_t hidden_dim = 4096;
  const uint64_t total_size = batch * seq_len * hidden_dim;

  std::vector<float> input(total_size);
  std::vector<float> gamma(hidden_dim, 1.0f);
  std::vector<float> output(total_size, 0.0f);

  norm_params params;
  params.batch      = batch * seq_len;   // flatten outer dims
  params.norm_size  = hidden_dim;        // normalize over hidden_dim
  params.norm_type  = norm_type_t::RMS_NORM;
  params.src_dt     = data_type_t::f32;
  params.dst_dt     = data_type_t::f32;
  params.epsilon    = 1e-6f;
  params.use_scale  = true;

  status_t status = normalization_direct(
      input.data(), output.data(),
      gamma.data(), nullptr,
      nullptr, nullptr,
      nullptr, params);

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 3: FusedAddRMSNorm (Transformer Decoder Block)

```cpp
int fused_add_rms_norm_example() {
  using namespace zendnnl::lowoha::normalization;

  const uint64_t batch      = 2;
  const uint64_t seq_len    = 512;
  const uint64_t hidden_dim = 4096;
  const uint64_t total_size = batch * seq_len * hidden_dim;

  // sublayer_output is the output of the attention or FFN sub-layer
  std::vector<float> sublayer_output(total_size);
  // residual carries the running residual stream across layers
  std::vector<float> residual(total_size);
  std::vector<float> gamma(hidden_dim, 1.0f);
  std::vector<float> hidden_states(total_size, 0.0f);

  // Fill sublayer_output and residual with data ...

  norm_params params;
  params.batch      = batch * seq_len;
  params.norm_size  = hidden_dim;
  params.norm_type  = norm_type_t::FUSED_ADD_RMS_NORM;
  params.src_dt     = data_type_t::f32;
  params.dst_dt     = data_type_t::f32;
  params.epsilon    = 1e-6f;
  params.use_scale  = true;

  status_t status = normalization_direct(
      sublayer_output.data(), hidden_states.data(),
      gamma.data(), /*beta=*/nullptr,
      /*running_mean=*/nullptr, /*running_var=*/nullptr,
      residual.data(), params);

  // After the call:
  //   residual[i]      == old_residual[i] + sublayer_output[i]
  //   hidden_states[i] == gamma[i] * residual[i] / rms

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 4: LayerNorm (FP32)

```cpp
int layer_norm_example() {
  using namespace zendnnl::lowoha::normalization;

  const uint64_t batch      = 4;
  const uint64_t hidden_dim = 768;
  const uint64_t total_size = batch * hidden_dim;

  std::vector<float> input(total_size);
  std::vector<float> gamma(hidden_dim, 1.0f);
  std::vector<float> beta(hidden_dim, 0.0f);
  std::vector<float> output(total_size, 0.0f);

  norm_params params;
  params.batch      = batch;
  params.norm_size  = hidden_dim;
  params.norm_type  = norm_type_t::LAYER_NORM;
  params.src_dt     = data_type_t::f32;
  params.dst_dt     = data_type_t::f32;
  params.epsilon    = 1e-5f;
  params.use_scale  = true;
  params.use_shift  = true;

  status_t status = normalization_direct(
      input.data(), output.data(),
      gamma.data(), beta.data(),
      nullptr, nullptr,
      nullptr, params);

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 5: RMSNorm (BF16 Input with BF16 Gamma)

```cpp
int rms_norm_bf16_example() {
  using namespace zendnnl::lowoha::normalization;

  const uint64_t batch      = 2;
  const uint64_t seq_len    = 512;
  const uint64_t hidden_dim = 4096;
  const uint64_t total_size = batch * seq_len * hidden_dim;

  // BF16 buffers stored as int16_t
  std::vector<int16_t> input(total_size);
  std::vector<int16_t> gamma(hidden_dim);
  std::vector<int16_t> output(total_size, 0);

  // Fill input and gamma with BF16 data ...

  norm_params params;
  params.batch      = batch * seq_len;
  params.norm_size  = hidden_dim;
  params.norm_type  = norm_type_t::RMS_NORM;
  params.src_dt     = data_type_t::bf16;
  params.dst_dt     = data_type_t::bf16;
  params.gamma_dt   = data_type_t::bf16;  // gamma buffer is BF16
  params.epsilon    = 1e-6f;
  params.use_scale  = true;

  status_t status = normalization_direct(
      input.data(), output.data(),
      gamma.data(), /*beta=*/nullptr,
      /*running_mean=*/nullptr, /*running_var=*/nullptr,
      /*residual=*/nullptr, params);

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 6: BatchNorm Inference

```cpp
int batch_norm_inference_example() {
  using namespace zendnnl::lowoha::normalization;

  const uint64_t N = 2, C = 64, H = 56, W = 56;
  const uint64_t total_size = N * C * H * W;

  std::vector<float> input(total_size);
  std::vector<float> gamma(C, 1.0f);
  std::vector<float> beta(C, 0.0f);
  std::vector<float> running_mean(C, 0.0f);
  std::vector<float> running_var(C, 1.0f);
  std::vector<float> output(total_size, 0.0f);

  norm_params params;
  params.batch        = N;
  params.num_channels = C;
  params.norm_size    = H * W;
  params.norm_type    = norm_type_t::BATCH_NORM;
  params.src_dt       = data_type_t::f32;
  params.dst_dt       = data_type_t::f32;
  params.epsilon      = 1e-5f;
  params.use_scale    = true;
  params.use_shift    = true;

  status_t status = normalization_direct(
      input.data(), output.data(),
      gamma.data(), beta.data(),
      running_mean.data(), running_var.data(),
      /*residual=*/nullptr, params);

  return (status == status_t::success) ? 0 : -1;
}
```

## Buffer Requirements

### Residual Buffer (FusedAddRMSNorm)

The `residual` parameter is unique to `FUSED_ADD_RMS_NORM`:

- **Required:** Must be non-null when `norm_type == FUSED_ADD_RMS_NORM`
- **Element type:** Same as `params.src_dt` (f32, bf16, or f16)
- **Access:** Read-write (modified in-place)
- **After the call:** `residual[i] = old_residual[i] + input[i]`

For all other norm types, pass `nullptr` for the residual parameter.

### Gamma and Beta

- **Gamma (scale):** Shape `[norm_size]` for LayerNorm/RMSNorm/FusedAddRMSNorm, `[num_channels]` for BatchNorm. Supports FP32 (default), BF16, or F16 — set `params.gamma_dt` to match the buffer's element type. Pass `nullptr` if `use_scale == false`.
- **Beta (shift):** Shape `[norm_size]` for LayerNorm, `[num_channels]` for BatchNorm. Unused by RMSNorm and FusedAddRMSNorm. Supports FP32 (default), BF16, or F16 — set `params.beta_dt` to match the buffer's element type. Pass `nullptr` if `use_shift == false` or not applicable.

If `gamma_dt` or `beta_dt` is not explicitly set, it defaults to `data_type_t::f32`. When providing BF16/F16 gamma/beta buffers, you **must** set the corresponding `gamma_dt`/`beta_dt` to `data_type_t::bf16` or `data_type_t::f16`; a mismatch between the field and the actual buffer type leads to undefined behavior.

## API Summary

| Function | Purpose |
|----------|---------|
| `normalization_direct` | Main execution API for all normalization variants |

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `const void*` | Input tensor pointer (read-only) |
| `output` | `void*` | Output tensor pointer |
| `gamma` | `const void*` | Scale parameter pointer (read-only) |
| `beta` | `const void*` | Shift parameter pointer (read-only) |
| `running_mean` | `const void*` | Pre-computed mean (BatchNorm only, FP32) |
| `running_var` | `const void*` | Pre-computed variance (BatchNorm only, FP32) |
| `residual` | `void*` | Residual buffer (FusedAddRMSNorm only, in-place) |
| `params` | `norm_params&` | Configuration parameters |

## Diagnostics and Profiling

- **Input validation** runs by default. It is gated by the environment variable `ZENDNNL_DIAGNOSTICS_ENABLE`, which defaults to enabled (anything other than `0`). To disable validation in production hot paths and reduce the gate to a single predicted-taken branch, set `ZENDNNL_DIAGNOSTICS_ENABLE=0` before launching the application.
- **Profiling** is controlled by the environment variable `ZENDNNL_ENABLE_PROFILER=1` and `ZENDNNL_PROFILE_LOG_LEVEL=4`. When active, `normalization_direct` reports execution time and operator parameters.
- **F16-FMA build-time toggle:** Build with `-DZENDNNL_NATIVE_F32_ACCUM=ON` to disable the native AVX512-FP16 fast path for LayerNorm/RMSNorm/FusedAddRMSNorm on AVX-512-FP16-capable hosts; those dispatches take the FP32-accumulating AVX-512 kernel instead. The flag also enables f16 inputs on hosts without AVX-512-FP16 (storage is handled via F16C convert in the FP32 kernel). It has no effect on BatchNorm or `bf16`-gamma/beta combos. See the [F16 FMA Mode](#f16-fma-mode-native-avx512-fp16) section for details.

## Error Handling

The `normalization_direct` function returns `status_t`:

- `status_t::success`: Operation completed successfully
- `status_t::failure`: Operation failed (check logs for details)
- `status_t::isa_unsupported`: F16 was requested but the platform lacks AVX512-FP16 ISA support

Failure causes checked unconditionally (production hot path):
- F16/BF16 cross-mixing between `src_dt` and `dst_dt`
- FusedAddRMSNorm with a null residual buffer

Additional failure causes checked only when `ZENDNNL_DIAGNOSTICS_ENABLE=1`:
- Null input or output pointers
- `norm_type` not specified (`NONE`)
- Unsupported `src_dt` or `dst_dt` (not f32, bf16, or f16)
- `batch == 0` or `norm_size == 0`
- Unsupported `gamma_dt` (not f32, bf16, or f16) when `use_scale == true`
- Unsupported `beta_dt` (not f32, bf16, or f16) when `use_shift == true`
- `use_scale == true` but gamma pointer is null
- `use_shift == true` but beta pointer is null (LayerNorm/BatchNorm only; RMSNorm and FusedAddRMSNorm skip this check)
- BatchNorm missing `running_mean` or `running_var`
- `epsilon <= 0`
