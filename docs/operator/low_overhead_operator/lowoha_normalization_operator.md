
(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA Normalization Operator

## Overview

The **LowOHA Normalization Operator** is a high-performance, low-overhead normalization operator designed for **latency-sensitive inference workloads**. It provides a single `normalization_direct` entry point that dispatches to four normalization variants — LayerNorm, RMSNorm, FusedAddRMSNorm, and BatchNorm — with minimal per-call overhead.

Unlike the standard operator factory pattern, LowOHA Normalization provides a **function-based interface** optimized for:
- Minimal execution overhead — no operator / context object lifecycle on the hot path
- FP32, BF16, and F16 data types for input, output, gamma, and beta (chosen independently)
- Four normalization variants behind a single API
- Native AVX-512 and AVX-512-FP16 vectorized kernels for LayerNorm, RMSNorm, and FusedAddRMSNorm
- Build-time switch to force FP32 accumulation for numerical reproducibility
- Direct control over execution parameters (data types, threads, epsilon)


## Normalization Formulas

### LayerNorm

Normalizes each sample across the trailing dimensions; mean and variance are computed on-the-fly from the input.

$$
y_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
$$

Where $\mu = \frac{1}{N}\sum_i x_i$ and $\sigma^2 = \frac{1}{N}\sum_i (x_i - \mu)^2$.

**Use case:** Transformer encoder blocks (BERT, GPT pre-norm layers).

### RMSNorm

Root Mean Square normalization — LayerNorm without mean subtraction or shift parameter.

$$
y_i = \gamma_i \cdot \frac{x_i}{\sqrt{\frac{1}{N}\sum_j x_j^2 + \epsilon}}
$$

**Use case:** LLaMA, Mistral, and other modern LLM architectures.

### FusedAddRMSNorm

Fuses a residual addition with RMSNorm in a single kernel call; the residual buffer is updated **in-place**, then the result is normalized.

$$
\text{residual}_i \mathrel{+}= \text{input}_i
$$

$$
y_i = \gamma_i \cdot \frac{\text{residual}_i}{\sqrt{\frac{1}{N}\sum_j \text{residual}_j^2 + \epsilon}}
$$

**Use case:** Transformer decoder sub-layers (LLaMA, Mistral) where each block adds to a running residual stream and immediately normalizes.

### BatchNorm (Inference)

Normalizes per channel using **pre-computed** running statistics from training.

$$
y_{n,c,s} = \gamma_c \cdot \frac{x_{n,c,s} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c
$$

**Use case:** CNN inference (ResNet, VGG, MobileNet).


## Core API: `normalization_direct`

The primary interface for LowOHA Normalization is the `normalization_direct` function:

```cpp
status_t normalization_direct(
  const void *input,         // Input tensor (read-only)
  void       *output,        // Output tensor (same shape as input)
  const void *gamma,         // Scale parameters (may be nullptr)
  const void *beta,          // Shift parameters (may be nullptr)
  const void *running_mean,  // Pre-computed mean (BatchNorm only, may be nullptr)
  const void *running_var,   // Pre-computed variance (BatchNorm only, may be nullptr)
  void       *residual,      // Residual buffer (FusedAddRMSNorm only, may be nullptr)
  norm_params &params        // Normalization parameters (non-const reference)
);
```

### Return Value

| Value                          | Description |
|--------------------------------|-------------|
| `status_t::success`            | Operation completed successfully |
| `status_t::failure`            | Validation or runtime error (check logs for details) |
| `status_t::isa_unsupported`    | F16 buffer was requested but host lacks AVX-512-FP16 (unless built with `-DZENDNNL_NATIVE_F32_ACCUM=ON`) |

### Dispatch Logic

- `LAYER_NORM` / `RMS_NORM` → AVX-512-FP16 (`__m512h`) when [F16 FMA Mode](#f16-fma-mode) eligibility holds, otherwise the FP32-accumulating AVX-512 kernel.
- `FUSED_ADD_RMS_NORM` → the FP32-accumulating AVX-512 kernel by default. When the library is built with `-DZENDNNL_FUSED_ADD_RMS_F16=ON`, it uses the AVX-512-FP16 (`__m512h`) kernel under a strict all-f16 gate (see [F16 FMA Mode](#f16-fma-mode)); otherwise it falls through to the FP32-accumulating AVX-512 kernel.
- `BATCH_NORM` → Reference (scalar) kernel.
- For non-F16 configurations, any path falls back to the reference kernel when AVX-512 is unavailable.
- F16 configurations (any actually-used buffer is `f16`) require AVX-512-FP16 up-front; the API returns `status_t::isa_unsupported` and does **not** fall back to the reference kernel.


## Parameters Structure

### `norm_params`

The main configuration structure for LowOHA Normalization. The caller must set `norm_type`, `batch`, `norm_size`, and (for BatchNorm) `num_channels`. The kernel treats the input as a logically 2-D `[batch, norm_size]` matrix and normalizes each row independently.

```cpp
struct norm_params {
  // --- Normalization variant ---
  norm_type_t  norm_type;       // LAYER_NORM / RMS_NORM / FUSED_ADD_RMS_NORM / BATCH_NORM

  // --- Flattened dimensions (set by caller) ---
  uint64_t     batch;           // Product of all outer (non-normalized) dims
  uint64_t     norm_size;       // Product of all normalized (trailing) dims
  uint64_t     num_channels;    // Channel count C (BatchNorm only)

  // --- Normalization parameters ---
  float        epsilon;         // Numerical-stability constant (default 1e-5)
  bool         use_scale;       // Apply gamma
  bool         use_shift;       // Apply beta (ignored by RMSNorm / FusedAddRMSNorm)

  // --- Data types ---
  data_type_t  src_dt;          // Source dtype:      f32 / bf16 / f16
  data_type_t  dst_dt;          // Destination dtype: f32 / bf16 / f16
  data_type_t  gamma_dt;        // Gamma dtype:       f32 (default) / bf16 / f16
  data_type_t  beta_dt;         // Beta dtype:        f32 (default) / bf16 / f16

  // --- Backend selection ---
  norm_algo_t  algorithm;       // Backend (default: none = auto)
  int32_t      num_threads;     // 0 = auto

  // --- Internal dispatch breadcrumb (callers should leave at default) ---
  data_type_t  accum_type;
};
```

### Normalization Type

```cpp
enum class norm_type_t : int {
  NONE               = -1,  // Not specified
  LAYER_NORM         =  0,
  BATCH_NORM         =  1,
  RMS_NORM           =  2,
  FUSED_ADD_RMS_NORM =  3
};
```

### Dimension Flattening Rules

The caller flattens tensor dimensions before calling `normalization_direct`. The total element count must equal `batch × norm_size` (or `batch × num_channels × norm_size` for BatchNorm).

| Tensor Shape       | Norm Dims   | `batch`   | `norm_size`  | `num_channels` |
|--------------------|-------------|-----------|--------------|----------------|
| `[B, D]`           | last 1      | `B`       | `D`          | —              |
| `[B, S, D]`        | last 1      | `B * S`   | `D`          | —              |
| `[B, S, H, D]`     | last 2      | `B * S`   | `H * D`      | —              |
| `[N, C, H, W]`     | per-channel | `N`       | `H * W`      | `C`            |

### Supported Data Type Combinations

**Input / Output (`src_dt`, `dst_dt`):**

| Input | Output |
|-------|--------|
| FP32  | FP32   |
| BF16  | BF16   |
| BF16  | FP32   |
| FP32  | BF16   |
| F16   | F16    |
| F16   | FP32   |
| FP32  | F16    |

> **Note:** F16 cannot be cross-mixed with BF16 between `src_dt` and `dst_dt` (e.g. `F16 → BF16` is rejected at validation). BF16 and non-native-F16 paths perform compute in FP32 internally, with conversions at the load/store boundary. Native AVX-512-FP16 kernels accumulate in `__m512h`.

**Gamma / Beta (`gamma_dt`, `beta_dt`):** `f32` (default), `bf16`, or `f16`. Chosen independently of `src_dt` / `dst_dt`.

**BatchNorm running mean / variance:** FP32 only (cast to `const float *` internally).

### Parameter Requirements by Norm Type

| Parameter      | LayerNorm     | RMSNorm      | FusedAddRMSNorm     | BatchNorm        |
|----------------|---------------|--------------|---------------------|------------------|
| `norm_type`    | `LAYER_NORM`  | `RMS_NORM`   | `FUSED_ADD_RMS_NORM`| `BATCH_NORM`     |
| `batch`        | required      | required     | required            | required (`N`)   |
| `norm_size`    | required      | required     | required            | required (`H*W`) |
| `num_channels` | unused        | unused       | unused              | required (`C`)   |
| `gamma`        | optional      | optional     | optional            | optional         |
| `beta`         | optional      | nullptr      | nullptr             | optional         |
| `running_mean` | nullptr       | nullptr      | nullptr             | required (FP32)  |
| `running_var`  | nullptr       | nullptr      | nullptr             | required (FP32)  |
| `residual`     | nullptr       | nullptr      | required            | nullptr          |
| `use_scale`    | true / false  | true / false | true / false        | true / false     |
| `use_shift`    | true / false  | unused       | unused              | true / false     |

### Backend Selection

```cpp
enum class norm_algo_t : int {
  none             = -1,  // Auto-select (default; routes to AVX-512 or reference)
  dynamic_dispatch =  0,  // Dynamic dispatch
  reference        =  1   // Reference (scalar) implementation
};
```


## Buffer Requirements

### Gamma and Beta

- **Gamma (scale):** Shape `[norm_size]` for LayerNorm / RMSNorm / FusedAddRMSNorm, `[num_channels]` for BatchNorm. Supports `f32` (default), `bf16`, or `f16` — set `params.gamma_dt` to match the buffer's element type. Pass `nullptr` if `use_scale == false`.
- **Beta (shift):** Shape `[norm_size]` for LayerNorm, `[num_channels]` for BatchNorm. Unused by RMSNorm and FusedAddRMSNorm. Supports `f32` (default), `bf16`, or `f16` — set `params.beta_dt` accordingly. Pass `nullptr` if `use_shift == false` or not applicable.

If `gamma_dt` / `beta_dt` is not explicitly set, it defaults to `data_type_t::f32`. When providing BF16/F16 gamma/beta buffers, you **must** set the corresponding dtype field — a mismatch between the field and the actual buffer leads to undefined behavior.

### Residual Buffer (FusedAddRMSNorm only)

The `residual` parameter is unique to `FUSED_ADD_RMS_NORM`:

- **Required:** non-null when `norm_type == FUSED_ADD_RMS_NORM`.
- **Element type:** same as `params.src_dt` (f32 / bf16 / f16).
- **Access:** read-write, modified in-place.
- **After the call:** `residual[i] = old_residual[i] + input[i]`.

For all other norm types, pass `nullptr`.

### BatchNorm Running Statistics

`running_mean` and `running_var` are required for `BATCH_NORM`, must be FP32, and have shape `[num_channels]`.


## F16 FMA Mode

Native AVX-512-FP16 acceleration is selected when **all** of the following hold:

- `norm_type` is `LAYER_NORM` or `RMS_NORM` (BatchNorm is never F16-eligible; FusedAddRMSNorm is F16-eligible only when built with `-DZENDNNL_FUSED_ADD_RMS_F16=ON`, and then only under a strict `src_dt == dst_dt == gamma_dt == f16` gate).
- At least one of `src_dt` / `dst_dt` is `f16`.
- Actually-used `gamma_dt` / `beta_dt` is `f16` or `f32` (`bf16` forces the FP32 path).
- Host CPU exposes AVX-512-FP16.
- Build was **not** configured with `-DZENDNNL_NATIVE_F32_ACCUM=ON`.

When eligible, the inner loop runs in `__m512h` registers (32 lanes/register, ~2× throughput vs the FP32-accumulating AVX-512 path). Conversions to/from FP32 happen only at the load/store boundary.

### Dispatch Matrix (LayerNorm / RMSNorm)

| `(src_dt, dst_dt)` | `gamma_dt` | `beta_dt` (LayerNorm + `use_shift`) | Kernel chosen |
|---|---|---|---|
| `(f16, f16)` / `(f16, f32)` / `(f32, f16)` | `f16` or `f32` | `f16` or `f32` | **AVX-512-FP16** |
| as above, but `gamma_dt = bf16` or `beta_dt = bf16` | bf16 | bf16 | AVX-512 (FP32-accumulating) |
| `(f32, f32)` (no F16 anywhere) | any | any | AVX-512 (FP32-accumulating) |
| `(bf16, bf16)` / `(bf16, f32)` / `(f32, bf16)` | any | any | AVX-512 (FP32-accumulating) |
| `(f16, bf16)` / `(bf16, f16)` | any | any | **Rejected** — `status_t::failure` |

By default `FUSED_ADD_RMS_NORM` has no AVX-512-FP16 fast path and always runs on the FP32-accumulating AVX-512 kernel. The in-place residual add plus F16 sum-of-squares accumulation produced unacceptable precision loss, so the native FP16 fused-add kernel is compiled out by default.

> **Opt-in fast path:** Build with `-DZENDNNL_FUSED_ADD_RMS_F16=ON` to re-enable the native AVX-512-FP16 (F16-accumulating) fused-add kernel. It uses a strict `src_dt == dst_dt == gamma_dt == f16` gate (the residual buffer aliases `src` and is read-modify-written in place, so it must share the f16 storage layout); mixed-dtype fused-add still falls through to the FP32-accumulating AVX-512 kernel. Intended for A/B precision experiments only — expect reduced accuracy vs. the FP32 path.

`BATCH_NORM` always uses the reference kernel (no AVX-512 BatchNorm fast path).

### Build-Time Opt-Out

Pass `-DZENDNNL_NATIVE_F32_ACCUM=ON` to CMake to disable the native AVX-512-FP16 fast path for LayerNorm / RMSNorm. With this flag set:

- Those calls take the FP32-accumulating AVX-512 kernel instead (useful for A/B precision comparisons and reproducibility studies).
- F16 inputs are also **enabled on hosts without AVX-512-FP16**: the FP32 kernel converts F16 storage via F16C (`_mm512_cvtph_ps` / `_mm512_cvtps_ph`). Without the flag, F16 calls on such hosts return `status_t::isa_unsupported`.
- It has no effect on BatchNorm (always reference) or `bf16`-gamma/beta combos (already on the FP32 path).
- If the library was also built with `-DZENDNNL_FUSED_ADD_RMS_F16=ON`, this flag takes precedence: `ZENDNNL_NATIVE_F32_ACCUM=ON` disables the opt-in FusedAddRMSNorm F16 path too (both share the `can_use_f16_fma_kernel()` gate), so FusedAddRMSNorm stays on the FP32-accumulating kernel.

The same switch also covers the F16 path in the [LowOHA EmbeddingBag operator](lowoha_embedding_bag_operator.md).


## Usage Examples

### Example 1: LayerNorm (FP32)

```cpp
#include "lowoha_operators/normalization/lowoha_normalization.hpp"

int layer_norm_fp32_example() {
  using namespace zendnnl::lowoha::normalization;

  const uint64_t batch      = 4;
  const uint64_t hidden_dim = 768;

  std::vector<float> input(batch * hidden_dim);
  std::vector<float> gamma(hidden_dim, 1.0f);
  std::vector<float> beta(hidden_dim, 0.0f);
  std::vector<float> output(batch * hidden_dim, 0.0f);

  // Configure parameters
  norm_params params;
  params.batch      = batch;
  params.norm_size  = hidden_dim;
  params.norm_type  = norm_type_t::LAYER_NORM;
  params.src_dt     = data_type_t::f32;
  params.dst_dt     = data_type_t::f32;
  params.epsilon    = 1e-5f;
  params.use_scale  = true;
  params.use_shift  = true;

  // Execute
  status_t status = normalization_direct(
      input.data(), output.data(),
      gamma.data(), beta.data(),
      /*running_mean=*/nullptr, /*running_var=*/nullptr,
      /*residual=*/nullptr, params);

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 2: RMSNorm (FP32) on a 2D Tensor

```cpp
#include "lowoha_operators/normalization/lowoha_normalization.hpp"

int rms_norm_fp32_example() {
  using namespace zendnnl::lowoha::normalization;

  const uint64_t batch      = 4;
  const uint64_t hidden_dim = 4096;

  std::vector<float> input(batch * hidden_dim);
  std::vector<float> gamma(hidden_dim, 1.0f);
  std::vector<float> output(batch * hidden_dim, 0.0f);

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

### Example 3: RMSNorm on a 3D Transformer Tensor

For a 3D tensor `[batch, seq_len, hidden_dim]` normalized over the last dim, flatten the outer dims into `params.batch`:

```cpp
#include "lowoha_operators/normalization/lowoha_normalization.hpp"

int rms_norm_3d_example() {
  using namespace zendnnl::lowoha::normalization;

  const uint64_t batch      = 2;
  const uint64_t seq_len    = 512;
  const uint64_t hidden_dim = 4096;

  std::vector<float> input(batch * seq_len * hidden_dim);
  std::vector<float> gamma(hidden_dim, 1.0f);
  std::vector<float> output(batch * seq_len * hidden_dim, 0.0f);

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
      nullptr, nullptr, nullptr, params);

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 4: FusedAddRMSNorm (Transformer Decoder Block)

```cpp
#include "lowoha_operators/normalization/lowoha_normalization.hpp"

int fused_add_rms_norm_example() {
  using namespace zendnnl::lowoha::normalization;

  const uint64_t batch      = 2;
  const uint64_t seq_len    = 512;
  const uint64_t hidden_dim = 4096;
  const uint64_t total      = batch * seq_len * hidden_dim;

  std::vector<float> sublayer_output(total);   // attention / FFN output
  std::vector<float> residual(total);          // running residual stream (in-place)
  std::vector<float> gamma(hidden_dim, 1.0f);
  std::vector<float> hidden_states(total, 0.0f);

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

### Example 5: RMSNorm with BF16 Storage and BF16 Gamma

```cpp
#include "lowoha_operators/normalization/lowoha_normalization.hpp"

int rms_norm_bf16_example() {
  using namespace zendnnl::lowoha::normalization;

  const uint64_t batch      = 2;
  const uint64_t seq_len    = 512;
  const uint64_t hidden_dim = 4096;
  const uint64_t total      = batch * seq_len * hidden_dim;

  // BF16 buffers stored as int16_t (or uint16_t)
  std::vector<int16_t> input(total);
  std::vector<int16_t> gamma(hidden_dim);
  std::vector<int16_t> output(total, 0);

  norm_params params;
  params.batch      = batch * seq_len;
  params.norm_size  = hidden_dim;
  params.norm_type  = norm_type_t::RMS_NORM;
  params.src_dt     = data_type_t::bf16;
  params.dst_dt     = data_type_t::bf16;
  params.gamma_dt   = data_type_t::bf16;   // gamma buffer is BF16
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

### Example 6: F16 LayerNorm (Native AVX-512-FP16)

```cpp
#include "lowoha_operators/normalization/lowoha_normalization.hpp"

int layer_norm_f16_example() {
  using namespace zendnnl::lowoha::normalization;

  const uint64_t batch      = 4;
  const uint64_t hidden_dim = 768;
  const uint64_t total      = batch * hidden_dim;

  // F16 buffers stored as uint16_t (IEEE 754 half-precision bit patterns)
  std::vector<uint16_t> input(total);
  std::vector<uint16_t> gamma(hidden_dim);
  std::vector<uint16_t> beta(hidden_dim);
  std::vector<uint16_t> output(total, 0);

  norm_params params;
  params.batch      = batch;
  params.norm_size  = hidden_dim;
  params.norm_type  = norm_type_t::LAYER_NORM;
  params.src_dt     = data_type_t::f16;
  params.dst_dt     = data_type_t::f16;
  params.gamma_dt   = data_type_t::f16;
  params.beta_dt    = data_type_t::f16;
  params.epsilon    = 1e-5f;
  params.use_scale  = true;
  params.use_shift  = true;

  status_t status = normalization_direct(
      input.data(), output.data(),
      gamma.data(), beta.data(),
      nullptr, nullptr, nullptr, params);

  if (status == status_t::isa_unsupported) {
    // Host lacks AVX-512-FP16 -- skip or fall back to FP32
    return 0;
  }
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 7: BatchNorm Inference (NCHW)

```cpp
#include "lowoha_operators/normalization/lowoha_normalization.hpp"

int batch_norm_inference_example() {
  using namespace zendnnl::lowoha::normalization;

  const uint64_t N = 2, C = 64, H = 56, W = 56;

  std::vector<float> input(N * C * H * W);
  std::vector<float> gamma(C, 1.0f);
  std::vector<float> beta(C, 0.0f);
  std::vector<float> running_mean(C, 0.0f);
  std::vector<float> running_var(C, 1.0f);
  std::vector<float> output(N * C * H * W, 0.0f);

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


## Performance Considerations

- **Algorithm Selection:** Leave `params.algorithm = norm_algo_t::none` for auto-selection (routes to the best available AVX-512 / AVX-512-FP16 kernel; falls back to reference on non-AVX-512 hosts for non-F16 paths).
- **Data-Type Selection:**

  | Dtype Combo        | Best For                                   | Notes                                                    |
  |--------------------|--------------------------------------------|----------------------------------------------------------|
  | `(f32, f32)`       | Numerical precision, reproducibility       | Reference path on non-AVX-512 hosts                      |
  | `(bf16, bf16)`     | Training-compatible inference              | FP32 internal accumulation                               |
  | `(f16, f16)`       | Lowest memory bandwidth                    | Requires AVX-512-FP16; ~2× kernel throughput             |
  | Mixed F32 ↔ F16    | F16 storage, downstream F32 compute        | One-sided F16 still triggers AVX-512-FP16 path           |

- **Threading:** `params.num_threads` controls intra-op parallelism (`0` = auto, honors `OMP_NUM_THREADS`). The kernel parallelizes across rows of the flattened `[batch, norm_size]` matrix — taller batches scale better than wider rows.
- **Variant-to-Architecture Mapping:**

  | Variant            | Where it shows up                                                       |
  |--------------------|-------------------------------------------------------------------------|
  | LayerNorm          | BERT, GPT pre-norm blocks                                               |
  | RMSNorm            | LLaMA, Mistral, Qwen, modern LLMs                                       |
  | FusedAddRMSNorm    | LLM decoder blocks with residual streams (fuses add + norm in one pass) |
  | BatchNorm          | CNN inference (ResNet, VGG, MobileNet)                                  |

- **Diagnostics Toggle:** Input validation runs by default. On verified production hot paths, set `ZENDNNL_DIAGNOSTICS_ENABLE=0` to reduce the gate to a single predicted-taken branch. Always-on failure causes (F16/BF16 cross-mixing, FusedAddRMSNorm with null residual) are still checked.


## Integration Workflow

1. **Flatten** the tensor: derive `batch`, `norm_size`, and (for BatchNorm) `num_channels` from the real tensor shape (see [Dimension Flattening Rules](#dimension-flattening-rules)).
2. **Populate** `norm_params` with the norm type, dimensions, dtypes, and epsilon. Match `gamma_dt` / `beta_dt` to the buffers you provide.
3. **Call** `normalization_direct`, passing `nullptr` for arguments not used by the chosen variant (see [Parameter Requirements by Norm Type](#parameter-requirements-by-norm-type)).
4. **Check** `status_t`. Treat `status_t::isa_unsupported` separately from `failure` so F16 paths can fall back gracefully on hosts without AVX-512-FP16.

```cpp
using namespace zendnnl::lowoha::normalization;

norm_params p;
p.norm_type = norm_type_t::RMS_NORM;          // 1. choose variant
p.batch     = batch * seq_len;                // 2. flatten outer dims
p.norm_size = hidden_dim;
p.src_dt    = data_type_t::f32;
p.dst_dt    = data_type_t::f32;
p.epsilon   = 1e-6f;
p.use_scale = true;

status_t s = normalization_direct(            // 3. invoke
    input, output, gamma, /*beta=*/nullptr,
    /*running_mean=*/nullptr, /*running_var=*/nullptr,
    /*residual=*/nullptr, p);

if (s == status_t::isa_unsupported) { /* fall back, e.g. promote to F32 */ }
else if (s != status_t::success)    { /* handle error */ }
```


## Validation

The operator performs the following validations:

**Always-on (production hot path):**

1. F16 / BF16 cross-mixing between `src_dt` and `dst_dt` (rejected).
2. `FUSED_ADD_RMS_NORM` with a null `residual` buffer (rejected).
3. F16 buffer requested on a host without AVX-512-FP16 (returns `status_t::isa_unsupported`, unless built with `-DZENDNNL_NATIVE_F32_ACCUM=ON`).

**Additional checks when `ZENDNNL_DIAGNOSTICS_ENABLE=1` (default):**

4. Null `input` or `output` pointer.
5. `norm_type == NONE` (not specified).
6. Unsupported `src_dt` / `dst_dt` (not one of `f32`, `bf16`, `f16`).
7. `batch == 0` or `norm_size == 0`.
8. Unsupported `gamma_dt` (when `use_scale == true`) or `beta_dt` (when `use_shift == true`).
9. `use_scale == true` but `gamma == nullptr`.
10. `use_shift == true` but `beta == nullptr` (LayerNorm / BatchNorm only; RMSNorm and FusedAddRMSNorm skip this check).
11. BatchNorm missing `running_mean` or `running_var`.
12. `epsilon <= 0`.

## Diagnostics and Profiling

- **Input validation** runs by default; toggle with `ZENDNNL_DIAGNOSTICS_ENABLE` (defaults to `1`). Set to `0` to skip optional validation on production hot paths — the always-on checks (F16/BF16 cross-mixing, FusedAddRMSNorm with null residual) remain in place.
- **Profiling** is controlled by `ZENDNNL_ENABLE_PROFILER=1` and `ZENDNNL_PROFILE_LOG_LEVEL=4`. When active, `normalization_direct` logs execution time and operator parameters.
- **F16-FMA build-time toggle:** Build with `-DZENDNNL_NATIVE_F32_ACCUM=ON` to disable the native AVX-512-FP16 fast path for LayerNorm / RMSNorm on AVX-512-FP16-capable hosts; those dispatches take the FP32-accumulating AVX-512 kernel instead. The flag also enables F16 inputs on hosts without AVX-512-FP16 (storage handled via F16C convert in the FP32 kernel). It has no effect on BatchNorm or `bf16`-gamma/beta combos. See the [F16 FMA Mode](#f16-fma-mode) section for details.
- **FusedAddRMSNorm F16 opt-in:** FusedAddRMSNorm has no native AVX-512-FP16 path by default. Build with `-DZENDNNL_FUSED_ADD_RMS_F16=ON` to enable it under a strict `src_dt == dst_dt == gamma_dt == f16` gate (A/B precision experiments only). `-DZENDNNL_NATIVE_F32_ACCUM=ON` overrides it and keeps FusedAddRMSNorm on the FP32-accumulating kernel.
