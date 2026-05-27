
(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA Reorder Operator

## Overview

The **LowOHA Reorder Operator** is a high-performance, low-overhead data type conversion operator designed for **quantization, dequantization, and data type conversion in workloads**. It provides a direct API to convert data between BF16/FP32/F16 and INT8/UINT8 formats, and for conversions among FP32, BF16, F16 in any direction, with configurable scale and zero-point parameters.

Unlike the standard Reorder operator which uses the operator factory pattern, LowOHA Reorder provides a **function-based interface** optimized for:
- Minimal execution overhead
- Quantization (BF16/FP32 → INT8/UINT8)
- Dequantization (INT8/UINT8 → BF16/FP32)
- Data type conversion (FP32 ⇔ BF16, FP32 ⇔ F16, BF16 ⇔ F16)
- **Dynamic quantization** (compute scale/zero-point from source data at runtime)
- Per-tensor, per-channel (row and column), and per-group (row and column) quantization granularities
- Strided (non-contiguous) source memory support


## Quantization/Dequantization/Conversion Formulas

### Quantization (BF16/FP32 → INT8)

$$
\mathrm{int8} = \mathrm{clamp}(\mathrm{round}(\frac{\mathrm{input}}{\mathrm{scale}}) + \mathrm{zp}, -128, 127)
$$

### Quantization (BF16/FP32 → UINT8)

$$
\mathrm{uint8} = \mathrm{clamp}(\mathrm{round}(\frac{\mathrm{input}}{\mathrm{scale}}) + \mathrm{zp}, 0, 255)
$$

### Dequantization (INT8/UINT8 → BF16/FP32)

$$
\mathrm{output} = (\mathrm{int8} - \mathrm{zp}) \times \mathrm{scale}
$$

### Data Type Conversion (FP32 → BF16)

**Simple conversion (no scale/zero-point):**

$$
\mathrm{bf16} = \mathrm{bf16}(\mathrm{f32})
$$

**With scale and zero-point:**

$$
\mathrm{bf16} = \mathrm{bf16}(\frac{\mathrm{f32}}{\mathrm{scale}} + \mathrm{zp})
$$

### Data Type Conversion (BF16 → FP32)

**Simple conversion (no scale/zero-point):**

$$
\mathrm{f32} = \mathrm{f32}(\mathrm{bf16})
$$

**With scale and zero-point:**

$$
\mathrm{f32} = (\mathrm{f32}(\mathrm{bf16}) - \mathrm{zp}) \times \mathrm{scale}
$$

### Dynamic Quantization Parameter Computation

When `dynamic_quant = true`, scale and zero-point are computed from the source data at runtime.

**Symmetric quantization (S8 destination, zero_point buffer = nullptr):**

$$
\mathrm{scale} = \frac{\max(|\min(A)|, |\max(A)|)}{127}
$$

$$
\mathrm{zp} = 0
$$

**Asymmetric quantization (U8 destination, zero_point buffer provided):**

$$
\mathrm{scale} = \frac{\max(A) - \min(A)}{255}
$$

$$
\mathrm{zp} = \mathrm{round}(\frac{-\min(A)}{\mathrm{scale}})
$$

Where $A$ is the set of source values within the quantization scope (per-tensor, per-channel, or per-group).


## Core API: `reorder_direct`

The primary interface for LowOHA Reorder is the `reorder_direct` function:

```cpp
status_t reorder_direct(
  const void *src,                      // Pointer to source data buffer
  void *dst,                            // Pointer to destination data buffer (nullptr allowed for dynamic quant compute-only)
  reorder_params_t params        // Reorder parameters
);
```

### Return Value

| Value | Description |
|-------|-------------|
| `status_t::success` | Operation completed successfully |
| `status_t::failure` | Operation failed (invalid parameters, null pointers, etc.) |

**Note:** When `dynamic_quant = true` and `dst = nullptr`, the function computes and fills the scale/zero-point output buffers without performing quantization. This is useful for pre-computing quantization parameters.


## Parameters Structure

### `reorder_params_t`

The main configuration structure for LowOHA Reorder:

```cpp
struct reorder_params_t {
  data_type_t src_dtype;                  // Source data type
  data_type_t dst_dtype;                  // Destination data type
  reorder_quant_params_t quant_params;    // Quantization parameters (scale, zero_point)
  reorder_algo_t algo;                    // Algorithm selection
  int32_t num_threads;                    // Number of threads (0 = auto)
  std::vector<int64_t> src_shape;         // Source shape: [N] or [M, N] or [batch, M, N] (mandatory)
  std::vector<int64_t> dst_shape;         // Destination shape: must match src_shape (mandatory)
  std::vector<int64_t> src_strides;       // Source strides for non-contiguous memory (optional)
  std::vector<int64_t> dst_strides;       // Destination strides (reserved for future, not currently supported)
  bool dynamic_quant;                     // Enable dynamic quantization (default: false)
  bool is_prepack;                        // Weight-prepack mode (see "Weight Prepack Mode" below)
  prepack_params_t prepack;               // Prepack request (used only when is_prepack = true)
};
```

### Shape Format

Both `src_shape` and `dst_shape` are **mandatory** and determine the tensor dimensionality:

| Shape Size | Format | Description |
|------------|--------|-------------|
| 1 | `[N]` | 1D array with N elements |
| 2 | `[M, N]` | 2D matrix with M rows and N columns |
| 3 | `[batch, M, N]` | 3D batched matrix |

The total number of elements is computed automatically from the shape.

**Important Constraint:** `src_shape` and `dst_shape` **must be identical**. An error will be thrown if they differ.

### Strides Format (Optional)

#### Source Strides (`src_strides`)

Source strides enable reading from non-contiguous source memory:

| Strides Size | Format | Description |
|--------------|--------|-------------|
| Empty | - | Contiguous memory (default) |
| 1 | `[stride]` | 1D with custom stride |
| 2 | `[stride_M, stride_N]` | 2D with row and column strides |
| 3 | `[stride_batch, stride_M, stride_N]` | 3D with batch, row, and column strides |

#### Destination Strides (`dst_strides`)

**Note:** `dst_strides` is reserved for future implementation and is **currently not supported**. The destination is always written in contiguous format. Providing `dst_strides` will result in an error.


### Supported Data Type Combinations

| Source Type | Destination Type | Operation |
|-------------|------------------|-----------|
| BF16 | S8 (INT8) | Quantization |
| S8 (INT8) | BF16 | Dequantization |
| BF16 | U8 (UINT8) | Quantization |
| U8 (UINT8) | BF16 | Dequantization |
| FP32 | S8 (INT8) | Quantization |
| S8 (INT8) | FP32 | Dequantization |
| FP32 | U8 (UINT8) | Quantization |
| U8 (UINT8) | FP32 | Dequantization |
| FP32 | BF16 | Data Type Conversion (optional scale/zp) |
| BF16 | FP32 | Data Type Conversion (optional scale/zp) |
| FP32 | F16  | Data Type Conversion (optional scale/zp) |
| F16  | FP32 | Data Type Conversion (optional scale/zp) |
| BF16 | F16  | Data Type Conversion (optional scale/zp, via FP32) |
| F16  | BF16 | Data Type Conversion (optional scale/zp, via FP32) |


### `reorder_quant_params_t`

Quantization parameters for scale and zero-point:

```cpp
struct reorder_quant_params_t {
  struct quant_t {
    void *buff;                    // Pointer to quantization data buffer (read for static, write for dynamic)
    data_type_t dt;                // Data type of the buffer
    std::vector<int64_t> dims;     // Dimensions (mandatory, must match tensor dims)
  };

  quant_t scale;        // Scale factor (f32 or bf16)
  quant_t zero_point;   // Zero point offset (s32 only)
};
```

**Currently Supported Data Types:**

| Parameter | Supported Type | Description |
|-----------|---------------|-------------|
| `scale` | `f32` or `bf16` | Scale factor (must be finite; bf16 is converted to f32 internally) |
| `zero_point` | `s32` | Zero point offset |

**Buffer Semantics:**

| Mode | `buff` Usage | Description |
|------|--------------|-------------|
| Static quantization (`dynamic_quant = false`) | **Read** | User provides pre-computed scale/zero-point values |
| Dynamic quantization (`dynamic_quant = true`) | **Write** | User provides output buffer; API fills it with computed values |

**Note on FP32 ↔ BF16 Conversion:**
- For FP32 ↔ BF16 data type conversion, the `quant_params` are **optional**
- If `quant_params.scale.buff` is `nullptr`, a simple direct conversion is performed without scaling
- When scale/zero-point are provided, the conversion formulas are applied (see [Data Type Conversion formulas](#data-type-conversion-fp32--bf16))


## Quantization Granularities

The `dims` field determines the quantization granularity. **dims is mandatory** and must match the tensor dimensionality.

### 1D Tensor (shape = [N])

| Granularity | dims | Total Values | Description |
|-------------|------|--------------|-------------|
| Per-tensor | `{1}` | 1 | Single scale/zp for all elements |
| Per-channel | `{N}` | N | Different scale/zp for each element |

### 2D Tensor (shape = [M, N])

| Granularity | dims | Total Values | Description |
|-------------|------|--------------|-------------|
| Per-tensor | `{1, 1}` | 1 | Single scale/zp for entire matrix |
| Per-col | `{1, N}` | N | Different scale/zp for each column |
| Per-row | `{M, 1}` | M | Different scale/zp for each row (per-token) |
| Per-group-row | `{G, N}` | G × N | G groups across rows, each with N values |
| Per-group-col | `{M, G}` | M × G | G groups across columns, each row has G values |

**Per-group-row constraint:** M must be divisible by G (M % G == 0)

**Per-group-col constraint:** N must be divisible by G (N % G == 0)

### 3D Tensor (shape = [batch, M, N])

| Granularity | dims | Total Values | Description |
|-------------|------|--------------|-------------|
| Per-tensor | `{1, 1, 1}` | 1 | Single scale/zp for entire tensor |
| Per-col | `{1, 1, N}` | N | Different scale/zp for each column |
| Per-row | `{1, M, 1}` | M | Different scale/zp for each row (per-token) |
| Per-group-row | `{1, G, N}` | G × N | G groups across rows, each with N values |
| Per-group-col | `{1, M, G}` | M × G | G groups across columns, each row has G values |

**Per-group-row constraint:** M must be divisible by G (M % G == 0)

**Per-group-col constraint:** N must be divisible by G (N % G == 0)

### Index Calculation

**Per-group-row** (dims `{G, N}`):
- `group_size = M / G`
- `group_idx = row / group_size`
- `index = group_idx * N + col`

**Per-group-col** (dims `{M, G}`):
- `group_size = N / G`
- `group_idx = col / group_size`
- `index = row * G + group_idx`


### `reorder_algo_t`

Algorithm selection for the reorder operation:

```cpp
enum class reorder_algo_t : int {
  none = -1,        // No specific algorithm
  DT = 0,           // Decision tree based algorithm selection (recommended)
  native = 1,       // Native vectorized implementation (AVX512)
  reference = 2,    // Reference scalar implementation
  algo_count        // Number of algorithms (must be last)
};
```

**Algorithm Selection:**

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `native` | AVX512 vectorized implementation | Large buffers (≥64 elements) |
| `reference` | Scalar implementation | Small buffers or debugging |
| `DT` | Decision tree based selection | General use (recommended) |


## Dynamic Quantization Mode

Dynamic quantization computes the quantization parameters (scale and zero-point) **at runtime from the source data**, rather than requiring pre-computed values. This is enabled by setting `dynamic_quant = true` in `reorder_params_t`.

### How It Works

1. The API scans the source data to find min/max values (respecting the configured granularity)
2. Scale and zero-point are computed from min/max and written to the user-provided output buffers
3. If `dst` is not `nullptr`, the quantization is then performed using the computed parameters
4. If `dst` is `nullptr`, only the scale/zero-point computation is performed (compute-only mode)

### Quantization Modes

The quantization mode is determined by the presence of the `zero_point` buffer:

| zero_point.buff | Mode | Destination Type | Formula |
|-----------------|------|------------------|---------|
| `nullptr` | **Symmetric** | S8 (INT8) | scale = max(\|min\|, \|max\|) / 127, zp = 0 |
| Provided | **Asymmetric** | U8 (UINT8) | scale = (max - min) / 255, zp = round(-min / scale) |

### Supported Configurations

| Source Type | Destination Type | Mode |
|-------------|------------------|------|
| BF16 | S8 (INT8) | Symmetric |
| FP32 | S8 (INT8) | Symmetric |
| BF16 | U8 (UINT8) | Asymmetric |
| FP32 | U8 (UINT8) | Asymmetric |

### Supported Granularities

All granularities are supported for dynamic quantization:

| Granularity | dims (2D) | Output Values | Description |
|-------------|-----------|---------------|-------------|
| Per-tensor | `{1, 1}` | 1 | Single scale/zp for all elements |
| Per-row (per-token) | `{M, 1}` | M | One scale/zp per row |
| Per-col | `{1, N}` | N | One scale/zp per column |
| Per-group-row | `{G, N}` | G × N | G groups across rows |
| Per-group-col | `{M, G}` | M × G | G groups across columns |

### Buffer Requirements

| Parameter | Required | Data Type | Buffer Size |
|-----------|----------|-----------|-------------|
| `scale.buff` | **Always** | `f32` or `bf16` | Product of `scale.dims` |
| `zero_point.buff` | Asymmetric only | `s32` | Product of `zero_point.dims` |

- For **symmetric** mode: leave `zero_point.buff = nullptr` (zero-point is implicitly 0)
- For **asymmetric** mode: `zero_point.dims` must have the same granularity as `scale.dims`

### Compute-Only Mode

When `dst = nullptr`, the API only computes scale and zero-point values without performing quantization. This is useful for:
- Pre-computing quantization parameters for later use
- Profiling the parameter computation overhead separately
- Using the computed parameters with a different quantization implementation


## Weight Prepack Mode

Weight prepack lets the caller produce a backend-blocked weight buffer **once** (e.g. at model load), then reuse it across many matmul calls without paying the per-call internal-reorder cost.

The mode is selected by setting `reorder_params_t::is_prepack = true` on the same `reorder_params_t` you hand to `reorder_direct`. The prepack-specific knobs (algo / dtypes / K / N / ldb / ...) live on the embedded `reorder_params_t::prepack` sub-struct; the rest of the standard reorder fields (`src_dtype`, `dst_dtype`, `quant_params`, `src_shape`, `dst_strides`, `dynamic_quant`, ...) are ignored in this mode.

### Supported Algos

Only **AOCL DLP blocked** is supported:

| `prepack.algo` | Supported | Notes |
|----------------|-----------|-------|
| `matmul_algo_t::aocl_dlp_blocked` | ✅ Yes | f32 / bf16 / f16 / s8 (+ s8 sym-quant variant) / s4 / u4 |
| `matmul_algo_t::libxsmm_blocked`  | ❌ No  | Layout would mis-match if the matmul-side partitioner falls back to AOCL DLP — silent wrong results. |
| `matmul_algo_t::onednn_blocked`   | ❌ No  | OneDNN's blocked layout depends on (M, K, N, dtypes, post-ops, ISA) — the prepack can't reproduce the matmul-time layout. |
| Anything else                     | ❌ No  | Non-blocked variants consume raw weights; no prepack needed. |

Anything other than `aocl_dlp_blocked` returns `status_t::unimplemented` with a clear error in the log.

### Two-Step Caller Workflow

The contract is: **call `weight_prepack_size` first, allocate exactly that many bytes, then call `reorder_direct`**.

1. **Query the required size** with `weight_prepack_size(rp)`. Returns the prepacked-buffer size in bytes (already rounded up to 64-byte alignment) on success, `0` on validation/dispatch failure.
2. **Allocate a buffer of at least that size.** No specific alignment is required, but 64-byte alignment is recommended for best AVX-512 throughput.
3. **Call `reorder_direct(weights, dst, rp)`** with the same `rp`. The library writes the prepacked layout into `dst`.

> ⚠️ **The buffer size MUST match (or exceed) the value returned by `weight_prepack_size`.**
> The library does **not** verify the buffer length (a `void *` carries no size information in C/C++). An under-sized buffer causes out-of-bounds writes — silent corruption, not an error. Always pair the size query with the allocation.

### `prepack_params_t`

```cpp
struct prepack_params_t {
  matmul_algo_t algo;           // Must be matmul_algo_t::aocl_dlp_blocked
  data_type_t   wei_dtype;      // Weight data type (f32 / bf16 / f16 / s8 / s4 / u4)
  data_type_t   src_dtype;      // Source (matmul A) dtype (disambiguates s8 vs u8 src)
  int64_t       K;              // Weight rows
  int64_t       N;              // Weight cols
  int64_t       ldb;            // Physical leading dimension of the input weights
  bool          transposed;     // true => weights are column-major ('ba')
  int           sym_group_size; // AOCL: >0 selects s8 sym-quant variant
  size_t        cached_size;    // (out, internal) — populated by weight_prepack_size
};
```

### Supported `wei_dtype` Routing

| `wei_dtype` | `src_dtype`        | AOCL DLP reorder kernel used         |
|-------------|--------------------|---------------------------------------|
| `f32`       | any                | `aocl_reorder_f32f32f32of32`         |
| `bf16`      | any                | `aocl_reorder_bf16bf16f32of32`       |
| `f16`       | any                | `aocl_reorder_f16f16f16of16` (DLP only) |
| `s4` / `u4` | any                | `aocl_reorder_bf16s4f32of32` (4-bit wei, bf16 act) |
| `s8`        | `s8` / `bf16` / `f32` | `aocl_reorder_s8s8s32os32`        |
| `s8`        | `u8`               | `aocl_reorder_u8s8s32os32`           |
| `s8` + `sym_group_size > 0` (with `src_dtype = bf16`) | — | `aocl_reorder_s8s8s32os32_sym_quant` (DLP only) |

### Consuming the Prepacked Buffer at Matmul Time

Hand the prepacked buffer to `matmul_direct` with:

- `matmul_params::lowoha_algo = matmul_algo_t::aocl_dlp_blocked`
- `matmul_params::mem_format_b = 'r'`

`mem_format_b = 'r'` tells the matmul backend "the `weight` pointer is already in AOCL DLP blocked layout, skip the internal reorder". `matmul_direct` validates that `lowoha_algo == aocl_dlp_blocked` whenever `mem_format_b == 'r'`; any other algo is rejected up front (would otherwise silently produce wrong results).

### End-to-End Example

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"

// --- Model load: prepack once ------------------------------------------
reorder_params_t rp;
rp.is_prepack         = true;
rp.prepack.algo       = matmul_algo_t::aocl_dlp_blocked;
rp.prepack.wei_dtype  = data_type_t::s8;
rp.prepack.src_dtype  = data_type_t::bf16;   // matmul A is bf16 -> sym-quant variant
rp.prepack.K          = K;
rp.prepack.N          = N;
rp.prepack.ldb        = N;                   // row-major leading dim
rp.prepack.transposed = false;
rp.prepack.sym_group_size = K;               // per-row scale (optional)

// Step 1: query the required size
size_t prepack_bytes = weight_prepack_size(rp);
if (prepack_bytes == 0) { /* validation error -- check apilog_error */ }

// Step 2: allocate a buffer of exactly that size
std::vector<uint8_t> prepacked_buf(prepack_bytes);

// Step 3: prepack
status_t st = reorder_direct(original_weight, prepacked_buf.data(), rp);
if (st != status_t::success) { /* prepack failed */ }

// --- Inference: reuse `prepacked_buf` across many matmuls --------------
matmul_params mp;
mp.lowoha_algo  = matmul_algo_t::aocl_dlp_blocked;
mp.mem_format_b = 'r';                       // <-- skip internal reorder
mp.dtypes.src   = data_type_t::bf16;
mp.dtypes.wei   = data_type_t::s8;
mp.dtypes.dst   = data_type_t::bf16;
// ... other matmul fields ...

matmul_batch_params_t bp;
matmul_direct(/*layout*/'r', /*transA*/false, /*transB*/false,
              M, N, K, /*alpha*/1.0f,
              activation, /*lda*/K,
              prepacked_buf.data(), /*ldb*/N,    // <-- prepacked weight
              bias, /*beta*/0.0f, dst, /*ldc*/N,
              /*is_weights_const*/true,
              bp, mp);
```

### Caller Responsibilities

The library trusts the caller to keep prepack-time and matmul-time parameters in sync. A mismatch produces **silently wrong results** — no error, no crash, just bad math. Make sure these match between the `reorder_direct` (prepack) call and the subsequent `matmul_direct` call:

- `K`, `N`, `ldb`, `transposed`
- `wei_dtype`, `src_dtype`
- `sym_group_size` (for the s8 sym-quant variant)

If you mutate any of these on `rp.prepack` between the size query and the prepack call (or between the prepack call and the matmul call), the buffer's layout will not match what the matmul kernel expects.

### Buffer Lifetime

The library never allocates the prepacked buffer; the caller owns it end-to-end. As long as the buffer outlives the last `matmul_direct` call that uses it (with `mem_format_b = 'r'`), there is no freeing or special teardown to remember.


## Usage Examples

### Example 1: Per-Tensor Quantization (BF16 → INT8)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int bf16_to_int8_per_tensor_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 128;
  constexpr int64_t N = 256;
  
  // Per-tensor: single scale and zero_point
  float scale = 0.5f;
  int32_t zero_point = 0;
  
  // Allocate buffers (BF16 stored as uint16_t)
  std::vector<uint16_t> input_bf16(M * N);
  std::vector<int8_t> output_int8(M * N);
  
  // Initialize input...
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::bf16;
  params.dst_dtype = data_type_t::s8;
  params.src_shape = {M, N};  // 2D matrix (mandatory)
  params.dst_shape = {M, N};  // Must match src_shape
  
  // Per-tensor: dims = {1, 1} for 2D
  params.quant_params.scale.buff = &scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {1, 1};  // per-tensor
  
  params.quant_params.zero_point.buff = &zero_point;
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.quant_params.zero_point.dims = {1, 1};  // per-tensor
  
  params.algo = reorder_algo_t::DT;
  
  // Execute quantization
  status_t status = reorder_direct(input_bf16.data(), output_int8.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 2: Per-Channel Quantization (BF16 → INT8)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int bf16_to_int8_per_channel_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 128;
  constexpr int64_t N = 4;
  
  // Per-channel: different scale/zp for each column (N values)
  std::vector<float> scales = {0.25f, 0.5f, 0.75f, 1.0f};
  std::vector<int32_t> zero_points = {0, 5, -5, 10};
  
  // Allocate buffers
  std::vector<uint16_t> input_bf16(M * N);
  std::vector<int8_t> output_int8(M * N);
  
  // Initialize input...
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::bf16;
  params.dst_dtype = data_type_t::s8;
  params.src_shape = {M, N};  // 2D matrix
  params.dst_shape = {M, N};  // Must match src_shape
  
  // Per-channel: dims = {1, N} for 2D (N values, one per column)
  params.quant_params.scale.buff = scales.data();
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {1, N};  // per-channel
  
  params.quant_params.zero_point.buff = zero_points.data();
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.quant_params.zero_point.dims = {1, N};  // per-channel
  
  params.algo = reorder_algo_t::DT;
  
  // Execute quantization
  status_t status = reorder_direct(input_bf16.data(), output_int8.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 3: Per-Group Quantization (BF16 → INT8)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int bf16_to_int8_per_group_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 8;   // Rows
  constexpr int64_t N = 4;   // Columns
  constexpr int64_t G = 2;   // Number of groups (M % G == 0)
  // group_size = M / G = 4 rows per group
  
  // Per-group: G × N total values (each group has N scale/zp values)
  // Layout: [group0_col0, group0_col1, ..., group0_colN-1, group1_col0, ...]
  std::vector<float> scales = {
    0.25f, 0.5f, 0.75f, 1.0f,    // Group 0: different per column
    0.5f, 1.0f, 1.5f, 2.0f       // Group 1: different per column
  };
  std::vector<int32_t> zero_points = {
    0, 5, -5, 10,                // Group 0
    -10, 0, 5, 15                // Group 1
  };
  
  // Allocate buffers
  std::vector<uint16_t> input_bf16(M * N);
  std::vector<int8_t> output_int8(M * N);
  
  // Initialize input...
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::bf16;
  params.dst_dtype = data_type_t::s8;
  params.src_shape = {M, N};  // 2D matrix
  params.dst_shape = {M, N};  // Must match src_shape
  
  // Per-group: dims = {G, N} for 2D (G×N total values)
  params.quant_params.scale.buff = scales.data();
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {G, N};  // per-group
  
  params.quant_params.zero_point.buff = zero_points.data();
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.quant_params.zero_point.dims = {G, N};  // per-group
  
  params.algo = reorder_algo_t::DT;
  
  // Execute quantization
  status_t status = reorder_direct(input_bf16.data(), output_int8.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 4: Dequantization (INT8 → BF16)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int int8_to_bf16_dequantization_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 128;
  constexpr int64_t N = 256;
  
  float scale = 0.5f;
  int32_t zero_point = 0;
  
  // Allocate buffers
  std::vector<int8_t> input_int8(M * N);
  std::vector<uint16_t> output_bf16(M * N);
  
  // Initialize input...
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::s8;
  params.dst_dtype = data_type_t::bf16;
  params.src_shape = {M, N};  // 2D matrix
  params.dst_shape = {M, N};  // Must match src_shape
  
  // Per-tensor: dims = {1, 1}
  params.quant_params.scale.buff = &scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {1, 1};
  
  params.quant_params.zero_point.buff = &zero_point;
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.quant_params.zero_point.dims = {1, 1};
  
  params.algo = reorder_algo_t::DT;
  
  // Execute dequantization
  status_t status = reorder_direct(input_int8.data(), output_bf16.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 5: Strided Source Memory

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int strided_reorder_example() {
  using namespace zendnnl::lowoha::reorder;
  
  // Logical shape: [4, 4] but embedded in [4, 8] physical memory
  constexpr int64_t M = 4;
  constexpr int64_t N = 4;
  constexpr int64_t physical_cols = 8;  // Padded for alignment
  
  float scale = 0.5f;
  int32_t zero_point = 0;
  
  // Source: [4 × 8] physical layout, reading [4 × 4] logical
  std::vector<uint16_t> input_bf16(M * physical_cols);
  // Destination: contiguous [4 × 4]
  std::vector<int8_t> output_int8(M * N);
  
  // Initialize input with data in columns 0-3, padding in columns 4-7...
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::bf16;
  params.dst_dtype = data_type_t::s8;
  params.src_shape = {M, N};  // Logical shape
  params.dst_shape = {M, N};  // Must match src_shape
  params.src_strides = {physical_cols, 1};  // stride_M=8, stride_N=1
  // dst_strides not set - destination is always contiguous
  
  params.quant_params.scale.buff = &scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {1, 1};
  
  params.quant_params.zero_point.buff = &zero_point;
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.quant_params.zero_point.dims = {1, 1};
  
  params.algo = reorder_algo_t::DT;
  
  // Execute - reads strided input, writes contiguous output
  status_t status = reorder_direct(input_bf16.data(), output_int8.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 6: 3D Batched Tensor with Per-Tensor Scale

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int batched_reorder_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t batch = 4;
  constexpr int64_t M = 32;
  constexpr int64_t N = 64;
  
  float scale = 0.5f;
  int32_t zero_point = 0;
  
  std::vector<uint16_t> input_bf16(batch * M * N);
  std::vector<int8_t> output_int8(batch * M * N);
  
  // Initialize input...
  
  reorder_params_t params;
  params.src_dtype = data_type_t::bf16;
  params.dst_dtype = data_type_t::s8;
  params.src_shape = {batch, M, N};  // 3D batched matrix
  params.dst_shape = {batch, M, N};  // Must match src_shape
  
  // Per-tensor: dims = {1, 1, 1} for 3D
  params.quant_params.scale.buff = &scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {1, 1, 1};
  
  params.quant_params.zero_point.buff = &zero_point;
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.quant_params.zero_point.dims = {1, 1, 1};
  
  params.algo = reorder_algo_t::DT;
  
  status_t status = reorder_direct(input_bf16.data(), output_int8.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 7: FP32 Quantization (FP32 → INT8)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int f32_to_int8_quantization_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 128;
  constexpr int64_t N = 256;
  
  float scale = 0.5f;
  int32_t zero_point = 0;
  
  // Allocate buffers
  std::vector<float> input_f32(M * N);
  std::vector<int8_t> output_int8(M * N);
  
  // Initialize input...
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::f32;
  params.dst_dtype = data_type_t::s8;
  params.src_shape = {M, N};  // 2D matrix
  params.dst_shape = {M, N};  // Must match src_shape
  
  // Per-tensor: dims = {1, 1} for 2D
  params.quant_params.scale.buff = &scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {1, 1};
  
  params.quant_params.zero_point.buff = &zero_point;
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.quant_params.zero_point.dims = {1, 1};
  
  params.algo = reorder_algo_t::DT;
  
  // Execute quantization
  status_t status = reorder_direct(input_f32.data(), output_int8.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 8: FP32 Dequantization (INT8 → FP32)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int int8_to_f32_dequantization_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 128;
  constexpr int64_t N = 256;
  
  float scale = 0.5f;
  int32_t zero_point = 0;
  
  // Allocate buffers
  std::vector<int8_t> input_int8(M * N);
  std::vector<float> output_f32(M * N);
  
  // Initialize input...
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::s8;
  params.dst_dtype = data_type_t::f32;
  params.src_shape = {M, N};  // 2D matrix
  params.dst_shape = {M, N};  // Must match src_shape
  
  // Per-tensor: dims = {1, 1}
  params.quant_params.scale.buff = &scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {1, 1};
  
  params.quant_params.zero_point.buff = &zero_point;
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.quant_params.zero_point.dims = {1, 1};
  
  params.algo = reorder_algo_t::DT;
  
  // Execute dequantization
  status_t status = reorder_direct(input_int8.data(), output_f32.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 9: FP32 Per-Channel Quantization (FP32 → UINT8)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int f32_to_uint8_per_channel_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 128;
  constexpr int64_t N = 4;
  
  // Per-channel: different scale/zp for each column
  std::vector<float> scales = {0.25f, 0.5f, 0.75f, 1.0f};
  std::vector<int32_t> zero_points = {128, 130, 125, 128};  // Typical for UINT8
  
  // Allocate buffers
  std::vector<float> input_f32(M * N);
  std::vector<uint8_t> output_uint8(M * N);
  
  // Initialize input...
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::f32;
  params.dst_dtype = data_type_t::u8;
  params.src_shape = {M, N};  // 2D matrix
  params.dst_shape = {M, N};  // Must match src_shape
  
  // Per-channel: dims = {1, N} for 2D
  params.quant_params.scale.buff = scales.data();
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {1, N};
  
  params.quant_params.zero_point.buff = zero_points.data();
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.quant_params.zero_point.dims = {1, N};
  
  params.algo = reorder_algo_t::DT;
  
  // Execute quantization
  status_t status = reorder_direct(input_f32.data(), output_uint8.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 10: FP32 to BF16 Simple Conversion (No Scale/Zero-Point)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int f32_to_bf16_simple_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 128;
  constexpr int64_t N = 256;
  
  // Allocate buffers (BF16 stored as uint16_t)
  std::vector<float> input_f32(M * N);
  std::vector<uint16_t> output_bf16(M * N);
  
  // Initialize input...
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::f32;
  params.dst_dtype = data_type_t::bf16;
  params.src_shape = {M, N};  // 2D matrix
  params.dst_shape = {M, N};  // Must match src_shape
  // No scale/zp parameters - simple type conversion
  
  params.algo = reorder_algo_t::DT;
  
  // Execute conversion
  status_t status = reorder_direct(input_f32.data(), output_bf16.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 11: FP32 to BF16 with Scale/Zero-Point

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int f32_to_bf16_with_scale_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 128;
  constexpr int64_t N = 256;
  
  float scale = 0.5f;
  int32_t zero_point = 2;
  
  // Allocate buffers
  std::vector<float> input_f32(M * N);
  std::vector<uint16_t> output_bf16(M * N);
  
  // Initialize input...
  
  // Configure reorder parameters
  // Formula: bf16_val = bf16(f32_val / scale + zero_point)
  reorder_params_t params;
  params.src_dtype = data_type_t::f32;
  params.dst_dtype = data_type_t::bf16;
  params.src_shape = {M, N};  // 2D matrix
  params.dst_shape = {M, N};  // Must match src_shape
  
  // Per-tensor: dims = {1, 1} for 2D
  params.quant_params.scale.buff = &scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {1, 1};
  
  params.quant_params.zero_point.buff = &zero_point;
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.quant_params.zero_point.dims = {1, 1};
  
  params.algo = reorder_algo_t::DT;
  
  // Execute conversion
  status_t status = reorder_direct(input_f32.data(), output_bf16.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 12: BF16 to FP32 Simple Conversion (No Scale/Zero-Point)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int bf16_to_f32_simple_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 128;
  constexpr int64_t N = 256;
  
  // Allocate buffers (BF16 stored as uint16_t)
  std::vector<uint16_t> input_bf16(M * N);
  std::vector<float> output_f32(M * N);
  
  // Initialize input...
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::bf16;
  params.dst_dtype = data_type_t::f32;
  params.src_shape = {M, N};  // 2D matrix
  params.dst_shape = {M, N};  // Must match src_shape
  // No scale/zp parameters - simple type conversion
  
  params.algo = reorder_algo_t::DT;
  
  // Execute conversion
  status_t status = reorder_direct(input_bf16.data(), output_f32.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 13: BF16 to FP32 with Scale/Zero-Point

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int bf16_to_f32_with_scale_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 128;
  constexpr int64_t N = 256;
  
  float scale = 0.5f;
  int32_t zero_point = 2;
  
  // Allocate buffers
  std::vector<uint16_t> input_bf16(M * N);
  std::vector<float> output_f32(M * N);
  
  // Initialize input...
  
  // Configure reorder parameters
  // Formula: f32_val = (bf16_as_f32 - zero_point) * scale
  reorder_params_t params;
  params.src_dtype = data_type_t::bf16;
  params.dst_dtype = data_type_t::f32;
  params.src_shape = {M, N};  // 2D matrix
  params.dst_shape = {M, N};  // Must match src_shape
  
  // Per-tensor: dims = {1, 1} for 2D
  params.quant_params.scale.buff = &scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {1, 1};
  
  params.quant_params.zero_point.buff = &zero_point;
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.quant_params.zero_point.dims = {1, 1};
  
  params.algo = reorder_algo_t::DT;
  
  // Execute conversion
  status_t status = reorder_direct(input_bf16.data(), output_f32.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 14: FP32 to F16 Simple Conversion (No Scale/Zero-Point)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int f32_to_f16_simple_example() {
  using namespace zendnnl::lowoha::reorder;

  constexpr int64_t M = 128;
  constexpr int64_t N = 256;

  // F16 destination buffer is stored as uint16_t.
  std::vector<float>    input_f32(M * N);
  std::vector<uint16_t> output_f16(M * N);

  // Initialize input...

  reorder_params_t params;
  params.src_dtype = data_type_t::f32;
  params.dst_dtype = data_type_t::f16;
  params.src_shape = {M, N};
  params.dst_shape = {M, N};
  // No scale/zp -> simple narrow with round-to-nearest-even (VCVTPS2PH).

  params.algo = reorder_algo_t::DT;

  status_t status = reorder_direct(input_f32.data(), output_f16.data(), params);
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 15: F16 to FP32 with Scale/Zero-Point

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int f16_to_f32_with_scale_example() {
  using namespace zendnnl::lowoha::reorder;

  constexpr int64_t M = 128;
  constexpr int64_t N = 256;

  float scale = 0.5f;
  int32_t zero_point = 2;

  std::vector<uint16_t> input_f16(M * N);   // F16 stored as uint16_t
  std::vector<float>    output_f32(M * N);

  // Initialize input...

  // Formula applied: f32_val = (f32(f16) - zp) * scale
  reorder_params_t params;
  params.src_dtype = data_type_t::f16;
  params.dst_dtype = data_type_t::f32;
  params.src_shape = {M, N};
  params.dst_shape = {M, N};

  params.quant_params.scale.buff = &scale;
  params.quant_params.scale.dt   = data_type_t::f32;
  params.quant_params.scale.dims = {1, 1};

  params.quant_params.zero_point.buff = &zero_point;
  params.quant_params.zero_point.dt   = data_type_t::s32;
  params.quant_params.zero_point.dims = {1, 1};

  params.algo = reorder_algo_t::DT;

  status_t status = reorder_direct(input_f16.data(), output_f32.data(), params);
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 16: BF16 to F16 Simple Conversion (No Scale/Zero-Point)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int bf16_to_f16_simple_example() {
  using namespace zendnnl::lowoha::reorder;

  constexpr int64_t M = 128;
  constexpr int64_t N = 256;

  // Both BF16 and F16 are stored as uint16_t in user buffers.
  std::vector<uint16_t> input_bf16(M * N);
  std::vector<uint16_t> output_f16(M * N);

  // Initialize input (BF16 layout)...

  reorder_params_t params;
  params.src_dtype = data_type_t::bf16;
  params.dst_dtype = data_type_t::f16;
  params.src_shape = {M, N};
  params.dst_shape = {M, N};
  // No scale/zp; the kernel widens BF16 -> FP32 then narrows FP32 -> F16.

  params.algo = reorder_algo_t::DT;

  status_t status = reorder_direct(input_bf16.data(), output_f16.data(), params);
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 17: F16 to BF16 with Scale/Zero-Point (Per-Tensor)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int f16_to_bf16_with_scale_example() {
  using namespace zendnnl::lowoha::reorder;

  constexpr int64_t M = 128;
  constexpr int64_t N = 256;

  float scale = 0.5f;
  int32_t zero_point = 2;

  std::vector<uint16_t> input_f16(M * N);
  std::vector<uint16_t> output_bf16(M * N);

  // Initialize input...

  // Formula applied: bf16_val = bf16( (f32(f16) - zp) * scale )
  reorder_params_t params;
  params.src_dtype = data_type_t::f16;
  params.dst_dtype = data_type_t::bf16;
  params.src_shape = {M, N};
  params.dst_shape = {M, N};

  params.quant_params.scale.buff = &scale;
  params.quant_params.scale.dt   = data_type_t::f32;
  params.quant_params.scale.dims = {1, 1};

  params.quant_params.zero_point.buff = &zero_point;
  params.quant_params.zero_point.dt   = data_type_t::s32;
  params.quant_params.zero_point.dims = {1, 1};

  params.algo = reorder_algo_t::DT;

  status_t status = reorder_direct(input_f16.data(), output_bf16.data(), params);
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 18: FP32 to F16 Per-Channel Conversion

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int f32_to_f16_per_channel_example() {
  using namespace zendnnl::lowoha::reorder;

  constexpr int64_t M = 128;
  constexpr int64_t N = 4;

  // One scale/zp per column.
  std::vector<float>   scales      = {0.25f, 0.5f, 0.75f, 1.0f};
  std::vector<int32_t> zero_points = {0, 1, 2, 3};

  std::vector<float>    input_f32(M * N);
  std::vector<uint16_t> output_f16(M * N);

  // Initialize input...

  reorder_params_t params;
  params.src_dtype = data_type_t::f32;
  params.dst_dtype = data_type_t::f16;
  params.src_shape = {M, N};
  params.dst_shape = {M, N};

  params.quant_params.scale.buff      = scales.data();
  params.quant_params.scale.dt        = data_type_t::f32;
  params.quant_params.scale.dims      = {1, N};        // per-channel-col

  params.quant_params.zero_point.buff = zero_points.data();
  params.quant_params.zero_point.dt   = data_type_t::s32;
  params.quant_params.zero_point.dims = {1, N};

  params.algo = reorder_algo_t::DT;

  status_t status = reorder_direct(input_f32.data(), output_f16.data(), params);
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 19: Dynamic Quantization — Symmetric Per-Tensor (BF16 → S8)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int dynamic_quant_symmetric_per_tensor_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 128;
  constexpr int64_t N = 256;
  
  // Allocate buffers
  std::vector<uint16_t> input_bf16(M * N);
  std::vector<int8_t> output_s8(M * N);
  
  // Initialize input...
  
  // Output buffer for computed scale (API will fill this)
  float computed_scale = 0.0f;
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::bf16;
  params.dst_dtype = data_type_t::s8;
  params.src_shape = {M, N};
  params.dst_shape = {M, N};
  params.dynamic_quant = true;  // Enable dynamic quantization
  
  // Provide output buffer for scale
  params.quant_params.scale.buff = &computed_scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {1, 1};  // per-tensor
  // zero_point.buff = nullptr → symmetric mode (zp = 0 implicitly)
  
  // Execute: computes scale from data, then quantizes
  status_t status = reorder_direct(input_bf16.data(), output_s8.data(), params);
  
  // computed_scale is now filled with: max(|min|, |max|) / 127
  // Dequantize: original ≈ output_s8[i] * computed_scale
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 20: Dynamic Quantization — Asymmetric Per-Tensor (FP32 → U8)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int dynamic_quant_asymmetric_per_tensor_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 128;
  constexpr int64_t N = 256;
  
  // Allocate buffers
  std::vector<float> input_f32(M * N);
  std::vector<uint8_t> output_u8(M * N);
  
  // Initialize input...
  
  // Output buffers for computed scale and zero_point (API will fill these)
  float computed_scale = 0.0f;
  int32_t computed_zp = 0;
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::f32;
  params.dst_dtype = data_type_t::u8;
  params.src_shape = {M, N};
  params.dst_shape = {M, N};
  params.dynamic_quant = true;  // Enable dynamic quantization
  
  // Provide output buffer for scale
  params.quant_params.scale.buff = &computed_scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {1, 1};  // per-tensor
  
  // Provide output buffer for zero_point → asymmetric mode
  params.quant_params.zero_point.buff = &computed_zp;
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.quant_params.zero_point.dims = {1, 1};  // per-tensor
  
  // Execute: computes scale/zp from data, then quantizes
  status_t status = reorder_direct(input_f32.data(), output_u8.data(), params);
  
  // computed_scale = (max - min) / 255
  // computed_zp = round(-min / scale)
  // Dequantize: original ≈ (output_u8[i] - computed_zp) * computed_scale
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 21: Dynamic Quantization — Per-Token / Per-Row (BF16 → S8)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int dynamic_quant_per_token_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 4;   // 4 tokens (rows)
  constexpr int64_t N = 256;
  
  // Allocate buffers
  std::vector<uint16_t> input_bf16(M * N);
  std::vector<int8_t> output_s8(M * N);
  
  // Initialize input (each row may have different value ranges)...
  
  // Output buffer: M scales, one per row
  std::vector<float> computed_scales(M, 0.0f);
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::bf16;
  params.dst_dtype = data_type_t::s8;
  params.src_shape = {M, N};
  params.dst_shape = {M, N};
  params.dynamic_quant = true;
  
  // Per-row (per-token): dims = {M, 1}
  params.quant_params.scale.buff = computed_scales.data();
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {M, 1};  // per-row (per-token)
  // zero_point.buff = nullptr → symmetric mode
  
  // Execute: computes M separate scales (one per row), then quantizes
  status_t status = reorder_direct(input_bf16.data(), output_s8.data(), params);
  
  // Each computed_scales[i] = max(|min_row_i|, |max_row_i|) / 127
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 22: Dynamic Quantization — Compute-Only Mode (dst = nullptr)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int dynamic_quant_compute_only_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 128;
  constexpr int64_t N = 256;
  
  // Allocate source buffer
  std::vector<float> input_f32(M * N);
  
  // Initialize input...
  
  // Output buffer for computed scale
  float computed_scale = 0.0f;
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::f32;
  params.dst_dtype = data_type_t::s8;
  params.src_shape = {M, N};
  params.dst_shape = {M, N};
  params.dynamic_quant = true;
  
  params.quant_params.scale.buff = &computed_scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {1, 1};  // per-tensor
  
  // dst = nullptr: ONLY compute scale, do NOT perform quantization
  status_t status = reorder_direct(input_f32.data(), nullptr, params);
  
  // computed_scale is now filled; no quantized output was produced
  // Use computed_scale later for actual quantization if needed
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 23: Dynamic Quantization — Asymmetric Per-Token (BF16 → U8)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int dynamic_quant_asymmetric_per_token_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 4;   // 4 tokens (rows)
  constexpr int64_t N = 256;
  
  // Allocate buffers
  std::vector<uint16_t> input_bf16(M * N);
  std::vector<uint8_t> output_u8(M * N);
  
  // Initialize input...
  
  // Output buffers: M values each, one per row
  std::vector<float> computed_scales(M, 0.0f);
  std::vector<int32_t> computed_zps(M, 0);
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::bf16;
  params.dst_dtype = data_type_t::u8;
  params.src_shape = {M, N};
  params.dst_shape = {M, N};
  params.dynamic_quant = true;
  
  // Per-row (per-token): dims = {M, 1}
  params.quant_params.scale.buff = computed_scales.data();
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {M, 1};
  
  // Providing zp buffer → asymmetric mode
  params.quant_params.zero_point.buff = computed_zps.data();
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.quant_params.zero_point.dims = {M, 1};
  
  // Execute: computes per-row scale/zp, then quantizes
  status_t status = reorder_direct(input_bf16.data(), output_u8.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 24: Dynamic Quantization — Per-Group-Row (BF16 → S8)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int dynamic_quant_per_group_row_example() {
  using namespace zendnnl::lowoha::reorder;
  
  constexpr int64_t M = 8;   // Rows
  constexpr int64_t N = 4;   // Columns
  constexpr int64_t G = 2;   // Number of groups (M % G == 0)
  // group_size = M / G = 4 rows per group
  
  // Allocate buffers
  std::vector<uint16_t> input_bf16(M * N);
  std::vector<int8_t> output_s8(M * N);
  
  // Initialize input...
  
  // Output buffer: G × N scales
  std::vector<float> computed_scales(G * N, 0.0f);
  
  // Configure reorder parameters
  reorder_params_t params;
  params.src_dtype = data_type_t::bf16;
  params.dst_dtype = data_type_t::s8;
  params.src_shape = {M, N};
  params.dst_shape = {M, N};
  params.dynamic_quant = true;
  
  // Per-group-row: dims = {G, N}
  params.quant_params.scale.buff = computed_scales.data();
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.scale.dims = {G, N};  // per-group-row
  // zero_point.buff = nullptr → symmetric mode
  
  // Execute: computes G×N separate scales, then quantizes
  status_t status = reorder_direct(input_bf16.data(), output_s8.data(), params);
  
  return (status == status_t::success) ? 0 : -1;
}
```


## Validation

The operator performs the following validations:

### Standard Reorder Validation

1. **Null pointer checks:** Source and destination buffers must not be null
2. **Shape validation:** src_shape and dst_shape must be non-empty with all positive dimensions
3. **Shape matching:** src_shape and dst_shape must be identical (error thrown if different)
4. **Data type validation:** Source/destination type combination must be supported
5. **Scale validation:** Must be finite (for f32 or bf16)
6. **Zero-point validation:** Zero-point values may be any `int32`; during quantization they are clamped to the valid range of the destination type, which can increase saturation and reduce dequantization accuracy if the specified zero-point lies outside that range.
7. **Dims validation:** Must match tensor dimensionality and follow granularity rules
8. **Per-group validation:** M must be divisible by G (row groups) or N must be divisible by G (column groups)
9. **Destination strides:** dst_strides must be empty (strided destination not currently supported)

### Dynamic Quantization Validation (when `dynamic_quant = true`)

10. **Source data type:** Must be BF16 or FP32
11. **Destination data type:** Must be S8 (symmetric) or U8 (asymmetric)
12. **Scale buffer:** Must be non-null (output buffer required)
13. **Scale dims:** Must be specified and valid for the tensor shape
14. **Scale data type:** Must be f32 or bf16
15. **Symmetric mode:** If zero_point.buff is nullptr, destination must be S8
16. **Asymmetric mode:** If zero_point.buff is provided, destination must be U8, and zero_point dims must match scale granularity
17. **Zero-point data type:** Must be s32 (when provided)

### Weight Prepack Validation (when `is_prepack = true`)

18. **Null pointer checks:** `weights` and `dst` buffers must not be null
19. **Dimensions:** `prepack.K > 0` and `prepack.N > 0`
20. **Leading dimension:** `prepack.ldb` must be ≥ `prepack.K` (transposed) or ≥ `prepack.N` (non-transposed)
21. **Algo:** `prepack.algo` must be `matmul_algo_t::aocl_dlp_blocked` (any other value returns `status_t::unimplemented` / `weight_prepack_size` returns 0)
22. **Buffer size:** the caller is responsible — `dst` must hold at least `weight_prepack_size(params)` bytes. The library does **not** verify this; an undersized buffer causes silent out-of-bounds writes.


## Implementation Support Matrix

The following table shows which combinations have optimized (AVX512) vs reference implementations:

### Static Reorder: BF16/FP32 ↔ S8/U8 and FP32 ↔ BF16

| Granularity | Source Contiguous | Source Strided (last_stride=1) | Source Strided (other) |
|-------------|-------------------|--------------------------------|------------------------|
| Per-tensor | ✅ Optimal | ✅ Optimal | ⚙️ Reference |
| Per-channel | ⚙️ Reference | ⚙️ Reference | ⚙️ Reference |
| Per-group | ⚙️ Reference | ⚙️ Reference | ⚙️ Reference |

### Dynamic Quantization: Parameter Computation

| Granularity | Implementation |
|-------------|---------------|
| Per-tensor | ⚙️ Reference (sequential scan) |
| Per-row | ⚙️ Reference (OpenMP parallelized) |
| Per-col | ⚙️ Reference (OpenMP parallelized) |
| Per-group-row | ⚙️ Reference (OpenMP parallelized, collapse(2)) |
| Per-group-col | ⚙️ Reference (OpenMP parallelized, collapse(2)) |

After computing dynamic quantization parameters, the standard reorder path is used for the actual quantization step, following the static reorder support matrix above.

**Legend:**
- ✅ **Optimal:** AVX512 vectorized implementation for best performance
- ⚙️ **Reference:** Scalar implementation (functionally correct, lower throughput)

**Notes:**
- Strided source memory with `stride_N = 1` (last dimension contiguous) can use optimal path for per-tensor
- Per-channel and per-group granularities currently use reference implementation
- The `DT` algorithm automatically selects the best available implementation
- Destination memory is always written contiguously (strided destination not currently supported)
- FP32 ↔ BF16 conversion supports both simple (no scale/zp) and scaled conversions
- Dynamic quantization parameter computation uses OpenMP for parallelism across rows/columns/groups


## Performance Considerations

- **Algorithm Selection:** Use `DT` (default) for automatic selection based on buffer size and configuration
- **Vectorization:** `native` algorithm uses AVX512 for large buffers (≥64 elements) with supported configurations
- **Threading:** Set `num_threads` to control parallelism (0 = use all available)
- **Source Memory Layout:** Contiguous source memory is fastest; strided source with last_stride=1 can still use optimal path
- **Destination Memory:** Always written contiguously (strided destination not currently supported)
- **Granularity:** Per-tensor is fastest with optimal support; per-channel/per-group use reference implementation
 - **Float ↔ Float Conversion (FP32, BF16, F16):** Simple conversion (no scale/zp) is fastest. The optimal `native` path is the AVX-512 per-tensor path with contiguous source memory (or 2D -> [x, 1]/ 3D -> [x, y, 1] padded stride). Availability of this path follows the implementation's current ISA/runtime selection logic for `native`; this documentation does not guarantee a separate F16C-specific dispatcher check or fallback for FP32 ↔ F16 / BF16 ↔ F16 conversions. BF16 ↔ F16 conversion goes through FP32 in registers, so it has the same per-element cost as two 16-bit ↔ FP32 conversions fused together.
- **Dynamic Quantization:** Adds a min/max scan pass over the source data before quantization. For per-tensor, this scans all elements sequentially. For per-channel and per-group, the scan is parallelized with OpenMP. The compute-only mode (`dst = nullptr`) can be used to separate the parameter computation from the quantization step.
