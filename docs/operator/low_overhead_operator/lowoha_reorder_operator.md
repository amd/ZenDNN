
(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA Reorder Operator

## Overview

The **LowOHA Reorder Operator** is a high-performance, low-overhead data type conversion operator designed for **quantization, dequantization, and data type conversion in workloads**. It provides a direct API to convert data between BF16/FP32/F16 and INT8/UINT8 formats, and for conversions among FP32, BF16, F16 in any direction, with configurable scale and zero-point parameters.

Unlike the standard Reorder operator which uses the operator factory pattern, LowOHA Reorder provides a **function-based interface** optimized for:
- Minimal execution overhead
- Quantization (BF16/FP32 → INT8/UINT8)
- Dequantization (INT8/UINT8 → BF16/FP32)
- Data type conversion (FP32 ⇔ BF16, FP32 ⇔ F16, BF16 ⇔ F16)
- Dynamic quantization (compute scale/zero-point from source data at runtime) — supports BF16, FP32, and FP16 sources
- Per-tensor, per-channel (row and column), and per-group (row and column) quantization granularities
- Strided (non-contiguous) source memory support
- Weight prepack (`is_prepack = true`) — produce a backend-blocked weight buffer once at model load and reuse it across matmul calls; for static-INT8 with a `u8` source it also precomputes the per-column weight sums the matmul needs for zero-point compensation (see *Weight Prepack Mode*)
- For per-tensor FP16 source/destination, the optimal AVX-512 path has two backends — F32-FMA (AVX-512F + F16C convert) and FP16-FMA (`__m512h`-native, AVX512-FP16 ISA) — auto-selected at dispatch time via `can_use_f16_fma_kernel()`. Build with `-DZENDNNL_NATIVE_F32_ACCUM=ON` to pin reorder to the F32-FMA path. No runtime env var.


## Quantization/Dequantization/Conversion Formulas

### Quantization (BF16/FP32/FP16 → INT8)

$$
\mathrm{int8} = \mathrm{clamp}(\mathrm{round}(\frac{\mathrm{input}}{\mathrm{scale}}) + \mathrm{zp}, -128, 127)
$$

`round` is round-to-nearest-even (banker's rounding). `clamp` saturates out-of-range quotients to the s8 endpoints; this is how a user-supplied `int32` `zp` outside the s8 range gets reduced — see *Validation Rules §Zero-point validation*.

### Quantization (BF16/FP32/FP16 → UINT8)

$$
\mathrm{uint8} = \mathrm{clamp}(\mathrm{round}(\frac{\mathrm{input}}{\mathrm{scale}}) + \mathrm{zp}, 0, 255)
$$

### Dequantization (INT8/UINT8 → BF16/FP32/FP16)

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

### Data Type Conversion (FP32 → FP16)

**Simple conversion (no scale/zero-point):**

$$
\mathrm{f16} = \mathrm{f16}(\mathrm{f32})
$$

**With scale and zero-point:**

$$
\mathrm{f16} = \mathrm{f16}(\frac{\mathrm{f32}}{\mathrm{scale}} + \mathrm{zp})
$$

### Data Type Conversion (FP16 → FP32)

**Simple conversion (no scale/zero-point):**

$$
\mathrm{f32} = \mathrm{f32}(\mathrm{f16})
$$

**With scale and zero-point:**

$$
\mathrm{f32} = (\mathrm{f32}(\mathrm{f16}) - \mathrm{zp}) \times \mathrm{scale}
$$

### Data Type Conversion (BF16 → FP16)

Routes through FP32 internally (`bf16 → f32 → f16`).

**Simple conversion (no scale/zero-point):**

$$
\mathrm{f16} = \mathrm{f16}(\mathrm{f32}(\mathrm{bf16}))
$$

**With scale and zero-point:**

$$
\mathrm{f16} = \mathrm{f16}(\frac{\mathrm{f32}(\mathrm{bf16})}{\mathrm{scale}} + \mathrm{zp})
$$

### Data Type Conversion (FP16 → BF16)

Routes through FP32 internally (`f16 → f32 → bf16`).

**Simple conversion (no scale/zero-point):**

$$
\mathrm{bf16} = \mathrm{bf16}(\mathrm{f32}(\mathrm{f16}))
$$

**With scale and zero-point:**

$$
\mathrm{bf16} = \mathrm{bf16}(\frac{\mathrm{f32}(\mathrm{f16})}{\mathrm{scale}} + \mathrm{zp})
$$

> **Note on the cross-format conversions (BF16 ↔ FP16):** Both directions go through FP32 as an intermediate. FP32 is a strict superset of both BF16 (which keeps the 8 exponent bits + 7 mantissa bits) and FP16 (5 exponent + 10 mantissa), so the intermediate is exact. The destination narrow then loses precision only at the destination dtype's representable limit — FP16's narrower exponent range (`~6.1e-5` to `65504`) is the binding constraint when converting BF16 → FP16, while FP16's wider mantissa (10 vs 7 bits) means FP16 → BF16 loses 3 mantissa bits to the BF16 narrow.

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
\mathrm{zp} = \mathrm{clamp}(\mathrm{round}(\frac{-\min(A)}{\mathrm{scale}}),\ \mathrm{INT32\_MIN},\ \mathrm{INT32\_MAX})
$$

Where $A$ is the set of source values within the quantization scope (per-tensor, per-channel, or per-group).

> **Scale floor:** Both symmetric and asymmetric `scale` values are clamped to a positive floor before being returned. The internal compute floor is `1e-10`. When the user-supplied scale buffer dtype is `f16`, the floor is raised to FP16's minimum positive normal `2^-14 ≈ 6.10e-5` before the narrow-to-f16 step so the stored f16 scale is always non-zero and the downstream static-dequant `round(val / scale)` cannot divide by zero. This applies on both the fused-quant fast path and the `compute_dynamic_quant_params` fall-through path — `f16` scale buffers route through the fall-through path to keep `dst` and the stored f16 scale consistent (see *Implementation Support Matrix §Dynamic Quantization: Fused fast paths*).

> **Non-finite source handling — statistics pass only:** Non-finite source values (NaN, ±Inf) are skipped when computing the per-scope min/max statistics in Pass 1 — they neither contribute to $\min(A)$ / $\max(A)$ nor to $\max(|\min(A)|, |\max(A)|)$. If a scope contains only non-finite values, the scope is treated as an empty set when computing the parameters: the result is a benign $(\mathrm{scale}, \mathrm{zp})$ reset. For asymmetric this gives $\mathrm{scale} = 1/255$ and $\mathrm{zp} = 0$ (via the `max := min + 1` constant-scope guard above). For symmetric this gives $\mathrm{scale} = 10^{-10}$ (the compute-floor; `abs_max` clamps to `1e-10`, then the final `scale < 1e-10` re-clamp pins it there) and $\mathrm{zp} = 0$. All three backends (scalar reference, F32-FMA, FP16-FMA) agree on this Pass-1 / parameter-reset behaviour. Pass 2 (quantization) does not mask non-finite source lanes. A NaN or ±Inf source element therefore still produces some destination value via the standard `nearbyint(v / scale) + zp` chain followed by saturation to the destination dtype range — the exact integer result for NaN/Inf source lanes is implementation-defined (saturating C++ cast / `VCVTPH2W` indefinite semantics) and is not guaranteed to equal the zero point. If your workload may contain non-finite values in the source tensor, sanitize them upstream (e.g. `where(isfinite(x), x, 0.0)`) for deterministic outputs.


## Core API: `reorder_direct`

The primary interface for LowOHA Reorder is the `reorder_direct` function:

```cpp
status_t reorder_direct(
  const void *src,                      // Pointer to source data buffer
  void *dst,                            // Pointer to destination data buffer (nullptr allowed for dynamic quant compute-only)
  reorder_params_t &params              // Reorder parameters (non-const reference)
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

**Note:** `dst_strides` is reserved for future implementation and is **currently not supported** — the destination is always written in contiguous format. The field is currently **ignored** (it is not validated), so populating it has no effect rather than raising an error.


### Supported Data Type Combinations

| Source Type | Destination Type | Operation |
|-------------|------------------|-----------|
| BF16 | S8 (INT8) | Quantization (static + dynamic) |
| S8 (INT8) | BF16 | Dequantization |
| BF16 | U8 (UINT8) | Quantization (static + dynamic) |
| U8 (UINT8) | BF16 | Dequantization |
| FP32 | S8 (INT8) | Quantization (static + dynamic) |
| S8 (INT8) | FP32 | Dequantization |
| FP32 | U8 (UINT8) | Quantization (static + dynamic) |
| U8 (UINT8) | FP32 | Dequantization |
| FP16 | S8 (INT8) | Quantization (static + dynamic) |
| S8 (INT8) | FP16 | Dequantization (static only) |
| FP16 | U8 (UINT8) | Quantization (static + dynamic) |
| U8 (UINT8) | FP16 | Dequantization (static only) |
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

  quant_t scale;        // Scale factor (f32, bf16, or f16)
  quant_t zero_point;   // Zero point offset (s32 only)
};
```

**Currently Supported Data Types:**

| Parameter | Supported Type | Description |
|-----------|---------------|-------------|
| `scale` | `f32`, `bf16`, or `f16` | Scale factor (must be finite; `bf16`/`f16` are widened to f32 transparently on read and narrowed back to the user's dtype on write in the dynamic-quant path) |
| `zero_point` | `s32` | Zero point offset |

> **Note on FP16 scale precision and range:** FP16 has a 10-bit mantissa (vs BF16's 7 and F32's 23) and a normal-number range of roughly `[6.1e-5, 65504]`. For typical workloads (activations in `[-2, 2]`, scales `≈ 0.0157`), FP16 storage is comfortably within the normal range; the narrowing introduces at most `~|scale| * 2^-11` of additional round-trip error, which sits well inside the shared BF16/FP16 test tolerance (`max_scale / 2 + 0.03`). When the reorder operator writes an `f16` scale buffer in the dynamic-quant path (any granularity, any backend) it floors the f32 scale at FP16's minimum positive normal (`2^-14 ≈ 6.10e-5`) before narrowing, so an "all-zero or very-small range" input produces a tiny but non-zero f16 scale and the downstream static-quant read-back never divides by zero. If you anticipate genuinely low-magnitude scales (`< 6.1e-5`) and the floor itself would be visibly lossy for your workload, use `f32` or `bf16` scale buffers instead.

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

**Per-group-row constraint:** M must be divisible by G (M % G == 0), and G must be a proper divisor — `1 < G < M` (G = 1 is per-tensor and G = M is per-row)

**Per-group-col constraint:** N must be divisible by G (N % G == 0), and G must be a proper divisor — `1 < G < N` (G = 1 is per-tensor and G = N is per-col)

### 3D Tensor (shape = [batch, M, N])

| Granularity | dims | Total Values | Description |
|-------------|------|--------------|-------------|
| Per-tensor | `{1, 1, 1}` | 1 | Single scale/zp for entire tensor |
| Per-col | `{1, 1, N}` | N | Different scale/zp for each column |
| Per-row | `{1, M, 1}` | M | Different scale/zp for each row (per-token) |
| Per-group-row | `{1, G, N}` | G × N | G groups across rows, each with N values |
| Per-group-col | `{1, M, G}` | M × G | G groups across columns, each row has G values |

**Per-group-row constraint:** M must be divisible by G (M % G == 0), and G must be a proper divisor — `1 < G < M` (G = 1 is per-tensor and G = M is per-row)

**Per-group-col constraint:** N must be divisible by G (N % G == 0), and G must be a proper divisor — `1 < G < N` (G = 1 is per-tensor and G = N is per-col)

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
| FP16 | S8 (INT8) | Symmetric |
| BF16 | U8 (UINT8) | Asymmetric |
| FP32 | U8 (UINT8) | Asymmetric |
| FP16 | U8 (UINT8) | Asymmetric |

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
| `scale.buff` | **Always** | `f32`, `bf16`, or `f16` | Product of `scale.dims` |
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

Two layouts are supported — **AOCL DLP blocked** and the **group_matmul
custom-kernel (CK) VNNI** layout:

| `prepack.algo` | Supported | Notes |
|----------------|-----------|-------|
| `matmul_algo_t::aocl_dlp_blocked` | ✅ Yes | f32 / bf16 / f16 / s8 (+ s8 sym-quant variant) / s4 / u4 |
| `matmul_algo_t::moe_custom_kernel` | ✅ Yes | bf16 (VDPBF16PS) / s8 (DQ-INT8 VPDPBUSD). Packs into the caller `dst`; consumed by `group_matmul` with `mem_format_b='r'` (see *Custom-Kernel (group_matmul) Prepack*). |
| `matmul_algo_t::libxsmm_blocked`  | ❌ No  | Layout would mis-match if the matmul-side partitioner falls back to AOCL DLP — silent wrong results. |
| `matmul_algo_t::onednn_blocked`   | ❌ No  | OneDNN's blocked layout depends on (M, K, N, dtypes, post-ops, ISA) — the prepack can't reproduce the matmul-time layout. |
| Anything else                     | ❌ No  | Non-blocked variants consume raw weights; no prepack needed. |

Anything other than `aocl_dlp_blocked` or `moe_custom_kernel` is rejected at validation: `weight_prepack_size` returns `0` and the `reorder_direct` prepack path returns `status_t::failure` (with a clear error in the log). `status_t::unimplemented` is reserved for an unsupported `wei_dtype` inside the AOCL DLP backend.

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
  matmul_algo_t algo;           // Prepack layout selector:
                                //   matmul_algo_t::aocl_dlp_blocked  -> AOCL DLP blocked
                                //   matmul_algo_t::moe_custom_kernel -> group_matmul CK VNNI
                                //                                       (packs into caller dst)
  data_type_t   wei_dtype;      // Weight data type (f32 / bf16 / f16 / s8 / s4 / u4)
  data_type_t   src_dtype;      // Source (matmul A) dtype. Disambiguates s8 vs u8
                                //   src, and (with wei_dtype = s8) decides whether
                                //   the column-sum buffer is appended -- see
                                //   "Appended Column-Sum Buffer". MUST equal the
                                //   matmul's dtypes.src.
  int64_t       K;              // Weight rows
  int64_t       N;              // Weight cols
  int64_t       ldb;            // Physical leading dimension of the input weights
  bool          transposed;     // true => weights are column-major ('ba')
  int           sym_group_size; // AOCL: >0 selects s8 sym-quant variant

  // --- group_matmul custom-kernel (CK) prepack: used only when
  //     algo == matmul_algo_t::moe_custom_kernel (see "Custom-Kernel
  //     (group_matmul) Prepack" above) ---
  int           pack_nr;                  // CK pack width; 0 = auto (plan_pack_nr)
  bool          interleave_split_halves;  // silu/gelu [gate|up] -> interleaved (even N)

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
| `s8`        | `u8`               | `aocl_reorder_u8s8s32os32` **+ appended column-sum buffer** (see below) |
| `s8` + `sym_group_size > 0` (with `src_dtype = bf16`) | — | `aocl_reorder_s8s8s32os32_sym_quant` (DLP only) |

### Appended Column-Sum Buffer (static-INT8, `wei_dtype = s8` + `src_dtype = u8`)

For the asymmetric static-INT8 path, the matmul has to subtract a source-zero-point
correction term built from the per-column weight sums:

```
dst[m,n] = sum_k (src[m,k] - src_zp) * wei[k,n]
         = sum_k src[m,k] * wei[k,n]  -  src_zp * colsum[n]

where    colsum[n] = sum_k wei[k,n]   <- depends only on the weights
```

`colsum[n]` depends only on the weights, so recomputing it on every matmul call
costs an avoidable O(K·N) reduction. More importantly, **once the weights are in
AOCL DLP blocked layout the matmul can no longer compute it**: the blocked bytes
are not addressable as a plain `[K, N]` matrix. The prepack therefore computes
`colsum[n] = sum_k wei[k,n]` from the raw weights and **appends it to the
prepacked buffer**, which is the only correct source for the term at matmul time.

**When it is appended.** Implicitly, with no caller flag, whenever all three hold:

| Condition | Why |
|---|---|
| `prepack.wei_dtype == s8` | static-INT8 weights |
| `prepack.src_dtype == u8` | the only source dtype whose compensation the matmul recovers from the packed buffer |
| `prepack.sym_group_size <= 0` | sym-quant carries its own metadata and uses a different reorder kernel |

Every other prepack (sym-quant, WOQ, float, and `s8` weights with an `s8` / `bf16` /
`f32` source) skips it and produces a buffer with no tail.

**Layout.** The buffer is `N * sizeof(int32_t)` bytes, placed at the 64-byte-aligned
offset immediately past the AOCL-packed weights, and its own length is rounded up to
64 bytes:

```
+------------------------------------------+  offset 0
|  AOCL DLP blocked weights                |
|  (size from aocl_get_reorder_buf_size_*) |
+------------------------------------------+  round_up(packed_size, 64)
|  colsum[0 .. N-1]   (N * int32)          |
|  padding up to the next 64B boundary     |
+------------------------------------------+  weight_prepack_size(rp)
```

The writer (prepack) and the reader (matmul) both compute that offset with the same
shared helper, `static_quant_colsum_offset()`, so the size formula and the 64-byte
alignment are defined in exactly one place.

**The stored value is the raw sum, not `-src_zp * colsum[n]`.** `src_zp` is a
runtime quantity that is unknown at pack time, so the prepacked buffer stays
independent of it. The same prepacked weights are therefore valid across calls that use
different source zero-points.

> ℹ️ `weight_prepack_size` **already includes** the appended buffer. Callers who
> follow the two-step workflow need no changes and no extra allocation. Do not
> size the allocation from a raw `aocl_get_reorder_buf_size_*` call — that value
> omits the tail and the prepack will write past the end of the buffer.

### Consuming the Prepacked Buffer at Matmul Time

**AOCL DLP blocked** — hand the prepacked buffer to `matmul_direct` with:

- `matmul_params::lowoha_algo = matmul_algo_t::aocl_dlp_blocked`
- `matmul_params::mem_format_b = 'r'`

`mem_format_b = 'r'` tells the matmul backend "the `weight` pointer is already in AOCL DLP blocked layout, skip the internal reorder". `matmul_direct` validates that `lowoha_algo == aocl_dlp_blocked` whenever `mem_format_b == 'r'`; any other algo is rejected up front (would otherwise silently produce wrong results).

**Static-INT8: `dtypes.src` must match the `prepack.src_dtype` you packed with.**
The matmul reads the appended column-sum buffer when *all* of the following hold:

- the weights were supplied prepacked (`mem_format_b == 'r'` on entry),
- `dtypes.src == u8`, and
- the source zero-point is non-zero.

Those are matmul-side facts. The library has no way to inspect a `void *` weight
buffer and discover whether a tail was actually appended, so it trusts that
`dtypes.src` matches the `prepack.src_dtype` used to build the buffer. Getting this
wrong is a **memory-safety bug, not just a numerical one**:

| Prepacked with | Matmul called with | Result |
|---|---|---|
| `src_dtype = u8` | `dtypes.src = u8` | ✅ Correct — tail present, tail read |
| `src_dtype = s8` / `bf16` / `f32` | `dtypes.src = u8`, `src_zp != 0` | ❌ **Out-of-bounds read** past the end of the buffer — no tail was appended, but the matmul reads `N * int32` where one would have been |
| `src_dtype = u8` | `dtypes.src = s8`, `src_zp != 0` | ⛔ Rejected with `status_t::unimplemented` (see below) |
| any | `src_zp == 0` | ✅ Symmetric — no compensation, tail (if present) is ignored |

The one case the library *can* detect is an `s8` source: `matmul_direct` rejects
`mem_format_b == 'r'` + `dtypes.src == s8` + non-zero per-tensor `src_zp` with
`status_t::unimplemented`, because no prepack configuration ever produces a tail for
an `s8` source and the blocked bytes cannot be reduced to recover one. Use a `u8`
source, a symmetric `s8` source (`src_zp == 0`), or non-prepacked weights
(`mem_format_b = 'n'`).

Transposed weights are supported: pass `transB = true` to `matmul_direct` when the
buffer was packed with `prepack.transposed = true`. The column-sum is always taken
over the logical `[K, N]` weight, so `colsum[n]` is the same value either way.

**Custom-kernel (group_matmul) VNNI** — hand the prepacked buffer to `group_matmul_direct` (per expert) with:

- `matmul_params::mem_format_b = 'r'`
- `is_weights_const = true`
- the decode custom-kernel path engaged (`ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=1` + ALGO 3)

`mem_format_b = 'r'` tells the custom kernel "the `weight` pointer is already in the CK VNNI layout — consume it directly, do not pack and do not cache". The kernel **aliases** the caller buffer (no copy, no LRU). A `'r'` weight is **CK-only**: if the call cannot be served by the custom kernel (CK off, unsupported shape/host, ALGO ≠ 3), `group_matmul_direct` **fails the call** (`status_t::failure`) rather than let a non-CK executor mis-read the packed bytes. See the group_matmul docs for the full *CK-only-or-fail* contract.

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

#### Static-INT8 Asymmetric (u8 source) — with the appended column-sum

Same two-step workflow; the only difference is `src_dtype = u8`, which makes the
prepack append the column-sum buffer. Nothing extra to allocate, read, or free —
`weight_prepack_size` covers it and the matmul finds it on its own.

```cpp
// --- Model load ---------------------------------------------------------
reorder_params_t rp;
rp.is_prepack             = true;
rp.prepack.algo           = matmul_algo_t::aocl_dlp_blocked;
rp.prepack.wei_dtype      = data_type_t::s8;
rp.prepack.src_dtype      = data_type_t::u8;  // <-- appends colsum[N]
rp.prepack.K              = K;
rp.prepack.N              = N;
rp.prepack.ldb            = N;
rp.prepack.transposed     = false;
rp.prepack.sym_group_size = 0;                // static quant, not sym-quant

size_t prepack_bytes = weight_prepack_size(rp);   // includes the colsum tail
std::vector<uint8_t> prepacked_buf(prepack_bytes);
reorder_direct(original_weight, prepacked_buf.data(), rp);

// --- Inference ----------------------------------------------------------
int32_t src_zp = 17;                          // may vary call to call
float   src_scale = 0.031f, wei_scale = 0.017f;

matmul_params mp;
mp.lowoha_algo  = matmul_algo_t::aocl_dlp_blocked;
mp.mem_format_b = 'r';
mp.dtypes.src   = data_type_t::u8;            // MUST match prepack.src_dtype
mp.dtypes.wei   = data_type_t::s8;
mp.dtypes.dst   = data_type_t::f32;

mp.quant_params.src_scale.buff = &src_scale;
mp.quant_params.src_scale.dt   = data_type_t::f32;
mp.quant_params.src_scale.dims = {1, 1};
mp.quant_params.wei_scale.buff = &wei_scale;
mp.quant_params.wei_scale.dt   = data_type_t::f32;
mp.quant_params.wei_scale.dims = {1, 1};
mp.quant_params.src_zp.buff    = &src_zp;     // per-tensor
mp.quant_params.src_zp.dt      = data_type_t::s32;
mp.quant_params.src_zp.dims    = {1, 1};

matmul_batch_params_t bp;
matmul_direct('r', /*transA*/false, /*transB*/false, M, N, K, /*alpha*/1.0f,
              activation, /*lda*/K, prepacked_buf.data(), /*ldb*/N,
              /*bias*/nullptr, /*beta*/0.0f, dst, /*ldc*/N,
              /*is_weights_const*/true, bp, mp);
```

`src_zp` may change between calls without re-prepacking: the buffer stores the raw
column sums and the matmul applies the current `-src_zp` as a post-op scale factor.

### Caller Responsibilities

The library trusts the caller to keep prepack-time and matmul-time parameters in sync. A mismatch produces **silently wrong results** — no error, no crash, just bad math. Make sure these match between the `reorder_direct` (prepack) call and the subsequent `matmul_direct` call:

- `K`, `N`, `ldb`, `transposed` (`prepack.transposed = true` pairs with `transB = true`)
- `wei_dtype`, `src_dtype`
- `sym_group_size` (for the s8 sym-quant variant)

> ⚠️ **`src_dtype` is the one field whose mismatch is worse than wrong math.** For
> `wei_dtype = s8`, it decides whether the column-sum buffer is appended at pack
> time, while the matmul decides whether to *read* one from `dtypes.src`. Packing
> with a non-`u8` `src_dtype` and then running the matmul with `dtypes.src = u8`
> and a non-zero source zero-point reads off the end of the buffer. See
> *Consuming the Prepacked Buffer at Matmul Time* for the full matrix.

If you mutate any of these on `rp.prepack` between the size query and the prepack call (or between the prepack call and the matmul call), the buffer's layout will not match what the matmul kernel expects.

### Buffer Lifetime

The library never allocates the prepacked buffer; the caller owns it end-to-end. As long as the buffer outlives the last `matmul_direct` call that uses it (with `mem_format_b = 'r'`), there is no freeing or special teardown to remember.

### Custom-Kernel (group_matmul) Prepack

In addition to the AOCL DLP blocked layout, the prepack path can produce the **group_matmul custom-kernel (CK) VNNI** layout. This is selected by `prepack.algo = matmul_algo_t::moe_custom_kernel`. It is a **memory-format change only**: like the AOCL path, it packs each weight into the caller's `dst` buffer (in the VNNI layout the CK microkernel consumes) and does **not** touch any cache. At inference, `group_matmul` consumes that buffer **directly** via `mem_format_b='r'` — no re-pack, no LRU.

| `prepack_params_t` field | Custom-kernel meaning |
|---|---|
| `algo` | `matmul_algo_t::moe_custom_kernel` selects the CK VNNI pack-into-`dst` |
| `wei_dtype` | `bf16` → VDPBF16PS K-pair pack; `s8` → DQ-INT8 VPDPBUSD K-quad pack (+ per-column int32 compensation row) |
| `pack_nr` | CK pack width; `0` = auto (`plan_pack_nr(K, N)`, prefers 32 else 64), else `32`/`64` |
| `interleave_split_halves` | `true` re-interleaves canonical `[gate_cols \| up_cols]` weight into `[g0,u0,g1,u1,...]` for fused silu/gelu (requires even N) |
| `K`, `N`, `ldb`, `transposed` | same meaning as the AOCL path |

Constraints (mirror the CK runtime gate in `prepare_for_call`): `pack_nr ∈ {32,64}` dividing N; the `s8` family requires `K % 4 == 0`; `interleave_split_halves` requires even N.

- The prepacked buffer should be 64-byte aligned (required for `wei_dtype = s8`; recommended for bf16). Use `std::aligned_alloc(64, bytes)`; `weight_prepack_size` already rounds `bytes` up to a multiple of 64.

A single CK prepack op covers one weight; prepacking a **group** of experts is done with `group_reorder` (below).


## Group Reorder

`group_reorder` applies `reorder_direct` to a **batch of independent reorder ops** in one call — the grouped counterpart to the single-op `reorder_direct`. It is a thin sequential wrapper: each op carries its own `reorder_params_t`, so a single group may freely mix standard reorder, dynamic-quant, and weight-prepack (including custom-kernel) ops.

Its primary use case is **MoE model load**: build one custom-kernel prepack `reorder_params_t` per expert and reorder every expert's weight into a caller-owned VNNI buffer ahead of the first decode step, so inference (`group_matmul` with `mem_format_b='r'`) consumes the pre-packed weights directly with no first-iteration pack latency.

### API

```cpp
status_t group_reorder(
  const std::vector<const void *> &src,      // per-op source buffers
  const std::vector<void *>       &dst,      // per-op destination buffers (may be null per the op's mode)
  std::vector<reorder_params_t>   &params    // per-op params (non-const: dynamic-quant writes scale/zp back)
);
```

### Behavior & Contract

| Aspect | Behavior |
|--------|----------|
| Loop | Sequential `reorder_direct(src[i], dst[i], params[i])` for `i` in `[0, params.size())` |
| Why sequential | `reorder_direct` parallelises internally (vector kernels / dynamic-quant dispatchers); an OMP outer loop would nest regions |
| Per-op mode | Selected by each `params[i]` (`is_prepack` / `dynamic_quant` / standard) — modes may be mixed in one group |
| Vector sizes | `src.size() == dst.size() == params.size()`; an empty group or a mismatch returns `status_t::failure` |
| `dst[i]` | May be null only where that op's mode permits (e.g. compute-only dynamic quant). Weight prepack (AOCL or custom-kernel) requires a non-null caller `dst`. |
| Failure | Aborts on the **first** failing op and returns its status; ops already completed keep their finished output (no rollback) |
| Return | `status_t::success` when every op succeeded |

### Example: Grouped Type Conversion

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int group_reorder_convert_example() {
  using namespace zendnnl::lowoha::reorder;

  constexpr int E = 4;  // ops in the group
  std::vector<std::vector<float>>    src_store(E);
  std::vector<std::vector<uint16_t>> dst_store(E);  // bf16 as uint16_t
  std::vector<const void *>          src(E);
  std::vector<void *>                dst(E);
  std::vector<reorder_params_t>      params(E);

  for (int i = 0; i < E; ++i) {
    const int64_t M = 16, N = 64;
    src_store[i].resize(M * N);
    dst_store[i].resize(M * N);
    // Initialize src_store[i] ...
    src[i] = src_store[i].data();
    dst[i] = dst_store[i].data();

    params[i].src_dtype = data_type_t::f32;
    params[i].dst_dtype = data_type_t::bf16;
    params[i].src_shape = {M, N};
    params[i].dst_shape = {M, N};
  }

  status_t status = group_reorder(src, dst, params);
  return (status == status_t::success) ? 0 : -1;
}
```

### Example: 8-Expert MoE Prepack (Custom-Kernel) + Inference

Prepack every expert's weight into a caller-owned VNNI buffer once via `group_reorder`, then run inference through the `group_matmul` API with `mem_format_b='r'`, which consumes the pre-packed buffers directly.

```cpp
#include <cstdlib>   // std::aligned_alloc / std::free
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"

void moe_prepack_then_infer(
    const std::vector<const void *> &expert_weights,  // E weight ptrs (bf16 [K,N])
    int K, int N) {
  using namespace zendnnl::lowoha::reorder;
  const int E = static_cast<int>(expert_weights.size());

  // ---- Model load: one CK-prepack op per expert ----------------------
  std::vector<reorder_params_t> rp(E);
  std::vector<const void *>     src(E);
  std::vector<void *>           prepacked(E, nullptr);  // caller-owned VNNI buffers
  for (int e = 0; e < E; ++e) {
    rp[e].is_prepack            = true;
    rp[e].prepack.algo          = matmul_algo_t::moe_custom_kernel;
    rp[e].prepack.wei_dtype     = data_type_t::bf16;
    rp[e].prepack.src_dtype     = data_type_t::bf16;
    rp[e].prepack.K             = K;
    rp[e].prepack.N             = N;          // multiple of pack_nr (32/64)
    rp[e].prepack.ldb           = N;          // row-major [K, N]
    rp[e].prepack.transposed    = false;
    rp[e].prepack.pack_nr       = 0;          // auto

    // Step 1: query the VNNI buffer size; Step 2: allocate 64B-aligned.
    const size_t bytes = weight_prepack_size(rp[e]);
    prepacked[e]       = std::aligned_alloc(64, bytes);
    src[e]             = expert_weights[e];
  }
  group_reorder(src, prepacked, rp);          // writes VNNI bytes into prepacked[]

  // ---- Inference: group_matmul consumes the prepacked buffers --------
  // With the decode path engaged (ZENDNNL_GRP_MATMUL_ALGO=3 +
  // ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=1, BF16) and per-expert
  // params[e].mem_format_b = 'r' + is_weights_const[e] = true, the CK
  // dispatcher aliases prepacked[e] directly — no re-pack, no cache.
  // group_matmul_direct(layouts, transAs, transBs, Ms, Ns, Ks, alphas,
  //                     srcs, ldas, prepacked, ldbs, biases, betas,
  //                     dsts, ldcs, is_wc, params, /*moe=*/nullptr);

  // prepacked[] must outlive the last inference call, then:
  // for (int e = 0; e < E; ++e) std::free(prepacked[e]);
}
```

> For the full prepack → direct-consume architecture, validation gates, the
> CK-only-or-fail contract, and the BF16/INT8 pack layouts, see the group_matmul
> `docs/group_reorder_article.md`.


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

### Example 24: FP16 Dynamic Quantization — Per-Token (FP16 → S8)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "common/float16.hpp"

int dynamic_quant_f16_per_token_example() {
  using namespace zendnnl::lowoha::reorder;
  using zendnnl::common::float16_t;

  constexpr int64_t M = 4;     // 4 tokens (rows)
  constexpr int64_t N = 256;

  // FP16 stored as uint16_t (bit-compatible with float16_t).
  std::vector<uint16_t> input_f16(M * N);
  std::vector<int8_t>   output_s8(M * N);

  // Initialize input...
  // (e.g., for (int i = 0; i < M*N; ++i)
  //          input_f16[i] = float16_t::f32_to_f16_val(some_f32_value(i));)

  // Per-row scales (one per token).
  std::vector<float> computed_scales(M, 0.0f);

  reorder_params_t params;
  params.src_dtype = data_type_t::f16;     // FP16 source
  params.dst_dtype = data_type_t::s8;      // symmetric quantization
  params.src_shape = {M, N};
  params.dst_shape = {M, N};
  params.dynamic_quant = true;

  // Per-token (per-channel-row): dims = {M, 1}
  params.quant_params.scale.buff = computed_scales.data();
  params.quant_params.scale.dt   = data_type_t::f32;
  params.quant_params.scale.dims = {M, 1};
  // zero_point.buff = nullptr → symmetric mode (zp = 0).

  // The kernel auto-picks between F32-FMA and FP16-FMA backends based on
  // can_use_f16_fma_kernel() (AVX512-FP16 ISA + GCC 12+ + the library was
  // not built with -DZENDNNL_NATIVE_F32_ACCUM=ON). To pin reorder to the
  // F32-FMA path, rebuild the library with that CMake flag.
  status_t status = reorder_direct(input_f16.data(), output_s8.data(), params);
  if (status != status_t::success) {
    return -1;
  }
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 25: FP16 Dynamic Quantization — Asymmetric Per-Token (FP16 → U8)

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "common/float16.hpp"

int dynamic_quant_f16_asymmetric_per_token_example() {
  using namespace zendnnl::lowoha::reorder;

  constexpr int64_t M = 4;
  constexpr int64_t N = 256;

  std::vector<uint16_t> input_f16(M * N);
  std::vector<uint8_t>  output_u8(M * N);

  // Initialize input...

  std::vector<float>   computed_scales(M, 0.0f);
  std::vector<int32_t> computed_zps(M, 0);

  reorder_params_t params;
  params.src_dtype = data_type_t::f16;
  params.dst_dtype = data_type_t::u8;      // asymmetric quantization
  params.src_shape = {M, N};
  params.dst_shape = {M, N};
  params.dynamic_quant = true;

  // Per-token: dims = {M, 1} for both scale and zp.
  params.quant_params.scale.buff = computed_scales.data();
  params.quant_params.scale.dt   = data_type_t::f32;
  params.quant_params.scale.dims = {M, 1};

  params.quant_params.zero_point.buff = computed_zps.data();
  params.quant_params.zero_point.dt   = data_type_t::s32;
  params.quant_params.zero_point.dims = {M, 1};

  status_t status = reorder_direct(input_f16.data(), output_u8.data(), params);
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 26: Dynamic Quantization — Per-Group-Row (BF16 → S8)

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
5. **Scale validation:** Must be finite (for f32, bf16, or f16)
6. **Zero-point validation:** Zero-point values may be any `int32`; during quantization they are clamped to the valid range of the destination type, which can increase saturation and reduce dequantization accuracy if the specified zero-point lies outside that range.
7. **Dims validation:** Must match tensor dimensionality and follow granularity rules
8. **Per-group validation:** M must be divisible by G (row groups) or N must be divisible by G (column groups)
9. **Destination strides:** strided destination is not currently supported; `dst_strides` is ignored and the destination is always written contiguously. (Note: this is not currently enforced by a validation check — a populated `dst_strides` is silently ignored rather than rejected.)

### Dynamic Quantization Validation (when `dynamic_quant = true`)

10. **Source data type:** Must be BF16, FP32, or FP16
11. **Destination data type:** Must be S8 (symmetric) or U8 (asymmetric)
12. **Scale buffer:** Must be non-null (output buffer required)
13. **Scale dims:** Must be specified and valid for the tensor shape
14. **Scale data type:** Must be f32, bf16, or f16
15. **Symmetric mode:** If zero_point.buff is nullptr, destination must be S8
16. **Asymmetric mode:** If zero_point.buff is provided, destination must be U8, and zero_point dims must match scale granularity
17. **Zero-point data type:** Must be s32 (when provided)

### Weight Prepack Validation (when `is_prepack = true`)

18. **Null pointer checks:** `weights` must not be null; `dst` must not be null (both the AOCL and custom-kernel paths write the prepacked layout into the caller's `dst`)
19. **Dimensions:** `prepack.K > 0` and `prepack.N > 0`
20. **Leading dimension:** `prepack.ldb` must be ≥ `prepack.K` (transposed) or ≥ `prepack.N` (non-transposed)
21. **Algo:** `prepack.algo` must be `matmul_algo_t::aocl_dlp_blocked` or `matmul_algo_t::moe_custom_kernel` (any other value is rejected at validation: `reorder_direct` returns `status_t::failure` and `weight_prepack_size` returns `0`)
22. **Buffer size:** the caller is responsible — `dst` must hold at least `weight_prepack_size(params)` bytes (both paths). The library does **not** verify this; an undersized buffer causes silent out-of-bounds writes.
23. **Appended column-sum buffer (AOCL, static-INT8):** for `wei_dtype = s8` + `src_dtype = u8` + `sym_group_size <= 0`, `weight_prepack_size` returns the packed-weight size **plus** `round_up(N * sizeof(int32_t), 64)` for the appended column-sum. This is implicit — there is no flag to enable or disable it, and no separate validation. Sizing the allocation from a raw `aocl_get_reorder_buf_size_*` call instead of `weight_prepack_size` under-allocates by exactly that tail.

> **Cross-check performed by matmul, not reorder:** `matmul_direct` rejects `mem_format_b = 'r'` + `dtypes.src = s8` + non-zero per-tensor `src_zp` with `status_t::unimplemented`, since no prepack configuration appends a column-sum for an `s8` source. The mirror-image error — packing with a non-`u8` `src_dtype` and running the matmul with `dtypes.src = u8` — is **not** detectable by either operator and reads past the end of the buffer. See *Consuming the Prepacked Buffer at Matmul Time*.

### Custom-Kernel Prepack Validation (when `is_prepack = true` and `prepack.algo == matmul_algo_t::moe_custom_kernel`)

24. **Weight dtype:** `prepack.wei_dtype` must be `bf16` or `s8`
25. **Pack width:** `prepack.pack_nr` resolves to `32` or `64` (explicit value must divide N; `0` auto-selects via `plan_pack_nr(K, N)`)
26. **INT8 K alignment:** for `wei_dtype = s8`, `prepack.K % 4 == 0` (VNNI K-quad)
27. **Interleave:** `interleave_split_halves = true` requires even N
28. **Alignment (caller):** `dst` should be 64-byte aligned so the CK microkernel's aligned AVX-512 loads succeed at inference (`weight_prepack_size` already rounds the byte size up to 64)

### Group Reorder Validation (`group_reorder`)

29. **Non-empty group:** `params` must not be empty
30. **Vector sizes:** `src.size() == dst.size() == params.size()`
31. **Per-op:** each op is validated by `reorder_direct` under its own mode; the first per-op failure aborts the group and returns that status


## Implementation Support Matrix

The following table shows which combinations have optimized (AVX512) vs reference implementations:

### Static Reorder: BF16/FP32/FP16 ↔ S8/U8 and FP32 ↔ BF16

| Granularity | Source Contiguous | Source Strided (last_stride=1) | Source Strided (other) |
|-------------|-------------------|--------------------------------|------------------------|
| Per-tensor | ✅ Optimal | ✅ Optimal | ⚙️ Reference |
| Per-channel | ⚙️ Reference | ⚙️ Reference | ⚙️ Reference |
| Per-group | ⚙️ Reference | ⚙️ Reference | ⚙️ Reference |

For per-tensor FP16 source (or FP16 destination on dequant), the optimal AVX-512 path has two backends:
- **F32-FMA** (default): F16C convert on load/store, math in `__m512`. Bit-exact with the scalar reference. Requires AVX-512F + F16C (the two CPUID bits are architecturally independent, but every shipping AVX-512F-capable CPU also has F16C, so the F32-FMA path is available wherever AVX-512F is available).
- **FP16-FMA** (`__m512h`-native): math fully in FP16. Requires AVX512-FP16 ISA (Granite Rapids / Sapphire Rapids / Turin). ~2× throughput; tolerates ±1 LSB drift vs the scalar reference in the common regime. The FP16-FMA quant kernels narrow the intermediate `round(val/scale)` to `int16` (`VCVTPH2W`) before applying the `int32` zero point; the FP16-FMA dequant kernels widen the `int32` `(input - zero_point)` difference to FP16 (`VCVTDQ2PH`) and narrow the user scale to `_Float16` before the FP16 scale multiply. To preserve the public `int32 zero_point` and `finite scale` contracts on both directions, the vector loop is gated on the following safe ranges; any failure routes the whole call through the scalar tail (which computes the chain in `f32` with the offset in `int32`, matching the F32-FMA / scalar reference bit-for-bit):
>
> | Direction | Guards |
> |---|---|
> | Quant (`f16 → s8/u8`) | `1/scale` fits in FP16 (`scale > ~3.05e-5`) **and** `|zero_point| ≤ 32512` |
> | Dequant (`s8/u8 → f16`) | `|zero_point| ≤ 65000` **and** (`scale = 0` or `FP16_MIN_NORMAL ≤ |scale| ≤ 65504`) |
>
> The dynamic-quant path almost never trips the quant-side zp guard in practice — `zp = round(-min/scale)` only exceeds 32512 when the source row is clustered far from zero in a very narrow band (e.g. an `f16` row with `min=-65000, max=-64500`). The dequant-side guards are only relevant when a user explicitly supplies an `int32 zero_point` or an `f32 scale` outside FP16's finite normal range to the static dequant API; both are permitted by the API contract.

The backend is auto-selected via the helper `can_use_f16_fma_kernel()` (defined in `lowoha_reorder_common.hpp`), which wraps `zendnnl_platform_info().get_avx512_f16_status()`. Reorder exposes **no runtime env var** for this choice — selection is purely build-time + runtime ISA detection.

Building the library with the CMake option `ZENDNNL_NATIVE_F32_ACCUM=ON` is the master kill-switch: it disables the FP16-FMA kernels and forces reorder onto its F32-accumulating AVX-512 path for numerical-reproducibility studies.

### Dynamic Quantization: Fused fast paths (no compute-then-quantize round-trip)

| Granularity | BF16 / FP32 source | FP16 source |
|-------------|--------------------|-------------|
| Per-token (per-channel-row), 2D contiguous | ✅ Fused vector (F32 math) | ✅ Fused vector — F32-FMA or FP16-FMA (auto-selected) |
| Per-group-col, 2D contiguous | ✅ Fused vector (F32 math) | ✅ Fused vector — F32-FMA or FP16-FMA (auto-selected) |
| Per-token, 2D contiguous, `ALGO=2` (unfused) | ✅ Unfused 2-pass vector (F32 math) | ✅ Unfused 2-pass vector — F32-FMA or FP16-FMA (auto-selected) |
| Per-token, `ALGO=3` (scalar fused) | ⚙️ Scalar | ⚙️ Scalar |

For granularities not in the fast-path table (per-tensor, per-channel-col, per-group-row, 1D / 3D, strided), the dynamic-quant path falls through to:
1. `compute_dynamic_quant_params(src, params)` — scans the source data and fills the scale/zp buffers.
2. Standard reorder using the freshly computed scale/zp (follows the static reorder support matrix above).

### Dynamic Quantization: Parameter Computation (fall-through path)

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
- **Weight Prepack (static-INT8, `u8` source):** The appended column-sum buffer moves the `sum_k wei[k,n]` reduction from every matmul call to the one-time prepack, removing an O(K·N) pass per call plus the LRU lookup that used to cache its result. The prepack itself costs one extra O(K·N) reduction (OpenMP-parallel over N) and `round_up(N * 4, 64)` bytes of buffer. The win grows with call count and shrinks as M grows, so it matters most for decode-shaped (small-M) inference.
- **Dynamic Quantization:** Adds a min/max scan pass over the source data before quantization. For per-tensor, this scans all elements sequentially. For per-channel and per-group, the scan is parallelized with OpenMP. The compute-only mode (`dst = nullptr`) can be used to separate the parameter computation from the quantization step.
