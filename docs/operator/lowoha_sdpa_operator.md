
(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA SDPA Operator

## Overview

The **LowOHA SDPA (Scaled Dot-Product Attention) Operator** is a high-performance, framework-agnostic implementation of the flash attention algorithm for CPU inference. It provides a direct C API that accepts raw pointers and stride metadata, eliminating any dependency on PyTorch ATen or other framework tensors.

The operator implements the standard multi-head attention computation:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q \cdot K^T}{\sqrt{d_k}} + M\right) \cdot V
$$

Where:
- *Q* ∈ ℝ<sup>B×H×S<sub>q</sub>×D</sup>: Query tensor
- *K* ∈ ℝ<sup>B×H×S<sub>kv</sub>×D</sup>: Key tensor
- *V* ∈ ℝ<sup>B×H×S<sub>kv</sub>×D</sup>: Value tensor
- *M*: Optional attention mask (additive, broadcastable 2-D or 4-D)
- *d<sub>k</sub>*: Head dimension (used for default scaling)

For self-attention S<sub>q</sub> == S<sub>kv</sub>. For cross-attention (e.g. encoder-decoder models like T5/MT5, or attention pooling in SigLIP) they may differ.

Key design goals:
- Zero framework overhead — operates on raw data pointers with explicit strides
- Runtime SIMD dispatch — AVX-512 when available, scalar fallback otherwise
- Tiled flash attention — O(S) memory instead of O(S²) for the attention matrix
- OpenMP parallelization across batch × heads × query tiles
- Thread-local scratch buffer reuse across calls


## Core API: `sdpa_direct`

The primary interface for LowOHA SDPA:

```cpp
status_t sdpa_direct(
  const void *query,      // Query tensor data pointer  [B, H, S_q,  D]
  const void *key,        // Key tensor data pointer    [B, H, S_kv, D]
  const void *value,      // Value tensor data pointer  [B, H, S_kv, D]
  const void *attn_mask,  // Optional attention mask (can be nullptr)
  void *output,           // Output tensor data pointer [B, H, S_q,  D]
  sdpa_params &params     // SDPA parameters (dimensions, strides, dtypes, etc.)
);
```

**Returns:** `status_t::success` on success, `status_t::failure` on error.

**Namespace:** `zendnnl::lowoha::sdpa`


## Parameters Structure

### `sdpa_params`

The unified parameter structure for all SDPA backends:

```cpp
struct sdpa_params {
  // Tensor dimensions
  int64_t batch;
  int64_t num_heads;
  int64_t seq_len;         // Q / Output sequence length (S_q)
  int64_t kv_seq_len;      // K / V sequence length (S_kv); 0 = same as seq_len
  int64_t head_dim;

  // Per-tensor BHSD strides
  int64_t q_stride_b, q_stride_h, q_stride_s, q_stride_d;
  int64_t k_stride_b, k_stride_h, k_stride_s, k_stride_d;
  int64_t v_stride_b, v_stride_h, v_stride_s, v_stride_d;
  int64_t o_stride_b, o_stride_h, o_stride_s, o_stride_d;

  // Mask parameters (raw 4-D sizes + strides)
  int mask_ndims;
  int64_t mask_sizes[4];
  int64_t mask_strides[4];

  // Data types
  data_type_t qkv_dt;     // Q/K/V data type (f32 or bf16)
  data_type_t out_dt;      // Output data type
  data_type_t mask_dt;     // Mask data type (f32 or bf16)

  // Computation parameters
  double scale;            // Attention scale (0 = auto: 1/sqrt(head_dim))
  bool is_causal;          // Enable causal (upper-triangular) masking
  double dropout_p;        // Dropout probability (must be 0)

  int32_t num_threads;     // Number of OpenMP threads (0 = auto)
};
```

#### `seq_len` vs `kv_seq_len`

| Field | Applies to | Description |
|-------|-----------|-------------|
| `seq_len` | Q, Output | Query sequence length (S<sub>q</sub>) |
| `kv_seq_len` | K, V | Key/Value sequence length (S<sub>kv</sub>). Set to `0` to use `seq_len` (self-attention). |

For **self-attention** (e.g. ViT encoder, GPT), set `kv_seq_len = 0` or `kv_seq_len = seq_len`.

For **cross-attention** (e.g. T5/MT5 decoder attending to encoder, SigLIP attention pooling), set `kv_seq_len` to the actual K/V sequence length.

### Stride Requirements

The flash backend uses per-tensor BHSD strides to support non-contiguous memory layouts. The following constraints must be satisfied:

| Stride | Requirement | Reason |
|--------|-------------|--------|
| `q_stride_d`, `k_stride_d`, `v_stride_d` | Must be `1` | GEMM requires contiguous head dimension |
| `q_stride_s`, `k_stride_s`, `v_stride_s` | Must be `> 0` | Sequence stride is the GEMM leading dimension |
| `o_stride_s`, `o_stride_d` | Must be `> 0` | Output must be writable |
| `o_stride_b` | Must be `> 0` when `batch > 1` | Parallel writes must not alias |
| `o_stride_h` | Must be `> 0` when `num_heads > 1` | Parallel writes must not alias |

### Supported Data Types

| Q/K/V Type | Mask Type | Output Type | Notes |
|------------|-----------|-------------|-------|
| FP32 | FP32 | FP32 | Standard floating-point |
| FP32 | None | FP32 | No attention mask |
| BF16 | FP32 | BF16 | Mixed-precision BFloat16 |
| BF16 | BF16 | BF16 | Full BF16 pipeline |
| BF16 | None | BF16 | No attention mask |

> **Note:** All internal accumulation is performed in FP32 regardless of the input data type.


### Attention Mask

The attention mask is an optional additive mask applied before the softmax. It supports two layouts:

| `mask_ndims` | Shape | Broadcasting |
|--------------|-------|-------------|
| `2` | `[S_q, S_kv]` | Broadcast across batch and heads |
| `4` | `[B, H, S_q, S_kv]` | Per-batch, per-head mask (dims of size 1 are broadcast) |

The last dimension (`S_kv`) must have stride 1 (contiguous). When `is_causal = true`, future positions are filled with `-inf` regardless of the mask.


## Execution Flow

```
sdpa_direct()
  │  Profiling / logging wrapper
  │
  ▼
sdpa_flash_cpu_standalone()
  │  1. Validate inputs (null checks, dimensions, strides, dtypes)
  │  2. Build lightweight tensor views from sdpa_params
  │  3. Build mask view (if mask provided)
  │
  ▼
sdpa_flash_cpu_run_internal()
  │  Runtime SIMD dispatch:
  │    if (AVX-512 available) → SimdOps<avx512_tag>  (16-lane __m512)
  │    else                   → SimdOps<scalar_tag>   (1-lane scalar)
  │
  ▼
flash_attention_kernel_sa_dispatch<SimdTag>()
  │  Select tile sizes based on Q sequence length (seq_len):
  │    seq_len >= 768  → q_split=256, kv_split=512
  │    seq_len >= 192  → q_split=64,  kv_split=512
  │    seq_len <  192  → q_split=32,  kv_split=512
  │    batch > 4       → q_split=512  (override)
  │
  ▼
cpu_flash_attention_sa<SimdTag, scalar_t, mask_t, q_split, kv_split>()
  │  OpenMP parallel loop over batch × heads × q_tiles
  │
  │  For each (batch_i, head_j, q_tile_k):
  │    ┌─ for each kv_tile:
  │    │    1. GEMM: Q_tile × K_tile^T          (via AOCL BLAS)
  │    │    2. Causal masking (fill future with -inf)
  │    │    3. Scale + mask fusion               (SIMD fused FMA)
  │    │    4. Row-wise max + exp + sum           (SIMD fused reductions)
  │    │    5. Rescale running output accumulator (SIMD)
  │    │    6. GEMM: softmax_tile × V_tile       (via AOCL BLAS)
  │    └─
  │    7. Write final output = accumulator / sum (SIMD scaled store)
```


## Flash Attention Algorithm

The kernel implements the online softmax flash attention algorithm, which avoids materializing the full S×S attention matrix:

1. **Tiling**: Q is split into tiles of `q_split_size` rows. K and V are split into tiles of `kv_split_size` rows. Only one Q×K tile is materialized at a time.

2. **Online Softmax**: For each Q tile, the kernel iterates over KV tiles and maintains running statistics (row-wise max and sum) to compute the softmax incrementally. When a new KV tile produces a larger max, the previously accumulated output is rescaled.

3. **Memory Efficiency**: Scratch memory per thread is `O(q_split × kv_split + q_split × head_dim)` instead of `O(S × S)` for the full attention matrix. Scratch buffers are thread-local and reused across calls.

4. **Parallelization**: The outer loop over `batch × heads × q_tiles` work items is parallelized with `#pragma omp parallel for schedule(static)`.


## SIMD Dispatch

The kernel uses a tag-based template dispatch to select the SIMD implementation at runtime:

```cpp
if (zendnnl::common::zendnnl_platform_info().get_avx512f_status()) {
    run(simd::avx512_tag{});   // 16-lane AVX-512
} else {
    run(simd::scalar_tag{});   // 1-lane scalar fallback
}
```

The `SimdTag` template parameter propagates through all helper functions, so the compiler generates separate instantiations for each ISA. The AVX-512 methods use `__attribute__((target("avx512f,avx512bw,avx512vl,fma")))`, allowing the binary to be compiled without global `-mavx512f` flags.

### `SimdOps<Tag>` Specializations

| Tag | `VecF32` Type | Lanes | ISA Requirement |
|-----|---------------|-------|-----------------|
| `avx512_tag` | `__m512` | 16 | AVX-512F + BW + VL + FMA |
| `scalar_tag` | `struct { float v; }` | 1 | None (portable) |

### SIMD-Accelerated Operations

| Operation | Function | Description |
|-----------|----------|-------------|
| Scale + Mask | `scale_attn_mask_fusion` | Fused `out = a * scale + mask` (FMA) |
| Scale + Max | `mul_reduce_max_fusion` | Fused multiply and row-wise max |
| Exp + Sum | `exp_reduce_sum_fusion` | Fused `exp(x - max)` and reduction sum |
| Row Max | `row_max` | SIMD row-wise maximum |
| Row Scale | `scale_dst_row` | SIMD element-wise multiply |
| Output Write | `write_scaled_output_row` | SIMD scaled store (FP32 or BF16 conversion) |
| Fast Exp | `vec_exp_u20` / `vec_fexp_u20` | ~20 ULP polynomial exp approximation |


## Usage Examples

### Example 1: Basic FP32 SDPA

```cpp
#include "lowoha_operators/sdpa/lowoha_sdpa.hpp"

int lowoha_sdpa_fp32_example() {
  using namespace zendnnl::lowoha::sdpa;

  int64_t B = 1, H = 12, S = 384, D = 64;

  // Allocate contiguous BHSD tensors
  std::vector<float> query(B * H * S * D, 0.1f);
  std::vector<float> key(B * H * S * D, 0.1f);
  std::vector<float> value(B * H * S * D, 0.1f);
  std::vector<float> output(B * H * S * D, 0.0f);

  // Configure parameters
  sdpa_params params;
  params.batch      = B;
  params.num_heads  = H;
  params.seq_len    = S;
  params.kv_seq_len = S;  // self-attention: Q and K/V have the same length
  params.head_dim   = D;

  // Contiguous BHSD strides
  params.q_stride_b = H * S * D;
  params.q_stride_h = S * D;
  params.q_stride_s = D;
  params.q_stride_d = 1;

  params.k_stride_b = H * S * D;
  params.k_stride_h = S * D;
  params.k_stride_s = D;
  params.k_stride_d = 1;

  params.v_stride_b = H * S * D;
  params.v_stride_h = S * D;
  params.v_stride_s = D;
  params.v_stride_d = 1;

  params.o_stride_b = H * S * D;
  params.o_stride_h = S * D;
  params.o_stride_s = D;
  params.o_stride_d = 1;

  params.qkv_dt = data_type_t::f32;
  params.out_dt = data_type_t::f32;
  params.scale  = 1.0 / std::sqrt(static_cast<double>(D));
  params.is_causal = false;
  params.dropout_p = 0.0;

  // Execute SDPA (no mask)
  status_t status = sdpa_direct(
    query.data(), key.data(), value.data(),
    nullptr,  // no attention mask
    output.data(),
    params
  );

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 2: BF16 SDPA with Causal Masking

```cpp
int lowoha_sdpa_bf16_causal_example() {
  using namespace zendnnl::lowoha::sdpa;

  int64_t B = 4, H = 16, S = 1024, D = 64;

  // BF16 stored as uint16_t
  std::vector<uint16_t> query(B * H * S * D);
  std::vector<uint16_t> key(B * H * S * D);
  std::vector<uint16_t> value(B * H * S * D);
  std::vector<uint16_t> output(B * H * S * D, 0);

  sdpa_params params;
  params.batch      = B;
  params.num_heads  = H;
  params.seq_len    = S;
  params.kv_seq_len = S;  // self-attention
  params.head_dim   = D;

  // Contiguous BHSD strides
  params.q_stride_b = H * S * D;
  params.q_stride_h = S * D;
  params.q_stride_s = D;
  params.q_stride_d = 1;

  params.k_stride_b = H * S * D;
  params.k_stride_h = S * D;
  params.k_stride_s = D;
  params.k_stride_d = 1;

  params.v_stride_b = H * S * D;
  params.v_stride_h = S * D;
  params.v_stride_s = D;
  params.v_stride_d = 1;

  params.o_stride_b = H * S * D;
  params.o_stride_h = S * D;
  params.o_stride_s = D;
  params.o_stride_d = 1;

  params.qkv_dt   = data_type_t::bf16;
  params.out_dt   = data_type_t::bf16;
  params.scale    = 1.0 / std::sqrt(static_cast<double>(D));
  params.is_causal = true;
  params.dropout_p = 0.0;

  status_t status = sdpa_direct(
    query.data(), key.data(), value.data(),
    nullptr,  // causal masking is applied internally
    output.data(),
    params
  );

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 3: FP32 SDPA with 4-D Attention Mask

```cpp
int lowoha_sdpa_with_mask_example() {
  using namespace zendnnl::lowoha::sdpa;

  int64_t B = 2, H = 8, S = 512, D = 64;

  std::vector<float> query(B * H * S * D, 0.1f);
  std::vector<float> key(B * H * S * D, 0.1f);
  std::vector<float> value(B * H * S * D, 0.1f);
  std::vector<float> output(B * H * S * D, 0.0f);

  // 4-D attention mask [B, H, S_q, S_kv]
  std::vector<float> mask(B * H * S * S, 0.0f);

  sdpa_params params;
  params.batch      = B;
  params.num_heads  = H;
  params.seq_len    = S;
  params.kv_seq_len = S;  // self-attention
  params.head_dim   = D;

  // Q/K/V strides (contiguous BHSD)
  params.q_stride_b = H * S * D;  params.q_stride_h = S * D;
  params.q_stride_s = D;          params.q_stride_d = 1;
  params.k_stride_b = H * S * D;  params.k_stride_h = S * D;
  params.k_stride_s = D;          params.k_stride_d = 1;
  params.v_stride_b = H * S * D;  params.v_stride_h = S * D;
  params.v_stride_s = D;          params.v_stride_d = 1;
  params.o_stride_b = H * S * D;  params.o_stride_h = S * D;
  params.o_stride_s = D;          params.o_stride_d = 1;

  params.qkv_dt   = data_type_t::f32;
  params.out_dt   = data_type_t::f32;
  params.mask_dt  = data_type_t::f32;
  params.scale    = 1.0 / std::sqrt(static_cast<double>(D));
  params.is_causal = false;
  params.dropout_p = 0.0;

  // Mask metadata: 4-D [B, H, S_q, S_kv]
  params.mask_ndims = 4;
  params.mask_sizes[0]   = B;      params.mask_strides[0] = H * S * S;
  params.mask_sizes[1]   = H;      params.mask_strides[1] = S * S;
  params.mask_sizes[2]   = S;      params.mask_strides[2] = S;
  params.mask_sizes[3]   = S;      params.mask_strides[3] = 1;

  status_t status = sdpa_direct(
    query.data(), key.data(), value.data(),
    mask.data(),
    output.data(),
    params
  );

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 4: SDPA with Broadcast 2-D Mask

```cpp
int lowoha_sdpa_broadcast_mask_example() {
  using namespace zendnnl::lowoha::sdpa;

  int64_t B = 8, H = 12, S = 384, D = 64;

  std::vector<float> query(B * H * S * D, 0.1f);
  std::vector<float> key(B * H * S * D, 0.1f);
  std::vector<float> value(B * H * S * D, 0.1f);
  std::vector<float> output(B * H * S * D, 0.0f);

  // 2-D mask [S_q, S_kv] — broadcast across all batches and heads
  std::vector<float> mask(S * S, 0.0f);

  sdpa_params params;
  params.batch      = B;
  params.num_heads  = H;
  params.seq_len    = S;
  params.kv_seq_len = S;  // self-attention
  params.head_dim   = D;

  // Contiguous BHSD strides (same pattern as above)
  params.q_stride_b = H * S * D;  params.q_stride_h = S * D;
  params.q_stride_s = D;          params.q_stride_d = 1;
  params.k_stride_b = H * S * D;  params.k_stride_h = S * D;
  params.k_stride_s = D;          params.k_stride_d = 1;
  params.v_stride_b = H * S * D;  params.v_stride_h = S * D;
  params.v_stride_s = D;          params.v_stride_d = 1;
  params.o_stride_b = H * S * D;  params.o_stride_h = S * D;
  params.o_stride_s = D;          params.o_stride_d = 1;

  params.qkv_dt   = data_type_t::f32;
  params.out_dt   = data_type_t::f32;
  params.mask_dt  = data_type_t::f32;
  params.scale    = 0.0;  // 0 = auto (1/sqrt(head_dim))
  params.is_causal = false;
  params.dropout_p = 0.0;

  // Mask metadata: 2-D [S_q, S_kv]
  params.mask_ndims = 2;
  params.mask_sizes[0]   = S;   params.mask_strides[0] = S;
  params.mask_sizes[1]   = S;   params.mask_strides[1] = 1;

  status_t status = sdpa_direct(
    query.data(), key.data(), value.data(),
    mask.data(),
    output.data(),
    params
  );

  return (status == status_t::success) ? 0 : -1;
}
```


### Example 5: Cross-Attention (Encoder-Decoder)

Cross-attention is used in encoder-decoder models (T5, MT5) where the decoder query attends to encoder key/value with a different sequence length.

```cpp
int lowoha_sdpa_cross_attention_example() {
  using namespace zendnnl::lowoha::sdpa;

  int64_t B = 2, H = 12, D = 64;
  int64_t S_q  = 1;     // decoder query length (e.g. current token)
  int64_t S_kv = 512;   // encoder key/value length

  std::vector<float> query(B * H * S_q * D, 0.1f);
  std::vector<float> key(B * H * S_kv * D, 0.1f);
  std::vector<float> value(B * H * S_kv * D, 0.1f);
  std::vector<float> output(B * H * S_q * D, 0.0f);

  // 4-D cross-attention mask [B, 1, S_q, S_kv]
  std::vector<float> mask(B * 1 * S_q * S_kv, 0.0f);

  sdpa_params params;
  params.batch      = B;
  params.num_heads  = H;
  params.seq_len    = S_q;    // query sequence length
  params.kv_seq_len = S_kv;   // key/value sequence length (different!)
  params.head_dim   = D;

  // Q strides [B, H, S_q, D]
  params.q_stride_b = H * S_q * D;   params.q_stride_h = S_q * D;
  params.q_stride_s = D;             params.q_stride_d = 1;

  // K strides [B, H, S_kv, D]
  params.k_stride_b = H * S_kv * D;  params.k_stride_h = S_kv * D;
  params.k_stride_s = D;             params.k_stride_d = 1;

  // V strides [B, H, S_kv, D]
  params.v_stride_b = H * S_kv * D;  params.v_stride_h = S_kv * D;
  params.v_stride_s = D;             params.v_stride_d = 1;

  // Output strides [B, H, S_q, D]
  params.o_stride_b = H * S_q * D;   params.o_stride_h = S_q * D;
  params.o_stride_s = D;             params.o_stride_d = 1;

  params.qkv_dt   = data_type_t::f32;
  params.out_dt   = data_type_t::f32;
  params.mask_dt  = data_type_t::f32;
  params.scale    = 1.0 / std::sqrt(static_cast<double>(D));
  params.is_causal = false;
  params.dropout_p = 0.0;

  // Mask metadata: 4-D [B, 1, S_q, S_kv] — broadcast across heads
  params.mask_ndims = 4;
  params.mask_sizes[0]   = B;      params.mask_strides[0] = S_q * S_kv;
  params.mask_sizes[1]   = 1;      params.mask_strides[1] = 0;
  params.mask_sizes[2]   = S_q;    params.mask_strides[2] = S_kv;
  params.mask_sizes[3]   = S_kv;   params.mask_strides[3] = 1;

  status_t status = sdpa_direct(
    query.data(), key.data(), value.data(),
    mask.data(),
    output.data(),
    params
  );

  return (status == status_t::success) ? 0 : -1;
}
```


## Tiling Heuristics

The flash kernel selects tile sizes based on the Q sequence length (`seq_len`)
to balance parallelism and cache efficiency. `kv_split_size` is clamped at
runtime to `min(512, kv_seq_len)`.

| Condition | `q_split_size` | `kv_split_size` | Rationale |
|-----------|---------------|-----------------|-----------|
| `seq_len >= 768` | 256 | 512 | Large tiles maximize GEMM efficiency |
| `seq_len >= 192` | 64 | 512 | Moderate tiles for medium sequences |
| `seq_len < 192` | 32 | 512 | Small tiles preserve parallelism for short sequences |
| `batch > 4` (override) | 512 | 512 | Larger Q tiles when batch parallelism is sufficient |


## Scratch Memory Management

Each thread uses a private scratch buffer for intermediate results:

| Buffer | Size | Purpose |
|--------|------|---------|
| `qk_data` | `q_split × kv_split` | Q×K^T tile (FP32 accumulation) |
| `qk_max_data` | `q_split` | Running row-wise max for online softmax |
| `qk_sum_data` | `q_split` | Running row-wise sum for online softmax |
| `dst_data` | `q_split × head_dim` | Running output accumulator (FP32) |
| `qk_reduced_data` | `q_split × kv_split` | BF16 softmax weights (BF16 path only) |

Scratch buffers are managed via a `thread_local` allocator (`flash_scratch_acquire`) that grows-only and reuses memory across calls. Call `sdpa_flash_cpu_free_scratch()` to eagerly release.

## Limitations

- **Dropout**: Only `dropout_p = 0.0` is supported. Non-zero dropout returns `status_t::failure`.
- **Head dimension contiguity**: `stride_d` must be 1 for Q, K, and V (required by the underlying GEMM).
- **Mask contiguity**: The last mask dimension (`S_kv`) must have stride 1.
