
(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA SDPA Operator

## Overview

The **SDPA (Scaled Dot-Product Attention) API** (`sdpa_direct`) is a high-performance, framework-agnostic implementation of the flash attention algorithm for CPU inference. It accepts raw pointers and stride metadata, eliminating any dependency on PyTorch ATen or other framework tensors.

It computes the standard multi-head attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q \cdot K^T}{\sqrt{d_k}} + M\right) \cdot V
$$

Where:
- *Q* ∈ ℝ<sup>B×H×S<sub>q</sub>×D</sup>: Query tensor
- *K* ∈ ℝ<sup>B×H<sub>kv</sub>×S<sub>kv</sub>×D</sup>: Key tensor
- *V* ∈ ℝ<sup>B×H<sub>kv</sub>×S<sub>kv</sub>×D</sup>: Value tensor
- *M*: Optional attention mask (additive, broadcastable 2-D or 4-D)
- *d<sub>k</sub>*: Head dimension (used for default scaling)

For self-attention S<sub>q</sub> == S<sub>kv</sub>. For cross-attention (e.g. encoder-decoder models like T5/MT5, or attention pooling in SigLIP) they may differ.

The `sdpa_params` structure is a *unified* parameter block designed for both a flash-style backend and a future BMM-based backend. Today only the **flash backend** is active (the BMM path is reserved and disabled in `lowoha_sdpa.cpp`); the flash backend selects its tile sizes and SIMD specialization automatically from the workload, with no manual algorithm knob.

Include `lowoha_operators/sdpa/lowoha_sdpa.hpp` (namespace `zendnnl::lowoha::sdpa`).

### Key benefits

- Zero framework overhead — operates on raw data pointers with explicit strides
- Runtime SIMD dispatch — AVX-512 when available, scalar fallback otherwise
- Tiled flash attention — O(S) memory instead of O(S²) for the attention matrix
- OpenMP parallelization across batch × heads × query tiles
- Thread-local scratch buffer reuse across calls
- Self- and cross-attention support, with optional additive 2-D / 4-D mask and causal masking
- FP32 / BF16 / FP16 inputs with FP32 internal precision for numerical stability
- Optional BF16 dynamic-INT8 compute for Q×K^T and/or probability×V using
  ZenDNN reorder and matmul APIs

## API signature

```cpp
status_t sdpa_direct(
  const void *query,      // Query tensor data pointer  [B, H, S_q,  D]
  const void *key,        // Key tensor data pointer    [B, H_kv, S_kv, D]
  const void *value,      // Value tensor data pointer  [B, H_kv, S_kv, D]
  const void *attn_mask,  // Optional attention mask (can be nullptr)
  void *output,           // Output tensor data pointer [B, H, S_q,  D]
  sdpa_params &params     // SDPA parameters (dimensions, strides, dtypes, etc.)
);
```

### Parameters

`sdpa_params` is the unified parameter structure for all SDPA backends:

```cpp
struct sdpa_params {
  // Tensor dimensions
  int64_t batch;
  int64_t num_heads;
  int64_t kv_num_heads;   // K/V heads for GQA/MQA; 0 = num_heads
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
  data_type_t qkv_dt;      // Q/K/V data type (f32, bf16, or f16)
  data_type_t out_dt;      // Output data type (must equal qkv_dt, or none)
  data_type_t mask_dt;     // Mask data type (f32 for any supported qkv_dt; bf16 only with bf16 Q/K/V; f16 only with f16 Q/K/V)

  // Computation parameters
  double scale;            // Attention scale (0 = auto: 1/sqrt(head_dim))
  bool is_causal;          // Enable causal (upper-triangular) masking
  bool sliding_window;          // Enable sliding-window attention (default false)
  int64_t sliding_window_size;  // Window width W; required when sliding_window
  double dropout_p;        // Dropout probability (must be 0)
  bool is_qk_quant;        // BF16 Q/K dynamic-INT8 Q×K^T compute (default false)
  bool is_pv_quant;        // BF16 V / u8 probability INT8 P×V compute (default false)

  int32_t num_threads;     // Number of OpenMP threads (0 = auto)
};
```

| Field | Type | Description |
|-------|------|-------------|
| `batch` | `int64_t` | Batch size (B) |
| `num_heads` | `int64_t` | Number of query/output attention heads (H) |
| `kv_num_heads` | `int64_t` | Number of key/value heads (H<sub>kv</sub>) for GQA/MQA; `0` = same as `num_heads` |
| `seq_len` | `int64_t` | Query / output sequence length (S<sub>q</sub>) |
| `kv_seq_len` | `int64_t` | Key / value sequence length (S<sub>kv</sub>); `0` = same as `seq_len` |
| `head_dim` | `int64_t` | Per-head feature dimension (D) |
| `q/k/v/o_stride_*` | `int64_t` | Per-tensor BHSD strides (see [Stride requirements](#stride-requirements)) |
| `mask_ndims` | `int` | Mask rank: `0` (none), `2`, or `4` |
| `mask_sizes` / `mask_strides` | `int64_t[4]` | Mask shape and strides |
| `qkv_dt` | `data_type_t` | Q/K/V data type (`f32`, `bf16`, or `f16`) |
| `out_dt` | `data_type_t` | Output data type (must equal `qkv_dt`, or `none`) |
| `mask_dt` | `data_type_t` | Mask data type (see [Supported data types](#supported-data-types)) |
| `scale` | `double` | Attention scale; `0` = auto (`1/sqrt(head_dim)`) |
| `is_causal` | `bool` | Enable causal (upper-triangular) masking |
| `sliding_window` | `bool` | Enable sliding-window attention; defaults to `false` |
| `sliding_window_size` | `int64_t` | Window width W, must be `> 0` when `sliding_window` is `true` |
| `dropout_p` | `double` | Dropout probability (must be `0`) |
| `is_qk_quant` | `bool` | Quantize BF16 Q/K per token and run Q×K^T in INT8; requires AOCL-DLP and AVX512-VNNI |
| `is_pv_quant` | `bool` | Quantize BF16 V per channel and the softmax tile to u8, and run probability×V in INT8; requires AOCL-DLP and AVX512-VNNI |
| `num_threads` | `int32_t` | OpenMP thread count; `0` = auto |

#### `seq_len` vs `kv_seq_len`

| Field | Applies to | Description |
|-------|-----------|-------------|
| `seq_len` | Q, Output | Query sequence length (S<sub>q</sub>) |
| `kv_seq_len` | K, V | Key/Value sequence length (S<sub>kv</sub>). Set to `0` to use `seq_len` (self-attention). |

For **self-attention** (e.g. ViT encoder, GPT), set `kv_seq_len = 0` or `kv_seq_len = seq_len`.

For **cross-attention** (e.g. T5/MT5 decoder attending to encoder, SigLIP attention pooling), set `kv_seq_len` to the actual K/V sequence length.

#### `num_heads` vs `kv_num_heads`

For standard MHA, set `kv_num_heads = 0` or `kv_num_heads = num_heads`.
For MQA/GQA, set `kv_num_heads` to the compact K/V head count. The flash
backend requires `num_heads % kv_num_heads == 0` and maps each query head `h`
to K/V head `h / (num_heads / kv_num_heads)`, so K/V do not need to be
expanded or copied before calling `sdpa_direct`.

#### Stride requirements

The flash backend uses per-tensor BHSD strides to support non-contiguous memory layouts. The following constraints must be satisfied:

| Stride | Requirement | Reason |
|--------|-------------|--------|
| `q_stride_d`, `k_stride_d`, `v_stride_d` | Must be `1` | GEMM requires contiguous head dimension |
| `q_stride_s`, `k_stride_s`, `v_stride_s` | Must be `> 0` | Sequence stride is the GEMM leading dimension |
| `o_stride_s`, `o_stride_d` | Must be `> 0` | Output must be writable |
| `o_stride_b` | Must be `> 0` when `batch > 1` | Parallel writes must not alias |
| `o_stride_h` | Must be `> 0` when `num_heads > 1` | Parallel writes must not alias |

#### Supported data types

| Q/K/V Type | Mask Type | Output Type | Notes |
|------------|-----------|-------------|-------|
| FP32 | FP32 | FP32 | Standard floating-point |
| FP32 | None | FP32 | No attention mask |
| BF16 | FP32 | BF16 | Mixed-precision BFloat16 |
| BF16 | BF16 | BF16 | Full BF16 pipeline |
| BF16 | None | BF16 | No attention mask |
| FP16 | FP32 | FP16 | Mixed-precision IEEE half (requires AVX512-FP16, see below) |
| FP16 | FP16 | FP16 | Full FP16 pipeline (requires AVX512-FP16, see below) |
| FP16 | None | FP16 | No attention mask (requires AVX512-FP16, see below) |

> **Note:** Internal precision is FP32 across all paths. Both the Q×K^T and softmax×V matmuls write FP32 outputs into the per-thread accumulators (`aocl_gemm_bf16bf16f32of32` for BF16 inputs, `aocl_gemm_f16f16f32of32` for FP16 inputs, `aocl_gemm_f32f32f32of32` for FP32 inputs). Online-softmax max/sum reductions and the running output accumulator stay in FP32. `out_dt` must either equal `qkv_dt` or be `data_type_t::none`.

#### Dynamic INT8 compute

`is_qk_quant` and `is_pv_quant` independently select INT8 compute for the two
attention matmuls. Both default to `false`, both require BF16 Q/K/V with BF16
output, and each may be set on its own:

| `is_qk_quant` | `is_pv_quant` | Q×K^T | probability×V |
| --- | --- | --- | --- |
| `false` | `false` | BF16 | BF16 |
| `true` | `false` | INT8 s8×s8 | BF16 |
| `false` | `true` | BF16 | INT8 u8×s8 |
| `true` | `true` | INT8 s8×s8 | INT8 u8×s8 |

Either flag selects INT8 only when the Q/K/V data type is BF16. FP32 and FP16
calls continue through their normal floating-point paths, so mixed-dtype
workloads are unaffected. Both flags require the flash backend; requesting
either with another kernel returns `status_t::unimplemented`.

The public tensor data types do not change. Internally, with `is_qk_quant`:

- Q and K are quantized once per head to symmetric s8 with one scale per token,
  through a single LOWOHA `group_dynamic_quant` call.
- Q×K^T uses s8×s8 via `matmul_direct` and writes dequantized FP32 scores.
- The attention scale is folded into the per-token Q dequantization scales.

With `is_pv_quant`:

- V is quantized once per KV head to per-channel s8. All V heads are processed
  by one grouped per-channel `group_dynamic_quant` call across the configured
  SDPA thread team.
- The online-softmax exponential loop writes its known `[0, 1]` output directly
  as u8 with scale `1/255`, avoiding a separate probability min/max scan or
  reorder.
- probability×V uses u8×s8 and accumulates dequantized FP32 output.

When `is_pv_quant` is set without `is_qk_quant`, Q×K^T stays on the BF16 GEMM
with the attention scale folded into its alpha, and only the V and probability
tensors are quantized.

Online softmax max/sum reductions and the running output accumulator remain
FP32 in every combination. Both paths require an AOCL-DLP-enabled build and
AVX512-F, AVX512-BW, AVX512-VL and AVX512-VNNI.

##### ISA requirement for FP16

The FP16 paths require **AVX512-FP16** at runtime (CPUID leaf 7, subleaf 0, EDX bit 23; available on Zen 5 / Sapphire Rapids and later). The operator probes this via `zendnnl_platform_info().get_avx512_f16_status()` and rejects FP16 calls early with `status_t::isa_unsupported` when the ISA is absent, mirroring the gate enforced by the matmul backend (`lowoha_matmul.cpp`).

#### Attention mask

The attention mask is an optional additive mask applied before the softmax. It supports two layouts:

| `mask_ndims` | Shape | Broadcasting |
|--------------|-------|-------------|
| `2` | `[S_q, S_kv]` | Broadcast across batch and heads |
| `4` | `[B, H, S_q, S_kv]` | Per-batch, per-head mask (dims of size 1 are broadcast) |

The last dimension (`S_kv`) must have stride 1 (contiguous). When `is_causal = true`, future positions are filled with `-inf` regardless of the mask.

#### Sliding window

Sliding-window attention (Gemma 3 / EmbeddingGemma SWA layers, vLLM's `sliding_window`) is off by default. Set `sliding_window = true` and `sliding_window_size = W` to restrict query position *i* to key positions

$$
i - (W - 1) \le j \le i + (W - 1)
$$

matching vLLM's `(W-1, W-1)` left/right convention for bidirectional encoder attention. Combining it with `is_causal = true` drops the right half of the band, leaving `i - (W-1) <= j <= i`.

The band is applied inside the kernel, so no `[S_q, S_kv]` mask tensor has to be built. The flash backend (floating-point and dynamic-INT8) skips whole KV tiles that fall outside the band and shrinks the remaining GEMM to the live `[band_lo, band_hi)` keys inside a visited tile, making the cost O(S·W) instead of O(S²). A window can still be combined with an additive `attn_mask` (e.g. a padding mask); the band is applied first, then the mask is added.

`sliding_window = true` with `sliding_window_size <= 0` is rejected with `status_t::failure`. The BMM backend does not support sliding window.

### Return value

- `status_t::success` — attention computed and written to `output`
- `status_t::failure` — validation or kernel failure (null pointers, invalid dims/strides, unsupported dtype combination, non-zero dropout, non-contiguous mask last dim)
- `status_t::isa_unsupported` — FP16 request on a host without **AVX512-FP16** (see [ISA requirement for FP16](#isa-requirement-for-fp16))

## Execution Flow

`sdpa_direct` is a thin profiling/logging wrapper around the flash backend. The backend validates inputs, builds lightweight tensor/mask views, then dispatches on the runtime ISA and the Q sequence length:

```
sdpa_direct()
  │  Profiling / logging wrapper
  │
  ▼
flash_sdpa()
  │  1. Validate inputs (null checks, dimensions, strides, dtypes)
  │  2. Build lightweight tensor views from sdpa_params
  │  3. Build mask view (if mask provided)
  │
  ▼
sdpa_flash_cpu_run_internal()
  │  1. Reject FP16 without AVX512-FP16
  │  2. INT8 ISA gate: is_qk_quant || is_pv_quant requires
  │     AVX512-F, AVX512-BW, AVX512-VL, AVX512-VNNI (else isa_unsupported)
  │  3. Runtime SIMD dispatch:
  │       if (AVX-512 available) → SimdOps<avx512_tag>  (16-lane __m512)
  │       else                   → SimdOps<scalar_tag>   (1-lane scalar)
  │  4. if (is_qk_quant || is_pv_quant) → sdpa_flash_cpu_run_int8()
  │       same KV loop as below; does not enter the FP dispatch
  │
  ▼
flash_attention_kernel_sa_dispatch<SimdTag>()   (floating point only)
  │  Select tile sizes based on Q sequence length (seq_len):
  │    seq_len >= 768  → q_split=256, kv_split=512
  │    seq_len >= 192  → q_split=64,  kv_split=512
  │    seq_len <  192  → q_split=32,  kv_split=512
  │    batch > 4       → q_split=512  (override)
  │
  ▼
cpu_flash_attention_sa / INT8 kernel
  │  OpenMP parallel loop over batch × heads × q_tiles
  │  Query block starts at m. KV tile start is n; GEMM start is n_gemm.
  │
  │  For each (batch_i, head_j, q_tile at m):
  │    if sliding_window:
  │      win_left = W-1; win_right = 0 if causal else W-1
  │      num_keys = min(kvSize, m + qBlock + win_right)
  │      n_start  = tile-align max(0, m - win_left)
  │    else:
  │      n_start = 0
  │      num_keys = m + qBlock if causal, else kvSize
  │    ┌─ for n = n_start; n < num_keys; n += kv_split:
  │    │    if sliding_window:
  │    │      shrink to live keys [band_lo, band_hi) inside the tile
  │    │      skip the tile when that span is empty
  │    │    1. GEMM: Q_tile × K[n_gemm:]^T       (via AOCL BLAS)
  │    │    2. if window: fill keys outside the row band with -inf
  │    │       else if causal on the last KV span: fill future with -inf
  │    │    3. Scale + additive mask fusion at column n_gemm
  │    │    4. Row-wise max + exp + sum           (SIMD fused reductions)
  │    │    5. Rescale the accumulator only after the first live tile
  │    │    6. GEMM: softmax_tile × V[n_gemm:]
  │    │       beta = 0 on the first live tile, else 1
  │    └─
  │    if no live KV tile: write zeros
  │    else: output = accumulator / sum          (SIMD scaled store)
```

## Flash Attention Algorithm

The kernel implements the online softmax flash attention algorithm, which avoids materializing the full S×S attention matrix:

1. **Tiling**: Q is split into tiles of `q_split_size` rows. K and V are split into tiles of `kv_split_size` rows. Only one Q×K tile is materialized at a time.

2. **Online Softmax**: For each Q tile, the kernel iterates over KV tiles and maintains running statistics (row-wise max and sum) to compute the softmax incrementally. When a new KV tile produces a larger max, the previously accumulated output is rescaled.

3. **Memory Efficiency**: Scratch memory per thread is `O(q_split × kv_split + q_split × head_dim)` instead of `O(S × S)` for the full attention matrix. Scratch buffers are thread-local and reused across calls. The dynamic-INT8 path is the exception: its quantized Q/K/V buffers are whole-tensor, `O(batch × heads × seq_len × head_dim)`, and therefore grow with sequence length — see **Scratch memory** below.

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
| Output Write | `write_scaled_output_row` | SIMD scaled store (FP32 / BF16 / FP16 conversion) |
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

### Example 3: FP16 SDPA with FP16 Attention Mask

The setup mirrors the BF16 example; only `qkv_dt`/`out_dt`/`mask_dt` change.
The call returns `status_t::isa_unsupported` if the host lacks **AVX512-FP16** — see
[ISA requirement for FP16](#isa-requirement-for-fp16).

```cpp
int lowoha_sdpa_fp16_example() {
  using namespace zendnnl::lowoha::sdpa;

  int64_t B = 2, H = 12, S = 512, D = 64;

  // FP16 stored as uint16_t (IEEE 754 half).
  std::vector<uint16_t> query(B * H * S * D);
  std::vector<uint16_t> key(B * H * S * D);
  std::vector<uint16_t> value(B * H * S * D);
  std::vector<uint16_t> output(B * H * S * D, 0);

  // FP16 mask, broadcast across batch and heads ([S_q, S_kv]).
  std::vector<uint16_t> mask(S * S, 0);

  sdpa_params params;
  params.batch      = B;
  params.num_heads  = H;
  params.seq_len    = S;
  params.kv_seq_len = S;
  params.head_dim   = D;

  // Contiguous BHSD strides
  params.q_stride_b = H * S * D;  params.q_stride_h = S * D;
  params.q_stride_s = D;          params.q_stride_d = 1;
  params.k_stride_b = H * S * D;  params.k_stride_h = S * D;
  params.k_stride_s = D;          params.k_stride_d = 1;
  params.v_stride_b = H * S * D;  params.v_stride_h = S * D;
  params.v_stride_s = D;          params.v_stride_d = 1;
  params.o_stride_b = H * S * D;  params.o_stride_h = S * D;
  params.o_stride_s = D;          params.o_stride_d = 1;

  params.qkv_dt    = data_type_t::f16;
  params.out_dt    = data_type_t::f16;   // must match qkv_dt
  params.mask_dt   = data_type_t::f16;   // f32 is also valid with FP16 Q/K/V
  params.scale     = 1.0 / std::sqrt(static_cast<double>(D));
  params.is_causal = false;
  params.dropout_p = 0.0;

  // 2-D mask metadata [S_q, S_kv]
  params.mask_ndims      = 2;
  params.mask_sizes[0]   = S;   params.mask_strides[0] = S;
  params.mask_sizes[1]   = S;   params.mask_strides[1] = 1;

  status_t status = sdpa_direct(
    query.data(), key.data(), value.data(),
    mask.data(),
    output.data(),
    params
  );

  // status == status_t::isa_unsupported if the runtime CPU lacks
  // AVX512-FP16.
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 4: FP32 SDPA with 4-D Attention Mask

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

### Example 5: SDPA with Broadcast 2-D Mask

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

### Example 6: Cross-Attention (Encoder-Decoder)

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

## Notes and best practices

1. **Tiling heuristics**: The flash kernel selects tile sizes from the Q sequence length (`seq_len`) to balance parallelism and cache efficiency; `kv_split_size` is clamped at runtime to `min(512, kv_seq_len)`.

   | Condition | `q_split_size` | `kv_split_size` | Rationale |
   |-----------|---------------|-----------------|-----------|
   | `seq_len >= 768` | 256 | 512 | Large tiles maximize GEMM efficiency |
   | `seq_len >= 192` | 64 | 512 | Moderate tiles for medium sequences |
   | `seq_len < 192` | 32 | 512 | Small tiles preserve parallelism for short sequences |
   | `batch > 4` (override) | 512 | 512 | Larger Q tiles when batch parallelism is sufficient |

2. **Scratch memory**: Each calling thread retains grow-only scratch storage
   that is reused across calls. The dynamic-INT8 path also retains its
   quantized Q/K/V, scale, FP32 tile, and BF16/u8 probability buffers, avoiding
   repeated allocation and value-initialization for stable or smaller shapes.
   Call `sdpa_flash_cpu_free_scratch()` on the owning thread to release all
   retained flash scratch storage eagerly.

   | Buffer | Size | Purpose |
   |--------|------|---------|
   | `qk_data` | `q_split × kv_split` | Q×K^T tile (FP32 accumulation) |
   | `qk_max_data` | `q_split` | Running row-wise max for online softmax |
   | `qk_sum_data` | `q_split` | Running row-wise sum for online softmax |
   | `dst_data` | `q_split × head_dim` | Running output accumulator (FP32) |
   | `qk_reduced_data` | `q_split × kv_split` | Reduced-precision softmax weights/tile feeding the softmax×V GEMM (BF16 / FP16 path) |

3. **Threading**: Set `params.num_threads` to control parallelism (`0` = auto: uses `OMP_NUM_THREADS` or the system default). Work is distributed across `batch × heads × query-tiles`, so larger batch/head counts expose more parallel work.

4. **Self- vs cross-attention**: For self-attention set `kv_seq_len = 0` (or `seq_len`); for cross-attention set `kv_seq_len` to the K/V length. K and V always share `S_kv`.

5. **Dropout**: Only `dropout_p = 0.0` is supported. Non-zero dropout returns `status_t::failure`.

6. **Head dimension contiguity**: `stride_d` must be `1` for Q, K, and V (required by the underlying GEMM).

7. **Mask contiguity**: The last mask dimension (`S_kv`) must have stride `1`.

8. **FP16 ISA gate**: FP16 requests on hosts without **AVX512-FP16** return `status_t::isa_unsupported`; check for this status and fall back or skip gracefully.
