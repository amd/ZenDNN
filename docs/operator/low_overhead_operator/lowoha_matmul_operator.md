(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA MatMul Operator

## Table of Contents

- [Overview](#overview)
- [General MatMul Operation](#general-matmul-operation)
- [Core API: `matmul_direct`](#core-api-matmul_direct)
- [Parameters Structure](#parameters-structure)
- [Usage Examples](#usage-examples)
- [Reorder Quantization](#reorder-quantization-source-quantization-via-reorder)
- [Weight Caching and Reordering](#weight-caching-and-reordering)
- [Clearing Weight Caches](#clearing-weight-caches)
- [Zero-Point Compensation Caching (INT8)](#zero-point-compensation-caching-int8)
- [Backend Selection](#backend-selection)

## Overview

The LowOHA MatMul Operator is the primary matrix multiplication API in ZenDNN — a high-performance, low-overhead matrix multiplication operator designed for latency-sensitive inference workloads. It provides a direct, function-based interface (`matmul_direct`) to backend libraries (AOCL-DLP, LibXSMM, OneDNN) with built-in weight caching and fused post-operations, making it the recommended entry point for most inference paths (model serving, LLM decoding, batched MatMul, INT8 / WOQ quantized inference).

Key characteristics:

- **Minimal execution overhead** — direct function call with no operator-factory or object-construction cost on the hot path
- **Repeated weight reuse** — reordered weights are cached across calls and keyed on the weight pointer
- **Backend-native post-operation fusion** — post-ops, INT8, and WOQ paths exposed directly to the backend
- **Direct control over execution parameters** — the caller picks dtypes, layout, threading, and (optionally) the backend algorithm

For the operator-factory-based API (`matmul_operator_t`, used for graph-style composition and framework integration patterns), see [MatMul Operator](../tensor_operator/matmul_operator.md).

## General MatMul Operation

Let:

- *A* ∈ ℝ<sup>mxk</sup> or ℝ<sup>bsxmxk</sup>: Input Matrix or Batched Input Matrix
- *B* ∈ ℝ<sup>kxn</sup> or ℝ<sup>bsxkxn</sup>: Weight Matrix or Batched Weight Matrix
- *C* ∈ ℝ<sup>mxn</sup> or ℝ<sup>bsxmxn</sup>: Output Matrix or Batched Output Matrix
- *Bias* ∈ ℝ<sup>1xn</sup>: Optional Bias vector
- *Alpha* (*α*): Scaling factor for the matrix product
- *Beta* (*β*): Scaling factor for the output (for accumulation)
- *Post_ops(x)*: Optional activation or binary post operations (e.g., ReLU, GELU, `binary_add`, etc.)
- *Transpose(A, B)*: Optional transpose operation on matrices A or B

The computation can be expressed as:

$$
C = \text{PostOps}(\alpha \cdot (A \cdot B) + \beta \cdot C + \text{Bias})
$$

### LowOHA MatMul Flow Diagram

```text
       Input A                            Weights B
  ([M x K] or [BS x M x K])         ([K x N] or [BS x K x N])
    (Optional:                          (Constant weights:
 dynamic quant to S8/U8)            reorder cached on first call)
       |                                    |
       +------------------x-----------------+
                          |
                       MatMul    ← alpha * (A x B) + beta * C
                          |
                   +------v------+
                   |   Add Bias  |
                   +------v------+
                   | Scale / ZP  |  ← INT8 dequant or WOQ dequant
                   +------v------+
                   |   Post-Op   |  ← Activation / Binary Op (fused)
                   +------v------+
                       Output C
```

## Core API: `matmul_direct`

The primary interface for LowOHA MatMul is the `matmul_direct` function:

```cpp
status_t matmul_direct(
  const char layout,           // 'r' for row-major
  const bool transA,           // Transpose input matrix A
  const bool transB,           // Transpose weight matrix B
  const int M,                 // Number of rows in A (and C)
  const int N,                 // Number of columns in B (and C)
  const int K,                 // Number of columns in A / rows in B
  const float alpha,           // Scaling factor for A*B
  const void *src,             // Input matrix A
  const int lda,               // Leading dimension of A
  const void *weight,          // Weight matrix B
  const int ldb,               // Leading dimension of B
  const void *bias,            // Optional bias vector (can be nullptr)
  const float beta,            // Scaling factor for output C
  void *dst,                   // Output matrix C
  const int ldc,               // Leading dimension of C
  const bool is_weights_const, // Whether weights are constant (enables caching)
  matmul_batch_params_t batch_params, // Batch parameters (batch sizes and strides)
  matmul_params params         // LowOHA parameters (dtypes, post-ops, quantization)
);
```

## Parameters Structure

### `matmul_batch_params_t`

Structure for batch dimensions and strides:

```cpp
struct matmul_batch_params_t {
  int Batch_A;                    // Batch size for source tensor (default: 1)
  int Batch_B;                    // Batch size for weight tensor (default: 1)
  size_t batch_stride_src;        // Byte stride between batches for source (-1 = auto-calculate)
  size_t batch_stride_wei;        // Byte stride between batches for weight (-1 = auto-calculate)
  size_t batch_stride_dst;        // Byte stride between batches for destination (-1 = auto-calculate)
};
```

### `matmul_params`

The main configuration structure for LowOHA MatMul:

```cpp
struct matmul_params {
  matmul_data_types dtypes;                  // Data types for tensors
  std::vector<matmul_post_op> postop_;       // Post-operations
  matmul_quantization_params_t quant_params; // Quantization parameters
  char mem_format_a;                         // Memory format for A ('n'=non-reordered, 'r'=reordered)
  char mem_format_b;                         // Memory format for B ('n'=non-reordered; 'r'=pre-reordered into AOCL-DLP blocked layout via a prior reorder_direct() prepack — backend skips its internal reorder)
  matmul_algo_t lowoha_algo;                 // Preferred backend algorithm (default: none → single uses aocl_dlp_blocked; batched uses libxsmm when built, otherwise aocl_dlp/reference)
  int32_t num_threads;                       // Number of threads (0 = auto)
  std::string plugin_op;                     // Plugin op name
  bool dynamic_quant;                        // Enable dynamic quantization of source (default: false)
  int32_t weight_cache_type;                 // Per-call weight cache mode (0=disabled, 1=out-of-place, 2=allow in-place, needs a writable weight buffer; capped by ZENDNNL_MATMUL_WEIGHT_CACHE env var)
  size_t wei_buffer_capacity_bytes;          // Writable bytes at the weight pointer (0 = exactly k*n). INT8 in-place needs this when the blocked size exceeds k*n.
};
```

A partitioned BMM that shares one weight and has a source zero-point stays out-of-place under mode 2, even when the blocked image already equals `k*n`.

### `matmul_data_types`

Specifies the data types for each tensor:

```cpp
struct matmul_data_types {
  data_type_t src;      // Input data type
  data_type_t wei;      // Weight data type
  data_type_t dst;      // Output data type
  data_type_t bias;     // Bias data type
  data_type_t compute;  // Computation type
};
```

**Supported Combinations:**

| Src Type | Weight Type | Bias Type | Dst Type | Notes |
|----------|-------------|-----------|-------------|-------|
| F32 | F32 | F32 | F32 | Standard floating-point |
| BF16 | BF16 | F32/BF16 | F32/BF16 | Mixed-precision BFloat16 |
| F16 | F16 | F32/F16 | F16/F32 | Half-precision (requires AVX512-FP16, except on the native BRGEMM). No scales or zero-points. |
| BF16 | S4/U4 | F32/BF16 | F32/BF16 | Weight-only (`dynamic_quant=false`). Source stays BF16. S4 is symmetric (no zero-point). U4 is asymmetric (weight zero-point required, S8 or BF16). |
| BF16/F32/S8 | S4 | F32/BF16 | F32/BF16 | W4A8 (`compute=s8`, no zero-points). A BF16 or F32 source requires `dynamic_quant=true`. S8 source requires `dynamic_quant=false` and a source scale. An F32 source writes an F32 destination. See [W4A8](#w4a8-symmetric-w4--a8). |
| S8 | S4 | F32/BF16 | F32 | GGML Q4_0. Packed symmetric S4 with an S8 source. Source scale is per-token or per-group. No zero-point. |
| U8/S8 | S8 | F32/BF16/S8/U8/S32 | F32/BF16/S8/U8/S32/F16 | INT8. U8 source scale is per-tensor only. An S8 per-token or per-group source scale writes F32 or BF16 only. S32, S8, U8, and F16 outputs need a per-tensor source scale. |
| BF16/F32 | S8 | F32/BF16/S8/U8 | F32/BF16/S8/U8 | INT8. No S32 or F16 output. A per-token source scale writes F32 or BF16 only. |

> **Note:** F16 MatMul requires AVX512-FP16. Without that ISA, F16 operations return `status_t::isa_unsupported`, except a call that asks for the native BRGEMM, which needs AVX-512 F instead (see [Native BRGEMM](#native-brgemm)). F32, BF16, and F16 matmul have no scales and no zero-points.

### `matmul_post_op`

Defines a single post-operation:

```cpp
struct matmul_post_op {
  post_op_type_t po_type;      // Type of post-operation
  void *buff;                  // Buffer for binary ops (nullptr for activations)
  data_type_t dtype;           // Data type of the buffer
  std::vector<int64_t> dims;   // Dimensions of the buffer
  float alpha;                 // Alpha parameter
  float beta;                  // Beta parameter
};
```

**Supported Post-Op Types:**

| Post-Op Type | Description | Requires Buffer |
|--------------|-------------|-----------------|
| `post_op_type_t::relu` | Rectified Linear Unit | No |
| `post_op_type_t::gelu_erf` | GELU (erf variant) | No |
| `post_op_type_t::gelu_tanh` | GELU (tanh variant) | No |
| `post_op_type_t::swish` | SiLU / Swish | No |
| `post_op_type_t::sigmoid` | Sigmoid | No |
| `post_op_type_t::tanh` | Hyperbolic Tangent | No |
| `post_op_type_t::clip` | Element-wise Clip to `[alpha, beta]` (lower, upper) | No |
| `post_op_type_t::mish` | Mish | No |
| `post_op_type_t::binary_add` | Element-wise Add | Yes |
| `post_op_type_t::binary_mul` | Element-wise Multiply | Yes |

> **Note:** On AOCL-DLP INT8 `sym_quant` / `dynamic_quant` kernel paths, the `mish` post-op is rejected by `validate_matmul_direct_inputs` (input validation runs by default unless `ZENDNNL_DIAGNOSTICS_ENABLE=0`). A call that asks for the native BRGEMM (see [Native BRGEMM](#native-brgemm)) takes it on those calls, except on GGML packed weights.

> **Note:** `matmul_direct` also takes the `post_op_type_t::elu`, `square`, `abs`, `sqrt`, `exp`, `log` and `softmax` post-ops (`softmax` over each output row). The native BRGEMM (`ZENDNNL_MATMUL_ALGO=10`, or `lowoha_algo = matmul_algo_t::native_brgemm`) applies them on the BF16, INT8 and FP32 calls it serves. The AOCL-DLP chain does not map them: a call on the AOCL-DLP route, the default with the algorithm unset, runs on the reference kernel when its chain holds one of them, or is refused where that kernel does not serve it, and a `pooling` post-op is refused; see [AOCL-DLP route](#aocl-dlp-route).

**Binary Post-Op Dimensions:**

For binary post-ops (`binary_add` and `binary_mul`), the `dims` field specifies the tensor shape:

| Operation | Dims | Shape | Description |
|-----------|------|-------|-------------|
| MM (Matrix Multiply) | 2D | `{M, N}` | Element-wise operation on output |
| BMM (Batched Matrix Multiply) | 2D | `{M, N}` | Same values broadcast to all batches |
| BMM (Batched Matrix Multiply) | 3D | `{Batch, M, N}` | Different values per batch |

### `matmul_quantization_params_t`

Quantization scale and zero-point parameters for quantized operations (WOQ and INT8):

```cpp
struct matmul_quantization_params_t {
  /**
   * Individual quantization parameter (scale or zero-point)
   * 
   * Dimensions determine quantization granularity:
   *   - Per-tensor:  dims = {1} or {1,1}  → single scale for entire tensor
   *   - Per-channel: dims = {1, N}        → one scale per output channel
   *   - Per-group:   dims = {G, N}        → G groups along K, where G = K/group_size
   */
  struct matmul_quant_t {
    const void *buff;              // Pointer to scale/zero-point data
    data_type_t dt;                // Scale: f32/bf16. INT8 zero-point: s8/u8/s32. WOQ U4 weight ZP: s8 or bf16.
    std::vector<int64_t> dims;     // Dimensions of the quantization tensor
  };
  
  matmul_quant_t src_scale;     // Source tensor scale (for INT8)
  matmul_quant_t wei_scale;     // Weight tensor scale (required for WOQ and INT8)
  matmul_quant_t dst_scale;     // Destination tensor scale (for INT8)
  matmul_quant_t src_zp;        // Source tensor zero-point (for INT8 asymmetric quantization)
  matmul_quant_t wei_zp;        // Weight tensor zero-point (for WOQ asymmetric or INT8)
  matmul_quant_t dst_zp;        // Destination tensor zero-point (for INT8)
};
```

**INT8 Quantization Granularity:**

| Buffer  | Scale Granularity (mandatory)              | Zero-Point Granularity (when present) |
|---------|--------------------------------------------|----------------------------------------|
| Input   | Per-tensor, per-token, or per-group        | Per-tensor                             |
| Weights | Per-tensor, per-channel, or per-group      | Per-tensor                             |
| Output  | Per-tensor                                 | Per-tensor                             |

> **Note:**
> - Zero-points are not supported when input or weight scales are per-token or per-group.
> - Only bf16 and f32 output data types are supported when input or weight scales are per-token or per-group; a call that asks for the native BRGEMM also takes u8, s8 and s32 outputs there (not f16).

**INT8 Source Scale Granularity Details:**

| Granularity | Scale Dims | Description |
|-------------|------------|-------------|
| Per-tensor  | `{1}` or `{1, 1}` | Single scale for entire source matrix |
| Per-token   | `{M, 1}` | One scale per row (token) of the source matrix |
| Per-group   | `{M, G}` | G groups along K dimension, where G = K/group_size |

**INT8 Weight Scale Granularity Details:**

| Granularity | Scale Dims | Description |
|-------------|------------|-------------|
| Per-tensor  | `{1}` or `{1, 1}` | Single scale for entire weight matrix |
| Per-channel | `{1, N}` | One scale per output channel (column) |
| Per-group   | `{G, N}` | G groups along K dimension (G = K/group_size) |

> **Note:** When both source and weight use per-group scales, the number of groups (G) must match. `group_size = K / G`. Per-tensor shapes may also be `{}`. Per-channel weight scales may also be `{N}`.

**INT8 source × weight scale pairs:**

Scales are F32 or BF16. A zero-point, when present, is per-tensor: source and weight zero-points are S32, S8, or U8; the destination zero-point is S8, U8, or S32. Destination scale and destination zero-point are per-tensor. A zero-point is not combined with a per-token or per-group scale.

| Source | Source scale | Weight scale | Zero-point | Output |
|--------|--------------|--------------|------------|--------|
| U8 or S8 | per-tensor | per-tensor | none, or per-tensor source and/or weight | U8, S8, S32, F32, BF16, F16 |
| U8 or S8 | per-tensor | per-channel | none, or per-tensor source and/or weight | U8, S8, S32, F32, BF16, F16 |
| S8 | per-token | per-channel | none | F32, BF16 |
| S8 | per-token | per-group | none | F32, BF16 |
| S8 | per-group | per-group (same G) | none | F32, BF16 |
| S8, M = 1 | per-token (one element) | per-group | none | F32, BF16 |
| S8 | per-tensor | per-group | none | F32, BF16 |
| BF16 or F32 | per-tensor | per-tensor | none, or per-tensor source | BF16 source: F32, BF16, S8, U8. F32 source: F32, BF16, S8, U8 |
| BF16 or F32 | per-tensor | per-channel | none, or per-tensor source | same as the row above |
| BF16 or F32 | per-token | per-channel | none | F32, BF16 |

A per-group BF16 or F32 source is quantized to S8 first, then uses the S8 per-group row. A per-token source is not paired with a per-tensor weight. A per-group source is not paired with a per-channel or per-tensor weight.

On the [AOCL-DLP route](#aocl-dlp-route), per-group weight scales `{G, N}` need a group size `K / G` that is a multiple of 4. The S8 rows with F32 or BF16 output run on AOCL-DLP's sym-quant kernel when the source scale is per-token or per-group, when the weight scale is per-group, or at M = 1 with one source value and a per-channel or per-group weight scale. That kernel reads F32 weight scales, or weight scales of the source scale's type; BF16 weight scales with F32 source scales return `status_t::unimplemented` there. The other rows take BF16 weight scales with either source scale type.

**WOQ Quantization Granularity for weights:**

The source is plain BF16. It has no scale and no zero-point. Weight scales are F32 or BF16. S4 has no zero-point. U4 requires a zero-point, S8 or BF16. The U4 zero-point granularity is chosen separately from the scale. `group_size` follows the scale: `K` for per-tensor and per-channel, `K / G` for per-group. A per-group size other than `K` is even.

| Weight scale | U4 zero-point |
|--------------|---------------|
| per-tensor | per-tensor, per-channel, or per-group |
| per-channel | per-tensor, per-channel, or per-group |
| per-group | per-tensor, per-channel, or per-group (same G when both are per-group) |

> **Note:** WOQ requires `is_weights_const = true` for weight reordering and caching.

### W4A8 (symmetric W4 + A8)

Symmetric 4-bit weights with **s8** activations at GEMM time (AOCL-DLP `sym_quant`, the custom kernel, or the native BRGEMM). Distinct from WOQ, which keeps **bf16** activations. Destination is **bf16** or **f32**. Scale tensors accept **f32** or **bf16**. No `src_zp` / `wei_zp`. `dtypes.compute` must be **s8**. A call that asks for the native BRGEMM (see [Native BRGEMM](#native-brgemm)) also takes **f16** weight and source scales.

Two entry contracts:

| Entry | Src dtype | `dynamic_quant` | Who quantizes A | `src_scale.buff` |
|-------|-----------|-----------------|-----------------|------------------|
| **Dynamic** | BF16 or F32 | `true` | Library at runtime | Optional (computed if null); `dims`/`dt` required |
| **Static** | S8 (pre-quantized) | `false` | Caller | **Required** |

`s8` source with `dynamic_quant = true` is rejected. `bf16` source with `dynamic_quant = false` is **WOQ**, not W4A8.

| Tensor  | Caller dtype | Compute dtype | Scale dtype | Scale granularity | Zero-point |
|---------|--------------|---------------|-------------|-------------------|------------|
| Source  | BF16 or F32 (dynamic), or S8 (static) | S8 | F32, BF16; F16 on the native BRGEMM | Per-token `{M, 1}` or per-group `{M, G}` (`{1, 1}` only when `M == 1`) | Not supported |
| Weights | S4 (packed)  | S8            | F32, BF16; F16 on the native BRGEMM | Per-group `{G, N}` (G = K / group_size; `{1, N}` is one group over K) | Not supported (symmetric) |
| Output  | BF16 or F32 (F32 for an F32 source) | —             | —           | —                 | —          |

> **Notes:**
> - **Backends:** with the algorithm unset, the custom-kernel route where it serves the call and AOCL-DLP (`aocl_dlp` / `aocl_dlp_blocked`) otherwise; the native BRGEMM for a call that asks for it (`ZENDNNL_MATMUL_ALGO=10`, or `lowoha_algo = matmul_algo_t::native_brgemm`).
> - **Shape constraints:** `src_scale` dims `{M, 1}` or `{M, G}` (`G` matching `wei_scale.dims[0]`); `wei_scale` dims `{G, N}` with `N` matching the matmul `N`; `K` divisible by `G`; `K / G` (group size) a multiple of 4. Compact `{M, 1}` scales are broadcast to `{M, G}` in the AOCL path before GEMM.
> - **vs WOQ:** WOQ is bf16×s4 with `dynamic_quant = false`. W4A8 uses the symmetric grouped blocked weight layout, not the weight-only s4 layout.

**Supported Data Types for Scales and Zero Points by Backend:**

| Quant Mode | Tensor  | Parameter  | DLP Backend                        | OneDNN Backend                      |
|------------|---------|------------|------------------------------------|-------------------------------------|
| INT8       | Source  | Scale      | F32, BF16                          | F32                                 |
| INT8       | Source  | Zero-Point | S32, S8, U8                        | S32                                 |
| INT8       | Weights | Scale      | F32, BF16                          | F32                                 |
| INT8       | Weights | Zero-Point | S32, S8, U8                        | S32                                 |
| INT8       | Output  | Scale      | F32, BF16                          | F32, BF16 *(BF16 converted to F32)* |
| INT8       | Output  | Zero-Point | S8, U8, S32                        | S32                                 |
| WOQ (S4)   | Weights | Scale      | F32, BF16                          | ✗ *(WOQ always dispatches to DLP)*  |
| WOQ (S4)   | Weights | Zero-Point | ✗ *(symmetric, no zero-point)*     | ✗ *(WOQ always dispatches to DLP)* |
| WOQ (U4)   | Weights | Scale      | F32, BF16                          | ✗ *(WOQ always dispatches to DLP)*  |
| WOQ (U4)   | Weights | Zero-Point | S8, BF16                           | ✗ *(WOQ always dispatches to DLP)*  |
| W4A8      | Source  | Scale      | F32, BF16 (F16 on the native BRGEMM) | ✗ *(W4A8 never runs on oneDNN)*    |
| W4A8      | Weights | Scale      | F32, BF16 (F16 on the native BRGEMM) | ✗ *(W4A8 never runs on oneDNN)*    |

> **Notes:**
> - **WOQ Routing:** WOQ is supported by the AOCL-DLP backend only.
> - **W4A8 Routing:** W4A8 runs on the custom-kernel route or AOCL-DLP with the algorithm unset, and on the native BRGEMM for a call that asks for it.
> - **S4 vs U4:** S4 weights use symmetric quantization (no zero-point). U4 weights use asymmetric quantization with zero-point.
> - **U4 Zero-Point Domain:** The dequantization formula for U4 depends on the zero-point data type. See the dequantization formulas in the WOQ key points section below for details.

## Usage Examples

### Example 1: BF16 MatMul with Multiple Post-Ops

```cpp
int lowoha_matmul_bf16_fused_ops_example() {
  using namespace zendnnl::lowoha::matmul;
  
  int M = 64, N = 128, K = 256;
  int lda = K, ldb = N, ldc = N;
  
  // Allocate BF16 matrices (stored as uint16_t)
  std::vector<uint16_t> A_bf16(M * K);
  std::vector<uint16_t> B_bf16(K * N);
  std::vector<uint16_t> C_bf16(M * N);
  std::vector<float> bias(N, 0.0f);
  
  // Binary add tensor
  std::vector<uint16_t> add_tensor(M * N);
  
  // Configure data types
  matmul_data_types dtypes;
  dtypes.src = data_type_t::bf16;
  dtypes.wei = data_type_t::bf16;
  dtypes.dst = data_type_t::bf16;
  dtypes.bias = data_type_t::f32;
  dtypes.compute = data_type_t::f32;
  
  // Configure post-operations: GELU -> Binary Add
  matmul_params params;
  params.dtypes = dtypes;
  
  // Post-op 1: GELU
  matmul_post_op gelu_op;
  gelu_op.po_type = post_op_type_t::gelu_erf;
  gelu_op.buff = nullptr;
  gelu_op.dtype = data_type_t::none;
  gelu_op.dims = {M, N};
  params.postop_.push_back(gelu_op);
  
  // Post-op 2: Binary Add
  matmul_post_op add_op;
  add_op.po_type = post_op_type_t::binary_add;
  add_op.buff = add_tensor.data();
  add_op.dtype = data_type_t::bf16;
  add_op.dims = {M, N};
  params.postop_.push_back(add_op);
  
  // Configure batch parameters
  matmul_batch_params_t batch_params;
  batch_params.Batch_A = 1;
  batch_params.Batch_B = 1;
  
  // Execute MatMul
  status_t status = matmul_direct(
    'r', false, false,
    M, N, K,
    1.0f,
    A_bf16.data(), lda,
    B_bf16.data(), ldb,
    bias.data(),
    0.0f,
    C_bf16.data(), ldc,
    true,  // is_weights_const (enables caching)
    batch_params,
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 2: Batched MatMul

```cpp
int lowoha_batched_matmul_example() {
  using namespace zendnnl::lowoha::matmul;
  
  int batch_size = 16;
  int M = 32, N = 64, K = 128;
  int lda = K, ldb = N, ldc = N;
  
  // Allocate batched matrices
  std::vector<float> A(batch_size * M * K, 1.0f);  // Batched input
  std::vector<float> B(K * N, 0.5f);               // Shared weights
  std::vector<float> C(batch_size * M * N, 0.0f);  // Batched output
  std::vector<float> bias(N, 0.0f);
  
  // Configure data types
  matmul_data_types dtypes;
  dtypes.src = data_type_t::f32;
  dtypes.wei = data_type_t::f32;
  dtypes.dst = data_type_t::f32;
  dtypes.bias = data_type_t::f32;
  dtypes.compute = data_type_t::f32;
  
  matmul_params params;
  params.dtypes = dtypes;
  
  // Configure batch parameters
  matmul_batch_params_t batch_params;
  batch_params.Batch_A = batch_size;  // Batched input
  batch_params.Batch_B = 1;           // Shared weights
  
  // Execute batched MatMul
  status_t status = matmul_direct(
    'r', false, false,
    M, N, K,
    1.0f,
    A.data(), lda,
    B.data(), ldb,
    bias.data(),
    0.0f,
    C.data(), ldc,
    true,  // is_weights_const (enables caching)
    batch_params,
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 3: WOQ MatMul with BF16 Input and S4 Weights (Symmetric)

This example demonstrates Weight-Only Quantization (WOQ) with signed 4-bit weights and per-group quantization, commonly used in LLM inference:

```cpp
int lowoha_woq_bf16s4_matmul_example() {
  using namespace zendnnl::lowoha::matmul;
  
  // Matrix dimensions
  constexpr int M = 16, K = 128, N = 64;
  constexpr int GROUP_SIZE = 32;              // Typical group size for LLM quantization
  constexpr int NUM_GROUPS = K / GROUP_SIZE;  // = 4 groups
  
  // Create weight scale tensor (per-group: {NUM_GROUPS, N})
  std::vector<float> wei_scale(NUM_GROUPS * N);
  for (int g = 0; g < NUM_GROUPS; ++g) {
    for (int n = 0; n < N; ++n) {
      wei_scale[g * N + n] = 1.0f + 0.1f * g;  // Varying scales per group
    }
  }
  
  // Create S4 packed weights (2 values per byte)
  size_t packed_weight_size = (K * N + 1) / 2;
  std::vector<int8_t> weights(packed_weight_size);
  int8_t s4_val = 1 & 0x0F;
  int8_t packed_val = s4_val | (s4_val << 4);  // Same value in both nibbles
  std::fill(weights.begin(), weights.end(), packed_val);
  
  // Create BF16 input (stored as int16_t)
  std::vector<int16_t> input(M * K);
  std::fill(input.begin(), input.end(), 0x3F80);  // BF16 representation of 1.0f
  
  // Output tensor
  std::vector<float> output(M * N, 0.0f);
  
  // Configure data types for WOQ
  matmul_data_types dtypes;
  dtypes.src = data_type_t::bf16;
  dtypes.wei = data_type_t::s4;
  dtypes.dst = data_type_t::f32;
  dtypes.bias = data_type_t::none;
  dtypes.compute = data_type_t::none;
  
  matmul_params params;
  params.dtypes = dtypes;
  
  // Setup per-group quantization parameters
  params.quant_params.wei_scale.buff = wei_scale.data();
  params.quant_params.wei_scale.dt = data_type_t::f32;
  params.quant_params.wei_scale.dims = {NUM_GROUPS, N};
  
  // Batch parameters
  matmul_batch_params_t batch_params;
  batch_params.Batch_A = 1;
  batch_params.Batch_B = 1;
  
  // Execute WOQ MatMul
  status_t status = matmul_direct(
    'r', false, false,
    M, N, K,
    1.0f,
    input.data(), K,
    weights.data(), N,
    nullptr,  // no bias
    0.0f,
    output.data(), N,
    true,  // is_weights_const (required for WOQ)
    batch_params,
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```

**Key Points for WOQ:**

- **S4 Weight Packing**: Weights are stored as 4-bit signed integers (range [-8, 7]), with 2 values packed per byte (low nibble and high nibble). S4 uses symmetric quantization — no zero-point is passed to the AOCL-DLP kernel.
- **U4 Weight Packing**: Weights are stored as 4-bit unsigned integers (range [0, 15]), with 2 values packed per byte. U4 uses asymmetric quantization with a zero-point. The zero-point data type can be `s8` (integer domain) or `bf16` (float domain), which determines the dequantization formula.
- **Quantization Granularity**: Per-group quantization divides K dimension into groups, each with its own scale (and zero-point for U4). The group size must be even and must divide K evenly.
- **Constant Weights**: `is_weights_const = true` is required for WOQ to enable weight reordering and caching.
- **Dequantization Formula (S4)**: `dequant_value = s4_weight * scale`
- **Dequantization Formulas (U4)**: The formula depends on the zero-point data type:

| Zero-Point Type | Formula | Description |
|-----------------|---------|-------------|
| `s8` (integer domain) | `dequant = (u4_weight - zp) * scale` | Standard affine dequantization with integer zero-point |
| `bf16` (float domain) | `dequant = (u4_weight - 8) * scale + zp` | Asymmetric dequantization with midpoint shift (8 = midpoint of [0, 15]) and float zero-point added post-scaling |

**Key Differences from S4 WOQ:**

- **Weight type**: `data_type_t::u4` (unsigned, range [0, 15]) vs `data_type_t::s4` (signed, range [-8, 7])
- **Zero-point**: Required for U4 (asymmetric). S4 does not use zero-point in the AOCL-DLP kernel. U4 zero-point can be `s8` or `bf16`.
- **Dequantization**: Depends on zero-point data type — see the dequantization formulas table above.

### Static per-tensor U8×S8 custom-kernel route

Eligible decode-class calls (`M <= 32`) can use the grouped N-tile VNNI custom
kernel when `ZENDNNL_MATMUL_CUSTOM_KERNEL=1` and
`ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_INT8=1`:

- source: U8 with one positive finite F32/BF16 scale and one S32 zero-point;
- weight: symmetric S8 with either one positive finite F32/BF16 scale or
  per-channel `{N}` / `{1,N}` F32/BF16 scales, and no weight zero-point;
- destination: BF16, F32, or U8;
- U8 destination additionally supplies one positive finite F32/BF16
  destination multiplier and one S32 zero-point;
- accepted per-tensor metadata shapes are `{}`, `{1}`, and `{1,1}`;
- row-major, `transA=false`, constant raw weights, `alpha=1`, `beta=0`, and
  either no post-op, one buffer-free ReLU/sigmoid, or the exact
  `binary_mul(F32[M,N]) -> binary_add(F32[M,N])` chain are required.

Prompt-sized calls and unsupported metadata remain on the standard scalar
backend. U8 output follows the AOCL/reference SCALE convention:
`RNE(real * dst_scale + dst_zp)`, saturated to `[0,255]`.

### Example 4: INT8 MatMul with Zero-Point Compensation Caching

This example demonstrates INT8 matmul with asymmetric quantization and automatic zero-point compensation caching:

```cpp
int lowoha_int8_matmul_example() {
  using namespace zendnnl::lowoha::matmul;
  
  // Matrix dimensions (realistic sizes for LLM inference)
  constexpr int M = 32;    // Batch size / sequence length
  constexpr int K = 4096;  // Hidden dimension
  constexpr int N = 4096;  // Output dimension
  
  // Create INT8 weights (s8)
  std::vector<int8_t> weights(K * N);
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] = static_cast<int8_t>((i % 7) - 3);  // Values: -3 to 3
  }
  
  // Create U8 input (asymmetric quantization)
  std::vector<uint8_t> input(M * K);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<uint8_t>(i % 256);
  }
  
  // Output tensor
  std::vector<float> output(M * N, 0.0f);
  
  // Quantization parameters
  float src_scale_val = 0.05f;   // Source scale (per-tensor)
  float dst_scale_val = 0.1f;    // Destination scale (per-tensor)
  int32_t src_zp_val = 128;      // Source zero-point (asymmetric u8)
  
  // Weight scale: per-channel (one scale per output column)
  std::vector<float> wei_scale(N);
  for (int n = 0; n < N; ++n) {
    wei_scale[n] = 0.01f + 0.0001f * (n % 100);
  }
  
  // Configure data types for INT8
  matmul_data_types dtypes;
  dtypes.src = data_type_t::u8;   // Unsigned 8-bit activations
  dtypes.wei = data_type_t::s8;   // Signed 8-bit weights
  dtypes.dst = data_type_t::f32;  // Float32 output
  dtypes.bias = data_type_t::none;
  dtypes.compute = data_type_t::none;
  
  matmul_params params;
  params.dtypes = dtypes;
  
  // Set source scale (per-tensor)
  params.quant_params.src_scale.buff = &src_scale_val;
  params.quant_params.src_scale.dt = data_type_t::f32;
  params.quant_params.src_scale.dims = {1};
  
  // Set weight scale (per-channel)
  params.quant_params.wei_scale.buff = wei_scale.data();
  params.quant_params.wei_scale.dt = data_type_t::f32;
  params.quant_params.wei_scale.dims = {1, N};
  
  // Set destination scale (per-tensor)
  params.quant_params.dst_scale.buff = &dst_scale_val;
  params.quant_params.dst_scale.dt = data_type_t::f32;
  params.quant_params.dst_scale.dims = {1};
  
  // Set source zero-point (triggers 1D compensation caching)
  params.quant_params.src_zp.buff = &src_zp_val;
  params.quant_params.src_zp.dt = data_type_t::s32;
  params.quant_params.src_zp.dims = {1};
  
  // Note: Weight zero-point is 0 (symmetric) - no need to set
  // This results in 1D compensation which is cached
  
  // Add ReLU post-op
  matmul_post_op relu_op;
  relu_op.po_type = post_op_type_t::relu;
  relu_op.buff = nullptr;
  relu_op.dtype = data_type_t::none;
  params.postop_.push_back(relu_op);
  
  // Batch parameters
  matmul_batch_params_t batch_params;
  batch_params.Batch_A = 1;
  batch_params.Batch_B = 1;
  
  // Execute INT8 MatMul
  // First execution: computes and caches zero-point compensation
  // Subsequent executions: reuses cached compensation
  status_t status = matmul_direct(
    'r', false, false,
    M, N, K,
    1.0f,
    input.data(), K,
    weights.data(), N,
    nullptr,  // no bias
    0.0f,
    output.data(), N,
    true,  // is_weights_const (required for caching)
    batch_params,
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 5: F16 (Half-Precision) MatMul

This example demonstrates F16 MatMul where all tensors (source, weights, destination) use half-precision floating point. F16 requires AVX512-FP16.

```cpp
int lowoha_matmul_f16_example() {
  using namespace zendnnl::lowoha::matmul;

  // Matrix dimensions
  constexpr int M = 64, N = 128, K = 256;

  // F16 is represented as uint16_t (IEEE 754 half-precision format)
  std::vector<uint16_t> src(M * K);
  std::vector<uint16_t> wei(K * N);
  std::vector<uint16_t> dst(M * N, 0);

  // Initialize with simple pattern using float16_t conversion
  std::fill(src.begin(), src.end(), float16_t(1.0f).raw());
  std::fill(wei.begin(), wei.end(), float16_t(1.0f).raw());

  // Setup data types - all F16
  matmul_data_types matmul_dtype;
  matmul_dtype.src = data_type_t::f16;
  matmul_dtype.wei = data_type_t::f16;
  matmul_dtype.dst = data_type_t::f16;
  matmul_dtype.bias = data_type_t::none;
  matmul_dtype.compute = data_type_t::none;

  matmul_params params;
  params.dtypes = matmul_dtype;

  matmul_batch_params_t batch_params;
  batch_params.Batch_A = 1;
  batch_params.Batch_B = 1;

  // Execute F16 matmul
  status_t status = matmul_direct(
    'r', false, false,
    M, N, K,
    1.0f, src.data(), K,
    wei.data(), N,
    nullptr,  // no bias
    0.0f, dst.data(), N,
    true,  // is_weights_const
    batch_params, params);

  // Handle ISA-unsupported gracefully
  if (status == status_t::isa_unsupported) {
    // F16 not supported on this platform - skip gracefully
    return 0;
  }
  return (status == status_t::success) ? 0 : -1;
}
```

**Key Points for F16:**

- **ISA Requirement**: F16 requires AVX512-FP16 (CPUID leaf 7, subleaf 0, EDX bit 23), except a call that asks for the native BRGEMM, which needs AVX-512 F, BW, VL and DQ and FMA, and multiplies in F16 with AVX512-FP16 where the CPU has it.
- **Backend**: AOCL-DLP (which accumulates in F16) and OneDNN; the native BRGEMM (`ZENDNNL_MATMUL_ALGO=10`, or `lowoha_algo = matmul_algo_t::native_brgemm`), which on a CPU with AVX512-FP16 accumulates in F16 over blocks of 256 K steps and adds the blocks in F32 (`ZENDNNL_MATMUL_NATIVE_F16_ACCUM=f32` keeps F32 accumulation there), and elsewhere accumulates in F32 (see [Native BRGEMM](#native-brgemm)).

## Reorder Quantization (Source Quantization via Reorder)

Reorder quantization enables **BF16 or F32 source tensors to be quantized to S8/U8 on-the-fly** before dispatching to INT8 matmul kernels. This is useful when the model provides floating-point activations but the weights are already in INT8 format.

To enable reorder quantization, `dtypes.compute` must be set to the target quantization type (`s8` for symmetric or `u8` for asymmetric), `dynamic_quant` must be enabled, and the source type must be BF16 or F32.

### How It Works

When eligible (src is BF16/F32, weight is S8, `dtypes.compute` is S8 or U8, and `dynamic_quant` is true), `matmul_direct` automatically invokes the reorder quantization path before the matmul dispatch:

1. **Quantize source**: Converts the BF16/F32 source tensor to S8/U8 using the LOWOHA reorder engine
2. **Execute matmul**: Dispatches the quantized source with INT8 weights to the selected backend (using per-group symmetric quantization when per-group scales are provided)
3. **Free buffers**: Releases the temporary quantized source, scale, and zero-point buffers

### Source Scale Granularity

The reorder quantization path supports multiple scale granularities for the source tensor:

| Granularity | Scale Dims | Description |
|-------------|------------|-------------|
| Per-tensor  | `{1, 1}` | Single scale for entire source matrix |
| Per-token   | `{M, 1}` | One scale per row; each token is independently quantized |
| Per-group   | `{M, G}` | G groups along K dimension (G = K/group_size); finer granularity reduces quantization error |

When `transA` is true, the reorder quantization wrapper automatically transposes the scale dimensions and scale buffer data to match the physical tensor layout, then transposes back for downstream consumption.

### Quantization Modes

Currently only **dynamic quantization** is supported. Static quantization support is planned for a future release.

| Mode | `dynamic_quant` | Scale/ZP Buffers | Status |
|------|----------------|-----------------|--------|
| **Dynamic** | `true` | Optional — allocated internally if `buff` is `nullptr` | Supported |
| **Static** | `false` | User must provide pre-computed `src_scale.buff` (and `src_zp.buff` for U8) | Not yet supported |

> **Post-op restriction:** the `mish` post-op is **not supported** on the reorder/dynamic-quantization path. `matmul_direct` returns `status_t::failure` for `dynamic_quant` configurations that include `mish` in `params.postop_`, because the underlying AOCL-DLP `sym_quant` kernel does not implement `mish` and would otherwise silently drop it. A call that asks for the native BRGEMM takes it: the native BRGEMM applies `mish` in its epilogue.

### Zero-Point Handling

Zero-point behavior depends on the target quantization type (`dtypes.compute`):

| Compute Type | Quantization | Zero-Point | Formula |
|--------------|-------------|------------|---------|
| `s8` (symmetric) | Signed 8-bit | Implicitly 0 (ignored) | `int8_val = clamp(round(src / scale), -128, 127)` |
| `u8` (asymmetric) | Unsigned 8-bit | Required (`src_zp` dims/dt must be set) | `uint8_val = clamp(round(src / scale) + zp, 0, 255)` |

### Required Parameters

For reorder quantization to activate, the following must be set:

| Parameter | Requirement |
|-----------|-------------|
| `dtypes.src` | `bf16` or `f32` |
| `dtypes.wei` | `s8` |
| `dtypes.compute` | `s8` or `u8` |
| `quant_params.src_scale.dims` | Must be non-empty (e.g., `{1, 1}` for per-tensor, `{M, 1}` for per-token, `{M, G}` for per-group) |
| `quant_params.src_scale.dt` | Must be set (`f32` or `bf16`) |
| `quant_params.src_scale.buff` | Required for static mode; optional for dynamic mode |
| `quant_params.src_zp.dims` | Required only for U8 (asymmetric) |
| `quant_params.src_zp.dt` | Required only for U8 (asymmetric) |
| `quant_params.src_zp.buff` | Required for static U8; optional for dynamic U8 |

### Example 6: Dynamic Reorder Quantization (BF16 → U8)

```cpp
int lowoha_dynamic_reorder_quant_example() {
  using namespace zendnnl::lowoha::matmul;

  constexpr int M = 64, K = 256, N = 128;

  // BF16 source activations (stored as uint16_t)
  std::vector<uint16_t> input_bf16(M * K);
  // S8 weights (pre-quantized)
  std::vector<int8_t> weights_s8(K * N);
  // F32 output
  std::vector<float> output(M * N, 0.0f);
  // BF16 bias
  std::vector<uint16_t> bias_bf16(N);

  // Configure data types
  matmul_data_types dtypes;
  dtypes.src = data_type_t::bf16;
  dtypes.wei = data_type_t::s8;
  dtypes.dst = data_type_t::f32;
  dtypes.bias = data_type_t::bf16;
  dtypes.compute = data_type_t::u8;  // Asymmetric quantization target

  matmul_params params;
  params.dtypes = dtypes;
  params.dynamic_quant = true;  // Enable dynamic quantization

  // Source scale: dims and dt must be set; buff is nullptr so it will be
  // allocated internally and filled by the dynamic quantization pass
  params.quant_params.src_scale.buff = nullptr;
  params.quant_params.src_scale.dt = data_type_t::f32;
  params.quant_params.src_scale.dims = {1, 1};  // Per-tensor

  // Source zero-point: required for U8 (asymmetric)
  params.quant_params.src_zp.buff = nullptr;
  params.quant_params.src_zp.dt = data_type_t::s32;
  params.quant_params.src_zp.dims = {1, 1};  // Per-tensor

  // Weight quantization params (pre-quantized weights)
  float wei_scale_val = 0.05f;
  int32_t wei_zp_val = 0;
  params.quant_params.wei_scale.buff = &wei_scale_val;
  params.quant_params.wei_scale.dt = data_type_t::f32;
  params.quant_params.wei_scale.dims = {1, 1};
  params.quant_params.wei_zp.buff = &wei_zp_val;
  params.quant_params.wei_zp.dt = data_type_t::s32;
  params.quant_params.wei_zp.dims = {1, 1};

  matmul_batch_params_t batch_params;

  status_t status = matmul_direct(
    'r', false, false,
    M, N, K,
    1.0f,
    input_bf16.data(), K,
    weights_s8.data(), N,
    bias_bf16.data(),
    0.0f,
    output.data(), N,
    true,
    batch_params,
    params
  );

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 7: INT8 Per-Group Symmetric Quantization with Dynamic Source Quantization

This example demonstrates per-group symmetric quantization where both source and weight scales are per-group, and the BF16 source is dynamically quantized to S8 before the matmul:

```cpp
int lowoha_int8_per_group_dynamic_quant_example() {
  using namespace zendnnl::lowoha::matmul;

  constexpr int M = 32, K = 256, N = 128;
  constexpr int GROUP_SIZE = 64;
  constexpr int NUM_GROUPS = K / GROUP_SIZE;  // = 4 groups

  // BF16 source activations
  std::vector<uint16_t> input_bf16(M * K);

  // S8 weights (pre-quantized with per-group scales)
  std::vector<int8_t> weights_s8(K * N);

  // Weight scale: per-group {NUM_GROUPS, N}
  std::vector<float> wei_scale(NUM_GROUPS * N);
  for (int g = 0; g < NUM_GROUPS; ++g)
    for (int n = 0; n < N; ++n)
      wei_scale[g * N + n] = 0.02f + 0.001f * g;

  // Source scale: per-group {M, NUM_GROUPS}
  // buff is nullptr — will be computed dynamically during quantization
  // Output
  std::vector<float> output(M * N, 0.0f);

  // Configure data types
  matmul_data_types dtypes;
  dtypes.src = data_type_t::bf16;
  dtypes.wei = data_type_t::s8;
  dtypes.dst = data_type_t::f32;
  dtypes.bias = data_type_t::none;
  dtypes.compute = data_type_t::s8;  // Symmetric quantization target

  matmul_params params;
  params.dtypes = dtypes;
  params.dynamic_quant = true;

  // Source scale: per-group {M, NUM_GROUPS}
  params.quant_params.src_scale.buff = nullptr;  // Dynamic — computed at runtime
  params.quant_params.src_scale.dt = data_type_t::f32;
  params.quant_params.src_scale.dims = {M, NUM_GROUPS};

  // Weight scale: per-group {NUM_GROUPS, N}
  params.quant_params.wei_scale.buff = wei_scale.data();
  params.quant_params.wei_scale.dt = data_type_t::f32;
  params.quant_params.wei_scale.dims = {NUM_GROUPS, N};

  matmul_batch_params_t batch_params;

  status_t status = matmul_direct(
    'r', false, false,
    M, N, K,
    1.0f,
    input_bf16.data(), K,
    weights_s8.data(), N,
    nullptr,
    0.0f,
    output.data(), N,
    true,
    batch_params,
    params
  );

  return (status == status_t::success) ? 0 : -1;
}
```

**Key Points for Per-Group INT8:**

- **Source and weight groups must match**: When both use per-group scales, `K / group_size` must be the same for source dims `{M, G}` and weight dims `{G, N}`.
- **Scale data types**: Both `f32` and `bf16` are supported for scale tensors.
- **Dynamic source quantization**: When `src_scale.buff` is `nullptr` with `dynamic_quant = true`, the scale is computed from the source data at runtime using per-group statistics.
- **Weight reorder caching**: The weight reorder cache key includes the group size, so different group sizes for the same weight tensor produce separate cache entries.

### Example 8: W4A8 dynamic (bf16 activations + symmetric s4 weights)

W4A8 runs on the custom-kernel route or **AOCL DLP** with the algorithm unset, and on the native BRGEMM under `ZENDNNL_MATMUL_ALGO=10`. Dynamic entry: the library quantizes **bf16** activations to **s8** at runtime (`dynamic_quant = true`, `dtypes.compute = s8`). Packed **s4** weights use **per-group** `wei_scale` `{G, N}`; `src_scale` is `{M, 1}` or `{M, G}`. Output is **bf16** or **f32** (**f32** for an **f32** source). Scale tensors may be **f32** or **bf16**; a call on the native BRGEMM also takes **f16** weight scales. No `src_zp` / `wei_zp`. Requires **`K` divisible by `G`** and **`(K/G) % 4 == 0`**.

```cpp
int lowoha_w4a8_matmul_example() {
  using namespace zendnnl::lowoha::matmul;
  using zendnnl::common::matmul_algo_t;

  constexpr int M = 8, K = 128, N = 64;
  constexpr int NUM_GROUPS = K / 32;  // G = 4, group_size = 32

  std::vector<bfloat16_t> src_scale(M);           // per-token {M, 1} as flat M elems
  for (int m = 0; m < M; ++m) {
    src_scale[m] = bfloat16_t(0.02f + 0.001f * static_cast<float>(m));
  }

  std::vector<bfloat16_t> wei_scale(NUM_GROUPS * N);
  for (int g = 0; g < NUM_GROUPS; ++g)
    for (int n = 0; n < N; ++n)
      wei_scale[g * N + n] = bfloat16_t(1.0f + 0.05f * static_cast<float>(g));

  std::vector<int8_t> weights((K * N + 1) / 2);    // packed s4
  const int8_t nib = 1 & 0x0F;
  std::fill(weights.begin(), weights.end(),
              static_cast<int8_t>(nib | (nib << 4)));

  std::vector<uint16_t> input(M * K, 0x3F80u);     // bf16 1.0
  std::vector<uint16_t> output(M * N, 0);

  matmul_data_types dtypes;
  dtypes.src = data_type_t::bf16;
  dtypes.wei = data_type_t::s4;
  dtypes.dst = data_type_t::bf16;
  dtypes.bias = data_type_t::none;
  dtypes.compute = data_type_t::s8;

  matmul_params params;
  params.dtypes = dtypes;
  params.dynamic_quant = true;
  params.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;

  params.quant_params.src_scale.buff = src_scale.data();
  params.quant_params.src_scale.dt = data_type_t::bf16;
  params.quant_params.src_scale.dims = {M, 1};

  params.quant_params.wei_scale.buff = wei_scale.data();
  params.quant_params.wei_scale.dt = data_type_t::bf16;
  params.quant_params.wei_scale.dims = {NUM_GROUPS, N};

  matmul_batch_params_t batch_params;

  status_t status = matmul_direct(
    'r', false, false,
    M, N, K,
    1.0f,
    input.data(), K,
    weights.data(), N,
    nullptr,
    0.0f,
    output.data(), N,
    true,   // constant weights (reorder/cache)
    batch_params,
    params);

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 8b: W4A8 static (pre-quantized s8 activations + symmetric s4 weights)

Static entry: the caller passes **s8** activations and populated `src_scale.buff` / `wei_scale.buff` with `dynamic_quant = false` and `dtypes.compute = s8`. Do **not** set `dynamic_quant = true` on an s8 source (validation fails). `src_scale` dims remain **logical** `{M, 1}` or `{M, G}` even if `transA` is true.

```cpp
int lowoha_w4a8_static_matmul_example() {
  using namespace zendnnl::lowoha::matmul;
  using zendnnl::common::matmul_algo_t;

  constexpr int M = 8, K = 128, N = 64;
  constexpr int NUM_GROUPS = K / 32;  // G = 4, group_size = 32

  std::vector<bfloat16_t> src_scale(M * NUM_GROUPS);  // per-group {M, G}
  for (int m = 0; m < M; ++m)
    for (int g = 0; g < NUM_GROUPS; ++g)
      src_scale[m * NUM_GROUPS + g]
              = bfloat16_t(0.02f + 0.001f * static_cast<float>(m));

  std::vector<bfloat16_t> wei_scale(NUM_GROUPS * N);
  for (int g = 0; g < NUM_GROUPS; ++g)
    for (int n = 0; n < N; ++n)
      wei_scale[g * N + n] = bfloat16_t(1.0f + 0.05f * static_cast<float>(g));

  std::vector<int8_t> weights((K * N + 1) / 2);
  const int8_t nib = 1 & 0x0F;
  std::fill(weights.begin(), weights.end(),
              static_cast<int8_t>(nib | (nib << 4)));

  std::vector<int8_t> input(M * K, 1);
  std::vector<uint16_t> output(M * N, 0);

  matmul_data_types dtypes;
  dtypes.src = data_type_t::s8;
  dtypes.wei = data_type_t::s4;
  dtypes.dst = data_type_t::bf16;
  dtypes.bias = data_type_t::none;
  dtypes.compute = data_type_t::s8;

  matmul_params params;
  params.dtypes = dtypes;
  params.dynamic_quant = false;
  params.lowoha_algo = matmul_algo_t::aocl_dlp_blocked;

  params.quant_params.src_scale.buff = src_scale.data();
  params.quant_params.src_scale.dt = data_type_t::bf16;
  params.quant_params.src_scale.dims = {M, NUM_GROUPS};

  params.quant_params.wei_scale.buff = wei_scale.data();
  params.quant_params.wei_scale.dt = data_type_t::bf16;
  params.quant_params.wei_scale.dims = {NUM_GROUPS, N};

  matmul_batch_params_t batch_params;

  status_t status = matmul_direct(
    'r', false, false,
    M, N, K,
    1.0f,
    input.data(), K,
    weights.data(), N,
    nullptr,
    0.0f,
    output.data(), N,
    true,
    batch_params,
    params);

  return (status == status_t::success) ? 0 : -1;
}
```

**Key points for W4A8:**

- **Not WOQ**: `dynamic_quant = false` with **bf16×s4** selects weight-only quantization, not W4A8.
- **Static vs dynamic**: pre-quantized **s8×s4** requires `dynamic_quant = false` and a populated `src_scale.buff`. Dynamic **bf16×s4** requires `dynamic_quant = true`. Mixing s8 src with `dynamic_quant = true` is rejected.
- **Source scales**: logical dims `{M, 1}` or `{M, G}` (or `{1, 1}` when `M == 1`); dtype **f32** or **bf16**. `{M, 1}` is broadcast to `{M, G}` before the sym_quant GEMM.
- **Weight scales**: `wei_scale` dims `{G, N}`; dtype **f32** or **bf16** (**f16** too on the native BRGEMM).
- **Weight prepack**: WOQ s4/u4 uses the weight-only blocked layout. Native W4A8 (`s4` weights, source `s8`) uses a separate blocked layout. Widening s4 to s8 before the GEMM uses the symmetric grouped s8 layout. The native BRGEMM keeps the s4 weight as nibbles in its own layout, the raw weight's size when `K` is a multiple of 8 and `N` of 32; for a contiguous weight (`ldb` equal to its row) of that size, weight cache mode 2 writes the layout over the weight under `ZENDNNL_MATMUL_ALGO=10`.
- **Bits on the native BRGEMM**: a call with alpha 1 and beta 0 into an f32 or bf16 destination sums its groups as the custom kernel does: with a per-token (or `{1, 1}`) source scale, one FMA of each group's sum by its weight scale, then the source scale, whose product is rounded before the bias is added; with an `{M, G}` source scale, one FMA of each group's sum by the product of the group's source and weight scales (zero where the sum is zero); then the bias. A call with alpha or beta sums them as the reference kernel (`lowoha_algo = matmul_algo_t::reference`) does, and so does a call into an integer destination (with the checks off): with one group over K (`{1, N}` weight scales), alpha applies to the integer sum before the source and weight scales, and beta * C after them into an f32, bf16 or f16 destination, `((alpha * S) * src_scale) * wei_scale + beta * C`, as `matmul_direct` defines it and AOCL-DLP computes it, or before them into an integer destination, `((alpha * S + beta * C) * src_scale) * wei_scale`, as with per-channel s8 weights; with more groups, after the sum of the scaled groups, `alpha * sum + beta * C`. Without a gated activation or a post-op of a transcendental function (such as gelu), an f32 or bf16 result equals that of the custom kernel (the first order) or of the reference kernel (the second) bit for bit where that kernel serves the same call, with one exception: a per-token (or `{1, 1}`) call keeps the custom kernel's bits only without a bias. With a bias, the custom kernel may fuse the source scale's product and the bias into one FMA on some rows of a call, so its result can differ from the native BRGEMM's in the last bit on those rows.
- **Runnable examples**: `run_lowoha_matmul_w4a8_test` (dynamic) and `run_lowoha_matmul_w4a8_static_test` (static) in `examples/lowoha_matmul_example.cpp`.

## Weight Caching and Reordering

One of the key features of LowOHA MatMul is **automatic weight reordering and caching**.

### How Weight Caching Works

1. **First Execution**:
   - Weights are reordered to the optimal format for the selected backend
   - Reordered weights are stored in an LRU cache
   - Cache key is generated from weight pointer, dimensions, data type, backend, and (for per-group symmetric quantization) group size

2. **Subsequent Executions**:
   - Cache is queried using the same key
   - On cache hit: reordered weights are retrieved (fast path)
   - On cache miss: weights are reordered and cached

3. **Cache Eviction**:
   - When cache is full, least recently used weights are evicted
   - Evicted weights are freed to make room for new entries

### Clearing Weight Caches

These APIs flush weight caches for the **AOCL-DLP, oneDNN, native, and grp_matmul** backends. Call them only in a quiescent window: no in-flight `matmul_direct` (or `group_matmul_direct`) and no thread holding cached weight pointers. They are not safe inside an OpenMP parallel region.

| API | What it clears |
|-----|----------------|
| `clear_matmul_weight_caches()` | Preferred entry: AOCL, oneDNN, native, and grp_matmul caches, plus the **calling thread's** AOCL post-op metadata |
| `clear_matmul_aocl_weight_caches()` | AOCL-DLP typed LRUs, symquant, WOQ, W4A8, GGML unpack, and zero-point compensation. No-op when built without AOCL-DLP (`ZENDNNL_DEPENDS_AOCLDLP=0`); GGML unpack is not dropped in that stub |
| `clear_matmul_aocl_postop_metadata_cache()` | Calling thread's AOCL post-op metadata LRU only. No-op without AOCL-DLP |
| `clear_matmul_onednn_weight_caches()` | oneDNN blocked weight cache. No-op when built without oneDNN (`ZENDNNL_DEPENDS_ONEDNN=0`) |
| `clear_matmul_native_weight_caches()` | Native FP32/BF16/F16/INT8 prepacked weight caches, and the buffers of widened F16 sources kept for later calls |
| `clear_grp_matmul_weight_caches()` | grp_matmul pack arenas (BF16, DQ-INT8, FP16, W4A8/S4), the N-tile f32 weight-scale memo, and the prepack fingerprint (including AUTO mixed-in-place warm latches). Under `ZENDNNL_MATMUL_WEIGHT_CACHE=2` this also drops in-place pack sentinels so the next call repacks |

They do **not** clear routed-MoE packed weights (`group_matmul_routed_moe_flush_weight_cache()`) or LibXSMM blocked weights. AOCL/oneDNN/native LRUs are process-wide, so a call can also drop entries that `group_matmul_direct` warmed on those same backends. The prepack fingerprint is cleared with the grp_matmul arenas so a later call, including SwiGLU-OAI on ALGO 3, re-warms instead of skipping on a stale "already warmed" record.

Declarations: `zendnnl::lowoha::matmul` in `lowoha_operators/matmul/lowoha_matmul.hpp`.

## Zero-Point Compensation Caching (INT8)

For INT8 MatMul with asymmetric quantization, LowOHA provides **automatic caching of 1D zero-point compensation**:

### How ZP Compensation Caching Works

1. **1D Compensation (src_zp only)**:
   - Compensation depends only on weight column sums: `comp[n] = -src_zp × Σ(B[k,n])`
   - Computed once on first execution and stored in LRU cache
   - Subsequent inferences reuse cached compensation

2. **2D Compensation (wei_zp ≠ 0)**:
   - Compensation depends on source row sums (changes per inference)
   - Cannot be cached; recomputed every execution

### Cache Configuration

| Environment Variable | Values | Description |
|---------------------|--------|-------------|
| `ZENDNNL_ZP_COMP_CACHE` | `1` (default) | Enable ZP compensation caching |
| `ZENDNNL_ZP_COMP_CACHE` | `0` | Disable ZP compensation caching |

## Backend Selection

LowOHA MatMul supports multiple backends. A backend (kernel) can be selected either programmatically via `matmul_params` or at runtime via an environment variable.

### Supported Algorithms

The same algorithms can be selected by either the `matmul_algo_t` enum value (in code) or the `ZENDNNL_MATMUL_ALGO` environment variable value (at runtime).

| `ZENDNNL_MATMUL_ALGO` | `matmul_algo_t` enum | Description |
|-----------------------|----------------------|-------------|
| `auto`                | `auto_tuner`         | Auto Tuner — selects the most performant backend at runtime |
| `0`                   | `dynamic_dispatch`   | Automatic backend selection based on heuristics |
| `1`                   | `aocl_dlp_blocked`   | Blocked AOCL-DLP backend (default for single MatMul) |
| `2`                   | `onednn_blocked`     | Blocked OneDNN backend |
| `3`                   | `libxsmm_blocked`    | Blocked LibXSMM backend |
| `4`                   | `aocl_dlp`           | AOCL-DLP backend |
| `5`                   | `onednn`             | OneDNN backend |
| `6`                   | `libxsmm`            | LibXSMM backend (initial batched default when LIBXSMM is built) |
| `7`                   | `batched_sgemm`      | AOCL-DLP's batch GEMM for batched calls (`ZENDNNL_BMM_ALGO`); a single call runs on AOCL-DLP |
| `8`                   | `auto_tuner`         | Auto Tuner, as `auto` |
| `9`                   | `reference`          | Reference kernel |
| `10`                  | `native_brgemm`      | Native BRGEMM over the native VNNI weight layout: serves the calls listed under [Native BRGEMM](#native-brgemm); another call takes the default route, or returns an error when `lowoha_algo` names the native BRGEMM |
| `11`                  | `moe_custom_kernel`  | Group matmul's MoE custom kernel. A weight packed for it (`reorder_direct` with `prepack.algo = moe_custom_kernel`) is read only by group matmul's custom-kernel route: a single call on such a weight returns `status_t::failure`, and a single call on any other weight runs on AOCL-DLP |

> **Note:** When `params.lowoha_algo` is `none` and the applicable runtime setting (`ZENDNNL_MATMUL_ALGO` for single MatMul or `ZENDNNL_BMM_ALGO` for batched MatMul) is unset, single MatMul falls back to `aocl_dlp_blocked`. Batched MatMul initially selects `libxsmm`, but uses it only in builds with `ZENDNNL_DEPENDS_LIBXSMM=ON`; otherwise it falls back to `aocl_dlp`, or to `reference` when AOCL-DLP is also unavailable. The default LIBXSMM-disabled build therefore runs LowOHA BMM with AOCL-DLP.

### Native BRGEMM

A call asks for the native BRGEMM with `params.lowoha_algo = matmul_algo_t::native_brgemm`, or, for a single (not batched) call, with `lowoha_algo` left `none` under `ZENDNNL_MATMUL_ALGO=10`, which selects the kernel of single calls: a batched call with `lowoha_algo` left `none` takes its kernel from `ZENDNNL_BMM_ALGO`. Dynamic dispatch (`0`) and the auto tuner never select it. The native BRGEMM runs a call that names it or returns an error: no other backend runs it. A call that `ZENDNNL_MATMUL_ALGO=10` brings to it and that it does not implement runs on the default route instead (see the end of this section).

It serves a single (not batched), row-major call with a plain source (`mem_format_a = 'n'`) that has:
- bf16 × bf16, u8 / s8 × s8 (or a bf16 / f32 source the call quantizes to u8 / s8: `dynamic_quant` with `dtypes.compute` u8 / s8), s8 × s4 (an s8 source, given or quantized by the call from bf16 / f32, with symmetric s4 weights: W4A8), f32 × f32 or f16 × f16 data types, into an f32, bf16, f16, u8, s8 or s32 dst (f32 or bf16 for f32 × f32 and for s4 weights with the checks on, f16 or f32 for f16 × f16), with a bias of f32, bf16, f16, s8, u8 or s32;
- a raw weight (`mem_format_b = 'n'`), an AOCL-DLP blocked prepack of a bf16 × bf16 or static u8 × s8 call, which it unreorders (in a build with AOCL-DLP, with no `ZENDNNL_DLP_M_HINT`, under `ZENDNNL_MATMUL_ALGO=10` only), or a buffer the native BRGEMM rewrote in place;
- N from 1 to 2147483616, the largest multiple of 32 an int holds, K ≥ 1 and ldb at least the weight row (a weight whose N is not a multiple of 32 is packed with zero columns up to whole 32-column blocks, about 32 / N times the weight's bytes for N below 32, never in place, and only its N columns are read and written); for bf16 weights, K at most 67108864 (2^26), whose K pairs the bf16 kernels offset in an int, and at M above 1 a source whose rows those kernels offset in an int too, (min(M, 12) - 1) * lda + K at most 2147483647 (a transposed source is copied into rows K apart); for f16 weights, K at most 2147483632, whose steps of 16 an int holds; for s8 and s4 weights, K within the int32 accumulation bound (for s4 weights, each group within it and K at most 2147483640, whose whole K octets an int holds) and per-group weight scales over groups of whole K quads (s4 weights always take `{G, N}` weight scales, `{1, N}` for one group over K). Every call reads the weight's one layout: an M = 1 call on the GEMV kernels of the native BRGEMM (below), every other M on the native tiles (the `[MATMUL.NATIVE_BRGEMM]` line names the GEMV kernels of an M = 1 call in its `gemv` field and shows `gemv=none` on the tiles). An M = 1 call whose layout is at most a quarter of the L2 runs on the calling thread (`team=1` on that line), as one core reads such a layout in less time than the call's team takes to start;
- for INT8 and W4A8, no weight zero point (and for s4 weights no u8 source and no source zero point, also with a source the call quantizes), and no source zero point with a source the call quantizes to s8; f32, bf16 or f16 scales per tensor or per row for the source, per tensor, per column or per group for the weight (for s4 weights `{G, N}` only, `{1, N}` for one group over K); s32, s8 or u8 source zero points per tensor or per row (with per-group weight scales, one per tensor and only with a u8 source); an `{M, G}` source scale with an s8 source and no zero point;
- the post-ops and gated activations of `matmul_direct`, except a softmax with a gated activation, and beta with a gated activation only over a dst of at least N columns.

In short, under `ZENDNNL_MATMUL_ALGO=10` (the lists below give every form):

| Calls | Where they run |
|---|---|
| bf16, DA8W8 (u8 / s8 sources, given or quantized from bf16 / f32), DA8W4 (s8 sources over symmetric s4 weights with `{G, N}` scales), f32 and f16 single row-major calls on raw weights, AOCL-DLP blocked prepacks of bf16 × bf16 or static u8 × s8 calls (in a build with AOCL-DLP, with no DLP M hint) or images of the native BRGEMM, any M, N and K within the bounds above, with the post-ops, gated activations, biases, alpha and beta of `matmul_direct` | the native BRGEMM |
| batched calls (on their `ZENDNNL_BMM_ALGO` kernel) and group matmul; GGML packed weights; an s8 weight's zero point of one value over a static u8 / s8 source; W4A16 (s4 or u4 weights) with per-column f32 or bf16 weight scales and no s4 weight zero point (a u4 zero point included); AOCL-DLP blocked prepacks of other calls (other types, or under a DLP M hint) and other prepack formats; N or K past the bounds above; a CPU without the instruction set of the call's types; a column-major call of float operands on a weight that changes; a softmax with a gated activation | the default route, as with the algorithm unset |
| the forms the default route computes wrongly or has no kernel for (W8A16, W4A16 with s4 or u4 weights and per-group or f16 weight scales, W4A16 with an s4 weight's zero point, a column-major call on a constant weight or with integer operands, a reordered source, mixed float types, `gelu_tanh_and_mul`, the other forms listed below) | keep the native BRGEMM's error, dst untouched |

**One layout per weight.** The native layout of a weight is made from the weight alone: its type, its bytes, K, N, ldb, transB and the format of its buffer (raw, or an AOCL-DLP blocked prepack the native BRGEMM unreorders), in blocks of 32 columns. Nothing of a call (M, the post-ops or gated activation, the thread count, the weight cache mode, the route) changes it, and calls in any order share it. The weight cache mode decides only where it lives:
- mode 0 (or a weight that is not constant): packed for each call, M = 1 included, and freed when it returns; a weight buffer that takes a new weight between calls gives the new weight's result;
- mode 1: cached for the weight pointer, next to the raw weight, and packed again from the buffer's new bytes when the two ends of the bytes it was packed from change;
- mode 2: as mode 1, except that under `ZENDNNL_MATMUL_ALGO=10` the first call rewrites a bf16, f16 or s4 weight (or an AOCL-DLP blocked prepack it unreorders) in place when its layout fits the buffer (a contiguous ldb and N a multiple of 32; bf16 with K even, s4 with K a multiple of 8), and every later call, of any M or post-op, reads that image. With `lowoha_algo = native_brgemm` alone mode 2 caches as mode 1.

**M = 1 calls.** An M = 1 call runs on the GEMV kernels of the native BRGEMM, which read the weight's layout as it is (the layout cached at weight cache mode 1, the image at mode 2, the pack of the call at mode 0), with no other pack or copy of the weight: one kernel per type (`gemv=bf16`, `f32`, `f16`, `f16_ph`, `s8` where the source scales are per tensor or per row and the weight scales per tensor or per column, `s8_group` where the source or the weight scales are per group, `s4`) reads a few 32-column blocks side by side, each a stream of its own, all fed by one load of the source per K step. Each output column takes the steps of the tile kernel of its type in K order (the dot products of K pairs of bf16, the fused multiply-adds of f32 and of widened f16, the f16 multiply-adds in blocks of 256 K steps, the exact int32 sums of INT8 and s4 with the tiles' dequantization and group steps), and the epilogue is the tiles' (alpha and beta, every bias type, post-ops, gated activations, requantization, the dst types), so an M = 1 call gives the bits the tiles give it. The columns go to the threads of the call's plan in whole blocks (a gate block of a gated call with its up block, or runs of 16 columns where the halves are not whole blocks), each block over all of K, the threads spread over the team when there are fewer blocks than threads. Each thread reads two blocks at a time, each continuing the stream of the block it read before, or four where its share of a bf16, f32 or f16 layout fits half its L2, which gives the multiply-adds more accumulators to hide their latency.

The earlier GEMV kernels of the native BRGEMM (narrow bf16, the bf16 and INT8 BKC kernels), with their own packs and caches, stay in the library, called only by their tests.

**Gated activations.** The native layout of a weight depends on the weight alone, not on the call: a gated call reads the layout a plain call on the same weight cached or wrote in place, and the two share it, so a weight a first call rewrote in place (weight cache mode 2) serves later plain and gated calls alike. `silu_and_mul` and `gelu_and_mul` take the `[gate | up]` halves of the weight's N columns, gate column i paired with up column N / 2 + i. With bf16, s8 or s4 weights and N a multiple of 32, each tile computes gate columns and their up columns side by side and pairs them. Tiles of 64 columns or more hold whole 32-column blocks of each half; a plan of narrower tiles runs tiles of 64 columns where those deal out evenly over the team (or number at least 8 per thread), and otherwise tiles of 32 columns, a run of 16 gate columns and its 16 up columns, whose kernel calls read the up run at a second address, keeping the tile count the plan was balanced for; a plan of 16-column tiles runs half as many, in row blocks of fewer rows where the team would otherwise have fewer units than threads. With any other even N, and for f32 and f16 weights, the call stages whole rows of its pre-activation in an M × N f32 buffer of the calling thread, kept across its calls up to 4 MiB; rows a call needs above that serve the thread's calls that need them, and its next staged call that fits in 4 MiB frees them. A check of the call (`native_brgemm_refusal`) allocates none. The threads of the tiles' parallel region pair the halves of all rows once every tile is done. The `[MATMUL.NATIVE_BRGEMM]` line of a call over the halves ends with `pairing=` and the tile width at which its tiles paired them, or `pairing=staged`. `swiglu_oai_mul` takes interleaved pairs, gate column 2i and up column 2i + 1. Either way alpha, beta, the bias and the post-ops act on the N pre-activation columns, then the activation, as `matmul_direct` defines it. `gelu_tanh_and_mul` is `gelu_and_mul` with GELU's tanh approximation, `0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))` (the `gelu_tanh` post-op's), over the same halves; `gelu_and_mul` is GELU's erf form on every route. Only the native BRGEMM serves `gelu_tanh_and_mul`: on any other route the call returns `status_t::unimplemented`, with dst untouched and one error line.

`matmul_direct` checks its inputs before any route runs, unless `ZENDNNL_DIAGNOSTICS_ENABLE=0` turns the checks off, and returns `status_t::failure` with an error line for a call they reject. With the checks on, four parts of the list above are narrower:
- source scales: per tensor, or per row or `{M, G}` only with an s8 source or a bf16 / f32 source the call quantizes to s8, no source zero point, an f32, bf16, u8, s8 or s32 dst (not f16), and weight scales per column or per group (`G × N` of them with an `{M, G}` source scale);
- W4A8 source scales: f32, bf16 or f16, per row or `{M, G}` (`{1, 1}` only at M = 1);
- source zero points: per tensor only;
- data types, by the type of the source whatever the weight (a bf16 / f32 source the call quantizes included): no f32 source into a bf16 or f16 dst, no bf16 source into an f16 dst (but for a bf16 × bf16 call that asks for the native BRGEMM, below), and no f16 source into a bf16 dst.

With the checks on they also refuse, with `status_t::failure` before any route, an f16 source the call quantizes for s4 weights, an f16 call with a weight zero point, and an f16 call into a bf16 dst; with the checks off the native BRGEMM refuses the first with `status_t::unimplemented`.

The native BRGEMM serves the whole list only with the checks off. On every other route an f16 source, weight, dst, bias or binary operand needs AVX512-FP16, which `matmul_direct` checks before the checks above: without it such a call returns `status_t::isa_unsupported`. A call that asks for the native BRGEMM takes them without AVX512-FP16: there the native BRGEMM widens f16 values to f32 and rounds f32 results to f16 with AVX-512 F.

An f16 × f16 call keeps the weight in f16 in the native layout (per block of 32 columns, K rows of 32 values: the raw weight's size when N is a multiple of 32, so weight cache mode 2 rewrites a contiguous weight in place under `ZENDNNL_MATMUL_ALGO=10`, as a bf16 one) and runs on one of two kernel families over that layout, which the CPU and `ZENDNNL_MATMUL_NATIVE_F16_ACCUM` pick once per process; the `[MATMUL.NATIVE_BRGEMM]` line of the call names them (`kernels=f16_ph` or `kernels=f16`):

- On a CPU with AVX512-FP16, in a build whose compiler has the AVX512-FP16 intrinsics, with `ZENDNNL_MATMUL_NATIVE_F16_ACCUM` unset or `f16` (the default), the f16_ph kernels read the call's f16 source rows as they are (a transposed source is copied once into rows K apart) and multiply-add in f16, 32 lanes to an instruction. From M 192 (the `f16_wide_min_m` key of `ZENDNNL_MATMUL_NATIVE_CONFIG`), where N is a multiple of 64, a tile spans 64 columns, two 32-column blocks whose multiply-adds share each broadcast of a source element; below M 192, for other N, and where the team would otherwise have fewer tiles than threads, the tiles span 32 or 16 columns. Each output accumulates in f16 over blocks of 256 K steps from the start of K, and each block's sum, widened to f32 exactly, is added in K order to the output's f32 sum. f16 has 10 mantissa bits: a running f16 sum over a long K loses the products that fall below its last bit (a sum of ones stops at 2048), and the blocks bound that loss by one block whatever K. A block of k products (k ≤ 256) is off by at most k·2⁻¹¹ / (1 − k·2⁻¹¹) of the sum of their magnitudes, 0.143 at 256 (a worst case; rounding errors of random sign mostly cancel), and the f32 additions of the blocks add about 2⁻²⁴ of the magnitudes per block. The bound holds while a block's partial sums stay in the normal range of f16, 2⁻¹⁴ to 65504 in magnitude. Above it a partial sum overflows to infinity. Below it each f16 step is off by up to 2⁻²⁵ absolutely, so a block of tiny products can lose them all: 256 products of 2⁻¹³ × 2⁻¹³ give 0 instead of 2⁻¹⁸. `ZENDNNL_MATMUL_NATIVE_F16_ACCUM=f32` sums such products in f32, where only the f16 dst then rounds. The K passes of every plan are whole blocks and every tile width takes the same steps for an output, so the bits do not depend on the plan, the threads or the M split. These calls widen nothing and keep no buffer.
- On a CPU without AVX512-FP16, in a build without the intrinsics, or with `ZENDNNL_MATMUL_NATIVE_F16_ACCUM=f32`, the f16 kernels widen the source to f32 once per call (into a buffer of M × K floats; the process keeps such buffers for later calls, at most 32 of them and 512 MiB over all threads, and `clear_matmul_native_weight_caches()` frees them) and add the products of the widened values in K order, one f32 fused multiply-add each, whatever the plan: the error of an f32 sum, at half the multiply-adds per instruction of the f16_ph kernels.

Either way alpha, beta, the bias, the post-ops and an f16 dst's rounding to nearest even follow, and M = 1 calls run on the GEMV kernels with the same steps. The AOCL-DLP route accumulates F16 in F16 too, over K blocks of its own (its measured error stops growing from K 2048 and stays above the native BRGEMM's), so its results differ from the native BRGEMM's.

The checks also take, for a call that asks for the native BRGEMM, the `mish` post-op on the INT8 sym-quant, dynamically quantized and W4A8 calls that the other routes refuse it on (not on GGML packed weights), and an f16 dst of a bf16 × bf16 call (not one the call quantizes), which the native BRGEMM rounds from its f32 results, on a CPU without AVX512-FP16 too.

Any other call with `lowoha_algo = native_brgemm` returns, with dst untouched and one error line naming the reason and that no other route takes the call:
- `status_t::unimplemented` for what the native BRGEMM does not implement, for example weight zero points, u4 weights, s4 weights with a u8 source, a source zero point or weight scales other than `{G, N}`, a source the call quantizes to s8 with a source zero point, an f16 source for s8 weights, an f16 call into a dst other than f16 or f32, batched calls, column-major calls, reordered sources, GGML packed weights, a softmax with a gated activation, an f16 dst with per-group weight scales, or an s8 source with a zero point on per-group weight scales;
- `status_t::isa_unsupported` on a CPU without the instruction set its data types need (AVX-512 F, BW, VL and DQ and FMA, with AVX512-BF16 for bf16 and AVX512-VNNI for INT8);
- `status_t::failure` for a weight buffer only another route reads (a custom-kernel image), for a call the native BRGEMM does not serve on a buffer it rewrote in place, which no other route reads, and for wrong arguments (a bad shape, leading dimension or buffer).

A single call without `lowoha_algo` under `ZENDNNL_MATMUL_ALGO=10` that the native BRGEMM refuses with `status_t::unimplemented` or `status_t::isa_unsupported` runs on the default route instead, as with the algorithm unset:
- on a CPU without AVX512-FP16, a call with an f16 source, weight, dst, bias or binary operand, which only the native BRGEMM takes there, returns `status_t::isa_unsupported` before any other check, with dst untouched and the error line of the default route's AVX512-FP16 check, as with the algorithm unset;
- `matmul_direct` runs the default route's input checks on it first: a call the checks take only for the native BRGEMM (f16 W4A8 scales, `mish` on a quantized call, a u8 / s8 / s32 dst of a per-token or per-group call) fails there with their error line, before any route reads it;
- the call caches its weight out of place (weight cache mode 1 at most), as a later call on the buffer may read the raw weight on the native BRGEMM;
- whatever the process's weight cache mode is now (an image written in place, and a native call that took mode 2, outlive a downgrade of the mode), the call holds the bytes of its raw weight of any type, as it lays the weight out (a column-major `[K, N]` weight spans `(N - 1) * ldb + K` elements, two to a byte for u4 and s4), until it returns: a concurrent native call that would rewrite the buffer in place waits for it, and if such a rewrite ran after the native BRGEMM looked at the call, the call returns `status_t::failure` with one error line instead, as no other route reads a buffer the native BRGEMM rewrote;
- one warning line per process for each reason says so: `matmul_direct: under ZENDNNL_MATMUL_ALGO=10 a call the native BRGEMM does not serve (<reason>) takes the default route (logged once per reason)`. A batched call without `lowoha_algo` logs the line for `a batched call`, and holds every weight it reads as such a call holds its weight: its `Batch_B` weights, `batch_stride_wei` elements apart (by default a weight's rows apart); one of them holding an image the native BRGEMM wrote in place fails the call with `status_t::failure` and one error line, with dst untouched.

Such a call keeps its error instead when its arguments are wrong (N or K below 1, an ldb below a weight row as the call lays the weight out, K elements for a row-major `[N, K]` weight (transB) or a column-major `[K, N]` one and N otherwise, a bad shape, leading dimension or buffer), when its buffer holds an image the native BRGEMM wrote in place (`status_t::failure`, the error line saying no other route reads it) or a custom-kernel image, and when the native BRGEMM fails after it took the call (`status_t::failure`). The default route serves the calls it takes as it does with the algorithm unset, with that route's own limits.

It also keeps the native BRGEMM's error (`status_t::unimplemented`, or `status_t::isa_unsupported`), with dst untouched, for the forms the default route returns success for with a dst that is wrong or not written, and for an activation it does not implement:
- a column-major call on a constant weight, or with an s8 / s4 / u4 weight or a u8 / s8 source;
- a reordered source (`mem_format_a = 'r'`);
- W8A16 (an s8 weight with a bf16, f32 or f16 source the call does not quantize);
- W4A16 (an s4 or u4 weight with a source the call does not quantize) with per-group `{G, N}` or f16 weight scales;
- a call of a float source the call does not quantize into an s32, s8 or u8 dst (with a float, s4 or u4 weight);
- source, weight and dst types the default route has no kernel for: a float source and weight of two types (a bf16 source with an f32 weight, for example), a bf16 × bf16 call into an f16 dst, a u8 / s8 source with a float or u4 weight, and an s4 weight with an f32 or u8 source;
- a source the call quantizes to s8 with a source zero point, or to u8 with a zero point per row;
- a weight zero point of an s4 weight (W4A16), or of an s8 weight with more than one value or over a source the call quantizes, and zero points with an alpha other than 1 over an s8 weight;
- f16 weight or source scales over an s8 weight, f32 weight scales with bf16 source scales, and bf16 weight scales with f32 source scales over an s4 weight (W4A8);
- `gelu_tanh_and_mul`, which only the native BRGEMM implements;
- in a build without AOCL-DLP, what the default route of that build, the reference kernel, does not compute: a column-major call, a reordered (`mem_format_b = 'r'`) or GGML packed weight, per-group weight scales of an s8 weight without source scales of the same groups, a binary operand that is neither `{1, N}` nor rows of N values, and a pooling post-op.

Its error line says why: `matmul_direct: native BRGEMM did not serve the call (<reason>); not handed back: the default route does not compute <form> correctly` (`...; not handed back: the default route does not compute a call of these source, weight and dst types` for the types it has no kernel for, `...; not handed back: the default route does not implement gelu_tanh_and_mul`, `...; not handed back: the default route of a build without AOCL-DLP does not compute <form>` for the reference kernel's forms), and no default-route line is logged.

Under `ZENDNNL_MATMUL_ALGO=10` a call on an AOCL-DLP blocked prepack whose `lowoha_algo` names AOCL-DLP blocked keeps its AOCL-DLP route when the native BRGEMM does not serve it. Group matmul never reaches the native BRGEMM: under `ZENDNNL_MATMUL_ALGO=10`, and for an op whose `lowoha_algo` is `native_brgemm`, it runs as with the algorithm unset.

### AOCL-DLP route

With `lowoha_algo` left `none` and the algorithm unset, a single call runs on `aocl_dlp_blocked`. `aocl_dlp`, `aocl_dlp_blocked`, and every other kernel except the reference kernel and oneDNN pass the calls they do not serve to AOCL-DLP. On that route:
- **Post-ops the chain does not map.** The AOCL-DLP chain applies relu, leaky relu, gelu tanh, gelu erf, sigmoid, swish, tanh, clip, mish, binary add and binary mul. A call whose chain holds elu, square, abs, sqrt, exp, log or softmax runs on the reference kernel instead, with one info line, and gets the results of the same call with `lowoha_algo = matmul_algo_t::reference`. It returns `status_t::unimplemented` where the reference kernel does not serve it: a column-major call, AOCL-prepacked or GGML packed weights, per-group weight scales of an S8 or S4 weight without a source scale of the same groups, a BF16 / F32 source with S8 weights and no source scale, or a binary operand that is neither `{1, N}` nor rows of N values. The reference kernel reads the weight buffer as given: every call on one weight buffer must take the same route, since at weight cache mode 2 AOCL-DLP may rewrite a constant weight in place.
- **Refused post-ops.** A `pooling` post-op returns `status_t::unimplemented`, and so does a gated activation with any of the post-ops above.
- **Weight scales.** With s8 weights, one weight scale value or one per column runs on any wiring. Per-group weight scales `{G, N}` run only on the S8 rows of the INT8 pair table with F32 or BF16 output, with no zero point, a group size `K / G` that is a multiple of 4 and `G × N` values. Those rows run on the sym-quant kernel, which reads F32 weight scales, or weight scales of the source scale's type. Any other weight scale returns `status_t::unimplemented`.
- **INT8 alpha and beta.** AOCL-DLP's GEMMs with s8 weights, or with s4 weights and a u8 / s8 source, take alpha and beta as int32. An alpha or beta that is not an integer int32 holds returns `status_t::unimplemented`. A nonzero beta with s8 weights is applied in two places. For a u8 / s8 source with one source scale value into an S32, S8 or U8 destination, beta * C is added to the int32 sum before the scales. On the sym-quant kernel (S8 source, no zero point, F32 or BF16 destination), it is added after the scales, with per-token source scales (`{1, 1}` at M = 1 included), per-group source scales or per-group weight scales. Any other nonzero beta with s8 weights returns `status_t::unimplemented`: for example into an F32, BF16 or F16 destination with one source scale value and per-tensor or per-channel weight scales, or with a BF16 / F32 source the call does not quantize. The reference kernel and the native BRGEMM place beta * C as these kernels do: after the scales into an F32, BF16 or F16 destination, `alpha * (A x B) + beta * C`. Into an S32, S8 or U8 destination they add it to the integer sum before the scales, `(alpha * S + beta * C) * src_scale * wei_scale`, when the source scale and the weight scale have one group over K each (per tensor or per token, over per-tensor or per-channel weight scales); with `{G, N}` weight scales it follows the sum of the scaled groups, `alpha * sum + beta * C`, and with `{M, G}` source scales over per-tensor or per-channel weight scales it precedes the weight scale only, `(alpha * sum + beta * C) * wei_scale`, where sum adds each group's integer sum times its source scale.
- **Auto tuner.** Under `ZENDNNL_MATMUL_ALGO=auto` (8) these rules apply to the calls that run on AOCL-DLP. When every candidate of `ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES` is oneDNN or the reference kernel, no rule applies. Otherwise a call with a post-op the chain does not map runs on the reference kernel as above, and a call AOCL-DLP refuses for its weight scales, alpha or beta runs on the first candidate off AOCL-DLP, or is refused when there is none.

A refused call leaves dst untouched and writes one error line. The one-op group route, which runs gated activations and the calls `single_matmul_group_routable` admits, applies the weight-scale count rule, the int32 alpha and beta rule, and the post-op rule of gated activations. `group_matmul_direct` refuses an operation with a post-op the chain does not map, or pooling, unless the operation runs on oneDNN. oneDNN and the reference kernel keep their own handling of these calls.

### Via `matmul_params`

Set the `lowoha_algo` field on the `matmul_params` struct using the `matmul_algo_t` enum:

```cpp
matmul_params params;
params.lowoha_algo = matmul_algo_t::aocl_dlp;
```

### Via Environment Variable

Export `ZENDNNL_MATMUL_ALGO` before running the application using the value from the table above:

```bash
export ZENDNNL_MATMUL_ALGO=1   # selects aocl_dlp_blocked
```
