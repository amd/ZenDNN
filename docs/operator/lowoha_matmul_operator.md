
(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA MatMul Operator

## Overview

The **LowOHA MatMul Operator** is a high-performance, low-overhead matrix multiplication operator designed for **latency-sensitive inference workloads**. It provides a direct API to backend libraries (AOCL, LibXSMM, OneDNN) with built-in weight caching, and fused post-operations.

Unlike the standard MatMul operator which uses the operator factory pattern, LowOHA MatMul provides a **function-based interface** optimized for:
- Minimal execution overhead
- Repeated weight reuse
- Backend-native post-operation fusion
- Direct control over execution parameters


# General MatMul Operation

Let:

- *A* ∈ ℝ<sup>mxk</sup> or ℝ<sup>bsxmxk</sup>: Input Matrix or Batched Input Matrix
- *B* ∈ ℝ<sup>kxn</sup> or ℝ<sup>bsxkxn</sup>: Weight Matrix or Batched Weight Matrix
- *C* ∈ ℝ<sup>mxn</sup> or ℝ<sup>bsxmxn</sup>: Output Matrix or Batched Output Matrix
- *Bias* ∈ ℝ¹ˣᴺ: Optional Bias vector
- *Alpha* (*α*): Scaling factor for the matrix product
- *Beta* (*β*): Scaling factor for the output (for accumulation)
- *Post_ops(x)*: Optional activation or binary post operations (Example: ReLU, GELU, Binary_add, etc.)
- *Transpose(A, B)*: Optional transpose operation on matrices A or B

The computation can be expressed as:

$$
C = \text{PostOps}(\alpha \cdot (A \cdot B) + \text{Bias} + \beta \cdot C)
$$



## Core API: `matmul_direct`

The primary interface for LowOHA MatMul is the `matmul_direct` function:

```cpp
status_t matmul_direct(
  const char layout,           // 'r' for row-major,
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
  matmul_data_types dtypes;                       // Data types for tensors
  std::vector<matmul_post_op> postop_;             // Post-operations
  matmul_quantization_params_t quant_params; // Quantization parameters
  char mem_format_a;                       // Memory format for A ('n'=non-reordered, 'r'=reordered)
  char mem_format_b;                       // Memory format for B ('n'=non-reordered, 'r'=reordered)
  matmul_algo_t lowoha_algo;               // Preferred backend algorithm
  uint64_t num_threads;                    // Number of threads (0 = auto)
};
```


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

| Src Type | Weight Type | Bias Type | dst Type | Notes |
|----------|-------------|-----------|-------------|-------|
| FP32 | FP32 | FP32 | FP32 | Standard floating-point |
| BF16 | BF16 | FP32/BF16 | FP32/BF16 | Mixed-precision BFloat16 |
| BF16 | S4 | FP32/BF16 | FP32/BF16 | Weight-Only Quantization (WOQ) |
| U8 | S8 | FP32/BF16/S8/U8/S32 | FP32/BF16/S8/U8/S32 | INT8 Quantization |
| S8 | S8 | FP32/BF16/S8/U8/S32 | FP32/BF16/S8/U8/S32 | INT8 Quantization |


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
| `post_op_type_t::binary_add` | Element-wise Add | Yes |
| `post_op_type_t::binary_mul` | Element-wise Multiply | Yes |


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
    data_type_t dt;                // Data type (f32/bf16 for scale, s8/u8/s32 for zero-point)
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

**WOQ Quantization Granularity:**

| Granularity | Scale Dims | Zero-Point Dims | Description |
|-------------|------------|-----------------|-------------|
| Per-tensor | `{1}` or `{1, 1}` | `{1}` or `{1, 1}` | Single scale/zp for entire weight matrix |
| Per-channel | `{1, N}` | `{1, N}` | One scale/zp per output channel |
| Per-group | `{G, N}` | `{G, N}` | G groups along K dimension (G = K/group_size) |

**Note:** WOQ requires `is_weights_const = true` for weight reordering and caching.


## Usage Examples

### Example 1: BF16 MatMul with Multiple Post-Ops

```cpp
int lowoha_matmul_bf16_fused_ops_example() {
  using namespace zendnnl::lowoha;
  
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
  using namespace zendnnl::lowoha;
  
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

### Example 3: WOQ MatMul with BF16 Input and S4 Weights

This example demonstrates Weight-Only Quantization (WOQ) with per-group quantization, commonly used in LLM inference:

```cpp
int lowoha_woq_bf16s4_matmul_example() {
  using namespace zendnnl::lowoha;
  
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
  
  // Create zero point tensor (per-group: {NUM_GROUPS, N})
  std::vector<int8_t> wei_zp(NUM_GROUPS * N);
  for (int g = 0; g < NUM_GROUPS; ++g) {
    for (int n = 0; n < N; ++n) {
      wei_zp[g * N + n] = static_cast<int8_t>(g % 4);  // zp = 0, 1, 2, 3
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
  
  params.quant_params.wei_zp.buff = wei_zp.data();
  params.quant_params.wei_zp.dt = data_type_t::s8;
  params.quant_params.wei_zp.dims = {NUM_GROUPS, N};
  
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
- **S4 Weight Packing**: Weights are stored as 4-bit signed integers, with 2 values packed per byte (low nibble and high nibble).
- **Quantization Granularity**: Per-group quantization divides K dimension into groups, each with its own scale/zero-point.
- **Constant Weights**: `is_weights_const = true` is required for WOQ to enable weight reordering and caching.
- **Dequantization Formula**: `dequant_value = (s4_weight - zero_point) * scale`


### Example 4: INT8 MatMul with Zero-Point Compensation Caching

This example demonstrates INT8 matmul with asymmetric quantization and automatic zero-point compensation caching:

```cpp
int lowoha_int8_matmul_example() {
  using namespace zendnnl::lowoha;
  
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


## Weight Caching and Reordering

One of the key features of LowOHA MatMul is **automatic weight reordering and caching**.

### How It Works

1. **First Execution**: 
   - Weights are reordered to the optimal format for the selected backend
   - Reordered weights are stored in an LRU cache
   - Cache key is generated from weight pointer, dimensions, data type, and backend

2. **Subsequent Executions**:
   - Cache is queried using the same key
   - If cache hit: reordered weights are retrieved (fast path)
   - If cache miss: weights are reordered and cached

3. **Cache Eviction**:
   - When cache is full, least recently used weights are evicted
   - Evicted weights are freed to make room for new entries


## Zero-Point Compensation Caching (INT8)

For INT8 matmul with asymmetric quantization, LowOHA provides **automatic caching of 1D zero-point compensation**:

### How It Works

1. **1D Compensation (src_zp only)**:
   - Compensation depends only on weight column sums: `comp[n] = -src_zp × Σ(B[k,n])`
   - Computed once on first execution and stored in LRU cache
   - Subsequent inferences reuse cached compensation

2. **2D Compensation (wei_zp ≠ 0)**:
   - Compensation depends on source row sums (changes per inference)
   - Cannot be cached, recomputed every execution

### Cache Configuration

| Environment Variable | Values | Description |
|---------------------|--------|-------------|
| `ZENDNNL_ZP_COMP_CACHE` | `1` (default) | Enable ZP compensation caching |
| `ZENDNNL_ZP_COMP_CACHE` | `0` | Disable ZP compensation caching |


## Backend Selection

LowOHA MatMul supports multiple backends. The backend can be selected:

### 1. Via `matmul_params`

```cpp
matmul_params params;
params.lowoha_algo = matmul_algo_t::aocl_dlp;  
```

**Available Algorithms:**
- `matmul_algo_t::auto_tuner` - Auto Tuner (selects performant backend at runtime)
- `matmul_algo_t::dynamic_dispatch` - Automatic backend selection based on heuristics
- `matmul_algo_t::aocl_dlp_blocked` - Blocked AOCL DLP backend
- `matmul_algo_t::onednn_blocked` - Blocked OneDNN backend
- `matmul_algo_t::libxsmm_blocked` - Blocked LibXSMM backend
- `matmul_algo_t::aocl_dlp` - AOCL DLP backend
- `matmul_algo_t::onednn` - OneDNN backend
- `matmul_algo_t::libxsmm` - LibXSMM backend


### 2. Via Environment Variable

```bash
export ZENDNNL_MATMUL_ALGO=1
```

### Supported LowOHA Matmul Kernels

| Algo |       Kernel          |
|------|-----------------------|
| auto | auto_tuner            |
| 0    | dynamic_dispatch      |
| 1    | aocl_dlp_blocked      |
| 2    | onednn_blocked        |
| 3    | libxsmm_blocked       |
| 4    | aocl_dlp              |
| 5    | onednn                |
| 6    | libxsmm               |
