
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
C = \text{Post\_Ops}(\alpha \cdot (A \cdot B) + \text{Bias} + \beta \cdot C)
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
  lowoha_params params,        // LowOHA parameters (dtypes, post-ops, quantization)
  int Batch_A = 1,             // Number of batches in A (default: 1)
  int Batch_B = 1              // Number of batches in B (default: 1)
);
```


## Parameters Structure

### `lowoha_params`

The main configuration structure for LowOHA MatMul:

```cpp
struct lowoha_params {
  data_types dtypes;                       // Data types for tensors
  std::vector<postop> postop_;             // Post-operations
  lowoha_quantization_params_t quant_params; // Quantization parameters
  char mem_format_a;                       // Memory format for A ('n'=reordered, 'r'=non-reordered)
  char mem_format_b;                       // Memory format for B ('n'=reordered, 'r'=non-reordered)
  matmul_algo_t lowoha_algo;               // Preferred backend algorithm
};
```


### `data_types`

Specifies the data types for each tensor:

```cpp
struct data_types {
  data_type_t src;      // Input data type (f32, bf16, s8, u8)
  data_type_t wei;      // Weight data type (f32, bf16, s8, u8)
  data_type_t dst;      // Output data type (f32, bf16, s8, u8, s32)
  data_type_t bias;     // Bias data type (f32, bf16, s8)
  data_type_t compute;  // Computation type (usually same as dst)
};
```

**Supported Combinations:**

| Src Type | Weight Type | Bias Type | Output Type | Notes |
|----------|-------------|-----------|-------------|-------|
| FP32 | FP32 | FP32 | FP32 | Standard floating-point |
| BF16 | BF16 | FP32/BF16 | FP32/BF16 | Mixed-precision BFloat16 |


### `postop`

Defines a single post-operation:

```cpp
struct postop {
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


### `lowoha_quantization_params_t`

Quantization scale and zero-point parameters:

```cpp
struct lowoha_quantization_params_t {
  struct quant_t {
    const void *buff;    // Pointer to scale/zero-point data
    data_type_t dt;      // Data type (f32 for scale, s8/u8/s32 for zero-point)
    size_t size;         // Size of buffer
  };
  
  quant_t src_scale;     // Source tensor scale
  quant_t wei_scale;     // Weight tensor scale
  quant_t dst_scale;     // Destination tensor scale
  quant_t src_zp;        // Source tensor zero-point
  quant_t wei_zp;        // Weight tensor zero-point
  quant_t dst_zp;        // Destination tensor zero-point
};
```


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
  data_types dtypes;
  dtypes.src = data_type_t::bf16;
  dtypes.wei = data_type_t::bf16;
  dtypes.dst = data_type_t::bf16;
  dtypes.bias = data_type_t::f32;
  dtypes.compute = data_type_t::f32;
  
  // Configure post-operations: GELU -> Binary Add
  lowoha_params params;
  params.dtypes = dtypes;
  
  // Post-op 1: GELU
  postop gelu_op;
  gelu_op.po_type = post_op_type_t::gelu_erf;
  gelu_op.buff = nullptr;
  gelu_op.dtype = data_type_t::none;
  gelu_op.dims = {M, N};
  params.postop_.push_back(gelu_op);
  
  // Post-op 2: Binary Add
  postop add_op;
  add_op.po_type = post_op_type_t::binary_add;
  add_op.buff = add_tensor.data();
  add_op.dtype = data_type_t::bf16;
  add_op.dims = {M, N};
  params.postop_.push_back(add_op);
  
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
    params,
    1, 1
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
  data_types dtypes;
  dtypes.src = data_type_t::f32;
  dtypes.wei = data_type_t::f32;
  dtypes.dst = data_type_t::f32;
  dtypes.bias = data_type_t::f32;
  dtypes.compute = data_type_t::f32;
  
  lowoha_params params;
  params.dtypes = dtypes;
  
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
    params,
    batch_size,  // Batch_A = 16 (batched input)
    1            // Batch_B = 1 (shared weights)
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



## Backend Selection

LowOHA MatMul supports multiple backends. The backend can be selected:

### 1. Via `lowoha_params`

```cpp
lowoha_params params;
params.lowoha_algo = matmul_algo_t::aocl_blis;  
```

**Available Algorithms:**
- `matmul_algo_t::dynamic_dispatch` (0) - Automatic backend selection based on heuristics
- `matmul_algo_t::aocl_blis` (1) - AOCL BLIS backend
- `matmul_algo_t::aocl_blis_blocked` (2) - Blocked AOCL BLIS
- `matmul_algo_t::onednn` (3) - OneDNN backend
- `matmul_algo_t::onednn_blocked` (4) - Blocked OneDNN
- `matmul_algo_t::libxsmm` (5) - LibXSMM backend

### 2. Via Environment Variable

```bash
export ZENDNNL_MATMUL_ALGO=1  # 0=Dynamic, 1=AOCL BLIS, 2=AOCL Blocked, 3=OneDNN, 4=OneDNN Blocked, 5=LibXSMM
```
