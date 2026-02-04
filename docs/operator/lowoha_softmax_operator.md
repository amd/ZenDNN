
(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# LOWOHA Softmax Operator

## Overview

The **LOWOHA Softmax Operator** is a high-performance, low-overhead softmax operator designed for **latency-sensitive inference workloads**. It provides a direct API to backend libraries (OneDNN, Reference) with minimal execution overhead.

Unlike the standard Softmax operator which uses the operator factory pattern, LOWOHA Softmax provides a **function-based interface** optimized for:
- Minimal execution overhead
- Support for multi-dimensional tensors
- Log-softmax variant for classification tasks
- Backend-native optimizations
- Direct control over execution parameters

## Softmax Operation

Let:

- *X* ∈ ℝ<sup>N<sub>1</sub>×N<sub>2</sub>×...×N<sub>k</sub></sup>: Input Tensor
- *Y* ∈ ℝ<sup>N<sub>1</sub>×N<sub>2</sub>×...×N<sub>k</sub></sup>: Output Tensor
- *axis*: Axis along which to compute softmax (default: -1, last axis)

The computation can be expressed as:

### Standard Softmax

For each element along the softmax axis:

$$
Y_i = \frac{\exp(X_i - \max(X))}{\sum_j \exp(X_j - \max(X))}
$$

Where:
- $X_i$ is the input element at position $i$
- $\max(X)$ is the maximum value along the axis
- The sum is computed over all elements along the axis

### Log Softmax

For each element along the softmax axis:

$$
Y_i = X_i - \max(X) - \log\left(\sum_j \exp(X_j - \max(X))\right)
$$

The $\max(X)$ term ensures numerical stability by preventing overflow in the exponential function.

## Core API: `softmax_direct`

The primary interface for LOWOHA Softmax is the `softmax_direct` function:

```cpp
status_t softmax_direct(
  const void *input,        // Input tensor
  void *output,             // Output tensor (same shape as input)
  softmax_params &params    // Softmax parameters
);
```

## Parameters Structure

### `softmax_params`

The main configuration structure for LOWOHA Softmax:

```cpp
struct softmax_params {
  uint64_t batch;                       // Batch size (outer dimensions product)
  uint64_t axis_dim;                    // Dimension size along softmax axis
  int axis;                             // Axis along which to compute softmax (-1 for last)
  bool log_softmax;                     // If true, compute log(softmax(x))
  data_type_t src_dt;                   // Source/input data type
  data_type_t dst_dt;                   // Destination/output data type
  softmax_algo_t algorithm;             // Selected algorithm
  uint64_t num_threads;                 // Number of threads (0 = auto)
  
  // Original tensor shape (for OneDNN backend)
  uint64_t shape[SOFTMAX_MAX_NDIMS];    // Original tensor dimensions
  int ndims;                            // Number of dimensions
};
```

### Data Type Support

**Supported Data Type Combinations:**

| Input Type | Output Type | Notes |
|------------|-------------|-------|
| FP32 | FP32 | Standard floating-point precision |
| BF16 | BF16 | Mixed-precision for inference |

### Helper Function: `setup_softmax_shape`

To simplify parameter initialization for N-dimensional tensors:

```cpp
status_t setup_softmax_shape(
  softmax_params &params,
  const uint64_t *shape,    // Array of tensor dimensions
  int ndims,                // Number of dimensions
  int axis                  // Axis for softmax (-1 for last axis)
);
```

**Example Usage:**
```cpp
softmax_params params;
uint64_t shape[] = {2, 64, 128};  // [batch, sequence, features]
setup_softmax_shape(params, shape, 3, -1);  // Softmax along last axis
```

## Usage Examples

### Example 1: Basic FP32 Softmax

```cpp
int softmax_fp32_example() {
  using namespace zendnnl::lowoha::softmax;
  
  // Input dimensions: [batch=2, features=5]
  uint64_t batch = 2;
  uint64_t axis_dim = 5;
  uint64_t total_size = batch * axis_dim;
  
  // Create input and output tensors
  std::vector<float> input = {
    1.0f, 2.0f, 3.0f, 4.0f, 5.0f,     // batch 0
    -1.0f, 0.0f, 1.0f, 2.0f, 3.0f     // batch 1
  };
  std::vector<float> output(total_size, 0.0f);
  
  // Setup parameters
  softmax_params params;
  uint64_t shape[] = {batch, axis_dim};
  setup_softmax_shape(params, shape, 2, -1);
  
  params.src_dt = data_type_t::f32;
  params.dst_dt = data_type_t::f32;
  params.log_softmax = false;
  params.algorithm = softmax_algo_t::none;  // Auto-select
  
  // Execute softmax
  status_t status = softmax_direct(
    input.data(),
    output.data(),
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 2: BF16 Softmax for Attention

```cpp
int softmax_bf16_attention_example() {
  using namespace zendnnl::lowoha::softmax;
  
  // Attention dimensions: [batch=4, seq_len=128]
  uint64_t batch = 4;
  uint64_t seq_len = 128;
  uint64_t total_size = batch * seq_len;
  
  // Allocate BF16 tensors (stored as uint16_t)
  std::vector<uint16_t> attention_scores(total_size);
  std::vector<uint16_t> attention_weights(total_size, 0);
  
  // Initialize attention scores (BF16 representation)
  for (uint64_t i = 0; i < total_size; ++i) {
    attention_scores[i] = 0x3F80 + static_cast<uint16_t>(i % 16);
  }
  
  // Setup parameters
  softmax_params params;
  uint64_t shape[] = {batch, seq_len};
  setup_softmax_shape(params, shape, 2, -1);
  
  params.src_dt = data_type_t::bf16;
  params.dst_dt = data_type_t::bf16;
  params.log_softmax = false;
  params.algorithm = softmax_algo_t::none;
  
  // Execute softmax on attention scores
  status_t status = softmax_direct(
    attention_scores.data(),
    attention_weights.data(),
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```

## Backend Selection

LOWOHA Softmax supports multiple backends:

### Available Algorithms

```cpp
enum class softmax_algo_t {
  none = -1,             // Auto-select (default)
  dynamic_dispatch = 0,  // Not implemented
  onednn = 1,            // OneDNN backend
  reference = 2          // Reference implementation
};
```

### Algorithm Selection Priority

1. **Auto-selection** (`algorithm = softmax_algo_t::none`):
   - If OneDNN is available: Uses OneDNN backend
   - Otherwise: Falls back to reference implementation

2. **Explicit selection**:
   ```cpp
   params.algorithm = softmax_algo_t::onednn;     // Use OneDNN
   params.algorithm = softmax_algo_t::reference;  // Use reference
   ```

## Performance Considerations

### 1. Numerical Stability

The implementation uses the **max-shift trick** for numerical stability:
- Computes $\max(X)$ before exponentials
- Prevents overflow: $\exp(x - \max(X))$ stays within valid range
- Prevents underflow in division

### 2. Threading

- Set `params.num_threads` to control parallelism
- Default (`num_threads = 0`): Uses OMP_NUM_THREADS or system default
- Batch parallelism: Outer dimensions are parallelized

### 3. Memory Layout

- Softmax operates on contiguous memory
- Axis dimension should be the last dimension for best performance
- OneDNN backend may optimize for specific layouts

### 4. Backend Performance

| Backend | Best For | Notes |
|---------|----------|-------|
| OneDNN | Large tensors, BF16 | Vectorized, optimized for AVX-512 |
| Reference | Small tensors, debugging | Simple implementation, portable |

## Common Use Cases

### 1. Attention Mechanisms (Transformers)

```cpp
// Attention weights: softmax(Q·K^T / √d_k)
softmax_params params;
uint64_t shape[] = {batch, num_heads, seq_len, seq_len};
setup_softmax_shape(params, shape, 4, -1);  // Softmax over last dim
params.src_dt = data_type_t::bf16;
params.dst_dt = data_type_t::bf16;
softmax_direct(attention_scores, attention_weights, params);
```

### 2. Classification Output Layer

```cpp
// Convert logits to probabilities
softmax_params params;
uint64_t shape[] = {batch_size, num_classes};
setup_softmax_shape(params, shape, 2, -1);
params.src_dt = data_type_t::f32;
params.dst_dt = data_type_t::f32;
softmax_direct(logits, probabilities, params);
```

## API Summary

| Function | Purpose |
|----------|---------|
| `softmax_direct` | Main execution API for softmax/log-softmax |
| `setup_softmax_shape` | Helper to initialize parameters from N-D shape |

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `const void*` | Input tensor pointer |
| `output` | `void*` | Output tensor pointer (same shape as input) |
| `params` | `softmax_params&` | Configuration parameters |

## Error Handling

The `softmax_direct` function returns `status_t`:

- `status_t::success`: Operation completed successfully
- `status_t::failure`: Operation failed (check logs for details)

Common failure causes:
- Null input/output pointers
- Invalid dimensions (batch=0 or axis_dim=0)
- Unsupported data type combination
- Invalid axis index
