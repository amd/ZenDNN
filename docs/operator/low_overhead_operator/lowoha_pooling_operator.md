
(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# LOWOHA Pooling Operator

## Overview

The **LOWOHA Pooling Operator** is a high-performance, low-overhead pooling operator designed for **latency-sensitive inference workloads**. It provides a direct API to backend library OneDNN with minimal execution overhead for both max pooling and average pooling operations.

Unlike the standard Pooling operator which uses the operator factory pattern, LOWOHA Pooling provides a **function-based interface** optimized for:
- Minimal execution overhead
- Support for both max and average pooling
- Configurable padding modes for average pooling
- Backend-native optimizations
- Direct control over execution parameters
- NHWC data format support

## Pooling Operations

Let:

- *X* ∈ ℝ<sup>N×H×W×C</sup>: Input Tensor (NHWC format)
- *Y* ∈ ℝ<sup>N×H<sub>out</sub>×W<sub>out</sub>×C</sup>: Output Tensor (NHWC format)
- $K_h, K_w$: Kernel (pooling window) dimensions
- $S_h, S_w$: Stride dimensions
- $P_t, P_l, P_b, P_r$: Padding (top, left, bottom, right)

### Output Dimensions

$$
H_{out} = \left\lfloor \frac{H + P_t + P_b - K_h}{S_h} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W + P_l + P_r - K_w}{S_w} \right\rfloor + 1
$$

### Max Pooling

For each output position $(n, h, w, c)$:

$$
Y[n, h, w, c] = \max_{i \in [0, K_h), j \in [0, K_w)} X[n, h \cdot S_h + i, w \cdot S_w + j, c]
$$

### Average Pooling

For each output position $(n, h, w, c)$:

$$
Y[n, h, w, c] = \frac{1}{N_{valid}} \sum_{i \in [0, K_h), j \in [0, K_w)} X[n, h \cdot S_h + i, w \cdot S_w + j, c]
$$

Where $N_{valid}$ is:
- **Exclude Padding**: Count of non-padding elements in the window
- **Include Padding**: Fixed kernel size $K_h \times K_w$

## Core API: `pooling_direct`

The primary interface for LOWOHA Pooling is the `pooling_direct` function:

```cpp
status_t pooling_direct(
  const void *input,        // Input tensor [N, H, W, C]
  void *output,             // Output tensor [N, H_out, W_out, C]
  pool_params &params       // Pooling parameters
);
```

## Parameters Structure

### `pool_params`

The main configuration structure for LOWOHA Pooling:

```cpp
struct pool_params {
  pooling_dims_t dims;           // Tensor dimensions
  pooling_data_types dtypes;     // Data types
  pooling_algo_t algo;           // Algorithm selection
  
  // Strides
  uint32_t stride_h;             // Stride along height
  uint32_t stride_w;             // Stride along width
  
  // Padding
  uint32_t pad_top;              // Top padding
  uint32_t pad_left;             // Left padding
  uint32_t pad_bottom;           // Bottom padding
  uint32_t pad_right;            // Right padding
  
  // Pooling type
  bool is_max_pooling;           // true = max, false = average
  
  // Average pooling mode
  avg_pooling_mode_t avg_mode;   // Include/exclude padding
  
  // Data format
  char data_format[8];           // "NHWC" (default)
  
  uint64_t num_threads;          // Number of threads (0 = auto)
};
```

### `pooling_dims_t`

Structure for tensor dimensions:

```cpp
struct pooling_dims_t {
  uint64_t batch;                // Batch size (N)
  uint64_t in_height;            // Input height (H)
  uint64_t in_width;             // Input width (W)
  uint64_t channels;             // Number of channels (C)
  uint64_t kernel_height;        // Kernel height (K_h)
  uint64_t kernel_width;         // Kernel width (K_w)
  uint64_t out_height;           // Output height (H_out)
  uint64_t out_width;            // Output width (W_out)
};
```

### `pooling_data_types`

Structure for data types:

```cpp
struct pooling_data_types {
  data_type_t src;               // Input data type
  data_type_t dst;               // Output data type
};
```

### Data Type Support

**Supported Data Type Combinations:**

| Input Type | Output Type | Max Pool | Avg Pool | Notes |
|------------|-------------|----------|----------|-------|
| FP32 | FP32 | ✓ | ✓ | Standard floating-point |
| BF16 | BF16 | ✓ | ✓ | Mixed-precision for inference |

### Average Pooling Modes

```cpp
enum class avg_pooling_mode_t {
  include_padding = 0,  // Include padding in average (TensorFlow SAME)
  exclude_padding = 1   // Exclude padding from average (PyTorch default)
};
```

**Mode Comparison:**
- **include_padding**: Divides sum by kernel size \(K_h \times K_w\)
- **exclude_padding**: Divides sum by count of non-padding elements

## Usage Examples

### Example 1: Basic Max Pooling (FP32)

```cpp
int maxpool_fp32_example() {
  using namespace zendnnl::lowoha::pooling;
  
  // Input dimensions [N=1, H=4, W=4, C=2]
  uint64_t N = 1, H = 4, W = 4, C = 2;
  uint64_t kernel_h = 2, kernel_w = 2;
  uint32_t stride_h = 2, stride_w = 2;
  uint32_t pad = 0;  // No padding (VALID)
  
  // Calculate output dimensions
  uint64_t H_out = (H - kernel_h) / stride_h + 1;  // = 2
  uint64_t W_out = (W - kernel_w) / stride_w + 1;  // = 2
  
  // Create input and output tensors (NHWC format)
  std::vector<float> input(N * H * W * C);
  std::vector<float> output(N * H_out * W_out * C, 0.0f);
  
  // Initialize input with simple pattern
  for (uint64_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i);
  }
  
  // Setup pooling parameters
  pool_params params;
  params.dims.batch = N;
  params.dims.in_height = H;
  params.dims.in_width = W;
  params.dims.channels = C;
  params.dims.kernel_height = kernel_h;
  params.dims.kernel_width = kernel_w;
  params.dims.out_height = H_out;
  params.dims.out_width = W_out;
  
  params.stride_h = stride_h;
  params.stride_w = stride_w;
  params.pad_top = pad;
  params.pad_left = pad;
  params.pad_bottom = pad;
  params.pad_right = pad;
  
  params.is_max_pooling = true;
  params.dtypes.src = data_type_t::f32;
  params.dtypes.dst = data_type_t::f32;
  params.algo = pooling_algo_t::none;  // Auto-select
  
  // Execute max pooling
  status_t status = pooling_direct(
    input.data(),
    output.data(),
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 2: Average Pooling with Padding (FP32)

```cpp
int avgpool_with_padding_example() {
  using namespace zendnnl::lowoha::pooling;
  
  // Input dimensions [N=2, H=8, W=8, C=3]
  uint64_t N = 2, H = 8, W = 8, C = 3;
  uint64_t kernel_h = 3, kernel_w = 3;
  uint32_t stride_h = 2, stride_w = 2;
  uint32_t pad_top = 1, pad_left = 1;
  uint32_t pad_bottom = 1, pad_right = 1;
  
  // Calculate output dimensions
  uint64_t H_out = (H + pad_top + pad_bottom - kernel_h) / stride_h + 1;
  uint64_t W_out = (W + pad_left + pad_right - kernel_w) / stride_w + 1;
  
  // Create tensors
  std::vector<float> input(N * H * W * C);
  std::vector<float> output(N * H_out * W_out * C, 0.0f);
  
  // Initialize input
  for (uint64_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i % 100) / 10.0f;
  }
  
  // Setup pooling parameters
  pool_params params;
  params.dims.batch = N;
  params.dims.in_height = H;
  params.dims.in_width = W;
  params.dims.channels = C;
  params.dims.kernel_height = kernel_h;
  params.dims.kernel_width = kernel_w;
  params.dims.out_height = H_out;
  params.dims.out_width = W_out;
  
  params.stride_h = stride_h;
  params.stride_w = stride_w;
  params.pad_top = pad_top;
  params.pad_left = pad_left;
  params.pad_bottom = pad_bottom;
  params.pad_right = pad_right;
  
  params.is_max_pooling = false;  // Average pooling
  params.avg_mode = avg_pooling_mode_t::exclude_padding;
  params.dtypes.src = data_type_t::f32;
  params.dtypes.dst = data_type_t::f32;
  params.algo = pooling_algo_t::none;
  
  // Execute average pooling
  status_t status = pooling_direct(
    input.data(),
    output.data(),
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 3: BF16 Max Pooling for CNNs

```cpp
int maxpool_bf16_cnn_example() {
  using namespace zendnnl::lowoha::pooling;
  
  // CNN feature map: [N=1, H=16, W=16, C=64]
  uint64_t N = 1, H = 16, W = 16, C = 64;
  uint64_t kernel_h = 2, kernel_w = 2;
  uint32_t stride_h = 2, stride_w = 2;
  
  uint64_t H_out = H / stride_h;  // = 8
  uint64_t W_out = W / stride_w;  // = 8
  
  // Allocate BF16 tensors (stored as uint16_t)
  std::vector<uint16_t> input(N * H * W * C);
  std::vector<uint16_t> output(N * H_out * W_out * C, 0);
  
  // Initialize with BF16 values
  for (uint64_t i = 0; i < input.size(); ++i) {
    input[i] = 0x3F80 + static_cast<uint16_t>(i % 256);
  }
  
  // Setup parameters
  pool_params params;
  params.dims.batch = N;
  params.dims.in_height = H;
  params.dims.in_width = W;
  params.dims.channels = C;
  params.dims.kernel_height = kernel_h;
  params.dims.kernel_width = kernel_w;
  params.dims.out_height = H_out;
  params.dims.out_width = W_out;
  
  params.stride_h = stride_h;
  params.stride_w = stride_w;
  params.pad_top = 0;
  params.pad_left = 0;
  params.pad_bottom = 0;
  params.pad_right = 0;
  
  params.is_max_pooling = true;
  params.dtypes.src = data_type_t::bf16;
  params.dtypes.dst = data_type_t::bf16;
  params.algo = pooling_algo_t::none;
  
  // Execute BF16 max pooling
  status_t status = pooling_direct(
    input.data(),
    output.data(),
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 4: Comparing Padding Modes

```cpp
int compare_padding_modes_example() {
  using namespace zendnnl::lowoha::pooling;
  
  // Small test case: [N=1, H=4, W=4, C=1]
  uint64_t N = 1, H = 4, W = 4, C = 1;
  uint64_t kernel_h = 2, kernel_w = 2;
  uint32_t stride_h = 2, stride_w = 2;
  uint32_t pad = 1;  // Padding on all sides
  
  uint64_t H_out = (H + 2*pad - kernel_h) / stride_h + 1;
  uint64_t W_out = (W + 2*pad - kernel_w) / stride_w + 1;
  
  // Input filled with constant value
  std::vector<float> input(N * H * W * C, 4.0f);
  std::vector<float> output_exclude(N * H_out * W_out * C, 0.0f);
  std::vector<float> output_include(N * H_out * W_out * C, 0.0f);
  
  // Setup base parameters
  pool_params params;
  params.dims.batch = N;
  params.dims.in_height = H;
  params.dims.in_width = W;
  params.dims.channels = C;
  params.dims.kernel_height = kernel_h;
  params.dims.kernel_width = kernel_w;
  params.dims.out_height = H_out;
  params.dims.out_width = W_out;
  params.stride_h = stride_h;
  params.stride_w = stride_w;
  params.pad_top = pad;
  params.pad_left = pad;
  params.pad_bottom = pad;
  params.pad_right = pad;
  params.is_max_pooling = false;
  params.dtypes.src = data_type_t::f32;
  params.dtypes.dst = data_type_t::f32;
  params.algo = pooling_algo_t::none;
  
  // Test 1: Exclude padding
  params.avg_mode = avg_pooling_mode_t::exclude_padding;
  pooling_direct(input.data(), output_exclude.data(), params);
  
  // Test 2: Include padding
  params.avg_mode = avg_pooling_mode_t::include_padding;
  pooling_direct(input.data(), output_include.data(), params);
  
  // Corner element will differ:
  // - exclude_padding: 4.0 (only 1 valid element)
  // - include_padding: 1.0 (4.0 / 4 kernel elements)
  
  return 0;
}
```

## Backend Selection

LOWOHA Pooling supports multiple backends:

### Available Algorithms

```cpp
enum class pooling_algo_t {
  none = -1,             // Auto-select (default)
  dynamic_dispatch = 0,  // Not implemented
  onednn = 1,            // OneDNN backend
  reference = 2          // Reference implementation
};
```

### Algorithm Selection Priority

1. **Auto-selection** (`algo = pooling_algo_t::none`):
   - If OneDNN is available: Uses OneDNN backend
   - Otherwise: Falls back to reference implementation

2. **Explicit selection**:
   ```cpp
   params.algo = pooling_algo_t::onednn;     // Use OneDNN
   params.algo = pooling_algo_t::reference;  // Use reference
   ```

## Performance Considerations

### 1. Memory Layout

- **NHWC format** is required (channels-last)
- Channels are the fastest-varying dimension
- Better cache locality for pooling across spatial dimensions

### 2. Threading

- Set `params.num_threads` to control parallelism
- Default (`num_threads = 0`): Uses OMP_NUM_THREADS or system default
- Parallelization is typically across batch and channels

### 3. Padding Strategy

**For Max Pooling:**
- Padding values are typically -∞ (don't affect max)

**For Average Pooling:**
- **exclude_padding**: More accurate (common in PyTorch)
- **include_padding**: Simpler (TensorFlow SAME padding)

### 4. Kernel and Stride Selection

| Configuration | Use Case | Downsampling Factor |
|---------------|----------|---------------------|
| Kernel 2×2, Stride 2×2 | Standard downsampling | 4× reduction |
| Kernel 3×3, Stride 2×2 | Overlapping pooling | 4× reduction |
| Kernel 3×3, Stride 1×1 | Smoothing, no downsample | 1× (no reduction) |

### 5. Backend Performance

| Backend | Best For | Notes |
|---------|----------|-------|
| OneDNN | Large feature maps, BF16 | Vectorized, optimized for AVX-512 |
| Reference | Small inputs, debugging | Simple implementation, portable |

## Common Use Cases

### 1. CNN Downsampling (Max Pooling)

```cpp
// After convolution layer, reduce spatial dimensions
pool_params params;
params.dims.batch = batch_size;
params.dims.in_height = 32;
params.dims.in_width = 32;
params.dims.channels = 64;
params.dims.kernel_height = 2;
params.dims.kernel_width = 2;
params.dims.out_height = 16;
params.dims.out_width = 16;
params.stride_h = 2;
params.stride_w = 2;
params.is_max_pooling = true;
params.dtypes.src = data_type_t::bf16;
params.dtypes.dst = data_type_t::bf16;
pooling_direct(conv_output, pooling_output, params);
```

### 2. Global Average Pooling

```cpp
// Reduce spatial dimensions to 1×1 for classification
// Input: [N, H, W, C] → Output: [N, 1, 1, C]
pool_params params;
params.dims.kernel_height = H;  // Pool entire height
params.dims.kernel_width = W;   // Pool entire width
params.dims.out_height = 1;
params.dims.out_width = 1;
params.stride_h = 1;
params.stride_w = 1;
params.is_max_pooling = false;  // Average pooling
params.avg_mode = avg_pooling_mode_t::exclude_padding;
pooling_direct(features, global_pool, params);
```

### 3. Region of Interest (RoI) Pooling

```cpp
// Extract fixed-size features from variable-size regions
// Common in object detection (Fast R-CNN, etc.)
pool_params params;
params.dims.in_height = roi_height;
params.dims.in_width = roi_width;
params.dims.kernel_height = roi_height / output_size;
params.dims.kernel_width = roi_width / output_size;
params.dims.out_height = output_size;  // e.g., 7
params.dims.out_width = output_size;   // e.g., 7
params.is_max_pooling = true;
pooling_direct(roi_features, pooled_features, params);
```

## Padding Modes Explained

### Example: 4×4 Input, 2×2 Kernel, Stride 2, Padding 1

**Input (constant value 4.0):**
```
    ┌──────────────────┐
pad │ 0  0  0  0  0  0 │
    │ 0  4  4  4  4  0 │
    │ 0  4  4  4  4  0 │
    │ 0  4  4  4  4  0 │
    │ 0  4  4  4  4  0 │
    │ 0  0  0  0  0  0 │
    └──────────────────┘
```

**Corner Window (top-left 2×2):**
```
┌────┬────┐
│ 0  │ 0  │  Corner element has only 1 valid value (4.0)
├────┼────┤
│ 0  │ 4  │
└────┴────┘
```

**Average Pooling Results:**
- **exclude_padding**: `4.0 / 1 = 4.0` (only count the valid element)
- **include_padding**: `4.0 / 4 = 1.0` (divide by kernel size)

## API Summary

| Function | Purpose |
|----------|---------|
| `pooling_direct` | Main execution API for max/average pooling |

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `const void*` | Input tensor [N, H, W, C] |
| `output` | `void*` | Output tensor [N, H_out, W_out, C] |
| `params` | `pool_params&` | Configuration parameters |

## Error Handling

The `pooling_direct` function returns `status_t`:

- `status_t::success`: Operation completed successfully
- `status_t::failure`: Operation failed (check logs for details)

Common failure causes:
- Null input/output pointers
- Invalid dimensions (zero values)
- Unsupported data type combination
- Output dimensions don't match calculated dimensions

## Typical Pooling Configurations

### ResNet-style Max Pooling
```cpp
kernel_h = 3, kernel_w = 3
stride_h = 2, stride_w = 2
pad_top = 1, pad_left = 1, pad_bottom = 1, pad_right = 1
is_max_pooling = true
```

### VGG-style Max Pooling
```cpp
kernel_h = 2, kernel_w = 2
stride_h = 2, stride_w = 2
pad_top = 0, pad_left = 0, pad_bottom = 0, pad_right = 0
is_max_pooling = true
```

### Inception-style Average Pooling
```cpp
kernel_h = 3, kernel_w = 3
stride_h = 1, stride_w = 1
pad_top = 1, pad_left = 1, pad_bottom = 1, pad_right = 1
is_max_pooling = false
avg_mode = exclude_padding
```
