
(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# LOWOHA Conv2D Operator

## Overview

The **LOWOHA Conv2D Operator** is a high-performance, low-overhead 2D convolution operator designed for **latency-sensitive inference workloads**. It provides a direct API to backend libraries (OneDNN, AOCL) with built-in weight caching and fused post-operations.

Unlike the standard Conv2D operator which uses the operator factory pattern, LOWOHA Conv2D provides a **function-based interface** optimized for:
- Minimal execution overhead
- Repeated weight reuse
- Backend-native post-operation fusion
- Direct control over execution parameters
- Support for depthwise and grouped convolutions


# General Conv2D Operation

Let:

- *Input* ∈ ℝ<sup>N×H×W×C</sup>: Input Tensor (NHWC format)
- *Filter* ∈ ℝ<sup>KH×KW×C_in×C_out</sup>: Filter/Weight Tensor
- *Bias* ∈ ℝ<sup>C_out</sup>: Optional Bias vector
- *Output* ∈ ℝ<sup>N×H_out×W_out×C_out</sup>: Output Tensor
- *stride*: Stride values (stride_h, stride_w)
- *pad*: Padding values (pad_top, pad_bottom, pad_left, pad_right)
- *dilation*: Dilation values (dilation_h, dilation_w)
- *Post_ops(x)*: Optional activation or binary post operations (Example: ReLU, ReLU6, Binary_add, etc.)

The computation can be expressed as:

$$
\text{Output}[n, h, w, c] = \text{PostOps}\left(\sum_{kh, kw, c_{in}} \text{Input}[n, h \cdot s_h + kh \cdot d_h - p_t, w \cdot s_w + kw \cdot d_w - p_l, c_{in}] \cdot \text{Filter}[kh, kw, c_{in}, c] + \text{Bias}[c]\right)
$$

Where:
- *s_h*, *s_w* are stride height and width
- *d_h*, *d_w* are dilation height and width
- *p_t*, *p_l* are padding top and left


## Core API: `conv_direct`

The primary interface for LOWOHA Conv2D is the `conv_direct` function:

```cpp
status_t conv_direct(
  const void *input,           // Input tensor [N, H, W, C] (NHWC format)
  const void *filter,          // Filter tensor [KH, KW, C_in, C_out]
  const void *bias,            // Optional bias vector [C_out] (can be nullptr)
  void *output,                // Output tensor [N, H_out, W_out, C_out]
  const bool is_weights_const, // Whether filters are constant (enables caching)
  conv_params &params          // LOWOHA parameters (dims, strides, padding, dtypes, post-ops)
);
```


## Parameters Structure

### `conv_params`

The main configuration structure for LOWOHA Conv2D:

```cpp
struct conv_params {
  // Strides [stride_h, stride_w]
  uint32_t stride_h;                      // Stride along height dimension
  uint32_t stride_w;                      // Stride along width dimension

  // Padding [top, left, bottom, right]
  uint32_t pad_top;                       // Padding at top (height)
  uint32_t pad_left;                      // Padding at left (width)
  uint32_t pad_bottom;                    // Padding at bottom (height)
  uint32_t pad_right;                     // Padding at right (width)

  // Dilations [dilation_h, dilation_w]
  uint32_t dilation_h;                    // Dilation along height dimension (default: 1)
  uint32_t dilation_w;                    // Dilation along width dimension (default: 1)

  depthwise_params depthwise;             // Depthwise convolution parameters

  char data_format[8];                    // Data format string (currently "NHWC" only)

  conv_dims_t dims;                       // Convolution dimensions
  conv_algo_t algo;                       // Convolution algorithm
  conv_data_types dtypes;                 // Data types for tensors
  std::vector<conv_postop> postop_;       // Post-operations
};
```


### `conv_dims_t`

Structure for convolution tensor dimensions:

```cpp
struct conv_dims_t {
  uint64_t batch;                // Batch size (N)
  uint64_t in_height;            // Input height (H)
  uint64_t in_width;             // Input width (W)
  uint64_t in_channels;          // Input channels (C_in)

  uint64_t filter_height;        // Filter height (KH)
  uint64_t filter_width;         // Filter width (KW)
  uint64_t out_channels;         // Output channels (C_out)

  uint64_t out_height;           // Output height
  uint64_t out_width;            // Output width
};
```

**Output Spatial Dimensions:**

The output spatial dimensions are computed as:

$$
H_{out} = \left\lfloor \frac{H + p_{top} + p_{bottom} - d_h \cdot (KH - 1) - 1}{s_h} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W + p_{left} + p_{right} - d_w \cdot (KW - 1) - 1}{s_w} \right\rfloor + 1
$$


### `conv_data_types`

Specifies the data types for each tensor:

```cpp
struct conv_data_types {
  data_type_t input;     // Input data type
  data_type_t filter;    // Filter data type
  data_type_t bias;      // Bias data type
  data_type_t output;    // Output data type
};
```

**Supported Combinations:**

| Input Type | Filter Type | Bias Type | Output Type | Notes |
|------------|-------------|-----------|-------------|-------|
| FP32 | FP32 | FP32 | FP32 | Standard floating-point |
| BF16 | BF16 | FP32/BF16 | FP32/BF16 | Mixed-precision BFloat16 |


### `conv_postop`

Defines a single post-operation:

```cpp
struct conv_postop {
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
| `post_op_type_t::clip` | Clip values (use for ReLU6: alpha=0, beta=6) | No |
| `post_op_type_t::gelu_erf` | GELU (erf variant) | No |
| `post_op_type_t::gelu_tanh` | GELU (tanh variant) | No |
| `post_op_type_t::swish` | SiLU / Swish | No |
| `post_op_type_t::sigmoid` | Sigmoid | No |
| `post_op_type_t::tanh` | Hyperbolic Tangent | No |
| `post_op_type_t::binary_add` | Element-wise Add | Yes |
| `post_op_type_t::binary_mul` | Element-wise Multiply | Yes |

**Binary Post-Op Dimensions:**

For binary post-ops (`binary_add` and `binary_mul`), the `dims` field specifies the tensor shape:

| Shape | Format | Description |
|-------|--------|-------------|
| `{N, H_out, W_out, C_out}` | Full tensor | Element-wise operation on entire output |
| `{1, 1, 1, C_out}` | Broadcast | Same values broadcast across spatial dimensions |
| `{1, H_out, W_out, 1}` | Spatial | Same spatial pattern across all channels |


### `depthwise_params`

Parameters for depthwise and grouped convolutions:

```cpp
struct depthwise_params {
  uint32_t groups;               // Number of groups for grouped convolution (1 = standard conv)
  bool is_depthwise;             // True if depthwise convolution (groups == in_channels)
  uint32_t depth_multiplier;     // Depth multiplier for depthwise (output_channels / in_channels)
};
```

**Convolution Types:**

| Type | Groups | is_depthwise | Description |
|------|--------|--------------|-------------|
| Standard | 1 | false | Regular convolution |
| Grouped | > 1 | false | Input/output channels split into groups |
| Depthwise | in_channels | true | Separate filter per input channel |

**Depthwise Convolution:**
- Used in MobileNet architectures
- Dramatically reduces parameters: `KH × KW × C_in × multiplier` vs `KH × KW × C_in × C_out`
- Often combined with pointwise (1×1) convolution


## Usage Examples

### Example 1: Basic FP32 Conv2D with ReLU

```cpp
int lowoha_conv2d_relu_fp32_example() {
  using namespace zendnnl::lowoha::conv;
  
  // Convolution dimensions
  int batch = 4;
  int in_h = 56, in_w = 56, in_c = 64;
  int filter_h = 3, filter_w = 3, out_c = 128;
  
  // Calculate output dimensions (stride=1, pad=1)
  int out_h = in_h;  // SAME padding
  int out_w = in_w;
  
  // Allocate tensors
  std::vector<float> input(batch * in_h * in_w * in_c, 1.0f);
  std::vector<float> filter(filter_h * filter_w * in_c * out_c, 0.5f);
  std::vector<float> bias(out_c, 0.1f);
  std::vector<float> output(batch * out_h * out_w * out_c, 0.0f);
  
  // Configure conv_params
  conv_params params;
  
  // Set dimensions
  params.dims.batch = batch;
  params.dims.in_height = in_h;
  params.dims.in_width = in_w;
  params.dims.in_channels = in_c;
  params.dims.filter_height = filter_h;
  params.dims.filter_width = filter_w;
  params.dims.out_channels = out_c;
  params.dims.out_height = out_h;
  params.dims.out_width = out_w;
  
  // Set stride and padding
  params.stride_h = 1;
  params.stride_w = 1;
  params.pad_top = 1;
  params.pad_bottom = 1;
  params.pad_left = 1;
  params.pad_right = 1;
  
  // Set data types
  params.dtypes.input = data_type_t::f32;
  params.dtypes.filter = data_type_t::f32;
  params.dtypes.bias = data_type_t::f32;
  params.dtypes.output = data_type_t::f32;
  
  // Add ReLU post-op
  conv_postop relu_op;
  relu_op.po_type = post_op_type_t::relu;
  relu_op.buff = nullptr;
  relu_op.dtype = data_type_t::none;
  params.postop_.push_back(relu_op);
  
  // Execute Conv2D
  status_t status = conv_direct(
    input.data(),
    filter.data(),
    bias.data(),
    output.data(),
    true,  // is_weights_const (enables caching)
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```


### Example 2: BF16 Conv2D with Multiple Post-Ops

```cpp
int lowoha_conv2d_bf16_fused_ops_example() {
  using namespace zendnnl::lowoha::conv;
  
  // Convolution dimensions
  int batch = 8;
  int in_h = 28, in_w = 28, in_c = 128;
  int filter_h = 3, filter_w = 3, out_c = 256;
  int out_h = 28, out_w = 28;
  
  // Allocate BF16 tensors (stored as uint16_t)
  std::vector<uint16_t> input_bf16(batch * in_h * in_w * in_c);
  std::vector<uint16_t> filter_bf16(filter_h * filter_w * in_c * out_c);
  std::vector<float> bias(out_c, 0.0f);
  std::vector<uint16_t> output_bf16(batch * out_h * out_w * out_c);
  
  // Binary add tensor (for residual connection)
  std::vector<uint16_t> add_tensor(batch * out_h * out_w * out_c);
  
  // Configure conv_params
  conv_params params;
  
  // Set dimensions
  params.dims.batch = batch;
  params.dims.in_height = in_h;
  params.dims.in_width = in_w;
  params.dims.in_channels = in_c;
  params.dims.filter_height = filter_h;
  params.dims.filter_width = filter_w;
  params.dims.out_channels = out_c;
  params.dims.out_height = out_h;
  params.dims.out_width = out_w;
  
  // Set stride and padding
  params.stride_h = 1;
  params.stride_w = 1;
  params.pad_top = 1;
  params.pad_bottom = 1;
  params.pad_left = 1;
  params.pad_right = 1;
  
  // Set data types
  params.dtypes.input = data_type_t::bf16;
  params.dtypes.filter = data_type_t::bf16;
  params.dtypes.bias = data_type_t::f32;
  params.dtypes.output = data_type_t::bf16;
  
  // Post-op 1: Binary Add (residual connection)
  conv_postop add_op;
  add_op.po_type = post_op_type_t::binary_add;
  add_op.buff = add_tensor.data();
  add_op.dtype = data_type_t::bf16;
  add_op.dims = {batch, out_h, out_w, out_c};
  params.postop_.push_back(add_op);
  
  // Post-op 2: ReLU
  conv_postop relu_op;
  relu_op.po_type = post_op_type_t::relu;
  relu_op.buff = nullptr;
  relu_op.dtype = data_type_t::none;
  params.postop_.push_back(relu_op);
  
  // Execute Conv2D
  status_t status = conv_direct(
    input_bf16.data(),
    filter_bf16.data(),
    bias.data(),
    output_bf16.data(),
    true,  // is_weights_const (enables caching)
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```


### Example 3: Depthwise Conv2D (MobileNet Pattern)

```cpp
int lowoha_depthwise_conv2d_example() {
  using namespace zendnnl::lowoha::conv;
  
  // Depthwise convolution dimensions
  int batch = 4;
  int in_h = 112, in_w = 112, in_c = 32;
  int filter_h = 3, filter_w = 3;
  int depth_multiplier = 1;
  int out_c = in_c * depth_multiplier;  // 32
  int out_h = 112, out_w = 112;
  
  // Allocate tensors
  std::vector<float> input(batch * in_h * in_w * in_c, 1.0f);
  std::vector<float> filter(filter_h * filter_w * in_c * depth_multiplier, 0.5f);
  std::vector<float> bias(out_c, 0.0f);
  std::vector<float> output(batch * out_h * out_w * out_c, 0.0f);
  
  // Configure conv_params
  conv_params params;
  
  // Set dimensions
  params.dims.batch = batch;
  params.dims.in_height = in_h;
  params.dims.in_width = in_w;
  params.dims.in_channels = in_c;
  params.dims.filter_height = filter_h;
  params.dims.filter_width = filter_w;
  params.dims.out_channels = out_c;
  params.dims.out_height = out_h;
  params.dims.out_width = out_w;
  
  // Set stride and padding
  params.stride_h = 1;
  params.stride_w = 1;
  params.pad_top = 1;
  params.pad_bottom = 1;
  params.pad_left = 1;
  params.pad_right = 1;
  
  // Configure depthwise convolution
  params.depthwise.groups = in_c;
  params.depthwise.is_depthwise = true;
  params.depthwise.depth_multiplier = depth_multiplier;
  
  // Set data types
  params.dtypes.input = data_type_t::f32;
  params.dtypes.filter = data_type_t::f32;
  params.dtypes.bias = data_type_t::f32;
  params.dtypes.output = data_type_t::f32;
  
  // Add ReLU6 post-op (common in MobileNet)
  conv_postop relu6_op;
  relu6_op.po_type = post_op_type_t::relu6;
  relu6_op.buff = nullptr;
  relu6_op.dtype = data_type_t::none;
  params.postop_.push_back(relu6_op);
  
  // Execute Depthwise Conv2D
  status_t status = conv_direct(
    input.data(),
    filter.data(),
    bias.data(),
    output.data(),
    true,  // is_weights_const (enables caching)
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```


### Example 4: Strided Conv2D for Downsampling

```cpp
int lowoha_strided_conv2d_example() {
  using namespace zendnnl::lowoha::conv;
  
  // Strided convolution for 2x downsampling
  int batch = 4;
  int in_h = 56, in_w = 56, in_c = 64;
  int filter_h = 3, filter_w = 3, out_c = 128;
  
  // Output dimensions with stride=2
  int out_h = (in_h + 2 - filter_h) / 2 + 1;  // 28
  int out_w = (in_w + 2 - filter_w) / 2 + 1;  // 28
  
  // Allocate tensors
  std::vector<float> input(batch * in_h * in_w * in_c, 1.0f);
  std::vector<float> filter(filter_h * filter_w * in_c * out_c, 0.5f);
  std::vector<float> bias(out_c, 0.0f);
  std::vector<float> output(batch * out_h * out_w * out_c, 0.0f);
  
  // Configure conv_params
  conv_params params;
  
  // Set dimensions
  params.dims.batch = batch;
  params.dims.in_height = in_h;
  params.dims.in_width = in_w;
  params.dims.in_channels = in_c;
  params.dims.filter_height = filter_h;
  params.dims.filter_width = filter_w;
  params.dims.out_channels = out_c;
  params.dims.out_height = out_h;
  params.dims.out_width = out_w;
  
  // Set stride=2 for downsampling
  params.stride_h = 2;
  params.stride_w = 2;
  params.pad_top = 1;
  params.pad_bottom = 1;
  params.pad_left = 1;
  params.pad_right = 1;
  
  // Set data types
  params.dtypes.input = data_type_t::f32;
  params.dtypes.filter = data_type_t::f32;
  params.dtypes.bias = data_type_t::f32;
  params.dtypes.output = data_type_t::f32;
  
  // Add ReLU post-op
  conv_postop relu_op;
  relu_op.po_type = post_op_type_t::relu;
  relu_op.buff = nullptr;
  relu_op.dtype = data_type_t::none;
  params.postop_.push_back(relu_op);
  
  // Execute Strided Conv2D
  status_t status = conv_direct(
    input.data(),
    filter.data(),
    bias.data(),
    output.data(),
    true,  // is_weights_const (enables caching)
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```


### Example 5: Dilated Conv2D (Atrous Convolution)

```cpp
int lowoha_dilated_conv2d_example() {
  using namespace zendnnl::lowoha::conv;
  
  // Dilated convolution (used in DeepLab, WaveNet)
  int batch = 2;
  int in_h = 64, in_w = 64, in_c = 128;
  int filter_h = 3, filter_w = 3, out_c = 128;
  
  // With dilation=2, effective kernel size is 5x5
  int dilation = 2;
  int effective_filter_h = (filter_h - 1) * dilation + 1;  // 5
  int effective_filter_w = (filter_w - 1) * dilation + 1;  // 5
  
  // Calculate output dimensions
  int out_h = (in_h + 4 - effective_filter_h) + 1;  // 64 (with pad=2)
  int out_w = (in_w + 4 - effective_filter_w) + 1;  // 64
  
  // Allocate tensors
  std::vector<float> input(batch * in_h * in_w * in_c, 1.0f);
  std::vector<float> filter(filter_h * filter_w * in_c * out_c, 0.5f);
  std::vector<float> bias(out_c, 0.0f);
  std::vector<float> output(batch * out_h * out_w * out_c, 0.0f);
  
  // Configure conv_params
  conv_params params;
  
  // Set dimensions
  params.dims.batch = batch;
  params.dims.in_height = in_h;
  params.dims.in_width = in_w;
  params.dims.in_channels = in_c;
  params.dims.filter_height = filter_h;
  params.dims.filter_width = filter_w;
  params.dims.out_channels = out_c;
  params.dims.out_height = out_h;
  params.dims.out_width = out_w;
  
  // Set stride and padding
  params.stride_h = 1;
  params.stride_w = 1;
  params.pad_top = 2;
  params.pad_bottom = 2;
  params.pad_left = 2;
  params.pad_right = 2;
  
  // Set dilation
  params.dilation_h = dilation;
  params.dilation_w = dilation;
  
  // Set data types
  params.dtypes.input = data_type_t::f32;
  params.dtypes.filter = data_type_t::f32;
  params.dtypes.bias = data_type_t::f32;
  params.dtypes.output = data_type_t::f32;
  
  // Add ReLU post-op
  conv_postop relu_op;
  relu_op.po_type = post_op_type_t::relu;
  relu_op.buff = nullptr;
  relu_op.dtype = data_type_t::none;
  params.postop_.push_back(relu_op);
  
  // Execute Dilated Conv2D
  status_t status = conv_direct(
    input.data(),
    filter.data(),
    bias.data(),
    output.data(),
    true,  // is_weights_const (enables caching)
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```


## Weight Caching and Reordering

One of the key features of LOWOHA Conv2D is **automatic filter reordering and caching**.

### How It Works

1. **First Execution**: 
   - Filters are reordered to the optimal format for the selected backend
   - Reordered filters are stored in an LRU cache
   - Cache key is generated from filter pointer, dimensions, data type, and backend

2. **Subsequent Executions**:
   - Cache is queried using the same key
   - If cache hit: reordered filters are retrieved (fast path)
   - If cache miss: filters are reordered and cached

3. **Cache Eviction**:
   - When cache is full, least recently used filters are evicted
   - Evicted filters are freed to make room for new entries

### Requirements

- Set `is_weights_const = true` to enable caching
- Filter pointer must remain stable across invocations
- Cache provides significant speedup for repeated inference


## Backend Selection

LowOHA Conv2D supports multiple backends. The backend is selected via `conv_params`:

```cpp
conv_params params;
params.algo = conv_algo_t::onednn_blocked;
```

**Available Algorithms:**
- `conv_algo_t::dynamic_dispatch` - Automatic backend selection based on heuristics
- `conv_algo_t::aocl_dlp_blocked` - Blocked AOCL DLP backend (not implemented)
- `conv_algo_t::onednn_blocked` - Blocked OneDNN backend
- `conv_algo_t::aocl_dlp` - AOCL DLP backend (not implemented)
- `conv_algo_t::onednn` - OneDNN backend
- `conv_algo_t::auto_tuner` - Auto Tuner (not implemented)
- `conv_algo_t::reference` - Reference implementation

### Supported LowOHA Conv2D Kernels

| Algo |       Kernel          |
|------|-----------------------|
| 0    | dynamic_dispatch      |
| 1    | aocl_dlp_blocked      |
| 2    | onednn_blocked        |
| 4    | aocl_dlp              |
| 5    | onednn                |
| 8    | auto_tuner            |
| 9    | reference             |

## Troubleshooting

### Common Issues

1. **Output dimensions mismatch:**
   - Verify padding and stride calculations
   - Use output dimension formulas provided above

2. **Performance not optimal:**
   - Ensure `is_weights_const = true` for repeated inference
   - Try different backends via `conv_algo_t`
   - Use BF16 on supported hardware

3. **Depthwise convolution errors:**
   - Verify `groups == in_channels`
   - Set `is_depthwise = true`
   - Ensure `out_channels = in_channels * depth_multiplier`

4. **Binary post-op shape mismatch:**
   - Verify binary buffer dims match output shape
   - Use `{1, 1, 1, C_out}` for channel-wise broadcast
