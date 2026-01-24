
(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA Reorder Operator

## Overview

The **LowOHA Reorder Operator** is a high-performance, low-overhead data type conversion operator designed for **quantization, dequantization, and data type conversion in workloads**. It provides a direct API to convert data between BF16/FP32 and INT8/UINT8 formats, as well as between FP32 and BF16 formats, with configurable scale and zero-point parameters.

Unlike the standard Reorder operator which uses the operator factory pattern, LowOHA Reorder provides a **function-based interface** optimized for:
- Minimal execution overhead
- Quantization (BF16/FP32 → INT8/UINT8)
- Dequantization (INT8/UINT8 → BF16/FP32)
- Data type conversion (FP32 ⇔ BF16)
- Per-tensor, per-channel, and per-group quantization granularities
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


## Core API: `reorder_direct`

The primary interface for LowOHA Reorder is the `reorder_direct` function:

```cpp
status_t reorder_direct(
  const void *src,                      // Pointer to source data buffer
  void *dst,                            // Pointer to destination data buffer
  reorder_params_t params        // Reorder parameters
);
```

### Return Value

| Value | Description |
|-------|-------------|
| `status_t::success` | Operation completed successfully |
| `status_t::failure` | Operation failed (invalid parameters, null pointers, etc.) |


## Parameters Structure

### `reorder_params_t`

The main configuration structure for LowOHA Reorder:

```cpp
struct reorder_params_t {
  data_type_t src_dtype;                  // Source data type
  data_type_t dst_dtype;                  // Destination data type
  reorder_quant_params_t quant_params;    // Quantization parameters (scale, zero_point)
  reorder_algo_t algo;                    // Algorithm selection
  uint64_t num_threads;                   // Number of threads (0 = auto)
  std::vector<int64_t> src_shape;         // Source shape: [N] or [M, N] or [batch, M, N] (mandatory)
  std::vector<int64_t> dst_shape;         // Destination shape: must match src_shape (mandatory)
  std::vector<int64_t> src_strides;       // Source strides for non-contiguous memory (optional)
  std::vector<int64_t> dst_strides;       // Destination strides (reserved for future, not currently supported)
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
| FP32 | BF16 | Data Type Conversion |
| BF16 | FP32 | Data Type Conversion |


### `reorder_quant_params_t`

Quantization parameters for scale and zero-point:

```cpp
struct reorder_quant_params_t {
  struct quant_t {
    const void *buff;              // Pointer to quantization data buffer
    data_type_t dt;                // Data type of the buffer
    std::vector<int64_t> dims;     // Dimensions (mandatory, must match tensor dims)
  };

  quant_t scale;        // Scale factor (f32 only)
  quant_t zero_point;   // Zero point offset (s32 only)
};
```

**Currently Supported Data Types:**

| Parameter | Supported Type | Description |
|-----------|---------------|-------------|
| `scale` | `f32` | Scale factor (must be positive and finite) |
| `zero_point` | `s32` | Zero point offset |

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
| Per-channel | `{1, N}` | N | Different scale/zp for each column |
| Per-group | `{G, N}` | G × N | G groups across rows, each with N values |

**Per-group constraint:** M must be divisible by G (M % G == 0)

### 3D Tensor (shape = [batch, M, N])

| Granularity | dims | Total Values | Description |
|-------------|------|--------------|-------------|
| Per-tensor | `{1, 1, 1}` | 1 | Single scale/zp for entire tensor |
| Per-channel | `{1, 1, N}` | N | Different scale/zp for each column |
| Per-group | `{1, G, N}` | G × N | G groups across rows, each with N values |

**Per-group constraint:** M must be divisible by G (M % G == 0)

### Per-Group Index Calculation

For per-group quantization with dims `{G, N}`:
- `group_size = M / G`
- `group_idx = row / group_size`
- `index = group_idx * N + col`


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


## Validation

The operator performs the following validations:

1. **Null pointer checks:** Source and destination buffers must not be null
2. **Shape validation:** src_shape and dst_shape must be non-empty with all positive dimensions
3. **Shape matching:** src_shape and dst_shape must be identical (error thrown if different)
4. **Data type validation:** Source/destination type combination must be supported
5. **Scale validation:** Must be finite (for f32)
6. **Zero-point validation:** Must be within valid range for target type
7. **Dims validation:** Must match tensor dimensionality and follow granularity rules
8. **Per-group validation:** M must be divisible by G
9. **Destination strides:** dst_strides must be empty (strided destination not currently supported)


## Implementation Support Matrix

The following table shows which combinations have optimized (AVX512) vs reference implementations:

### BF16/FP32 ↔ S8/U8 and FP32 ↔ BF16

| Granularity | Source Contiguous | Source Strided (last_stride=1) | Source Strided (other) |
|-------------|-------------------|--------------------------------|------------------------|
| Per-tensor | ✅ Optimal | ✅ Optimal | ⚙️ Reference |
| Per-channel | ⚙️ Reference | ⚙️ Reference | ⚙️ Reference |
| Per-group | ⚙️ Reference | ⚙️ Reference | ⚙️ Reference |

**Legend:**
- ✅ **Optimal:** AVX512 vectorized implementation for best performance
- ⚙️ **Reference:** Scalar implementation (functionally correct, lower throughput)

**Notes:**
- Strided source memory with `stride_N = 1` (last dimension contiguous) can use optimal path for per-tensor
- Per-channel and per-group granularities currently use reference implementation
- The `DT` algorithm automatically selects the best available implementation
- Destination memory is always written contiguously (strided destination not currently supported)
- FP32 ↔ BF16 conversion supports both simple (no scale/zp) and scaled conversions


## Performance Considerations

- **Algorithm Selection:** Use `DT` (default) for automatic selection based on buffer size and configuration
- **Vectorization:** `native` algorithm uses AVX512 for large buffers (≥64 elements) with supported configurations
- **Threading:** Set `num_threads` to control parallelism (0 = use all available)
- **Source Memory Layout:** Contiguous source memory is fastest; strided source with last_stride=1 can still use optimal path
- **Destination Memory:** Always written contiguously (strided destination not currently supported)
- **Granularity:** Per-tensor is fastest with optimal support; per-channel/per-group use reference implementation
- **FP32 ↔ BF16 Conversion:** Simple conversion (no scale/zp) is fastest; scaled conversion follows the same granularity performance characteristics as quantization/dequantization
