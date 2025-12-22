
(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA Reorder Operator

## Overview

The **LowOHA Reorder Operator** is a high-performance, low-overhead data type conversion operator designed for **quantization and dequantization workloads**. It provides a direct API to convert data between BF16 and INT8/UINT8 formats with configurable scale and zero-point parameters.

Unlike the standard Reorder operator which uses the operator factory pattern, LowOHA Reorder provides a **function-based interface** optimized for:
- Minimal execution overhead
- Quantization (BF16 → INT8/UINT8)
- Dequantization (INT8/UINT8 → BF16)
- Direct control over quantization parameters


## Quantization/Dequantization Formulas

### Quantization (BF16 → INT8)

$$
\mathrm{int8} = \mathrm{clamp}(\mathrm{round}(\frac{\mathrm{bf16}}{\mathrm{scale}}) + \mathrm{zp}, -128, 127)
$$

### Quantization (BF16 → UINT8)

$$
\mathrm{uint8} = \mathrm{clamp}(\mathrm{round}(\frac{\mathrm{bf16}}{\mathrm{scale}}) + \mathrm{zp}, 0, 255)
$$

### Dequantization (INT8/UINT8 → BF16)

$$
\mathrm{bf16} = (\mathrm{int8} - \mathrm{zp}) \times \mathrm{scale}
$$


## Core API: `reorder_direct`

The primary interface for LowOHA Reorder is the `reorder_direct` function:

```cpp
status_t reorder_direct(
  const void *src,                  // Pointer to source data buffer
  void *dst,                        // Pointer to destination data buffer
  size_t nelems,                    // Number of elements to convert
  lowoha_reorder_params_t params    // Reorder parameters
);
```

### Return Value

| Value | Description |
|-------|-------------|
| `status_t::success` | Operation completed successfully |
| `status_t::failure` | Operation failed (invalid parameters, null pointers, etc.) |


## Parameters Structure

### `lowoha_reorder_params_t`

The main configuration structure for LowOHA Reorder:

```cpp
struct lowoha_reorder_params_t {
  reorder_data_types_t dtypes;        // Source and destination data types
  reorder_quant_params_t quant_params; // Quantization parameters (scale, zero_point)
  reorder_algo_t algo;                // Algorithm selection
  uint64_t num_threads;               // Number of threads (0 = auto)
};
```

### `reorder_data_types_t`

Specifies the source and destination data types:

```cpp
struct reorder_data_types_t {
  data_type_t src;  // Source data type
  data_type_t dst;  // Destination data type
};
```

**Supported Combinations:**

| Source Type | Destination Type | Operation |
|-------------|------------------|-----------|
| BF16 | S8 (INT8) | Quantization |
| S8 (INT8) | BF16 | Dequantization |
| BF16 | U8 (UINT8) | Quantization |
| U8 (UINT8) | BF16 | Dequantization |


### `reorder_quant_params_t`

Quantization parameters for scale and zero-point, using a flexible structure for future extensibility:

```cpp
struct reorder_quant_params_t {
  /**
   * Individual quantization parameter (scale or zero-point)
   */
  struct quant_t {
    const void *buff;              // Pointer to quantization data buffer
    data_type_t dt;                // Data type of the buffer
    std::vector<int64_t> dims;     // Dimensions of the quantization tensor
  };

  quant_t scale;        // Scale factor (currently f32 only)
  quant_t zero_point;   // Zero point offset (currently s32 only)
};
```

**Currently Supported Data Types:**

| Parameter | Supported Type | Description |
|-----------|---------------|-------------|
| `scale` | `f32` | Scale factor (must be positive and finite) |
| `zero_point` | `s32` | Zero point offset (must be in [-128, 127] for INT8, [0, 255] for UINT8) |

**Quantization Granularities (via `dims`):**

| Granularity | `dims` Value | Description |
|-------------|--------------|-------------|
| Per-tensor | `{}` (empty) | Single scale/zp for entire tensor (currently supported) |
| Per-channel | `{num_channels}` | One scale/zp per channel (future) |
| Per-group | `{num_groups, group_size}` | Grouped quantization (future) |

**Default Values (when `buff` is nullptr):**
- `scale`: 1.0f
- `zero_point`: 0


### `reorder_algo_t`

Algorithm selection for the reorder operation:

```cpp
enum class reorder_algo_t : int {
  none = -1,        // No specific algorithm
  DT = 0,           // Decision tree based algorithm selection
  native = 1,       // Native vectorized implementation (AVX512)
  reference = 2,    // Reference scalar implementation
  algo_count        // Number of algorithms (must be last)
};
```

**Algorithm Selection:**

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `native` | AVX512 vectorized implementation | Large buffers |
| `reference` | Scalar implementation | Small buffers or debugging |
| `DT` | Decision tree based selection | General use (recommended) |


## Usage Examples

### Example 1: BF16 to INT8 Quantization

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int bf16_to_int8_quantization_example() {
  using namespace zendnnl::lowoha;
  
  constexpr size_t nelems = 1024;
  float scale = 0.5f;
  int32_t zero_point = 0;
  
  // Allocate buffers
  // BF16 data is stored as uint16_t
  std::vector<uint16_t> input_bf16(nelems);
  std::vector<int8_t> output_int8(nelems);
  
  // Initialize BF16 input (example: fill with some values)
  for (size_t i = 0; i < nelems; ++i) {
    float val = static_cast<float>(i) * 0.1f - 50.0f;
    // Convert float to bf16 (simplified)
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(float));
    input_bf16[i] = static_cast<uint16_t>(bits >> 16);
  }
  
  // Configure reorder parameters
  lowoha_reorder_params_t params;
  params.dtypes.src = data_type_t::bf16;
  params.dtypes.dst = data_type_t::s8;
  params.quant_params.scale.buff = &scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.zero_point.buff = &zero_point;
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.algo = reorder_algo_t::DT;
  
  // Execute quantization
  status_t status = reorder_direct(
    input_bf16.data(),
    output_int8.data(),
    nelems,
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 2: INT8 to BF16 Dequantization

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int int8_to_bf16_dequantization_example() {
  using namespace zendnnl::lowoha;
  
  constexpr size_t nelems = 1024;
  float scale = 0.5f;
  int32_t zero_point = 0;
  
  // Allocate buffers
  std::vector<int8_t> input_int8(nelems);
  std::vector<uint16_t> output_bf16(nelems);
  
  // Initialize INT8 input
  for (size_t i = 0; i < nelems; ++i) {
    input_int8[i] = static_cast<int8_t>((i % 256) - 128);
  }
  
  // Configure reorder parameters
  lowoha_reorder_params_t params;
  params.dtypes.src = data_type_t::s8;
  params.dtypes.dst = data_type_t::bf16;
  params.quant_params.scale.buff = &scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.zero_point.buff = &zero_point;
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.algo = reorder_algo_t::DT;
  
  // Execute dequantization
  status_t status = reorder_direct(
    input_int8.data(),
    output_bf16.data(),
    nelems,
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 3: BF16 to UINT8 Quantization

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int bf16_to_uint8_quantization_example() {
  using namespace zendnnl::lowoha;
  
  constexpr size_t nelems = 1024;
  float scale = 0.5f;
  int32_t zero_point = 128;  // Typical zero-point for unsigned quantization
  
  // Allocate buffers
  // BF16 data is stored as uint16_t
  std::vector<uint16_t> input_bf16(nelems);
  std::vector<uint8_t> output_uint8(nelems);
  
  // Initialize BF16 input (example: fill with some values)
  for (size_t i = 0; i < nelems; ++i) {
    float val = static_cast<float>(i) * 0.1f - 50.0f;
    // Convert float to bf16 (simplified)
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(float));
    input_bf16[i] = static_cast<uint16_t>(bits >> 16);
  }
  
  // Configure reorder parameters
  lowoha_reorder_params_t params;
  params.dtypes.src = data_type_t::bf16;
  params.dtypes.dst = data_type_t::u8;
  params.quant_params.scale.buff = &scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.zero_point.buff = &zero_point;
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.algo = reorder_algo_t::DT;
  
  // Execute quantization
  status_t status = reorder_direct(
    input_bf16.data(),
    output_uint8.data(),
    nelems,
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 4: UINT8 to BF16 Dequantization

```cpp
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

int uint8_to_bf16_dequantization_example() {
  using namespace zendnnl::lowoha;
  
  constexpr size_t nelems = 1024;
  float scale = 0.5f;
  int32_t zero_point = 128;  // Typical zero-point for unsigned quantization
  
  // Allocate buffers
  std::vector<uint8_t> input_uint8(nelems);
  std::vector<uint16_t> output_bf16(nelems);
  
  // Initialize UINT8 input
  for (size_t i = 0; i < nelems; ++i) {
    input_uint8[i] = static_cast<uint8_t>(i % 256);
  }
  
  // Configure reorder parameters
  lowoha_reorder_params_t params;
  params.dtypes.src = data_type_t::u8;
  params.dtypes.dst = data_type_t::bf16;
  params.quant_params.scale.buff = &scale;
  params.quant_params.scale.dt = data_type_t::f32;
  params.quant_params.zero_point.buff = &zero_point;
  params.quant_params.zero_point.dt = data_type_t::s32;
  params.algo = reorder_algo_t::DT;
  
  // Execute dequantization
  status_t status = reorder_direct(
    input_uint8.data(),
    output_bf16.data(),
    nelems,
    params
  );
  
  return (status == status_t::success) ? 0 : -1;
}
```
