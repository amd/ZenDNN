/*******************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

#ifndef LOWOHA_REORDER_UTILS_HPP
#define LOWOHA_REORDER_UTILS_HPP

#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"
#include "memory/memory_utils.hpp"
#include <cstring>
#include <cmath>
#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace reorder {

using zendnnl::memory::status_t;

//==============================================================================
// Granularity type enum
//==============================================================================

/**
 * @brief Quantization granularity type
 */
enum class granularity_type_t {
  invalid = -1,    ///< Invalid dims configuration
  per_tensor,      ///< All dims = 1 - single scale/zp for all elements
  per_channel,     ///< Different value per column (N values)
  per_group,       ///< G groups × N columns (G*N values)
  mixed            ///< Different granularity for scale vs zp
};

//==============================================================================
// Inline helper functions for data type conversion
//==============================================================================

/**
 * @brief Convert bf16 (stored as uint16_t) to float32
 */
static inline float bf16_to_float(uint16_t val) {
  uint32_t bits = static_cast<uint32_t>(val) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(float));
  return result;
}

/**
 * @brief Convert float32 to bf16 (stored as uint16_t) with round-to-nearest-even
 */
static inline uint16_t float_to_bf16(float val) {
  uint32_t bits;
  std::memcpy(&bits, &val, sizeof(float));
  uint32_t lsb = (bits >> 16) & 1;
  uint32_t rounding_bias = 0x7FFF + lsb;
  bits += rounding_bias;
  return static_cast<uint16_t>(bits >> 16);
}

//==============================================================================
// Inline helper functions for quantization parameter access
//==============================================================================

/**
 * @brief Extract scale value from quant_t at given index
 * @param scale_param The scale quant_t parameter
 * @param index Index into the scale array (0 for per-tensor)
 * @return Scale value (default 1.0f if buffer is null)
 */
static inline float get_scale_value(const reorder_quant_params_t::quant_t &scale_param, 
                                     size_t index = 0) {
  if (scale_param.buff == nullptr) {
    return 1.0f;
  }
  return static_cast<const float *>(scale_param.buff)[index];
}

/**
 * @brief Extract zero_point value from quant_t at given index
 * @param zp_param The zero_point quant_t parameter
 * @param index Index into the zero_point array (0 for per-tensor)
 * @return Zero point value (default 0 if buffer is null)
 */
static inline int get_zero_point_value(const reorder_quant_params_t::quant_t &zp_param,
                                        size_t index = 0) {
  if (zp_param.buff == nullptr) {
    return 0;
  }
  return static_cast<const int32_t *>(zp_param.buff)[index];
}

//==============================================================================
// Inline helper functions for scalar quantization/dequantization
//==============================================================================

/**
 * @brief Quantize a single bf16 value to int8
 * @param bf16_val Input bf16 value (as uint16_t)
 * @param scale Scale factor
 * @param zp Zero point
 * @return Quantized int8 value
 */
static inline int8_t quantize_bf16_to_s8_scalar(uint16_t bf16_val, float scale, int zp) {
  float f32_val = bf16_to_float(bf16_val);
  int32_t quantized = static_cast<int32_t>(std::round(f32_val / scale)) + zp;
  return static_cast<int8_t>(std::max(-128, std::min(127, quantized)));
}

/**
 * @brief Quantize a single bf16 value to uint8
 * @param bf16_val Input bf16 value (as uint16_t)
 * @param scale Scale factor
 * @param zp Zero point
 * @return Quantized uint8 value
 */
static inline uint8_t quantize_bf16_to_u8_scalar(uint16_t bf16_val, float scale, int zp) {
  float f32_val = bf16_to_float(bf16_val);
  int32_t quantized = static_cast<int32_t>(std::round(f32_val / scale)) + zp;
  return static_cast<uint8_t>(std::max(0, std::min(255, quantized)));
}

/**
 * @brief Dequantize a single int8 value to bf16
 * @param s8_val Input int8 value
 * @param scale Scale factor
 * @param zp Zero point
 * @return Dequantized bf16 value (as uint16_t)
 */
static inline uint16_t dequantize_s8_to_bf16_scalar(int8_t s8_val, float scale, int zp) {
  float f32_val = (static_cast<float>(s8_val) - zp) * scale;
  return float_to_bf16(f32_val);
}

/**
 * @brief Dequantize a single uint8 value to bf16
 * @param u8_val Input uint8 value
 * @param scale Scale factor
 * @param zp Zero point
 * @return Dequantized bf16 value (as uint16_t)
 */
static inline uint16_t dequantize_u8_to_bf16_scalar(uint8_t u8_val, float scale, int zp) {
  float f32_val = (static_cast<float>(u8_val) - zp) * scale;
  return float_to_bf16(f32_val);
}

//==============================================================================
// Inline helper functions for FP32 scalar quantization/dequantization
//==============================================================================

/**
 * @brief Quantize a single fp32 value to int8
 * @param f32_val Input fp32 value
 * @param scale Scale factor
 * @param zp Zero point
 * @return Quantized int8 value
 */
static inline int8_t quantize_f32_to_s8_scalar(float f32_val, float scale, int zp) {
  int32_t quantized = static_cast<int32_t>(std::round(f32_val / scale)) + zp;
  return static_cast<int8_t>(std::max(-128, std::min(127, quantized)));
}

/**
 * @brief Quantize a single fp32 value to uint8
 * @param f32_val Input fp32 value
 * @param scale Scale factor
 * @param zp Zero point
 * @return Quantized uint8 value
 */
static inline uint8_t quantize_f32_to_u8_scalar(float f32_val, float scale, int zp) {
  int32_t quantized = static_cast<int32_t>(std::round(f32_val / scale)) + zp;
  return static_cast<uint8_t>(std::max(0, std::min(255, quantized)));
}

/**
 * @brief Dequantize a single int8 value to fp32
 * @param s8_val Input int8 value
 * @param scale Scale factor
 * @param zp Zero point
 * @return Dequantized fp32 value
 */
static inline float dequantize_s8_to_f32_scalar(int8_t s8_val, float scale, int zp) {
  return (static_cast<float>(s8_val) - zp) * scale;
}

/**
 * @brief Dequantize a single uint8 value to fp32
 * @param u8_val Input uint8 value
 * @param scale Scale factor
 * @param zp Zero point
 * @return Dequantized fp32 value
 */
static inline float dequantize_u8_to_f32_scalar(uint8_t u8_val, float scale, int zp) {
  return (static_cast<float>(u8_val) - zp) * scale;
}

//==============================================================================
// Inline helper functions for granularity detection and index calculation
//==============================================================================

/**
 * @brief Helper to check if dims represent per-tensor (all dimensions are 1)
 * 
 * Per-tensor dims:
 *   - 1D: {1}
 *   - 2D: {1, 1}
 *   - 3D: {1, 1, 1}
 */
static inline bool is_per_tensor_dims(const std::vector<int64_t> &dims) {
  if (dims.empty()) return false;  // Empty dims is invalid in new convention
  for (int64_t d : dims) {
    if (d != 1) return false;
  }
  return true;
}

/**
 * @brief Helper to check if dims represent per-channel quantization
 * @param dims Quantization parameter dims
 * @param shape Tensor shape
 * 
 * Per-channel dims (different value for each column):
 *   - 1D: dims = {N} where dims[0] == shape[0] (N values)
 *   - 2D: dims = {1, N} where dims[1] == shape[1] (N values)
 *   - 3D: dims = {1, 1, N} where dims[2] == shape[2] (N values)
 */
static inline bool is_per_channel_dims(const std::vector<int64_t> &dims,
                                        const std::vector<int64_t> &shape) {
  if (dims.empty() || dims.size() != shape.size()) return false;
  
  if (dims.size() == 1) {
    // 1D: per-channel means dims[0] == shape[0] (N values)
    return dims[0] == shape[0];
  } else if (dims.size() == 2) {
    // 2D: per-channel means dims = {1, N}
    return dims[0] == 1 && dims[1] == shape[1];
  } else if (dims.size() == 3) {
    // 3D: per-channel means dims = {1, 1, N}
    return dims[0] == 1 && dims[1] == 1 && dims[2] == shape[2];
  }
  return false;
}

/**
 * @brief Helper to check if dims represent per-group quantization
 * @param dims Quantization parameter dims
 * @param shape Tensor shape
 * 
 * Per-group dims (G groups across rows, each group has N values):
 *   - 1D: Not supported (use per-channel instead)
 *   - 2D: dims = {G, N} where M % G == 0 and G > 1 (G*N total values)
 *   - 3D: dims = {1, G, N} where M % G == 0 and G > 1 (G*N total values)
 */
static inline bool is_per_group_dims(const std::vector<int64_t> &dims,
                                      const std::vector<int64_t> &shape) {
  if (dims.empty() || dims.size() != shape.size()) return false;
  
  // 1D: no per-group support
  if (dims.size() == 1) {
    return false;
  } else if (dims.size() == 2) {
    // 2D: per-group means dims = {G, N} where G > 1 (G*N total values)
    int64_t G = dims[0];
    return G > 1 && dims[1] == shape[1] && (shape[0] % G == 0);
  } else if (dims.size() == 3) {
    // 3D: per-group means dims = {1, G, N} where G > 1 (G*N total values)
    int64_t G = dims[1];
    return dims[0] == 1 && G > 1 && dims[2] == shape[2] && (shape[1] % G == 0);
  }
  return false;
}

/**
 * @brief Determine granularity type for a single quant_t parameter
 * @param dims Quantization parameter dims
 * @param shape Tensor shape
 * @return Granularity type enum value
 */
static inline granularity_type_t get_single_granularity(const std::vector<int64_t> &dims,
                                                         const std::vector<int64_t> &shape) {
  if (is_per_tensor_dims(dims)) {
    return granularity_type_t::per_tensor;
  } else if (is_per_channel_dims(dims, shape)) {
    return granularity_type_t::per_channel;
  } else if (is_per_group_dims(dims, shape)) {
    return granularity_type_t::per_group;
  }
  return granularity_type_t::invalid;  // Invalid dims configuration
}

/**
 * @brief Determine the combined granularity type from reorder params
 * @param params Reorder parameters
 * @return Granularity type enum value (mixed if scale and zp have different granularities)
 */
static inline granularity_type_t get_granularity_type(const reorder_params_t &params) {
  const auto &shape = params.src_shape;
  const auto &scale_dims = params.quant_params.scale.dims;
  const auto &zp_dims = params.quant_params.zero_point.dims;
  
  granularity_type_t scale_gran = get_single_granularity(scale_dims, shape);
  granularity_type_t zp_gran = get_single_granularity(zp_dims, shape);
  
  // If both are the same, return that granularity
  if (scale_gran == zp_gran) {
    return scale_gran;
  }
  
  // Different granularities for scale and zp
  return granularity_type_t::mixed;
}

/**
 * @brief Get number of groups from dims for per-group quantization
 * @param dims Quantization parameter dims
 * @return Number of groups (G), or 1 if not per-group
 */
static inline int64_t get_num_groups(const std::vector<int64_t> &dims) {
  // 1D: no per-group support, return 1
  if (dims.size() == 1) {
    return 1;
  } else if (dims.size() == 2) {
    return dims[0];  // 2D: dims = {G, N}
  } else if (dims.size() == 3) {
    return dims[1];  // 3D: dims = {1, G, N}
  }
  return 1;
}

/**
 * @brief Get scale index for per-channel/per-group quantization
 * @param params Reorder parameters
 * @param row Row index (M dimension, used for per-group)
 * @param col Column index (N dimension, used for per-channel)
 * @param N Number of columns
 * @return Index into the scale array
 */
static inline size_t get_scale_index(const reorder_params_t &params, 
                                      int64_t row, int64_t col, int64_t N) {
  const auto &dims = params.quant_params.scale.dims;
  const auto &shape = params.src_shape;
  
  if (is_per_tensor_dims(dims)) {
    return 0;
  } else if (is_per_channel_dims(dims, shape)) {
    // Per-channel: index by column
    if (dims.size() == 1) {
      return static_cast<size_t>(col);  // 1D: treat as column index
    }
    return static_cast<size_t>(col);  // 2D/3D: column index
  } else if (is_per_group_dims(dims, shape)) {
    // Per-group: index = group_idx * N + col
    int64_t G = get_num_groups(dims);
    int64_t M = params.M();
    int64_t group_size = M / G;
    int64_t group_idx = row / group_size;
    return static_cast<size_t>(group_idx * N + col);
  }
  return 0;
}

/**
 * @brief Get zero-point index for per-channel/per-group quantization
 * @param params Reorder parameters
 * @param row Row index (M dimension, used for per-group)
 * @param col Column index (N dimension, used for per-channel)
 * @param N Number of columns
 * @return Index into the zero_point array
 */
static inline size_t get_zp_index(const reorder_params_t &params,
                                   int64_t row, int64_t col, int64_t N) {
  const auto &dims = params.quant_params.zero_point.dims;
  const auto &shape = params.src_shape;
  
  if (is_per_tensor_dims(dims)) {
    return 0;
  } else if (is_per_channel_dims(dims, shape)) {
    // Per-channel: index by column
    if (dims.size() == 1) {
      return static_cast<size_t>(col);  // 1D: treat as column index
    }
    return static_cast<size_t>(col);  // 2D/3D: column index
  } else if (is_per_group_dims(dims, shape)) {
    // Per-group: index = group_idx * N + col
    int64_t G = get_num_groups(dims);
    int64_t M = params.M();
    int64_t group_size = M / G;
    int64_t group_idx = row / group_size;
    return static_cast<size_t>(group_idx * N + col);
  }
  return 0;
}

//==============================================================================
// Non-inline utility function declarations
//==============================================================================

/**
 * @brief Validates input parameters for reorder operation.
 */
status_t validate_reorder_inputs(const void *src, void *dst, size_t nelems,
                                  const reorder_params_t &params);

/**
 * @brief Convert data_type_t enum to string representation.
 */
const char *reorder_data_type_to_string(data_type_t dtype);

/**
 * @brief Convert reorder_algo_t enum to string representation.
 */
const char *reorder_algo_to_string(reorder_algo_t algo);

/**
 * @brief Select the optimal reorder algorithm based on parameters.
 */
reorder_algo_t select_reorder_algo(const reorder_params_t &params, size_t nelems);

/**
 * @brief Check if the given data type combination is supported for reorder.
 */
bool is_reorder_supported(data_type_t src_dtype, data_type_t dst_dtype);

/**
 * @brief Validate shape parameters for reorder.
 */
status_t validate_reorder_shape(const reorder_params_t &params);

/**
 * @brief Validate stride parameters.
 */
status_t validate_reorder_strides(const reorder_params_t &params);

/**
 * @brief Validate quantization parameters (scale and zero_point dimensions).
 * 
 * Checks that:
 * - Per-channel: dims match N (number of columns)
 * - Per-group: M is divisible by group_size and num_groups = M / group_size
 */
status_t validate_reorder_quant_params(const reorder_params_t &params);

/**
 * @brief Check if strides represent contiguous memory layout.
 */
bool is_stride_contiguous(const reorder_params_t &params);

/**
 * @brief Convert granularity_type_t to string for logging
 */
const char *granularity_to_string(granularity_type_t granularity);

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_REORDER_UTILS_HPP
