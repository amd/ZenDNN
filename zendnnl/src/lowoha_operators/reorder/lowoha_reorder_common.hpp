/********************************************************************************
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

#ifndef _LOWOHA_REORDER_COMMON_HPP
#define _LOWOHA_REORDER_COMMON_HPP

#include "memory/memory_utils.hpp"
#include <vector>
#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace reorder {

using namespace zendnnl::memory;

/**
 * @brief Supported reorder algorithms
 */
enum class reorder_algo_t : int {
  none = -1,          ///< No specific algorithm
  DT = 0,             ///< Decision tree based algorithm selection
  native = 1,         ///< Native vectorized implementation (AVX512)
  reference = 2,      ///< Reference scalar implementation
  algo_count          ///< Number of algorithms (must be last)
};

/**
 * @brief Structure for reorder quantization parameters
 *
 * Used for quantization (bf16 -> int8/uint8) and dequantization (int8/uint8 -> bf16).
 * For quantization (s8):   int8_val = clamp(round(bf16_val / scale) + zero_point, -128, 127)
 * For quantization (u8):   uint8_val = clamp(round(bf16_val / scale) + zero_point, 0, 255)
 * For dequantization: bf16_val = (int_val - zero_point) * scale
 *
 * Granularity convention for dims (must match tensor dimensionality):
 *
 * For 1D tensor (shape = [N]):
 *   - per-tensor:  dims = {1}       (1 value for all elements)
 *   - per-channel: dims = {N}       (N values, one per element)
 *
 * For 2D tensor (shape = [M, N]):
 *   - per-tensor:  dims = {1, 1}    (1 value for all elements)
 *   - per-channel: dims = {1, N}    (N values, one per column)
 *   - per-group:   dims = {G, N}    (G*N values, M % G == 0, each group has N values)
 *
 * For 3D tensor (shape = [batch, M, N]):
 *   - per-tensor:  dims = {1, 1, 1} (1 value for all elements)
 *   - per-channel: dims = {1, 1, N} (N values, one per column)
 *   - per-group:   dims = {1, G, N} (G*N values, M % G == 0, each group has N values)
 *
 * Per-group indexing: index = group_idx * N + col, where group_idx = row / (M / G)
 *
 * Currently supported:
 *   - scale: f32 only
 *   - zero_point: s32 only
 */
struct reorder_quant_params_t {
  /**
   * @brief Individual quantization parameter (scale or zero-point)
   *
   * Supports different data types and quantization granularities.
   */
  struct quant_t {
    const void *buff;              ///< Pointer to quantization data buffer
    data_type_t dt;                ///< Data type of the buffer (f32 for scale, s32 for zp)
    std::vector<int64_t> dims;     ///< Dimensions matching tensor dimensionality

    /**
     * @brief Default constructor
     */
    quant_t() : buff(nullptr), dt(data_type_t::none), dims() {}
  };

  quant_t scale;                              ///< Scale factor (currently f32 only)
  quant_t zero_point;                         ///< Zero point offset (currently s32 only)

  /**
   * @brief Default constructor
   */
  reorder_quant_params_t() : scale(), zero_point() {}
};

/**
 * @brief Main parameter structure for LOWOHA reorder operation
 *
 * Shape format (vector of int64_t):
 *   - Size 1: 1D array [nelems]
 *   - Size 2: 2D matrix [M, N]
 *   - Size 3: 3D batched matrix [batch, M, N]
 *
 * Strides format (vector of int64_t):
 *   - Empty: contiguous memory
 *   - Size 1: 1D array with stride
 *   - Size 2: 2D matrix with strides [stride_M, stride_N]
 *   - Size 3: 3D batched with strides [stride_batch, stride_M, stride_N]
 *
 * @note src_shape and dst_shape must be identical. An error will be thrown if they differ.
 * @note dst_strides is reserved for future implementation and is currently not supported.
 *       The destination is always written in contiguous format.
 */
struct reorder_params_t {
  data_type_t src_dtype;                  ///< Source data type
  data_type_t dst_dtype;                  ///< Destination data type
  reorder_quant_params_t quant_params;    ///< Quantization parameters
  reorder_algo_t algo;                    ///< Selected algorithm
  uint64_t num_threads;                   ///< Number of threads (0 = auto)
  std::vector<int64_t> src_shape;         ///< Source shape: [nelems] or [M, N] or [batch, M, N]
  std::vector<int64_t> dst_shape;         ///< Destination shape: must match src_shape
  std::vector<int64_t> src_strides;       ///< Source strides for non-contiguous memory access
  std::vector<int64_t> dst_strides;       ///< Destination strides (reserved for future, not currently supported)

  /**
   * @brief Default constructor
   */
  reorder_params_t()
      : src_dtype(data_type_t::none), dst_dtype(data_type_t::none),
        quant_params(), algo(reorder_algo_t::DT), num_threads(0),
        src_shape(), dst_shape(), src_strides(), dst_strides() {}
  
  /**
   * @brief Check if this is a 1D shape
   */
  bool is_1d() const {
    return src_shape.size() == 1;
  }

  /**
   * @brief Check if this is a 2D shape
   */
  bool is_2d() const {
    return src_shape.size() == 2;
  }
  
  /**
   * @brief Check if this is a 3D shape (batched)
   */
  bool is_3d() const {
    return src_shape.size() == 3;
  }

  /**
   * @brief Check if shape is provided and valid
   */
  bool is_shaped() const {
    if (src_shape.empty()) return false;
    for (auto dim : src_shape) {
      if (dim <= 0) return false;
    }
    return true;
  }

  /**
   * @brief Check if src_shape and dst_shape match
   * @return true if shapes are identical, false otherwise
   */
  bool shapes_match() const {
    return src_shape == dst_shape;
  }

  /**
   * @brief Get total number of elements
   */
  int64_t nelems() const {
    if (src_shape.empty()) return 0;
    int64_t n = 1;
    for (auto dim : src_shape) {
      n *= dim;
    }
    return n;
  }

  /**
   * @brief Get batch dimension (1 for 1D/2D)
   */
  int64_t batch() const {
    return is_3d() ? src_shape[0] : 1;
  }

  /**
   * @brief Get M dimension (rows)
   */
  int64_t M() const {
    if (is_1d()) return src_shape[0];
    if (is_2d()) return src_shape[0];
    if (is_3d()) return src_shape[1];
    return 0;
  }

  /**
   * @brief Get N dimension (columns)
   */
  int64_t N() const {
    if (is_1d()) return 1;
    if (is_2d()) return src_shape[1];
    if (is_3d()) return src_shape[2];
    return 0;
  }
  
  /**
   * @brief Check if source strides are specified
   */
  bool has_src_strides() const {
    return !src_strides.empty();
  }

  /**
   * @brief Check if destination strides are specified (reserved for future)
   */
  bool has_dst_strides() const {
    return !dst_strides.empty();
  }
  
  /**
   * @brief Check if source memory layout is contiguous
   */
  bool is_src_contiguous() const {
    if (!has_src_strides() || !is_shaped()) {
      return true;  // No strides or no shape means contiguous
    }
    
    // Check if strides match contiguous layout
    if (src_strides.size() == 1) {
      return src_strides[0] == 1;
    } else if (src_strides.size() == 2 && is_2d()) {
      return src_strides[0] == N() && src_strides[1] == 1;
    } else if (src_strides.size() == 3 && is_3d()) {
      return src_strides[0] == (M() * N()) && 
             src_strides[1] == N() && 
             src_strides[2] == 1;
    }
    return false;
  }
};

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_REORDER_COMMON_HPP
