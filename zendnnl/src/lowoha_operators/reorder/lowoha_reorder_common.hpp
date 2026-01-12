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
 * @brief Structure to hold data types for reorder operation
 */
struct reorder_data_types_t {
  data_type_t src = data_type_t::none;     ///< Source data type
  data_type_t dst = data_type_t::none;     ///< Destination data type
};

/**
 * @brief Structure for reorder quantization parameters
 *
 * Used for quantization (bf16 -> int8/uint8) and dequantization (int8/uint8 -> bf16).
 * For quantization (s8):   int8_val = clamp(round(bf16_val / scale) + zero_point, -128, 127)
 * For quantization (u8):   uint8_val = clamp(round(bf16_val / scale) + zero_point, 0, 255)
 * For dequantization: bf16_val = (int_val - zero_point) * scale
 *
 * Currently supported:
 *   - scale: f32 only
 *   - zero_point: s32 only
 */
struct reorder_quant_params_t {
  /**
   * @brief Individual quantization parameter (scale or zero-point)
   *
   * data types and quantization granularities (per-tensor, per-channel, per-group).
   */
  struct quant_t {
    const void *buff;              ///< Pointer to quantization data buffer
    data_type_t dt;                ///< Data type of the buffer (f32 for scale, s32 for zp)
    std::vector<int64_t> dims;     ///< Dimensions of the quantization tensor

    /**
     * @brief Default constructor
     *
     * Default dims is empty, indicating per-tensor quantization (single value).
     * For per-channel: dims = {num_channels}
     * For per-group: dims = {num_groups, group_size} or similar
     */
    quant_t() : buff(nullptr), dt(data_type_t::none), dims() {}
  };

  quant_t scale;        ///< Scale factor (currently f32 only)
  quant_t zero_point;   ///< Zero point offset (currently s32 only)

  /**
   * @brief Default constructor
   */
  reorder_quant_params_t() : scale(), zero_point() {}
};

/**
 * @brief Main parameter structure for LOWOHA reorder operation
 */
struct lowoha_reorder_params_t {
  reorder_data_types_t dtypes;            ///< Data types for source and destination
  reorder_quant_params_t quant_params;    ///< Quantization parameters
  reorder_algo_t algo;                    ///< Selected algorithm
  uint64_t num_threads;                   ///< Number of threads (0 = auto)

  /**
   * @brief Default constructor
   */
  lowoha_reorder_params_t()
      : dtypes(), quant_params(), algo(reorder_algo_t::DT), num_threads(0) {}
};

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_REORDER_COMMON_HPP

