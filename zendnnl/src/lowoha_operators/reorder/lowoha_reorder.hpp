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

#ifndef _LOWOHA_REORDER_HPP
#define _LOWOHA_REORDER_HPP

#include <omp.h>
#include <cstddef>

#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace reorder {

using zendnnl::memory::status_t;
using zendnnl::memory::data_type_t;

/**
 * @brief Execute data type conversion (reorder) with quantization support
 *
 * Performs element-wise data type conversion from source to destination buffer.
 *
 * Supported conversions:
 * - BF16 ↔ S8/U8: Quantization/Dequantization with scale and zero-point
 * - F32  ↔ S8/U8: Quantization/Dequantization with scale and zero-point
 * - F32  ↔ BF16:  Type conversion with optional scale and zero-point
 *
 * Quantization formulas:
 * - Quantize:   int_val = clamp(round(src_val / scale) + zero_point, min, max)
 * - Dequantize: dst_val = (int_val - zero_point) * scale
 * - F32->BF16:  bf16_val = bf16((f32_val / scale) + zero_point)  [scale/zp optional]
 * - BF16->F32:  f32_val = (bf16_as_f32 - zero_point) * scale     [scale/zp optional]
 *
 * For F32 ↔ BF16 conversions:
 * - Scale and zero_point are OPTIONAL
 * - If not provided (buff = nullptr), simple type conversion is performed
 * - Default values when not provided: scale = 1.0, zero_point = 0
 *
 * Shape: [nelems] for 1D, [M, N] for 2D, [batch, M, N] for 3D (mandatory)
 * Strides: Optional for non-contiguous source memory
 * Granularity: Per-tensor, per-channel, or per-group (inferred from scale/zp dims)
 *
 * @param src    Source data buffer
 * @param dst    Destination data buffer (always contiguous output)
 * @param params Reorder parameters (data types, shape, quantization params, strides)
 *
 * @return status_t::success on success, status_t::failure otherwise
 *
 * @note Shape is mandatory; nelems is computed automatically from shape.
 * @note Destination is always written in contiguous format.
 * @note Buffers must not overlap.
 */
status_t reorder_direct(const void *src, void *dst,
                         reorder_params_t params);


/**
 * @brief RAII helper to temporarily set OpenMP thread count.
 *        Automatically restores original thread count on scope exit.
 *
 * @example
 *   {
 *       reorder_threadlimit guard(4);   // Set to 4 threads
 *       // ... parallel work ...
 *   }  // Restored to original
 *
 */
struct reorder_threadlimit {
  int old_num_threads;
  bool is_modified;

  reorder_threadlimit(int num_threads) : old_num_threads(0), is_modified(false) {
    if (num_threads != omp_get_max_threads()) {
      old_num_threads = omp_get_max_threads();
      omp_set_num_threads(num_threads);
      is_modified = true;
    }
  }

  ~reorder_threadlimit() {
    if (is_modified) {
      omp_set_num_threads(old_num_threads);
    }
  }
};


} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_REORDER_HPP
