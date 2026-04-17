/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef _REORDER_DTYPE_DISPATCH_HPP
#define _REORDER_DTYPE_DISPATCH_HPP

#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"
#include <cstddef>

namespace zendnnl {
namespace lowoha {
namespace reorder {

void reorder_wrapper(const void *src, void *dst, size_t nelems,
                     const reorder_params_t &params,
                     reorder_algo_t algo);

void reorder_granular_scaler_impl_2d(const void *src, void *dst,
                                      const reorder_params_t &params);

void reorder_granular_scaler_impl_3d(const void *src, void *dst,
                                      const reorder_params_t &params);

/**
 * @brief Helper function to get element size for a given data type
 */
inline size_t get_dtype_size(data_type_t dtype) {
  switch (dtype) {
    case data_type_t::f32:  return sizeof(float);
    case data_type_t::bf16: return sizeof(uint16_t);
    case data_type_t::s8:   return sizeof(int8_t);
    case data_type_t::u8:   return sizeof(uint8_t);
    default:                return 1;
  }
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // _REORDER_DTYPE_DISPATCH_HPP
