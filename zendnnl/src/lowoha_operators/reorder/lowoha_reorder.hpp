/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

using zendnnl::memory::status_t;
using zendnnl::memory::data_type_t;

/**
 * @brief Execute data type conversion (reorder) between buffers
 *
 * This function performs element-wise data type conversion from source buffer
 * to destination buffer. Currently supports:
 * - BF16 to INT8 (quantization)
 * - INT8 to BF16 (dequantization)
 * - BF16 to UINT8 (quantization)
 * - UINT8 to BF16 (dequantization)
 *
 * For quantization (bf16 -> int8):
 *   int8_val = clamp(round(bf16_val / scale) + zero_point, -128, 127)
 *
 * For quantization (bf16 -> uint8):
 *   uint8_val = clamp(round(bf16_val / scale) + zero_point, 0, 255)
 *
 * For dequantization (int8/uint8 -> bf16):
 *   bf16_val = (int8_val - zero_point) * scale
 *
 * @param src       Pointer to source data buffer
 * @param dst       Pointer to destination data buffer
 * @param nelems    Number of elements to convert
 * @param params    Reorder parameters including data types and quantization params
 *
 * @return status_t::success on successful execution, status_t::failure otherwise
 *
 * @note The source and destination buffers must not overlap.
 * @note For bf16 data, the buffer should be uint16_t* type-punned.
 * @note For int8 data, the buffer should be int8_t*.
 * @note For uint8 data, the buffer should be uint8_t*.
 *
 * @example
 * @code
 * // BF16 to INT8 quantization
 * float scale = 0.5f;
 * int32_t zero_point = 0;
 *
 * lowoha_reorder_params_t params;
 * params.dtypes.src = data_type_t::bf16;
 * params.dtypes.dst = data_type_t::s8;
 * params.quant_params.scale.buff = &scale;
 * params.quant_params.scale.dt = data_type_t::f32;
 * params.quant_params.zero_point.buff = &zero_point;
 * params.quant_params.zero_point.dt = data_type_t::s32;
 *
 * status_t status = reorder_direct(bf16_buffer, int8_buffer, nelems, params);
 *
 * // INT8 to BF16 dequantization
 * lowoha_reorder_params_t dequant_params;
 * dequant_params.dtypes.src = data_type_t::s8;
 * dequant_params.dtypes.dst = data_type_t::bf16;
 * dequant_params.quant_params.scale.buff = &scale;
 * dequant_params.quant_params.scale.dt = data_type_t::f32;
 * dequant_params.quant_params.zero_point.buff = &zero_point;
 * dequant_params.quant_params.zero_point.dt = data_type_t::s32;
 *
 * status = reorder_direct(int8_buffer, bf16_buffer, nelems, dequant_params);
 *
 * // BF16 to UINT8 quantization
 * int32_t u8_zero_point = 128;
 * lowoha_reorder_params_t u8_quant_params;
 * u8_quant_params.dtypes.src = data_type_t::bf16;
 * u8_quant_params.dtypes.dst = data_type_t::u8;
 * u8_quant_params.quant_params.scale.buff = &scale;
 * u8_quant_params.quant_params.scale.dt = data_type_t::f32;
 * u8_quant_params.quant_params.zero_point.buff = &u8_zero_point;
 * u8_quant_params.quant_params.zero_point.dt = data_type_t::s32;
 *
 * status = reorder_direct(bf16_buffer, uint8_buffer, nelems, u8_quant_params);
 *
 * // UINT8 to BF16 dequantization
 * lowoha_reorder_params_t u8_dequant_params;
 * u8_dequant_params.dtypes.src = data_type_t::u8;
 * u8_dequant_params.dtypes.dst = data_type_t::bf16;
 * u8_dequant_params.quant_params.scale.buff = &scale;
 * u8_dequant_params.quant_params.scale.dt = data_type_t::f32;
 * u8_dequant_params.quant_params.zero_point.buff = &u8_zero_point;
 * u8_dequant_params.quant_params.zero_point.dt = data_type_t::s32;
 *
 * status = reorder_direct(uint8_buffer, bf16_buffer, nelems, u8_dequant_params);
 * @endcode
 */
status_t reorder_direct(const void *src, void *dst, size_t nelems,
                         lowoha_reorder_params_t params);

} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_REORDER_HPP

