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

#include <cstddef>
#include <vector>

#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace reorder {

using zendnnl::memory::status_t;
using zendnnl::memory::data_type_t;

/**
 * @brief Multi-mode reorder entry point.
 *
 * Dispatches a reorder request to one of several pipelines based on
 * fields in @p params. New modes can be added over time without
 * changing this signature -- callers select a mode by populating the
 * corresponding sub-struct / flag in @ref reorder_params_t.
 *
 * Currently registered modes (checked in this order; the first match
 * wins):
 *
 *   - Weight prepack
 *       Selected when @c params.is_prepack is true. Reorders a weight
 *       matrix into the backend-specific blocked layout consumed by
 *       the matching matmul algo (aocl_dlp_blocked, libxsmm_blocked,
 *       onednn_blocked) named in @c params.prepack.algo. The caller
 *       must have queried the destination size via
 *       @c weight_prepack_size(params) and allocated at least that
 *       many bytes at @p dst.
 *
 *   - Dynamic quantization
 *       Selected when @c params.dynamic_quant is true. Computes
 *       scale / zero-point from the source data and (optionally)
 *       quantizes into @p dst.
 *
 *   - Standard reorder (default)
 *       Element-wise dtype conversion / static (de)quantization. See
 *       "Standard reorder" notes below for supported conversions and
 *       formulas.
 *
 * --- Standard reorder ---
 *
 * Supported conversions:
 * - BF16 ↔ S8/U8: Quantization/Dequantization with scale and zero-point
 * - F32  ↔ S8/U8: Quantization/Dequantization with scale and zero-point
 * - F32  ↔ BF16:  Type conversion with optional scale and zero-point
 * - F32  ↔ F16:   Type conversion with optional scale and zero-point
 * - BF16 ↔ F16:   Type conversion with optional scale and zero-point (via f32)
 *
 * Quantization formulas:
 * - Quantize:   int_val = clamp(round(src_val / scale) + zero_point, min, max)
 * - Dequantize: dst_val = (int_val - zero_point) * scale
 * - F32->BF16:  bf16_val = bf16((f32_val / scale) + zero_point)  [scale/zp optional]
 * - BF16->F32:  f32_val = (bf16_as_f32 - zero_point) * scale     [scale/zp optional]
 * - F32->F16:   f16_val = f16((f32_val / scale) + zero_point)    [scale/zp optional]
 * - F16->F32:   f32_val = (f16_as_f32 - zero_point) * scale      [scale/zp optional]
 * - BF16->F16:  f16_val = f16((bf16_as_f32 / scale) + zero_point) [scale/zp optional]
 * - F16->BF16:  bf16_val = bf16((f16_as_f32 - zero_point) * scale) [scale/zp optional]
 *
 * For float-only conversions (F32 ↔ BF16, F32 ↔ F16, BF16 ↔ F16):
 * - Scale and zero_point are OPTIONAL
 * - If not provided (buff = nullptr), simple type conversion is performed
 * - Default values when not provided: scale = 1.0, zero_point = 0
 *
 * Shape: [nelems] for 1D, [M, N] for 2D, [batch, M, N] for 3D (mandatory)
 * Strides: Optional for non-contiguous source memory
 * Granularity: Per-tensor, per-channel, or per-group (inferred from scale/zp dims)
 *
 * @param src    Source data buffer.
 * @param dst    Destination data buffer (caller-allocated and sized
 *               according to the selected mode's contract).
 * @param params Reorder parameters. The fields consulted depend on the
 *               selected mode (see per-mode descriptions above).
 *
 * @return status_t::success on success, status_t::failure otherwise.
 *
 * @note Mode-specific contracts (mandatory shape, alignment, capacity,
 *       etc.) are documented with each mode above.
 * @note Buffers must not overlap.
 */
status_t reorder_direct(const void *src, void *dst,
                        reorder_params_t &params);

/**
 * @brief Grouped per-token dynamic quantization for MoE/group GEMM sources.
 *
 * Treats the rows from all active source matrices as one logical collection
 * for scheduling. Each source matrix may live at a different base address, but
 * each row must be contiguous. Strides follow the same convention as
 * reorder_direct: empty means contiguous, otherwise 2D strides are
 * `{row_stride, col_stride}` in elements. For this per-token path
 * `col_stride` must be 1 and `row_stride >= K[i]`. Scale buffers are
 * per-expert and indexed by local row: `scale[i][m]`.
 *
 * Current implementation supports symmetric per-token bf16/f32 -> s8 dynamic
 * quantization. Callers own all destination and scale buffers.
 */
status_t group_dynamic_quant(
    const std::vector<const void *> &src,
    const std::vector<int> &M,
    const std::vector<int> &K,
    const std::vector<std::vector<int64_t>> &src_strides,
    const std::vector<void *> &dst,
    const std::vector<std::vector<int64_t>> &dst_strides,
    const std::vector<void *> &scale,
    const group_dynamic_quant_params_t &params);

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_REORDER_HPP
