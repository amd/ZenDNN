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
#include <cstdint>
#include <vector>

#include "common/zendnnl_api.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace reorder {

using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;

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
 * - F16  ↔ S8/U8: Quantization/Dequantization with scale and zero-point
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
ZENDNNL_API status_t reorder_direct(
        const void *src, void *dst, reorder_params_t &params);

/**
 * @brief Direct fused per-token BF16-to-S8 dynamic quantization kernel.
 *
 * Quantizes a contiguous row-major @c [M,N] BF16 matrix. For each row, the
 * kernel computes one F32 scale from the largest finite absolute input value:
 *
 * @code
 * scales[m] = max(abs(src[m, :])) / 127
 * dst[m, n] = nearbyint(src[m, n] / scales[m])
 * @endcode
 *
 * The scale is lower-bounded by @c 1e-10f. Under the default masked
 * floating-point exception environment, non-finite inputs do not contribute
 * to the scale and are quantized to zero.
 *
 * This low-level entry point bypasses @ref reorder_direct validation and ISA
 * dispatch. The caller must ensure that AVX-512F, AVX-512BW, and AVX-512VL are
 * available. Use @c zendnnl::common::zendnnl_platform_info() to query support,
 * or use @ref reorder_direct when a portable fallback is required.
 *
 * @param src    Input buffer containing at least @c M*N raw BF16 bit patterns.
 * @param dst    Output buffer with capacity for at least @c M*N S8 values.
 * @param scales Output buffer with capacity for at least @c M F32 scales.
 * @param M      Number of rows (tokens); must be positive.
 * @param N      Number of contiguous elements per row; must be positive.
 *
 * @note The kernel uses the active OpenMP thread configuration.
 * @note Quantization follows the active floating-point rounding mode; the
 *       usual @c FE_TONEAREST mode gives round-to-nearest-even behavior.
 * @note Input and output buffers must not overlap.
 */
ZENDNNL_API void dynamic_per_token_quant_bf16_s8_native(
        const uint16_t *src, int8_t *dst, float *scales, int64_t M, int64_t N);

/**
 * @brief Grouped dynamic quantization for independent source matrices.
 *
 * Treats work from all active source matrices as one logical collection for
 * scheduling. Each source matrix may live at a different base address, but
 * each row must be contiguous. Strides follow the same convention as
 * reorder_direct: empty means contiguous, otherwise 2D strides are
 * `{row_stride, col_stride}` in elements. `col_stride` must be 1 and
 * `row_stride >= K[i]`.
 *
 * Scale layout is selected by group_dynamic_quant_params_t::granularity:
 * per-token uses `scale[i][m]`, per-channel uses `scale[i][k]`, and
 * per-group uses `scale[i][m * num_groups + g]`. Callers own all destination
 * and scale buffers.
 */
status_t group_dynamic_quant(const std::vector<const void *> &src,
        const std::vector<int> &M, const std::vector<int> &K,
        const std::vector<std::vector<int64_t>> &src_strides,
        const std::vector<void *> &dst,
        const std::vector<std::vector<int64_t>> &dst_strides,
        const std::vector<void *> &scale,
        const group_dynamic_quant_params_t &params);

/**
 * @brief Grouped reorder — apply @ref reorder_direct to a group of
 *        independent reorder operations.
 *
 * Thin wrapper that calls @c reorder_direct(src[i], dst[i], params[i])
 * for each op @c i in @c [0, params.size()) in a sequential loop. Every
 * mode @ref reorder_direct supports is available per-op, selected by that
 * op's own @ref reorder_params_t, so a single group may freely mix:
 *   - Weight prepack            (@c params[i].is_prepack == true)
 *   - Dynamic quantization      (@c params[i].dynamic_quant == true)
 *   - Standard reorder / (de)quant / type conversion (default)
 *
 * This is the grouped building block the higher-level weight-prepack
 * grouping is composed from: a caller that wants to pre-pack a group of
 * expert weights builds one prepack-mode @ref reorder_params_t per expert
 * and hands the batch here.
 *
 * Threading: @ref reorder_direct parallelises internally, so the outer
 * loop here stays sequential to avoid nested OpenMP regions.
 *
 * Contract:
 *   - @p src, @p dst and @p params must all have the same length
 *     (@c params.size()); a mismatch or an empty group returns failure.
 *   - @c params is taken by non-const reference because the dynamic-quant
 *     mode writes computed scale/zero-point back into each op's params
 *     (mirrors @ref reorder_direct's signature).
 *   - @c dst[i] may be @c nullptr only where the selected per-op mode
 *     permits it (e.g. compute-only dynamic quant); otherwise the per-op
 *     validation inside @ref reorder_direct rejects it.
 *   - On the first per-op failure the loop stops and returns that op's
 *     status. Ops already completed keep their finished state in the
 *     caller-owned buffers (no rollback).
 *
 * @param src    Per-op source buffers.
 * @param dst    Per-op destination buffers (caller-allocated/sized per the
 *               selected per-op mode).
 * @param params Per-op reorder parameters (one @ref reorder_params_t each).
 *
 * @return status_t::success when every op succeeded; otherwise the status
 *         of the first op that failed.
 */
status_t group_reorder(const std::vector<const void *> &src,
        const std::vector<void *> &dst, std::vector<reorder_params_t> &params);

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_REORDER_HPP
