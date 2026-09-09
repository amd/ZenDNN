/*******************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _LOWOHA_PREPACK_HPP
#define _LOWOHA_PREPACK_HPP

#include <cstddef>
#include <cstdint>

#include "common/op_config.hpp"
#include "memory/memory_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace reorder {

using zendnnl::common::matmul_algo_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;

// Forward declaration: the public prepack API
struct reorder_params_t;

// Algo selection reuses zendnnl::common::matmul_algo_t directly
// (common/op_config.hpp). The prepacked output produced
// here is exactly the layout consumed by the matching matmul algo.
//
// Supported algos:
//   - matmul_algo_t::aocl_dlp_blocked
//       AOCL DLP blocked weight layout. Consumed by the single-op /
//       group ALGO 1 AOCL DLP path. To consume the prepacked buffer at
//       matmul time, set @c matmul_params::mem_format_b = 'r' (with
//       @c lowoha_algo == matmul_algo_t::aocl_dlp_blocked).
//   - matmul_algo_t::moe_custom_kernel
//       group_matmul custom-kernel VNNI weight layout (bf16, f16, or s8).
//       Consumed DIRECTLY by the ALGO 3 (N-tile) custom kernel: set
//       @c matmul_params::mem_format_b = 'r' AND
//       @c matmul_params::lowoha_algo == matmul_algo_t::moe_custom_kernel
//       on the group_matmul_direct call. `mem_format_b == 'r'` alone is
//       ambiguous (it also marks the aocl_dlp_blocked layout above); the
//       dispatcher keys the CK-VNNI classification on `lowoha_algo`, so
//       omitting it leaves the weight treated as a raw/AOCL buffer. When
//       both are set the dispatcher aliases the caller-owned buffer
//       instead of re-packing and does NOT touch the process pack cache.
//       Such a prepacked weight is CK-only — a call that cannot route to
//       the custom kernel (CK disabled / unsupported host / non-ALGO-3)
//       fails rather than silently mis-reading the VNNI bytes.
//
// Any other matmul_algo_t passed to the prepack API returns
// status_t::unimplemented.

/**
 * @brief Parameters describing a single weight prepack request.
 *
 * The caller fills this struct on @c reorder_params_t::prepack, sets
 * @c reorder_params_t::is_prepack to true, queries the destination
 * size with @ref weight_prepack_size, then calls @c reorder_direct
 * with the same params. The library never allocates the prepacked
 * buffer itself; the caller owns @c dst end-to-end.
 *
 * Layout / transpose convention matches @c reorderAndCacheWeights in
 * the AOCL matmul backend:
 *   - @c K, @c N are the logical weight dimensions
 *     (rows x cols of the un-transposed weight matrix B).
 *   - @c transposed = true if the input pointer is column-major (order = "ba").
 *   - @c ldb is the physical leading dimension of the input pointer.
 *   - The AOCL "order" param is always row-major ('r') and is not exposed.
 *
 * Backend-specific fields:
 *
 *   AOCL (s8 sym-quant variant):
 *     - @c sym_group_size > 0 selects the s8 sym-quant variant
 *       (s8s8s32os32_sym_quant). The value populates
 *       @c dlp_quant_op_t::group_size on the metadata's @c b_quant_op;
 *       the caller should compute it from the B-side weight scale
 *       granularity (for wei_scale {G, N}, group_size = K / G).
 */
struct prepack_params_t {
    matmul_algo_t algo; ///< Target prepack layout:
    ///< @c matmul_algo_t::aocl_dlp_blocked   — AOCL DLP blocked weight; OR
    ///< @c matmul_algo_t::moe_custom_kernel  — group_matmul custom-kernel VNNI
    data_type_t wei_dtype; ///< Weight data type
    ///< @brief Effective A dtype consumed by the target packed-weight
    ///< layout, after any dynamic quantization -- NOT necessarily the
    ///< dtype of the caller's matmul A tensor.
    ///<
    ///< W4A8 native prepack: pass src_dtype=s8 (runtime quantizes bf16->s8 first).
    data_type_t src_dtype;
    int64_t K; ///< Weight rows
    int64_t N; ///< Weight cols
    int64_t ldb; ///< Physical leading dimension
    bool transposed; ///< true => 't', false => 'n'
    int sym_group_size; ///< AOCL: >0 selects s8 sym-quant variant
    ///< @brief CK pack width (32 or 64). 0 => auto via plan_pack_nr(K,N).
    ///< Used only when @c algo == matmul_algo_t::moe_custom_kernel.
    int pack_nr;
    ///< @brief CK split-halves interleave (silu_and_mul / gelu_and_mul
    ///< re-interleave [gate|up] -> [g0,u0,...]; requires even N).
    ///< Used only when @c algo == matmul_algo_t::moe_custom_kernel.
    bool interleave_split_halves;

    /**
   * @brief Last prepacked-buffer size (in bytes) computed by
   *        @c weight_prepack_size.
   *
   * Reset to 0 at the start of every @c weight_prepack_size call and
   * overwritten with the freshly-computed size on success. Read-only
   * for callers; useful as an out-of-band way to observe the size
   * without keeping a separate copy.
   *
   * @c mutable so the field can be updated through a
   * @c const reorder_params_t& reference.
   */
    mutable size_t cached_size;

    prepack_params_t()
        : algo(matmul_algo_t::none)
        , wei_dtype(data_type_t::none)
        , src_dtype(data_type_t::none)
        , K(0)
        , N(0)
        , ldb(0)
        , transposed(false)
        , sym_group_size(0)
        , pack_nr(0)
        , interleave_split_halves(false)
        , cached_size(0) {}
};

// =====================================================================
// Public API: caller-managed two-step prepack
//
// Both entry points take the same reorder_params_t the caller is going
// to hand to reorder_direct -- the prepack-specific fields live in the
// embedded reorder_params_t::prepack sub-struct, so the caller fills
// one struct and reuses it for both steps.
//
// Step 1 - weight_prepack_size(reorder_params_t&) : query the required
//                                                    prepacked-buffer
//                                                    size in bytes.
// Step 2 - reorder_direct(src, dst, reorder_params_t&) : fill the
//                                                    caller-provided
//                                                    buffer (prepack
//                                                    mode is selected
//                                                    by the
//                                                    rp.is_prepack
//                                                    flag).
//
// The library never allocates the prepacked buffer itself — the caller
// always brings its own storage.
// This keeps ownership and lifetime entirely on the caller side, with
// no special-purpose free helper to remember.
//
// Currently supported algos (set @c params.algo to):
//   - matmul_algo_t::aocl_dlp_blocked
//       wei_dtype = f32   -> aocl_reorder_f32f32f32of32
//       wei_dtype = bf16  -> aocl_reorder_bf16bf16f32of32
//       wei_dtype = f16   -> aocl_reorder_f16f16f16of16  (DLP only)
//       wei_dtype = s4/u4 -> aocl_reorder_bf16s4f32of32
//                            (4-bit weights, bf16 activations; the
//                             GEMM dequantizes on the fly)
//       wei_dtype = s4 + src_dtype = s8 -> aocl_reorder_s8s4s32os32 (native W4A8)
//       wei_dtype = s8 + src_dtype = s8/bf16/f32
//                         -> aocl_reorder_s8s8s32os32
//       wei_dtype = s8 + src_dtype = u8
//                         -> aocl_reorder_u8s8s32os32
//       wei_dtype = s8 with sym_group_size > 0
//                         -> aocl_reorder_s8s8s32os32_sym_quant (DLP only)
//
//   - matmul_algo_t::moe_custom_kernel
//       Custom-kernel VNNI pack for the group_matmul ALGO 3 (N-tile)
//       path. Pack family chosen by wei_dtype:
//         wei_dtype = bf16 -> VDPBF16PS VNNI pack
//                             (layout [O/pack_nr][K/2][pack_nr][2])
//         wei_dtype = f16  -> native AVX-512-FP16 plain slab
//                             (layout [O/pack_nr][K][pack_nr]; no K-pair
//                             VNNI doubling and no compensation row)
//         wei_dtype = s8   -> DQ-INT8 VPDPBUSD VNNI-quad pack
//                             + per-column int32 compensation row
//                             (requires K % 4 == 0)
//       pack_nr auto-selects (plan_pack_nr) unless params.pack_nr pins
//       32 or 64 (must divide N). Consumed by setting
//       @c matmul_params::mem_format_b = 'r' AND
//       @c matmul_params::lowoha_algo == matmul_algo_t::moe_custom_kernel
//       on group_matmul_direct; see the "Supported algos" block at the
//       top of this file for the CK-only consumption contract.
//
// All other matmul_algo_t values return status_t::unimplemented.
// =====================================================================

/**
 * @brief Query the byte size of the prepacked output buffer.
 *
 * Inspects @c params.prepack to determine the required size. Returned
 * value is already rounded up to a 64-byte alignment boundary (the
 * alignment AOCL expects), so a buffer allocated to exactly this many
 * bytes is guaranteed to be large enough.
 *
 * @param params  Reorder parameters with @c params.prepack populated.
 * @return Required buffer size in bytes (64B-aligned) on success;
 *         @c 0 on validation error or unsupported algo/dtype (the
 *         underlying cause is logged via apilog_error).
 */
size_t weight_prepack_size(const reorder_params_t &params);

/**
 * @internal
 * @brief Prepack (reorder) a weight matrix into a caller-provided buffer.
 *
 * NOT a user-facing entry point. User code should call @c reorder_direct,
 * which delegates here when @c params.is_prepack is true. Inspects
 * @c params.prepack for the actual prepack request.
 *
 * The caller MUST have allocated at least @c weight_prepack_size(params)
 * bytes at @p dst; no capacity recheck is performed here (re-querying
 * the size on every call would add a redundant per-call cost).
 *
 * @param weights   Pointer to the original (un-prepacked) weight buffer.
 * @param params    Reorder parameters with @c params.prepack populated.
 * @param dst       Destination buffer. No specific alignment is required
 *                  by any backend, but 64-byte alignment is recommended
 *                  for best AVX-512 throughput.
 * @return @c status_t::success on success;
 *         @c status_t::unimplemented for unsupported algos / dtypes;
 *         @c status_t::failure on validation errors.
 */
status_t weight_prepack_into(
        const void *weights, const reorder_params_t &params, void *dst);

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_PREPACK_HPP
