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

#ifndef GGML_WEIGHT_UNPACK_HPP
#define GGML_WEIGHT_UNPACK_HPP

#include <cstdint>
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

/**
 * @brief Returns the minimum buffer size in bytes needed for the unpacked
 *        GGML weights + scales layout.
 *
 * With use_bf16_scales=true the packed and unpacked totals are equal, so the
 * inplace unpack in ggml_unpack_weight_buffer is safe to run directly on the
 * caller's packed-weights allocation. With use_bf16_scales=false (fp32 scales)
 * the unpacked layout is larger and a fresh allocation is required.
 *
 * @param ggml_type       GGML quantization type (8 = Q8_0, 2 = Q4_0)
 * @param use_bf16_scales true to use bf16 scales, false for fp32
 * @param N               number of rows in the weight matrix (output channels)
 * @param K               number of columns (must be divisible by 32)
 * @return buffer size in bytes, or -1 on invalid parameters
 */
int64_t ggml_unpack_weight_buffer_size(
        int ggml_type, bool use_bf16_scales, int64_t N, int64_t K);

/**
 * @brief Unpack GGML quantised weights into a flat weight region + a flat
 *        scale region.
 *
 * If @p unpack_buffer is null, the caller's @p weight_data buffer is
 * reinterpreted as both the source of packed blocks and the destination of the
 * unpacked layout. Otherwise, @p unpack_buffer receives the unpacked layout and
 * @p weight_data remains unchanged.
 *
 * On success:
 *   *wei_ptr  -> start of the buffer (int8 weight bytes)
 *   *scl_ptr  -> weight region + weight_bytes (fp32 or bf16 scale bytes)
 *
 * Q4_0 weights are emitted as packed SIGNED nibbles: the GGML +8 bias is
 * removed here so the downstream sign-extending s4 -> s8 upcast is correct.
 *
 * @return 0 on success, -1 on error.
 */
int ggml_unpack_weight_buffer(const void *weight_data, int ggml_type,
        bool use_bf16_scales, int64_t N, int64_t K, int8_t **wei_ptr,
        void **scl_ptr, void *unpack_buffer = nullptr);

/**
 * @brief Returns true when @p params is configured for sym-quant per-group
 *        int8 matmul, the only mode supported by the GGML packed-weight path.
 *
 * Callers should validate this BEFORE invoking unpack_ggml_weights_and_cache
 * to fail-fast on misconfigured matmul setups.
 */
bool ggml_is_sym_quant(const matmul_params &params);

/**
 * @brief Pre-quantization validation for the GGML packed-weight matmul path.
 *
 * No-op (returns success) when params.packing.pack_format_b != 1. Otherwise
 * verifies the static preconditions that the GGML unpack/reorder pipeline
 * requires:
 *   - weights must be constant (so the LRU cache can keep them across calls)
 *   - Batch_B must be 1 (single-buffer unpack/reorder cache)
 *   - transB must be true (GGML stores weights as N x K row-major,
 *     i.e. the transpose of B)
 *
 * Designed to be invoked through op_instrumentation::validate(...).
 */
status_t validate_ggml_packed_inputs(const matmul_params &params,
        bool is_weights_const, int Batch_B, bool transB);

/**
 * @brief Unpack GGML packed weights out of place, reorder them for AOCL
 *        blocked sym-quant execution, and cache the final reordered buffer.
 *
 * The block format is taken from params.dtypes.wei: s8 (or none, the default
 * carried by the fused-MoE Op2 scratch params) selects Q8_0, which unpacks
 * straight to s8, and s4 selects Q4_0, whose nibbles are unpacked to packed s4
 * and then widened to s8 so both formats share one reorder.  Any other weight
 * dtype is rejected.
 *
 * Uses an LRU cache keyed on the tuple
 *   (trans, K, N, ldb, weight pointer, algo = aocl_dlp_blocked)
 * to avoid re-running the unpack and reorder if the same packed buffer is
 * presented again with the same geometry. The activation row count M and
 * the (constant) sym-quant group size are intentionally NOT part of the
 * cache key, since the reordered weight bytes do not depend on them.
 *
 * On success, @p weight is redirected to the cached AOCL sym-quant-reordered
 * int8 weight buffer, params.mem_format_b is set to reordered, and
 * params.quant_params.wei_scale is populated with the cached bf16 scale array.
 * The caller-owned packed GGML buffer is not modified.
 *
 * Precondition: trans must be 't' (GGML stores weights as N x K row-major,
 * i.e. the transpose of B).  ggml_is_sym_quant(params) is required and
 * validated by the caller only for ACTIVE experts; cold-expert cache warming
 * (M==0) may call this before src_scale is populated — the unpack needs only
 * the packed weight + shape, not the src_scale, so the sym-quant check does
 * not gate warming.
 *
 * @p skip_reorder selects the downstream weight layout:
 *   - false (default) — REORDER path: unpack + AOCL sym-quant reorder +
 *     cache; @p weight -> reordered buffer, mem_format_b = 'r'.  Consumed
 *     directly by the AOCL full-weight GEMM (ALGO 1 etc.).
 *   - true — RAW-S8 path: unpack to a cached raw s8 buffer (N x K) +
 *     {K/32, N} bf16 scales and hand it back UN-reordered
 *     (mem_format_b = 'n', packing.pack_format_b cleared to 0).  The weight
 *     is then a plain per-group s8 weight, so the flat_n_tile (ALGO 3)
 *     per-group path reorders it INTERNALLY per N-tile
 *     (`do_tile`'s `{G, n_tile}` sym-quant repack) — the GEMM handles the
 *     reorder after N-tiling, instead of this function pre-reordering the
 *     full weight.  Identical to how a caller-provided per-group s8 weight
 *     already flows.  The two layouts are cached under DISTINCT keys so
 *     they never alias.  Callers set this when the call may run on N-tile
 *     (ALGO 3 pinned, or AUTO where the selector can pick ALGO 3).
 *
 * @param weight  [in/out] Pointer to packed weight data; redirected to the
 *                cached reordered (or raw s8) weights on success.
 * @param N       Number of output channels (rows in the GGML weight matrix)
 * @param K       Number of input features (columns, must be divisible by 32)
 * @param ldb     Leading dimension for the unpacked weight matrix
 * @param trans   AOCL transpose flag for matrix B (must be 't' for GGML)
 * @param params  [in/out] matmul_params whose wei_scale is populated on success
 * @param skip_reorder  hand back raw s8 (mem_format 'n') for the N-tile
 *                      per-group DLP path instead of pre-reordering
 * @param is_weights_const  when false the unpack still runs but bypasses the
 *                      cache entirely, re-unpacking into a thread-private
 *                      buffer that stays valid for the rest of this call
 * @return status_t::success on success, status_t::failure on error
 */
status_t unpack_ggml_weights_and_cache(const void *&weight, int N, int K,
        int ldb, char trans, matmul_params &params, bool skip_reorder = false,
        bool is_weights_const = true);

/** Clear cached GGML unpacked/reordered weight buffers. */
void clear_ggml_weight_unpack_cache();

/** Number of cached GGML unpacked/reordered weight buffers currently resident.
 *  One entry is added per distinct (weight pointer, K, N, ldb, trans) unpacked
 *  weight, so a fully-warmed fused-MoE layer holds 2 * num_experts entries
 *  (gate/up + down).  Intended for tests / diagnostics. */
size_t ggml_weight_unpack_cache_size();

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // GGML_WEIGHT_UNPACK_HPP
