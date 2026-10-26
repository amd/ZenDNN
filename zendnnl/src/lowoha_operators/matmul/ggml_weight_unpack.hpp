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
 * Plain Q4_0 weights are emitted as packed SIGNED nibbles: the GGML +8 bias
 * is removed here so the s4 GEMM sign-extends the true code.  Superblock
 * Q4_0 (block_q4_0x8, @p is_superblock) already stores signed nibbles, so
 * that bias is left in place.  Superblock requires ggml_type 2 and N
 * divisible by 8.
 *
 * @return 0 on success, -1 on error.
 */
int ggml_unpack_weight_buffer(const void *weight_data, int ggml_type,
        bool use_bf16_scales, int64_t N, int64_t K, int8_t **wei_ptr,
        void **scl_ptr, void *unpack_buffer = nullptr,
        bool is_superblock = false);

/**
 * @brief Returns true when @p params is already a sym-quant per-group int8
 *        matmul: s8 source, s4 or s8 weight, no source zero-point, more than
 *        one source scale, f32 or bf16 destination.
 *
 * This describes the GEMM after source quantization.  It is not a
 * precondition of unpack_ggml_weights_and_cache.  An f32 or bf16 activation
 * with dynamic_quant set is a valid entry to the packed-weight path; the
 * source is quantized later.
 */
bool ggml_is_sym_quant(const matmul_params &params);

/**
 * @brief True when a GGML packed weight can run the sym-quant GEMM.
 *
 * Accepts the post-quantization shape from ggml_is_sym_quant, and the
 * pre-quantization entry llama.cpp uses: an f32 or bf16 activation with
 * dynamic_quant set and a non-scalar source scale.  A scalar source scale
 * is rejected in both forms.  Cold-expert warming does not call this; unpack
 * itself still ignores the activation.
 */
bool ggml_packed_gemm_supported(const matmul_params &params);

/**
 * @brief Pre-quantization validation for the GGML packed-weight matmul path.
 *
 * No-op (returns success) when params.packing.pack_format_b != 1. Otherwise
 * verifies the static preconditions that the GGML out-of-place unpack
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
 * @brief Unpack GGML packed weights out of place and cache the raw result.
 *
 * The block format is taken from params.dtypes.wei: s8 (or none, the default
 * carried by the fused-MoE Op2 scratch params) selects Q8_0, which unpacks
 * straight to s8, and s4 selects Q4_0, which stays packed signed s4 (two
 * nibbles per byte).  Any other weight dtype is rejected.
 * params.packing.is_superblock selects the 8-row block_q4_0x8 layout for an
 * s4 weight; it is rejected for every other dtype.
 *
 * This function never reorders.  On success @p weight points at a freshly
 * allocated raw matrix — s8 [N, K] for Q8_0, packed s4 [N, K/2] for Q4_0 —
 * with writable capacity covering the matching AOCL sym-quant blocked image
 * (Q8_0 also covers the custom-kernel INT8 pack), then {K/32, N} bf16 scales
 * past that capacity.  params.mem_format_b is 'n', packing.pack_format_b /
 * is_superblock are cleared, dtypes.wei stays s8 or s4, weight_cache_type
 * is 2 because the call entered with pack_format_b == 1, and
 * wei_buffer_capacity_bytes is that capacity so a later AOCL reorder can
 * write its blocked image into this buffer.  The caller-owned packed GGML
 * buffer is not modified (llama.cpp may map it read-only).  bf16 / fp32
 * weights never enter this path and keep the caller's cache mode.
 *
 * Uses an LRU cache keyed on the tuple
 *   (trans, K, N, ldb, weight pointer, ggml type, is_superblock)
 * so the same packed buffer is unpacked once.  M is not part of the key.
 *
 * Precondition: trans must be 't' (GGML stores weights as N x K row-major,
 * i.e. the transpose of B).  The activation dtype is not an input.  Cold
 * experts (M==0) may call this before src_scale is populated; the unpack
 * needs only the packed weight and its shape.
 *
 * @param weight  [in/out] Pointer to packed weight data; redirected to the
 *                cached raw s8 weights on success.
 * @param N       Number of output channels (rows in the GGML weight matrix)
 * @param K       Number of input features (columns, must be divisible by 32)
 * @param ldb     Leading dimension for the unpacked weight matrix
 * @param trans   Transpose flag for matrix B (must be 't' for GGML)
 * @param params  [in/out] matmul_params whose wei_scale is populated on success
 * @param is_weights_const  when false the unpack still runs but bypasses the
 *                cache entirely, re-unpacking into a thread-private buffer
 *                that stays valid for the rest of this call
 * @return status_t::success on success, status_t::failure on error
 */
status_t unpack_ggml_weights_and_cache(const void *&weight, int N, int K,
        int ldb, char trans, matmul_params &params,
        bool is_weights_const = true);

/** Clear cached GGML unpacked weight buffers. */
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
