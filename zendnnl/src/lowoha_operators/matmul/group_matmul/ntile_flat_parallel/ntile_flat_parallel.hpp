/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************/

/**
 * @file ntile_flat_parallel.hpp
 * @brief Internal W8A8 grouped-MoE executor and cache lifecycle.
 *
 * Library-internal: not installed, not part of the public API surface, and
 * reachable only from the existing vector-based `group_matmul_direct`
 * overload.  No new public symbol is introduced by this feature.
 *
 * ## What this path is for
 *
 * The generic fused-MoE dispatcher builds per-expert GEMM descriptors and
 * consults the planner, which forces the quantized activation to be
 * re-derived (or re-read from memory) for every output-channel block.  For
 * a small-batch decode step -- a handful of tokens per expert against
 * weights far larger than L2 -- the whole cost of the layer is moving
 * weights, so the only thing that matters is that each weight byte is
 * touched once, in a VNNI-ready layout, against an activation that is
 * already resident in registers.
 *
 * This path recognizes exactly that call shape from the arguments the
 * existing contract already carries, then runs the whole
 * gate/up -> SiLU x mul -> requantize -> down chain itself, bypassing the
 * planner and the descriptor build. Calls it does not recognize may continue
 * through main's generic fused-MoE path, including caller-prequantized S8.
 *
 * ## Why it needs no extra arguments
 *
 * The caller has already gathered each expert's rows into a contiguous
 * `src[i]` buffer, and `moe_postop->row_ptrs` already states where each
 * (token, slot) result must land.  Together those remove the need for the
 * routing ids and the original token matrix: the per-expert row order the
 * caller chose *is* the schedule.
 *
 * There are two source/ownership modes:
 *
 *   - `bf16_dynamic`: BF16 source, runtime BF16 -> biased-U8 quantization,
 *     and Op2 writes BF16 back into the reusable source rows.
 *   - `s8_prequantized`: caller-owned signed-S8 source plus a positive
 *     per-token BF16/F32 scale. Stage 0 is skipped, and Op2 writes BF16 into
 *     caller-owned `fused_moe.dst_down` rows. The destination must be the
 *     tight BF16 view over the exact same BF16-sized backing whose leading
 *     bytes held the S8 source prefix.
 *
 * In either mode `row_ptrs` must be a one-to-one permutation of the BF16
 * rows Op2 writes. The existing weighted-reduce post-op then runs
 * unmodified.
 *
 * Nothing here references a framework: raw pointers, OpenMP, and AVX-512.
 */

#ifndef LOWOHA_NTILE_FLAT_PARALLEL_HPP
#define LOWOHA_NTILE_FLAT_PARALLEL_HPP

#include <cstddef>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {

/// Source encodings accepted by the private whole-MoE executor.
///
/// This deliberately classifies only the source/dynamic-quant pair. The
/// remaining dtype, scale, ownership, and geometry requirements are validated
/// by `try_execute`.
enum class input_mode_t {
    unsupported,
    mixed,
    bf16_dynamic,
    s8_prequantized,
};

inline input_mode_t classify_input_mode(const matmul_params &p) noexcept {
    if (p.dtypes.src == data_type_t::bf16 && p.dynamic_quant) {
        return input_mode_t::bf16_dynamic;
    }
    if (p.dtypes.src == data_type_t::s8 && !p.dynamic_quant) {
        return input_mode_t::s8_prequantized;
    }
    return input_mode_t::unsupported;
}

/// Return one mode shared by every positive-M entry in `[0, count)`.
///
/// `mixed` is distinct from `unsupported` so callers can diagnose a
/// heterogeneous active set separately from an individually invalid
/// BF16/S8 + dynamic-quant pairing.
inline input_mode_t classify_uniform_input_mode(const std::vector<int> &M,
        const std::vector<matmul_params> &params, size_t count) noexcept {
    if (M.size() < count || params.size() < count) {
        return input_mode_t::unsupported;
    }
    input_mode_t uniform = input_mode_t::unsupported;
    bool found = false;
    for (size_t i = 0; i < count; ++i) {
        if (M[i] <= 0) { continue; }
        const input_mode_t current = classify_input_mode(params[i]);
        if (current == input_mode_t::unsupported) {
            return input_mode_t::unsupported;
        }
        if (found && current != uniform) { return input_mode_t::mixed; }
        uniform = current;
        found = true;
    }
    return found ? uniform : input_mode_t::unsupported;
}

/**
 * @brief Try to run this `group_matmul_direct` call as a W8A8 grouped MoE.
 *
 * Arguments are the caller's, forwarded verbatim from the vector-based
 * overload.  The function first proves the call is one it can serve --
 * one uniform source mode, BF16 destination, symmetric per-output-channel
 * int8 w13/w2, per-token activation scale, SiLU-and-mul, no bias, no zero
 * point or Op1 post-op chain, raw/unpacked operand formats, supported aligned
 * shapes, safe source/destination ownership, and a row-count/`row_ptrs`
 * agreement -- and only then touches any data.
 *
 * `bf16_dynamic` preserves the original contract and code path:
 * `src=bf16`, `compute=s8`, `dynamic_quant=true`, private stage-0
 * quantization, publication to a caller-provided source-scale buffer using
 * its configured BF16/F32 dtype, and source reuse for the BF16 W2 destination.
 *
 * `s8_prequantized` requires `src=s8`, `wei=s8`, `dst=bf16`, `compute=s8`,
 * `dynamic_quant=false`, no source zero-point, and a non-null positive finite
 * source scale with exact shape `{M[i], 1}` and dtype BF16 or F32. Op1 output
 * remains library-owned, while every active Op2 destination is caller-owned
 * through `fused_moe.dst_down`. It must equal `src[i]` and use
 * `ldc_down == hidden`, with the S8 prefix residing in a BF16-sized
 * `[M, hidden]` backing allocation. W13 consumes the complete S8 prefix before
 * W2 overwrites the backing as BF16; allocation capacity is the caller's
 * responsibility because raw pointers do not expose it. Separate, offset,
 * partial, padded, and cross-expert destinations are rejected. The BF16
 * SiLU-times-up intermediate is still dynamically requantized before W2.
 *
 * The path requires both `ZENDNNL_MATMUL_WEIGHT_CACHE != 0` and a nonzero
 * effective per-call `matmul_params::weight_cache_type`; with caching disabled
 * it declines before writing output. The dispatcher may then use main's
 * generic BF16 or caller-prequantized-S8 fused-MoE implementation.
 * The complete-tensor LRU honors `ZENDNNL_LRU_CACHE_CAPACITY` exactly. While
 * caching is enabled,
 * W13/W2 weight allocations are immutable cache identities: keep them live
 * and unchanged until `clear_fused_moe_scratch()` completes in a quiescent
 * window. Weight scales are copied into execution-private scratch on every
 * call and may change between calls without a cache flush. Call the clear API
 * before releasing, replacing, or reusing a weight allocation. Changing cache
 * mode to 0 does not clear prior entries; if weights change while disabled,
 * clear before changing back to mode 1 or 2.
 *
 * @return @c status_t::unimplemented when the call is not eligible, in
 *         which case NOTHING has been written and the caller may continue
 *         through the generic fused-MoE path. Otherwise the terminal status
 *         of the fast path (@c status_t::success, or an allocation / ISA
 *         failure).
 */
status_t try_execute(const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        const std::vector<matmul_params> &params,
        const group_matmul_moe_postop_params *moe_postop,
        const grp_matmul_gated_act_params *gated_act,
        const grp_matmul_fused_moe_params *fused_moe);

/**
 * @brief Drop every packed weight this path has cached.
 *
 * Called from `clear_fused_moe_scratch()` so the library's existing
 * MoE-state teardown covers the packed weights too, and a host that
 * already calls it needs no new API.  Takes the exclusive side of the
 * execution lifecycle lock, so it cannot retire a pack while a fast-path
 * call is in flight. This flush is the generation boundary that must precede
 * destruction, mutation, or allocator reuse of cached W13/W2 weight storage.
 */
void flush_packed_weight_cache();

/// Number of live entries in the packed-weight cache.  Test observability
/// for the pack/flush lifecycle; not used on the hot path.
size_t packed_weight_cache_size();

/// Release this thread's fast-path scratch capacity.
void reset_thread_local_scratch();

} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_NTILE_FLAT_PARALLEL_HPP
