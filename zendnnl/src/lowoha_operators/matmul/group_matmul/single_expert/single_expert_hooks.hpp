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
 *******************************************************************************/

/// Single-expert EXECUTION HOOKS.
///
/// The single-expert branches the shared executors call (custom-kernel
/// dispatch, the N-tile executor, the prompt tile planner), so single-expert
/// tuning lives in one place.  Each hook is a small inline predicate; the
/// selection policy they build on is in `single_expert_policy.hpp`.
///
/// Library-internal.  Must not be included by
/// `group_matmul_parallel_common.hpp` or `single_expert_policy.hpp`.

#ifndef ZENDNNL_GROUP_MATMUL_SINGLE_EXPERT_HOOKS_HPP
#define ZENDNNL_GROUP_MATMUL_SINGLE_EXPERT_HOOKS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/group_matmul_n_tile_planner.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/n_tile_knobs.hpp"
#include "single_expert_policy.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// AUTO deep-K K-blocking: engage for the single-expert (dense-FFN)
// decode-class call only (see `is_dense_ffn_decode`, the decode-gated
// twin of the adaptive N-tile sizer's scope predicate). MoE (num_ops>1)
// and prompt (m_max>kDecodeMaxM) leave this false; the BF16/INT8
// dtype-specific, act=none, bias-free, deep-K, and M>max_mr guards in
// dispatch_tile still apply, so only eligible W2 decode calls K-block.
inline bool se_kblock_auto(int num_ops, int m_max) {
    return is_dense_ffn_decode(num_ops, m_max);
}

// A dense BF16 prompt at pack NR 32 runs 12-row calls: 24 accumulator
// chains instead of 16 keep VDPBF16PS issue full.  Each output element
// still sums K in the same order, so the result is unchanged.
inline bool se_bf16_prompt(bool single_expert, int NV, int m_max,
        custom_kernel::KernelVariant variant) {
    return single_expert && NV == 2 && m_max > kDecodeMaxM
            && (variant == custom_kernel::KernelVariant::kBF16_BF16_BF16
                    || variant == custom_kernel::KernelVariant::kBF16_BF16_F32)
            && (get_grp_matmul_single_expert_prompt_tile_dtypes()
                    & kSingleExpertBf16);
}

// A dense FP16 prompt at pack NR 32 runs the prompt microkernel: 12 rows
// over two o-blocks per call, so 24 FP16 accumulator chains instead of the
// decode kernel's 8.  Each output element sums K in the same order.
inline bool se_f16_prompt(bool single_expert, int NV, int m_max,
        custom_kernel::KernelVariant variant) {
    return single_expert && NV == 2 && m_max > kDecodeMaxM
            && custom_kernel::is_f16_variant(variant)
            && (get_grp_matmul_single_expert_prompt_tile_dtypes()
                    & kSingleExpertF16);
}

// MR bound of a call whose prompt runs the BF16 or FP16 prompt kernels;
// `max_mr` unchanged otherwise.
inline int se_prompt_max_mr(int max_mr, bool bf16_prompt, bool f16_prompt) {
    if (bf16_prompt) return custom_kernel::kMaxMrNv2Prompt;
    if (f16_prompt) return custom_kernel::kMaxF16PromptMr;
    return max_mr;
}

// Whether a prompt tile may walk its K in chunks: an act=none BF16 prompt
// with an even K and no bias (its chunks accumulate in f32 scratch), or an
// act=none FP16 prompt (its FP16 partial sums reload exactly from the
// output).  A gated epilogue needs the whole K at once.
inline bool se_prompt_kc_eligible(
        bool bf16_prompt, bool f16_prompt, int K, bool gated, bool has_bias) {
    if (gated) return false;
    if (f16_prompt) return true;
    return bf16_prompt && (K % 2) == 0 && !has_bias;
}

// Default K chunk of an eligible prompt tile.  When the B of one call (one
// pack-width column block for BF16, `kF16PromptNb` for FP16) over the whole
// K overflows `l2_budget`, every MR call of an M-block re-reads it from
// memory; chunks of `..._PROMPT_KC` keep each chunk of B resident across the
// M-block, and the prompt tile planner sizes its tiles for one chunk.
// Returns 0 (the full K) otherwise.
inline int se_prompt_kc(
        bool eligible, bool f16_prompt, int pack_nr, int K, int64_t l2_budget) {
    if (!eligible) return 0;
    const int kc = get_grp_matmul_single_expert_prompt_kc();
    if (kc <= 0 || K <= kc) return 0;
    const int blocks = f16_prompt ? custom_kernel::kF16PromptNb : 1;
    const int64_t call_b_bytes = static_cast<int64_t>(K) * pack_nr * blocks * 2;
    return call_b_bytes > l2_budget ? kc : 0;
}

// Shallowest K chunk a prompt tile walks; shallower chunks only add
// accumulator reloads.
inline constexpr int kPromptKcMin = 64;

// K chunk of a prompt tile: the prompt tile's `kc` knob when set (a depth of
// K or more keeps the full K), else the call's default chunk.  A tile that
// cannot chunk keeps the full K.
inline int se_prompt_tile_kc(
        const custom_kernel::CallContext &ctx, int kc_knob, int K) {
    if (kc_knob <= 0) return ctx.prompt_kc;
    return ctx.prompt_kc_eligible && kc_knob < K ? kc_knob : 0;
}

// A prompt the prompt tile executor does not take runs the multi-expert
// kernels and loop orders: the prompt MR bound, K chunks and FP16 prompt
// kernel are fitted to the prompt tile's blocking only.  A bound the prompt
// kernels did not raise (a decode policy's) is kept.
inline void se_leave_prompt_tile(custom_kernel::CallContext &ctx) {
    if (ctx.base_max_mr > 0) ctx.max_mr = ctx.base_max_mr;
    ctx.prompt_kc = 0;
    ctx.prompt_kc_eligible = false;
    ctx.f16_prompt = false;
}

// Minimum MR-call count at which the custom-kernel tile loops
// (`dispatch_tile` for BF16, `dispatch_tile_int8` for DQ-INT8) switch to
// o-block-outer nesting.  The swap trades redundant walks of the sub-tile's
// B strip for re-reading the A panel once per o-block; with two MR calls it
// saves a single B walk and does not pay, so it starts at three.
inline constexpr int kOblockOuterMinCalls = 3;

// Whether a custom-kernel tile runs o-block outer.  Single-expert only
// (`CallContext::single_expert`, num_ops == 1).  Both orders compute
// identical results; the MoE exclusion is explained at the BF16 use site in
// `dispatch_tile`.
inline bool se_oblock_outer(bool single_expert, int n_calls) {
    return single_expert && n_calls >= kOblockOuterMinCalls;
}

// N-tile column split: a single expert on the custom kernel balances the
// remainder columns across threads; native W4A8 keeps even boundaries.
inline bool se_balance_remainder(
        bool single_expert, bool use_custom, bool native_s4_wei) {
    return single_expert && use_custom && !native_s4_wei;
}

// Whether a call is a single-expert prompt the CK prompt tile executor may
// take.  `prompt_rounds`: the round executor would run it in the prompt
// phase.  The cheap checks precede the policy classifier.
inline bool se_prompt_tile_call(int num_ops,
        const std::vector<matmul_params> &params, const std::vector<int> &K,
        const std::vector<int> &N, int num_threads, bool prompt_rounds) {
    return prompt_rounds && num_ops == 1 && !params.empty()
            && !(params[0].active_matmul > 0 && params[0].total_matmul > 1)
            && grp_matmul_single_expert_ck(params[0], K[0], N[0], num_threads)
                       .prompt;
}

// Whether the prompt tile planner uses the dense single-expert plan.
inline bool se_dense_prompt_plan(size_t num_ops, bool bf16_family) {
    // Only BF16 has the dense equal-split plan; other dtypes use the
    // multi-expert tile sizing.
    return num_ops == 1 && bf16_family;
}

// ── Adaptive N-tile floor (single-expert only) ───────────────────────
// SCOPE: `num_ops == 1`.  Real MoE groups keep the legacy fixed
// `phase_cap`, so multi-expert round scheduling and per-expert tiling
// are untouched; multi-expert decode has its own `ccd_fill_min_n_tile`.
//
// `phase_cap` bounds tile WIDTH for microkernel efficiency, but
// `max_N / phase_cap` can yield far fewer tiles than the thread team the
// lone expert receives (N=4096 / 512 = 8 tiles on 64 cores leaves 56
// threads idle).  So size the tile by target COUNT instead, tracking the
// thread count across core counts and N.  Never below `floor`, and
// bounded by `phase_cap` except in `n_tile_from_target_count`'s
// one-wave fallback.
inline int se_adaptive_min_n_tile(
        const GroupNTileTopology &topo, int phase_cap) {
    const int oversub = get_grp_n_tile_oversub();
    if (oversub <= 0) return phase_cap; // adaptive sizing disabled
    if (!is_single_dense_expert(topo.num_ops)) return phase_cap;
    const int team_per_expert = topo.num_threads / std::max(1, topo.num_ops);
    if (team_per_expert <= 1) return phase_cap; // one-thread-per-expert
    // A single expert can never usefully run more concurrent N-tiles than
    // there are hardware threads: surplus tiles just queue behind the first
    // wave.  Hence the `num_threads` cap, which in this `num_ops == 1` scope
    // (`team_per_expert == num_threads`) collapses every OVERSUB >= 1 onto
    // the same target — see the knob's doc-block in the header for why the
    // multiplier is kept for the multi-expert generalisation.  Widened to
    // 64-bit first: OVERSUB accepts any non-negative int, so the product
    // would overflow before the cap.
    const int64_t target = static_cast<int64_t>(oversub) * team_per_expert;
    const int target_tiles
            = static_cast<int>(std::min<int64_t>(target, topo.num_threads));
    return n_tile_from_target_count(
            topo, phase_cap, target_tiles, get_grp_n_tile_floor());
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_SINGLE_EXPERT_HOOKS_HPP
