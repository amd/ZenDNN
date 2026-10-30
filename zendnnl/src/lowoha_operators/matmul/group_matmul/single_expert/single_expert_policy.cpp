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

/// Out-of-line half of the single-expert selection policy declared in
/// `single_expert_policy.hpp`: the dtype-tuple classifier, its call-level
/// form, the Rule 0.46 / 0.45 AUTO helper, the prepack-side layer test, the
/// knob warning, and `grp_matmul_single_dense_call`.

#include <string>
#include <vector>

#include "single_expert_policy.hpp"

#include "../common/group_matmul_parallel_common.hpp"
#include "../custom_kernel/dispatch.hpp"
#include "../executors/n_tile/prompt_tile/prompt_tile_tunables.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/n_tile_knobs.hpp" // get_grp_n_tile_strategy

namespace zendnnl {
namespace lowoha {
namespace matmul {

void warn_single_expert_knob_value(
        const char *name, const std::string &unknown) {
    apilog_warning(name, ": ignoring unknown value(s) '", unknown,
            "'; expected bf16, f16, int8, u8s8, w4a8, all or none");
}

single_expert_ck_t grp_matmul_single_expert_ck(data_type_t src, data_type_t wei,
        data_type_t dst, bool dynamic_quant, data_type_t compute,
        bool static_u8s8, bool aocl_prepacked, int K, int N, int num_threads) {
    const unsigned force_dt = get_grp_matmul_single_expert_force_ck_dtypes();
    const unsigned prompt_dt
            = get_grp_matmul_single_expert_prompt_tile_dtypes();
    const unsigned decode_dt
            = get_grp_matmul_single_expert_decode_tile_dtypes();
    if ((prompt_dt | decode_dt) == 0) return {};
    if (!get_grp_matmul_custom_kernel()) return {};
    // Strategy 0 plans a lone expert as Sequential AOCL.
    if (get_grp_n_tile_strategy() == 0) return {};
    using custom_kernel::KernelVariant;
    const KernelVariant v = grp_matmul_ck_weight_variant(src, wei, dst,
            dynamic_quant, compute, static_u8s8, aocl_prepacked, K, N);
    if (v == KernelVariant::kUnsupported) return {};
    prompt_tile::kernel_type_t fam;
    unsigned dt = 0;
    if (custom_kernel::is_int8_variant(v)) {
        fam = prompt_tile::kernel_type_t::kInt8;
        dt = static_u8s8 ? kSingleExpertU8s8 : kSingleExpertInt8;
    } else if (custom_kernel::is_w4a8_variant(v)) {
        fam = prompt_tile::kernel_type_t::kW4a8;
        dt = kSingleExpertW4a8;
    } else if (custom_kernel::is_f16_variant(v)) {
        fam = prompt_tile::kernel_type_t::kF16;
        dt = kSingleExpertF16;
    } else if (v == KernelVariant::kBF16_BF16_BF16
            || v == KernelVariant::kBF16_BF16_F32) {
        fam = prompt_tile::kernel_type_t::kBf16;
        dt = kSingleExpertBf16;
    } else {
        return {};
    }
    if (((prompt_dt | decode_dt) & dt) == 0) return {};

    // The phase knobs choose the phases; `..._FORCE_CK` lifts decode's
    // N-tile count for the dtypes it lists.
    const bool wide_n = num_threads <= 0 || K <= 0 || N <= 0
            || has_enough_single_expert_tiles(N, K, num_threads);
    const bool n_ok = wide_n || (force_dt & dt) != 0;
    if (!n_ok) return {};
    single_expert_ck_t r;
    r.decode = (decode_dt & dt) != 0;
    // The executor does not tile static U8xS8 (per-tensor source scale), and
    // `decode_d` never plans the rounds strategies it takes over.
    r.prompt = (prompt_dt & dt & ~kSingleExpertU8s8) != 0
            && prompt_tile::global_value(prompt_tile::kEnable) != 0
            && !(prompt_tile::facts(fam).kblock
                    && grp_matmul_kblock_enabled(/*kblock_auto=*/false))
            && get_grp_n_tile_strategy() != 1;
    r.forced = !wide_n && r.any();
    return r;
}

single_expert_ck_t grp_matmul_single_expert_ck(
        const matmul_params &p, int K, int N, int num_threads) {
    if (K > 0 && p.dtypes.wei == data_type_t::s4) {
        const auto &ws = p.quant_params.wei_scale.dims;
        if (ws.size() == 2 && ws[0] > 0
                && (K % ws[0] != 0
                        || (K / ws[0]) % custom_kernel::kS4Octet != 0))
            return {};
    }
    return grp_matmul_single_expert_ck(p.dtypes.src, p.dtypes.wei, p.dtypes.dst,
            p.dynamic_quant, p.dtypes.compute, is_static_u8s8_quant_config(p),
            is_aocl_prepacked_weight(p), K, N, num_threads);
}

single_expert_ck_t grp_matmul_single_expert_call_ck(const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K,
        const std::vector<matmul_params> &params,
        grp_matmul_gated_act_t fused_act, int num_threads) {
    if (M.size() != 1 || M[0] <= 0 || params.empty()) return {};
    const auto &p = params[0];
    if (p.active_matmul > 0 && p.total_matmul > 1) return {};
    // The custom kernel's gated activations store the family's 16-bit type:
    // FP16 for an FP16 layer, BF16 otherwise.
    const data_type_t gated_dst = p.dtypes.src == data_type_t::f16
            ? data_type_t::f16
            : data_type_t::bf16;
    if (fused_act != grp_matmul_gated_act_t::none && p.dtypes.dst != gated_dst)
        return {};
    const single_expert_ck_t sck
            = grp_matmul_single_expert_ck(p, K[0], N[0], num_threads);
    if (!sck.any()
            || !group_matmul_custom_kernel_candidate(params, M, N, fused_act))
        return {};
    return sck;
}

single_expert_rule_pick grp_matmul_single_expert_auto_rule(
        const single_expert_ck_t &single_ck, bool is_decode,
        bool phase_env_pinned, bool n_tile_safe, int num_ops, int max_M) {
    // Rule 0.46 — SINGLE DENSE EXPERT → ALGO 3 (custom kernel), per phase.
    // `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_{PROMPT,DECODE}_TILE` cover the
    // prompt / decode and `..._FORCE_CK` both, each for the dtypes it lists
    // (`grp_matmul_single_expert_ck`; decode on by default).  A lone expert would otherwise
    // take Rule 0.7's ALGO 1 (full-team AOCL) in prompt and Rule 0.45 /
    // 0.5 in decode.  The prompt tile executor splits a prompt's M into
    // L3-team blocks; for a fused FFN each team keeps its rows local from W13
    // to W2.  With both phases covered, prompt and decode share one
    // custom-kernel pack, which WEIGHT_CACHE=2 may store in place.  Honours
    // an explicit phase pin and the n_tile_safe clamp.
    if (!phase_env_pinned && single_ck.covers(is_decode)) {
        if (single_ck.forced) {
            return {true, 3, 3,
                    is_decode ? "auto_rule046_single_expert_force_ck_decode"
                              : "auto_rule046_single_expert_force_ck"};
        }
        return {true, 3, 3,
                is_decode ? "auto_rule046_single_dense_decode"
                          : "auto_rule046_single_dense_prompt"};
    }

    // Rule 0.45 — SINGLE DENSE EXPERT DECODE → ALGO 3 (N-tile).
    // A lone expert (`num_ops == 1`) in decode would otherwise be diverted by
    // Rule 0.5's low-occupancy arm to ALGO 1 because one active expert cannot
    // satisfy the four-threads-per-expert threshold on a normal team. That
    // path never reaches the N-tile planner where the single-expert
    // optimisations live (adaptive tiling, ragged-N fallback, K-blocking), so
    // route it to ALGO 3, mirroring the multi-expert decode default (Rule 2c).
    // `is_dense_ffn_decode` is the scope predicate shared with the adaptive
    // N-tile sizer and the auto-K-blocking gate, keeping MoE and prompt
    // untouched. Honours an explicit decode pin and the n_tile_safe clamp;
    // env-gated (default ON) for A/B. Default and gates are pinned by the
    // `SingleDenseExpertDecode*` tests in test_algos.cpp.
    if (is_dense_ffn_decode(num_ops, max_M) && !phase_env_pinned && n_tile_safe
            && get_grp_matmul_dense_decode_ntile()) {
        return {true, 3, 3, "auto_rule045_single_dense_decode"};
    }
    return {};
}

bool grp_matmul_single_expert_prepack_layer(data_type_t src, data_type_t wei,
        data_type_t dst, bool dynamic_quant, data_type_t compute, int K, int N,
        int num_threads) {
    const bool static_u8s8 = !dynamic_quant && src == data_type_t::u8
            && wei == data_type_t::s8
            && (compute == data_type_t::none || compute == data_type_t::u8);
    return grp_matmul_single_expert_ck(src, wei, dst, dynamic_quant, compute,
            static_u8s8, /*aocl_prepacked=*/false, K, N, num_threads)
            .both();
}

bool grp_matmul_single_dense_call(const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K,
        const std::vector<matmul_params> &params, int num_threads) {
    if (M.size() != 1 || M[0] <= 0 || params.empty() || N.empty() || K.empty())
        return false;
    const auto &p = params[0];
    if (p.active_matmul > 0 && p.total_matmul > 1) return false;
    if (is_aocl_prepacked_weight(p)) return false;
    const bool float_family = (p.dtypes.src == data_type_t::bf16
                                      && p.dtypes.wei == data_type_t::bf16)
            || (p.dtypes.src == data_type_t::f16
                    && p.dtypes.wei == data_type_t::f16);
    return float_family
            && grp_matmul_single_expert_ck(p, K[0], N[0], num_threads).both();
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
