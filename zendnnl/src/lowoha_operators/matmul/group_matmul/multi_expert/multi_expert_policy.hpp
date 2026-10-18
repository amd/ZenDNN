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

/// Multi-expert (MoE) selection policy shared by the selector, prepack
/// and the executors.

#ifndef ZENDNNL_GROUP_MATMUL_MULTI_EXPERT_POLICY_HPP
#define ZENDNNL_GROUP_MATMUL_MULTI_EXPERT_POLICY_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_env.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

/// Does unpinned AUTO run this layer on ALGO 3 in both phases?  BF16 and
/// FP16 layers of at least two experts do: the custom kernel serves both
/// phases from one family-specific pack.  A single-expert call, such as a
/// routed `matmul_direct`, and every other dtype keep the shape rules of
/// `auto_select_algo`, so their prompt and decode may run different ALGOs.
/// `layer_experts` counts the pool (`total_matmul` under the framework
/// opt-in), so a call with one active expert of a larger pool stays with its
/// layer.
inline bool grp_matmul_auto_algo3_layer(
        bool supported_float_family, int layer_experts) {
    return supported_float_family && layer_experts >= 2;
}

/// `grp_matmul_auto_algo3_layer` for one call: every active expert is
/// uniformly bf16 x bf16 or f16 x f16 on a weight the caller did not
/// AOCL-prepack (such a weight keeps the AOCL prompt path).
inline bool grp_matmul_auto_algo3_call(
        const std::vector<int> &M, const std::vector<matmul_params> &params) {
    if (params.size() < M.size()) { return false; }
    bool any_active = false;
    data_type_t layer_dtype = data_type_t::none;
    for (size_t i = 0; i < M.size(); ++i) {
        if (M[i] <= 0) { continue; }
        const auto &p = params[i];
        const data_type_t dt = p.dtypes.src;
        if ((dt != data_type_t::bf16 && dt != data_type_t::f16)
                || p.dtypes.wei != dt
                || (layer_dtype != data_type_t::none && layer_dtype != dt)
                || is_aocl_prepacked_weight(p)) {
            return false;
        }
        layer_dtype = dt;
        any_active = true;
    }
    int layer_experts = static_cast<int>(M.size());
    if (!params.empty() && params[0].active_matmul > 0) {
        layer_experts = std::max(
                layer_experts, static_cast<int>(params[0].total_matmul));
    }
    return any_active
            && grp_matmul_auto_algo3_layer(
                    layer_dtype != data_type_t::none, layer_experts);
}

// ── DecodeDynamic generic decision-tree knobs ────────────────────────
//
// These take effect whenever DecodeDynamic is eligible for selection,
// i.e. under BOTH:
//   * `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=3` (forced DecodeDynamic, the
//     production default), and
//   * `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=0` (smart AUTO), where the
//     planner may route decode-class shapes to DecodeDynamic.
// Under `=2` (Rounds) and `=1` (DecodeD) the flow is untouched and these
// getters are never consulted.
//
// The route DecodeDynamic-vs-Rounds is decided PER OP from two signals
// available in `topo` — the active expert count, the machine CCD count,
// and this op's per-expert weight bytes vs a CCD's L3.  No layer/model
// identity is used:
//
//   use_decode_dynamic =
//        (active_ops  >= EPC_MULT  * num_ccds)         // enough experts/CCD
//     || (wei_per_expert >= WEI_L3_MULT * kL3PerCcdBytes) // weight >> CCD L3
//
// Structural rationale for the two branches:
//   * experts-per-CCD (epc = active_ops/num_ccds): the CCD-cohesive map
//     assigns whole experts to CCDs, so its load balance is governed by
//     how evenly `active_ops` divides `num_ccds`.  At low epc a single
//     extra expert overloads one CCD (the imbalance is ~ceil(epc)/epc),
//     so DecodeDynamic only pays off once epc is large enough that this
//     rounding imbalance is small — EPC_MULT (default 4).  epc grows with
//     batch size, so this is the decode-throughput-sensitive branch.
//   * per-expert weight vs CCD L3: the Rounds path batches experts to
//     fit their weights in aggregate L3, so when a single expert's
//     weight already exceeds a CCD's L3 it serialises down to ~1 expert
//     per round; DecodeDynamic instead streams experts concurrently, one
//     per CCD.  WEI_L3_MULT (default 2, i.e. >= 2 CCD-L3 worth) selects
//     that regime.
// Both thresholds are env-overridable for tuning to other shapes.

// ZENDNNL_GRP_MATMUL_DECDYN_EPC_MULT — cached, default 4.
//   DecodeDynamic when active_ops >= this * num_ccds (experts-per-CCD).
inline int get_grp_decdyn_epc_mult() {
    static constexpr int kDefault = 4;
    static const int v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_DECDYN_EPC_MULT");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed) || parsed < 1) return kDefault;
        return parsed;
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_DECDYN_WEI_L3_MULT — cached, default 2.
//   DecodeDynamic when wei_per_expert >= this * kL3PerCcdBytes (32 MB),
//   i.e. the per-expert weight is large enough that Rounds' L3-batching
//   serialises.  0 disables this branch (epc rule only).
inline int get_grp_decdyn_wei_l3_mult() {
    static constexpr int kDefault = 2;
    static const int v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_DECDYN_WEI_L3_MULT");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed) || parsed < 0) return kDefault;
        return parsed;
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_DECDYN_CK_INT8_FUSED — cached, default 1 (ON).
//   Whether a `use_custom` DQ-INT8 FUSED call may engage DecodeDynamic,
//   moving int8 fused-MoE Op1 (gate+up) off the barriered Rounds executor.
//   Set 0 to restore the old unconditional exclusion if a shape regresses;
//   `dyn_single_pool_safe` in plan_group_n_tile says why it is redundant.
inline bool get_grp_decdyn_ck_int8_fused() {
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_DECDYN_CK_INT8_FUSED");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return true; // default / junk: On
        return parsed != 0;
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_N_TILE_CCD_FILL = { 0, 1 } — cached, default 1 (ON).
//   Kill switch for the multi-expert CCD-fill N-tile sizer
//   (`ccd_fill_min_n_tile`), used only by the DecodeDynamic executor: it
//   shrinks the tile for experts whose fixed tile yields fewer tiles than
//   the CCD has lanes, leaving wider-N experts on `phase_cap`.  Set to 0 to
//   restore the legacy fixed decode tile.  Non-numeric input → default 1.
inline bool get_grp_n_tile_ccd_fill() {
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_N_TILE_CCD_FILL");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return true; // default / junk: On
        return parsed != 0;
    }();
    return v;
}

// DecodeDynamic's expert-count gate in `plan_group_n_tile`.
// MULTI-EXPERT ONLY.  `execute_decode_dynamic` assigns whole experts
// to individual CCDs, so a lone expert would run on ONE CCD and idle
// the rest of the team.  `huge_weight` alone is clearable by a single
// dense W13, so without this clause such a call would be pulled in on
// weight size and bypass `se_adaptive_min_n_tile`, its proper sizer.
// ACTIVE count, not `topo.num_ops`: the latter is the padded slot
// count, so a 128-slot call with one firing expert would read as
// multi-expert, clear `huge_weight`, and enter DecodeDynamic with a
// single CCD working — precisely the case this clause excludes.
inline bool grp_matmul_decode_dynamic_multi_expert(int active_ops) {
    return active_ops > 1;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_MULTI_EXPERT_POLICY_HPP
