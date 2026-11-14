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

/// Single-expert SELECTION POLICY.
///
/// Owns every decision specific to a group-matmul call of exactly one expert
/// (`num_ops == 1`), so it lives with that workload rather than in the shared
/// dispatcher: the single-op scope predicates, the Rule 0.45 dense-decode
/// knob, the three single-expert custom-kernel dtype-set knobs and their
/// classifier (`grp_matmul_single_expert_ck`), the Rule 0.46 / 0.45 AUTO
/// helper, the prepack-side layer test, the custom-kernel epilogue terms, and
/// the `test_api` atoms behind the knobs.
///
/// It also declares `grp_matmul_single_dense_call`, which the WC=2 phase-split
/// check in `selection/algo_select.hpp` needs.
///
/// Library-internal.  Included by the dispatcher, the N-tile executor, the
/// custom-kernel dispatch and `matmul_direct` route, prepack, the fused-MoE
/// entry and `gtests/group_matmul/moe_test_utils.hpp` (the RAII override
/// guards).

#ifndef ZENDNNL_GROUP_MATMUL_SINGLE_EXPERT_POLICY_HPP
#define ZENDNNL_GROUP_MATMUL_SINGLE_EXPERT_POLICY_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_env.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// ── Single-op (num_ops == 1) specialisation scope ───────────────────────
// SINGLE SOURCE OF TRUTH for the optimisations added for a lone expert.
//
// WHAT IS ACTUALLY TESTED: `num_ops == 1`, plus `max_M <= kDecodeMaxM` for
// the decode refinement.  Nothing here inspects operand shapes, so "dense
// FFN" / "W13/W2" in these doc-blocks names the workload the paths were
// TUNED for, not a property the predicates verify: "dense" means "a
// single-op group" (i.e. not MoE), and any other single-op grouped matmul
// in the decode M band is in scope too.  That is deliberate — all three
// consumers are shape-generic perf heuristics with their own guards (Rule
// 0.45 also requires `n_tile_safe`; the kblock gate keys on per-expert M
// and dtype) — so a non-FFN single-op caller gets a differently-tuned but
// equally valid plan.  Add operand-shape inputs here if a consumer ever
// needs FFN specifically.
//
// Two predicates, so each consumer takes exactly the scope it needs:
//   `is_single_dense_expert` (phase-agnostic) guards adaptive N-tile sizing
//       in `se_adaptive_min_n_tile`, which serves BOTH phases and so must not
//       be decode-gated.
//   `is_dense_ffn_decode` (adds the decode gate) guards Rule 0.45 ALGO-3
//       routing and the auto deep-K K-blocking in `prepare_for_call`; both
//       must fire only in decode (a single-expert prompt takes Rule 0.46 or
//       Rule 0.7).
inline bool is_single_dense_expert(int num_ops) {
    return num_ops == 1;
}
inline bool is_dense_ffn_decode(int num_ops, int max_M) {
    return is_single_dense_expert(num_ops) && max_M <= kDecodeMaxM;
}

namespace test_api {

// Sentinel `-1` = no override (use the cached env path, default ON).
// `0` / `1` force Rule 0.45's env gate
// (`ZENDNNL_GRP_MATMUL_DENSE_DECODE_NTILE`) off / on for the lifetime of
// a test.  Needed because the getter caches its env read in a
// function-local `static const`, so a gtest cannot A/B the rule by
// setting the env var after the first call has already latched it.
inline std::atomic<int> s_grp_matmul_dense_decode_ntile_override {-1};

// A `single_expert_dtype` mask for
// `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_TILE` for the lifetime of a test;
// `-1` uses the cached env value.
inline std::atomic<int> s_grp_matmul_single_expert_prompt_tile_override {-1};

// A `single_expert_dtype` mask for
// `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_DECODE_TILE` for the lifetime of a test;
// `-1` uses the cached env value.
inline std::atomic<int> s_grp_matmul_single_expert_decode_tile_override {-1};

// A `single_expert_dtype` mask for `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_FORCE_CK`
// for the lifetime of a test; `-1` uses the cached env value.
inline std::atomic<int> s_grp_matmul_single_expert_force_ck_override {-1};

// `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_KC` for the lifetime of a test;
// `-1` uses the cached env value.
inline std::atomic<int> s_grp_matmul_single_expert_prompt_kc_override {-1};

// `single_expert_dtype` masks for
// `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_{PROMPT,DECODE}_BALANCE` for the lifetime
// of a test; `-1` uses the cached env value.
inline std::atomic<int> s_grp_matmul_single_expert_prompt_balance_override {-1};
inline std::atomic<int> s_grp_matmul_single_expert_decode_balance_override {-1};

// `ZENDNNL_GRP_MATMUL_N_TILE_OVERSUB` for the lifetime of a test; `-1` uses
// the cached env value.
inline std::atomic<int> s_grp_n_tile_oversub_override {-1};

} // namespace test_api

inline std::atomic<int> &test_api_dense_decode_ntile_override() {
    return test_api::s_grp_matmul_dense_decode_ntile_override;
}
inline std::atomic<int> &test_api_single_expert_prompt_tile_override() {
    return test_api::s_grp_matmul_single_expert_prompt_tile_override;
}
inline std::atomic<int> &test_api_single_expert_decode_tile_override() {
    return test_api::s_grp_matmul_single_expert_decode_tile_override;
}
inline std::atomic<int> &test_api_single_expert_force_ck_override() {
    return test_api::s_grp_matmul_single_expert_force_ck_override;
}

// ZENDNNL_GRP_MATMUL_DENSE_DECODE_NTILE = { 0, 1 } — cached, default 1 (ON).
//   AUTO-only (`ALGO=0`) routing knob for Rule 0.45: a lone expert
//   (`num_ops == 1`) in decode goes to ALGO 3 rather than being diverted by
//   Rule 0.5's low-occupancy arm to ALGO 1, so it reaches the N-tile planner.
//   Set `=0` to restore the generic occupancy decision for A/B measurement.
//   Scope:
//   num_ops==1 + decode + n_tile_safe + phase env NOT explicitly pinned.
//   Non-numeric input → default 1.  The `test_api` override atom above lets
//   a gtest A/B the rule mid-process, which the cached env read cannot.
inline bool get_grp_matmul_dense_decode_ntile() {
    const int ovr = test_api_dense_decode_ntile_override().load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_DENSE_DECODE_NTILE");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return true; // default / junk: On
        return parsed != 0;
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_N_TILE_OVERSUB = { >= 0 } — cached, default 2.
//   Oversubscription factor for adaptive N-tile sizing
//   (`se_adaptive_min_n_tile`), which targets
//   `min(OVERSUB * team_per_expert, num_threads)` N-tiles so the tile COUNT
//   tracks the thread count instead of a fixed constant.  SCOPE:
//   `num_ops == 1` only; real MoE keeps the legacy fixed tile.
//
//   EFFECTIVELY AN ON/OFF SWITCH TODAY: in that scope
//   `team_per_expert == num_threads`, so the cap collapses every value >= 1
//   onto the same target and `0` is the only distinct setting (it disables
//   adaptive sizing).  The multiplier is kept rather than reduced to a bool
//   because the multi-expert generalisation needs it: once `num_ops > 1` is
//   in scope, `team_per_expert < num_threads` leaves real headroom under the
//   cap.  Non-numeric input → default 2.
inline int get_grp_n_tile_oversub() {
    const int ovr = test_api::s_grp_n_tile_oversub_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr;
    static const int v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_N_TILE_OVERSUB");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed)) return 2; // default
        return (parsed >= 0) ? parsed : 2;
    }();
    return v;
}

// Single-expert custom-kernel policy — three cached dtype-set knobs for
// group-matmul calls of exactly one expert (one expert in the pool, not one
// active expert of an MoE pool) at an N the custom-kernel pack divides.  A
// covered phase runs ALGO 3 on the custom kernel (Rule 0.46), and
// `matmul_direct` routes the layer.
//   * ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_TILE (default `all`) — prompt,
//     on the CK prompt tile executor; not static U8xS8, which the executor
//     does not tile; not under `ZENDNNL_GRP_MATMUL_PROMPT_TILE=0`,
//     K-blocking or `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=1`.
//   * ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_DECODE_TILE (default `all`) — decode,
//     on the CK N-tile path.
//   * ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_FORCE_CK (default `all`) — the phase
//     knobs cover a dtype it lists at any N; any other dtype only at decode's
//     N-tile count (`has_enough_single_expert_tiles`).  It never adds a phase.
// Each value is a list of tokens joined by `,`, `+` or whitespace
// (case-insensitive): `bf16`, `f16`, `int8` (DQ-INT8), `u8s8` (static
// U8xS8), `w4a8`, `all` / `1` / `on` / `true` / `yes` (every dtype) and
// `none` / `0` / `off` / `false` / `no` (nothing).  Unknown tokens are
// ignored; a value with no known token keeps the default.  Both are warned
// once.  Read once per process.  No knob applies under
// `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=0`.  With both phases covered the layer
// has one custom-kernel pack, which WEIGHT_CACHE=2 may store in place.  An
// explicit `ZENDNNL_GRP_MATMUL_AUTO_{PROMPT,DECODE}_ALGO` pin and the CK
// sub-knobs still apply.
enum single_expert_dtype : unsigned {
    kSingleExpertBf16 = 1u << 0,
    kSingleExpertF16 = 1u << 1,
    kSingleExpertInt8 = 1u << 2,
    kSingleExpertU8s8 = 1u << 3,
    kSingleExpertW4a8 = 1u << 4,
    kSingleExpertAll = (1u << 5) - 1,
};

/// Parse a single-expert dtype-set value.  Unset → `fallback`; a value with
/// no known token → `fallback`, with `*unknown` set.  Unknown tokens are
/// appended to `*unknown` (space-separated) when it is non-null.
inline unsigned parse_grp_matmul_single_expert_dtypes(
        const char *value, unsigned fallback, std::string *unknown = nullptr) {
    if (value == nullptr) return fallback;
    static constexpr std::pair<const char *, unsigned> kTokens[]
            = {{"0", 0u}, {"none", 0u}, {"off", 0u}, {"false", 0u}, {"no", 0u},
                    {"1", kSingleExpertAll}, {"all", kSingleExpertAll},
                    {"on", kSingleExpertAll}, {"true", kSingleExpertAll},
                    {"yes", kSingleExpertAll}, {"bf16", kSingleExpertBf16},
                    {"f16", kSingleExpertF16}, {"int8", kSingleExpertInt8},
                    {"u8s8", kSingleExpertU8s8}, {"w4a8", kSingleExpertW4a8}};
    const auto is_sep = [](char c) {
        return c == ',' || c == '+' || c == ' ' || c == '\t' || c == '\n'
                || c == '\r';
    };
    unsigned mask = 0;
    bool known = false;
    bool any_token = false;
    for (const char *p = value; *p != '\0';) {
        while (*p != '\0' && is_sep(*p))
            ++p;
        const char *start = p;
        while (*p != '\0' && !is_sep(*p))
            ++p;
        const size_t len = static_cast<size_t>(p - start);
        if (len == 0) continue;
        any_token = true;
        bool matched = false;
        for (const auto &[name, bits] : kTokens) {
            if (std::strlen(name) != len) continue;
            matched = true;
            for (size_t i = 0; i < len && matched; ++i) {
                const char c = start[i];
                matched = (c >= 'A' && c <= 'Z'
                                          ? static_cast<char>(c - 'A' + 'a')
                                          : c)
                        == name[i];
            }
            if (matched) {
                mask |= bits;
                break;
            }
        }
        known |= matched;
        if (!matched && unknown != nullptr) {
            if (!unknown->empty()) unknown->push_back(' ');
            unknown->append(start, len);
        }
    }
    if (!known && !any_token && unknown != nullptr) *unknown = "<empty>";
    return known ? mask : fallback;
}

/// Logs the unknown tokens of a knob value once, when its getter first reads
/// it.  Defined in single_expert_policy.cpp.
void warn_single_expert_knob_value(
        const char *name, const std::string &unknown);
/// The cached dtype set of one knob.
inline unsigned read_grp_matmul_single_expert_knob(
        const char *name, unsigned fallback) {
    std::string unknown;
    const unsigned v = parse_grp_matmul_single_expert_dtypes(
            std::getenv(name), fallback, &unknown);
    if (!unknown.empty()) warn_single_expert_knob_value(name, unknown);
    return v;
}
/// Dtype sets of the three knobs; a test override `>= 0` is the mask itself.
inline unsigned get_grp_matmul_single_expert_prompt_tile_dtypes() {
    const int ovr = test_api_single_expert_prompt_tile_override().load(
            std::memory_order_relaxed);
    if (ovr >= 0) return static_cast<unsigned>(ovr) & kSingleExpertAll;
    // Explicit `ZENDNNL_GRP_MATMUL_ALGO=0` keeps every prompt on AOCL-DLP.
    if (get_grp_matmul_algo_legacy()) return 0u;
    static const unsigned v = read_grp_matmul_single_expert_knob(
            "ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_TILE", kSingleExpertAll);
    return v;
}
inline unsigned get_grp_matmul_single_expert_decode_tile_dtypes() {
    const int ovr = test_api_single_expert_decode_tile_override().load(
            std::memory_order_relaxed);
    if (ovr >= 0) return static_cast<unsigned>(ovr) & kSingleExpertAll;
    static const unsigned v = read_grp_matmul_single_expert_knob(
            "ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_DECODE_TILE", kSingleExpertAll);
    return v;
}
inline unsigned get_grp_matmul_single_expert_force_ck_dtypes() {
    const int ovr = test_api_single_expert_force_ck_override().load(
            std::memory_order_relaxed);
    if (ovr >= 0) return static_cast<unsigned>(ovr) & kSingleExpertAll;
    static const unsigned v = read_grp_matmul_single_expert_knob(
            "ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_FORCE_CK", kSingleExpertAll);
    return v;
}

// Thread-count-exact work split for a single expert on the custom kernel,
// two cached dtype-set knobs (same tokens as above):
//   * ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_BALANCE (default `bf16`) — the
//     prompt tile planner gives every thread of every L3 team one N slice of
//     its team's rows, the rows sized so the busiest thread of each team
//     carries the same work (`plan_balanced`).  Otherwise BF16 keeps the
//     dense equal split, sized to `ceil(M / n_teams)`, whose whole tiles
//     leave one thread a second tile at 15, 31 or 63 threads.  The other
//     families' shrink loop (>= 8 tiles per thread) already scales at those
//     counts, and their kernels run some single large tiles slower.  `u8s8`
//     has no effect: the prompt tile does not take static U8xS8.
//   * ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_DECODE_BALANCE (default `all`) —
//     decode splits N in whole `pack_nr` blocks spread evenly over the
//     thread count when that leaves the busiest thread fewer blocks than
//     the exact-divisor tile, which idles threads that do not divide N.
//     Each thread keeps at least `ZENDNNL_GRP_MATMUL_N_TILE_FLOOR` columns;
//     an explicit `..._CUSTOM_KERNEL_N_TILE`, `..._N_TILE_OVERSUB=0` or
//     `..._N_TILE_STRATEGY` 0 or 1 keeps the tile they select.
inline unsigned get_grp_matmul_single_expert_prompt_balance_dtypes() {
    const int ovr
            = test_api::s_grp_matmul_single_expert_prompt_balance_override.load(
                    std::memory_order_relaxed);
    if (ovr >= 0) return static_cast<unsigned>(ovr) & kSingleExpertAll;
    static const unsigned v = read_grp_matmul_single_expert_knob(
            "ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_BALANCE",
            kSingleExpertBf16);
    return v;
}
inline unsigned get_grp_matmul_single_expert_decode_balance_dtypes() {
    const int ovr
            = test_api::s_grp_matmul_single_expert_decode_balance_override.load(
                    std::memory_order_relaxed);
    if (ovr >= 0) return static_cast<unsigned>(ovr) & kSingleExpertAll;
    static const unsigned v = read_grp_matmul_single_expert_knob(
            "ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_DECODE_BALANCE",
            kSingleExpertAll);
    return v;
}

/// Default K chunk of a deep-K single-expert prompt tile.
inline constexpr int kSingleExpertPromptKc = 4096;

/// ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_KC = { 0, n } — cached, default
///   `kSingleExpertPromptKc`.
///   K chunk of a single-expert BF16 / FP16 prompt tile that can chunk and
///   whose B of one call over the whole K overflows L2 (`se_prompt_kc`).
///   `0` keeps the full K.  Rounded down to an even depth; non-numeric or negative
///   input → default.
inline int get_grp_matmul_single_expert_prompt_kc() {
    const int ovr
            = test_api::s_grp_matmul_single_expert_prompt_kc_override.load(
                    std::memory_order_relaxed);
    if (ovr >= 0) return ovr & ~1;
    static const int v = []() {
        const char *e
                = std::getenv("ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_KC");
        int parsed = 0;
        if (!parse_env_int_strict(e, parsed) || parsed < 0)
            return kSingleExpertPromptKc;
        return parsed & ~1;
    }();
    return v;
}

inline constexpr int kSingleExpertTilesPerThread = 2;

/// Decode's N-tile count for a single expert: at least
/// `kSingleExpertTilesPerThread` custom-kernel column blocks per thread.
inline bool has_enough_single_expert_tiles(
        int N, int K, int num_threads) noexcept {
    const int pack_nr = custom_kernel::plan_pack_nr(K, N);
    if (pack_nr != custom_kernel::kNRMin && pack_nr != custom_kernel::kNRMax)
        return false;
    const int usable_n_tiles = N / pack_nr;
    const int requested_threads = std::max(1, num_threads);
    return usable_n_tiles >= requested_threads * kSingleExpertTilesPerThread;
}

/// The phases a one-expert layer runs on the custom kernel under the
/// single-expert policy; `forced` marks coverage below decode's N-tile count,
/// which only `..._FORCE_CK` grants.
struct single_expert_ck_t {
    bool prompt = false;
    bool decode = false;
    bool forced = false;
    bool any() const { return prompt || decode; }
    bool both() const { return prompt && decode; }
    bool covers(bool is_decode) const { return is_decode ? decode : prompt; }
};

/// The policy for one weight's dtype tuple.  `K`/`N` > 0 also require a
/// custom-kernel pack width that divides N.  When `num_threads` > 0, a dtype
/// `..._FORCE_CK` does not list also needs
/// `has_enough_single_expert_tiles(N, K, num_threads)`.  `static_u8s8` marks the static
/// U8xS8 form; `aocl_prepacked` an AOCL-DLP blocked weight, which the custom
/// kernel converts only for BF16 and static U8xS8.  Defined in
/// single_expert_policy.cpp.
single_expert_ck_t grp_matmul_single_expert_ck(data_type_t src, data_type_t wei,
        data_type_t dst, bool dynamic_quant, data_type_t compute,
        bool static_u8s8, bool aocl_prepacked, int K = 0, int N = 0,
        int num_threads = 0);

/// `grp_matmul_single_expert_ck` for one call's representative params.
single_expert_ck_t grp_matmul_single_expert_ck(
        const matmul_params &p, int K = 0, int N = 0, int num_threads = 0);

// Rule 0.46 scope for one call: one expert in the pool under the
// single-expert CK policy, on a call the custom kernel will engage for.
single_expert_ck_t grp_matmul_single_expert_call_ck(const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K,
        const std::vector<matmul_params> &params,
        grp_matmul_gated_act_t fused_act, int num_threads);

/// What the single-expert AUTO rules (Rule 0.46, then Rule 0.45) answer for
/// one call.  `hit` false means neither matched and AUTO continues with the
/// next rule; otherwise `algo` / `reason` / `want` are what `auto_select_algo`
/// reports for the matched rule.
struct single_expert_rule_pick {
    bool hit = false;
    int algo = 0;
    int want = 0;
    const char *reason = nullptr;
};

/// Rule 0.46 and Rule 0.45 of `auto_select_algo`, in that order.
/// `single_ck` is the call's resolved single-expert policy (empty when the
/// call is not n_tile_safe or the custom kernel would refuse its epilogue).
single_expert_rule_pick grp_matmul_single_expert_auto_rule(
        const single_expert_ck_t &single_ck, bool is_decode,
        bool phase_env_pinned, bool n_tile_safe, int num_ops, int max_M);

/// Prepack's view of a Rule 0.46 single dense layer: the single-expert CK
/// policy covers both phases of this raw (not AOCL-prepacked) weight.
bool grp_matmul_single_expert_prepack_layer(data_type_t src, data_type_t wei,
        data_type_t dst, bool dynamic_quant, data_type_t compute, int K, int N,
        int num_threads);

/// A call of exactly one expert in the pool (not one active expert of an
/// MoE pool) on a raw BF16/FP16 weight whose both phases the single-expert
/// CK policy covers: the layers whose custom-kernel pack may own the weight
/// buffer at WEIGHT_CACHE=2.  Defined in
/// `single_expert/single_expert_policy.cpp`.
bool grp_matmul_single_dense_call(const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K,
        const std::vector<matmul_params> &params, int num_threads);

/// Epilogue terms of `custom_kernel::prepare_for_call` that the selector
/// cannot see.  Rule 0.46 needs them, or it could pick ALGO 3 for a call the
/// CK then refuses, which would run N-tile rounds on AOCL tiles.  `tight` is
/// whether a gated output is in the halved-width layout the CK writes.
inline bool single_expert_ck_epilogue_supported(float alpha, float beta,
        const void *bias, bool tight, grp_matmul_gated_act_t act,
        data_type_t act_dtype) {
    if (alpha != 1.0f || beta != 0.0f) return false;
    if (act == grp_matmul_gated_act_t::none) return true;
    if ((act_dtype != data_type_t::bf16 && act_dtype != data_type_t::f16)
            || !tight)
        return false;
    // Split-halves gated activations have no fused bias yet.
    return bias == nullptr || act == grp_matmul_gated_act_t::swiglu_oai_mul;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_SINGLE_EXPERT_POLICY_HPP
