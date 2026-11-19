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

#include "prompt_tile_executor.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <string>

#include "../../../common/group_matmul_parallel_common.hpp"
#include "../../../single_expert/single_expert_hooks.hpp"
#include "common/zendnnl_compat.hpp"
#include "prompt_tile_planner.hpp"
#include "prompt_tile_topology.hpp"
#include "prompt_tile_tunables.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace prompt_tile {

namespace {

using custom_kernel::KernelVariant;

struct alignas(64) Cursor {
    std::atomic<int> next {0};
    int end = 0;
};

/// Per-calling-thread scratch reused across calls (no per-call allocation
/// once warmed).
struct Scratch {
    Plan plan;
    std::unique_ptr<Cursor[]> cursors;
    int cursor_cap = 0;
    /// First owner slot of each team (`Plan::owner_begin`).
    std::vector<int> slot_of_team;
    /// A `src_u8` call's source rows (`prompt_tile_execute`).  Bounded like
    /// `reorder_quant_buffers_t`: a block 8x the request is reallocated, and
    /// one above `kU8RowsKeepBytes` is freed after its call, so a long
    /// prompt's rows are not held through the decode that follows.
    std::unique_ptr<uint8_t[]> u8_rows;
    size_t u8_cap = 0;
    static constexpr size_t kU8RowsKeepBytes = size_t {64} << 20;

    Cursor *cursors_for(int n) {
        if (n > cursor_cap) {
            cursors.reset(new Cursor[n]);
            cursor_cap = n;
        }
        return cursors.get();
    }

    uint8_t *u8_rows_for(size_t bytes) {
        const bool fits = u8_rows != nullptr && bytes <= u8_cap;
        if (fits && (u8_cap <= (size_t {1} << 20) || bytes >= u8_cap / 8))
            return u8_rows.get();
        uint8_t *p = new (std::nothrow) uint8_t[bytes];
        if (p == nullptr) return fits ? u8_rows.get() : nullptr;
        u8_rows.reset(p);
        u8_cap = bytes;
        return p;
    }

    void u8_rows_done() {
        if (u8_cap > kU8RowsKeepBytes) {
            u8_rows.reset();
            u8_cap = 0;
        }
    }
};

Scratch &scratch() {
    static thread_local Scratch s;
    return s;
}

/// The family the executor plans `v` as; false when it does not serve `v`.
bool resolve_kernel_type(KernelVariant v, kernel_type_t &fam) {
    if (v == KernelVariant::kBF16_BF16_BF16
            || v == KernelVariant::kBF16_BF16_F32)
        fam = kernel_type_t::kBf16;
    else if (custom_kernel::is_int8_variant(v))
        fam = kernel_type_t::kInt8;
    else if (custom_kernel::is_w4a8_variant(v))
        fam = kernel_type_t::kW4a8;
    else if (custom_kernel::is_f16_variant(v))
        fam = kernel_type_t::kF16;
    else
        return false;
    return true;
}

size_t scale_bytes(const custom_kernel::CallContext &kctx) {
    return kctx.scale_kind == custom_kernel::ScaleKind::kBf16 ? 2 : 4;
}

/// Bytes the CK streams per weight column beyond its K weights: the INT8
/// compensation, W4A8's per-group compensation and weight scale, or
/// MXFP4's one in-pack scale byte per group.
int64_t wei_col_extra_bytes(
        const custom_kernel::CallContext &kctx, kernel_type_t fam, int64_t K) {
    if (fam == kernel_type_t::kInt8) {
        // Per-group Q8 stores one int32 compensation row per group.
        if (kctx.group_size > 0 && K > 0 && (K % kctx.group_size) == 0)
            return (K / kctx.group_size)
                    * static_cast<int64_t>(sizeof(int32_t));
        return static_cast<int64_t>(sizeof(int32_t));
    }
    if (fam != kernel_type_t::kW4a8 || kctx.group_size <= 0) return 0;
    const custom_kernel::W4Format fmt
            = custom_kernel::w4_format_for_variant(kctx.variant);
    const int64_t per_group
            = static_cast<int64_t>(
                      custom_kernel::w4_tail_bytes_per_group_col(fmt))
            + (custom_kernel::w4_scales_in_pack(fmt)
                            ? 0
                            : static_cast<int64_t>(scale_bytes(kctx)));
    return K / kctx.group_size * per_group;
}

ShapeKey shape_key(kernel_type_t fam, const CallView &call, int T) {
    ShapeKey key;
    key.fam = fam;
    key.T = T;
    for (size_t e = 0; e < call.M.size(); ++e) {
        const int m = call.M[e];
        if (m <= 0) continue;
        if (key.M == 0) {
            key.K = call.K[e];
            key.N = call.N[e];
        }
        key.M += m;
        key.Mmax = std::max<int64_t>(key.Mmax, m);
    }
    return key;
}

/// What `plan_call` planned with.
struct PlannedCall {
    kernel_type_t fam = kernel_type_t::kBf16;
    ShapeKey key;
    Resolved r;
    const TeamMap *tm = nullptr;
    int kc = 0; ///< K chunk of the call's tiles (`se_prompt_tile_kc`).
};

/// Resolves the call's knobs, fitted to its row alignment, and plans it.
/// Flattened, so `plan_tiles` is inlined into the per-call planning step.
ZENDNNL_FLATTEN PlannedCall plan_call(const custom_kernel::CallContext &kctx,
        const CallView &call, int T, Plan &plan) {
    PlannedCall p;
    resolve_kernel_type(kctx.variant, p.fam);
    const kernel_traits_t &ff = facts(p.fam);
    const int row_align
            = ff.row_align > 0 ? ff.row_align : std::max(kctx.max_mr, 1);
    p.key = shape_key(p.fam, call, T);
    p.r = resolve(p.key);
    sanitize(p.r.t, row_align, ff.min_mr);
    p.tm = &get_team_map(T, p.r.t[kTeam], p.r.t[kSubteam]);

    PlannerConfig cfg;
    cfg.tun = p.r.t;
    cfg.pack_nr = kctx.pack_nr;
    cfg.max_mr = kctx.max_mr;
    cfg.l2_bytes = p.tm->l2_bytes;
    cfg.l3_bytes_per_team = p.tm->l3_bytes;
    cfg.a_bytes = ff.a_bytes;
    cfg.wei_bits = ff.wei_bits;
    cfg.wei_col_extra_bytes = wei_col_extra_bytes(kctx, p.fam, p.key.K);
    cfg.min_mr = ff.min_mr;
    cfg.row_align = row_align;
    cfg.num_threads = T;
    cfg.single_dense = se_dense_prompt_plan(
            call.M.size(), p.fam == kernel_type_t::kBf16);
    cfg.balanced = se_balanced_prompt_plan(call.M.size(), kctx.variant);
    p.kc = se_prompt_tile_kc(kctx, p.r.t[kKc], p.key.K);
    cfg.kc = p.kc;

    plan_tiles(call.M.data(), call.N.data(), call.K.data(),
            static_cast<int>(call.M.size()), cfg, p.tm->team_size.data(),
            p.tm->n_teams, plan);
    return p;
}

std::string join_sizes(const std::vector<int> &v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ',';
        s += std::to_string(v[i]);
    }
    return s;
}

std::string join_knobs(const Tunables &k) {
    std::string s;
    for (int i = 0; i < kNumKnobs; ++i) {
        if (i) s += ',';
        s += kKnobs[i].name;
        s += ':';
        s += std::to_string(k.v[i]);
    }
    return s;
}

} // namespace

bool should_execute_prompt_tile(const custom_kernel::CallContext &kctx,
        bool use_custom, bool rounds_strategy, bool prompt_phase,
        const std::vector<int> &M, const std::vector<int> &N,
        bool single_expert_prompt) {
    if (!use_custom || !kctx.enabled || !rounds_strategy || !prompt_phase)
        return false;
    kernel_type_t fam = kernel_type_t::kBf16;
    if (!resolve_kernel_type(kctx.variant, fam)) return false;
    if (facts(fam).kblock && grp_matmul_kblock_enabled(kctx.kblock_auto))
        return false;
    if (kctx.pack_nr <= 0) return false;
    int active = 0;
    for (size_t e = 0; e < M.size(); ++e) {
        if (M[e] <= 0) continue;
        if (N[e] <= 0 || N[e] % kctx.pack_nr != 0) return false;
        ++active;
    }
    // A single-expert call (a dense layer) splits its M across teams the same
    // way a heavy shared expert does, under the single-expert CK policy
    // rather than the multi-expert family mask; `enable=0` still keeps every
    // prompt on rounds.  One active expert among idle MoE slots keeps
    // the multi-expert scope.
    if (active == 1) return single_expert_prompt && global_value(kEnable) != 0;
    return active > 1 && enabled(fam);
}

void prompt_tile_execute(custom_kernel::CallContext &kctx, const CallView &call,
        int num_threads, bool log_plan) {
    const int T = std::max(num_threads, 1);
    Scratch &s = scratch();
    Plan &plan = s.plan;
    const PlannedCall p = plan_call(kctx, call, T, plan);
    kctx.prompt_kc = p.kc;
    const TeamMap &tm = *p.tm;
    const Tunables &k = p.r.t;
    const int num_ops = static_cast<int>(call.M.size());

    const int n_teams = tm.n_teams;
    const int ntiles = static_cast<int>(plan.tiles.size());
    // Slot `n_teams` is the whole list, used when the runtime hands the
    // region a different team size than the map was built for.
    // Owner slots follow: one per thread, team-major (`Plan::owner_begin`).
    const bool owned = !plan.owner_begin.empty()
            && static_cast<int>(plan.owner_begin.size()) == tm.num_threads + 1;
    Cursor *cur = s.cursors_for(n_teams + 1 + (owned ? tm.num_threads : 0));
    for (int t = 0; t < n_teams; ++t) {
        cur[t].next.store(plan.team_begin[t], std::memory_order_relaxed);
        cur[t].end = plan.team_begin[t + 1];
    }
    cur[n_teams].next.store(0, std::memory_order_relaxed);
    cur[n_teams].end = ntiles;
    Cursor *slot_cur = cur + n_teams + 1;
    if (owned) {
        s.slot_of_team.resize(static_cast<size_t>(n_teams) + 1);
        s.slot_of_team[0] = 0;
        for (int t = 0; t < n_teams; ++t)
            s.slot_of_team[t + 1] = s.slot_of_team[t] + tm.team_size[t];
        for (int o = 0; o < tm.num_threads; ++o) {
            slot_cur[o].next.store(
                    plan.owner_begin[o], std::memory_order_relaxed);
            slot_cur[o].end = plan.owner_begin[o + 1];
        }
    }
    const int *slot_of_team = s.slot_of_team.data();

    if (log_plan) {
        const ShapeKey &key = p.key;
        apilog_info("[GRP_MATMUL.PLAN.PTILE] fam=", facts(p.fam).name, " key=K",
                key.K, ",N", key.N, ",M", key.M, ",Mmax", key.Mmax, ",T", key.T,
                " match=env:", p.r.env_entry, " tiles=", ntiles,
                " mb=", plan.mb, " ns=", plan.ns, " teams=", n_teams,
                " team_src=", (tm.from_l3 ? "l3" : "contiguous"),
                " team_sizes=", join_sizes(tm.team_size),
                " team_domains=", join_sizes(tm.domain_of_team),
                " l2=", tm.l2_bytes, " l3=", tm.l3_bytes,
                " est_imbalance=", plan.est_imbalance, " makespan=",
                plan_makespan(plan, call.K.data(), k[kRb], tm.team_size.data(),
                        n_teams),
                " owned=", owned ? 1 : 0, " num_ops=", num_ops,
                " num_threads=", T, " knobs=", join_knobs(k));
    }
    test_api::s_calls_served.fetch_add(1, std::memory_order_relaxed);
    if (ntiles == 0) return;

    const Tile *tiles = plan.tiles.data();
    const int *victims = tm.steal_order.data();
    const int grab = k[kGrab];
    const bool steal = k[kSteal] != 0;
    const size_t dst_elem = custom_kernel::dst_elem_bytes(kctx.variant);
    const size_t src_elem = facts(p.fam).a_bytes;
    const size_t scale_elem = scale_bytes(kctx);
    const QuantizedInputView *quant = call.quant;
    // Per-row source and per-channel weight scales only: static U8xS8, whose
    // epilogue requantizes and applies post-ops, stays on the round executor.
    const custom_kernel::Int8EpilogueConfig int8_epilogue {};
    // Q8 source scales are one per K-group. W4A8 per-token stays at one
    // scale per row; matching `{M,G}` uses `kctx.src_scale_groups`.
    const bool q8_src_groups = kctx.group_size == 32
            && custom_kernel::is_int8_variant(kctx.variant);

    // DQ-INT8 / W4A8 NR32 single expert: the team copies the source rows
    // XOR-ed with 0x80 (u8) at the top of the parallel region, so the DQ-INT8
    // K-loop skips that XOR per broadcast.  The copy is one pass over the
    // rows; the XOR it saves recurs for every NR block.  Below
    // `kU8SrcMinMacsPerThread` the copy and its barrier cost more than that.
    constexpr int64_t kU8SrcMinMacsPerThread = int64_t {1} << 24;
    int u8_ld = 0;
    uint8_t *u8_base = nullptr;
    if (quant != nullptr && num_ops == 1 && call.M[0] > 0
            && static_cast<int64_t>(call.M[0]) * call.N[0] * call.K[0]
                    >= kU8SrcMinMacsPerThread * T
            && custom_kernel::u8_src_eligible(kctx)) {
        u8_ld = (call.K[0] + 63) / 64 * 64;
        u8_base = s.u8_rows_for(static_cast<size_t>(call.M[0]) * u8_ld);
        if (u8_base != nullptr && !custom_kernel::use_u8_src(kctx))
            u8_base = nullptr;
    }

    auto run_tile = [&](const Tile &t) {
        const int e = t.e;
        const size_t m0 = static_cast<size_t>(t.m0);
        auto *d = static_cast<char *>(call.dst[e])
                + m0 * call.ldc[e] * dst_elem;
        if (quant != nullptr) {
            const QuantizedInputView &q = quant[e];
            custom_kernel::Int8EpilogueConfig epi = int8_epilogue;
            int src_groups = 1;
            if (q8_src_groups) {
                epi.group_k = 32;
                epi.src_row_groups = call.K[e] / 32;
                epi.wei_group_stride = call.N[e];
                src_groups = epi.src_row_groups;
            }
            const char *a = static_cast<const char *>(q.src) + m0 * q.lda;
            int lda = q.lda;
            if (u8_base != nullptr) {
                a = reinterpret_cast<const char *>(u8_base) + m0 * u8_ld;
                lda = u8_ld;
            }
            const int32_t *sums = nullptr;
            if (q.src_group_sums != nullptr) {
                const size_t groups
                        = static_cast<size_t>(call.K[e] / kctx.group_size);
                sums = q.src_group_sums + m0 * groups;
            }
            custom_kernel::dispatch_tile(kctx, e, t.rows, call.K[e], t.ncols,
                    t.n0, a, lda, call.bias[e], d, call.ldc[e],
                    static_cast<const char *>(q.src_scale)
                            + m0
                                    * static_cast<size_t>(q8_src_groups
                                                    ? src_groups
                                                    : kctx.src_scale_groups)
                                    * scale_elem,
                    q.src_zp != nullptr ? q.src_zp + m0 : nullptr, q.wei_scale,
                    epi, sums);
            return;
        }
        const auto *a = static_cast<const char *>(call.src[e])
                + m0 * call.lda[e] * src_elem;
        custom_kernel::dispatch_tile(kctx, e, t.rows, call.K[e], t.ncols, t.n0,
                a, call.lda[e], call.bias[e], d, call.ldc[e],
                /*src_scale=*/nullptr, /*src_zp=*/nullptr,
                /*wei_scale=*/nullptr, int8_epilogue);
    };
    auto drain = [&](Cursor &c) {
        const int end = c.end;
        for (;;) {
            const int i = c.next.fetch_add(grab, std::memory_order_relaxed);
            if (i >= end) return;
            const int stop = std::min(end, i + grab);
            for (int j = i; j < stop; ++j)
                run_tile(tiles[j]);
        }
    };

#pragma omp parallel num_threads(T)
    {
        if (u8_base != nullptr) {
            const int rows = call.M[0];
            const int per = (rows + omp_get_num_threads() - 1)
                    / omp_get_num_threads();
            const int r0 = std::min(rows, omp_get_thread_num() * per);
            const int r1 = std::min(rows, r0 + per);
            if (r1 > r0) {
                custom_kernel::copy_s8_rows_as_u8(
                        static_cast<const int8_t *>(quant[0].src)
                                + static_cast<size_t>(r0) * quant[0].lda,
                        quant[0].lda, u8_base + static_cast<size_t>(r0) * u8_ld,
                        u8_ld, r1 - r0, call.K[0]);
            }
#pragma omp barrier
        }
        if (owned && omp_get_num_threads() == tm.num_threads) {
            // Own slot, then teammates' (same L3) when stealing.  Not other
            // teams': a slot holds a thread's few tiles, and an atomic on
            // every slot of a 64-thread call costs ~50 us across domains.
            const int tid = omp_get_thread_num();
            const int my = tm.team_of[tid];
            const int base = slot_of_team[my];
            const int size = tm.team_size[my];
            const int rank = tm.rank_in_team[tid];
            drain(slot_cur[base + rank]);
            for (int i = 1; steal && i < size; ++i) {
                Cursor &c = slot_cur[base + (rank + i) % size];
                if (c.next.load(std::memory_order_relaxed) < c.end) drain(c);
            }
        } else if (omp_get_num_threads() == tm.num_threads) {
            const int my = tm.team_of[omp_get_thread_num()];
            drain(cur[my]);
            if (steal && n_teams > 1) {
                const int *v
                        = victims + static_cast<size_t>(my) * (n_teams - 1);
                for (int i = 0; i < n_teams - 1; ++i)
                    drain(cur[v[i]]);
            }
        } else {
            drain(cur[n_teams]);
        }
    }
    if (u8_base != nullptr) {
        custom_kernel::use_s8_src(kctx);
        s.u8_rows_done();
    }
}

namespace test_api {
const Plan &plan_only(const custom_kernel::CallContext &kctx,
        const CallView &call, int num_threads) {
    Plan &plan = scratch().plan;
    plan_call(kctx, call, std::max(num_threads, 1), plan);
    return plan;
}
} // namespace test_api

const char *execution_path_label(
        bool fused_epilogue, grp_matmul_gated_act_t act) {
    if (!fused_epilogue) return "flat_n_tile_ptile_custom";
    switch (act) {
        case grp_matmul_gated_act_t::silu_and_mul:
            return "flat_n_tile_ptile_fused_silu_and_mul_tight_custom";
        case grp_matmul_gated_act_t::gelu_and_mul:
            return "flat_n_tile_ptile_fused_gelu_and_mul_tight_custom";
        default: return "flat_n_tile_ptile_fused_swiglu_oai_tight_custom";
    }
}

} // namespace prompt_tile
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
