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
#include <string>

#include "../group_matmul_parallel_common.hpp"
#include "../n_tile/group_matmul_n_tile.hpp"
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

    Cursor *cursors_for(int n) {
        if (n > cursor_cap) {
            cursors.reset(new Cursor[n]);
            cursor_cap = n;
        }
        return cursors.get();
    }
};

Scratch &scratch() {
    static thread_local Scratch s;
    return s;
}

/// The family the executor plans `v` as; false when it does not serve `v`.
bool family_of(KernelVariant v, Family &fam) {
    if (v == KernelVariant::kBF16_BF16_BF16
            || v == KernelVariant::kBF16_BF16_F32)
        fam = Family::kBf16;
    else if (custom_kernel::is_int8_variant(v))
        fam = Family::kInt8;
    else if (custom_kernel::is_s4_variant(v))
        fam = Family::kW4a8;
    else if (custom_kernel::is_f16_variant(v))
        fam = Family::kF16;
    else
        return false;
    return true;
}

size_t scale_bytes(const custom_kernel::CallContext &kctx) {
    return kctx.scale_kind == custom_kernel::ScaleKind::kBf16 ? 2 : 4;
}

/// Bytes the CK streams per weight column beyond its K weights: the INT8
/// compensation, or W4A8's per-group compensation and weight scale.
int64_t wei_col_extra_bytes(
        const custom_kernel::CallContext &kctx, Family fam, int64_t K) {
    if (fam == Family::kInt8) return 4;
    if (fam != Family::kW4a8 || kctx.group_size <= 0) return 0;
    return K / kctx.group_size * static_cast<int64_t>(4 + scale_bytes(kctx));
}

ShapeKey shape_key(Family fam, const CallView &call, int T) {
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
struct Planned {
    Family fam = Family::kBf16;
    ShapeKey key;
    Resolved r;
    const TeamMap *tm = nullptr;
};

/// Resolves the call's knobs, fitted to its row alignment, and plans it.
/// Flattened, so `plan_tiles` is inlined into the per-call planning step.
ZENDNNL_FLATTEN Planned plan_call(const custom_kernel::CallContext &kctx,
        const CallView &call, int T, Plan &plan) {
    Planned p;
    family_of(kctx.variant, p.fam);
    const FamilyFacts &ff = facts(p.fam);
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

bool should_run(const custom_kernel::CallContext &kctx, bool use_custom,
        bool rounds_strategy, bool prompt_phase, const std::vector<int> &M,
        const std::vector<int> &N) {
    if (!use_custom || !kctx.enabled || !rounds_strategy || !prompt_phase)
        return false;
    Family fam = Family::kBf16;
    if (!family_of(kctx.variant, fam) || !enabled(fam)) return false;
    if (facts(fam).kblock && grp_matmul_kblock_enabled(kctx.kblock_auto))
        return false;
    if (kctx.pack_nr <= 0) return false;
    int active = 0;
    for (size_t e = 0; e < M.size(); ++e) {
        if (M[e] <= 0) continue;
        if (N[e] <= 0 || N[e] % kctx.pack_nr != 0) return false;
        ++active;
    }
    return active > 1;
}

void execute(const custom_kernel::CallContext &kctx, const CallView &call,
        int num_threads, bool log_plan) {
    const int T = std::max(num_threads, 1);
    Scratch &s = scratch();
    Plan &plan = s.plan;
    const Planned p = plan_call(kctx, call, T, plan);
    const TeamMap &tm = *p.tm;
    const Tunables &k = p.r.t;
    const int num_ops = static_cast<int>(call.M.size());

    const int n_teams = tm.n_teams;
    const int ntiles = static_cast<int>(plan.tiles.size());
    // Slot `n_teams` is the whole list, used when the runtime hands the
    // region a different team size than the map was built for.
    Cursor *cur = s.cursors_for(n_teams + 1);
    for (int t = 0; t < n_teams; ++t) {
        cur[t].next.store(plan.team_begin[t], std::memory_order_relaxed);
        cur[t].end = plan.team_begin[t + 1];
    }
    cur[n_teams].next.store(0, std::memory_order_relaxed);
    cur[n_teams].end = ntiles;

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
                " est_imbalance=", plan.est_imbalance, " num_ops=", num_ops,
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
    const QuantRow *quant = call.quant;
    // Per-row source and per-channel weight scales only: static U8xS8, whose
    // epilogue requantizes and applies post-ops, stays on the round executor.
    const custom_kernel::Int8EpilogueConfig int8_epilogue {};

    auto run_tile = [&](const Tile &t) {
        const int e = t.e;
        const size_t m0 = static_cast<size_t>(t.m0);
        auto *d = static_cast<char *>(call.dst[e])
                + m0 * call.ldc[e] * dst_elem;
        if (quant != nullptr) {
            const QuantRow &q = quant[e];
            custom_kernel::dispatch_tile(kctx, e, t.rows, call.K[e], t.ncols,
                    t.n0, static_cast<const char *>(q.src) + m0 * q.lda, q.lda,
                    call.bias[e], d, call.ldc[e],
                    static_cast<const char *>(q.src_scale) + m0 * scale_elem,
                    q.src_zp != nullptr ? q.src_zp + m0 : nullptr, q.wei_scale,
                    int8_epilogue);
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
        if (omp_get_num_threads() == tm.num_threads) {
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
}

namespace test_api {
const Plan &plan_only(const custom_kernel::CallContext &kctx,
        const CallView &call, int num_threads) {
    Plan &plan = scratch().plan;
    plan_call(kctx, call, std::max(num_threads, 1), plan);
    return plan;
}
} // namespace test_api

const char *gemm_mode_label(bool fused_epilogue, grp_matmul_gated_act_t act) {
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
