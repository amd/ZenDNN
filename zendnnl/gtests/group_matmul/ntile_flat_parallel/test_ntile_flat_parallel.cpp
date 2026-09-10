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

/// @file test_ntile_flat_parallel.cpp
/// @brief Unit tests for the internal W8A8 grouped-MoE fast path.
///
/// The fast path is reached through the existing vector-based
/// `group_matmul_direct` overload when global
/// `ZENDNNL_GRP_MATMUL_ALGO=4`, or under global AUTO when the matching
/// decode/prompt phase setting requests 4. These tests build the argument set
/// a framework MoE call produces -- per-expert grouped sources, an
/// active-prefix weight vector, a `row_ptrs` post-op -- and drive the public
/// entry point through both request sources.
/// Coverage:
///
///   * byte-exact packed layout against an independent scalar packer
///   * every eligibility rule, each of which must decline with
///     `unimplemented` and leave the output untouched
///   * executor numerics against a scalar reference that rounds at the same
///     points the kernels do, across MoE geometries and every `block_m` tail
///   * direct caller-prequantized S8 input with ZenTorch-style same-backing
///     reuse, plus signed-A microkernel equivalence
///   * source-scale and destination/row-pointer ownership rejection matrices
///   * direct-S8 eligibility decline into main's generic S8 fallback
///   * per-expert row ordering / `row_ptrs` permutation semantics
///   * the weighted-reduce post-op, including `skip_weighted`
///   * packed-weight cache reuse and its release by
///     `clear_fused_moe_scratch()`

#include <iostream>
#include <limits>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ntile_flat_parallel_pack.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ukernel/ntile_flat_parallel_microkernel.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"
#include "lowoha_operators/matmul/group_matmul/ntile_flat_parallel/ntile_flat_parallel.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "moe_test_utils.hpp"

namespace {

using zendnnl::common::data_type_t;
using status_t = zendnnl::error_handling::status_t;
using zendnnl::lowoha::matmul::group_matmul_direct;
using zendnnl::lowoha::matmul::group_matmul_moe_postop_params;
using zendnnl::lowoha::matmul::grp_matmul_fused_moe_params;
using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
using zendnnl::lowoha::matmul::matmul_params;

namespace w8a8 = zendnnl::lowoha::matmul::ntile_flat_parallel;

// The fast path emits AVX-512 VNNI + BF16 intrinsics, so every test that can
// reach a kernel is gated at runtime.
#define SKIP_IF_NO_ISA() \
    do { \
        if (!w8a8::isa_supported()) { \
            GTEST_SKIP() << "host lacks AVX-512 VNNI/BF16"; \
        } \
    } while (0)

// ---------------------------------------------------------------------------
// bf16 helpers, written independently of the library's own so a divergence in
// either shows up here.
// ---------------------------------------------------------------------------
uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if (((u >> 23) & 0xff) == 0xff && (u & 0x7fffffu) != 0) {
        return static_cast<uint16_t>((u >> 16) | 0x40u); // quiet NaN
    }
    const uint32_t bias = 0x7fffu + ((u >> 16) & 1u); // round to nearest even
    return static_cast<uint16_t>((u + bias) >> 16);
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// Problem description + generated data.
//
// Mirrors what a framework MoE layer hands to `group_matmul_direct`: weights
// for every expert with the firing ones first, one grouped [M_e, H] source
// per firing expert, and a `row_ptrs` array addressing those same rows.
// ---------------------------------------------------------------------------
struct MoEProblem {
    MoEProblem() = default;
    MoEProblem(const MoEProblem &) = default;
    MoEProblem(MoEProblem &&) = default;
    MoEProblem &operator=(const MoEProblem &) = default;
    MoEProblem &operator=(MoEProblem &&) = default;

    // This fixture owns the weight storage used as the packed-cache identity.
    // Mirror the public host contract by ending the cache generation
    // in the destructor body, before vector members release their allocations.
    ~MoEProblem() { w8a8::flush_packed_weight_cache(); }

    int64_t num_experts = 0; // E
    int64_t hidden = 0; // H
    int64_t inter = 0; // I
    int64_t num_tokens = 0; // T
    int64_t topk = 0;

    // Weights in tensor-expert order: w13 [E, 2I, H], w2 [E, H, I].
    std::vector<int8_t> w13;
    std::vector<int8_t> w2;
    std::vector<uint16_t> w13_scale; // [E, 2I] bf16
    std::vector<uint16_t> w2_scale; // [E, H]  bf16

    std::vector<int32_t> topk_ids; // [T, topk]
    std::vector<float> topk_weights; // [T, topk]

    // Firing experts in first-encounter order, and the token that fills each
    // of their grouped rows.
    std::vector<int32_t> active_expert_ids;
    std::vector<std::vector<int32_t>> rows_of_active;
    // (active slot, row within that slot) for each flat (t, k) pair.
    std::vector<std::pair<int32_t, int32_t>> slot_of_pair;

    std::vector<std::vector<uint16_t>> grouped_src; // per active slot
    std::vector<uint16_t> tokens; // [T, H] bf16

    int64_t num_active() const {
        return static_cast<int64_t>(active_expert_ids.size());
    }
};

MoEProblem make_problem(int64_t E, int64_t H, int64_t I, int64_t T,
        int64_t topk, uint32_t seed, bool distinct_experts_per_token = true) {
    MoEProblem p;
    p.num_experts = E;
    p.hidden = H;
    p.inter = I;
    p.num_tokens = T;
    p.topk = topk;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.004f, 0.03f);
    std::uniform_real_distribution<float> ad(-1.5f, 1.5f);
    std::uniform_real_distribution<float> wd(0.05f, 1.0f);

    p.w13.resize(static_cast<size_t>(E) * 2 * I * H);
    p.w2.resize(static_cast<size_t>(E) * H * I);
    for (auto &v : p.w13) {
        v = static_cast<int8_t>(qd(rng));
    }
    for (auto &v : p.w2) {
        v = static_cast<int8_t>(qd(rng));
    }

    p.w13_scale.resize(static_cast<size_t>(E) * 2 * I);
    p.w2_scale.resize(static_cast<size_t>(E) * H);
    for (auto &v : p.w13_scale) {
        v = f32_to_bf16(sd(rng));
    }
    for (auto &v : p.w2_scale) {
        v = f32_to_bf16(sd(rng));
    }

    p.tokens.resize(static_cast<size_t>(T) * H);
    for (auto &v : p.tokens) {
        v = f32_to_bf16(ad(rng));
    }

    // Routing: `topk` distinct experts per token by default, so no token
    // contributes twice to the same expert.
    p.topk_ids.resize(static_cast<size_t>(T) * topk);
    p.topk_weights.resize(static_cast<size_t>(T) * topk);
    std::vector<int32_t> pool(E);
    std::iota(pool.begin(), pool.end(), 0);
    for (int64_t t = 0; t < T; ++t) {
        if (distinct_experts_per_token) {
            std::shuffle(pool.begin(), pool.end(), rng);
            for (int64_t k = 0; k < topk; ++k) {
                p.topk_ids[t * topk + k] = pool[static_cast<size_t>(k % E)];
            }
        } else {
            for (int64_t k = 0; k < topk; ++k) {
                p.topk_ids[t * topk + k] = static_cast<int32_t>(
                        rng() % static_cast<uint32_t>(E));
            }
        }
        for (int64_t k = 0; k < topk; ++k) {
            p.topk_weights[t * topk + k] = wd(rng);
        }
    }

    // Grouping, in the first-encounter order a framework produces.
    p.slot_of_pair.resize(static_cast<size_t>(T) * topk);
    for (int64_t i = 0; i < T * topk; ++i) {
        const int32_t e = p.topk_ids[i];
        const int32_t t = static_cast<int32_t>(i / topk);
        auto it = std::find(
                p.active_expert_ids.begin(), p.active_expert_ids.end(), e);
        int32_t a;
        if (it == p.active_expert_ids.end()) {
            a = static_cast<int32_t>(p.active_expert_ids.size());
            p.active_expert_ids.push_back(e);
            p.rows_of_active.emplace_back();
        } else {
            a = static_cast<int32_t>(it - p.active_expert_ids.begin());
        }
        const int32_t pos = static_cast<int32_t>(p.rows_of_active[a].size());
        p.rows_of_active[a].push_back(t);
        p.slot_of_pair[i] = {a, pos};
    }

    p.grouped_src.resize(p.rows_of_active.size());
    for (size_t a = 0; a < p.rows_of_active.size(); ++a) {
        const auto &rows = p.rows_of_active[a];
        p.grouped_src[a].resize(rows.size() * static_cast<size_t>(H));
        for (size_t r = 0; r < rows.size(); ++r) {
            std::memcpy(p.grouped_src[a].data() + r * H,
                    p.tokens.data() + static_cast<size_t>(rows[r]) * H,
                    static_cast<size_t>(H) * sizeof(uint16_t));
        }
    }
    return p;
}

// ---------------------------------------------------------------------------
// The full argument set for one `group_matmul_direct` call, owning every
// vector so a test can mutate one field and re-issue the call.
// ---------------------------------------------------------------------------
struct CallArgs {
    std::vector<char> layout;
    std::vector<bool> transA, transB, is_weights_const;
    std::vector<int> M, N, K, lda, ldb, ldc;
    std::vector<float> alpha, beta;
    std::vector<const void *> src, weight, bias;
    std::vector<void *> dst;
    std::vector<matmul_params> params;
    grp_matmul_gated_act_params gated_act;
    grp_matmul_fused_moe_params fused_moe;
    group_matmul_moe_postop_params postop;

    // Buffers the call reads or writes.
    std::vector<std::vector<uint16_t>> grouped; // mutated in place by Op2
    std::vector<std::vector<int8_t>>
            grouped_s8; // exact-size source/reference copy
    std::vector<std::vector<uint16_t>>
            grouped_dst; // S8 prefix followed by same-backing BF16 W2 dst
    std::vector<uint16_t> output; // [T, H] bf16
    std::vector<const void *> row_ptrs;
    std::vector<std::vector<uint16_t>> src_scale_scratch;
    std::vector<std::vector<float>> src_scale_f32;
    std::vector<std::vector<float>> weight_scale_f32;
    std::vector<std::vector<float>> down_scale_f32;

    status_t run_from_environment() {
        return group_matmul_direct(layout, transA, transB, M, N, K, alpha, src,
                lda, weight, ldb, bias, beta, dst, ldc, is_weights_const,
                params, &postop, &gated_act, &fused_moe);
    }

    status_t run(int algo = 4) {
        moe_test_utils::AlgoEnvGuard algo_guard(algo);
        return run_from_environment();
    }

    status_t run_fastpath_only() {
        return w8a8::try_execute(layout, transA, transB, M, N, K, alpha, src,
                lda, weight, ldb, bias, beta, dst, ldc, is_weights_const,
                params, &postop, &gated_act, &fused_moe);
    }
};

bool environment_requests_ntile_flat_parallel(const CallArgs &call) {
    const size_t active = call.params[0].active_matmul > 0
            ? static_cast<size_t>(call.params[0].active_matmul)
            : call.M.size();
    const auto phase = zendnnl::lowoha::matmul::classify_grp_matmul_phase(
            call.M, active);
    return zendnnl::lowoha::matmul::
                   resolve_grp_matmul_ntile_flat_parallel_request(phase)
            != zendnnl::lowoha::matmul::
                    grp_matmul_ntile_flat_parallel_request_source::none;
}

/// Build the call the way `FusedMoE.cpp` does: firing experts occupy the
/// leading slots of every weight-side vector, the inactive tail follows, and
/// `row_ptrs` addresses the grouped source rows (which Op2 overwrites).
std::unique_ptr<CallArgs> build_call(const MoEProblem &p, int num_threads = 0) {
    auto a = std::make_unique<CallArgs>();
    const int64_t E = p.num_experts;
    const int64_t Ea = p.num_active();
    const int64_t H = p.hidden;
    const int64_t I = p.inter;
    const int64_t two_i = 2 * I;

    a->grouped = p.grouped_src;
    a->output.assign(static_cast<size_t>(p.num_tokens) * H, 0);

    a->layout.assign(static_cast<size_t>(Ea), 'r');
    a->transA.assign(static_cast<size_t>(Ea), false);
    a->alpha.assign(static_cast<size_t>(Ea), 1.0f);
    a->beta.assign(static_cast<size_t>(Ea), 0.0f);
    a->bias.assign(static_cast<size_t>(Ea), nullptr);
    a->dst.assign(static_cast<size_t>(Ea), nullptr);
    a->M.resize(static_cast<size_t>(Ea));
    a->lda.assign(static_cast<size_t>(Ea), static_cast<int>(H));
    a->ldc.assign(static_cast<size_t>(Ea), static_cast<int>(two_i));
    a->src.resize(static_cast<size_t>(Ea));
    a->src_scale_scratch.resize(static_cast<size_t>(Ea));

    a->transB.assign(static_cast<size_t>(E), true);
    a->is_weights_const.assign(static_cast<size_t>(E), true);
    a->N.assign(static_cast<size_t>(E), static_cast<int>(two_i));
    a->K.assign(static_cast<size_t>(E), static_cast<int>(H));
    a->ldb.assign(static_cast<size_t>(E), static_cast<int>(H));
    a->weight.resize(static_cast<size_t>(E));
    a->params.assign(static_cast<size_t>(E), matmul_params {});

    a->fused_moe.down_weight.resize(static_cast<size_t>(E));
    a->fused_moe.N_down.assign(static_cast<size_t>(E), static_cast<int>(H));
    a->fused_moe.ldb_down.assign(static_cast<size_t>(E), static_cast<int>(I));
    a->fused_moe.bias_down.assign(static_cast<size_t>(Ea), nullptr);
    a->fused_moe.bias_dt_down = data_type_t::none;
    a->fused_moe.down_scale.resize(static_cast<size_t>(Ea));

    // Weight-side metadata for every expert: firing first, then the rest.
    std::vector<int8_t> is_active(static_cast<size_t>(E), 0);
    auto place = [&](int64_t slot, int64_t e) {
        a->weight[static_cast<size_t>(slot)]
                = p.w13.data() + static_cast<size_t>(e) * two_i * H;
        a->fused_moe.down_weight[static_cast<size_t>(slot)]
                = p.w2.data() + static_cast<size_t>(e) * H * I;
        a->params[static_cast<size_t>(slot)].dtypes.wei = data_type_t::s8;
    };
    for (int64_t s = 0; s < Ea; ++s) {
        const int64_t e = p.active_expert_ids[static_cast<size_t>(s)];
        is_active[static_cast<size_t>(e)] = 1;
        place(s, e);
    }
    for (int64_t e = 0, fill = Ea; e < E; ++e) {
        if (is_active[static_cast<size_t>(e)]) { continue; }
        place(fill++, e);
    }

    // Per-firing-expert input side.
    for (int64_t s = 0; s < Ea; ++s) {
        const int64_t e = p.active_expert_ids[static_cast<size_t>(s)];
        const int64_t Me = static_cast<int64_t>(p.rows_of_active[s].size());
        a->M[static_cast<size_t>(s)] = static_cast<int>(Me);
        a->src[static_cast<size_t>(s)] = a->grouped[s].data();

        auto &pr = a->params[static_cast<size_t>(s)];
        pr.dtypes.src = data_type_t::bf16;
        pr.dtypes.dst = data_type_t::bf16;
        pr.dtypes.wei = data_type_t::s8;
        pr.dtypes.bias = data_type_t::none;
        pr.dtypes.compute = data_type_t::s8;
        pr.dynamic_quant = true;
        pr.num_threads = num_threads;
        pr.plugin_op = "gtest::ntile_flat_parallel";
        pr.quant_params.wei_scale.buff
                = p.w13_scale.data() + static_cast<size_t>(e) * two_i;
        pr.quant_params.wei_scale.dt = data_type_t::bf16;
        pr.quant_params.wei_scale.dims = {1, two_i};

        a->src_scale_scratch[static_cast<size_t>(s)].assign(
                static_cast<size_t>(Me), 0);
        pr.quant_params.src_scale.buff
                = a->src_scale_scratch[static_cast<size_t>(s)].data();
        pr.quant_params.src_scale.dt = data_type_t::bf16;
        pr.quant_params.src_scale.dims = {Me, 1};

        auto &ds = a->fused_moe.down_scale[static_cast<size_t>(s)];
        ds.buff = p.w2_scale.data() + static_cast<size_t>(e) * H;
        ds.dt = data_type_t::bf16;
        ds.dims = {1, H};
    }
    a->params[0].active_matmul = static_cast<uint32_t>(Ea);
    a->params[0].total_matmul = static_cast<uint32_t>(E);

    a->gated_act.act = grp_matmul_gated_act_t::silu_and_mul;

    // Op2 writes back into the grouped buffers, so `row_ptrs` addresses them.
    a->row_ptrs.resize(static_cast<size_t>(p.num_tokens * p.topk));
    for (int64_t i = 0; i < p.num_tokens * p.topk; ++i) {
        const auto [slot, pos] = p.slot_of_pair[static_cast<size_t>(i)];
        a->row_ptrs[static_cast<size_t>(i)]
                = a->grouped[static_cast<size_t>(slot)].data() + pos * H;
    }
    a->postop.num_tokens = static_cast<int>(p.num_tokens);
    a->postop.topk = static_cast<int>(p.topk);
    a->postop.output = a->output.data();
    a->postop.ldc_output = static_cast<int>(H);
    a->postop.topk_weights = p.topk_weights.data();
    a->postop.skip_weighted = false;
    a->postop.row_ptrs = a->row_ptrs.data();
    return a;
}

// ---------------------------------------------------------------------------
// Scalar reference.
//
// Rounds to bf16 at exactly the points the kernels do (after silu x mul, and
// after the down projection), and accumulates the GEMMs in exact integer
// arithmetic. What remains between this and the kernels is the SiLU
// polynomial plus the rcp14 reciprocal, both well under 1e-3 relative.
// ---------------------------------------------------------------------------
void quantize_row_ref(
        const uint16_t *row, int64_t K, std::vector<int32_t> &q, float &scale) {
    q.resize(static_cast<size_t>(K));
    float amax = 0.f;
    for (int64_t k = 0; k < K; ++k) {
        amax = std::max(amax, std::fabs(bf16_to_f32(row[k])));
    }
    amax = std::max(amax, 1e-7f);
    scale = amax / 127.f;
    const float inv = 127.f / amax;
    for (int64_t k = 0; k < K; ++k) {
        const float v = bf16_to_f32(row[k]) * inv;
        q[static_cast<size_t>(k)] = static_cast<int32_t>(std::nearbyintf(v));
    }
}

/// Convert the existing broad BF16 fixture into ZenTorch's direct-S8 ALGO-4
/// contract: the tight S8 prefix and BF16 W2 destination share one BF16-sized
/// backing allocation per active expert.
std::unique_ptr<CallArgs> build_prequantized_call(const MoEProblem &p,
        data_type_t scale_dt = data_type_t::f32, int dst_ldc = 0) {
    auto a = build_call(p);
    const size_t active = static_cast<size_t>(p.num_active());
    const int64_t H = p.hidden;
    if (dst_ldc == 0) { dst_ldc = static_cast<int>(H); }
    a->grouped_s8.resize(active);
    a->grouped_dst.resize(active);
    a->src_scale_f32.resize(active);
    a->fused_moe.dst_down.resize(active);
    a->fused_moe.ldc_down.assign(active, dst_ldc);

    std::vector<int32_t> q;
    for (size_t slot = 0; slot < active; ++slot) {
        const int64_t rows = a->M[slot];
        a->grouped_s8[slot].resize(
                static_cast<size_t>(rows) * static_cast<size_t>(H));
        a->grouped_dst[slot].assign(
                static_cast<size_t>(rows) * static_cast<size_t>(dst_ldc), 0);
        a->src_scale_f32[slot].resize(static_cast<size_t>(rows));
        a->src_scale_scratch[slot].resize(static_cast<size_t>(rows));

        for (int64_t row = 0; row < rows; ++row) {
            float scale = 0.f;
            quantize_row_ref(p.grouped_src[slot].data() + row * H, H, q, scale);
            a->src_scale_f32[slot][static_cast<size_t>(row)] = scale;
            a->src_scale_scratch[slot][static_cast<size_t>(row)]
                    = f32_to_bf16(scale);
            for (int64_t k = 0; k < H; ++k) {
                a->grouped_s8[slot][static_cast<size_t>(row * H + k)]
                        = static_cast<int8_t>(q[static_cast<size_t>(k)]);
            }
        }

        if (dst_ldc == H) {
            std::memcpy(a->grouped_dst[slot].data(), a->grouped_s8[slot].data(),
                    a->grouped_s8[slot].size() * sizeof(int8_t));
            a->src[slot] = a->grouped_dst[slot].data();
        } else {
            // Construct an invalid padded/separate call for rejection tests.
            a->src[slot] = a->grouped_s8[slot].data();
        }
        a->fused_moe.dst_down[slot] = a->grouped_dst[slot].data();
        auto &param = a->params[slot];
        param.dtypes.src = data_type_t::s8;
        param.dynamic_quant = false;
        param.quant_params.src_scale.dt = scale_dt;
        param.quant_params.src_scale.buff = scale_dt == data_type_t::bf16
                ? static_cast<const void *>(a->src_scale_scratch[slot].data())
                : static_cast<const void *>(a->src_scale_f32[slot].data());
        param.quant_params.src_scale.dims = {rows, 1};
    }

    for (int64_t pair = 0; pair < p.num_tokens * p.topk; ++pair) {
        const auto [slot, row] = p.slot_of_pair[static_cast<size_t>(pair)];
        a->row_ptrs[static_cast<size_t>(pair)]
                = a->grouped_dst[static_cast<size_t>(slot)].data()
                + row * dst_ldc;
    }
    a->postop.row_ptrs = a->row_ptrs.data();
    return a;
}

/// Convert the fixture's W13/W2 scales to owned FP32 buffers so fallback tests
/// can cover the valid all-FP32 scale configuration without dropping the
/// existing all-BF16 coverage.
void promote_weight_scales_to_f32(CallArgs &a) {
    const size_t active = a.M.size();
    a.weight_scale_f32.resize(active);
    a.down_scale_f32.resize(active);
    for (size_t i = 0; i < active; ++i) {
        const auto *w13_bf16 = static_cast<const uint16_t *>(
                a.params[i].quant_params.wei_scale.buff);
        a.weight_scale_f32[i].resize(static_cast<size_t>(a.N[i]));
        for (size_t n = 0; n < a.weight_scale_f32[i].size(); ++n) {
            a.weight_scale_f32[i][n] = bf16_to_f32(w13_bf16[n]);
        }
        a.params[i].quant_params.wei_scale.buff = a.weight_scale_f32[i].data();
        a.params[i].quant_params.wei_scale.dt = data_type_t::f32;

        const auto *w2_bf16
                = static_cast<const uint16_t *>(a.fused_moe.down_scale[i].buff);
        a.down_scale_f32[i].resize(static_cast<size_t>(a.fused_moe.N_down[i]));
        for (size_t n = 0; n < a.down_scale_f32[i].size(); ++n) {
            a.down_scale_f32[i][n] = bf16_to_f32(w2_bf16[n]);
        }
        a.fused_moe.down_scale[i].buff = a.down_scale_f32[i].data();
        a.fused_moe.down_scale[i].dt = data_type_t::f32;
    }
}

std::vector<uint16_t> reference_moe(
        const MoEProblem &p, const CallArgs *prequantized = nullptr) {
    const int64_t H = p.hidden;
    const int64_t I = p.inter;
    const int64_t two_i = 2 * I;
    std::vector<uint16_t> out(static_cast<size_t>(p.num_tokens) * H, 0);
    // [T*topk, H] bf16 per-slot down-projection results.
    std::vector<uint16_t> per_slot(
            static_cast<size_t>(p.num_tokens * p.topk) * H, 0);

    std::vector<int32_t> aq;
    std::vector<int32_t> aq2;
    for (int64_t i = 0; i < p.num_tokens * p.topk; ++i) {
        const int64_t t = i / p.topk;
        const int64_t e = p.topk_ids[static_cast<size_t>(i)];

        float as = 0.f;
        if (prequantized == nullptr) {
            quantize_row_ref(p.tokens.data() + t * H, H, aq, as);
            // build_call declares BF16 dynamic source scales. Match the
            // configured scale tensor precision before the GEMM consumes it.
            as = bf16_to_f32(f32_to_bf16(as));
        } else {
            const auto [slot, row] = p.slot_of_pair[static_cast<size_t>(i)];
            const auto &src
                    = prequantized->grouped_s8[static_cast<size_t>(slot)];
            aq.resize(static_cast<size_t>(H));
            for (int64_t k = 0; k < H; ++k) {
                aq[static_cast<size_t>(k)] = static_cast<int32_t>(
                        src[static_cast<size_t>(row * H + k)]);
            }
            const auto &scale = prequantized->params[static_cast<size_t>(slot)]
                                        .quant_params.src_scale;
            as = scale.dt == data_type_t::bf16
                    ? bf16_to_f32(
                              static_cast<const uint16_t *>(scale.buff)[row])
                    : static_cast<const float *>(scale.buff)[row];
        }

        // gate/up + silu x mul -> bf16 intermediate
        std::vector<uint16_t> mid(static_cast<size_t>(I));
        const int8_t *w13e = p.w13.data() + static_cast<size_t>(e) * two_i * H;
        const uint16_t *s13e
                = p.w13_scale.data() + static_cast<size_t>(e) * two_i;
        for (int64_t n = 0; n < I; ++n) {
            int64_t acc_g = 0;
            int64_t acc_u = 0;
            for (int64_t k = 0; k < H; ++k) {
                acc_g += static_cast<int64_t>(aq[static_cast<size_t>(k)])
                        * w13e[n * H + k];
                acc_u += static_cast<int64_t>(aq[static_cast<size_t>(k)])
                        * w13e[(n + I) * H + k];
            }
            const float x
                    = static_cast<float>(acc_g) * as * bf16_to_f32(s13e[n]);
            const float y
                    = static_cast<float>(acc_u) * as * bf16_to_f32(s13e[n + I]);
            const float silu = x / (1.f + std::exp(-x));
            mid[static_cast<size_t>(n)] = f32_to_bf16(silu * y);
        }

        // requantize, then down projection -> bf16
        float as2 = 0.f;
        quantize_row_ref(mid.data(), I, aq2, as2);
        const int8_t *w2e = p.w2.data() + static_cast<size_t>(e) * H * I;
        const uint16_t *s2e = p.w2_scale.data() + static_cast<size_t>(e) * H;
        for (int64_t n = 0; n < H; ++n) {
            int64_t acc = 0;
            for (int64_t k = 0; k < I; ++k) {
                acc += static_cast<int64_t>(aq2[static_cast<size_t>(k)])
                        * w2e[n * I + k];
            }
            per_slot[static_cast<size_t>(i) * H + n] = f32_to_bf16(
                    static_cast<float>(acc) * as2 * bf16_to_f32(s2e[n]));
        }
    }

    // weighted reduce over topk, accumulated in f32
    for (int64_t t = 0; t < p.num_tokens; ++t) {
        for (int64_t n = 0; n < H; ++n) {
            float sum = 0.f;
            for (int64_t k = 0; k < p.topk; ++k) {
                const int64_t i = t * p.topk + k;
                sum += p.topk_weights[static_cast<size_t>(i)]
                        * bf16_to_f32(per_slot[static_cast<size_t>(i) * H + n]);
            }
            out[static_cast<size_t>(t) * H + n] = f32_to_bf16(sum);
        }
    }
    return out;
}

void expect_close(const std::vector<uint16_t> &got,
        const std::vector<uint16_t> &want, float rtol, const char *what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    double max_rel = 0.0;
    float scale = 0.f;
    for (size_t i = 0; i < want.size(); ++i) {
        scale = std::max(scale, std::fabs(bf16_to_f32(want[i])));
    }
    ASSERT_GT(scale, 0.f) << what << ": reference is all zero";
    for (size_t i = 0; i < want.size(); ++i) {
        const float a = bf16_to_f32(got[i]);
        const float b = bf16_to_f32(want[i]);
        max_rel = std::max<double>(max_rel, std::fabs(a - b) / scale);
    }
    EXPECT_LT(max_rel, rtol) << what << ": max relative deviation " << max_rel;
}

} // namespace

// ---------------------------------------------------------------------------
// Packed layout
// ---------------------------------------------------------------------------

// The packed form must be byte-identical to an independently written scalar
// packer: that layout is the contract shared with the reference int8 MoE
// kernels, so a silent drift would be a wrong-results bug, not a slow one.
TEST(W8A8MoEPack, ByteExactAgainstScalarPacker) {
    SKIP_IF_NO_ISA();
    constexpr int64_t E = 3;
    constexpr int64_t OC = 64; // two block_n blocks
    constexpr int64_t IC = 128;

    std::vector<int8_t> src(static_cast<size_t>(E) * OC * IC);
    std::mt19937 rng(7);
    for (auto &v : src) {
        v = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
    }

    const int64_t oc_stride = w8a8::packed_bytes_per_oc(IC);
    std::vector<int8_t> got(
            static_cast<size_t>(E) * OC * static_cast<size_t>(oc_stride), 0);
    ASSERT_EQ(w8a8::pack_weights(src.data(), got.data(), E, OC, IC, 1),
            status_t::success);

    // Independent scalar packer: per 32-output-channel block, K/4 groups of
    // 32 channels x 4 k-values, then the 32 int32 compensation values.
    std::vector<int8_t> want(got.size(), 0);
    const int64_t blocks = OC / w8a8::block_n;
    for (int64_t e = 0; e < E; ++e) {
        for (int64_t b = 0; b < blocks; ++b) {
            int8_t *dst
                    = want.data() + (e * OC + b * w8a8::block_n) * oc_stride;
            const int8_t *s = src.data() + (e * OC + b * w8a8::block_n) * IC;
            for (int64_t k4 = 0; k4 < IC / 4; ++k4) {
                for (int64_t n = 0; n < w8a8::block_n; ++n) {
                    for (int64_t j = 0; j < 4; ++j) {
                        dst[(k4 * w8a8::block_n + n) * 4 + j]
                                = s[n * IC + k4 * 4 + j];
                    }
                }
            }
            auto *comp = reinterpret_cast<int32_t *>(dst + w8a8::block_n * IC);
            for (int64_t n = 0; n < w8a8::block_n; ++n) {
                int32_t sum = 0;
                for (int64_t k = 0; k < IC; ++k) {
                    sum += s[n * IC + k];
                }
                comp[n] = 128 * sum;
            }
        }
    }
    EXPECT_EQ(std::memcmp(got.data(), want.data(), got.size()), 0);
}

TEST(W8A8MoEPack, RejectsUnalignedGeometry) {
    std::vector<int8_t> src(32 * 32);
    std::vector<int8_t> dst(32 * 64);
    // out_channels must be a multiple of block_n.
    EXPECT_EQ(w8a8::pack_weights(src.data(), dst.data(), 1, 31, 32, 1),
            status_t::memory_bad_size);
    // in_channels must be a multiple of the VNNI step.
    EXPECT_EQ(w8a8::pack_weights(src.data(), dst.data(), 1, 32, 33, 1),
            status_t::memory_bad_size);
    EXPECT_EQ(w8a8::pack_weights(src.data(), dst.data(),
                      std::numeric_limits<int64_t>::max(), 32, 32, 1),
            status_t::memory_bad_size);
    EXPECT_EQ(w8a8::pack_weights(nullptr, dst.data(), 1, 32, 32, 1),
            status_t::op_bad_io);
}

TEST(W8A8MoEMicrokernel, SignedS8MatchesBiasedU8GateUp) {
    SKIP_IF_NO_ISA();
    constexpr int64_t N = w8a8::block_n;
    constexpr int64_t K = 32;
    const int8_t corners[] = {-128, -127, -1, 0, 1, 126, 127};
    constexpr size_t num_corners = sizeof(corners) / sizeof(corners[0]);

    std::vector<int8_t> w_gate(static_cast<size_t>(N * K));
    std::vector<int8_t> w_up(static_cast<size_t>(N * K));
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            w_gate[static_cast<size_t>(n * K + k)]
                    = corners[static_cast<size_t>(n + k) % num_corners];
            w_up[static_cast<size_t>(n * K + k)]
                    = corners[static_cast<size_t>(2 * n + 3 * k + 1)
                            % num_corners];
        }
    }

    const int64_t packed_size = N * w8a8::packed_bytes_per_oc(K);
    std::vector<int8_t> packed_gate(static_cast<size_t>(packed_size));
    std::vector<int8_t> packed_up(static_cast<size_t>(packed_size));
    ASSERT_EQ(w8a8::pack_weights(w_gate.data(), packed_gate.data(), 1, N, K, 1),
            status_t::success);
    ASSERT_EQ(w8a8::pack_weights(w_up.data(), packed_up.data(), 1, N, K, 1),
            status_t::success);
    const auto *comp_gate = reinterpret_cast<const int32_t *>(
            packed_gate.data() + static_cast<size_t>(N * K));
    const auto *comp_up = reinterpret_cast<const int32_t *>(
            packed_up.data() + static_cast<size_t>(N * K));

    std::vector<float> bs_gate(static_cast<size_t>(N));
    std::vector<float> bs_up(static_cast<size_t>(N));
    for (int64_t n = 0; n < N; ++n) {
        bs_gate[static_cast<size_t>(n)]
                = 0.002f * static_cast<float>(1 + n % 5);
        bs_up[static_cast<size_t>(n)] = 0.003f * static_cast<float>(1 + n % 7);
    }

    for (int64_t rows = 1; rows <= 4; ++rows) {
        std::vector<int8_t> signed_storage(static_cast<size_t>(rows * K) + 1);
        int8_t *const signed_a = signed_storage.data() + 1;
        ASSERT_NE(reinterpret_cast<uintptr_t>(signed_a) % alignof(int32_t), 0u);
        std::vector<uint8_t> biased_a(static_cast<size_t>(rows * K));
        std::vector<float> as(static_cast<size_t>(rows));
        for (int64_t m = 0; m < rows; ++m) {
            as[static_cast<size_t>(m)] = 0.01f * static_cast<float>(m + 1);
            for (int64_t k = 0; k < K; ++k) {
                const int8_t value
                        = corners[static_cast<size_t>(m * K + k) % num_corners];
                signed_a[static_cast<size_t>(m * K + k)] = value;
                biased_a[static_cast<size_t>(m * K + k)] = static_cast<uint8_t>(
                        static_cast<int32_t>(value) + 128);
            }
        }

        std::vector<uint16_t> from_signed(static_cast<size_t>(rows * N), 0);
        std::vector<uint16_t> from_biased(static_cast<size_t>(rows * N), 0);
        w8a8::tinygemm_gate_up<true>(
                reinterpret_cast<const uint8_t *>(signed_a), packed_gate.data(),
                packed_up.data(), from_signed.data(), as.data(), bs_gate.data(),
                bs_up.data(), comp_gate, comp_up, rows, K, K, N, N);
        w8a8::tinygemm_gate_up<false>(biased_a.data(), packed_gate.data(),
                packed_up.data(), from_biased.data(), as.data(), bs_gate.data(),
                bs_up.data(), comp_gate, comp_up, rows, K, K, N, N);
        EXPECT_EQ(from_signed, from_biased) << "rows=" << rows;
    }
}

// ---------------------------------------------------------------------------
// Numerics
// ---------------------------------------------------------------------------

// Relative tolerance, measured against the largest reference magnitude.
//
// The reference rounds to bf16 exactly where the kernels do, so the residual
// is the SiLU polynomial plus the rcp14 reciprocal the epilogue uses instead
// of a divide. Those differ from an exact sigmoid by well under a bf16 ULP,
// but that is enough to flip the rounding of an occasional intermediate
// element by one ULP (2^-8 relative), which then propagates through the down
// projection. Observed worst case across these cases is ~3e-3, and it grows
// slowly with element count as more elements sit near a rounding boundary.
//
// 1e-2 sits comfortably above that floor while staying two orders of
// magnitude below a structural error: when the packed-weight cache was
// returning a stale entry, this same check reported deviations of 1.2-2.0.
constexpr float kRtol = 1e-2f;

TEST(W8A8MoEExecute, MatchesScalarReference) {
    SKIP_IF_NO_ISA();
    zendnnl::lowoha::matmul::clear_fused_moe_scratch();
    const auto p = make_problem(/*E=*/8, /*H=*/128, /*I=*/64, /*T=*/6,
            /*topk=*/3, /*seed=*/11);
    auto call = build_call(p);
    if (!environment_requests_ntile_flat_parallel(*call)) {
        GTEST_SKIP() << "set global ALGO=4 or the matching AUTO phase ALGO=4";
    }
    ASSERT_EQ(call->run_from_environment(), status_t::success);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u);
    expect_close(call->output, reference_moe(p), kRtol, "decode-shaped MoE");
}

TEST(W8A8MoEExecute, PrequantizedS8SeparateCallerDstDeclinesBeforeWrites) {
    SKIP_IF_NO_ISA();
    zendnnl::lowoha::matmul::clear_fused_moe_scratch();
    const auto p = make_problem(/*E=*/8, /*H=*/128, /*I=*/64, /*T=*/6,
            /*topk=*/3, /*seed=*/13);
    auto call = build_prequantized_call(p, data_type_t::f32);
    for (size_t slot = 0; slot < call->src.size(); ++slot) {
        call->src[slot] = call->grouped_s8[slot].data();
    }
    const auto source_before = call->grouped_s8;
    const auto destination_before = call->grouped_dst;
    const auto output_before = call->output;

    EXPECT_EQ(call->run_fastpath_only(), status_t::unimplemented);
    EXPECT_EQ(call->grouped_s8, source_before)
            << "declined separate S8 source was modified";
    EXPECT_EQ(call->grouped_dst, destination_before)
            << "declined separate BF16 destination was modified";
    EXPECT_EQ(call->output, output_before)
            << "declined separate-buffer call wrote reduce output";
}

TEST(W8A8MoEExecute, PrequantizedS8SameBackingMatchesScalarReference) {
    SKIP_IF_NO_ISA();
    zendnnl::lowoha::matmul::clear_fused_moe_scratch();
    const auto p = make_problem(/*E=*/8, /*H=*/128, /*I=*/64, /*T=*/6,
            /*topk=*/3, /*seed=*/14);
    auto call = build_prequantized_call(p, data_type_t::f32);
    const auto reference = reference_moe(p, call.get());

    for (size_t slot = 0; slot < call->src.size(); ++slot) {
        ASSERT_EQ(call->src[slot], call->fused_moe.dst_down[slot])
                << "slot=" << slot;
        ASSERT_EQ(call->fused_moe.ldc_down[slot], p.hidden)
                << "same-backing reuse must be tight";
    }

    ASSERT_EQ(call->run(), status_t::success);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u);
    expect_close(call->output, reference, kRtol,
            "prequantized-S8 same-backing weighted MoE");
}

TEST(W8A8MoEExecute, PublishesConfiguredDynamicSourceScales) {
    SKIP_IF_NO_ISA();
    const auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/5,
            /*topk=*/2, /*seed=*/15);

    for (const data_type_t scale_dt : {data_type_t::bf16, data_type_t::f32}) {
        auto call = build_call(p);
        if (scale_dt == data_type_t::f32) {
            call->src_scale_f32.resize(static_cast<size_t>(p.num_active()));
            for (size_t slot = 0; slot < call->M.size(); ++slot) {
                call->src_scale_f32[slot].assign(
                        static_cast<size_t>(call->M[slot]), 0.f);
                call->params[slot].quant_params.src_scale.buff
                        = call->src_scale_f32[slot].data();
                call->params[slot].quant_params.src_scale.dt = data_type_t::f32;
            }
        }

        ASSERT_EQ(call->run_fastpath_only(), status_t::success)
                << "scale_dt=" << static_cast<int>(scale_dt);

        std::vector<int32_t> q;
        for (size_t slot = 0; slot < call->M.size(); ++slot) {
            for (int64_t row = 0; row < call->M[slot]; ++row) {
                float expected = 0.f;
                quantize_row_ref(p.grouped_src[slot].data() + row * p.hidden,
                        p.hidden, q, expected);
                if (scale_dt == data_type_t::bf16) {
                    EXPECT_EQ(call->src_scale_scratch[slot][row],
                            f32_to_bf16(expected));
                } else {
                    EXPECT_FLOAT_EQ(call->src_scale_f32[slot][row], expected);
                }
            }
        }
    }
}

TEST(W8A8MoEExecute, PrequantizedS8PaddedCallerDstDeclinesBeforeWrites) {
    SKIP_IF_NO_ISA();
    const auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/5,
            /*topk=*/2, /*seed=*/17);
    constexpr int padded_ldc = 71;
    constexpr uint16_t sentinel = 0x5A5A;
    auto call = build_prequantized_call(p, data_type_t::f32, padded_ldc);
    for (auto &dst : call->grouped_dst) {
        std::fill(dst.begin(), dst.end(), sentinel);
    }
    const auto source_before = call->grouped_s8;
    const auto destination_before = call->grouped_dst;
    const auto output_before = call->output;

    EXPECT_EQ(call->run_fastpath_only(), status_t::unimplemented);
    EXPECT_EQ(call->grouped_s8, source_before);
    EXPECT_EQ(call->grouped_dst, destination_before);
    EXPECT_EQ(call->output, output_before);
}

// The number of rows a firing expert receives is set by routing, so every
// remainder against the kernel's 1..4 row instantiations and against block_m
// has to be exercised. topk=1 with T tokens all on one expert gives direct
// control of that row count.
TEST(W8A8MoEExecute, RowCountTails) {
    SKIP_IF_NO_ISA();
    for (int64_t T : {2, 3, 4, 5, 7, 8, 31, 32, 33, 47, 64, 65}) {
        MoEProblem p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, T,
                /*topk=*/2, /*seed=*/static_cast<uint32_t>(100 + T));
        auto call = build_call(p);
        ASSERT_EQ(call->run(), status_t::success) << "T=" << T;
        expect_close(call->output, reference_moe(p), kRtol,
                ("row tail T=" + std::to_string(T)).c_str());
    }
}

TEST(W8A8MoEExecute, GeometryVariations) {
    SKIP_IF_NO_ISA();
    struct Case {
        int64_t E, H, I, T, topk;
    };
    // Unrelated MoE geometries: the path is expressed only in terms of
    // hidden / intermediate size, so none of these should be special.
    const Case cases[] = {
            {2, 32, 32, 1, 1},
            {4, 96, 32, 4, 2},
            {6, 160, 96, 5, 3},
            {16, 64, 128, 8, 4},
            {8, 256, 32, 3, 8},
    };
    for (const auto &c : cases) {
        MoEProblem p = make_problem(c.E, c.H, c.I, c.T, c.topk,
                /*seed=*/static_cast<uint32_t>(c.H * 31 + c.I));
        auto call = build_call(p);
        ASSERT_EQ(call->run(), status_t::success)
                << "E" << c.E << " H" << c.H << " I" << c.I;
        expect_close(call->output, reference_moe(p), kRtol, "geometry");
    }
}

// Two slots of one token may route to the same expert, which puts two rows
// with identical content but distinct router weights in the same grouped
// buffer. The per-expert row ordering and the reduce must keep them apart.
TEST(W8A8MoEExecute, RepeatedExpertPerToken) {
    SKIP_IF_NO_ISA();
    MoEProblem p = make_problem(/*E=*/3, /*H=*/64, /*I=*/32, /*T=*/5,
            /*topk=*/4, /*seed=*/29, /*distinct_experts_per_token=*/false);
    auto call = build_call(p);
    ASSERT_EQ(call->run(), status_t::success);
    expect_close(call->output, reference_moe(p), kRtol, "repeated expert");
}

// The fast path writes Op2 output into the grouped source buffers and lets
// the post-op gather through row_ptrs. Reversing the order in which experts
// are presented changes which buffer each token's row lands in, and must not
// change the answer.
TEST(W8A8MoEExecute, PerExpertOrderingIsRespected) {
    SKIP_IF_NO_ISA();
    const auto p = make_problem(/*E=*/6, /*H=*/64, /*I=*/32, /*T=*/6,
            /*topk=*/3, /*seed=*/41);
    auto call = build_call(p);
    ASSERT_EQ(call->run(), status_t::success);
    const std::vector<uint16_t> baseline = call->output;

    // Same problem, firing experts presented in reverse order.
    MoEProblem q = p;
    std::reverse(q.active_expert_ids.begin(), q.active_expert_ids.end());
    std::reverse(q.rows_of_active.begin(), q.rows_of_active.end());
    const int32_t last = static_cast<int32_t>(q.active_expert_ids.size()) - 1;
    for (auto &sp : q.slot_of_pair) {
        sp.first = last - sp.first;
    }
    q.grouped_src.assign(q.rows_of_active.size(), {});
    for (size_t a = 0; a < q.rows_of_active.size(); ++a) {
        const auto &rows = q.rows_of_active[a];
        q.grouped_src[a].resize(rows.size() * static_cast<size_t>(q.hidden));
        for (size_t r = 0; r < rows.size(); ++r) {
            std::memcpy(q.grouped_src[a].data() + r * q.hidden,
                    q.tokens.data() + static_cast<size_t>(rows[r]) * q.hidden,
                    static_cast<size_t>(q.hidden) * sizeof(uint16_t));
        }
    }
    auto reordered = build_call(q);
    ASSERT_EQ(reordered->run(), status_t::success);
    EXPECT_EQ(reordered->output, baseline);
}

// skip_weighted makes the post-op a plain gather-sum; the router weights are
// then the caller's business, so the reference drops them too.
TEST(W8A8MoEExecute, SkipWeightedReduce) {
    SKIP_IF_NO_ISA();
    MoEProblem p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/4,
            /*topk=*/2, /*seed=*/53);
    auto call = build_call(p);
    call->postop.skip_weighted = true;
    call->postop.topk_weights = nullptr;
    ASSERT_EQ(call->run(), status_t::success);

    MoEProblem unweighted = p;
    std::fill(unweighted.topk_weights.begin(), unweighted.topk_weights.end(),
            1.0f);
    expect_close(
            call->output, reference_moe(unweighted), kRtol, "skip_weighted");
}

TEST(W8A8MoEExecute, ThreadCountDoesNotChangeResult) {
    SKIP_IF_NO_ISA();
    const auto p = make_problem(/*E=*/8, /*H=*/128, /*I=*/64, /*T=*/8,
            /*topk=*/4, /*seed=*/67);
    auto single = build_call(p, /*num_threads=*/1);
    ASSERT_EQ(single->run(), status_t::success);
    for (int nt : {2, 4, 8}) {
        auto multi = build_call(p, nt);
        ASSERT_EQ(multi->run(), status_t::success) << "threads=" << nt;
        EXPECT_EQ(multi->output, single->output) << "threads=" << nt;
    }
}

TEST(W8A8MoEDispatch, RequiresAlgo4OptIn) {
    SKIP_IF_NO_ISA();
    zendnnl::lowoha::matmul::clear_fused_moe_scratch();
    ASSERT_EQ(w8a8::packed_weight_cache_size(), 0u);

    {
        moe_test_utils::AlgoEnvGuard automatic(/*algo=*/0);
        EXPECT_FALSE(
                zendnnl::lowoha::matmul ::get_grp_matmul_ntile_flat_parallel());
        EXPECT_EQ(zendnnl::lowoha::matmul::get_grp_matmul_algo(), 0);
    }
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 0u);

    {
        moe_test_utils::AlgoEnvGuard explicit_opt_in(/*algo=*/4);
        EXPECT_TRUE(
                zendnnl::lowoha::matmul ::get_grp_matmul_ntile_flat_parallel());
        // A declined ALGO-4 call must enter the generic AUTO planner.
        EXPECT_EQ(zendnnl::lowoha::matmul::get_grp_matmul_algo(), 0);
    }

    const auto p = make_problem(/*E=*/8, /*H=*/128, /*I=*/64, /*T=*/8,
            /*topk=*/4, /*seed=*/71);
    auto explicit_fastpath = build_call(p);
    moe_test_utils::GemmModeCaptureGuard capture;
    ASSERT_EQ(explicit_fastpath->run(/*algo=*/4), status_t::success);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u);
    const char *mode = test_api::s_last_group_matmul_direct_gemm_mode.load(
            std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr);
    EXPECT_STREQ(mode, "ntile_flat_parallel");
    EXPECT_EQ(zendnnl::lowoha::matmul::executed_algo_from_gemm_mode(mode), 4);
    expect_close(explicit_fastpath->output, reference_moe(p), kRtol, "algo 4");

    zendnnl::lowoha::matmul::clear_fused_moe_scratch();
}

TEST(W8A8MoEDispatch, LateDeclinePreservesIncomingThreadCap) {
    SKIP_IF_NO_ISA();
    const int32_t cached_baseline
            = zendnnl::lowoha::thread_guard::max_threads();
    const int32_t current = omp_get_max_threads();
    ASSERT_EQ(current, cached_baseline);
    if (current <= 1) { GTEST_SKIP() << "requires a reducible thread cap"; }
    const int32_t cap = std::max<int32_t>(1, current / 2);

    const auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/4,
            /*topk=*/2, /*seed=*/72);
    auto call = build_call(p, cap);
    call->row_ptrs[0] = call->row_ptrs[1];

    zendnnl::lowoha::thread_guard outer_cap(cap, current);
    ASSERT_EQ(omp_get_max_threads(), cap);
    EXPECT_EQ(call->run_fastpath_only(), status_t::unimplemented);
    EXPECT_EQ(omp_get_max_threads(), cap)
            << "a late fast-path decline must not undo the caller cap";
}

TEST(W8A8MoEDispatch, AutoDecodePhase4ExecutesEligibleCall) {
    SKIP_IF_NO_ISA();
    using namespace zendnnl::lowoha::matmul;
    clear_fused_moe_scratch();
    ASSERT_EQ(w8a8::packed_weight_cache_size(), 0u);

    const auto p = make_problem(/*E=*/8, /*H=*/128, /*I=*/64, /*T=*/8,
            /*topk=*/4, /*seed=*/73);
    auto call = build_call(p);
    moe_test_utils::AutoPromptAlgoOverride inactive_prompt(6);
    moe_test_utils::AutoDecodeAlgoOverride decode_w8a8(4);
    ASSERT_EQ(call->run(/*algo=*/0), status_t::success);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u);
    expect_close(call->output, reference_moe(p), kRtol, "AUTO decode phase 4");

    clear_fused_moe_scratch();
}

TEST(W8A8MoEDispatch, AutoPromptPhase4ExecutesEligibleCall) {
    SKIP_IF_NO_ISA();
    using namespace zendnnl::lowoha::matmul;
    clear_fused_moe_scratch();
    ASSERT_EQ(w8a8::packed_weight_cache_size(), 0u);

    // One expert with M=40 is prompt-class (strictly above 32).
    const auto p = make_problem(/*E=*/1, /*H=*/64, /*I=*/32, /*T=*/40,
            /*topk=*/1, /*seed=*/79);
    auto call = build_call(p, /*num_threads=*/2);
    moe_test_utils::AutoPromptAlgoOverride prompt_w8a8(4);
    moe_test_utils::AutoDecodeAlgoOverride inactive_decode(6);
    ASSERT_EQ(call->run(/*algo=*/0), status_t::success);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u);
    expect_close(call->output, reference_moe(p), kRtol, "AUTO prompt phase 4");

    clear_fused_moe_scratch();
}

TEST(W8A8MoEDispatch, AutoDecodePhase4ClassifiesActivePrefixBeforeFallback) {
    SKIP_IF_NO_ISA();
    using namespace zendnnl::lowoha::matmul;
    clear_fused_moe_scratch();
    ASSERT_EQ(w8a8::packed_weight_cache_size(), 0u);

    const auto p = make_problem(/*E=*/8, /*H=*/64, /*I=*/32, /*T=*/4,
            /*topk=*/1, /*seed=*/81);
    auto generic_baseline = build_call(p);
    ASSERT_LT(generic_baseline->M.size(), static_cast<size_t>(p.num_experts));
    // Framework prepack extras may extend M beyond active_matmul. These rows
    // are not part of this call; a large cold-expert M must not select prompt.
    generic_baseline->M.resize(static_cast<size_t>(p.num_experts), 4096);
    ASSERT_EQ(generic_baseline->run(/*algo=*/1), status_t::success);

    auto call = build_call(p);
    call->M.resize(static_cast<size_t>(p.num_experts), 4096);
    moe_test_utils::AutoPromptAlgoOverride prompt_multilevel(6);
    moe_test_utils::AutoDecodeAlgoOverride decode_w8a8(4);
    moe_test_utils::GemmModeCaptureGuard capture;
    ASSERT_EQ(call->run(/*algo=*/0), status_t::success);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 0u)
            << "the existing hook declines mismatched M/src vector lengths";
    const char *mode = test_api::s_last_group_matmul_direct_gemm_mode.load(
            std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr);
    // The inherited decode default is refined by the existing few-expert
    // wide-N rule to ALGO 1. Prompt ALGO 6 would prove tail leakage.
    EXPECT_EQ(executed_algo_from_gemm_mode(mode), 1) << "mode=" << mode;
    EXPECT_EQ(call->output, generic_baseline->output)
            << "the active decode prefix must inherit its refined default "
               "policy, not prompt ALGO 6";

    clear_fused_moe_scratch();
}

TEST(W8A8MoEDispatch, GlobalGenericPinSuppressesPhase4) {
    SKIP_IF_NO_ISA();
    using namespace zendnnl::lowoha::matmul;
    clear_fused_moe_scratch();
    ASSERT_EQ(w8a8::packed_weight_cache_size(), 0u);

    const auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/4,
            /*topk=*/2, /*seed=*/83);
    auto generic_baseline = build_call(p);
    ASSERT_EQ(generic_baseline->run(/*algo=*/1), status_t::success);

    auto call = build_call(p);
    moe_test_utils::AutoDecodeAlgoOverride decode_w8a8(4);
    moe_test_utils::GemmModeCaptureGuard capture;
    ASSERT_EQ(call->run(/*algo=*/1), status_t::success);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 0u)
            << "global generic ALGO 1 must suppress the phase W8A8 hook";
    const char *mode = test_api::s_last_group_matmul_direct_gemm_mode.load(
            std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr);
    EXPECT_EQ(executed_algo_from_gemm_mode(mode), 1) << "mode=" << mode;
    EXPECT_EQ(call->output, generic_baseline->output)
            << "phase 4 must not perturb a global generic ALGO 1 call";

    clear_fused_moe_scratch();
}

// ---------------------------------------------------------------------------
// Eligibility: every rule must decline, and declining must not write
// ---------------------------------------------------------------------------

class W8A8MoEEligibility : public ::testing::Test {
protected:
    void SetUp() override {
        problem_ = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/4,
                /*topk=*/2, /*seed=*/97);
    }

    /// Apply `mutate`, then assert the fast path declines without writing.
    template <typename F>
    void expect_declined(F &&mutate, const char *what) {
        auto call = build_call(problem_);
        const std::vector<uint16_t> grouped_before = call->grouped[0];
        mutate(*call);
        EXPECT_EQ(call->run_fastpath_only(), status_t::unimplemented) << what;
        EXPECT_EQ(call->output, std::vector<uint16_t>(call->output.size(), 0))
                << what << ": output was written";
        EXPECT_EQ(call->grouped[0], grouped_before)
                << what << ": source was overwritten";
    }

    MoEProblem problem_;
};

class W8A8MoEPrequantizedEligibility : public ::testing::Test {
protected:
    void SetUp() override {
        // topk == E makes every expert active with exactly three rows, which
        // keeps all rejection-matrix pointer/range mutations deterministic.
        problem_ = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/3,
                /*topk=*/4, /*seed=*/101);
    }

    template <typename F>
    void expect_declined(F &&mutate, const char *what) {
        auto call = build_prequantized_call(problem_);
        for (auto &dst : call->grouped_dst) {
            std::fill(dst.begin(), dst.end(), static_cast<uint16_t>(0x5A5A));
        }
        std::fill(call->output.begin(), call->output.end(),
                static_cast<uint16_t>(0x6B6B));
        mutate(*call);
        const auto source_before = call->grouped_s8;
        const auto destination_before = call->grouped_dst;
        const auto output_before = call->output;

        EXPECT_EQ(call->run_fastpath_only(), status_t::unimplemented) << what;
        EXPECT_EQ(call->grouped_s8, source_before)
                << what << ": source bytes changed";
        EXPECT_EQ(call->grouped_dst, destination_before)
                << what << ": destination sentinel changed";
        EXPECT_EQ(call->output, output_before)
                << what << ": reduce output sentinel changed";
    }

    MoEProblem problem_;
};

TEST_F(W8A8MoEEligibility, BaselineIsAccepted) {
    SKIP_IF_NO_ISA();
    auto call = build_call(problem_);
    EXPECT_EQ(call->run_fastpath_only(), status_t::success);
}

TEST(W8A8MoEPrequantized, AcceptsF32AndTailSafeBf16SourceScales) {
    SKIP_IF_NO_ISA();
    const auto p = make_problem(/*E=*/1, /*H=*/64, /*I=*/32, /*T=*/17,
            /*topk=*/1, /*seed=*/103);
    for (data_type_t scale_dt : {data_type_t::f32, data_type_t::bf16}) {
        auto call = build_prequantized_call(p, scale_dt);
        const auto source_before = call->grouped_s8;
        ASSERT_EQ(call->M[0], 17);
        ASSERT_EQ(call->run_fastpath_only(), status_t::success)
                << "scale dtype=" << static_cast<int>(scale_dt);
        EXPECT_EQ(call->grouped_s8, source_before);
        expect_close(call->output, reference_moe(p, call.get()), kRtol,
                scale_dt == data_type_t::f32 ? "F32 source scale"
                                             : "tail-safe BF16 source scale");
    }
}

TEST_F(W8A8MoEPrequantizedEligibility, RejectsBadSourceScaleMetadataAndValues) {
    expect_declined([](CallArgs &c) {
        c.params[0].quant_params.src_scale.buff = nullptr;
    }, "null source scale");
    expect_declined([](CallArgs &c) {
        c.params[0].quant_params.src_scale.dt = data_type_t::f16;
    }, "unsupported source scale dtype");
    expect_declined([](CallArgs &c) {
        c.params[0].quant_params.src_scale.dims = {c.M[0]};
    }, "rank-one source scale");
    expect_declined([](CallArgs &c) {
        c.params[0].quant_params.src_scale.dims = {c.M[0] + 1, 1};
    }, "wrong source scale row count");
    expect_declined([](CallArgs &c) {
        c.params[0].quant_params.src_scale.dims = {c.M[0], 2};
    }, "non-per-token source scale");
    expect_declined([](CallArgs &c) { c.src_scale_f32[0][0] = 0.f; },
            "zero source scale");
    expect_declined([](CallArgs &c) { c.src_scale_f32[0][0] = -0.25f; },
            "negative source scale");
    expect_declined([](CallArgs &c) {
        c.src_scale_f32[0][0] = std::numeric_limits<float>::infinity();
    }, "infinite source scale");
    expect_declined([](CallArgs &c) {
        c.src_scale_f32[0][0] = std::numeric_limits<float>::quiet_NaN();
    }, "NaN source scale");
    expect_declined([](CallArgs &c) {
        c.src_scale_scratch[0][0]
                = f32_to_bf16(std::numeric_limits<float>::infinity());
        c.params[0].quant_params.src_scale.buff = c.src_scale_scratch[0].data();
        c.params[0].quant_params.src_scale.dt = data_type_t::bf16;
    }, "BF16 infinite source scale");
}

TEST_F(W8A8MoEPrequantizedEligibility, RejectsNonUniformOrContradictoryModes) {
    expect_declined([](CallArgs &c) {
        c.params[1].dtypes.src = data_type_t::bf16;
        c.params[1].dynamic_quant = true;
        c.src[1] = c.grouped[1].data();
    }, "mixed active BF16/S8 modes");
    expect_declined([](CallArgs &c) { c.params[0].dynamic_quant = true; },
            "S8 with dynamic_quant=true");
    expect_declined([](CallArgs &c) {
        for (size_t i = 0; i < c.M.size(); ++i) {
            c.params[i].dtypes.src = data_type_t::bf16;
            c.params[i].dynamic_quant = false;
            c.src[i] = c.grouped[i].data();
        }
    }, "BF16 with dynamic_quant=false");
    expect_declined([](CallArgs &c) {
        c.params[0].quant_params.src_zp.buff = c.src_scale_f32[0].data();
        c.params[0].quant_params.src_zp.dt = data_type_t::f32;
        c.params[0].quant_params.src_zp.dims = {c.M[0], 1};
    }, "S8 source zero point");
}

TEST_F(W8A8MoEPrequantizedEligibility,
        RejectsUnsafeDestinationsAndRowPointers) {
    expect_declined([](CallArgs &c) {
        c.fused_moe.dst_down.clear();
        c.fused_moe.ldc_down.clear();
    }, "missing dst_down");
    expect_declined([](CallArgs &c) {
        c.fused_moe.dst_down.resize(c.fused_moe.dst_down.size() - 1);
    }, "short dst_down");
    expect_declined([](CallArgs &c) {
        c.fused_moe.ldc_down.resize(c.fused_moe.ldc_down.size() - 1);
    }, "short ldc_down");
    expect_declined([](CallArgs &c) { c.fused_moe.dst_down[1] = nullptr; },
            "mixed-null dst_down");
    expect_declined([](CallArgs &c) { --c.fused_moe.ldc_down[0]; },
            "undersized dst_down stride");
    expect_declined([](CallArgs &c) { c.src[0] = c.grouped_s8[0].data(); },
            "separate source and dst_down storage");
    expect_declined([](CallArgs &c) { c.row_ptrs[0] = c.row_ptrs[1]; },
            "duplicate row pointer");
    expect_declined([](CallArgs &c) {
        c.row_ptrs[0] = static_cast<const uint16_t *>(c.row_ptrs[0]) + 1;
    }, "partial destination row pointer");
    expect_declined([](CallArgs &c) { c.row_ptrs[0] = c.output.data(); },
            "outside destination row pointer");
    expect_declined([](CallArgs &c) {
        c.fused_moe.dst_down[1] = c.fused_moe.dst_down[0];
    }, "overlapping destination expert ranges");
    expect_declined([](CallArgs &c) {
        c.fused_moe.dst_down[0]
                = const_cast<uint8_t *>(static_cast<const uint8_t *>(c.src[0]))
                + 1;
    }, "offset source/destination overlap");
    expect_declined([](CallArgs &c) {
        c.fused_moe.dst_down[0] = const_cast<void *>(c.src[1]);
    }, "cross-expert source/destination overlap");
    expect_declined([](CallArgs &c) {
        c.fused_moe.dst_down[0] = const_cast<void *>(c.src[0]);
        ++c.fused_moe.ldc_down[0];
    }, "padded same-base source/destination overlap");
    expect_declined([](CallArgs &c) {
        c.postop.output = const_cast<void *>(c.src[0]);
    }, "source/reduced-output byte overlap");
    expect_declined([](CallArgs &c) {
        c.postop.output = c.fused_moe.dst_down[0];
    }, "dst_down/reduced-output byte overlap");
}

TEST_F(W8A8MoEEligibility, RejectsNonSiluActivation) {
    expect_declined([](CallArgs &c) {
        c.gated_act.act = grp_matmul_gated_act_t::gelu_and_mul;
    }, "gelu_and_mul");
    expect_declined([](CallArgs &c) {
        c.gated_act.act = grp_matmul_gated_act_t::swiglu_oai_mul;
    }, "swiglu_oai_mul");
    expect_declined([](CallArgs &c) {
        c.gated_act.act = grp_matmul_gated_act_t::none;
    }, "act=none");
}

TEST_F(W8A8MoEEligibility, RejectsNonInt8OrNonBf16) {
    expect_declined([](CallArgs &c) {
        c.params[0].dtypes.wei = data_type_t::bf16;
    }, "bf16 weights");
    expect_declined([](CallArgs &c) {
        c.params[0].dtypes.src = data_type_t::f32;
    }, "f32 source");
    expect_declined([](CallArgs &c) {
        c.params[0].dtypes.dst = data_type_t::f32;
    }, "f32 dst");
    expect_declined([](CallArgs &c) {
        c.params[0].dtypes.compute = data_type_t::bf16;
    }, "bf16 compute");
    expect_declined([](CallArgs &c) { c.params[0].dynamic_quant = false; },
            "static quant");
}

TEST_F(W8A8MoEEligibility, RejectsPrepackedFormatsAndOp1Postops) {
    expect_declined([](CallArgs &c) {
        c.params[0].mem_format_b = 'r';
        c.params[0].lowoha_algo
                = zendnnl::ops::matmul_algo_t::moe_custom_kernel;
    }, "prepacked W13/W2 weights");
    expect_declined([](CallArgs &c) { c.params[0].packing.pack_format_b = 1; },
            "GGML-packed W13/W2 weights");
    expect_declined([](CallArgs &c) { c.params[0].mem_format_a = 'r'; },
            "preformatted source");
    expect_declined([](CallArgs &c) {
        zendnnl::lowoha::matmul::matmul_post_op relu;
        relu.po_type = zendnnl::ops::post_op_type_t::relu;
        c.params[0].postop_.push_back(relu);
    }, "Op1 relu post-op");
}

TEST_F(W8A8MoEEligibility, RejectsBias) {
    expect_declined([this](CallArgs &c) {
        c.bias[0] = problem_.w13_scale.data();
        c.params[0].dtypes.bias = data_type_t::bf16;
    }, "Op1 bias");
    expect_declined([this](CallArgs &c) {
        c.fused_moe.bias_down[0] = problem_.w2_scale.data();
        c.fused_moe.bias_dt_down = data_type_t::bf16;
    }, "Op2 bias");
}

TEST_F(W8A8MoEEligibility, RejectsZeroPointsAndCoarseScales) {
    expect_declined([this](CallArgs &c) {
        c.params[0].quant_params.wei_zp.buff = problem_.w13_scale.data();
        c.params[0].quant_params.wei_zp.dims = {1, 2 * problem_.inter};
    }, "weight zero point");
    expect_declined([](CallArgs &c) {
        c.params[0].quant_params.wei_scale.dims = {}; // per-tensor
    }, "per-tensor weight scale");
    expect_declined([this](CallArgs &c) {
        // per-group along K rather than per-token
        c.params[0].quant_params.src_scale.dims = {c.M[0], 4};
        (void)this;
    }, "per-group source scale");
}

TEST_F(W8A8MoEEligibility, RejectsUnsupportedShapes) {
    expect_declined([](CallArgs &c) {
        // hidden size not a multiple of block_n
        for (auto &k : c.K) {
            k -= 1;
        }
    }, "unaligned hidden size");
    expect_declined([](CallArgs &c) {
        // gate/up output width not even -> no gate/up split
        for (auto &n : c.N) {
            n -= 1;
        }
    }, "odd gate/up width");
    expect_declined([](CallArgs &c) {
        c.N[1] = c.N[0] + 32; // non-uniform across experts
    }, "non-uniform N");
    expect_declined([](CallArgs &c) {
        c.ldb[0] = c.ldb[0] + 8; // weight rows not tight
    }, "padded weight rows");
    expect_declined([](CallArgs &c) {
        c.lda[0] = c.lda[0] + 8; // source rows not tight
    }, "padded source rows");
    expect_declined([](CallArgs &c) { c.transB[0] = false; }, "transB=false");
    expect_declined([](CallArgs &c) { c.transA[0] = true; }, "transA=true");
    expect_declined([](CallArgs &c) { c.alpha[0] = 2.0f; }, "alpha != 1");
    expect_declined([](CallArgs &c) { c.beta[0] = 1.0f; }, "beta != 0");
    expect_declined([](CallArgs &c) { c.layout[0] = 'c'; }, "column major");
    expect_declined([](CallArgs &c) { c.is_weights_const[0] = false; },
            "non-const weights");
}

TEST_F(W8A8MoEEligibility, RejectsCallerOwnedOp2Destination) {
    expect_declined([](CallArgs &c) {
        c.fused_moe.dst_down.assign(c.M.size(), nullptr);
        c.fused_moe.dst_down[0] = c.grouped[0].data();
        c.fused_moe.ldc_down.assign(c.M.size(), c.lda[0]);
    }, "caller-provided dst_down");
    expect_declined([](CallArgs &c) { c.dst[0] = c.grouped[0].data(); },
            "caller-provided Op1 dst");
}

TEST_F(W8A8MoEEligibility, RejectsInconsistentPostop) {
    expect_declined(
            [](CallArgs &c) { c.postop.row_ptrs = nullptr; }, "no row_ptrs");
    expect_declined(
            [](CallArgs &c) { c.postop.output = nullptr; }, "no output");
    expect_declined([](CallArgs &c) { c.postop.topk_weights = nullptr; },
            "weighted reduce without weights");
    // Grouped rows must account for exactly num_tokens * topk routed slots.
    expect_declined([](CallArgs &c) { c.postop.topk += 1; }, "topk mismatch");
    expect_declined([](CallArgs &c) { c.postop.num_tokens += 1; },
            "token count mismatch");
    // A row_ptr that does not name a grouped row would make the post-op read
    // something this path never wrote.
    expect_declined([](CallArgs &c) { c.row_ptrs[0] = c.output.data(); },
            "row_ptr outside the grouped buffers");
    expect_declined([](CallArgs &c) {
        c.row_ptrs[0] = static_cast<const uint16_t *>(c.row_ptrs[0]) + 1;
    }, "row_ptr not on a row boundary");
    // Two slots naming the same row: one grouped row would be double-counted
    // and another never read.
    expect_declined([](CallArgs &c) { c.row_ptrs[0] = c.row_ptrs[1]; },
            "duplicated row_ptr");
    expect_declined([](CallArgs &c) {
        c.postop.output = const_cast<void *>(c.src[0]);
    }, "reduce output aliases a BF16 W2 row");
}

TEST_F(W8A8MoEEligibility, Bf16OutputOverlapDeclinesWithoutFastpathWrites) {
    auto call = build_call(problem_);
    call->postop.output = const_cast<void *>(call->src[0]);
    const auto grouped_before = call->grouped;
    const auto output_before = call->output;

    EXPECT_EQ(call->run_fastpath_only(), status_t::unimplemented);
    EXPECT_EQ(call->grouped, grouped_before);
    EXPECT_EQ(call->output, output_before);
}

TEST_F(W8A8MoEEligibility, PublicValidationPrecedesFastpath) {
    auto expect_public_failure = [&](auto &&mutate, const char *what) {
        zendnnl::lowoha::matmul::clear_fused_moe_scratch();
        auto call = build_call(problem_);
        mutate(*call);
        const auto grouped_before = call->grouped;
        const auto output_before = call->output;

        EXPECT_EQ(call->run(), status_t::failure) << what;
        EXPECT_EQ(call->grouped, grouped_before) << what;
        EXPECT_EQ(call->output, output_before) << what;
        EXPECT_EQ(w8a8::packed_weight_cache_size(), 0u) << what;
    };

    expect_public_failure([](CallArgs &c) { c.bias.clear(); },
            "missing required bias vector");
    expect_public_failure([](CallArgs &c) {
        ASSERT_GT(c.params[0].active_matmul, 1u);
        c.params[0].total_matmul = c.params[0].active_matmul - 1;
    }, "total_matmul smaller than active_matmul");
}

TEST_F(W8A8MoEEligibility, IndependentWeightAllocationsDeclineCleanly) {
    auto call = build_call(problem_);
    const size_t gate_up_bytes = static_cast<size_t>(2 * problem_.inter)
            * static_cast<size_t>(problem_.hidden);
    const size_t down_bytes = static_cast<size_t>(problem_.hidden)
            * static_cast<size_t>(problem_.inter);
    std::vector<std::vector<int8_t>> gate_up_storage(call->weight.size());
    std::vector<std::vector<int8_t>> down_storage(
            call->fused_moe.down_weight.size());

    for (size_t i = 0; i < call->weight.size(); ++i) {
        const size_t offset = i + 1;
        gate_up_storage[i].resize(gate_up_bytes + offset + 1);
        std::memcpy(gate_up_storage[i].data() + offset, call->weight[i],
                gate_up_bytes);
        call->weight[i] = gate_up_storage[i].data() + offset;

        down_storage[i].resize(down_bytes + offset + 1);
        std::memcpy(down_storage[i].data() + offset,
                call->fused_moe.down_weight[i], down_bytes);
        call->fused_moe.down_weight[i] = down_storage[i].data() + offset;
    }

    const auto grouped_before = call->grouped;
    EXPECT_EQ(call->run_fastpath_only(), status_t::unimplemented);
    EXPECT_EQ(call->grouped, grouped_before);
    EXPECT_EQ(call->output, std::vector<uint16_t>(call->output.size(), 0));
}

// A declined call must still produce the right answer through the generic
// path: the fast path is an optimization, never a gate on functionality.
TEST(W8A8MoEFallback, DeclinedCallStillComputes) {
    SKIP_IF_NO_ISA();
    const auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/4,
            /*topk=*/2, /*seed=*/131);
    auto fast = build_call(p);
    ASSERT_EQ(fast->run(), status_t::success);

    // gelu_and_mul is declined by the fast path, so this call goes through
    // the generic dispatcher and must succeed there.
    auto declined = build_call(p);
    declined->gated_act.act = grp_matmul_gated_act_t::gelu_and_mul;
    ASSERT_EQ(declined->run_fastpath_only(), status_t::unimplemented);
    EXPECT_EQ(declined->run(), status_t::success);
}

TEST(W8A8MoEFallback, PrequantizedS8GeometryMissUsesMainS8Path) {
    SKIP_IF_NO_ISA();
    using namespace zendnnl::lowoha::matmul;
    // H=48 and padded dst_ldc=53 miss ALGO4 eligibility but are valid for
    // main's generic caller-prequantized-S8 fused-MoE implementation.
    const auto p = make_problem(/*E=*/1, /*H=*/48, /*I=*/32, /*T=*/5,
            /*topk=*/1, /*seed=*/137);
    auto call = build_prequantized_call(p, data_type_t::f32, /*dst_ldc=*/53);
    promote_weight_scales_to_f32(*call);
    const auto source_before = call->grouped_s8;
    const auto reference = reference_moe(p, call.get());
    ASSERT_EQ(call->run_fastpath_only(), status_t::unimplemented);
    moe_test_utils::AutoPromptAlgoOverride inactive_prompt(6);
    moe_test_utils::AutoDecodeAlgoOverride decode_w8a8(4);
    ASSERT_EQ(call->run(/*algo=*/0), status_t::success);
    EXPECT_EQ(call->grouped_s8, source_before);
    expect_close(call->output, reference, /*rtol=*/2e-2f,
            "ALGO4 S8 geometry decline through main generic fallback");
}

TEST(W8A8MoEPrequantized,
        GenericAlgorithmsRetainMainS8SupportForBothDestinationLayouts) {
    using namespace zendnnl::lowoha::matmul;
    // Generic fused MoE and ALGO4 round the activated intermediate at
    // different points. Keep this main-compatibility check separate from
    // ALGO4's tighter, executor-specific kRtol envelope.
    constexpr float generic_rtol = 2e-2f;
    const auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/4,
            /*topk=*/2, /*seed=*/138);
    moe_test_utils::AutoPromptAlgoOverride prompt_generic(0);
    moe_test_utils::AutoDecodeAlgoOverride decode_generic(0);

    // PR #659 supports caller-prequantized S8 whenever dst_down is explicit.
    // Exercise both a separate BF16 destination and ALGO4's same-backing view.
    // The latter is safe on the generic path because its two passes are
    // separated by a barrier; an enabled vertical-fusion attempt detects the
    // byte-stride mismatch and declines to that two-pass path.
    for (const data_type_t scale_dt : {data_type_t::bf16, data_type_t::f32}) {
        for (const bool same_backing : {false, true}) {
            for (const int algo : {0, 1, 2, 3, 5, 6}) {
                reset_grp_matmul_caches();
                auto call = build_prequantized_call(p, scale_dt);
                if (scale_dt == data_type_t::f32) {
                    promote_weight_scales_to_f32(*call);
                }
                if (!same_backing) {
                    for (size_t slot = 0; slot < call->src.size(); ++slot) {
                        call->src[slot] = call->grouped_s8[slot].data();
                    }
                }
                ASSERT_EQ(call->run(algo), status_t::success)
                        << "scale_dt=" << static_cast<int>(scale_dt)
                        << " same_backing=" << same_backing << " algo=" << algo;
                expect_close(call->output, reference_moe(p, call.get()),
                        generic_rtol,
                        ("generic prequantized S8 scale_dt="
                                + std::to_string(static_cast<int>(scale_dt))
                                + " same_backing="
                                + std::to_string(same_backing)
                                + " algo=" + std::to_string(algo))
                                .c_str());
            }
        }
    }
}

TEST(W8A8MoEFallback, PromptGeometryMissInheritsAlgo1Policy) {
    SKIP_IF_NO_ISA();
    using namespace zendnnl::lowoha::matmul;
    clear_fused_moe_scratch();

    // M=64 is prompt-class. H=48 deliberately misses ALGO4 alignment, while
    // latest main's unpinned prompt policy routes the declined call to ALGO1.
    const auto p = make_problem(/*E=*/1, /*H=*/48, /*I=*/32, /*T=*/64,
            /*topk=*/1, /*seed=*/139);
    auto generic_baseline = build_call(p, /*num_threads=*/2);
    ASSERT_EQ(generic_baseline->run(/*algo=*/1), status_t::success);

    auto call = build_call(p, /*num_threads=*/2);
    ASSERT_EQ(call->run_fastpath_only(), status_t::unimplemented);
    moe_test_utils::AutoPromptAlgoOverride prompt_w8a8(4);
    moe_test_utils::AutoDecodeAlgoOverride inactive_decode(6);
    moe_test_utils::GemmModeCaptureGuard capture;
    ASSERT_EQ(call->run(/*algo=*/0), status_t::success);
    const char *mode = test_api::s_last_group_matmul_direct_gemm_mode.load(
            std::memory_order_relaxed);
    ASSERT_NE(mode, nullptr);
    EXPECT_EQ(executed_algo_from_gemm_mode(mode), 1) << "mode=" << mode;
    EXPECT_EQ(call->output, generic_baseline->output)
            << "phase-4 decline must preserve forced-ALGO1 numerics";

    clear_fused_moe_scratch();
}

// ---------------------------------------------------------------------------
// Packed-weight cache lifecycle
// ---------------------------------------------------------------------------

TEST(W8A8MoECache, PacksOncePerWeightAndFlushes) {
    SKIP_IF_NO_ISA();
    zendnnl::lowoha::matmul::clear_fused_moe_scratch();
    WeightCacheGuard cache_on(/*mode=*/1);
    ASSERT_EQ(w8a8::packed_weight_cache_size(), 0u);

    const auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/4,
            /*topk=*/2, /*seed=*/149);
    auto first = build_call(p);
    ASSERT_EQ(first->run(), status_t::success);
    // One entry per role (gate/up and down).
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u);

    // Re-issuing against the same weights must reuse the pack, and must
    // reproduce the answer bit-for-bit.
    auto second = build_call(p);
    ASSERT_EQ(second->run(), status_t::success);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u);
    EXPECT_EQ(second->output, first->output);

    // A different weight tensor is a different entry.
    const auto q = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/4,
            /*topk=*/2, /*seed=*/151);
    auto third = build_call(q);
    ASSERT_EQ(third->run(), status_t::success);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 4u);

    // The library's existing MoE teardown must release them.
    zendnnl::lowoha::matmul::clear_fused_moe_scratch();
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 0u);

    // And a post-flush call must repack and still be correct.
    auto after = build_call(p);
    ASSERT_EQ(after->run(), status_t::success);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u);
    EXPECT_EQ(after->output, first->output);
}

TEST(W8A8MoECache, ModesOneAndTwoReusePackedEntries) {
    SKIP_IF_NO_ISA();
    using zendnnl::lowoha::matmul::clear_fused_moe_scratch;

    for (const int mode : {1, 2}) {
        clear_fused_moe_scratch();
        WeightCacheGuard cache_mode(mode);
        const auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/4,
                /*topk=*/2, /*seed=*/153);
        auto first = build_call(p);
        ASSERT_EQ(first->run(), status_t::success) << "mode=" << mode;
        EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u) << "mode=" << mode;

        auto second = build_call(p);
        ASSERT_EQ(second->run(), status_t::success) << "mode=" << mode;
        EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u) << "mode=" << mode;
        EXPECT_EQ(second->output, first->output) << "mode=" << mode;
    }
    clear_fused_moe_scratch();
}

TEST(W8A8MoECache, RuntimeDisableRequiresExplicitGenerationClear) {
    SKIP_IF_NO_ISA();
    using zendnnl::lowoha::matmul::clear_fused_moe_scratch;

    clear_fused_moe_scratch();
    WeightCacheGuard restore(/*mode=*/1);
    auto &config = zendnnl::common::matmul_config_t::instance();
    const auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/4,
            /*topk=*/2, /*seed=*/155);
    auto first = build_call(p);
    ASSERT_EQ(first->run(), status_t::success);
    ASSERT_EQ(w8a8::packed_weight_cache_size(), 2u);

    config.set_weight_cache(0);
    auto disabled = build_call(p);
    EXPECT_EQ(disabled->run_fastpath_only(), status_t::unimplemented);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u)
            << "changing mode must not retire entries an in-flight call may "
               "still own";

    clear_fused_moe_scratch();
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 0u);
    config.set_weight_cache(1);
    auto next_generation = build_call(p);
    ASSERT_EQ(next_generation->run(), status_t::success);
    EXPECT_EQ(next_generation->output, first->output);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u);
    clear_fused_moe_scratch();
}

TEST(W8A8MoECache, DisabledDeclinesWithoutPublishingOrWriting) {
    SKIP_IF_NO_ISA();
    using zendnnl::lowoha::matmul::clear_fused_moe_scratch;

    clear_fused_moe_scratch();
    WeightCacheGuard cache_off(/*mode=*/0);
    const auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/4,
            /*topk=*/2, /*seed=*/157);

    auto direct = build_call(p);
    const auto output_before = direct->output;
    const auto grouped_before = direct->grouped;
    EXPECT_EQ(direct->run_fastpath_only(), status_t::unimplemented);
    EXPECT_EQ(direct->output, output_before);
    EXPECT_EQ(direct->grouped, grouped_before);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 0u);

    // The public dispatcher must still complete through its generic fallback;
    // disabling cache disables this optimization, not the operation. Four
    // total experts select the normal few-expert ALGO-2 refinement, so compare
    // against that generic scheduler directly rather than against ALGO 4's
    // intentionally different rounding sequence.
    auto generic_baseline = build_call(p);
    ASSERT_EQ(generic_baseline->run(/*algo=*/2), status_t::success);
    auto fallback = build_call(p);
    ASSERT_EQ(fallback->run(), status_t::success);
    EXPECT_EQ(fallback->output, generic_baseline->output);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 0u);
    clear_fused_moe_scratch();
}

TEST(W8A8MoECache, DisabledDirectS8UsesMainFallback) {
    SKIP_IF_NO_ISA();
    using zendnnl::lowoha::matmul::clear_fused_moe_scratch;

    clear_fused_moe_scratch();
    WeightCacheGuard cache_off(/*mode=*/0);
    const auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/4,
            /*topk=*/2, /*seed=*/158);
    auto call = build_prequantized_call(p, data_type_t::f32);
    promote_weight_scales_to_f32(*call);
    const auto reference = reference_moe(p, call.get());

    EXPECT_EQ(call->run_fastpath_only(), status_t::unimplemented);
    ASSERT_EQ(call->run(), status_t::success);
    expect_close(call->output, reference, /*rtol=*/2e-2f,
            "cache-disabled ALGO4 S8 call through main generic fallback");
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 0u);
    clear_fused_moe_scratch();
}

TEST(W8A8MoECache, PerCallDisableDeclinesWithoutPublishingOrWriting) {
    SKIP_IF_NO_ISA();
    using zendnnl::lowoha::matmul::clear_fused_moe_scratch;

    clear_fused_moe_scratch();
    WeightCacheGuard process_cache_on(/*mode=*/1);
    const auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/4,
            /*topk=*/2, /*seed=*/158);
    auto call = build_call(p);
    for (auto &param : call->params) {
        param.weight_cache_type = 0;
    }
    const auto output_before = call->output;
    const auto grouped_before = call->grouped;

    EXPECT_EQ(call->run_fastpath_only(), status_t::unimplemented);
    EXPECT_EQ(call->output, output_before);
    EXPECT_EQ(call->grouped, grouped_before);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 0u);
    clear_fused_moe_scratch();
}

TEST(W8A8MoECache, HonorsConfiguredLruCapacityBelowModelWorkingSet) {
    SKIP_IF_NO_ISA();
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    setenv("ZENDNNL_LRU_CACHE_CAPACITY", "1", /*overwrite=*/1);

    EXPECT_EXIT(
            {
                const auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32,
                        /*T=*/4, /*topk=*/2, /*seed=*/159);
                auto call = build_call(p);
                const status_t st = call->run(/*algo=*/4);
                std::exit(st == status_t::success
                                        && w8a8::packed_weight_cache_size()
                                                == 1u
                                ? 0
                                : 1);
            },
            ::testing::ExitedWithCode(0), "");

    unsetenv("ZENDNNL_LRU_CACHE_CAPACITY");
}

TEST(W8A8MoECache, LiveScalesRefreshWithoutFlushingPackedWeights) {
    SKIP_IF_NO_ISA();
    using zendnnl::lowoha::matmul::clear_fused_moe_scratch;

    clear_fused_moe_scratch();
    WeightCacheGuard cache_on(/*mode=*/1);
    auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/2,
            /*topk=*/4, /*seed=*/159);
    auto first = build_call(p);
    ASSERT_EQ(first->run(), status_t::success);
    expect_close(first->output, reference_moe(p), kRtol, "original scales");
    ASSERT_EQ(w8a8::packed_weight_cache_size(), 2u);

    const int8_t *const w13_base = p.w13.data();
    const int8_t *const w2_base = p.w2.data();
    const uint16_t *const w13_scale_base = p.w13_scale.data();
    const uint16_t *const w2_scale_base = p.w2_scale.data();
    const auto replacement = make_problem(/*E=*/4, /*H=*/64, /*I=*/32,
            /*T=*/2, /*topk=*/4, /*seed=*/161);

    // Scales are current-call metadata, not part of the packed-weight cache
    // generation. Change them in place without flushing the weight pack.
    std::copy(replacement.w13_scale.begin(), replacement.w13_scale.end(),
            p.w13_scale.begin());
    std::copy(replacement.w2_scale.begin(), replacement.w2_scale.end(),
            p.w2_scale.begin());
    ASSERT_EQ(p.w13.data(), w13_base);
    ASSERT_EQ(p.w2.data(), w2_base);
    ASSERT_EQ(p.w13_scale.data(), w13_scale_base);
    ASSERT_EQ(p.w2_scale.data(), w2_scale_base);

    auto second = build_call(p);
    ASSERT_EQ(second->run(), status_t::success);
    expect_close(second->output, reference_moe(p), kRtol, "replacement scales");
    EXPECT_NE(second->output, first->output);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u);
    clear_fused_moe_scratch();
}

TEST(W8A8MoECache, ConcurrentCallsUseIndependentLiveScales) {
    SKIP_IF_NO_ISA();
    using zendnnl::lowoha::matmul::clear_fused_moe_scratch;

    clear_fused_moe_scratch();
    WeightCacheGuard cache_on(/*mode=*/1);
    const auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/2,
            /*topk=*/4, /*seed=*/173);
    auto alternate = p;
    const auto replacement = make_problem(/*E=*/4, /*H=*/64, /*I=*/32,
            /*T=*/2, /*topk=*/4, /*seed=*/175);
    alternate.w13_scale = replacement.w13_scale;
    alternate.w2_scale = replacement.w2_scale;

    auto first = build_call(p, /*num_threads=*/2);
    auto second = build_call(p, /*num_threads=*/2);
    for (size_t slot = 0; slot < second->M.size(); ++slot) {
        const size_t expert = static_cast<size_t>(p.active_expert_ids[slot]);
        second->params[slot].quant_params.wei_scale.buff
                = alternate.w13_scale.data() + expert * 2 * p.inter;
        second->fused_moe.down_scale[slot].buff
                = alternate.w2_scale.data() + expert * p.hidden;
    }

    status_t first_status = status_t::failure;
    status_t second_status = status_t::failure;
    std::thread first_thread(
            [&] { first_status = first->run_fastpath_only(); });
    std::thread second_thread(
            [&] { second_status = second->run_fastpath_only(); });
    first_thread.join();
    second_thread.join();

    ASSERT_EQ(first_status, status_t::success);
    ASSERT_EQ(second_status, status_t::success);
    expect_close(first->output, reference_moe(p), kRtol, "concurrent scales A");
    expect_close(second->output, reference_moe(alternate), kRtol,
            "concurrent scales B");
    EXPECT_NE(first->output, second->output);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u);
    clear_fused_moe_scratch();
}

TEST(W8A8MoECache, FlushStartsNewGenerationAtSameAddress) {
    SKIP_IF_NO_ISA();
    using zendnnl::lowoha::matmul::clear_fused_moe_scratch;

    clear_fused_moe_scratch();
    WeightCacheGuard cache_on(/*mode=*/1);
    auto p = make_problem(/*E=*/4, /*H=*/64, /*I=*/32, /*T=*/4,
            /*topk=*/2, /*seed=*/163);
    auto first = build_call(p);
    ASSERT_EQ(first->run(), status_t::success);
    expect_close(first->output, reference_moe(p), kRtol, "generation one");
    ASSERT_EQ(w8a8::packed_weight_cache_size(), 2u);

    const int8_t *const w13_base = p.w13.data();
    const int8_t *const w2_base = p.w2.data();
    const uint16_t *const w13_scale_base = p.w13_scale.data();
    const uint16_t *const w2_scale_base = p.w2_scale.data();
    const auto replacement = make_problem(/*E=*/4, /*H=*/64, /*I=*/32,
            /*T=*/4, /*topk=*/2, /*seed=*/167);

    // The documented lifecycle boundary must precede any mutation or reuse.
    clear_fused_moe_scratch();
    std::copy(replacement.w13.begin(), replacement.w13.end(), p.w13.begin());
    std::copy(replacement.w2.begin(), replacement.w2.end(), p.w2.begin());
    std::copy(replacement.w13_scale.begin(), replacement.w13_scale.end(),
            p.w13_scale.begin());
    std::copy(replacement.w2_scale.begin(), replacement.w2_scale.end(),
            p.w2_scale.begin());
    ASSERT_EQ(p.w13.data(), w13_base);
    ASSERT_EQ(p.w2.data(), w2_base);
    ASSERT_EQ(p.w13_scale.data(), w13_scale_base);
    ASSERT_EQ(p.w2_scale.data(), w2_scale_base);

    auto second = build_call(p);
    ASSERT_EQ(second->run(), status_t::success);
    expect_close(second->output, reference_moe(p), kRtol,
            "same-address generation two");
    EXPECT_NE(second->output, first->output);
    EXPECT_EQ(w8a8::packed_weight_cache_size(), 2u);
    clear_fused_moe_scratch();
}

// ---------------------------------------------------------------------------
// Gate/up epilogue: SiLU value classes
// ---------------------------------------------------------------------------
//
// These tests pin the epilogue's activation contract independently of the
// implementation behind it, so the same text can be compiled against the
// file-private `silu_ps` and against the shared
// `group_matmul_act_avx512::silu_avx512` and must pass either way.  Both are
// approximations with different error shapes (different exp polynomial, and
// only the shared one refines its reciprocal with a Newton step), so anything
// asserted here is a property the epilogue owes its caller rather than a
// property of one polynomial.
//
// The gate is NOT the file's end-to-end `kRtol`.  The epilogue rounds
// `silu(gate) * up` to bf16 immediately, and bf16 has an 8-bit significand, so
// output quantization alone costs up to 2^-8 = 3.9e-3 relative -- comparing a
// bf16 store against an unrounded f32 reference measures mostly that, and any
// tolerance derived from it would be too slack to detect a bad polynomial.
//
// So the reference is rounded to bf16 the same way the kernel rounds, and the
// requirement is that the kernel land on the correctly-rounded answer or one
// bf16 ULP away.  That is roughly 5x tighter than `kRtol` in the places where
// it binds, and it isolates activation error from output quantization: an
// activation whose f32 error exceeded half a bf16 ULP would start pushing
// results two ULPs out and fail.
namespace {

// Slack for the inequality bounds below (0 <= silu(x) <= x and friends):
// one bf16 ULP relative, since the compared value has been through a bf16
// store.  Not an accuracy tolerance.
constexpr float kBf16Slack = 4e-3f;

// Distance to the next representable bf16 magnitude above `b` -- one ULP at
// `b`'s exponent.
float bf16_ulp(uint16_t b) {
    const uint16_t mag = static_cast<uint16_t>(b & 0x7FFF);
    return std::fabs(
            bf16_to_f32(static_cast<uint16_t>(mag + 1)) - bf16_to_f32(mag));
}

// Drives the real row-count dispatch (`tinygemm_gate_up`) so the epilogue sees
// exactly the gate values `xs` against the up values `ys`, and returns the
// bf16 it stored.
//
// Uses only the documented packed layout.  With the biased activation row set
// to 128 everywhere except a single +1 at k = 0, and every output channel's
// weight set to 1 at k = 0 and 0 elsewhere, `acc - compensation` is exactly 1
// for each channel: acc = 129 * 1 and compensation = 128 * 1.  The epilogue
// then forms x = As * 1 * Bs0[n] and y = As * 1 * Bs1[n], so with As = 1 the
// two per-channel scale vectors are a direct handle on the activation input.
std::vector<uint16_t> run_gate_up_epilogue(
        const std::vector<float> &xs, const std::vector<float> &ys) {
    const int64_t N = w8a8::block_n;
    const int64_t K = 32; // multiple of the VNNI step and of 32
    EXPECT_EQ(static_cast<int64_t>(xs.size()), N);
    EXPECT_EQ(static_cast<int64_t>(ys.size()), N);

    std::vector<int8_t> src(static_cast<size_t>(N) * K, 0);
    for (int64_t n = 0; n < N; ++n) {
        src[static_cast<size_t>(n) * K] = 1;
    }
    const int64_t oc_stride = w8a8::packed_bytes_per_oc(K);
    std::vector<int8_t> packed(
            static_cast<size_t>(N) * static_cast<size_t>(oc_stride), 0);
    EXPECT_EQ(w8a8::pack_weights(src.data(), packed.data(), 1, N, K, 1),
            status_t::success);

    std::vector<uint8_t> A(static_cast<size_t>(K), 128);
    A[0] = 129;
    const std::vector<float> As(1, 1.0f);
    const auto *comp = reinterpret_cast<const int32_t *>(
            packed.data() + static_cast<size_t>(N) * K);

    std::vector<uint16_t> C(static_cast<size_t>(N), 0);
    w8a8::tinygemm_gate_up(A.data(), packed.data(), packed.data(), C.data(),
            As.data(), xs.data(), ys.data(), comp, comp, /*M=*/1, K,
            /*lda=*/K, /*ldb=*/w8a8::block_n, /*ldc=*/N);
    return C;
}

double silu_exact(double x) {
    return x / (1.0 + std::exp(-x));
}

} // namespace

// Dense-ish sweep of the band the epilogue actually operates in.  Both
// implementations are well inside tolerance here; this is the test that would
// catch a wrong polynomial, a dropped `* up`, or a lane permutation.
TEST(W8A8MoESiluEpilogue, MatchesExactSiluOverOperatingRange) {
    SKIP_IF_NO_ISA();
    const int64_t N = w8a8::block_n;
    // -40 .. +40 in 0.05 steps, taken 32 lanes at a time, plus the awkward
    // small magnitudes where silu(x) ~ x/2 and rounding is most delicate.
    std::vector<float> all;
    for (int i = -800; i <= 800; ++i) {
        all.push_back(static_cast<float>(i) * 0.05f);
    }
    for (float v : {1e-3f, -1e-3f, 1e-2f, -1e-2f, 0.125f, -0.125f, 1.f, -1.f}) {
        all.push_back(v);
    }
    while (static_cast<int64_t>(all.size()) % N != 0) {
        all.push_back(0.f);
    }

    // A non-trivial `up` vector, so a silently dropped multiply cannot pass.
    std::vector<float> ys(static_cast<size_t>(N));
    for (int64_t n = 0; n < N; ++n) {
        ys[static_cast<size_t>(n)] = 0.5f + 0.125f * static_cast<float>(n % 7);
    }

    size_t checked = 0, exact_hits = 0, one_ulp = 0;
    double worst_ulps = 0.0;
    float worst_x = 0.f;
    for (size_t base = 0; base < all.size(); base += static_cast<size_t>(N)) {
        const std::vector<float> xs(all.begin() + static_cast<long>(base),
                all.begin() + static_cast<long>(base) + N);
        const std::vector<uint16_t> got = run_gate_up_epilogue(xs, ys);
        for (int64_t n = 0; n < N; ++n) {
            const float x = xs[static_cast<size_t>(n)];
            const float y = ys[static_cast<size_t>(n)];
            // Reference rounded exactly where the kernel rounds.
            const uint16_t want_b = f32_to_bf16(
                    static_cast<float>(silu_exact(x) * static_cast<double>(y)));
            const float want = bf16_to_f32(want_b);
            const float have = bf16_to_f32(got[static_cast<size_t>(n)]);
            const float ulp = bf16_ulp(want_b);

            if (got[static_cast<size_t>(n)] == want_b) {
                ++exact_hits;
            } else {
                ++one_ulp;
            }
            const double ulps = ulp > 0.f ? std::fabs(have - want) / ulp : 0.0;
            if (ulps > worst_ulps) {
                worst_ulps = ulps;
                worst_x = x;
            }
            ASSERT_LE(std::fabs(have - want), 1.001f * ulp + 1e-30f)
                    << "gate x=" << x << " up=" << y << " want=" << want
                    << " got=" << have << " (more than one bf16 ULP out)";
            ++checked;
        }
    }
    EXPECT_GE(checked, 1600u);
    std::cout << "[silu] operating range: " << checked << " lanes, "
              << exact_hits << " bit-exact vs correctly-rounded bf16, "
              << one_ulp << " one ULP out, worst " << worst_ulps
              << " ULP at x=" << worst_x << "\n";
}

// Value classes that are not about accuracy but about not producing garbage:
// tiny and subnormal magnitudes, signed zero, and magnitudes past the point
// where exp(-x) saturates.  The bounds asserted are the ones any SiLU owes
// its caller; the two implementations disagree on the exact encoding of the
// deep negative tail, and that disagreement is recorded rather than pinned.
TEST(W8A8MoESiluEpilogue, TinyExtremeAndNonFiniteAreWellBehaved) {
    SKIP_IF_NO_ISA();
    const int64_t N = w8a8::block_n;
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    std::vector<float> xs = {0.f, -0.f, 1.4e-45f, -1.4e-45f, 1e-40f, -1e-40f,
            1.17549435e-38f, -1.17549435e-38f, 1e-20f, -1e-20f, 1e-8f, -1e-8f,
            0.5f, -0.5f, 20.f, -20.f, 87.f, -87.f, 87.336548f, -87.336548f,
            88.f, -88.f, 88.722839f, -88.722839f, 100.f, -100.f, 200.f, -200.f,
            300.f, -300.f, qnan, inf};
    ASSERT_EQ(static_cast<int64_t>(xs.size()), N);
    const std::vector<float> ys(static_cast<size_t>(N), 1.0f);

    const std::vector<uint16_t> got = run_gate_up_epilogue(xs, ys);

    for (int64_t n = 0; n < N; ++n) {
        const float x = xs[static_cast<size_t>(n)];
        const float have = bf16_to_f32(got[static_cast<size_t>(n)]);

        if (std::isnan(x)) {
            EXPECT_TRUE(std::isnan(have)) << "NaN gate must propagate";
            continue;
        }
        if (std::isinf(x)) {
            // silu(+inf) = +inf; the epilogue must not turn it into a NaN.
            EXPECT_FALSE(std::isnan(have)) << "+Inf gate produced NaN";
            EXPECT_GT(have, 0.f);
            continue;
        }
        // Finite input, and an exact result well inside bf16's finite range:
        // the output must be finite.
        const double want = silu_exact(x);
        ASSERT_LT(std::fabs(want), 3.3e38);
        EXPECT_TRUE(std::isfinite(have))
                << "finite gate x=" << x << " produced " << have;

        if (x >= 0.f) {
            // 0 <= silu(x) <= x for x >= 0, up to bf16 rounding.
            EXPECT_GE(have, 0.f) << "x=" << x;
            EXPECT_LE(have, x * (1.f + kBf16Slack) + 1e-30f) << "x=" << x;
        } else if (x >= -88.f) {
            // silu has a single minimum of ~-0.2785 on the negative axis.
            EXPECT_LE(have, 0.f) << "x=" << x;
            EXPECT_GE(have, -0.30f) << "x=" << x;
        } else {
            // Past ~-88 both implementations underflow to a numerically-zero
            // magnitude, but not to the same encoding: one clamps exp's
            // argument and lands on -0, the other clamps the exponent and
            // lands on ~-5e-37.  Both are zero for every downstream purpose;
            // pinning either would bake in one polynomial's quirk.
            EXPECT_LE(std::fabs(have), 1e-30f)
                    << "deep negative tail x=" << x << " gave " << have;
        }
    }
    std::cout << "[silu] tail encodings (recorded, not pinned): x=-100 -> "
              << bf16_to_f32(got[25]) << ", x=-200 -> " << bf16_to_f32(got[27])
              << ", x=-300 -> " << bf16_to_f32(got[29]) << "\n";
}
