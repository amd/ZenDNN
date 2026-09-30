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
 * @file test_routed_moe.cpp
 * @brief Tests for the routed-MoE direct APIs.
 *
 * Covers the capability query, every rejection the validator can
 * produce, byte-exact agreement of the packed weight layout against an
 * independent scalar packer, numerical agreement of the executor against
 * a scalar reference across several MoE geometries and token counts, the
 * expert-map masking path, and the packed-weight cache flush.
 *
 * The geometries under test are deliberately unrelated to one another
 * (expert count, topk, hidden size, intermediate size and the token
 * tails all differ) because the point of this contract is that it is not
 * tuned to one model.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

#include <omp.h>

#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/routed_moe/routed_moe_internal.hpp"
#include "lowoha_operators/matmul/routed_moe/routed_moe_kernels.hpp"

// gtest_utils.cpp.  Declared here rather than by including gtest_utils.hpp,
// whose namespace-scope using-directives would make this file's aliases
// ambiguous.
void reset_grp_matmul_caches();

// The CRT has no POSIX setenv/unsetenv. _putenv_s(name, "") removes the
// variable, matching unsetenv. Linux keeps the real functions.
#if defined(_MSC_VER)
static inline int setenv(const char *name, const char *value, int overwrite) {
    if (!overwrite && std::getenv(name) != nullptr) { return 0; }
    return _putenv_s(name, value);
}
static inline int unsetenv(const char *name) {
    return _putenv_s(name, "");
}
#endif

namespace {

using zendnnl::lowoha::matmul::group_matmul_projection_params;
using zendnnl::lowoha::matmul::group_matmul_routed_moe_flush_weight_cache;
using zendnnl::lowoha::matmul::group_matmul_routed_moe_pack_weights;
using zendnnl::lowoha::matmul::group_matmul_routed_moe_packed_row_bytes;
using zendnnl::lowoha::matmul::group_matmul_routed_moe_query;
using zendnnl::lowoha::matmul::group_matmul_routed_moe_validate;
using zendnnl::lowoha::matmul::group_matmul_routing_params;
using zendnnl::lowoha::matmul::routed_fused_moe_direct;
using zendnnl::lowoha::matmul::routed_moe_activation_t;
using zendnnl::lowoha::matmul::routed_moe_capability;
using zendnnl::lowoha::matmul::routed_moe_params;
using zendnnl::lowoha::matmul::routed_moe_quant_t;
using zendnnl::lowoha::matmul::routed_moe::execute;

using data_type_t = zendnnl::common::data_type_t;
using status_t = zendnnl::error_handling::status_t;

// ---------------------------------------------------------------------------
// bf16 helpers (round-to-nearest-even, matching _mm512_cvtne2ps_pbh)
// ---------------------------------------------------------------------------
uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    return static_cast<uint16_t>((u + 0x7fffu + lsb) >> 16);
}

float bf16_to_f32(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// A routed-MoE problem, owning its buffers.
// ---------------------------------------------------------------------------
struct moe_problem {
    int64_t M = 0, K = 0, N = 0, E = 0, topk = 0;
    routed_moe_activation_t act = routed_moe_activation_t::silu_and_mul;

    std::vector<uint16_t> src, dst;
    std::vector<int8_t> w13, w2;
    std::vector<float> s13, s2;
    std::vector<int32_t> ids;
    std::vector<float> weights;

    void build(int64_t M_, int64_t K_, int64_t N_, int64_t E_, int64_t topk_,
            uint32_t seed) {
        M = M_;
        K = K_;
        N = N_;
        E = E_;
        topk = topk_;

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> act(-2.f, 2.f);
        std::uniform_int_distribution<int> q(-127, 127);
        std::uniform_real_distribution<float> sc(0.002f, 0.02f);
        std::uniform_int_distribution<int> pick(0, static_cast<int>(E) - 1);
        std::uniform_real_distribution<float> rw(0.05f, 0.95f);

        src.resize(static_cast<size_t>(M * K));
        for (auto &v : src) {
            v = f32_to_bf16(act(rng));
        }
        dst.assign(static_cast<size_t>(M * K), 0);

        w13.resize(static_cast<size_t>(E * 2 * N * K));
        for (auto &v : w13) {
            v = static_cast<int8_t>(q(rng));
        }
        w2.resize(static_cast<size_t>(E * K * N));
        for (auto &v : w2) {
            v = static_cast<int8_t>(q(rng));
        }

        s13.resize(static_cast<size_t>(E * 2 * N));
        for (auto &v : s13) {
            v = sc(rng);
        }
        s2.resize(static_cast<size_t>(E * K));
        for (auto &v : s2) {
            v = sc(rng);
        }

        ids.resize(static_cast<size_t>(M * topk));
        for (auto &v : ids) {
            v = pick(rng);
        }
        weights.resize(static_cast<size_t>(M * topk));
        for (auto &v : weights) {
            v = rw(rng);
        }
    }

    /// Rescale the gate/up scales so the pre-activation gate is O(1) instead
    /// of O(sqrt(K)).  At the default scales |gate| ~ 10 * sqrt(K / 256),
    /// where SiLU and GELU both reduce to max(x, 0) and an activation mix-up
    /// would stay inside the pipeline tolerance.
    void unit_gate() {
        const float f = 1.25f / std::sqrt(static_cast<float>(K));
        for (auto &v : s13) {
            v *= f;
        }
    }

    routed_moe_params params() {
        routed_moe_params p;
        p.num_tokens = M;
        p.hidden_size = K;
        p.intermediate_size = N;
        p.num_local_experts = E;
        p.topk = topk;

        p.src = src.data();
        p.src_stride = K;
        p.src_dt = data_type_t::bf16;
        p.dst = dst.data();
        p.dst_stride = K;
        p.dst_dt = data_type_t::bf16;

        p.gate_up_weight = w13.data();
        p.down_weight = w2.data();
        p.wei_dt = data_type_t::s8;

        p.gate_up_scale = s13.data();
        p.down_scale = s2.data();
        p.scale_dt = data_type_t::f32;

        p.topk_ids = ids.data();
        p.topk_ids_stride = topk;
        p.topk_weights = weights.data();
        p.topk_weights_stride = topk;

        p.activation = act;
        p.quant_scheme = routed_moe_quant_t::sym_per_oc_w8a8_dynamic_per_token;
        p.num_threads = 0;
        return p;
    }
};

// ---------------------------------------------------------------------------
// Scalar reference.
//
// Mirrors the executor's pipeline including where it rounds to bf16 and the
// order in which it applies the two scales, so the only expected divergence is
// the exp/reciprocal approximation inside SiLU, or the erf approximation inside
// GELU (exact std::erf here).  Written independently of the kernels (plain
// loops, no intrinsics, no shared helpers).
//
// `expert_map` follows the same convention as the contract: null means the ids
// are already local, and a negative entry masks a slot out.
// ---------------------------------------------------------------------------
std::vector<uint16_t> reference_moe(
        const moe_problem &pb, const int32_t *expert_map = nullptr) {
    const int64_t M = pb.M, K = pb.K, N = pb.N, topk = pb.topk;
    std::vector<uint16_t> out(static_cast<size_t>(M * K), 0);

    std::vector<int32_t> qa(static_cast<size_t>(K));
    std::vector<float> inter(static_cast<size_t>(N));
    std::vector<int32_t> qi(static_cast<size_t>(N));
    std::vector<float> acc(static_cast<size_t>(K));

    for (int64_t m = 0; m < M; ++m) {
        float amax = 0.f;
        for (int64_t k = 0; k < K; ++k) {
            amax = std::max(amax, std::fabs(bf16_to_f32(pb.src[m * K + k])));
        }
        amax = std::max(amax, 1e-7f);
        const float sa = amax / 127.f;
        const float inv = 127.f / amax;
        for (int64_t k = 0; k < K; ++k) {
            qa[k] = static_cast<int32_t>(
                    std::nearbyintf(bf16_to_f32(pb.src[m * K + k]) * inv));
        }

        std::fill(acc.begin(), acc.end(), 0.f);
        for (int64_t t = 0; t < topk; ++t) {
            const int32_t raw = pb.ids[m * topk + t];
            const int64_t e = expert_map != nullptr ? expert_map[raw] : raw;
            if (e < 0) { continue; } // slot not resident on this rank

            const int8_t *w13e = pb.w13.data() + e * 2 * N * K;
            const float *s13e = pb.s13.data() + e * 2 * N;
            for (int64_t j = 0; j < N; ++j) {
                int32_t g = 0, u = 0;
                for (int64_t k = 0; k < K; ++k) {
                    g += qa[k] * w13e[j * K + k];
                    u += qa[k] * w13e[(N + j) * K + k];
                }
                const float gf = (static_cast<float>(g) * sa) * s13e[j];
                const float uf = (static_cast<float>(u) * sa) * s13e[N + j];
                const float act
                        = pb.act == routed_moe_activation_t::gelu_and_mul
                        ? 0.5f * gf * (1.f + std::erf(gf * 0.70710678f))
                        : gf / (1.f + std::exp(-gf));
                inter[j] = bf16_to_f32(f32_to_bf16(act * uf));
            }

            float amax2 = 0.f;
            for (int64_t j = 0; j < N; ++j) {
                amax2 = std::max(amax2, std::fabs(inter[j]));
            }
            amax2 = std::max(amax2, 1e-7f);
            const float sa2 = amax2 / 127.f;
            const float inv2 = 127.f / amax2;
            for (int64_t j = 0; j < N; ++j) {
                qi[j] = static_cast<int32_t>(std::nearbyintf(inter[j] * inv2));
            }

            const int8_t *w2e = pb.w2.data() + e * K * N;
            const float *s2e = pb.s2.data() + e * K;
            const float rw = pb.weights[m * topk + t];
            for (int64_t k = 0; k < K; ++k) {
                int32_t d = 0;
                for (int64_t j = 0; j < N; ++j) {
                    d += qi[j] * w2e[k * N + j];
                }
                const float df = (static_cast<float>(d) * sa2) * s2e[k];
                acc[k] += bf16_to_f32(f32_to_bf16(df * rw));
            }
        }
        for (int64_t k = 0; k < K; ++k) {
            out[m * K + k] = f32_to_bf16(acc[k]);
        }
    }
    return out;
}

/// Relative mean absolute error against a reference, in bf16 space.
double rel_mae(
        const std::vector<uint16_t> &got, const std::vector<uint16_t> &want) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double w = bf16_to_f32(want[i]);
        num += std::fabs(bf16_to_f32(got[i]) - w);
        den += std::fabs(w);
    }
    return den > 0.0 ? num / den : num;
}

// ---------------------------------------------------------------------------
// Independent scalar packer, for a byte-for-byte layout diff.
// ---------------------------------------------------------------------------
std::vector<int8_t> reference_pack(
        const int8_t *src, int64_t E, int64_t OC, int64_t IC) {
    const int64_t row = IC + 4;
    std::vector<int8_t> out(static_cast<size_t>(E * OC * row), 0);
    for (int64_t e = 0; e < E; ++e) {
        for (int64_t nb = 0; nb < OC / 32; ++nb) {
            int8_t *blk = out.data() + e * OC * row + nb * 32 * row;
            const int8_t *sblk = src + e * OC * IC + nb * 32 * IC;
            for (int64_t k4 = 0; k4 < IC / 4; ++k4) {
                for (int64_t n = 0; n < 32; ++n) {
                    for (int64_t b = 0; b < 4; ++b) {
                        blk[k4 * 128 + n * 4 + b] = sblk[n * IC + k4 * 4 + b];
                    }
                }
            }
            for (int64_t n = 0; n < 32; ++n) {
                int32_t comp = 0;
                for (int64_t k = 0; k < IC; ++k) {
                    comp += 128 * static_cast<int32_t>(sblk[n * IC + k]);
                }
                std::memcpy(blk + 32 * IC + n * 4, &comp, sizeof(comp));
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Geometries.  Two are shaped like well-known MoE layers (small enough to
// stay a unit test); the rest exist to break any assumption that E, topk,
// hidden and intermediate are related, and to hit token tails.
// ---------------------------------------------------------------------------
struct geometry {
    const char *name;
    int64_t K, N, E, topk;
};

const geometry kGeometries[] = {
        {"qwen_like", 256, 96, 32, 8},
        {"mixtral_like", 256, 448, 8, 2},
        {"wide_expert_few_topk", 512, 32, 4, 1},
        {"many_experts_high_topk", 128, 64, 64, 16},
        {"skewed_hidden_intermediate", 96, 320, 6, 3},
};

const int64_t kTokenCounts[] = {1, 2, 3, 5, 8, 31, 33, 64, 127};

// ---------------------------------------------------------------------------
// The packed-weight cache keys on the weight address and lives for the
// process, which is right for model parameters but wrong for test fixtures:
// one case's freed `w13` allocation is very often reused as the next case's,
// so without a flush a case would silently run against its predecessor's
// pack.  A host does the same thing on model reload -- flushing here is the
// documented contract, not a workaround.  The grouped fallback has its own
// pointer-keyed caches (custom-kernel packs, AOCL reorders), which the
// fallback-parity cases would otherwise hit with a predecessor's weights.
// ---------------------------------------------------------------------------
struct routed_moe_test : public ::testing::Test {
    void SetUp() override {
        group_matmul_routed_moe_flush_weight_cache();
        reset_grp_matmul_caches();
    }
};

struct routed_moe_isa_test : public routed_moe_test {
    void SetUp() override {
        routed_moe_test::SetUp();
        routed_moe_capability cap;
        const status_t status = group_matmul_routed_moe_query(&cap);
        if (status == status_t::isa_unsupported) {
            GTEST_SKIP() << "requires AVX-512 VNNI and BF16";
        }
        ASSERT_EQ(status, status_t::success);
    }
};

#if ZENDNNL_ROUTED_MOE_KERNELS_COMPILED
/// Run the routed gate/up epilogue activation over `n` (a multiple of 16)
/// lanes.
__attribute__((
        target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,"
               "avx512bf16,fma"))) void
routed_epilogue_act(
        routed_moe_activation_t act, const float *x, float *y, int64_t n) {
    namespace rm = zendnnl::lowoha::matmul::routed_moe;
    for (int64_t i = 0; i < n; i += 16) {
        const __m512 v = _mm512_loadu_ps(x + i);
        _mm512_storeu_ps(y + i,
                act == routed_moe_activation_t::gelu_and_mul
                        ? rm::gated_act_ps<
                                  routed_moe_activation_t::gelu_and_mul>(v)
                        : rm::gated_act_ps<
                                  routed_moe_activation_t::silu_and_mul>(v));
    }
}
#endif

/// DA8W8 descriptors for the explicit routed superset overload, built from a
/// moe_problem. `bf16_scales` exercises the routed executor's one-time
/// bf16-to-f32 scale conversion and cache.
struct da8w8_call {
    group_matmul_projection_params primary;
    group_matmul_projection_params secondary;
    group_matmul_routing_params routing;
    zendnnl::lowoha::matmul::grp_matmul_gated_act_params act;
    std::vector<float> src_scale_primary, src_scale_secondary;
    std::vector<uint16_t> s13_bf16, s2_bf16;

    da8w8_call(moe_problem &pb, bool bf16_scales = false) {
        src_scale_primary.assign(static_cast<size_t>(pb.M), 0.f);
        src_scale_secondary.assign(static_cast<size_t>(pb.M), 0.f);
        if (bf16_scales) {
            // Round the problem's scales so the scalar reference sees the
            // exact values the kernel is given.
            for (auto &v : pb.s13) {
                s13_bf16.push_back(f32_to_bf16(v));
                v = bf16_to_f32(s13_bf16.back());
            }
            for (auto &v : pb.s2) {
                s2_bf16.push_back(f32_to_bf16(v));
                v = bf16_to_f32(s2_bf16.back());
            }
        }
        const data_type_t scale_dt
                = bf16_scales ? data_type_t::bf16 : data_type_t::f32;

        const auto init = [&](group_matmul_projection_params &proj, int out,
                                  int in, const int8_t *w, const void *scale,
                                  std::vector<float> &src_scale) {
            proj.output_size = out;
            proj.input_size = in;
            proj.trans_weight = true;
            proj.weight = w;
            proj.ldb = in;
            proj.params.dtypes.src = data_type_t::bf16;
            proj.params.dtypes.wei = data_type_t::s8;
            proj.params.dtypes.dst = data_type_t::bf16;
            proj.params.dtypes.compute = data_type_t::s8;
            proj.params.dynamic_quant = true;
            proj.params.quant_params.src_scale.buff = src_scale.data();
            proj.params.quant_params.src_scale.dt = scale_dt;
            proj.params.quant_params.src_scale.dims = {pb.M, 1};
            proj.params.quant_params.wei_scale.buff = scale;
            proj.params.quant_params.wei_scale.dt = scale_dt;
            proj.params.quant_params.wei_scale.dims = {pb.E, out};
        };
        init(primary, static_cast<int>(2 * pb.N), static_cast<int>(pb.K),
                pb.w13.data(),
                bf16_scales ? static_cast<const void *>(s13_bf16.data())
                            : pb.s13.data(),
                src_scale_primary);
        init(secondary, static_cast<int>(pb.K), static_cast<int>(pb.N),
                pb.w2.data(),
                bf16_scales ? static_cast<const void *>(s2_bf16.data())
                            : pb.s2.data(),
                src_scale_secondary);

        routing.topk_ids = pb.ids.data();
        routing.topk_ids_stride = static_cast<int>(pb.topk);
        routing.topk_weights = pb.weights.data();
        routing.topk_weights_stride = static_cast<int>(pb.topk);

        act.act = pb.act == routed_moe_activation_t::gelu_and_mul
                ? zendnnl::lowoha::matmul::grp_matmul_gated_act_t::gelu_and_mul
                : zendnnl::lowoha::matmul::grp_matmul_gated_act_t::silu_and_mul;
    }

    status_t run(moe_problem &pb, uint16_t *dst) {
        return routed_fused_moe_direct('r', false, pb.src.data(),
                static_cast<int>(pb.K), static_cast<int>(pb.M),
                static_cast<int>(pb.E), static_cast<int>(pb.topk), dst,
                static_cast<int>(pb.K), primary, routing, &secondary, &act);
    }
};

/// Records whether a call reached the vector group_matmul_direct surface.
struct grouped_call_probe {
    std::atomic<bool> &capture
            = zendnnl::lowoha::matmul::test_api::s_capture_gemm_mode;
    std::atomic<const char *> &last_mode = zendnnl::lowoha::matmul::test_api::
            s_last_group_matmul_direct_gemm_mode;

    grouped_call_probe() { arm(); }
    ~grouped_call_probe() { capture.store(false, std::memory_order_relaxed); }
    void arm() {
        last_mode.store(nullptr, std::memory_order_relaxed);
        capture.store(true, std::memory_order_relaxed);
    }
    bool grouped_ran() const {
        return last_mode.load(std::memory_order_relaxed) != nullptr;
    }
};

struct disable_routed_guard {
    disable_routed_guard() {
        EXPECT_EQ(setenv("ZENDNNL_ENABLE_ROUTED_MOE", "0", 1), 0);
    }
    ~disable_routed_guard() { unsetenv("ZENDNNL_ENABLE_ROUTED_MOE"); }
};

struct RoutedMoECapability : routed_moe_test {};
struct RoutedMoEKernel : routed_moe_isa_test {};
struct RoutedMoEValidate : routed_moe_isa_test {};
struct RoutedMoEPack : routed_moe_isa_test {};
struct RoutedMoEExecute : routed_moe_isa_test {};
struct RoutedMoECache : routed_moe_isa_test {};
struct RoutedMoESuperset : routed_moe_isa_test {};
struct RoutedMoEGeneric : routed_moe_test {};

} // namespace

// ===========================================================================
// Capability query
// ===========================================================================

TEST_F(RoutedMoECapability, ReportsTheImplementedEnvelope) {
    routed_moe_capability cap;
    const status_t query_status = group_matmul_routed_moe_query(&cap);
    if (query_status == status_t::isa_unsupported) {
        EXPECT_EQ(cap.activation_mask, 0u);
        EXPECT_EQ(cap.quant_mask, 0u);
        EXPECT_EQ(cap.src_dtype_mask, 0u);
        EXPECT_EQ(cap.wei_dtype_mask, 0u);
        EXPECT_EQ(cap.scale_dtype_mask, 0u);
        return;
    }
    ASSERT_EQ(query_status, status_t::success);

    EXPECT_EQ(cap.block_m, 32);
    EXPECT_EQ(cap.block_n, 32);
    EXPECT_EQ(cap.vnni_step, 4);
    EXPECT_EQ(cap.max_kernel_rows, 8);
    EXPECT_EQ(cap.max_s4_kernel_rows, 6);
    EXPECT_EQ(cap.s4_group_size_align, 8);
    EXPECT_EQ(cap.hidden_size_align, 32);
    EXPECT_EQ(cap.intermediate_size_align, 32);

    // SiLU- and GELU-and-mul, and nothing else claimed.
    EXPECT_EQ(cap.activation_mask,
            (1u << static_cast<uint32_t>(routed_moe_activation_t::silu_and_mul))
                    | (1u << static_cast<uint32_t>(
                               routed_moe_activation_t::gelu_and_mul)));
    EXPECT_EQ(cap.activation_mask
                    & (1u << static_cast<uint32_t>(
                               routed_moe_activation_t::swiglu_oai_mul)),
            0u);
    EXPECT_NE(cap.quant_mask
                    & (1u << static_cast<uint32_t>(routed_moe_quant_t::
                                       sym_per_oc_w8a8_dynamic_per_token)),
            0u);
    EXPECT_EQ(
            cap.src_dtype_mask, 1u << static_cast<uint32_t>(data_type_t::bf16));
    EXPECT_NE(
            cap.wei_dtype_mask & (1u << static_cast<uint32_t>(data_type_t::s8)),
            0u);
    EXPECT_EQ(cap.scale_dtype_mask,
            (1u << static_cast<uint32_t>(data_type_t::f32))
                    | (1u << static_cast<uint32_t>(data_type_t::bf16)));

    EXPECT_EQ(cap.supports_expert_map, 1);
    EXPECT_EQ(cap.supports_bias, 0);
    EXPECT_EQ(cap.supports_router_weight_on_input, 0);
    EXPECT_GT(cap.max_topk, 0);
    EXPECT_GT(cap.max_local_experts, 0);
}

TEST_F(RoutedMoECapability, RejectsNull) {
    EXPECT_EQ(group_matmul_routed_moe_query(nullptr), status_t::op_bad_io);
}

TEST(RoutedMoEApi, ProjectionWeightCapacityIsIndependentMetadata) {
    group_matmul_projection_params primary;
    group_matmul_projection_params secondary;

    EXPECT_EQ(primary.wei_buffer_capacity_bytes, 0u);
    EXPECT_EQ(secondary.wei_buffer_capacity_bytes, 0u);

    primary.wei_buffer_capacity_bytes = 4096;
    secondary.wei_buffer_capacity_bytes = 8192;
    EXPECT_EQ(primary.wei_buffer_capacity_bytes, 4096u);
    EXPECT_EQ(secondary.wei_buffer_capacity_bytes, 8192u);
}

// ===========================================================================
// Epilogue activation
// ===========================================================================

#if ZENDNNL_ROUTED_MOE_KERNELS_COMPILED
// The routed GELU must be the grouped path's GELU, bit for bit, so a call
// that falls back to group_matmul computes the same activation.  Compared
// against the f32 separate-pass helper with up == 1, which is exact.
TEST_F(RoutedMoEKernel, GeluIsBitIdenticalToGroupedPostPass) {
    constexpr int64_t dim = 4096;
    std::vector<float> x(static_cast<size_t>(dim));
    for (int64_t i = 0; i < dim; ++i) {
        x[i] = -12.f + 24.f * static_cast<float>(i) / (dim - 1);
    }
    x[0] = 0.f;
    x[1] = -0.f;
    x[2] = 1e-30f;
    x[3] = -1e-30f;
    x[4] = 60.f;
    x[5] = -60.f;

    std::vector<float> routed(static_cast<size_t>(dim));
    routed_epilogue_act(routed_moe_activation_t::gelu_and_mul, x.data(),
            routed.data(), dim);

    std::vector<float> row(static_cast<size_t>(2 * dim), 1.f);
    std::copy(x.begin(), x.end(), row.begin());
    zendnnl::lowoha::matmul::apply_gated_act_inplace(
            zendnnl::lowoha::matmul::grp_matmul_gated_act_t::gelu_and_mul,
            row.data(), 0, 1, static_cast<int>(2 * dim),
            static_cast<int>(2 * dim), data_type_t::f32);

    for (int64_t i = 0; i < dim; ++i) {
        uint32_t a, b;
        std::memcpy(&a, &routed[i], sizeof(a));
        std::memcpy(&b, &row[i], sizeof(b));
        EXPECT_EQ(a, b) << "x=" << x[i];
    }
}

// Accuracy of the shared erf-GELU against a double-precision reference.
// A&S 7.1.26 contributes <= 1.5e-7 to erf and the polynomial exp ~5e-5
// relative on the exp(-x^2) term, so |gelu error| <= 0.5 |x| * 5e-5.  The bound
// below leaves 2x margin on that and is ~80x below one bf16 ulp at |x| = 1.
TEST_F(RoutedMoEKernel, GeluTracksErfReference) {
    constexpr int64_t n = 16 * 256;
    std::vector<float> x(static_cast<size_t>(n)), y(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        x[i] = -10.f + 20.f * static_cast<float>(i) / (n - 1);
    }
    routed_epilogue_act(
            routed_moe_activation_t::gelu_and_mul, x.data(), y.data(), n);

    double worst = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double xd = x[i];
        const double want = 0.5 * xd * (1.0 + std::erf(xd / std::sqrt(2.0)));
        const double err = std::fabs(y[i] - want);
        const double bound = 5e-5 * std::max(1.0, std::fabs(xd));
        EXPECT_LE(err, bound) << "x=" << x[i];
        worst = std::max(worst, err / std::max(1.0, std::fabs(xd)));
    }
    RecordProperty("worst_scaled_abs_error", std::to_string(worst));

    // Saturation and propagation: gelu(x) -> x for large x, -> 0 for large
    // negative x, and a NaN gate stays NaN.
    alignas(64) float edge[16] = {60.f, -60.f, 0.f, -0.f,
            std::numeric_limits<float>::quiet_NaN(), 1e-30f, -1e-30f, 8.f, -8.f,
            3.f, -3.f, 0.5f, -0.5f, 1.f, -1.f, 2.f};
    alignas(64) float out[16];
    routed_epilogue_act(routed_moe_activation_t::gelu_and_mul, edge, out, 16);
    EXPECT_EQ(out[0], 60.f);
    EXPECT_EQ(out[1], 0.f);
    EXPECT_EQ(out[2], 0.f);
    EXPECT_TRUE(std::isnan(out[4]));
    for (int i = 0; i < 16; ++i) {
        if (i != 4) { EXPECT_TRUE(std::isfinite(out[i])) << "x=" << edge[i]; }
    }
}

TEST_F(RoutedMoEKernel, SiluEpilogueIsUnchanged) {
    // The SiLU epilogue keeps its own exp/rcp14 form rather than the shared
    // sigmoid; the template parameter must still select it.
    constexpr int64_t n = 16 * 64;
    std::vector<float> x(static_cast<size_t>(n)), y(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        x[i] = -10.f + 20.f * static_cast<float>(i) / (n - 1);
    }
    routed_epilogue_act(
            routed_moe_activation_t::silu_and_mul, x.data(), y.data(), n);
    for (int64_t i = 0; i < n; ++i) {
        const double xd = x[i];
        const double want = xd / (1.0 + std::exp(-xd));
        EXPECT_NEAR(y[i], want, 2e-3 * std::max(1.0, std::fabs(xd)))
                << "x=" << x[i];
    }
}
#endif

// ===========================================================================
// Validation: accepted geometries
// ===========================================================================

TEST_F(RoutedMoEValidate, AcceptsEveryGeometryAndTokenCount) {
    for (const auto &g : kGeometries) {
        for (int64_t M : kTokenCounts) {
            moe_problem pb;
            pb.build(M, g.K, g.N, g.E, g.topk, 7u);
            const auto p = pb.params();
            EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::success)
                    << g.name << " M=" << M;
        }
    }
}

TEST_F(RoutedMoEValidate, AcceptsPaddedActivationAndRoutingRows) {
    // Row-padded src/dst and routing rows are legal: only the innermost
    // element must be unit-stride.
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 11u);

    std::vector<uint16_t> padded_src(static_cast<size_t>(8 * 160), 0);
    auto p = pb.params();
    p.src = padded_src.data();
    p.src_stride = 160;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::success);

    std::vector<int32_t> padded_ids(static_cast<size_t>(8 * 4), 0);
    p = pb.params();
    p.topk_ids = padded_ids.data();
    p.topk_ids_stride = 4; // > topk, so the trailing entries are ignored
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::success);
}

// ===========================================================================
// Validation: rejections
// ===========================================================================

TEST_F(RoutedMoEValidate, RejectsUnsupportedActivation) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 3u);

    // Named-but-unimplemented gated activations are reported as
    // `unimplemented`, never silently run as SiLU or GELU.
    auto p = pb.params();
    p.activation = routed_moe_activation_t::swiglu_oai_mul;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::unimplemented);

    // Out-of-range enumerator values are not a supported activation either.
    p = pb.params();
    p.activation = static_cast<routed_moe_activation_t>(7);
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::unimplemented);

    // An unset activation is malformed input rather than a missing feature.
    p = pb.params();
    p.activation = routed_moe_activation_t::undef;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::op_bad_io);
}

TEST_F(RoutedMoEValidate, AcceptsGeluAndMulOnEveryGeometry) {
    for (const auto &g : kGeometries) {
        moe_problem pb;
        pb.build(5, g.K, g.N, g.E, g.topk, 11u);
        pb.act = routed_moe_activation_t::gelu_and_mul;
        EXPECT_EQ(group_matmul_routed_moe_validate(pb.params()),
                status_t::success)
                << g.name;
    }
}

TEST_F(RoutedMoEValidate, GeluKeepsEveryNonActivationRejection) {
    // Supporting a second activation must not relax any other rule.
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 3u);
    pb.act = routed_moe_activation_t::gelu_and_mul;
    std::vector<float> bias(static_cast<size_t>(8 * 128), 0.f);

    auto p = pb.params();
    p.gate_up_bias = bias.data();
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::unimplemented);

    p = pb.params();
    p.apply_router_weight_on_input = 1;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::unimplemented);

    p = pb.params();
    p.gate_up_scale_stride_expert = 2 * pb.N + 1;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_stride);

    p = pb.params();
    p.intermediate_size = 48; // not a multiple of block_n
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_size);

    p = pb.params();
    p.wei_dt = data_type_t::s4;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_quant);

    p = pb.params();
    p.topk_ids = nullptr;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::op_bad_io);
}

TEST_F(RoutedMoEValidate, RejectsUnsupportedQuantization) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 3u);

    for (auto q : {routed_moe_quant_t::sym_per_group_w8a8_dynamic_per_token,
                 routed_moe_quant_t::asym_per_oc_w8a8_dynamic_per_token,
                 routed_moe_quant_t::sym_per_oc_w4a8_dynamic_per_token}) {
        auto p = pb.params();
        p.quant_scheme = q;
        EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::unimplemented);
    }

    auto p = pb.params();
    p.quant_scheme = routed_moe_quant_t::undef;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::op_bad_io);

    // 4-bit weights under an 8-bit scheme is an inconsistent request.
    p = pb.params();
    p.wei_dt = data_type_t::s4;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_quant);
}

TEST_F(RoutedMoEValidate, RejectsBiasAndRouterWeightOnInput) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 3u);
    std::vector<float> bias(static_cast<size_t>(8 * 128), 0.f);

    auto p = pb.params();
    p.gate_up_bias = bias.data();
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::unimplemented);

    p = pb.params();
    p.down_bias = bias.data();
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::unimplemented);

    p = pb.params();
    p.bias_dt = data_type_t::f32;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::unimplemented);

    p = pb.params();
    p.apply_router_weight_on_input = 1;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::unimplemented);
}

TEST_F(RoutedMoEValidate, RejectsUnsupportedDataTypes) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 3u);

    auto p = pb.params();
    p.src_dt = data_type_t::f32;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::unimplemented);

    p = pb.params();
    p.dst_dt = data_type_t::f16;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::unimplemented);

    p = pb.params();
    p.scale_dt = data_type_t::f16;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::unimplemented);
}

TEST_F(RoutedMoEValidate, RejectsMisalignedDimensions) {
    // hidden and intermediate are both reduction *and* output-channel extents,
    // so both must be a multiple of the 32-wide tile.
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 3u);

    auto p = pb.params();
    p.hidden_size = 120;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_size);

    p = pb.params();
    p.intermediate_size = 48;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_size);

    p = pb.params();
    p.hidden_size = 0;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::op_bad_io);

    p = pb.params();
    p.topk = 0;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::op_bad_io);
}

TEST_F(RoutedMoEValidate, RejectsBadStrides) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 3u);

    auto p = pb.params();
    p.src_stride = 64; // < hidden_size
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_stride);

    p = pb.params();
    p.dst_stride = 1;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_stride);

    p = pb.params();
    p.topk_ids_stride = 1; // < topk
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_stride);

    // Padded weight rows would interleave foreign bytes into a packed block.
    p = pb.params();
    p.gate_up_stride_oc = 160;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_stride);

    p = pb.params();
    p.down_stride_expert = 99;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_stride);
}

TEST_F(RoutedMoEValidate, AcceptsTrailingExpertWeightPadding) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 3u);

    auto p = pb.params();
    p.gate_up_stride_expert = 2 * pb.N * pb.K + 64;
    p.down_stride_expert = pb.K * pb.N + 64;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::success);
}

TEST_F(RoutedMoEValidate, RejectsDerivedCountAndAddressOverflows) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 3u);

    const auto expect_size_rejection = [&](const routed_moe_params &p) {
        EXPECT_EQ(
                group_matmul_routed_moe_validate(p), status_t::memory_bad_size);
        EXPECT_EQ(execute(p), status_t::memory_bad_size);
    };

    auto p = pb.params();
    p.num_tokens = std::numeric_limits<int64_t>::max();
    p.topk = 2; // M * topk overflows int64.
    expect_size_rejection(p);

    p = pb.params();
    p.num_tokens = std::numeric_limits<int32_t>::max();
    p.topk = 1; // Flat ids fit, but expert padding no longer fits int32.
    expect_size_rejection(p);

    p = pb.params();
    p.num_local_experts = std::numeric_limits<int32_t>::max();
    expect_size_rejection(p);

    p = pb.params();
    p.hidden_size = 65824; // First aligned K beyond safe vpdpbusd capacity.
    expect_size_rejection(p);

    p = pb.params();
    p.num_tokens = 2;
    p.src_stride = std::numeric_limits<int64_t>::max();
    expect_size_rejection(p);

    p = pb.params();
    p.gate_up_stride_expert = std::numeric_limits<int64_t>::max();
    expect_size_rejection(p);

    p = pb.params();
    p.num_tokens = 2;
    p.topk_ids_stride = std::numeric_limits<int64_t>::max();
    expect_size_rejection(p);
}

TEST_F(RoutedMoEValidate, RejectsNullBuffers) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 3u);

    auto p = pb.params();
    p.src = nullptr;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::op_bad_io);

    p = pb.params();
    p.gate_up_scale = nullptr;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::op_bad_io);

    p = pb.params();
    p.topk_ids = nullptr;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::op_bad_io);
}

TEST_F(RoutedMoEValidate, RejectsOutOfRangeRoutingIds) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 3u);

    pb.ids[5] = static_cast<int32_t>(pb.E); // one past the last local expert
    auto p = pb.params();
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_index);

    pb.ids[5] = -1;
    EXPECT_EQ(group_matmul_routed_moe_validate(pb.params()),
            status_t::memory_bad_index);

    // Without an expert map the ids must already be local, so an id beyond the
    // local range is rejected even if it would be a valid global id.
    pb.ids[5] = static_cast<int32_t>(pb.E) + 100;
    EXPECT_EQ(group_matmul_routed_moe_validate(pb.params()),
            status_t::memory_bad_index);
}

TEST_F(RoutedMoEValidate, RejectsInconsistentExpertMap) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 3u);
    std::vector<int32_t> map(16, 0);

    auto p = pb.params();
    p.expert_map = map.data();
    p.expert_map_size = 0; // map given but no size
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::op_bad_io);

    p = pb.params();
    p.expert_map_size = 16; // size given but no map
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::op_bad_io);

    // A map entry pointing past the local expert range is a caller bug, unlike
    // a negative entry which legitimately masks the slot out.
    map[0] = static_cast<int32_t>(pb.E);
    p = pb.params();
    p.expert_map = map.data();
    p.expert_map_size = 16;
    pb.ids[0] = 0;
    p.topk_ids = pb.ids.data();
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_index);
}

// ===========================================================================
// Packing
// ===========================================================================

TEST_F(RoutedMoEPack, MatchesReferenceLayoutByteForByte) {
    struct {
        int64_t E, OC, IC;
    } cases[] = {
            {1, 32, 32}, {2, 64, 128}, {3, 96, 256},
            {8, 896, 256}, // mixtral-like gate/up block
            {4, 128, 96}, // IC not a power of two
    };

    for (const auto &c : cases) {
        std::mt19937 rng(0x51ee);
        std::uniform_int_distribution<int> q(-128, 127);
        std::vector<int8_t> src(static_cast<size_t>(c.E * c.OC * c.IC));
        for (auto &v : src) {
            v = static_cast<int8_t>(q(rng));
        }

        const int64_t row = group_matmul_routed_moe_packed_row_bytes(
                c.IC, data_type_t::s8);
        ASSERT_EQ(row, c.IC + 4);

        std::vector<int8_t> got(static_cast<size_t>(c.E * c.OC * row), 0);
        ASSERT_EQ(group_matmul_routed_moe_pack_weights(src.data(), got.data(),
                          c.E, c.OC, c.IC, data_type_t::s8, 0),
                status_t::success);

        const auto want = reference_pack(src.data(), c.E, c.OC, c.IC);
        ASSERT_EQ(got.size(), want.size());
        EXPECT_EQ(std::memcmp(got.data(), want.data(), got.size()), 0)
                << "E=" << c.E << " OC=" << c.OC << " IC=" << c.IC;
    }
}

TEST_F(RoutedMoEPack, RejectsUnsupportedShapesAndTypes) {
    std::vector<int8_t> src(1024, 0), dst(4096, 0);

    // 4-bit weights are not packable by this layout.
    EXPECT_EQ(group_matmul_routed_moe_pack_weights(
                      src.data(), dst.data(), 1, 32, 32, data_type_t::s4, 0),
            status_t::unimplemented);
    EXPECT_EQ(group_matmul_routed_moe_packed_row_bytes(32, data_type_t::s4), 0);

    // Output channels must tile by 32, input width by the VNNI step.
    EXPECT_EQ(group_matmul_routed_moe_pack_weights(
                      src.data(), dst.data(), 1, 16, 32, data_type_t::s8, 0),
            status_t::memory_bad_size);
    EXPECT_EQ(group_matmul_routed_moe_pack_weights(
                      src.data(), dst.data(), 1, 32, 30, data_type_t::s8, 0),
            status_t::memory_bad_size);
    EXPECT_EQ(group_matmul_routed_moe_packed_row_bytes(30, data_type_t::s8), 0);

    EXPECT_EQ(group_matmul_routed_moe_pack_weights(
                      nullptr, dst.data(), 1, 32, 32, data_type_t::s8, 0),
            status_t::op_bad_io);

    // Products are rejected before either tiny test buffer is touched.
    EXPECT_EQ(group_matmul_routed_moe_pack_weights(src.data(), dst.data(),
                      std::numeric_limits<int64_t>::max(), 32, 32,
                      data_type_t::s8, 1),
            status_t::memory_bad_size);
    EXPECT_EQ(group_matmul_routed_moe_pack_weights(src.data(), dst.data(), 1,
                      std::numeric_limits<int64_t>::max() - 31, 32,
                      data_type_t::s8, 1),
            status_t::memory_bad_size);
    EXPECT_EQ(group_matmul_routed_moe_packed_row_bytes(
                      std::numeric_limits<int64_t>::max(), data_type_t::s8),
            0);
}

// ===========================================================================
// Execution
// ===========================================================================

TEST_F(RoutedMoEExecute, MatchesReferenceAcrossGeometriesAndTokenCounts) {
    for (const auto &g : kGeometries) {
        for (int64_t M : kTokenCounts) {
            moe_problem pb;
            pb.build(M, g.K, g.N, g.E, g.topk, 0x1234u);
            // Each case allocates fresh weights that may land on a freed
            // predecessor's address, so drop the cache between cases.
            group_matmul_routed_moe_flush_weight_cache();

            auto p = pb.params();
            ASSERT_EQ(execute(p), status_t::success) << g.name << " M=" << M;

            const auto want = reference_moe(pb);
            // The only divergence from the reference is the polynomial exp and
            // the rcp14 reciprocal inside SiLU.
            EXPECT_LT(rel_mae(pb.dst, want), 3e-3) << g.name << " M=" << M;
        }
    }
}

// GELU replaces SiLU only in the epilogue, so the same pipeline tolerance
// applies: the reference uses exact std::erf, the kernel the A&S 7.1.26 erf
// whose error (< 1e-4 absolute on gelu, see RoutedMoEKernel) is far below the
// bf16 rounding of the intermediate that dominates rel_mae.
TEST_F(RoutedMoEExecute, GeluMatchesReferenceAcrossGeometriesAndTokenCounts) {
    for (const auto &g : kGeometries) {
        for (int64_t M : kTokenCounts) {
            moe_problem pb;
            pb.build(M, g.K, g.N, g.E, g.topk, 0x4321u);
            pb.act = routed_moe_activation_t::gelu_and_mul;
            pb.unit_gate();
            group_matmul_routed_moe_flush_weight_cache();

            ASSERT_EQ(execute(pb.params()), status_t::success)
                    << g.name << " M=" << M;
            EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3)
                    << g.name << " M=" << M;
        }
    }
}

// Gemma-4-26B-A4B MoE geometry (hidden 2816, intermediate 704, topk 8) at a
// decode batch, a full BS32 decode step and a prompt chunk with a partial
// routing block.  The expert count is reduced from 128 to 16 to keep the
// weights at ~95 MB; it only changes how many routing blocks exist.
TEST_F(RoutedMoEExecute, GeluGemmaShapedDecodeAndPrompt) {
    constexpr int64_t K = 2816, N = 704, E = 16, topk = 8;
    for (int64_t M : {int64_t {1}, int64_t {32}, int64_t {77}}) {
        moe_problem pb;
        pb.build(M, K, N, E, topk, 0x6e44u + static_cast<uint32_t>(M));
        pb.act = routed_moe_activation_t::gelu_and_mul;
        pb.unit_gate();
        group_matmul_routed_moe_flush_weight_cache();

        auto p = pb.params();
        ASSERT_EQ(execute(p), status_t::success) << "M=" << M;
        EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3) << "M=" << M;

        // The tile schedule depends on the team width; the result must not.
        const auto full_team = pb.dst;
        std::fill(pb.dst.begin(), pb.dst.end(), 0);
        p.num_threads = 1;
        ASSERT_EQ(execute(p), status_t::success) << "M=" << M;
        EXPECT_EQ(pb.dst, full_team) << "M=" << M;
    }
}

TEST_F(RoutedMoEExecute, GeluAndSiluShareOnePackAndDiffer) {
    // The two activations read the same packed W13; switching between them
    // on one weight must neither repack into a different layout nor leak the
    // other epilogue.
    moe_problem pb;
    pb.build(21, 256, 96, 32, 8, 0x5151u);
    pb.unit_gate();
    ASSERT_EQ(execute(pb.params()), status_t::success);
    const auto silu = pb.dst;
    EXPECT_LT(rel_mae(silu, reference_moe(pb)), 3e-3);

    pb.act = routed_moe_activation_t::gelu_and_mul;
    std::fill(pb.dst.begin(), pb.dst.end(), 0);
    ASSERT_EQ(execute(pb.params()), status_t::success);
    EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
    EXPECT_GT(rel_mae(pb.dst, silu), 5e-2);

    pb.act = routed_moe_activation_t::silu_and_mul;
    std::fill(pb.dst.begin(), pb.dst.end(), 0);
    ASSERT_EQ(execute(pb.params()), status_t::success);
    EXPECT_EQ(pb.dst, silu);
}

TEST_F(RoutedMoEExecute, GeluHonorsExpertMapAndPaddedStrides) {
    moe_problem pb;
    pb.build(13, 128, 64, 8, 4, 0x7171u);
    pb.act = routed_moe_activation_t::gelu_and_mul;
    pb.unit_gate();

    std::vector<int32_t> map(16);
    for (int i = 0; i < 16; ++i) {
        map[i] = (i % 2 == 0) ? i / 2 : -1;
    }
    std::mt19937 rng(9);
    std::uniform_int_distribution<int> pick(0, 15);
    for (auto &v : pb.ids) {
        v = pick(rng);
    }

    const int64_t dst_stride = pb.K + 13;
    std::vector<uint16_t> dst(static_cast<size_t>(pb.M * dst_stride), 0x5a5a);
    auto p = pb.params();
    p.expert_map = map.data();
    p.expert_map_size = static_cast<int64_t>(map.size());
    p.dst = dst.data();
    p.dst_stride = dst_stride;
    ASSERT_EQ(execute(p), status_t::success);

    std::vector<uint16_t> got(static_cast<size_t>(pb.M * pb.K));
    for (int64_t m = 0; m < pb.M; ++m) {
        std::memcpy(got.data() + m * pb.K, dst.data() + m * dst_stride,
                static_cast<size_t>(pb.K) * sizeof(uint16_t));
        for (int64_t k = pb.K; k < dst_stride; ++k) {
            EXPECT_EQ(dst[m * dst_stride + k], 0x5a5a);
        }
    }
    EXPECT_LT(rel_mae(got, reference_moe(pb, map.data())), 3e-3);
}

TEST_F(RoutedMoEExecute, IsDeterministicAcrossRepeatedCalls) {
    moe_problem pb;
    pb.build(37, 256, 96, 32, 8, 0x99u);
    auto p = pb.params();

    ASSERT_EQ(execute(p), status_t::success);
    const auto first = pb.dst;

    for (int i = 0; i < 3; ++i) {
        std::fill(pb.dst.begin(), pb.dst.end(), 0);
        ASSERT_EQ(execute(p), status_t::success);
        EXPECT_EQ(std::memcmp(pb.dst.data(), first.data(),
                          first.size() * sizeof(uint16_t)),
                0)
                << "repeat " << i;
    }
}

TEST_F(RoutedMoEExecute, MatchesAcrossOneTwoAndRuntimeMaxThreads) {
    moe_problem pb;
    pb.build(33, 128, 64, 8, 3, 0x31u);
    const auto want = reference_moe(pb);

    std::vector<int> thread_counts = {1, 2, std::max(1, omp_get_max_threads())};
    std::sort(thread_counts.begin(), thread_counts.end());
    thread_counts.erase(std::unique(thread_counts.begin(), thread_counts.end()),
            thread_counts.end());

    std::vector<uint16_t> first;
    for (int threads : thread_counts) {
        std::fill(pb.dst.begin(), pb.dst.end(), 0);
        auto p = pb.params();
        p.num_threads = threads;
        ASSERT_EQ(execute(p), status_t::success) << "threads=" << threads;
        EXPECT_LT(rel_mae(pb.dst, want), 3e-3) << "threads=" << threads;
        if (first.empty()) {
            first = pb.dst;
        } else {
            EXPECT_EQ(pb.dst, first) << "threads=" << threads;
        }
    }
}

TEST_F(RoutedMoEExecute, HonorsPaddedActivationAndRoutingStrides) {
    moe_problem pb;
    pb.build(9, 128, 64, 8, 3, 0x41u);

    const int64_t src_stride = pb.K + 17;
    const int64_t dst_stride = pb.K + 11;
    const int64_t ids_stride = pb.topk + 2;
    const int64_t weights_stride = pb.topk + 3;
    std::vector<uint16_t> src(static_cast<size_t>(pb.M * src_stride), 0x7f7f);
    std::vector<uint16_t> dst(static_cast<size_t>(pb.M * dst_stride), 0x5a5a);
    std::vector<int32_t> ids(static_cast<size_t>(pb.M * ids_stride), -777);
    std::vector<float> weights(
            static_cast<size_t>(pb.M * weights_stride), -777.f);
    for (int64_t m = 0; m < pb.M; ++m) {
        std::memcpy(src.data() + m * src_stride, pb.src.data() + m * pb.K,
                static_cast<size_t>(pb.K) * sizeof(uint16_t));
        std::memcpy(ids.data() + m * ids_stride, pb.ids.data() + m * pb.topk,
                static_cast<size_t>(pb.topk) * sizeof(int32_t));
        std::memcpy(weights.data() + m * weights_stride,
                pb.weights.data() + m * pb.topk,
                static_cast<size_t>(pb.topk) * sizeof(float));
    }

    auto p = pb.params();
    p.src = src.data();
    p.src_stride = src_stride;
    p.dst = dst.data();
    p.dst_stride = dst_stride;
    p.topk_ids = ids.data();
    p.topk_ids_stride = ids_stride;
    p.topk_weights = weights.data();
    p.topk_weights_stride = weights_stride;
    ASSERT_EQ(execute(p), status_t::success);

    std::vector<uint16_t> got(static_cast<size_t>(pb.M * pb.K));
    for (int64_t m = 0; m < pb.M; ++m) {
        std::memcpy(got.data() + m * pb.K, dst.data() + m * dst_stride,
                static_cast<size_t>(pb.K) * sizeof(uint16_t));
        for (int64_t k = pb.K; k < dst_stride; ++k) {
            EXPECT_EQ(dst[m * dst_stride + k], 0x5a5a);
        }
    }
    EXPECT_LT(rel_mae(got, reference_moe(pb)), 3e-3);
}

TEST_F(RoutedMoEExecute, HandlesEveryTokenRoutedToOneExpert) {
    // Degenerate routing: a single expert owns every slot, so the sorted order
    // is one long run and the block tail is the only partial block.
    moe_problem pb;
    pb.build(19, 128, 64, 8, 2, 0x2au);
    std::fill(pb.ids.begin(), pb.ids.end(), 3);

    auto p = pb.params();
    ASSERT_EQ(execute(p), status_t::success);
    EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
}

TEST_F(RoutedMoEExecute, ExpertMapMasksNonResidentSlots) {
    // Expert-parallel shard: global ids 0..15, only 8 experts resident.  A
    // negative map entry must contribute nothing rather than aliasing expert 0.
    moe_problem pb;
    pb.build(16, 128, 64, 8, 4, 0x77u);

    std::vector<int32_t> map(16);
    for (int i = 0; i < 16; ++i) {
        map[i] = (i % 2 == 0) ? i / 2 : -1;
    }
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> pick(0, 15);
    for (auto &v : pb.ids) {
        v = pick(rng);
    }

    auto p = pb.params();
    p.expert_map = map.data();
    p.expert_map_size = static_cast<int64_t>(map.size());
    ASSERT_EQ(group_matmul_routed_moe_validate(p), status_t::success);
    ASSERT_EQ(execute(p), status_t::success);

    EXPECT_LT(rel_mae(pb.dst, reference_moe(pb, map.data())), 3e-3);
}

TEST_F(RoutedMoEExecute, ProducesZeroWhenNoSlotIsResident) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0x81u);
    const std::vector<int32_t> map(8, -1);
    std::fill(pb.dst.begin(), pb.dst.end(), 0xffff);

    auto p = pb.params();
    p.expert_map = map.data();
    p.expert_map_size = static_cast<int64_t>(map.size());
    ASSERT_EQ(execute(p), status_t::success);

    for (uint16_t v : pb.dst) {
        EXPECT_EQ(bf16_to_f32(v), 0.f);
    }
}

TEST_F(RoutedMoEExecute, RejectsInvalidExpertMapIdsAndEntries) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0x91u);
    std::vector<int32_t> map(16, 0);
    for (int i = 0; i < 8; ++i) {
        map[static_cast<size_t>(i)] = i;
    }

    auto p = pb.params();
    p.expert_map = map.data();
    p.expert_map_size = static_cast<int64_t>(map.size());

    pb.ids[0] = -1;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_index);
    EXPECT_EQ(execute(p), status_t::memory_bad_index);

    pb.ids[0] = static_cast<int32_t>(map.size());
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_index);
    EXPECT_EQ(execute(p), status_t::memory_bad_index);

    pb.ids[0] = 0;
    map[0] = static_cast<int32_t>(pb.E);
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_index);
    EXPECT_EQ(execute(p), status_t::memory_bad_index);
}

TEST_F(RoutedMoEExecute, ForwardsValidationRejections) {
    moe_problem pb;
    pb.build(8, 120, 64, 8, 2, 0x12u);
    auto p = pb.params();
    // Execution must reject exactly what validation rejects, so a caller that
    // skipped the gate still cannot run an unsupported problem.
    EXPECT_EQ(execute(p), status_t::memory_bad_size);

    pb.build(8, 128, 64, 8, 2, 0x12u);
    p = pb.params();
    p.activation = routed_moe_activation_t::swiglu_oai_mul;
    EXPECT_EQ(execute(p), status_t::unimplemented);

    pb.ids[0] = static_cast<int32_t>(pb.E);
    p = pb.params();
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_index);
    EXPECT_EQ(execute(p), status_t::memory_bad_index);

    pb.ids[0] = -1;
    p = pb.params();
    std::fill(pb.dst.begin(), pb.dst.end(), 0x6b6b);
    const auto untouched = pb.dst;
    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::memory_bad_index);
    EXPECT_EQ(execute(p), status_t::memory_bad_index);
    EXPECT_EQ(pb.dst, untouched);
}

TEST_F(RoutedMoEExecute, MirrorsValidationForActivationQuantAndBias) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0x22u);
    std::vector<float> bias(static_cast<size_t>(2 * pb.N), 0.f);

    const auto expect_both = [&](routed_moe_params p, status_t expected) {
        EXPECT_EQ(group_matmul_routed_moe_validate(p), expected);
        EXPECT_EQ(execute(p), expected);
    };

    auto p = pb.params();
    p.activation = routed_moe_activation_t::swiglu_oai_mul;
    expect_both(p, status_t::unimplemented);

    p = pb.params();
    p.activation = routed_moe_activation_t::gelu_and_mul;
    p.gate_up_bias = bias.data();
    expect_both(p, status_t::unimplemented);

    p = pb.params();
    p.quant_scheme = routed_moe_quant_t::sym_per_group_w8a8_dynamic_per_token;
    expect_both(p, status_t::unimplemented);

    p = pb.params();
    p.gate_up_bias = bias.data();
    expect_both(p, status_t::unimplemented);

    p = pb.params();
    p.down_bias = bias.data();
    expect_both(p, status_t::unimplemented);

    p = pb.params();
    p.bias_dt = data_type_t::f32;
    expect_both(p, status_t::unimplemented);

    p = pb.params();
    p.apply_router_weight_on_input = 1;
    expect_both(p, status_t::unimplemented);
}

// ===========================================================================
// Packed-weight cache
// ===========================================================================

TEST_F(RoutedMoECache, CompositeKeySeparatesTensorRoleAndGeometry) {
    int shared_identity = 0;

    moe_problem first;
    first.build(11, 128, 64, 4, 2, 0x101u);
    auto first_params = first.params();
    // Deliberately use one explicit identity for both projections.
    first_params.gate_up_cache_key = &shared_identity;
    first_params.down_cache_key = &shared_identity;
    ASSERT_EQ(execute(first_params), status_t::success);
    EXPECT_LT(rel_mae(first.dst, reference_moe(first)), 3e-3);

    moe_problem second;
    second.build(7, 256, 96, 6, 3, 0x202u);
    auto second_params = second.params();
    // Reuse that identity under a different E/OC/K geometry as well.
    second_params.gate_up_cache_key = &shared_identity;
    second_params.down_cache_key = &shared_identity;
    ASSERT_EQ(execute(second_params), status_t::success);
    EXPECT_LT(rel_mae(second.dst, reference_moe(second)), 3e-3);

    std::fill(first.dst.begin(), first.dst.end(), 0);
    ASSERT_EQ(execute(first_params), status_t::success);
    EXPECT_LT(rel_mae(first.dst, reference_moe(first)), 3e-3);
}

TEST_F(RoutedMoECache, ConcurrentPackAndHitAreThreadSafe) {
    moe_problem pb;
    pb.build(31, 128, 64, 8, 3, 0x303u);
    std::vector<uint16_t> second_dst(pb.dst.size(), 0);

    auto first_params = pb.params();
    auto second_params = first_params;
    second_params.dst = second_dst.data();

    status_t first_status = status_t::failure;
    status_t second_status = status_t::failure;
    std::thread first_thread([&]() { first_status = execute(first_params); });
    std::thread second_thread(
            [&]() { second_status = execute(second_params); });
    first_thread.join();
    second_thread.join();

    ASSERT_EQ(first_status, status_t::success);
    ASSERT_EQ(second_status, status_t::success);
    const auto want = reference_moe(pb);
    EXPECT_LT(rel_mae(pb.dst, want), 3e-3);
    EXPECT_LT(rel_mae(second_dst, want), 3e-3);
    EXPECT_EQ(pb.dst, second_dst);
}

TEST_F(RoutedMoECache, FlushForcesRepackWithIdenticalResults) {
    moe_problem pb;
    pb.build(24, 128, 64, 8, 2, 0xa5u);
    auto p = pb.params();

    ASSERT_EQ(execute(p), status_t::success);
    const auto cached = pb.dst;

    group_matmul_routed_moe_flush_weight_cache();
    std::fill(pb.dst.begin(), pb.dst.end(), 0);
    ASSERT_EQ(execute(p), status_t::success);
    EXPECT_EQ(std::memcmp(pb.dst.data(), cached.data(),
                      cached.size() * sizeof(uint16_t)),
            0);
}

TEST_F(RoutedMoECache, FlushIsIdempotentAndSafeWhenEmpty) {
    group_matmul_routed_moe_flush_weight_cache();
    group_matmul_routed_moe_flush_weight_cache();

    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0xb6u);
    auto p = pb.params();
    EXPECT_EQ(execute(p), status_t::success);
}

TEST_F(RoutedMoECache, RepacksAfterWeightsChangeUnderTheSameAddress) {
    // A model reload can hand back the same allocation with different bytes.
    // Without the flush the cache would legitimately keep serving the old
    // pack, so this asserts the flush is what makes reload safe.
    moe_problem pb;
    pb.build(16, 128, 64, 8, 2, 0xc7u);
    auto p = pb.params();
    ASSERT_EQ(execute(p), status_t::success);

    std::mt19937 rng(0xd8);
    std::uniform_int_distribution<int> q(-127, 127);
    for (auto &v : pb.w13) {
        v = static_cast<int8_t>(q(rng));
    }
    for (auto &v : pb.w2) {
        v = static_cast<int8_t>(q(rng));
    }

    group_matmul_routed_moe_flush_weight_cache();
    std::fill(pb.dst.begin(), pb.dst.end(), 0);
    ASSERT_EQ(execute(p), status_t::success);
    EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
}

// ===========================================================================
// Explicit routed superset overload
// ===========================================================================

TEST_F(RoutedMoESuperset, DA8W8DecodeUsesExplicitContract) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0x91u);
    std::vector<float> src_scale_primary(static_cast<size_t>(pb.M), 0.0f);
    std::vector<float> src_scale_secondary(static_cast<size_t>(pb.M), 0.0f);

    group_matmul_projection_params primary;
    primary.output_size = static_cast<int>(2 * pb.N);
    primary.input_size = static_cast<int>(pb.K);
    primary.trans_weight = true;
    primary.weight = pb.w13.data();
    primary.ldb = static_cast<int>(pb.K);
    primary.params.dtypes.src = data_type_t::bf16;
    primary.params.dtypes.wei = data_type_t::s8;
    primary.params.dtypes.dst = data_type_t::bf16;
    primary.params.dtypes.compute = data_type_t::s8;
    primary.params.dynamic_quant = true;
    primary.params.quant_params.src_scale.buff = src_scale_primary.data();
    primary.params.quant_params.src_scale.dt = data_type_t::f32;
    primary.params.quant_params.src_scale.dims = {pb.M, 1};
    primary.params.quant_params.wei_scale.buff = pb.s13.data();
    primary.params.quant_params.wei_scale.dt = data_type_t::f32;
    primary.params.quant_params.wei_scale.dims = {pb.E, 2 * pb.N};

    group_matmul_projection_params secondary;
    secondary.output_size = static_cast<int>(pb.K);
    secondary.input_size = static_cast<int>(pb.N);
    secondary.trans_weight = true;
    secondary.weight = pb.w2.data();
    secondary.ldb = static_cast<int>(pb.N);
    secondary.params.dtypes.src = data_type_t::bf16;
    secondary.params.dtypes.wei = data_type_t::s8;
    secondary.params.dtypes.dst = data_type_t::bf16;
    secondary.params.dtypes.compute = data_type_t::s8;
    secondary.params.dynamic_quant = true;
    secondary.params.quant_params.src_scale.buff = src_scale_secondary.data();
    secondary.params.quant_params.src_scale.dt = data_type_t::f32;
    secondary.params.quant_params.src_scale.dims = {pb.M, 1};
    secondary.params.quant_params.wei_scale.buff = pb.s2.data();
    secondary.params.quant_params.wei_scale.dt = data_type_t::f32;
    secondary.params.quant_params.wei_scale.dims = {pb.E, pb.K};

    group_matmul_routing_params routing;
    routing.topk_ids = pb.ids.data();
    routing.topk_ids_stride = static_cast<int>(pb.topk);
    routing.topk_weights = pb.weights.data();
    routing.topk_weights_stride = static_cast<int>(pb.topk);

    zendnnl::lowoha::matmul::grp_matmul_gated_act_params act;
    act.act = zendnnl::lowoha::matmul::grp_matmul_gated_act_t::silu_and_mul;

    ASSERT_EQ(routed_fused_moe_direct('r', false, pb.src.data(),
                      static_cast<int>(pb.K), static_cast<int>(pb.M),
                      static_cast<int>(pb.E), static_cast<int>(pb.topk),
                      pb.dst.data(), static_cast<int>(pb.K), primary, routing,
                      &secondary, &act),
            status_t::success);
    EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);

    // Disabling routed MoE must bypass the eligible fast executor and enter
    // the existing vector group_matmul_direct path.
    struct env_guard_t {
        ~env_guard_t() { unsetenv("ZENDNNL_ENABLE_ROUTED_MOE"); }
    } env_guard;
    ASSERT_EQ(setenv("ZENDNNL_ENABLE_ROUTED_MOE", "0", 1), 0);

    auto &capture = zendnnl::lowoha::matmul::test_api::s_capture_gemm_mode;
    auto &last_mode = zendnnl::lowoha::matmul::test_api::
            s_last_group_matmul_direct_gemm_mode;
    struct capture_guard_t {
        std::atomic<bool> &capture;
        ~capture_guard_t() { capture.store(false, std::memory_order_relaxed); }
    } capture_guard {capture};
    last_mode.store(nullptr, std::memory_order_relaxed);
    capture.store(true, std::memory_order_relaxed);

    std::fill(pb.dst.begin(), pb.dst.end(), 0);
    ASSERT_EQ(routed_fused_moe_direct('r', false, pb.src.data(),
                      static_cast<int>(pb.K), static_cast<int>(pb.M),
                      static_cast<int>(pb.E), static_cast<int>(pb.topk),
                      pb.dst.data(), static_cast<int>(pb.K), primary, routing,
                      &secondary, &act),
            status_t::success);
    ASSERT_NE(last_mode.load(std::memory_order_relaxed), nullptr);
    EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 2e-2);
}

TEST_F(RoutedMoESuperset, DA8W8PromptUsesRoutedExecutor) {
    moe_problem pb;
    pb.build(33, 128, 64, 8, 2, 0x92u);
    std::vector<uint16_t> routed_output(pb.dst.size(), 0);
    auto routed_params = pb.params();
    routed_params.dst = routed_output.data();
    ASSERT_EQ(execute(routed_params), status_t::success);
    std::vector<float> src_scale_primary(static_cast<size_t>(pb.M), 0.0f);
    std::vector<float> src_scale_secondary(static_cast<size_t>(pb.M), 0.0f);

    group_matmul_projection_params primary;
    primary.output_size = static_cast<int>(2 * pb.N);
    primary.input_size = static_cast<int>(pb.K);
    primary.trans_weight = true;
    primary.weight = pb.w13.data();
    primary.ldb = static_cast<int>(pb.K);
    primary.params.dtypes.src = data_type_t::bf16;
    primary.params.dtypes.wei = data_type_t::s8;
    primary.params.dtypes.dst = data_type_t::bf16;
    primary.params.dtypes.compute = data_type_t::s8;
    primary.params.dynamic_quant = true;
    primary.params.quant_params.src_scale.buff = src_scale_primary.data();
    primary.params.quant_params.src_scale.dt = data_type_t::f32;
    primary.params.quant_params.src_scale.dims = {pb.M, 1};
    primary.params.quant_params.wei_scale.buff = pb.s13.data();
    primary.params.quant_params.wei_scale.dt = data_type_t::f32;
    primary.params.quant_params.wei_scale.dims = {pb.E, 2 * pb.N};

    group_matmul_projection_params secondary;
    secondary.output_size = static_cast<int>(pb.K);
    secondary.input_size = static_cast<int>(pb.N);
    secondary.trans_weight = true;
    secondary.weight = pb.w2.data();
    secondary.ldb = static_cast<int>(pb.N);
    secondary.params.dtypes.src = data_type_t::bf16;
    secondary.params.dtypes.wei = data_type_t::s8;
    secondary.params.dtypes.dst = data_type_t::bf16;
    secondary.params.dtypes.compute = data_type_t::s8;
    secondary.params.dynamic_quant = true;
    secondary.params.quant_params.src_scale.buff = src_scale_secondary.data();
    secondary.params.quant_params.src_scale.dt = data_type_t::f32;
    secondary.params.quant_params.src_scale.dims = {pb.M, 1};
    secondary.params.quant_params.wei_scale.buff = pb.s2.data();
    secondary.params.quant_params.wei_scale.dt = data_type_t::f32;
    secondary.params.quant_params.wei_scale.dims = {pb.E, pb.K};

    group_matmul_routing_params routing;
    routing.topk_ids = pb.ids.data();
    routing.topk_weights = pb.weights.data();

    zendnnl::lowoha::matmul::grp_matmul_gated_act_params act;
    act.act = zendnnl::lowoha::matmul::grp_matmul_gated_act_t::silu_and_mul;

    ASSERT_EQ(routed_fused_moe_direct('r', false, pb.src.data(),
                      static_cast<int>(pb.K), static_cast<int>(pb.M),
                      static_cast<int>(pb.E), static_cast<int>(pb.topk),
                      pb.dst.data(), static_cast<int>(pb.K), primary, routing,
                      &secondary, &act),
            status_t::success);
    EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 2e-2);
    EXPECT_EQ(pb.dst, routed_output);
}

TEST_F(RoutedMoESuperset, PaddedExpertWeightsUseRoutedExecutor) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0xc4bu);
    const auto expected = reference_moe(pb);
    da8w8_call call(pb);

    const size_t experts = static_cast<size_t>(pb.E);
    const size_t w13_logical = pb.w13.size() / experts;
    const size_t w2_logical = pb.w2.size() / experts;
    const size_t w13_stride = w13_logical + 64;
    const size_t w2_stride = w2_logical + 64;
    std::vector<int8_t> padded_w13(experts * w13_stride, int8_t {0x5a});
    std::vector<int8_t> padded_w2(experts * w2_stride, int8_t {0x5a});
    for (size_t e = 0; e < experts; ++e) {
        std::memcpy(padded_w13.data() + e * w13_stride,
                pb.w13.data() + e * w13_logical, w13_logical);
        std::memcpy(padded_w2.data() + e * w2_stride,
                pb.w2.data() + e * w2_logical, w2_logical);
    }

    call.primary.weight = padded_w13.data();
    call.primary.wei_buffer_capacity_bytes = w13_stride;
    call.secondary.weight = padded_w2.data();
    call.secondary.wei_buffer_capacity_bytes = w2_stride;

    grouped_call_probe probe;
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_FALSE(probe.grouped_ran());
    EXPECT_LT(rel_mae(pb.dst, expected), 3e-3);
}

// The explicit DA8W8 superset call with gelu_and_mul must take the routed
// executor (bit-identical to calling it directly, and never entering the
// grouped surface) and agree with the forced grouped path.  The grouped path
// quantizes and rounds at different points, so the parity bound is the same
// 2e-2 the SiLU superset test uses.
TEST_F(RoutedMoESuperset, DA8W8GeluDecodeAndPromptUseRoutedExecutor) {
    for (int64_t M : {int64_t {8}, int64_t {65}}) {
        moe_problem pb;
        pb.build(M, 256, 96, 32, 8, 0xa11u + static_cast<uint32_t>(M));
        pb.act = routed_moe_activation_t::gelu_and_mul;
        pb.unit_gate();
        group_matmul_routed_moe_flush_weight_cache();
        reset_grp_matmul_caches();

        std::vector<uint16_t> direct(pb.dst.size(), 0);
        auto direct_params = pb.params();
        direct_params.dst = direct.data();
        ASSERT_EQ(execute(direct_params), status_t::success) << "M=" << M;
        const auto want = reference_moe(pb);

        da8w8_call call(pb);
        grouped_call_probe probe;
        ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success) << "M=" << M;
        EXPECT_FALSE(probe.grouped_ran()) << "M=" << M;
        EXPECT_EQ(pb.dst, direct) << "M=" << M;
        EXPECT_LT(rel_mae(pb.dst, want), 3e-3) << "M=" << M;

        std::vector<uint16_t> generic(pb.dst.size(), 0);
        {
            disable_routed_guard disable;
            probe.arm();
            ASSERT_EQ(call.run(pb, generic.data()), status_t::success)
                    << "M=" << M;
            EXPECT_TRUE(probe.grouped_ran()) << "M=" << M;
        }
        EXPECT_LT(rel_mae(generic, want), 2e-2) << "M=" << M;
        EXPECT_LT(rel_mae(pb.dst, generic), 2e-2) << "M=" << M;
    }
}

TEST_F(RoutedMoESuperset, CapacityMetadataKeepsGroupedFallbackCorrect) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0xc4au);
    const auto expected = reference_moe(pb);
    da8w8_call call(pb);

    const size_t experts = static_cast<size_t>(pb.E);
    const size_t w13_logical = pb.w13.size() / experts;
    const size_t w2_logical = pb.w2.size() / experts;
    const size_t w13_stride = w13_logical + 64;
    const size_t w2_stride = w2_logical + 64;
    std::vector<int8_t> padded_w13(experts * w13_stride, int8_t {0x5a});
    std::vector<int8_t> padded_w2(experts * w2_stride, int8_t {0x5a});
    for (size_t e = 0; e < experts; ++e) {
        std::memcpy(padded_w13.data() + e * w13_stride,
                pb.w13.data() + e * w13_logical, w13_logical);
        std::memcpy(padded_w2.data() + e * w2_stride,
                pb.w2.data() + e * w2_logical, w2_logical);
    }

    call.primary.weight = padded_w13.data();
    call.primary.wei_buffer_capacity_bytes = w13_stride;
    call.secondary.weight = padded_w2.data();
    call.secondary.wei_buffer_capacity_bytes = w2_stride;

    disable_routed_guard disable;
    grouped_call_probe probe;
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_TRUE(probe.grouped_ran());
    EXPECT_LT(rel_mae(pb.dst, expected), 2e-2);
}

// bf16 checkpoint scales are accepted by the routed fast path, converted once
// to f32 and cached for subsequent calls.
TEST_F(RoutedMoESuperset, GeluBf16ScalesUseRoutedFastPath) {
    moe_problem pb;
    pb.build(9, 128, 64, 8, 2, 0xb16u);
    pb.act = routed_moe_activation_t::gelu_and_mul;
    pb.unit_gate();
    da8w8_call call(pb, /*bf16_scales=*/true);

    grouped_call_probe probe;
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_FALSE(probe.grouped_ran());
    EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 2e-2);
}

TEST_F(RoutedMoESuperset, Bf16ScaleCacheFollowsPackedWeightIdentity) {
    moe_problem pb;
    pb.build(9, 128, 64, 8, 2, 0xb17u);
    da8w8_call call(pb, /*bf16_scales=*/true);

    grouped_call_probe probe;
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_FALSE(probe.grouped_ran());
    const auto first = pb.dst;

    // A scale tensor may be rematerialized at another address while the
    // immutable model weight keeps the same cache identity. The converted
    // scales remain attached to that weight until the explicit model-lifetime
    // cache flush.
    std::vector<uint16_t> alternate_down_scale = call.s2_bf16;
    for (auto &value : alternate_down_scale) {
        value = f32_to_bf16(2.0f * bf16_to_f32(value));
    }
    call.secondary.params.quant_params.wei_scale.buff
            = alternate_down_scale.data();

    std::fill(pb.dst.begin(), pb.dst.end(), uint16_t {0});
    probe.arm();
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_FALSE(probe.grouped_ran());
    EXPECT_EQ(pb.dst, first);

    group_matmul_routed_moe_flush_weight_cache();
    std::fill(pb.dst.begin(), pb.dst.end(), uint16_t {0});
    probe.arm();
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_FALSE(probe.grouped_ran());
    EXPECT_GT(rel_mae(pb.dst, first), 0.1);
}

TEST_F(RoutedMoESuperset, Bf16ScalesOutsideFastGeometryUseGroupedFallback) {
    moe_problem pb;
    pb.build(4, 128, 48, 8, 2, 0xbf16u);
    const auto expected = reference_moe(pb);
    da8w8_call call(pb, /*bf16_scales=*/true);

    grouped_call_probe probe;
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_TRUE(probe.grouped_ran());
    EXPECT_LT(rel_mae(pb.dst, expected), 2e-2);
}

TEST_F(RoutedMoESuperset, GeluWithBiasUsesGroupedFallback) {
    // Zero weights reduce the MoE to its down-projection bias, so the grouped
    // result can be checked exactly without modelling the biased epilogue.
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0xb1a5u);
    pb.act = routed_moe_activation_t::gelu_and_mul;
    std::fill(pb.w13.begin(), pb.w13.end(), 0);
    std::fill(pb.w2.begin(), pb.w2.end(), 0);
    da8w8_call call(pb, /*bf16_scales=*/true);

    std::vector<uint16_t> b13(static_cast<size_t>(pb.E * 2 * pb.N));
    std::vector<uint16_t> b2(static_cast<size_t>(pb.E * pb.K));
    for (size_t i = 0; i < b13.size(); ++i) {
        b13[i] = f32_to_bf16(0.002f * static_cast<float>(i % 7) - 0.006f);
    }
    for (int64_t e = 0; e < pb.E; ++e) {
        for (int64_t h = 0; h < pb.K; ++h) {
            b2[static_cast<size_t>(e * pb.K + h)]
                    = f32_to_bf16(0.01f * static_cast<float>(e + 1)
                            + 0.0001f * static_cast<float>(h));
        }
    }
    call.primary.bias = b13.data();
    call.primary.params.dtypes.bias = data_type_t::bf16;
    call.secondary.bias = b2.data();
    call.secondary.params.dtypes.bias = data_type_t::bf16;

    grouped_call_probe probe;
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    ASSERT_TRUE(probe.grouped_ran());
    for (int64_t m = 0; m < pb.M; ++m) {
        for (int64_t h = 0; h < pb.K; ++h) {
            float expected = 0.f;
            for (int64_t t = 0; t < pb.topk; ++t) {
                const size_t slot = static_cast<size_t>(m * pb.topk + t);
                expected += pb.weights[slot]
                        * bf16_to_f32(b2[static_cast<size_t>(
                                pb.ids[slot] * pb.K + h)]);
            }
            EXPECT_NEAR(bf16_to_f32(pb.dst[static_cast<size_t>(m * pb.K + h)]),
                    expected, 1e-3f);
        }
    }
}

TEST_F(RoutedMoESuperset, UnsupportedSwigluOaiBiasUsesGroupedFallback) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0x93u);
    std::fill(pb.w13.begin(), pb.w13.end(), 0);
    std::fill(pb.w2.begin(), pb.w2.end(), 0);

    std::vector<uint16_t> s13(
            static_cast<size_t>(pb.E * 2 * pb.N), f32_to_bf16(0.01f));
    std::vector<uint16_t> s2(
            static_cast<size_t>(pb.E * pb.K), f32_to_bf16(0.01f));
    std::vector<uint16_t> b13(static_cast<size_t>(pb.E * 2 * pb.N));
    std::vector<uint16_t> b2(static_cast<size_t>(pb.E * pb.K));
    for (int64_t e = 0; e < pb.E; ++e) {
        for (int64_t n = 0; n < 2 * pb.N; ++n) {
            b13[static_cast<size_t>(e * 2 * pb.N + n)]
                    = f32_to_bf16(0.002f * static_cast<float>(n % 7 - 3));
        }
        for (int64_t h = 0; h < pb.K; ++h) {
            b2[static_cast<size_t>(e * pb.K + h)]
                    = f32_to_bf16(0.01f * static_cast<float>(e + 1)
                            + 0.0001f * static_cast<float>(h));
        }
    }

    group_matmul_projection_params primary;
    primary.output_size = static_cast<int>(2 * pb.N);
    primary.input_size = static_cast<int>(pb.K);
    primary.trans_weight = true;
    primary.weight = pb.w13.data();
    primary.ldb = static_cast<int>(pb.K);
    primary.bias = b13.data();
    primary.params.dtypes.src = data_type_t::bf16;
    primary.params.dtypes.wei = data_type_t::s8;
    primary.params.dtypes.dst = data_type_t::bf16;
    primary.params.dtypes.bias = data_type_t::bf16;
    primary.params.dtypes.compute = data_type_t::s8;
    primary.params.dynamic_quant = true;
    primary.params.quant_params.src_scale.dt = data_type_t::bf16;
    primary.params.quant_params.src_scale.dims = {pb.M, 1};
    primary.params.quant_params.wei_scale.buff = s13.data();
    primary.params.quant_params.wei_scale.dt = data_type_t::bf16;
    primary.params.quant_params.wei_scale.dims = {pb.E, 2 * pb.N};

    group_matmul_projection_params secondary;
    secondary.output_size = static_cast<int>(pb.K);
    secondary.input_size = static_cast<int>(pb.N);
    secondary.trans_weight = true;
    secondary.weight = pb.w2.data();
    secondary.ldb = static_cast<int>(pb.N);
    secondary.bias = b2.data();
    secondary.params.dtypes.src = data_type_t::bf16;
    secondary.params.dtypes.wei = data_type_t::s8;
    secondary.params.dtypes.dst = data_type_t::bf16;
    secondary.params.dtypes.bias = data_type_t::bf16;
    secondary.params.dtypes.compute = data_type_t::s8;
    secondary.params.dynamic_quant = true;
    secondary.params.quant_params.src_scale.dt = data_type_t::bf16;
    secondary.params.quant_params.src_scale.dims = {pb.M, 1};
    secondary.params.quant_params.wei_scale.buff = s2.data();
    secondary.params.quant_params.wei_scale.dt = data_type_t::bf16;
    secondary.params.quant_params.wei_scale.dims = {pb.E, pb.K};

    group_matmul_routing_params routing;
    routing.topk_ids = pb.ids.data();
    routing.topk_weights = pb.weights.data();

    zendnnl::lowoha::matmul::grp_matmul_gated_act_params act;
    act.act = zendnnl::lowoha::matmul::grp_matmul_gated_act_t::swiglu_oai_mul;

    auto &capture = zendnnl::lowoha::matmul::test_api::s_capture_gemm_mode;
    auto &last_mode = zendnnl::lowoha::matmul::test_api::
            s_last_group_matmul_direct_gemm_mode;
    struct capture_guard_t {
        std::atomic<bool> &capture;
        ~capture_guard_t() { capture.store(false, std::memory_order_relaxed); }
    } capture_guard {capture};
    last_mode.store(nullptr, std::memory_order_relaxed);
    capture.store(true, std::memory_order_relaxed);

    const status_t status = routed_fused_moe_direct('r', false, pb.src.data(),
            static_cast<int>(pb.K), static_cast<int>(pb.M),
            static_cast<int>(pb.E), static_cast<int>(pb.topk), pb.dst.data(),
            static_cast<int>(pb.K), primary, routing, &secondary, &act);
    const char *grouped_mode = last_mode.load(std::memory_order_relaxed);

    ASSERT_EQ(status, status_t::success);
    ASSERT_NE(grouped_mode, nullptr)
            << "unsupported SwiGLU-OAI+bias must invoke group_matmul_direct";
    for (int64_t m = 0; m < pb.M; ++m) {
        for (int64_t h = 0; h < pb.K; ++h) {
            float expected = 0.0f;
            for (int64_t k = 0; k < pb.topk; ++k) {
                const size_t slot = static_cast<size_t>(m * pb.topk + k);
                const int32_t expert = pb.ids[slot];
                expected += pb.weights[slot]
                        * bf16_to_f32(
                                b2[static_cast<size_t>(expert * pb.K + h)]);
            }
            EXPECT_NEAR(bf16_to_f32(pb.dst[static_cast<size_t>(m * pb.K + h)]),
                    expected, 1e-3f);
        }
    }
}

TEST_F(RoutedMoEGeneric, PrimaryOnlyF32SupportsReducedAndSlotOutput) {
    constexpr int M = 5;
    constexpr int E = 3;
    constexpr int K = 4;
    constexpr int N = 6;
    constexpr int topk = 2;
    const std::vector<float> src = {1.f, 2.f, 3.f, 4.f, 2.f, 3.f, 4.f, 5.f, 3.f,
            4.f, 5.f, 6.f, 4.f, 5.f, 6.f, 7.f, 5.f, 6.f, 7.f, 8.f};
    std::vector<float> weights(static_cast<size_t>(E * N * K));
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<float>((static_cast<int>(i) % 11) - 5) / 16.f;
    }
    std::vector<float> bias(static_cast<size_t>(E * N));
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] = static_cast<float>(i % N) / 32.f;
    }
    const std::vector<int32_t> ids = {0, 1, 1, 2, 2, 0, 0, 2, 1, 0};
    const std::vector<float> route_weights
            = {0.75f, 0.25f, 0.6f, 0.4f, 0.2f, 0.8f, 0.5f, 0.5f, 0.3f, 0.7f};

    group_matmul_projection_params primary;
    primary.output_size = N;
    primary.input_size = K;
    primary.trans_weight = true;
    primary.weight = weights.data();
    primary.ldb = K;
    primary.bias = bias.data();
    primary.params.dtypes.src = data_type_t::f32;
    primary.params.dtypes.wei = data_type_t::f32;
    primary.params.dtypes.dst = data_type_t::f32;
    primary.params.dtypes.bias = data_type_t::f32;
    primary.params.num_threads = 2;
    primary.params.weight_cache_type = 0;

    group_matmul_routing_params routing;
    routing.topk_ids = ids.data();
    routing.topk_weights = route_weights.data();

    const auto slot_reference = [&](int token, int slot, int column) {
        const int expert = ids[static_cast<size_t>(token) * topk + slot];
        float value = bias[static_cast<size_t>(expert) * N + column];
        for (int k = 0; k < K; ++k) {
            value += src[static_cast<size_t>(token) * K + k]
                    * weights[(static_cast<size_t>(expert) * N + column) * K
                            + k];
        }
        return value;
    };

    std::vector<float> reduced(static_cast<size_t>(M * N), 0.f);
    ASSERT_EQ(routed_fused_moe_direct('r', false, src.data(), K, M, E, topk,
                      reduced.data(), N, primary, routing),
            status_t::success);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float expected = 0.f;
            for (int t = 0; t < topk; ++t) {
                expected += route_weights[static_cast<size_t>(m) * topk + t]
                        * slot_reference(m, t, n);
            }
            EXPECT_NEAR(
                    reduced[static_cast<size_t>(m) * N + n], expected, 1e-5f);
        }
    }

    routing.reduce_output = false;
    std::vector<float> slots(static_cast<size_t>(M * topk * N), 0.f);
    ASSERT_EQ(routed_fused_moe_direct('r', false, src.data(), K, M, E, topk,
                      slots.data(), N, primary, routing),
            status_t::success);
    for (int m = 0; m < M; ++m) {
        for (int t = 0; t < topk; ++t) {
            for (int n = 0; n < N; ++n) {
                EXPECT_NEAR(slots[(static_cast<size_t>(m) * topk + t) * N + n],
                        slot_reference(m, t, n), 1e-5f);
            }
        }
    }
}

TEST_F(RoutedMoEGeneric, F32GatedTwoProjectionPipeline) {
    constexpr int M = 4;
    constexpr int E = 3;
    constexpr int H = 4;
    constexpr int I = 3;
    constexpr int topk = 2;
    const std::vector<float> src = {1.f, -2.f, 0.5f, 3.f, 0.5f, 1.f, -1.f, 2.f,
            2.f, 0.25f, 1.f, -0.5f, 1.5f, -1.f, 2.f, 0.75f};
    std::vector<float> w13(static_cast<size_t>(E * 2 * I * H));
    std::vector<float> w2(static_cast<size_t>(E * H * I));
    std::vector<float> b13(static_cast<size_t>(E * 2 * I));
    std::vector<float> b2(static_cast<size_t>(E * H));
    for (size_t i = 0; i < w13.size(); ++i) {
        w13[i] = static_cast<float>((static_cast<int>(i) % 13) - 6) / 32.f;
    }
    for (size_t i = 0; i < w2.size(); ++i) {
        w2[i] = static_cast<float>((static_cast<int>(i) % 9) - 4) / 24.f;
    }
    for (size_t i = 0; i < b13.size(); ++i) {
        b13[i] = static_cast<float>(i % (2 * I)) / 64.f;
    }
    for (size_t i = 0; i < b2.size(); ++i) {
        b2[i] = static_cast<float>(i % H) / 48.f;
    }
    const std::vector<int32_t> ids = {0, 1, 1, 2, 2, 0, 0, 2};
    const std::vector<float> route_weights
            = {0.7f, 0.3f, 0.4f, 0.6f, 0.2f, 0.8f, 0.55f, 0.45f};

    group_matmul_projection_params primary;
    primary.output_size = 2 * I;
    primary.input_size = H;
    primary.weight = w13.data();
    primary.ldb = H;
    primary.bias = b13.data();
    primary.params.dtypes.src = data_type_t::f32;
    primary.params.dtypes.wei = data_type_t::f32;
    primary.params.dtypes.dst = data_type_t::f32;
    primary.params.dtypes.bias = data_type_t::f32;
    primary.params.num_threads = 2;
    primary.params.weight_cache_type = 0;

    group_matmul_projection_params secondary;
    secondary.output_size = H;
    secondary.input_size = I;
    secondary.weight = w2.data();
    secondary.ldb = I;
    secondary.bias = b2.data();
    secondary.params.dtypes.src = data_type_t::f32;
    secondary.params.dtypes.wei = data_type_t::f32;
    secondary.params.dtypes.dst = data_type_t::f32;
    secondary.params.dtypes.bias = data_type_t::f32;
    secondary.params.num_threads = 1;
    secondary.params.weight_cache_type = 0;

    group_matmul_routing_params routing;
    routing.topk_ids = ids.data();
    routing.topk_weights = route_weights.data();

    zendnnl::lowoha::matmul::grp_matmul_gated_act_params act;
    act.act = zendnnl::lowoha::matmul::grp_matmul_gated_act_t::silu_and_mul;

    std::vector<float> output(static_cast<size_t>(M * H), 0.f);
    ASSERT_EQ(routed_fused_moe_direct('r', false, src.data(), H, M, E, topk,
                      output.data(), H, primary, routing, &secondary, &act),
            status_t::success);

    for (int m = 0; m < M; ++m) {
        std::vector<float> expected(static_cast<size_t>(H), 0.f);
        for (int t = 0; t < topk; ++t) {
            const int expert = ids[static_cast<size_t>(m) * topk + t];
            float inter[I] = {};
            for (int i = 0; i < I; ++i) {
                float gate = b13[static_cast<size_t>(expert) * 2 * I + i];
                float up = b13[static_cast<size_t>(expert) * 2 * I + I + i];
                for (int h = 0; h < H; ++h) {
                    const float x = src[static_cast<size_t>(m) * H + h];
                    gate += x
                            * w13[(static_cast<size_t>(expert) * 2 * I + i) * H
                                    + h];
                    up += x
                            * w13[(static_cast<size_t>(expert) * 2 * I + I + i)
                                            * H
                                    + h];
                }
                inter[i] = (gate / (1.f + std::exp(-gate))) * up;
            }
            for (int h = 0; h < H; ++h) {
                float down = b2[static_cast<size_t>(expert) * H + h];
                for (int i = 0; i < I; ++i) {
                    down += inter[i]
                            * w2[(static_cast<size_t>(expert) * H + h) * I + i];
                }
                expected[static_cast<size_t>(h)]
                        += route_weights[static_cast<size_t>(m) * topk + t]
                        * down;
            }
        }
        for (int h = 0; h < H; ++h) {
            EXPECT_NEAR(output[static_cast<size_t>(m) * H + h],
                    expected[static_cast<size_t>(h)], 2e-5f);
        }
    }
}

// The generic fallback keeps its grouped-token, descriptor and quant scratch
// in a thread-local that survives the call and only ever grows, so a call is
// only correct if it re-seeds every element it reads rather than inheriting
// the tail of a wider predecessor.  Drive one thread through a sequence that
// shrinks and re-grows every dimension the scratch is indexed by -- token
// count, active-expert count and slot count -- alternate the reduced and slot
// output modes, which take the fused and the two-projection arm respectively
// and size the destination differently, and alternate bias presence, which is
// what distinguishes re-seeding the per-expert operand vectors from merely
// resizing them.  Each call is checked against a reference computed from
// scratch, so a stale carry-over shows up as a numeric mismatch.
TEST_F(RoutedMoEGeneric, ReusedScratchSurvivesShrinkingAndGrowingShapes) {
    constexpr int E = 3;
    constexpr int H = 4;
    constexpr int I = 3;
    constexpr int topk = 2;
    constexpr int max_M = 8;

    std::vector<float> src(static_cast<size_t>(max_M * H));
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<float>((static_cast<int>(i) % 7) - 3) / 4.f;
    }
    std::vector<float> w13(static_cast<size_t>(E * 2 * I * H));
    std::vector<float> w2(static_cast<size_t>(E * H * I));
    for (size_t i = 0; i < w13.size(); ++i) {
        w13[i] = static_cast<float>((static_cast<int>(i) % 13) - 6) / 32.f;
    }
    for (size_t i = 0; i < w2.size(); ++i) {
        w2[i] = static_cast<float>((static_cast<int>(i) % 9) - 4) / 24.f;
    }
    std::vector<float> b13(static_cast<size_t>(E * 2 * I));
    std::vector<float> b2(static_cast<size_t>(E * H));
    for (size_t i = 0; i < b13.size(); ++i) {
        b13[i] = 0.5f + static_cast<float>(i % 5) / 8.f;
    }
    for (size_t i = 0; i < b2.size(); ++i) {
        b2[i] = -0.75f + static_cast<float>(i % 3) / 6.f;
    }

    group_matmul_projection_params primary;
    primary.output_size = 2 * I;
    primary.input_size = H;
    primary.weight = w13.data();
    primary.ldb = H;
    primary.params.dtypes.src = data_type_t::f32;
    primary.params.dtypes.wei = data_type_t::f32;
    primary.params.dtypes.dst = data_type_t::f32;
    primary.params.dtypes.bias = data_type_t::f32;
    primary.params.num_threads = 2;
    primary.params.weight_cache_type = 0;

    group_matmul_projection_params secondary;
    secondary.output_size = H;
    secondary.input_size = I;
    secondary.weight = w2.data();
    secondary.ldb = I;
    secondary.params.dtypes.src = data_type_t::f32;
    secondary.params.dtypes.wei = data_type_t::f32;
    secondary.params.dtypes.dst = data_type_t::f32;
    secondary.params.dtypes.bias = data_type_t::f32;
    secondary.params.num_threads = 2;
    secondary.params.weight_cache_type = 0;

    zendnnl::lowoha::matmul::grp_matmul_gated_act_params act;
    act.act = zendnnl::lowoha::matmul::grp_matmul_gated_act_t::silu_and_mul;

    // Reference for one routed slot: gate/up projection, SiLU-and-mul, then
    // the down projection.  Recomputed per call so it cannot itself go stale.
    const auto slot_reference = [&](int token, int expert, bool with_bias,
                                        std::vector<float> &out) {
        std::vector<float> inter(static_cast<size_t>(I), 0.f);
        for (int i = 0; i < I; ++i) {
            float gate = with_bias
                    ? b13[static_cast<size_t>(expert) * 2 * I + i]
                    : 0.f;
            float up = with_bias
                    ? b13[static_cast<size_t>(expert) * 2 * I + I + i]
                    : 0.f;
            for (int h = 0; h < H; ++h) {
                const float x = src[static_cast<size_t>(token) * H + h];
                gate += x
                        * w13[(static_cast<size_t>(expert) * 2 * I + i) * H
                                + h];
                up += x
                        * w13[(static_cast<size_t>(expert) * 2 * I + I + i) * H
                                + h];
            }
            inter[static_cast<size_t>(i)]
                    = (gate / (1.f + std::exp(-gate))) * up;
        }
        for (int h = 0; h < H; ++h) {
            float down
                    = with_bias ? b2[static_cast<size_t>(expert) * H + h] : 0.f;
            for (int i = 0; i < I; ++i) {
                down += inter[static_cast<size_t>(i)]
                        * w2[(static_cast<size_t>(expert) * H + h) * I + i];
            }
            out[static_cast<size_t>(h)] = down;
        }
    };

    const auto run_case
            = [&](const char *what, int M, const std::vector<int32_t> &ids,
                      bool reduce_output, bool with_bias) {
        SCOPED_TRACE(what);
        primary.bias = with_bias ? b13.data() : nullptr;
        secondary.bias = with_bias ? b2.data() : nullptr;
        ASSERT_EQ(ids.size(), static_cast<size_t>(M) * topk);
        std::vector<float> route_weights(ids.size());
        for (size_t i = 0; i < route_weights.size(); ++i) {
            route_weights[i] = 0.25f + 0.125f * static_cast<float>(i % 4);
        }

        group_matmul_routing_params routing;
        routing.topk_ids = ids.data();
        routing.topk_weights = route_weights.data();
        routing.reduce_output = reduce_output;

        const size_t rows = reduce_output ? static_cast<size_t>(M)
                                          : static_cast<size_t>(M) * topk;
        std::vector<float> output(rows * static_cast<size_t>(H), -7.f);
        ASSERT_EQ(routed_fused_moe_direct('r', false, src.data(), H, M, E, topk,
                          output.data(), H, primary, routing, &secondary, &act),
                status_t::success);

        std::vector<float> slot(static_cast<size_t>(H), 0.f);
        for (int m = 0; m < M; ++m) {
            std::vector<float> reduced(static_cast<size_t>(H), 0.f);
            for (int t = 0; t < topk; ++t) {
                const size_t index = static_cast<size_t>(m) * topk + t;
                slot_reference(m, ids[index], with_bias, slot);
                for (int h = 0; h < H; ++h) {
                    reduced[static_cast<size_t>(h)] += route_weights[index]
                            * slot[static_cast<size_t>(h)];
                }
                if (!reduce_output) {
                    for (int h = 0; h < H; ++h) {
                        EXPECT_NEAR(output[index * H + static_cast<size_t>(h)],
                                slot[static_cast<size_t>(h)], 2e-5f)
                                << "m=" << m << " t=" << t << " h=" << h;
                    }
                }
            }
            if (reduce_output) {
                for (int h = 0; h < H; ++h) {
                    EXPECT_NEAR(output[static_cast<size_t>(m) * H
                                        + static_cast<size_t>(h)],
                            reduced[static_cast<size_t>(h)], 2e-5f)
                            << "m=" << m << " h=" << h;
                }
            }
        }
    };

    // Widest and biased first, so every later call runs against scratch that
    // is larger than it needs and carries live per-expert bias pointers that
    // an unbiased successor must not inherit.
    const std::vector<int32_t> wide
            = {0, 1, 1, 2, 2, 0, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2};
    run_case("wide/all experts/reduced/bias", max_M, wide, true, true);
    // One active expert and two tokens: shrinks tokens, slots and the active
    // set at once, drops the bias, and leaves two experts inactive.
    run_case("narrow/single expert/reduced/no bias", 2, {1, 1, 1, 1}, true,
            false);
    run_case("wide again/all experts/reduced/bias", max_M, wide, true, true);
    // Slot output re-enters through the two-projection arm and the other tail
    // of finish_output.
    run_case("narrow/single expert/slots/bias", 1, {2, 2}, false, true);
    run_case("mid/two experts/slots/no bias", 3, {0, 2, 2, 0, 0, 2}, false,
            false);
    run_case("wide again/all experts/slots/bias", max_M, wide, false, true);
    run_case(
            "narrow/all experts/reduced/no bias", 2, {0, 1, 2, 0}, true, false);
    run_case("narrow/all experts/reduced/bias", 2, {0, 1, 2, 0}, true, true);
}
