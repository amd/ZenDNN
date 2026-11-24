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
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <omp.h>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/group_matmul/common/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_env.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/nr32_pack.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel_config.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack.hpp"
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
// GELU (exact std::erf here, std::tanh for the tanh form).  Written
// independently of the kernels (plain loops, no intrinsics, no shared
// helpers).
//
// `expert_map` follows the same convention as the contract: null means the ids
// are already local, and a negative entry masks a slot out.
// ---------------------------------------------------------------------------
float gelu_ref(float x) {
    if (zendnnl::lowoha::matmul::get_grp_matmul_gelu_is_tanh()) {
        return 0.5f * x
                * (1.f
                        + std::tanh(
                                0.7978845608f * (x + 0.044715f * x * x * x)));
    }
    return 0.5f * x * (1.f + std::erf(x * 0.70710678f));
}

/// `b13` (`[E, 2N]`) and `b2` (`[E, K]`), when given, are the gate/up and the
/// down biases.
std::vector<uint16_t> reference_moe(const moe_problem &pb,
        const int32_t *expert_map = nullptr, const float *b13 = nullptr,
        const float *b2 = nullptr) {
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
                float gf = (static_cast<float>(g) * sa) * s13e[j];
                float uf = (static_cast<float>(u) * sa) * s13e[N + j];
                if (b13 != nullptr) {
                    gf += b13[e * 2 * N + j];
                    uf += b13[e * 2 * N + N + j];
                }
                const float act
                        = pb.act == routed_moe_activation_t::gelu_and_mul
                        ? gelu_ref(gf)
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
                float df = (static_cast<float>(d) * sa2) * s2e[k];
                if (b2 != nullptr) df += b2[e * K + k];
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

/// A BF16 NaN written over an output before a call, so that an element the
/// call leaves unwritten stays visible.
constexpr uint16_t kOutputPoison = 0x7fc1u;

void poison(std::vector<uint16_t> &out) {
    std::fill(out.begin(), out.end(), kOutputPoison);
}

void expect_output_written(const std::vector<uint16_t> &out) {
    EXPECT_EQ(std::count(out.begin(), out.end(), kOutputPoison), 0)
            << "elements left unwritten";
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
ZENDNNL_TARGET("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx512bf16,fma")
void routed_epilogue_act(
        routed_moe_activation_t act, const float *x, float *y, int64_t n) {
    namespace rm = zendnnl::lowoha::matmul::routed_moe;
    const auto mode
            = zendnnl::lowoha::matmul::group_matmul_act_avx512::act_mode();
    for (int64_t i = 0; i < n; i += 16) {
        const __m512 v = _mm512_loadu_ps(x + i);
        _mm512_storeu_ps(y + i,
                act == routed_moe_activation_t::gelu_and_mul
                        ? rm::gated_act_ps<
                                  routed_moe_activation_t::gelu_and_mul>(
                                  v, mode)
                        : rm::gated_act_ps<
                                  routed_moe_activation_t::silu_and_mul>(
                                  v, mode));
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
            = zendnnl::lowoha::matmul::test_api::s_capture_execution_path;
    std::atomic<const char *> &last_mode = zendnnl::lowoha::matmul::test_api::
            s_last_group_matmul_direct_execution_path;

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

struct routed_cache_flush_guard {
    ~routed_cache_flush_guard() {
        group_matmul_routed_moe_flush_weight_cache();
    }
};

struct routed_weight_cache_mode_guard {
    zendnnl::common::matmul_config_t &config
            = zendnnl::common::matmul_config_t::instance();
    int32_t previous_cache = config.get_weight_cache();
    bool previous_mixed = config.get_grp_auto_mixed_inplace();

    routed_weight_cache_mode_guard() {
        config.set_weight_cache(2);
        config.set_grp_auto_mixed_inplace(false);
    }
    ~routed_weight_cache_mode_guard() {
        config.set_weight_cache(previous_cache);
        config.set_grp_auto_mixed_inplace(previous_mixed);
    }
};

/// Pins the weight cache mode for its scope.
struct weight_cache_mode_pin {
    zendnnl::common::matmul_config_t &config
            = zendnnl::common::matmul_config_t::instance();
    int32_t previous = config.get_weight_cache();
    explicit weight_cache_mode_pin(int32_t mode) {
        config.set_weight_cache(mode);
    }
    ~weight_cache_mode_pin() { config.set_weight_cache(previous); }
};

/// Sets one group-matmul selector override (`-1` = unset) for its scope.
struct selector_pin {
    std::atomic<int> &slot;
    int previous;
    selector_pin(std::atomic<int> &s, int value)
        : slot(s), previous(s.exchange(value, std::memory_order_relaxed)) {}
    ~selector_pin() { slot.store(previous, std::memory_order_relaxed); }
};

/// Requests N-tile flat parallel for decode calls (the
/// `ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO=4` override) for its scope, with the
/// global selector on AUTO, the prompt setting unset (7), weight cache mode 1
/// and the custom-kernel switches on, whatever the environment sets, and
/// counts the calls N-tile flat parallel completes and the routed layer entries
/// it makes.
struct ntile_flat_parallel_decode_guard {
    selector_pin global_algo {
            zendnnl::lowoha::matmul::test_api_algo_override(), 0};
    selector_pin prompt_algo {zendnnl::lowoha::matmul::test_api::
                                      s_grp_matmul_auto_prompt_algo_override,
            7};
    selector_pin decode_algo {zendnnl::lowoha::matmul::test_api::
                                      s_grp_matmul_auto_decode_algo_override,
            4};
    selector_pin ck {zendnnl::lowoha::matmul::test_api::
                             s_grp_matmul_custom_kernel_override,
            1};
    selector_pin ck_int8 {zendnnl::lowoha::matmul::test_api::
                                  s_grp_matmul_custom_kernel_int8_override,
            1};
    weight_cache_mode_pin cache_mode {1};
    std::atomic<bool> &capture
            = zendnnl::lowoha::matmul::test_api::s_capture_ntile_flat_parallel;
    std::atomic<uint64_t> &completed = zendnnl::lowoha::matmul::test_api::
            s_ntile_flat_parallel_completed;
    std::atomic<uint64_t> &captured = zendnnl::lowoha::matmul::test_api::
            s_ntile_flat_parallel_layer_entries;

    ntile_flat_parallel_decode_guard() {
        completed.store(0, std::memory_order_relaxed);
        captured.store(0, std::memory_order_relaxed);
        capture.store(true, std::memory_order_relaxed);
    }
    ~ntile_flat_parallel_decode_guard() {
        capture.store(false, std::memory_order_relaxed);
    }
    uint64_t calls() const { return completed.load(std::memory_order_relaxed); }
    uint64_t layer_entries() const {
        return captured.load(std::memory_order_relaxed);
    }
};

struct RoutedMoECapability : routed_moe_test {};
struct RoutedMoEKernel : routed_moe_isa_test {};
struct RoutedMoEValidate : routed_moe_isa_test {};
struct RoutedMoEPack : routed_moe_isa_test {};
struct RoutedMoEExecute : routed_moe_isa_test {};
struct RoutedMoECache : routed_moe_isa_test {};
struct RoutedMoESuperset : routed_moe_isa_test {};
struct RoutedMoEGeneric : routed_moe_test {};
/// Pins the selectors these tests assume, whatever the environment sets: the
/// global selector on AUTO and both phase settings unset (7), so the routes
/// they compare N-tile flat parallel with are the defaults.
struct RoutedMoENTileFlatParallel : routed_moe_isa_test {
    selector_pin global_algo {
            zendnnl::lowoha::matmul::test_api_algo_override(), 0};
    selector_pin prompt_algo {zendnnl::lowoha::matmul::test_api::
                                      s_grp_matmul_auto_prompt_algo_override,
            7};
    selector_pin decode_algo {zendnnl::lowoha::matmul::test_api::
                                      s_grp_matmul_auto_decode_algo_override,
            7};
};

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
    if (zendnnl::lowoha::matmul::get_grp_matmul_gelu_is_tanh()) {
        GTEST_SKIP() << "pins the erf form; a GELU knob selects the tanh form";
    }
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

TEST_F(RoutedMoEExecute, RejectsF32OutputForW8WithoutWritingDestination) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0xF32u);
    std::vector<float> output(pb.dst.size(), -99.0f);
    const auto untouched = output;

    auto p = pb.params();
    p.dst = output.data();
    p.dst_dt = data_type_t::f32;

    EXPECT_EQ(group_matmul_routed_moe_validate(p), status_t::unimplemented);
    EXPECT_EQ(execute(p), status_t::unimplemented);
    EXPECT_EQ(output, untouched);
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

    auto &capture = zendnnl::lowoha::matmul::test_api::s_capture_execution_path;
    auto &last_mode = zendnnl::lowoha::matmul::test_api::
            s_last_group_matmul_direct_execution_path;
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

TEST_F(RoutedMoESuperset, WeightCache2PacksDA8W8ExpertsInPlace) {
    routed_weight_cache_mode_guard cache_mode_guard;
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0xc4cu);
    const auto expected_output = reference_moe(pb);
    da8w8_call call(pb);

    const size_t experts = static_cast<size_t>(pb.E);
    const size_t w13_logical = pb.w13.size() / experts;
    const size_t w2_logical = pb.w2.size() / experts;
    const size_t w13_capacity = static_cast<size_t>(2 * pb.N)
            * static_cast<size_t>(group_matmul_routed_moe_packed_row_bytes(
                    pb.K, data_type_t::s8));
    const size_t w2_capacity = static_cast<size_t>(pb.K)
            * static_cast<size_t>(group_matmul_routed_moe_packed_row_bytes(
                    pb.N, data_type_t::s8));
    ASSERT_GT(w13_capacity, w13_logical);
    ASSERT_GT(w2_capacity, w2_logical);

    std::vector<int8_t> padded_w13(experts * w13_capacity, int8_t {0x5a});
    std::vector<int8_t> padded_w2(experts * w2_capacity, int8_t {0x5a});
    routed_cache_flush_guard cache_guard;
    for (size_t e = 0; e < experts; ++e) {
        std::memcpy(padded_w13.data() + e * w13_capacity,
                pb.w13.data() + e * w13_logical, w13_logical);
        std::memcpy(padded_w2.data() + e * w2_capacity,
                pb.w2.data() + e * w2_logical, w2_logical);
    }

    std::vector<int8_t> expected_w13(experts * w13_capacity);
    std::vector<int8_t> expected_w2(experts * w2_capacity);
    ASSERT_EQ(group_matmul_routed_moe_pack_weights(pb.w13.data(),
                      expected_w13.data(), pb.E, 2 * pb.N, pb.K,
                      data_type_t::s8, 4),
            status_t::success);
    ASSERT_EQ(group_matmul_routed_moe_pack_weights(pb.w2.data(),
                      expected_w2.data(), pb.E, pb.K, pb.N, data_type_t::s8, 4),
            status_t::success);

    call.primary.weight = padded_w13.data();
    call.primary.wei_buffer_capacity_bytes = w13_capacity;
    call.primary.params.weight_cache_type = 2;
    call.secondary.weight = padded_w2.data();
    call.secondary.wei_buffer_capacity_bytes = w2_capacity;
    call.secondary.params.weight_cache_type = 2;

    const auto original_w13 = padded_w13;
    const auto original_w2 = padded_w2;
    grouped_call_probe probe;

    auto &fail_registration = zendnnl::lowoha::matmul::routed_moe::test_api::
            s_fail_inplace_registration_after_sentinel;
    fail_registration.store(true, std::memory_order_relaxed);
    probe.arm();
    EXPECT_EQ(call.run(pb, pb.dst.data()), status_t::memory_bad_storage);
    EXPECT_FALSE(probe.grouped_ran());
    EXPECT_EQ(padded_w13, original_w13);
    EXPECT_EQ(padded_w2, original_w2);

    // The failed registration rolled both maps back and left source bytes
    // untouched, so an immediate retry can pack and execute normally.
    probe.arm();
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_FALSE(probe.grouped_ran());
    EXPECT_LT(rel_mae(pb.dst, expected_output), 3e-3);
    for (size_t e = 0; e < experts; ++e) {
        EXPECT_EQ(std::memcmp(padded_w13.data() + e * w13_capacity,
                          expected_w13.data() + e * w13_capacity, w13_capacity),
                0);
        EXPECT_EQ(std::memcmp(padded_w2.data() + e * w2_capacity,
                          expected_w2.data() + e * w2_capacity, w2_capacity),
                0);
    }

    // A cache hit must consume the caller-owned pack directly, without trying
    // to interpret the mutated buffer as row-major weights.
    std::fill(pb.dst.begin(), pb.dst.end(), 0);
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_LT(rel_mae(pb.dst, expected_output), 3e-3);

    // A live process downgrade cannot revoke an existing caller-owned pack:
    // the row-major image is gone until flush, so the matching routed layout
    // must remain usable even though new in-place mutations are disabled.
    cache_mode_guard.config.set_weight_cache(1);
    std::fill(pb.dst.begin(), pb.dst.end(), 0);
    probe.arm();
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_FALSE(probe.grouped_ran());
    EXPECT_LT(rel_mae(pb.dst, expected_output), 3e-3);
    cache_mode_guard.config.set_weight_cache(2);

    // Once caller storage is packed in place, no generic fallback may consume
    // it as row-major weights. A later eligibility/configuration change must
    // fail closed before entering grouped Matmul.
    {
        disable_routed_guard disable;
        probe.arm();
        EXPECT_EQ(call.run(pb, pb.dst.data()), status_t::memory_bad_storage);
        EXPECT_FALSE(probe.grouped_ran());
    }

    // Flush must reconstruct the row-major image before dropping its sentinel;
    // otherwise this same model would be packed a second time on reuse.
    group_matmul_routed_moe_flush_weight_cache();
    for (size_t e = 0; e < experts; ++e) {
        EXPECT_EQ(std::memcmp(padded_w13.data() + e * w13_capacity,
                          original_w13.data() + e * w13_capacity, w13_logical),
                0);
        EXPECT_EQ(std::memcmp(padded_w2.data() + e * w2_capacity,
                          original_w2.data() + e * w2_capacity, w2_logical),
                0);
    }
    std::fill(pb.dst.begin(), pb.dst.end(), uint16_t {0});
    probe.arm();
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_FALSE(probe.grouped_ran());
    EXPECT_LT(rel_mae(pb.dst, expected_output), 3e-3);
}

TEST_F(RoutedMoESuperset,
        GenericFirstWeightCache2PreservesRowMajorForRoutedPacking) {
    for (int64_t M : {int64_t {8}, int64_t {65}}) {
        SCOPED_TRACE(::testing::Message() << "M=" << M);
        routed_weight_cache_mode_guard cache_mode_guard;
        group_matmul_routed_moe_flush_weight_cache();
        reset_grp_matmul_caches();

        moe_problem pb;
        pb.build(M, 128, 64, 8, 2, 0xc4e0u + static_cast<uint32_t>(M));
        const auto expected_output = reference_moe(pb);
        da8w8_call call(pb);

        const size_t experts = static_cast<size_t>(pb.E);
        const size_t w13_logical = pb.w13.size() / experts;
        const size_t w2_logical = pb.w2.size() / experts;
        const size_t w13_capacity = static_cast<size_t>(2 * pb.N)
                * static_cast<size_t>(group_matmul_routed_moe_packed_row_bytes(
                        pb.K, data_type_t::s8));
        const size_t w2_capacity = static_cast<size_t>(pb.K)
                * static_cast<size_t>(group_matmul_routed_moe_packed_row_bytes(
                        pb.N, data_type_t::s8));
        std::vector<int8_t> padded_w13(experts * w13_capacity, int8_t {0x5a});
        std::vector<int8_t> padded_w2(experts * w2_capacity, int8_t {0x5a});
        routed_cache_flush_guard cache_guard;
        for (size_t e = 0; e < experts; ++e) {
            std::memcpy(padded_w13.data() + e * w13_capacity,
                    pb.w13.data() + e * w13_logical, w13_logical);
            std::memcpy(padded_w2.data() + e * w2_capacity,
                    pb.w2.data() + e * w2_logical, w2_logical);
        }
        const auto original_w13 = padded_w13;
        const auto original_w2 = padded_w2;

        call.primary.weight = padded_w13.data();
        call.primary.wei_buffer_capacity_bytes = w13_capacity;
        call.primary.params.weight_cache_type = 2;
        call.secondary.weight = padded_w2.data();
        call.secondary.wei_buffer_capacity_bytes = w2_capacity;
        call.secondary.params.weight_cache_type = 2;

        grouped_call_probe probe;
        {
            disable_routed_guard disable;
            ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
            EXPECT_TRUE(probe.grouped_ran());
        }
        EXPECT_LT(rel_mae(pb.dst, expected_output), 2e-2);
        EXPECT_EQ(padded_w13, original_w13);
        EXPECT_EQ(padded_w2, original_w2);

        // The generic call may update process-wide grouped policy. Restore WC=2
        // and prove the routed miss still sees untouched row-major weights.
        cache_mode_guard.config.set_weight_cache(2);
        cache_mode_guard.config.set_grp_auto_mixed_inplace(false);
        std::fill(pb.dst.begin(), pb.dst.end(), uint16_t {0});
        probe.arm();
        ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
        EXPECT_FALSE(probe.grouped_ran());
        EXPECT_LT(rel_mae(pb.dst, expected_output), 3e-3);
        EXPECT_NE(padded_w13, original_w13);
        EXPECT_NE(padded_w2, original_w2);

        group_matmul_routed_moe_flush_weight_cache();
        reset_grp_matmul_caches();
    }
}

// The generic path quantizes each token once while grouping it: large enough
// here to group in parallel, with enough experts to seed their descriptors in
// parallel, and with tokens routed only to another rank's experts.
TEST_F(RoutedMoESuperset, GenericQuantizingGatherMatchesReference) {
    constexpr int kLocal = 48;
    constexpr int kGlobal = 64;
    std::vector<int32_t> map(kGlobal);
    for (int g = 0; g < kGlobal; ++g) {
        map[static_cast<size_t>(g)] = g < kLocal ? g : -1;
    }
    for (const bool bf16_scales : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "bf16_scales=" << bf16_scales);
        moe_problem pb;
        pb.build(512, 256, 128, kLocal, 4, 0x9a7eu);
        pb.unit_gate();
        std::mt19937 rng(0x51u);
        std::uniform_int_distribution<int> pick(0, kGlobal - 1);
        for (size_t s = 0; s < pb.ids.size(); ++s) {
            // The first 8 tokens route only to remote experts.
            pb.ids[s] = s < 8 * static_cast<size_t>(pb.topk)
                    ? kLocal + static_cast<int>(s % pb.topk)
                    : pick(rng);
        }
        da8w8_call call(pb, bf16_scales);
        call.routing.expert_map = map.data();
        call.routing.expert_map_size = kGlobal;
        const auto expected_output = reference_moe(pb, map.data());

        disable_routed_guard disable;
        grouped_call_probe probe;
        ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
        EXPECT_TRUE(probe.grouped_ran());
        EXPECT_LT(rel_mae(pb.dst, expected_output), 2e-2);
        for (size_t i = 0; i < 8 * static_cast<size_t>(pb.K); ++i) {
            ASSERT_EQ(pb.dst[i], 0) << "remote-only token, element " << i;
        }
    }
}

TEST_F(RoutedMoESuperset, ConcurrentGenericReadSerializesFirstInplacePacking) {
    routed_weight_cache_mode_guard cache_mode_guard;
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0xc4du);
    const auto expected_output = reference_moe(pb);
    da8w8_call call(pb);
    call.primary.params.num_threads = 1;
    call.secondary.params.num_threads = 1;

    const size_t experts = static_cast<size_t>(pb.E);
    const size_t w13_logical = pb.w13.size() / experts;
    const size_t w2_logical = pb.w2.size() / experts;
    const size_t w13_capacity = static_cast<size_t>(2 * pb.N)
            * static_cast<size_t>(group_matmul_routed_moe_packed_row_bytes(
                    pb.K, data_type_t::s8));
    const size_t w2_capacity = static_cast<size_t>(pb.K)
            * static_cast<size_t>(group_matmul_routed_moe_packed_row_bytes(
                    pb.N, data_type_t::s8));
    std::vector<int8_t> padded_w13(experts * w13_capacity, int8_t {0x5a});
    std::vector<int8_t> padded_w2(experts * w2_capacity, int8_t {0x5a});
    routed_cache_flush_guard cache_guard;
    for (size_t e = 0; e < experts; ++e) {
        std::memcpy(padded_w13.data() + e * w13_capacity,
                pb.w13.data() + e * w13_logical, w13_logical);
        std::memcpy(padded_w2.data() + e * w2_capacity,
                pb.w2.data() + e * w2_logical, w2_logical);
    }
    const auto original_w13 = padded_w13;
    const auto original_w2 = padded_w2;

    call.primary.weight = padded_w13.data();
    call.primary.wei_buffer_capacity_bytes = w13_capacity;
    call.primary.params.weight_cache_type = 2;
    call.secondary.weight = padded_w2.data();
    call.secondary.wei_buffer_capacity_bytes = w2_capacity;
    call.secondary.params.weight_cache_type = 2;

    // Bias makes this call generic-ineligible for the routed kernel while
    // preserving the same zero-bias numerical result.
    std::vector<uint16_t> zero_b13(
            static_cast<size_t>(pb.E * 2 * pb.N), uint16_t {0});
    std::vector<uint16_t> zero_b2(
            static_cast<size_t>(pb.E * pb.K), uint16_t {0});
    da8w8_call generic_call = call;
    generic_call.primary.bias = zero_b13.data();
    generic_call.primary.params.dtypes.bias = data_type_t::bf16;
    generic_call.secondary.bias = zero_b2.data();
    generic_call.secondary.params.dtypes.bias = data_type_t::bf16;

    auto &pause = zendnnl::lowoha::matmul::routed_moe::test_api::
            s_pause_generic_with_storage_guard;
    auto &generic_entered = zendnnl::lowoha::matmul::routed_moe::test_api::
            s_generic_storage_guard_entered;
    auto &capture_request = zendnnl::lowoha::matmul::routed_moe::test_api::
            s_capture_inplace_pack_request;
    auto &pack_requested = zendnnl::lowoha::matmul::routed_moe::test_api::
            s_inplace_pack_requested;
    struct hook_guard_t {
        std::atomic<bool> &pause;
        std::atomic<bool> &capture;
        ~hook_guard_t() {
            pause.store(false, std::memory_order_release);
            capture.store(false, std::memory_order_relaxed);
        }
    } hook_guard {pause, capture_request};

    generic_entered.store(false, std::memory_order_relaxed);
    pack_requested.store(false, std::memory_order_relaxed);
    capture_request.store(true, std::memory_order_relaxed);
    pause.store(true, std::memory_order_release);

    const auto wait_until = [](const std::atomic<bool> &flag) {
        const auto deadline
                = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!flag.load(std::memory_order_acquire)
                && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        return flag.load(std::memory_order_acquire);
    };

    std::vector<uint16_t> generic_output(pb.dst.size(), uint16_t {0});
    std::vector<uint16_t> fast_output(pb.dst.size(), uint16_t {0});
    status_t generic_status = status_t::failure;
    status_t fast_status = status_t::failure;
    std::thread generic_thread([&]() {
        generic_status = generic_call.run(pb, generic_output.data());
    });

    const bool saw_generic_guard = wait_until(generic_entered);
    std::thread fast_thread;
    bool saw_pack_request = false;
    bool storage_stayed_raw = false;
    if (saw_generic_guard) {
        fast_thread = std::thread(
                [&]() { fast_status = call.run(pb, fast_output.data()); });
        saw_pack_request = wait_until(pack_requested);
        storage_stayed_raw
                = padded_w13 == original_w13 && padded_w2 == original_w2;
    }

    pause.store(false, std::memory_order_release);
    generic_thread.join();
    if (fast_thread.joinable()) { fast_thread.join(); }
    capture_request.store(false, std::memory_order_relaxed);

    ASSERT_TRUE(saw_generic_guard);
    ASSERT_TRUE(saw_pack_request);
    EXPECT_TRUE(storage_stayed_raw);
    ASSERT_EQ(generic_status, status_t::success);
    ASSERT_EQ(fast_status, status_t::success);
    EXPECT_LT(rel_mae(generic_output, expected_output), 2e-2);
    EXPECT_LT(rel_mae(fast_output, expected_output), 3e-3);
    EXPECT_NE(padded_w13, original_w13);
    EXPECT_NE(padded_w2, original_w2);
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

    auto &capture = zendnnl::lowoha::matmul::test_api::s_capture_execution_path;
    auto &last_mode = zendnnl::lowoha::matmul::test_api::
            s_last_group_matmul_direct_execution_path;
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

// The routed executor never packs a 16-bit weight, so the generic fallback
// keeps the caller's WEIGHT_CACHE=2: group Matmul writes the custom-kernel pack
// of a BF16 or FP16 MoE layer into the caller's buffers, a second call reads
// it, and the output matches an out-of-place run bit for bit.
TEST_F(RoutedMoEGeneric, SixteenBitFallbackKeepsTheWeightCacheMode) {
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    std::vector<data_type_t> types;
    if (ck::bf16_isa_supported()) types.push_back(data_type_t::bf16);
    if (ck::avx512f16_available()) types.push_back(data_type_t::f16);
    if (types.empty()) {
        GTEST_SKIP() << "no 16-bit custom kernel on this host";
    }
    constexpr int M = 8;
    constexpr int E = 4;
    constexpr int H = 256;
    constexpr int I = 256;
    constexpr int topk = 2;
    std::vector<int32_t> ids(static_cast<size_t>(M) * topk);
    std::vector<float> route_weights(ids.size());
    for (int m = 0; m < M; ++m) {
        for (int t = 0; t < topk; ++t) {
            ids[static_cast<size_t>(m) * topk + t] = (m + t) % E;
            route_weights[static_cast<size_t>(m) * topk + t]
                    = t == 0 ? 0.75f : 0.25f;
        }
    }
    group_matmul_routing_params routing;
    routing.topk_ids = ids.data();
    routing.topk_weights = route_weights.data();
    zendnnl::lowoha::matmul::grp_matmul_gated_act_params act;
    act.act = zendnnl::lowoha::matmul::grp_matmul_gated_act_t::silu_and_mul;

    for (const data_type_t dt : types) {
        SCOPED_TRACE(dt == data_type_t::f16 ? "FP16" : "BF16");
        const auto from_f32 = [dt](float f) {
            return dt == data_type_t::f16
                    ? zendnnl::common::float16_t::f32_to_f16_val(f)
                    : f32_to_bf16(f);
        };
        std::vector<uint16_t> src(static_cast<size_t>(M) * H);
        std::vector<uint16_t> w13_raw(static_cast<size_t>(E) * 2 * I * H);
        std::vector<uint16_t> w2_raw(static_cast<size_t>(E) * H * I);
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = from_f32(
                    static_cast<float>(static_cast<int>(i % 7) - 3) / 8.f);
        }
        for (size_t i = 0; i < w13_raw.size(); ++i) {
            w13_raw[i] = from_f32(
                    static_cast<float>(static_cast<int>(i % 13) - 6) / 64.f);
        }
        for (size_t i = 0; i < w2_raw.size(); ++i) {
            w2_raw[i] = from_f32(
                    static_cast<float>(static_cast<int>(i % 9) - 4) / 48.f);
        }

        // The caller asks for WEIGHT_CACHE=2 on both projections.
        auto run = [&](std::vector<uint16_t> &w13, std::vector<uint16_t> &w2,
                           std::vector<uint16_t> &out) {
            auto projection = [dt](std::vector<uint16_t> &w, int n, int k) {
                group_matmul_projection_params p;
                p.output_size = n;
                p.input_size = k;
                p.weight = w.data();
                p.ldb = k;
                p.params.dtypes.src = dt;
                p.params.dtypes.wei = dt;
                p.params.dtypes.dst = dt;
                p.params.num_threads = 8;
                p.params.weight_cache_type = 2;
                return p;
            };
            const auto primary = projection(w13, 2 * I, H);
            const auto secondary = projection(w2, H, I);
            out.assign(static_cast<size_t>(M) * H, 0);
            return routed_fused_moe_direct('r', false, src.data(), H, M, E,
                    topk, out.data(), H, primary, routing, &secondary, &act);
        };

        std::vector<uint16_t> ref, out, again;
        reset_grp_matmul_caches();
        {
            routed_weight_cache_mode_guard restore;
            restore.config.set_weight_cache(1);
            auto w13 = w13_raw;
            auto w2 = w2_raw;
            ASSERT_EQ(run(w13, w2, ref), status_t::success);
            EXPECT_TRUE(w13 == w13_raw && w2 == w2_raw)
                    << "WEIGHT_CACHE=1 must leave the weights raw";
        }
        reset_grp_matmul_caches();
        routed_weight_cache_mode_guard wc2;
        auto w13 = w13_raw;
        auto w2 = w2_raw;
        ASSERT_EQ(run(w13, w2, out), status_t::success);
        EXPECT_FALSE(w13 == w13_raw)
                << "the gate/up pack must be written in place";
        EXPECT_FALSE(w2 == w2_raw) << "the down pack must be written in place";
        EXPECT_TRUE(out == ref);
        ASSERT_EQ(run(w13, w2, again), status_t::success);
        EXPECT_TRUE(again == ref);
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

// ===========================================================================
// N-tile flat parallel decode (ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO=4)
// ===========================================================================

TEST_F(RoutedMoENTileFlatParallel,
        MatchesReferenceAcrossGeometriesAndDecodeTokens) {
    for (const geometry &g : kGeometries) {
        for (const int64_t M : {1, 2, 3, 8, 31}) {
            SCOPED_TRACE(std::string(g.name) + " M=" + std::to_string(M));
            // Each case's buffers can reuse its predecessor's addresses.
            group_matmul_routed_moe_flush_weight_cache();
            reset_grp_matmul_caches();
            moe_problem pb;
            pb.build(M, g.K, g.N, g.E, g.topk,
                    0x5a00u + static_cast<uint32_t>(M));
            da8w8_call call(pb);
            ntile_flat_parallel_decode_guard guard;
            poison(pb.dst);
            ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
            EXPECT_EQ(guard.calls(), 1u);
            expect_output_written(pb.dst);
            EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
        }
    }
}

TEST_F(RoutedMoENTileFlatParallel, OneExpertOwnsEveryRowAndTheRestAreInactive) {
    moe_problem pb;
    pb.build(19, 128, 64, 8, 2, 0x5b01u);
    std::fill(pb.ids.begin(), pb.ids.end(), 3);
    da8w8_call call(pb);
    ntile_flat_parallel_decode_guard guard;
    poison(pb.dst);
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_EQ(guard.calls(), 1u);
    expect_output_written(pb.dst);
    EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
}

// A call is decode-shaped while no expert has more than kDecodeMaxM rows,
// whatever its token count.
TEST_F(RoutedMoENTileFlatParallel, DecodeShapeEndsAtTheDecodeRowLimit) {
    const int limit = zendnnl::lowoha::matmul::kDecodeMaxM;
    for (const int rows : {limit, limit + 1}) {
        SCOPED_TRACE("rows=" + std::to_string(rows));
        group_matmul_routed_moe_flush_weight_cache();
        reset_grp_matmul_caches();
        moe_problem pb;
        pb.build(rows, 128, 64, 2, 1, 0x5b0eu);
        std::fill(pb.ids.begin(), pb.ids.end(), 0);
        da8w8_call call(pb);
        ntile_flat_parallel_decode_guard guard;
        poison(pb.dst);
        ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
        EXPECT_EQ(guard.calls(), rows <= limit ? 1u : 0u);
        expect_output_written(pb.dst);
        EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
    }
}

// N-tile flat parallel keeps one pack per expert, so a layer of any expert
// count runs on it.
TEST_F(RoutedMoENTileFlatParallel, ServesLayersOfAnyExpertCount) {
    for (const int experts : {1, 256, 257, 300}) {
        SCOPED_TRACE("experts=" + std::to_string(experts));
        group_matmul_routed_moe_flush_weight_cache();
        reset_grp_matmul_caches();
        moe_problem pb;
        pb.build(8, 128, 64, experts, std::min(experts, 8), 0x5b0fu);
        da8w8_call call(pb);
        ntile_flat_parallel_decode_guard guard;
        poison(pb.dst);
        ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
        EXPECT_EQ(guard.calls(), 1u);
        expect_output_written(pb.dst);
        EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
    }
}

/// The most slots any expert of `pb` takes.
int64_t max_expert_rows(const moe_problem &pb) {
    std::vector<int64_t> rows(static_cast<size_t>(pb.E), 0);
    for (const int32_t e : pb.ids)
        ++rows[static_cast<size_t>(e)];
    return *std::max_element(rows.begin(), rows.end());
}

// Hidden and intermediate sizes off the kernels' blocks: W2's columns end
// inside a block, W13's up columns start inside a kernel vector, and K ends
// inside a dot-product group.
TEST_F(RoutedMoENTileFlatParallel, UnalignedShapesMatchTheReference) {
    struct shape_t {
        int64_t K, N;
    };
    for (const shape_t s : {shape_t {200, 72}, shape_t {96, 100},
                 shape_t {130, 37}, shape_t {72, 200}}) {
        for (const routed_moe_activation_t act :
                {routed_moe_activation_t::silu_and_mul,
                        routed_moe_activation_t::gelu_and_mul}) {
            for (const int64_t tokens : {5, 200, 520}) {
                SCOPED_TRACE(::testing::Message()
                        << "H=" << s.K << " I=" << s.N << " tokens=" << tokens
                        << (act == routed_moe_activation_t::gelu_and_mul
                                           ? " gelu"
                                           : " silu"));
                group_matmul_routed_moe_flush_weight_cache();
                reset_grp_matmul_caches();
                moe_problem pb;
                pb.build(tokens, s.K, s.N, 8, 2, 0x5b60u);
                if (tokens == 520) {
                    ASSERT_GT(max_expert_rows(pb),
                            zendnnl::lowoha::matmul::kDecodeMaxM);
                }
                pb.unit_gate();
                pb.act = act;
                da8w8_call call(pb);
                ntile_flat_parallel_decode_guard guard;
                selector_pin prompt(
                        zendnnl::lowoha::matmul::test_api::
                                s_grp_matmul_auto_prompt_algo_override,
                        4);
                poison(pb.dst);
                ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
                EXPECT_EQ(guard.calls(), 1u);
                expect_output_written(pb.dst);
                EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
            }
        }
    }
    group_matmul_routed_moe_flush_weight_cache();
}

// With `unit_gate()` the gate is O(1), where SiLU and GELU differ enough
// that each output misses the other activation's reference.
TEST_F(RoutedMoENTileFlatParallel, GatedActivationsMatchTheirReference) {
    const routed_moe_activation_t acts[]
            = {routed_moe_activation_t::silu_and_mul,
                    routed_moe_activation_t::gelu_and_mul};
    for (const routed_moe_activation_t act : acts) {
        SCOPED_TRACE(
                act == routed_moe_activation_t::silu_and_mul ? "silu" : "gelu");
        group_matmul_routed_moe_flush_weight_cache();
        moe_problem pb;
        pb.build(8, 256, 96, 32, 8, 0x5b08u);
        pb.unit_gate();
        pb.act = act;
        da8w8_call call(pb);
        ntile_flat_parallel_decode_guard guard;
        poison(pb.dst);
        ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
        EXPECT_EQ(guard.calls(), 1u);
        expect_output_written(pb.dst);
        EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
        moe_problem other = pb;
        other.act = act == routed_moe_activation_t::silu_and_mul
                ? routed_moe_activation_t::gelu_and_mul
                : routed_moe_activation_t::silu_and_mul;
        EXPECT_GT(rel_mae(pb.dst, reference_moe(other)), 3e-2);
    }
}

// The routed reference has no SwiGLU-OAI, so the default route is the oracle.
TEST_F(RoutedMoENTileFlatParallel, SwigluOaiMatchesTheDefaultRoute) {
    moe_problem pb;
    pb.build(8, 256, 96, 32, 8, 0x5b09u);
    pb.unit_gate();
    da8w8_call call(pb);
    call.act.act
            = zendnnl::lowoha::matmul::grp_matmul_gated_act_t::swiglu_oai_mul;
    std::vector<uint16_t> routed_output(pb.dst.size(), 0);
    ASSERT_EQ(call.run(pb, routed_output.data()), status_t::success);
    group_matmul_routed_moe_flush_weight_cache();
    reset_grp_matmul_caches();

    ntile_flat_parallel_decode_guard guard;
    poison(pb.dst);
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_EQ(guard.calls(), 1u);
    expect_output_written(pb.dst);
    EXPECT_LT(rel_mae(pb.dst, routed_output), 3e-3);
}

// Biases on both projections under each gated activation, as in GPT-OSS, and
// on the down projection alone: N-tile flat parallel serves them in decode and
// prompt calls and matches the reference, or for SwiGLU-OAI, which the
// reference lacks, the default route; the same call without biases misses it.
TEST_F(RoutedMoENTileFlatParallel, BiasesMatchTheDefaultRoute) {
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
    struct bias_case {
        const char *name;
        grp_matmul_gated_act_t act;
        bool gate_up;
        data_type_t dt;
        bool prompt;
    };
    const bias_case cases[] = {
            {"SwiGLU-OAI, BF16 biases", grp_matmul_gated_act_t::swiglu_oai_mul,
                    true, data_type_t::bf16, false},
            {"SwiGLU-OAI, F32 biases", grp_matmul_gated_act_t::swiglu_oai_mul,
                    true, data_type_t::f32, false},
            {"SiLU, BF16 down bias", grp_matmul_gated_act_t::silu_and_mul,
                    false, data_type_t::bf16, false},
            {"SiLU, BF16 biases", grp_matmul_gated_act_t::silu_and_mul, true,
                    data_type_t::bf16, false},
            {"GELU, F32 biases", grp_matmul_gated_act_t::gelu_and_mul, true,
                    data_type_t::f32, false},
            {"SwiGLU-OAI, BF16 biases, prompt",
                    grp_matmul_gated_act_t::swiglu_oai_mul, true,
                    data_type_t::bf16, true},
            {"GELU, BF16 biases, prompt", grp_matmul_gated_act_t::gelu_and_mul,
                    true, data_type_t::bf16, true},
    };
    for (const bias_case &c : cases) {
        SCOPED_TRACE(c.name);
        group_matmul_routed_moe_flush_weight_cache();
        reset_grp_matmul_caches();
        moe_problem pb;
        if (c.prompt) {
            pb.build(200, 256, 96, 4, 2, 0x5b0cu);
        } else {
            pb.build(8, 256, 96, 32, 8, 0x5b0cu);
        }
        pb.unit_gate();
        da8w8_call call(pb);
        call.act.act = c.act;
        std::vector<uint16_t> unbiased(pb.dst.size(), 0);
        ASSERT_EQ(call.run(pb, unbiased.data()), status_t::success);

        std::mt19937 rng(0x5b0du);
        std::uniform_real_distribution<float> unit(-0.5f, 0.5f);
        std::vector<float> b13(static_cast<size_t>(pb.E * 2 * pb.N));
        std::vector<float> b2(static_cast<size_t>(pb.E * pb.K));
        for (auto *b : {&b13, &b2}) {
            for (auto &v : *b) {
                v = bf16_to_f32(f32_to_bf16(unit(rng)));
            }
        }
        std::vector<uint16_t> b13_bf16, b2_bf16;
        for (const float v : b13)
            b13_bf16.push_back(f32_to_bf16(v));
        for (const float v : b2)
            b2_bf16.push_back(f32_to_bf16(v));
        const bool f32 = c.dt == data_type_t::f32;
        if (c.gate_up) {
            call.primary.bias = f32
                    ? static_cast<const void *>(b13.data())
                    : static_cast<const void *>(b13_bf16.data());
            call.primary.params.dtypes.bias = c.dt;
        }
        call.secondary.bias = f32 ? static_cast<const void *>(b2.data())
                                  : static_cast<const void *>(b2_bf16.data());
        call.secondary.params.dtypes.bias = c.dt;
        group_matmul_routed_moe_flush_weight_cache();
        reset_grp_matmul_caches();
        std::vector<uint16_t> routed(pb.dst.size(), 0);
        ASSERT_EQ(call.run(pb, routed.data()), status_t::success);
        group_matmul_routed_moe_flush_weight_cache();
        reset_grp_matmul_caches();

        ntile_flat_parallel_decode_guard guard;
        selector_pin prompt(zendnnl::lowoha::matmul::test_api::
                                    s_grp_matmul_auto_prompt_algo_override,
                4);
        poison(pb.dst);
        ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
        EXPECT_EQ(guard.calls(), 1u);
        expect_output_written(pb.dst);
        if (c.act == grp_matmul_gated_act_t::swiglu_oai_mul) {
            EXPECT_LT(rel_mae(pb.dst, routed), 3e-3);
        } else {
            pb.act = c.act == grp_matmul_gated_act_t::gelu_and_mul
                    ? routed_moe_activation_t::gelu_and_mul
                    : routed_moe_activation_t::silu_and_mul;
            EXPECT_LT(rel_mae(pb.dst,
                              reference_moe(pb, nullptr,
                                      c.gate_up ? b13.data() : nullptr,
                                      b2.data())),
                    3e-3);
        }
        // A down bias alone moves this output by about 3%.
        EXPECT_GT(rel_mae(pb.dst, unbiased), 1e-2);
    }
}

// Each phase follows its own knob, which outranks the global selector, and a
// prompt call N-tile flat parallel does not serve keeps the routed executor's
// output bit for bit.
TEST_F(RoutedMoENTileFlatParallel,
        EachNTileFlatParallelRequestServesItsPhases) {
    namespace mm = zendnnl::lowoha::matmul;
    struct request {
        const char *name;
        int global, prompt, decode;
        bool decode_served, prompt_served;
    };
    // 0 is AUTO for the global selector, and 7 an unset phase setting,
    // whatever the environment sets.
    const request requests[] = {
            {"unset", 0, 7, 7, false, false},
            {"ALGO=4", 4, 7, 7, true, true},
            {"AUTO_DECODE_ALGO=4", 0, 7, 4, true, false},
            {"AUTO_PROMPT_ALGO=4", 0, 4, 7, false, true},
            {"ALGO=3 AUTO_DECODE_ALGO=4", 3, 7, 4, true, false},
            {"ALGO=4 AUTO_DECODE_ALGO=3", 4, 7, 3, false, true},
    };
    moe_problem decode_pb;
    decode_pb.build(8, 256, 96, 32, 8, 0x5b10u);
    moe_problem prompt_pb;
    prompt_pb.build(200, 128, 64, 2, 1, 0x5b11u);
    std::vector<uint16_t> prompt_routed;
    for (const request &r : requests) {
        SCOPED_TRACE(r.name);
        ntile_flat_parallel_decode_guard guard;
        selector_pin global(mm::test_api_algo_override(), r.global);
        selector_pin prompt(
                mm::test_api::s_grp_matmul_auto_prompt_algo_override, r.prompt);
        selector_pin decode(guard.decode_algo.slot, r.decode);

        da8w8_call decode_call(decode_pb);
        poison(decode_pb.dst);
        ASSERT_EQ(decode_call.run(decode_pb, decode_pb.dst.data()),
                status_t::success);
        const unsigned decode_calls = r.decode_served ? 1u : 0u;
        EXPECT_EQ(guard.calls(), decode_calls);
        EXPECT_LT(rel_mae(decode_pb.dst, reference_moe(decode_pb)), 3e-3);

        da8w8_call prompt_call(prompt_pb);
        poison(prompt_pb.dst);
        ASSERT_EQ(prompt_call.run(prompt_pb, prompt_pb.dst.data()),
                status_t::success);
        EXPECT_EQ(guard.calls(), decode_calls + (r.prompt_served ? 1u : 0u));
        EXPECT_LT(rel_mae(prompt_pb.dst, reference_moe(prompt_pb)), 3e-3);
        if (r.prompt_served) { continue; }
        if (prompt_routed.empty()) {
            prompt_routed = prompt_pb.dst; // the "unset" row
        } else {
            EXPECT_EQ(prompt_pb.dst, prompt_routed);
        }
    }
}

// Prompt calls cut each expert's rows into blocks of at most kDecodeMaxM rows;
// a one-row tail block and an expert owning every row included.
TEST_F(RoutedMoENTileFlatParallel, PromptCallsMatchTheReference) {
    namespace mm = zendnnl::lowoha::matmul;
    struct shape_t {
        int64_t tokens, K, N, E, topk;
        bool one_expert;
    };
    const shape_t shapes[]
            = {{65, 128, 64, 8, 8, false}, {200, 256, 96, 4, 2, false},
                    {520, 128, 64, 16, 4, false}, {130, 128, 64, 8, 1, true}};
    uint32_t seed = 0x5b40u;
    for (const shape_t s : shapes) {
        for (const auto act : {routed_moe_activation_t::silu_and_mul,
                     routed_moe_activation_t::gelu_and_mul}) {
            SCOPED_TRACE(::testing::Message()
                    << "tokens=" << s.tokens << " E=" << s.E
                    << " topk=" << s.topk << " one_expert=" << s.one_expert
                    << (act == routed_moe_activation_t::gelu_and_mul
                                       ? " GELU"
                                       : " SiLU"));
            group_matmul_routed_moe_flush_weight_cache();
            reset_grp_matmul_caches();
            moe_problem pb;
            pb.build(s.tokens, s.K, s.N, s.E, s.topk, seed++);
            pb.act = act;
            if (s.one_expert) { std::fill(pb.ids.begin(), pb.ids.end(), 3); }
            ASSERT_GT(max_expert_rows(pb), mm::kDecodeMaxM);
            da8w8_call call(pb);
            ntile_flat_parallel_decode_guard guard;
            selector_pin prompt(
                    mm::test_api::s_grp_matmul_auto_prompt_algo_override, 4);
            poison(pb.dst);
            ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
            EXPECT_EQ(guard.calls(), 1u);
            expect_output_written(pb.dst);
            EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
        }
    }
}

/// Per-group weight scales for `pb`'s layer, gate/up `[E][K / gate_group][2N]`
/// and down `[E][N / down_group][K]`, with source scales per token or, with
/// `src_groups`, per (token, weight group) of each projection.
struct grouped_scales {
    int64_t gate_group = 0, down_group = 0;
    bool src_groups = false;
    std::vector<float> s13, s2;
    /// The call's source scale buffers.
    std::vector<float> src13, src2;

    void build(const moe_problem &pb, const int64_t gate, const int64_t down,
            const bool per_group_source, const uint32_t seed) {
        gate_group = gate;
        down_group = down;
        src_groups = per_group_source;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> sc(0.002f, 0.02f);
        s13.resize(static_cast<size_t>(pb.E * (pb.K / gate) * 2 * pb.N));
        s2.resize(static_cast<size_t>(pb.E * (pb.N / down) * pb.K));
        for (float &v : s13)
            v = sc(rng);
        for (float &v : s2)
            v = sc(rng);
        src13.assign(
                static_cast<size_t>(pb.M * source_groups(pb.K, gate)), 0.f);
        src2.assign(static_cast<size_t>(pb.M * source_groups(pb.N, down)), 0.f);
    }

    int64_t source_groups(const int64_t K, const int64_t group) const {
        return src_groups ? K / group : 1;
    }

    void apply(const moe_problem &pb, da8w8_call &call) {
        auto &q1 = call.primary.params.quant_params;
        auto &q2 = call.secondary.params.quant_params;
        q1.wei_scale.buff = s13.data();
        q1.wei_scale.dims = {pb.E, pb.K / gate_group, 2 * pb.N};
        q2.wei_scale.buff = s2.data();
        q2.wei_scale.dims = {pb.E, pb.N / down_group, pb.K};
        q1.src_scale.buff = src13.data();
        q1.src_scale.dims = {pb.M, source_groups(pb.K, gate_group)};
        q2.src_scale.buff = src2.data();
        q2.src_scale.dims = {pb.M, source_groups(pb.N, down_group)};
    }
};

/// Quantizes `n` values to s8 in runs of `run`, each with its scale.
void quantize_runs(const float *x, const int64_t n, const int64_t run,
        int32_t *q, float *scale) {
    for (int64_t r0 = 0; r0 < n; r0 += run) {
        float amax = 0.f;
        for (int64_t k = r0; k < r0 + run; ++k)
            amax = std::max(amax, std::fabs(x[k]));
        amax = std::max(amax, 1e-7f);
        const float inv = 127.f / amax;
        scale[r0 / run] = amax / 127.f;
        for (int64_t k = r0; k < r0 + run; ++k)
            q[k] = static_cast<int32_t>(std::nearbyintf(x[k] * inv));
    }
}

/// `reference_moe` with `gs`'s per-group weight scales and source scales.
std::vector<uint16_t> reference_moe_grouped(
        const moe_problem &pb, const grouped_scales &gs) {
    const int64_t M = pb.M, K = pb.K, N = pb.N, topk = pb.topk;
    const int64_t gg = gs.gate_group, dg = gs.down_group;
    const int64_t G1 = K / gg, G2 = N / dg;
    const int64_t run1 = gs.src_groups ? gg : K, run2 = gs.src_groups ? dg : N;
    std::vector<uint16_t> out(static_cast<size_t>(M * K), 0);
    std::vector<float> x(static_cast<size_t>(K)), sa(static_cast<size_t>(G1));
    std::vector<float> inter(static_cast<size_t>(N)),
            sa2(static_cast<size_t>(G2)), acc(static_cast<size_t>(K));
    std::vector<int32_t> qa(static_cast<size_t>(K)), qi(static_cast<size_t>(N));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k)
            x[k] = bf16_to_f32(pb.src[m * K + k]);
        quantize_runs(x.data(), K, run1, qa.data(), sa.data());
        std::fill(acc.begin(), acc.end(), 0.f);
        for (int64_t t = 0; t < topk; ++t) {
            const int64_t e = pb.ids[m * topk + t];
            const int8_t *w13e = pb.w13.data() + e * 2 * N * K;
            const float *s13e = gs.s13.data() + e * G1 * 2 * N;
            for (int64_t j = 0; j < N; ++j) {
                float gf = 0.f, uf = 0.f;
                for (int64_t g = 0; g < G1; ++g) {
                    int32_t gd = 0, ud = 0;
                    for (int64_t k = g * gg; k < (g + 1) * gg; ++k) {
                        gd += qa[k] * w13e[j * K + k];
                        ud += qa[k] * w13e[(N + j) * K + k];
                    }
                    const float a = sa[g * gg / run1];
                    gf += static_cast<float>(gd) * a * s13e[g * 2 * N + j];
                    uf += static_cast<float>(ud) * a * s13e[g * 2 * N + N + j];
                }
                const float act
                        = pb.act == routed_moe_activation_t::gelu_and_mul
                        ? gelu_ref(gf)
                        : gf / (1.f + std::exp(-gf));
                inter[j] = bf16_to_f32(f32_to_bf16(act * uf));
            }
            quantize_runs(inter.data(), N, run2, qi.data(), sa2.data());
            const int8_t *w2e = pb.w2.data() + e * K * N;
            const float *s2e = gs.s2.data() + e * G2 * K;
            const float rw = pb.weights[m * topk + t];
            for (int64_t k = 0; k < K; ++k) {
                float df = 0.f;
                for (int64_t g = 0; g < G2; ++g) {
                    int32_t d = 0;
                    for (int64_t j = g * dg; j < (g + 1) * dg; ++j)
                        d += qi[j] * w2e[k * N + j];
                    df += static_cast<float>(d) * sa2[g * dg / run2]
                            * s2e[g * K + k];
                }
                acc[k] += bf16_to_f32(f32_to_bf16(df * rw));
            }
        }
        for (int64_t k = 0; k < K; ++k)
            out[m * K + k] = f32_to_bf16(acc[k]);
    }
    return out;
}

// DA8W8 with per-group weight scales, against source scales per token and
// per (token, group), in decode and prompt calls.  Source magnitudes that
// alternate by group make the two source scalings differ.
TEST_F(RoutedMoENTileFlatParallel, PerGroupScalesMatchTheReference) {
    namespace mm = zendnnl::lowoha::matmul;
    uint32_t seed = 0x5c10u;
    for (const bool src_groups : {false, true}) {
        for (const int64_t tokens : {1, 8, 130, 300, 520}) {
            SCOPED_TRACE(::testing::Message()
                    << "tokens=" << tokens << " src_groups=" << src_groups);
            group_matmul_routed_moe_flush_weight_cache();
            reset_grp_matmul_caches();
            moe_problem pb;
            pb.build(tokens, 256, 96, 8, 2, seed++);
            if (tokens >= 300) {
                ASSERT_GT(max_expert_rows(pb), mm::kDecodeMaxM);
            }
            for (int64_t m = 0; m < pb.M; ++m) {
                for (int64_t k = 32; k < pb.K; k += 64) {
                    for (int64_t i = k; i < k + 32; ++i) {
                        uint16_t &v = pb.src[m * pb.K + i];
                        v = f32_to_bf16(4.f * bf16_to_f32(v));
                    }
                }
            }
            grouped_scales gs;
            gs.build(pb, 32, 32, src_groups, seed++);
            da8w8_call call(pb);
            gs.apply(pb, call);
            ntile_flat_parallel_decode_guard guard;
            selector_pin prompt(
                    mm::test_api::s_grp_matmul_auto_prompt_algo_override, 4);
            poison(pb.dst);
            ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
            EXPECT_EQ(guard.calls(), 1u);
            expect_output_written(pb.dst);
            const double err = rel_mae(pb.dst, reference_moe_grouped(pb, gs));
            EXPECT_LT(err, 3e-3);
            grouped_scales other = gs;
            other.src_groups = !src_groups;
            EXPECT_GT(
                    rel_mae(pb.dst, reference_moe_grouped(pb, other)), 2 * err);
        }
    }
}

// A layer's prompt and decode calls share one layer entry and its packs.
TEST_F(RoutedMoENTileFlatParallel, PromptAndDecodeCallsShareOneLayerEntry) {
    namespace mm = zendnnl::lowoha::matmul;
    moe_problem pb;
    pb.build(200, 128, 64, 4, 2, 0x5b48u);
    ntile_flat_parallel_decode_guard guard;
    selector_pin prompt(
            mm::test_api::s_grp_matmul_auto_prompt_algo_override, 4);
    {
        da8w8_call call(pb);
        poison(pb.dst);
        ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
        EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
    }
    pb.M = 8;
    pb.src.resize(static_cast<size_t>(pb.M * pb.K));
    pb.dst.resize(pb.src.size());
    pb.ids.resize(static_cast<size_t>(pb.M * pb.topk));
    pb.weights.resize(pb.ids.size());
    da8w8_call call(pb);
    poison(pb.dst);
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
    EXPECT_EQ(guard.calls(), 2u);
    EXPECT_EQ(guard.layer_entries(), 1u);
}

// A prompt unit sweeps its expert's blocks one L2-sized chunk of columns at a
// time.  With a 1 MiB L2 one thread's full-width W13 units span two chunks and
// sixteen threads' units one; both write the same bytes.
TEST_F(RoutedMoENTileFlatParallel, PromptUnitsWriteTheSameAcrossTeamSizes) {
    namespace mm = zendnnl::lowoha::matmul;
    moe_problem pb;
    pb.build(320, 4096, 128, 32, 8, 0x5b51u);
    ASSERT_GT(max_expert_rows(pb), mm::kDecodeMaxM);
    da8w8_call call(pb);
    ntile_flat_parallel_decode_guard guard;
    selector_pin prompt(
            mm::test_api::s_grp_matmul_auto_prompt_algo_override, 4);
    std::vector<uint16_t> out[2];
    for (const int threads : {1, 16}) {
        call.primary.params.num_threads = threads;
        call.secondary.params.num_threads = threads;
        std::vector<uint16_t> &o = out[threads == 1 ? 0 : 1];
        o.resize(pb.dst.size());
        poison(o);
        ASSERT_EQ(call.run(pb, o.data()), status_t::success);
    }
    EXPECT_EQ(guard.calls(), 2u);
    expect_output_written(out[0]);
    EXPECT_EQ(out[0], out[1]);
    group_matmul_routed_moe_flush_weight_cache();
}

// Where the kernels that read source bytes XOR-ed with 0x80 pay, N-tile flat
// parallel runs them and flips its s8 rows; decode and prompt calls write what
// the s8-source kernels write.
TEST_F(RoutedMoENTileFlatParallel,
        U8SourceKernelsWriteWhatS8SourceKernelsWrite) {
    namespace mm = zendnnl::lowoha::matmul;
    for (const int64_t tokens : {8, 200}) {
        SCOPED_TRACE(::testing::Message() << "tokens=" << tokens);
        moe_problem pb;
        pb.build(tokens, 256, 128, 4, 2, 0x5b52u);
        da8w8_call call(pb);
        ntile_flat_parallel_decode_guard guard;
        selector_pin prompt(
                mm::test_api::s_grp_matmul_auto_prompt_algo_override, 4);
        std::vector<uint16_t> out[2];
        auto &u8_calls = mm::test_api::s_ntile_flat_parallel_u8_src_calls;
        for (const int u8 : {0, 1}) {
            group_matmul_routed_moe_flush_weight_cache();
            selector_pin u8_src(
                    mm::test_api::s_ntile_flat_parallel_u8_src_override, u8);
            out[u8].resize(pb.dst.size());
            poison(out[u8]);
            const uint64_t before = u8_calls.load();
            ASSERT_EQ(call.run(pb, out[u8].data()), status_t::success);
            EXPECT_EQ(u8_calls.load() - before, u8 ? 1u : 0u)
                    << "the pinned source kernels ran";
        }
        EXPECT_EQ(guard.calls(), 2u);
        expect_output_written(out[0]);
        expect_output_written(out[1]);
        EXPECT_EQ(out[1], out[0]);
    }
    group_matmul_routed_moe_flush_weight_cache();
}

TEST_F(RoutedMoENTileFlatParallel, ReusesItsPacksUntilTheRoutedFlush) {
    moe_problem pb;
    pb.build(8, 256, 96, 32, 8, 0x5b03u);
    da8w8_call call(pb);
    ntile_flat_parallel_decode_guard guard;
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_EQ(guard.calls(), 2u);
    EXPECT_EQ(guard.layer_entries(), 1u);

    // New weights under the same addresses, after the routed flush alone:
    // the documented boundary for a reload.
    for (auto &v : pb.w13) {
        v = static_cast<int8_t>(-v);
    }
    for (auto &v : pb.w2) {
        v = static_cast<int8_t>(-v);
    }
    group_matmul_routed_moe_flush_weight_cache();
    poison(pb.dst);
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_EQ(guard.calls(), 3u);
    EXPECT_EQ(guard.layer_entries(), 2u);
    expect_output_written(pb.dst);
    EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
}

// Scales at a new address with new values, as a host that reloads them: the
// call reads the new values, the layer keeps one entry, and the experts keep
// one pack each.
TEST_F(RoutedMoENTileFlatParallel, MovedWeightScalesReplaceTheLayersState) {
    namespace mm = zendnnl::lowoha::matmul;
    group_matmul_routed_moe_flush_weight_cache();
    moe_problem pb;
    pb.build(8, 128, 64, 16, 2, 0x5b05u);
    da8w8_call call(pb);
    ntile_flat_parallel_decode_guard guard;
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    std::vector<std::vector<float>> moved;
    moved.reserve(6);
    for (int i = 0; i < 3; ++i) {
        for (float &v : pb.s13)
            v *= 1.25f;
        for (float &v : pb.s2)
            v *= 0.8f;
        moved.push_back(pb.s13);
        call.primary.params.quant_params.wei_scale.buff = moved.back().data();
        moved.push_back(pb.s2);
        call.secondary.params.quant_params.wei_scale.buff = moved.back().data();
        poison(pb.dst);
        ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
        expect_output_written(pb.dst);
        EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
    }
    EXPECT_EQ(guard.calls(), 4u);
    EXPECT_EQ(guard.layer_entries(), 4u);
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_layers(), 1u);
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(),
            static_cast<size_t>(2 * pb.E));
    group_matmul_routed_moe_flush_weight_cache();
}

// A pack that runs out of memory fails its call with a memory status and
// writes nothing; the layer is not marked unserved, so the next call packs
// and runs.
TEST_F(RoutedMoENTileFlatParallel,
        RunningOutOfMemoryWhilePackingFailsOnlyThatCall) {
    namespace mm = zendnnl::lowoha::matmul;
    struct pack_failure_pin {
        std::atomic<bool> &fails = mm::ntile_flat_parallel::nr32::test_api::
                s_pack_allocation_fails;
        pack_failure_pin() { fails.store(true); }
        ~pack_failure_pin() { fails.store(false); }
    };
    group_matmul_routed_moe_flush_weight_cache();
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0x5b0au);
    da8w8_call call(pb);
    ntile_flat_parallel_decode_guard guard;
    std::vector<uint16_t> out(pb.dst.size());
    poison(out);
    const std::vector<uint16_t> untouched = out;
    {
        pack_failure_pin fails;
        EXPECT_EQ(call.run(pb, out.data()), status_t::memory_bad_storage);
    }
    EXPECT_EQ(out, untouched);
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_layers(), 0u);
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), 0u);
    ASSERT_EQ(call.run(pb, out.data()), status_t::success);
    EXPECT_EQ(guard.calls(), 1u);
    expect_output_written(out);
    EXPECT_LT(rel_mae(out, reference_moe(pb)), 3e-3);
    group_matmul_routed_moe_flush_weight_cache();
}

// The routed flush waits for an N-tile flat parallel call in flight, which
// reads the layer's packs and the caller's buffers until it returns.
TEST_F(RoutedMoENTileFlatParallel, RoutedFlushWaitsForACallInFlight) {
    namespace mm = zendnnl::lowoha::matmul;
    struct pause_pin {
        std::atomic<bool> &pause = mm::test_api::s_pause_ntile_flat_parallel;
        pause_pin() {
            mm::test_api::s_ntile_flat_parallel_paused.store(false);
            pause.store(true);
        }
        ~pause_pin() { pause.store(false); }
    };
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0x5b0bu);
    da8w8_call call(pb);
    ntile_flat_parallel_decode_guard guard;
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    status_t status = status_t::failure;
    std::atomic<bool> flushed {false};
    std::thread caller, flusher;
    {
        pause_pin pause;
        caller = std::thread([&] { status = call.run(pb, pb.dst.data()); });
        const auto deadline
                = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!mm::test_api::s_ntile_flat_parallel_paused.load()
                && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        EXPECT_TRUE(mm::test_api::s_ntile_flat_parallel_paused.load());
        flusher = std::thread([&] {
            group_matmul_routed_moe_flush_weight_cache();
            flushed.store(true);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        EXPECT_FALSE(flushed.load())
                << "the flush returned while the call was in flight";
    }
    caller.join();
    flusher.join();
    EXPECT_TRUE(flushed.load());
    EXPECT_EQ(status, status_t::success);
    EXPECT_EQ(guard.calls(), 2u);
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_layers(), 0u);
}

// Weights stored [in, out] (`trans_weight = false`), and the 'R' layout, are
// served like the default call, byte for byte.
TEST_F(RoutedMoENTileFlatParallel,
        UntransposedWeightsAndLayoutRMatchTheDefaultCall) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0x5b0du);
    da8w8_call call(pb);
    ntile_flat_parallel_decode_guard guard;
    std::vector<uint16_t> want(pb.dst.size()), out(pb.dst.size());
    poison(want);
    ASSERT_EQ(call.run(pb, want.data()), status_t::success);
    expect_output_written(want);
    EXPECT_LT(rel_mae(want, reference_moe(pb)), 3e-3);

    poison(out);
    ASSERT_EQ(routed_fused_moe_direct('R', false, pb.src.data(),
                      static_cast<int>(pb.K), static_cast<int>(pb.M),
                      static_cast<int>(pb.E), static_cast<int>(pb.topk),
                      out.data(), static_cast<int>(pb.K), call.primary,
                      call.routing, &call.secondary, &call.act),
            status_t::success);
    EXPECT_EQ(out, want);

    const int64_t H = pb.K, I = pb.N;
    std::vector<int8_t> w13(pb.w13.size()), w2(pb.w2.size());
    for (int64_t e = 0; e < pb.E; ++e) {
        for (int64_t n = 0; n < 2 * I; ++n) {
            for (int64_t k = 0; k < H; ++k) {
                w13[(e * H + k) * 2 * I + n] = pb.w13[(e * 2 * I + n) * H + k];
            }
        }
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t i = 0; i < I; ++i) {
                w2[(e * I + i) * H + h] = pb.w2[(e * H + h) * I + i];
            }
        }
    }
    da8w8_call untransposed(pb);
    untransposed.primary.trans_weight = false;
    untransposed.primary.weight = w13.data();
    untransposed.primary.ldb = static_cast<int>(2 * I);
    untransposed.secondary.trans_weight = false;
    untransposed.secondary.weight = w2.data();
    untransposed.secondary.ldb = static_cast<int>(H);
    poison(out);
    ASSERT_EQ(untransposed.run(pb, out.data()), status_t::success);
    EXPECT_EQ(out, want);
    EXPECT_EQ(guard.calls(), 3u);
    group_matmul_routed_moe_flush_weight_cache();
}

// Calls on the same buffers with opposite `trans_weight`, at strides that fit
// either layout, are two layers: the second writes what it writes alone.
TEST_F(RoutedMoENTileFlatParallel,
        OppositeTranspositionsOfOneBufferAreTwoLayers) {
    namespace mm = zendnnl::lowoha::matmul;
    group_matmul_routed_moe_flush_weight_cache();
    // H = 2I, so W13's rows hold H values either way; W2's are padded to H.
    moe_problem pb;
    pb.build(8, 128, 64, 4, 2, 0x5b12u);
    const int64_t E = pb.E, H = pb.K, I = pb.N;
    std::vector<int8_t> w2(static_cast<size_t>(E * H * H), 0);
    for (int64_t e = 0; e < E; ++e) {
        for (int64_t h = 0; h < H; ++h) {
            std::copy_n(&pb.w2[(e * H + h) * I], I, &w2[(e * H + h) * H]);
        }
    }
    da8w8_call call(pb);
    call.primary.wei_buffer_capacity_bytes = static_cast<size_t>(2 * I * H);
    call.secondary.weight = w2.data();
    call.secondary.ldb = static_cast<int>(H);
    call.secondary.wei_buffer_capacity_bytes = static_cast<size_t>(H * H);
    da8w8_call other = call;
    other.primary.trans_weight = false;
    other.primary.ldb = static_cast<int>(2 * I);
    other.secondary.trans_weight = false;
    ntile_flat_parallel_decode_guard guard;
    std::vector<uint16_t> out(pb.dst.size()), alone(pb.dst.size());
    ASSERT_EQ(call.run(pb, out.data()), status_t::success);
    poison(out);
    ASSERT_EQ(other.run(pb, out.data()), status_t::success);
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_layers(), 2u);
    group_matmul_routed_moe_flush_weight_cache();
    poison(alone);
    ASSERT_EQ(other.run(pb, alone.data()), status_t::success);
    EXPECT_EQ(out, alone);
    EXPECT_EQ(guard.calls(), 3u);
    group_matmul_routed_moe_flush_weight_cache();
}

// Both decode schedules, at every team size, write the same bytes.
TEST_F(RoutedMoENTileFlatParallel,
        DecodeSchedulesAndTeamSizesWriteTheSameBytes) {
    namespace mm = zendnnl::lowoha::matmul;
    moe_problem pb;
    pb.build(8, 256, 96, 32, 8, 0x5b0eu);
    da8w8_call call(pb);
    ntile_flat_parallel_decode_guard guard;
    std::vector<uint16_t> first;
    for (const int threads : {1, 3, 7}) {
        for (const int own : {0, 1}) {
            SCOPED_TRACE(::testing::Message()
                    << threads << " threads own_units_first=" << own);
            selector_pin schedule(
                    mm::test_api::s_ntile_flat_parallel_own_units_override,
                    own);
            call.primary.params.num_threads = threads;
            call.secondary.params.num_threads = threads;
            std::vector<uint16_t> out(pb.dst.size());
            poison(out);
            ASSERT_EQ(call.run(pb, out.data()), status_t::success);
            expect_output_written(out);
            if (first.empty()) {
                first = out;
                EXPECT_LT(rel_mae(first, reference_moe(pb)), 3e-3);
            } else {
                EXPECT_EQ(out, first);
            }
        }
    }
    EXPECT_EQ(guard.calls(), 6u);
    group_matmul_routed_moe_flush_weight_cache();
}

/// Restores the environment's N-tile flat-parallel config when it goes.
struct config_restore {
    ~config_restore() {
        zendnnl::lowoha::matmul::test_api::set_ntile_flat_parallel_config(
                nullptr);
    }
};

// The settings change how a routed call's work is cut, ordered and staged,
// not what it writes: DA8W8 decode and prompt calls write the same bytes
// under each, through the shared gather of the source rows too.
TEST_F(RoutedMoENTileFlatParallel, SettingsWriteTheSameBytes) {
    namespace mm = zendnnl::lowoha::matmul;
    const char *const configs[] = {"", "gather_rows=1", "sweep_rows=1",
            "sweep_rows=100000", "l2_frac=0.25", "subtile_rows=1", "u8_src=on",
            "u8_src=off", "units=1,prompt_units=1", "unit_order=own",
            "unit_order=queue", "heavy_light=off"};
    const config_restore restore;
    for (const int64_t tokens : {int64_t {8}, int64_t {300}}) {
        moe_problem pb;
        pb.build(
                tokens, 256, 96, 8, 2, 0x5b60u + static_cast<uint32_t>(tokens));
        ASSERT_EQ(max_expert_rows(pb) > mm::kDecodeMaxM, tokens > 8);
        da8w8_call call(pb);
        ntile_flat_parallel_decode_guard guard;
        selector_pin prompt(
                mm::test_api::s_grp_matmul_auto_prompt_algo_override, 4);
        std::vector<uint16_t> first;
        for (const char *config : configs) {
            SCOPED_TRACE(::testing::Message()
                    << "tokens=" << tokens << " config='" << config << "'");
            group_matmul_routed_moe_flush_weight_cache();
            mm::test_api::set_ntile_flat_parallel_config(config);
            std::vector<uint16_t> out(pb.dst.size());
            poison(out);
            ASSERT_EQ(call.run(pb, out.data()), status_t::success);
            expect_output_written(out);
            if (first.empty()) {
                first = out;
                EXPECT_LT(rel_mae(first, reference_moe(pb)), 3e-3);
            } else {
                EXPECT_EQ(out, first);
            }
        }
        EXPECT_EQ(guard.calls(), std::size(configs));
    }
    group_matmul_routed_moe_flush_weight_cache();
}

// Source and output rows with padding past their K values: the padding of
// the source is never read, and that of the output never written.
TEST_F(RoutedMoENTileFlatParallel, PaddedRowsMatchTheUnpaddedCall) {
    namespace mm = zendnnl::lowoha::matmul;
    for (const int64_t tokens : {8, 300}) {
        SCOPED_TRACE(::testing::Message() << "tokens=" << tokens);
        group_matmul_routed_moe_flush_weight_cache();
        moe_problem pb;
        pb.build(tokens, 128, 64, 4, 2, 0x5b0cu);
        if (tokens > 8) { ASSERT_GT(max_expert_rows(pb), mm::kDecodeMaxM); }
        da8w8_call call(pb);
        ntile_flat_parallel_decode_guard guard;
        selector_pin prompt(
                mm::test_api::s_grp_matmul_auto_prompt_algo_override, 4);
        std::vector<uint16_t> want(pb.dst.size());
        poison(want);
        ASSERT_EQ(call.run(pb, want.data()), status_t::success);
        const int src_ld = static_cast<int>(pb.K) + 16;
        const int out_ld = static_cast<int>(pb.K) + 24;
        std::vector<uint16_t> src(static_cast<size_t>(pb.M) * src_ld,
                f32_to_bf16(std::numeric_limits<float>::quiet_NaN()));
        for (int64_t m = 0; m < pb.M; ++m) {
            std::copy_n(&pb.src[m * pb.K], pb.K, &src[m * src_ld]);
        }
        std::vector<uint16_t> out(static_cast<size_t>(pb.M) * out_ld);
        poison(out);
        const std::vector<uint16_t> untouched = out;
        ASSERT_EQ(
                routed_fused_moe_direct('r', false, src.data(), src_ld,
                        static_cast<int>(pb.M), static_cast<int>(pb.E),
                        static_cast<int>(pb.topk), out.data(), out_ld,
                        call.primary, call.routing, &call.secondary, &call.act),
                status_t::success);
        EXPECT_EQ(guard.calls(), 2u);
        for (int64_t m = 0; m < pb.M; ++m) {
            ASSERT_TRUE(std::equal(
                    &want[m * pb.K], &want[m * pb.K] + pb.K, &out[m * out_ld]))
                    << "row " << m;
            ASSERT_TRUE(std::equal(&out[m * out_ld] + pb.K,
                    &out[m * out_ld] + out_ld, &untouched[m * out_ld] + pb.K))
                    << "row " << m << "'s padding";
        }
    }
    group_matmul_routed_moe_flush_weight_cache();
}

// With `skip_weighted` the top-1 router weights are not applied.
TEST_F(RoutedMoENTileFlatParallel, SkipWeightedLeavesTheRouterWeightsOut) {
    moe_problem pb;
    pb.build(8, 128, 64, 8, 1, 0x5b0fu);
    std::fill(pb.weights.begin(), pb.weights.end(), 0.37f);
    da8w8_call call(pb);
    call.routing.skip_weighted = true;
    ntile_flat_parallel_decode_guard guard;
    poison(pb.dst);
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_EQ(guard.calls(), 1u);
    expect_output_written(pb.dst);
    const std::vector<uint16_t> weighted = reference_moe(pb);
    std::fill(pb.weights.begin(), pb.weights.end(), 1.0f);
    const double err = rel_mae(pb.dst, reference_moe(pb));
    EXPECT_LT(err, 3e-3);
    EXPECT_GT(rel_mae(pb.dst, weighted), 100 * err);
    group_matmul_routed_moe_flush_weight_cache();
}

/// `pb`'s layer as one fused `group_matmul_direct` call of its experts, each
/// fed the rows of the tokens routed to it in token order and reading its
/// slice of the routed call's weight and scale buffers; the weighted reduce
/// goes to `out`.
status_t run_layer_grouped(const moe_problem &pb, std::vector<uint16_t> &out) {
    namespace mm = zendnnl::lowoha::matmul;
    const int E = static_cast<int>(pb.E), H = static_cast<int>(pb.K);
    const int I = static_cast<int>(pb.N), T = static_cast<int>(pb.M);
    const int topk = static_cast<int>(pb.topk);
    std::vector<std::vector<int>> tokens(E);
    std::vector<std::pair<int, int>> slot_rows;
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < topk; ++k) {
            const int e = pb.ids[static_cast<size_t>(t) * topk + k];
            slot_rows.push_back({e, static_cast<int>(tokens[e].size())});
            tokens[e].push_back(t);
        }
    }
    std::vector<std::vector<uint16_t>> src(E), op1(E), down(E);
    std::vector<std::vector<float>> src_scale(E);
    std::vector<char> layout(E, 'r');
    std::vector<bool> transA(E, false), transB(E, true), wconst(E, true);
    std::vector<int> M(E), N(E, 2 * I), K(E, H), lda(E, H), ldb(E, H),
            ldc(E, 2 * I);
    std::vector<float> alpha(E, 1.0f), beta(E, 0.0f);
    std::vector<const void *> srcv(E), wei(E), bias(E, nullptr);
    std::vector<void *> dst(E);
    std::vector<mm::matmul_params> params(E);
    mm::grp_matmul_fused_moe_params fused;
    for (int e = 0; e < E; ++e) {
        M[e] = static_cast<int>(tokens[e].size());
        const size_t rows = static_cast<size_t>(std::max(M[e], 1));
        src[e].assign(rows * H, 0);
        for (int j = 0; j < M[e]; ++j) {
            std::copy_n(pb.src.data() + static_cast<size_t>(tokens[e][j]) * H,
                    H, src[e].data() + static_cast<size_t>(j) * H);
        }
        src_scale[e].assign(rows, 0.0f);
        op1[e].assign(rows * 2 * I, 0);
        down[e].assign(rows * H, 0);
        srcv[e] = src[e].data();
        wei[e] = pb.w13.data() + static_cast<size_t>(e) * 2 * I * H;
        dst[e] = op1[e].data();
        mm::matmul_params &p = params[e];
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::bf16;
        p.dtypes.compute = data_type_t::s8;
        p.dynamic_quant = true;
        p.weight_cache_type = 1;
        p.quant_params.src_scale.buff = src_scale[e].data();
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.dims = {static_cast<int64_t>(rows), 1};
        p.quant_params.wei_scale.buff
                = pb.s13.data() + static_cast<size_t>(e) * 2 * I;
        p.quant_params.wei_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.dims = {1, 2 * I};
        fused.down_weight.push_back(
                pb.w2.data() + static_cast<size_t>(e) * H * I);
        fused.N_down.push_back(H);
        fused.ldb_down.push_back(I);
        fused.bias_down.push_back(nullptr);
        mm::grp_matmul_fused_moe_params::down_weight_quant_t scale;
        scale.buff = pb.s2.data() + static_cast<size_t>(e) * H;
        scale.dt = data_type_t::f32;
        scale.dims = {1, H};
        fused.down_scale.push_back(scale);
        fused.dst_down.push_back(down[e].data());
        fused.ldc_down.push_back(H);
    }
    std::vector<const void *> row_ptrs;
    for (const auto &[e, j] : slot_rows)
        row_ptrs.push_back(down[e].data() + static_cast<size_t>(j) * H);
    out.assign(static_cast<size_t>(T) * H, 0);
    mm::group_matmul_moe_postop_params reduce;
    reduce.num_tokens = T;
    reduce.topk = topk;
    reduce.output = out.data();
    reduce.ldc_output = H;
    reduce.topk_weights = pb.weights.data();
    reduce.row_ptrs = row_ptrs.data();
    mm::grp_matmul_gated_act_params act;
    act.act = pb.act == routed_moe_activation_t::gelu_and_mul
            ? mm::grp_matmul_gated_act_t::gelu_and_mul
            : mm::grp_matmul_gated_act_t::silu_and_mul;
    return mm::group_matmul_direct(layout, transA, transB, M, N, K, alpha, srcv,
            lda, wei, ldb, bias, beta, dst, ldc, wconst, params, &reduce, &act,
            &fused);
}

// A routed layer and a grouped call of the same experts run on one set of
// packs, whichever call packs them, and write the same output; either flush
// drops every pack and the routed layer's state.
TEST_F(RoutedMoENTileFlatParallel, RoutedAndGroupedCallsShareOnePackPerWeight) {
    namespace mm = zendnnl::lowoha::matmul;
    moe_problem pb;
    pb.build(8, 256, 96, 8, 2, 0x5b07u);
    // Two rows per expert, so either call reads every weight.
    for (int64_t t = 0; t < pb.M; ++t) {
        pb.ids[t * 2] = static_cast<int32_t>(t % pb.E);
        pb.ids[t * 2 + 1] = static_cast<int32_t>((t + 3) % pb.E);
    }
    da8w8_call call(pb);
    ntile_flat_parallel_decode_guard guard;
    std::atomic<uint64_t> &grouped
            = mm::test_api::s_ntile_flat_parallel_grouped_completed;
    grouped.store(0, std::memory_order_relaxed);
    group_matmul_routed_moe_flush_weight_cache();
    const size_t every_weight = 2 * static_cast<size_t>(pb.E);
    for (const bool routed_first : {true, false}) {
        SCOPED_TRACE(routed_first ? "routed first" : "grouped first");
        ASSERT_EQ(mm::test_api::ntile_flat_parallel_packs(), 0u);
        std::vector<uint16_t> out;
        poison(pb.dst);
        if (routed_first) {
            ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
            EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), every_weight);
            ASSERT_EQ(run_layer_grouped(pb, out), status_t::success);
        } else {
            ASSERT_EQ(run_layer_grouped(pb, out), status_t::success);
            EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), every_weight);
            ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
        }
        EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), every_weight);
        EXPECT_EQ(mm::test_api::ntile_flat_parallel_layers(), 1u);
        expect_output_written(pb.dst);
        EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
        EXPECT_EQ(out, pb.dst);
        if (routed_first) {
            mm::clear_grp_matmul_weight_caches();
        } else {
            group_matmul_routed_moe_flush_weight_cache();
        }
        EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), 0u);
        EXPECT_EQ(mm::test_api::ntile_flat_parallel_layers(), 0u);
    }
    EXPECT_EQ(guard.calls(), 2u);
    EXPECT_EQ(guard.layer_entries(), 2u);
    EXPECT_EQ(grouped.load(std::memory_order_relaxed), 2u);
}

// A layer called with a second activation keeps a second routed state, with
// its own plans, over the packs the first one made.
TEST_F(RoutedMoENTileFlatParallel, ASecondActivationReusesTheLayersPacks) {
    namespace mm = zendnnl::lowoha::matmul;
    moe_problem pb;
    pb.build(8, 128, 64, 16, 2, 0x5b09u);
    pb.unit_gate();
    da8w8_call call(pb);
    ntile_flat_parallel_decode_guard guard;
    group_matmul_routed_moe_flush_weight_cache();
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
    const size_t packs = mm::test_api::ntile_flat_parallel_packs();
    EXPECT_EQ(packs, 2 * static_cast<size_t>(pb.E));
    pb.act = routed_moe_activation_t::gelu_and_mul;
    call.act.act = mm::grp_matmul_gated_act_t::gelu_and_mul;
    poison(pb.dst);
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    expect_output_written(pb.dst);
    EXPECT_LT(rel_mae(pb.dst, reference_moe(pb)), 3e-3);
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), packs);
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_layers(), 2u);
    EXPECT_EQ(guard.calls(), 2u);
    EXPECT_EQ(guard.layer_entries(), 2u);
    group_matmul_routed_moe_flush_weight_cache();
}

// A call N-tile flat parallel owns and cannot serve fails before it writes
// anything, rather than running elsewhere; the control row shows the setup is
// one N-tile flat parallel serves.  Weights a caller may change between calls
// without a flush are among them.
TEST_F(RoutedMoENTileFlatParallel, CallsItCannotServeFail) {
    const auto unchanged = [](da8w8_call &) {};
    const auto non_constant = [](da8w8_call &c) {
        c.primary.weight_is_const = false;
        c.secondary.weight_is_const = false;
    };
    const auto scale_dims = [](const std::vector<int64_t> &dims) {
        return [dims](da8w8_call &c) {
            c.primary.params.quant_params.wei_scale.dims = dims;
            c.secondary.params.quant_params.wei_scale.dims = dims;
        };
    };
    float output_scale = 2.0f;
    const auto destination_scale = [&output_scale](da8w8_call &c) {
        auto &s = c.secondary.params.quant_params.dst_scale;
        s.buff = &output_scale;
        s.dt = data_type_t::f32;
        s.dims = {1};
    };
    std::vector<int32_t> map(8);
    const auto non_resident = [&map](da8w8_call &c) {
        for (int e = 0; e < 8; ++e) {
            map[e] = e;
        }
        map[c.routing.topk_ids[0]] = -1;
        c.routing.expert_map = map.data();
        c.routing.expert_map_size = 8;
    };
    int32_t zero_point = 1;
    const auto weight_zero_point = [&zero_point](da8w8_call &c) {
        auto &zp = c.primary.params.quant_params.wei_zp;
        zp.buff = &zero_point;
        zp.dt = data_type_t::s32;
        zp.dims = {1};
    };
    const auto post_op = [](da8w8_call &c) {
        zendnnl::lowoha::matmul::matmul_post_op relu;
        relu.po_type = zendnnl::common::post_op_type_t::relu;
        c.secondary.params.postop_.push_back(relu);
    };
    struct unserved_case {
        const char *name;
        int cache_mode;
        std::function<void(da8w8_call &)> edit;
        char layout = 'r';
        bool trans_src = false;
    };
    const unserved_case cases[] = {
            {"control", 1, unchanged},
            {"non-constant weights", 1, non_constant},
            {"one weight scale per expert", 1, scale_dims({8, 1})},
            {"one weight scale per tensor", 1, scale_dims({1})},
            {"destination scale", 1, destination_scale},
            {"weight cache mode 0", 0, unchanged},
            {"non-resident expert", 1, non_resident},
            {"unreduced output", 1,
                    [](da8w8_call &c) { c.routing.reduce_output = false; }},
            {"column-major source", 1, unchanged, 'c'},
            {"transposed source", 1, unchanged, 'r', true},
            {"alpha", 1, [](da8w8_call &c) { c.primary.alpha = 2.0f; }},
            {"weight zero point", 1, weight_zero_point},
            {"post-op", 1, post_op},
            {"a compute type other than the kernels' s8", 1,
                    [](da8w8_call &c) {
        c.primary.params.dtypes.compute = data_type_t::bf16;
        c.secondary.params.dtypes.compute = data_type_t::bf16;
    }},
    };
    for (const unserved_case &c : cases) {
        SCOPED_TRACE(c.name);
        const bool control = c.name == std::string("control");
        group_matmul_routed_moe_flush_weight_cache();
        reset_grp_matmul_caches();
        moe_problem pb;
        pb.build(8, 128, 64, 8, 2, 0x5b06u);
        da8w8_call call(pb);
        c.edit(call);
        ntile_flat_parallel_decode_guard guard;
        weight_cache_mode_pin mode(c.cache_mode);
        // Room for an unreduced output's row per routed slot.
        std::vector<uint16_t> out(pb.dst.size() * pb.topk);
        poison(out);
        const std::vector<uint16_t> untouched = out;
        const status_t status = routed_fused_moe_direct(c.layout, c.trans_src,
                pb.src.data(), static_cast<int>(pb.K), static_cast<int>(pb.M),
                static_cast<int>(pb.E), static_cast<int>(pb.topk), out.data(),
                static_cast<int>(pb.K), call.primary, call.routing,
                &call.secondary, &call.act);
        out.resize(pb.dst.size() * (control ? 1 : pb.topk));
        EXPECT_EQ(guard.calls(), control ? 1u : 0u);
        if (control) {
            EXPECT_EQ(status, status_t::success);
            EXPECT_LT(rel_mae(out, reference_moe(pb)), 3e-3);
        } else {
            EXPECT_EQ(status, status_t::unimplemented);
            EXPECT_EQ(out, untouched);
        }
    }

    // W13 without an activation: its output is W2's input.
    SCOPED_TRACE("no gated activation");
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0x5b06u);
    da8w8_call call(pb);
    call.primary.output_size /= 2;
    ntile_flat_parallel_decode_guard guard;
    std::vector<uint16_t> out(pb.dst.size());
    poison(out);
    const std::vector<uint16_t> untouched = out;
    EXPECT_EQ(routed_fused_moe_direct('r', false, pb.src.data(),
                      static_cast<int>(pb.K), static_cast<int>(pb.M),
                      static_cast<int>(pb.E), static_cast<int>(pb.topk),
                      out.data(), static_cast<int>(pb.K), call.primary,
                      call.routing, &call.secondary, nullptr),
            status_t::unimplemented);
    EXPECT_EQ(guard.calls(), 0u);
    EXPECT_EQ(out, untouched);
    group_matmul_routed_moe_flush_weight_cache();
}

// Packing a layer warms no other weight layout: no AOCL-DLP or custom-kernel
// prepack runs for it, even under `ZENDNNL_GRP_MATMUL_ALGO=0`.
TEST_F(RoutedMoENTileFlatParallel, PackingWarmsNoOtherWeightLayout) {
    using zendnnl::lowoha::matmul::test_api_algo_legacy_override;
    using zendnnl::lowoha::matmul::test_api_algo_override;
    namespace prepack_api
            = zendnnl::lowoha::matmul::group_matmul_prepack::test_api;
    struct algo0_guard {
        int algo = test_api_algo_override().exchange(0);
        int legacy = test_api_algo_legacy_override().exchange(1);
        bool stats = prepack_api::s_capture_last_invocation.exchange(true);
        algo0_guard() { prepack_api::clear_last_invocation_stats(); }
        ~algo0_guard() {
            prepack_api::s_capture_last_invocation.store(stats);
            test_api_algo_legacy_override().store(legacy);
            test_api_algo_override().store(algo);
        }
    } algo0;
    moe_problem pb;
    pb.build(8, 128, 64, 8, 2, 0x5b0bu);
    da8w8_call call(pb);
    ntile_flat_parallel_decode_guard guard;
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_EQ(guard.calls(), 1u);
    EXPECT_FALSE(prepack_api::get_last_invocation_stats().valid);
}

// A one-thread team, in process and under an OpenMP thread limit.  The limit
// takes effect only when the runtime starts, so that call runs in a
// re-executed child process.
TEST_F(RoutedMoENTileFlatParallel, OneThreadTeamMatchesReference) {
    const auto one_thread_call = [] {
        moe_problem pb;
        pb.build(8, 256, 96, 32, 8, 0x5b07u);
        da8w8_call call(pb);
        call.primary.params.num_threads = 1;
        call.secondary.params.num_threads = 1;
        ntile_flat_parallel_decode_guard guard;
        poison(pb.dst);
        return call.run(pb, pb.dst.data()) == status_t::success
                && guard.calls() == 1
                && rel_mae(pb.dst, reference_moe(pb)) < 3e-3;
    };
    EXPECT_TRUE(one_thread_call());
#if GTEST_HAS_DEATH_TEST
    struct env_pin {
        const char *name;
        bool had_value = false;
        std::string old_value;
        env_pin(const char *n, const char *value) : name(n) {
            if (const char *v = std::getenv(name)) {
                had_value = true;
                old_value = v;
            }
            setenv(name, value, 1);
        }
        ~env_pin() {
            if (had_value) {
                setenv(name, old_value.c_str(), 1);
            } else {
                unsetenv(name);
            }
        }
    };
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    env_pin limit("OMP_THREAD_LIMIT", "16");
    env_pin threads("OMP_NUM_THREADS", "32");
    EXPECT_EXIT(std::exit(one_thread_call() ? 0 : 1),
            ::testing::ExitedWithCode(0), "");
#endif
}

// At weight cache mode 2 a DA8W8 layer that declares its buffer capacity is
// packed in place by the fast executor, except while N-tile flat parallel owns
// its decode calls: then a prompt leaves the weights raw for N-tile flat
// parallel to pack, whichever way the custom kernel's switches are set.
TEST_F(RoutedMoENTileFlatParallel,
        PromptLeavesTheWeightsRawWhileNTileFlatParallelOwnsTheDecode) {
    for (const bool int8_kernel : {true, false}) {
        SCOPED_TRACE(int8_kernel ? "INT8 kernel on" : "INT8 kernel off");
        group_matmul_routed_moe_flush_weight_cache();
        // Two experts and 200 tokens put more than kDecodeMaxM rows on an
        // expert; the decode reuses the weights with 8 tokens.
        moe_problem prompt;
        prompt.build(200, 128, 64, 2, 1, 0x5b0du);
        moe_problem decode;
        decode.build(8, 128, 64, 2, 1, 0x5b0eu);
        decode.w13 = prompt.w13;
        decode.w2 = prompt.w2;
        decode.s13 = prompt.s13;
        decode.s2 = prompt.s2;
        const size_t experts = static_cast<size_t>(prompt.E);
        const size_t w13_logical = prompt.w13.size() / experts;
        const size_t w2_logical = prompt.w2.size() / experts;
        const size_t w13_capacity = static_cast<size_t>(2 * prompt.N)
                * static_cast<size_t>(group_matmul_routed_moe_packed_row_bytes(
                        prompt.K, data_type_t::s8));
        const size_t w2_capacity = static_cast<size_t>(prompt.K)
                * static_cast<size_t>(group_matmul_routed_moe_packed_row_bytes(
                        prompt.N, data_type_t::s8));
        std::vector<int8_t> padded_w13(experts * w13_capacity, int8_t {0x5a});
        std::vector<int8_t> padded_w2(experts * w2_capacity, int8_t {0x5a});
        for (size_t e = 0; e < experts; ++e) {
            std::memcpy(padded_w13.data() + e * w13_capacity,
                    prompt.w13.data() + e * w13_logical, w13_logical);
            std::memcpy(padded_w2.data() + e * w2_capacity,
                    prompt.w2.data() + e * w2_logical, w2_logical);
        }
        const auto in_padded_buffers = [&](da8w8_call &call) {
            call.primary.weight = padded_w13.data();
            call.primary.wei_buffer_capacity_bytes = w13_capacity;
            call.primary.params.weight_cache_type = 2;
            call.secondary.weight = padded_w2.data();
            call.secondary.wei_buffer_capacity_bytes = w2_capacity;
            call.secondary.params.weight_cache_type = 2;
        };
        da8w8_call prompt_call(prompt);
        in_padded_buffers(prompt_call);
        da8w8_call decode_call(decode);
        in_padded_buffers(decode_call);
        const auto original_w13 = padded_w13;

        // The decode guard restores the INT8 override it pinned.
        ntile_flat_parallel_decode_guard guard;
        guard.ck_int8.slot.store(int8_kernel ? 1 : 0);
        routed_weight_cache_mode_guard cache_mode_guard;
        routed_cache_flush_guard cache_guard;
        ASSERT_EQ(
                prompt_call.run(prompt, prompt.dst.data()), status_t::success);
        EXPECT_LT(rel_mae(prompt.dst, reference_moe(prompt)), 3e-3);
        EXPECT_TRUE(padded_w13 == original_w13);
        ASSERT_EQ(
                decode_call.run(decode, decode.dst.data()), status_t::success);
        EXPECT_EQ(guard.calls(), 1u);
        EXPECT_LT(rel_mae(decode.dst, reference_moe(decode)), 3e-3);
    }
}

/// A routed MoE layer of 16-bit experts (BF16 or FP16), owning its buffers.
/// The weights are about 1/sqrt(fan-in), so the gates stay O(1) and SiLU and
/// GELU stay apart.
struct half_moe_problem {
    data_type_t dt = data_type_t::bf16;
    int M = 0, K = 0, N = 0, E = 0, topk = 0;
    zendnnl::lowoha::matmul::grp_matmul_gated_act_t act
            = zendnnl::lowoha::matmul::grp_matmul_gated_act_t::silu_and_mul;
    std::vector<uint16_t> src, w13, w2;
    std::vector<int32_t> ids;
    std::vector<float> weights;
    /// The problem whose `w13` and `w2` the calls read, when not this one.
    half_moe_problem *weights_owner = nullptr;
    /// `[E, 2N]` and `[E, K]` biases in the problem's type; none when empty.
    std::vector<uint16_t> b13, b2;

    void add_biases(uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> unit(-0.5f, 0.5f);
        b13.resize(static_cast<size_t>(E) * 2 * N);
        b2.resize(static_cast<size_t>(E) * K);
        for (auto *b : {&b13, &b2}) {
            for (auto &v : *b)
                v = from_f32(unit(rng));
        }
    }

    float to_f32(uint16_t v) const {
        return dt == data_type_t::f16
                ? zendnnl::common::float16_t::f16_to_f32_val(v)
                : bf16_to_f32(v);
    }
    uint16_t from_f32(float f) const {
        return dt == data_type_t::f16
                ? zendnnl::common::float16_t::f32_to_f16_val(f)
                : f32_to_bf16(f);
    }

    void build(data_type_t dt_, int M_, int K_, int N_, int E_, int topk_,
            uint32_t seed) {
        dt = dt_;
        M = M_;
        K = K_;
        N = N_;
        E = E_;
        topk = topk_;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> unit(-1.f, 1.f);
        std::uniform_real_distribution<float> rw(0.05f, 0.95f);
        std::uniform_int_distribution<int> pick(0, E - 1);
        const float a13 = 1.7f / std::sqrt(static_cast<float>(K));
        const float a2 = 1.7f / std::sqrt(static_cast<float>(N));
        src.resize(static_cast<size_t>(M) * K);
        for (auto &v : src)
            v = from_f32(unit(rng));
        w13.resize(static_cast<size_t>(E) * 2 * N * K);
        for (auto &v : w13)
            v = from_f32(a13 * unit(rng));
        w2.resize(static_cast<size_t>(E) * K * N);
        for (auto &v : w2)
            v = from_f32(a2 * unit(rng));
        ids.resize(static_cast<size_t>(M) * topk);
        weights.resize(ids.size());
        for (int m = 0; m < M; ++m) {
            int32_t *const row = ids.data() + static_cast<size_t>(m) * topk;
            for (int t = 0; t < topk; ++t) {
                do {
                    row[t] = pick(rng);
                } while (std::find(row, row + t, row[t]) != row + t);
                weights[static_cast<size_t>(m) * topk + t] = rw(rng);
            }
        }
    }

    /// Keeps the first `M_` tokens, on the same weights.
    void keep_tokens(int M_) {
        M = M_;
        src.resize(static_cast<size_t>(M) * K);
        ids.resize(static_cast<size_t>(M) * topk);
        weights.resize(ids.size());
    }

    /// One routed call into `out`, sized and poisoned here, at the caller's
    /// weight cache mode.  Rows of `src_ld` and `out_ld` values (0 for K)
    /// pad the source with NaN and the output with the poison.
    status_t run(std::vector<uint16_t> &out, int num_threads = 8,
            int src_ld = 0, int out_ld = 0) {
        using zendnnl::lowoha::matmul::grp_matmul_gated_act_params;
        const auto projection = [&](std::vector<uint16_t> &w, int n, int k) {
            group_matmul_projection_params p;
            p.output_size = n;
            p.input_size = k;
            p.weight = w.data();
            p.ldb = k;
            p.params.dtypes.src = dt;
            p.params.dtypes.wei = dt;
            p.params.dtypes.dst = dt;
            p.params.num_threads = num_threads;
            return p;
        };
        half_moe_problem &owner
                = weights_owner != nullptr ? *weights_owner : *this;
        auto primary = projection(owner.w13, 2 * N, K);
        auto secondary = projection(owner.w2, K, N);
        for (auto [p, b] :
                {std::pair {&primary, &b13}, std::pair {&secondary, &b2}}) {
            if (b->empty()) continue;
            p->bias = b->data();
            p->params.dtypes.bias = dt;
        }
        group_matmul_routing_params routing;
        routing.topk_ids = ids.data();
        routing.topk_weights = weights.data();
        grp_matmul_gated_act_params gated;
        gated.act = act;
        src_ld = src_ld > 0 ? src_ld : K;
        out_ld = out_ld > 0 ? out_ld : K;
        std::vector<uint16_t> padded;
        if (src_ld != K) {
            padded.assign(static_cast<size_t>(M) * src_ld, 0x7fc0u);
            for (int m = 0; m < M; ++m) {
                std::copy_n(&src[static_cast<size_t>(m) * K], K,
                        &padded[static_cast<size_t>(m) * src_ld]);
            }
        }
        out.resize(static_cast<size_t>(M) * out_ld);
        poison(out);
        return routed_fused_moe_direct('r', false,
                padded.empty() ? src.data() : padded.data(), src_ld, M, E, topk,
                out.data(), out_ld, primary, routing, &secondary, &gated);
    }

    /// Relative mean absolute error of `got` against `want`.
    double rel_mae(const std::vector<uint16_t> &got,
            const std::vector<uint16_t> &want) const {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < want.size(); ++i) {
            const double w = to_f32(want[i]);
            num += std::fabs(to_f32(got[i]) - w);
            den += std::fabs(w);
        }
        return den > 0.0 ? num / den : num;
    }

    /// The reference's tolerance: FP16 kernels accumulate in FP16 within a
    /// K block, about 0.6% from an exact reference at K = 512.
    double tolerance() const { return dt == data_type_t::f16 ? 1e-2 : 2e-3; }
};

/// Scalar reference for `half_moe_problem`: rounds the intermediate and each
/// slot's W2 output to the problem's type where the decode does, then
/// reduces in FP32.
std::vector<uint16_t> reference_moe_half(const half_moe_problem &pb) {
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
    const int M = pb.M, K = pb.K, N = pb.N, topk = pb.topk;
    std::vector<uint16_t> out(static_cast<size_t>(M) * K, 0);
    std::vector<float> inter(static_cast<size_t>(N));
    std::vector<float> acc(static_cast<size_t>(K));
    for (int m = 0; m < M; ++m) {
        std::fill(acc.begin(), acc.end(), 0.f);
        const uint16_t *const x = pb.src.data() + static_cast<size_t>(m) * K;
        for (int t = 0; t < topk; ++t) {
            const size_t slot = static_cast<size_t>(m) * topk + t;
            const size_t e = static_cast<size_t>(pb.ids[slot]);
            const uint16_t *const w13e = pb.w13.data() + e * 2 * N * K;
            for (int j = 0; j < N; ++j) {
                double g = 0.0, u = 0.0;
                for (int k = 0; k < K; ++k) {
                    const double xv = pb.to_f32(x[k]);
                    g += xv * pb.to_f32(w13e[static_cast<size_t>(j) * K + k]);
                    u += xv
                            * pb.to_f32(
                                    w13e[static_cast<size_t>(N + j) * K + k]);
                }
                const float gf = static_cast<float>(g);
                const float a = pb.act == grp_matmul_gated_act_t::gelu_and_mul
                        ? gelu_ref(gf)
                        : gf / (1.f + std::exp(-gf));
                inter[j] = pb.to_f32(pb.from_f32(a * static_cast<float>(u)));
            }
            const uint16_t *const w2e = pb.w2.data() + e * K * N;
            for (int k = 0; k < K; ++k) {
                double d = 0.0;
                for (int j = 0; j < N; ++j) {
                    d += static_cast<double>(inter[j])
                            * pb.to_f32(w2e[static_cast<size_t>(k) * N + j]);
                }
                acc[k] += pb.weights[slot]
                        * pb.to_f32(pb.from_f32(static_cast<float>(d)));
            }
        }
        for (int k = 0; k < K; ++k)
            out[static_cast<size_t>(m) * K + k] = pb.from_f32(acc[k]);
    }
    return out;
}

/// The 16-bit types the custom kernel serves on this host.
std::vector<data_type_t> served_half_types() {
    namespace ck = zendnnl::lowoha::matmul::custom_kernel;
    std::vector<data_type_t> types;
    if (ck::bf16_isa_supported()) types.push_back(data_type_t::bf16);
    if (ck::avx512f16_available()) types.push_back(data_type_t::f16);
    return types;
}

const char *half_name(data_type_t dt) {
    return dt == data_type_t::f16 ? "FP16" : "BF16";
}

#define SKIP_WITHOUT_HALF_CUSTOM_KERNEL() \
    do { \
        if (served_half_types().empty()) { \
            GTEST_SKIP() << "no 16-bit custom kernel on this host"; \
        } \
    } while (0)

TEST_F(RoutedMoENTileFlatParallel,
        Bf16AndFp16MatchReferenceAcrossGeometriesAndDecodeTokens) {
    SKIP_WITHOUT_HALF_CUSTOM_KERNEL();
    weight_cache_mode_pin cache_mode(1);
    struct shape_t {
        int K, N, E, topk;
    };
    uint32_t seed = 0x5b10u;
    for (const data_type_t dt : served_half_types()) {
        for (const shape_t s : {shape_t {256, 128, 8, 2},
                     shape_t {384, 192, 16, 4}, shape_t {512, 256, 32, 8}}) {
            for (const int tokens : {1, 5, 16}) {
                for (const int threads : {8, 1}) {
                    SCOPED_TRACE(::testing::Message()
                            << half_name(dt) << " H=" << s.K << " I=" << s.N
                            << " E=" << s.E << " topk=" << s.topk
                            << " tokens=" << tokens << " threads=" << threads);
                    // The custom-kernel pack cache is keyed by weight
                    // address, which the next problem's buffers may reuse.
                    reset_grp_matmul_caches();
                    half_moe_problem pb;
                    pb.build(dt, tokens, s.K, s.N, s.E, s.topk, seed++);
                    const auto expected = reference_moe_half(pb);
                    std::vector<uint16_t> routed;
                    ASSERT_EQ(pb.run(routed, threads), status_t::success);
                    group_matmul_routed_moe_flush_weight_cache();

                    ntile_flat_parallel_decode_guard guard;
                    std::vector<uint16_t> out(expected.size());
                    poison(out);
                    ASSERT_EQ(pb.run(out, threads), status_t::success);
                    EXPECT_EQ(guard.calls(), 1u);
                    expect_output_written(out);
                    EXPECT_LT(pb.rel_mae(out, expected), pb.tolerance());
                    EXPECT_LT(pb.rel_mae(out, routed), 2e-3);
                    group_matmul_routed_moe_flush_weight_cache();
                }
            }
        }
    }
}

TEST_F(RoutedMoENTileFlatParallel,
        Bf16AndFp16GatedActivationsMatchTheirReference) {
    SKIP_WITHOUT_HALF_CUSTOM_KERNEL();
    using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;
    weight_cache_mode_pin cache_mode(1);
    for (const data_type_t dt : served_half_types()) {
        for (const auto act : {grp_matmul_gated_act_t::silu_and_mul,
                     grp_matmul_gated_act_t::gelu_and_mul}) {
            SCOPED_TRACE(::testing::Message()
                    << half_name(dt) << " "
                    << (act == grp_matmul_gated_act_t::gelu_and_mul ? "GELU"
                                                                    : "SiLU"));
            reset_grp_matmul_caches();
            half_moe_problem pb;
            pb.build(dt, 8, 256, 128, 8, 2, 0x5b20u);
            pb.act = act;
            const auto expected = reference_moe_half(pb);
            half_moe_problem other = pb;
            other.act = act == grp_matmul_gated_act_t::gelu_and_mul
                    ? grp_matmul_gated_act_t::silu_and_mul
                    : grp_matmul_gated_act_t::gelu_and_mul;
            const auto other_expected = reference_moe_half(other);

            ntile_flat_parallel_decode_guard guard;
            std::vector<uint16_t> out;
            ASSERT_EQ(pb.run(out), status_t::success);
            expect_output_written(out);
            EXPECT_EQ(guard.calls(), 1u);
            EXPECT_LT(pb.rel_mae(out, expected), pb.tolerance());
            EXPECT_GT(pb.rel_mae(out, other_expected), 2e-2);
            group_matmul_routed_moe_flush_weight_cache();
        }

        SCOPED_TRACE(::testing::Message() << half_name(dt) << " SwiGLU-OAI");
        reset_grp_matmul_caches();
        half_moe_problem pb;
        pb.build(dt, 8, 256, 128, 8, 2, 0x5b21u);
        pb.act = grp_matmul_gated_act_t::swiglu_oai_mul;
        std::vector<uint16_t> routed, out;
        ASSERT_EQ(pb.run(routed), status_t::success);
        group_matmul_routed_moe_flush_weight_cache();
        ntile_flat_parallel_decode_guard guard;
        ASSERT_EQ(pb.run(out), status_t::success);
        expect_output_written(out);
        EXPECT_EQ(guard.calls(), 1u);
        EXPECT_LT(pb.rel_mae(out, routed), 2e-3);
        group_matmul_routed_moe_flush_weight_cache();
    }
}

// SwiGLU-OAI with biases on both projections, as in GPT-OSS: N-tile flat
// parallel matches the default route in decode and prompt calls, which the same
// call without biases misses.
TEST_F(RoutedMoENTileFlatParallel,
        Bf16AndFp16SwigluOaiBiasesMatchTheDefaultRoute) {
    SKIP_WITHOUT_HALF_CUSTOM_KERNEL();
    namespace mm = zendnnl::lowoha::matmul;
    weight_cache_mode_pin cache_mode(1);
    struct shape_t {
        int tokens, experts;
    };
    for (const data_type_t dt : served_half_types()) {
        for (const shape_t s : {shape_t {8, 8}, shape_t {200, 4}}) {
            SCOPED_TRACE(::testing::Message()
                    << half_name(dt) << " tokens=" << s.tokens);
            reset_grp_matmul_caches();
            half_moe_problem pb;
            pb.build(dt, s.tokens, 256, 128, s.experts, 2, 0x5b25u);
            pb.act = mm::grp_matmul_gated_act_t::swiglu_oai_mul;
            std::vector<uint16_t> unbiased, routed, out;
            ASSERT_EQ(pb.run(unbiased), status_t::success);
            group_matmul_routed_moe_flush_weight_cache();
            pb.add_biases(0x5b26u);
            ASSERT_EQ(pb.run(routed), status_t::success);
            group_matmul_routed_moe_flush_weight_cache();
            ntile_flat_parallel_decode_guard guard;
            selector_pin prompt(
                    mm::test_api::s_grp_matmul_auto_prompt_algo_override, 4);
            ASSERT_EQ(pb.run(out), status_t::success);
            expect_output_written(out);
            EXPECT_EQ(guard.calls(), 1u);
            EXPECT_LT(pb.rel_mae(out, routed), 2e-3);
            EXPECT_GT(pb.rel_mae(out, unbiased), 2e-2);
            group_matmul_routed_moe_flush_weight_cache();
        }
    }
}

TEST_F(RoutedMoENTileFlatParallel,
        Bf16AndFp16PromptShapedCallsKeepTheGenericRoute) {
    SKIP_WITHOUT_HALF_CUSTOM_KERNEL();
    // Two experts and 200 tokens put more than kDecodeMaxM rows on an expert,
    // and N-tile flat parallel is requested for decode calls only.
    weight_cache_mode_pin cache_mode(1);
    for (const data_type_t dt : served_half_types()) {
        SCOPED_TRACE(half_name(dt));
        reset_grp_matmul_caches();
        half_moe_problem pb;
        pb.build(dt, 200, 256, 128, 2, 1, 0x5b22u);
        std::vector<uint16_t> routed, out;
        ASSERT_EQ(pb.run(routed), status_t::success);
        group_matmul_routed_moe_flush_weight_cache();
        ntile_flat_parallel_decode_guard guard;
        ASSERT_EQ(pb.run(out), status_t::success);
        expect_output_written(out);
        EXPECT_EQ(guard.calls(), 0u);
        EXPECT_TRUE(out == routed);
        group_matmul_routed_moe_flush_weight_cache();
    }
}

// With N-tile flat parallel requested for prompts, 16-bit prompt calls run on
// it.
TEST_F(RoutedMoENTileFlatParallel, Bf16AndFp16PromptCallsMatchTheReference) {
    SKIP_WITHOUT_HALF_CUSTOM_KERNEL();
    namespace mm = zendnnl::lowoha::matmul;
    weight_cache_mode_pin cache_mode(1);
    for (const data_type_t dt : served_half_types()) {
        for (const auto act : {mm::grp_matmul_gated_act_t::silu_and_mul,
                     mm::grp_matmul_gated_act_t::gelu_and_mul}) {
            SCOPED_TRACE(::testing::Message()
                    << half_name(dt)
                    << (act == mm::grp_matmul_gated_act_t::gelu_and_mul
                                       ? " GELU"
                                       : " SiLU"));
            reset_grp_matmul_caches();
            half_moe_problem pb;
            pb.build(dt, 200, 256, 128, 4, 2, 0x5b49u);
            pb.act = act;
            const auto expected = reference_moe_half(pb);
            ntile_flat_parallel_decode_guard guard;
            selector_pin prompt(
                    mm::test_api::s_grp_matmul_auto_prompt_algo_override, 4);
            std::vector<uint16_t> out;
            ASSERT_EQ(pb.run(out), status_t::success);
            expect_output_written(out);
            EXPECT_EQ(guard.calls(), 1u);
            EXPECT_LT(pb.rel_mae(out, expected), pb.tolerance());
            group_matmul_routed_moe_flush_weight_cache();
        }
    }
}

// At weight cache mode 2 a 16-bit prompt would pack the layer in place; while
// N-tile flat parallel owns the decode, the prompt leaves the weights raw for
// it to pack.  Whichever phase runs first, both stay correct.
TEST_F(RoutedMoENTileFlatParallel, Bf16AndFp16PhasesShareAWeightCache2Layer) {
    SKIP_WITHOUT_HALF_CUSTOM_KERNEL();
    // A global pin of 3 holds the prompt on ALGO 3 while decode requests 4.
    struct order_t {
        int global;
        bool prompt_first;
    };
    const order_t orders[] = {{-1, true}, {-1, false}, {3, true}, {3, false}};
    for (const data_type_t dt : served_half_types()) {
        for (const order_t o : orders) {
            const bool prompt_first = o.prompt_first;
            SCOPED_TRACE(::testing::Message()
                    << half_name(dt) << " ALGO=" << o.global
                    << (prompt_first ? " prompt first" : " decode first"));
            selector_pin global_algo(
                    zendnnl::lowoha::matmul::test_api_algo_override(),
                    o.global);
            reset_grp_matmul_caches();
            half_moe_problem pb;
            pb.build(dt, 200, 256, 128, 4, 2, 0x5b23u);
            const half_moe_problem prompt = pb;
            const auto prompt_expected = reference_moe_half(prompt);
            half_moe_problem decode = pb;
            decode.keep_tokens(8);
            const auto decode_expected = reference_moe_half(decode);
            const auto use_tokens = [&](const half_moe_problem &from) {
                pb.M = from.M;
                pb.src = from.src;
                pb.ids = from.ids;
                pb.weights = from.weights;
            };
            ntile_flat_parallel_decode_guard guard;
            routed_weight_cache_mode_guard cache_mode_guard;
            std::vector<uint16_t> out;
            const auto run_decode = [&] {
                use_tokens(decode);
                const uint64_t served = guard.calls();
                ASSERT_EQ(pb.run(out), status_t::success);
                expect_output_written(out);
                EXPECT_EQ(guard.calls(), served + 1);
                EXPECT_LT(pb.rel_mae(out, decode_expected), pb.tolerance());
            };
            if (!prompt_first) { run_decode(); }
            use_tokens(prompt);
            ASSERT_EQ(pb.run(out), status_t::success);
            expect_output_written(out);
            EXPECT_EQ(guard.calls(), prompt_first ? 0u : 1u);
            EXPECT_TRUE(pb.w13 == prompt.w13 && pb.w2 == prompt.w2)
                    << "the prompt must leave the weights raw";
            EXPECT_LT(pb.rel_mae(out, prompt_expected), pb.tolerance());
            run_decode();
            group_matmul_routed_moe_flush_weight_cache();
        }
    }
}

// A layer's first decode waits for a generic call in flight before it packs
// the layer from the weights, which that call leaves raw.
TEST_F(RoutedMoENTileFlatParallel, FirstCallWaitsForAGenericCallInFlight) {
    SKIP_WITHOUT_HALF_CUSTOM_KERNEL();
    namespace rm_api = zendnnl::lowoha::matmul::routed_moe::test_api;
    for (const data_type_t dt : served_half_types()) {
        SCOPED_TRACE(half_name(dt));
        reset_grp_matmul_caches();
        half_moe_problem prompt;
        prompt.build(dt, 200, 256, 128, 4, 2, 0x5b24u);
        const half_moe_problem raw = prompt;
        const auto prompt_expected = reference_moe_half(raw);
        half_moe_problem decode = raw;
        decode.keep_tokens(8);
        const auto decode_expected = reference_moe_half(decode);
        decode.weights_owner = &prompt;

        ntile_flat_parallel_decode_guard guard;
        routed_weight_cache_mode_guard cache_mode_guard;
        struct pause_guard {
            std::atomic<bool> &pause
                    = rm_api::s_pause_generic_with_storage_guard;
            pause_guard() { pause.store(true, std::memory_order_release); }
            ~pause_guard() { pause.store(false, std::memory_order_release); }
        } paused;
        auto &entered = rm_api::s_generic_storage_guard_entered;
        entered.store(false, std::memory_order_relaxed);

        std::vector<uint16_t> prompt_out, decode_out;
        status_t prompt_status = status_t::failure;
        status_t decode_status = status_t::failure;
        std::thread prompt_thread(
                [&] { prompt_status = prompt.run(prompt_out); });
        const auto deadline
                = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!entered.load(std::memory_order_acquire)
                && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        const bool prompt_entered = entered.load(std::memory_order_acquire);
        std::thread decode_thread(
                [&] { decode_status = decode.run(decode_out); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const uint64_t packed_while_paused = guard.layer_entries();
        paused.pause.store(false, std::memory_order_release);
        prompt_thread.join();
        decode_thread.join();

        ASSERT_TRUE(prompt_entered);
        EXPECT_EQ(packed_while_paused, 0u);
        ASSERT_EQ(prompt_status, status_t::success);
        ASSERT_EQ(decode_status, status_t::success);
        EXPECT_TRUE(prompt.w13 == raw.w13 && prompt.w2 == raw.w2)
                << "the prompt must leave the weights raw";
        EXPECT_EQ(guard.calls(), 1u);
        EXPECT_EQ(guard.layer_entries(), 1u);
        EXPECT_LT(prompt.rel_mae(prompt_out, prompt_expected),
                prompt.tolerance());
        EXPECT_LT(decode.rel_mae(decode_out, decode_expected),
                decode.tolerance());
        group_matmul_routed_moe_flush_weight_cache();
    }
}

// A prompt pinned to ALGO 1 runs AOCL-DLP, whose reorder would land in the
// caller's 16-bit buffers at weight cache mode 2; while N-tile flat parallel
// owns the decode it reorders out of place, and the decode packs the raw
// weights.
TEST_F(RoutedMoENTileFlatParallel,
        Bf16AndFp16DecodeAfterAnAoclPromptMatchesReference) {
    SKIP_WITHOUT_HALF_CUSTOM_KERNEL();
    namespace mm = zendnnl::lowoha::matmul;
    for (const data_type_t dt : served_half_types()) {
        for (const bool mixed : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                    << half_name(dt) << " mixed_inplace=" << mixed);
            reset_grp_matmul_caches();
            half_moe_problem pb;
            pb.build(dt, 200, 256, 128, 4, 2, 0x5b31u);
            half_moe_problem raw = pb;
            ntile_flat_parallel_decode_guard guard;
            routed_weight_cache_mode_guard cache_mode_guard;
            cache_mode_guard.config.set_grp_auto_mixed_inplace(mixed);
            selector_pin prompt_algo(
                    mm::test_api::s_grp_matmul_auto_prompt_algo_override, 1);
            std::vector<uint16_t> out;
            ASSERT_EQ(pb.run(out), status_t::success);
            expect_output_written(out);
            EXPECT_EQ(guard.calls(), 0u);
            EXPECT_TRUE(pb.w13 == raw.w13 && pb.w2 == raw.w2)
                    << "the prompt must leave the weights raw";

            pb.keep_tokens(8);
            raw.keep_tokens(8);
            ASSERT_EQ(pb.run(out), status_t::success);
            expect_output_written(out);
            EXPECT_EQ(guard.calls(), 1u);
            EXPECT_LT(
                    raw.rel_mae(out, reference_moe_half(raw)), raw.tolerance());
            group_matmul_routed_moe_flush_weight_cache();
        }
    }
}

// Weights another path packed in place before N-tile flat parallel was
// requested no longer hold the layer's weights: N-tile flat parallel refuses
// the layer rather than read them.
TEST_F(RoutedMoENTileFlatParallel, Bf16AndFp16RefuseWeightsPackedInPlace) {
    SKIP_WITHOUT_HALF_CUSTOM_KERNEL();
    namespace mm = zendnnl::lowoha::matmul;
    for (const data_type_t dt : served_half_types()) {
        SCOPED_TRACE(half_name(dt));
        reset_grp_matmul_caches();
        half_moe_problem pb;
        pb.build(dt, 200, 256, 128, 4, 2, 0x5b32u);
        const half_moe_problem raw = pb;
        std::vector<uint16_t> out;
        {
            routed_weight_cache_mode_guard cache_mode_guard;
            selector_pin algo3(mm::test_api_algo_override(), 3);
            selector_pin prompt_unset(
                    mm::test_api::s_grp_matmul_auto_prompt_algo_override, 7);
            selector_pin ck(
                    mm::test_api::s_grp_matmul_custom_kernel_override, 1);
            ASSERT_EQ(pb.run(out), status_t::success);
            expect_output_written(out);
        }
        ASSERT_FALSE(pb.w13 == raw.w13) << "the prompt must pack in place";

        ntile_flat_parallel_decode_guard guard;
        pb.keep_tokens(8);
        out.assign(static_cast<size_t>(pb.M) * pb.K, 0);
        poison(out);
        const std::vector<uint16_t> untouched = out;
        EXPECT_EQ(pb.run(out), status_t::unimplemented);
        EXPECT_EQ(guard.calls(), 0u);
        EXPECT_EQ(out, untouched);
        group_matmul_routed_moe_flush_weight_cache();
        reset_grp_matmul_caches();
    }
}

// The same for an AOCL-DLP reorder into the caller's buffers.
TEST_F(RoutedMoENTileFlatParallel,
        Bf16AndFp16RefuseWeightsAoclReorderedInPlace) {
    SKIP_WITHOUT_HALF_CUSTOM_KERNEL();
    namespace mm = zendnnl::lowoha::matmul;
    for (const data_type_t dt : served_half_types()) {
        SCOPED_TRACE(half_name(dt));
        reset_grp_matmul_caches();
        half_moe_problem pb;
        pb.build(dt, 200, 256, 128, 4, 2, 0x5b35u);
        const half_moe_problem raw = pb;
        std::vector<uint16_t> out;
        {
            routed_weight_cache_mode_guard cache_mode_guard;
            selector_pin global_auto(mm::test_api_algo_override(), 0);
            selector_pin prompt_algo(
                    mm::test_api::s_grp_matmul_auto_prompt_algo_override, 1);
            ASSERT_EQ(pb.run(out), status_t::success);
            expect_output_written(out);
        }
        ASSERT_FALSE(pb.w13 == raw.w13) << "the prompt must reorder in place";

        ntile_flat_parallel_decode_guard guard;
        pb.keep_tokens(8);
        EXPECT_EQ(pb.run(out), status_t::unimplemented);
        std::vector<uint16_t> untouched(out.size());
        poison(untouched);
        EXPECT_EQ(guard.calls(), 0u);
        EXPECT_EQ(out, untouched);
        group_matmul_routed_moe_flush_weight_cache();
        reset_grp_matmul_caches();
    }
}

// An FP16 layer on a host without AVX512-FP16 fails before anything is
// packed.
TEST_F(RoutedMoENTileFlatParallel, Fp16WithoutItsKernelsFailsBeforePacking) {
    namespace mm = zendnnl::lowoha::matmul;
    if (mm::custom_kernel::avx512f16_available()) {
        GTEST_SKIP() << "this host has the FP16 kernels";
    }
    group_matmul_routed_moe_flush_weight_cache();
    half_moe_problem pb;
    pb.build(data_type_t::f16, 8, 256, 128, 4, 2, 0x5b34u);
    ntile_flat_parallel_decode_guard guard;
    std::vector<uint16_t> out;
    EXPECT_EQ(pb.run(out), status_t::unimplemented);
    std::vector<uint16_t> untouched(out.size());
    poison(untouched);
    EXPECT_EQ(out, untouched);
    EXPECT_EQ(guard.calls(), 0u);
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_packs(), 0u);
}

// 16-bit source and output rows with padding past their K values: the
// padding of the source is never read, and that of the output never written.
TEST_F(RoutedMoENTileFlatParallel, Bf16AndFp16PaddedRowsMatchTheUnpaddedCall) {
    SKIP_WITHOUT_HALF_CUSTOM_KERNEL();
    namespace mm = zendnnl::lowoha::matmul;
    for (const data_type_t dt : served_half_types()) {
        for (const int tokens : {8, 300}) {
            SCOPED_TRACE(::testing::Message()
                    << half_name(dt) << " tokens=" << tokens);
            group_matmul_routed_moe_flush_weight_cache();
            half_moe_problem pb;
            pb.build(dt, tokens, 256, 128, 4, 2, 0x5b36u);
            ntile_flat_parallel_decode_guard guard;
            selector_pin prompt(
                    mm::test_api::s_grp_matmul_auto_prompt_algo_override, 4);
            std::vector<uint16_t> want, out;
            ASSERT_EQ(pb.run(want), status_t::success);
            expect_output_written(want);
            const int src_ld = pb.K + 16, out_ld = pb.K + 24;
            ASSERT_EQ(pb.run(out, 8, src_ld, out_ld), status_t::success);
            EXPECT_EQ(guard.calls(), 2u);
            std::vector<uint16_t> pad(static_cast<size_t>(out_ld - pb.K));
            poison(pad);
            for (int m = 0; m < tokens; ++m) {
                const uint16_t *row = &out[static_cast<size_t>(m) * out_ld];
                ASSERT_TRUE(std::equal(
                        row, row + pb.K, &want[static_cast<size_t>(m) * pb.K]))
                        << "row " << m;
                ASSERT_TRUE(std::equal(row + pb.K, row + out_ld, pad.begin()))
                        << "row " << m << "'s padding";
            }
        }
    }
    group_matmul_routed_moe_flush_weight_cache();
}
