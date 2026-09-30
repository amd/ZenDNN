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
 * @file test_routed_moe_s4.cpp
 * @brief End-to-end routed-MoE per-group signed-W4 coverage.
 *
 * The numerical oracle in this file is intentionally independent of the
 * production S4 packer, unpack helpers, micro-kernel selector, and activation
 * helpers.  It decodes canonical nibbles and evaluates the complete quantized
 * MoE pipeline with scalar loops.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "lowoha_operators/matmul/routed_moe/routed_moe_internal.hpp"

// The MSVC CRT does not provide POSIX setenv/unsetenv.
#if defined(_MSC_VER)
static inline int setenv(const char *name, const char *value, int overwrite) {
    if (!overwrite && std::getenv(name) != nullptr) { return 0; }
    return _putenv_s(name, value);
}
static inline int unsetenv(const char *name) {
    return _putenv_s(name, "");
}
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
void clear_ggml_weight_unpack_cache();
void clear_grp_wei_scale_f32_cache();
namespace custom_kernel {
void clear_custom_kernel_pack_cache_s4();
}
namespace group_matmul_prepack {
void clear_fingerprint_cache_for_test();
}
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

namespace {

using zendnnl::lowoha::matmul::group_matmul_projection_params;
using zendnnl::lowoha::matmul::group_matmul_routed_moe_flush_weight_cache;
using zendnnl::lowoha::matmul::group_matmul_routed_moe_query;
using zendnnl::lowoha::matmul::group_matmul_routed_moe_validate;
using zendnnl::lowoha::matmul::group_matmul_routing_params;
using zendnnl::lowoha::matmul::routed_fused_moe_direct;
using zendnnl::lowoha::matmul::routed_moe_activation_t;
using zendnnl::lowoha::matmul::routed_moe_capability;
using zendnnl::lowoha::matmul::routed_moe_params;
using zendnnl::lowoha::matmul::routed_moe_quant_t;
using zendnnl::lowoha::matmul::routed_moe::execute;
using w4_reduction_path_t
        = zendnnl::lowoha::matmul::routed_moe::test_api::w4_reduction_path_t;

using data_type_t = zendnnl::common::data_type_t;
using status_t = zendnnl::error_handling::status_t;

constexpr uint16_t kOutputPoison = 0x7fc1u;

void reset_grouped_s4_caches() {
    zendnnl::lowoha::matmul::custom_kernel::clear_custom_kernel_pack_cache_s4();
    zendnnl::lowoha::matmul::group_matmul_prepack::
            clear_fingerprint_cache_for_test();
    zendnnl::lowoha::matmul::clear_ggml_weight_unpack_cache();
    zendnnl::lowoha::matmul::clear_grp_wei_scale_f32_cache();
    zendnnl::lowoha::matmul::clear_matmul_weight_caches();
}

uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

float bf16_to_f32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<uint16_t> f32_vector_to_bf16(const std::vector<float> &source) {
    std::vector<uint16_t> result(source.size());
    std::transform(source.begin(), source.end(), result.begin(),
            [](float value) { return f32_to_bf16(value); });
    return result;
}

std::vector<float> bf16_vector_to_f32(const std::vector<uint16_t> &source) {
    std::vector<float> result(source.size());
    std::transform(source.begin(), source.end(), result.begin(),
            [](uint16_t value) { return bf16_to_f32(value); });
    return result;
}

std::vector<int8_t> pack_signed_nibbles(const std::vector<int8_t> &logical) {
    std::vector<int8_t> packed((logical.size() + 1) / 2, 0);
    for (size_t i = 0; i < logical.size(); ++i) {
        const uint8_t nibble = static_cast<uint8_t>(logical[i]) & 0x0fu;
        uint8_t byte = static_cast<uint8_t>(packed[i / 2]);
        if ((i & 1u) == 0) {
            byte = static_cast<uint8_t>((byte & 0xf0u) | nibble);
        } else {
            byte = static_cast<uint8_t>((byte & 0x0fu) | (nibble << 4));
        }
        packed[i / 2] = static_cast<int8_t>(byte);
    }
    return packed;
}

int32_t unpack_signed_nibble(
        const std::vector<int8_t> &packed, size_t logical_index) {
    const uint8_t byte = static_cast<uint8_t>(packed[logical_index / 2]);
    const uint8_t nibble = (logical_index & 1u) == 0
            ? static_cast<uint8_t>(byte & 0x0fu)
            : static_cast<uint8_t>(byte >> 4);
    return nibble < 8 ? static_cast<int32_t>(nibble)
                      : static_cast<int32_t>(nibble) - 16;
}

struct s4_problem {
    int64_t M = 0;
    int64_t K = 128;
    int64_t N = 64;
    int64_t E = 8;
    int64_t topk = 2;
    int64_t gate_group_size = 32;
    int64_t down_group_size = 32;
    int64_t src_stride = 0;
    int64_t dst_stride = 0;
    int64_t ids_stride = 0;
    int64_t weights_stride = 0;
    int32_t threads = 1;
    routed_moe_activation_t activation = routed_moe_activation_t::silu_and_mul;

    std::vector<uint16_t> src;
    std::vector<uint16_t> dst;
    std::vector<int8_t> w13;
    std::vector<int8_t> w2;
    std::vector<float> s13;
    std::vector<float> s2;
    std::vector<int32_t> ids;
    std::vector<float> weights;

    void build(int64_t tokens, uint32_t seed) {
        M = tokens;
        src_stride = src_stride > 0 ? src_stride : K;
        dst_stride = dst_stride > 0 ? dst_stride : K;
        ids_stride = ids_stride > 0 ? ids_stride : topk;
        weights_stride = weights_stride > 0 ? weights_stride : topk;

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> source_dist(-1.f, 1.f);
        std::uniform_int_distribution<int> quant_dist(-8, 7);
        std::uniform_real_distribution<float> scale_dist(0.003f, 0.012f);

        src.assign(static_cast<size_t>((M - 1) * src_stride + K),
                f32_to_bf16(-9.f));
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t k = 0; k < K; ++k) {
                src[static_cast<size_t>(m * src_stride + k)]
                        = f32_to_bf16(source_dist(rng));
            }
        }
        dst.assign(
                static_cast<size_t>((M - 1) * dst_stride + K), kOutputPoison);

        std::vector<int8_t> logical13(static_cast<size_t>(E * 2 * N * K));
        std::vector<int8_t> logical2(static_cast<size_t>(E * K * N));
        for (auto &value : logical13) {
            value = static_cast<int8_t>(quant_dist(rng));
        }
        for (auto &value : logical2) {
            value = static_cast<int8_t>(quant_dist(rng));
        }
        // Every end-to-end case carries both representable S4 extremes.
        logical13[0] = -8;
        logical13[1] = 7;
        logical13[logical13.size() - 2] = -8;
        logical13[logical13.size() - 1] = 7;
        logical2[0] = 7;
        logical2[1] = -8;
        logical2[logical2.size() - 2] = 7;
        logical2[logical2.size() - 1] = -8;
        w13 = pack_signed_nibbles(logical13);
        w2 = pack_signed_nibbles(logical2);

        const int64_t gate_groups = K / gate_group_size;
        const int64_t down_groups = N / down_group_size;
        s13.resize(static_cast<size_t>(E * gate_groups * 2 * N));
        s2.resize(static_cast<size_t>(E * down_groups * K));
        for (auto &value : s13) {
            value = scale_dist(rng);
        }
        for (auto &value : s2) {
            value = scale_dist(rng);
        }

        ids.assign(static_cast<size_t>((M - 1) * ids_stride + topk), -99);
        weights.assign(
                static_cast<size_t>((M - 1) * weights_stride + topk), -99.f);
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t t = 0; t < topk; ++t) {
                ids[static_cast<size_t>(m * ids_stride + t)]
                        = static_cast<int32_t>((m * (t + 1) + 3 * t) % E);
                weights[static_cast<size_t>(m * weights_stride + t)] = topk == 1
                        ? 1.f
                        : (t == 0 ? 0.65f
                                  : 0.35f / static_cast<float>(topk - 1));
            }
        }
    }

    void poison_output() { std::fill(dst.begin(), dst.end(), kOutputPoison); }

    std::vector<uint16_t> logical_output() const {
        std::vector<uint16_t> result(static_cast<size_t>(M * K));
        for (int64_t m = 0; m < M; ++m) {
            std::copy_n(dst.data() + m * dst_stride, K, result.data() + m * K);
        }
        return result;
    }

    routed_moe_params params() {
        routed_moe_params p;
        p.num_tokens = M;
        p.hidden_size = K;
        p.intermediate_size = N;
        p.num_local_experts = E;
        p.topk = topk;
        p.src = src.data();
        p.src_stride = src_stride;
        p.src_dt = data_type_t::bf16;
        p.dst = dst.data();
        p.dst_stride = dst_stride;
        p.dst_dt = data_type_t::bf16;
        p.gate_up_weight = w13.data();
        p.down_weight = w2.data();
        p.wei_dt = data_type_t::s4;
        p.gate_up_scale = s13.data();
        p.down_scale = s2.data();
        p.scale_dt = data_type_t::f32;
        p.topk_ids = ids.data();
        p.topk_ids_stride = ids_stride;
        p.topk_weights = weights.data();
        p.topk_weights_stride = weights_stride;
        p.activation = activation;
        p.quant_scheme
                = routed_moe_quant_t::sym_per_group_w4a8_dynamic_per_token;
        p.gate_up_group_size = gate_group_size;
        p.down_group_size = down_group_size;
        p.num_threads = threads;
        return p;
    }
};

struct s4_call {
    group_matmul_projection_params primary;
    group_matmul_projection_params secondary;
    group_matmul_routing_params routing;
    zendnnl::lowoha::matmul::grp_matmul_gated_act_params gated_act;
    std::vector<float> primary_src_scale;
    std::vector<float> secondary_src_scale;
    std::vector<uint16_t> primary_wei_scale_bf16;
    std::vector<uint16_t> secondary_wei_scale_bf16;

    explicit s4_call(s4_problem &pb, bool bf16_scales = false)
        : primary_src_scale(static_cast<size_t>(pb.M), 0.f)
        , secondary_src_scale(static_cast<size_t>(pb.M), 0.f) {
        if (bf16_scales) {
            primary_wei_scale_bf16 = f32_vector_to_bf16(pb.s13);
            secondary_wei_scale_bf16 = f32_vector_to_bf16(pb.s2);
        }
        const auto init
                = [&](group_matmul_projection_params &projection, int output,
                          int input, const void *weight, const void *scale,
                          data_type_t scale_dt, int64_t groups,
                          std::vector<float> &src_scale) {
            projection.output_size = output;
            projection.input_size = input;
            projection.trans_weight = true;
            projection.weight = weight;
            projection.ldb = input;
            projection.params.dtypes.src = data_type_t::bf16;
            projection.params.dtypes.wei = data_type_t::s4;
            projection.params.dtypes.dst = data_type_t::bf16;
            projection.params.dtypes.compute = data_type_t::s8;
            projection.params.dynamic_quant = true;
            projection.params.weight_cache_type = 1;
            projection.params.quant_params.src_scale.buff = src_scale.data();
            projection.params.quant_params.src_scale.dt = data_type_t::f32;
            projection.params.quant_params.src_scale.dims = {pb.M, 1};
            projection.params.quant_params.wei_scale.buff = scale;
            projection.params.quant_params.wei_scale.dt = scale_dt;
            projection.params.quant_params.wei_scale.dims
                    = {pb.E, groups, output};
            projection.params.num_threads = pb.threads;
        };
        const data_type_t scale_dt
                = bf16_scales ? data_type_t::bf16 : data_type_t::f32;
        init(primary, static_cast<int>(2 * pb.N), static_cast<int>(pb.K),
                pb.w13.data(),
                bf16_scales ? static_cast<const void *>(
                                      primary_wei_scale_bf16.data())
                            : static_cast<const void *>(pb.s13.data()),
                scale_dt, pb.K / pb.gate_group_size, primary_src_scale);
        init(secondary, static_cast<int>(pb.K), static_cast<int>(pb.N),
                pb.w2.data(),
                bf16_scales ? static_cast<const void *>(
                                      secondary_wei_scale_bf16.data())
                            : static_cast<const void *>(pb.s2.data()),
                scale_dt, pb.N / pb.down_group_size, secondary_src_scale);
        routing.topk_ids = pb.ids.data();
        routing.topk_ids_stride = static_cast<int>(pb.ids_stride);
        routing.topk_weights = pb.weights.data();
        routing.topk_weights_stride = static_cast<int>(pb.weights_stride);
        gated_act.act = pb.activation == routed_moe_activation_t::gelu_and_mul
                ? zendnnl::lowoha::matmul::grp_matmul_gated_act_t::gelu_and_mul
                : zendnnl::lowoha::matmul::grp_matmul_gated_act_t::silu_and_mul;
    }

    status_t run(s4_problem &pb, uint16_t *output, int64_t output_stride = 0) {
        const int64_t stride
                = output_stride > 0 ? output_stride : pb.dst_stride;
        return routed_fused_moe_direct('r', false, pb.src.data(),
                static_cast<int>(pb.src_stride), static_cast<int>(pb.M),
                static_cast<int>(pb.E), static_cast<int>(pb.topk), output,
                static_cast<int>(stride), primary, routing, &secondary,
                &gated_act);
    }
};

enum class oracle_reduction_t { fused_f32, scatter_bf16, grouped_bf16 };

struct oracle_options {
    oracle_reduction_t reduction = oracle_reduction_t::fused_f32;
    const int32_t *expert_map = nullptr;
    const float *gate_scale = nullptr;
    const float *down_scale = nullptr;
};

struct scalar_quantized_row {
    std::vector<int32_t> values;
    float scale = 0.f;
};

template <typename Load>
scalar_quantized_row quantize_scalar(int64_t width, const Load &load) {
    float amax = 0.f;
    for (int64_t i = 0; i < width; ++i) {
        amax = std::max(amax, std::fabs(load(i)));
    }
    amax = std::max(amax, 1e-7f);
    scalar_quantized_row result;
    result.values.resize(static_cast<size_t>(width));
    result.scale = amax / 127.f;
    const float inverse = 127.f / amax;
    for (int64_t i = 0; i < width; ++i) {
        const int32_t rounded
                = static_cast<int32_t>(std::nearbyintf(load(i) * inverse));
        result.values[static_cast<size_t>(i)]
                = std::max<int32_t>(-127, std::min<int32_t>(127, rounded));
    }
    return result;
}

float scalar_activation(routed_moe_activation_t activation, float value) {
    if (activation == routed_moe_activation_t::gelu_and_mul) {
        return 0.5f * value * (1.f + std::erf(value * 0.7071067811865475f));
    }
    return value / (1.f + std::exp(-value));
}

std::vector<uint16_t> scalar_s4_moe(
        const s4_problem &pb, const oracle_options &options = {}) {
    const int64_t gate_groups = pb.K / pb.gate_group_size;
    const int64_t down_groups = pb.N / pb.down_group_size;
    const int64_t gate_group_stride = 2 * pb.N;
    const int64_t down_group_stride = pb.K;
    const int64_t gate_expert_stride = gate_groups * gate_group_stride;
    const int64_t down_expert_stride = down_groups * down_group_stride;
    const float *gate_scale = options.gate_scale != nullptr ? options.gate_scale
                                                            : pb.s13.data();
    const float *down_scale
            = options.down_scale != nullptr ? options.down_scale : pb.s2.data();

    std::vector<uint16_t> output(static_cast<size_t>(pb.M * pb.K), 0);
    std::vector<float> intermediate(static_cast<size_t>(pb.N));
    std::vector<float> down(static_cast<size_t>(pb.topk * pb.K), 0.f);
    std::vector<int32_t> local_experts(static_cast<size_t>(pb.topk), -1);

    for (int64_t m = 0; m < pb.M; ++m) {
        const scalar_quantized_row source_quant
                = quantize_scalar(pb.K, [&](int64_t k) {
            return bf16_to_f32(
                    pb.src[static_cast<size_t>(m * pb.src_stride + k)]);
        });
        std::fill(down.begin(), down.end(), 0.f);
        std::fill(local_experts.begin(), local_experts.end(), -1);

        for (int64_t t = 0; t < pb.topk; ++t) {
            const int32_t raw
                    = pb.ids[static_cast<size_t>(m * pb.ids_stride + t)];
            const int32_t expert = options.expert_map != nullptr
                    ? options.expert_map[raw]
                    : raw;
            local_experts[static_cast<size_t>(t)] = expert;
            if (expert < 0) { continue; }

            for (int64_t j = 0; j < pb.N; ++j) {
                float gate = 0.f;
                float up = 0.f;
                for (int64_t group = 0; group < gate_groups; ++group) {
                    int32_t gate_dot = 0;
                    int32_t up_dot = 0;
                    for (int64_t kk = 0; kk < pb.gate_group_size; ++kk) {
                        const int64_t k = group * pb.gate_group_size + kk;
                        const int32_t a
                                = source_quant.values[static_cast<size_t>(k)];
                        const size_t gate_index = static_cast<size_t>(
                                (expert * 2 * pb.N + j) * pb.K + k);
                        const size_t up_index = static_cast<size_t>(
                                (expert * 2 * pb.N + pb.N + j) * pb.K + k);
                        gate_dot
                                += a * unpack_signed_nibble(pb.w13, gate_index);
                        up_dot += a * unpack_signed_nibble(pb.w13, up_index);
                    }
                    const float *group_scale = gate_scale
                            + expert * gate_expert_stride
                            + group * gate_group_stride;
                    gate += static_cast<float>(gate_dot) * group_scale[j];
                    up += static_cast<float>(up_dot) * group_scale[pb.N + j];
                }
                gate *= source_quant.scale;
                up *= source_quant.scale;
                intermediate[static_cast<size_t>(j)] = bf16_to_f32(f32_to_bf16(
                        scalar_activation(pb.activation, gate) * up));
            }

            const scalar_quantized_row inter_quant
                    = quantize_scalar(pb.N, [&](int64_t j) {
                return intermediate[static_cast<size_t>(j)];
            });
            float *down_row = down.data() + t * pb.K;
            for (int64_t k = 0; k < pb.K; ++k) {
                float value = 0.f;
                for (int64_t group = 0; group < down_groups; ++group) {
                    int32_t dot = 0;
                    for (int64_t jj = 0; jj < pb.down_group_size; ++jj) {
                        const int64_t j = group * pb.down_group_size + jj;
                        const size_t weight_index = static_cast<size_t>(
                                (expert * pb.K + k) * pb.N + j);
                        dot += inter_quant.values[static_cast<size_t>(j)]
                                * unpack_signed_nibble(pb.w2, weight_index);
                    }
                    const float scale = down_scale[expert * down_expert_stride
                            + group * down_group_stride + k];
                    value += static_cast<float>(dot) * scale;
                }
                down_row[k] = value * inter_quant.scale;
            }
        }

        for (int64_t k = 0; k < pb.K; ++k) {
            float result = 0.f;
            if (options.reduction == oracle_reduction_t::fused_f32) {
                // Native fused reduction visits routed blocks by local expert,
                // retaining original slot order within one expert.
                for (int64_t expert = 0; expert < pb.E; ++expert) {
                    for (int64_t t = 0; t < pb.topk; ++t) {
                        if (local_experts[static_cast<size_t>(t)] != expert) {
                            continue;
                        }
                        const float route_weight
                                = pb.weights[static_cast<size_t>(
                                        m * pb.weights_stride + t)];
                        result = std::fma(
                                down[static_cast<size_t>(t * pb.K + k)],
                                route_weight, result);
                    }
                }
            } else {
                for (int64_t t = 0; t < pb.topk; ++t) {
                    if (local_experts[static_cast<size_t>(t)] < 0) { continue; }
                    const float route_weight = pb.weights[static_cast<size_t>(
                            m * pb.weights_stride + t)];
                    float value = down[static_cast<size_t>(t * pb.K + k)];
                    if (options.reduction == oracle_reduction_t::scatter_bf16) {
                        value = bf16_to_f32(f32_to_bf16(value * route_weight));
                        result += value;
                    } else {
                        // The grouped fallback writes W2 to BF16, then its
                        // generic MoE post-op applies router weights in FP32.
                        value = bf16_to_f32(f32_to_bf16(value));
                        result = std::fma(value, route_weight, result);
                    }
                }
            }
            output[static_cast<size_t>(m * pb.K + k)] = f32_to_bf16(result);
        }
    }
    return output;
}

double rel_mae(
        const std::vector<uint16_t> &got, const std::vector<uint16_t> &want) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double expected = bf16_to_f32(want[i]);
        numerator += std::fabs(bf16_to_f32(got[i]) - expected);
        denominator += std::fabs(expected);
    }
    return denominator > 0.0 ? numerator / denominator : numerator;
}

void expect_numerically_close(const std::vector<uint16_t> &got,
        const std::vector<uint16_t> &want, double rel_limit,
        float absolute_floor, float element_relative) {
    ASSERT_EQ(got.size(), want.size());
    EXPECT_LT(rel_mae(got, want), rel_limit);
    for (size_t i = 0; i < want.size(); ++i) {
        const float actual = bf16_to_f32(got[i]);
        const float expected = bf16_to_f32(want[i]);
        ASSERT_TRUE(std::isfinite(actual)) << "element " << i;
        EXPECT_LE(std::fabs(actual - expected),
                absolute_floor + element_relative * std::fabs(expected))
                << "element " << i << " actual=" << actual
                << " expected=" << expected;
    }
}

void expect_output_written(const std::vector<uint16_t> &output) {
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NE(output[i], kOutputPoison) << "element " << i;
        EXPECT_TRUE(std::isfinite(bf16_to_f32(output[i]))) << "element " << i;
    }
}

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

struct native_w4_capture_guard {
    std::atomic<bool> &capture = zendnnl::lowoha::matmul::routed_moe::test_api::
            s_capture_native_w4;
    std::atomic<w4_reduction_path_t> &path = zendnnl::lowoha::matmul::
            routed_moe::test_api::s_last_w4_reduction_path;

    native_w4_capture_guard() {
        path.store(w4_reduction_path_t::none, std::memory_order_relaxed);
        capture.store(true, std::memory_order_relaxed);
    }

    ~native_w4_capture_guard() {
        capture.store(false, std::memory_order_relaxed);
    }
};

struct disable_routed_guard {
    bool had_value = false;
    std::string old_value;

    disable_routed_guard() {
        if (const char *value = std::getenv("ZENDNNL_ENABLE_ROUTED_MOE")) {
            had_value = true;
            old_value = value;
        }
        EXPECT_EQ(setenv("ZENDNNL_ENABLE_ROUTED_MOE", "0", 1), 0);
    }

    ~disable_routed_guard() {
        if (had_value) {
            setenv("ZENDNNL_ENABLE_ROUTED_MOE", old_value.c_str(), 1);
        } else {
            unsetenv("ZENDNNL_ENABLE_ROUTED_MOE");
        }
    }
};

void route_first_slot_to_expert_zero(s4_problem &pb) {
    for (int64_t m = 0; m < pb.M; ++m) {
        pb.ids[static_cast<size_t>(m * pb.ids_stride)] = 0;
        if (pb.topk > 1
                && pb.ids[static_cast<size_t>(m * pb.ids_stride + 1)] == 0) {
            pb.ids[static_cast<size_t>(m * pb.ids_stride + 1)] = 1;
        }
    }
}

struct routed_s4_test : public ::testing::Test {
    bool had_enable_value = false;
    std::string old_enable_value;

    void SetUp() override {
        if (const char *value = std::getenv("ZENDNNL_ENABLE_ROUTED_MOE")) {
            had_enable_value = true;
            old_enable_value = value;
        }
        unsetenv("ZENDNNL_ENABLE_ROUTED_MOE");
        group_matmul_routed_moe_flush_weight_cache();
        reset_grouped_s4_caches();
        zendnnl::lowoha::matmul::routed_moe::test_api::s_capture_native_w4
                .store(false, std::memory_order_relaxed);
        zendnnl::lowoha::matmul::routed_moe::test_api::s_last_w4_reduction_path
                .store(w4_reduction_path_t::none, std::memory_order_relaxed);

        routed_moe_capability capability;
        const status_t status = group_matmul_routed_moe_query(&capability);
        if (status == status_t::isa_unsupported) {
            GTEST_SKIP() << "requires AVX-512 VNNI and BF16";
        }
        ASSERT_EQ(status, status_t::success);
    }

    void TearDown() override {
        zendnnl::lowoha::matmul::routed_moe::test_api::s_capture_native_w4
                .store(false, std::memory_order_relaxed);
        if (had_enable_value) {
            setenv("ZENDNNL_ENABLE_ROUTED_MOE", old_enable_value.c_str(), 1);
        } else {
            unsetenv("ZENDNNL_ENABLE_ROUTED_MOE");
        }
    }
};

struct RoutedMoES4Capability : routed_s4_test {};
struct RoutedMoES4Validate : routed_s4_test {};
struct RoutedMoES4Execute : routed_s4_test {};
struct RoutedMoES4Routing : routed_s4_test {};
struct RoutedMoES4Reduction : routed_s4_test {};
struct RoutedMoES4Cache : routed_s4_test {};
struct RoutedMoES4Fallback : routed_s4_test {};
struct RoutedMoES4Scratch : routed_s4_test {};

void check_native_high_level(s4_problem &pb, s4_call &call,
        const oracle_options &oracle, w4_reduction_path_t expected_path,
        std::vector<uint16_t> *captured_output = nullptr) {
    pb.poison_output();
    grouped_call_probe grouped_probe;
    auto &completed = zendnnl::lowoha::matmul::routed_moe::test_api::
            s_native_w4_completed;
    const uint64_t before = completed.load(std::memory_order_relaxed);
    native_w4_capture_guard capture;

    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_FALSE(grouped_probe.grouped_ran());
    EXPECT_EQ(completed.load(std::memory_order_relaxed), before + 1);
    EXPECT_EQ(capture.path.load(std::memory_order_relaxed), expected_path);

    const std::vector<uint16_t> got = pb.logical_output();
    expect_output_written(got);
    expect_numerically_close(
            got, scalar_s4_moe(pb, oracle), 1.2e-2, 0.04f, 0.04f);
    if (captured_output != nullptr) { *captured_output = got; }
}

void check_native_direct(s4_problem &pb, routed_moe_params params,
        const oracle_options &oracle, w4_reduction_path_t expected_path,
        std::vector<uint16_t> *captured_output = nullptr) {
    pb.poison_output();
    params.dst = pb.dst.data();
    grouped_call_probe grouped_probe;
    auto &completed = zendnnl::lowoha::matmul::routed_moe::test_api::
            s_native_w4_completed;
    const uint64_t before = completed.load(std::memory_order_relaxed);
    native_w4_capture_guard capture;

    ASSERT_EQ(execute(params), status_t::success);
    EXPECT_FALSE(grouped_probe.grouped_ran());
    EXPECT_EQ(completed.load(std::memory_order_relaxed), before + 1);
    EXPECT_EQ(capture.path.load(std::memory_order_relaxed), expected_path);

    const std::vector<uint16_t> got = pb.logical_output();
    expect_output_written(got);
    expect_numerically_close(
            got, scalar_s4_moe(pb, oracle), 1.2e-2, 0.04f, 0.04f);
    if (captured_output != nullptr) { *captured_output = got; }
}

void check_grouped_fallback(s4_problem &pb, s4_call &call) {
    std::vector<uint16_t> output(
            static_cast<size_t>(pb.M * pb.K), kOutputPoison);
    grouped_call_probe grouped_probe;
    auto &completed = zendnnl::lowoha::matmul::routed_moe::test_api::
            s_native_w4_completed;
    const uint64_t before = completed.load(std::memory_order_relaxed);
    native_w4_capture_guard capture;

    ASSERT_EQ(call.run(pb, output.data(), pb.K), status_t::success);
    EXPECT_TRUE(grouped_probe.grouped_ran());
    EXPECT_EQ(completed.load(std::memory_order_relaxed), before);
    EXPECT_EQ(capture.path.load(std::memory_order_relaxed),
            w4_reduction_path_t::none);
    expect_output_written(output);

    oracle_options oracle;
    oracle.reduction = oracle_reduction_t::grouped_bf16;
    expect_numerically_close(
            output, scalar_s4_moe(pb, oracle), 3.5e-2, 0.12f, 0.10f);
}

} // namespace

TEST_F(RoutedMoES4Capability, ReportsPerGroupSignedW4Envelope) {
    routed_moe_capability capability;
    ASSERT_EQ(group_matmul_routed_moe_query(&capability), status_t::success);
    EXPECT_EQ(capability.max_s4_kernel_rows, 6);
    EXPECT_EQ(capability.s4_group_size_align, 8);
    EXPECT_EQ(capability.quant_mask,
            (1u << static_cast<uint32_t>(
                     routed_moe_quant_t::sym_per_oc_w8a8_dynamic_per_token))
                    | (1u << static_cast<uint32_t>(routed_moe_quant_t::
                                       sym_per_group_w4a8_dynamic_per_token)));
    EXPECT_EQ(capability.wei_dtype_mask,
            (1u << static_cast<uint32_t>(data_type_t::s8))
                    | (1u << static_cast<uint32_t>(data_type_t::s4)));
    EXPECT_NE(capability.scale_dtype_mask
                    & (1u << static_cast<uint32_t>(data_type_t::f32)),
            0u);
    EXPECT_NE(capability.scale_dtype_mask
                    & (1u << static_cast<uint32_t>(data_type_t::bf16)),
            0u);
}

TEST_F(RoutedMoES4Validate, AcceptsTightPerGroupContract) {
    s4_problem pb;
    pb.build(8, 0x4433u);
    auto params = pb.params();
    EXPECT_EQ(group_matmul_routed_moe_validate(params), status_t::success);
    EXPECT_EQ(execute(params), status_t::success);

    params = pb.params();
    params.gate_up_scale_stride_expert = (pb.K / pb.gate_group_size) * 2 * pb.N;
    params.down_scale_stride_expert = (pb.N / pb.down_group_size) * pb.K;
    EXPECT_EQ(group_matmul_routed_moe_validate(params), status_t::success);
}

TEST_F(RoutedMoES4Validate, InvalidGroupsDtypesAndStridesHaveExecuteParity) {
    s4_problem pb;
    pb.build(8, 0x4434u);
    const auto expect_rejected
            = [&](routed_moe_params params, status_t expected) {
        EXPECT_EQ(group_matmul_routed_moe_validate(params), expected);
        EXPECT_EQ(execute(params), expected);
    };

    auto params = pb.params();
    params.gate_up_group_size = 0;
    expect_rejected(params, status_t::memory_bad_quant);

    params = pb.params();
    params.gate_up_group_size = -8;
    expect_rejected(params, status_t::memory_bad_quant);

    params = pb.params();
    params.gate_up_group_size = 12;
    expect_rejected(params, status_t::memory_bad_quant);

    params = pb.params();
    params.gate_up_group_size = 24; // aligned to eight, but does not divide K.
    expect_rejected(params, status_t::memory_bad_quant);

    params = pb.params();
    params.scale_dt = data_type_t::f16;
    expect_rejected(params, status_t::unimplemented);

    params = pb.params();
    params.wei_dt = data_type_t::s8;
    expect_rejected(params, status_t::memory_bad_quant);

    params = pb.params();
    params.src_dt = data_type_t::f32;
    expect_rejected(params, status_t::unimplemented);

    params = pb.params();
    params.gate_up_scale_stride_expert
            = (pb.K / pb.gate_group_size) * 2 * pb.N + 1;
    expect_rejected(params, status_t::memory_bad_stride);

    params = pb.params();
    params.down_scale_stride_expert = (pb.N / pb.down_group_size) * pb.K + 1;
    expect_rejected(params, status_t::memory_bad_stride);

    params = pb.params();
    params.topk_weights_stride = pb.topk - 1;
    expect_rejected(params, status_t::memory_bad_stride);

    params = pb.params();
    params.quant_scheme
            = routed_moe_quant_t::asym_per_oc_w8a8_dynamic_per_token;
    expect_rejected(params, status_t::unimplemented);
}

TEST_F(RoutedMoES4Execute,
        NativeDecodeMatchesScalarOracleAndGenericSupplement) {
    for (int64_t tokens : {int64_t {1}, int64_t {9}, int64_t {16}}) {
        SCOPED_TRACE(::testing::Message() << "tokens=" << tokens);
        group_matmul_routed_moe_flush_weight_cache();
        reset_grouped_s4_caches();
        s4_problem pb;
        pb.build(tokens, 0x4400u + static_cast<uint32_t>(tokens));
        if (tokens == 16) { std::fill(pb.ids.begin(), pb.ids.end(), 0); }
        s4_call call(pb);
        oracle_options native_oracle;
        native_oracle.reduction = oracle_reduction_t::fused_f32;
        std::vector<uint16_t> native;
        ASSERT_NO_FATAL_FAILURE(check_native_high_level(pb, call, native_oracle,
                w4_reduction_path_t::fused_f32, &native));

        std::vector<uint16_t> generic(
                static_cast<size_t>(pb.M * pb.K), kOutputPoison);
        {
            disable_routed_guard disable;
            grouped_call_probe probe;
            ASSERT_EQ(call.run(pb, generic.data(), pb.K), status_t::success);
            EXPECT_TRUE(probe.grouped_ran());
        }
        oracle_options grouped_oracle;
        grouped_oracle.reduction = oracle_reduction_t::grouped_bf16;
        expect_numerically_close(generic, scalar_s4_moe(pb, grouped_oracle),
                3.5e-2, 0.12f, 0.10f);
        EXPECT_LT(rel_mae(native, generic), 3e-2);
    }
}

TEST_F(RoutedMoES4Execute,
        PromptRoutingMatchesScalarOracleAndGenericSupplement) {
    for (int64_t tokens : {int64_t {33}, int64_t {97}, int64_t {160}}) {
        SCOPED_TRACE(::testing::Message() << "tokens=" << tokens);
        group_matmul_routed_moe_flush_weight_cache();
        reset_grouped_s4_caches();
        s4_problem pb;
        pb.build(tokens, 0x4500u + static_cast<uint32_t>(tokens));
        route_first_slot_to_expert_zero(pb);
        s4_call call(pb);
        oracle_options native_oracle;
        native_oracle.reduction = oracle_reduction_t::fused_f32;
        std::vector<uint16_t> native;
        ASSERT_NO_FATAL_FAILURE(check_native_high_level(pb, call, native_oracle,
                w4_reduction_path_t::fused_f32, &native));

        std::vector<uint16_t> generic(
                static_cast<size_t>(pb.M * pb.K), kOutputPoison);
        {
            disable_routed_guard disable;
            grouped_call_probe probe;
            ASSERT_EQ(call.run(pb, generic.data(), pb.K), status_t::success);
            EXPECT_TRUE(probe.grouped_ran());
        }
        oracle_options grouped_oracle;
        grouped_oracle.reduction = oracle_reduction_t::grouped_bf16;
        expect_numerically_close(generic, scalar_s4_moe(pb, grouped_oracle),
                3.5e-2, 0.12f, 0.10f);
        EXPECT_LT(rel_mae(native, generic), 3e-2);
    }
}

TEST_F(RoutedMoES4Execute, ValidGroupSizesAndSignedExtremesMatchOracle) {
    struct group_case {
        int64_t K;
        int64_t N;
        int64_t gate_group;
        int64_t down_group;
    };
    const group_case cases[] = {
            {64, 96, 8, 24},
            {96, 64, 24, 8},
            {128, 64, 32, 64},
            {128, 128, 64, 32},
    };

    for (const auto &test_case : cases) {
        SCOPED_TRACE(::testing::Message()
                << "K=" << test_case.K << " N=" << test_case.N
                << " gate_group=" << test_case.gate_group
                << " down_group=" << test_case.down_group);
        group_matmul_routed_moe_flush_weight_cache();
        reset_grouped_s4_caches();
        s4_problem pb;
        pb.K = test_case.K;
        pb.N = test_case.N;
        pb.gate_group_size = test_case.gate_group;
        pb.down_group_size = test_case.down_group;
        pb.E = 4;
        pb.build(5, 0x4511u + static_cast<uint32_t>(test_case.K));
        EXPECT_EQ(unpack_signed_nibble(pb.w13, 0), -8);
        EXPECT_EQ(unpack_signed_nibble(pb.w13, 1), 7);
        EXPECT_EQ(unpack_signed_nibble(pb.w2, 0), 7);
        EXPECT_EQ(unpack_signed_nibble(pb.w2, 1), -8);

        s4_call call(pb);
        oracle_options oracle;
        oracle.reduction = oracle_reduction_t::fused_f32;
        ASSERT_NO_FATAL_FAILURE(check_native_high_level(
                pb, call, oracle, w4_reduction_path_t::fused_f32));
    }
}

TEST_F(RoutedMoES4Execute, GeluMatchesIndependentScalarActivation) {
    s4_problem pb;
    pb.activation = routed_moe_activation_t::gelu_and_mul;
    pb.build(8, 0x4454u);
    s4_call call(pb);
    oracle_options oracle;
    oracle.reduction = oracle_reduction_t::fused_f32;
    ASSERT_NO_FATAL_FAILURE(check_native_high_level(
            pb, call, oracle, w4_reduction_path_t::fused_f32));
}

TEST_F(RoutedMoES4Routing, LiveRowAndRoutingBlockBoundariesMatchOracle) {
    for (int64_t rows :
            {int64_t {1}, int64_t {5}, int64_t {6}, int64_t {7}, int64_t {31},
                    int64_t {32}, int64_t {33}, int64_t {34}, int64_t {65}}) {
        SCOPED_TRACE(::testing::Message() << "live_rows=" << rows);
        group_matmul_routed_moe_flush_weight_cache();
        reset_grouped_s4_caches();
        s4_problem pb;
        pb.K = 32;
        pb.N = 32;
        pb.E = 2;
        pb.topk = 1;
        pb.gate_group_size = 8;
        pb.down_group_size = 8;
        pb.build(rows, 0x4600u + static_cast<uint32_t>(rows));
        std::fill(pb.ids.begin(), pb.ids.end(), 0);
        s4_call call(pb);
        oracle_options oracle;
        oracle.reduction = oracle_reduction_t::fused_f32;
        ASSERT_NO_FATAL_FAILURE(check_native_high_level(
                pb, call, oracle, w4_reduction_path_t::fused_f32));
    }
}

TEST_F(RoutedMoES4Routing, DuplicateSlotsAndOneExpertSkewMatchOracle) {
    s4_problem pb;
    pb.E = 4;
    pb.build(34, 0x4610u);
    for (int64_t m = 0; m < pb.M; ++m) {
        pb.ids[static_cast<size_t>(m * pb.ids_stride)] = 0;
        pb.ids[static_cast<size_t>(m * pb.ids_stride + 1)] = m % 2 == 0 ? 0 : 1;
    }
    s4_call call(pb);
    oracle_options oracle;
    oracle.reduction = oracle_reduction_t::fused_f32;
    ASSERT_NO_FATAL_FAILURE(check_native_high_level(
            pb, call, oracle, w4_reduction_path_t::fused_f32));
}

TEST_F(RoutedMoES4Routing, ExpertMapMasksInactiveSlotsOnNativePath) {
    s4_problem pb;
    pb.build(8, 0x4455u);
    std::vector<int32_t> map(16);
    for (int32_t i = 0; i < 16; ++i) {
        map[static_cast<size_t>(i)] = i % 2 == 0 ? i / 2 : -1;
    }
    for (size_t i = 0; i < pb.ids.size(); ++i) {
        pb.ids[i] = static_cast<int32_t>(i % map.size());
    }
    s4_call call(pb);
    call.routing.expert_map = map.data();
    call.routing.expert_map_size = static_cast<int>(map.size());
    oracle_options oracle;
    oracle.reduction = oracle_reduction_t::fused_f32;
    oracle.expert_map = map.data();
    ASSERT_NO_FATAL_FAILURE(check_native_high_level(
            pb, call, oracle, w4_reduction_path_t::fused_f32));
}

TEST_F(RoutedMoES4Routing, AllInactiveTakesDeliberateNativeEarlyReturn) {
    s4_problem pb;
    pb.build(8, 0x4457u);
    std::vector<int32_t> map(static_cast<size_t>(pb.E), -1);
    s4_call call(pb);
    call.routing.expert_map = map.data();
    call.routing.expert_map_size = static_cast<int>(map.size());

    pb.poison_output();
    grouped_call_probe grouped_probe;
    auto &completed = zendnnl::lowoha::matmul::routed_moe::test_api::
            s_native_w4_completed;
    const uint64_t before = completed.load(std::memory_order_relaxed);
    native_w4_capture_guard capture;
    ASSERT_EQ(call.run(pb, pb.dst.data()), status_t::success);
    EXPECT_FALSE(grouped_probe.grouped_ran());
    EXPECT_EQ(completed.load(std::memory_order_relaxed), before);
    EXPECT_EQ(capture.path.load(std::memory_order_relaxed),
            w4_reduction_path_t::all_inactive);

    oracle_options oracle;
    oracle.reduction = oracle_reduction_t::fused_f32;
    oracle.expert_map = map.data();
    const auto got = pb.logical_output();
    EXPECT_EQ(got, scalar_s4_moe(pb, oracle));
    EXPECT_TRUE(std::all_of(
            got.begin(), got.end(), [](uint16_t value) { return value == 0; }));
}

TEST_F(RoutedMoES4Routing, PaddedActivationAndRoutingRowsPreservePadding) {
    s4_problem pb;
    pb.K = 64;
    pb.N = 64;
    pb.src_stride = 71;
    pb.dst_stride = 73;
    pb.ids_stride = 5;
    pb.weights_stride = 6;
    pb.build(7, 0x4620u);
    s4_call call(pb);
    oracle_options oracle;
    oracle.reduction = oracle_reduction_t::fused_f32;
    ASSERT_NO_FATAL_FAILURE(check_native_high_level(
            pb, call, oracle, w4_reduction_path_t::fused_f32));

    for (int64_t m = 0; m + 1 < pb.M; ++m) {
        for (int64_t k = pb.K; k < pb.dst_stride; ++k) {
            EXPECT_EQ(pb.dst[static_cast<size_t>(m * pb.dst_stride + k)],
                    kOutputPoison);
        }
    }
}

TEST_F(RoutedMoES4Reduction, CapturesFusedPathAndIsThreadStable) {
    s4_problem pb;
    pb.K = 128;
    pb.N = 64;
    pb.build(33, 0x4510u);
    route_first_slot_to_expert_zero(pb);

    oracle_options oracle;
    oracle.reduction = oracle_reduction_t::fused_f32;
    std::vector<uint16_t> reference;
    auto params = pb.params();
    params.num_threads = 1;
    ASSERT_NO_FATAL_FAILURE(check_native_direct(
            pb, params, oracle, w4_reduction_path_t::fused_f32, &reference));

    for (int threads : {2, 3, 4}) {
        if (threads > omp_get_max_threads() || threads > pb.K / 32) {
            continue;
        }
        SCOPED_TRACE(::testing::Message() << "threads=" << threads);
        params = pb.params();
        params.num_threads = threads;
        std::vector<uint16_t> output;
        ASSERT_NO_FATAL_FAILURE(check_native_direct(
                pb, params, oracle, w4_reduction_path_t::fused_f32, &output));
        EXPECT_EQ(output, reference);
    }
}

TEST_F(RoutedMoES4Reduction, CapturesScatterPathAndMatchesScatterOracle) {
    constexpr int requested_threads = 3;
    if (omp_get_max_threads() < requested_threads) {
        GTEST_SKIP() << "scatter requires more than K/32 effective threads";
    }

    s4_problem pb;
    pb.K = 64; // Two output-column blocks; three threads force scatter.
    pb.N = 64;
    pb.build(33, 0x4512u);
    route_first_slot_to_expert_zero(pb);
    oracle_options oracle;
    oracle.reduction = oracle_reduction_t::scatter_bf16;

    auto params = pb.params();
    params.num_threads = requested_threads;
    std::vector<uint16_t> reference;
    ASSERT_NO_FATAL_FAILURE(check_native_direct(
            pb, params, oracle, w4_reduction_path_t::scatter_bf16, &reference));

    if (omp_get_max_threads() >= 4) {
        params = pb.params();
        params.num_threads = 4;
        std::vector<uint16_t> output;
        ASSERT_NO_FATAL_FAILURE(check_native_direct(pb, params, oracle,
                w4_reduction_path_t::scatter_bf16, &output));
        EXPECT_EQ(output, reference);
    }
}

TEST_F(RoutedMoES4Execute, Bf16ScalesUseNativePathAndMatchOracle) {
    s4_problem pb;
    pb.build(8, 0x4467u);
    s4_call call(pb, /*bf16_scales=*/true);
    const std::vector<float> gate_scale
            = bf16_vector_to_f32(call.primary_wei_scale_bf16);
    const std::vector<float> down_scale
            = bf16_vector_to_f32(call.secondary_wei_scale_bf16);
    oracle_options oracle;
    oracle.reduction = oracle_reduction_t::fused_f32;
    oracle.gate_scale = gate_scale.data();
    oracle.down_scale = down_scale.data();
    ASSERT_NO_FATAL_FAILURE(check_native_high_level(
            pb, call, oracle, w4_reduction_path_t::fused_f32));
}

TEST_F(RoutedMoES4Fallback, K96N48UsesGroupedPathAndMatchesOracle) {
    s4_problem pb;
    pb.K = 96;
    pb.N = 48;
    pb.gate_group_size = 24;
    pb.down_group_size = 24;
    pb.build(8, 0x4468u);
    pb.s13 = bf16_vector_to_f32(f32_vector_to_bf16(pb.s13));
    pb.s2 = bf16_vector_to_f32(f32_vector_to_bf16(pb.s2));
    s4_call call(pb, /*bf16_scales=*/true);
    ASSERT_NO_FATAL_FAILURE(check_grouped_fallback(pb, call));
}

TEST_F(RoutedMoES4Fallback, ZeroPointDescriptorsBypassNative) {
    s4_problem pb;
    pb.build(8, 0x4469u);
    s4_call call(pb);
    const int8_t zero = 0;
    call.primary.params.quant_params.wei_zp.buff = &zero;
    call.primary.params.quant_params.wei_zp.dt = data_type_t::s8;
    call.primary.params.quant_params.wei_zp.dims = {1};
    call.secondary.params.quant_params.wei_zp.buff = &zero;
    call.secondary.params.quant_params.wei_zp.dt = data_type_t::s8;
    call.secondary.params.quant_params.wei_zp.dims = {1};
    ASSERT_NO_FATAL_FAILURE(check_grouped_fallback(pb, call));
}

TEST_F(RoutedMoES4Cache, FlushRebuildsMutatedWeightsAtSameAddress) {
    s4_problem pb;
    pb.build(8, 0x4477u);
    oracle_options oracle;
    oracle.reduction = oracle_reduction_t::fused_f32;

    std::vector<uint16_t> original;
    ASSERT_NO_FATAL_FAILURE(check_native_direct(pb, pb.params(), oracle,
            w4_reduction_path_t::fused_f32, &original));
    for (auto &byte : pb.w13) {
        byte = static_cast<int8_t>(static_cast<uint8_t>(byte) ^ 0x11u);
    }
    for (auto &byte : pb.w2) {
        byte = static_cast<int8_t>(static_cast<uint8_t>(byte) ^ 0x11u);
    }

    pb.poison_output();
    ASSERT_EQ(execute(pb.params()), status_t::success);
    EXPECT_EQ(pb.logical_output(), original)
            << "a warm pointer key must retain its packed image";

    group_matmul_routed_moe_flush_weight_cache();
    std::vector<uint16_t> rebuilt;
    ASSERT_NO_FATAL_FAILURE(check_native_direct(
            pb, pb.params(), oracle, w4_reduction_path_t::fused_f32, &rebuilt));
    EXPECT_NE(rebuilt, original);
    EXPECT_TRUE(std::any_of(rebuilt.begin(), rebuilt.end(),
            [](uint16_t value) { return value != 0; }));
    expect_numerically_close(
            rebuilt, scalar_s4_moe(pb, oracle), 1.2e-2, 0.04f, 0.04f);
}

TEST_F(RoutedMoES4Cache, Bf16ScaleCacheFollowsPackedWeightIdentity) {
    s4_problem pb;
    pb.build(8, 0x4478u);
    s4_call call(pb, /*bf16_scales=*/true);
    const std::vector<float> gate_scale
            = bf16_vector_to_f32(call.primary_wei_scale_bf16);
    const std::vector<float> down_scale
            = bf16_vector_to_f32(call.secondary_wei_scale_bf16);
    oracle_options original_oracle;
    original_oracle.reduction = oracle_reduction_t::fused_f32;
    original_oracle.gate_scale = gate_scale.data();
    original_oracle.down_scale = down_scale.data();

    std::vector<uint16_t> original;
    ASSERT_NO_FATAL_FAILURE(check_native_high_level(pb, call, original_oracle,
            w4_reduction_path_t::fused_f32, &original));

    std::vector<uint16_t> alternate_down_scale = call.secondary_wei_scale_bf16;
    for (auto &value : alternate_down_scale) {
        value = f32_to_bf16(2.0f * bf16_to_f32(value));
    }
    call.secondary.params.quant_params.wei_scale.buff
            = alternate_down_scale.data();

    std::vector<uint16_t> cached;
    ASSERT_NO_FATAL_FAILURE(check_native_high_level(pb, call, original_oracle,
            w4_reduction_path_t::fused_f32, &cached));
    EXPECT_EQ(cached, original);

    group_matmul_routed_moe_flush_weight_cache();
    const std::vector<float> alternate_down_f32
            = bf16_vector_to_f32(alternate_down_scale);
    oracle_options rebuilt_oracle = original_oracle;
    rebuilt_oracle.down_scale = alternate_down_f32.data();
    std::vector<uint16_t> rebuilt;
    ASSERT_NO_FATAL_FAILURE(check_native_high_level(pb, call, rebuilt_oracle,
            w4_reduction_path_t::fused_f32, &rebuilt));
    EXPECT_GT(rel_mae(rebuilt, original), 0.1);
}

TEST_F(RoutedMoES4Scratch, SmallLargeSmallReuseStaysNumericallyClean) {
    s4_problem small_problem;
    small_problem.K = 64;
    small_problem.N = 32;
    small_problem.E = 4;
    small_problem.build(3, 0x4701u);
    oracle_options small_oracle;
    small_oracle.reduction = oracle_reduction_t::fused_f32;
    std::vector<uint16_t> first_small;
    ASSERT_NO_FATAL_FAILURE(check_native_direct(small_problem,
            small_problem.params(), small_oracle,
            w4_reduction_path_t::fused_f32, &first_small));

    s4_problem large;
    large.K = 256;
    large.N = 128;
    large.E = 8;
    large.topk = 4;
    large.build(65, 0x4702u);
    oracle_options large_oracle;
    large_oracle.reduction = oracle_reduction_t::fused_f32;
    ASSERT_NO_FATAL_FAILURE(check_native_direct(large, large.params(),
            large_oracle, w4_reduction_path_t::fused_f32));

    std::vector<uint16_t> second_small;
    ASSERT_NO_FATAL_FAILURE(check_native_direct(small_problem,
            small_problem.params(), small_oracle,
            w4_reduction_path_t::fused_f32, &second_small));
    EXPECT_EQ(second_small, first_small);
}
