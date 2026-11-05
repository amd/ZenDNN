/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

// The scale and zero-point layout rules of lowoha_quant_layout.hpp, and the
// AOCL-DLP wrapper's count and sym-quant rules of aocl_postop.hpp, over every
// granularity and dims edge, and what matmul_direct's validation, the AOCL-DLP
// post-op metadata and the native BRGEMM INT8 checks decide through them.

#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "lowoha_operators/matmul/backends/aocl/aocl_postop.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/lowoha_quant_layout.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_call.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "lowoha_operators/matmul/matmul_native/native_matmul.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
using zendnnl::common::data_type_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::error_handling::status_t;
using dims_t = std::vector<int64_t>;
using quant_t = mm::matmul_quantization_params_t::matmul_quant_t;
using granularity_t = mm::quant_count_granularity_t;

constexpr int64_t kBig = int64_t(1) << 32;
constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

float g_scales[1 << 14];
int32_t g_zero_points[1 << 14];

quant_t quant(const dims_t &dims, const void *buff = g_scales,
        data_type_t dt = data_type_t::f32) {
    quant_t q;
    q.buff = buff;
    q.dt = dt;
    q.dims = dims;
    return q;
}

mm::matmul_quantization_params_t scales(const dims_t &src, const dims_t &wei) {
    mm::matmul_quantization_params_t qp;
    qp.src_scale = quant(src);
    qp.wei_scale = quant(wei);
    return qp;
}

} // namespace

// ── Element counts ──────────────────────────────────────────────────────

TEST(QuantLayout, CheckedCountRejectsBadDims) {
    EXPECT_EQ(mm::checked_quant_param_num_elements({}), 1);
    EXPECT_EQ(mm::checked_quant_param_num_elements({1}), 1);
    EXPECT_EQ(mm::checked_quant_param_num_elements({1, 1}), 1);
    EXPECT_EQ(mm::checked_quant_param_num_elements({64}), 64);
    EXPECT_EQ(mm::checked_quant_param_num_elements({4, 8}), 32);
    EXPECT_EQ(mm::checked_quant_param_num_elements({2, 3, 4}), 24);
    EXPECT_EQ(mm::checked_quant_param_num_elements({kMax}), kMax);
    EXPECT_EQ(mm::checked_quant_param_num_elements({0}), -1);
    EXPECT_EQ(mm::checked_quant_param_num_elements({4, 0}), -1);
    EXPECT_EQ(mm::checked_quant_param_num_elements({-1}), -1);
    EXPECT_EQ(mm::checked_quant_param_num_elements({-1, -1}), -1);
    EXPECT_EQ(mm::checked_quant_param_num_elements({kBig, kBig}), -1)
            << "overflow";
    EXPECT_EQ(mm::checked_quant_param_num_elements({kMax, 2}), -1)
            << "overflow";
}

TEST(QuantLayout, AoclCountIsUnchecked) {
    EXPECT_EQ(mm::get_num_elements({}), 1u);
    EXPECT_EQ(mm::get_num_elements({1}), 1u);
    EXPECT_EQ(mm::get_num_elements({1, 1}), 1u);
    EXPECT_EQ(mm::get_num_elements({4, 8}), 32u);
    EXPECT_EQ(mm::get_num_elements({4, 0}), 0u);
    EXPECT_EQ(mm::get_num_elements({kBig, kBig}), 0u) << "wraps";
    EXPECT_EQ(mm::get_num_elements({-1, -1}), 1u) << "wraps";
}

// ── Per-tensor, per-channel, per-token ──────────────────────────────────

TEST(QuantLayout, PerTensorIsOneElement) {
    for (const dims_t &d :
            {dims_t {}, dims_t {1}, dims_t {1, 1}, dims_t {1, 1, 1}})
        EXPECT_TRUE(mm::quant_param_dims_are_per_tensor(d)) << d.size();
    for (const dims_t &d : {dims_t {2}, dims_t {1, 2}, dims_t {2, 1},
                 dims_t {0}, dims_t {-1, -1}, dims_t {kBig, kBig}})
        EXPECT_FALSE(mm::quant_param_dims_are_per_tensor(d));
    EXPECT_TRUE(mm::quant_param_is_per_tensor(quant({1, 1})));
    EXPECT_FALSE(mm::quant_param_is_per_tensor(quant({1, 1}, nullptr)))
            << "needs a buffer";
}

TEST(QuantLayout, PerChannelIsNOrOneByN) {
    const int N = 32;
    EXPECT_TRUE(mm::quant_param_is_per_channel(quant({N}), N));
    EXPECT_TRUE(mm::quant_param_is_per_channel(quant({1, N}), N));
    EXPECT_FALSE(mm::quant_param_is_per_channel(quant({N, 1}), N));
    EXPECT_FALSE(mm::quant_param_is_per_channel(quant({1, N, 1}), N));
    EXPECT_FALSE(mm::quant_param_is_per_channel(quant({2, N}), N));
    EXPECT_FALSE(mm::quant_param_is_per_channel(quant({N + 1}), N));
    EXPECT_FALSE(mm::quant_param_is_per_channel(quant({1, 1}), N));
    EXPECT_FALSE(mm::quant_param_is_per_channel(quant({}), N));
    EXPECT_FALSE(mm::quant_param_is_per_channel(quant({N}, nullptr), N));
    EXPECT_FALSE(mm::quant_param_is_per_channel(quant({0}), 0)) << "N <= 0";
    EXPECT_TRUE(mm::quant_param_is_per_channel(quant({1}), 1))
            << "N == 1: {1} is per-channel too";
}

TEST(QuantLayout, PerTokenIsMByOne) {
    const int M = 6;
    EXPECT_TRUE(mm::quant_param_is_per_token(quant({M, 1}), M));
    EXPECT_TRUE(mm::quant_param_is_per_token(quant({M, 1}, nullptr), M))
            << "judged by dims alone";
    EXPECT_FALSE(mm::quant_param_is_per_token(quant({M}), M));
    EXPECT_FALSE(mm::quant_param_is_per_token(quant({1, M}), M));
    EXPECT_FALSE(mm::quant_param_is_per_token(quant({M, 1, 1}), M));
    EXPECT_FALSE(mm::quant_param_is_per_token(quant({M, 2}), M));
    EXPECT_FALSE(mm::quant_param_is_per_token(quant({M + 1, 1}), M));
    EXPECT_FALSE(mm::quant_param_is_per_token(quant({0, 1}), 0)) << "M <= 0";
    EXPECT_TRUE(mm::quant_param_is_per_token(quant({1, 1}), 1));
}

TEST(QuantLayout, TensorOrRowValues) {
    const int64_t M = 6;
    EXPECT_EQ(mm::quant_param_tensor_or_row_values({}, M), 1);
    EXPECT_EQ(mm::quant_param_tensor_or_row_values({1}, M), 1);
    EXPECT_EQ(mm::quant_param_tensor_or_row_values({1, 1}, M), 1);
    EXPECT_EQ(mm::quant_param_tensor_or_row_values({1, 1, 1}, M), 1);
    EXPECT_EQ(mm::quant_param_tensor_or_row_values({M}, M), M);
    EXPECT_EQ(mm::quant_param_tensor_or_row_values({M, 1}, M), M);
    EXPECT_EQ(mm::quant_param_tensor_or_row_values({1, M}, M), 0);
    EXPECT_EQ(mm::quant_param_tensor_or_row_values({M, 1, 1}, M), 0);
    EXPECT_EQ(mm::quant_param_tensor_or_row_values({M, 2}, M), 0);
    EXPECT_EQ(mm::quant_param_tensor_or_row_values({M + 1}, M), 0);
    EXPECT_EQ(mm::quant_param_tensor_or_row_values({0}, M), 0);
    EXPECT_EQ(mm::quant_param_tensor_or_row_values({kBig, kBig}, M), 0);
    EXPECT_EQ(mm::quant_param_tensor_or_row_values({1}, 1), 1);
}

// ── Groups along K, by dims ─────────────────────────────────────────────

TEST(QuantLayout, WeightGroupsAreGByN) {
    const int N = 32, G = 4;
    EXPECT_TRUE(mm::quant_param_dims_are_per_group_wei({G, N}, N));
    EXPECT_FALSE(mm::quant_param_dims_are_per_group_wei({1, N}, N));
    EXPECT_FALSE(mm::quant_param_dims_are_per_group_wei({G, N + 1}, N));
    EXPECT_FALSE(mm::quant_param_dims_are_per_group_wei({G, 1}, N));
    EXPECT_FALSE(mm::quant_param_dims_are_per_group_wei({G, N, 1}, N));
    EXPECT_FALSE(mm::quant_param_dims_are_per_group_wei({G * N}, N));
    EXPECT_FALSE(mm::quant_param_dims_are_per_group_wei({0, N}, N));

    EXPECT_TRUE(mm::has_per_group_wei_scale(scales({}, {G, N}), N));
    EXPECT_FALSE(mm::has_per_group_wei_scale(scales({}, {1, N}), N));
    EXPECT_TRUE(mm::has_per_channel_wei_scale(scales({}, {1, N}), N));
    EXPECT_FALSE(mm::has_per_channel_wei_scale(scales({}, {N}), N))
            << "the 2-D form only";
    EXPECT_FALSE(mm::has_per_channel_wei_scale(scales({}, {G, N}), N));
    auto unset = scales({}, {G, N});
    unset.wei_scale.buff = nullptr;
    EXPECT_FALSE(mm::has_per_group_wei_scale(unset, N));
    unset.wei_scale.dims = {1, N};
    EXPECT_FALSE(mm::has_per_channel_wei_scale(unset, N));
    // Over 2-D {g >= 1, N} scales exactly one of the two holds.
    for (int64_t g = 1; g <= 8; ++g) {
        const auto qp = scales({}, {g, N});
        EXPECT_NE(mm::has_per_group_wei_scale(qp, N),
                mm::has_per_channel_wei_scale(qp, N))
                << g;
    }
}

TEST(QuantLayout, SourceGroupsAreMByG) {
    const int64_t M = 6, G = 4;
    EXPECT_TRUE(mm::quant_param_is_per_group_src(quant({M, G}), M, G));
    EXPECT_TRUE(mm::quant_param_is_per_group_src(quant({M, G}, nullptr), M, G))
            << "judged by dims alone";
    EXPECT_FALSE(mm::quant_param_is_per_group_src(quant({M, 1}), M, 1))
            << "one group is per-token";
    EXPECT_FALSE(mm::quant_param_is_per_group_src(quant({M, 8}), M, G));
    EXPECT_FALSE(mm::quant_param_is_per_group_src(quant({G, M}), M, G));
    EXPECT_FALSE(mm::quant_param_is_per_group_src(quant({M * G}), M, G));
    EXPECT_FALSE(mm::quant_param_is_per_group_src(quant({M, G, 1}), M, G));
}

// ── Layouts by element count ────────────────────────────────────────────

TEST(QuantLayout, CountGranularity) {
    // The values are part of the AOCL-DLP post-op cache signature.
    EXPECT_EQ(static_cast<int>(granularity_t::per_tensor), 0);
    EXPECT_EQ(static_cast<int>(granularity_t::per_axis), 1);
    EXPECT_EQ(static_cast<int>(granularity_t::per_group), 2);

    const int64_t M = 6;
    EXPECT_EQ(mm::quant_count_granularity(0, M), granularity_t::per_tensor);
    EXPECT_EQ(mm::quant_count_granularity(1, M), granularity_t::per_tensor);
    EXPECT_EQ(mm::quant_count_granularity(1, 1), granularity_t::per_tensor)
            << "one element over one row";
    EXPECT_EQ(mm::quant_count_granularity(M, M), granularity_t::per_axis);
    EXPECT_EQ(mm::quant_count_granularity(4 * M, M), granularity_t::per_group);
    EXPECT_EQ(mm::quant_count_granularity(M + 1, M), granularity_t::per_group);
    EXPECT_EQ(mm::quant_count_granularity(2, 1), granularity_t::per_group);
    EXPECT_EQ(mm::quant_count_granularity(mm::get_num_elements({-1, 2}), M),
            granularity_t::per_group);

    EXPECT_TRUE(mm::quant_count_is_per_axis(M, M));
    EXPECT_TRUE(mm::quant_count_is_per_axis(1, 1))
            << "a single element over an extent of one";
    EXPECT_FALSE(mm::quant_count_is_per_axis(1, M));
    EXPECT_FALSE(mm::quant_count_is_per_axis(4 * M, M));
    EXPECT_FALSE(mm::quant_count_is_per_axis(0, M));
}

TEST(QuantLayout, CountSourceGroups) {
    const int64_t M = 6, K = 64;
    EXPECT_EQ(mm::quant_count_src_groups(M, M, K), 1);
    EXPECT_EQ(mm::quant_count_src_groups(4 * M, M, K), 4);
    EXPECT_EQ(mm::quant_count_src_groups(16 * M, M, K), 16);
    EXPECT_EQ(mm::quant_count_src_groups(3 * M, M, K), 0)
            << "3 does not split K";
    EXPECT_EQ(mm::quant_count_src_groups(4 * M + 1, M, K), 0);
    EXPECT_EQ(mm::quant_count_src_groups(M - 1, M, K), 0);
    EXPECT_EQ(mm::quant_count_src_groups(2 * M, M, 30), 2);
    EXPECT_EQ(mm::quant_count_src_groups(4 * M, M, 30), 0);
    EXPECT_EQ(mm::quant_count_src_groups(1, 1, K), 1);
    EXPECT_EQ(mm::quant_count_src_groups(M, 0, K), 0) << "M <= 0";
}

// ── Group size along K ──────────────────────────────────────────────────

TEST(QuantLayout, GroupsAlongK) {
    EXPECT_TRUE(mm::quant_groups_divide(128, 4));
    EXPECT_TRUE(mm::quant_groups_divide(30, 3));
    EXPECT_FALSE(mm::quant_groups_divide(30, 4));
    EXPECT_FALSE(mm::quant_groups_divide(128, 0));
    EXPECT_FALSE(mm::quant_groups_divide(128, -4));

    EXPECT_TRUE(mm::quant_groups_are_whole_quads(128, 4)) << "groups of 32";
    EXPECT_TRUE(mm::quant_groups_are_whole_quads(128, 32)) << "groups of 4";
    EXPECT_TRUE(mm::quant_groups_are_whole_quads(96, 3)) << "groups of 32";
    EXPECT_TRUE(mm::quant_groups_are_whole_quads(128, 1)) << "one group";
    EXPECT_FALSE(mm::quant_groups_are_whole_quads(128, 64)) << "groups of 2";
    EXPECT_FALSE(mm::quant_groups_are_whole_quads(30, 3)) << "groups of 10";
    EXPECT_FALSE(mm::quant_groups_are_whole_quads(30, 4)) << "4 does not split";
    EXPECT_FALSE(mm::quant_groups_are_whole_quads(128, 0));

    EXPECT_TRUE(mm::quant_group_size_is_whole_quads(128, 32));
    EXPECT_TRUE(mm::quant_group_size_is_whole_quads(128, 128));
    EXPECT_FALSE(mm::quant_group_size_is_whole_quads(128, 6));
    EXPECT_FALSE(mm::quant_group_size_is_whole_quads(100, 8));
    EXPECT_FALSE(mm::quant_group_size_is_whole_quads(128, 0));
    EXPECT_FALSE(mm::quant_group_size_is_whole_quads(128, -4));
}

TEST(QuantLayout, SymQuantGroupSize) {
    const int M = 6, N = 32, K = 128;
    EXPECT_EQ(mm::sym_quant_group_size(scales({M, 1}, {4, N}), M, N, K), 32);
    EXPECT_EQ(mm::sym_quant_group_size(scales({M, 4}, {4, N}), M, N, K), 32);
    EXPECT_EQ(mm::sym_quant_group_size(scales({M, 1}, {1, N}), M, N, K), K);
    EXPECT_EQ(mm::sym_quant_group_size(scales({M, 1}, {N}), M, N, K), K);
    // A weight scale other than {G, N} with G dividing K: the source's groups.
    EXPECT_EQ(mm::sym_quant_group_size(scales({M, 4}, {3, N}), M, N, K), 32);
    EXPECT_EQ(mm::sym_quant_group_size(scales({M, 8}, {N}), M, N, K), 16);
    EXPECT_EQ(
            mm::sym_quant_group_size(scales({M, 4}, {4, N + 1}), M, N, K), 32);
    EXPECT_EQ(mm::sym_quant_group_size(scales({}, {N}), M, N, K), K);
    EXPECT_EQ(mm::sym_quant_group_size(scales({1, 1}, {N}), M, N, K), K);
}

// ── Source scales of the AOCL-DLP sym-quant kernel ──────────────────────

TEST(QuantLayout, CollapsedPerTokenAndSymQuantSource) {
    const int N = 32;
    const auto count = [](const mm::matmul_quantization_params_t &qp) {
        return mm::get_num_elements(qp.src_scale.dims);
    };
    for (const dims_t &w : {dims_t {1, N}, dims_t {4, N}}) {
        const auto qp = scales({1, 1}, w);
        EXPECT_TRUE(mm::src_scale_is_collapsed_per_token(qp, 1, N));
        EXPECT_TRUE(mm::src_scale_selects_sym_quant(qp, count(qp), 1, N));
        EXPECT_FALSE(mm::src_scale_is_collapsed_per_token(qp, 2, N))
                << "M > 1: one element is per-tensor";
        EXPECT_EQ(mm::src_scale_is_broadcast_per_token(qp, 2, N), w[0] > 1);
        EXPECT_EQ(
                mm::src_scale_selects_sym_quant(qp, count(qp), 2, N), w[0] > 1)
                << "per-group weight scales take it once per row";
    }
    for (const dims_t &w : {dims_t {N}, dims_t {1, 1}, dims_t {N, 1}}) {
        const auto qp = scales({1}, w);
        EXPECT_FALSE(mm::src_scale_is_collapsed_per_token(qp, 1, N))
                << "the weight scale must be 2-D {G, N}";
        EXPECT_FALSE(mm::src_scale_selects_sym_quant(qp, count(qp), 1, N));
    }
    auto unset = scales({1, 1}, {1, N});
    unset.src_scale.buff = nullptr;
    EXPECT_FALSE(mm::src_scale_is_collapsed_per_token(unset, 1, N));
    for (const dims_t &s :
            {dims_t {4, 1}, dims_t {4}, dims_t {1, 4}, dims_t {4, 2}}) {
        const auto qp = scales(s, {N});
        EXPECT_TRUE(mm::src_scale_selects_sym_quant(qp, count(qp), 4, N))
                << "more than one element";
    }
    const auto zero = scales({0}, {1, N});
    EXPECT_FALSE(mm::src_scale_selects_sym_quant(zero, count(zero), 1, N));
}

// ── matmul_direct's validation ──────────────────────────────────────────

namespace {

struct validation_call_t {
    int M = 6, N = 32, K = 64;
    mm::matmul_params params;
    validation_call_t(data_type_t src, data_type_t wei, data_type_t dst) {
        params.dtypes.src = src;
        params.dtypes.wei = wei;
        params.dtypes.dst = dst;
    }
    status_t validate() const {
        static int8_t buffer[1 << 16];
        return mm::validate_matmul_direct_inputs(buffer, buffer, buffer, M, N,
                K, 1, 1, params, /*is_weights_const=*/true);
    }
    status_t with(const dims_t &src, const dims_t &wei) {
        params.quant_params.src_scale = quant(src);
        params.quant_params.wei_scale = quant(wei);
        return validate();
    }
};

} // namespace

TEST(QuantLayoutValidation, SourceScalesJudgedByCount) {
    validation_call_t c(data_type_t::s8, data_type_t::s8, data_type_t::f32);
    const int64_t M = c.M, N = c.N;
    // Per-token: M values whatever the dims, with N weight values or {G, N}.
    EXPECT_EQ(c.with({M, 1}, {N}), status_t::success);
    EXPECT_EQ(c.with({M, 1}, {1, N}), status_t::success);
    EXPECT_EQ(c.with({M, 1}, {N, 1}), status_t::success);
    EXPECT_EQ(c.with({M}, {1, N}), status_t::success);
    EXPECT_EQ(c.with({1, M}, {1, N}), status_t::success);
    EXPECT_EQ(c.with({M, 1}, {4, N}), status_t::success);
    EXPECT_EQ(c.with({M, 1}, {1, 1}), status_t::failure);
    EXPECT_EQ(c.with({M, 1}, {4 * N}), status_t::failure)
            << "a flat per-group weight scale needs its {G, N} dims";
    // Per-group: a whole number of values per row splitting K, with as many
    // weight values per column.
    EXPECT_EQ(c.with({M, 4}, {4, N}), status_t::success);
    EXPECT_EQ(c.with({M * 4}, {4 * N}), status_t::success);
    EXPECT_EQ(c.with({M, 4}, {1, N}), status_t::failure);
    EXPECT_EQ(c.with({M, 4}, {8, N}), status_t::failure);
    EXPECT_EQ(c.with({M, 3}, {3, N}), status_t::failure)
            << "3 does not split K";
    EXPECT_EQ(c.with({M * 4 + 1}, {N}), status_t::failure);
    EXPECT_EQ(c.with({kBig, kBig}, {N}), status_t::failure) << "overflow";
    EXPECT_EQ(c.with({0}, {N}), status_t::failure);
    // Per-tensor source scales pair with any weight scale.
    EXPECT_EQ(c.with({1, 1}, {N}), status_t::success);
    EXPECT_EQ(c.with({}, {3, 5}), status_t::success);
}

TEST(QuantLayoutValidation, ZeroPointsAndDestinationScalesPerTensor) {
    validation_call_t c(data_type_t::u8, data_type_t::s8, data_type_t::u8);
    c.params.quant_params.src_scale = quant({1});
    c.params.quant_params.wei_scale = quant({c.N});
    c.params.quant_params.src_zp = quant({1}, g_zero_points, data_type_t::s32);
    c.params.quant_params.dst_scale = quant({1, 1});
    c.params.quant_params.dst_zp = quant({}, g_zero_points, data_type_t::s32);
    EXPECT_EQ(c.validate(), status_t::success);
    c.params.quant_params.src_zp.dims = {c.M, 1};
    EXPECT_EQ(c.validate(), status_t::failure);
    c.params.quant_params.src_zp.dims = {1};
    c.params.quant_params.dst_scale.dims = {c.N};
    EXPECT_EQ(c.validate(), status_t::failure);
    c.params.quant_params.dst_scale.dims = {1};
    c.params.quant_params.dst_zp.dims = {kBig, kBig};
    EXPECT_EQ(c.validate(), status_t::failure);
}

TEST(QuantLayoutValidation, W4A8Groups) {
    validation_call_t c(data_type_t::s8, data_type_t::s4, data_type_t::bf16);
    c.params.dtypes.compute = data_type_t::s8;
    const int64_t M = c.M, N = c.N;
    EXPECT_EQ(c.with({M, 1}, {4, N}), status_t::success);
    EXPECT_EQ(c.with({M, 4}, {4, N}), status_t::success);
    EXPECT_EQ(c.with({M, 1}, {1, N}), status_t::success) << "one group";
    EXPECT_EQ(c.with({M, 8}, {4, N}), status_t::failure);
    EXPECT_EQ(c.with({1, 1}, {4, N}), status_t::failure) << "per-tensor source";
    EXPECT_EQ(c.with({M, 1}, {3, N}), status_t::failure)
            << "3 does not split K";
    EXPECT_EQ(c.with({M, 1}, {32, N}), status_t::failure) << "groups of 2";
    EXPECT_EQ(c.with({M, 1}, {4, N + 1}), status_t::failure);
    c.K = 96;
    EXPECT_EQ(c.with({M, 3}, {3, N}), status_t::success) << "groups of 32";
}

// ── AOCL-DLP post-op metadata ───────────────────────────────────────────

#if ZENDNNL_DEPENDS_AOCLDLP
namespace {

class QuantLayoutAoclMetadata : public ::testing::Test {
protected:
    void SetUp() override { mm::clear_aocl_postop_metadata_cache(); }
    void TearDown() override { mm::clear_aocl_postop_metadata_cache(); }

    int8_t weight_[64] = {};

    dlp_metadata_t *create(const mm::matmul_params &p, int M, int N, int K,
            matmul_algo_t kernel = matmul_algo_t::aocl_dlp_blocked) {
        return mm::create_dlp_post_op(p, /*bias=*/nullptr, p.dtypes, N, K, M,
                /*zp_comp_acc=*/nullptr, /*zp_comp_ndim=*/0, kernel, weight_,
                /*is_weights_const=*/true);
    }
};

mm::matmul_params s8s8(const dims_t &src, const dims_t &wei) {
    mm::matmul_params p;
    p.dtypes.src = data_type_t::s8;
    p.dtypes.wei = data_type_t::s8;
    p.dtypes.dst = data_type_t::bf16;
    p.quant_params = scales(src, wei);
    return p;
}

} // namespace

TEST_F(QuantLayoutAoclMetadata, SymQuantScaleDims) {
    const int M = 4, N = 32, K = 64, G = 4;
    dlp_metadata_t *md = create(s8s8({M, 1}, {1, N}), M, N, K);
    ASSERT_NE(md, nullptr);
    ASSERT_NE(md->a_quant_op, nullptr);
    ASSERT_NE(md->b_quant_op, nullptr);
    EXPECT_EQ(md->a_quant_op->dequant_scale_factors->outer_dim,
            DLP_PARAM_DIM_PER_TOKEN);
    EXPECT_EQ(md->a_quant_op->dequant_scale_factors->len, M);
    EXPECT_EQ(md->b_quant_op->dequant_scale_factors->outer_dim,
            DLP_PARAM_DIM_PER_CHANNEL);
    EXPECT_EQ(md->b_quant_op->group_size, K);

    md = create(s8s8({M, G}, {G, N}), M, N, K);
    ASSERT_NE(md, nullptr);
    ASSERT_NE(md->a_quant_op, nullptr);
    EXPECT_EQ(md->a_quant_op->dequant_scale_factors->outer_dim,
            DLP_PARAM_DIM_PER_GROUP);
    EXPECT_EQ(md->b_quant_op->dequant_scale_factors->outer_dim,
            DLP_PARAM_DIM_PER_GROUP);
    EXPECT_EQ(md->b_quant_op->dequant_scale_factors->len, G * N);
    EXPECT_EQ(md->b_quant_op->group_size, K / G);
}

TEST_F(QuantLayoutAoclMetadata, CollapsedPerTokenAtOneRow) {
    const int N = 32, K = 64;
    // One source value over one row with a 2-D weight scale: the sym-quant
    // kernel's per-token source, one value per row.
    dlp_metadata_t *md = create(s8s8({1, 1}, {1, N}), 1, N, K);
    ASSERT_NE(md, nullptr);
    ASSERT_NE(md->a_quant_op, nullptr);
    EXPECT_EQ(md->a_quant_op->dequant_scale_factors->outer_dim,
            DLP_PARAM_DIM_PER_TOKEN);
    EXPECT_EQ(md->seq_length, 0);
    // A 1-D weight scale leaves it per-tensor: both scales are SCALE post-ops.
    // Both weight scales hold N values, so the two calls share a cache key.
    mm::clear_aocl_postop_metadata_cache();
    md = create(s8s8({1, 1}, {N}), 1, N, K);
    ASSERT_NE(md, nullptr);
    EXPECT_EQ(md->a_quant_op, nullptr);
    ASSERT_EQ(md->seq_length, 2);
    EXPECT_EQ(md->seq_vector[0], SCALE);
    EXPECT_EQ(md->scale[0].sf->scale_factor_dim, DLP_PARAM_DIM_PER_TENSOR);
    EXPECT_EQ(md->seq_vector[1], SCALE);
    EXPECT_EQ(md->scale[1].sf->scale_factor_dim, DLP_PARAM_DIM_PER_CHANNEL);
    EXPECT_EQ(md->scale[1].sf->scale_factor_len, N);
}

TEST_F(QuantLayoutAoclMetadata, OneColumnWeightScaleIsPerChannel) {
    const int M = 4, N = 1, K = 64;
    dlp_metadata_t *md = create(s8s8({M, 1}, {1, 1}), M, N, K);
    ASSERT_NE(md, nullptr);
    ASSERT_NE(md->b_quant_op, nullptr);
    EXPECT_EQ(md->b_quant_op->dequant_scale_factors->outer_dim,
            DLP_PARAM_DIM_PER_CHANNEL);
}

TEST_F(QuantLayoutAoclMetadata, WeightOnlyQuantScaleDims) {
    const int M = 4, N = 32, K = 64;
    const auto woq = [&](const dims_t &wei) {
        mm::matmul_params p;
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = data_type_t::s4;
        p.dtypes.dst = data_type_t::bf16;
        p.quant_params.wei_scale = quant(wei);
        return create(p, M, N, K);
    };
    struct expect_t {
        dims_t dims;
        DLP_PARAM_DIM_TYPE dim;
        md_t len, group;
    };
    for (const expect_t &e :
            {expect_t {{4, N}, DLP_PARAM_DIM_PER_GROUP, 4 * N, K / 4},
                    expect_t {{N}, DLP_PARAM_DIM_PER_CHANNEL, N, K},
                    expect_t {{1, N}, DLP_PARAM_DIM_PER_CHANNEL, N, K},
                    expect_t {{1, 1}, DLP_PARAM_DIM_PER_TENSOR, 1, K},
                    expect_t {{}, DLP_PARAM_DIM_PER_TENSOR, 1, K},
                    expect_t {{3, N}, DLP_PARAM_DIM_PER_GROUP, 3 * N, K},
                    expect_t {{64, N}, DLP_PARAM_DIM_PER_GROUP, 64 * N, K}}) {
        dlp_metadata_t *md = woq(e.dims);
        ASSERT_NE(md, nullptr);
        ASSERT_NE(md->b_quant_op, nullptr);
        EXPECT_EQ(md->b_quant_op->dequant_scale_factors->outer_dim, e.dim)
                << e.dims.size();
        EXPECT_EQ(md->b_quant_op->dequant_scale_factors->len, e.len)
                << e.dims.size();
        EXPECT_EQ(md->b_quant_op->group_size, e.group) << e.dims.size();
    }
}
#endif

// ── Native BRGEMM INT8 checks ───────────────────────────────────────────

namespace {

struct native_call_t {
    nat::GemmDescriptor desc;
    mm::matmul_params params;
    native_call_t(data_type_t src, int M, int N, int K) {
        params.dtypes.src = src;
        params.dtypes.wei = data_type_t::s8;
        params.dtypes.dst = data_type_t::f32;
        desc.M = M;
        desc.N = N;
        desc.K = K;
        desc.lda = K;
        desc.ldb = N;
        desc.ldc = N;
        desc.src_dt = src;
        desc.wei_dt = data_type_t::s8;
        desc.dst_dt = data_type_t::f32;
    }
    status_t check(const dims_t &src, const dims_t &wei) {
        params.quant_params.src_scale = quant(src);
        params.quant_params.wei_scale = quant(wei);
        return nat::check_int8_call(
                desc, params, nat::decide_int8_call(desc, params, nullptr));
    }
};

} // namespace

TEST(NativeQuantLayout, PerChannelScaleDims) {
    const int M = 6, N = 32, K = 64;
    native_call_t c(data_type_t::s8, M, N, K);
    // Weight scales: per-tensor or {N} / {1, N}.
    EXPECT_EQ(c.check({1}, {N}), status_t::success);
    EXPECT_EQ(c.check({1}, {1, N}), status_t::success);
    EXPECT_EQ(c.check({1}, {1, 1}), status_t::success);
    EXPECT_EQ(c.check({1}, {}), status_t::success);
    EXPECT_EQ(c.check({1}, {N, 1}), status_t::unimplemented);
    EXPECT_EQ(c.check({1}, {1, N, 1}), status_t::unimplemented);
    EXPECT_EQ(c.check({1}, {kBig, kBig}), status_t::unimplemented);
    // Source scales of a u8 / s8 source: per-tensor, {M} or {M, 1}.
    EXPECT_EQ(c.check({M}, {N}), status_t::success);
    EXPECT_EQ(c.check({M, 1}, {N}), status_t::success);
    EXPECT_EQ(c.check({1, M}, {N}), status_t::unimplemented);
    EXPECT_EQ(c.check({M, 1, 1}, {N}), status_t::unimplemented);
    EXPECT_EQ(c.check({0}, {N}), status_t::unimplemented);
    // A dynamically quantized source takes per-tensor or {M, 1} only.
    native_call_t d(data_type_t::bf16, M, N, K);
    d.params.dtypes.dst = data_type_t::bf16;
    d.desc.dst_dt = data_type_t::bf16;
    d.params.dynamic_quant = true;
    d.params.dtypes.compute = data_type_t::s8;
    EXPECT_EQ(d.check({M, 1}, {N}), status_t::success);
    EXPECT_EQ(d.check({1, 1}, {N}), status_t::success);
    EXPECT_EQ(d.check({M}, {N}), status_t::unimplemented);
}

TEST(NativeQuantLayout, PerGroupScaleDims) {
    const int M = 6, N = 32, K = 128;
    native_call_t c(data_type_t::s8, M, N, K);
    EXPECT_EQ(nat::int8_scale_groups(c.params, N), 0);
    EXPECT_EQ(c.check({M, 4}, {4, N}), status_t::success);
    EXPECT_EQ(nat::int8_scale_groups(c.params, N), 4);
    EXPECT_EQ(c.check({M, 1}, {4, N}), status_t::success);
    EXPECT_EQ(c.check({1, 1}, {4, N}), status_t::success);
    EXPECT_EQ(c.check({M}, {4, N}), status_t::success);
    EXPECT_EQ(c.check({M, 8}, {4, N}), status_t::unimplemented);
    EXPECT_EQ(c.check({1, M}, {4, N}), status_t::unimplemented);
    EXPECT_EQ(c.check({M, 32}, {32, N}), status_t::success) << "groups of 4";
    EXPECT_EQ(c.check({M, 64}, {64, N}), status_t::unimplemented)
            << "groups of 2";
    EXPECT_EQ(c.check({M, 3}, {3, N}), status_t::unimplemented)
            << "3 does not split K";
    EXPECT_EQ(nat::int8_scale_groups(c.params, N), 3);
}

TEST(NativeQuantLayout, LayoutGroupSize) {
    const int N = 32, K = 128;
    mm::matmul_params p;
    p.dtypes.src = data_type_t::s8;
    p.dtypes.wei = data_type_t::s8;
    p.dtypes.dst = data_type_t::f32;
    const auto group_size = [&](const dims_t &wei) {
        p.quant_params.wei_scale = quant(wei);
        return nat::native_layout_query_for(p, K, N, N, false).wei_group_size;
    };
    EXPECT_EQ(group_size({4, N}), 32);
    EXPECT_EQ(group_size({64, N}), 2);
    EXPECT_EQ(group_size({3, N}), -1) << "3 does not split K";
    EXPECT_EQ(group_size({1, N}), 0);
    EXPECT_EQ(group_size({N}), 0);

    nat::native_layout_query_t q
            = nat::native_layout_query_for(p, K, N, N, false);
    q.wei_group_size = 0;
    if (!nat::decide_native_layout(q).native)
        GTEST_SKIP() << "the native layout is not served here";
    for (const int gs : {32, 4, K}) {
        q.wei_group_size = gs;
        EXPECT_TRUE(nat::decide_native_layout(q).native) << gs;
    }
    for (const int gs : {2, 6, 24, -1}) {
        q.wei_group_size = gs;
        EXPECT_FALSE(nat::decide_native_layout(q).native) << gs;
    }
}
