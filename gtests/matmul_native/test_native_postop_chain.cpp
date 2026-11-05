/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

// Post-op chains of two ops, an activation the kernel epilogues can fuse and
// an op they cannot, in both orders: the native BRGEMM's BF16, INT8 and FP32
// calls apply them in chain order.

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "native_brgemm_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::post_op_type_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;

using chain_t = std::vector<post_op_type_t>;

/// The chains of the report, in both orders.
const std::vector<chain_t> &chains() {
    static const std::vector<chain_t> c = {
            {post_op_type_t::binary_add, post_op_type_t::sigmoid},
            {post_op_type_t::sigmoid, post_op_type_t::binary_add},
            {post_op_type_t::binary_mul, post_op_type_t::relu},
            {post_op_type_t::relu, post_op_type_t::binary_mul},
            {post_op_type_t::clip, post_op_type_t::tanh},
            {post_op_type_t::tanh, post_op_type_t::clip},
            {post_op_type_t::binary_add, post_op_type_t::gelu_erf},
            {post_op_type_t::gelu_erf, post_op_type_t::binary_add},
    };
    return c;
}

std::string chain_name(const chain_t &chain) {
    std::string s;
    for (post_op_type_t t : chain) {
        if (!s.empty()) s += ":";
        switch (t) {
            case post_op_type_t::binary_add: s += "binary_add"; break;
            case post_op_type_t::binary_mul: s += "binary_mul"; break;
            case post_op_type_t::sigmoid: s += "sigmoid"; break;
            case post_op_type_t::relu: s += "relu"; break;
            case post_op_type_t::clip: s += "clip"; break;
            case post_op_type_t::tanh: s += "tanh"; break;
            case post_op_type_t::gelu_erf: s += "gelu_erf"; break;
            default: s += "?"; break;
        }
    }
    return s;
}

/// One call of a chain: f32 binary operands of [M, N], or of [1, N] for a
/// row operand, clip to [-0.5, 0.75], relu with alpha 0, every alpha
/// explicit.
struct chain_call_t {
    nt::call_t c;
    nt::buffer_t src, wei, bias, dst, ref, operand;
    float src_scale = 0.5f, wei_scale = 0.25f;

    chain_call_t(int M, int N, int K, data_type_t src_dt, data_type_t wei_dt,
            data_type_t dst_dt, const chain_t &chain, float alpha,
            uint32_t seed, bool row_operand = false) {
        const bool int8 = wei_dt == data_type_t::s8;
        src = nt::random_buffer(src_dt, static_cast<size_t>(M) * K, seed,
                int8 ? -8.0f : -1.0f, int8 ? 8.0f : 1.0f);
        wei = nt::random_buffer(wei_dt, static_cast<size_t>(K) * N, seed + 1,
                int8 ? -8.0f : -1.0f, int8 ? 8.0f : 1.0f);
        bias = nt::random_buffer(data_type_t::f32, N, seed + 2, -1.0f, 1.0f);
        operand = nt::random_buffer(data_type_t::f32,
                static_cast<size_t>(row_operand ? 1 : M) * N, seed + 3, -1.0f,
                1.0f);
        dst = nt::make_buffer(static_cast<size_t>(M) * N * 4);
        ref = nt::make_buffer(static_cast<size_t>(M) * N * 4);
        c.M = M;
        c.N = N;
        c.K = K;
        c.alpha = alpha;
        c.src = src.get();
        c.wei = wei.get();
        c.bias = bias.get();
        c.dst = dst.get();
        c.is_weights_const = false;
        c.params.dtypes.src = src_dt;
        c.params.dtypes.wei = wei_dt;
        c.params.dtypes.dst = dst_dt;
        c.params.dtypes.bias = data_type_t::f32;
        if (int8) {
            auto &q = c.params.quant_params;
            q.src_scale.buff = &src_scale;
            q.src_scale.dt = data_type_t::f32;
            q.src_scale.dims = {1, 1};
            q.wei_scale.buff = &wei_scale;
            q.wei_scale.dt = data_type_t::f32;
            q.wei_scale.dims = {1, 1};
        }
        for (post_op_type_t t : chain) {
            mm::matmul_post_op po;
            po.po_type = t;
            po.alpha = 0.0f;
            if (t == post_op_type_t::binary_add
                    || t == post_op_type_t::binary_mul) {
                po.buff = operand.get();
                po.dtype = data_type_t::f32;
                po.dims = {row_operand ? 1 : M, N};
                po.leading_dim = N;
            } else if (t == post_op_type_t::clip) {
                po.alpha = -0.5f;
                po.beta = 0.75f;
            }
            c.params.postop_.push_back(po);
        }
    }

    /// Runs the reference into ref and compares dst with it.
    ::testing::AssertionResult matches_reference() {
        if (nt::run_reference(c, wei.get(), ref.get()) != status_t::success)
            return ::testing::AssertionFailure() << "reference failed";
        const bool low = c.params.dtypes.src != data_type_t::f32;
        return nt::within_band(dst.get(), c.N, ref.get(), c.N,
                c.params.dtypes.dst, c.M, c.N, low ? 1e-2f : 1e-4f,
                low ? 2e-2f : 1e-4f);
    }
};

struct route_t {
    const char *name;
    matmul_algo_t algo;
    data_type_t src, wei, dst;
    int M, N, K, threads;
};

/// The CPU has the instruction sets the route's kernels need.
bool runs_here(const route_t &r) {
    const nat::UarchParams &u = nat::detect_uarch();
    if (r.src == data_type_t::bf16 && !u.avx512bf16) return false;
    if (r.wei == data_type_t::s8 && !u.avx512vnni) return false;
    return r.algo != matmul_algo_t::native_brgemm || nat::native_base_isa();
}

// Through matmul_direct: native_brgemm (the native BRGEMM's tiles and
// epilogue: BF16 at M = 7, 8 and 40, INT8 at M = 1 on one thread and at
// M = 40, FP32 at M = 7 and 40), each with alpha 1 and 1.5.
TEST(NativePostopChain, EveryNativeRouteKeepsTheChainOrder) {
    const route_t routes[] = {
            {"native_brgemm bf16 M 8", matmul_algo_t::native_brgemm,
                    data_type_t::bf16, data_type_t::bf16, data_type_t::f32, 8,
                    96, 64, 4},
            {"native_brgemm bf16 M 7", matmul_algo_t::native_brgemm,
                    data_type_t::bf16, data_type_t::bf16, data_type_t::f32, 7,
                    96, 48, 1},
            {"native_brgemm s8 M 1", matmul_algo_t::native_brgemm,
                    data_type_t::s8, data_type_t::s8, data_type_t::f32, 1, 96,
                    64, 1},
            {"native_brgemm bf16", matmul_algo_t::native_brgemm,
                    data_type_t::bf16, data_type_t::bf16, data_type_t::f32, 40,
                    96, 64, 4},
            {"native_brgemm s8", matmul_algo_t::native_brgemm, data_type_t::s8,
                    data_type_t::s8, data_type_t::f32, 40, 96, 64, 4},
            {"native_brgemm f32 M 7", matmul_algo_t::native_brgemm,
                    data_type_t::f32, data_type_t::f32, data_type_t::f32, 7, 96,
                    48, 4},
            {"native_brgemm f32", matmul_algo_t::native_brgemm,
                    data_type_t::f32, data_type_t::f32, data_type_t::f32, 40,
                    96, 64, 4},
    };
    uint32_t seed = 300;
    for (const route_t &r : routes) {
        if (!runs_here(r)) continue;
        for (const chain_t &chain : chains())
            for (const float alpha : {1.0f, 1.5f}) {
                SCOPED_TRACE(std::string(r.name) + " " + chain_name(chain)
                        + " alpha " + std::to_string(alpha));
                chain_call_t cc(r.M, r.N, r.K, r.src, r.wei, r.dst, chain,
                        alpha, seed += 8);
                cc.c.params.lowoha_algo = r.algo;
                cc.c.params.num_threads = r.threads;
                ASSERT_EQ(nt::run_matmul(cc.c), status_t::success);
                EXPECT_TRUE(cc.matches_reference());
            }
    }
}

// A binary operand of [M, N] or [1, N], alone or after an activation, on the
// native BRGEMM's tiles, with M not a multiple of their tiles.
TEST(NativePostopChain, BinaryOperandsFollowTheTileOnTheNativeEntry) {
    const route_t routes[] = {
            {"native_brgemm bf16 M 7", matmul_algo_t::native_brgemm,
                    data_type_t::bf16, data_type_t::bf16, data_type_t::f32, 7,
                    96, 48, 4},
            {"native_brgemm s8 M 1", matmul_algo_t::native_brgemm,
                    data_type_t::s8, data_type_t::s8, data_type_t::f32, 1, 96,
                    48, 1},
            {"native_brgemm bf16", matmul_algo_t::native_brgemm,
                    data_type_t::bf16, data_type_t::bf16, data_type_t::f32, 29,
                    96, 48, 4},
            {"native_brgemm s8", matmul_algo_t::native_brgemm, data_type_t::s8,
                    data_type_t::s8, data_type_t::f32, 29, 96, 48, 4},
            {"native_brgemm f32 M 7", matmul_algo_t::native_brgemm,
                    data_type_t::f32, data_type_t::f32, data_type_t::f32, 7, 96,
                    48, 4},
            {"native_brgemm f32", matmul_algo_t::native_brgemm,
                    data_type_t::f32, data_type_t::f32, data_type_t::f32, 29,
                    96, 48, 4},
    };
    const std::vector<chain_t> binary = {
            {post_op_type_t::binary_add},
            {post_op_type_t::binary_mul},
            {post_op_type_t::relu, post_op_type_t::binary_add},
    };
    uint32_t seed = 1100;
    for (const route_t &r : routes) {
        if (!runs_here(r)) continue;
        for (const bool row_operand : {false, true})
            for (const chain_t &chain : binary) {
                SCOPED_TRACE(std::string(r.name) + " " + chain_name(chain)
                        + (row_operand ? " [1, N]" : " [M, N]"));
                chain_call_t cc(r.M, r.N, r.K, r.src, r.wei, r.dst, chain, 1.5f,
                        seed += 8, row_operand);
                cc.c.params.lowoha_algo = r.algo;
                cc.c.params.num_threads = r.threads;
                ASSERT_EQ(nt::run_matmul(cc.c), status_t::success);
                EXPECT_TRUE(cc.matches_reference());
            }
    }
}

} // namespace
