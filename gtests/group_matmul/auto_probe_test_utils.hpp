/********************************************************************************
# * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// @file auto_probe_test_utils.hpp
/// @brief Builders shared by `test_algos.cpp` and
///        `single_expert/test_single_expert_policy.cpp`:
///   * `AutoProbeShape` / `build_auto_probe` and the
///     `apply_{w4a8_auto,shape_rules}_probe` rewrites — minimal
///     `(layout, M, N, K, params)` tuples for `group_matmul_algo_select`.
///   * `Int8AlgoCustomParam` / `build_int8_dq_inputs` — DQ-INT8 symmetric
///     per-expert input tensors for `group_matmul_kernel_test`.

#ifndef ZENDNNL_GTESTS_GROUP_MATMUL_AUTO_PROBE_TEST_UTILS_HPP
#define ZENDNNL_GTESTS_GROUP_MATMUL_AUTO_PROBE_TEST_UTILS_HPP

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

#include "gtest_utils.hpp"

namespace auto_probe_test_utils {

struct Int8AlgoCustomParam {
    int act_int; // 0=none, 1=silu, 2=gelu, 3=swiglu_oai
    int M, num_ops,
            dim; // dim = single-half output width; N_op1 = 2*dim for gated
};

// Build per-expert (src, wei_s8, scale) tensors for the int8 DQ
// symmetric path.  The src is bf16 with attached per-token
// src_scale; the weight is s8 with per-channel wei_scale.  Returns
// the assembled tensor vectors and the dst buffer for the run.
// Caller invokes `group_matmul_kernel_test` twice — once with
// CK_INT8 OFF (reference) and once ON (test) — without rebuilding
// the inputs between runs.
inline void build_int8_dq_inputs(tensor_factory_t &tf,
        const Int8AlgoCustomParam &p, int N_op1, int K,
        std::vector<tensor_t> &src_tensors, std::vector<tensor_t> &wei_tensors,
        std::vector<tensor_t> &bias_tensors, std::vector<tensor_t> &dst_tensors,
        // Scale dtype for BOTH the per-channel wei_scale and the per-token
        // src_scale.  Defaults to f32 (the original coverage); pass bf16 to
        // exercise the reorder's native bf16 scale output, which the CK
        // microkernel must materialise to f32 before consuming (reading a
        // bf16 scale buffer as f32 would silently corrupt every output —
        // see the f32-scale-view materialisation in n_tile_execute).
        data_type_t scale_dt = data_type_t::f32) {
    src_tensors.assign(p.num_ops, tensor_t {});
    wei_tensors.assign(p.num_ops, tensor_t {});
    bias_tensors.assign(p.num_ops, tensor_t {});
    dst_tensors.assign(p.num_ops, tensor_t {});

    // Per-channel wei_scale, per-token src_scale — the granularity
    // Phase 1's CK admits.  Scale dtype is f32 or bf16 (see `scale_dt`).
    const std::vector<int64_t> wei_scale_dims {1, N_op1};
    const std::vector<uint64_t> src_scale_shape {
            static_cast<uint64_t>(p.M), static_cast<uint64_t>(1)};

    for (int e = 0; e < p.num_ops; ++e) {
        auto wei_ref = tf.uniform_dist_tensor(
                {static_cast<uint64_t>(K), static_cast<uint64_t>(N_op1)},
                data_type_t::bf16, /*range=*/2.0);
        tensor_t wei_s8, wei_scale, wei_zp;
        ASSERT_EQ(quant_params_compute(tf, wei_ref, data_type_t::bf16,
                          data_type_t::s8, wei_scale_dims, scale_dt, wei_scale,
                          wei_zp, &wei_s8),
                status_t::success)
                << "DQ-INT8 weight quant setup failed";
        wei_tensors[e] = wei_s8;

        auto src_scale = tf.zero_tensor(src_scale_shape, scale_dt);
        src_tensors[e] = tf.uniform_dist_tensor(
                {static_cast<uint64_t>(p.M), static_cast<uint64_t>(K)},
                data_type_t::bf16, /*range=*/2.0, /*trans=*/false, src_scale,
                tensor_t());

        // Bias is intentionally OMITTED — Phase 1 refuses gated-act +
        // bias for both bf16 and int8 CK paths, so a bias-free baseline
        // keeps every act cell inside the kernel envelope.
        bias_tensors[e] = tensor_t {};

        dst_tensors[e] = tf.uniform_dist_tensor(
                {static_cast<uint64_t>(p.M), static_cast<uint64_t>(N_op1)},
                data_type_t::bf16, /*range=*/2.0);
    }
}

// Build a minimal valid `(layout, M, N, K, params)` tuple for an
// auto-select probe.  Row-major BF16, no quant, no packed-B.
struct AutoProbeShape {
    std::vector<char> layout;
    std::vector<int> M;
    std::vector<int> N;
    std::vector<int> K;
    std::vector<zendnnl::lowoha::matmul::matmul_params> params;
    int num_threads;
};

inline AutoProbeShape build_auto_probe(
        int M_val, int K_val, int N_val, int num_ops, int num_threads) {
    using namespace zendnnl::lowoha::matmul;
    AutoProbeShape s;
    s.layout.assign(num_ops, 'r');
    s.M.assign(num_ops, M_val);
    s.N.assign(num_ops, N_val);
    s.K.assign(num_ops, K_val);
    s.params.resize(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        s.params[i].dtypes.src = data_type_t::bf16;
        s.params[i].dtypes.wei = data_type_t::bf16;
        s.params[i].dtypes.dst = data_type_t::bf16;
        s.params[i].dtypes.bias = data_type_t::bf16;
        s.params[i].mem_format_a = 'n';
        s.params[i].mem_format_b = 'n';
        s.params[i].dynamic_quant = false;
        s.params[i].packing.pack_format_b = 0;
    }
    s.num_threads = num_threads;
    return s;
}

// Rewrite a BF16 auto-select probe as n_tile-safe W4A8 (dynamic bf16 src,
// packed s4 weights, per-group wei scale).  This lets the generic AUTO rules
// be tested with W4A8 metadata without standing up the full GEMM stack.
inline void apply_w4a8_auto_probe(AutoProbeShape &s) {
    using namespace zendnnl::lowoha::matmul;
    static float wei_scale_sentinel = 1.0f;
    for (size_t i = 0; i < s.params.size(); ++i) {
        auto &p = s.params[i];
        p.dtypes.src = data_type_t::bf16;
        p.dtypes.wei = data_type_t::s4;
        p.dtypes.dst = data_type_t::bf16;
        p.dtypes.compute = data_type_t::s8;
        p.dynamic_quant = true;
        p.mem_format_b = 'n';
        p.quant_params.src_zp = {};
        p.quant_params.wei_zp = {};
        const int m = (i < s.M.size() && s.M[i] > 0) ? s.M[i] : 1;
        const int n = (i < s.N.size()) ? s.N[i] : 2;
        p.quant_params.src_scale.dims = {static_cast<int64_t>(m), 1};
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.buff = nullptr;
        p.quant_params.wei_scale.dims
                = {2, static_cast<int64_t>(n > 1 ? n : 2)};
        p.quant_params.wei_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.buff = &wei_scale_sentinel;
    }
}

// Rewrite a BF16 auto-select probe as f32.  Only BF16/FP16 layers of several
// experts run ALGO 3 in both phases, so an f32 probe of the same shape
// exercises the shape rules.
inline void apply_shape_rules_probe(AutoProbeShape &s) {
    using namespace zendnnl::lowoha::matmul;
    for (auto &p : s.params) {
        p.dtypes.src = data_type_t::f32;
        p.dtypes.wei = data_type_t::f32;
        p.dtypes.dst = data_type_t::f32;
        p.dtypes.bias = data_type_t::f32;
    }
}

} // namespace auto_probe_test_utils

#endif // ZENDNNL_GTESTS_GROUP_MATMUL_AUTO_PROBE_TEST_UTILS_HPP
