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

/// Which multi-expert layers AUTO keeps on ALGO 3 in both phases.

#include "multi_expert_policy.hpp"

#include "lowoha_operators/matmul/group_matmul/custom_kernel/ck_enable.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/n_tile_knobs.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/prompt_tile/prompt_tile_tunables.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_policy.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

unsigned get_grp_matmul_multi_expert_algo3_dtypes() {
    const int ovr = test_api::s_grp_matmul_multi_expert_algo3_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return static_cast<unsigned>(ovr) & kSingleExpertAll;
    // A value with no known token keeps the default, as unset does.
    static constexpr unsigned kDefault = ~0u;
    static const unsigned v = read_grp_matmul_single_expert_knob(
            "ZENDNNL_GRP_MATMUL_MULTI_EXPERT_ALGO3", kDefault);
    if (v != kDefault) return v;
    return get_grp_matmul_algo_legacy() ? kSingleExpertBf16 | kSingleExpertF16
                                        : kSingleExpertAll;
}

bool grp_matmul_auto_algo3_quant_expert(const matmul_params &p, int K, int N) {
    const auto &q = p.quant_params;
    // The kernel's BF16 / F32 epilogue applies no destination quantization.
    if (K <= 0 || N <= 0 || !get_grp_matmul_custom_kernel()
            || !p.postop_.empty() || p.dtypes.src == data_type_t::u8
            || p.dtypes.compute != data_type_t::s8 || q.src_zp.buff != nullptr
            || q.wei_zp.buff != nullptr || q.dst_scale.buff != nullptr
            || q.dst_zp.buff != nullptr) {
        return false;
    }
    const auto v = grp_matmul_ck_weight_variant(p.dtypes.src, p.dtypes.wei,
            p.dtypes.dst, p.dynamic_quant, p.dtypes.compute,
            /*static_u8s8=*/false, is_aocl_prepacked_weight(p), K, N);
    // The kernel reads one source scale per row.
    const auto &ss = q.src_scale.dims;
    if (ss.size() == 2 && ss[1] > 1) { return false; }
    const auto &ws = q.wei_scale.dims;
    prompt_tile::kernel_type_t fam;
    unsigned dt = 0;
    if (custom_kernel::is_int8_variant(v)) {
        if (ws.size() == 2 && ws[0] > 1) { return false; }
        fam = prompt_tile::kernel_type_t::kInt8;
        dt = kSingleExpertInt8;
    } else if (custom_kernel::is_w4a8_variant(v)) {
        if (ws.size() == 2 && ws[0] > 0
                && (K % ws[0] != 0
                        || (K / ws[0]) % custom_kernel::kW4Octet != 0)) {
            return false;
        }
        fam = prompt_tile::kernel_type_t::kW4a8;
        dt = kSingleExpertW4a8;
    } else {
        return false;
    }
    // The prompt runs on the prompt tile executor; a family it does not serve
    // keeps the shape rules in both phases.
    return (get_grp_matmul_multi_expert_algo3_dtypes() & dt) != 0
            && prompt_tile::enabled(fam) && get_grp_n_tile_strategy() != 1;
}

namespace {

int layer_experts_of(
        const std::vector<int> &M, const std::vector<matmul_params> &params) {
    int layer_experts = static_cast<int>(M.size());
    if (!params.empty() && params[0].active_matmul > 0) {
        layer_experts = std::max(
                layer_experts, static_cast<int>(params[0].total_matmul));
    }
    return layer_experts;
}

} // namespace

bool grp_matmul_float_moe_layer(
        const std::vector<int> &M, const std::vector<matmul_params> &params) {
    if (params.size() < M.size()) { return false; }
    data_type_t layer_dtype = data_type_t::none;
    for (size_t i = 0; i < M.size(); ++i) {
        if (M[i] <= 0) { continue; }
        const auto &p = params[i];
        const data_type_t dt = p.dtypes.src;
        if ((dt != data_type_t::bf16 && dt != data_type_t::f16)
                || p.dtypes.wei != dt
                || (layer_dtype != data_type_t::none && layer_dtype != dt)
                || is_aocl_prepacked_weight(p)) {
            return false;
        }
        layer_dtype = dt;
    }
    return grp_matmul_auto_algo3_layer(
            layer_dtype != data_type_t::none, layer_experts_of(M, params));
}

grp_matmul_algo3_family grp_matmul_auto_algo3_family(const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K,
        const std::vector<matmul_params> &params) {
    using family = grp_matmul_algo3_family;
    if (params.size() < M.size()) { return family::none; }
    bool any_active = false;
    data_type_t layer_src = data_type_t::none;
    data_type_t layer_wei = data_type_t::none;
    // The experts of a layer share their quant metadata, so an admitted
    // expert's verdict stands for the next one of the same shape.
    const matmul_params *admitted = nullptr;
    int admitted_K = 0, admitted_N = 0;
    // The source scale's row count is the expert's M; only its columns count.
    const auto src_cols = [](const matmul_params &p) {
        const auto &d = p.quant_params.src_scale.dims;
        return d.size() == 2 ? d[1] : -static_cast<int64_t>(d.size());
    };
    const auto same_quant
            = [&](const matmul_params &a, const matmul_params &b) {
        return a.dtypes.dst == b.dtypes.dst
                && a.dynamic_quant == b.dynamic_quant
                && a.dtypes.compute == b.dtypes.compute
                && a.mem_format_b == b.mem_format_b
                && a.lowoha_algo == b.lowoha_algo && a.postop_.empty()
                && src_cols(a) == src_cols(b)
                && a.quant_params.wei_scale.dims
                == b.quant_params.wei_scale.dims
                && (a.quant_params.src_zp.buff == nullptr)
                == (b.quant_params.src_zp.buff == nullptr)
                && (a.quant_params.wei_zp.buff == nullptr)
                == (b.quant_params.wei_zp.buff == nullptr)
                && (a.quant_params.dst_scale.buff == nullptr)
                == (b.quant_params.dst_scale.buff == nullptr)
                && (a.quant_params.dst_zp.buff == nullptr)
                == (b.quant_params.dst_zp.buff == nullptr);
    };
    for (size_t i = 0; i < M.size(); ++i) {
        if (M[i] <= 0) { continue; }
        const auto &p = params[i];
        const data_type_t src = p.dtypes.src;
        const data_type_t wei = p.dtypes.wei;
        const bool float_family
                = (src == data_type_t::bf16 || src == data_type_t::f16)
                && wei == src && !is_aocl_prepacked_weight(p)
                && (get_grp_matmul_multi_expert_algo3_dtypes()
                           & (src == data_type_t::bf16 ? kSingleExpertBf16
                                                       : kSingleExpertF16))
                        != 0;
        if (!float_family) {
            if (i >= N.size() || i >= K.size()) { return family::none; }
            // The executor takes one weight-scale layout (W4A8's group size)
            // from the first active expert for the whole layer.
            if (admitted != nullptr
                    && p.quant_params.wei_scale.dims
                            != admitted->quant_params.wei_scale.dims) {
                return family::none;
            }
            const bool reuse = admitted != nullptr && K[i] == admitted_K
                    && N[i] == admitted_N && same_quant(p, *admitted);
            if (!reuse && !grp_matmul_auto_algo3_quant_expert(p, K[i], N[i])) {
                return family::none;
            }
            admitted = &p;
            admitted_K = K[i];
            admitted_N = N[i];
        }
        if (any_active && (src != layer_src || wei != layer_wei)) {
            return family::none;
        }
        layer_src = src;
        layer_wei = wei;
        any_active = true;
    }
    if (!any_active
            || !grp_matmul_auto_algo3_layer(
                    true, layer_experts_of(M, params))) {
        return family::none;
    }
    return admitted != nullptr ? family::quant_family : family::float_family;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
