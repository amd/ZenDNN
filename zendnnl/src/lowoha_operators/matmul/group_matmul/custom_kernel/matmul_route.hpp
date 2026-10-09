/*******************************************************************************
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

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_MATMUL_ROUTE_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_MATMUL_ROUTE_HPP

#include <algorithm>
#include <cmath>

#include <omp.h>

#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/n_tile/group_matmul_n_tile.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

inline constexpr int kSingleExpertTilesPerThread = 2;

inline bool matmul_custom_kernel_bias_supported(
        const void *bias, data_type_t bias_dtype) noexcept {
    const bool has_bias = bias != nullptr;
    const bool has_bias_dtype = bias_dtype != data_type_t::none;
    if (has_bias != has_bias_dtype) return false;
    if (!has_bias) return true;
    return bias_dtype == data_type_t::bf16 || bias_dtype == data_type_t::f32
            || bias_dtype == data_type_t::f16;
}

inline bool matmul_custom_kernel_quant_supported(int M, int N, int K,
        custom_kernel::KernelVariant variant,
        const matmul_params &params) noexcept {
    const auto &qp = params.quant_params;
    const bool src_per_token = quant_param_is_per_token(qp.src_scale, M);

    if (custom_kernel::is_int8_variant(variant)) {
        if (!get_grp_matmul_custom_kernel_int8()) return false;
        if ((K % custom_kernel::kVNNIInt8Quad) != 0) return false;
        if (K > custom_kernel::kMaxSafeInt8AccumK) return false;

        // Static asymmetric U8xS8 uses scalar source metadata and scalar or
        // per-channel symmetric weight scales.
        if (is_static_u8s8_quant_config(params))
            return static_u8s8_quant_supported(params, N);

        const bool signed_compute
                = variant == custom_kernel::KernelVariant::kS8_S8_BF16_SYM
                || variant == custom_kernel::KernelVariant::kS8_S8_F32_SYM;
        if (signed_compute && qp.src_zp.buff != nullptr) return false;
        if (qp.dst_scale.buff != nullptr || qp.dst_zp.buff != nullptr)
            return false;
        if (!src_per_token || qp.wei_scale.buff == nullptr
                || qp.wei_zp.buff != nullptr)
            return false;
        if (!params.dynamic_quant && qp.src_scale.buff == nullptr) return false;

        return quant_param_is_per_channel(qp.wei_scale, N);
    }

    if (custom_kernel::is_s4_variant(variant)) {
        if (!is_w4a8_config(params) || (K % custom_kernel::kS4Octet) != 0) {
            return false;
        }
        if (qp.dst_scale.buff != nullptr || qp.dst_zp.buff != nullptr)
            return false;
        if (!src_per_token || qp.wei_scale.buff == nullptr
                || qp.src_zp.buff != nullptr || qp.wei_zp.buff != nullptr) {
            return false;
        }
        if (!params.dynamic_quant && qp.src_scale.buff == nullptr) return false;

        const bool src_scale_dt_supported = qp.src_scale.dt == data_type_t::f32
                || qp.src_scale.dt == data_type_t::bf16;
        const bool wei_scale_dt_supported = qp.wei_scale.dt == data_type_t::f32
                || qp.wei_scale.dt == data_type_t::bf16;
        const auto &wei_dims = qp.wei_scale.dims;
        if (!src_scale_dt_supported || !wei_scale_dt_supported
                || wei_dims.size() != 2 || wei_dims[0] <= 1
                || wei_dims[1] != static_cast<int64_t>(N)
                || static_cast<int64_t>(K) % wei_dims[0] != 0) {
            return false;
        }
        const int64_t group_size = static_cast<int64_t>(K) / wei_dims[0];
        return group_size > 0 && (group_size % custom_kernel::kS4Octet) == 0;
    }

    return true;
}

inline bool has_enough_single_expert_tiles(
        int N, int K, int num_threads) noexcept {
    const int pack_nr = custom_kernel::plan_pack_nr(K, N);
    if (pack_nr != custom_kernel::kNRMin && pack_nr != custom_kernel::kNRMax)
        return false;
    const int usable_n_tiles = N / pack_nr;
    const int requested_threads = std::max(1, num_threads);
    return usable_n_tiles >= requested_threads * kSingleExpertTilesPerThread;
}

/// True when a raw scalar weight can become eligible for CK on another phase
/// or thread configuration in this process. This deliberately ignores
/// execution-only fields (M, current thread count, alpha/beta, bias, post-ops):
/// when the current call falls back to scalar AOCL under WC=2, those fields may
/// change on a later call that reuses the same weight. The caller uses this
/// verdict only to cap the CURRENT scalar fallback at WC=1 and preserve raw
/// bytes for that future CK pack.
inline bool custom_kernel_cross_phase_weight_candidate(
        const matmul_params &params) noexcept {
    auto &config = zendnnl::common::matmul_config_t::instance();
    if (!config.get_custom_kernel_route() || !get_grp_matmul_custom_kernel())
        return false;
    if (config.get_weight_cache() != 2
            || effective_weight_cache_type(params.weight_cache_type) != 2) {
        return false;
    }
    if (params.mem_format_b != 'n') return false;

    // Weight-centric by design: src/dst/compute metadata can change between
    // calls while the same weight pointer is reused. Current transA, A format,
    // batching, backend, constness, N/K view, and variant rejection therefore
    // must not license an irreversible in-place reorder that a future
    // supported scalar tuple would CK-pack as raw. Capping a non-AOCL or
    // non-constant current call is harmless; those paths do not consume the
    // AOCL in-place permission in the first place.
    const bool bf16_route = params.dtypes.wei == data_type_t::bf16
            && custom_kernel::dispatch_supported();
    const bool int8_route = params.dtypes.wei == data_type_t::s8
            && get_grp_matmul_custom_kernel_int8()
            && custom_kernel::avx512vnni_available();
    return bf16_route || int8_route;
}

/// Return true when a single MatMul can be reproduced by one-op grouped AUTO.
///
/// The grouped dispatcher owns prompt/decode selection and cache-mode-2
/// cross-warming once this boundary accepts the call.
inline bool custom_kernel_routable(char layout, int M, int N, int K,
        int num_threads, int batch_count, bool transA, bool transB, float alpha,
        float beta, const void *bias, int ldb, bool is_weights_const,
        const matmul_params &params) {
    auto &config = zendnnl::common::matmul_config_t::instance();
    if (!config.get_custom_kernel_route()) return false;

    // Gate on the effective fallback backend, not the raw environment value:
    // unset/-1 resolves to AOCL-DLP blocked on AOCL builds, while explicit
    // oneDNN/libxsmm/reference selections remain excluded. A per-operation
    // algorithm, when present, is stronger than the process-wide setting.
    const matmul_algo_t effective_kernel
            = params.lowoha_algo != matmul_algo_t::none ? params.lowoha_algo
                                                        : resolve_kernel();
    if (effective_kernel != matmul_algo_t::aocl_dlp_blocked) return false;

    const bool aocl_prepacked = is_aocl_prepacked_weight(params);
    const int32_t global_cache = config.get_weight_cache();
    if (global_cache < 0 || global_cache > 2) return false;
    if (!get_grp_matmul_custom_kernel()) return false;
    if (get_grp_n_tile_strategy() == 0) return false;

    // Prompt calls intentionally enter one-op grouped AUTO: BF16/INT8 execute
    // full-weight AOCL (ALGO 1) so mixed-WC2 preparation can warm a future
    // decode CK layout, while W4A8 executes its supported ALGO 2 prompt path.
    // Decode still has to resolve to ALGO 3 below.
    if (!has_enough_single_expert_tiles(N, K, num_threads)) return false;
    if (M <= 0 || batch_count != 1 || transA || !is_weights_const) return false;
    if (layout != 'r' && layout != 'R') return false;
    if (params.mem_format_a != 'n') return false;
    if (omp_in_parallel() || omp_get_dynamic()) return false;

    // Raw scalar inputs must not override a per-operation algorithm. AOCL
    // prepacked inputs may carry the matching aocl_dlp_blocked provenance, or
    // leave it unset for scalar ZenTorch compatibility.
    if (!aocl_prepacked && params.lowoha_algo != matmul_algo_t::none)
        return false;

    const int32_t effective_cache
            = effective_weight_cache_type(params.weight_cache_type);
    if (effective_cache != global_cache) return false;
    if (aocl_prepacked
            && (effective_cache == 0 || config.get_lru_cache_capacity() == 0)) {
        return false;
    }

    if (alpha != 1.0f || beta != 0.0f) return false;
    if (!matmul_custom_kernel_bias_supported(bias, params.dtypes.bias))
        return false;

    const custom_kernel::KernelVariant variant = custom_kernel::resolve_variant(
            params.dtypes.src, params.dtypes.wei, params.dtypes.dst,
            params.dynamic_quant, params.dtypes.compute);
    const bool bf16_variant
            = variant == custom_kernel::KernelVariant::kBF16_BF16_BF16;
    const bool int_variant = custom_kernel::is_int8_variant(variant);
    const bool s4_variant = custom_kernel::is_s4_variant(variant);
    if (!bf16_variant && !int_variant && !s4_variant) return false;
    if (!params.postop_.empty()
            && !(int_variant && !params.dynamic_quant
                    && params.dtypes.src == data_type_t::u8
                    && int8_ck_postops_supported(params, M, N)))
        return false;
    if (bf16_variant && !custom_kernel::dispatch_supported()) return false;
    if ((int_variant || s4_variant) && !custom_kernel::avx512vnni_available()) {
        return false;
    }
    if (!matmul_custom_kernel_quant_supported(M, N, K, variant, params))
        return false;
    if (params.packing.pack_format_b == 1) return false;
    if (!aocl_prepacked && params.mem_format_b != 'n') return false;

    const int pack_nr = custom_kernel::plan_pack_nr(K, N);
    if (pack_nr != custom_kernel::kNRMin && pack_nr != custom_kernel::kNRMax)
        return false;
    const int min_ldb = transB ? K : N;
    if (ldb < min_ldb) return false;

    // Reuse the grouped selector itself rather than approximating its policy.
    // In particular, one thread, a global ALGO 1 pin, or a decode-phase ALGO 1
    // pin all make the one-op wrapper succeed without ever reaching CK.
    const std::vector<char> grouped_layout {layout};
    const std::vector<int> grouped_m {M};
    const std::vector<int> grouped_n {N};
    const std::vector<int> grouped_k {K};
    const std::vector<matmul_params> grouped_params {params};
    const int selected = select_grp_matmul_algo(grouped_layout, grouped_m,
            grouped_n, grouped_k, grouped_params, num_threads);
    return selected == 3 || (s4_variant && selected == 2)
            || (M > kDecodeMaxM && selected == 1);
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_MATMUL_ROUTE_HPP
