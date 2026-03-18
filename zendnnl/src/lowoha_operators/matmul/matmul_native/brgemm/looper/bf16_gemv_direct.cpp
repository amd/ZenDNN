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

#include "lowoha_operators/matmul/matmul_native/brgemm/looper/bf16_gemv_direct.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_gemv_kcontiguous.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "operators/matmul/matmul_config.hpp"
#include "common/zendnnl_global.hpp"

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstdlib>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using namespace zendnnl::error_handling;

static inline fused_postop_t detect_fused_postop(const matmul_params &params) {
    for (auto &po : params.postop_) {
        if (po.po_type == post_op_type_t::relu && po.alpha == 0.0f)
            return fused_postop_t::relu;
        if (po.po_type == post_op_type_t::gelu_tanh)
            return fused_postop_t::gelu_tanh;
        if (po.po_type == post_op_type_t::gelu_erf)
            return fused_postop_t::gelu_erf;
        if (po.po_type == post_op_type_t::sigmoid)
            return fused_postop_t::sigmoid;
        if (po.po_type == post_op_type_t::tanh)
            return fused_postop_t::tanh_op;
        if (po.po_type == post_op_type_t::swish)
            return fused_postop_t::swish;
    }
    return fused_postop_t::none;
}

__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,fma")))
bool bf16_gemv_direct(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;
    const int ldb = desc.ldb;
    const float alpha = desc.alpha, beta = desc.beta;

    if (M != 1) return false;
    if (!uarch.avx512bf16) return false;
    if (alpha != 1.0f) return false;
    if (desc.transA) return false;

    // ── Eligibility gates (cheap checks first, no allocations) ──
    const int K_padded = (K + 1) & ~1;
    const int N_padded = ((N + NR_PACK - 1) / NR_PACK) * NR_PACK;
    const size_t b_packed_bytes =
        static_cast<size_t>(K_padded) * N_padded * sizeof(uint16_t);
    const size_t l2_cap = static_cast<size_t>(uarch.l2_bytes);

    // N<NR_PACK pads to 64 columns (mostly zeros), but dpbf16ps's 2x compute
    // efficiency over FP32 FMA still wins. Benchmarked: KC beats DLP/GEMM
    // for all N=1..256 when B fits in L2.
    if (N <= 0 || N > 256) return false;
    if (desc.num_threads > 1) return false;
    if (b_packed_bytes > l2_cap) return false;
    if (desc.ldc != N) return false;

    fused_postop_t kc_fused_op = fused_postop_t::none;
    bool has_unfuseable_postops = false;
    for (int i = 0; i < static_cast<int>(params.postop_.size()); ++i) {
        auto pt = params.postop_[i].po_type;
        if (pt == post_op_type_t::none) continue;
        bool is_activation = (pt == post_op_type_t::relu && params.postop_[i].alpha == 0.0f)
                          || pt == post_op_type_t::gelu_tanh
                          || pt == post_op_type_t::gelu_erf
                          || pt == post_op_type_t::sigmoid
                          || pt == post_op_type_t::tanh
                          || pt == post_op_type_t::swish;
        if (is_activation && kc_fused_op == fused_postop_t::none) {
            kc_fused_op = detect_fused_postop(params);
        } else {
            has_unfuseable_postops = true;
        }
    }
    if (has_unfuseable_postops) return false;

    // ── Past all gates: shape is eligible. Convert bias, pack B, compute. ──
    const bool has_bias = (desc.bias != nullptr);
    const bool dst_is_bf16 = (desc.dst_dt == data_type_t::bf16);
    const bool bias_is_bf16 = (desc.bias_dt == data_type_t::bf16);
    const uint16_t *A = static_cast<const uint16_t *>(src);

    static thread_local float *s_bias_g = nullptr;
    static thread_local size_t s_bias_g_cap = 0;
    const float *bias_f = nullptr;
    if (has_bias) {
        if (bias_is_bf16) {
            if (s_bias_g_cap < static_cast<size_t>(N)) {
                std::free(s_bias_g);
                s_bias_g = static_cast<float *>(std::aligned_alloc(
                    64, ((N * sizeof(float) + 63) & ~size_t(63))));
                s_bias_g_cap = s_bias_g ? N : 0;
                if (!s_bias_g) return false;
            }
            const uint16_t *bb = static_cast<const uint16_t *>(bias);
            for (int n = 0; n < N; ++n) {
                uint32_t bits = static_cast<uint32_t>(bb[n]) << 16;
                std::memcpy(&s_bias_g[n], &bits, sizeof(float));
            }
            bias_f = s_bias_g;
        } else {
            bias_f = static_cast<const float *>(bias);
        }
    }

    const uint16_t *B_raw = static_cast<const uint16_t *>(weight);
    const uint16_t *B_kc = nullptr;
    [[maybe_unused]] const char *pack_source = "none";

    using zendnnl::ops::matmul_config_t;
    static int32_t s_weight_cache =
        matmul_config_t::instance().get_weight_cache();

    // Global cache keyed by (ptr, K, N, ldb, transB). Safe when
    // is_weights_const is true (caller guarantees data at ptr won't change).
    // For model swapping, call BF16KContiguousWeightCache::instance().clear().
    if (desc.is_weights_const && s_weight_cache != 0) {
        PrepackedWeightKey key{weight, K, N, ldb, desc.transB};
        const BF16KContiguousWeight *cached =
            BF16KContiguousWeightCache::instance().get_or_pack(key, B_raw);
        if (!cached) return false;
        B_kc = cached->data;
        pack_source = "global_cache";
    } else {
        static thread_local uint16_t *s_bkc = nullptr;
        static thread_local size_t s_bkc_cap = 0;
        size_t need = static_cast<size_t>(K_padded) * N_padded;
        if (s_bkc_cap < need) {
            std::free(s_bkc);
            s_bkc = static_cast<uint16_t *>(std::aligned_alloc(
                64, ((need * sizeof(uint16_t) + 63) & ~size_t(63))));
            s_bkc_cap = s_bkc ? need : 0;
        }
        if (!s_bkc) return false;
        pack_b_kcontiguous_ext(B_raw, ldb, K, N, desc.transB, s_bkc);
        B_kc = s_bkc;
        pack_source = "thread_local";
    }

    bf16_gemv_kcontiguous(
        A, B_kc,
        dst_is_bf16 ? static_cast<uint16_t *>(dst) : nullptr,
        !dst_is_bf16 ? static_cast<float *>(dst) : nullptr,
        (has_bias && bias_f) ? bias_f : nullptr,
        kc_fused_op,
        beta,
        dst_is_bf16,
        K, N);

    // Log check cached to avoid per-call global singleton lookup overhead.
    static bool s_log = apilog_info_enabled();
    if (s_log) {
        apilog_info("Native BF16 KC-GEMV: M=1 K=", K, " N=", N,
                    " K_padded=", K_padded, " N_padded=", N_padded,
                    " packed_B=", b_packed_bytes / 1024, "KB",
                    " beta=", beta,
                    " transB=", desc.transB ? "true" : "false",
                    " bias=", has_bias ? "yes" : "no",
                    " fused_op=", static_cast<int>(kc_fused_op),
                    " dst=", dst_is_bf16 ? "bf16" : "fp32",
                    " pack=", pack_source);
    }
    return true;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
