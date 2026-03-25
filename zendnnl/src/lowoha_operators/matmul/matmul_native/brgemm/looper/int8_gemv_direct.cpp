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

#include "lowoha_operators/matmul/matmul_native/brgemm/looper/int8_gemv_direct.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/int8_gemv_bkc.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"
#include "operators/matmul/matmul_config.hpp"
#include "common/zendnnl_global.hpp"

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using namespace zendnnl::error_handling;

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
bool int8_gemv_direct(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;
    const int ldb = desc.ldb;

    // ── Eligibility gates ──
    if (M != 1) return false;
    if (!uarch.avx512vnni) return false;
    if (desc.transA) return false;
    if (desc.alpha != 1.0f) return false;
    if (desc.beta != 0.0f) return false;

    // INT8 source: u8 or s8; weight: s8; dst: bf16 or fp32
    const bool src_is_u8  = (desc.src_dt == data_type_t::u8);
    const bool src_is_s8  = (desc.src_dt == data_type_t::s8);
    if (!src_is_u8 && !src_is_s8) return false;
    if (desc.wei_dt != data_type_t::s8) return false;
    const bool dst_is_bf16 = (desc.dst_dt == data_type_t::bf16);
    const bool dst_is_fp32 = (desc.dst_dt == data_type_t::f32);
    if (!dst_is_bf16 && !dst_is_fp32) return false;

    const int K_padded = (K + 3) & ~3;
    const int N_padded = ((N + NR_PACK - 1) / NR_PACK) * NR_PACK;
    const size_t b_packed_bytes =
        static_cast<size_t>(K_padded) * N_padded * sizeof(int8_t);

    if (N <= 0) return false;
    if (desc.num_threads > 1) return false;
    if (desc.ldc != N) return false;

    // L2 gate: large shapes fall through to BRGEMM looper
    const size_t l2_cap = static_cast<size_t>(uarch.l2_bytes);
    if (b_packed_bytes > l2_cap) return false;

    // Check for unfuseable post-ops
    fused_postop_t kc_fused_op = fused_postop_t::none;
    bool has_unfuseable_postops = false;
    for (int i = 0; i < static_cast<int>(params.postop_.size()); ++i) {
        auto pt = params.postop_[i].po_type;
        if (pt == post_op_type_t::none) continue;
        if (kc_fused_op == fused_postop_t::none) {
            if (pt == post_op_type_t::relu && params.postop_[i].alpha == 0.0f)
                { kc_fused_op = fused_postop_t::relu; continue; }
            if (pt == post_op_type_t::gelu_tanh)
                { kc_fused_op = fused_postop_t::gelu_tanh; continue; }
            if (pt == post_op_type_t::gelu_erf)
                { kc_fused_op = fused_postop_t::gelu_erf; continue; }
            if (pt == post_op_type_t::sigmoid)
                { kc_fused_op = fused_postop_t::sigmoid; continue; }
            if (pt == post_op_type_t::tanh)
                { kc_fused_op = fused_postop_t::tanh_op; continue; }
            if (pt == post_op_type_t::swish)
                { kc_fused_op = fused_postop_t::swish; continue; }
        }
        has_unfuseable_postops = true;
    }
    if (has_unfuseable_postops) return false;

    // ── Extract quantization parameters ──
    auto qp = extract_int8_quant(params);
    float src_scale = qp.src_scale;
    int32_t src_zp = qp.src_zp;
    float wei_scale_default = 1.0f;
    const float *wei_scale_ptr = qp.wei_scale ? qp.wei_scale : &wei_scale_default;
    int wei_scale_count = qp.wei_scale_count;
    if (wei_scale_count != 0 && wei_scale_count != 1 && wei_scale_count != N)
        return false;

    // ── Prepare A as u8 ──
    // If source is s8, convert to u8 by adding 128 and adjust zero point.
    const uint8_t *A_u8 = nullptr;
    static thread_local uint8_t *s_a_buf = nullptr;
    static thread_local size_t s_a_cap = 0;
    int32_t effective_zp = src_zp;

    if (src_is_u8) {
        A_u8 = static_cast<const uint8_t *>(src);
    } else {
        // s8 → u8: add 128, adjust zp
        if (s_a_cap < static_cast<size_t>(K)) {
            std::free(s_a_buf);
            s_a_buf = static_cast<uint8_t *>(std::aligned_alloc(
                64, ((K + 63) & ~size_t(63))));
            s_a_cap = s_a_buf ? K : 0;
        }
        if (!s_a_buf) return false;
        const int8_t *a_s8 = static_cast<const int8_t *>(src);
        for (int k = 0; k < K; ++k)
            s_a_buf[k] = static_cast<uint8_t>(static_cast<int>(a_s8[k]) + 128);
        A_u8 = s_a_buf;
        effective_zp = src_zp + 128;
    }

    // ── Bias → fp32 ──
    const bool has_bias = (desc.bias != nullptr);
    const float *bias_f = nullptr;
    static thread_local float *s_bias_f = nullptr;
    static thread_local size_t s_bias_cap = 0;
    static thread_local const void *s_bias_ptr = nullptr;
    static thread_local int s_bias_N = 0;

    if (has_bias) {
        if (desc.bias_dt == data_type_t::bf16) {
            if (s_bias_cap < static_cast<size_t>(N)) {
                std::free(s_bias_f);
                s_bias_f = static_cast<float *>(std::aligned_alloc(
                    64, ((N * sizeof(float) + 63) & ~size_t(63))));
                s_bias_cap = s_bias_f ? N : 0;
                if (!s_bias_f) return false;
                s_bias_ptr = nullptr;
            }
            if (s_bias_ptr != bias || s_bias_N != N) {
                const uint16_t *bb = static_cast<const uint16_t *>(bias);
                for (int n = 0; n < N; ++n) {
                    uint32_t bits = static_cast<uint32_t>(bb[n]) << 16;
                    std::memcpy(&s_bias_f[n], &bits, sizeof(float));
                }
                s_bias_ptr = bias;
                s_bias_N = N;
            }
            bias_f = s_bias_f;
        } else if (desc.bias_dt == data_type_t::f32) {
            bias_f = static_cast<const float *>(bias);
        } else {
            return false;
        }
    }

    // ── Pack + cache ──
    const int8_t *B_raw = static_cast<const int8_t *>(weight);
    const int8_t *B_kc = nullptr;
    const float *combined_scale = nullptr;
    const float *effective_bias = nullptr;
    [[maybe_unused]] const char *pack_source = "none";

    using zendnnl::ops::matmul_config_t;
    static int32_t s_weight_cache =
        matmul_config_t::instance().get_weight_cache();

    if (desc.is_weights_const && s_weight_cache != 0) {
        static thread_local const void *s_gc_wt = nullptr;
        static thread_local int s_gc_K = 0, s_gc_N = 0;
        static thread_local float s_gc_scale = 0;
        static thread_local int32_t s_gc_zp = 0;
        static thread_local const INT8KContiguousWeight *s_gc_entry = nullptr;

        if (s_gc_wt == weight && s_gc_K == K && s_gc_N == N
            && s_gc_scale == src_scale && s_gc_zp == effective_zp
            && s_gc_entry) {
            B_kc = s_gc_entry->data;
            combined_scale = s_gc_entry->combined_scale;
            effective_bias = s_gc_entry->effective_bias;
            pack_source = "tl_cache";
        } else {
            PrepackedWeightKey key{weight, K, N, ldb, desc.transB};
            const INT8KContiguousWeight *cached =
                INT8KContiguousWeightCache::instance().get_or_pack(
                    key, B_raw, src_scale, effective_zp,
                    bias_f, wei_scale_ptr, wei_scale_count);
            if (cached) {
                B_kc = cached->data;
                combined_scale = cached->combined_scale;
                effective_bias = cached->effective_bias;
                s_gc_wt = weight;
                s_gc_K = K;
                s_gc_N = N;
                s_gc_scale = src_scale;
                s_gc_zp = effective_zp;
                s_gc_entry = cached;
                pack_source = "global_cache";
            }
        }
    }
    if (!B_kc) {
        static thread_local int8_t *s_bkc = nullptr;
        static thread_local size_t s_bkc_cap = 0;
        static thread_local int32_t *s_cs = nullptr;
        static thread_local float *s_cscale = nullptr;
        static thread_local float *s_ebias = nullptr;
        static thread_local size_t s_dq_cap = 0;

        const size_t need_pack = static_cast<size_t>(K_padded) * N_padded;
        if (s_bkc_cap < need_pack) {
            std::free(s_bkc);
            s_bkc = static_cast<int8_t *>(std::aligned_alloc(
                64, ((need_pack + 63) & ~size_t(63))));
            s_bkc_cap = s_bkc ? need_pack : 0;
        }
        if (!s_bkc) return false;

        if (s_dq_cap < static_cast<size_t>(N_padded)) {
            std::free(s_cs); std::free(s_cscale); std::free(s_ebias);
            size_t alloc_n = ((N_padded * sizeof(float) + 63) & ~size_t(63));
            s_cs     = static_cast<int32_t *>(std::aligned_alloc(64, alloc_n));
            s_cscale = static_cast<float *>(std::aligned_alloc(64, alloc_n));
            s_ebias  = static_cast<float *>(std::aligned_alloc(64, alloc_n));
            s_dq_cap = (s_cs && s_cscale && s_ebias) ? N_padded : 0;
        }
        if (!s_cs || !s_cscale || !s_ebias) return false;

        pack_b_int8_bkc(B_raw, ldb, K, N, desc.transB, s_bkc, s_cs);
        precompute_int8_dequant(
            s_cs, bias_f, src_scale, effective_zp,
            wei_scale_ptr, wei_scale_count,
            N, N_padded, s_cscale, s_ebias);

        B_kc = s_bkc;
        combined_scale = s_cscale;
        effective_bias = s_ebias;
        pack_source = "thread_local";
    }

    // ── Compute ──
    int8_gemv_bkc(
        A_u8, B_kc,
        combined_scale, effective_bias,
        dst_is_bf16 ? static_cast<uint16_t *>(dst) : nullptr,
        dst_is_fp32 ? static_cast<float *>(dst) : nullptr,
        kc_fused_op, dst_is_bf16, K, N);

    static bool s_log = apilog_info_enabled();
    if (s_log) {
        apilog_info("Native INT8 BKC-GEMV: M=1 K=", K, " N=", N,
                    " K_padded=", K_padded, " N_padded=", N_padded,
                    " packed_B=", b_packed_bytes / 1024, "KB",
                    " src=", src_is_u8 ? "u8" : "s8",
                    " dst=", dst_is_bf16 ? "bf16" : "fp32",
                    " src_zp=", effective_zp,
                    " wei_scale=", wei_scale_count > 1 ? "per_channel" : "per_tensor",
                    " fused_op=", static_cast<int>(kc_fused_op),
                    " pack=", pack_source);
    }
    return true;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
