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
#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/lowoha_quant_layout.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/int8_gemv_bkc.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <memory>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using namespace zendnnl::error_handling;

namespace {

/// BF16/FP32 bias → FP32 scratch (thread_local), converted on every call:
/// the caller's buffer may hold new values at the same address.
bool int8_resolve_bias_f(const GemmDescriptor &desc, int N, const void *bias,
        bool has_bias, const float **bias_f_out) {
    *bias_f_out = nullptr;
    if (!has_bias) return true;
    if (desc.bias_dt == data_type_t::f32) {
        *bias_f_out = static_cast<const float *>(bias);
        return true;
    }
    if (desc.bias_dt != data_type_t::bf16) return false;

    static thread_local std::unique_ptr<float, native_aligned_free_t> s_bias_f;
    static thread_local size_t s_bias_cap = 0;

    if (s_bias_cap < static_cast<size_t>(N)) {
        s_bias_f.reset(static_cast<float *>(zendnnl_aligned_alloc(64,
                ((static_cast<size_t>(N) * sizeof(float) + 63)
                        & ~size_t(63)))));
        s_bias_cap = s_bias_f ? static_cast<size_t>(N) : 0;
        if (!s_bias_f) return false;
    }
    float *bias_f = s_bias_f.get();
    const uint16_t *bb = static_cast<const uint16_t *>(bias);
    for (int n = 0; n < N; ++n) {
        uint32_t bits = static_cast<uint32_t>(bb[n]) << 16;
        std::memcpy(&bias_f[n], &bits, sizeof(float));
    }
    *bias_f_out = bias_f;
    return true;
}

} // namespace

native_gemv_plan_t int8_gemv_choice(const GemmDescriptor &desc,
        const UarchParams &uarch, const matmul_params &params) {
    native_gemv_plan_t plan;
    const int M = desc.M, N = desc.N, K = desc.K;
    if (M != 1 || !uarch.avx512vnni || desc.transA) return plan;
    if (desc.src_dt != data_type_t::u8 && desc.src_dt != data_type_t::s8)
        return plan;
    if (desc.wei_dt != data_type_t::s8) return plan;
    if (desc.dst_dt != data_type_t::bf16 && desc.dst_dt != data_type_t::f32)
        return plan;
    if (N <= 0 || desc.num_threads > 1 || desc.ldc != N) return plan;
    // The kernel scales the dequantized value, with its bias and zero-point
    // term, by alpha and adds beta * C after the scales; the reference order
    // scales only the integer sum by alpha and adds beta * C before them.
    if (desc.alpha != 1.0f || desc.beta != 0.0f) return plan;

    const int K_padded = (K + 3) & ~3;
    const int N_padded = ((N + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
    const size_t b_packed_bytes
            = static_cast<size_t>(K_padded) * N_padded * sizeof(int8_t);
    if (b_packed_bytes > static_cast<size_t>(uarch.l2_bytes)) return plan;

    fused_postop_t fused = fused_postop_t::none;
    bool has_unfuseable = false;
    if (!scan_gemv_postops(params, &fused, &has_unfuseable)) return plan;

    // A per-tensor f32 source scale and an s32 / s8 / u8 zero point; f32
    // weight scales per tensor or per column (the kernel reads every scale
    // as f32); no weight zero point and no dst scale or zero point, which
    // the kernel does not apply.
    const auto &qp = params.quant_params;
    const auto &ws = qp.wei_scale;
    if (ws.buff != nullptr
            && (ws.dt != data_type_t::f32
                    || (!quant_param_dims_are_per_tensor(ws.dims)
                            && !quant_param_is_per_channel(ws, N))))
        return plan;
    const auto &src_sc_dims = qp.src_scale.dims;
    if (!src_sc_dims.empty() && src_sc_dims.back() > 1) return plan;
    if (qp.src_scale.buff != nullptr && qp.src_scale.dt != data_type_t::f32)
        return plan;
    const auto &src_zp_dims = qp.src_zp.dims;
    if (!src_zp_dims.empty() && src_zp_dims.back() > 1) return plan;
    if (qp.src_zp.buff != nullptr && qp.src_zp.dt != data_type_t::s32
            && qp.src_zp.dt != data_type_t::s8
            && qp.src_zp.dt != data_type_t::u8)
        return plan;
    if (qp.wei_zp.buff != nullptr || qp.dst_scale.buff != nullptr
            || qp.dst_zp.buff != nullptr)
        return plan;

    plan.kernel = native_gemv_t::int8_bkc;
    plan.fused = fused;
    return plan;
}

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl,avx512vnni")
status_t int8_gemv_run(const native_gemv_plan_t &plan,
        const GemmDescriptor &desc, const void *src, const void *weight,
        void *dst, const void *bias, const matmul_params &params,
        int32_t weight_cache_mode) {
    const int N = desc.N, K = desc.K;
    const int ldb = desc.ldb;
    const bool src_is_u8 = (desc.src_dt == data_type_t::u8);
    const bool dst_is_bf16 = (desc.dst_dt == data_type_t::bf16);
    const bool dst_is_fp32 = (desc.dst_dt == data_type_t::f32);
    const int K_padded = (K + 3) & ~3;
    const int N_padded = ((N + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
    const fused_postop_t kc_fused_op = plan.fused;

    const Int8QuantParams qp = extract_int8_quant(params);
    float src_scale = qp.src_scale;
    int32_t src_zp = qp.src_zp;
    float wei_scale_default = 1.0f;
    const float *wei_scale_ptr
            = qp.wei_scale ? qp.wei_scale : &wei_scale_default;
    int wei_scale_count = qp.wei_scale_count;

    // ── Prepare A as u8 ──
    // If source is s8, convert to u8 by adding 128 and adjust zero point.
    const uint8_t *A_u8 = nullptr;
    int32_t effective_zp = src_zp;

    if (src_is_u8) {
        A_u8 = static_cast<const uint8_t *>(src);
    } else {
        static thread_local std::unique_ptr<uint8_t, native_aligned_free_t>
                s_a_buf;
        static thread_local size_t s_a_cap = 0;
        // s8 → u8: add 128, adjust zp
        if (s_a_cap < static_cast<size_t>(K)) {
            s_a_buf.reset(static_cast<uint8_t *>(
                    zendnnl_aligned_alloc(64, ((K + 63) & ~size_t(63)))));
            s_a_cap = s_a_buf ? K : 0;
        }
        if (!s_a_buf) return status_t::failure;
        uint8_t *a_buf = s_a_buf.get();
        const int8_t *a_s8 = static_cast<const int8_t *>(src);
        const __m512i v128 = _mm512_set1_epi8(static_cast<int8_t>(-128));
        int k = 0;
        for (; k + 63 < K; k += 64) {
            __m512i s = _mm512_loadu_si512(a_s8 + k);
            _mm512_storeu_si512(a_buf + k, _mm512_sub_epi8(s, v128));
        }
        for (; k < K; ++k)
            a_buf[k] = static_cast<uint8_t>(static_cast<int>(a_s8[k]) + 128);
        A_u8 = a_buf;
        effective_zp = src_zp + 128;
    }

    const bool has_bias = (desc.bias != nullptr);
    const float *bias_f = nullptr;
    if (!int8_resolve_bias_f(desc, N, bias, has_bias, &bias_f))
        return status_t::failure;

    // ── Pack + cache: the pack and its column sums depend on the weight
    // alone ──
    const int8_t *B_raw = static_cast<const int8_t *>(weight);
    const int8_t *B_kc = nullptr;
    const int32_t *col_sum = nullptr;
    [[maybe_unused]] const char *pack_source = "none";

    if (weight_cache_mode != 0) {
        // The whole key of the cache: one buffer read as [K, N] and as
        // [N, K] (a square weight) has two packs; a pack serves while the
        // weight's bytes hold the words of their two ends that it read.
        static thread_local const void *s_gc_wt = nullptr;
        static thread_local int s_gc_K = 0, s_gc_N = 0, s_gc_ldb = 0;
        static thread_local bool s_gc_tr = false;
        static thread_local const INT8KContiguousWeight *s_gc_entry = nullptr;
        static thread_local uint64_t s_gc_gen = 0;

        const size_t extent = native_raw_weight_bytes(
                data_type_t::s8, K, N, ldb, desc.transB);
        const uint64_t cur_gen
                = weight_cache_generation().load(std::memory_order_relaxed);
        if (s_gc_wt == weight && s_gc_K == K && s_gc_N == N && s_gc_ldb == ldb
                && s_gc_tr == desc.transB && s_gc_entry && s_gc_gen == cur_gen
                && weight_end_words_hold(
                        s_gc_entry->source_ends, weight, extent)) {
            B_kc = s_gc_entry->data;
            col_sum = s_gc_entry->col_sum;
            pack_source = "tl_cache";
        } else {
            PrepackedWeightKey key {weight, K, N, ldb, desc.transB};
            const INT8KContiguousWeight *cached
                    = INT8KContiguousWeightCache::instance().get_or_pack(
                            key, B_raw, extent);
            if (cached) {
                B_kc = cached->data;
                col_sum = cached->col_sum;
                s_gc_wt = weight;
                s_gc_K = K;
                s_gc_N = N;
                s_gc_ldb = ldb;
                s_gc_tr = desc.transB;
                s_gc_entry = cached;
                s_gc_gen = cur_gen;
                pack_source = "global_cache";
            }
        }
    }
    if (!B_kc) {
        static thread_local std::unique_ptr<int8_t, native_aligned_free_t>
                s_bkc;
        static thread_local size_t s_bkc_cap = 0;
        static thread_local std::unique_ptr<int32_t, native_aligned_free_t>
                s_cs;
        static thread_local size_t s_cs_cap = 0;

        const size_t need_pack = static_cast<size_t>(K_padded) * N_padded;
        constexpr size_t bkc_guard = NR_PACK * INT8_VNNI_GRP;
        if (s_bkc_cap < need_pack + bkc_guard) {
            s_bkc.reset(static_cast<int8_t *>(zendnnl_aligned_alloc(
                    64, ((need_pack + bkc_guard + 63) & ~size_t(63)))));
            s_bkc_cap = s_bkc ? need_pack + bkc_guard : 0;
        }
        if (!s_bkc) return status_t::failure;

        if (s_cs_cap < static_cast<size_t>(N_padded)) {
            s_cs.reset(static_cast<int32_t *>(zendnnl_aligned_alloc(
                    64, ((N_padded * sizeof(int32_t) + 63) & ~size_t(63)))));
            s_cs_cap = s_cs ? N_padded : 0;
        }
        if (!s_cs) return status_t::failure;

        pack_b_int8_bkc(B_raw, ldb, K, N, desc.transB, s_bkc.get(), s_cs.get());
        std::memset(s_bkc.get() + need_pack, 0, bkc_guard);

        B_kc = s_bkc.get();
        col_sum = s_cs.get();
        pack_source = "thread_local";
    }

    // ── Dequantization: calls on one weight may carry other source scales
    // and zero points, weight scales and biases (also new values in the same
    // buffers), so the kernel applies this call's own ──
    int8_gemv_dequant_t dq;
    dq.col_sum = col_sum;
    dq.bias = bias_f;
    dq.wei_scale = wei_scale_ptr;
    dq.per_channel = wei_scale_count > 1;
    dq.src_scale = src_scale;
    dq.src_zp = effective_zp;

    // ── Compute ──
    int8_gemv_bkc(A_u8, B_kc, dq,
            dst_is_bf16 ? static_cast<uint16_t *>(dst) : nullptr,
            dst_is_fp32 ? static_cast<float *>(dst) : nullptr, kc_fused_op,
            desc.alpha, desc.beta, dst_is_bf16, K, N);

    static bool s_log = apilog_info_enabled();
    if (s_log) {
        const size_t b_packed_bytes
                = static_cast<size_t>(K_padded) * N_padded * sizeof(int8_t);
        apilog_info("Native INT8 BKC-GEMV: M=1 K=", K, " N=", N,
                " K_padded=", K_padded, " N_padded=", N_padded,
                " packed_B=", b_packed_bytes / 1024, "KB",
                " src=", src_is_u8 ? "u8" : "s8",
                " dst=", dst_is_bf16 ? "bf16" : "fp32",
                " src_zp=", effective_zp, " wei_scale=",
                wei_scale_count > 1 ? "per_channel" : "per_tensor",
                " fused_op=", static_cast<int>(kc_fused_op),
                " pack=", pack_source);
    }
    return status_t::success;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
