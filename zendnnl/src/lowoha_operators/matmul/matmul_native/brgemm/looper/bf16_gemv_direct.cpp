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
#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_gemv_bkc.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_gemv_narrow.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using namespace zendnnl::error_handling;

namespace {

inline void bf16_bias_to_f32_span(const uint16_t *bb, int count, float *out) {
    for (int n = 0; n < count; ++n) {
        uint32_t bits = static_cast<uint32_t>(bb[n]) << 16;
        std::memcpy(&out[n], &bits, sizeof(float));
    }
}

/// BF16 bias → FP32 scratch (thread_local), full row length N, converted on
/// every call: the caller's buffer may hold new values at the same address.
bool bkc_resolve_bias_f(const GemmDescriptor &desc, int N, const void *bias,
        bool has_bias, const float **bias_f_out) {
    *bias_f_out = nullptr;
    if (!has_bias) return true;
    const bool bias_is_bf16 = (desc.bias_dt == data_type_t::bf16);
    if (desc.bias_dt == data_type_t::f32) {
        *bias_f_out = static_cast<const float *>(bias);
        return true;
    }
    if (!bias_is_bf16) return false;

    static thread_local std::unique_ptr<float, native_aligned_free_t>
            s_bias_fp32;
    static thread_local size_t s_bias_cap = 0;

    if (s_bias_cap < static_cast<size_t>(N)) {
        s_bias_fp32.reset(static_cast<float *>(zendnnl_aligned_alloc(64,
                ((static_cast<size_t>(N) * sizeof(float) + 63)
                        & ~size_t(63)))));
        s_bias_cap = s_bias_fp32 ? static_cast<size_t>(N) : 0;
        if (!s_bias_fp32) return false;
    }
    float *bias_f = s_bias_fp32.get();
    bf16_bias_to_f32_span(static_cast<const uint16_t *>(bias), N, bias_f);
    *bias_f_out = bias_f;
    return true;
}

/// Single-thread packed B under the call's weight cache mode: at 1 or 2 the
/// global weight cache, keyed by \c PrepackedWeightKey (pointer, K, N, ldb,
/// transB), whose pack serves while the weight's bytes hold the words of
/// their two ends that it read; at 0, or when the cache refuses a new entry
/// (ZENDNNL_LRU_CACHE_CAPACITY), a thread-local buffer packed for this call
/// and never reused: a caller refills a weight buffer at mode 0.  Returns
/// success, failure when a buffer cannot be allocated, or unimplemented,
/// with nothing packed, when an in-place rewrite of the buffer replaced the
/// raw weight since the call found it raw.
status_t bkc_resolve_packed_B_st(const GemmDescriptor &desc, int K, int N,
        int K_padded, int N_padded, const void *weight, const uint16_t *B_raw,
        int32_t weight_cache_mode, const uint16_t **B_bkc_out,
        const char **pack_source_out) {
    *B_bkc_out = nullptr;
    *pack_source_out = "none";
    const int ldb = desc.ldb;
    const size_t need = static_cast<size_t>(K_padded) * N_padded;

    const auto pack_for_call = [&]() {
        static thread_local std::unique_ptr<uint16_t, native_aligned_free_t>
                tl_buf;
        static thread_local size_t tl_cap = 0;
        if (tl_cap < need) {
            tl_buf.reset(static_cast<uint16_t *>(zendnnl_aligned_alloc(
                    64, ((need * sizeof(uint16_t) + 63) & ~size_t(63)))));
            tl_cap = tl_buf ? need : 0;
        }
        if (!tl_buf) return status_t::failure;
        pack_b_bkc_ext(B_raw, ldb, K, N, desc.transB, tl_buf.get());
        *B_bkc_out = tl_buf.get();
        *pack_source_out = "thread_local";
        return status_t::success;
    };

    if (weight_cache_mode != 0) {
        // The whole key of the cache: one buffer read as [K, N] and as
        // [N, K] (a square weight) has two packs.
        static thread_local const void *tl_g_wt = nullptr;
        static thread_local int tl_g_K = 0, tl_g_N = 0, tl_g_ldb = 0;
        static thread_local bool tl_g_tr = false;
        static thread_local const BF16BKCWeight *tl_g_entry = nullptr;
        static thread_local uint64_t tl_g_gen = 0;

        const size_t extent = native_raw_weight_bytes(
                data_type_t::bf16, K, N, ldb, desc.transB);
        const uint64_t cur_gen
                = weight_cache_generation().load(std::memory_order_relaxed);
        if (tl_g_wt == weight && tl_g_K == K && tl_g_N == N && tl_g_ldb == ldb
                && tl_g_tr == desc.transB && tl_g_entry && tl_g_gen == cur_gen
                && weight_end_words_hold(
                        tl_g_entry->source_ends, weight, extent)) {
            *B_bkc_out = tl_g_entry->data;
            *pack_source_out = "tl_hit_global_ptr";
            return status_t::success;
        }
        const PrepackedWeightKey key {weight, K, N, ldb, desc.transB};
        const BF16BKCWeight *cached
                = BF16BKCWeightCache::instance().find(key, extent);
        if (!cached) {
            // A first call of the tiles at weight cache mode 2 may be
            // rewriting this buffer in place: the pack reads the raw weight
            // while the native weight cache holds the buffer, which it does
            // only while the buffer holds no image.
            status_t packed = status_t::failure;
            switch (native_weight_read_raw(weight, extent, extent, [&] {
                cached = BF16BKCWeightCache::instance().get_or_pack(
                        key, B_raw, extent);
                if (!cached) packed = pack_for_call();
            })) {
                case native_raw_read_t::done: break;
                case native_raw_read_t::image: return status_t::unimplemented;
                case native_raw_read_t::failed: return status_t::failure;
            }
            if (!cached) return packed;
        }
        *B_bkc_out = cached->data;
        tl_g_wt = weight;
        tl_g_K = K;
        tl_g_N = N;
        tl_g_ldb = ldb;
        tl_g_tr = desc.transB;
        tl_g_entry = cached;
        tl_g_gen = cur_gen;
        *pack_source_out = "global_cache";
        return status_t::success;
    }
    return pack_for_call();
}

} // namespace

native_gemv_plan_t bf16_gemv_choice(const GemmDescriptor &desc,
        const UarchParams &uarch, const matmul_params &params) {
    native_gemv_plan_t plan;
    const int M = desc.M, N = desc.N, K = desc.K;
    if (M != 1 || desc.num_threads > 1 || !uarch.avx512bf16 || desc.transA
            || N <= 0)
        return plan;
    if (desc.dst_dt != data_type_t::bf16 && desc.dst_dt != data_type_t::f32)
        return plan;
    fused_postop_t fused = fused_postop_t::none;
    bool has_unfuseable = false;
    if (!scan_gemv_postops(params, &fused, &has_unfuseable)) return plan;
    if (desc.bias != nullptr && desc.bias_dt != data_type_t::bf16
            && desc.bias_dt != data_type_t::f32)
        return plan;
    if (desc.ldc < N) return plan;

    const int K_padded = (K + 1) & ~1;
    const size_t l2_cap = static_cast<size_t>(uarch.l2_bytes);
    native_gemv_t kernel = native_gemv_t::bf16_bkc;
    if (N <= kBf16GemvNarrowMaxN && !desc.transB) {
        // BKC pads N to BKC_NR_PAD columns; the narrow kernel reads the
        // caller's row-major weight in place and keeps the N columns in one
        // xmm register.
        kernel = native_gemv_t::bf16_narrow;
    } else {
        const int N_padded = ((N + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
        const size_t b_packed_bytes
                = static_cast<size_t>(K_padded) * N_padded * sizeof(uint16_t);
        // >L2 shapes stream through the cached BKC data directly — the
        // kernel's sequential k-pair access lets the HW prefetcher handle
        // L3/DRAM traffic with accumulators in ZMM registers. Requires const
        // weights for caching.
        if (b_packed_bytes > l2_cap && !desc.is_weights_const) return plan;
    }
    plan.kernel = kernel;
    plan.fused = fused;
    return plan;
}

ZENDNNL_TARGET("avx512f,avx512bf16,avx512bw,avx512vl,fma")
status_t bf16_gemv_run(const native_gemv_plan_t &plan,
        const GemmDescriptor &desc, const void *src, const void *weight,
        void *dst, const void *bias, int32_t weight_cache_mode) {
    const int N = desc.N, K = desc.K;
    const float alpha = desc.alpha, beta = desc.beta;
    const int K_padded = (K + 1) & ~1;
    const fused_postop_t kc_fused_op = plan.fused;
    const bool has_bias = (desc.bias != nullptr);
    const bool dst_is_bf16 = (desc.dst_dt == data_type_t::bf16);
    const uint16_t *A = static_cast<const uint16_t *>(src);
    const uint16_t *B_raw = static_cast<const uint16_t *>(weight);

    // The kernels' epilogues take an FP32 bias; nullptr without a bias.
    const float *bias_f = nullptr;
    if (!bkc_resolve_bias_f(desc, N, bias, has_bias, &bias_f))
        return status_t::failure;

    if (plan.kernel == native_gemv_t::bf16_narrow) {
        uint16_t *C_bf16_ptr
                = dst_is_bf16 ? static_cast<uint16_t *>(dst) : nullptr;
        float *C_fp32_ptr = !dst_is_bf16 ? static_cast<float *>(dst) : nullptr;
        bf16_gemv_narrow(A, K, B_raw, desc.ldb, C_bf16_ptr, C_fp32_ptr, bias_f,
                kc_fused_op, alpha, beta, dst_is_bf16, N);

        static bool s_log_narrow = apilog_info_enabled();
        if (s_log_narrow) {
            apilog_info("Native BF16 narrow-GEMV: M=1 K=", K, " N=", N,
                    " beta=", beta, " bias=", has_bias ? "yes" : "no",
                    " fused_op=", static_cast<int>(kc_fused_op),
                    " dst=", dst_is_bf16 ? "bf16" : "fp32");
        }
        return status_t::success;
    }

    const int N_padded = ((N + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
    const uint16_t *B_bkc = nullptr;
    const char *pack_source = nullptr;
    const status_t packed = bkc_resolve_packed_B_st(desc, K, N, K_padded,
            N_padded, weight, B_raw, weight_cache_mode, &B_bkc, &pack_source);
    if (packed != status_t::success) return packed;

    bf16_gemv_bkc(A, B_bkc,
            dst_is_bf16 ? static_cast<uint16_t *>(dst) : nullptr,
            !dst_is_bf16 ? static_cast<float *>(dst) : nullptr, bias_f,
            kc_fused_op, alpha, beta, dst_is_bf16, K, N);

    static bool s_log = apilog_info_enabled();
    if (s_log) {
        // Diagnostic-only scalars — computed inside the logging gate
        // so the hot path doesn't pay for them when APILOG is off.
        const size_t b_packed_bytes
                = static_cast<size_t>(K_padded) * N_padded * sizeof(uint16_t);
        const int waste_cols = N_padded - N;
        const int waste_pct
                = (N_padded > 0) ? (waste_cols * 100 / N_padded) : 0;
        apilog_info("Native BF16 BKC-GEMV: M=1 K=", K, " N=", N,
                " K_padded=", K_padded, " N_padded=", N_padded,
                " waste=", waste_cols, "cols(", waste_pct, "%)",
                " packed_B=", b_packed_bytes / 1024, "KB", " beta=", beta,
                " transB=", desc.transB ? "true" : "false",
                " bias=", has_bias ? "yes" : "no",
                " fused_op=", static_cast<int>(kc_fused_op),
                " dst=", dst_is_bf16 ? "bf16" : "fp32", " pack=", pack_source);
    }
    return status_t::success;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
