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
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_gemv_bkc.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"
#include "operators/matmul/matmul_config.hpp"
#include "common/zendnnl_global.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <omp.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using namespace zendnnl::error_handling;
using zendnnl::ops::matmul_config_t;

namespace {

/// Worst per-thread packed column-slice size (bytes) for team size \p nt
/// (\p chunk = ceil(N/nt), padded to BKC_NR_PAD).
inline size_t bf16_bkc_mt_worst_slice_packed_bytes(int K_padded, int N, int nt) {
    if (nt < 1) nt = 1;
    const int chunk = (N + nt - 1) / nt;
    const int n_pad = ((chunk + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
    return static_cast<size_t>(K_padded) * n_pad * sizeof(uint16_t);
}

/// Max fraction of L2 used for each thread's packed B slice in the MT path
/// (non-const weights only). Leaves headroom for A, outputs, bias, and
/// prefetch. Const-weight shapes bypass this gate — per-thread BKC packing
/// is cached thread-local, and first-touch gives NUMA-local L3 residency.
inline constexpr unsigned kBf16BkcMtL2SliceBudgetPct = 85;

/// Map OpenMP \p tid → column slice index (same \p chunk = ceil(N/nt) layout as
/// linear scheduling, but permuted so consecutive \p tid values map to the same
/// CCX). When \p nt % ccx_cores == 0 and \p nt >= ccx_cores, indices \p s are a
/// permutation of 0..nt-1 (every slice assigned once). Otherwise returns \p tid
/// (linear). \p nt == ccx_cores gives the identity map.
inline int bf16_gemv_mt_slice_index(int tid, int nt, int ccx_cores) {
    if (ccx_cores < 2 || nt < ccx_cores || (nt % ccx_cores) != 0)
        return tid;
    const int num_ccxs = nt / ccx_cores;
    const int ccx_id = tid / ccx_cores;
    const int local_id = tid % ccx_cores;
    return ccx_id + local_id * num_ccxs;
}

/// Column range for slice \p s with fixed \p chunk (same as linear scheduling).
inline void bf16_gemv_mt_slice_column_range(int s, int N, int chunk,
                                            int *n0_out, int *n1_out) {
    const int n0 = s * chunk;
    *n0_out = n0;
    *n1_out = std::min(N, n0 + chunk);
}

inline void bf16_bias_to_f32_span(const uint16_t *bb, int count, float *out) {
    for (int n = 0; n < count; ++n) {
        uint32_t bits = static_cast<uint32_t>(bb[n]) << 16;
        std::memcpy(&out[n], &bits, sizeof(float));
    }
}

/// BF16 bias → FP32 scratch (thread_local). Full row length N.
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

    static thread_local float *s_bias_fp32 = nullptr;
    static thread_local size_t s_bias_cap = 0;
    static thread_local const void *s_bias_ptr = nullptr;
    static thread_local int s_bias_N = 0;
    static thread_local uint64_t s_bias_gen = 0;

    const uint64_t cur_gen = weight_cache_generation().load(
        std::memory_order_relaxed);
    if (s_bias_fp32 && s_bias_ptr == bias && s_bias_N == N
        && s_bias_gen == cur_gen) {
        *bias_f_out = s_bias_fp32;
        return true;
    }
    if (s_bias_cap < static_cast<size_t>(N)) {
        std::free(s_bias_fp32);
        s_bias_fp32 = static_cast<float *>(std::aligned_alloc(
            64, ((static_cast<size_t>(N) * sizeof(float) + 63) & ~size_t(63))));
        s_bias_cap = s_bias_fp32 ? static_cast<size_t>(N) : 0;
        if (!s_bias_fp32) return false;
    }
    bf16_bias_to_f32_span(static_cast<const uint16_t *>(bias), N, s_bias_fp32);
    s_bias_ptr = bias;
    s_bias_N = N;
    s_bias_gen = cur_gen;
    *bias_f_out = s_bias_fp32;
    return true;
}

/// Single-thread packed B: global weight cache when const+enabled, else
/// thread-local buffer. Keyed by \c PrepackedWeightKey (pointer, K, N, ldb, transB).
bool bkc_resolve_packed_B_st(
    const GemmDescriptor &desc,
    int K, int N, int K_padded, int N_padded,
    const void *weight, const uint16_t *B_raw,
    const uint16_t **B_bkc_out, const char **pack_source_out) {
    *B_bkc_out = nullptr;
    *pack_source_out = "none";
    const int ldb = desc.ldb;
    const size_t need = static_cast<size_t>(K_padded) * N_padded;

    const int32_t wcache = matmul_config_t::instance().get_weight_cache();
    if (desc.is_weights_const && wcache != 0) {
        static thread_local const void *tl_g_wt = nullptr;
        static thread_local int tl_g_K = 0, tl_g_N = 0;
        static thread_local const uint16_t *tl_g_data = nullptr;
        static thread_local uint64_t tl_g_gen = 0;

        const uint64_t cur_gen = weight_cache_generation().load(
            std::memory_order_relaxed);
        if (tl_g_wt == weight && tl_g_K == K && tl_g_N == N
            && tl_g_data && tl_g_gen == cur_gen) {
            *B_bkc_out = tl_g_data;
            *pack_source_out = "tl_hit_global_ptr";
            return true;
        }
        const PrepackedWeightKey key{weight, K, N, ldb, desc.transB};
        const BF16BKCWeight *cached =
            BF16BKCWeightCache::instance().get_or_pack(key, B_raw);
        if (!cached) return false;
        *B_bkc_out = cached->data;
        tl_g_wt = weight;
        tl_g_K = K;
        tl_g_N = N;
        tl_g_data = cached->data;
        tl_g_gen = cur_gen;
        *pack_source_out = "global_cache";
        return true;
    }

    static thread_local uint16_t *tl_buf = nullptr;
    static thread_local size_t tl_cap = 0;
    static thread_local const void *tl_wt = nullptr;
    static thread_local int tl_K = 0, tl_N = 0, tl_ldb = 0;
    static thread_local bool tl_tr = false;
    static thread_local uint64_t tl_gen = 0;

    const uint64_t cur_gen = weight_cache_generation().load(
        std::memory_order_relaxed);
    const bool hit = (tl_buf && tl_cap >= need && desc.is_weights_const
                      && tl_wt == weight && tl_K == K && tl_N == N
                      && tl_ldb == ldb && tl_tr == desc.transB
                      && tl_gen == cur_gen);
    if (!hit) {
        if (tl_cap < need) {
            std::free(tl_buf);
            tl_buf = static_cast<uint16_t *>(std::aligned_alloc(
                64, ((need * sizeof(uint16_t) + 63) & ~size_t(63))));
            tl_cap = tl_buf ? need : 0;
        }
        if (!tl_buf) return false;
        pack_b_bkc_ext(B_raw, ldb, K, N, desc.transB, tl_buf);
        tl_wt = weight;
        tl_K = K;
        tl_N = N;
        tl_ldb = ldb;
        tl_tr = desc.transB;
        tl_gen = cur_gen;
    }
    *B_bkc_out = tl_buf;
    *pack_source_out = "thread_local";
    return true;
}

/// Per-thread N-column slice pack (multi-thread GEMV). No global cache.
bool bkc_pack_column_slice(
    const GemmDescriptor &desc, int K, int K_padded,
    int n0, int n_cols,
    const void *weight, const uint16_t *B_raw,
    const uint16_t **packed_out) {
    const int ldb = desc.ldb;
    const int n_pad = ((n_cols + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
    const size_t need = static_cast<size_t>(K_padded) * n_pad;

    static thread_local uint16_t *buf = nullptr;
    static thread_local size_t cap = 0;
    static thread_local const void *w = nullptr;
    static thread_local int k_stored = 0, ldb_stored = 0;
    static thread_local int n0_stored = 0, nloc_stored = 0;
    static thread_local bool tr_stored = false;
    static thread_local uint64_t gen_stored = 0;

    const uint64_t cur_gen = weight_cache_generation().load(
        std::memory_order_relaxed);
    const bool can_reuse = desc.is_weights_const
        && buf && cap >= need
        && w == weight && k_stored == K && ldb_stored == ldb
        && n0_stored == n0 && nloc_stored == n_cols
        && tr_stored == desc.transB && gen_stored == cur_gen;

    if (!can_reuse) {
        if (cap < need) {
            std::free(buf);
            buf = static_cast<uint16_t *>(std::aligned_alloc(
                64, ((need * sizeof(uint16_t) + 63) & ~size_t(63))));
            cap = buf ? need : 0;
        }
        if (!buf) return false;
        pack_b_bkc_ext(B_raw, ldb, K, n_cols, desc.transB, buf, n0);
        w = weight;
        k_stored = K;
        ldb_stored = ldb;
        n0_stored = n0;
        nloc_stored = n_cols;
        tr_stored = desc.transB;
        gen_stored = cur_gen;
    }
    *packed_out = buf;
    return true;
}

} // namespace

__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,fma")))
bool bf16_gemv_direct(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;
    const float alpha = desc.alpha, beta = desc.beta;

    if (M != 1) return false;
    if (!uarch.avx512bf16) return false;
    if (desc.transA) return false;
    if (N <= 0) return false;

    const int K_padded = (K + 1) & ~1;
    const size_t l2_cap = static_cast<size_t>(uarch.l2_bytes);
    const int num_threads = desc.num_threads;

    fused_postop_t kc_fused_op = fused_postop_t::none;
    bool has_unfuseable = false;
    if (!scan_gemv_postops(params, &kc_fused_op, &has_unfuseable))
        return false;

    const bool has_bias = (desc.bias != nullptr);
    const bool dst_is_bf16 = (desc.dst_dt == data_type_t::bf16);
    const bool bias_is_bf16 = (desc.bias_dt == data_type_t::bf16);
    const uint16_t *A = static_cast<const uint16_t *>(src);
    const uint16_t *B_raw = static_cast<const uint16_t *>(weight);

    if (has_bias && !bias_is_bf16 && desc.bias_dt != data_type_t::f32)
        return false;

    if (num_threads > 1) {
        if (desc.ldc < N) return false;

        const int nt = m1_gemv_cap_threads(N, num_threads);
        const int chunk = (N + nt - 1) / nt;
        const size_t slice_bytes =
            bf16_bkc_mt_worst_slice_packed_bytes(K_padded, N, nt);
        const size_t l2_slice_budget =
            (l2_cap * kBf16BkcMtL2SliceBudgetPct) / 100;
        // Const weights: allow >L2 slices. Per-thread BKC packing is cached
        // thread-local (one-time cost), and each thread's first-touch places
        // pages in its local CCD's DRAM → L3-resident with NUMA locality.
        // The kernel's sequential k-pair access lets HW prefetcher stream from
        // L3/DRAM. Non-const weights repack every call → gate on L2.
        if (slice_bytes > l2_slice_budget && !desc.is_weights_const)
            return false;

        const float *bias_f_row = nullptr;
        if (!bkc_resolve_bias_f(desc, N, bias, has_bias, &bias_f_row))
            return false;

        uint16_t *C_bf16 = dst_is_bf16 ? static_cast<uint16_t *>(dst) : nullptr;
        float *C_fp32 = !dst_is_bf16 ? static_cast<float *>(dst) : nullptr;

        std::atomic<int> pack_fail{0};
        const int ccx_w = uarch.ccx_cores;
        #pragma omp parallel num_threads(nt)
        {
            const int tid = omp_get_thread_num();
            const int s = bf16_gemv_mt_slice_index(tid, nt, ccx_w);
            int n0 = 0, n1 = 0;
            bf16_gemv_mt_slice_column_range(s, N, chunk, &n0, &n1);
            if (n0 < N) {
                const int n_loc = n1 - n0;
                const uint16_t *B_slice = nullptr;
                if (!bkc_pack_column_slice(desc, K, K_padded, n0, n_loc,
                                           weight, B_raw, &B_slice))
                    pack_fail.store(1, std::memory_order_relaxed);
                else
                    bf16_gemv_bkc(
                        A, B_slice,
                        C_bf16 ? C_bf16 + n0 : nullptr,
                        C_fp32 ? C_fp32 + n0 : nullptr,
                        bias_f_row ? bias_f_row + n0 : nullptr,
                        kc_fused_op, alpha, beta, dst_is_bf16, K, n_loc);
            }
        }
        return pack_fail.load(std::memory_order_relaxed) == 0;
    }

    // Single-thread
    if (desc.ldc < N) return false;
    const int N_padded = ((N + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
    const size_t b_packed_bytes =
        static_cast<size_t>(K_padded) * N_padded * sizeof(uint16_t);

    // >L2 shapes stream through the cached BKC data directly — the kernel's
    // sequential k-pair access lets the HW prefetcher handle L3/DRAM traffic
    // with accumulators in ZMM registers. Requires const weights for caching.
    if (b_packed_bytes > l2_cap && !desc.is_weights_const)
        return false;

    const float *bias_f = nullptr;
    if (!bkc_resolve_bias_f(desc, N, bias, has_bias, &bias_f))
        return false;

    const uint16_t *B_bkc = nullptr;
    const char *pack_source = nullptr;
    if (!bkc_resolve_packed_B_st(desc, K, N, K_padded, N_padded,
                                 weight, B_raw, &B_bkc, &pack_source))
        return false;

    bf16_gemv_bkc(
        A, B_bkc,
        dst_is_bf16 ? static_cast<uint16_t *>(dst) : nullptr,
        !dst_is_bf16 ? static_cast<float *>(dst) : nullptr,
        (has_bias && bias_f) ? bias_f : nullptr,
        kc_fused_op, alpha, beta, dst_is_bf16, K, N);

    static bool s_log = apilog_info_enabled();
    if (s_log) {
        apilog_info("Native BF16 BKC-GEMV: M=1 K=", K, " N=", N,
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
