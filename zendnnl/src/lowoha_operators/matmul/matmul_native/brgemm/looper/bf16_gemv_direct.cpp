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
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "operators/matmul/matmul_config.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_brgemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/postop.hpp"

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstdlib>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

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

// ── Small-N GEMV: row-major B, FP32 FMA, no packing ──────────────────
// For N < 64 (1-3 ZMM vectors), reads B in natural layout.
// Eliminates NR=64 padding waste and packing overhead entirely.
//
// Both GEMV fast paths (small_n and prepacked) bail out when alpha != 1.0
// to avoid incorrect epilogue ordering. The standard BRGEMM/GEMM path
// handles alpha != 1.0 correctly via deferred scale_tile + postop reapply.
__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,fma")))
static bool bf16_gemv_small_n(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params) {

    (void)uarch;
    const int N = desc.N, K = desc.K;
    const int ldb = desc.ldb;
    const float alpha = desc.alpha, beta = desc.beta;

    if (alpha != 1.0f) return false;

    const bool has_bias = (desc.bias != nullptr);
    const bool dst_is_bf16 = (desc.dst_dt == data_type_t::bf16);
    const bool bias_is_bf16 = (desc.bias_dt == data_type_t::bf16);
    const bool can_fuse = true;  // alpha==1.0 guaranteed by early return above

    const uint16_t *A = static_cast<const uint16_t *>(src);
    const uint16_t *B = static_cast<const uint16_t *>(weight);

    int fused_idx = -1;
    std::vector<matmul_post_op> remaining_postops;
    for (int i = 0; i < static_cast<int>(params.postop_.size()); ++i) {
        auto pt = params.postop_[i].po_type;
        bool is_fused = (pt == post_op_type_t::relu && params.postop_[i].alpha == 0.0f)
                     || pt == post_op_type_t::gelu_tanh
                     || pt == post_op_type_t::gelu_erf
                     || pt == post_op_type_t::sigmoid
                     || pt == post_op_type_t::tanh
                     || pt == post_op_type_t::swish;
        if (is_fused && fused_idx < 0) fused_idx = i;
        else remaining_postops.push_back(params.postop_[i]);
    }
    const bool has_remaining_postops = !remaining_postops.empty();
    const fused_postop_t fused_op = (can_fuse && fused_idx >= 0)
        ? detect_fused_postop(params) : fused_postop_t::none;

    // Bias BF16→FP32
    const float *bias_f = nullptr;
    if (has_bias) {
        if (bias_is_bf16) {
            static thread_local float *s_bf = nullptr;
            static thread_local size_t s_cap = 0;
            if (s_cap < static_cast<size_t>(N)) {
                std::free(s_bf);
                s_bf = static_cast<float *>(std::aligned_alloc(
                    64, ((N * sizeof(float) + 63) & ~size_t(63))));
                s_cap = s_bf ? N : 0;
            }
            if (!s_bf) return false;
            const uint16_t *bb = static_cast<const uint16_t *>(bias);
            for (int n = 0; n < N; ++n) {
                uint32_t bits = static_cast<uint32_t>(bb[n]) << 16;
                std::memcpy(&s_bf[n], &bits, sizeof(float));
            }
            bias_f = s_bf;
        } else {
            bias_f = static_cast<const float *>(bias);
        }
    }

    // FP32 C buffer
    static thread_local float *s_c = nullptr;
    static thread_local size_t s_ccap = 0;
    float *C_fp32;
    if (dst_is_bf16) {
        if (s_ccap < static_cast<size_t>(N)) {
            std::free(s_c);
            s_c = static_cast<float *>(std::aligned_alloc(
                64, ((N * sizeof(float) + 63) & ~size_t(63))));
            s_ccap = s_c ? N : 0;
        }
        if (!s_c) return false;
        C_fp32 = s_c;
        if (beta != 0.0f) {
            const uint16_t *cb = static_cast<const uint16_t *>(dst);
            for (int n = 0; n < N; ++n) {
                uint32_t bits = static_cast<uint32_t>(cb[n]) << 16;
                std::memcpy(&C_fp32[n], &bits, sizeof(float));
            }
        }
    } else {
        C_fp32 = static_cast<float *>(dst);
    }

    // Compute: C[1×N] = A[1×K] × B[K×N], row-major B, FP32 FMA
    const int nv_full = N / 16;
    const int nr_tail = N % 16;
    const int nv = nv_full + (nr_tail > 0 ? 1 : 0);
    const __mmask16 tail_mask = nr_tail
        ? static_cast<__mmask16>((1u << nr_tail) - 1) : 0xFFFF;
    __m512 acc[4];

    if (beta != 0.0f) {
        __m512 bv = _mm512_set1_ps(beta);
        for (int v = 0; v < nv_full; ++v)
            acc[v] = _mm512_mul_ps(bv, _mm512_loadu_ps(C_fp32 + v * 16));
        if (nr_tail)
            acc[nv_full] = _mm512_mul_ps(bv,
                _mm512_maskz_loadu_ps(tail_mask, C_fp32 + nv_full * 16));
    } else {
        for (int v = 0; v < nv; ++v)
            acc[v] = _mm512_setzero_ps();
    }

    for (int k = 0; k < K; ++k) {
        __m512 av = _mm512_castsi512_ps(
            _mm512_set1_epi32(static_cast<int>(A[k]) << 16));
        const uint16_t *b_row = B + k * ldb;
        for (int v = 0; v < nv_full; ++v) {
            __m512 bv = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(b_row + v * 16))),
                16));
            acc[v] = _mm512_fmadd_ps(av, bv, acc[v]);
        }
        if (nr_tail) {
            __m256i bb = _mm256_maskz_loadu_epi16(tail_mask,
                reinterpret_cast<const __m256i *>(b_row + nv_full * 16));
            __m512 bv = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(bb), 16));
            acc[nv_full] = _mm512_fmadd_ps(av, bv, acc[nv_full]);
        }
    }

    // Store: bias + fused postop
    const bool can_direct_bf16 = dst_is_bf16 && can_fuse && !has_remaining_postops;
    for (int v = 0; v < nv; ++v) {
        const int n_off = v * 16;
        const int elems = std::min(16, N - n_off);
        __m512 val = acc[v];

        if (has_bias && can_fuse && bias_f) {
            __m512 bv = (elems == 16)
                ? _mm512_loadu_ps(bias_f + n_off)
                : _mm512_maskz_loadu_ps(
                    static_cast<__mmask16>((1u << elems) - 1),
                    bias_f + n_off);
            val = _mm512_add_ps(val, bv);
        }
        if (fused_op != fused_postop_t::none)
            val = apply_fused_postop(val, fused_op);

        if (can_direct_bf16) {
            __m256bh bf = _mm512_cvtneps_pbh(val);
            if (elems == 16)
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(
                        static_cast<uint16_t *>(dst) + n_off), (__m256i)bf);
            else
                _mm256_mask_storeu_epi16(
                    static_cast<uint16_t *>(dst) + n_off,
                    (1u << elems) - 1u, (__m256i)bf);
        } else {
            if (elems == 16)
                _mm512_storeu_ps(C_fp32 + n_off, val);
            else
                _mm512_mask_storeu_ps(C_fp32 + n_off, (1u << elems) - 1u, val);
        }
    }

    // Non-fused postops
    if (has_remaining_postops && C_fp32) {
        int ldc_fp32 = dst_is_bf16 ? N : desc.ldc;
        apply_postops_tile(C_fp32, ldc_fp32, 1, N, 0, 0,
                           nullptr, remaining_postops);
    }

    // FP32→BF16 conversion for non-direct path
    if (dst_is_bf16 && !can_direct_bf16) {
        uint16_t *out = static_cast<uint16_t *>(dst);
        for (int n = 0; n < N; ++n) {
            float val = C_fp32[n];
            if (alpha != 1.0f) val *= alpha;
            if (has_bias && !can_fuse)
                val += bias_f ? bias_f[n] : 0.0f;
            uint32_t bits;
            std::memcpy(&bits, &val, sizeof(bits));
            out[n] = static_cast<uint16_t>(
                (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
        }
    }

    return true;
}


__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,fma")))
bool bf16_gemv_direct(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;
    const int ldb = desc.ldb, ldc = desc.ldc;
    const float alpha = desc.alpha, beta = desc.beta;

    if (M != 1) return false;
    if (!uarch.avx512bf16) return false;
    if (alpha != 1.0f) return false;

    // ── Decision: use direct GEMV fast path or fall through to standard
    //    3-stage BRGEMM (Planner → Looper → Kernel)?
    //
    // Fast path wins when:
    //   1. B fits in L1 (overhead-dominated, planner/looper cost > compute)
    //   2. N >= K (wide shapes where flat N scan is natural)
    //   3. N >= 64 (NR=64 panels are fully utilized, no zero-padding waste)
    //
    // Standard BRGEMM wins when:
    //   - B > L1: planner's BK tiling keeps B slices in L1 for better reuse
    //   - K > N: tall-skinny shapes need K-tiling from the looper
    //   - N < 64: NR=64 panels waste bandwidth on zero-padded columns
    //
    // For L2-resident shapes (128KB-1MB), standard BRGEMM is better because
    // the planner's BK and NB blocking decisions are critical for L1/L2 reuse.
    // These will be tuned further in the next optimization phase.
    constexpr int NR = 64;
    const size_t b_bytes = static_cast<size_t>(K) * N * sizeof(uint16_t);
    const size_t l1_capacity = static_cast<size_t>(uarch.l1d_bytes);

    // ── Small-N path: direct row-major FP32 FMA (no packing) ──
    // For N < 64, the NR=64 prepacked layout wastes 50-97% of loads on zeros.
    // Instead, read B in natural row-major layout using BF16→FP32 FMA.
    // Only 1-2 ZMM accumulators needed (N=16: 1 vec, N=32: 2 vecs).
    // Small-N direct GEMV: only for small K where the FP32 FMA approach
    // is fast enough. For large K, the 2× compute disadvantage vs dpbf16ps
    // makes this slower than even the padded-NR=64 packed path.
    if (N < NR && N > 0 && K <= 128 && !desc.transB) {
        return bf16_gemv_small_n(desc, uarch, src, weight, dst, bias, params);
    }

    if (N < NR) return false;       // shouldn't reach here, but safety
    if (N <= K) return false;       // square/tall shapes: looper's BK tiling is better
    if (b_bytes > l1_capacity) return false;  // larger shapes need planner's BK tiling

    const bool has_bias = (desc.bias != nullptr);
    const bool dst_is_bf16 = (desc.dst_dt == data_type_t::bf16);
    const bool bias_is_bf16 = (desc.bias_dt == data_type_t::bf16);
    const bool can_fuse = (alpha == 1.0f);

    // Separate fused (activation) and non-fused (binary) post-ops.
    // Fused ops are handled inside the BRGEMM kernel epilogue.
    // Non-fused ops (binary add/mul) are applied after the kernel via apply_postops_tile.
    int fused_idx = -1;
    std::vector<matmul_post_op> remaining_postops;
    for (int i = 0; i < static_cast<int>(params.postop_.size()); ++i) {
        auto pt = params.postop_[i].po_type;
        bool is_fused = (pt == post_op_type_t::relu && params.postop_[i].alpha == 0.0f)
                     || pt == post_op_type_t::gelu_tanh
                     || pt == post_op_type_t::gelu_erf
                     || pt == post_op_type_t::sigmoid
                     || pt == post_op_type_t::tanh
                     || pt == post_op_type_t::swish;
        if (is_fused && fused_idx < 0) {
            fused_idx = i;
        } else {
            remaining_postops.push_back(params.postop_[i]);
        }
    }
    const bool has_remaining_postops = !remaining_postops.empty();

    using zendnnl::ops::matmul_config_t;
    static int32_t s_weight_cache =
        matmul_config_t::instance().get_weight_cache();
    if (!desc.is_weights_const || s_weight_cache == 0) return false;

    const uint16_t *B_raw = static_cast<const uint16_t *>(weight);
    PrepackedWeightKey key{B_raw, K, N, ldb, desc.transB};
    const BF16PrepackedWeight *ppw =
        BF16PrepackedWeightCache::instance().get_or_prepack(key, B_raw);
    if (!ppw) return false;

    const uint16_t *A = static_cast<const uint16_t *>(src);
    const int vnni_stride = BF16PrepackedWeight::stride();

    // Bias BF16→FP32 conversion
    static thread_local float *s_bias_fp32 = nullptr;
    static thread_local size_t s_bias_cap = 0;
    const float *bias_f = nullptr;
    if (has_bias) {
        if (bias_is_bf16) {
            if (s_bias_cap < static_cast<size_t>(N)) {
                std::free(s_bias_fp32);
                s_bias_fp32 = static_cast<float *>(std::aligned_alloc(
                    64, ((N * sizeof(float) + 63) & ~size_t(63))));
                s_bias_cap = s_bias_fp32 ? N : 0;
            }
            if (!s_bias_fp32) return false;
            const uint16_t *bb = static_cast<const uint16_t *>(bias);
            for (int n = 0; n < N; ++n) {
                uint32_t bits = static_cast<uint32_t>(bb[n]) << 16;
                std::memcpy(&s_bias_fp32[n], &bits, sizeof(float));
            }
            bias_f = s_bias_fp32;
        } else {
            bias_f = static_cast<const float *>(bias);
        }
    }

    const fused_postop_t fused_op = can_fuse
        ? detect_fused_postop(params) : fused_postop_t::none;
    const bool can_direct_bf16 = dst_is_bf16 && can_fuse && !has_remaining_postops;

    // FP32 accumulation buffer for BF16 output
    static thread_local float *s_c_fp32 = nullptr;
    static thread_local size_t s_c_cap = 0;
    float *C_fp32 = nullptr;
    int ldc_fp32 = 0;

    if (dst_is_bf16) {
        if (s_c_cap < static_cast<size_t>(N)) {
            std::free(s_c_fp32);
            s_c_fp32 = static_cast<float *>(std::aligned_alloc(
                64, ((N * sizeof(float) + 63) & ~size_t(63))));
            s_c_cap = s_c_fp32 ? N : 0;
        }
        if (!s_c_fp32) return false;
        C_fp32 = s_c_fp32;
        ldc_fp32 = N;
        if (beta != 0.0f) {
            const uint16_t *cb = static_cast<const uint16_t *>(dst);
            for (int n = 0; n < N; ++n) {
                uint32_t bits = static_cast<uint32_t>(cb[n]) << 16;
                std::memcpy(&C_fp32[n], &bits, sizeof(float));
            }
        }
    } else {
        C_fp32 = static_cast<float *>(dst);
        ldc_fp32 = ldc;
    }

    uint16_t *C_bf16_dst = dst_is_bf16
        ? static_cast<uint16_t *>(dst) : nullptr;

    bf16_brgemm_fn_t gemv_kernel = select_bf16_brgemm_kernel(1, NR);

    for (int jp = 0; jp < ppw->n_panels; ++jp) {
        const int col = jp * NR;
        const int nr_act = std::min(NR, N - col);
        const uint16_t *pb = ppw->get_panel(0, jp);
        const float *tb = (has_bias && can_fuse) ? (bias_f + col) : nullptr;
        uint16_t *dbf = can_direct_bf16 ? C_bf16_dst + col : nullptr;
        int dbf_ldc = can_direct_bf16 ? ldc : 0;

        if (nr_act == NR)
            gemv_kernel(A, desc.lda, pb, vnni_stride,
                C_fp32 + col, ldc_fp32, K, K, beta,
                tb, fused_op, dbf, dbf_ldc);
        else
            bf16_brgemm_tail_kernel(A, desc.lda, pb, vnni_stride,
                C_fp32 + col, ldc_fp32, K, K, 1, nr_act, beta,
                tb, fused_op, dbf, dbf_ldc);
    }

    // Apply non-fused post-ops (binary add/mul) on FP32 C buffer.
    // These run after the kernel because they need the full C result.
    if (has_remaining_postops && C_fp32) {
        apply_postops_tile(C_fp32, ldc_fp32, 1, N, 0, 0,
                           nullptr, remaining_postops);
    }

    // Convert FP32 → BF16 for non-direct paths
    if (dst_is_bf16 && !can_direct_bf16) {
        uint16_t *out = static_cast<uint16_t *>(dst);
        for (int n = 0; n < N; ++n) {
            float val = C_fp32[n];
            if (alpha != 1.0f) val *= alpha;
            if (has_bias && !can_fuse)
                val += bias_f ? bias_f[n] : 0.0f;
            uint32_t bits;
            std::memcpy(&bits, &val, sizeof(bits));
            out[n] = static_cast<uint16_t>(
                (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
        }
    }

    return true;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
