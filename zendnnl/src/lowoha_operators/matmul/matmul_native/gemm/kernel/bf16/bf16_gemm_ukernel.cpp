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

//
// BF16 GEMM microkernels — pure SIMD compute, zero allocations, zero OMP.
//
// Architecture: AVX-512 with BF16 VNNI (vdpbf16ps).
// All kernels operate on VNNI-interleaved prepacked B panels.
//
// Kernels:
//   - bf16_ukernel<MR,NV>     Template intrinsics kernel for all MR×NR tiles.
//   - bf16_tail_kernel         Dynamic MR/NR with masked ops for edge tiles.
//   - select_bf16_ukernel      Dispatch table: MR×NR → function pointer.
//
// ═══════════════════════════════════════════════════════════════════════════
// ZMM Register Pressure Analysis (AMD Zen 5: 32 ZMM registers)
// ═══════════════════════════════════════════════════════════════════════════
//
// Per-kernel register budget in the K-loop hot path:
//   Accumulators : MR × NV  ZMM registers (persistent across K iterations)
//   B loads      : NV       ZMM registers (loaded fresh each K-pair)
//   A broadcast  : 1        ZMM register  (loaded fresh each K-pair per M)
//   ─────────────────────────────────────────────────────────
//   Total        : MR×NV + NV + 1 ZMM
//
// Recommendation: keep total ≤ 26 (81%) to leave 6 free for compiler
// temporaries (beta scaling, bias loads, postop computation, BF16 convert).
// At 29 (91%), the compiler may spill 1-2 accumulators to stack.
//
// ┌──────┬──────────┬──────────┬──────────┬──────────────────────────────┐
// │  MR  │ NR=16    │ NR=32    │ NR=64    │ Notes                        │
// │      │ (NV=1)   │ (NV=2)   │ (NV=4)   │                              │
// ├──────┼──────────┼──────────┼──────────┼──────────────────────────────┤
// │   1  │  3 (9%)  │  5 (16%) │  9 (28%) │ UF=4, Citadel GEMV path      │
// │   2  │  5 (16%) │  7 (22%) │ 13 (41%) │ UF=4, decode M=2             │
// │   3  │  7 (22%) │ 10 (31%) │ 17 (53%) │ UF=4, decode M=3             │
// │   4  │  9 (28%) │ 13 (41%) │ 21 (66%) │ UF=4/4/2, decode M=4         │
// │   6  │ 13 (41%) │ 15 (47%) │ 29 (91%) │ UF=4/4/2, NR=64 tight        │
// │   8  │ 17 (53%) │ 19 (59%) │  INVALID │ UF=4/4, future large-M       │
// │  12  │ 14 (44%) │  INVALID │  INVALID │ UF=4, future large-M NR=16   │
// └──────┴──────────┴──────────┴──────────┴──────────────────────────────┘
//
// Unroll factor (UF) — number of K-pairs processed per loop iteration:
//   UF=4: total ≤ 20 regs → 12+ free for 4 unrolled B load sets
//   UF=2: total ≤ 28 regs → 4+ free for 2 unrolled B load sets
//   UF=1: total >  28 regs → minimal unroll to avoid spills
//
// Epilogue register usage (outside K-loop, reuses accumulator regs):
//   beta scaling : 1 ZMM (broadcast beta)
//   bias add     : NV ZMM (loaded per NV-group, reused across M rows)
//   fused postop : 2-4 ZMM temporaries (sigmoid/gelu need exp approx)
//   BF16 convert : 1 YMM per store (cvtneps_pbh, uses lower half of ZMM)
//
// ═══════════════════════════════════════════════════════════════════════════
//

#include "lowoha_operators/matmul/matmul_native/gemm/kernel/bf16/bf16_gemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// ============================================================================
// Template microkernel: MR x (NV*16) with VNNI dpbf16ps
//
// Adaptive K-loop unroll factor based on register pressure:
//   regs_used = MR*NV (accumulators) + NV (B loads) + 1 (A broadcast)
//   UF=4 when regs_used <= 20  (lots of headroom → max ILP)
//   UF=2 when regs_used <= 28  (tight but fits)
//   UF=1 when regs_used >  28  (near limit, no unroll)
// ============================================================================

template<int MR, int NV>
__attribute__((target("avx512f,avx512bf16,fma"), noinline))
void bf16_ukernel(
    const uint16_t *__restrict__ A, int lda,
    const uint16_t *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C, int ldc,
    int k, float beta,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16) {

    // K-loop live registers: MR*NV accumulators + NV B loads + 1 A broadcast.
    // B loads and A broadcast are reused each K-pair, not accumulated.
    // Epilogue needs ~4-6 temporaries (beta, bias, postop, BF16 convert).
    // Target: REGS ≤ 26 for comfortable UF=4, ≤ 28 for UF=2.
    static constexpr int REGS = MR * NV + NV + 1;
    static constexpr int UF = (REGS <= 20) ? 4 : (REGS <= 28) ? 2 : 1;

    __m512 acc[MR][NV];

    if (beta != 0.0f) {
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_loadu_ps(C + m * ldc + v * 16);
        if (beta != 1.0f) {
            __m512 vbeta = _mm512_set1_ps(beta);
            for (int m = 0; m < MR; ++m)
                for (int v = 0; v < NV; ++v)
                    acc[m][v] = _mm512_mul_ps(acc[m][v], vbeta);
        }
    } else {
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_setzero_ps();
    }

    const int k_pairs = k / 2;
    const int k_rem = k & 1;

    int kk = 0;
    for (; kk + UF - 1 < k_pairs; kk += UF) {
        for (int u = 0; u < UF; ++u) {
            const uint16_t *b_kp = B_vnni + (kk + u) * b_stride;
            __m512bh bv[NV];
            for (int v = 0; v < NV; ++v)
                bv[v] = (__m512bh)_mm512_loadu_si512(b_kp + v * 16 * VNNI_PAIR);
            for (int m = 0; m < MR; ++m) {
                uint32_t a_pair;
                std::memcpy(&a_pair, &A[m * lda + 2 * (kk + u)], sizeof(a_pair));
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(a_pair));
                for (int v = 0; v < NV; ++v)
                    acc[m][v] = _mm512_dpbf16_ps(acc[m][v], a_bf16, bv[v]);
            }
        }
    }
    for (; kk < k_pairs; ++kk) {
        const uint16_t *b_kp = B_vnni + kk * b_stride;
        __m512bh b_vec[NV];
        for (int v = 0; v < NV; ++v)
            b_vec[v] = (__m512bh)_mm512_loadu_si512(b_kp + v * 16 * VNNI_PAIR);
        for (int m = 0; m < MR; ++m) {
            uint32_t a_pair;
            std::memcpy(&a_pair, &A[m * lda + 2 * kk], sizeof(a_pair));
            __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_dpbf16_ps(acc[m][v], a_bf16, b_vec[v]);
        }
    }
    if (k_rem) {
        const uint16_t *b_kp = B_vnni + k_pairs * b_stride;
        __m512bh b_vec[NV];
        for (int v = 0; v < NV; ++v)
            b_vec[v] = (__m512bh)_mm512_loadu_si512(b_kp + v * 16 * VNNI_PAIR);
        for (int m = 0; m < MR; ++m) {
            uint32_t a_pair = static_cast<uint32_t>(A[m * lda + 2 * k_pairs]);
            __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_dpbf16_ps(acc[m][v], a_bf16, b_vec[v]);
        }
    }

    for (int m = 0; m < MR; ++m) {
        for (int v = 0; v < NV; ++v) {
            __m512 val = acc[m][v];
            if (bias) val = _mm512_add_ps(val, _mm512_loadu_ps(bias + v * 16));
            if (fused_op != fused_postop_t::none)
                val = apply_fused_postop(val, fused_op);
            if (C_bf16) {
                __m256bh bf = _mm512_cvtneps_pbh(val);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(C_bf16 + m * ldc_bf16 + v * 16),
                    (__m256i)bf);
            } else {
                _mm512_storeu_ps(C + m * ldc + v * 16, val);
            }
        }
    }
}

// Explicit instantiations
//
// MR × NV matrix — valid combinations respect 32-ZMM register limit.
// REGS = MR*NV + NV + 1.  Free = 32 - REGS (available for epilogue).
//
// ┌──────┬─────────────────┬─────────────────┬─────────────────┐
// │  MR  │ NV=1 (NR=16)    │ NV=2 (NR=32)    │ NV=4 (NR=64)    │
// │      │ REGS/Free/UF    │ REGS/Free/UF    │ REGS/Free/UF    │
// ├──────┼─────────────────┼─────────────────┼─────────────────┤
// │   1  │  3 / 29 / UF=4  │  5 / 27 / UF=4  │  9 / 23 / UF=4  │
// │   2  │  5 / 27 / UF=4  │  7 / 25 / UF=4  │ 13 / 19 / UF=4  │
// │   3  │  7 / 25 / UF=4  │ 10 / 22 / UF=4  │ 17 / 15 / UF=4  │
// │   4  │  9 / 23 / UF=4  │ 13 / 19 / UF=4  │ 21 / 11 / UF=2  │
// │   6  │ 13 / 19 / UF=4  │ 15 / 17 / UF=4  │ 29 /  3 / UF=2  │  ← tight
// │   8  │ 17 / 15 / UF=4  │ 19 / 13 / UF=4  │     INVALID      │
// │  12  │ 14 / 18 / UF=4  │     INVALID      │     INVALID      │
// └──────┴─────────────────┴─────────────────┴─────────────────┘
//
// INVALID = would exceed 32 ZMM registers (not instantiated).
// 6×4 (29 regs, 3 free) is the tightest valid kernel. The compiler
// may need to spill 1-2 regs during the epilogue (bias+postop),
// but the K-loop hot path fits without spills.

#define INST(MR, NV) \
    template void bf16_ukernel<MR,NV>( \
        const uint16_t*, int, const uint16_t*, int, float*, int, \
        int, float, const float*, fused_postop_t, uint16_t*, int);

INST(1,4) INST(2,4) INST(3,4) INST(4,4) INST(6,4)
INST(1,2) INST(2,2) INST(3,2) INST(4,2) INST(6,2) INST(8,2)
INST(1,1) INST(2,1) INST(3,1) INST(4,1) INST(6,1) INST(8,1) INST(12,1)
#undef INST

// ============================================================================
// Microkernel dispatch
//
// Selects the best kernel for given MR and NR, respecting register limits.
// For MR values without a matching NR=64 kernel, falls back to NR=32 or 16.
// ============================================================================
__attribute__((target("avx512f,avx512bf16,fma")))
bf16_ukernel_fn_t select_bf16_ukernel(int MR, int NR) {
    switch (NR) {
    case 64:
        switch (MR) {
        case 1:  return bf16_ukernel<1, 4>;
        case 2:  return bf16_ukernel<2, 4>;
        case 3:  return bf16_ukernel<3, 4>;
        case 4:  return bf16_ukernel<4, 4>;
        case 6:  return bf16_ukernel<6, 4>;
        }
        break;
    case 32:
        switch (MR) {
        case 1:  return bf16_ukernel<1, 2>;
        case 2:  return bf16_ukernel<2, 2>;
        case 3:  return bf16_ukernel<3, 2>;
        case 4:  return bf16_ukernel<4, 2>;
        case 6:  return bf16_ukernel<6, 2>;
        case 8:  return bf16_ukernel<8, 2>;
        }
        break;
    case 16:
        switch (MR) {
        case 1:  return bf16_ukernel<1, 1>;
        case 2:  return bf16_ukernel<2, 1>;
        case 3:  return bf16_ukernel<3, 1>;
        case 4:  return bf16_ukernel<4, 1>;
        case 6:  return bf16_ukernel<6, 1>;
        case 8:  return bf16_ukernel<8, 1>;
        case 12: return bf16_ukernel<12, 1>;
        }
        break;
    }
    return nullptr;
}

// ============================================================================
// Tail microkernel for edge tiles (dynamic MR/NR, masked operations)
// ============================================================================
__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,fma")))
void bf16_tail_kernel(
    const uint16_t *__restrict__ A, int lda,
    const uint16_t *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C, int ldc,
    int k, int mr_act, int nr_act, float beta,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16) {

    const int k_pairs = k / 2;
    const int k_rem = k & 1;
    const int nv_full = nr_act / 16;
    const int nr_tail = nr_act % 16;

    for (int m = 0; m < mr_act; ++m) {
        for (int v = 0; v < nv_full; ++v) {
            __m512 acc = (beta != 0.0f)
                ? _mm512_mul_ps(_mm512_loadu_ps(C + m * ldc + v * 16),
                                _mm512_set1_ps(beta))
                : _mm512_setzero_ps();

            for (int kk = 0; kk < k_pairs; ++kk) {
                uint32_t a_pair = static_cast<uint32_t>(A[m * lda + 2 * kk])
                    | (static_cast<uint32_t>(A[m * lda + 2 * kk + 1]) << 16);
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
                __m512bh b_bf16 = (__m512bh)_mm512_loadu_si512(
                    B_vnni + kk * b_stride + v * 16 * VNNI_PAIR);
                acc = _mm512_dpbf16_ps(acc, a_bf16, b_bf16);
            }
            if (k_rem) {
                uint32_t a_pair = static_cast<uint32_t>(A[m * lda + 2 * k_pairs]);
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
                __m512bh b_bf16 = (__m512bh)_mm512_loadu_si512(
                    B_vnni + k_pairs * b_stride + v * 16 * VNNI_PAIR);
                acc = _mm512_dpbf16_ps(acc, a_bf16, b_bf16);
            }
            if (bias) acc = _mm512_add_ps(acc, _mm512_loadu_ps(bias + v * 16));
            if (fused_op != fused_postop_t::none) acc = apply_fused_postop(acc, fused_op);

            if (C_bf16) {
                __m256bh bf = _mm512_cvtneps_pbh(acc);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(C_bf16 + m * ldc_bf16 + v * 16), (__m256i)bf);
            } else {
                _mm512_storeu_ps(C + m * ldc + v * 16, acc);
            }
        }

        if (nr_tail > 0) {
            __mmask16 mask = (1u << nr_tail) - 1;
            int col_off = nv_full * 16;
            __m512 acc = (beta != 0.0f)
                ? _mm512_mul_ps(
                    _mm512_maskz_loadu_ps(mask, C + m * ldc + col_off),
                    _mm512_set1_ps(beta))
                : _mm512_setzero_ps();

            for (int kk = 0; kk < k_pairs; ++kk) {
                uint32_t a_pair = static_cast<uint32_t>(A[m * lda + 2 * kk])
                    | (static_cast<uint32_t>(A[m * lda + 2 * kk + 1]) << 16);
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
                __m512bh b_bf16 = (__m512bh)_mm512_loadu_si512(
                    B_vnni + kk * b_stride + nv_full * 16 * VNNI_PAIR);
                acc = _mm512_dpbf16_ps(acc, a_bf16, b_bf16);
            }
            if (k_rem) {
                uint32_t a_pair = static_cast<uint32_t>(A[m * lda + 2 * k_pairs]);
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
                __m512bh b_bf16 = (__m512bh)_mm512_loadu_si512(
                    B_vnni + k_pairs * b_stride + nv_full * 16 * VNNI_PAIR);
                acc = _mm512_dpbf16_ps(acc, a_bf16, b_bf16);
            }
            if (bias) acc = _mm512_add_ps(acc, _mm512_maskz_loadu_ps(mask, bias + col_off));
            if (fused_op != fused_postop_t::none) acc = apply_fused_postop(acc, fused_op);

            if (C_bf16) {
                __m256bh bf = _mm512_cvtneps_pbh(acc);
                _mm256_mask_storeu_epi16(C_bf16 + m * ldc_bf16 + col_off, mask, (__m256i)bf);
            } else {
                _mm512_mask_storeu_ps(C + m * ldc + col_off, mask, acc);
            }
        }
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
