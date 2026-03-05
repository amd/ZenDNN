/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_operators/matmul/matmul_ai/gemm/intrinsic/bf16/avx512_bf16_gemm.hpp"
#include "lowoha_operators/matmul/matmul_ai/gemm/gemm_planner.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/postop.hpp"
#include "operators/matmul/matmul_config.hpp"
#include "common/zendnnl_global.hpp"
#include "common/bfloat16.hpp"

#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

using namespace zendnnl::error_handling;
using zendnnl::ops::matmul_config_t;
using zendnnl::ops::post_op_type_t;

// ============================================================================
// Scale tile: multiply all elements by alpha
// ============================================================================
__attribute__((target("avx512f")))
static void scale_tile(float *C, int ldc, int m_count, int n_count, float alpha) {
    __m512 av = _mm512_set1_ps(alpha);
    for (int m = 0; m < m_count; ++m) {
        float *row = C + m * ldc;
        int n = 0;
        for (; n + 15 < n_count; n += 16)
            _mm512_storeu_ps(row + n, _mm512_mul_ps(_mm512_loadu_ps(row + n), av));
        for (; n < n_count; ++n)
            row[n] *= alpha;
    }
}

// VNNI_PAIR, NR_PACK, BF16PrepackedWeight, BF16PrepackedWeightCache
// are defined in kernel_cache.hpp (shared with kernel_cache.cpp).

// ============================================================================
// Hand-scheduled BF16 6×64 microkernel (inline assembly K-loop)
//
// Register map (29 of 32 ZMM):
//   zmm0  - zmm23  : 24 FP32 accumulators (6 rows × 4 NV columns)
//   zmm24 - zmm27  : 4 B loads (VNNI-packed, 16 BF16 pairs each)
//   zmm28           : A broadcast (BF16 pair as 32-bit, replicated)
//   zmm29 - zmm31  : free for epilogue
//
// K-loop: 2× k-pair unrolled (4 BF16 elements per iteration).
// Per k-pair: 4 vmovups(B) + 6 vpbroadcastd(A) + 24 vdpbf16ps = 34 instr.
// 2 k-pairs = 68 compute + 7 ptr advances + 1 branch = 76 instructions.
// dpbf16-bound: 48 dpbf16 / 2 ports = 24 cycles + 1 = 25 cycles.
// 1536 FLOPs / 25 cycles = 61.4 FLOPs/cycle (96% of peak).
// ============================================================================
__attribute__((target("avx512f,avx512bf16,fma"), noinline))
static void bf16_ukernel_6x64_asm(
    const uint16_t *__restrict__ A, int lda,
    const uint16_t *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C, int ldc,
    int k, float beta,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16) {

    // A row pointers (advance by 4 bytes = 1 BF16 pair per k-pair)
    const uint16_t *a0 = A;
    const uint16_t *a1 = A + lda;
    const uint16_t *a2 = A + 2 * lda;
    const uint16_t *a3 = A + 3 * lda;
    const uint16_t *a4 = A + 4 * lda;
    const uint16_t *a5 = A + 5 * lda;
    const uint16_t *bp = B_vnni;
    // VNNI stride in bytes: NR_PACK * VNNI_PAIR * sizeof(uint16_t) = 256
    const long bs = static_cast<long>(b_stride) * static_cast<long>(sizeof(uint16_t));

    // Stack buffer: 24 accumulators × 64 bytes = 1536 bytes (L1-resident)
    __m512 c_acc[24] __attribute__((aligned(64)));

    const int k_pairs = k / 2;
    int k2 = k_pairs >> 1;  // 2x k-pair unrolled iterations

    __asm__ __volatile__ (
        // Zero 24 accumulators
        "vpxord %%zmm0,  %%zmm0,  %%zmm0\n\t"  "vpxord %%zmm1,  %%zmm1,  %%zmm1\n\t"
        "vpxord %%zmm2,  %%zmm2,  %%zmm2\n\t"  "vpxord %%zmm3,  %%zmm3,  %%zmm3\n\t"
        "vpxord %%zmm4,  %%zmm4,  %%zmm4\n\t"  "vpxord %%zmm5,  %%zmm5,  %%zmm5\n\t"
        "vpxord %%zmm6,  %%zmm6,  %%zmm6\n\t"  "vpxord %%zmm7,  %%zmm7,  %%zmm7\n\t"
        "vpxord %%zmm8,  %%zmm8,  %%zmm8\n\t"  "vpxord %%zmm9,  %%zmm9,  %%zmm9\n\t"
        "vpxord %%zmm10, %%zmm10, %%zmm10\n\t" "vpxord %%zmm11, %%zmm11, %%zmm11\n\t"
        "vpxord %%zmm12, %%zmm12, %%zmm12\n\t" "vpxord %%zmm13, %%zmm13, %%zmm13\n\t"
        "vpxord %%zmm14, %%zmm14, %%zmm14\n\t" "vpxord %%zmm15, %%zmm15, %%zmm15\n\t"
        "vpxord %%zmm16, %%zmm16, %%zmm16\n\t" "vpxord %%zmm17, %%zmm17, %%zmm17\n\t"
        "vpxord %%zmm18, %%zmm18, %%zmm18\n\t" "vpxord %%zmm19, %%zmm19, %%zmm19\n\t"
        "vpxord %%zmm20, %%zmm20, %%zmm20\n\t" "vpxord %%zmm21, %%zmm21, %%zmm21\n\t"
        "vpxord %%zmm22, %%zmm22, %%zmm22\n\t" "vpxord %%zmm23, %%zmm23, %%zmm23\n\t"

        "testl %[k2], %[k2]\n\t"
        "jle 2f\n\t"

        // ── 2× k-pair unrolled main loop ──
        ".p2align 5\n"
        "1:\n\t"
        // ── k-pair 0: B from [bp], A from [a*+0] ──
        "vmovups     (%[bp]),    %%zmm24\n\t"
        "vmovups   64(%[bp]),    %%zmm25\n\t"
        "vmovups  128(%[bp]),    %%zmm26\n\t"
        "vmovups  192(%[bp]),    %%zmm27\n\t"
        "vpbroadcastd  (%[a0]),  %%zmm28\n\t"
        "vdpbf16ps %%zmm24, %%zmm28, %%zmm0\n\t"  "vdpbf16ps %%zmm25, %%zmm28, %%zmm1\n\t"
        "vdpbf16ps %%zmm26, %%zmm28, %%zmm2\n\t"  "vdpbf16ps %%zmm27, %%zmm28, %%zmm3\n\t"
        "vpbroadcastd  (%[a1]),  %%zmm28\n\t"
        "vdpbf16ps %%zmm24, %%zmm28, %%zmm4\n\t"  "vdpbf16ps %%zmm25, %%zmm28, %%zmm5\n\t"
        "vdpbf16ps %%zmm26, %%zmm28, %%zmm6\n\t"  "vdpbf16ps %%zmm27, %%zmm28, %%zmm7\n\t"
        "vpbroadcastd  (%[a2]),  %%zmm28\n\t"
        "vdpbf16ps %%zmm24, %%zmm28, %%zmm8\n\t"  "vdpbf16ps %%zmm25, %%zmm28, %%zmm9\n\t"
        "vdpbf16ps %%zmm26, %%zmm28, %%zmm10\n\t" "vdpbf16ps %%zmm27, %%zmm28, %%zmm11\n\t"
        "vpbroadcastd  (%[a3]),  %%zmm28\n\t"
        "vdpbf16ps %%zmm24, %%zmm28, %%zmm12\n\t" "vdpbf16ps %%zmm25, %%zmm28, %%zmm13\n\t"
        "vdpbf16ps %%zmm26, %%zmm28, %%zmm14\n\t" "vdpbf16ps %%zmm27, %%zmm28, %%zmm15\n\t"
        "vpbroadcastd  (%[a4]),  %%zmm28\n\t"
        "vdpbf16ps %%zmm24, %%zmm28, %%zmm16\n\t" "vdpbf16ps %%zmm25, %%zmm28, %%zmm17\n\t"
        "vdpbf16ps %%zmm26, %%zmm28, %%zmm18\n\t" "vdpbf16ps %%zmm27, %%zmm28, %%zmm19\n\t"
        "vpbroadcastd  (%[a5]),  %%zmm28\n\t"
        "vdpbf16ps %%zmm24, %%zmm28, %%zmm20\n\t" "vdpbf16ps %%zmm25, %%zmm28, %%zmm21\n\t"
        "vdpbf16ps %%zmm26, %%zmm28, %%zmm22\n\t" "vdpbf16ps %%zmm27, %%zmm28, %%zmm23\n\t"

        // ── k-pair 1: B from [bp+bs], A from [a*+4] ──
        "vmovups     (%[bp],%[bs],1),    %%zmm24\n\t"
        "vmovups   64(%[bp],%[bs],1),    %%zmm25\n\t"
        "vmovups  128(%[bp],%[bs],1),    %%zmm26\n\t"
        "vmovups  192(%[bp],%[bs],1),    %%zmm27\n\t"
        "vpbroadcastd 4(%[a0]),  %%zmm28\n\t"
        "vdpbf16ps %%zmm24, %%zmm28, %%zmm0\n\t"  "vdpbf16ps %%zmm25, %%zmm28, %%zmm1\n\t"
        "vdpbf16ps %%zmm26, %%zmm28, %%zmm2\n\t"  "vdpbf16ps %%zmm27, %%zmm28, %%zmm3\n\t"
        "vpbroadcastd 4(%[a1]),  %%zmm28\n\t"
        "vdpbf16ps %%zmm24, %%zmm28, %%zmm4\n\t"  "vdpbf16ps %%zmm25, %%zmm28, %%zmm5\n\t"
        "vdpbf16ps %%zmm26, %%zmm28, %%zmm6\n\t"  "vdpbf16ps %%zmm27, %%zmm28, %%zmm7\n\t"
        "vpbroadcastd 4(%[a2]),  %%zmm28\n\t"
        "vdpbf16ps %%zmm24, %%zmm28, %%zmm8\n\t"  "vdpbf16ps %%zmm25, %%zmm28, %%zmm9\n\t"
        "vdpbf16ps %%zmm26, %%zmm28, %%zmm10\n\t" "vdpbf16ps %%zmm27, %%zmm28, %%zmm11\n\t"
        "vpbroadcastd 4(%[a3]),  %%zmm28\n\t"
        "vdpbf16ps %%zmm24, %%zmm28, %%zmm12\n\t" "vdpbf16ps %%zmm25, %%zmm28, %%zmm13\n\t"
        "vdpbf16ps %%zmm26, %%zmm28, %%zmm14\n\t" "vdpbf16ps %%zmm27, %%zmm28, %%zmm15\n\t"
        "vpbroadcastd 4(%[a4]),  %%zmm28\n\t"
        "vdpbf16ps %%zmm24, %%zmm28, %%zmm16\n\t" "vdpbf16ps %%zmm25, %%zmm28, %%zmm17\n\t"
        "vdpbf16ps %%zmm26, %%zmm28, %%zmm18\n\t" "vdpbf16ps %%zmm27, %%zmm28, %%zmm19\n\t"
        "vpbroadcastd 4(%[a5]),  %%zmm28\n\t"
        "vdpbf16ps %%zmm24, %%zmm28, %%zmm20\n\t" "vdpbf16ps %%zmm25, %%zmm28, %%zmm21\n\t"
        "vdpbf16ps %%zmm26, %%zmm28, %%zmm22\n\t" "vdpbf16ps %%zmm27, %%zmm28, %%zmm23\n\t"

        // Advance: A by 8 bytes (2 k-pairs × 4 bytes), B by 2×bs
        "addq $8, %[a0]\n\t"  "addq $8, %[a1]\n\t"  "addq $8, %[a2]\n\t"
        "addq $8, %[a3]\n\t"  "addq $8, %[a4]\n\t"  "addq $8, %[a5]\n\t"
        "leaq (%[bp],%[bs],2), %[bp]\n\t"

        "decl %[k2]\n\t"
        "jnz 1b\n\t"

        // ── Store 24 accumulators to stack buffer ──
        "2:\n\t"
        "vmovaps %%zmm0,      (%[ca])\n\t"  "vmovaps %%zmm1,    64(%[ca])\n\t"
        "vmovaps %%zmm2,   128(%[ca])\n\t"  "vmovaps %%zmm3,   192(%[ca])\n\t"
        "vmovaps %%zmm4,   256(%[ca])\n\t"  "vmovaps %%zmm5,   320(%[ca])\n\t"
        "vmovaps %%zmm6,   384(%[ca])\n\t"  "vmovaps %%zmm7,   448(%[ca])\n\t"
        "vmovaps %%zmm8,   512(%[ca])\n\t"  "vmovaps %%zmm9,   576(%[ca])\n\t"
        "vmovaps %%zmm10,  640(%[ca])\n\t"  "vmovaps %%zmm11,  704(%[ca])\n\t"
        "vmovaps %%zmm12,  768(%[ca])\n\t"  "vmovaps %%zmm13,  832(%[ca])\n\t"
        "vmovaps %%zmm14,  896(%[ca])\n\t"  "vmovaps %%zmm15,  960(%[ca])\n\t"
        "vmovaps %%zmm16, 1024(%[ca])\n\t"  "vmovaps %%zmm17, 1088(%[ca])\n\t"
        "vmovaps %%zmm18, 1152(%[ca])\n\t"  "vmovaps %%zmm19, 1216(%[ca])\n\t"
        "vmovaps %%zmm20, 1280(%[ca])\n\t"  "vmovaps %%zmm21, 1344(%[ca])\n\t"
        "vmovaps %%zmm22, 1408(%[ca])\n\t"  "vmovaps %%zmm23, 1472(%[ca])\n\t"

        : [a0]"+r"(a0), [a1]"+r"(a1), [a2]"+r"(a2),
          [a3]"+r"(a3), [a4]"+r"(a4), [a5]"+r"(a5),
          [bp]"+r"(bp), [k2]"+r"(k2)
        : [bs]"r"(bs), [ca]"r"(c_acc)
        : "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7",
          "zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",
          "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23",
          "zmm24","zmm25","zmm26","zmm27","zmm28", "memory"
    );

    // Load accumulators from stack buffer
    __m512 c00=c_acc[0],  c01=c_acc[1],  c02=c_acc[2],  c03=c_acc[3];
    __m512 c10=c_acc[4],  c11=c_acc[5],  c12=c_acc[6],  c13=c_acc[7];
    __m512 c20=c_acc[8],  c21=c_acc[9],  c22=c_acc[10], c23=c_acc[11];
    __m512 c30=c_acc[12], c31=c_acc[13], c32=c_acc[14], c33=c_acc[15];
    __m512 c40=c_acc[16], c41=c_acc[17], c42=c_acc[18], c43=c_acc[19];
    __m512 c50=c_acc[20], c51=c_acc[21], c52=c_acc[22], c53=c_acc[23];

    // Remainder: odd k-pair (if k_pairs is odd) + odd k element
    {
        int kk_done = (k_pairs >> 1) * 2;  // k-pairs processed by asm
        for (int kk = kk_done; kk < k_pairs; ++kk) {
            __m512bh b0 = (__m512bh)_mm512_loadu_si512(bp);
            __m512bh b1 = (__m512bh)_mm512_loadu_si512(bp + 16 * VNNI_PAIR);
            __m512bh b2 = (__m512bh)_mm512_loadu_si512(bp + 32 * VNNI_PAIR);
            __m512bh b3 = (__m512bh)_mm512_loadu_si512(bp + 48 * VNNI_PAIR);
            uint32_t ap; __m512bh av;
#define DO_ROW_4(R, row) \
            std::memcpy(&ap, &a##R[0], 4); av = (__m512bh)_mm512_set1_epi32((int)ap); \
            c##R##0 = _mm512_dpbf16_ps(c##R##0, av, b0); c##R##1 = _mm512_dpbf16_ps(c##R##1, av, b1); \
            c##R##2 = _mm512_dpbf16_ps(c##R##2, av, b2); c##R##3 = _mm512_dpbf16_ps(c##R##3, av, b3);
            DO_ROW_4(0, 0) DO_ROW_4(1, 1) DO_ROW_4(2, 2)
            DO_ROW_4(3, 3) DO_ROW_4(4, 4) DO_ROW_4(5, 5)
#undef DO_ROW_4
            a0 += VNNI_PAIR; a1 += VNNI_PAIR; a2 += VNNI_PAIR;
            a3 += VNNI_PAIR; a4 += VNNI_PAIR; a5 += VNNI_PAIR;
            bp += b_stride;
        }
        // Handle odd K (single BF16 element, zero-padded second slot)
        if (k & 1) {
            __m512bh b0 = (__m512bh)_mm512_loadu_si512(bp);
            __m512bh b1 = (__m512bh)_mm512_loadu_si512(bp + 16 * VNNI_PAIR);
            __m512bh b2 = (__m512bh)_mm512_loadu_si512(bp + 32 * VNNI_PAIR);
            __m512bh b3 = (__m512bh)_mm512_loadu_si512(bp + 48 * VNNI_PAIR);
            uint32_t ap; __m512bh av;
#define DO_ROW_REM(R) \
            ap = static_cast<uint32_t>(a##R[0]); av = (__m512bh)_mm512_set1_epi32((int)ap); \
            c##R##0 = _mm512_dpbf16_ps(c##R##0, av, b0); c##R##1 = _mm512_dpbf16_ps(c##R##1, av, b1); \
            c##R##2 = _mm512_dpbf16_ps(c##R##2, av, b2); c##R##3 = _mm512_dpbf16_ps(c##R##3, av, b3);
            DO_ROW_REM(0) DO_ROW_REM(1) DO_ROW_REM(2)
            DO_ROW_REM(3) DO_ROW_REM(4) DO_ROW_REM(5)
#undef DO_ROW_REM
        }
    }

    // ── Epilogue ──
    // Beta accumulation
    if (beta != 0.0f) {
        __m512 bv = _mm512_set1_ps(beta);
#define BETA_ROW(R, off) \
        c##R##0=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+(off)),c##R##0); \
        c##R##1=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+(off)+16),c##R##1); \
        c##R##2=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+(off)+32),c##R##2); \
        c##R##3=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+(off)+48),c##R##3);
        BETA_ROW(0, 0) BETA_ROW(1, ldc) BETA_ROW(2, 2*ldc)
        BETA_ROW(3, 3*ldc) BETA_ROW(4, 4*ldc) BETA_ROW(5, 5*ldc)
#undef BETA_ROW
    }
    // Bias
    if (bias) {
        __m512 bv0=_mm512_loadu_ps(bias), bv1=_mm512_loadu_ps(bias+16),
               bv2=_mm512_loadu_ps(bias+32), bv3=_mm512_loadu_ps(bias+48);
#define BIAS_ROW(R) \
        c##R##0=_mm512_add_ps(c##R##0,bv0); c##R##1=_mm512_add_ps(c##R##1,bv1); \
        c##R##2=_mm512_add_ps(c##R##2,bv2); c##R##3=_mm512_add_ps(c##R##3,bv3);
        BIAS_ROW(0) BIAS_ROW(1) BIAS_ROW(2) BIAS_ROW(3) BIAS_ROW(4) BIAS_ROW(5)
#undef BIAS_ROW
    }
    // Fused activation
    if (fused_op != fused_postop_t::none) {
#define ACT_ROW(R) \
        c##R##0=apply_fused_postop(c##R##0,fused_op); c##R##1=apply_fused_postop(c##R##1,fused_op); \
        c##R##2=apply_fused_postop(c##R##2,fused_op); c##R##3=apply_fused_postop(c##R##3,fused_op);
        ACT_ROW(0) ACT_ROW(1) ACT_ROW(2) ACT_ROW(3) ACT_ROW(4) ACT_ROW(5)
#undef ACT_ROW
    }
    // Store: BF16 direct or FP32
    if (C_bf16) {
#define STORE_BF16_ROW(R, off) { \
        _mm256_storeu_si256((__m256i*)(C_bf16+(off)),   (__m256i)_mm512_cvtneps_pbh(c##R##0)); \
        _mm256_storeu_si256((__m256i*)(C_bf16+(off)+16),(__m256i)_mm512_cvtneps_pbh(c##R##1)); \
        _mm256_storeu_si256((__m256i*)(C_bf16+(off)+32),(__m256i)_mm512_cvtneps_pbh(c##R##2)); \
        _mm256_storeu_si256((__m256i*)(C_bf16+(off)+48),(__m256i)_mm512_cvtneps_pbh(c##R##3)); }
        STORE_BF16_ROW(0, 0) STORE_BF16_ROW(1, ldc_bf16)
        STORE_BF16_ROW(2, 2*ldc_bf16) STORE_BF16_ROW(3, 3*ldc_bf16)
        STORE_BF16_ROW(4, 4*ldc_bf16) STORE_BF16_ROW(5, 5*ldc_bf16)
#undef STORE_BF16_ROW
    } else {
#define STORE_FP32_ROW(R, off) \
        _mm512_storeu_ps(C+(off),c##R##0);    _mm512_storeu_ps(C+(off)+16,c##R##1); \
        _mm512_storeu_ps(C+(off)+32,c##R##2); _mm512_storeu_ps(C+(off)+48,c##R##3);
        STORE_FP32_ROW(0, 0) STORE_FP32_ROW(1, ldc) STORE_FP32_ROW(2, 2*ldc)
        STORE_FP32_ROW(3, 3*ldc) STORE_FP32_ROW(4, 4*ldc) STORE_FP32_ROW(5, 5*ldc)
#undef STORE_FP32_ROW
    }
}

// ============================================================================
// BF16 A micro-panel packing: row-major A → contiguous MR×KB BF16 blocks
// ============================================================================
static void pack_a_bf16_block(
    const uint16_t *A_src, uint16_t *pack_buf,
    int ic, int pc, int M, int K, int lda,
    int mb, int kb, int MR) {

    const int m_actual = std::min(mb, M - ic);
    const int k_actual = std::min(kb, K - pc);
    const int m_panels = (m_actual + MR - 1) / MR;
    const uint16_t *A_block = A_src + ic * lda + pc;

    for (int ip = 0; ip < m_panels; ++ip) {
        const int i0 = ip * MR;
        const int mr = std::min(MR, m_actual - i0);
        uint16_t *dst = pack_buf + ip * MR * k_actual;
        for (int m = 0; m < mr; ++m)
            std::memcpy(dst + m * k_actual,
                        A_block + (i0 + m) * lda,
                        k_actual * sizeof(uint16_t));
        for (int m = mr; m < MR; ++m)
            std::memset(dst + m * k_actual, 0, k_actual * sizeof(uint16_t));
    }
}

// ============================================================================
// BF16 AVX-512 microkernel: MR x (NV*16) with VNNI dpbf16ps
//
//   A:           BF16 source tile, row-major, lda stride (in BF16 elements)
//   B_vnni:      VNNI-packed BF16 panel, stride = NR_PACK * 2 BF16 per k-pair
//   C:           FP32 output tile (accumulation always in FP32)
//   k:           K-dimension for this block
//   beta:        0.0 or 1.0 (accumulate into C)
//   bias:        FP32 bias pointer (nullptr if no bias or not final K-block)
//   fused_op:    fused post-op to apply in epilogue
//
// K-loop processes 2 elements at a time via _mm512_dpbf16_ps.
// Handles odd K by zero-padding the last pair.
// ============================================================================
template<int MR, int NV>
__attribute__((target("avx512f,avx512bf16,fma")))
__attribute__((noinline))
static void bf16_ukernel(
    const uint16_t *__restrict__ A, int lda,
    const uint16_t *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C, int ldc,
    int k, float beta,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16) {

    // b_stride is in uint16_t units: NR_PACK * VNNI_PAIR per k-pair
    // Each k-pair row: b_stride uint16_t values

    // Accumulators: MR rows x NV ZMM registers (each 16 FP32)
    __m512 acc[MR][NV];

    // Initialize accumulators
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

    // Main K-loop: 2x k-pair unrolled for better ILP and branch amortization.
    // Each k-pair processes 2 BF16 elements via dpbf16ps.
    // 2x unroll = 4 BF16 elements per iteration, reducing branch overhead.
    const int k_pairs = k / 2;
    const int k_rem = k & 1;

    int kk = 0;
    for (; kk + 1 < k_pairs; kk += 2) {
        // ── k-pair 0 ──
        const uint16_t *b_kp0 = B_vnni + kk * b_stride;
        __m512bh bv0[NV];
        for (int v = 0; v < NV; ++v)
            bv0[v] = (__m512bh)_mm512_loadu_si512(b_kp0 + v * 16 * VNNI_PAIR);

        for (int m = 0; m < MR; ++m) {
            uint32_t a_pair;
            std::memcpy(&a_pair, &A[m * lda + 2 * kk], sizeof(a_pair));
            __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_dpbf16_ps(acc[m][v], a_bf16, bv0[v]);
        }

        // ── k-pair 1 ──
        const uint16_t *b_kp1 = B_vnni + (kk + 1) * b_stride;
        __m512bh bv1[NV];
        for (int v = 0; v < NV; ++v)
            bv1[v] = (__m512bh)_mm512_loadu_si512(b_kp1 + v * 16 * VNNI_PAIR);

        for (int m = 0; m < MR; ++m) {
            uint32_t a_pair;
            std::memcpy(&a_pair, &A[m * lda + 2 * (kk + 1)], sizeof(a_pair));
            __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(static_cast<int>(a_pair));
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_dpbf16_ps(acc[m][v], a_bf16, bv1[v]);
        }
    }
    // Remainder k-pair (if k_pairs is odd)
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

    // Handle odd K remainder (single BF16 element, zero-padded second slot)
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

    // Epilogue: bias + fused activation + store
    for (int m = 0; m < MR; ++m) {
        for (int v = 0; v < NV; ++v) {
            __m512 val = acc[m][v];

            // Add bias (FP32)
            if (bias) {
                val = _mm512_add_ps(val, _mm512_loadu_ps(bias + v * 16));
            }

            // Fused activation
            if (fused_op != fused_postop_t::none)
                val = apply_fused_postop(val, fused_op);

            // Store: fused FP32→BF16 conversion or FP32 accumulator
            if (C_bf16) {
                __m256bh bf = _mm512_cvtneps_pbh(val);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(
                        C_bf16 + m * ldc_bf16 + v * 16),
                    (__m256i)bf);
            } else {
                _mm512_storeu_ps(C + m * ldc + v * 16, val);
            }
        }
    }
}

// ============================================================================
// Explicit instantiations — ONLY active kernels
//
// Decode (M<=4, LLM decode):  MR=M, NR=64 (NV=4)  — 4..16 acc, max N-throughput
// GEMM no activation:         MR=6, NR=48 (NV=3)   — 18 acc, max GEMM throughput
// GEMM with activation:       MR=6, NR=32 (NV=2)   — 12 acc, safe epilogue
// Small N fallback:           MR=6, NR=16 (NV=1)   — 6 acc, minimal
// ============================================================================

// Decode: MR=1..4 with all NR widths (NV=1,2,4) to cover any N
template void bf16_ukernel<1,4>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<2,4>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<3,4>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<4,4>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<1,2>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<2,2>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<3,2>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<4,2>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<1,1>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<2,1>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<3,1>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<4,1>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
// GEMM high-throughput: MR=6, NV=4 (NR=64) — 24 acc, relu/no-act only
// NR=NR_PACK=64: zero panel crossing, perfect alignment
template void bf16_ukernel<6,4>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
// GEMM safe: MR=6, NV=2 (NR=32) — 12 acc, any postop
template void bf16_ukernel<6,2>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
// Fallback: MR=6, NV=1 (NR=16) — 64/16=4, zero panel crossing
template void bf16_ukernel<6,1>(const uint16_t*, int, const uint16_t*, int,
    float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);

// Function pointer type for BF16 microkernel dispatch
using bf16_ukernel_fn_t = void (*)(const uint16_t *, int, const uint16_t *, int,
                                   float *, int, int, float, const float *,
                                   fused_postop_t, uint16_t *, int);

/// Select BF16 microkernel based on MR and NR from planner.
/// All active NR values (16, 32, 64) divide NR_PACK=64 evenly → zero panel crossings.
static bf16_ukernel_fn_t select_bf16_ukernel(int MR, int NR) {
    switch (NR) {
    case 64:
        switch (MR) {
        case 1:  return bf16_ukernel<1, 4>;
        case 2:  return bf16_ukernel<2, 4>;
        case 3:  return bf16_ukernel<3, 4>;
        case 4:  return bf16_ukernel<4, 4>;
        case 6:  return bf16_ukernel_6x64_asm;
        }
        break;
    case 32:
        switch (MR) {
        case 1:  return bf16_ukernel<1, 2>;
        case 2:  return bf16_ukernel<2, 2>;
        case 3:  return bf16_ukernel<3, 2>;
        case 4:  return bf16_ukernel<4, 2>;
        case 6:  return bf16_ukernel<6, 2>;
        }
        break;
    case 16:
        switch (MR) {
        case 1:  return bf16_ukernel<1, 1>;
        case 2:  return bf16_ukernel<2, 1>;
        case 3:  return bf16_ukernel<3, 1>;
        case 4:  return bf16_ukernel<4, 1>;
        case 6:  return bf16_ukernel<6, 1>;
        }
        break;
    }
    return bf16_ukernel<6, 1>;
}

// ============================================================================
// BF16 tail microkernel for edge tiles (dynamic MR/NR)
// ============================================================================
__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,fma")))
static void bf16_tail_kernel(
    const uint16_t *__restrict__ A, int lda,
    const uint16_t *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C, int ldc,
    int k, int mr_act, int nr_act, float beta,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16) {

    // Use masked operations for partial tiles
    const int k_pairs = k / 2;
    const int k_rem = k & 1;

    // Process column groups of 16
    const int nv_full = nr_act / 16;
    const int nr_tail = nr_act % 16;

    for (int m = 0; m < mr_act; ++m) {
        // Full 16-wide column groups
        for (int v = 0; v < nv_full; ++v) {
            __m512 acc = (beta != 0.0f)
                ? _mm512_mul_ps(_mm512_loadu_ps(C + m * ldc + v * 16),
                                _mm512_set1_ps(beta))
                : _mm512_setzero_ps();

            for (int kk = 0; kk < k_pairs; ++kk) {
                uint32_t a_pair = static_cast<uint32_t>(A[m * lda + 2 * kk])
                    | (static_cast<uint32_t>(A[m * lda + 2 * kk + 1]) << 16);
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(a_pair));
                __m512bh b_bf16 = (__m512bh)_mm512_loadu_si512(
                    B_vnni + kk * b_stride + v * 16 * VNNI_PAIR);
                acc = _mm512_dpbf16_ps(acc, a_bf16, b_bf16);
            }
            if (k_rem) {
                uint32_t a_pair = static_cast<uint32_t>(
                    A[m * lda + 2 * k_pairs]);
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(a_pair));
                __m512bh b_bf16 = (__m512bh)_mm512_loadu_si512(
                    B_vnni + k_pairs * b_stride + v * 16 * VNNI_PAIR);
                acc = _mm512_dpbf16_ps(acc, a_bf16, b_bf16);
            }

            if (bias) acc = _mm512_add_ps(acc,
                _mm512_loadu_ps(bias + v * 16));
            if (fused_op != fused_postop_t::none)
                acc = apply_fused_postop(acc, fused_op);

            if (C_bf16) {
                __m256bh bf = _mm512_cvtneps_pbh(acc);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(
                        C_bf16 + m * ldc_bf16 + v * 16),
                    (__m256i)bf);
            } else {
                _mm512_storeu_ps(C + m * ldc + v * 16, acc);
            }
        }

        // Tail columns (masked)
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
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(a_pair));
                __m512bh b_bf16 = (__m512bh)_mm512_loadu_si512(
                    B_vnni + kk * b_stride + nv_full * 16 * VNNI_PAIR);
                acc = _mm512_dpbf16_ps(acc, a_bf16, b_bf16);
            }
            if (k_rem) {
                uint32_t a_pair = static_cast<uint32_t>(
                    A[m * lda + 2 * k_pairs]);
                __m512bh a_bf16 = (__m512bh)_mm512_set1_epi32(
                    static_cast<int>(a_pair));
                __m512bh b_bf16 = (__m512bh)_mm512_loadu_si512(
                    B_vnni + k_pairs * b_stride + nv_full * 16 * VNNI_PAIR);
                acc = _mm512_dpbf16_ps(acc, a_bf16, b_bf16);
            }

            if (bias) acc = _mm512_add_ps(acc,
                _mm512_maskz_loadu_ps(mask, bias + col_off));
            if (fused_op != fused_postop_t::none)
                acc = apply_fused_postop(acc, fused_op);

            if (C_bf16) {
                __m256bh bf = _mm512_cvtneps_pbh(acc);
                _mm256_mask_storeu_epi16(
                    C_bf16 + m * ldc_bf16 + col_off, mask, (__m256i)bf);
            } else {
                _mm512_mask_storeu_ps(C + m * ldc + col_off, mask, acc);
            }
        }
    }
}

// ============================================================================
// FP32-to-BF16 output conversion for a tile
// ============================================================================
__attribute__((target("avx512f,avx512bf16")))
static void convert_fp32_to_bf16_tile(
    const float *src_fp32, int ldc_fp32,
    uint16_t *dst_bf16, int ldc_bf16,
    int rows, int cols) {

    for (int m = 0; m < rows; ++m) {
        int n = 0;
        // Process 16 FP32 values → 16 BF16 values at a time
        for (; n + 16 <= cols; n += 16) {
            __m512 v = _mm512_loadu_ps(src_fp32 + m * ldc_fp32 + n);
            __m256bh bf = _mm512_cvtneps_pbh(v);
            _mm256_storeu_si256(
                reinterpret_cast<__m256i *>(dst_bf16 + m * ldc_bf16 + n),
                (__m256i)bf);
        }
        // Scalar tail
        for (; n < cols; ++n) {
            // Round-to-nearest-even to match vectorized _mm512_cvtneps_pbh
            float val = src_fp32[m * ldc_fp32 + n];
            uint32_t bits;
            std::memcpy(&bits, &val, sizeof(bits));
            uint32_t lsb = (bits >> 16) & 1;
            uint32_t rounding_bias = 0x7FFF + lsb;
            bits += rounding_bias;
            dst_bf16[m * ldc_bf16 + n] = static_cast<uint16_t>(bits >> 16);
        }
    }
}

// ============================================================================
// VNNI B prepacking: pack BF16 B matrix into VNNI format
//
// Input:  B in row-major BF16, B[k][n] at offset k * ldb + n
//         (or transposed: B_logical[k][n] = B_storage[n * ldb + k])
// Output: VNNI panels of NR_PACK width, k-pairs interleaved:
//         panel[kp][n] = { B[2*kp][n], B[2*kp+1][n] } as uint16_t pair
//
// K is rounded up to even (zero-padded if odd).
// ============================================================================
__attribute__((target("avx512f,avx512bw")))
[[maybe_unused]] static void pack_b_vnni(
    const uint16_t *B, int K, int N, int ldb, bool transB,
    uint16_t *packed, int K_padded) {

    const int np = (N + NR_PACK - 1) / NR_PACK;
    const int k_pairs = K_padded / 2;
    const int vnni_stride = NR_PACK * VNNI_PAIR;

    for (int jp = 0; jp < np; ++jp) {
        const int j0 = jp * NR_PACK;
        const int nr_act = std::min(NR_PACK, N - j0);
        uint16_t *dst = packed
            + static_cast<size_t>(jp) * k_pairs * vnni_stride;

        if (!transB && nr_act == NR_PACK) {
            // AVX-512 fast path: row-major B, full NR_PACK=64 panel.
            // Interleave two consecutive K rows into VNNI pairs using
            // unpacklo/hi to produce {b0[n], b1[n]} pairs 32 elements
            // at a time.
            for (int kp = 0; kp < k_pairs; ++kp) {
                uint16_t *d = dst + kp * vnni_stride;
                const int k0 = kp * 2;
                const int k1 = k0 + 1;
                const uint16_t *row0 = (k0 < K) ? B + k0 * ldb + j0 : nullptr;
                const uint16_t *row1 = (k1 < K) ? B + k1 * ldb + j0 : nullptr;

                for (int n = 0; n < NR_PACK; n += 32) {
                    __m512i r0 = row0 ? _mm512_loadu_si512(row0 + n)
                                      : _mm512_setzero_si512();
                    __m512i r1 = row1 ? _mm512_loadu_si512(row1 + n)
                                      : _mm512_setzero_si512();
                    __m512i lo = _mm512_unpacklo_epi16(r0, r1);
                    __m512i hi = _mm512_unpackhi_epi16(r0, r1);
                    // unpacklo/hi work per 128-bit lane; need to fix lane order
                    // After unpack: lane0 has pairs 0-3,16-19; lane1 has 4-7,20-23; etc.
                    // Rearrange with permutex2var to get contiguous VNNI order.
                    // idx selects: 0-7 from lo, 8-15 from hi (32-bit granularity)
                    const __m512i idx_lo = _mm512_setr_epi32(
                        0,1,2,3, 16,17,18,19, 4,5,6,7, 20,21,22,23);
                    const __m512i idx_hi = _mm512_setr_epi32(
                        8,9,10,11, 24,25,26,27, 12,13,14,15, 28,29,30,31);
                    __m512i out0 = _mm512_permutex2var_epi32(lo, idx_lo, hi);
                    __m512i out1 = _mm512_permutex2var_epi32(lo, idx_hi, hi);
                    _mm512_storeu_si512(d + n * VNNI_PAIR, out0);
                    _mm512_storeu_si512(d + (n + 16) * VNNI_PAIR, out1);
                }
            }
        } else {
            // Scalar fallback for transposed B or partial panels
            for (int kp = 0; kp < k_pairs; ++kp) {
                uint16_t *d = dst + kp * vnni_stride;
                const int k0 = kp * 2;
                const int k1 = k0 + 1;

                for (int n = 0; n < nr_act; ++n) {
                    uint16_t v0, v1;
                    if (!transB) {
                        v0 = (k0 < K) ? B[k0 * ldb + (j0 + n)] : 0;
                        v1 = (k1 < K) ? B[k1 * ldb + (j0 + n)] : 0;
                    } else {
                        v0 = (k0 < K) ? B[(j0 + n) * ldb + k0] : 0;
                        v1 = (k1 < K) ? B[(j0 + n) * ldb + k1] : 0;
                    }
                    d[n * VNNI_PAIR + 0] = v0;
                    d[n * VNNI_PAIR + 1] = v1;
                }
                for (int n = nr_act; n < NR_PACK; ++n) {
                    d[n * VNNI_PAIR + 0] = 0;
                    d[n * VNNI_PAIR + 1] = 0;
                }
            }
        }
    }
}

// ============================================================================
// BF16 5-loop BLIS-style thread loop (internal)
//
// Accumulates into a temporary FP32 C buffer, then optionally converts
// to BF16 for the final output.
// ============================================================================
// ============================================================================
// On-the-fly VNNI pack: pack NR_PACK columns of B for a K-block into VNNI.
// Used when weight caching is disabled — each thread packs only its strip.
// ============================================================================
__attribute__((target("avx512f,avx512bw")))
static void pack_b_vnni_strip(
    const uint16_t *B, int ldb, bool transB,
    int col_start, int nr_act, int K, int K_padded,
    int pc, int kb,
    uint16_t *packed) {

    const int kb_padded = (kb + 1) & ~1;
    const int k_pairs = kb_padded / 2;
    const int out_stride = NR_PACK * VNNI_PAIR;

    for (int kp = 0; kp < k_pairs; ++kp) {
        uint16_t *d = packed + kp * out_stride;
        const int k0 = pc + kp * 2;
        const int k1 = k0 + 1;

        if (!transB && nr_act == NR_PACK) {
            const uint16_t *row0 = (k0 < K) ? B + k0 * ldb + col_start
                                             : nullptr;
            const uint16_t *row1 = (k1 < K) ? B + k1 * ldb + col_start
                                             : nullptr;
            for (int n = 0; n < NR_PACK; n += 32) {
                __m512i r0 = row0 ? _mm512_loadu_si512(row0 + n)
                                  : _mm512_setzero_si512();
                __m512i r1 = row1 ? _mm512_loadu_si512(row1 + n)
                                  : _mm512_setzero_si512();
                __m512i lo = _mm512_unpacklo_epi16(r0, r1);
                __m512i hi = _mm512_unpackhi_epi16(r0, r1);
                const __m512i idx_lo = _mm512_setr_epi32(
                    0,1,2,3, 16,17,18,19, 4,5,6,7, 20,21,22,23);
                const __m512i idx_hi = _mm512_setr_epi32(
                    8,9,10,11, 24,25,26,27, 12,13,14,15, 28,29,30,31);
                _mm512_storeu_si512(d + n * VNNI_PAIR,
                    _mm512_permutex2var_epi32(lo, idx_lo, hi));
                _mm512_storeu_si512(d + (n + 16) * VNNI_PAIR,
                    _mm512_permutex2var_epi32(lo, idx_hi, hi));
            }
        } else {
            for (int n = 0; n < nr_act; ++n) {
                uint16_t v0, v1;
                if (!transB) {
                    v0 = (k0 < K) ? B[k0 * ldb + (col_start + n)] : 0;
                    v1 = (k1 < K) ? B[k1 * ldb + (col_start + n)] : 0;
                } else {
                    v0 = (k0 < K) ? B[(col_start + n) * ldb + k0] : 0;
                    v1 = (k1 < K) ? B[(col_start + n) * ldb + k1] : 0;
                }
                d[n * VNNI_PAIR + 0] = v0;
                d[n * VNNI_PAIR + 1] = v1;
            }
            for (int n = nr_act; n < NR_PACK; ++n) {
                d[n * VNNI_PAIR + 0] = 0;
                d[n * VNNI_PAIR + 1] = 0;
            }
        }
    }
}

__attribute__((target("avx512f,avx512bf16,fma")))
static void bf16_thread_loop(
    const GemmDescriptor &desc,
    const BlockPlan &plan,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params,
    const BF16PrepackedWeight *prepacked_b) {

    const uint16_t *A = static_cast<const uint16_t *>(src);
    const bool dst_is_bf16 = (desc.dst_dt == data_type_t::bf16);
    const bool bias_is_bf16 = (desc.bias_dt == data_type_t::bf16);

    const int M = desc.M, N = desc.N, K = desc.K;
    const int lda = desc.lda, ldc = desc.ldc;
    const float alpha = desc.alpha;
    const float beta = (desc.alpha != 1.0f && desc.beta != 0.0f)
                       ? (desc.beta / desc.alpha) : desc.beta;
    const int MB = plan.MB, NB = plan.NB, KB = plan.KB;
    const int MR = plan.MR, NR = plan.NR;
    const int num_threads = plan.num_threads;
    const bool has_bias = (bias != nullptr);
    const bool b_prepacked = (prepacked_b != nullptr);

    // On-the-fly packing support: raw B pointer for when weights aren't cached
    const uint16_t *B_raw = static_cast<const uint16_t *>(weight);
    const int ldb_raw = desc.ldb;
    const bool transB_raw = desc.transB;
    const int K_padded_raw = (K + 1) & ~1;

    // Convert bias to FP32 if needed (cached across calls if pointer unchanged).
    // Microkernels always receive FP32 bias pointers.
    static thread_local float *s_bias_fp32 = nullptr;
    static thread_local size_t s_bias_cap = 0;
    static thread_local const void *s_last_bias_ptr = nullptr;
    static thread_local int s_last_bias_N = 0;
    const float *bias_f = nullptr;

    if (has_bias) {
        if (bias_is_bf16) {
            // BF16 bias → convert to FP32 (skip if pointer & N unchanged)
            const bool bias_changed =
                (bias != s_last_bias_ptr) || (N != s_last_bias_N);
            if (bias_changed || s_bias_cap < static_cast<size_t>(N)) {
                if (s_bias_cap < static_cast<size_t>(N)) {
                    std::free(s_bias_fp32);
                    s_bias_fp32 = static_cast<float *>(std::aligned_alloc(
                        64, ((N * sizeof(float) + 63) & ~size_t(63))));
                    s_bias_cap = N;
                }
                if (s_bias_fp32) {
                    const uint16_t *bias_bf16 =
                        static_cast<const uint16_t *>(bias);
                    int n = 0;
                    // Vectorized BF16→FP32: shift left by 16 bits
                    for (; n + 16 <= N; n += 16) {
                        __m256i bf = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(bias_bf16 + n));
                        __m512i wide = _mm512_cvtepu16_epi32(bf);
                        __m512i shifted = _mm512_slli_epi32(wide, 16);
                        _mm512_storeu_ps(s_bias_fp32 + n,
                            _mm512_castsi512_ps(shifted));
                    }
                    // Scalar tail
                    for (; n < N; ++n) {
                        uint32_t bits = static_cast<uint32_t>(bias_bf16[n]) << 16;
                        std::memcpy(&s_bias_fp32[n], &bits, sizeof(float));
                    }
                    s_last_bias_ptr = bias;
                    s_last_bias_N = N;
                }
            }
            bias_f = s_bias_fp32;
        } else {
            // FP32 bias → use directly
            bias_f = static_cast<const float *>(bias);
        }
    }

    // Fused post-op detection
    fused_postop_t fused_op = fused_postop_t::none;
    int fused_idx = -1;
    for (int i = 0; i < static_cast<int>(params.postop_.size()); ++i) {
        auto pt = params.postop_[i].po_type;
        if (pt == post_op_type_t::relu) {
            if (params.postop_[i].alpha == 0.0f) {
                fused_op = fused_postop_t::relu; fused_idx = i; break;
            }
        } else if (pt == post_op_type_t::gelu_tanh) {
            fused_op = fused_postop_t::gelu_tanh; fused_idx = i; break;
        } else if (pt == post_op_type_t::gelu_erf) {
            fused_op = fused_postop_t::gelu_erf; fused_idx = i; break;
        } else if (pt == post_op_type_t::sigmoid) {
            fused_op = fused_postop_t::sigmoid; fused_idx = i; break;
        } else if (pt == post_op_type_t::tanh) {
            fused_op = fused_postop_t::tanh_op; fused_idx = i; break;
        }
    }

    std::vector<matmul_post_op> remaining_postops;
    remaining_postops.reserve(params.postop_.size());
    for (int i = 0; i < static_cast<int>(params.postop_.size()); ++i) {
        if (i != fused_idx)
            remaining_postops.push_back(params.postop_[i]);
    }
    const bool has_remaining_postops = !remaining_postops.empty();

    const int vnni_stride = NR_PACK * VNNI_PAIR;

    // FP32 output buffer (accumulation always in FP32)
    // If dst is BF16, we write FP32 here then convert at the end.
    // If dst is FP32, we write directly to dst.
    float *C_fp32;
    int ldc_fp32;
    bool need_fp32_buf = dst_is_bf16;
    static thread_local float *s_c_buf = nullptr;
    static thread_local size_t s_c_cap = 0;

    if (need_fp32_buf) {
        size_t needed = static_cast<size_t>(M) * N;
        if (s_c_cap < needed) {
            std::free(s_c_buf);
            s_c_buf = static_cast<float *>(std::aligned_alloc(
                64, ((needed * sizeof(float) + 63) & ~size_t(63))));
            s_c_cap = s_c_buf ? needed : 0;
        }
        if (!s_c_buf) {
            commonlog_error("BF16 GEMM: failed to allocate FP32 C buffer");
            return;
        }
        C_fp32 = s_c_buf;
        ldc_fp32 = N;
        if (beta != 0.0f) {
            const uint16_t *C_bf16 = static_cast<const uint16_t *>(dst);
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N; ++n) {
                    uint32_t bits = static_cast<uint32_t>(
                        C_bf16[m * ldc + n]) << 16;
                    float val;
                    std::memcpy(&val, &bits, sizeof(val));
                    C_fp32[m * ldc_fp32 + n] = val;
                }
        }
    } else {
        C_fp32 = static_cast<float *>(dst);
        ldc_fp32 = ldc;
    }

    // Select hot kernel based on planner MR and NR
    bf16_ukernel_fn_t hot_kernel = select_bf16_ukernel(MR, NR);

    // Fuse bias/activation only when alpha==1 (epilogue order is correct).
    // When alpha!=1, bias/activation must come AFTER scale_tile.
    const bool can_fuse_epilogue = (alpha == 1.0f);

    // Can we store BF16 directly from microkernel epilogue?
    // Requires: BF16 output + alpha==1 (correct epilogue order) + no unfused postops
    const bool can_direct_bf16 = dst_is_bf16 && can_fuse_epilogue
                                 && !has_remaining_postops;
    // BF16 output pointer (only used when can_direct_bf16)
    uint16_t *C_bf16_dst = dst_is_bf16
        ? static_cast<uint16_t *>(dst) : nullptr;

    // A-packing: auto-enable when row stride exceeds L1 prefetcher limit
    const bool do_pack_a = (lda * static_cast<int>(sizeof(uint16_t)) > 4096);

    // Thread loop: parallel over MC x NC tiles
    if (num_threads <= 1) {
        // Single-threaded path
        for (int pc = 0; pc < K; pc += KB) {
            const int kb_act = std::min(KB, K - pc);
            const int pc_pair = pc / 2;
            const bool is_last_k = (pc + KB >= K);
            const float beta_k = (pc == 0) ? beta : 1.0f;
            const bool can_fuse = is_last_k && can_fuse_epilogue;

            for (int jc = 0; jc < N; jc += NB) {
                const int nb_act = std::min(NB, N - jc);

                for (int ic = 0; ic < M; ic += MB) {
                    const int mb_act = std::min(MB, M - ic);
                    const int m_panels = (mb_act + MR - 1) / MR;

                    for (int jr = 0; jr < nb_act; jr += NR) {
                        const int nr_act = std::min(NR, nb_act - jr);
                        const int col = jc + jr;
                        const bool full_nr = (nr_act == NR);

                        const fused_postop_t tile_fop =
                            can_fuse ? fused_op : fused_postop_t::none;
                        const float *tile_bias =
                            (has_bias && can_fuse) ? (bias_f + col) : nullptr;

                        // Resolve B pointer once per NR-tile (outside M-panel loop)
                        const uint16_t *pb;
                        int pb_stride;
                        if (b_prepacked) {
                            int panel_idx = col / NR_PACK;
                            int in_panel_off = col % NR_PACK;
                            pb = prepacked_b->get_panel(pc_pair, panel_idx)
                                 + in_panel_off * VNNI_PAIR;
                            pb_stride = vnni_stride;
                        } else {
                            static thread_local uint16_t *s_otf = nullptr;
                            static thread_local size_t s_otf_cap = 0;
                            int kb_pad = (kb_act + 1) & ~1;
                            size_t need = static_cast<size_t>(
                                kb_pad / 2) * NR_PACK * VNNI_PAIR;
                            if (s_otf_cap < need) {
                                std::free(s_otf);
                                s_otf = static_cast<uint16_t *>(
                                    std::aligned_alloc(64,
                                        ((need * sizeof(uint16_t) + 63)
                                         & ~size_t(63))));
                                s_otf_cap = s_otf ? need : 0;
                            }
                            if (!s_otf) continue;
                            pack_b_vnni_strip(B_raw, ldb_raw, transB_raw,
                                col, std::min(static_cast<int>(NR_PACK),
                                              N - col),
                                K, K_padded_raw, pc, kb_act, s_otf);
                            pb = s_otf;
                            pb_stride = vnni_stride;
                        }

                        for (int ip = 0; ip < m_panels; ++ip) {
                            const int ir = ip * MR;
                            const int mr_act = std::min(MR, mb_act - ir);
                            const uint16_t *At = A + (ic + ir) * lda + pc;
                            float *Ct = C_fp32 + (ic + ir) * ldc_fp32 + col;

                            uint16_t *tile_bf16 = nullptr;
                            int tile_ldc_bf16 = 0;
                            if (can_direct_bf16 && is_last_k) {
                                tile_bf16 = C_bf16_dst
                                    + (ic + ir) * ldc + col;
                                tile_ldc_bf16 = ldc;
                            }

                            if (hot_kernel && full_nr && mr_act == MR) {
                                hot_kernel(
                                    At, lda, pb, pb_stride,
                                    Ct, ldc_fp32, kb_act, beta_k,
                                    tile_bias, tile_fop,
                                    tile_bf16, tile_ldc_bf16);
                            } else if (full_nr && mr_act >= 1 && mr_act <= 4) {
                                auto tail_uk = select_bf16_ukernel(mr_act, NR);
                                if (tail_uk) {
                                    tail_uk(
                                        At, lda, pb, pb_stride,
                                        Ct, ldc_fp32, kb_act, beta_k,
                                        tile_bias, tile_fop,
                                        tile_bf16, tile_ldc_bf16);
                                } else {
                                    bf16_tail_kernel(
                                        At, lda, pb, pb_stride,
                                        Ct, ldc_fp32, kb_act,
                                        mr_act, nr_act, beta_k,
                                        tile_bias, tile_fop,
                                        tile_bf16, tile_ldc_bf16);
                                }
                            } else {
                                bf16_tail_kernel(
                                    At, lda, pb, pb_stride,
                                    Ct, ldc_fp32, kb_act,
                                    mr_act, nr_act, beta_k,
                                    tile_bias, tile_fop,
                                    tile_bf16, tile_ldc_bf16);
                            }
                        }
                    }
                }
            }

            // Post-tile: alpha scaling, then bias/postops if not fused
            if (is_last_k) {
                if (alpha != 1.0f) {
                    scale_tile(C_fp32, ldc_fp32, M, N, alpha);
                    // Bias and fused postop deferred to here
                    if (has_bias)
                        apply_postops_tile(C_fp32, ldc_fp32, M, N,
                                           0, 0, bias_f, {});
                    if (fused_op != fused_postop_t::none && fused_idx >= 0)
                        apply_postops_tile(C_fp32, ldc_fp32, M, N,
                                           0, 0, nullptr,
                                           {params.postop_[fused_idx]});
                }
                if (has_remaining_postops) {
                    apply_postops_tile(C_fp32, ldc_fp32, M, N,
                                       0, 0, nullptr, remaining_postops);
                }
            }
        }
    } else {
        // BRGEMM-style multi-threaded: tile-outer, K-inner.
        // Each thread owns disjoint M×N tiles and processes ALL K-blocks
        // for its tiles before moving on. This eliminates inter-K-block
        // barriers and keeps C data hot in L1/registers across K-blocks.
        const int ic_tiles = (M + MB - 1) / MB;
        const int jc_tiles = (N + NB - 1) / NB;
        const int total_2d_tiles = ic_tiles * jc_tiles;
        const int active_threads = std::min(num_threads, total_2d_tiles);

        // Skip A-packing for small M when total A fits in L2.
        // Avoids N_threads redundant copies of the same small A matrix.
        const bool skip_pack_a = do_pack_a
            && (static_cast<size_t>(M) * K * sizeof(uint16_t)
                <= static_cast<size_t>(uarch.l2_bytes));
        const bool actual_pack_a = do_pack_a && !skip_pack_a;

        #pragma omp parallel num_threads(active_threads)
        {
            static thread_local uint16_t *tl_pa = nullptr;
            static thread_local size_t tl_pa_cap = 0;

            #pragma omp for schedule(static)
            for (int tid = 0; tid < total_2d_tiles; ++tid) {
                const int ic_idx = tid / jc_tiles;
                const int jc_idx = tid % jc_tiles;
                const int ic = ic_idx * MB;
                const int jc = jc_idx * NB;
                const int mb_act = std::min(MB, M - ic);
                const int nb_act = std::min(NB, N - jc);
                const int m_panels = (mb_act + MR - 1) / MR;

                // BRGEMM K-loop: accumulate ALL K-blocks for this tile
                for (int pc = 0; pc < K; pc += KB) {
                    const int kb_act = std::min(KB, K - pc);
                    const int pc_pair = pc / 2;
                    const bool is_last_k = (pc + KB >= K);
                    const float beta_k = (pc == 0) ? beta : 1.0f;
                    const bool can_fuse = is_last_k && can_fuse_epilogue;

                    // A source for this K-block
                    const uint16_t *a_base;
                    int at_stride_base;
                    if (actual_pack_a) {
                        size_t need = static_cast<size_t>(
                            ((mb_act + MR - 1) / MR) * MR) * kb_act;
                        if (tl_pa_cap < need) {
                            std::free(tl_pa);
                            tl_pa = static_cast<uint16_t *>(std::aligned_alloc(
                                64, ((need * sizeof(uint16_t) + 63) & ~size_t(63))));
                            tl_pa_cap = tl_pa ? need : 0;
                        }
                        if (tl_pa) {
                            pack_a_bf16_block(A, tl_pa, ic, pc, M, K, lda,
                                              mb_act, kb_act, MR);
                            a_base = tl_pa;
                            at_stride_base = kb_act;
                        } else {
                            a_base = A + ic * lda + pc;
                            at_stride_base = lda;
                        }
                    } else {
                        a_base = A + ic * lda + pc;
                        at_stride_base = lda;
                    }

                    for (int jr = 0; jr < nb_act; jr += NR) {
                        const int nr_act = std::min(NR, nb_act - jr);
                        const int col = jc + jr;
                        const bool full_nr = (nr_act == NR);

                        const fused_postop_t tile_fop =
                            can_fuse ? fused_op : fused_postop_t::none;
                        const float *tile_bias =
                            (has_bias && can_fuse) ? (bias_f + col) : nullptr;

                        // Resolve B once per NR-tile (outside M-panel loop)
                        const uint16_t *pb;
                        int pb_stride;
                        if (b_prepacked) {
                            int panel_idx = col / NR_PACK;
                            int in_panel_off = col % NR_PACK;
                            pb = prepacked_b->get_panel(pc_pair, panel_idx)
                                 + in_panel_off * VNNI_PAIR;
                            pb_stride = vnni_stride;
                        } else {
                            static thread_local uint16_t *tl_otf = nullptr;
                            static thread_local size_t tl_otf_cap = 0;
                            int kb_pad = (kb_act + 1) & ~1;
                            size_t need = static_cast<size_t>(
                                kb_pad / 2) * NR_PACK * VNNI_PAIR;
                            if (tl_otf_cap < need) {
                                std::free(tl_otf);
                                tl_otf = static_cast<uint16_t *>(
                                    std::aligned_alloc(64,
                                        ((need * sizeof(uint16_t) + 63)
                                         & ~size_t(63))));
                                tl_otf_cap = tl_otf ? need : 0;
                            }
                            if (!tl_otf) continue;
                            pack_b_vnni_strip(B_raw, ldb_raw,
                                transB_raw, col,
                                std::min(static_cast<int>(NR_PACK),
                                         N - col),
                                K, K_padded_raw, pc, kb_act, tl_otf);
                            pb = tl_otf;
                            pb_stride = vnni_stride;
                        }

                        for (int ip = 0; ip < m_panels; ++ip) {
                            const int ir = ip * MR;
                            const int mr_act = std::min(MR, mb_act - ir);
                            float *Ct = C_fp32 + (ic + ir) * ldc_fp32 + col;

                            const uint16_t *At;
                            int at_stride;
                            if (actual_pack_a) {
                                At = a_base + ip * MR * kb_act;
                                at_stride = kb_act;
                            } else {
                                At = a_base + ir * at_stride_base;
                                at_stride = at_stride_base;
                            }

                            uint16_t *tile_bf16 = nullptr;
                            int tile_ldc_bf16 = 0;
                            if (can_direct_bf16 && is_last_k) {
                                tile_bf16 = C_bf16_dst
                                    + (ic + ir) * ldc + col;
                                tile_ldc_bf16 = ldc;
                            }

                            if (hot_kernel && full_nr && mr_act == MR) {
                                hot_kernel(
                                    At, at_stride, pb, pb_stride,
                                    Ct, ldc_fp32, kb_act, beta_k,
                                    tile_bias, tile_fop,
                                    tile_bf16, tile_ldc_bf16);
                            } else if (full_nr && mr_act >= 1 && mr_act <= 4) {
                                auto tail_uk = select_bf16_ukernel(mr_act, NR);
                                if (tail_uk) {
                                    tail_uk(
                                        At, at_stride, pb, pb_stride,
                                        Ct, ldc_fp32, kb_act, beta_k,
                                        tile_bias, tile_fop,
                                        tile_bf16, tile_ldc_bf16);
                                } else {
                                    bf16_tail_kernel(
                                        At, at_stride, pb, pb_stride,
                                        Ct, ldc_fp32, kb_act,
                                        mr_act, nr_act, beta_k,
                                        tile_bias, tile_fop,
                                        tile_bf16, tile_ldc_bf16);
                                }
                            } else {
                                bf16_tail_kernel(
                                    At, at_stride, pb, pb_stride,
                                    Ct, ldc_fp32, kb_act,
                                    mr_act, nr_act, beta_k,
                                    tile_bias, tile_fop,
                                    tile_bf16, tile_ldc_bf16);
                            }
                        }
                    }

                    // Per-tile epilogue after last K-block
                    if (is_last_k) {
                        float *Ctile = C_fp32 + ic * ldc_fp32 + jc;
                        if (alpha != 1.0f) {
                            scale_tile(Ctile, ldc_fp32, mb_act, nb_act, alpha);
                            if (has_bias)
                                apply_postops_tile(Ctile, ldc_fp32, mb_act, nb_act,
                                                   jc, ic, bias_f, {});
                            if (fused_op != fused_postop_t::none && fused_idx >= 0)
                                apply_postops_tile(Ctile, ldc_fp32, mb_act, nb_act,
                                                   jc, ic, nullptr,
                                                   {params.postop_[fused_idx]});
                        }
                        if (has_remaining_postops) {
                            apply_postops_tile(Ctile, ldc_fp32, mb_act, nb_act,
                                               jc, ic, nullptr, remaining_postops);
                        }
                    }
                } // K-loop (BRGEMM batch-reduce)
            } // omp for
        } // omp parallel
    }

    // Convert FP32 accumulator to BF16 output if not already fused
    if (need_fp32_buf && !can_direct_bf16 && s_c_buf) {
        uint16_t *C_bf16 = static_cast<uint16_t *>(dst);
        convert_fp32_to_bf16_tile(C_fp32, ldc_fp32, C_bf16, ldc, M, N);
    }
}

// ============================================================================
// BF16 GEMM execute: plan caching + VNNI B prepacking + thread loop
// ============================================================================

void bf16_gemm_execute(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;
    const bool transB = desc.transB;
    const bool is_weights_const = desc.is_weights_const;
    const int K_padded = (K + 1) & ~1;  // Round up to even for VNNI

    // ── 1. Block plan (reuse FP32 planner, cache aware) ──
    // BF16 elements are 2 bytes vs 4 for FP32, so effectively
    // we can fit 2x more data in cache. The planner computes
    // based on FP32 element size; we adjust KB by 2x.
    static thread_local struct {
        int M, N, K, threads; bool transA, transB;
        BlockPlan plan;
    } s_plan_cache = {0, 0, 0, 0, false, false, {}};

    // Classify activation post-ops for NR selection.
    // "Simple" activations (relu) need only 1 ZMM scratch → NV=4 (NR=64) safe
    // "Complex" activations (gelu, sigmoid, tanh...) need 6-10 scratch → NV=2 (NR=32)
    bool has_activation = false;
    bool has_complex_activation = false;
    for (const auto &po : params.postop_) {
        auto pt = po.po_type;
        if (pt == post_op_type_t::relu || pt == post_op_type_t::leaky_relu) {
            has_activation = true;
            // relu/leaky_relu: simple, NV=4 safe
        } else if (pt == post_op_type_t::gelu_tanh ||
                   pt == post_op_type_t::gelu_erf ||
                   pt == post_op_type_t::sigmoid ||
                   pt == post_op_type_t::tanh ||
                   pt == post_op_type_t::swish ||
                   pt == post_op_type_t::elu) {
            has_activation = true;
            has_complex_activation = true;
            break;
        }
    }

    BlockPlan plan;
    if (s_plan_cache.M == M && s_plan_cache.N == N && s_plan_cache.K == K &&
        s_plan_cache.threads == desc.num_threads &&
        s_plan_cache.transA == desc.transA &&
        s_plan_cache.transB == transB) {
        plan = s_plan_cache.plan;
    } else {
        plan = plan_blocks(desc, uarch);
        // Ensure KB is even for VNNI pairs (dpbf16ps processes k-pairs)
        plan.KB = (plan.KB + 1) & ~1;

        // BF16 KB selection: minimize K-blocks to reduce synchronization.
        // For multi-threaded: use L2 for A (accept L2 latency, avoid barriers).
        // For single-threaded: use L1 constraint (minimize latency).
        {
            int b_full_k_bytes = K_padded * NR_PACK
                                 * static_cast<int>(sizeof(uint16_t));
            if (desc.num_threads <= 1) {
                // Single-threaded: A must fit in L1 for low latency
                int a_panel_bytes = 6 * K_padded
                                    * static_cast<int>(sizeof(uint16_t));
                int l1_limit = static_cast<int>(0.8 * uarch.l1d_bytes);
                if (b_full_k_bytes <= uarch.l2_bytes / 2
                    && a_panel_bytes <= l1_limit) {
                    plan.KB = K_padded;
                }
            } else {
                // Multi-threaded: prefer fewer K-blocks to reduce sync overhead.
                // Two constraints:
                //  (a) A panel (MR*KB*2) must fit in L2 (accept L2 access latency)
                //  (b) B panel (KB*NR_PACK*2) must fit in L2 (avoid L3 streaming)
                int l2_for_a = uarch.l2_bytes / 2;
                int kb_a = l2_for_a / (6 * static_cast<int>(sizeof(uint16_t)));
                int kb_b = (uarch.l2_bytes / 2)
                           / (NR_PACK * static_cast<int>(sizeof(uint16_t)));
                int kb_max_mt = std::min(kb_a, kb_b);
                kb_max_mt = (kb_max_mt + 1) & ~1;
                if (K_padded <= kb_max_mt) {
                    plan.KB = K_padded;
                } else if (kb_max_mt > plan.KB) {
                    int n_blocks = (K_padded + kb_max_mt - 1) / kb_max_mt;
                    int even_kb = ((K_padded + n_blocks - 1) / n_blocks + 7) & ~7;
                    plan.KB = (even_kb + 1) & ~1;
                }
            }
        }

        s_plan_cache = {M, N, K, desc.num_threads, desc.transA, transB, plan};
    }

    // BF16 MR/NR selection (M-aware + post-op aware):
    //
    // DECODE path (M <= 4): MR=M, NR=64 (NV=4)
    //   - Maximizes N-throughput for GEMV-like shapes
    //   - MR=1..4 with NV=4: 4..16 acc, max N-throughput
    //   - NR=64 = NR_PACK: zero panel crossing, perfect alignment
    //
    // GEMM path (M >= 5):
    //   - NR=64 (NV=4): relu/no-act — 24 acc, max throughput, zero crossing
    //   - NR=32 (NV=2): complex activation — 12 acc, safe epilogue
    //   - NR=16 (NV=1): small N, avoid tile waste
    //
    const bool is_decode = (M <= 4);
    const char *path_name = "gemm";

    if (is_decode) {
        // Decode: MR=M (exact fit, no M-waste) for ALL N values.
        // Picks the widest NR that N supports for max N-throughput.
        plan.MR = M;
        plan.MB = M;
        if (N >= 64)       plan.NR = 64;
        else if (N >= 32)  plan.NR = 32;
        else               plan.NR = 16;
        path_name = "decode";
    } else if (N >= 64 && !has_complex_activation) {
        // MR selection: prefer MR=6 for throughput, but use MR=4 when
        // MR=6 creates badly unbalanced IC tiles (e.g. M=8 → 6+2).
        // MR=4 with NV=4 gives 16 accumulators — still efficient.
        if (M % 6 == 0 || M >= 18) {
            plan.MR = 6;
        } else if (M % 4 == 0) {
            plan.MR = 4;
        } else if (M % 6 <= 3 && M > 12) {
            plan.MR = 4;
        } else {
            plan.MR = 6;
        }
        plan.NR = 64;
        path_name = "gemm-nr64";
    } else if (N >= 32) {
        plan.MR = (M % 6 == 0 || M >= 18) ? 6
                : (M % 4 == 0) ? 4 : 6;
        plan.NR = 32;
    } else {
        plan.MR = (M % 6 == 0 || M >= 18) ? 6
                : (M % 4 == 0) ? 4 : 6;
        plan.NR = 16;
    }
    // Re-align NB to be a multiple of the final NR
    plan.NB = std::max(plan.NB / plan.NR * plan.NR, plan.NR);
    plan.NB = std::min(plan.NB, N);

    // Recompute MB: L2-aware when A is packed (packed_A + B_panel <= L2).
    if (!is_decode) {
        bool will_pack_a = (desc.lda * static_cast<int>(sizeof(uint16_t)) > 4096);
        if (will_pack_a) {
            int b_lines_per_krow = ((plan.NR + 15) / 16) * 64;
            int k_pairs_kb = (plan.KB + 1) / 2;
            int b_accessed_bytes = k_pairs_kb * b_lines_per_krow;
            int l2_for_a = std::max(uarch.l2_bytes - b_accessed_bytes, 0);
            plan.MB = std::max(l2_for_a / (plan.KB * 2), plan.MR);
        } else {
            plan.MB = std::max((uarch.l1d_bytes + uarch.l2_bytes)
                               / (plan.NB * 4 + plan.KB * 2), plan.MR);
        }
        plan.MB = plan.MB / plan.MR * plan.MR;
        plan.MB = std::min(plan.MB, M);
    }

    // Load-balance: ensure enough tiles for all threads, with even-sized tiles.
    if (!is_decode && plan.num_threads > 1) {
        int jc_tiles = (N + plan.NB - 1) / plan.NB;
        int ic_tiles = (M + plan.MB - 1) / plan.MB;

        // Need at least num_threads total tiles. Compute how many IC tiles
        // are needed given the JC tile count.
        int needed_ic = (plan.num_threads + jc_tiles - 1) / jc_tiles;
        if (needed_ic > ic_tiles && needed_ic > 1) {
            // Split M into needed_ic even blocks, each a multiple of MR.
            int m_panels = (M + plan.MR - 1) / plan.MR;
            int panels_per_block = std::max(m_panels / needed_ic, 1);
            plan.MB = panels_per_block * plan.MR;
            plan.MB = std::min(plan.MB, M);
            ic_tiles = (M + plan.MB - 1) / plan.MB;
        }

        // If still not enough tiles, also shrink NB
        if (ic_tiles * jc_tiles < plan.num_threads && plan.NB > plan.NR) {
            int needed_jc = (plan.num_threads + ic_tiles - 1) / ic_tiles;
            int nb_target = (N + needed_jc - 1) / needed_jc;
            nb_target = std::max((nb_target / plan.NR) * plan.NR, plan.NR);
            plan.NB = std::min(nb_target, plan.NB);
        }
    }

    if (apilog_info_enabled()) {
        apilog_info("AI BF16 GEMM plan: M=", M, " N=", N, " K=", K,
                    " MB=", plan.MB, " NB=", plan.NB, " KB=", plan.KB,
                    " MR=", plan.MR, " NR=", plan.NR,
                    " NV=", plan.NR / 16,
                    " path=", path_name,
                    " activation=", has_activation ? "yes" : "no",
                    " complex_act=", has_complex_activation ? "yes" : "no",
                    " threads=", plan.num_threads);
    }

    // ── 2. B matrix VNNI prepacking ──
    //
    // (a) WEIGHT_CACHE + is_weights_const: one-time pack, cached forever.
    // (b) Otherwise: prepacked_b = nullptr → thread loop does on-the-fly
    //     VNNI packing per NR-wide strip per thread (parallelized, L1-hot).
    //
    static int32_t s_weight_cache =
        matmul_config_t::instance().get_weight_cache();
    const BF16PrepackedWeight *prepacked_b = nullptr;
    const bool can_cache = is_weights_const && (s_weight_cache != 0);

    if (can_cache) {
        PrepackedWeightKey bk{weight, K, N, desc.ldb, transB};
        prepacked_b = BF16PrepackedWeightCache::instance().get_or_prepack(
            bk, static_cast<const uint16_t *>(weight));
    }
    // else: prepacked_b stays nullptr → on-the-fly packing in thread loop

    // ── 3. Thread loop ──
    bf16_thread_loop(desc, plan, uarch, src, weight, dst, bias, params,
                     prepacked_b);
}

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
