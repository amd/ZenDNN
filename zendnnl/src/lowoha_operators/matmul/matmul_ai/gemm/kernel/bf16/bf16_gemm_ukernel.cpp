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

//
// BF16 GEMM microkernels — pure SIMD compute, zero allocations, zero OMP.
//
// This file contains ONLY the register-tile compute kernels:
//   - bf16_ukernel_6x64_asm   (hand-scheduled inline asm, peak path)
//   - bf16_ukernel<MR,NV>     (C++ intrinsics template, all shapes)
//   - bf16_tail_kernel         (dynamic MR/NR with masked ops)
//   - select_bf16_ukernel      (dispatch table)
//
// The driver (thread loops, packing, plan caching) is in avx512_bf16_gemm.cpp.
// Shared packing routines are in common/bf16_packing.hpp.

//

#include "lowoha_operators/matmul/matmul_ai/gemm/kernel/bf16/bf16_gemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/avx512_math.hpp"

#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

// ============================================================================
// Hand-scheduled BF16 6x64 microkernel (inline assembly K-loop)
//
// Register map (29 of 32 ZMM):
//   zmm0  - zmm23  : 24 FP32 accumulators (6 rows x 4 NV columns)
//   zmm24 - zmm27  : 4 B loads (VNNI-packed, 16 BF16 pairs each)
//   zmm28           : A broadcast (BF16 pair as 32-bit, replicated)
//   zmm29 - zmm31  : free for epilogue
//
// K-loop: 2x k-pair unrolled (4 BF16 elements per iteration).
// Per k-pair: 4 vmovups(B) + 6 vpbroadcastd(A) + 24 vdpbf16ps = 34 instr.
// 2 k-pairs = 68 compute + 7 ptr advances + 1 branch = 76 instructions.
// dpbf16-bound: 48 dpbf16 / 2 ports = 24 cycles + 1 = 25 cycles.
// 1536 FLOPs / 25 cycles = 61.4 FLOPs/cycle (96% of peak).
// ============================================================================
__attribute__((target("avx512f,avx512bf16,fma"), noinline))
void bf16_ukernel_6x64_asm(
    const uint16_t *__restrict__ A, int lda,
    const uint16_t *__restrict__ B_vnni, int b_stride,
    float *__restrict__ C, int ldc,
    int k, float beta,
    const float *__restrict__ bias,
    fused_postop_t fused_op,
    uint16_t *__restrict__ C_bf16, int ldc_bf16) {

    const uint16_t *a0 = A;
    const uint16_t *a1 = A + lda;
    const uint16_t *a2 = A + 2 * lda;
    const uint16_t *a3 = A + 3 * lda;
    const uint16_t *a4 = A + 4 * lda;
    const uint16_t *a5 = A + 5 * lda;
    const uint16_t *bp = B_vnni;
    const long bs = static_cast<long>(b_stride) * static_cast<long>(sizeof(uint16_t));

    __m512 c_acc[24] __attribute__((aligned(64)));

    const int k_pairs = k / 2;
    int k2 = k_pairs >> 1;

    __asm__ __volatile__ (
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

        ".p2align 5\n"
        "1:\n\t"
        // k-pair 0
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

        // k-pair 1
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

        "addq $8, %[a0]\n\t"  "addq $8, %[a1]\n\t"  "addq $8, %[a2]\n\t"
        "addq $8, %[a3]\n\t"  "addq $8, %[a4]\n\t"  "addq $8, %[a5]\n\t"
        "leaq (%[bp],%[bs],2), %[bp]\n\t"

        "decl %[k2]\n\t"
        "jnz 1b\n\t"

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

    __m512 c00=c_acc[0],  c01=c_acc[1],  c02=c_acc[2],  c03=c_acc[3];
    __m512 c10=c_acc[4],  c11=c_acc[5],  c12=c_acc[6],  c13=c_acc[7];
    __m512 c20=c_acc[8],  c21=c_acc[9],  c22=c_acc[10], c23=c_acc[11];
    __m512 c30=c_acc[12], c31=c_acc[13], c32=c_acc[14], c33=c_acc[15];
    __m512 c40=c_acc[16], c41=c_acc[17], c42=c_acc[18], c43=c_acc[19];
    __m512 c50=c_acc[20], c51=c_acc[21], c52=c_acc[22], c53=c_acc[23];

    // Remainder: odd k-pair + odd k element
    {
        int kk_done = (k_pairs >> 1) * 2;
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

    // Epilogue: beta, bias, activation, store
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
    if (bias) {
        __m512 bv0=_mm512_loadu_ps(bias), bv1=_mm512_loadu_ps(bias+16),
               bv2=_mm512_loadu_ps(bias+32), bv3=_mm512_loadu_ps(bias+48);
#define BIAS_ROW(R) \
        c##R##0=_mm512_add_ps(c##R##0,bv0); c##R##1=_mm512_add_ps(c##R##1,bv1); \
        c##R##2=_mm512_add_ps(c##R##2,bv2); c##R##3=_mm512_add_ps(c##R##3,bv3);
        BIAS_ROW(0) BIAS_ROW(1) BIAS_ROW(2) BIAS_ROW(3) BIAS_ROW(4) BIAS_ROW(5)
#undef BIAS_ROW
    }
    if (fused_op != fused_postop_t::none) {
#define ACT_ROW(R) \
        c##R##0=apply_fused_postop(c##R##0,fused_op); c##R##1=apply_fused_postop(c##R##1,fused_op); \
        c##R##2=apply_fused_postop(c##R##2,fused_op); c##R##3=apply_fused_postop(c##R##3,fused_op);
        ACT_ROW(0) ACT_ROW(1) ACT_ROW(2) ACT_ROW(3) ACT_ROW(4) ACT_ROW(5)
#undef ACT_ROW
    }
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
// Template microkernel: MR x (NV*16) with VNNI dpbf16ps
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
    for (; kk + 1 < k_pairs; kk += 2) {
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
template void bf16_ukernel<1,4>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<2,4>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<3,4>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<4,4>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<1,2>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<2,2>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<3,2>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<4,2>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<1,1>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<2,1>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<3,1>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<4,1>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<6,4>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<6,2>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);
template void bf16_ukernel<6,1>(const uint16_t*, int, const uint16_t*, int, float*, int, int, float, const float*, fused_postop_t, uint16_t*, int);

// ============================================================================
// Microkernel dispatch
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

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
