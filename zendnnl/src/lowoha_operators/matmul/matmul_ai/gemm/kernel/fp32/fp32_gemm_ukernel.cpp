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
// FP32 GEMM microkernels -- pure SIMD compute, zero allocations, zero OMP.
//
// Layer 3 in the Planner/Looper/Kernel architecture.
//

#include "lowoha_operators/matmul/matmul_ai/gemm/kernel/fp32/fp32_gemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/avx512_math.hpp"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

__attribute__((target("avx512f,fma"), noinline))
void avx512_ukernel_6x64_asm(
    const float * __restrict__ pa, int a_stride,
    const float * __restrict__ pb, int b_stride,
    float * __restrict__ C, int ldc, int k, float beta,
    const float * __restrict__ bias, fused_postop_t fused_op) {

    const float *a0 = pa;
    const float *a1 = pa + a_stride;
    const float *a2 = pa + 2 * a_stride;
    const float *a3 = pa + 3 * a_stride;
    const float *a4 = pa + 4 * a_stride;
    const float *a5 = pa + 5 * a_stride;
    const float *bp = pb;
    const long bs = static_cast<long>(b_stride) * 4;

    // Stack buffer: 24 accumulators × 64 bytes = 1536 bytes (L1-resident)
    __m512 c_acc[24] __attribute__((aligned(64)));
    int k2 = k >> 1;

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

        // ── 2× unrolled K-loop ──
        ".p2align 5\n"
        "1:\n\t"
        // ── k-step 0: B from [bp], A from [a*+0] ──
        "vmovups     (%[bp]),    %%zmm24\n\t"
        "vmovups   64(%[bp]),    %%zmm25\n\t"
        "vmovups  128(%[bp]),    %%zmm26\n\t"
        "vmovups  192(%[bp]),    %%zmm27\n\t"
        "vbroadcastss  (%[a0]),  %%zmm28\n\t"
        "vfmadd231ps %%zmm24, %%zmm28, %%zmm0\n\t"  "vfmadd231ps %%zmm25, %%zmm28, %%zmm1\n\t"
        "vfmadd231ps %%zmm26, %%zmm28, %%zmm2\n\t"  "vfmadd231ps %%zmm27, %%zmm28, %%zmm3\n\t"
        "vbroadcastss  (%[a1]),  %%zmm28\n\t"
        "vfmadd231ps %%zmm24, %%zmm28, %%zmm4\n\t"  "vfmadd231ps %%zmm25, %%zmm28, %%zmm5\n\t"
        "vfmadd231ps %%zmm26, %%zmm28, %%zmm6\n\t"  "vfmadd231ps %%zmm27, %%zmm28, %%zmm7\n\t"
        "vbroadcastss  (%[a2]),  %%zmm28\n\t"
        "vfmadd231ps %%zmm24, %%zmm28, %%zmm8\n\t"  "vfmadd231ps %%zmm25, %%zmm28, %%zmm9\n\t"
        "vfmadd231ps %%zmm26, %%zmm28, %%zmm10\n\t" "vfmadd231ps %%zmm27, %%zmm28, %%zmm11\n\t"
        "vbroadcastss  (%[a3]),  %%zmm28\n\t"
        "vfmadd231ps %%zmm24, %%zmm28, %%zmm12\n\t" "vfmadd231ps %%zmm25, %%zmm28, %%zmm13\n\t"
        "vfmadd231ps %%zmm26, %%zmm28, %%zmm14\n\t" "vfmadd231ps %%zmm27, %%zmm28, %%zmm15\n\t"
        "vbroadcastss  (%[a4]),  %%zmm28\n\t"
        "vfmadd231ps %%zmm24, %%zmm28, %%zmm16\n\t" "vfmadd231ps %%zmm25, %%zmm28, %%zmm17\n\t"
        "vfmadd231ps %%zmm26, %%zmm28, %%zmm18\n\t" "vfmadd231ps %%zmm27, %%zmm28, %%zmm19\n\t"
        "vbroadcastss  (%[a5]),  %%zmm28\n\t"
        "vfmadd231ps %%zmm24, %%zmm28, %%zmm20\n\t" "vfmadd231ps %%zmm25, %%zmm28, %%zmm21\n\t"
        "vfmadd231ps %%zmm26, %%zmm28, %%zmm22\n\t" "vfmadd231ps %%zmm27, %%zmm28, %%zmm23\n\t"

        // ── k-step 1: B from [bp+bs], A from [a*+4] ──
        "vmovups     (%[bp],%[bs],1),    %%zmm24\n\t"
        "vmovups   64(%[bp],%[bs],1),    %%zmm25\n\t"
        "vmovups  128(%[bp],%[bs],1),    %%zmm26\n\t"
        "vmovups  192(%[bp],%[bs],1),    %%zmm27\n\t"
        "vbroadcastss 4(%[a0]),  %%zmm28\n\t"
        "vfmadd231ps %%zmm24, %%zmm28, %%zmm0\n\t"  "vfmadd231ps %%zmm25, %%zmm28, %%zmm1\n\t"
        "vfmadd231ps %%zmm26, %%zmm28, %%zmm2\n\t"  "vfmadd231ps %%zmm27, %%zmm28, %%zmm3\n\t"
        "vbroadcastss 4(%[a1]),  %%zmm28\n\t"
        "vfmadd231ps %%zmm24, %%zmm28, %%zmm4\n\t"  "vfmadd231ps %%zmm25, %%zmm28, %%zmm5\n\t"
        "vfmadd231ps %%zmm26, %%zmm28, %%zmm6\n\t"  "vfmadd231ps %%zmm27, %%zmm28, %%zmm7\n\t"
        "vbroadcastss 4(%[a2]),  %%zmm28\n\t"
        "vfmadd231ps %%zmm24, %%zmm28, %%zmm8\n\t"  "vfmadd231ps %%zmm25, %%zmm28, %%zmm9\n\t"
        "vfmadd231ps %%zmm26, %%zmm28, %%zmm10\n\t" "vfmadd231ps %%zmm27, %%zmm28, %%zmm11\n\t"
        "vbroadcastss 4(%[a3]),  %%zmm28\n\t"
        "vfmadd231ps %%zmm24, %%zmm28, %%zmm12\n\t" "vfmadd231ps %%zmm25, %%zmm28, %%zmm13\n\t"
        "vfmadd231ps %%zmm26, %%zmm28, %%zmm14\n\t" "vfmadd231ps %%zmm27, %%zmm28, %%zmm15\n\t"
        "vbroadcastss 4(%[a4]),  %%zmm28\n\t"
        "vfmadd231ps %%zmm24, %%zmm28, %%zmm16\n\t" "vfmadd231ps %%zmm25, %%zmm28, %%zmm17\n\t"
        "vfmadd231ps %%zmm26, %%zmm28, %%zmm18\n\t" "vfmadd231ps %%zmm27, %%zmm28, %%zmm19\n\t"
        "vbroadcastss 4(%[a5]),  %%zmm28\n\t"
        "vfmadd231ps %%zmm24, %%zmm28, %%zmm20\n\t" "vfmadd231ps %%zmm25, %%zmm28, %%zmm21\n\t"
        "vfmadd231ps %%zmm26, %%zmm28, %%zmm22\n\t" "vfmadd231ps %%zmm27, %%zmm28, %%zmm23\n\t"

        // Advance pointers by 2 k-steps
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

    // Load 24 accumulators from stack buffer
    __m512 c00=c_acc[0],  c01=c_acc[1],  c02=c_acc[2],  c03=c_acc[3];
    __m512 c10=c_acc[4],  c11=c_acc[5],  c12=c_acc[6],  c13=c_acc[7];
    __m512 c20=c_acc[8],  c21=c_acc[9],  c22=c_acc[10], c23=c_acc[11];
    __m512 c30=c_acc[12], c31=c_acc[13], c32=c_acc[14], c33=c_acc[15];
    __m512 c40=c_acc[16], c41=c_acc[17], c42=c_acc[18], c43=c_acc[19];
    __m512 c50=c_acc[20], c51=c_acc[21], c52=c_acc[22], c53=c_acc[23];

    // Remainder (0 or 1 k-steps)
    if (k & 1) {
        __m512 b0=_mm512_loadu_ps(bp), b1=_mm512_loadu_ps(bp+16),
               b2=_mm512_loadu_ps(bp+32), b3=_mm512_loadu_ps(bp+48);
        __m512 a;
        a=_mm512_set1_ps(*a0); c00=_mm512_fmadd_ps(a,b0,c00); c01=_mm512_fmadd_ps(a,b1,c01); c02=_mm512_fmadd_ps(a,b2,c02); c03=_mm512_fmadd_ps(a,b3,c03);
        a=_mm512_set1_ps(*a1); c10=_mm512_fmadd_ps(a,b0,c10); c11=_mm512_fmadd_ps(a,b1,c11); c12=_mm512_fmadd_ps(a,b2,c12); c13=_mm512_fmadd_ps(a,b3,c13);
        a=_mm512_set1_ps(*a2); c20=_mm512_fmadd_ps(a,b0,c20); c21=_mm512_fmadd_ps(a,b1,c21); c22=_mm512_fmadd_ps(a,b2,c22); c23=_mm512_fmadd_ps(a,b3,c23);
        a=_mm512_set1_ps(*a3); c30=_mm512_fmadd_ps(a,b0,c30); c31=_mm512_fmadd_ps(a,b1,c31); c32=_mm512_fmadd_ps(a,b2,c32); c33=_mm512_fmadd_ps(a,b3,c33);
        a=_mm512_set1_ps(*a4); c40=_mm512_fmadd_ps(a,b0,c40); c41=_mm512_fmadd_ps(a,b1,c41); c42=_mm512_fmadd_ps(a,b2,c42); c43=_mm512_fmadd_ps(a,b3,c43);
        a=_mm512_set1_ps(*a5); c50=_mm512_fmadd_ps(a,b0,c50); c51=_mm512_fmadd_ps(a,b1,c51); c52=_mm512_fmadd_ps(a,b2,c52); c53=_mm512_fmadd_ps(a,b3,c53);
    }

    // ── Epilogue ──
    if (beta != 0.0f) {
        __m512 bv = _mm512_set1_ps(beta);
        c00=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C),c00);       c01=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+16),c01);
        c02=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+32),c02);    c03=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+48),c03);
        c10=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+ldc),c10);   c11=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+ldc+16),c11);
        c12=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+ldc+32),c12);c13=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+ldc+48),c13);
        c20=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+2*ldc),c20);   c21=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+2*ldc+16),c21);
        c22=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+2*ldc+32),c22);c23=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+2*ldc+48),c23);
        c30=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+3*ldc),c30);   c31=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+3*ldc+16),c31);
        c32=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+3*ldc+32),c32);c33=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+3*ldc+48),c33);
        c40=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+4*ldc),c40);   c41=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+4*ldc+16),c41);
        c42=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+4*ldc+32),c42);c43=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+4*ldc+48),c43);
        c50=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+5*ldc),c50);   c51=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+5*ldc+16),c51);
        c52=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+5*ldc+32),c52);c53=_mm512_fmadd_ps(bv,_mm512_loadu_ps(C+5*ldc+48),c53);
    }
    if (bias) {
        __m512 bv0=_mm512_loadu_ps(bias), bv1=_mm512_loadu_ps(bias+16),
               bv2=_mm512_loadu_ps(bias+32), bv3=_mm512_loadu_ps(bias+48);
        c00=_mm512_add_ps(c00,bv0); c01=_mm512_add_ps(c01,bv1); c02=_mm512_add_ps(c02,bv2); c03=_mm512_add_ps(c03,bv3);
        c10=_mm512_add_ps(c10,bv0); c11=_mm512_add_ps(c11,bv1); c12=_mm512_add_ps(c12,bv2); c13=_mm512_add_ps(c13,bv3);
        c20=_mm512_add_ps(c20,bv0); c21=_mm512_add_ps(c21,bv1); c22=_mm512_add_ps(c22,bv2); c23=_mm512_add_ps(c23,bv3);
        c30=_mm512_add_ps(c30,bv0); c31=_mm512_add_ps(c31,bv1); c32=_mm512_add_ps(c32,bv2); c33=_mm512_add_ps(c33,bv3);
        c40=_mm512_add_ps(c40,bv0); c41=_mm512_add_ps(c41,bv1); c42=_mm512_add_ps(c42,bv2); c43=_mm512_add_ps(c43,bv3);
        c50=_mm512_add_ps(c50,bv0); c51=_mm512_add_ps(c51,bv1); c52=_mm512_add_ps(c52,bv2); c53=_mm512_add_ps(c53,bv3);
    }
    if (fused_op != fused_postop_t::none) {
        c00=apply_fused_postop(c00,fused_op); c01=apply_fused_postop(c01,fused_op); c02=apply_fused_postop(c02,fused_op); c03=apply_fused_postop(c03,fused_op);
        c10=apply_fused_postop(c10,fused_op); c11=apply_fused_postop(c11,fused_op); c12=apply_fused_postop(c12,fused_op); c13=apply_fused_postop(c13,fused_op);
        c20=apply_fused_postop(c20,fused_op); c21=apply_fused_postop(c21,fused_op); c22=apply_fused_postop(c22,fused_op); c23=apply_fused_postop(c23,fused_op);
        c30=apply_fused_postop(c30,fused_op); c31=apply_fused_postop(c31,fused_op); c32=apply_fused_postop(c32,fused_op); c33=apply_fused_postop(c33,fused_op);
        c40=apply_fused_postop(c40,fused_op); c41=apply_fused_postop(c41,fused_op); c42=apply_fused_postop(c42,fused_op); c43=apply_fused_postop(c43,fused_op);
        c50=apply_fused_postop(c50,fused_op); c51=apply_fused_postop(c51,fused_op); c52=apply_fused_postop(c52,fused_op); c53=apply_fused_postop(c53,fused_op);
    }
    _mm512_storeu_ps(C,c00);          _mm512_storeu_ps(C+16,c01);        _mm512_storeu_ps(C+32,c02);        _mm512_storeu_ps(C+48,c03);
    _mm512_storeu_ps(C+ldc,c10);      _mm512_storeu_ps(C+ldc+16,c11);    _mm512_storeu_ps(C+ldc+32,c12);    _mm512_storeu_ps(C+ldc+48,c13);
    _mm512_storeu_ps(C+2*ldc,c20);    _mm512_storeu_ps(C+2*ldc+16,c21);  _mm512_storeu_ps(C+2*ldc+32,c22);  _mm512_storeu_ps(C+2*ldc+48,c23);
    _mm512_storeu_ps(C+3*ldc,c30);    _mm512_storeu_ps(C+3*ldc+16,c31);  _mm512_storeu_ps(C+3*ldc+32,c32);  _mm512_storeu_ps(C+3*ldc+48,c33);
    _mm512_storeu_ps(C+4*ldc,c40);    _mm512_storeu_ps(C+4*ldc+16,c41);  _mm512_storeu_ps(C+4*ldc+32,c42);  _mm512_storeu_ps(C+4*ldc+48,c43);
    _mm512_storeu_ps(C+5*ldc,c50);    _mm512_storeu_ps(C+5*ldc+16,c51);  _mm512_storeu_ps(C+5*ldc+32,c52);  _mm512_storeu_ps(C+5*ldc+48,c53);
}

// ============================================================================
// Template AVX-512 FP32 microkernel: MR x (NV*16) with fused bias
//
//   pa, a_stride: A tile pointer + row stride
//   pb:           packed B tile (k-contiguous, NV*16 wide)
//   C, ldc:       output tile
//   k:            K-dimension for this block
//   beta:         0.0 or 1.0
//   bias:         pointer to NV*16 bias values at this column offset,
//                 or nullptr if no bias / not final K-block
//
// Epilogue order: acc += beta * C_old, acc += bias, acc = max(acc,0) if relu, store.
// K-loop unrolled 4x. All MR/NV loops unrolled at compile time.
// ============================================================================
template<int MR, int NV>
__attribute__((target("avx512f,fma"), noinline))
void avx512_ukernel(
    const float * __restrict__ pa, int a_stride,
    const float * __restrict__ pb, int b_stride,
    float * __restrict__ C, int ldc, int k, float beta,
    const float * __restrict__ bias, fused_postop_t fused_op) {

    __m512 acc[MR][NV];
    for (int m = 0; m < MR; ++m)
        for (int v = 0; v < NV; ++v)
            acc[m][v] = _mm512_setzero_ps();

    const float *a_row[MR];
    for (int m = 0; m < MR; ++m)
        a_row[m] = pa + m * a_stride;

    // K-loop: 4x unrolled for instruction-level parallelism.
    // bv[NV] is scoped per u-step so the compiler reuses registers.
    constexpr int KUNROLL = 4;

    int kk = 0;
    for (; kk + (KUNROLL - 1) < k; kk += KUNROLL) {
        for (int u = 0; u < KUNROLL; ++u) {
            const float *bp = pb + (kk + u) * b_stride;
            __m512 bv[NV];
            for (int v = 0; v < NV; ++v)
                bv[v] = _mm512_loadu_ps(bp + v * 16);
            for (int m = 0; m < MR; ++m) {
                __m512 a = _mm512_set1_ps(a_row[m][kk + u]);
                for (int v = 0; v < NV; ++v)
                    acc[m][v] = _mm512_fmadd_ps(a, bv[v], acc[m][v]);
            }
        }
    }
    for (; kk < k; ++kk) {
        __m512 bv[NV];
        for (int v = 0; v < NV; ++v)
            bv[v] = _mm512_loadu_ps(pb + kk * b_stride + v * 16);
        for (int m = 0; m < MR; ++m) {
            __m512 a = _mm512_set1_ps(a_row[m][kk]);
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_fmadd_ps(a, bv[v], acc[m][v]);
        }
    }

    // Epilogue: beta → bias → activation → store (kept tight for code gen)
    if (beta != 0.0f) {
        __m512 bv = _mm512_set1_ps(beta);
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_fmadd_ps(
                    bv, _mm512_loadu_ps(C + m * ldc + v * 16), acc[m][v]);
    }

    if (bias) {
        __m512 bias_v[NV];
        for (int v = 0; v < NV; ++v)
            bias_v[v] = _mm512_loadu_ps(bias + v * 16);
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = _mm512_add_ps(acc[m][v], bias_v[v]);
    }

    if (fused_op != fused_postop_t::none) {
        for (int m = 0; m < MR; ++m)
            for (int v = 0; v < NV; ++v)
                acc[m][v] = apply_fused_postop(acc[m][v], fused_op);
    }

    for (int m = 0; m < MR; ++m)
        for (int v = 0; v < NV; ++v)
            _mm512_storeu_ps(C + m * ldc + v * 16, acc[m][v]);
}

// ============================================================================
// Explicit instantiations — ONLY active kernels
//
// GEMM high-throughput:  MR=6, NR=64 (NV=4) — 24 acc, relu/no-act only
// GEMM safe:             MR=6, NR=32 (NV=2) — 12 acc, complex activation
// Small N fallback:      MR=6, NR=16 (NV=1) — 6 acc, minimal
// ============================================================================
template void avx512_ukernel<6,4>(const float*, int, const float*, int, float*, int, int, float, const float*, fused_postop_t);
template void avx512_ukernel<6,2>(const float*, int, const float*, int, float*, int, int, float, const float*, fused_postop_t);
template void avx512_ukernel<6,1>(const float*, int, const float*, int, float*, int, int, float, const float*, fused_postop_t);

// ============================================================================
// AVX-512 masked tail kernel with fused bias
// ============================================================================
__attribute__((target("avx512f,avx512bw,fma"), noinline))
void avx512_tail_kernel(
    const float * __restrict__ pa, int a_stride,
    const float * __restrict__ pb, int b_stride,
    float * __restrict__ C, int ldc, int k,
    int mr_count, int nr_count, float beta,
    const float * __restrict__ bias, fused_postop_t fused_op) {

    const int full_vecs = nr_count / 16;
    const int rem = nr_count & 15;
    const __mmask16 rem_mask = rem ? static_cast<__mmask16>((1u << rem) - 1)
                                   : static_cast<__mmask16>(0);

    __m512 acc[12][4];
    const int nv = (nr_count + 15) / 16;
    for (int m = 0; m < mr_count; ++m)
        for (int v = 0; v < nv; ++v)
            acc[m][v] = _mm512_setzero_ps();

    for (int kk = 0; kk < k; ++kk) {
        __m512 bv[4];
        for (int v = 0; v < full_vecs; ++v)
            bv[v] = _mm512_loadu_ps(pb + kk * b_stride + v * 16);
        if (rem)
            bv[full_vecs] = _mm512_maskz_loadu_ps(
                rem_mask, pb + kk * b_stride + full_vecs * 16);
        for (int m = 0; m < mr_count; ++m) {
            __m512 a = _mm512_set1_ps(pa[m * a_stride + kk]);
            for (int v = 0; v < full_vecs; ++v)
                acc[m][v] = _mm512_fmadd_ps(a, bv[v], acc[m][v]);
            if (rem)
                acc[m][full_vecs] = _mm512_fmadd_ps(a, bv[full_vecs],
                                                     acc[m][full_vecs]);
        }
    }

    // ── Epilogue: beta → bias → binary → activation → store ──

    // 1. Beta: acc += beta * C_old
    if (beta != 0.0f) {
        __m512 bv = _mm512_set1_ps(beta);
        for (int m = 0; m < mr_count; ++m) {
            for (int v = 0; v < full_vecs; ++v)
                acc[m][v] = _mm512_fmadd_ps(
                    bv, _mm512_loadu_ps(C + m * ldc + v * 16), acc[m][v]);
            if (rem)
                acc[m][full_vecs] = _mm512_fmadd_ps(
                    bv, _mm512_maskz_loadu_ps(rem_mask, C + m * ldc + full_vecs * 16),
                    acc[m][full_vecs]);
        }
    }

    // 2. Fused bias
    if (bias) {
        for (int v = 0; v < full_vecs; ++v) {
            __m512 bv = _mm512_loadu_ps(bias + v * 16);
            for (int m = 0; m < mr_count; ++m)
                acc[m][v] = _mm512_add_ps(acc[m][v], bv);
        }
        if (rem) {
            __m512 bv = _mm512_maskz_loadu_ps(rem_mask, bias + full_vecs * 16);
            for (int m = 0; m < mr_count; ++m)
                acc[m][full_vecs] = _mm512_add_ps(acc[m][full_vecs], bv);
        }
    }

    // 3. Fused activation
    if (fused_op != fused_postop_t::none) {
        for (int m = 0; m < mr_count; ++m)
            for (int v = 0; v < nv; ++v)
                acc[m][v] = apply_fused_postop(acc[m][v], fused_op);
    }

    // 5. Store
    for (int m = 0; m < mr_count; ++m) {
        for (int v = 0; v < full_vecs; ++v)
            _mm512_storeu_ps(C + m * ldc + v * 16, acc[m][v]);
        if (rem)
            _mm512_mask_storeu_ps(C + m * ldc + full_vecs * 16,
                                  rem_mask, acc[m][full_vecs]);
    }
}

// ============================================================================
// Scalar fallback with fused bias
// ============================================================================
void scalar_microkernel(
    const float *pa, int a_stride, const float *pb, int b_stride,
    float *C, int ldc, int k,
    int mr_count, int nr_count, float beta,
    const float *bias, fused_postop_t fused_op) {

    if (beta == 0.0f) {
        for (int m = 0; m < mr_count; ++m)
            for (int n = 0; n < nr_count; ++n)
                C[m * ldc + n] = 0.0f;
    }
    for (int kk = 0; kk < k; ++kk)
        for (int m = 0; m < mr_count; ++m) {
            float a = pa[m * a_stride + kk];
            for (int n = 0; n < nr_count; ++n)
                C[m * ldc + n] += a * pb[kk * b_stride + n];
        }
    // Bias
    if (bias) {
        for (int m = 0; m < mr_count; ++m)
            for (int n = 0; n < nr_count; ++n)
                C[m * ldc + n] += bias[n];
    }
    // Scalar fused activation (fallback path)
    if (fused_op == fused_postop_t::relu) {
        for (int m = 0; m < mr_count; ++m)
            for (int n = 0; n < nr_count; ++n)
                if (C[m * ldc + n] < 0.0f) C[m * ldc + n] = 0.0f;
    } else if (fused_op == fused_postop_t::gelu_tanh) {
        for (int m = 0; m < mr_count; ++m)
            for (int n = 0; n < nr_count; ++n) {
                float &v = C[m * ldc + n];
                float x3 = v * v * v;
                v = 0.5f * v * (1.0f + std::tanh(0.7978845608028654f * (v + 0.044715f * x3)));
            }
    } else if (fused_op == fused_postop_t::gelu_erf) {
        for (int m = 0; m < mr_count; ++m)
            for (int n = 0; n < nr_count; ++n) {
                float &v = C[m * ldc + n];
                v = 0.5f * v * (1.0f + std::erf(v * 0.7071067811865476f));
            }
    } else if (fused_op == fused_postop_t::sigmoid) {
        for (int m = 0; m < mr_count; ++m)
            for (int n = 0; n < nr_count; ++n)
                C[m * ldc + n] = 1.0f / (1.0f + std::exp(-C[m * ldc + n]));
    } else if (fused_op == fused_postop_t::tanh_op) {
        for (int m = 0; m < mr_count; ++m)
            for (int n = 0; n < nr_count; ++n)
                C[m * ldc + n] = std::tanh(C[m * ldc + n]);
    } else if (fused_op == fused_postop_t::swish) {
        for (int m = 0; m < mr_count; ++m)
            for (int n = 0; n < nr_count; ++n) {
                float v = C[m * ldc + n];
                C[m * ldc + n] = v / (1.0f + std::exp(-v));
            }
    }
}

// ============================================================================
// Kernel dispatch
// ============================================================================
/// Select FP32 microkernel — asm kernel for NR=64 (primary hot path).
__attribute__((target("avx512f,fma")))
ukernel_fn_t select_ukernel([[maybe_unused]] int MR, int NR) {
    switch (NR) {
    case 64: return avx512_ukernel_6x64_asm; // Hand-scheduled asm, 24 acc
    case 32: return avx512_ukernel<6, 2>;     // Template fallback (NV=2)
    case 16: return avx512_ukernel<6, 1>;     // Small N fallback
    }
    return avx512_ukernel<6, 1>;
}

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
