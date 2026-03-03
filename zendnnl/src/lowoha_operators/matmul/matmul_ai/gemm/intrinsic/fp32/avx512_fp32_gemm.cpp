/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "lowoha_operators/matmul/matmul_ai/gemm/intrinsic/fp32/avx512_fp32_gemm.hpp"
#include "lowoha_operators/matmul/matmul_ai/gemm/gemm_planner.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_ai/common/postop.hpp"
#include "operators/matmul/matmul_config.hpp"
#include "common/zendnnl_global.hpp"

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
// Hand-scheduled FP32 6×64 microkernel (inline assembly K-loop)
//
// Register map (uses ALL 32 ZMM):
//   zmm0  - zmm23  : 24 accumulators (6 rows × 4 cols)
//   zmm24 - zmm27  : B tile loads (4 × 16 FP32 = 64 columns)
//   zmm28           : A broadcast (reused per row)
//   zmm29 - zmm31  : free for epilogue
//
// K-loop: 2× unrolled. 24 FMA per k-step = 12 cycles (FMA-bound).
// 2 k-steps = 24 cycles + 1 branch = 25 cycles, 1536 FLOPs → 61 FLOPs/cyc.
// ============================================================================
__attribute__((target("avx512f,fma"), noinline))
static void avx512_ukernel_6x64_asm(
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
static void avx512_ukernel(
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
static void avx512_tail_kernel(
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
static void scalar_microkernel(
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
    }
}

// ============================================================================
// Kernel dispatch
// ============================================================================
using ukernel_fn_t = void (*)(const float *, int, const float *, int,
                              float *, int, int, float, const float *,
                              fused_postop_t);

// Vectorized alpha scaling
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

/// Select FP32 microkernel — asm kernel for NR=64 (primary hot path).
static ukernel_fn_t select_ukernel([[maybe_unused]] int MR, int NR) {
    switch (NR) {
    case 64: return avx512_ukernel_6x64_asm; // Hand-scheduled asm, 24 acc
    case 32: return avx512_ukernel<6, 2>;     // Template fallback (NV=2)
    case 16: return avx512_ukernel<6, 1>;     // Small N fallback
    }
    return avx512_ukernel<6, 1>;
}

// ============================================================================
// Packing helpers
// ============================================================================

static inline void pack_a_block(
    const float *A_src, float *pack_buf,
    int ic, int pc, int M, int K, int lda, bool transA,
    int mb, int kb, int MR) {

    const int m_actual = std::min(mb, M - ic);
    const int k_actual = std::min(kb, K - pc);
    const int m_panels = (m_actual + MR - 1) / MR;

    if (!transA) {
        const float *A_block = A_src + ic * lda + pc;
        for (int ip = 0; ip < m_panels; ++ip) {
            const int i0 = ip * MR;
            const int mr = std::min(MR, m_actual - i0);
            float *dst = pack_buf + ip * MR * k_actual;
            for (int m = 0; m < mr; ++m)
                std::memcpy(dst + m * k_actual,
                            A_block + (i0 + m) * lda,
                            k_actual * sizeof(float));
            for (int m = mr; m < MR; ++m)
                std::memset(dst + m * k_actual, 0, k_actual * sizeof(float));
        }
    } else {
        const float *A_block = A_src + pc * lda + ic;
        for (int ip = 0; ip < m_panels; ++ip) {
            const int i0 = ip * MR;
            const int mr = std::min(MR, m_actual - i0);
            float *dst = pack_buf + ip * MR * k_actual;
            for (int m = 0; m < mr; ++m)
                for (int kk = 0; kk < k_actual; ++kk)
                    dst[m * k_actual + kk] = A_block[kk * lda + (i0 + m)];
            for (int m = mr; m < MR; ++m)
                std::memset(dst + m * k_actual, 0, k_actual * sizeof(float));
        }
    }
}

// ============================================================================
// Aligned buffer
// ============================================================================
struct AlignedBuffer {
    float *ptr;
    explicit AlignedBuffer(size_t n)
        : ptr(n > 0 ? static_cast<float *>(
              std::aligned_alloc(64, ((n * sizeof(float) + 63) & ~size_t(63))))
              : nullptr) {}
    ~AlignedBuffer() { if (ptr) std::free(ptr); }
    AlignedBuffer(const AlignedBuffer &) = delete;
    AlignedBuffer &operator=(const AlignedBuffer &) = delete;
};

// ============================================================================
// Main 5-loop BLIS-style thread loop (internal)
// ============================================================================

static void ai_thread_loop(
    const GemmDescriptor &desc,
    const BlockPlan &plan,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params,
    const PrepackedWeight *prepacked_b) {

    const float *A = static_cast<const float *>(src);
    const float *B = static_cast<const float *>(weight);
    float *C       = static_cast<float *>(dst);
    const float *bias_f = static_cast<const float *>(bias);

    const int M = desc.M, N = desc.N, K = desc.K;
    const int lda = desc.lda, ldb = desc.ldb, ldc = desc.ldc;
    const bool transA = desc.transA;
    const float alpha = desc.alpha;
    // When alpha != 1 and beta != 0, pass beta/alpha to the microkernel.
    // After scale_tile: alpha*(A*B + (beta/alpha)*C_old) = alpha*A*B + beta*C_old.
    const float beta  = (desc.alpha != 1.0f && desc.beta != 0.0f)
                        ? (desc.beta / desc.alpha) : desc.beta;
    const int MB = plan.MB, NB = plan.NB, KB = plan.KB;
    const int MR = plan.MR, NR = plan.NR;
    const int num_threads = plan.num_threads;
    const bool use_avx512 = uarch.avx512f;
    const bool has_bias = (bias_f != nullptr);

    // Detect fusable post-op (first eltwise op in the chain).
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
        if (i != fused_idx) remaining_postops.push_back(params.postop_[i]);
    }
    const bool has_remaining_postops = !remaining_postops.empty();

    // Pack controls (read once from singleton, cached across calls).
    // Auto-enable A packing when row stride exceeds L1 stride prefetcher
    // limit (~4KB on Zen4/5). For lda=3584 FP32: 14KB stride → pack.
    const bool do_pack_a = transA
                           || (lda * static_cast<int>(sizeof(float)) > 4096);

    ukernel_fn_t hot_ukernel = use_avx512 ? select_ukernel(MR, NR) : nullptr;

    const int jc_tiles = (N + NB - 1) / NB;
    const size_t pa_elems = static_cast<size_t>(((MB + MR - 1) / MR) * MR) * KB;

    auto run_loop = [&](float *pa_buf) {
        for (int jc_idx = 0; jc_idx < jc_tiles; ++jc_idx) {
            const int jc = jc_idx * NB;
            const int nb_act = std::min(NB, N - jc);

            for (int pc = 0; pc < K; pc += KB) {
                const int kb_act = std::min(KB, K - pc);
                const bool is_last_k = (pc + kb_act >= K);

                // B data: prepacked (panel-wide K-contiguous) or direct access
                const bool b_use_prepacked = (prepacked_b != nullptr);

                for (int ic = 0; ic < M; ic += MB) {
                    const int mb_act = std::min(MB, M - ic);

                    // -- Pack A (or skip) --
                    const float *a_base;
                    if (do_pack_a) {
                        pack_a_block(A, pa_buf, ic, pc, M, K, lda, transA,
                                     mb_act, kb_act, MR);
                        a_base = pa_buf;
                    } else {
                        a_base = A + ic * lda + pc;
                    }

                    const float tile_beta = (pc == 0) ? beta : 1.0f;
                    const int m_panels = (mb_act + MR - 1) / MR;

                    for (int jr = 0; jr < nb_act; jr += NR) {
                        const int nr_act = std::min(NR, nb_act - jr);
                        const bool full_nr = (nr_act == NR);

                        const float *pb_tile;
                        int pb_stride;
                        bool panel_crossing = false;
                        int nr_part1 = 0;

                        if (b_use_prepacked) {
                            int col = jc + jr;
                            int panel_idx = col / NR_PACK;
                            int in_panel_off = col % NR_PACK;
                            if (in_panel_off + nr_act <= NR_PACK) {
                                pb_tile = prepacked_b->get_panel(pc, panel_idx) + in_panel_off;
                                pb_stride = NR_PACK;
                            } else {
                                panel_crossing = true;
                                nr_part1 = NR_PACK - in_panel_off;
                                pb_tile = prepacked_b->get_panel(pc, panel_idx) + in_panel_off;
                                pb_stride = NR_PACK;
                            }
                        } else {
                            pb_tile = B + pc * ldb + (jc + jr);
                            pb_stride = ldb;
                        }

                        // Fuse bias/binary/activation only when alpha==1 (epilogue order is correct).
                        // When alpha!=1, bias/binary/activation must come AFTER scale_tile.
                        const bool can_fuse = (alpha == 1.0f);
                        const float *tile_bias =
                            (has_bias && is_last_k && can_fuse) ? (bias_f + jc + jr) : nullptr;
                        const fused_postop_t tile_fop =
                            (is_last_k && can_fuse) ? fused_op : fused_postop_t::none;

                        for (int ip = 0; ip < m_panels; ++ip) {
                            const int ir = ip * MR;
                            const int mr_act = std::min(MR, mb_act - ir);
                            float *Ct = C + (ic + ir) * ldc + (jc + jr);

                            const float *pa;
                            int pa_stride;
                            if (do_pack_a) {
                                pa = a_base + ip * MR * kb_act;
                                pa_stride = kb_act;
                            } else {
                                pa = a_base + ir * lda;
                                pa_stride = lda;
                            }

                            if (!panel_crossing) {
                                if (hot_ukernel && full_nr && mr_act == MR) {
                                    hot_ukernel(pa, pa_stride, pb_tile, pb_stride,
                                                Ct, ldc, kb_act, tile_beta,
                                                tile_bias, tile_fop);
                                } else if (use_avx512) {
                                    avx512_tail_kernel(pa, pa_stride, pb_tile, pb_stride,
                                                       Ct, ldc, kb_act, mr_act, nr_act,
                                                       tile_beta, tile_bias, tile_fop);
                                } else {
                                    scalar_microkernel(pa, pa_stride, pb_tile, pb_stride,
                                                       Ct, ldc, kb_act, mr_act, nr_act,
                                                       tile_beta, tile_bias, tile_fop);
                                }
                            } else {
                                int nr_part2 = nr_act - nr_part1;

                                // Panel crossing only happens with prepacked B
                                int col = jc + jr;
                                int pidx = col / NR_PACK;

                                const float *pb1 = prepacked_b->get_panel(pc, pidx) + (col % NR_PACK);
                                const float *pb2 = prepacked_b->get_panel(pc, pidx + 1);

                                const float *bias1 = tile_bias;
                                avx512_tail_kernel(pa, pa_stride, pb1, NR_PACK,
                                                   Ct, ldc, kb_act, mr_act, nr_part1,
                                                   tile_beta, bias1, tile_fop);

                                const float *bias2 = tile_bias ? (tile_bias + nr_part1) : nullptr;
                                avx512_tail_kernel(pa, pa_stride, pb2, NR_PACK,
                                                   Ct + nr_part1, ldc, kb_act, mr_act, nr_part2,
                                                   tile_beta, bias2, tile_fop);
                            }
                        }
                    }

                    // Post-K-block: alpha scaling, unfused postops
                    if (is_last_k) {
                        float *Ctile = C + ic * ldc + jc;
                        if (alpha != 1.0f) {
                            scale_tile(Ctile, ldc, mb_act, nb_act, alpha);
                            if (has_bias)
                                apply_postops_tile(Ctile, ldc, mb_act, nb_act,
                                                   jc, ic, bias_f, {});
                            if (fused_op != fused_postop_t::none && fused_idx >= 0)
                                apply_postops_tile(Ctile, ldc, mb_act, nb_act,
                                                   jc, ic, nullptr,
                                                   {params.postop_[fused_idx]});
                        }
                        if (has_remaining_postops) {
                            apply_postops_tile(Ctile, ldc, mb_act, nb_act,
                                               jc, ic, nullptr, remaining_postops);
                        }
                    }
                } // ic
            } // pc
        } // jc
    }; // end run_loop lambda (used for single-thread only)

    if (num_threads <= 1) {
        // Single-thread: skip OpenMP, use thread-local reusable buffer
        static thread_local float *s_pa = nullptr;
        static thread_local size_t s_pa_cap = 0;
        if (do_pack_a && s_pa_cap < pa_elems) {
            std::free(s_pa);
            s_pa = static_cast<float *>(
                std::aligned_alloc(64, ((pa_elems * 4 + 63) & ~size_t(63))));
            s_pa_cap = s_pa ? pa_elems : 0;
        }
        run_loop((do_pack_a && s_pa) ? s_pa : nullptr);
    } else {
        const int ic_tiles = (M + MB - 1) / MB;
        const int total_2d_tiles = ic_tiles * jc_tiles;
        const bool b_use_prepacked = (prepacked_b != nullptr);

        // BRGEMM-style: tile-outer, K-inner. Each thread processes ALL
        // K-blocks for its tiles, eliminating inter-K-block barriers.
        const int active_threads = std::min(num_threads, total_2d_tiles);

        const bool skip_pack_a = do_pack_a
            && (static_cast<size_t>(M) * K * sizeof(float)
                <= static_cast<size_t>(uarch.l2_bytes));
        const bool actual_pack_a = do_pack_a && !skip_pack_a;

        #pragma omp parallel num_threads(active_threads)
        {
            static thread_local float *tl_pa = nullptr;
            static thread_local size_t tl_pa_cap = 0;

            #pragma omp for schedule(static)
            for (int tile_idx = 0; tile_idx < total_2d_tiles; ++tile_idx) {
                const int ic_idx = tile_idx / jc_tiles;
                const int jc_idx = tile_idx % jc_tiles;
                const int ic = ic_idx * MB;
                const int jc = jc_idx * NB;
                const int mb_act = std::min(MB, M - ic);
                const int nb_act = std::min(NB, N - jc);
                const int m_panels = (mb_act + MR - 1) / MR;

                for (int pc = 0; pc < K; pc += KB) {
                    const int kb_act = std::min(KB, K - pc);
                    const bool is_last_k = (pc + kb_act >= K);
                    const float tile_beta = (pc == 0) ? beta : 1.0f;

                    const float *a_base;
                    int a_stride_base;
                    if (actual_pack_a) {
                        size_t need = static_cast<size_t>(
                            ((mb_act + MR - 1) / MR) * MR) * kb_act;
                        if (tl_pa_cap < need) {
                            std::free(tl_pa);
                            tl_pa = static_cast<float *>(std::aligned_alloc(
                                64, ((need * 4 + 63) & ~size_t(63))));
                            tl_pa_cap = tl_pa ? need : 0;
                        }
                        if (tl_pa) {
                            pack_a_block(A, tl_pa, ic, pc, M, K, lda, transA,
                                         mb_act, kb_act, MR);
                            a_base = tl_pa;
                            a_stride_base = kb_act;
                        } else {
                            a_base = A + ic * lda + pc;
                            a_stride_base = lda;
                        }
                    } else {
                        a_base = A + ic * lda + pc;
                        a_stride_base = lda;
                    }

                    for (int jr = 0; jr < nb_act; jr += NR) {
                        const int nr_act = std::min(NR, nb_act - jr);
                        const bool full_nr = (nr_act == NR);

                        const float *pb_tile;
                        int pb_stride;
                        bool panel_crossing = false;
                        int nr_part1 = 0;

                        if (b_use_prepacked) {
                            int col = jc + jr;
                            int panel_idx = col / NR_PACK;
                            int in_panel_off = col % NR_PACK;
                            if (in_panel_off + nr_act <= NR_PACK) {
                                pb_tile = prepacked_b->get_panel(pc, panel_idx) + in_panel_off;
                                pb_stride = NR_PACK;
                            } else {
                                panel_crossing = true;
                                nr_part1 = NR_PACK - in_panel_off;
                                pb_tile = prepacked_b->get_panel(pc, panel_idx) + in_panel_off;
                                pb_stride = NR_PACK;
                            }
                        } else {
                            pb_tile = B + pc * ldb + (jc + jr);
                            pb_stride = ldb;
                        }

                        const bool can_fuse = (alpha == 1.0f);
                        const float *tile_bias =
                            (has_bias && is_last_k && can_fuse) ? (bias_f + jc + jr) : nullptr;
                        const fused_postop_t tile_fop =
                            (is_last_k && can_fuse) ? fused_op : fused_postop_t::none;

                        for (int ip = 0; ip < m_panels; ++ip) {
                            const int ir = ip * MR;
                            const int mr_act = std::min(MR, mb_act - ir);
                            float *Ct = C + (ic + ir) * ldc + (jc + jr);

                            const float *pa;
                            int pa_stride;
                            if (actual_pack_a) {
                                pa = a_base + ip * MR * kb_act;
                                pa_stride = kb_act;
                            } else {
                                pa = a_base + ir * a_stride_base;
                                pa_stride = a_stride_base;
                            }

                            if (!panel_crossing) {
                                if (hot_ukernel && full_nr && mr_act == MR) {
                                    hot_ukernel(pa, pa_stride, pb_tile, pb_stride,
                                                Ct, ldc, kb_act, tile_beta,
                                                tile_bias, tile_fop);
                                } else if (use_avx512) {
                                    avx512_tail_kernel(pa, pa_stride, pb_tile, pb_stride,
                                                       Ct, ldc, kb_act, mr_act, nr_act,
                                                       tile_beta, tile_bias, tile_fop);
                                } else {
                                    scalar_microkernel(pa, pa_stride, pb_tile, pb_stride,
                                                       Ct, ldc, kb_act, mr_act, nr_act,
                                                       tile_beta, tile_bias, tile_fop);
                                }
                            } else {
                                int nr_part2 = nr_act - nr_part1;
                                int col = jc + jr;
                                int pidx = col / NR_PACK;

                                const float *pb1 = prepacked_b->get_panel(pc, pidx) + (col % NR_PACK);
                                const float *pb2 = prepacked_b->get_panel(pc, pidx + 1);

                                avx512_tail_kernel(pa, pa_stride, pb1, NR_PACK,
                                                   Ct, ldc, kb_act, mr_act, nr_part1,
                                                   tile_beta, tile_bias, tile_fop);

                                const float *bias2 = tile_bias ? (tile_bias + nr_part1) : nullptr;
                                avx512_tail_kernel(pa, pa_stride, pb2, NR_PACK,
                                                   Ct + nr_part1, ldc, kb_act, mr_act, nr_part2,
                                                   tile_beta, bias2, tile_fop);
                            }
                        }
                    }

                    if (is_last_k) {
                        float *Ctile = C + ic * ldc + jc;
                        if (alpha != 1.0f) {
                            scale_tile(Ctile, ldc, mb_act, nb_act, alpha);
                            if (has_bias)
                                apply_postops_tile(Ctile, ldc, mb_act, nb_act,
                                                   jc, ic, bias_f, {});
                            if (fused_op != fused_postop_t::none && fused_idx >= 0)
                                apply_postops_tile(Ctile, ldc, mb_act, nb_act,
                                                   jc, ic, nullptr,
                                                   {params.postop_[fused_idx]});
                        }
                        if (has_remaining_postops) {
                            apply_postops_tile(Ctile, ldc, mb_act, nb_act,
                                               jc, ic, nullptr, remaining_postops);
                        }
                    }
                } // K-loop (BRGEMM)
            } // omp for
        } // omp parallel
    }
}

// ============================================================================
// GEMM execute: plan caching + B prepacking + thread loop
// ============================================================================

void gemm_execute(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;
    const bool transB = desc.transB;
    const bool is_weights_const = desc.is_weights_const;

    // Classify activation post-ops for NR selection.
    // "Simple" activations (relu) need 1 ZMM scratch → NV=4 (NR=64) safe
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

    // ── 1. Block plan (cached per shape, zero cost on repeated calls) ──
    static thread_local struct {
        int M, N, K, threads; bool transA, transB;
        BlockPlan plan;
    } s_plan_cache = {0, 0, 0, 0, false, false, {}};

    BlockPlan plan;
    if (s_plan_cache.M == M && s_plan_cache.N == N && s_plan_cache.K == K &&
        s_plan_cache.threads == desc.num_threads &&
        s_plan_cache.transA == desc.transA &&
        s_plan_cache.transB == transB) {
        plan = s_plan_cache.plan;
    } else {
        plan = plan_blocks(desc, uarch);

        // KB: minimize K-blocks while keeping B panel in L2.
        if (desc.num_threads <= 1) {
            int b_bytes = K * NR_PACK * static_cast<int>(sizeof(float));
            if (b_bytes <= uarch.l2_bytes / 2)
                plan.KB = K;
        } else {
            int kb_b = (uarch.l2_bytes / 2)
                       / (NR_PACK * static_cast<int>(sizeof(float)));
            int kb_a = (uarch.l2_bytes / 2)
                       / (6 * static_cast<int>(sizeof(float)));
            int kb_max_mt = std::min(kb_a, kb_b);
            if (K <= kb_max_mt) {
                plan.KB = K;
            } else if (kb_max_mt > plan.KB) {
                int n_blocks = (K + kb_max_mt - 1) / kb_max_mt;
                plan.KB = ((K + n_blocks - 1) / n_blocks + 7) & ~7;
            }
        }

        s_plan_cache = {M, N, K, desc.num_threads, desc.transA, transB, plan};
    }

    // FP32 MR/NR selection:
    //   NR=64 (NV=4): relu/no-act — 24 acc, 2x fewer tiles vs NR=32
    //   NR=32 (NV=2): complex activation or fallback — 12 acc, asm kernel
    //   NR=16 (NV=1): small N
    plan.MR = 6;
    if (N >= 64 && !has_complex_activation) {
        plan.NR = 64;
    } else if (N >= 32) {
        plan.NR = 32;
    } else {
        plan.NR = 16;
    }
    // Re-align NB to be a multiple of the final NR
    plan.NB = std::max(plan.NB / plan.NR * plan.NR, plan.NR);
    plan.NB = std::min(plan.NB, N);
    // Recompute MB for the final NB.
    // When A is packed, packed_A + B_panel must fit in L2:
    //   packed_A = MB × KB × 4, B_panel = KB × NR_PACK × 4
    //   MB = (L2 - B_panel) / (KB × 4)
    // When A is NOT packed, use original C+A formula.
    {
        bool will_pack_a = desc.transA
                           || (desc.lda * static_cast<int>(sizeof(float)) > 4096);
        if (will_pack_a) {
            // Use actual B data loaded (NR cols, not full NR_PACK panel)
            int b_lines_per_krow = ((plan.NR + 15) / 16) * 64; // bytes
            int b_accessed_bytes = plan.KB * b_lines_per_krow;
            int l2_for_a = std::max(uarch.l2_bytes - b_accessed_bytes, 0);
            plan.MB = std::max(l2_for_a / (plan.KB * 4), plan.MR);
        } else {
            plan.MB = std::max((uarch.l1d_bytes + uarch.l2_bytes)
                               / (plan.NB * 4 + plan.KB * 4), plan.MR);
        }
        plan.MB = plan.MB / plan.MR * plan.MR;
        plan.MB = std::min(plan.MB, M);
    }

    // Even-split load balance: ensure enough tiles for all threads.
    if (plan.num_threads > 1) {
        int jc_tiles = (N + plan.NB - 1) / plan.NB;
        int ic_tiles = (M + plan.MB - 1) / plan.MB;
        int needed_ic = (plan.num_threads + jc_tiles - 1) / jc_tiles;
        if (needed_ic > ic_tiles && needed_ic > 1) {
            int m_panels = (M + plan.MR - 1) / plan.MR;
            int panels_per_block = std::max(m_panels / needed_ic, 1);
            plan.MB = panels_per_block * plan.MR;
            plan.MB = std::min(plan.MB, M);
        }
    }

    if (apilog_info_enabled()) {
        apilog_info("AI GEMM plan: M=", M, " N=", N, " K=", K,
                    " MB=", plan.MB, " NB=", plan.NB, " KB=", plan.KB,
                    " MR=", plan.MR, " NR=", plan.NR,
                    " NV=", plan.NR / 16,
                    " activation=", has_activation ? "yes" : "no",
                    " complex_act=", has_complex_activation ? "yes" : "no",
                    " threads=", plan.num_threads);
    }

    // ── 2. B matrix prepacking ──
    //
    // Three cases:
    //   (a) WEIGHT_CACHE=1 + is_weights_const: one-time prepack, cached forever.
    //       Subsequent calls just look up the pointer (mutex + hash, ~100ns).
    //   (b) !can_cache + transB=true: must pack every call (transposed layout).
    //       Uses thread-local reusable buffer (no malloc/free overhead).
    //   (c) !can_cache + transB=false + multi-threaded: per-call prepack
    //       into thread-local buffer for sequential B access (critical for
    //       large N where ldb stride kills L2 prefetcher).
    //   (d) !can_cache + transB=false + single-threaded: direct B access.
    //
    static int32_t s_weight_cache =
        matmul_config_t::instance().get_weight_cache();
    const PrepackedWeight *prepacked_b = nullptr;
    const bool can_cache = is_weights_const && (s_weight_cache != 0);

    if (can_cache) {
        // Case (a): cached prepack
        PrepackedWeightKey bk{weight, K, N, desc.ldb, transB};
        prepacked_b = PrepackedWeightCache::instance().get_or_prepack(
            bk, static_cast<const float *>(weight));
    } else if (transB || (desc.num_threads > 1 && desc.ldb > NR_PACK)) {
        // Case (b): transB requires packing (microkernel needs row-major B).
        static thread_local float *s_tb = nullptr;
        static thread_local size_t s_tb_cap = 0;
        static thread_local PrepackedWeight s_tb_pw;

        const int np = (N + NR_PACK - 1) / NR_PACK;
        const size_t total = static_cast<size_t>(np) * K * NR_PACK;

        if (s_tb_cap < total) {
            std::free(s_tb);
            s_tb = static_cast<float *>(std::aligned_alloc(
                64, ((total * sizeof(float) + 63) & ~size_t(63))));
            s_tb_cap = total;
        }

        if (s_tb) {
            const float *Bf = static_cast<const float *>(weight);
            for (int jp = 0; jp < np; ++jp) {
                const int j0 = jp * NR_PACK;
                const int nr_act = std::min(NR_PACK, N - j0);
                float *dst_p = s_tb + static_cast<size_t>(jp) * K * NR_PACK;
                for (int kk = 0; kk < K; ++kk) {
                    float *d = dst_p + kk * NR_PACK;
                    if (transB) {
                        for (int n = 0; n < nr_act; ++n)
                            d[n] = Bf[(j0 + n) * desc.ldb + kk];
                    } else {
                        for (int n = 0; n < nr_act; ++n)
                            d[n] = Bf[kk * desc.ldb + (j0 + n)];
                    }
                    for (int n = nr_act; n < NR_PACK; ++n)
                        d[n] = 0.0f;
                }
            }

            s_tb_pw.data = s_tb;
            s_tb_pw.K = K;
            s_tb_pw.N = N;
            s_tb_pw.n_panels = np;
            prepacked_b = &s_tb_pw;
        }
    }

    // ── 3. Thread loop ──
    ai_thread_loop(desc, plan, uarch, src, weight, dst, bias, params,
                   prepacked_b);
}

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
