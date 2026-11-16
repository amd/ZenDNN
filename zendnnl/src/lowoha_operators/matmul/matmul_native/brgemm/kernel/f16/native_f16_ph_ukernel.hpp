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

#ifndef MATMUL_NATIVE_BRGEMM_KERNEL_F16_NATIVE_F16_PH_UKERNEL_HPP
#define MATMUL_NATIVE_BRGEMM_KERNEL_F16_NATIVE_F16_PH_UKERNEL_HPP

#include <cstddef>
#include <cstdint>

// 1 where the compiler's headers have the AVX512-FP16 intrinsics, as
// common/float16.hpp takes them: the f16_ph tile kernels and the f16_ph GEMV
// kernels (kernel/gemv/native_gemv_f16_ph.cpp) exist under this one
// condition.
#if defined(ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS) \
        || (defined(__GNUC__) && (__GNUC__ >= 12))
#define ZENDNNL_NATIVE_F16_PH_KERNELS 1
#else
#define ZENDNNL_NATIVE_F16_PH_KERNELS 0
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// K steps of one block of the f16 accumulators of the AVX512-FP16 kernels.
inline constexpr int kNativeF16AccumBlockK = 256;

/// One MR x NR tile (NR = 16 * nv) over native-layout f16 blocks, from an
/// f16 source, with AVX512-FP16 multiply-adds: C[m][n] = beta * C[m][n] + the
/// sum over k of A[m][k] * B[k][n].  A holds MR rows lda apart, B the K rows
/// of the tile's first pack column b_stride elements apart, and at nv 4 the
/// rows of its columns 32 .. 63 b_next elements after those of columns
/// 0 .. 31 (b_next is not read at nv 1 and 2); C holds MR rows ldc apart;
/// rows are offset in size_t.  K runs in blocks of kNativeF16AccumBlockK steps
/// from its start (the last one shorter): in a block every element takes one
/// f16 fused multiply-add per k in K order, from +0.0, each rounded to f16;
/// the block's sum is then widened to f32 exactly and added to the element's
/// f32 sum, which starts at beta * C, rounded once (C is not read when beta
/// is 0, where it starts at +0.0).  Equal inputs give equal bits whatever MR
/// and nv, and a K split into calls of whole blocks, every call after the
/// first at beta 1, gives the bits of one call.
using native_f16_ph_ukernel_fn_t = void (*)(const uint16_t *__restrict A,
        int lda, const uint16_t *__restrict B, int b_stride, size_t b_next,
        float *__restrict C, int ldc, int K, float beta);

/// Largest MR served for NR = 16 * nv: 12 at nv 1, 2 and 4, 0 otherwise and
/// in a build whose compiler lacks the AVX512-FP16 intrinsics.  The f16
/// accumulators of the rows (one or, at nv 4, two per row), the B vectors and
/// the f32 sums of one row fit the 32 vector registers.
int native_f16_ph_ukernel_max_mr(int nv);

/// nullptr outside MR 1 .. native_f16_ph_ukernel_max_mr(nv).  The kernels
/// need AVX-512 F, BW and VL, AVX512-FP16 and FMA.
native_f16_ph_ukernel_fn_t select_native_f16_ph_ukernel(int mr, int nv);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_KERNEL_F16_NATIVE_F16_PH_UKERNEL_HPP
