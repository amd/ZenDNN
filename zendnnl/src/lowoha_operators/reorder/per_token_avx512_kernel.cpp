/*******************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

#include "lowoha_operators/reorder/per_token_avx512_kernel.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

#include <immintrin.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>
#include <omp.h>

namespace zendnnl {
namespace lowoha {
namespace reorder {

using zendnnl::lowoha::matmul::zendnnl_parallel_for;

//==============================================================================
// BF16 <-> F32 conversion helpers (AVX-512F)
//==============================================================================

/**
 * Convert 16 BF16 values (packed in a 256-bit register) to 16 F32 values.
 * BF16 is the upper 16 bits of IEEE 754 float32, so zero-extending and
 * shifting left by 16 reconstructs the original float.
 */
__attribute__((target("avx512f")))
static inline __m512 bf16x16_to_f32(__m256i bf16) {
  return _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16), 16));
}

/** Scalar BF16-to-F32 fallback for tail elements that don't fill a vector. */
static inline float bf16_scalar_to_f32(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float r;
  std::memcpy(&r, &bits, sizeof(float));
  return r;
}

/**
 * Returns a 16-bit mask where lane k is 1 iff v[k] is finite (not NaN/Inf).
 * Works by checking |v| < Inf in IEEE 754 representation.
 */
__attribute__((target("avx512f")))
static inline __mmask16 finite_mask(__m512 v, __m512i abs_mask, __m512 vinf) {
  __m512 absv = _mm512_castsi512_ps(
      _mm512_and_si512(_mm512_castps_si512(v), abs_mask));
  return _mm512_cmp_ps_mask(absv, vinf, _CMP_LT_OQ);
}

//==============================================================================
// Scale / zero-point computation from row statistics
//==============================================================================

/**
 * Symmetric quantization: scale = absmax / 127.
 * Clamps absmax to a minimum epsilon to avoid division by zero downstream.
 */
static inline void compute_symmetric_scale_from_absmax(float absmax,
                                                        float &scale) {
  if (absmax < 1e-10f) absmax = 1e-10f;
  scale = absmax / 127.0f;
  if (scale < 1e-10f) scale = 1e-10f;
}

/**
 * Asymmetric quantization: scale = (max - min) / 255, zp = round(-min / scale).
 * Handles edge cases where all values are identical or no finite values exist.
 */
static inline void compute_asymmetric_scale_zp(float min_val, float max_val,
                                                float &scale, int32_t &zp) {
  if (min_val == std::numeric_limits<float>::max() &&
      max_val == std::numeric_limits<float>::lowest()) {
    min_val = 0.0f;
    max_val = 0.0f;
  }
  if (max_val <= min_val) max_val = min_val + 1.0f;
  scale = (max_val - min_val) / 255.0f;
  if (scale < 1e-10f) scale = 1e-10f;

  double zp_d = std::round(static_cast<double>(-min_val) /
                            static_cast<double>(scale));
  constexpr double lo = static_cast<double>(std::numeric_limits<int32_t>::min());
  constexpr double hi = static_cast<double>(std::numeric_limits<int32_t>::max());
  if (zp_d < lo) zp = std::numeric_limits<int32_t>::min();
  else if (zp_d > hi) zp = std::numeric_limits<int32_t>::max();
  else zp = static_cast<int32_t>(zp_d);
}

//==============================================================================
// Cache-line-wide store
//
// Packs 4 x 16-byte narrowed results into a single 64-byte (cache-line)
// write using _mm512_store_si512.
//
// Requires 64-byte alignment; falls back to 4 x unaligned stores otherwise.
//==============================================================================

__attribute__((target("avx512f")))
static inline void store_4x16_s8(int8_t *dst,
                                  __m128i i0, __m128i i1,
                                  __m128i i2, __m128i i3,
                                  bool cacheline_aligned) {
  if (cacheline_aligned) {
    __m256i lo = _mm256_set_m128i(i1, i0);
    __m256i hi = _mm256_set_m128i(i3, i2);
    __m512i pack = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
    _mm512_store_si512(reinterpret_cast<__m512i *>(dst), pack);
  } else {
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst),      i0);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 16), i1);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 32), i2);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 48), i3);
  }
}

__attribute__((target("avx512f")))
static inline void store_4x16_u8(uint8_t *dst,
                                  __m128i i0, __m128i i1,
                                  __m128i i2, __m128i i3,
                                  bool cacheline_aligned) {
  store_4x16_s8(reinterpret_cast<int8_t *>(dst), i0, i1, i2, i3,
                cacheline_aligned);
}

//==============================================================================
// Tuning constants
//==============================================================================

//==============================================================================
// KERNEL 1:  BF16 -> S8 Symmetric Per-Token Quantization  (AVX-512F)
//==============================================================================
//
// Steps (per row, fused):
//   1. Pass 1 — Absmax reduction: scan all N BF16 elements, convert to F32,
//      compute absmax = max(|val|).  Optionally stage converted F32 values
//      into a scratch buffer for Pass 2.
//   2. Compute scale: scale = absmax / 127 (clamped to >= 1e-10).
//   3. Write scale to scales[m].
//   4. Pass 2 — Quantize: re-scan row, compute Q[j] = round(val[j] / scale),
//      narrow to int8 with signed saturation, store to dst.
//
// Register usage (peak, per thread):
//   zmm (512-bit):  ~13 total
//     Pass 1:  abs_mask(1) + vinf(1) + f0-f3(4) + a0-a3(4, overlapped)
//              + vam0-vam3(4) = 14, but a0-a3 are temporaries
//     Pass 2:  vscale(1) + f0-f3(4) + r0-r3(4) = 9
//   k-mask:    4  (finite_mask results for f0-f3)
//   Scalar:    scale, absmax, j, cl_ok
//
// Optimizations:
//   1. Fused per-row:  Pass 1 + Pass 2 back-to-back keeps row in L1/L2,
//      cutting DRAM reads ~2x vs two full-matrix sweeps.
//   2. Absmax via AND+MAX:  clears sign bit with VPANDD then VMAXPS,
//      avoiding separate min/max/abs operations (1 op vs 4).
//   3. Non-finite masking:  finite_mask() + VMAXPS{k} skips NaN/Inf lanes,
//      matching the reference path's std::isfinite() behavior.
//   4. L1 re-read:  Pass 2 re-loads BF16 from L1 (still hot from Pass 1)
//      and re-converts to F32.  Avoids per-call heap allocation that would
//      hurt multi-threaded scaling (OMP fork/join + NUMA placement).
//   5. 4x unrolling:  64 elements/iter across 4 independent accumulators
//      breaks dependency chains for out-of-order execution (ILP).
//   6. VPMOVSDB saturation:  _mm512_cvtsepi32_epi8 narrows int32->int8
//      with hardware saturation to [-128,127], no explicit clamp needed.
//   7. Cache-line stores:  packs 4x __m128i into a single 64B aligned
//      store for efficient cache-line writes.
//   8. OMP threading:  the parallel-for uses the default OMP thread count,
//      typically up to M threads (one row per thread) for NUMA-distributed
//      writes.
//   9. True division:  VDIVPS for exact match with reference scalar path
//      (no reciprocal multiply rounding differences).
//  10. Banker's rounding:  VCVTPS2DQ matches std::nearbyint() for
//      bit-exact agreement with reference output.
//==============================================================================

__attribute__((target("avx512f")))
void dynamic_per_token_quant_bf16_s8_native(const uint16_t *src, int8_t *dst,
                                             float *scales,
                                             int64_t M, int64_t N) {
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());

  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const uint16_t *row_src = src + m * N;
    int8_t         *row_dst = dst + m * N;

    // -- Pass 1: absmax reduction -----------------------------------------
    __m512 vam0 = _mm512_setzero_ps();
    __m512 vam1 = _mm512_setzero_ps();
    __m512 vam2 = _mm512_setzero_ps();
    __m512 vam3 = _mm512_setzero_ps();

    int64_t j = 0;
    for (; j + 63 < N; j += 64) {
      __m512 f0 = bf16x16_to_f32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row_src + j)));
      __m512 f1 = bf16x16_to_f32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row_src + j + 16)));
      __m512 f2 = bf16x16_to_f32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row_src + j + 32)));
      __m512 f3 = bf16x16_to_f32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row_src + j + 48)));

      __m512 a0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f0), abs_mask));
      __m512 a1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f1), abs_mask));
      __m512 a2 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f2), abs_mask));
      __m512 a3 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f3), abs_mask));
      vam0 = _mm512_mask_max_ps(vam0, finite_mask(f0, abs_mask, vinf), vam0, a0);
      vam1 = _mm512_mask_max_ps(vam1, finite_mask(f1, abs_mask, vinf), vam1, a1);
      vam2 = _mm512_mask_max_ps(vam2, finite_mask(f2, abs_mask, vinf), vam2, a2);
      vam3 = _mm512_mask_max_ps(vam3, finite_mask(f3, abs_mask, vinf), vam3, a3);
    }

    for (; j + 15 < N; j += 16) {
      __m512 f = bf16x16_to_f32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row_src + j)));
      __m512 af = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f), abs_mask));
      vam0 = _mm512_mask_max_ps(vam0, finite_mask(f, abs_mask, vinf), vam0, af);
    }

    vam0 = _mm512_max_ps(_mm512_max_ps(vam0, vam1),
                          _mm512_max_ps(vam2, vam3));
    float absmax = _mm512_reduce_max_ps(vam0);

    for (; j < N; ++j) {
      float v = bf16_scalar_to_f32(row_src[j]);
      if (std::isfinite(v))
        absmax = std::max(absmax, std::abs(v));
    }

    // -- Compute per-row scale --------------------------------------------
    float scale;
    compute_symmetric_scale_from_absmax(absmax, scale);
    scales[m] = scale;

    // -- Pass 2: quantize (re-load BF16 from L1, still hot from Pass 1) --
    __m512 vscale = _mm512_set1_ps(scale);
    bool cl_ok = (reinterpret_cast<uintptr_t>(row_dst) & 63) == 0;

    j = 0;
    for (; j + 63 < N; j += 64) {
      __m512 f0 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j)));
      __m512 f1 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j + 16)));
      __m512 f2 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j + 32)));
      __m512 f3 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j + 48)));

      __m512i r0 = _mm512_cvtps_epi32(_mm512_div_ps(f0, vscale));
      __m512i r1 = _mm512_cvtps_epi32(_mm512_div_ps(f1, vscale));
      __m512i r2 = _mm512_cvtps_epi32(_mm512_div_ps(f2, vscale));
      __m512i r3 = _mm512_cvtps_epi32(_mm512_div_ps(f3, vscale));

      store_4x16_s8(row_dst + j,
                    _mm512_cvtsepi32_epi8(r0), _mm512_cvtsepi32_epi8(r1),
                    _mm512_cvtsepi32_epi8(r2), _mm512_cvtsepi32_epi8(r3),
                    cl_ok);
    }
    for (; j + 15 < N; j += 16) {
      __m512 f = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j)));
      __m512i r = _mm512_cvtps_epi32(_mm512_div_ps(f, vscale));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(row_dst + j),
                       _mm512_cvtsepi32_epi8(r));
    }
    for (; j < N; ++j) {
      float v = bf16_scalar_to_f32(row_src[j]);
      int32_t q = static_cast<int32_t>(std::nearbyint(v / scale));
      q = std::max(-128, std::min(127, q));
      row_dst[j] = static_cast<int8_t>(q);
    }
  }
}

//==============================================================================
// KERNEL 2:  F32 -> S8 Symmetric Per-Token Quantization  (AVX-512F)
//==============================================================================
//
// Steps (per row, fused):
//   1. Pass 1 — Absmax reduction: scan all N F32 elements, compute
//      absmax = max(|val|) using AND+MAX with non-finite masking.
//   2. Compute scale: scale = absmax / 127 (clamped to >= 1e-10).
//   3. Write scale to scales[m].
//   4. Pass 2 — Quantize: re-read F32 from L1 cache (still hot from
//      Pass 1), compute Q[j] = round(val[j] / scale), narrow to int8
//      with signed saturation, store to dst.
//
// Register usage (peak, per thread):
//   zmm (512-bit):  ~14 total
//     Pass 1:  abs_mask(1) + vinf(1) + f0-f3(4) + a0-a3(4)
//              + vam0-vam3(4) = 14
//     Pass 2:  vscale(1) + r0-r3(4) = 5  (f32 loaded into r temporaries)
//   k-mask:    4  (finite_mask results)
//   Scalar:    scale, absmax, j, cl_ok
//   Stack:     none (F32 source needs no conversion buffer)
//
// Optimizations:
//   1. Fused per-row:  same L1 cache reuse as BF16->S8.
//   2. No scratch buffer:  source is already F32, Pass 2 re-reads directly
//      from L1 with zero conversion overhead.
//   3. Absmax via AND+MAX:  same as Kernel 1.
//   4. Non-finite masking:  finite_mask() + VMAXPS{k}.
//   5. 4x unrolling:  64 elements/iter for ILP.
//   6. VPMOVSDB saturation:  hardware int32->int8 clamping.
//   7. Cache-line stores:  64B aligned writes.
//   8. OMP parallelization: uses the default OMP thread count.
//   9. True division + banker's rounding:  bit-exact with reference.
//==============================================================================

__attribute__((target("avx512f")))
void dynamic_per_token_quant_f32_s8_native(const float *src, int8_t *dst,
                                            float *scales,
                                            int64_t M, int64_t N) {
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const float *row_src = src + m * N;
    int8_t      *row_dst = dst + m * N;

    // -- Pass 1: absmax reduction (skipping non-finite values) ---------------
    __m512 vam0 = _mm512_setzero_ps();
    __m512 vam1 = _mm512_setzero_ps();
    __m512 vam2 = _mm512_setzero_ps();
    __m512 vam3 = _mm512_setzero_ps();

    int64_t j = 0;
    for (; j + 63 < N; j += 64) {

      __m512 f0 = _mm512_loadu_ps(row_src + j);
      __m512 f1 = _mm512_loadu_ps(row_src + j + 16);
      __m512 f2 = _mm512_loadu_ps(row_src + j + 32);
      __m512 f3 = _mm512_loadu_ps(row_src + j + 48);

      __m512 a0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f0), abs_mask));
      __m512 a1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f1), abs_mask));
      __m512 a2 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f2), abs_mask));
      __m512 a3 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f3), abs_mask));
      vam0 = _mm512_mask_max_ps(vam0, finite_mask(f0, abs_mask, vinf), vam0, a0);
      vam1 = _mm512_mask_max_ps(vam1, finite_mask(f1, abs_mask, vinf), vam1, a1);
      vam2 = _mm512_mask_max_ps(vam2, finite_mask(f2, abs_mask, vinf), vam2, a2);
      vam3 = _mm512_mask_max_ps(vam3, finite_mask(f3, abs_mask, vinf), vam3, a3);
    }

    for (; j + 15 < N; j += 16) {
      __m512 f = _mm512_loadu_ps(row_src + j);
      __m512 af = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f), abs_mask));
      vam0 = _mm512_mask_max_ps(vam0, finite_mask(f, abs_mask, vinf), vam0, af);
    }

    vam0 = _mm512_max_ps(_mm512_max_ps(vam0, vam1),
                          _mm512_max_ps(vam2, vam3));
    float absmax = _mm512_reduce_max_ps(vam0);

    for (; j < N; ++j)
      if (std::isfinite(row_src[j]))
        absmax = std::max(absmax, std::abs(row_src[j]));

    float scale;
    compute_symmetric_scale_from_absmax(absmax, scale);
    scales[m] = scale;

    // -- Pass 2: quantize (F32 re-read from L1, no conversion needed) -----
    __m512 vscale = _mm512_set1_ps(scale);
    bool cl_ok = (reinterpret_cast<uintptr_t>(row_dst) & 63) == 0;

    j = 0;
    for (; j + 63 < N; j += 64) {
      __m512i r0 = _mm512_cvtps_epi32(
          _mm512_div_ps(_mm512_loadu_ps(row_src + j), vscale));
      __m512i r1 = _mm512_cvtps_epi32(
          _mm512_div_ps(_mm512_loadu_ps(row_src + j + 16), vscale));
      __m512i r2 = _mm512_cvtps_epi32(
          _mm512_div_ps(_mm512_loadu_ps(row_src + j + 32), vscale));
      __m512i r3 = _mm512_cvtps_epi32(
          _mm512_div_ps(_mm512_loadu_ps(row_src + j + 48), vscale));

      store_4x16_s8(row_dst + j,
                    _mm512_cvtsepi32_epi8(r0), _mm512_cvtsepi32_epi8(r1),
                    _mm512_cvtsepi32_epi8(r2), _mm512_cvtsepi32_epi8(r3),
                    cl_ok);
    }

    for (; j + 15 < N; j += 16) {
      __m512i r = _mm512_cvtps_epi32(
          _mm512_div_ps(_mm512_loadu_ps(row_src + j), vscale));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(row_dst + j),
                       _mm512_cvtsepi32_epi8(r));
    }

    for (; j < N; ++j) {
      int32_t q = static_cast<int32_t>(std::nearbyint(row_src[j] / scale));
      q = std::max(-128, std::min(127, q));
      row_dst[j] = static_cast<int8_t>(q);
    }
  }
}

//==============================================================================
// KERNEL 3:  BF16 -> U8 Asymmetric Per-Token Quantization  (AVX-512F)
//==============================================================================
//
// Steps (per row, fused):
//   1. Pass 1 — Min/Max reduction: scan all N BF16 elements, convert to
//      F32, compute row_min and row_max (skipping non-finite values).
//      Optionally stage converted F32 into scratch buffer.
//   2. Compute scale and zero-point:
//        scale = (row_max - row_min) / 255  (clamped to >= 1e-10)
//        zp    = round(-row_min / scale)    (clamped to int32 range)
//   3. Write scale to scales[m], zp to zps[m].
//   4. Pass 2 — Quantize with clamp: re-scan row, compute
//      Q[j] = clamp(round(val[j] / scale) + zp, 0, 255),
//      narrow to uint8, store to dst.
//
// Register usage (peak, per thread):
//   zmm (512-bit):  ~14 total
//     Pass 1:  abs_mask(1) + vinf(1) + f0-f3(4) + vmin0-3(4)
//              + vmax0-3(4) = 14
//     Pass 2:  vscale(1) + vzp(1) + vlo(1) + vhi(1) + f0-f3(4)
//              + r0-r3(4) = 12
//   k-mask:    4  (finite_mask results)
//   Scalar:    scale, zp, row_min, row_max, j, cl_ok
//
// Optimizations:
//   1. Fused per-row:  same L1 cache reuse as Kernel 1.
//   2. Min/Max with non-finite masking:  VMINPS{k}/VMAXPS{k} skip
//      NaN/Inf lanes during reduction.
//   3. L1 re-read:  same as Kernel 1, avoids per-call heap allocation.
//   4. 4x unrolling:  64 elements/iter with 4 independent min/max
//      accumulator pairs (8 zmm accumulators total).
//   5. Explicit [0,255] clamp:  VPMAXSD(0) + VPMINSD(255) before
//      VPMOVUSDB, because VPMOVUSDB treats negative int32 as large
//      unsigned and saturates to 255 instead of 0.
//   6. Cache-line stores:  64B aligned writes via store_4x16_u8.
//   7. OMP parallelization with default thread pool + true division +
//      banker's rounding.
//==============================================================================

__attribute__((target("avx512f")))
void dynamic_per_token_quant_bf16_u8_native(const uint16_t *src, uint8_t *dst,
                                             float *scales, int32_t *zps,
                                             int64_t M, int64_t N) {
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const uint16_t *row_src = src + m * N;
    uint8_t        *row_dst = dst + m * N;

    // -- Pass 1: min/max reduction (skipping non-finite) ------------------
    __m512 vmin0 = _mm512_set1_ps(std::numeric_limits<float>::max());
    __m512 vmax0 = _mm512_set1_ps(std::numeric_limits<float>::lowest());
    __m512 vmin1 = vmin0, vmax1 = vmax0;
    __m512 vmin2 = vmin0, vmax2 = vmax0;
    __m512 vmin3 = vmin0, vmax3 = vmax0;

    int64_t j = 0;
    for (; j + 63 < N; j += 64) {
      __m512 f0 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j)));
      __m512 f1 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j + 16)));
      __m512 f2 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j + 32)));
      __m512 f3 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j + 48)));

      __mmask16 k0 = finite_mask(f0, abs_mask, vinf);
      __mmask16 k1 = finite_mask(f1, abs_mask, vinf);
      __mmask16 k2 = finite_mask(f2, abs_mask, vinf);
      __mmask16 k3 = finite_mask(f3, abs_mask, vinf);
      vmin0 = _mm512_mask_min_ps(vmin0, k0, vmin0, f0);
      vmax0 = _mm512_mask_max_ps(vmax0, k0, vmax0, f0);
      vmin1 = _mm512_mask_min_ps(vmin1, k1, vmin1, f1);
      vmax1 = _mm512_mask_max_ps(vmax1, k1, vmax1, f1);
      vmin2 = _mm512_mask_min_ps(vmin2, k2, vmin2, f2);
      vmax2 = _mm512_mask_max_ps(vmax2, k2, vmax2, f2);
      vmin3 = _mm512_mask_min_ps(vmin3, k3, vmin3, f3);
      vmax3 = _mm512_mask_max_ps(vmax3, k3, vmax3, f3);
    }

    for (; j + 15 < N; j += 16) {
      __m512 f = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j)));
      __mmask16 k = finite_mask(f, abs_mask, vinf);
      vmin0 = _mm512_mask_min_ps(vmin0, k, vmin0, f);
      vmax0 = _mm512_mask_max_ps(vmax0, k, vmax0, f);
    }

    vmin0 = _mm512_min_ps(_mm512_min_ps(vmin0, vmin1),
                           _mm512_min_ps(vmin2, vmin3));
    vmax0 = _mm512_max_ps(_mm512_max_ps(vmax0, vmax1),
                           _mm512_max_ps(vmax2, vmax3));

    float row_min = _mm512_reduce_min_ps(vmin0);
    float row_max = _mm512_reduce_max_ps(vmax0);

    for (; j < N; ++j) {
      float v = bf16_scalar_to_f32(row_src[j]);
      if (std::isfinite(v)) {
        row_min = std::min(row_min, v);
        row_max = std::max(row_max, v);
      }
    }

    // -- Compute per-row scale and zero point -----------------------------
    float scale;
    int32_t zp;
    compute_asymmetric_scale_zp(row_min, row_max, scale, zp);
    scales[m] = scale;
    zps[m]    = zp;

    // -- Pass 2: quantize with clamp (re-load BF16 from L1) --------------
    __m512  vscale = _mm512_set1_ps(scale);
    __m512i vzp  = _mm512_set1_epi32(zp);
    __m512i vlo  = _mm512_set1_epi32(0);
    __m512i vhi  = _mm512_set1_epi32(255);
    bool cl_ok = (reinterpret_cast<uintptr_t>(row_dst) & 63) == 0;

    j = 0;
    for (; j + 63 < N; j += 64) {
      __m512 f0 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j)));
      __m512 f1 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j + 16)));
      __m512 f2 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j + 32)));
      __m512 f3 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j + 48)));
      __m512i r0 = _mm512_add_epi32(_mm512_cvtps_epi32(
          _mm512_div_ps(f0, vscale)), vzp);
      __m512i r1 = _mm512_add_epi32(_mm512_cvtps_epi32(
          _mm512_div_ps(f1, vscale)), vzp);
      __m512i r2 = _mm512_add_epi32(_mm512_cvtps_epi32(
          _mm512_div_ps(f2, vscale)), vzp);
      __m512i r3 = _mm512_add_epi32(_mm512_cvtps_epi32(
          _mm512_div_ps(f3, vscale)), vzp);
      r0 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r0));
      r1 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r1));
      r2 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r2));
      r3 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r3));
      store_4x16_u8(row_dst + j,
                    _mm512_cvtusepi32_epi8(r0), _mm512_cvtusepi32_epi8(r1),
                    _mm512_cvtusepi32_epi8(r2), _mm512_cvtusepi32_epi8(r3),
                    cl_ok);
    }
    for (; j + 15 < N; j += 16) {
      __m512 f = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j)));
      __m512i r = _mm512_add_epi32(
          _mm512_cvtps_epi32(_mm512_div_ps(f, vscale)), vzp);
      r = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(row_dst + j),
                       _mm512_cvtusepi32_epi8(r));
    }
    for (; j < N; ++j) {
      float v = bf16_scalar_to_f32(row_src[j]);
      int32_t q = static_cast<int32_t>(std::nearbyint(v / scale)) + zp;
      q = std::max(0, std::min(255, q));
      row_dst[j] = static_cast<uint8_t>(q);
    }
  }
}

//==============================================================================
// KERNEL 4:  F32 -> U8 Asymmetric Per-Token Quantization  (AVX-512F)
//==============================================================================
//
// Steps (per row, fused):
//   1. Pass 1 — Min/Max reduction: scan all N F32 elements, compute
//      row_min and row_max (skipping non-finite values).
//   2. Compute scale and zero-point:
//        scale = (row_max - row_min) / 255  (clamped to >= 1e-10)
//        zp    = round(-row_min / scale)    (clamped to int32 range)
//   3. Write scale to scales[m], zp to zps[m].
//   4. Pass 2 — Quantize with clamp: re-read F32 from L1 cache,
//      compute Q[j] = clamp(round(val[j] / scale) + zp, 0, 255),
//      narrow to uint8, store to dst.
//
// Register usage (peak, per thread):
//   zmm (512-bit):  ~14 total
//     Pass 1:  abs_mask(1) + vinf(1) + f0-f3(4) + vmin0-3(4)
//              + vmax0-3(4) = 14
//     Pass 2:  vscale(1) + vzp(1) + vlo(1) + vhi(1) + r0-r3(4) = 8
//   k-mask:    4  (finite_mask results)
//   Scalar:    scale, zp, row_min, row_max, j, cl_ok
//   Stack:     none (F32 source needs no conversion buffer)
//
// Optimizations:
//   1. Fused per-row:  same L1 cache reuse as Kernel 1.
//   2. No scratch buffer:  source is already F32, Pass 2 re-reads
//      directly from L1 cache.
//   3. Min/Max with non-finite masking:  VMINPS{k}/VMAXPS{k}.
//   4. 4x unrolling:  64 elements/iter for ILP.
//   5. Explicit [0,255] clamp:  same as Kernel 3.
//   6. Cache-line stores:  64B aligned writes.
//   7. OMP parallelization with default thread pool + true division +
//      banker's rounding.
//==============================================================================

__attribute__((target("avx512f")))
void dynamic_per_token_quant_f32_u8_native(const float *src, uint8_t *dst,
                                            float *scales, int32_t *zps,
                                            int64_t M, int64_t N) {
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const float *row_src = src + m * N;
    uint8_t     *row_dst = dst + m * N;

    // -- Pass 1: min/max reduction (skipping non-finite values) ---------------
    __m512 vmin0 = _mm512_set1_ps(std::numeric_limits<float>::max());
    __m512 vmax0 = _mm512_set1_ps(std::numeric_limits<float>::lowest());
    __m512 vmin1 = vmin0, vmax1 = vmax0;
    __m512 vmin2 = vmin0, vmax2 = vmax0;
    __m512 vmin3 = vmin0, vmax3 = vmax0;

    int64_t j = 0;
    for (; j + 63 < N; j += 64) {

      __m512 f0 = _mm512_loadu_ps(row_src + j);
      __m512 f1 = _mm512_loadu_ps(row_src + j + 16);
      __m512 f2 = _mm512_loadu_ps(row_src + j + 32);
      __m512 f3 = _mm512_loadu_ps(row_src + j + 48);

      __mmask16 k0 = finite_mask(f0, abs_mask, vinf);
      __mmask16 k1 = finite_mask(f1, abs_mask, vinf);
      __mmask16 k2 = finite_mask(f2, abs_mask, vinf);
      __mmask16 k3 = finite_mask(f3, abs_mask, vinf);
      vmin0 = _mm512_mask_min_ps(vmin0, k0, vmin0, f0);
      vmax0 = _mm512_mask_max_ps(vmax0, k0, vmax0, f0);
      vmin1 = _mm512_mask_min_ps(vmin1, k1, vmin1, f1);
      vmax1 = _mm512_mask_max_ps(vmax1, k1, vmax1, f1);
      vmin2 = _mm512_mask_min_ps(vmin2, k2, vmin2, f2);
      vmax2 = _mm512_mask_max_ps(vmax2, k2, vmax2, f2);
      vmin3 = _mm512_mask_min_ps(vmin3, k3, vmin3, f3);
      vmax3 = _mm512_mask_max_ps(vmax3, k3, vmax3, f3);
    }

    for (; j + 15 < N; j += 16) {
      __m512 f = _mm512_loadu_ps(row_src + j);
      __mmask16 k = finite_mask(f, abs_mask, vinf);
      vmin0 = _mm512_mask_min_ps(vmin0, k, vmin0, f);
      vmax0 = _mm512_mask_max_ps(vmax0, k, vmax0, f);
    }

    vmin0 = _mm512_min_ps(_mm512_min_ps(vmin0, vmin1),
                           _mm512_min_ps(vmin2, vmin3));
    vmax0 = _mm512_max_ps(_mm512_max_ps(vmax0, vmax1),
                           _mm512_max_ps(vmax2, vmax3));

    float row_min = _mm512_reduce_min_ps(vmin0);
    float row_max = _mm512_reduce_max_ps(vmax0);

    for (; j < N; ++j) {
      if (std::isfinite(row_src[j])) {
        row_min = std::min(row_min, row_src[j]);
        row_max = std::max(row_max, row_src[j]);
      }
    }

    float scale;
    int32_t zp;
    compute_asymmetric_scale_zp(row_min, row_max, scale, zp);
    scales[m] = scale;
    zps[m]    = zp;

    // -- Pass 2: quantize with clamp (F32 re-read from L1) ----------------
    __m512  vscale = _mm512_set1_ps(scale);
    __m512i vzp  = _mm512_set1_epi32(zp);
    __m512i vlo  = _mm512_set1_epi32(0);
    __m512i vhi  = _mm512_set1_epi32(255);
    bool cl_ok = (reinterpret_cast<uintptr_t>(row_dst) & 63) == 0;

    j = 0;
    for (; j + 63 < N; j += 64) {
      __m512i r0 = _mm512_add_epi32(_mm512_cvtps_epi32(
          _mm512_div_ps(_mm512_loadu_ps(row_src + j), vscale)), vzp);
      __m512i r1 = _mm512_add_epi32(_mm512_cvtps_epi32(
          _mm512_div_ps(_mm512_loadu_ps(row_src + j + 16), vscale)), vzp);
      __m512i r2 = _mm512_add_epi32(_mm512_cvtps_epi32(
          _mm512_div_ps(_mm512_loadu_ps(row_src + j + 32), vscale)), vzp);
      __m512i r3 = _mm512_add_epi32(_mm512_cvtps_epi32(
          _mm512_div_ps(_mm512_loadu_ps(row_src + j + 48), vscale)), vzp);

      r0 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r0));
      r1 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r1));
      r2 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r2));
      r3 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r3));

      store_4x16_u8(row_dst + j,
                    _mm512_cvtusepi32_epi8(r0), _mm512_cvtusepi32_epi8(r1),
                    _mm512_cvtusepi32_epi8(r2), _mm512_cvtusepi32_epi8(r3),
                    cl_ok);
    }

    for (; j + 15 < N; j += 16) {
      __m512i r = _mm512_add_epi32(_mm512_cvtps_epi32(
          _mm512_div_ps(_mm512_loadu_ps(row_src + j), vscale)), vzp);
      r = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(row_dst + j),
                       _mm512_cvtusepi32_epi8(r));
    }

    for (; j < N; ++j) {
      int32_t q = static_cast<int32_t>(
          std::nearbyint(row_src[j] / scale)) + zp;
      q = std::max(0, std::min(255, q));
      row_dst[j] = static_cast<uint8_t>(q);
    }
  }
}

//==============================================================================
// UNFUSED 2-PASS:  Per-Token Dynamic Quantization  (AVX-512F)
//==============================================================================
//
// Two-pass vectorized kernels with different parallelization per pass:
//   Pass 1 — Statistics:  parallel over M rows.  Each thread computes
//            absmax (symmetric) or min/max (asymmetric) for its rows
//            using AVX-512, then writes scale (and zp) to the output arrays.
//   Pass 2 — Quantize:   parallel over M*N contiguous elements.  Each
//            thread quantizes its chunk using the pre-computed per-row
//            scales, with AVX-512 vectorization.
//
// The key advantage over fused kernels is Pass 2 parallelism: when M is
// small (e.g. M=1) but N is large, the fused kernel can only use M threads,
// while this unfused Pass 2 distributes M*N elements across all threads.
//==============================================================================

// --- BF16 -> S8 Symmetric (unfused 2-pass AVX-512) ---

__attribute__((target("avx512f")))
void dynamic_per_token_quant_bf16_s8_unfused_native(const uint16_t *src,
                                                     int8_t *dst,
                                                     float *scales,
                                                     int64_t M, int64_t N) {
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());

  // -- Pass 1: per-row absmax + scale (parallel over M) -------------------
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const uint16_t *row_src = src + m * N;

    __m512 vam0 = _mm512_setzero_ps();
    __m512 vam1 = _mm512_setzero_ps();
    __m512 vam2 = _mm512_setzero_ps();
    __m512 vam3 = _mm512_setzero_ps();

    int64_t j = 0;
    for (; j + 63 < N; j += 64) {
      __m512 f0 = bf16x16_to_f32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row_src + j)));
      __m512 f1 = bf16x16_to_f32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row_src + j + 16)));
      __m512 f2 = bf16x16_to_f32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row_src + j + 32)));
      __m512 f3 = bf16x16_to_f32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row_src + j + 48)));

      __m512 a0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f0), abs_mask));
      __m512 a1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f1), abs_mask));
      __m512 a2 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f2), abs_mask));
      __m512 a3 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f3), abs_mask));
      vam0 = _mm512_mask_max_ps(vam0, finite_mask(f0, abs_mask, vinf), vam0, a0);
      vam1 = _mm512_mask_max_ps(vam1, finite_mask(f1, abs_mask, vinf), vam1, a1);
      vam2 = _mm512_mask_max_ps(vam2, finite_mask(f2, abs_mask, vinf), vam2, a2);
      vam3 = _mm512_mask_max_ps(vam3, finite_mask(f3, abs_mask, vinf), vam3, a3);
    }
    for (; j + 15 < N; j += 16) {
      __m512 f = bf16x16_to_f32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row_src + j)));
      __m512 af = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f), abs_mask));
      vam0 = _mm512_mask_max_ps(vam0, finite_mask(f, abs_mask, vinf), vam0, af);
    }

    vam0 = _mm512_max_ps(_mm512_max_ps(vam0, vam1),
                          _mm512_max_ps(vam2, vam3));
    float absmax = _mm512_reduce_max_ps(vam0);
    for (; j < N; ++j) {
      float v = bf16_scalar_to_f32(row_src[j]);
      if (std::isfinite(v))
        absmax = std::max(absmax, std::abs(v));
    }

    float scale;
    compute_symmetric_scale_from_absmax(absmax, scale);
    scales[m] = scale;
  }

  // -- Pass 2: quantize (parallel over M*N contiguous elements) -----------
  const int64_t total = M * N;
  constexpr int64_t grain_size = LOWOHA_REORDER_GRAIN_SIZE;
  zendnnl_parallel_for(0, total, grain_size,
      [&](int64_t begin, int64_t end) __attribute__((target("avx512f"))) {
    while (begin < end) {
      const int64_t m = begin / N;
      const int64_t row_end = std::min((m + 1) * N, end);
      const int64_t count = row_end - begin;
      const uint16_t *csrc = src + begin;
      int8_t *cdst = dst + begin;
      const __m512 vscale = _mm512_set1_ps(scales[m]);
      const bool cl_ok = (reinterpret_cast<uintptr_t>(cdst) & 63) == 0;

      int64_t k = 0;
      for (; k + 63 < count; k += 64) {
        __m512 f0 = bf16x16_to_f32(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(csrc + k)));
        __m512 f1 = bf16x16_to_f32(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(csrc + k + 16)));
        __m512 f2 = bf16x16_to_f32(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(csrc + k + 32)));
        __m512 f3 = bf16x16_to_f32(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(csrc + k + 48)));
        __m512i r0 = _mm512_cvtps_epi32(_mm512_div_ps(f0, vscale));
        __m512i r1 = _mm512_cvtps_epi32(_mm512_div_ps(f1, vscale));
        __m512i r2 = _mm512_cvtps_epi32(_mm512_div_ps(f2, vscale));
        __m512i r3 = _mm512_cvtps_epi32(_mm512_div_ps(f3, vscale));
        __m128i s0 = _mm512_cvtsepi32_epi8(r0);
        __m128i s1 = _mm512_cvtsepi32_epi8(r1);
        __m128i s2 = _mm512_cvtsepi32_epi8(r2);
        __m128i s3 = _mm512_cvtsepi32_epi8(r3);
        store_4x16_s8(cdst + k, s0, s1, s2, s3, cl_ok);
      }
      for (; k + 15 < count; k += 16) {
        __m512 f = bf16x16_to_f32(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(csrc + k)));
        __m512i r = _mm512_cvtps_epi32(_mm512_div_ps(f, vscale));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(cdst + k),
                         _mm512_cvtsepi32_epi8(r));
      }
      for (; k < count; ++k) {
        float v = bf16_scalar_to_f32(csrc[k]);
        int32_t q = static_cast<int32_t>(std::nearbyint(v / scales[m]));
        q = std::max(-128, std::min(127, q));
        cdst[k] = static_cast<int8_t>(q);
      }
      begin = row_end;
    }
  });
}

// --- F32 -> S8 Symmetric (unfused 2-pass AVX-512) ---

__attribute__((target("avx512f")))
void dynamic_per_token_quant_f32_s8_unfused_native(const float *src,
                                                    int8_t *dst,
                                                    float *scales,
                                                    int64_t M, int64_t N) {
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());

  // -- Pass 1: per-row absmax + scale (parallel over M) -------------------
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const float *row_src = src + m * N;

    __m512 vam0 = _mm512_setzero_ps();
    __m512 vam1 = _mm512_setzero_ps();
    __m512 vam2 = _mm512_setzero_ps();
    __m512 vam3 = _mm512_setzero_ps();

    int64_t j = 0;
    for (; j + 63 < N; j += 64) {
      __m512 f0 = _mm512_loadu_ps(row_src + j);
      __m512 f1 = _mm512_loadu_ps(row_src + j + 16);
      __m512 f2 = _mm512_loadu_ps(row_src + j + 32);
      __m512 f3 = _mm512_loadu_ps(row_src + j + 48);
      __m512 a0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f0), abs_mask));
      __m512 a1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f1), abs_mask));
      __m512 a2 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f2), abs_mask));
      __m512 a3 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f3), abs_mask));
      vam0 = _mm512_mask_max_ps(vam0, finite_mask(f0, abs_mask, vinf), vam0, a0);
      vam1 = _mm512_mask_max_ps(vam1, finite_mask(f1, abs_mask, vinf), vam1, a1);
      vam2 = _mm512_mask_max_ps(vam2, finite_mask(f2, abs_mask, vinf), vam2, a2);
      vam3 = _mm512_mask_max_ps(vam3, finite_mask(f3, abs_mask, vinf), vam3, a3);
    }
    for (; j + 15 < N; j += 16) {
      __m512 f = _mm512_loadu_ps(row_src + j);
      __m512 af = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f), abs_mask));
      vam0 = _mm512_mask_max_ps(vam0, finite_mask(f, abs_mask, vinf), vam0, af);
    }

    vam0 = _mm512_max_ps(_mm512_max_ps(vam0, vam1),
                          _mm512_max_ps(vam2, vam3));
    float absmax = _mm512_reduce_max_ps(vam0);
    for (; j < N; ++j)
      if (std::isfinite(row_src[j]))
        absmax = std::max(absmax, std::abs(row_src[j]));

    float scale;
    compute_symmetric_scale_from_absmax(absmax, scale);
    scales[m] = scale;
  }

  // -- Pass 2: quantize (parallel over M*N contiguous elements) -----------
  const int64_t total = M * N;
  constexpr int64_t grain_size = LOWOHA_REORDER_GRAIN_SIZE;
  zendnnl_parallel_for(0, total, grain_size,
      [&](int64_t begin, int64_t end) __attribute__((target("avx512f"))) {
    while (begin < end) {
      const int64_t m = begin / N;
      const int64_t row_end = std::min((m + 1) * N, end);
      const int64_t count = row_end - begin;
      const float *csrc = src + begin;
      int8_t *cdst = dst + begin;
      const __m512 vscale = _mm512_set1_ps(scales[m]);
      const bool cl_ok = (reinterpret_cast<uintptr_t>(cdst) & 63) == 0;

      int64_t k = 0;
      for (; k + 63 < count; k += 64) {
        __m512i r0 = _mm512_cvtps_epi32(_mm512_div_ps(_mm512_loadu_ps(csrc + k), vscale));
        __m512i r1 = _mm512_cvtps_epi32(_mm512_div_ps(_mm512_loadu_ps(csrc + k + 16), vscale));
        __m512i r2 = _mm512_cvtps_epi32(_mm512_div_ps(_mm512_loadu_ps(csrc + k + 32), vscale));
        __m512i r3 = _mm512_cvtps_epi32(_mm512_div_ps(_mm512_loadu_ps(csrc + k + 48), vscale));
        __m128i s0 = _mm512_cvtsepi32_epi8(r0);
        __m128i s1 = _mm512_cvtsepi32_epi8(r1);
        __m128i s2 = _mm512_cvtsepi32_epi8(r2);
        __m128i s3 = _mm512_cvtsepi32_epi8(r3);
        store_4x16_s8(cdst + k, s0, s1, s2, s3, cl_ok);
      }
      for (; k + 15 < count; k += 16) {
        __m512i r = _mm512_cvtps_epi32(_mm512_div_ps(_mm512_loadu_ps(csrc + k), vscale));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(cdst + k), _mm512_cvtsepi32_epi8(r));
      }
      for (; k < count; ++k) {
        int32_t q = static_cast<int32_t>(std::nearbyint(csrc[k] / scales[m]));
        q = std::max(-128, std::min(127, q));
        cdst[k] = static_cast<int8_t>(q);
      }
      begin = row_end;
    }
  });
}

// --- BF16 -> U8 Asymmetric (unfused 2-pass AVX-512) ---

__attribute__((target("avx512f")))
void dynamic_per_token_quant_bf16_u8_unfused_native(const uint16_t *src,
                                                     uint8_t *dst,
                                                     float *scales,
                                                     int32_t *zps,
                                                     int64_t M, int64_t N) {
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());

  // -- Pass 1: per-row min/max + scale/zp (parallel over M) ---------------
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const uint16_t *row_src = src + m * N;

    __m512 vmin0 = _mm512_set1_ps(std::numeric_limits<float>::max());
    __m512 vmax0 = _mm512_set1_ps(std::numeric_limits<float>::lowest());
    __m512 vmin1 = vmin0, vmax1 = vmax0;
    __m512 vmin2 = vmin0, vmax2 = vmax0;
    __m512 vmin3 = vmin0, vmax3 = vmax0;

    int64_t j = 0;
    for (; j + 63 < N; j += 64) {
      __m512 f0 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j)));
      __m512 f1 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j + 16)));
      __m512 f2 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j + 32)));
      __m512 f3 = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j + 48)));
      __mmask16 k0 = finite_mask(f0, abs_mask, vinf);
      __mmask16 k1 = finite_mask(f1, abs_mask, vinf);
      __mmask16 k2 = finite_mask(f2, abs_mask, vinf);
      __mmask16 k3 = finite_mask(f3, abs_mask, vinf);
      vmin0 = _mm512_mask_min_ps(vmin0, k0, vmin0, f0);
      vmax0 = _mm512_mask_max_ps(vmax0, k0, vmax0, f0);
      vmin1 = _mm512_mask_min_ps(vmin1, k1, vmin1, f1);
      vmax1 = _mm512_mask_max_ps(vmax1, k1, vmax1, f1);
      vmin2 = _mm512_mask_min_ps(vmin2, k2, vmin2, f2);
      vmax2 = _mm512_mask_max_ps(vmax2, k2, vmax2, f2);
      vmin3 = _mm512_mask_min_ps(vmin3, k3, vmin3, f3);
      vmax3 = _mm512_mask_max_ps(vmax3, k3, vmax3, f3);
    }
    for (; j + 15 < N; j += 16) {
      __m512 f = bf16x16_to_f32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row_src + j)));
      __mmask16 k = finite_mask(f, abs_mask, vinf);
      vmin0 = _mm512_mask_min_ps(vmin0, k, vmin0, f);
      vmax0 = _mm512_mask_max_ps(vmax0, k, vmax0, f);
    }

    vmin0 = _mm512_min_ps(_mm512_min_ps(vmin0, vmin1),
                           _mm512_min_ps(vmin2, vmin3));
    vmax0 = _mm512_max_ps(_mm512_max_ps(vmax0, vmax1),
                           _mm512_max_ps(vmax2, vmax3));
    float row_min = _mm512_reduce_min_ps(vmin0);
    float row_max = _mm512_reduce_max_ps(vmax0);
    for (; j < N; ++j) {
      float v = bf16_scalar_to_f32(row_src[j]);
      if (std::isfinite(v)) {
        row_min = std::min(row_min, v);
        row_max = std::max(row_max, v);
      }
    }

    float scale;
    int32_t zp;
    compute_asymmetric_scale_zp(row_min, row_max, scale, zp);
    scales[m] = scale;
    zps[m]    = zp;
  }

  // -- Pass 2: quantize (parallel over M*N contiguous elements) -----------
  const int64_t total = M * N;
  constexpr int64_t grain_size = LOWOHA_REORDER_GRAIN_SIZE;
  zendnnl_parallel_for(0, total, grain_size,
      [&](int64_t begin, int64_t end) __attribute__((target("avx512f"))) {
    while (begin < end) {
      const int64_t m = begin / N;
      const int64_t row_end = std::min((m + 1) * N, end);
      const int64_t count = row_end - begin;
      const uint16_t *csrc = src + begin;
      uint8_t *cdst = dst + begin;
      const __m512  vscale = _mm512_set1_ps(scales[m]);
      const __m512i vzp = _mm512_set1_epi32(zps[m]);
      const __m512i vlo = _mm512_set1_epi32(0);
      const __m512i vhi = _mm512_set1_epi32(255);
      const bool cl_ok = (reinterpret_cast<uintptr_t>(cdst) & 63) == 0;

      int64_t k = 0;
      for (; k + 63 < count; k += 64) {
        __m512 f0 = bf16x16_to_f32(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(csrc + k)));
        __m512 f1 = bf16x16_to_f32(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(csrc + k + 16)));
        __m512 f2 = bf16x16_to_f32(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(csrc + k + 32)));
        __m512 f3 = bf16x16_to_f32(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(csrc + k + 48)));
        __m512i r0 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi,
            _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_div_ps(f0, vscale)), vzp)));
        __m512i r1 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi,
            _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_div_ps(f1, vscale)), vzp)));
        __m512i r2 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi,
            _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_div_ps(f2, vscale)), vzp)));
        __m512i r3 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi,
            _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_div_ps(f3, vscale)), vzp)));
        __m128i u0 = _mm512_cvtusepi32_epi8(r0);
        __m128i u1 = _mm512_cvtusepi32_epi8(r1);
        __m128i u2 = _mm512_cvtusepi32_epi8(r2);
        __m128i u3 = _mm512_cvtusepi32_epi8(r3);
        store_4x16_u8(cdst + k, u0, u1, u2, u3, cl_ok);
      }
      for (; k + 15 < count; k += 16) {
        __m512 f = bf16x16_to_f32(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(csrc + k)));
        __m512i r = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi,
            _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_div_ps(f, vscale)), vzp)));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(cdst + k),
                         _mm512_cvtusepi32_epi8(r));
      }
      for (; k < count; ++k) {
        float v = bf16_scalar_to_f32(csrc[k]);
        int32_t q = static_cast<int32_t>(std::nearbyint(v / scales[m])) + zps[m];
        q = std::max(0, std::min(255, q));
        cdst[k] = static_cast<uint8_t>(q);
      }
      begin = row_end;
    }
  });
}

// --- F32 -> U8 Asymmetric (unfused 2-pass AVX-512) ---

__attribute__((target("avx512f")))
void dynamic_per_token_quant_f32_u8_unfused_native(const float *src,
                                                    uint8_t *dst,
                                                    float *scales,
                                                    int32_t *zps,
                                                    int64_t M, int64_t N) {
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());

  // -- Pass 1: per-row min/max + scale/zp (parallel over M) ---------------
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const float *row_src = src + m * N;

    __m512 vmin0 = _mm512_set1_ps(std::numeric_limits<float>::max());
    __m512 vmax0 = _mm512_set1_ps(std::numeric_limits<float>::lowest());
    __m512 vmin1 = vmin0, vmax1 = vmax0;
    __m512 vmin2 = vmin0, vmax2 = vmax0;
    __m512 vmin3 = vmin0, vmax3 = vmax0;

    int64_t j = 0;
    for (; j + 63 < N; j += 64) {
      __m512 f0 = _mm512_loadu_ps(row_src + j);
      __m512 f1 = _mm512_loadu_ps(row_src + j + 16);
      __m512 f2 = _mm512_loadu_ps(row_src + j + 32);
      __m512 f3 = _mm512_loadu_ps(row_src + j + 48);
      __mmask16 k0 = finite_mask(f0, abs_mask, vinf);
      __mmask16 k1 = finite_mask(f1, abs_mask, vinf);
      __mmask16 k2 = finite_mask(f2, abs_mask, vinf);
      __mmask16 k3 = finite_mask(f3, abs_mask, vinf);
      vmin0 = _mm512_mask_min_ps(vmin0, k0, vmin0, f0);
      vmax0 = _mm512_mask_max_ps(vmax0, k0, vmax0, f0);
      vmin1 = _mm512_mask_min_ps(vmin1, k1, vmin1, f1);
      vmax1 = _mm512_mask_max_ps(vmax1, k1, vmax1, f1);
      vmin2 = _mm512_mask_min_ps(vmin2, k2, vmin2, f2);
      vmax2 = _mm512_mask_max_ps(vmax2, k2, vmax2, f2);
      vmin3 = _mm512_mask_min_ps(vmin3, k3, vmin3, f3);
      vmax3 = _mm512_mask_max_ps(vmax3, k3, vmax3, f3);
    }
    for (; j + 15 < N; j += 16) {
      __m512 f = _mm512_loadu_ps(row_src + j);
      __mmask16 k = finite_mask(f, abs_mask, vinf);
      vmin0 = _mm512_mask_min_ps(vmin0, k, vmin0, f);
      vmax0 = _mm512_mask_max_ps(vmax0, k, vmax0, f);
    }

    vmin0 = _mm512_min_ps(_mm512_min_ps(vmin0, vmin1),
                           _mm512_min_ps(vmin2, vmin3));
    vmax0 = _mm512_max_ps(_mm512_max_ps(vmax0, vmax1),
                           _mm512_max_ps(vmax2, vmax3));
    float row_min = _mm512_reduce_min_ps(vmin0);
    float row_max = _mm512_reduce_max_ps(vmax0);
    for (; j < N; ++j) {
      if (std::isfinite(row_src[j])) {
        row_min = std::min(row_min, row_src[j]);
        row_max = std::max(row_max, row_src[j]);
      }
    }

    float scale;
    int32_t zp;
    compute_asymmetric_scale_zp(row_min, row_max, scale, zp);
    scales[m] = scale;
    zps[m]    = zp;
  }

  // -- Pass 2: quantize (parallel over M*N contiguous elements) -----------
  const int64_t total = M * N;
  constexpr int64_t grain_size = LOWOHA_REORDER_GRAIN_SIZE;
  zendnnl_parallel_for(0, total, grain_size,
      [&](int64_t begin, int64_t end) __attribute__((target("avx512f"))) {
    while (begin < end) {
      const int64_t m = begin / N;
      const int64_t row_end = std::min((m + 1) * N, end);
      const int64_t count = row_end - begin;
      const float *csrc = src + begin;
      uint8_t *cdst = dst + begin;
      const __m512  vscale = _mm512_set1_ps(scales[m]);
      const __m512i vzp = _mm512_set1_epi32(zps[m]);
      const __m512i vlo = _mm512_set1_epi32(0);
      const __m512i vhi = _mm512_set1_epi32(255);
      const bool cl_ok = (reinterpret_cast<uintptr_t>(cdst) & 63) == 0;

      int64_t k = 0;
      for (; k + 63 < count; k += 64) {
        __m512i r0 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi,
            _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_div_ps(_mm512_loadu_ps(csrc + k), vscale)), vzp)));
        __m512i r1 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi,
            _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_div_ps(_mm512_loadu_ps(csrc + k + 16), vscale)), vzp)));
        __m512i r2 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi,
            _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_div_ps(_mm512_loadu_ps(csrc + k + 32), vscale)), vzp)));
        __m512i r3 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi,
            _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_div_ps(_mm512_loadu_ps(csrc + k + 48), vscale)), vzp)));
        __m128i u0 = _mm512_cvtusepi32_epi8(r0);
        __m128i u1 = _mm512_cvtusepi32_epi8(r1);
        __m128i u2 = _mm512_cvtusepi32_epi8(r2);
        __m128i u3 = _mm512_cvtusepi32_epi8(r3);
        store_4x16_u8(cdst + k, u0, u1, u2, u3, cl_ok);
      }
      for (; k + 15 < count; k += 16) {
        __m512i r = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi,
            _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_div_ps(_mm512_loadu_ps(csrc + k), vscale)), vzp)));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(cdst + k),
                         _mm512_cvtusepi32_epi8(r));
      }
      for (; k < count; ++k) {
        int32_t q = static_cast<int32_t>(std::nearbyint(csrc[k] / scales[m])) + zps[m];
        q = std::max(0, std::min(255, q));
        cdst[k] = static_cast<uint8_t>(q);
      }
      begin = row_end;
    }
  });
}

//==============================================================================
// SCALAR REFERENCE:  Fused Per-Token Dynamic Quantization
//==============================================================================
//
// Scalar C++ implementations of the same fused per-row logic as the native
// AVX-512 kernels.  Each row is processed in two back-to-back passes
// (statistics + quantize) to keep row data in L1 cache, identical to the
// native path but without SIMD intrinsics.
//
// Used when algo == reference for correctness testing or platforms without
// AVX-512.
//==============================================================================

// --- BF16 -> S8 Symmetric (scalar fused) ---

void dynamic_per_token_quant_bf16_s8_ref(const uint16_t *src, int8_t *dst,
                                          float *scales,
                                          int64_t M, int64_t N) {
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const uint16_t *row_src = src + m * N;
    int8_t         *row_dst = dst + m * N;

    float absmax = 0.0f;
    for (int64_t j = 0; j < N; ++j) {
      float v = bf16_scalar_to_f32(row_src[j]);
      if (std::isfinite(v))
        absmax = std::max(absmax, std::abs(v));
    }

    float scale;
    compute_symmetric_scale_from_absmax(absmax, scale);
    scales[m] = scale;

    for (int64_t j = 0; j < N; ++j) {
      float v = bf16_scalar_to_f32(row_src[j]);
      int32_t q = static_cast<int32_t>(std::nearbyint(v / scale));
      q = std::max(-128, std::min(127, q));
      row_dst[j] = static_cast<int8_t>(q);
    }
  }
}

// --- F32 -> S8 Symmetric (scalar fused) ---

void dynamic_per_token_quant_f32_s8_ref(const float *src, int8_t *dst,
                                         float *scales,
                                         int64_t M, int64_t N) {
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const float *row_src = src + m * N;
    int8_t      *row_dst = dst + m * N;

    float absmax = 0.0f;
    for (int64_t j = 0; j < N; ++j) {
      if (std::isfinite(row_src[j]))
        absmax = std::max(absmax, std::abs(row_src[j]));
    }

    float scale;
    compute_symmetric_scale_from_absmax(absmax, scale);
    scales[m] = scale;

    for (int64_t j = 0; j < N; ++j) {
      int32_t q = static_cast<int32_t>(std::nearbyint(row_src[j] / scale));
      q = std::max(-128, std::min(127, q));
      row_dst[j] = static_cast<int8_t>(q);
    }
  }
}

// --- BF16 -> U8 Asymmetric (scalar fused) ---

void dynamic_per_token_quant_bf16_u8_ref(const uint16_t *src, uint8_t *dst,
                                          float *scales, int32_t *zps,
                                          int64_t M, int64_t N) {
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const uint16_t *row_src = src + m * N;
    uint8_t        *row_dst = dst + m * N;

    float row_min = std::numeric_limits<float>::max();
    float row_max = std::numeric_limits<float>::lowest();
    for (int64_t j = 0; j < N; ++j) {
      float v = bf16_scalar_to_f32(row_src[j]);
      if (std::isfinite(v)) {
        row_min = std::min(row_min, v);
        row_max = std::max(row_max, v);
      }
    }

    float scale;
    int32_t zp;
    compute_asymmetric_scale_zp(row_min, row_max, scale, zp);
    scales[m] = scale;
    zps[m]    = zp;

    for (int64_t j = 0; j < N; ++j) {
      float v = bf16_scalar_to_f32(row_src[j]);
      int32_t q = static_cast<int32_t>(std::nearbyint(v / scale)) + zp;
      q = std::max(0, std::min(255, q));
      row_dst[j] = static_cast<uint8_t>(q);
    }
  }
}

// --- F32 -> U8 Asymmetric (scalar fused) ---

void dynamic_per_token_quant_f32_u8_ref(const float *src, uint8_t *dst,
                                         float *scales, int32_t *zps,
                                         int64_t M, int64_t N) {
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const float *row_src = src + m * N;
    uint8_t     *row_dst = dst + m * N;

    float row_min = std::numeric_limits<float>::max();
    float row_max = std::numeric_limits<float>::lowest();
    for (int64_t j = 0; j < N; ++j) {
      if (std::isfinite(row_src[j])) {
        row_min = std::min(row_min, row_src[j]);
        row_max = std::max(row_max, row_src[j]);
      }
    }

    float scale;
    int32_t zp;
    compute_asymmetric_scale_zp(row_min, row_max, scale, zp);
    scales[m] = scale;
    zps[m]    = zp;

    for (int64_t j = 0; j < N; ++j) {
      int32_t q = static_cast<int32_t>(std::nearbyint(row_src[j] / scale)) + zp;
      q = std::max(0, std::min(255, q));
      row_dst[j] = static_cast<uint8_t>(q);
    }
  }
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
