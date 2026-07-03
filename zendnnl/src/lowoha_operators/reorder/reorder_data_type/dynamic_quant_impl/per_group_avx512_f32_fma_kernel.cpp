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

#include "lowoha_operators/reorder/reorder_data_type/dynamic_quant_impl/dynamic_kernels.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "common/bfloat16.hpp"
#include "common/float16.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
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
__attribute__((target("avx512f,avx512bw,avx512vl")))
static inline __m512 bf16x16_to_f32_pg(__m256i bf16) {
  return _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16), 16));
}

// Scalar BF16-to-F32 fallback: call common::bfloat16_t::bf16_to_f32_val
// directly with a static_cast<int16_t>(...) on the raw u16 input (BF16 is
// a bit-pattern type; signedness of the 16-bit container is irrelevant).

//==============================================================================
// FP16 <-> F32 conversion: call _mm512_cvtph_ps (F16C; AVX-512F + F16C)
// for vector loads, and common::float16_t::f16_to_f32_val for scalar tail
// elements. No local wrappers needed -- the intrinsic and central API
// already serve directly.
//==============================================================================

/**
 * Returns a 16-bit mask where lane k is 1 iff v[k] is finite (not NaN/Inf).
 * Works by checking |v| < Inf in IEEE 754 representation.
 */
__attribute__((target("avx512f,avx512bw,avx512vl")))
static inline __mmask16 finite_mask_pg(__m512 v, __m512i abs_mask,
                                        __m512 vinf) {
  __m512 absv = _mm512_castsi512_ps(
      _mm512_and_si512(_mm512_castps_si512(v), abs_mask));
  return _mm512_cmp_ps_mask(absv, vinf, _CMP_LT_OQ);
}

//==============================================================================
// Scale / zero-point computation from per-group statistics
//==============================================================================

/**
 * Symmetric quantization: scale = absmax / 127.
 * Clamps absmax to a minimum epsilon to avoid division by zero downstream.
 */
static inline void compute_symmetric_scale_pg(float absmax, float &scale) {
  if (absmax < 1e-10f) absmax = 1e-10f;
  scale = absmax / 127.0f;
  if (scale < 1e-10f) scale = 1e-10f;
}

/**
 * Asymmetric quantization: scale = (max - min) / 255, zp = round(-min / scale).
 * Handles edge cases where all values are identical or no finite values exist.
 */
static inline void compute_asymmetric_scale_zp_pg(float min_val, float max_val,
                                                   float &scale, int32_t &zp) {
  // Empty-group / all-non-finite-group reset. Catches both the F32 init
  // sentinels (numeric_limits<float>::max / lowest) and any non-finite
  // bounds that could appear if a future code path produces NaN/Inf
  // before reaching this helper.
  if (min_val > max_val ||
      !std::isfinite(min_val) || !std::isfinite(max_val)) {
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

__attribute__((target("avx512f,avx512bw,avx512vl")))
static inline void store_4x16_s8_pg(int8_t *dst, __m128i i0, __m128i i1,
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

/** Unsigned byte variant of the cache-line-wide store helper. */
__attribute__((target("avx512f,avx512bw,avx512vl")))
static inline void store_4x16_u8_pg(uint8_t *dst, __m128i i0, __m128i i1,
                                     __m128i i2, __m128i i3,
                                     bool cacheline_aligned) {
  store_4x16_s8_pg(reinterpret_cast<int8_t *>(dst), i0, i1, i2, i3,
                   cacheline_aligned);
}

//==============================================================================
// Per-group statistics reductions
//==============================================================================

/**
 * Compute absmax for one group using 4x unrolled AVX-512 reductions.
 * The caller provides load_f32 so the same reduction can be reused for F32
 * sources and BF16 sources converted to F32 on load.
 */
template <typename LoadF32>
__attribute__((target("avx512f,avx512bw,avx512vl")))
static inline float group_absmax(const LoadF32 &load_f32, int64_t group_size,
                                  __m512i abs_mask, __m512 vinf) {
  __m512 vam0 = _mm512_setzero_ps();
  __m512 vam1 = _mm512_setzero_ps();
  __m512 vam2 = _mm512_setzero_ps();
  __m512 vam3 = _mm512_setzero_ps();

  int64_t j = 0;
  for (; j + 63 < group_size; j += 64) {
    __m512 f0 = load_f32(j);
    __m512 f1 = load_f32(j + 16);
    __m512 f2 = load_f32(j + 32);
    __m512 f3 = load_f32(j + 48);
    __m512 a0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f0), abs_mask));
    __m512 a1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f1), abs_mask));
    __m512 a2 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f2), abs_mask));
    __m512 a3 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f3), abs_mask));
    vam0 = _mm512_mask_max_ps(vam0, finite_mask_pg(f0, abs_mask, vinf), vam0, a0);
    vam1 = _mm512_mask_max_ps(vam1, finite_mask_pg(f1, abs_mask, vinf), vam1, a1);
    vam2 = _mm512_mask_max_ps(vam2, finite_mask_pg(f2, abs_mask, vinf), vam2, a2);
    vam3 = _mm512_mask_max_ps(vam3, finite_mask_pg(f3, abs_mask, vinf), vam3, a3);
  }

  for (; j + 15 < group_size; j += 16) {
    __m512 f = load_f32(j);
    __m512 af = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(f), abs_mask));
    vam0 = _mm512_mask_max_ps(vam0, finite_mask_pg(f, abs_mask, vinf), vam0, af);
  }

  vam0 = _mm512_max_ps(_mm512_max_ps(vam0, vam1),
                        _mm512_max_ps(vam2, vam3));
  return _mm512_reduce_max_ps(vam0);
}

/**
 * Compute min and max for one group using 4x unrolled AVX-512 reductions.
 * Non-finite lanes are masked out to match the scalar reference behavior.
 */
template <typename LoadF32>
__attribute__((target("avx512f,avx512bw,avx512vl")))
static inline void group_minmax(const LoadF32 &load_f32, int64_t group_size,
                                 __m512i abs_mask, __m512 vinf,
                                 float &row_min, float &row_max) {
  __m512 vmin0 = _mm512_set1_ps(std::numeric_limits<float>::max());
  __m512 vmax0 = _mm512_set1_ps(std::numeric_limits<float>::lowest());
  __m512 vmin1 = vmin0, vmax1 = vmax0;
  __m512 vmin2 = vmin0, vmax2 = vmax0;
  __m512 vmin3 = vmin0, vmax3 = vmax0;

  int64_t j = 0;
  for (; j + 63 < group_size; j += 64) {
    __m512 f0 = load_f32(j);
    __m512 f1 = load_f32(j + 16);
    __m512 f2 = load_f32(j + 32);
    __m512 f3 = load_f32(j + 48);
    __mmask16 k0 = finite_mask_pg(f0, abs_mask, vinf);
    __mmask16 k1 = finite_mask_pg(f1, abs_mask, vinf);
    __mmask16 k2 = finite_mask_pg(f2, abs_mask, vinf);
    __mmask16 k3 = finite_mask_pg(f3, abs_mask, vinf);
    vmin0 = _mm512_mask_min_ps(vmin0, k0, vmin0, f0);
    vmax0 = _mm512_mask_max_ps(vmax0, k0, vmax0, f0);
    vmin1 = _mm512_mask_min_ps(vmin1, k1, vmin1, f1);
    vmax1 = _mm512_mask_max_ps(vmax1, k1, vmax1, f1);
    vmin2 = _mm512_mask_min_ps(vmin2, k2, vmin2, f2);
    vmax2 = _mm512_mask_max_ps(vmax2, k2, vmax2, f2);
    vmin3 = _mm512_mask_min_ps(vmin3, k3, vmin3, f3);
    vmax3 = _mm512_mask_max_ps(vmax3, k3, vmax3, f3);
  }

  for (; j + 15 < group_size; j += 16) {
    __m512 f = load_f32(j);
    __mmask16 k = finite_mask_pg(f, abs_mask, vinf);
    vmin0 = _mm512_mask_min_ps(vmin0, k, vmin0, f);
    vmax0 = _mm512_mask_max_ps(vmax0, k, vmax0, f);
  }

  vmin0 = _mm512_min_ps(_mm512_min_ps(vmin0, vmin1),
                         _mm512_min_ps(vmin2, vmin3));
  vmax0 = _mm512_max_ps(_mm512_max_ps(vmax0, vmax1),
                         _mm512_max_ps(vmax2, vmax3));
  row_min = _mm512_reduce_min_ps(vmin0);
  row_max = _mm512_reduce_max_ps(vmax0);
}

//==============================================================================
// Single-group symmetric s8 quant body (one contiguous K-block of
// `group_size` elements).  Shared by the single-tensor per-group kernels
// and the grouped (MoE) per-group scheduler so both produce bit-identical
// results.  `scale_out` receives the computed group scale.
//==============================================================================

__attribute__((target("avx512f")))
static inline void quant_one_group_f32_s8(const float *grp_src, int8_t *grp_dst,
                                          float *scale_out, int64_t group_size,
                                          __m512i abs_mask, __m512 vinf) {
  auto load_f32 = [&](int64_t j) __attribute__((target("avx512f"))) {
    return _mm512_loadu_ps(grp_src + j);
  };

  float absmax = group_absmax(load_f32, group_size, abs_mask, vinf);
  int64_t j = group_size & ~int64_t{15};
  for (; j < group_size; ++j) {
    if (std::isfinite(grp_src[j]))
      absmax = std::max(absmax, std::abs(grp_src[j]));
  }

  float scale;
  compute_symmetric_scale_pg(absmax, scale);
  *scale_out = scale;

  const __m512 vscale = _mm512_set1_ps(scale);
  const bool cl_ok = (reinterpret_cast<uintptr_t>(grp_dst) & 63) == 0;

  j = 0;
  for (; j + 63 < group_size; j += 64) {
    __m512i r0 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j), vscale));
    __m512i r1 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 16), vscale));
    __m512i r2 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 32), vscale));
    __m512i r3 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 48), vscale));
    store_4x16_s8_pg(grp_dst + j,
                      _mm512_cvtepi32_epi8(r0), _mm512_cvtepi32_epi8(r1),
                      _mm512_cvtepi32_epi8(r2), _mm512_cvtepi32_epi8(r3),
                      cl_ok);
  }
  for (; j + 15 < group_size; j += 16) {
    __m512i r = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j), vscale));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(grp_dst + j),
                     _mm512_cvtepi32_epi8(r));
  }
  for (; j < group_size; ++j) {
    if (!std::isfinite(grp_src[j])) { grp_dst[j] = 0; continue; }
    int32_t q = static_cast<int32_t>(std::nearbyint(grp_src[j] / scale));
    grp_dst[j] = static_cast<int8_t>(q);
  }
}

__attribute__((target("avx512f")))
static inline void quant_one_group_bf16_s8(const uint16_t *grp_src,
                                           int8_t *grp_dst, float *scale_out,
                                           int64_t group_size,
                                           __m512i abs_mask, __m512 vinf) {
  auto load_f32 = [&](int64_t j) __attribute__((target("avx512f"))) {
    return bf16x16_to_f32_pg(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(grp_src + j)));
  };

  float absmax = group_absmax(load_f32, group_size, abs_mask, vinf);
  int64_t j = group_size & ~int64_t{15};
  for (; j < group_size; ++j) {
    float v = common::bfloat16_t::bf16_to_f32_val(static_cast<int16_t>(grp_src[j]));
    if (std::isfinite(v))
      absmax = std::max(absmax, std::abs(v));
  }

  float scale;
  compute_symmetric_scale_pg(absmax, scale);
  *scale_out = scale;

  const __m512 vscale = _mm512_set1_ps(scale);
  const bool cl_ok = (reinterpret_cast<uintptr_t>(grp_dst) & 63) == 0;

  j = 0;
  for (; j + 63 < group_size; j += 64) {
    __m512i r0 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j), vscale));
    __m512i r1 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 16), vscale));
    __m512i r2 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 32), vscale));
    __m512i r3 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 48), vscale));
    store_4x16_s8_pg(grp_dst + j,
                      _mm512_cvtepi32_epi8(r0), _mm512_cvtepi32_epi8(r1),
                      _mm512_cvtepi32_epi8(r2), _mm512_cvtepi32_epi8(r3),
                      cl_ok);
  }
  for (; j + 15 < group_size; j += 16) {
    __m512i r = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j), vscale));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(grp_dst + j),
                     _mm512_cvtepi32_epi8(r));
  }
  for (; j < group_size; ++j) {
    float v = common::bfloat16_t::bf16_to_f32_val(static_cast<int16_t>(grp_src[j]));
    if (!std::isfinite(v)) { grp_dst[j] = 0; continue; }
    int32_t q = static_cast<int32_t>(std::nearbyint(v / scale));
    grp_dst[j] = static_cast<int8_t>(q);
  }
}

//==============================================================================
// KERNEL 1:  F32 -> S8 Symmetric Per-Group Quantization  (AVX-512F)
//==============================================================================
//
// Steps (per group, fused):
//   1. Pass 1 - Absmax reduction: scan K/G F32 elements in the group,
//      compute absmax = max(|val|) using AND+MAX with non-finite masking.
//   2. Compute scale: scale = absmax / 127 (clamped to >= 1e-10).
//   3. Write scale to scales[m * G + g].
//   4. Pass 2 - Quantize: re-read F32 from L1 cache, compute
//      Q[j] = round(val[j] / scale), narrow to int8 via truncation
//      (VPMOVDB, no saturation), store to dst.
//
// Register usage (peak, per thread):
//   zmm (512-bit):  ~14 total
//     Pass 1:  abs_mask(1) + vinf(1) + f0-f3(4) + a0-a3(4)
//              + vam0-vam3(4) = 14
//     Pass 2:  vscale(1) + r0-r3(4) = 5  (F32 loaded into r temporaries)
//   k-mask:    4  (finite_mask_pg results)
//   Scalar:    scale, absmax, task, m, g, j, cl_ok
//
// Optimizations:
//   1. Fused per-group:  statistics and quantization are done back-to-back
//      while the group data is still hot in cache.
//   2. Shared reduction helper:  load_f32 abstracts F32 vs BF16 load paths
//      while preserving the same vectorized reduction logic.
//   3. Absmax via AND+MAX:  clears sign bit with VPANDD then VMAXPS.
//   4. Non-finite masking:  finite_mask_pg() + VMAXPS{k} skips NaN/Inf lanes.
//   5. 4x unrolling:  64 elements/iter across independent accumulators for ILP.
//   6. VPMOVDB truncating narrow:  int32->int8 low-byte narrow; finite values
//      are in [-127,127] by construction, non-finite -> 0 (matching vLLM).
//   7. Cache-line stores:  packs 4x __m128i into a single 64B aligned store.
//   8. True division + banker's rounding:  bit-exact with reference behavior.
//==============================================================================

__attribute__((target("avx512f,avx512bw,avx512vl")))
void dynamic_per_group_quant_f32_s8_native(const float *src, int8_t *dst,
                                            float *scales,
                                            int64_t M, int64_t K, int64_t G) {
  if (M <= 0 || K <= 0 || G <= 0 || (K % G) != 0) return;

  const int64_t group_size = K / G;
  const int64_t total_groups = M * G;
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());

  zendnnl_parallel_for(0, total_groups, 1,
      [&](int64_t begin, int64_t end) __attribute__((target("avx512f,avx512bw,avx512vl"))) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t m = task / G;
      const int64_t g = task - m * G;
      const int64_t offset = m * K + g * group_size;
      const float *grp_src = src + offset;
      int8_t *grp_dst = dst + offset;

      auto load_f32 = [&](int64_t j) __attribute__((target("avx512f,avx512bw,avx512vl"))) {
        return _mm512_loadu_ps(grp_src + j);
      };

      float absmax = group_absmax(load_f32, group_size, abs_mask, vinf);
      int64_t j = group_size & ~int64_t{15};
      for (; j < group_size; ++j) {
        if (std::isfinite(grp_src[j]))
          absmax = std::max(absmax, std::abs(grp_src[j]));
      }

      float scale;
      compute_symmetric_scale_pg(absmax, scale);
      scales[task] = scale;

      const __m512 vscale = _mm512_set1_ps(scale);
      const bool cl_ok = (reinterpret_cast<uintptr_t>(grp_dst) & 63) == 0;

      j = 0;
      for (; j + 63 < group_size; j += 64) {
        __m512i r0 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j), vscale));
        __m512i r1 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 16), vscale));
        __m512i r2 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 32), vscale));
        __m512i r3 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 48), vscale));
        store_4x16_s8_pg(grp_dst + j,
                          _mm512_cvtepi32_epi8(r0), _mm512_cvtepi32_epi8(r1),
                          _mm512_cvtepi32_epi8(r2), _mm512_cvtepi32_epi8(r3),
                          cl_ok);
      }
      for (; j + 15 < group_size; j += 16) {
        __m512i r = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j), vscale));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(grp_dst + j),
                         _mm512_cvtepi32_epi8(r));
      }
      for (; j < group_size; ++j) {
        if (!std::isfinite(grp_src[j])) { grp_dst[j] = 0; continue; }
        int32_t q = static_cast<int32_t>(std::nearbyint(grp_src[j] / scale));
        grp_dst[j] = static_cast<int8_t>(q);
      }
    }
  });
}

//==============================================================================
// KERNEL 2:  BF16 -> S8 Symmetric Per-Group Quantization  (AVX-512F)
//==============================================================================
//
// Steps (per group, fused):
//   1. Pass 1 - Absmax reduction: scan K/G BF16 elements, convert to F32,
//      compute absmax = max(|val|) using AND+MAX with non-finite masking.
//   2. Compute scale: scale = absmax / 127 (clamped to >= 1e-10).
//   3. Write scale to scales[m * G + g].
//   4. Pass 2 - Quantize: re-read BF16 from L1, convert to F32, compute
//      Q[j] = round(val[j] / scale), narrow to int8 via truncation
//      (VPMOVDB, no saturation), store to dst.
//
// Register usage (peak, per thread):
//   zmm (512-bit):  ~14 total
//     Pass 1:  abs_mask(1) + vinf(1) + f0-f3(4) + a0-a3(4)
//              + vam0-vam3(4) = 14
//     Pass 2:  vscale(1) + f0-f3(4) + r0-r3(4) = 9
//   k-mask:    4  (finite_mask_pg results)
//   Scalar:    scale, absmax, task, m, g, j, cl_ok
//
// Optimizations:
//   1. Fused per-group:  Pass 1 + Pass 2 back-to-back keeps group data hot.
//   2. BF16 conversion in registers:  zero-extend BF16 lanes and shift left
//      to reconstruct F32 values without scratch buffers.
//   3. Absmax via AND+MAX with non-finite masking.
//   4. 4x unrolling:  64 elements/iter for ILP.
//   5. VPMOVDB truncating narrow:  int32->int8 low-byte narrow; finite values
//      are in [-127,127] by construction, non-finite -> 0 (matching vLLM).
//   6. Cache-line stores:  64B aligned writes when destination is aligned.
//   7. True division + banker's rounding:  bit-exact with reference behavior.
//==============================================================================

__attribute__((target("avx512f,avx512bw,avx512vl")))
void dynamic_per_group_quant_bf16_s8_native(const uint16_t *src, int8_t *dst,
                                             float *scales,
                                             int64_t M, int64_t K, int64_t G) {
  if (M <= 0 || K <= 0 || G <= 0 || (K % G) != 0) return;

  const int64_t group_size = K / G;
  const int64_t total_groups = M * G;
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());

  zendnnl_parallel_for(0, total_groups, 1,
      [&](int64_t begin, int64_t end) __attribute__((target("avx512f,avx512bw,avx512vl"))) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t m = task / G;
      const int64_t g = task - m * G;
      const int64_t offset = m * K + g * group_size;
      const uint16_t *grp_src = src + offset;
      int8_t *grp_dst = dst + offset;

      auto load_f32 = [&](int64_t j) __attribute__((target("avx512f,avx512bw,avx512vl"))) {
        return bf16x16_to_f32_pg(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(grp_src + j)));
      };

      float absmax = group_absmax(load_f32, group_size, abs_mask, vinf);
      int64_t j = group_size & ~int64_t{15};
      for (; j < group_size; ++j) {
        float v = common::bfloat16_t::bf16_to_f32_val(static_cast<int16_t>(grp_src[j]));
        if (std::isfinite(v))
          absmax = std::max(absmax, std::abs(v));
      }

      float scale;
      compute_symmetric_scale_pg(absmax, scale);
      scales[task] = scale;

      const __m512 vscale = _mm512_set1_ps(scale);
      const bool cl_ok = (reinterpret_cast<uintptr_t>(grp_dst) & 63) == 0;

      j = 0;
      for (; j + 63 < group_size; j += 64) {
        __m512i r0 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j), vscale));
        __m512i r1 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 16), vscale));
        __m512i r2 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 32), vscale));
        __m512i r3 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 48), vscale));
        store_4x16_s8_pg(grp_dst + j,
                          _mm512_cvtepi32_epi8(r0), _mm512_cvtepi32_epi8(r1),
                          _mm512_cvtepi32_epi8(r2), _mm512_cvtepi32_epi8(r3),
                          cl_ok);
      }
      for (; j + 15 < group_size; j += 16) {
        __m512i r = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j), vscale));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(grp_dst + j),
                         _mm512_cvtepi32_epi8(r));
      }
      for (; j < group_size; ++j) {
        float v = common::bfloat16_t::bf16_to_f32_val(static_cast<int16_t>(grp_src[j]));
        if (!std::isfinite(v)) { grp_dst[j] = 0; continue; }
        int32_t q = static_cast<int32_t>(std::nearbyint(v / scale));
        grp_dst[j] = static_cast<int8_t>(q);
      }
    }
  });
}

//==============================================================================
// KERNEL 3:  F32 -> U8 Asymmetric Per-Group Quantization  (AVX-512F)
//==============================================================================
//
// Steps (per group, fused):
//   1. Pass 1 - Min/Max reduction: scan K/G F32 elements and compute
//      row_min and row_max while skipping non-finite values.
//   2. Compute scale and zero-point:
//        scale = (row_max - row_min) / 255  (clamped to >= 1e-10)
//        zp    = round(-row_min / scale)    (clamped to int32 range)
//   3. Write scale to scales[m * G + g], zp to zps[m * G + g].
//   4. Pass 2 - Quantize with clamp: re-read F32 from L1, compute
//      Q[j] = clamp(round(val[j] / scale) + zp, 0, 255), narrow to uint8,
//      store to dst.
//
// Register usage (peak, per thread):
//   zmm (512-bit):  ~14 total
//     Pass 1:  abs_mask(1) + vinf(1) + f0-f3(4) + vmin0-3(4)
//              + vmax0-3(4) = 14
//     Pass 2:  vscale(1) + vzp(1) + vlo(1) + vhi(1) + r0-r3(4) = 8
//   k-mask:    4  (finite_mask_pg results)
//   Scalar:    scale, zp, row_min, row_max, task, m, g, j, cl_ok
//
// Optimizations:
//   1. Fused per-group:  min/max and quantization are done while data is hot.
//   2. Min/Max with non-finite masking:  VMINPS{k}/VMAXPS{k} skip NaN/Inf.
//   3. Shared reduction helper keeps F32 and BF16 paths aligned.
//   4. 4x unrolling:  64 elements/iter with independent min/max accumulators.
//   5. Explicit [0,255] clamp before VPMOVUSDB to avoid negative values
//      saturating to 255.
//   6. Cache-line stores:  64B aligned writes when destination is aligned.
//   7. True division + banker's rounding:  bit-exact with reference behavior.
//==============================================================================

__attribute__((target("avx512f,avx512bw,avx512vl")))
void dynamic_per_group_quant_f32_u8_native(const float *src, uint8_t *dst,
                                            float *scales, int32_t *zps,
                                            int64_t M, int64_t K, int64_t G) {
  if (M <= 0 || K <= 0 || G <= 0 || (K % G) != 0) return;

  const int64_t group_size = K / G;
  const int64_t total_groups = M * G;
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());

  zendnnl_parallel_for(0, total_groups, 1,
      [&](int64_t begin, int64_t end) __attribute__((target("avx512f,avx512bw,avx512vl"))) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t m = task / G;
      const int64_t g = task - m * G;
      const int64_t offset = m * K + g * group_size;
      const float *grp_src = src + offset;
      uint8_t *grp_dst = dst + offset;

      auto load_f32 = [&](int64_t j) __attribute__((target("avx512f,avx512bw,avx512vl"))) {
        return _mm512_loadu_ps(grp_src + j);
      };

      float row_min, row_max;
      group_minmax(load_f32, group_size, abs_mask, vinf, row_min, row_max);
      int64_t j = group_size & ~int64_t{15};
      for (; j < group_size; ++j) {
        if (std::isfinite(grp_src[j])) {
          row_min = std::min(row_min, grp_src[j]);
          row_max = std::max(row_max, grp_src[j]);
        }
      }

      float scale;
      int32_t zp;
      compute_asymmetric_scale_zp_pg(row_min, row_max, scale, zp);
      scales[task] = scale;
      zps[task] = zp;

      const __m512 vscale = _mm512_set1_ps(scale);
      const __m512i vzp = _mm512_set1_epi32(zp);
      const __m512i vlo = _mm512_set1_epi32(0);
      const __m512i vhi = _mm512_set1_epi32(255);
      const bool cl_ok = (reinterpret_cast<uintptr_t>(grp_dst) & 63) == 0;

      j = 0;
      for (; j + 63 < group_size; j += 64) {
        __m512i r0 = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j), vscale)), vzp);
        __m512i r1 = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j + 16), vscale)), vzp);
        __m512i r2 = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j + 32), vscale)), vzp);
        __m512i r3 = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j + 48), vscale)), vzp);
        r0 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r0));
        r1 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r1));
        r2 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r2));
        r3 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r3));
        store_4x16_u8_pg(grp_dst + j,
                          _mm512_cvtusepi32_epi8(r0), _mm512_cvtusepi32_epi8(r1),
                          _mm512_cvtusepi32_epi8(r2), _mm512_cvtusepi32_epi8(r3),
                          cl_ok);
      }
      for (; j + 15 < group_size; j += 16) {
        __m512i r = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j), vscale)), vzp);
        r = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(grp_dst + j),
                         _mm512_cvtusepi32_epi8(r));
      }
      for (; j < group_size; ++j) {
        if (!std::isfinite(grp_src[j])) { grp_dst[j] = 0; continue; }
        int32_t q = static_cast<int32_t>(std::nearbyint(grp_src[j] / scale)) + zp;
        q = std::max(0, std::min(255, q));
        grp_dst[j] = static_cast<uint8_t>(q);
      }
    }
  });
}

//==============================================================================
// KERNEL 4:  BF16 -> U8 Asymmetric Per-Group Quantization  (AVX-512F)
//==============================================================================
//
// Steps (per group, fused):
//   1. Pass 1 - Min/Max reduction: scan K/G BF16 elements, convert to F32,
//      compute row_min and row_max while skipping non-finite values.
//   2. Compute scale and zero-point:
//        scale = (row_max - row_min) / 255  (clamped to >= 1e-10)
//        zp    = round(-row_min / scale)    (clamped to int32 range)
//   3. Write scale to scales[m * G + g], zp to zps[m * G + g].
//   4. Pass 2 - Quantize with clamp: re-read BF16 from L1, convert to F32,
//      compute Q[j] = clamp(round(val[j] / scale) + zp, 0, 255), narrow to
//      uint8, store to dst.
//
// Register usage (peak, per thread):
//   zmm (512-bit):  ~14 total
//     Pass 1:  abs_mask(1) + vinf(1) + f0-f3(4) + vmin0-3(4)
//              + vmax0-3(4) = 14
//     Pass 2:  vscale(1) + vzp(1) + vlo(1) + vhi(1) + f0-f3(4)
//              + r0-r3(4) = 12
//   k-mask:    4  (finite_mask_pg results)
//   Scalar:    scale, zp, row_min, row_max, task, m, g, j, cl_ok
//
// Optimizations:
//   1. Fused per-group:  Pass 1 + Pass 2 back-to-back keeps group data hot.
//   2. BF16 conversion in registers avoids per-call scratch buffers.
//   3. Min/Max with non-finite masking:  VMINPS{k}/VMAXPS{k}.
//   4. 4x unrolling:  64 elements/iter for ILP.
//   5. Explicit [0,255] clamp before VPMOVUSDB to preserve reference behavior.
//   6. Cache-line stores:  64B aligned writes when destination is aligned.
//   7. True division + banker's rounding:  bit-exact with reference behavior.
//==============================================================================

__attribute__((target("avx512f,avx512bw,avx512vl")))
void dynamic_per_group_quant_bf16_u8_native(const uint16_t *src, uint8_t *dst,
                                             float *scales, int32_t *zps,
                                             int64_t M, int64_t K, int64_t G) {
  if (M <= 0 || K <= 0 || G <= 0 || (K % G) != 0) return;

  const int64_t group_size = K / G;
  const int64_t total_groups = M * G;
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());

  zendnnl_parallel_for(0, total_groups, 1,
      [&](int64_t begin, int64_t end) __attribute__((target("avx512f,avx512bw,avx512vl"))) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t m = task / G;
      const int64_t g = task - m * G;
      const int64_t offset = m * K + g * group_size;
      const uint16_t *grp_src = src + offset;
      uint8_t *grp_dst = dst + offset;

      auto load_f32 = [&](int64_t j) __attribute__((target("avx512f,avx512bw,avx512vl"))) {
        return bf16x16_to_f32_pg(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(grp_src + j)));
      };

      float row_min, row_max;
      group_minmax(load_f32, group_size, abs_mask, vinf, row_min, row_max);
      int64_t j = group_size & ~int64_t{15};
      for (; j < group_size; ++j) {
        float v = common::bfloat16_t::bf16_to_f32_val(static_cast<int16_t>(grp_src[j]));
        if (std::isfinite(v)) {
          row_min = std::min(row_min, v);
          row_max = std::max(row_max, v);
        }
      }

      float scale;
      int32_t zp;
      compute_asymmetric_scale_zp_pg(row_min, row_max, scale, zp);
      scales[task] = scale;
      zps[task] = zp;

      const __m512 vscale = _mm512_set1_ps(scale);
      const __m512i vzp = _mm512_set1_epi32(zp);
      const __m512i vlo = _mm512_set1_epi32(0);
      const __m512i vhi = _mm512_set1_epi32(255);
      const bool cl_ok = (reinterpret_cast<uintptr_t>(grp_dst) & 63) == 0;

      j = 0;
      for (; j + 63 < group_size; j += 64) {
        __m512i r0 = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j), vscale)), vzp);
        __m512i r1 = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j + 16), vscale)), vzp);
        __m512i r2 = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j + 32), vscale)), vzp);
        __m512i r3 = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j + 48), vscale)), vzp);
        r0 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r0));
        r1 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r1));
        r2 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r2));
        r3 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r3));
        store_4x16_u8_pg(grp_dst + j,
                          _mm512_cvtusepi32_epi8(r0), _mm512_cvtusepi32_epi8(r1),
                          _mm512_cvtusepi32_epi8(r2), _mm512_cvtusepi32_epi8(r3),
                          cl_ok);
      }
      for (; j + 15 < group_size; j += 16) {
        __m512i r = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j), vscale)), vzp);
        r = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(grp_dst + j),
                         _mm512_cvtusepi32_epi8(r));
      }
      for (; j < group_size; ++j) {
        float v = common::bfloat16_t::bf16_to_f32_val(static_cast<int16_t>(grp_src[j]));
        if (!std::isfinite(v)) { grp_dst[j] = 0; continue; }
        int32_t q = static_cast<int32_t>(std::nearbyint(v / scale)) + zp;
        q = std::max(0, std::min(255, q));
        grp_dst[j] = static_cast<uint8_t>(q);
      }
    }
  });
}

//==============================================================================
// FP16 PER-GROUP KERNELS — F32-FMA backend (Strategy B)
//
// Layout mirrors the BF16 per-group kernels above 1:1 with two mechanical
// swaps in the load lambda and the scalar tail:
// Identical structure to the BF16 per-group kernels above, with two
// mechanical swaps in the load lambda and the scalar tail (the BF16 path
// uses bf16x16_to_f32_pg + common::bfloat16_t::bf16_to_f32_val, the FP16
// path uses _mm512_cvtph_ps + common::float16_t::f16_to_f32_val).
//
// The companion __m512h-native FP16-FMA kernels live in
// per_group_avx512_f16_fma_kernel.cpp.
//==============================================================================

//==============================================================================
// KERNEL 5:  F16 -> S8 Symmetric Per-Group Quantization  (AVX-512F + F16C)
//==============================================================================
//
// Steps (per group, fused):
//   1. Pass 1 - Absmax reduction: scan K/G F16 elements in the group,
//      convert to F32 via VCVTPH2PS (F16C), compute absmax = max(|val|)
//      with non-finite masking.
//   2. Compute scale: scale = absmax / 127 (clamped to >= 1e-10).
//   3. Write scale to scales[m * G + g].
//   4. Pass 2 - Quantize: re-read F16 from L1 cache (still hot from Pass 1),
//      convert to F32, compute Q[j] = round(val[j] / scale), narrow to int8
//      via truncation (VPMOVDB, no saturation), store to dst.
//
// Register usage (peak, per thread):
//   zmm (512-bit):  ~14 total
//     Pass 1:  abs_mask(1) + vinf(1) + f0-f3(4) + a0-a3(4)
//              + vam0-vam3(4) = 14
//     Pass 2:  vscale(1) + f0-f3(4) + r0-r3(4) = 9
//   k-mask:    4  (finite_mask_pg results)
//   Scalar:    scale, absmax, task, m, g, j
//
// Optimizations:
//   1. Fused per-group:  Pass 1 + Pass 2 back-to-back keeps the group's
//      F16 data hot in cache, avoiding a second DRAM read.
//   2. F16C widen:  VCVTPH2PS converts 16 F16 lanes per instruction.
//      Requires F16C; available on every shipping AVX-512F host (CPUID
//      bits are independent but the superset relation holds in practice).
//      No AVX512-FP16 ISA needed.
//   3. Shared reduction helper:  load_f32 lambda abstracts the F16->F32
//      load path while reusing the generic group_absmax template.
//   4. Absmax via AND+MAX, non-finite masking, 4x unroll, VPMOVDB truncating
//      narrow (non-finite -> 0), cache-line stores -- same techniques as KERNEL 2.
//   5. zendnnl_parallel_for over M*G total groups for thread-pool reuse
//      across granular tasks.
//==============================================================================

__attribute__((target("avx512f,avx512bw,avx512vl,f16c")))
void dynamic_per_group_quant_f16_s8_native(const uint16_t *src, int8_t *dst,
                                            float *scales,
                                            int64_t M, int64_t K, int64_t G) {
  if (M <= 0 || K <= 0 || G <= 0 || (K % G) != 0) return;

  const int64_t group_size = K / G;
  const int64_t total_groups = M * G;
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());

  zendnnl_parallel_for(0, total_groups, 1,
      [&](int64_t begin, int64_t end) __attribute__((target("avx512f,avx512bw,avx512vl,f16c"))) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t m = task / G;
      const int64_t g = task - m * G;
      const int64_t offset = m * K + g * group_size;
      const uint16_t *grp_src = src + offset;
      int8_t *grp_dst = dst + offset;

      auto load_f32 = [&](int64_t j) __attribute__((target("avx512f,avx512bw,avx512vl,f16c"))) {
        return _mm512_cvtph_ps(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(grp_src + j)));
      };

      float absmax = group_absmax(load_f32, group_size, abs_mask, vinf);
      int64_t j = group_size & ~int64_t{15};
      for (; j < group_size; ++j) {
        float v = common::float16_t::f16_to_f32_val(grp_src[j]);
        if (std::isfinite(v))
          absmax = std::max(absmax, std::abs(v));
      }

      float scale;
      compute_symmetric_scale_pg(absmax, scale);
      scales[task] = scale;

      const __m512 vscale = _mm512_set1_ps(scale);
      const bool cl_ok = (reinterpret_cast<uintptr_t>(grp_dst) & 63) == 0;

      j = 0;
      for (; j + 63 < group_size; j += 64) {
        __m512i r0 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j), vscale));
        __m512i r1 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 16), vscale));
        __m512i r2 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 32), vscale));
        __m512i r3 = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j + 48), vscale));
        store_4x16_s8_pg(grp_dst + j,
                          _mm512_cvtepi32_epi8(r0), _mm512_cvtepi32_epi8(r1),
                          _mm512_cvtepi32_epi8(r2), _mm512_cvtepi32_epi8(r3),
                          cl_ok);
      }
      for (; j + 15 < group_size; j += 16) {
        __m512i r = _mm512_cvtps_epi32(_mm512_div_ps(load_f32(j), vscale));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(grp_dst + j),
                         _mm512_cvtepi32_epi8(r));
      }
      for (; j < group_size; ++j) {
        float v = common::float16_t::f16_to_f32_val(grp_src[j]);
        if (!std::isfinite(v)) { grp_dst[j] = 0; continue; }
        int32_t q = static_cast<int32_t>(std::nearbyint(v / scale));
        grp_dst[j] = static_cast<int8_t>(q);
      }
    }
  });
}

//==============================================================================
// KERNEL 6:  F16 -> U8 Asymmetric Per-Group Quantization  (AVX-512F + F16C)
//==============================================================================
//
// Steps (per group, fused):
//   1. Pass 1 - Min/Max reduction: scan K/G F16 elements in the group,
//      convert to F32 via VCVTPH2PS, compute row_min/row_max with non-finite
//      masking.
//   2. Compute scale + zp: scale = (max - min) / 255 (clamped to >= 1e-10);
//      zp = round(-min / scale), clamped to int32 range.
//   3. Write scale to scales[m * G + g] and zp to zps[m * G + g].
//   4. Pass 2 - Quantize: re-read F16 from L1, convert to F32, compute
//      Q[j] = round(val[j] / scale) + zp, clamp to [0, 255] in signed s32
//      (VPMAXSD/VPMINSD), narrow to uint8 with VPMOVUSDB and cache-line
//      store.
//
// Register usage (peak, per thread):
//   zmm (512-bit):  ~16 total
//     Pass 1:  abs_mask(1) + vinf(1) + f0-f3(4) + vmin0-3(4) + vmax0-3(4) = 14
//     Pass 2:  vscale(1) + vzp(1) + vlo32(1) + vhi32(1) + f0-f3(4) + r0-r3(4) = 12
//   k-mask:    4  (finite_mask_pg results)
//   Scalar:    scale, zp, row_min, row_max, task, m, g, j
//
// Optimizations:
//   1. Fused per-group:  same L1 cache reuse as KERNEL 5.
//   2. F16C widen, min/max with non-finite masking, 4x unroll -- same as
//      KERNEL 5.
//   3. Explicit [0, 255] clamp:  VPMAXSD/VPMINSD before VPMOVUSDB, since
//      VPMOVUSDB treats negative int32 as large unsigned and saturates
//      to 255 instead of 0.
//   4. Cache-line stores:  64B aligned writes.
//   5. zendnnl_parallel_for over M*G total groups for thread-pool reuse.
//==============================================================================

__attribute__((target("avx512f,avx512bw,avx512vl,f16c")))
void dynamic_per_group_quant_f16_u8_native(const uint16_t *src, uint8_t *dst,
                                            float *scales, int32_t *zps,
                                            int64_t M, int64_t K, int64_t G) {
  if (M <= 0 || K <= 0 || G <= 0 || (K % G) != 0) return;

  const int64_t group_size = K / G;
  const int64_t total_groups = M * G;
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());

  zendnnl_parallel_for(0, total_groups, 1,
      [&](int64_t begin, int64_t end) __attribute__((target("avx512f,avx512bw,avx512vl,f16c"))) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t m = task / G;
      const int64_t g = task - m * G;
      const int64_t offset = m * K + g * group_size;
      const uint16_t *grp_src = src + offset;
      uint8_t *grp_dst = dst + offset;

      auto load_f32 = [&](int64_t j) __attribute__((target("avx512f,avx512bw,avx512vl,f16c"))) {
        return _mm512_cvtph_ps(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(grp_src + j)));
      };

      float row_min, row_max;
      group_minmax(load_f32, group_size, abs_mask, vinf, row_min, row_max);
      int64_t j = group_size & ~int64_t{15};
      for (; j < group_size; ++j) {
        float v = common::float16_t::f16_to_f32_val(grp_src[j]);
        if (std::isfinite(v)) {
          row_min = std::min(row_min, v);
          row_max = std::max(row_max, v);
        }
      }

      float scale;
      int32_t zp;
      compute_asymmetric_scale_zp_pg(row_min, row_max, scale, zp);
      scales[task] = scale;
      zps[task] = zp;

      const __m512 vscale = _mm512_set1_ps(scale);
      const __m512i vzp = _mm512_set1_epi32(zp);
      const __m512i vlo = _mm512_set1_epi32(0);
      const __m512i vhi = _mm512_set1_epi32(255);
      const bool cl_ok = (reinterpret_cast<uintptr_t>(grp_dst) & 63) == 0;

      j = 0;
      for (; j + 63 < group_size; j += 64) {
        __m512i r0 = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j), vscale)), vzp);
        __m512i r1 = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j + 16), vscale)), vzp);
        __m512i r2 = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j + 32), vscale)), vzp);
        __m512i r3 = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j + 48), vscale)), vzp);
        r0 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r0));
        r1 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r1));
        r2 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r2));
        r3 = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r3));
        store_4x16_u8_pg(grp_dst + j,
                          _mm512_cvtusepi32_epi8(r0), _mm512_cvtusepi32_epi8(r1),
                          _mm512_cvtusepi32_epi8(r2), _mm512_cvtusepi32_epi8(r3),
                          cl_ok);
      }
      for (; j + 15 < group_size; j += 16) {
        __m512i r = _mm512_add_epi32(_mm512_cvtps_epi32(
            _mm512_div_ps(load_f32(j), vscale)), vzp);
        r = _mm512_max_epi32(vlo, _mm512_min_epi32(vhi, r));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(grp_dst + j),
                         _mm512_cvtusepi32_epi8(r));
      }
      for (; j < group_size; ++j) {
        float v = common::float16_t::f16_to_f32_val(grp_src[j]);
        if (!std::isfinite(v)) { grp_dst[j] = 0; continue; }
        int32_t q = static_cast<int32_t>(std::nearbyint(v / scale)) + zp;
        q = std::max(0, std::min(255, q));
        grp_dst[j] = static_cast<uint8_t>(q);
      }
    }
  });
}

//==============================================================================
// Grouped (MoE) per-group symmetric s8 dynamic quantization.
//
// Independent [M_i, K_i] sources, each carrying a {M_i, G} scale layout
// (one scale per (row, K-block), linear index m*G + g).  All experts share
// a single OpenMP schedule over sum(M_i) rows; each row task walks its G
// K-blocks serially (contiguous in memory), computing one scale and
// quantizing one block at a time via the shared single-group body.  Because
// every block is still produced by that identical body, the grouped result
// is bit-identical to the single-tensor per-group kernels regardless of how
// rows are distributed across threads (parity contract).  G is uniform
// across experts; group_size = K_i / G (must divide K_i exactly).
//==============================================================================

static int pg_omp_team_size(int num_threads) {
  return num_threads > 0 ? num_threads : omp_get_max_threads();
}

template <typename RowFn>
static void dynamic_per_group_group_quant_s8_impl(const std::vector<int> &M,
                                                  int num_threads,
                                                  RowFn row_fn) {
  const size_t num_ops = M.size();
  std::vector<int64_t> row_prefix(num_ops);
  int64_t total_rows = 0;
  for (size_t i = 0; i < num_ops; ++i) {
    total_rows += static_cast<int64_t>(std::max(0, M[i]));
    row_prefix[i] = total_rows;
  }
  if (total_rows <= 0) return;

  const int nt = std::min<int64_t>(pg_omp_team_size(num_threads), total_rows);
  #pragma omp parallel num_threads(nt)
  {
    const int tid = omp_get_thread_num();
    const int nthr = omp_get_num_threads();
    const int64_t begin = total_rows * tid / nthr;
    const int64_t end = total_rows * (tid + 1) / nthr;

    for (int64_t gr = begin; gr < end; ++gr) {
      const auto it = std::upper_bound(row_prefix.begin(), row_prefix.end(), gr);
      const size_t op = static_cast<size_t>(it - row_prefix.begin());
      const int64_t op_base = (op == 0) ? 0 : row_prefix[op - 1];
      const int64_t local_m = gr - op_base;
      row_fn(op, local_m);
    }
  }
}

__attribute__((target("avx512f")))
void dynamic_per_group_group_quant_bf16_s8_native(
    const std::vector<const void *> &src, const std::vector<int> &M,
    const std::vector<int> &K, const std::vector<int> &lda,
    const std::vector<void *> &dst, const std::vector<int> &dst_lda,
    const std::vector<float *> &scales, int64_t G, int num_threads) {
  if (G <= 0) return;
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());
  dynamic_per_group_group_quant_s8_impl(
      M, num_threads,
      [&](size_t op, int64_t local_m) __attribute__((target("avx512f"))) {
        const int64_t group_size = K[op] / G;
        const uint16_t *row_src =
            static_cast<const uint16_t *>(src[op]) + local_m * lda[op];
        int8_t *row_dst = static_cast<int8_t *>(dst[op]) + local_m * dst_lda[op];
        float *row_scales = scales[op] + local_m * G;
        for (int64_t g = 0; g < G; ++g) {
          quant_one_group_bf16_s8(row_src + g * group_size,
                                  row_dst + g * group_size, row_scales + g,
                                  group_size, abs_mask, vinf);
        }
      });
}

__attribute__((target("avx512f")))
void dynamic_per_group_group_quant_f32_s8_native(
    const std::vector<const void *> &src, const std::vector<int> &M,
    const std::vector<int> &K, const std::vector<int> &lda,
    const std::vector<void *> &dst, const std::vector<int> &dst_lda,
    const std::vector<float *> &scales, int64_t G, int num_threads) {
  if (G <= 0) return;
  const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
  const __m512  vinf     = _mm512_set1_ps(std::numeric_limits<float>::infinity());
  dynamic_per_group_group_quant_s8_impl(
      M, num_threads,
      [&](size_t op, int64_t local_m) __attribute__((target("avx512f"))) {
        const int64_t group_size = K[op] / G;
        const float *row_src =
            static_cast<const float *>(src[op]) + local_m * lda[op];
        int8_t *row_dst = static_cast<int8_t *>(dst[op]) + local_m * dst_lda[op];
        float *row_scales = scales[op] + local_m * G;
        for (int64_t g = 0; g < G; ++g) {
          quant_one_group_f32_s8(row_src + g * group_size,
                                 row_dst + g * group_size, row_scales + g,
                                 group_size, abs_mask, vinf);
        }
      });
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
