/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

//==============================================================================
// FP16 PER-GROUP DYNAMIC QUANTIZATION — FP16-FMA backend (Strategy A)
//
// Native __m512h per-group kernels. Same layout convention as the F32-FMA
// per_group_avx512_f32_fma_kernel.cpp counterpart: scales are written at
// scales[m * G + g], group_size = K / G, parallelism is one group per
// task across (M*G) tasks via zendnnl_parallel_for.
//
// Requires AVX512-FP16 ISA (gated upstream by the dispatcher). On
// toolchains older than GCC 12, the kernels compile to empty stubs and
// the dispatcher falls back to the F32-FMA kernels.
//==============================================================================

#include "lowoha_operators/reorder/reorder_data_type/dynamic_quant_impl/dynamic_kernels.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "common/float16.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>

namespace zendnnl {
namespace lowoha {
namespace reorder {

#if defined(__GNUC__) && (__GNUC__ >= 12)

using zendnnl::lowoha::matmul::zendnnl_parallel_for;
using zendnnl::common::finite_mask_ph;

//==============================================================================
// Scale / zero-point computation (F32, identical to F32-FMA path).
//==============================================================================

static inline void compute_symmetric_scale_pg_ph(float absmax, float &scale) {
  if (absmax < 1e-10f) absmax = 1e-10f;
  scale = absmax / 127.0f;
  if (scale < 1e-10f) scale = 1e-10f;
}

static inline void compute_asymmetric_scale_zp_pg_ph(float min_val,
                                                      float max_val,
                                                      float &scale,
                                                      int32_t &zp) {
  // Empty-group / all-non-finite-group reset. Two sentinel pairs reach
  // here: (i) the F32 init sentinels (numeric_limits<float>::max /
  // lowest) used by the F32-FMA backend, and (ii) (+Inf, -Inf) from the
  // FP16-FMA reduce — group_minmax_ph initializes vmin/vmax to FP16
  // +Inf/-Inf because numeric_limits<float>::max() is not representable
  // in FP16. Any time min > max or either bound is non-finite, the row
  // had no finite samples; treat it as an all-zero group, matching the
  // scalar reference.
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
// Per-group statistics reductions (__m512h, 32 lanes per vector).
//==============================================================================

/** Compute per-group absmax in __m512h with 4x unrolling. Widens to F32
 *  for the final horizontal reduce to keep precision parity with the
 *  F32-FMA path. */
__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline float group_absmax_ph(const uint16_t *grp_src,
                                     int64_t group_size) {
  __m512h vam0 = _mm512_setzero_ph();
  __m512h vam1 = _mm512_setzero_ph();
  __m512h vam2 = _mm512_setzero_ph();
  __m512h vam3 = _mm512_setzero_ph();

  int64_t j = 0;
  for (; j + 127 < group_size; j += 128) {
    __m512h v0 = _mm512_loadu_ph(grp_src + j);
    __m512h v1 = _mm512_loadu_ph(grp_src + j + 32);
    __m512h v2 = _mm512_loadu_ph(grp_src + j + 64);
    __m512h v3 = _mm512_loadu_ph(grp_src + j + 96);
    __m512h a0 = _mm512_abs_ph(v0);
    __m512h a1 = _mm512_abs_ph(v1);
    __m512h a2 = _mm512_abs_ph(v2);
    __m512h a3 = _mm512_abs_ph(v3);
    vam0 = _mm512_mask_max_ph(vam0, finite_mask_ph(v0), vam0, a0);
    vam1 = _mm512_mask_max_ph(vam1, finite_mask_ph(v1), vam1, a1);
    vam2 = _mm512_mask_max_ph(vam2, finite_mask_ph(v2), vam2, a2);
    vam3 = _mm512_mask_max_ph(vam3, finite_mask_ph(v3), vam3, a3);
  }
  for (; j + 31 < group_size; j += 32) {
    __m512h v = _mm512_loadu_ph(grp_src + j);
    __m512h a = _mm512_abs_ph(v);
    vam0 = _mm512_mask_max_ph(vam0, finite_mask_ph(v), vam0, a);
  }

  vam0 = _mm512_max_ph(_mm512_max_ph(vam0, vam1),
                        _mm512_max_ph(vam2, vam3));
  return static_cast<float>(_mm512_reduce_max_ph(vam0));
}

/** Compute per-group min/max in __m512h with 4x unrolling. Widens to F32
 *  for the final horizontal reduce. */
__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline void group_minmax_ph(const uint16_t *grp_src, int64_t group_size,
                                    float &row_min, float &row_max) {
  const _Float16 f16_pos_inf =  std::numeric_limits<_Float16>::infinity();
  const _Float16 f16_neg_inf = -std::numeric_limits<_Float16>::infinity();
  __m512h vmin0 = _mm512_set1_ph(f16_pos_inf);
  __m512h vmax0 = _mm512_set1_ph(f16_neg_inf);
  __m512h vmin1 = vmin0, vmax1 = vmax0;
  __m512h vmin2 = vmin0, vmax2 = vmax0;
  __m512h vmin3 = vmin0, vmax3 = vmax0;

  int64_t j = 0;
  for (; j + 127 < group_size; j += 128) {
    __m512h v0 = _mm512_loadu_ph(grp_src + j);
    __m512h v1 = _mm512_loadu_ph(grp_src + j + 32);
    __m512h v2 = _mm512_loadu_ph(grp_src + j + 64);
    __m512h v3 = _mm512_loadu_ph(grp_src + j + 96);
    __mmask32 k0 = finite_mask_ph(v0);
    __mmask32 k1 = finite_mask_ph(v1);
    __mmask32 k2 = finite_mask_ph(v2);
    __mmask32 k3 = finite_mask_ph(v3);
    vmin0 = _mm512_mask_min_ph(vmin0, k0, vmin0, v0);
    vmax0 = _mm512_mask_max_ph(vmax0, k0, vmax0, v0);
    vmin1 = _mm512_mask_min_ph(vmin1, k1, vmin1, v1);
    vmax1 = _mm512_mask_max_ph(vmax1, k1, vmax1, v1);
    vmin2 = _mm512_mask_min_ph(vmin2, k2, vmin2, v2);
    vmax2 = _mm512_mask_max_ph(vmax2, k2, vmax2, v2);
    vmin3 = _mm512_mask_min_ph(vmin3, k3, vmin3, v3);
    vmax3 = _mm512_mask_max_ph(vmax3, k3, vmax3, v3);
  }
  for (; j + 31 < group_size; j += 32) {
    __m512h v = _mm512_loadu_ph(grp_src + j);
    __mmask32 k = finite_mask_ph(v);
    vmin0 = _mm512_mask_min_ph(vmin0, k, vmin0, v);
    vmax0 = _mm512_mask_max_ph(vmax0, k, vmax0, v);
  }

  vmin0 = _mm512_min_ph(_mm512_min_ph(vmin0, vmin1),
                         _mm512_min_ph(vmin2, vmin3));
  vmax0 = _mm512_max_ph(_mm512_max_ph(vmax0, vmax1),
                         _mm512_max_ph(vmax2, vmax3));
  row_min = static_cast<float>(_mm512_reduce_min_ph(vmin0));
  row_max = static_cast<float>(_mm512_reduce_max_ph(vmax0));
}

//==============================================================================
// Pass-2 building block: quantize one __m512h vector to s16 with banker's
// rounding (matches the per-token FP16-FMA path's quantize_to_s16_ph).
//==============================================================================

__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline __m512i quantize_to_s16_ph_pg(__m512h v, __m512h vinv_scale) {
  __m512h q_ph = _mm512_mul_ph(v, vinv_scale);
  return _mm512_cvt_roundph_epi16(q_ph,
      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

//==============================================================================
// KERNEL 1: F16 -> S8 Symmetric Per-Group Quantization (FP16-FMA)
//==============================================================================
//
// Steps (per group, fused):
//   1. Pass 1 - Absmax: scan K/G F16 elements in __m512h (32 lanes/iter)
//      via group_absmax_ph (mask-only VMAXPH using finite_mask_ph). Widen
//      the final reduce to F32; the scalar tail folds remaining < 32 lanes
//      via float16_t::f16_to_f32_val.
//   2. Compute scale: scale = absmax_f32 / 127 (clamped to >= 1e-10) via
//      compute_symmetric_scale_pg_ph, written to scales[m * G + g].
//   3. Pass 2 - Quantize: broadcast 1/scale into __m512h, compute
//      q_ph = val * inv_scale_ph, narrow to s16 with VCVTPH2W (banker's
//      rounding), then narrow s16 -> s8 with VPMOVSWB and aligned/unaligned
//      store.
//
// Register usage (peak, per thread):
//   zmm (PH, 32 lanes):  vam0-vam3 + v0-v3 (~8) in Pass 1; vinv + v + q_ph
//                        (~3) in Pass 2.
//   __mmask32:  finite_mask_ph results in Pass 1.
//   Scalar:    absmax_f32, scale_f32, j, m, g.
//
// Optimizations:
//   1. Fused per-group:  Pass 1 + Pass 2 back-to-back keeps the group's
//      F16 data hot in L1/L2.
//   2. Native __m512h math:  32 lanes/iter at FP16 rate, ~2x throughput
//      vs the F32-FMA fallback in per_group_avx512_f32_fma_kernel.cpp.
//   3. Tiny-scale fallback:  fp16_inv_scale_is_finite guards the 1/scale
//      broadcast; when scale is below the FP16-reciprocal-safe threshold
//      the Pass-2 vector loop is bypassed and the scalar tail handles the
//      group in f32. Keeps the result bit-equivalent to the F32-FMA path.
//   4. Aligned/unaligned store split:  aligned32 flag picks _mm256_store
//      vs _mm256_storeu per group to avoid the unaligned-store penalty
//      where possible.
//   5. zendnnl_parallel_for over M*G total groups for thread-pool reuse
//      across granular tasks.
//==============================================================================

__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
void dynamic_per_group_quant_f16_s8_avx512fp16(const uint16_t *src, int8_t *dst,
                                             float *scales,
                                             int64_t M, int64_t K, int64_t G) {
  if (M <= 0 || K <= 0 || G <= 0 || (K % G) != 0) return;

  const int64_t group_size = K / G;
  const int64_t total_groups = M * G;

  zendnnl_parallel_for(0, total_groups, 1,
      [&](int64_t begin, int64_t end)
          __attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16"))) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t m = task / G;
      const int64_t g = task - m * G;
      const int64_t offset = m * K + g * group_size;
      const uint16_t *grp_src = src + offset;
      int8_t         *grp_dst = dst + offset;

      float absmax = group_absmax_ph(grp_src, group_size);
      int64_t j = group_size & ~int64_t{31};
      for (; j < group_size; ++j) {
        float v = common::float16_t::f16_to_f32_val(grp_src[j]);
        if (std::isfinite(v))
          absmax = std::max(absmax, std::abs(v));
      }

      float scale_f32;
      compute_symmetric_scale_pg_ph(absmax, scale_f32);
      scales[task] = scale_f32;

      const bool aligned32 = (reinterpret_cast<uintptr_t>(grp_dst) & 31) == 0;

      j = 0;
      // Guard: tiny scales make 1/scale overflow FP16; fall back to the
      // scalar tail in f32 (matches the F32-FMA / scalar reference).
      if (common::fp16_inv_scale_is_finite(scale_f32)) {
        const _Float16 inv_scale_f16 = static_cast<_Float16>(1.0f / scale_f32);
        const __m512h vinv = _mm512_set1_ph(inv_scale_f16);
        for (; j + 31 < group_size; j += 32) {
          __m512h v = _mm512_loadu_ph(grp_src + j);
          __m512i s16 = quantize_to_s16_ph_pg(v, vinv);
          __m256i s8  = _mm512_cvtsepi16_epi8(s16);
          if (aligned32)
            _mm256_store_si256(reinterpret_cast<__m256i *>(grp_dst + j), s8);
          else
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(grp_dst + j), s8);
        }
      }
      for (; j < group_size; ++j) {
        float v = common::float16_t::f16_to_f32_val(grp_src[j]);
        int32_t q = static_cast<int32_t>(std::nearbyint(v / scale_f32));
        q = std::max(-128, std::min(127, q));
        grp_dst[j] = static_cast<int8_t>(q);
      }
    }
  });
}

//==============================================================================
// KERNEL 2: F16 -> U8 Asymmetric Per-Group Quantization (FP16-FMA)
//==============================================================================
//
// Steps (per group, fused):
//   1. Pass 1 - Min/Max: scan K/G F16 elements in __m512h via group_minmax_ph
//      (mask-only VMINPH/VMAXPH using finite_mask_ph). Widen the final reduce
//      to F32; scalar tail folds remaining < 32 lanes via
//      float16_t::f16_to_f32_val.
//   2. Compute scale + zp: scale = (max - min) / 255 (clamped to >= 1e-10)
//      and zp = round(-min / scale), clamped to int32 via
//      compute_asymmetric_scale_zp_pg_ph. Written to scales[task] and
//      zps[task].
//   3. Pass 2 - Quantize: broadcast 1/scale into __m512h, compute
//      q_ph = val * inv_scale_ph, narrow to s16 with VCVTPH2W, widen to
//      s32 (lo/hi halves via VPMOVSXWD), add vzp32 in int32, clamp to
//      [0, 255] in s32 (VPMAXSD/VPMINSD), then narrow s32 -> u8 with
//      VPMOVUSDB and store.
//
// Register usage (peak, per thread):
//   zmm (PH, 32 lanes):  vmin/vmax 0-3 + v0-v3 (~12) in Pass 1; vinv + v +
//                        q_ph (~3) in Pass 2.
//   zmm (s32, 16 lanes): vzp32 + vlo32 + vhi32 + s32_lo + s32_hi (~5) in
//                        Pass 2.
//   __mmask32:  finite_mask_ph results in Pass 1.
//   Scalar:    row_min_f32, row_max_f32, scale_f32, zp, j, m, g.
//
// Optimizations:
//   1. Fused per-group:  same L1/L2 cache reuse as KERNEL 1.
//   2. Native __m512h math:  32 lanes/iter, ~2x throughput vs F32-FMA.
//   3. Twin guards on the Pass-2 vector loop (same shape as the per-token
//      asymmetric kernel):
//        - fp16_inv_scale_is_finite(scale)  rules out 1/scale FP16 overflow.
//        - fp16_zp_safe_for_s16_narrow(zp)  rules out the s16 narrow
//          pre-saturating an out-of-int16 quotient that a large zp would
//          otherwise rescue.
//      Either guard failure falls back to the scalar tail (f32 quotient
//      + int32 zp), bit-equivalent to the F32-FMA / scalar reference.
//   4. Explicit [0, 255] clamp before VPMOVUSDB (which treats negative
//      s32 as large unsigned and would saturate to 255 instead of 0).
//   5. zendnnl_parallel_for over M*G total groups for thread-pool reuse.
//==============================================================================

__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
void dynamic_per_group_quant_f16_u8_avx512fp16(const uint16_t *src, uint8_t *dst,
                                             float *scales, int32_t *zps,
                                             int64_t M, int64_t K, int64_t G) {
  if (M <= 0 || K <= 0 || G <= 0 || (K % G) != 0) return;

  const int64_t group_size = K / G;
  const int64_t total_groups = M * G;

  zendnnl_parallel_for(0, total_groups, 1,
      [&](int64_t begin, int64_t end)
          __attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16"))) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t m = task / G;
      const int64_t g = task - m * G;
      const int64_t offset = m * K + g * group_size;
      const uint16_t *grp_src = src + offset;
      uint8_t        *grp_dst = dst + offset;

      float row_min, row_max;
      group_minmax_ph(grp_src, group_size, row_min, row_max);
      int64_t j = group_size & ~int64_t{31};
      for (; j < group_size; ++j) {
        float v = common::float16_t::f16_to_f32_val(grp_src[j]);
        if (std::isfinite(v)) {
          row_min = std::min(row_min, v);
          row_max = std::max(row_max, v);
        }
      }

      float scale_f32;
      int32_t zp;
      compute_asymmetric_scale_zp_pg_ph(row_min, row_max, scale_f32, zp);
      scales[task] = scale_f32;
      zps[task]    = zp;

      j = 0;
      // Twin guards: (i) tiny scales make 1/scale overflow FP16, and
      // (ii) large |zp| make VCVTPH2W pre-saturate out-of-int16
      // quotients before the int32 zp add can bring them back into
      // range. Either failure forces the scalar tail in f32 with int32
      // zp, which matches the F32-FMA / scalar reference. See the
      // per-token fused-asym kernel for the full divergence rationale.
      if (common::fp16_inv_scale_is_finite(scale_f32) &&
          common::fp16_zp_safe_for_s16_narrow(zp)) {
        const _Float16 inv_scale_f16 = static_cast<_Float16>(1.0f / scale_f32);
        const __m512h vinv  = _mm512_set1_ph(inv_scale_f16);
        const __m512i vzp32 = _mm512_set1_epi32(zp);
        const __m512i vlo32 = _mm512_setzero_si512();
        const __m512i vhi32 = _mm512_set1_epi32(255);

        for (; j + 31 < group_size; j += 32) {
          __m512h v   = _mm512_loadu_ph(grp_src + j);
          __m512i s16 = quantize_to_s16_ph_pg(v, vinv);
          __m512i s32_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(s16));
          __m512i s32_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(s16, 1));
          s32_lo = _mm512_add_epi32(s32_lo, vzp32);
          s32_hi = _mm512_add_epi32(s32_hi, vzp32);
          // Signed [0, 255] clamp BEFORE VPMOVUSDB (which treats its
          // input as unsigned and would otherwise wrap negative s32 to
          // 255 instead of clamping to 0).
          s32_lo = _mm512_max_epi32(vlo32, _mm512_min_epi32(vhi32, s32_lo));
          s32_hi = _mm512_max_epi32(vlo32, _mm512_min_epi32(vhi32, s32_hi));
          __m128i u8_lo = _mm512_cvtusepi32_epi8(s32_lo);
          __m128i u8_hi = _mm512_cvtusepi32_epi8(s32_hi);
          _mm_storeu_si128(reinterpret_cast<__m128i *>(grp_dst + j),      u8_lo);
          _mm_storeu_si128(reinterpret_cast<__m128i *>(grp_dst + j + 16), u8_hi);
        }
      }
      for (; j < group_size; ++j) {
        float v = common::float16_t::f16_to_f32_val(grp_src[j]);
        int32_t q = static_cast<int32_t>(std::nearbyint(v / scale_f32)) + zp;
        q = std::max(0, std::min(255, q));
        grp_dst[j] = static_cast<uint8_t>(q);
      }
    }
  });
}

#else  // !(GCC >= 12) — Strategy A not buildable on this toolchain
       //
       // The FP16-FMA kernels cannot be compiled (no __m512h intrinsics on
       // toolchains older than GCC 12), but we still emit symbols for the
       // declared functions so the link succeeds. Instead of leaving them as
       // no-op stubs (which would silently corrupt output if the dispatcher
       // ever mis-routed to them), delegate to the always-available F32-FMA
       // siblings in per_group_avx512_f32_fma_kernel.cpp. The can_use_f16_fma_kernel()
       // helper returns false on this toolchain (gated on __GNUC__ >= 12 in
       // lowoha_reorder_common.hpp), so the dispatcher should never select
       // these in practice -- the delegation is defense in depth.

void dynamic_per_group_quant_f16_s8_avx512fp16(const uint16_t *src,
                                             int8_t *dst,
                                             float *scales,
                                             int64_t M, int64_t K,
                                             int64_t G) {
  dynamic_per_group_quant_f16_s8_native(src, dst, scales, M, K, G);
}

void dynamic_per_group_quant_f16_u8_avx512fp16(const uint16_t *src,
                                             uint8_t *dst,
                                             float *scales,
                                             int32_t *zps,
                                             int64_t M, int64_t K,
                                             int64_t G) {
  dynamic_per_group_quant_f16_u8_native(src, dst, scales, zps, M, K, G);
}

#endif  // __GNUC__ >= 12

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
