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
// FP16 PER-TOKEN DYNAMIC QUANTIZATION — FP16-FMA backend (Strategy A)
//
// Native __m512h kernels. The full statistics + quantize chain stays in
// FP16 (32 lanes per vector register) using the AVX-512-FP16 ISA. The host
// CPU MUST support `avx512fp16` (verified upstream by the dispatcher via
// zendnnl_platform_info().get_avx512_f16_status()) or these kernels will
// SIGILL. On toolchains older than GCC 12, this TU compiles to empty.
//
// Companion F32-FMA kernels (per_token_avx512_f32_fma_kernel.cpp) implement the
// same dtype/granularity combinations using F16C convert + __m512 math;
// they run on every AVX-512F + F16C host (every shipping AVX-512F CPU
// also has F16C in practice) and are the default backend.
//
// Precision contract:
//   - Per-row absmax / min / max reductions happen lane-wise in __m512h,
//     then widen to F32 for the final horizontal reduce (matches the
//     F32-FMA path's bit-pattern for stats).
//   - scale / zp are computed in F32 (identical helper as F32-FMA path).
//   - Pass 2 multiplies by (1/scale_f16) in __m512h (FMA-class op), then
//     converts to int16 with banker's rounding. The S8 path narrows int16 ->
//     int8 with the truncating VPMOVWB (non-finite -> 0, matching vLLM); the
//     U8 path widens to int32, clamps to [0,255], and narrows with VPMOVUSDB.
//     The FP16-FMA path is intentionally NOT bit-exact with the scalar
//     reference — tolerate |delta| <= 1.
//==============================================================================

#include "lowoha_operators/reorder/reorder_data_type/dynamic_quant_impl/dynamic_kernels.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "common/float16.hpp"

#include <immintrin.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>
#include <omp.h>

namespace zendnnl {
namespace lowoha {
namespace reorder {

#if defined(__GNUC__) && (__GNUC__ >= 12)

using zendnnl::lowoha::matmul::zendnnl_parallel_for;
using zendnnl::common::finite_mask_ph;

//==============================================================================
// Scale / zero-point computation from row statistics (F32, identical to
// the F32-FMA path so scales/zps are bit-equal between backends).
//==============================================================================

static inline void compute_symmetric_scale_from_absmax_ph(float absmax,
                                                           float &scale) {
  if (absmax < 1e-10f) absmax = 1e-10f;
  scale = absmax / 127.0f;
  if (scale < 1e-10f) scale = 1e-10f;
}

static inline void compute_asymmetric_scale_zp_ph(float min_val, float max_val,
                                                   float &scale, int32_t &zp) {
  // Empty-row / all-non-finite-row reset. Two sentinel pairs reach here:
  // (i) the F32 init sentinels (numeric_limits<float>::max / lowest)
  // used by the F32-FMA backend, and (ii) (+Inf, -Inf) from the
  // FP16-FMA reduce — the row-stat collector initializes vmin/vmax to
  // FP16 +Inf/-Inf because numeric_limits<float>::max() is not
  // representable in FP16. Any time min > max or either bound is
  // non-finite, the row had no finite samples; treat it as an all-zero
  // row, matching the scalar reference.
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
// 32-lane store helpers (mirror the F32-FMA path's cache-line stores).
//==============================================================================

/** Store 32 int8 lanes from a __m256i with optional 32B aligned
 *  store. The destination is uint16_t-spaced (1 byte per lane). */
__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline void store_32x_s8_ph(int8_t *dst, __m256i v, bool aligned32) {
  if (aligned32) {
    _mm256_store_si256(reinterpret_cast<__m256i *>(dst), v);
  } else {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), v);
  }
}

__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline void store_32x_u8_ph(uint8_t *dst, __m256i v, bool aligned32) {
  store_32x_s8_ph(reinterpret_cast<int8_t *>(dst), v, aligned32);
}

//==============================================================================
// FP16-FMA quantize helpers (Pass 2 building blocks)
//
// quantize_to_s16_ph: q_i16 = round(v * vinv_scale), s16 saturated.
// quantize_to_s16_ph_with_zp: q_i16 = clamp(round(v * vinv_scale) + zp,
//                                            0, 255), s16 lanes.
//
// Inputs are __m512h source data; vinv_scale is broadcast 1/scale in FP16.
// Output is __m512i with each lane holding a signed int16 in the low 16
// bits of each 16-bit slot. The S8 path narrows with the truncating VPMOVWB
// (_mm512_cvtepi16_epi8) so non-finite lanes (int16 0x8000) -> low byte 0x00
// -> 0, matching vLLM; the U8 path widens to s32 and uses VPMOVUSDB.
//==============================================================================

__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline __m512i quantize_to_s16_ph(__m512h v, __m512h vinv_scale) {
  // q = round_to_nearest_even(v * vinv_scale), result in __m512h.
  __m512h q_ph = _mm512_mul_ph(v, vinv_scale);
  // Convert PH -> 32 signed int16 lanes with banker's rounding, no exc.
  return _mm512_cvt_roundph_epi16(q_ph,
      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

//==============================================================================
// KERNEL 1: F16 -> S8 Symmetric Per-Token Quantization (FP16-FMA)
//==============================================================================
//
// Steps (per row, fused):
//   1. Pass 1 — Absmax reduction in __m512h, 32 lanes/iter with 4x unroll
//      (128 elements/iter). Non-finite lanes masked via finite_mask_ph().
//      Widen to F32 for horizontal reduce (precision).
//   2. Compute scale = absmax / 127 in F32. Write scales[m].
//   3. Pass 2 — Multiply by (1/scale_f16) in __m512h, convert PH->S16 with
//      banker's rounding, narrow S16->S8 with the truncating VPMOVWB. Finite
//      values are in [-127,127] by construction; non-finite int16 lanes
//      (0x8000) narrow to low byte 0x00 -> 0, matching vLLM.
//
// FMA cadence: VMULPH on PH operands is the FP16-FMA building block; it
// pairs with the round-narrow chain for ~2x throughput over the F32-FMA
// path on AVX512-FP16 capable CPUs (Granite Rapids, Turin).
//==============================================================================

__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
void dynamic_per_token_quant_f16_s8_avx512fp16(const uint16_t *src, int8_t *dst,
                                             float *scales,
                                             int64_t M, int64_t N) {
  auto row_loop = [&](int64_t m)
      __attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16"))) {
    const uint16_t *row_src = src + m * N;
    int8_t         *row_dst = dst + m * N;

    // -- Pass 1: absmax reduction in __m512h, 128 lanes / iter ----------
    __m512h vam0 = _mm512_setzero_ph();
    __m512h vam1 = _mm512_setzero_ph();
    __m512h vam2 = _mm512_setzero_ph();
    __m512h vam3 = _mm512_setzero_ph();

    int64_t j = 0;
    for (; j + 127 < N; j += 128) {
      __m512h v0 = _mm512_loadu_ph(row_src + j);
      __m512h v1 = _mm512_loadu_ph(row_src + j + 32);
      __m512h v2 = _mm512_loadu_ph(row_src + j + 64);
      __m512h v3 = _mm512_loadu_ph(row_src + j + 96);
      __m512h a0 = _mm512_abs_ph(v0);
      __m512h a1 = _mm512_abs_ph(v1);
      __m512h a2 = _mm512_abs_ph(v2);
      __m512h a3 = _mm512_abs_ph(v3);
      vam0 = _mm512_mask_max_ph(vam0, finite_mask_ph(v0), vam0, a0);
      vam1 = _mm512_mask_max_ph(vam1, finite_mask_ph(v1), vam1, a1);
      vam2 = _mm512_mask_max_ph(vam2, finite_mask_ph(v2), vam2, a2);
      vam3 = _mm512_mask_max_ph(vam3, finite_mask_ph(v3), vam3, a3);
    }

    for (; j + 31 < N; j += 32) {
      __m512h v = _mm512_loadu_ph(row_src + j);
      __m512h a = _mm512_abs_ph(v);
      vam0 = _mm512_mask_max_ph(vam0, finite_mask_ph(v), vam0, a);
    }

    vam0 = _mm512_max_ph(_mm512_max_ph(vam0, vam1),
                          _mm512_max_ph(vam2, vam3));
    float absmax = static_cast<float>(_mm512_reduce_max_ph(vam0));

    for (; j < N; ++j) {
      float v = common::float16_t::f16_to_f32_val(row_src[j]);
      if (std::isfinite(v))
        absmax = std::max(absmax, std::abs(v));
    }

    // -- Compute per-row scale in F32 (identical to F32-FMA path) ---------
    float scale_f32;
    compute_symmetric_scale_from_absmax_ph(absmax, scale_f32);
    scales[m] = scale_f32;

    // -- Pass 2: quantize in __m512h --------------------------------------
    // Use 1/scale broadcast in FP16 so the inner loop is one VMULPH.
    //
    // Guard: when scale_f32 is so small that 1/scale_f32 overflows the
    // FP16 normal range, _Float16(1/scale_f32) becomes +Inf and every
    // nonzero lane of VMULPH saturates. In that case we skip the vector
    // loop and let the scalar tail below process the whole row in f32,
    // matching the F32-FMA / scalar-reference output bit-for-bit.
    const bool aligned32 = (reinterpret_cast<uintptr_t>(row_dst) & 31) == 0;

    j = 0;
    if (common::fp16_inv_scale_is_finite(scale_f32)) {
      const _Float16 inv_scale_f16 = static_cast<_Float16>(1.0f / scale_f32);
      const __m512h vinv = _mm512_set1_ph(inv_scale_f16);

      for (; j + 31 < N; j += 32) {
        __m512h v = _mm512_loadu_ph(row_src + j);
        __m512i s16 = quantize_to_s16_ph(v, vinv);     // 32 x int16 lanes
        __m256i s8  = _mm512_cvtepi16_epi8(s16);      // truncating narrow -> int8 (non-finite -> 0)
        store_32x_s8_ph(row_dst + j, s8, aligned32);
      }
    }

    // Tail: scalar fallback matches the scalar reference behavior. When
    // the guard above tripped, this loop processes the entire row.
    for (; j < N; ++j) {
      float v = common::float16_t::f16_to_f32_val(row_src[j]);
      if (!std::isfinite(v)) { row_dst[j] = 0; continue; }
      int32_t q = static_cast<int32_t>(std::nearbyint(v / scale_f32));
      row_dst[j] = static_cast<int8_t>(q);
    }
  };

  if (M == 1) {
    row_loop(0);
    return;
  }

  // Same CCX-aware threading topology as the BF16/F32 symmetric s8 kernel.
  const int nthreads = omp_get_max_threads();
  const int cores_per_ccx = 8;

  if (nthreads >= cores_per_ccx && M < nthreads) {
    const int num_ccxs = std::max(1, nthreads / cores_per_ccx);

    #pragma omp parallel
    {
      const int tid      = omp_get_thread_num();
      const int ccx_id   = tid / cores_per_ccx;
      const int local_id = tid % cores_per_ccx;

      const int64_t rows_per_ccx = (M + num_ccxs - 1) / num_ccxs;
      const int64_t ccx_start    = ccx_id * rows_per_ccx;
      const int64_t ccx_end      = std::min(ccx_start + rows_per_ccx, M);

      for (int64_t m = ccx_start + local_id; m < ccx_end; m += cores_per_ccx)
        row_loop(m);
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m)
      row_loop(m);
  }
}

//==============================================================================
// KERNEL 2: F16 -> U8 Asymmetric Per-Token Quantization (FP16-FMA)
//==============================================================================
//
// Steps (per row, fused):
//   1. Pass 1 — Min/Max reduction in __m512h, 128 lanes/iter with 4x
//      independent vmin/vmax accumulators. Non-finite lanes masked.
//   2. Compute scale = (max - min) / 255 and zp = round(-min/scale) in F32.
//      Write scales[m], zps[m].
//   3. Pass 2 — Multiply v * (1/scale_f16) in __m512h, round to s16, add
//      zp in s16 lanes, clamp to [0,255] s16, narrow to u8 with VPMOVUSWB.
//      Note: explicit clamp before VPMOVUSWB is needed because PMOVUSWB
//      treats negative int16 as huge unsigned and saturates to 255.
//==============================================================================

__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
void dynamic_per_token_quant_f16_u8_avx512fp16(const uint16_t *src, uint8_t *dst,
                                             float *scales, int32_t *zps,
                                             int64_t M, int64_t N) {
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const uint16_t *row_src = src + m * N;
    uint8_t        *row_dst = dst + m * N;

    // -- Pass 1: min/max reduction in __m512h -----------------------------
    // Initialize to FP16 +Inf / -Inf so non-finite lanes (masked) never
    // disturb the accumulator.
    const _Float16 f16_pos_inf =  std::numeric_limits<_Float16>::infinity();
    const _Float16 f16_neg_inf = -std::numeric_limits<_Float16>::infinity();
    __m512h vmin0 = _mm512_set1_ph(f16_pos_inf);
    __m512h vmax0 = _mm512_set1_ph(f16_neg_inf);
    __m512h vmin1 = vmin0, vmax1 = vmax0;
    __m512h vmin2 = vmin0, vmax2 = vmax0;
    __m512h vmin3 = vmin0, vmax3 = vmax0;

    int64_t j = 0;
    for (; j + 127 < N; j += 128) {
      __m512h v0 = _mm512_loadu_ph(row_src + j);
      __m512h v1 = _mm512_loadu_ph(row_src + j + 32);
      __m512h v2 = _mm512_loadu_ph(row_src + j + 64);
      __m512h v3 = _mm512_loadu_ph(row_src + j + 96);
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

    for (; j + 31 < N; j += 32) {
      __m512h v = _mm512_loadu_ph(row_src + j);
      __mmask32 k = finite_mask_ph(v);
      vmin0 = _mm512_mask_min_ph(vmin0, k, vmin0, v);
      vmax0 = _mm512_mask_max_ph(vmax0, k, vmax0, v);
    }

    vmin0 = _mm512_min_ph(_mm512_min_ph(vmin0, vmin1),
                           _mm512_min_ph(vmin2, vmin3));
    vmax0 = _mm512_max_ph(_mm512_max_ph(vmax0, vmax1),
                           _mm512_max_ph(vmax2, vmax3));
    float row_min = static_cast<float>(_mm512_reduce_min_ph(vmin0));
    float row_max = static_cast<float>(_mm512_reduce_max_ph(vmax0));

    for (; j < N; ++j) {
      float v = common::float16_t::f16_to_f32_val(row_src[j]);
      if (std::isfinite(v)) {
        row_min = std::min(row_min, v);
        row_max = std::max(row_max, v);
      }
    }

    // -- Compute scale + zp in F32 ----------------------------------------
    float scale_f32;
    int32_t zp;
    compute_asymmetric_scale_zp_ph(row_min, row_max, scale_f32, zp);
    scales[m] = scale_f32;
    zps[m]    = zp;

    // -- Pass 2: quantize with zp add in s32 lanes ------------------------
    // Guard: skip the vector loop when 1/scale_f32 would overflow FP16.
    // The scalar tail below stays in f32 and matches the F32-FMA path
    // bit-for-bit when that happens.

    j = 0;
    // Two guards must both hold for the vector path: (i) 1/scale must
    // fit in FP16 so the VMULPH doesn't overflow lanes, and (ii) |zp|
    // must be small enough that the int16-saturating VCVTPH2W narrow
    // does NOT pre-saturate values that the subsequent int32 zp add
    // would have brought back into the destination range. Without (ii),
    // an out-of-int16 quotient gets clamped to ±32767 BEFORE the zp
    // add, producing an arbitrarily wrong final u8 — diverges from
    // scalar / F32-FMA by far more than the documented ±1 LSB
    // tolerance. The fp16_zp_safe_for_s16_narrow check rules out this
    // "narrow range far from zero" regime (e.g. f16 row min=-65000,
    // max=-64500 -> zp ≈ 33150 outside int16); when it fails, the
    // scalar tail below handles the whole row in f32 with int32 zp.
    if (common::fp16_inv_scale_is_finite(scale_f32) &&
        common::fp16_zp_safe_for_s16_narrow(zp)) {
      const _Float16 inv_scale_f16 = static_cast<_Float16>(1.0f / scale_f32);
      const __m512h  vinv  = _mm512_set1_ph(inv_scale_f16);
      const __m512i  vzp32 = _mm512_set1_epi32(zp);
      const __m512i  vlo32 = _mm512_setzero_si512();
      const __m512i  vhi32 = _mm512_set1_epi32(255);

      for (; j + 31 < N; j += 32) {
        __m512h v = _mm512_loadu_ph(row_src + j);
        __m512i s16 = quantize_to_s16_ph(v, vinv);
        // Widen s16 -> s32 (lo/hi 256-bit halves), add zp in int32 (no
        // saturation), clamp to [0,255] in s32, then narrow s32 -> u8.
        // Explicit signed clamp is required because VPMOVUSDB treats its
        // input as unsigned: a negative s32 (e.g. -1) would wrap to a
        // huge positive value and saturate to 255 instead of clamping to
        // 0. After the clamp every lane is in [0,255] so the narrow is
        // unambiguous.
        __m512i s32_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(s16));
        __m512i s32_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(s16, 1));
        s32_lo = _mm512_add_epi32(s32_lo, vzp32);
        s32_hi = _mm512_add_epi32(s32_hi, vzp32);
        s32_lo = _mm512_max_epi32(vlo32, _mm512_min_epi32(vhi32, s32_lo));
        s32_hi = _mm512_max_epi32(vlo32, _mm512_min_epi32(vhi32, s32_hi));
        __m128i u8_lo = _mm512_cvtusepi32_epi8(s32_lo);
        __m128i u8_hi = _mm512_cvtusepi32_epi8(s32_hi);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(row_dst + j),      u8_lo);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(row_dst + j + 16), u8_hi);
      }
    }

    for (; j < N; ++j) {
      float v = common::float16_t::f16_to_f32_val(row_src[j]);
      if (!std::isfinite(v)) { row_dst[j] = 0; continue; }
      int32_t q = static_cast<int32_t>(std::nearbyint(v / scale_f32)) + zp;
      q = std::max(0, std::min(255, q));
      row_dst[j] = static_cast<uint8_t>(q);
    }
  }
}

//==============================================================================
// UNFUSED 2-PASS Per-Token Dynamic Quantization (FP16-FMA)
//
// Two passes with different parallelization to extract more parallelism
// when M < num_threads (typical for batch=1 LLM inference):
//   Pass 1 — per-row absmax / min-max + scale (+ zp). Parallel over M rows.
//            Uses the same FP16-domain reductions as the fused kernel but
//            does NOT quantize. Writes scales[m] (and zps[m] for u8 asym).
//   Pass 2 — quantize. Parallel over M*N contiguous elements via
//            zendnnl_parallel_for, so all threads can participate even
//            when M is tiny. Each thread iterates row-by-row inside its
//            chunk and uses the row's pre-computed scale_f32 to broadcast
//            an FP16 1/scale per row.
//
// Precision contract: same as fused FP16-FMA (PH multiply rounds in FP16,
// PH->s16 conversion rounds again — tolerate |delta| <= 1 LSB vs scalar
// reference). The test helper `compare_lowoha_quant_output` already
// widens its tolerance for f16 source to absorb this.
//==============================================================================

//==============================================================================
// KERNEL 3: F16 -> S8 Symmetric Per-Token Quantization (unfused 2-pass,
//           FP16-FMA)
//==============================================================================
//
// Steps (split-pass, not fused):
//   1. Pass 1 (parallel over M) - Absmax: for each row, scan all N F16
//      elements in __m512h (32 lanes/iter) with finite_mask_ph, compute
//      absmax_ph = max(|val|) via VABSPH + VMAXPH{k}. Widen the final
//      reduce to F32 to compute scale = absmax_f32 / 127 (clamped to
//      >= 1e-10) and write to scales[m].
//   2. Pass 2 (parallel over M*N via zendnnl_parallel_for) - Quantize:
//      every chunk picks up its rows' scales from scales[], broadcasts
//      1/scale into __m512h, computes q_ph = val * inv_scale_ph,
//      narrows to s16 with VCVTPH2W (banker's rounding), then narrows
//      s16 -> s8 with the truncating VPMOVWB (non-finite -> 0) and stores.
//
// Why split: when M is small (e.g. M=1) but N is large, the fused per-row
// kernel can only parallelize across M threads while Pass 2 here distributes
// the M*N quantize work across the whole thread pool. Pays one extra L2/L3
// re-read of the source vs the fused path.
//
// Register usage (peak, per thread):
//   zmm (PH, 32 lanes): ~10 across Pass 1 (vam0-vam3 + v0-v3 + a0-a3
//                       overlapped) and ~6 across Pass 2 (vinv + v + q_ph
//                       per unroll).
//   __mmask32:  finite_mask_ph results in Pass 1.
//   Scalar:    absmax_f32, scale_f32, j (per-row in Pass 1; per-chunk in
//              Pass 2).
//
// Optimizations:
//   1. Pass-1 OMP parallel-for over M:  scales when M is large.
//   2. Pass-2 zendnnl_parallel_for over M*N:  decouples Pass-2 parallelism
//      from M for the M-small / N-large regime.
//   3. Native __m512h math:  32 lanes/iter at FP16 rate -- ~2x throughput
//      vs the F32-FMA fallback in per_token_avx512_f32_fma_kernel.cpp.
//   4. Tiny-scale fallback:  fp16_inv_scale_is_finite guards the 1/scale
//      broadcast; when scale is below the FP16-reciprocal-safe threshold
//      the Pass-2 vector loop is bypassed and the scalar tail (f32) handles
//      the row. Keeps the result bit-equivalent to the F32-FMA path in the
//      tiny-scale regime.
//   5. Banker's rounding via _MM_FROUND_TO_NEAREST_INT in VCVTPH2W matches
//      the scalar reference's std::nearbyint().
//   6. No scratch buffer: scales[] is the only state between passes.
//==============================================================================

__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
void dynamic_per_token_quant_f16_s8_unfused_avx512fp16(const uint16_t *src,
                                                     int8_t *dst,
                                                     float *scales,
                                                     int64_t M, int64_t N) {
  // -- Pass 1: per-row absmax + scale (parallel over M) -------------------
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const uint16_t *row_src = src + m * N;

    __m512h vam0 = _mm512_setzero_ph();
    __m512h vam1 = _mm512_setzero_ph();
    __m512h vam2 = _mm512_setzero_ph();
    __m512h vam3 = _mm512_setzero_ph();

    int64_t j = 0;
    for (; j + 127 < N; j += 128) {
      __m512h v0 = _mm512_loadu_ph(row_src + j);
      __m512h v1 = _mm512_loadu_ph(row_src + j + 32);
      __m512h v2 = _mm512_loadu_ph(row_src + j + 64);
      __m512h v3 = _mm512_loadu_ph(row_src + j + 96);
      __m512h a0 = _mm512_abs_ph(v0);
      __m512h a1 = _mm512_abs_ph(v1);
      __m512h a2 = _mm512_abs_ph(v2);
      __m512h a3 = _mm512_abs_ph(v3);
      vam0 = _mm512_mask_max_ph(vam0, finite_mask_ph(v0), vam0, a0);
      vam1 = _mm512_mask_max_ph(vam1, finite_mask_ph(v1), vam1, a1);
      vam2 = _mm512_mask_max_ph(vam2, finite_mask_ph(v2), vam2, a2);
      vam3 = _mm512_mask_max_ph(vam3, finite_mask_ph(v3), vam3, a3);
    }
    for (; j + 31 < N; j += 32) {
      __m512h v = _mm512_loadu_ph(row_src + j);
      __m512h a = _mm512_abs_ph(v);
      vam0 = _mm512_mask_max_ph(vam0, finite_mask_ph(v), vam0, a);
    }

    vam0 = _mm512_max_ph(_mm512_max_ph(vam0, vam1),
                          _mm512_max_ph(vam2, vam3));
    float absmax = static_cast<float>(_mm512_reduce_max_ph(vam0));

    for (; j < N; ++j) {
      float v = common::float16_t::f16_to_f32_val(row_src[j]);
      if (std::isfinite(v))
        absmax = std::max(absmax, std::abs(v));
    }

    float scale_f32;
    compute_symmetric_scale_from_absmax_ph(absmax, scale_f32);
    scales[m] = scale_f32;
  }

  // -- Pass 2: quantize (parallel over M*N contiguous elements) -----------
  const int64_t total = M * N;
  constexpr int64_t grain_size = LOWOHA_REORDER_GRAIN_SIZE;
  zendnnl_parallel_for(0, total, grain_size,
      [&](int64_t begin, int64_t end)
          __attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16"))) {
    while (begin < end) {
      const int64_t m = begin / N;
      const int64_t row_end = std::min((m + 1) * N, end);
      const int64_t count = row_end - begin;
      const uint16_t *csrc = src + begin;
      int8_t *cdst = dst + begin;
      const bool aligned32 = (reinterpret_cast<uintptr_t>(cdst) & 31) == 0;

      int64_t k = 0;
      // Guard: 1/scale broadcast in FP16 -> +Inf when scale is tiny, so
      // skip the vector loop and let the scalar tail (in f32) handle the
      // whole row. Matches the F32-FMA / scalar reference output.
      if (common::fp16_inv_scale_is_finite(scales[m])) {
        const _Float16 inv_scale_f16 = static_cast<_Float16>(1.0f / scales[m]);
        const __m512h vinv = _mm512_set1_ph(inv_scale_f16);
        for (; k + 31 < count; k += 32) {
          __m512h v   = _mm512_loadu_ph(csrc + k);
          __m512i s16 = quantize_to_s16_ph(v, vinv);
          __m256i s8  = _mm512_cvtepi16_epi8(s16);     // truncating narrow -> s8 (non-finite -> 0)
          store_32x_s8_ph(cdst + k, s8, aligned32);
        }
      }
      for (; k < count; ++k) {
        float v = common::float16_t::f16_to_f32_val(csrc[k]);
        if (!std::isfinite(v)) { cdst[k] = 0; continue; }
        int32_t q = static_cast<int32_t>(std::nearbyint(v / scales[m]));
        cdst[k] = static_cast<int8_t>(q);
      }
      begin = row_end;
    }
  });
}

//==============================================================================
// KERNEL 4: F16 -> U8 Asymmetric Per-Token Quantization (unfused 2-pass,
//           FP16-FMA)
//==============================================================================
//
// Steps (split-pass, not fused):
//   1. Pass 1 (parallel over M) - Min/Max: for each row, scan all N F16
//      elements in __m512h with finite_mask_ph, compute row_min/row_max
//      via VMINPH{k}/VMAXPH{k}. Widen the final reduce to F32 to compute
//      scale = (max - min) / 255 (clamped to >= 1e-10) and
//      zp = round(-min / scale) (clamped to int32); write to scales[m]
//      and zps[m].
//   2. Pass 2 (parallel over M*N via zendnnl_parallel_for) - Quantize:
//      every chunk picks up its rows' (scale, zp), broadcasts 1/scale into
//      __m512h, computes q_ph = val * inv_scale_ph, narrows to s16 via
//      VCVTPH2W, widens to s32, adds vzp32 in int32, clamps to [0, 255]
//      in s32 (VPMAXSD/VPMINSD), then narrows s32 -> u8 with VPMOVUSDB
//      and cache-line stores.
//
// Why split: same rationale as KERNEL 3 -- decouples Pass-2 parallelism
// from M so small-M / large-N workloads (typical single-token decode)
// scale across the full thread pool.
//
// Register usage (peak, per thread):
//   zmm (PH, 32 lanes): ~14 across Pass 1 (vmin/vmax 0-3 + v0-v3).
//   zmm (s32, 16 lanes): ~12 across Pass 2 (vzp32 + vlo32 + vhi32 +
//                        s32_lo/s32_hi + intermediates).
//   __mmask32:  finite_mask_ph results in Pass 1.
//   Scalar:    row_min_f32, row_max_f32, scale_f32, zp, j.
//
// Optimizations:
//   1. Pass-1 OMP parallel-for over M:  scales when M is large.
//   2. Pass-2 zendnnl_parallel_for over M*N:  decouples from M for the
//      M-small / N-large regime.
//   3. Native __m512h math in both passes, ~2x throughput vs F32-FMA.
//   4. Twin guards on the Pass-2 vector loop:
//        - fp16_inv_scale_is_finite(scale)  rules out 1/scale FP16 overflow.
//        - fp16_zp_safe_for_s16_narrow(zp)  rules out the s16 narrow
//          pre-saturating an out-of-int16 quotient that a large user zp
//          would otherwise rescue.
//      Either guard failure falls back to the scalar tail (f32 quotient
//      + int32 zp), bit-equivalent to the F32-FMA / scalar reference.
//   5. Explicit [0, 255] clamp before VPMOVUSDB (which treats negative
//      s32 as large unsigned and would saturate to 255 instead of 0).
//   6. Cache-line stores and 4x unroll (where applicable) for ILP.
//   7. No scratch buffer: scales[] + zps[] are the only state between passes.
//==============================================================================

__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
void dynamic_per_token_quant_f16_u8_unfused_avx512fp16(const uint16_t *src,
                                                     uint8_t *dst,
                                                     float *scales,
                                                     int32_t *zps,
                                                     int64_t M, int64_t N) {
  // -- Pass 1: per-row min/max + scale/zp (parallel over M) ---------------
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; ++m) {
    const uint16_t *row_src = src + m * N;

    // Init to FP16 ±Inf so masked non-finite lanes never disturb accums.
    const _Float16 f16_pos_inf =  std::numeric_limits<_Float16>::infinity();
    const _Float16 f16_neg_inf = -std::numeric_limits<_Float16>::infinity();
    __m512h vmin0 = _mm512_set1_ph(f16_pos_inf);
    __m512h vmax0 = _mm512_set1_ph(f16_neg_inf);
    __m512h vmin1 = vmin0, vmax1 = vmax0;
    __m512h vmin2 = vmin0, vmax2 = vmax0;
    __m512h vmin3 = vmin0, vmax3 = vmax0;

    int64_t j = 0;
    for (; j + 127 < N; j += 128) {
      __m512h v0 = _mm512_loadu_ph(row_src + j);
      __m512h v1 = _mm512_loadu_ph(row_src + j + 32);
      __m512h v2 = _mm512_loadu_ph(row_src + j + 64);
      __m512h v3 = _mm512_loadu_ph(row_src + j + 96);
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
    for (; j + 31 < N; j += 32) {
      __m512h v = _mm512_loadu_ph(row_src + j);
      __mmask32 k = finite_mask_ph(v);
      vmin0 = _mm512_mask_min_ph(vmin0, k, vmin0, v);
      vmax0 = _mm512_mask_max_ph(vmax0, k, vmax0, v);
    }

    vmin0 = _mm512_min_ph(_mm512_min_ph(vmin0, vmin1),
                           _mm512_min_ph(vmin2, vmin3));
    vmax0 = _mm512_max_ph(_mm512_max_ph(vmax0, vmax1),
                           _mm512_max_ph(vmax2, vmax3));
    float row_min = static_cast<float>(_mm512_reduce_min_ph(vmin0));
    float row_max = static_cast<float>(_mm512_reduce_max_ph(vmax0));

    for (; j < N; ++j) {
      float v = common::float16_t::f16_to_f32_val(row_src[j]);
      if (std::isfinite(v)) {
        row_min = std::min(row_min, v);
        row_max = std::max(row_max, v);
      }
    }

    float scale_f32;
    int32_t zp;
    compute_asymmetric_scale_zp_ph(row_min, row_max, scale_f32, zp);
    scales[m] = scale_f32;
    zps[m]    = zp;
  }

  // -- Pass 2: quantize (parallel over M*N contiguous elements) -----------
  const int64_t total = M * N;
  constexpr int64_t grain_size = LOWOHA_REORDER_GRAIN_SIZE;
  zendnnl_parallel_for(0, total, grain_size,
      [&](int64_t begin, int64_t end)
          __attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16"))) {
    while (begin < end) {
      const int64_t m = begin / N;
      const int64_t row_end = std::min((m + 1) * N, end);
      const int64_t count = row_end - begin;
      const uint16_t *csrc = src + begin;
      uint8_t *cdst = dst + begin;

      int64_t k = 0;
      // Twin guards: (i) tiny scales make 1/scale overflow FP16, and
      // (ii) large |zp| make VCVTPH2W pre-saturate out-of-int16
      // quotients before the int32 zp add can bring them back into
      // range. Either failure forces the scalar tail in f32 with int32
      // zp, which matches the F32-FMA / scalar reference. See the
      // fused-asym kernel above for the full divergence rationale.
      if (common::fp16_inv_scale_is_finite(scales[m]) &&
          common::fp16_zp_safe_for_s16_narrow(zps[m])) {
        const _Float16 inv_scale_f16 = static_cast<_Float16>(1.0f / scales[m]);
        const __m512h vinv  = _mm512_set1_ph(inv_scale_f16);
        const __m512i vzp32 = _mm512_set1_epi32(zps[m]);
        const __m512i vlo32 = _mm512_setzero_si512();
        const __m512i vhi32 = _mm512_set1_epi32(255);

        for (; k + 31 < count; k += 32) {
          __m512h v   = _mm512_loadu_ph(csrc + k);
          __m512i s16 = quantize_to_s16_ph(v, vinv);
          __m512i s32_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(s16));
          __m512i s32_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(s16, 1));
          s32_lo = _mm512_add_epi32(s32_lo, vzp32);
          s32_hi = _mm512_add_epi32(s32_hi, vzp32);
          // Signed [0, 255] clamp BEFORE VPMOVUSDB; that intrinsic
          // interprets its input as unsigned and would wrap negative
          // s32 values to 255 instead of clamping to 0.
          s32_lo = _mm512_max_epi32(vlo32, _mm512_min_epi32(vhi32, s32_lo));
          s32_hi = _mm512_max_epi32(vlo32, _mm512_min_epi32(vhi32, s32_hi));
          __m128i u8_lo = _mm512_cvtusepi32_epi8(s32_lo);
          __m128i u8_hi = _mm512_cvtusepi32_epi8(s32_hi);
          _mm_storeu_si128(reinterpret_cast<__m128i *>(cdst + k),      u8_lo);
          _mm_storeu_si128(reinterpret_cast<__m128i *>(cdst + k + 16), u8_hi);
        }
      }
      for (; k < count; ++k) {
        float v = common::float16_t::f16_to_f32_val(csrc[k]);
        if (!std::isfinite(v)) { cdst[k] = 0; continue; }
        int32_t q = static_cast<int32_t>(std::nearbyint(v / scales[m])) + zps[m];
        q = std::max(0, std::min(255, q));
        cdst[k] = static_cast<uint8_t>(q);
      }
      begin = row_end;
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
       // siblings in per_token_avx512_f32_fma_kernel.cpp. The can_use_f16_fma_kernel()
       // helper returns false on this toolchain (gated on __GNUC__ >= 12 in
       // lowoha_reorder_common.hpp), so the dispatcher should never select
       // these in practice -- the delegation is defense in depth.

void dynamic_per_token_quant_f16_s8_avx512fp16(const uint16_t *src,
                                             int8_t *dst,
                                             float *scales,
                                             int64_t M, int64_t N) {
  dynamic_per_token_quant_f16_s8_native(src, dst, scales, M, N);
}

void dynamic_per_token_quant_f16_u8_avx512fp16(const uint16_t *src,
                                             uint8_t *dst,
                                             float *scales,
                                             int32_t *zps,
                                             int64_t M, int64_t N) {
  dynamic_per_token_quant_f16_u8_native(src, dst, scales, zps, M, N);
}

void dynamic_per_token_quant_f16_s8_unfused_avx512fp16(const uint16_t *src,
                                                     int8_t *dst,
                                                     float *scales,
                                                     int64_t M, int64_t N) {
  dynamic_per_token_quant_f16_s8_unfused_native(src, dst, scales, M, N);
}

void dynamic_per_token_quant_f16_u8_unfused_avx512fp16(const uint16_t *src,
                                                     uint8_t *dst,
                                                     float *scales,
                                                     int32_t *zps,
                                                     int64_t M, int64_t N) {
  dynamic_per_token_quant_f16_u8_unfused_native(src, dst, scales, zps, M, N);
}

#endif  // __GNUC__ >= 12

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
