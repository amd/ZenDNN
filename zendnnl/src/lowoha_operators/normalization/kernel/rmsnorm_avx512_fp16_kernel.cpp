/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rmsnorm_avx512_fp16_kernel.hpp"

#if defined(__GNUC__) && (__GNUC__ >= 12)

#include "common/float16.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>

namespace zendnnl {
namespace lowoha {
namespace normalization {

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,avx512fp16,fma")
#endif

using zendnnl::common::float16_t;
using zendnnl::common::f16x32_load_typed;
using zendnnl::common::f16x32_load_tail_typed;
using zendnnl::common::f16x32_load_mask_typed;
using zendnnl::common::f16x32_store_typed;
using zendnnl::common::f16x32_store_mask_typed;
using zendnnl::common::f16x32_store_tail_typed;
using zendnnl::common::reduce_add_ph_to_fp32;

// =============================================================================
// Plain RMS Norm (native FP16) — single row, processes one [1, norm_size] slice.
//
//   rms    = sqrt( (1/N) * Σ input[i]² + eps )
//   out[i] = gamma[i] * input[i] / rms
//
// Pass 1 — sum-of-squares (4×32 = 128 elements/iter):
//   4 vmovdqu64(src) + 4 vfmadd231ph(acc) = 8 instructions.
//   FMA-bound (Zen5 / SPR 2 FP16 FMA ports): 4 fmadd / 2 = 2 cycles/iter.
//   Reduction is widened to FP32 once (precision; long rows).
//
// Pass 2 — normalize + optional gamma (4×32 = 128 elements/iter):
//   With scale: 4 load(γ) + 4 vmulph(γ×inv_rms) + 4 load(src) + 4 vmulph
//               + 4 store = 20 instructions.
//   Without:    4 load + 4 vmulph + 4 store = 12 instructions.
// =============================================================================

// Templated on the in-memory dtypes of input, output, and gamma. The dispatch
// in lowoha_normalization.cpp gates on InType/OutType/GammaType ∈ {f16, f32}.
// For non-f16 operand types the typed load/store helpers emit one vcvtps2ph /
// vcvtph2ps per 32-lane block; the FMA inner loop stays in __m512h regardless.
template <typename InType, typename OutType, typename GammaType>
static inline void rms_norm_row_fp16(
  const void *__restrict__ in_row,
  void       *__restrict__ out_row,
  const void *__restrict__ gamma,
  uint64_t norm_size,
  float    inv_n,
  float    epsilon,
  bool     use_scale
) {
  const InType    *in_p = static_cast<const InType *>(in_row);
  OutType         *out_p = static_cast<OutType *>(out_row);
  const GammaType *g_p   = static_cast<const GammaType *>(gamma);

  __m512h acc0 = _mm512_setzero_ph(), acc1 = _mm512_setzero_ph();
  __m512h acc2 = _mm512_setzero_ph(), acc3 = _mm512_setzero_ph();

  uint64_t i = 0;
  const uint64_t vec128 = norm_size & ~127ULL;

  for (; i < vec128; i += 128) {
    __m512h s0 = f16x32_load_typed<InType>(in_p + i +   0);
    __m512h s1 = f16x32_load_typed<InType>(in_p + i +  32);
    __m512h s2 = f16x32_load_typed<InType>(in_p + i +  64);
    __m512h s3 = f16x32_load_typed<InType>(in_p + i +  96);

    acc0 = _mm512_fmadd_ph(s0, s0, acc0);
    acc1 = _mm512_fmadd_ph(s1, s1, acc1);
    acc2 = _mm512_fmadd_ph(s2, s2, acc2);
    acc3 = _mm512_fmadd_ph(s3, s3, acc3);
  }
  for (; i + 31 < norm_size; i += 32) {
    __m512h s0 = f16x32_load_typed<InType>(in_p + i);
    acc0 = _mm512_fmadd_ph(s0, s0, acc0);
  }
  if (i < norm_size) {
    const int tail = static_cast<int>(norm_size - i);
    __mmask32 mask = (__mmask32)((1ULL << tail) - 1ULL);
    __m512h s0 = f16x32_load_tail_typed<InType>(in_p + i, mask, tail);
    acc0 = _mm512_fmadd_ph(s0, s0, acc0);
  }

  float total_sq = reduce_add_ph_to_fp32(acc0)
                   + reduce_add_ph_to_fp32(acc1)
                   + reduce_add_ph_to_fp32(acc2)
                   + reduce_add_ph_to_fp32(acc3);
  float inv_rms = 1.0f / std::sqrt(total_sq * inv_n + epsilon);
  __m512h inv_rms_h = _mm512_set1_ph((_Float16)inv_rms);

  i = 0;
  if (use_scale) {
    for (; i < vec128; i += 128) {
      __m512h g0 = _mm512_mul_ph(f16x32_load_typed<GammaType>(g_p + i +   0),
                                 inv_rms_h);
      __m512h g1 = _mm512_mul_ph(f16x32_load_typed<GammaType>(g_p + i +  32),
                                 inv_rms_h);
      __m512h g2 = _mm512_mul_ph(f16x32_load_typed<GammaType>(g_p + i +  64),
                                 inv_rms_h);
      __m512h g3 = _mm512_mul_ph(f16x32_load_typed<GammaType>(g_p + i +  96),
                                 inv_rms_h);

      f16x32_store_typed<OutType>(out_p + i +   0,
                                  _mm512_mul_ph(f16x32_load_typed<InType>(in_p + i +   0), g0));
      f16x32_store_typed<OutType>(out_p + i +  32,
                                  _mm512_mul_ph(f16x32_load_typed<InType>(in_p + i +  32), g1));
      f16x32_store_typed<OutType>(out_p + i +  64,
                                  _mm512_mul_ph(f16x32_load_typed<InType>(in_p + i +  64), g2));
      f16x32_store_typed<OutType>(out_p + i +  96,
                                  _mm512_mul_ph(f16x32_load_typed<InType>(in_p + i +  96), g3));
    }
    for (; i + 31 < norm_size; i += 32) {
      __m512h g0 = _mm512_mul_ph(f16x32_load_typed<GammaType>(g_p + i), inv_rms_h);
      f16x32_store_typed<OutType>(out_p + i,
                                  _mm512_mul_ph(f16x32_load_typed<InType>(in_p + i), g0));
    }
    if (i < norm_size) {
      const int tail = static_cast<int>(norm_size - i);
      __mmask32 mask = (__mmask32)((1ULL << tail) - 1ULL);
      __m512h g0 = _mm512_mul_ph(f16x32_load_mask_typed<GammaType>(g_p + i, mask),
                                 inv_rms_h);
      f16x32_store_tail_typed<OutType>(out_p + i,
                                       _mm512_mul_ph(f16x32_load_tail_typed<InType>(in_p + i, mask, tail), g0),
                                       mask, tail);
    }
  }
  else {
    for (; i < vec128; i += 128) {
      f16x32_store_typed<OutType>(out_p + i +   0,
                                  _mm512_mul_ph(f16x32_load_typed<InType>(in_p + i +   0), inv_rms_h));
      f16x32_store_typed<OutType>(out_p + i +  32,
                                  _mm512_mul_ph(f16x32_load_typed<InType>(in_p + i +  32), inv_rms_h));
      f16x32_store_typed<OutType>(out_p + i +  64,
                                  _mm512_mul_ph(f16x32_load_typed<InType>(in_p + i +  64), inv_rms_h));
      f16x32_store_typed<OutType>(out_p + i +  96,
                                  _mm512_mul_ph(f16x32_load_typed<InType>(in_p + i +  96), inv_rms_h));
    }
    for (; i + 31 < norm_size; i += 32) {
      f16x32_store_typed<OutType>(out_p + i,
                                  _mm512_mul_ph(f16x32_load_typed<InType>(in_p + i), inv_rms_h));
    }
    if (i < norm_size) {
      const int tail = static_cast<int>(norm_size - i);
      __mmask32 mask = (__mmask32)((1ULL << tail) - 1ULL);
      f16x32_store_tail_typed<OutType>(out_p + i,
                                       _mm512_mul_ph(f16x32_load_tail_typed<InType>(in_p + i, mask, tail),
                                           inv_rms_h),
                                       mask, tail);
    }
  }
}

// =============================================================================
// Fused Add + RMS Norm (native FP16) — single row.
//
//   residual[i] += input[i]           (in-place, FP16)
//   rms    = sqrt( (1/N) * Σ residual[i]² + eps )
//   out[i] = gamma[i] * residual[i] / rms
//
// Pass 2 re-reads residual from L1 (warm from pass 1 stores). This matches
// the reference precision semantics: out is computed from the FP16-rounded
// in-place sum, not the unrounded FP32 sum.
// =============================================================================

static inline void fused_add_rms_row_fp16(
  const void *__restrict__ in_row,
  void       *__restrict__ out_row,
  void       *__restrict__ res_row,
  const void *__restrict__ gamma,
  uint64_t norm_size,
  float    inv_n,
  float    epsilon,
  bool     use_scale
) {
  const uint16_t *in_u16  = static_cast<const uint16_t *>(in_row);
  uint16_t       *out_u16 = static_cast<uint16_t *>(out_row);
  uint16_t       *res_u16 = static_cast<uint16_t *>(res_row);
  const uint16_t *g_u16   = static_cast<const uint16_t *>(gamma);

  __m512h acc0 = _mm512_setzero_ph(), acc1 = _mm512_setzero_ph();
  __m512h acc2 = _mm512_setzero_ph(), acc3 = _mm512_setzero_ph();

  uint64_t i = 0;
  const uint64_t vec128 = norm_size & ~127ULL;

  for (; i < vec128; i += 128) {
    __m512h s0 = _mm512_add_ph(f16x32_load_typed<float16_t>(res_u16 + i +   0),
                               f16x32_load_typed<float16_t>(in_u16  + i +   0));
    __m512h s1 = _mm512_add_ph(f16x32_load_typed<float16_t>(res_u16 + i +  32),
                               f16x32_load_typed<float16_t>(in_u16  + i +  32));
    __m512h s2 = _mm512_add_ph(f16x32_load_typed<float16_t>(res_u16 + i +  64),
                               f16x32_load_typed<float16_t>(in_u16  + i +  64));
    __m512h s3 = _mm512_add_ph(f16x32_load_typed<float16_t>(res_u16 + i +  96),
                               f16x32_load_typed<float16_t>(in_u16  + i +  96));

    f16x32_store_typed<float16_t>(res_u16 + i +   0, s0);
    f16x32_store_typed<float16_t>(res_u16 + i +  32, s1);
    f16x32_store_typed<float16_t>(res_u16 + i +  64, s2);
    f16x32_store_typed<float16_t>(res_u16 + i +  96, s3);

    acc0 = _mm512_fmadd_ph(s0, s0, acc0);
    acc1 = _mm512_fmadd_ph(s1, s1, acc1);
    acc2 = _mm512_fmadd_ph(s2, s2, acc2);
    acc3 = _mm512_fmadd_ph(s3, s3, acc3);
  }
  for (; i + 31 < norm_size; i += 32) {
    __m512h s0 = _mm512_add_ph(f16x32_load_typed<float16_t>(res_u16 + i),
                               f16x32_load_typed<float16_t>(in_u16  + i));
    f16x32_store_typed<float16_t>(res_u16 + i, s0);
    acc0 = _mm512_fmadd_ph(s0, s0, acc0);
  }
  if (i < norm_size) {
    __mmask32 mask = (__mmask32)((1ULL << (norm_size - i)) - 1ULL);
    __m512h s0 = _mm512_add_ph(f16x32_load_mask_typed<float16_t>(res_u16 + i, mask),
                               f16x32_load_mask_typed<float16_t>(in_u16  + i, mask));
    f16x32_store_mask_typed<float16_t>(res_u16 + i, s0, mask);
    acc0 = _mm512_fmadd_ph(s0, s0, acc0);
  }

  float total_sq = reduce_add_ph_to_fp32(acc0)
                   + reduce_add_ph_to_fp32(acc1)
                   + reduce_add_ph_to_fp32(acc2)
                   + reduce_add_ph_to_fp32(acc3);
  float inv_rms = 1.0f / std::sqrt(total_sq * inv_n + epsilon);
  __m512h inv_rms_h = _mm512_set1_ph((_Float16)inv_rms);

  i = 0;
  if (use_scale) {
    for (; i < vec128; i += 128) {
      __m512h g0 = _mm512_mul_ph(f16x32_load_typed<float16_t>(g_u16 + i +   0),
                                 inv_rms_h);
      __m512h g1 = _mm512_mul_ph(f16x32_load_typed<float16_t>(g_u16 + i +  32),
                                 inv_rms_h);
      __m512h g2 = _mm512_mul_ph(f16x32_load_typed<float16_t>(g_u16 + i +  64),
                                 inv_rms_h);
      __m512h g3 = _mm512_mul_ph(f16x32_load_typed<float16_t>(g_u16 + i +  96),
                                 inv_rms_h);

      f16x32_store_typed<float16_t>(out_u16 + i +   0,
                                    _mm512_mul_ph(f16x32_load_typed<float16_t>(res_u16 + i +   0), g0));
      f16x32_store_typed<float16_t>(out_u16 + i +  32,
                                    _mm512_mul_ph(f16x32_load_typed<float16_t>(res_u16 + i +  32), g1));
      f16x32_store_typed<float16_t>(out_u16 + i +  64,
                                    _mm512_mul_ph(f16x32_load_typed<float16_t>(res_u16 + i +  64), g2));
      f16x32_store_typed<float16_t>(out_u16 + i +  96,
                                    _mm512_mul_ph(f16x32_load_typed<float16_t>(res_u16 + i +  96), g3));
    }
    for (; i + 31 < norm_size; i += 32) {
      __m512h g0 = _mm512_mul_ph(f16x32_load_typed<float16_t>(g_u16 + i), inv_rms_h);
      f16x32_store_typed<float16_t>(out_u16 + i,
                                    _mm512_mul_ph(f16x32_load_typed<float16_t>(res_u16 + i), g0));
    }
    if (i < norm_size) {
      __mmask32 mask = (__mmask32)((1ULL << (norm_size - i)) - 1ULL);
      __m512h g0 = _mm512_mul_ph(f16x32_load_mask_typed<float16_t>(g_u16 + i, mask),
                                 inv_rms_h);
      f16x32_store_mask_typed<float16_t>(out_u16 + i,
                                         _mm512_mul_ph(f16x32_load_mask_typed<float16_t>(res_u16 + i, mask), g0),
                                         mask);
    }
  }
  else {
    for (; i < vec128; i += 128) {
      f16x32_store_typed<float16_t>(out_u16 + i +   0,
                                    _mm512_mul_ph(f16x32_load_typed<float16_t>(res_u16 + i +   0), inv_rms_h));
      f16x32_store_typed<float16_t>(out_u16 + i +  32,
                                    _mm512_mul_ph(f16x32_load_typed<float16_t>(res_u16 + i +  32), inv_rms_h));
      f16x32_store_typed<float16_t>(out_u16 + i +  64,
                                    _mm512_mul_ph(f16x32_load_typed<float16_t>(res_u16 + i +  64), inv_rms_h));
      f16x32_store_typed<float16_t>(out_u16 + i +  96,
                                    _mm512_mul_ph(f16x32_load_typed<float16_t>(res_u16 + i +  96), inv_rms_h));
    }
    for (; i + 31 < norm_size; i += 32) {
      f16x32_store_typed<float16_t>(out_u16 + i,
                                    _mm512_mul_ph(f16x32_load_typed<float16_t>(res_u16 + i), inv_rms_h));
    }
    if (i < norm_size) {
      __mmask32 mask = (__mmask32)((1ULL << (norm_size - i)) - 1ULL);
      f16x32_store_mask_typed<float16_t>(out_u16 + i,
                                         _mm512_mul_ph(f16x32_load_mask_typed<float16_t>(res_u16 + i, mask),
                                             inv_rms_h),
                                         mask);
    }
  }
}

// =====================================================================
// Dispatches RMS_NORM (with mixed-dtype f16/f32 boundaries) and
// FUSED_ADD_RMS_NORM (all-f16, residual aliases src).
//
// Plain RMS_NORM supports three (src_dt, dst_dt) combos:
//   (f16, f16) — no boundary conversion (fastest)
//   (f16, f32) — store boundary widens __m512h -> __m512 + f32 store
//   (f32, f16) — load boundary narrows __m512 -> __m512h, then f16 store
// orthogonally crossed with two gamma_dt options:
//   gamma_dt = f16 — pure load (no conversion)
//   gamma_dt = f32 — load boundary narrows __m512 -> __m512h
// for a total of 6 templated instantiations. The FMA inner loop stays in
// __m512h regardless; the conversions happen only at load/store boundaries.
//
// FUSED_ADD_RMS_NORM keeps the all-f16 restriction because residual is
// read-modify-written in place and must share src dtype; mixed-dtype
// fused-add falls back to the F32-accumulating AVX-512 kernel via the
// dispatch eligibility check.
// =====================================================================

status_t rms_norm_avx512_fp16(
  const void *input,
  void       *output,
  void       *residual,
  const void *gamma,
  norm_params &params
) {
  const float    inv_n  = 1.0f / static_cast<float>(params.norm_size);
  const uint64_t N      = params.norm_size;
  const int64_t  batch  = static_cast<int64_t>(params.batch);
  const bool     is_fused = (params.norm_type == norm_type_t::FUSED_ADD_RMS_NORM
                             && residual);

  // Pick the right template instantiation based on (src_dt, dst_dt, gamma_dt).
  // The dispatch upstream guarantees gamma_dt ∈ {f16, f32} here.
  using rms_row_fn_t = void(*)(const void *, void *, const void *,
                               uint64_t, float, float, bool);
  rms_row_fn_t row_fn = nullptr;
  size_t in_elem_sz   = 0;
  size_t out_elem_sz  = 0;

  if (is_fused) {
    if (params.src_dt   != data_type_t::f16 ||
        params.dst_dt   != data_type_t::f16 ||
        params.gamma_dt != data_type_t::f16) {
      return status_t::unimplemented;
    }
  }
  else {
    const data_type_t src_dt   = params.src_dt;
    const data_type_t dst_dt   = params.dst_dt;
    const data_type_t gamma_dt = params.gamma_dt;

    if (src_dt == data_type_t::f16 && dst_dt == data_type_t::f16 &&
        gamma_dt == data_type_t::f16) {
      row_fn = &rms_norm_row_fp16<float16_t, float16_t, float16_t>;
      in_elem_sz = sizeof(float16_t);
      out_elem_sz = sizeof(float16_t);
    }
    else if (src_dt == data_type_t::f16 && dst_dt == data_type_t::f16 &&
             gamma_dt == data_type_t::f32) {
      row_fn = &rms_norm_row_fp16<float16_t, float16_t, float>;
      in_elem_sz = sizeof(float16_t);
      out_elem_sz = sizeof(float16_t);
    }
    else if (src_dt == data_type_t::f16 && dst_dt == data_type_t::f32 &&
             gamma_dt == data_type_t::f16) {
      row_fn = &rms_norm_row_fp16<float16_t, float, float16_t>;
      in_elem_sz = sizeof(float16_t);
      out_elem_sz = sizeof(float);
    }
    else if (src_dt == data_type_t::f16 && dst_dt == data_type_t::f32 &&
             gamma_dt == data_type_t::f32) {
      row_fn = &rms_norm_row_fp16<float16_t, float, float>;
      in_elem_sz = sizeof(float16_t);
      out_elem_sz = sizeof(float);
    }
    else if (src_dt == data_type_t::f32 && dst_dt == data_type_t::f16 &&
             gamma_dt == data_type_t::f16) {
      row_fn = &rms_norm_row_fp16<float, float16_t, float16_t>;
      in_elem_sz = sizeof(float);
      out_elem_sz = sizeof(float16_t);
    }
    else if (src_dt == data_type_t::f32 && dst_dt == data_type_t::f16 &&
             gamma_dt == data_type_t::f32) {
      row_fn = &rms_norm_row_fp16<float, float16_t, float>;
      in_elem_sz = sizeof(float);
      out_elem_sz = sizeof(float16_t);
    }
    else {
      // Combo unsupported by the FP16-FMA path; caller will fall through
      // to the FP32-accumulating AVX-512 kernel.
      return status_t::unimplemented;
    }
  }

  auto row_loop = [&](int64_t b) {
    if (!is_fused) {
      const char *in_byte_ptr  = static_cast<const char *>(input)
                                 + b * N * in_elem_sz;
      char *out_byte_ptr       = static_cast<char *>(output)
                                 + b * N * out_elem_sz;
      row_fn(in_byte_ptr, out_byte_ptr, gamma,
             N, inv_n, params.epsilon, params.use_scale);
    }
    else {
      fused_add_rms_row_fp16(
        static_cast<const uint16_t *>(input)  + b * N,
        static_cast<uint16_t *>(output)       + b * N,
        static_cast<uint16_t *>(residual)     + b * N,
        gamma, N, inv_n, params.epsilon, params.use_scale);
    }
  };

  if (batch <= 1) {
    for (int64_t b = 0; b < batch; ++b) {
      row_loop(b);
    }
  }
  else {
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < batch; ++b) {
      row_loop(b);
    }
  }
  return status_t::success;
}

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

#else // __GNUC__ < 12

namespace zendnnl {
namespace lowoha {
namespace normalization {

status_t rms_norm_avx512_fp16(
  const void *,
  void *,
  void *,
  const void *,
  norm_params &
) {
  return status_t::isa_unsupported;
}

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

#endif // __GNUC__ >= 12
