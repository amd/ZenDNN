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

#include "layernorm_avx512_fp16_kernel.hpp"

#if defined(__GNUC__) && (__GNUC__ >= 12)

#include "common/float16.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>

namespace zendnnl {
namespace lowoha {
namespace normalization {

#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,avx512fp16,fma")

using zendnnl::common::float16_t;
using zendnnl::common::f16x32_load_typed;
using zendnnl::common::f16x32_load_tail_typed;
using zendnnl::common::f16x32_load_mask_typed;
using zendnnl::common::f16x32_store_typed;
using zendnnl::common::f16x32_store_tail_typed;
using zendnnl::common::reduce_add_ph_to_fp32;

// =============================================================================
// Layer Norm (native FP16) — single row, processes one [1, norm_size] slice.
//
//   mean    = (1/N) * Σ x[i]
//   var     = (1/N) * Σ x[i]² - mean²          (clamped to ≥ 0)
//   inv_std = 1 / sqrt(var + eps)
//   y[i]   = gamma[i] * (x[i] - mean) * inv_std + beta[i]
//
// Pass 1 — simultaneous sum and sum-of-squares (4×32 = 128 elements/iter):
//   Register map (12 of 32 ZMM used at peak):
//     zmm0  - zmm3  : 4 sum   accumulators (sum0–sum3, __m512h)
//     zmm4  - zmm7  : 4 sumsq accumulators (sq0–sq3,   __m512h)
//     zmm8  - zmm11 : 4 data loads (s0–s3, 32 FP16 lanes each)
//
//   4 vmovdqu64(src) + 4 vaddph(sum) + 4 vfmadd231ph(sq) = 12 instructions.
//   FMA-bound (Zen5 / SPR 2 FP16 FMA ports): 4 fmadd / 2 = 2 cycles/iter.
//
//   Reductions of sum and sum-of-squares are widened to FP32 once before the
//   horizontal sum (see reduce_add_ph_to_fp32) to avoid FP16 precision loss
//   across long rows.
//
// Pass 2 — normalize + optional gamma/beta (4×32 = 128 elements/iter):
//   With scale+shift, per 4×32-lane block:
//     4 load(γ) + 4 vmulph(γ×inv_std) + 4 load(src) + 4 vsubph(mean)
//     + 4 load(β) + 4 vfmaddph + 4 store = 28 instructions.
// =============================================================================

// Templated on the in-memory dtypes of input, output, gamma and beta. The
// dispatch in lowoha_normalization.cpp gates each operand type on {f16, f32}.
// For any non-f16 operand the typed load/store helpers emit one vcvtps2ph /
// vcvtph2ps per 32-lane block; the inner-loop arithmetic stays in __m512h.
template <typename InType, typename OutType,
          typename GammaType, typename BetaType>
static inline void layer_norm_row_fp16(
  const void *__restrict__ in_row,
  void       *__restrict__ out_row,
  const void *__restrict__ gamma,
  const void *__restrict__ beta,
  uint64_t norm_size,
  float    inv_n,
  float    epsilon,
  bool     use_scale,
  bool     use_shift
) {
  const InType    *in_p = static_cast<const InType *>(in_row);
  OutType         *out_p = static_cast<OutType *>(out_row);
  const GammaType *g_p   = static_cast<const GammaType *>(gamma);
  const BetaType  *b_p   = static_cast<const BetaType *>(beta);

  __m512h sum0 = _mm512_setzero_ph(), sum1 = _mm512_setzero_ph();
  __m512h sum2 = _mm512_setzero_ph(), sum3 = _mm512_setzero_ph();
  __m512h sq0  = _mm512_setzero_ph(), sq1  = _mm512_setzero_ph();
  __m512h sq2  = _mm512_setzero_ph(), sq3  = _mm512_setzero_ph();

  uint64_t i = 0;
  const uint64_t vec128 = norm_size & ~127ULL;

  for (; i < vec128; i += 128) {
    __m512h s0 = f16x32_load_typed<InType>(in_p + i +   0);
    __m512h s1 = f16x32_load_typed<InType>(in_p + i +  32);
    __m512h s2 = f16x32_load_typed<InType>(in_p + i +  64);
    __m512h s3 = f16x32_load_typed<InType>(in_p + i +  96);

    sum0 = _mm512_add_ph(sum0, s0);
    sum1 = _mm512_add_ph(sum1, s1);
    sum2 = _mm512_add_ph(sum2, s2);
    sum3 = _mm512_add_ph(sum3, s3);

    sq0 = _mm512_fmadd_ph(s0, s0, sq0);
    sq1 = _mm512_fmadd_ph(s1, s1, sq1);
    sq2 = _mm512_fmadd_ph(s2, s2, sq2);
    sq3 = _mm512_fmadd_ph(s3, s3, sq3);
  }
  for (; i + 31 < norm_size; i += 32) {
    __m512h s0 = f16x32_load_typed<InType>(in_p + i);
    sum0 = _mm512_add_ph(sum0, s0);
    sq0  = _mm512_fmadd_ph(s0, s0, sq0);
  }
  if (i < norm_size) {
    const int tail = static_cast<int>(norm_size - i);
    __mmask32 mask = (__mmask32)((1ULL << tail) - 1ULL);
    __m512h s0 = f16x32_load_tail_typed<InType>(in_p + i, mask, tail);
    sum0 = _mm512_add_ph(sum0, s0);
    sq0  = _mm512_fmadd_ph(s0, s0, sq0);
  }

  // Widen to FP32 once for the horizontal reduce (precision; see header).
  float total_sum = reduce_add_ph_to_fp32(sum0)
                    + reduce_add_ph_to_fp32(sum1)
                    + reduce_add_ph_to_fp32(sum2)
                    + reduce_add_ph_to_fp32(sum3);
  float total_sq  = reduce_add_ph_to_fp32(sq0)
                    + reduce_add_ph_to_fp32(sq1)
                    + reduce_add_ph_to_fp32(sq2)
                    + reduce_add_ph_to_fp32(sq3);

  float mean    = total_sum * inv_n;
  float var     = std::max(0.0f, total_sq * inv_n - mean * mean);
  float inv_std = 1.0f / std::sqrt(var + epsilon);

  // Round mean / inv_std to FP16 once before broadcasting, matching the
  // precision the per-element compute will see.
  __m512h mean_h    = _mm512_set1_ph((_Float16)mean);
  __m512h inv_std_h = _mm512_set1_ph((_Float16)inv_std);

  i = 0;
  if (use_scale) {
    for (; i < vec128; i += 128) {
      __m512h g0 = _mm512_mul_ph(f16x32_load_typed<GammaType>(g_p + i +   0),
                                 inv_std_h);
      __m512h g1 = _mm512_mul_ph(f16x32_load_typed<GammaType>(g_p + i +  32),
                                 inv_std_h);
      __m512h g2 = _mm512_mul_ph(f16x32_load_typed<GammaType>(g_p + i +  64),
                                 inv_std_h);
      __m512h g3 = _mm512_mul_ph(f16x32_load_typed<GammaType>(g_p + i +  96),
                                 inv_std_h);

      __m512h d0 = _mm512_sub_ph(f16x32_load_typed<InType>(in_p + i +   0), mean_h);
      __m512h d1 = _mm512_sub_ph(f16x32_load_typed<InType>(in_p + i +  32), mean_h);
      __m512h d2 = _mm512_sub_ph(f16x32_load_typed<InType>(in_p + i +  64), mean_h);
      __m512h d3 = _mm512_sub_ph(f16x32_load_typed<InType>(in_p + i +  96), mean_h);

      if (use_shift) {
        __m512h b0 = f16x32_load_typed<BetaType>(b_p + i +   0);
        __m512h b1 = f16x32_load_typed<BetaType>(b_p + i +  32);
        __m512h b2 = f16x32_load_typed<BetaType>(b_p + i +  64);
        __m512h b3 = f16x32_load_typed<BetaType>(b_p + i +  96);
        d0 = _mm512_fmadd_ph(d0, g0, b0);
        d1 = _mm512_fmadd_ph(d1, g1, b1);
        d2 = _mm512_fmadd_ph(d2, g2, b2);
        d3 = _mm512_fmadd_ph(d3, g3, b3);
      }
      else {
        d0 = _mm512_mul_ph(d0, g0);
        d1 = _mm512_mul_ph(d1, g1);
        d2 = _mm512_mul_ph(d2, g2);
        d3 = _mm512_mul_ph(d3, g3);
      }

      f16x32_store_typed<OutType>(out_p + i +   0, d0);
      f16x32_store_typed<OutType>(out_p + i +  32, d1);
      f16x32_store_typed<OutType>(out_p + i +  64, d2);
      f16x32_store_typed<OutType>(out_p + i +  96, d3);
    }
    for (; i + 31 < norm_size; i += 32) {
      __m512h g0 = _mm512_mul_ph(f16x32_load_typed<GammaType>(g_p + i), inv_std_h);
      __m512h d0 = _mm512_sub_ph(f16x32_load_typed<InType>(in_p + i), mean_h);
      if (use_shift) {
        __m512h b0 = f16x32_load_typed<BetaType>(b_p + i);
        d0 = _mm512_fmadd_ph(d0, g0, b0);
      }
      else {
        d0 = _mm512_mul_ph(d0, g0);
      }
      f16x32_store_typed<OutType>(out_p + i, d0);
    }
    if (i < norm_size) {
      const int tail = static_cast<int>(norm_size - i);
      __mmask32 mask = (__mmask32)((1ULL << tail) - 1ULL);
      __m512h g0 = _mm512_mul_ph(f16x32_load_mask_typed<GammaType>(g_p + i, mask),
                                 inv_std_h);
      __m512h d0 = _mm512_sub_ph(f16x32_load_tail_typed<InType>(in_p + i, mask, tail),
                                 mean_h);
      if (use_shift) {
        __m512h b0 = f16x32_load_mask_typed<BetaType>(b_p + i, mask);
        d0 = _mm512_fmadd_ph(d0, g0, b0);
      }
      else {
        d0 = _mm512_mul_ph(d0, g0);
      }
      f16x32_store_tail_typed<OutType>(out_p + i, d0, mask, tail);
    }
  }
  else {
    for (; i < vec128; i += 128) {
      __m512h d0 = _mm512_mul_ph(_mm512_sub_ph(f16x32_load_typed<InType>
                                 (in_p + i +   0),
                                 mean_h), inv_std_h);
      __m512h d1 = _mm512_mul_ph(_mm512_sub_ph(f16x32_load_typed<InType>
                                 (in_p + i +  32),
                                 mean_h), inv_std_h);
      __m512h d2 = _mm512_mul_ph(_mm512_sub_ph(f16x32_load_typed<InType>
                                 (in_p + i +  64),
                                 mean_h), inv_std_h);
      __m512h d3 = _mm512_mul_ph(_mm512_sub_ph(f16x32_load_typed<InType>
                                 (in_p + i +  96),
                                 mean_h), inv_std_h);

      if (use_shift) {
        d0 = _mm512_add_ph(d0, f16x32_load_typed<BetaType>(b_p + i +   0));
        d1 = _mm512_add_ph(d1, f16x32_load_typed<BetaType>(b_p + i +  32));
        d2 = _mm512_add_ph(d2, f16x32_load_typed<BetaType>(b_p + i +  64));
        d3 = _mm512_add_ph(d3, f16x32_load_typed<BetaType>(b_p + i +  96));
      }

      f16x32_store_typed<OutType>(out_p + i +   0, d0);
      f16x32_store_typed<OutType>(out_p + i +  32, d1);
      f16x32_store_typed<OutType>(out_p + i +  64, d2);
      f16x32_store_typed<OutType>(out_p + i +  96, d3);
    }
    for (; i + 31 < norm_size; i += 32) {
      __m512h d0 = _mm512_mul_ph(_mm512_sub_ph(f16x32_load_typed<InType>(in_p + i),
                                 mean_h), inv_std_h);
      if (use_shift) {
        d0 = _mm512_add_ph(d0, f16x32_load_typed<BetaType>(b_p + i));
      }
      f16x32_store_typed<OutType>(out_p + i, d0);
    }
    if (i < norm_size) {
      const int tail = static_cast<int>(norm_size - i);
      __mmask32 mask = (__mmask32)((1ULL << tail) - 1ULL);
      __m512h d0 = _mm512_mul_ph(_mm512_sub_ph(
                                   f16x32_load_tail_typed<InType>(in_p + i, mask, tail),
                                   mean_h), inv_std_h);
      if (use_shift) {
        d0 = _mm512_add_ph(d0, f16x32_load_mask_typed<BetaType>(b_p + i, mask));
      }
      f16x32_store_tail_typed<OutType>(out_p + i, d0, mask, tail);
    }
  }
}

// =====================================================================
// Entry point — dispatches LAYER_NORM (native FP16) over the batch.
//
// Supports three (src_dt, dst_dt) combos:
//   (f16, f16) — no boundary conversion
//   (f16, f32) — store boundary widens __m512h -> __m512
//   (f32, f16) — load boundary narrows __m512 -> __m512h
// orthogonally crossed with gamma_dt ∈ {f16, f32} and beta_dt ∈ {f16, f32}
// for a total of 12 templated instantiations. The FMA inner loop stays in
// __m512h regardless; conversions happen only at load/store boundaries.
//
// When use_shift is false, the beta path is dead and BetaType is irrelevant
// at runtime — but we still dispatch on params.beta_dt to keep the table
// regular (the dead beta loads disappear after inlining).
// =====================================================================

status_t layer_norm_avx512_fp16(
  const void *input,
  void       *output,
  const void *gamma,
  const void *beta,
  norm_params &params
) {
  const float    inv_n  = 1.0f / static_cast<float>(params.norm_size);
  const uint64_t N      = params.norm_size;
  const int64_t  batch  = static_cast<int64_t>(params.batch);

  using ln_row_fn_t = void(*)(const void *, void *, const void *, const void *,
                              uint64_t, float, float, bool, bool);
  ln_row_fn_t row_fn = nullptr;
  size_t in_elem_sz  = 0;
  size_t out_elem_sz = 0;

  // Compact 4-component dtype key: (src, dst, gamma, beta). Each component is
  // f16 or f32. When use_shift is false the beta buffer is never derefed, so
  // any leftover beta_dt is safe and we collapse to BetaType=float16_t. When
  // use_shift is true beta is read for real, so reject any beta_dt outside
  // {f16, f32} up front — the upstream dispatcher already gates this, but
  // honour the kernel's contract independently in case a caller invokes the
  // kernel directly.
  const data_type_t src_dt   = params.src_dt;
  const data_type_t dst_dt   = params.dst_dt;
  const data_type_t gamma_dt = params.gamma_dt;
  if (params.use_shift &&
      params.beta_dt != data_type_t::f16 &&
      params.beta_dt != data_type_t::f32) {
    return status_t::unimplemented;
  }
  const data_type_t beta_dt = (params.use_shift &&
                               params.beta_dt == data_type_t::f32)
                              ? data_type_t::f32 : data_type_t::f16;

  // (src, dst) drives the element sizes used to advance the row pointers.
  if (src_dt == data_type_t::f16 && dst_dt == data_type_t::f16) {
    in_elem_sz = sizeof(float16_t);
    out_elem_sz = sizeof(float16_t);
  }
  else if (src_dt == data_type_t::f16 && dst_dt == data_type_t::f32) {
    in_elem_sz = sizeof(float16_t);
    out_elem_sz = sizeof(float);
  }
  else if (src_dt == data_type_t::f32 && dst_dt == data_type_t::f16) {
    in_elem_sz = sizeof(float);
    out_elem_sz = sizeof(float16_t);
  }
  else {
    return status_t::unimplemented;
  }

  // Select the (gamma, beta) template specialization. 4 combinations per
  // (src, dst) tile; 12 total entries below.
  if (src_dt == data_type_t::f16 && dst_dt == data_type_t::f16) {
    if (gamma_dt == data_type_t::f16 && beta_dt == data_type_t::f16) {
      row_fn = &layer_norm_row_fp16<float16_t, float16_t, float16_t, float16_t>;
    }
    else if (gamma_dt == data_type_t::f16 && beta_dt == data_type_t::f32) {
      row_fn = &layer_norm_row_fp16<float16_t, float16_t, float16_t, float>;
    }
    else if (gamma_dt == data_type_t::f32 && beta_dt == data_type_t::f16) {
      row_fn = &layer_norm_row_fp16<float16_t, float16_t, float, float16_t>;
    }
    else if (gamma_dt == data_type_t::f32 && beta_dt == data_type_t::f32) {
      row_fn = &layer_norm_row_fp16<float16_t, float16_t, float, float>;
    }
  }
  else if (src_dt == data_type_t::f16 && dst_dt == data_type_t::f32) {
    if (gamma_dt == data_type_t::f16 && beta_dt == data_type_t::f16) {
      row_fn = &layer_norm_row_fp16<float16_t, float, float16_t, float16_t>;
    }
    else if (gamma_dt == data_type_t::f16 && beta_dt == data_type_t::f32) {
      row_fn = &layer_norm_row_fp16<float16_t, float, float16_t, float>;
    }
    else if (gamma_dt == data_type_t::f32 && beta_dt == data_type_t::f16) {
      row_fn = &layer_norm_row_fp16<float16_t, float, float, float16_t>;
    }
    else if (gamma_dt == data_type_t::f32 && beta_dt == data_type_t::f32) {
      row_fn = &layer_norm_row_fp16<float16_t, float, float, float>;
    }
  }
  else if (src_dt == data_type_t::f32 && dst_dt == data_type_t::f16) {
    if (gamma_dt == data_type_t::f16 && beta_dt == data_type_t::f16) {
      row_fn = &layer_norm_row_fp16<float, float16_t, float16_t, float16_t>;
    }
    else if (gamma_dt == data_type_t::f16 && beta_dt == data_type_t::f32) {
      row_fn = &layer_norm_row_fp16<float, float16_t, float16_t, float>;
    }
    else if (gamma_dt == data_type_t::f32 && beta_dt == data_type_t::f16) {
      row_fn = &layer_norm_row_fp16<float, float16_t, float, float16_t>;
    }
    else if (gamma_dt == data_type_t::f32 && beta_dt == data_type_t::f32) {
      row_fn = &layer_norm_row_fp16<float, float16_t, float, float>;
    }
  }

  if (row_fn == nullptr) {
    // Combo unsupported by the FP16-FMA path; caller will fall through
    // to the FP32-accumulating AVX-512 kernel.
    return status_t::unimplemented;
  }

  auto row_loop = [&](int64_t b) {
    const char *in_byte_ptr  = static_cast<const char *>(input)
                               + b * N * in_elem_sz;
    char *out_byte_ptr       = static_cast<char *>(output)
                               + b * N * out_elem_sz;
    row_fn(in_byte_ptr, out_byte_ptr, gamma, beta,
           N, inv_n, params.epsilon, params.use_scale, params.use_shift);
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

// Toolchain too old for __m512h / _ph intrinsics. The dispatch in
// lowoha_normalization.cpp will fall back to the FP32-accumulating
// layer_norm_avx512 in this case; this stub exists only so the symbol
// resolves at link time.

namespace zendnnl {
namespace lowoha {
namespace normalization {

status_t layer_norm_avx512_fp16(
  const void *,
  void *,
  const void *,
  const void *,
  norm_params &
) {
  return status_t::isa_unsupported;
}

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

#endif // __GNUC__ >= 12
