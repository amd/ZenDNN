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

#ifndef _LOWOHA_NORMALIZATION_AVX512_UTILS_HPP
#define _LOWOHA_NORMALIZATION_AVX512_UTILS_HPP

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include "common/data_types.hpp"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,avx512bf16,fma")
#endif

namespace zendnnl {
namespace lowoha {
namespace normalization {
namespace avx512 {

using zendnnl::common::data_type_t;

// =============================================================================
// BF16 ↔ FP32 conversion helpers
// =============================================================================

inline __m512 bf16x16_to_fp32(const int16_t *ptr) {
  __m256i bf16 = _mm256_loadu_si256((const __m256i *)ptr);
  return _mm512_castsi512_ps(
           _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16), 16));
}

inline __m512 bf16x16_to_fp32_mask(const int16_t *ptr, __mmask16 mask) {
  __m256i bf16 = _mm256_maskz_loadu_epi16(mask, ptr);
  return _mm512_castsi512_ps(
           _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16), 16));
}

inline void store_fp32_as_bf16(int16_t *ptr, __m512 val) {
  _mm256_storeu_si256((__m256i *)ptr, (__m256i)_mm512_cvtneps_pbh(val));
}

inline void store_fp32_as_bf16_mask(int16_t *ptr, __m512 val,
                                    __mmask16 mask) {
  _mm256_mask_storeu_epi16(ptr, mask, (__m256i)_mm512_cvtneps_pbh(val));
}

// =============================================================================
// FP16 ↔ FP32 conversion helpers (F16C / AVX-512)
//
// Uses _mm512_cvtph_ps for F16 → FP32 (16 lanes per call) and _mm512_cvtps_ph
// for FP32 → F16 with round-to-nearest-even, no exceptions. The 16-lane AVX-512
// cvtph_ps is available on every AVX-512 CPU; AVX512-FP16 is NOT required for
// this F32-accumulation path.
// =============================================================================

inline __m512 f16x16_to_fp32(const uint16_t *ptr) {
  return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)ptr));
}

inline __m512 f16x16_to_fp32_mask(const uint16_t *ptr, __mmask16 mask) {
  return _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, ptr));
}

inline void store_fp32_as_f16(uint16_t *ptr, __m512 val) {
  _mm256_storeu_si256((__m256i *)ptr,
                      _mm512_cvtps_ph(val,
                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

inline void store_fp32_as_f16_mask(uint16_t *ptr, __m512 val,
                                   __mmask16 mask) {
  _mm256_mask_storeu_epi16(ptr, mask,
                           _mm512_cvtps_ph(val,
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// =============================================================================
// Dtype-aware load / store helpers
//
// These wrap FP32, BF16, and F16 intrinsics behind a runtime data_type_t. The
// dtype is loop-invariant so the branch predictor locks on after the first
// iteration, making the dispatch effectively free. This avoids maintaining
// separate kernel specializations for each {src, dst} × {FP32, BF16, F16}
// combination.
//
// load16   / load16_mask   — load 16 FP32 lanes from FP32, BF16, or F16 memory
// store16  / store16_mask  — store 16 FP32 lanes to FP32, BF16, or F16 memory
// =============================================================================

inline __m512 load16(const void *p, data_type_t dt) {
  switch (dt) {
  case data_type_t::bf16:
    return bf16x16_to_fp32(static_cast<const int16_t *>(p));
  case data_type_t::f16:
    return f16x16_to_fp32(static_cast<const uint16_t *>(p));
  default:
    return _mm512_loadu_ps(static_cast<const float *>(p));
  }
}

inline __m512 load16_mask(const void *p, __mmask16 m, data_type_t dt) {
  switch (dt) {
  case data_type_t::bf16:
    return bf16x16_to_fp32_mask(static_cast<const int16_t *>(p), m);
  case data_type_t::f16:
    return f16x16_to_fp32_mask(static_cast<const uint16_t *>(p), m);
  default:
    return _mm512_maskz_loadu_ps(m, static_cast<const float *>(p));
  }
}

inline void store16(void *p, __m512 v, data_type_t dt) {
  switch (dt) {
  case data_type_t::bf16:
    store_fp32_as_bf16(static_cast<int16_t *>(p), v);
    break;
  case data_type_t::f16:
    store_fp32_as_f16(static_cast<uint16_t *>(p), v);
    break;
  default:
    _mm512_storeu_ps(static_cast<float *>(p), v);
    break;
  }
}

inline void store16_mask(void *p, __m512 v, __mmask16 m, data_type_t dt) {
  switch (dt) {
  case data_type_t::bf16:
    store_fp32_as_bf16_mask(static_cast<int16_t *>(p), v, m);
    break;
  case data_type_t::f16:
    store_fp32_as_f16_mask(static_cast<uint16_t *>(p), v, m);
    break;
  default:
    _mm512_mask_storeu_ps(static_cast<float *>(p), m, v);
    break;
  }
}

// =============================================================================
// Element-size helper
//
// Returns the size in bytes of one element of the given dtype:
//   bf16, f16 -> 2 bytes
//   anything else (including f32) -> 4 bytes
// =============================================================================

inline size_t elem_size(data_type_t dt) {
  switch (dt) {
  case data_type_t::bf16:
  case data_type_t::f16:
    return 2;
  default:
    return 4;
  }
}

} // namespace avx512
} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_NORMALIZATION_AVX512_UTILS_HPP
