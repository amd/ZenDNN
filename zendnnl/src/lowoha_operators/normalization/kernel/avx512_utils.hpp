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

#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,avx512bf16,fma")

namespace zendnnl {
namespace lowoha {
namespace normalization {
namespace avx512 {

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
// Dtype-aware load / store helpers
//
// These wrap FP32 and BF16 intrinsics behind a runtime bool.  The bool is
// loop-invariant so the branch predictor locks on after the first iteration,
// making the branch effectively free.  This avoids maintaining separate
// kernel specializations for each {src, dst} × {FP32, BF16} combination.
//
// load16   / load16_mask   — load 16 FP32 lanes from FP32 or BF16 memory
// store16  / store16_mask  — store 16 FP32 lanes to FP32 or BF16 memory
// =============================================================================

inline __m512 load16(const void *p, bool bf16) {
  if (bf16) {
    return bf16x16_to_fp32(static_cast<const int16_t *>(p));
  }
  return _mm512_loadu_ps(static_cast<const float *>(p));
}

inline __m512 load16_mask(const void *p, __mmask16 m, bool bf16) {
  if (bf16) {
    return bf16x16_to_fp32_mask(static_cast<const int16_t *>(p), m);
  }
  return _mm512_maskz_loadu_ps(m, static_cast<const float *>(p));
}

inline void store16(void *p, __m512 v, bool bf16) {
  if (bf16) {
    store_fp32_as_bf16(static_cast<int16_t *>(p), v);
  }
  else {
    _mm512_storeu_ps(static_cast<float *>(p), v);
  }
}

inline void store16_mask(void *p, __m512 v, __mmask16 m, bool bf16) {
  if (bf16) {
    store_fp32_as_bf16_mask(static_cast<int16_t *>(p), v, m);
  }
  else {
    _mm512_mask_storeu_ps(static_cast<float *>(p), m, v);
  }
}

// =============================================================================
// Element-size helper
// =============================================================================

inline size_t elem_size(bool bf16) {
  return bf16 ? 2 : 4;
}

} // namespace avx512
} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_NORMALIZATION_AVX512_UTILS_HPP
