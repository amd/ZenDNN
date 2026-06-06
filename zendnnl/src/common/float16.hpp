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
#ifndef _FLOAT16_HPP_
#define _FLOAT16_HPP_

#include <cstdint>
#include <cmath>
#include <initializer_list>
#include <immintrin.h>

namespace zendnnl {
namespace common {

/** @union fp32fp16_t
 *  @brief Bit-level representation for float32 ↔ float16 conversion.
 *
 *  IEEE 754 half-precision (float16) layout:
 *    - 1 bit  sign
 *    - 5 bits exponent (bias 15)
 *    - 10 bits mantissa
 *
 *  IEEE 754 single-precision (float32) layout:
 *    - 1 bit  sign
 *    - 8 bits exponent (bias 127)
 *    - 23 bits mantissa
 */
union fp32fp16_t {
  /** @brief float constructor */
  fp32fp16_t(float ff) : fp32{ff} {}
  /** @brief uint16_t raw-bits constructor */
  fp32fp16_t(uint16_t hf) : u16{hf} {}

  float    fp32;  /**< float32 value */
  uint16_t u16;   /**< float16 raw bits */
  uint32_t u32;   /**< corresponding u32 value */
};

/** @class float16_t
 *  @brief Implements an IEEE 754 half-precision (float16) type.
 *
 *  Float16 is a numeric type not directly supported in C++. This class
 *  provides this numeric type using uint16_t as internal storage.
 *
 *  Conversion from other numeric types like float32 and integer types
 *  are supported.
 *
 *  Unlike bfloat16 (which shares the float32 exponent range), float16
 *  has a smaller exponent range (5 bits, bias 15) and larger mantissa
 *  (10 bits), requiring exponent rebasing during conversion.
 */
class float16_t {
 public:
  /** @name Constructors, Destructors and Assignment
   */
  /**@{*/
  /** @brief Default constructor, initializes to zero. */
  float16_t();

  /** @brief Conversion constructor from float32 to float16.
   * @param f : float32 value.
   */
  float16_t(float f);

  /** @brief Conversion assignment from float32 to float16.
   * @param f : float32 value.
   * @return A reference to converted float16 value.
   */
  float16_t &operator=(float f);

  /** @brief Conversion constructor from an integer type to float16.
   * @param i : an integer type value.
   */
  template<typename integer_type,
           typename SFINAE = std::enable_if_t<std::is_integral_v<integer_type>>>
               float16_t(integer_type i): float16_t{float(i)} {
  }

  /** @brief Conversion assignment from an integer type to float16.
   * @param i : an integer type value.
   * @return A reference to converted float16 value.
   */
  template<typename integer_type,
           typename SFINAE = std::enable_if_t<std::is_integral_v<integer_type>>>
  float16_t &operator=(integer_type i) {
    return (*this) = float16_t(i);
  }
  /**@}*/

  /** @name Conversion Operators
   */
  /**@{*/
  /** @brief Conversion from float16 to float. */
  operator float() const;

  /** @brief Conversion from float16 to int. */
  operator int()   const;
  /**@}*/

  /** @brief Get the raw 16-bit representation.
   * @return The raw uint16_t bits of this float16 value.
   */
  uint16_t raw() const {
    return raw_bits_;
  }

  /**
   * @brief Convert float16 raw value to float32 value.
   * @param f16_val The float16 value (as uint16_t) to be converted.
   * @return The converted float32 value.
   */
  static float f16_to_f32_val(uint16_t f16_val);

  /**
   * @brief Convert float32 value to float16 value using rounding to
   *        nearest-even.
   * @param val The float32 value to be converted.
   * @return The converted float16 value as uint16_t.
   */
  static uint16_t f32_to_f16_val(float val);

  /**
   * @brief Convert a float16 buffer to float32 buffer.
   * @param f16_buf Pointer to the float16 buffer.
   * @param f32_buf Pointer to the output float32 buffer.
   * @param size_ Size of the buffer.
   */
  static void f16_to_f32_buf(const uint16_t *f16_buf, float *f32_buf,
                             int64_t size_);

  /**
   * @brief Convert a float32 buffer to float16 buffer.
   * @param f32_buf Pointer to the float32 buffer.
   * @param f16_buf Pointer to the output float16 buffer.
   * @param size_ Size of the buffer.
   */
  static void f32_to_f16(const float *f32_buf, uint16_t *f16_buf,
                         int64_t size_);

#if defined(__GNUC__) && (__GNUC__ >= 12)
  //===--------------------------------------------------------------------===//
  // SIMD vector conversions (AVX-512 + AVX-512-FP16)
  //
  // These are pure register-to-register conversions: the caller is
  // responsible for the F32 load/store (with full or masked memory
  // access). Because the conversion itself is uniform across all 32
  // lanes, a single API serves both the regular (32-lane) and tail
  // (< 32 lanes) paths in vectorized kernels.
  //
  // The host CPU must support AVX-512 + AVX-512-FP16; the caller is
  // responsible for ISA dispatch before invoking these.
  //===--------------------------------------------------------------------===//

  /**
   * @brief Convert 32 float32 lanes (split as two __m512) to a single
   *        __m512h. Round mode is nearest-even, no exception bits raised.
   * @param lo Lanes 0..15 as float32.
   * @param hi Lanes 16..31 as float32.
   * @return The 32 converted float16 lanes as a __m512h.
   */
  static __m512h cvt_f32_to_f16_vec(__m512 lo, __m512 hi);

  /**
   * @brief Convert a __m512h to 32 float32 lanes split as two __m512.
   * @param val The float16 source vector.
   * @param lo  [out] Lanes 0..15 widened to float32.
   * @param hi  [out] Lanes 16..31 widened to float32.
   */
  static void cvt_f16_to_f32_vec(__m512h val, __m512 &lo, __m512 &hi);
#endif

 private:
  uint16_t raw_bits_; /*!< float16 raw bits (IEEE 754 half-precision) */
};

}//namespace common

namespace interface {
using float16_t = zendnnl::common::float16_t;
}//interface

}//namespace zendnnl

#endif
