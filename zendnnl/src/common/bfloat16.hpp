/********************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _BFLOAT16_HPP_
#define _BFLOAT16_HPP_

#include <cstdint>
#include <cmath>
#include <initializer_list>
#include <immintrin.h>

namespace zendnnl {
namespace common {

/** @union fp32bf16_t
 *  @brief Conversion from float32 to bfloat16 and vice-versa.
 */
union fp32bf16_t {
  /** @brief float constructor */
  fp32bf16_t(float ff):fp32{ff} {}
  /** @brief bf16 constructor */
  fp32bf16_t(uint16_t hf):bf16{0,hf} {}

  float       fp32;      /**< float value */
  uint16_t    bf16[2];   /**< equivalent bf16 value */
  uint32_t    u32;       /**< corresponding u32 value */
};

/** @class bfloat16_t
 *  @brief Implements a bfloat16 type
 *
 *  Bfloat16 type is a numeric type not directly supported in C++. This class
 *  provides this numeric type.
 *
 *  Conversion from other numeric types like float32 and integer type ae supported.
 *
 *  @todo Add support for basic arithmetic and comparison operators.
 */
class bfloat16_t {
 public:
  /** @name Constructors, Destructors and Assignment
   */
  /**@{*/
  /** @brief Default constructor, initializes to zero. */
  bfloat16_t();

  /** @brief Convertion constructor from float32 to bfloat16.
   * @param f : float32 value.
   */
  bfloat16_t(float f);

  /** @brief Convertion assignment from float32 to bfloat16.
   * @param f : float32 value.
   * @return A reference to converted bfloat16 value.
   */
  bfloat16_t &operator=(float f);

  /** @brief Convertion constructor from an integer type to bfloat16.
   * @param i : an integer type value.
   */
  template<typename integer_type,
           typename SFINAE = std::enable_if_t<std::is_integral_v<integer_type>>>
             bfloat16_t(integer_type i): bfloat16_t{float(i)} {
  }

  /** @brief Convertion assignment from an integer type to bfloat16.
   * @param i : an integer type value.
   * @return A reference to converted bfloat16 value.
   */
  template<typename integer_type,
           typename SFINAE = std::enable_if_t<std::is_integral_v<integer_type>>>
  bfloat16_t &operator=(integer_type i) {
    return (*this) = bfloat16_t(i);
  }
  /**@}*/

  /** @name Conversion Operators
   */
  /**@{*/
  /** @brief Conversion from bfloat16 to float. */
  operator float() const;

  /** @brief Conversion from bfloat16 to int. */
  operator int()   const;
  /**@}*/

  /**
   * @brief Convert BF16 value to float32 value using rounding to nearest-even.
   * @param bf16_val The BF16 value to be converted.
   * @return The converted float32 value.
   */
  static float bf16_to_f32_val(int16_t bf16_val);

  /**
   * @brief Convert float32 value to bf16 value using rounding to nearest-even.
   * @param val The float32 value to be converted.
   * @return The converted bf16 value.
   */
  static int16_t f32_to_bf16_val(float val);

  /**
   * @brief Convert 16 float32 values to 16 BF16 values using AVX512 instructions.
   * @param val The 16 float32 values packed in an AVX512 register.
   * @return The converted 16 BF16 values packed in an AVX512 register.
   */
  static __m256i f32_to_bf16_avx512(__m512 val);

  /**
   * @brief Convert an array of float32 values to BF16 values with rounding.
   * @param input Pointer to the input array of float32 values.
   * @param output Pointer to the output array of BF16 values.
   * @param count Number of elements to convert.
   */
  static void f32_to_bf16(const float *input, int16_t *output,
                               size_t count);

  /**
   * @brief Convert a BF16 buffer to float32.
   * @param bf16_buf Pointer to the BF16 buffer.
   * @param f32_buf Pointer to the output float32 buffer.
   * @param size size of the buffer.
   */
  static void bf16_to_f32_buf(const uint16_t *bf16_buf, float *f32_buf,
                              int64_t size_);

 private:
  uint16_t raw_bits_; /*!< bfloat16 raw bits */
};

}//namespace common

namespace interface {
using bfloat16_t = zendnnl::common::bfloat16_t;
}//interface

}//namespace zendnnl

#endif
