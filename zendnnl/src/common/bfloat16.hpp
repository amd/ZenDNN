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

namespace zendnnl {
namespace common {

/** @union fp32bf16_t
 *  @brief Conversion from float32 to bfloat16 and vice-versa.
 */
union fp32bf16_t {
  /** @brief float constructor */
  fp32bf16_t(float ff):fp32{ff}{}
  /** @brief bf16 constructor */
  fp32bf16_t(uint16_t hf):bf16{0,hf}{}

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
  bfloat16_t& operator=(float f);

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
  bfloat16_t& operator=(integer_type i) {
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

private:
  uint16_t raw_bits_; /*!< bfloat16 raw bits */
};

}//namespace common

namespace interface {
using bfloat16_t = zendnnl::common::bfloat16_t;
}//interface

}//namespace zendnnl

#endif
