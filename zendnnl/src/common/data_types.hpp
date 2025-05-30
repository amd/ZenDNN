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
#ifndef _DATA_TYPES_HPP_
#define _DATA_TYPES_HPP_

#include <cstdint>
#include <string>
#include "bfloat16.hpp"

/** @namespace zendnnl
 *  @brief ZenDNNL top level namespace.
 */
namespace zendnnl {
/** @namespace zendnnl::common
 *  @brief A namespace to contain classes, functions, variable and enumerations
 *  common to all ZenDNNL.
 */
namespace common {
/** @enum data_type_t
 *  @brief data types supported by ZenDNNL
 *
 * Data type refers to quantization level or precisison with which tensor data is
 * represented.
 *
 * It is tempting to think of data type as a template parameter to tensor_t class,
 * but data type is needed in many runtime checks, for example, kernel selection.
 * Also, with templated data type, quantization/dequantization routines will become
 * more cumbersome.
 */
enum class data_type_t : uint8_t {
  f32,  /*!< float 32bit */
  f16,  /*!< float 16bit */
  bf16, /*!< brain float 16bit */
  s32,  /*!< signed integer 32 bit */
  s16,  /*!< signed integer 16 bit */
  s8,   /*!< signed integer 8 bit */
  s4,   /*!< signed integer 4 bit */
  u32,  /*!< unsigned integer 32 bit */
  u16,  /*!< unsigned integer 16 bit */
  u8,   /*!< unsigned integer 8 bit */
  u4    /*!< unsigned integer 4 bit */
};

/** @brief conversion from data_type_t to corresponding C++ type */
template <data_type_t>
struct prec_traits {};

/** @brief f32 to float */
template <>
struct prec_traits<data_type_t::f32> {
    typedef float type;
};

/** @brief f16 to float */
template <>
struct prec_traits<data_type_t::f16> {
    typedef float type;
};

/** @brief bf16 to bfloat16_t */
template <>
struct prec_traits<data_type_t::bf16> {
    typedef bfloat16_t type;
};

/** @brief s32 to int32_t */
template <>
struct prec_traits<data_type_t::s32> {
    typedef int32_t type;
};

/** @brief s16 to int16_t */
template <>
struct prec_traits<data_type_t::s16> {
    typedef int16_t type;
};

/** @brief s8 to int8_t */
template <>
struct prec_traits<data_type_t::s8> {
    typedef int8_t type;
};

/** @brief s4 to uint8_t */
template <>
struct prec_traits<data_type_t::s4> {
    typedef uint8_t type;
};

/** @brief u32 to uint32_t */
template <>
struct prec_traits<data_type_t::u32> {
    typedef uint32_t type;
};

/** @brief u16 to uint16_t */
template <>
struct prec_traits<data_type_t::u16> {
    typedef uint16_t type;
};

/** @brief u8 to uint8_t */
template <>
struct prec_traits<data_type_t::u8> {
    typedef uint8_t type;
};

/** @brief u4 to uint8_t */
template <>
struct prec_traits<data_type_t::u4> {
    typedef uint8_t type;
};

/** @brief Conversion from C++ types to data_type_t */
template <typename>
struct data_traits {};

/** @brief float to f32 */
template <>
struct data_traits<float> {
    static constexpr data_type_t data_type = data_type_t::f32;
};

// template <>
// struct data_traits<float16_t> {
//     static constexpr data_type_t data_type = data_type_t::f16;
// };

/** @brief bfloat16_t to bf16 */
template <>
struct data_traits<bfloat16_t> {
    static constexpr data_type_t data_type = data_type_t::bf16;
};

/** @brief int32_t to s32 */
template <>
struct data_traits<int32_t> {
    static constexpr data_type_t data_type = data_type_t::s32;
};

/** @brief int16_t to s16 */
template <>
struct data_traits<int16_t> {
    static constexpr data_type_t data_type = data_type_t::s16;
};

/** @brief int8_t to s8 */
template <>
struct data_traits<int8_t> {
    static constexpr data_type_t data_type = data_type_t::s8;
};

/** @brief uint32_t to u32 */
template <>
struct data_traits<uint32_t> {
    static constexpr data_type_t data_type = data_type_t::u32;
};

/** @brief uint16_t to u16 */
template <>
struct data_traits<uint16_t> {
    static constexpr data_type_t data_type = data_type_t::u16;
};

/** @brief uint8_t to u8 */
template <>
struct data_traits<uint8_t> {
    static constexpr data_type_t data_type = data_type_t::u8;
};

// template <>
// struct data_traits<int4_t> {
//     static constexpr data_type_t data_type = data_type_t::s4;
// };
// template <>
// struct data_traits<uint4_t> {
//     static constexpr data_type_t data_type = data_type_t::u4;
// };

/** @brief Get size of a data type
 *  @param data_type : the data type
 *  @return Size of the data type
 */
uint32_t size_of(data_type_t data_type);

/** @brief Get name of the data type
 *  @param data_type : the data type
 *  @return Name of the data type
 */
std::string dtype_info(data_type_t data_type);

}//common

/** @namespace zendnnl::interface
 *  @brief A namespace that provides classes, functions, variables and enums to
 *  interface with external code.
 *
 *  This namespace exports all classes, functions, variables and enums that an
 *  external code will need to interface with ZenDNNL. Though an external code
 *  can refer to other namespaces, by convension other namespaces are internal to
 *  ZenDNNL and external code should refer to only zendnnl::interface namespace.
 */
namespace interface {
using data_type_t = zendnnl::common::data_type_t;
} //export

}//zendnnl


#endif
