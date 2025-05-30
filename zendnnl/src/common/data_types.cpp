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

#include <string>
#include "data_types.hpp"

namespace zendnnl {
namespace common {

uint32_t size_of(data_type_t data_type) {
  switch(data_type) {
  case data_type_t::f32 :
    return sizeof(prec_traits<data_type_t::f32>::type);
  case data_type_t::f16 :
    return sizeof(prec_traits<data_type_t::f16>::type);
  case data_type_t::bf16 :
    return sizeof(prec_traits<data_type_t::bf16>::type);
  case data_type_t::s32 :
    return sizeof(prec_traits<data_type_t::s32>::type);
  case data_type_t::s16 :
    return sizeof(prec_traits<data_type_t::s16>::type);
  case data_type_t::s8 :
    return sizeof(prec_traits<data_type_t::s8>::type);
  case data_type_t::s4 :
    return sizeof(prec_traits<data_type_t::s4>::type);
  case data_type_t::u32 :
    return sizeof(prec_traits<data_type_t::u32>::type);
  case data_type_t::u16 :
    return sizeof(prec_traits<data_type_t::u16>::type);
  case data_type_t::u8 :
    return sizeof(prec_traits<data_type_t::u8>::type);
  case data_type_t::u4 :
    return sizeof(prec_traits<data_type_t::u4>::type);
  }

  return 0;
}

std::string dtype_info(data_type_t data_type) {
switch(data_type) {
  case data_type_t::f32 :
    return "f32";
  case data_type_t::f16 :
    return "f16";
  case data_type_t::bf16 :
    return "bf16";
  case data_type_t::s32 :
    return "s32";
  case data_type_t::s16 :
    return "s16";
  case data_type_t::s8 :
    return "s8";
  case data_type_t::s4 :
    return "s4";
  case data_type_t::u32 :
    return "u32";
  case data_type_t::u16 :
    return "u16";
  case data_type_t::u8 :
    return "u8";
  case data_type_t::u4 :
    return "u4";
  }

  return 0;
}

} //common
} //zendnnl
