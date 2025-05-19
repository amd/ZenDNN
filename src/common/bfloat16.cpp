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

#include "bfloat16.hpp"

namespace zendnnl {
namespace common {

//implementation
bfloat16_t::bfloat16_t():raw_bits_{0} {
}

bfloat16_t::bfloat16_t(float f) {
  switch (std::fpclassify(f)) {
  case FP_SUBNORMAL:
  case FP_ZERO:
    // sign preserving zero (denormal go to zero)
    raw_bits_  = fp32bf16_t{f}.bf16[1];
    raw_bits_ &= 0x8000;
    break;
  case FP_INFINITE:
    raw_bits_ = fp32bf16_t{f}.bf16[1];
    break;
  case FP_NAN:
    // truncate and set MSB of the mantissa force QNAN
    raw_bits_  = fp32bf16_t{f}.bf16[1];;
    raw_bits_ |= (0x01 << 6);
    break;
  case FP_NORMAL:
    // round to nearest even and truncate
    fp32bf16_t  rbits{f};
    const uint32_t rounding_bias = 0x00007FFF + (rbits.bf16[1] & 0x1);
    rbits.u32                   += rounding_bias;
    raw_bits_                    = rbits.bf16[1];
    break;
  }
}

bfloat16_t::operator float() const {
  return fp32bf16_t{raw_bits_}.fp32;
}

bfloat16_t::operator int() const {
  return int(fp32bf16_t{raw_bits_}.fp32);
}

bfloat16_t& bfloat16_t::operator=(float f) {
  return (*this) = bfloat16_t(f);
}

}//namespace common
}//namespace zendnnl

