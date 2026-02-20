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

#include "float16.hpp"
#include <cstring>
#include <immintrin.h>

namespace zendnnl {
namespace common {

// IEEE 754 half-precision (float16) constants
// Layout: 1 sign | 5 exponent (bias 15) | 10 mantissa
static constexpr uint16_t F16_SIGN_MASK     = 0x8000;
static constexpr uint16_t F16_EXP_MASK      = 0x7C00;
static constexpr uint16_t F16_MANT_MASK     = 0x03FF;
static constexpr int      F16_EXP_BIAS      = 15;
static constexpr int      F16_EXP_MAX       = 31;     // all-ones exponent
static constexpr int      F16_MANT_BITS     = 10;

// IEEE 754 single-precision (float32) constants
static constexpr uint32_t F32_SIGN_MASK     = 0x80000000u;
static constexpr uint32_t F32_EXP_MASK      = 0x7F800000u;
static constexpr uint32_t F32_MANT_MASK     = 0x007FFFFFu;
static constexpr int      F32_EXP_BIAS      = 127;
static constexpr int      F32_MANT_BITS     = 23;

// Derived constants
static constexpr int      MANT_SHIFT        = F32_MANT_BITS -
    F16_MANT_BITS; // 13
static constexpr int      EXP_REBIAS        = F32_EXP_BIAS -
    F16_EXP_BIAS;  // 112

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

float16_t::float16_t() : raw_bits_{0} {
}

float16_t::float16_t(float f) : raw_bits_{0} {
  raw_bits_ = f32_to_f16_val(f);
}

float16_t &float16_t::operator=(float f) {
  return (*this) = float16_t(f);
}

//===----------------------------------------------------------------------===//
// Conversion Operators
//===----------------------------------------------------------------------===//

float16_t::operator float() const {
  return f16_to_f32_val(raw_bits_);
}

float16_t::operator int() const {
  return int(f16_to_f32_val(raw_bits_));
}

//===----------------------------------------------------------------------===//
// Scalar Conversions
//===----------------------------------------------------------------------===//

float float16_t::f16_to_f32_val(uint16_t f16_val) {
  const uint16_t sign = (f16_val & F16_SIGN_MASK);
  const uint16_t exp  = (f16_val & F16_EXP_MASK) >> F16_MANT_BITS;
  const uint16_t mant = (f16_val & F16_MANT_MASK);

  uint32_t f32_bits = 0;

  if (exp == 0) {
    if (mant == 0) {
      // +-Zero: preserve sign
      f32_bits = static_cast<uint32_t>(sign) << 16;
    }
    else {
      // Subnormal: value = (-1)^sign * 2^(-14) * (mant / 1024)
      // Normalize by shifting mantissa until the implicit 1 appears
      float value = static_cast<float>(mant) / (1 << F16_MANT_BITS);
      value *= (1.0f / (1 << (F16_EXP_BIAS - 1))); // * 2^(-14)
      if (sign) {
        value = -value;
      }
      std::memcpy(&f32_bits, &value, sizeof(float));
    }
  }
  else if (exp == F16_EXP_MAX) {
    // Infinity or NaN
    f32_bits = (static_cast<uint32_t>(sign) << 16)
               | F32_EXP_MASK
               | (static_cast<uint32_t>(mant) << MANT_SHIFT);
    if (mant != 0) {
      // Force quiet NaN (set MSB of mantissa)
      f32_bits |= (1u << (F32_MANT_BITS - 1));
    }
  }
  else {
    // Normal number: rebias exponent
    uint32_t f32_exp  = static_cast<uint32_t>(exp) + EXP_REBIAS;
    uint32_t f32_mant = static_cast<uint32_t>(mant) << MANT_SHIFT;
    f32_bits = (static_cast<uint32_t>(sign) << 16)
               | (f32_exp << F32_MANT_BITS)
               | f32_mant;
  }

  float result;
  std::memcpy(&result, &f32_bits, sizeof(float));
  return result;
}

uint16_t float16_t::f32_to_f16_val(float val) {
  uint32_t f32_bits;
  std::memcpy(&f32_bits, &val, sizeof(float));

  const uint32_t sign = (f32_bits & F32_SIGN_MASK) >> 16; // move to bit 15
  const uint32_t exp  = (f32_bits & F32_EXP_MASK) >> F32_MANT_BITS;
  const uint32_t mant = (f32_bits & F32_MANT_MASK);

  uint16_t f16_bits = 0;

  if (exp == 0) {
    // f32 zero or subnormal → f16 zero (preserve sign)
    f16_bits = static_cast<uint16_t>(sign);
  }
  else if (exp == 255) {
    // f32 Infinity or NaN
    f16_bits = static_cast<uint16_t>(sign) | F16_EXP_MASK;
    if (mant != 0) {
      // NaN: truncate mantissa and force quiet NaN
      uint16_t f16_mant = static_cast<uint16_t>(mant >> MANT_SHIFT);
      f16_bits |= f16_mant;
      f16_bits |= (1 << (F16_MANT_BITS - 1)); // force QNAN
    }
  }
  else {
    // Normal number: rebias exponent
    int32_t new_exp = static_cast<int32_t>(exp) - EXP_REBIAS;

    if (new_exp >= F16_EXP_MAX) {
      // Overflow → Infinity
      f16_bits = static_cast<uint16_t>(sign) | F16_EXP_MASK;
    }
    else if (new_exp <= 0) {
      if (new_exp < -F16_MANT_BITS) {
        // Too small even for subnormal → zero
        f16_bits = static_cast<uint16_t>(sign);
      }
      else {
        // Subnormal: shift mantissa right, include implicit 1 bit
        uint32_t full_mant = mant | (1u << F32_MANT_BITS); // add implicit 1
        int shift = MANT_SHIFT + 1 - new_exp; // total right shift

        // Round-to-nearest-even
        uint32_t round_bit = 1u << (shift - 1);
        uint32_t sticky    = (full_mant & (round_bit - 1)) ? 1u : 0u;
        uint32_t shifted   = full_mant >> shift;

        if ((full_mant & round_bit) && (sticky || (shifted & 1))) {
          shifted += 1;
        }

        f16_bits = static_cast<uint16_t>(sign)
                   | static_cast<uint16_t>(shifted & F16_MANT_MASK);
      }
    }
    else {
      // Normal range: round-to-nearest-even
      uint32_t round_bit = 1u << (MANT_SHIFT - 1);
      uint32_t sticky    = (mant & (round_bit - 1)) ? 1u : 0u;
      uint32_t truncated = mant >> MANT_SHIFT;

      if ((mant & round_bit) && (sticky || (truncated & 1))) {
        truncated += 1;
        if (truncated > F16_MANT_MASK) {
          // Mantissa overflow: increment exponent
          truncated = 0;
          new_exp += 1;
          if (new_exp >= F16_EXP_MAX) {
            // Overflow to Infinity after rounding
            f16_bits = static_cast<uint16_t>(sign) | F16_EXP_MASK;
            return f16_bits;
          }
        }
      }

      f16_bits = static_cast<uint16_t>(sign)
                 | static_cast<uint16_t>(new_exp << F16_MANT_BITS)
                 | static_cast<uint16_t>(truncated & F16_MANT_MASK);
    }
  }

  return f16_bits;
}

void float16_t::f16_to_f32_buf(const uint16_t *f16_buf, float *f32_buf,
                               int64_t size_) {
  for (int64_t j = 0; j < size_; ++j) {
    f32_buf[j] = float16_t::f16_to_f32_val(f16_buf[j]);
  }
}

}//namespace common
}//namespace zendnnl
