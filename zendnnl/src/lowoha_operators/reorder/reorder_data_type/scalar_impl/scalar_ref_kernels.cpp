/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "lowoha_operators/reorder/reorder_data_type/scalar_impl/scalar_kernels.hpp"

#include <cstring>
#include <cmath>

namespace zendnnl {
namespace lowoha {
namespace reorder {

void quantize_bf16_to_int8_ref(const uint16_t *input, int8_t *output,
                                size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Convert BF16 to float32
    uint32_t bits = static_cast<uint32_t>(input[i]) << 16;
    float val;
    std::memcpy(&val, &bits, sizeof(float));

    // Apply quantization: (val / scale) + zero_point (use nearbyint for consistent rounding)
    int32_t q = static_cast<int32_t>(std::nearbyint(val / scale)) + zero_point;
    q = std::max(-128, std::min(127, q));
    output[i] = static_cast<int8_t>(q);
  }
}

void dequantize_int8_to_bf16_ref(const int8_t *input, uint16_t *output,
                                  size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Dequantize: (x - zp) * scale
    float val = (static_cast<float>(input[i]) - zero_point) * scale;

    // Convert float32 to BF16 with rounding
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(float));
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    output[i] = static_cast<uint16_t>(bits >> 16);
  }
}
void quantize_bf16_to_uint8_ref(const uint16_t *input, uint8_t *output,
                                 size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Convert BF16 to float32
    uint32_t bits = static_cast<uint32_t>(input[i]) << 16;
    float val;
    std::memcpy(&val, &bits, sizeof(float));

    // Apply quantization: (val / scale) + zero_point
    int32_t q = static_cast<int32_t>(std::nearbyint(val / scale)) + zero_point;
    q = std::max(0, std::min(255, q));
    output[i] = static_cast<uint8_t>(q);
  }
}

void dequantize_uint8_to_bf16_ref(const uint8_t *input, uint16_t *output,
                                   size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Dequantize: (x - zp) * scale
    float val = (static_cast<float>(input[i]) - zero_point) * scale;

    // Convert float32 to BF16 with rounding
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(float));
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    output[i] = static_cast<uint16_t>(bits >> 16);
  }
}
void quantize_f32_to_int8_ref(const float *input, int8_t *output,
                               size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Apply quantization: (val / scale) + zero_point (use nearbyint for consistent rounding)
    int32_t q = static_cast<int32_t>(std::nearbyint(input[i] / scale)) + zero_point;
    q = std::max(-128, std::min(127, q));
    output[i] = static_cast<int8_t>(q);
  }
}

void dequantize_int8_to_f32_ref(const int8_t *input, float *output,
                                 size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Dequantize: (x - zp) * scale
    output[i] = (static_cast<float>(input[i]) - zero_point) * scale;
  }
}
void quantize_f32_to_uint8_ref(const float *input, uint8_t *output,
                                size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Apply quantization: (val / scale) + zero_point
    int32_t q = static_cast<int32_t>(std::nearbyint(input[i] / scale)) + zero_point;
    q = std::max(0, std::min(255, q));
    output[i] = static_cast<uint8_t>(q);
  }
}

void dequantize_uint8_to_f32_ref(const uint8_t *input, float *output,
                                  size_t nelems, float scale, int zero_point) {
  for (size_t i = 0; i < nelems; ++i) {
    // Dequantize: (x - zp) * scale
    output[i] = (static_cast<float>(input[i]) - zero_point) * scale;
  }
}
void convert_f32_to_bf16_ref(const float *input, uint16_t *output,
                              size_t nelems, float scale, int zero_point) {
  bool apply_scaling = (scale != 1.0f || zero_point != 0);
  
  for (size_t i = 0; i < nelems; ++i) {
    float val = input[i];
    if (apply_scaling) {
      val = val / scale + static_cast<float>(zero_point);
    }
    // Convert to BF16 using bit manipulation with rounding
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
    bits += rounding;
    output[i] = static_cast<uint16_t>(bits >> 16);
  }
}

void convert_bf16_to_f32_ref(const uint16_t *input, float *output,
                              size_t nelems, float scale, int zero_point) {
  bool apply_scaling = (scale != 1.0f || zero_point != 0);
  
  for (size_t i = 0; i < nelems; ++i) {
    // Convert BF16 to float32
    uint32_t bits = static_cast<uint32_t>(input[i]) << 16;
    float val;
    std::memcpy(&val, &bits, sizeof(val));
    
    if (apply_scaling) {
      val = (val - static_cast<float>(zero_point)) * scale;
    }
    output[i] = val;
  }
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
