/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_reorder_example.hpp"

#include <cstring>
#include <cmath>
#include <iomanip>
#include <chrono>

namespace zendnnl {
namespace examples {

// Helper function to convert float32 to bf16 (as uint16_t)
static inline uint16_t float_to_bf16(float val) {
  uint32_t bits;
  std::memcpy(&bits, &val, sizeof(float));
  // Round-to-nearest-even
  uint32_t lsb = (bits >> 16) & 1;
  uint32_t rounding_bias = 0x7FFF + lsb;
  bits += rounding_bias;
  return static_cast<uint16_t>(bits >> 16);
}

// Helper function to convert bf16 (as uint16_t) to float32
static inline float bf16_to_float(uint16_t val) {
  uint32_t bits = static_cast<uint32_t>(val) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(float));
  return result;
}

int run_lowoha_reorder_bf16_to_int8_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: BF16 to INT8 Quantization");
    log_info("========================================");

    // Test parameters
    constexpr size_t nelems = 16;
    float scale = 0.5f;
    int32_t zero_point = 0;

    // Create BF16 input data
    std::vector<uint16_t> input_bf16(nelems);
    std::vector<float> input_f32_ref = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Input values (float32): ");
    for (size_t i = 0; i < nelems; ++i) {
      input_bf16[i] = float_to_bf16(input_f32_ref[i]);
      // std::cout << std::setw(6) << input_f32_ref[i] << " ";
      // if ((i + 1) % 8 == 0) std::cout << std::endl;
    }

    // Output buffer
    std::vector<int8_t> output_int8(nelems, 0);

    // Setup LOWOHA reorder parameters
    lowoha_reorder_params_t params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.dst = data_type_t::s8;
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;

    log_info("Quantization parameters: scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: int8_val = clamp(round(bf16_val / scale) + zero_point, -128, 127)");

    // Execute reorder
    status_t status = reorder_direct(input_bf16.data(), output_int8.data(), nelems, params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // // Print results
    // log_info("Output INT8 values: ");
    // for (size_t i = 0; i < nelems; ++i) {
    //   std::cout << std::setw(5) << static_cast<int>(output_int8[i]) << " ";
    //   if ((i + 1) % 8 == 0) std::cout << std::endl;
    // }

    // Verify results
    log_info("Verification (expected vs actual):");
    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      int32_t expected = static_cast<int32_t>(std::round(input_f32_ref[i] / scale) + zero_point);
      expected = std::max(-128, std::min(127, expected));
      if (output_int8[i] != static_cast<int8_t>(expected)) {
        log_error("Mismatch at index ", i, ": expected ", expected,
                  ", got ", static_cast<int>(output_int8[i]));
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("BF16 to INT8 quantization test PASSED!");
    } else {
      log_error("BF16 to INT8 quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_int8_to_bf16_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: INT8 to BF16 Dequantization");
    log_info("========================================");

    // Test parameters
    constexpr size_t nelems = 16;
    float scale = 0.5f;
    int32_t zero_point = 0;

    // Create INT8 input data
    std::vector<int8_t> input_int8 = {
      -4, -3, -2, -1,
       0,  1,  2,  3,
       4,  5,  6,  7,
       8,  9, 10, 11
    };

    // log_info("Input INT8 values: ");
    // for (size_t i = 0; i < nelems; ++i) {
    //   std::cout << std::setw(5) << static_cast<int>(input_int8[i]) << " ";
    //   if ((i + 1) % 8 == 0) std::cout << std::endl;
    // }

    // Output buffer
    std::vector<uint16_t> output_bf16(nelems, 0);

    // Setup LOWOHA reorder parameters
    lowoha_reorder_params_t params;
    params.dtypes.src = data_type_t::s8;
    params.dtypes.dst = data_type_t::bf16;
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;

    log_info("Dequantization parameters: scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: bf16_val = (int8_val - zero_point) * scale");

    // Execute reorder
    status_t status = reorder_direct(input_int8.data(), output_bf16.data(), nelems, params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Print results
    log_info("Output BF16 values (as float32): ");
    for (size_t i = 0; i < nelems; ++i) {
      //float val = bf16_to_float(output_bf16[i]);
      // std::cout << std::setw(6) << val << " ";
      // if ((i + 1) % 8 == 0) std::cout << std::endl;
    }

    // Verify results
    log_info("Verification (expected vs actual):");
    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      float expected = (static_cast<float>(input_int8[i]) - zero_point) * scale;
      float actual = bf16_to_float(output_bf16[i]);
      // Allow small tolerance due to bf16 precision
      if (std::abs(actual - expected) > 0.0001f) {
        log_error("Mismatch at index ", i, ": expected ", expected, ", got ", actual);
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("INT8 to BF16 dequantization test PASSED!");
    } else {
      log_error("INT8 to BF16 dequantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_bf16_to_uint8_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: BF16 to UINT8 Quantization");
    log_info("========================================");

    // Test parameters
    constexpr size_t nelems = 16;
    float scale = 0.5f;
    int32_t zero_point = 128;  // Typical zero-point for uint8

    // Create BF16 input data
    std::vector<uint16_t> input_bf16(nelems);
    std::vector<float> input_f32_ref = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Input values (float32): ");
    for (size_t i = 0; i < nelems; ++i) {
      input_bf16[i] = float_to_bf16(input_f32_ref[i]);
    }

    // Output buffer
    std::vector<uint8_t> output_uint8(nelems, 0);

    // Setup LOWOHA reorder parameters
    lowoha_reorder_params_t params;
    params.dtypes.src = data_type_t::bf16;
    params.dtypes.dst = data_type_t::u8;
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;

    log_info("Quantization parameters: scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: uint8_val = clamp(round(bf16_val / scale) + zero_point, 0, 255)");

    // Execute reorder
    status_t status = reorder_direct(input_bf16.data(), output_uint8.data(), nelems, params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results
    log_info("Verification (expected vs actual):");
    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      int32_t expected = static_cast<int32_t>(std::round(input_f32_ref[i] / scale) + zero_point);
      expected = std::max(0, std::min(255, expected));
      if (output_uint8[i] != static_cast<uint8_t>(expected)) {
        log_error("Mismatch at index ", i, ": expected ", expected,
                  ", got ", static_cast<int>(output_uint8[i]));
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("BF16 to UINT8 quantization test PASSED!");
    } else {
      log_error("BF16 to UINT8 quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_uint8_to_bf16_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: UINT8 to BF16 Dequantization");
    log_info("========================================");

    // Test parameters
    constexpr size_t nelems = 16;
    float scale = 0.5f;
    int32_t zero_point = 128;  // Typical zero-point for uint8

    // Create UINT8 input data
    std::vector<uint8_t> input_uint8 = {
      120, 122, 124, 126,
      128, 130, 132, 134,
      136, 138, 140, 142,
      144, 146, 148, 150
    };

    // Output buffer
    std::vector<uint16_t> output_bf16(nelems, 0);

    // Setup LOWOHA reorder parameters
    lowoha_reorder_params_t params;
    params.dtypes.src = data_type_t::u8;
    params.dtypes.dst = data_type_t::bf16;
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;

    log_info("Dequantization parameters: scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: bf16_val = (uint8_val - zero_point) * scale");

    // Execute reorder
    status_t status = reorder_direct(input_uint8.data(), output_bf16.data(), nelems, params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Print results
    log_info("Output BF16 values (as float32): ");
    for (size_t i = 0; i < nelems; ++i) {
      // float val = bf16_to_float(output_bf16[i]);
    }

    // Verify results
    log_info("Verification (expected vs actual):");
    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      float expected = (static_cast<float>(input_uint8[i]) - zero_point) * scale;
      float actual = bf16_to_float(output_bf16[i]);
      // Allow small tolerance due to bf16 precision
      if (std::abs(actual - expected) > 0.0001f) {
        log_error("Mismatch at index ", i, ": expected ", expected, ", got ", actual);
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("UINT8 to BF16 dequantization test PASSED!");
    } else {
      log_error("UINT8 to BF16 dequantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

} // namespace examples
} // namespace zendnnl

