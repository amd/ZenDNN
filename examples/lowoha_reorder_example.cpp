/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
    reorder_params_t params;
    params.src_dtype = data_type_t::bf16;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{4, 4};  // 4x4 matrix = 16 elements
    params.dst_shape = std::vector<int64_t>{4, 4};  // Must match src_shape
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D

    log_info("Quantization parameters: scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: int8_val = clamp(round(bf16_val / scale) + zero_point, -128, 127)");

    // Execute reorder
    status_t status = reorder_direct(input_bf16.data(), output_int8.data(), params);

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
    reorder_params_t params;
    params.src_dtype = data_type_t::s8;
    params.dst_dtype = data_type_t::bf16;
    params.src_shape = std::vector<int64_t>{4, 4};  // 4x4 matrix = 16 elements
    params.dst_shape = std::vector<int64_t>{4, 4};  // Must match src_shape
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D

    log_info("Dequantization parameters: scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: bf16_val = (int8_val - zero_point) * scale");

    // Execute reorder
    status_t status = reorder_direct(input_int8.data(), output_bf16.data(), params);

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
    reorder_params_t params;
    params.src_dtype = data_type_t::bf16;
    params.dst_dtype = data_type_t::u8;
    params.src_shape = std::vector<int64_t>{4, 4};  // 4x4 matrix = 16 elements
    params.dst_shape = std::vector<int64_t>{4, 4};  // Must match src_shape
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D

    log_info("Quantization parameters: scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: uint8_val = clamp(round(bf16_val / scale) + zero_point, 0, 255)");

    // Execute reorder
    status_t status = reorder_direct(input_bf16.data(), output_uint8.data(), params);

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
    reorder_params_t params;
    params.src_dtype = data_type_t::u8;
    params.dst_dtype = data_type_t::bf16;
    params.src_shape = std::vector<int64_t>{4, 4};  // 4x4 matrix = 16 elements
    params.dst_shape = std::vector<int64_t>{4, 4};  // Must match src_shape
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D

    log_info("Dequantization parameters: scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: bf16_val = (uint8_val - zero_point) * scale");

    // Execute reorder
    status_t status = reorder_direct(input_uint8.data(), output_bf16.data(), params);

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

//==============================================================================
// Test 1: Per-Tensor Scale and Zero-Point
//==============================================================================
int run_lowoha_reorder_bf16_to_s8_per_tensor_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Per-Tensor Quantization");
    log_info("========================================");

    // Test parameters: 2D matrix [M=4, N=4]
    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    // Per-tensor: single scale and zero_point for entire tensor
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

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    for (size_t i = 0; i < nelems; ++i) {
      input_bf16[i] = float_to_bf16(input_f32_ref[i]);
    }

    // Output buffer
    std::vector<int8_t> output_int8(nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::bf16;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{M, N};  // 2D matrix
    params.dst_shape = std::vector<int64_t>{M, N};  // Must match src_shape

    // Per-tensor: dims = {1, 1} for 2D
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = {1, 1};  // per-tensor for 2D

    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = {1, 1};  // per-tensor for 2D

    log_info("Granularity: per-tensor (scale.dims={1,1}, zero_point.dims={1,1})");
    log_info("scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: int8_val = clamp(round(bf16_val / scale) + zero_point, -128, 127)");

    // Execute reorder
    status_t status = reorder_direct(input_bf16.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results
    log_info("Verification:");
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
      log_info("Per-Tensor quantization test PASSED!");
    } else {
      log_error("Per-Tensor quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// Test 2: Per-Channel Scale and Zero-Point
//==============================================================================
int run_lowoha_reorder_bf16_to_s8_per_channel_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Per-Channel Quantization");
    log_info("========================================");

    // Test parameters: 2D matrix [M=4, N=4]
    // Per-channel means one scale/zp per column (N channels)
    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    // Per-channel: different scale and zero_point per channel (column)
    std::vector<float> scales = {0.25f, 0.5f, 0.75f, 1.0f};
    std::vector<int32_t> zero_points = {0, 10, -10, 5};

    // Create BF16 input data
    std::vector<uint16_t> input_bf16(nelems);
    std::vector<float> input_f32_ref = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Per-channel scales: [", scales[0], ", ", scales[1], ", ", scales[2], ", ", scales[3], "]");
    log_info("Per-channel zero_points: [", zero_points[0], ", ", zero_points[1], ", ", zero_points[2], ", ", zero_points[3], "]");

    for (size_t i = 0; i < nelems; ++i) {
      input_bf16[i] = float_to_bf16(input_f32_ref[i]);
    }

    // Output buffer
    std::vector<int8_t> output_int8(nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::bf16;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{M, N};  // 2D matrix
    params.dst_shape = std::vector<int64_t>{M, N};  // Must match src_shape

    // Per-channel: dims = {1, N} for 2D (N values, one per column)
    params.quant_params.scale.buff = scales.data();
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, N};  // per-channel for 2D

    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, N};  // per-channel for 2D

    log_info("Granularity: per-channel (scale.dims={1,", N, "}, zero_point.dims={1,", N, "})");

    // Execute reorder
    status_t status = reorder_direct(input_bf16.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results - each column uses its own scale/zp
    log_info("Verification:");
    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        float scale_j = scales[j];
        int32_t zp_j = zero_points[j];
        int32_t expected = static_cast<int32_t>(std::round(input_f32_ref[idx] / scale_j) + zp_j);
        expected = std::max(-128, std::min(127, expected));
        if (output_int8[idx] != static_cast<int8_t>(expected)) {
          log_error("Mismatch at [", i, ",", j, "]: expected ", expected,
                    ", got ", static_cast<int>(output_int8[idx]),
                    " (scale=", scale_j, ", zp=", zp_j, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("Per-Channel quantization test PASSED!");
    } else {
      log_error("Per-Channel quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// Test 3: Per-Group Scale and Zero-Point
//==============================================================================
int run_lowoha_reorder_bf16_to_s8_per_group_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Per-Group Quantization");
    log_info("========================================");

    // Test parameters: 2D matrix [M=8, N=4]
    // Per-group: divide M (rows) into G groups
    // dims = {G, N} means G*N total values (each group has N scale/zp values)
    // group_size = M/G
    constexpr int64_t M = 8;
    constexpr int64_t N = 4;
    constexpr int64_t G = 2;  // Number of groups
    constexpr int64_t group_size = M / G;  // 4 rows per group
    constexpr size_t nelems = M * N;

    // Per-group: G*N values (each group has different scale/zp per column)
    // Layout: [group0_col0, group0_col1, group0_col2, group0_col3, group1_col0, ...]
    std::vector<float> scales = {
      0.25f, 0.5f, 0.75f, 1.0f,   // Group 0: different scale per column
      0.5f, 1.0f, 1.5f, 2.0f      // Group 1: different scale per column
    };
    std::vector<int32_t> zero_points = {
      0, 5, -5, 10,      // Group 0: different zp per column
      -10, 0, 5, 15      // Group 1: different zp per column
    };

    // Create BF16 input data
    std::vector<uint16_t> input_bf16(nelems);
    std::vector<float> input_f32_ref = {
      // Group 0 (rows 0-3)
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f,
      // Group 1 (rows 4-7)
       1.0f,  1.5f,  2.0f,  2.5f,
       3.0f,  3.5f,  4.0f,  4.5f,
       0.5f,  1.0f,  1.5f,  2.0f,
       2.5f,  3.0f,  3.5f,  4.0f
    };

    log_info("Input shape: [M=", M, ", N=", N, "] = ", nelems, " elements");
    log_info("Groups: G=", G, " groups of ", group_size, " rows each");
    log_info("Per-group dims: {", G, ", ", N, "} = ", G * N, " total scale/zp values");

    for (size_t i = 0; i < nelems; ++i) {
      input_bf16[i] = float_to_bf16(input_f32_ref[i]);
    }

    // Output buffer
    std::vector<int8_t> output_int8(nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::bf16;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{M, N};  // 2D matrix
    params.dst_shape = std::vector<int64_t>{M, N};  // Must match src_shape

    // Per-group: dims = {G, N} for 2D (G*N total values)
    params.quant_params.scale.buff = scales.data();
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{G, N};  // G groups × N columns

    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{G, N};  // G groups × N columns

    log_info("Granularity: per-group (dims={", G, ", ", N, "})");

    // Execute reorder
    status_t status = reorder_direct(input_bf16.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results - each row-group uses its own scale/zp per column
    // Index into scale/zp: group_idx * N + col
    log_info("Verification:");
    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        int64_t group_idx = i / group_size;  // Group by row
        size_t scale_zp_idx = group_idx * N + j;  // Index: group_idx * N + col
        float scale_g = scales[scale_zp_idx];
        int32_t zp_g = zero_points[scale_zp_idx];
        int32_t expected = static_cast<int32_t>(std::round(input_f32_ref[idx] / scale_g) + zp_g);
        expected = std::max(-128, std::min(127, expected));
        if (output_int8[idx] != static_cast<int8_t>(expected)) {
          log_error("Mismatch at [", i, ",", j, "] (group ", group_idx, "): expected ", expected,
                    ", got ", static_cast<int>(output_int8[idx]),
                    " (scale=", scale_g, ", zp=", zp_g, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("Per-Group quantization test PASSED!");
    } else {
      log_error("Per-Group quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// Test 4: Mixed Granularity - Per-Tensor Scale + Per-Channel Zero-Point
//==============================================================================
int run_lowoha_reorder_bf16_to_s8_mixed_granularity_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Mixed Granularity");
    log_info("(Per-Tensor Scale + Per-Channel Zero-Point)");
    log_info("========================================");

    // Test parameters: 2D matrix [M=4, N=4]
    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    // Mixed: per-tensor scale, per-channel zero_point
    float scale = 0.5f;  // Single scale for all
    std::vector<int32_t> zero_points = {0, 5, -5, 10};  // Different zp per channel

    // Create BF16 input data
    std::vector<uint16_t> input_bf16(nelems);
    std::vector<float> input_f32_ref = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Per-tensor scale: ", scale);
    log_info("Per-channel zero_points: [", zero_points[0], ", ", zero_points[1], ", ", zero_points[2], ", ", zero_points[3], "]");

    for (size_t i = 0; i < nelems; ++i) {
      input_bf16[i] = float_to_bf16(input_f32_ref[i]);
    }

    // Output buffer
    std::vector<int8_t> output_int8(nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::bf16;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{M, N};  // 2D matrix
    params.dst_shape = std::vector<int64_t>{M, N};  // Must match src_shape

    // Per-tensor scale: dims = {1, 1} for 2D
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D

    // Per-channel zero_point: dims = {1, N} for 2D
    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, N};  // per-channel for 2D

    log_info("Granularity: mixed (scale.dims={1,1} per-tensor, zero_point.dims={1,", N, "} per-channel)");

    // Execute reorder
    status_t status = reorder_direct(input_bf16.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results
    log_info("Verification:");
    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        int32_t zp_j = zero_points[j];  // per-channel zp
        int32_t expected = static_cast<int32_t>(std::round(input_f32_ref[idx] / scale) + zp_j);
        expected = std::max(-128, std::min(127, expected));
        if (output_int8[idx] != static_cast<int8_t>(expected)) {
          log_error("Mismatch at [", i, ",", j, "]: expected ", expected,
                    ", got ", static_cast<int>(output_int8[idx]),
                    " (scale=", scale, ", zp=", zp_j, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("Mixed Granularity quantization test PASSED!");
    } else {
      log_error("Mixed Granularity quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// Test 5: Batched MatMul - Single Per-Tensor Scale/ZP for All Batches
//==============================================================================
int run_lowoha_reorder_bf16_to_s8_batched_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Batched with Shared Scale/ZP");
    log_info("========================================");

    // Test parameters: 3D batched matrix [batch=4, M=2, N=4]
    constexpr int64_t batch = 4;
    constexpr int64_t M = 2;
    constexpr int64_t N = 4;
    constexpr size_t nelems = batch * M * N;

    // Single per-tensor scale and zero_point shared across ALL batches
    float scale = 0.5f;
    int32_t zero_point = 0;

    // Create BF16 input data (4 batches, each 2x4 matrix)
    std::vector<uint16_t> input_bf16(nelems);
    std::vector<float> input_f32_ref = {
      // Batch 0
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
      // Batch 1
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f,
      // Batch 2
      -1.0f, -0.5f,  0.0f,  0.5f,
       1.0f,  1.5f,  2.0f,  2.5f,
      // Batch 3
       3.0f,  3.5f,  4.0f,  4.5f,
       5.0f,  5.5f,  6.0f,  6.5f
    };

    log_info("Input shape: [", batch, ", ", M, ", ", N, "] = ", nelems, " elements");
    log_info("Single scale=", scale, " and zero_point=", zero_point, " for ALL batches");

    for (size_t i = 0; i < nelems; ++i) {
      input_bf16[i] = float_to_bf16(input_f32_ref[i]);
    }

    // Output buffer
    std::vector<int8_t> output_int8(nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::bf16;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{batch, M, N};  // 3D batched matrix
    params.dst_shape = std::vector<int64_t>{batch, M, N};  // Must match src_shape

    // Per-tensor: dims = {1, 1, 1} for 3D
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1, 1};  // per-tensor for 3D

    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1, 1};  // per-tensor for 3D

    log_info("Granularity: per-tensor (dims={1,1,1}, shared across all ", batch, " batches)");

    // Execute reorder
    status_t status = reorder_direct(input_bf16.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results - all batches use the same scale/zp
    log_info("Verification:");
    bool all_correct = true;
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          size_t idx = b * (M * N) + i * N + j;
          int32_t expected = static_cast<int32_t>(std::round(input_f32_ref[idx] / scale) + zero_point);
          expected = std::max(-128, std::min(127, expected));
          if (output_int8[idx] != static_cast<int8_t>(expected)) {
            log_error("Mismatch at [batch=", b, ", ", i, ", ", j, "]: expected ", expected,
                      ", got ", static_cast<int>(output_int8[idx]));
            all_correct = false;
          }
        }
      }
    }

    if (all_correct) {
      log_info("Batched quantization test PASSED!");
      log_info("Successfully applied single scale/zp to all ", batch, " batches");
    } else {
      log_error("Batched quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// Dequantization Tests (INT8 -> BF16)
//==============================================================================

//==============================================================================
// Test 6: Per-Tensor Dequantization (S8 -> BF16)
//==============================================================================
int run_lowoha_reorder_s8_to_bf16_per_tensor_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Per-Tensor Dequantization (S8->BF16)");
    log_info("========================================");

    // Test parameters: 2D matrix [M=4, N=4]
    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    // Per-tensor: single scale and zero_point for entire tensor
    float scale = 0.5f;
    int32_t zero_point = 0;

    // Create INT8 input data
    std::vector<int8_t> input_int8 = {
      -4, -3, -2, -1,
       0,  1,  2,  3,
       4,  5,  6,  7,
       8,  9, 10, 11
    };

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");

    // Output buffer
    std::vector<uint16_t> output_bf16(nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::s8;
    params.dst_dtype = data_type_t::bf16;
    params.src_shape = std::vector<int64_t>{M, N};  // 2D matrix
    params.dst_shape = std::vector<int64_t>{M, N};  // Must match src_shape

    // Per-tensor: dims = {1, 1} for 2D
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = {1, 1};  // per-tensor for 2D

    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = {1, 1};  // per-tensor for 2D

    log_info("Granularity: per-tensor (scale.dims={1,1}, zero_point.dims={1,1})");
    log_info("scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: bf16_val = (int8_val - zero_point) * scale");

    // Execute reorder
    status_t status = reorder_direct(input_int8.data(), output_bf16.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results
    log_info("Verification:");
    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      float expected = (static_cast<float>(input_int8[i]) - zero_point) * scale;
      float actual = bf16_to_float(output_bf16[i]);
      if (std::abs(actual - expected) > 0.01f) {
        log_error("Mismatch at index ", i, ": expected ", expected, ", got ", actual);
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("Per-Tensor dequantization test PASSED!");
    } else {
      log_error("Per-Tensor dequantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// Test 7: Per-Channel Dequantization (S8 -> BF16)
//==============================================================================
int run_lowoha_reorder_s8_to_bf16_per_channel_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Per-Channel Dequantization (S8->BF16)");
    log_info("========================================");

    // Test parameters: 2D matrix [M=4, N=4]
    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    // Per-channel: different scale and zero_point per channel (column)
    std::vector<float> scales = {0.25f, 0.5f, 0.75f, 1.0f};
    std::vector<int32_t> zero_points = {0, 10, -10, 5};

    // Create INT8 input data
    std::vector<int8_t> input_int8 = {
      -8, -3,  0,  5,
       0,  1, -2,  6,
       4,  5,  4,  7,
       8, 15, 10, 10
    };

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Per-channel scales: [", scales[0], ", ", scales[1], ", ", scales[2], ", ", scales[3], "]");
    log_info("Per-channel zero_points: [", zero_points[0], ", ", zero_points[1], ", ", zero_points[2], ", ", zero_points[3], "]");

    // Output buffer
    std::vector<uint16_t> output_bf16(nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::s8;
    params.dst_dtype = data_type_t::bf16;
    params.src_shape = std::vector<int64_t>{M, N};  // 2D matrix
    params.dst_shape = std::vector<int64_t>{M, N};  // Must match src_shape

    // Per-channel: dims = {1, N} for 2D (N values, one per column)
    params.quant_params.scale.buff = scales.data();
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, N};  // per-channel for 2D

    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, N};  // per-channel for 2D

    log_info("Granularity: per-channel (scale.dims={1,", N, "}, zero_point.dims={1,", N, "})");

    // Execute reorder
    status_t status = reorder_direct(input_int8.data(), output_bf16.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results - each column uses its own scale/zp
    log_info("Verification:");
    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        float scale_j = scales[j];
        int32_t zp_j = zero_points[j];
        float expected = (static_cast<float>(input_int8[idx]) - zp_j) * scale_j;
        float actual = bf16_to_float(output_bf16[idx]);
        if (std::abs(actual - expected) > 0.01f) {
          log_error("Mismatch at [", i, ",", j, "]: expected ", expected,
                    ", got ", actual, " (scale=", scale_j, ", zp=", zp_j, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("Per-Channel dequantization test PASSED!");
    } else {
      log_error("Per-Channel dequantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// Test 8: Per-Group Dequantization (S8 -> BF16)
//==============================================================================
int run_lowoha_reorder_s8_to_bf16_per_group_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Per-Group Dequantization (S8->BF16)");
    log_info("========================================");

    // Test parameters: 2D matrix [M=8, N=4]
    // Per-group: divide M (rows) into G groups
    // dims = {G, N} means G*N total values (each group has N scale/zp values)
    // group_size = M/G
    constexpr int64_t M = 8;
    constexpr int64_t N = 4;
    constexpr int64_t G = 2;  // Number of groups
    constexpr int64_t group_size = M / G;  // 4 rows per group
    constexpr size_t nelems = M * N;

    // Per-group: G*N values (each group has different scale/zp per column)
    // Layout: [group0_col0, group0_col1, group0_col2, group0_col3, group1_col0, ...]
    std::vector<float> scales = {
      0.25f, 0.5f, 0.75f, 1.0f,   // Group 0: different scale per column
      0.5f, 1.0f, 1.5f, 2.0f      // Group 1: different scale per column
    };
    std::vector<int32_t> zero_points = {
      0, 5, -5, 10,      // Group 0: different zp per column
      -10, 0, 5, 15      // Group 1: different zp per column
    };

    // Create INT8 input data
    std::vector<int8_t> input_int8 = {
      // Group 0 (rows 0-3)
      -4, -3, -2, -1,
       4,  5,  6,  7,
       2,  3,  4,  5,
       1,  2,  3,  4,
      // Group 1 (rows 4-7)
      10, 11, 12, 13,
      14, 15, 16, 17,
      13, 14, 15, 16,
      12, 13, 14, 15
    };

    log_info("Input shape: [M=", M, ", N=", N, "] = ", nelems, " elements");
    log_info("Groups: G=", G, " groups of ", group_size, " rows each");
    log_info("Per-group dims: {", G, ", ", N, "} = ", G * N, " total scale/zp values");

    // Output buffer
    std::vector<uint16_t> output_bf16(nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::s8;
    params.dst_dtype = data_type_t::bf16;
    params.src_shape = std::vector<int64_t>{M, N};  // 2D matrix
    params.dst_shape = std::vector<int64_t>{M, N};  // Must match src_shape

    // Per-group: dims = {G, N} for 2D (G*N total values)
    params.quant_params.scale.buff = scales.data();
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{G, N};  // G groups × N columns

    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{G, N};  // G groups × N columns

    log_info("Granularity: per-group (dims={", G, ", ", N, "})");

    // Execute reorder
    status_t status = reorder_direct(input_int8.data(), output_bf16.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results - each row-group uses its own scale/zp per column
    // Index into scale/zp: group_idx * N + col
    log_info("Verification:");
    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        int64_t group_idx = i / group_size;  // Group by row
        size_t scale_zp_idx = group_idx * N + j;  // Index: group_idx * N + col
        float scale_g = scales[scale_zp_idx];
        int32_t zp_g = zero_points[scale_zp_idx];
        float expected = (static_cast<float>(input_int8[idx]) - zp_g) * scale_g;
        float actual = bf16_to_float(output_bf16[idx]);
        if (std::abs(actual - expected) > 0.01f) {
          log_error("Mismatch at [", i, ",", j, "] (group ", group_idx, "): expected ", expected,
                    ", got ", actual, " (scale=", scale_g, ", zp=", zp_g, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("Per-Group dequantization test PASSED!");
    } else {
      log_error("Per-Group dequantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// Test 9: Mixed Granularity Dequantization (S8 -> BF16)
//==============================================================================
int run_lowoha_reorder_s8_to_bf16_mixed_granularity_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Mixed Granularity Dequantization (S8->BF16)");
    log_info("(Per-Tensor Scale + Per-Channel Zero-Point)");
    log_info("========================================");

    // Test parameters: 2D matrix [M=4, N=4]
    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    // Mixed: per-tensor scale, per-channel zero_point
    float scale = 0.5f;  // Single scale for all
    std::vector<int32_t> zero_points = {0, 5, -5, 10};  // Different zp per channel

    // Create INT8 input data
    std::vector<int8_t> input_int8 = {
      -4,  2, -7,  9,
       0,  6, -3, 13,
       4, 10,  1, 17,
       8, 14,  5, 21
    };

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Per-tensor scale: ", scale);
    log_info("Per-channel zero_points: [", zero_points[0], ", ", zero_points[1], ", ", zero_points[2], ", ", zero_points[3], "]");

    // Output buffer
    std::vector<uint16_t> output_bf16(nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::s8;
    params.dst_dtype = data_type_t::bf16;
    params.src_shape = std::vector<int64_t>{M, N};  // 2D matrix
    params.dst_shape = std::vector<int64_t>{M, N};  // Must match src_shape

    // Per-tensor scale: dims = {1, 1} for 2D
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D

    // Per-channel zero_point: dims = {1, N} for 2D
    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, N};  // per-channel for 2D

    log_info("Granularity: mixed (scale.dims={1,1} per-tensor, zero_point.dims={1,", N, "} per-channel)");

    // Execute reorder
    status_t status = reorder_direct(input_int8.data(), output_bf16.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results
    log_info("Verification:");
    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        int32_t zp_j = zero_points[j];  // per-channel zp
        float expected = (static_cast<float>(input_int8[idx]) - zp_j) * scale;
        float actual = bf16_to_float(output_bf16[idx]);
        if (std::abs(actual - expected) > 0.01f) {
          log_error("Mismatch at [", i, ",", j, "]: expected ", expected,
                    ", got ", actual, " (scale=", scale, ", zp=", zp_j, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("Mixed Granularity dequantization test PASSED!");
    } else {
      log_error("Mixed Granularity dequantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// Test 10: Batched Dequantization (S8 -> BF16)
//==============================================================================
int run_lowoha_reorder_s8_to_bf16_batched_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Batched Dequantization (S8->BF16)");
    log_info("========================================");

    // Test parameters: 3D batched matrix [batch=4, M=2, N=4]
    constexpr int64_t batch = 4;
    constexpr int64_t M = 2;
    constexpr int64_t N = 4;
    constexpr size_t nelems = batch * M * N;

    // Single per-tensor scale and zero_point shared across ALL batches
    float scale = 0.5f;
    int32_t zero_point = 0;

    // Create INT8 input data (4 batches, each 2x4 matrix)
    std::vector<int8_t> input_int8 = {
      // Batch 0
      -4, -3, -2, -1,
       0,  1,  2,  3,
      // Batch 1
       4,  5,  6,  7,
       8,  9, 10, 11,
      // Batch 2
      -2, -1,  0,  1,
       2,  3,  4,  5,
      // Batch 3
       6,  7,  8,  9,
      10, 11, 12, 13
    };

    log_info("Input shape: [", batch, ", ", M, ", ", N, "] = ", nelems, " elements");
    log_info("Single scale=", scale, " and zero_point=", zero_point, " for ALL batches");

    // Output buffer
    std::vector<uint16_t> output_bf16(nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::s8;
    params.dst_dtype = data_type_t::bf16;
    params.src_shape = std::vector<int64_t>{batch, M, N};  // 3D batched matrix
    params.dst_shape = std::vector<int64_t>{batch, M, N};  // Must match src_shape

    // Per-tensor: dims = {1, 1, 1} for 3D
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1, 1};  // per-tensor for 3D

    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1, 1};  // per-tensor for 3D

    log_info("Granularity: per-tensor (dims={1,1,1}, shared across all ", batch, " batches)");

    // Execute reorder
    status_t status = reorder_direct(input_int8.data(), output_bf16.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results - all batches use the same scale/zp
    log_info("Verification:");
    bool all_correct = true;
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          size_t idx = b * (M * N) + i * N + j;
          float expected = (static_cast<float>(input_int8[idx]) - zero_point) * scale;
          float actual = bf16_to_float(output_bf16[idx]);
          if (std::abs(actual - expected) > 0.01f) {
            log_error("Mismatch at [batch=", b, ", ", i, ", ", j, "]: expected ", expected,
                      ", got ", actual);
            all_correct = false;
          }
        }
      }
    }

    if (all_correct) {
      log_info("Batched dequantization test PASSED!");
      log_info("Successfully applied single scale/zp to all ", batch, " batches");
    } else {
      log_error("Batched dequantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// Strided Memory Layout Tests
//==============================================================================

//==============================================================================
// Test 11: Strided 2D Matrix (BF16 -> S8)
//==============================================================================
int run_lowoha_reorder_bf16_to_s8_strided_2d_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Strided 2D Matrix (BF16->S8)");
    log_info("========================================");

    // Test parameters: 2D matrix [M=4, N=4] embedded in larger memory [M=4, N=8]
    // We want to extract every other column (stride_N = 2)
    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr int64_t src_row_stride = 8;  // Source has 8 columns per row
    constexpr int64_t src_col_stride = 2;  // Extract every 2nd column
    constexpr size_t nelems = M * N;
    constexpr size_t src_total_size = M * src_row_stride;

    // Per-tensor scale and zero_point
    float scale = 0.5f;
    int32_t zero_point = 0;

    // Create source BF16 data in strided layout (4x8 matrix)
    // We'll read columns 0, 2, 4, 6 from each row
    std::vector<uint16_t> src_bf16(src_total_size);
    
    // Fill entire source buffer with recognizable pattern
    std::vector<float> src_f32_full = {
      // Row 0: cols 0,1,2,3,4,5,6,7 (we extract 0,2,4,6)
      -2.0f, 99.0f, -1.5f, 99.0f, -1.0f, 99.0f, -0.5f, 99.0f,
      // Row 1
       0.0f, 99.0f,  0.5f, 99.0f,  1.0f, 99.0f,  1.5f, 99.0f,
      // Row 2
       2.0f, 99.0f,  2.5f, 99.0f,  3.0f, 99.0f,  3.5f, 99.0f,
      // Row 3
       4.0f, 99.0f,  4.5f, 99.0f,  5.0f, 99.0f,  5.5f, 99.0f
    };

    // Expected values (extracted columns: 0, 2, 4, 6 from each row)
    std::vector<float> expected_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Source layout: [4, 8] (contiguous)");
    log_info("Logical shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Strides: [", src_row_stride, ", ", src_col_stride, "] (extract every 2nd column)");

    for (size_t i = 0; i < src_total_size; ++i) {
      src_bf16[i] = float_to_bf16(src_f32_full[i]);
    }

    // Output buffer (contiguous)
    std::vector<int8_t> output_int8(nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::bf16;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{M, N};  // Logical shape
    params.dst_shape = std::vector<int64_t>{M, N};  // Must match src_shape
    params.src_strides = std::vector<int64_t>{src_row_stride, src_col_stride};  // Strided access

    // Per-tensor quantization: dims = {1, 1} for 2D
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D

    log_info("scale=", scale, ", zero_point=", zero_point);

    // Execute reorder
    status_t status = reorder_direct(src_bf16.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results
    log_info("Verification:");
    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        int32_t expected = static_cast<int32_t>(std::round(expected_f32[idx] / scale) + zero_point);
        expected = std::max(-128, std::min(127, expected));
        if (output_int8[idx] != static_cast<int8_t>(expected)) {
          log_error("Mismatch at [", i, ",", j, "]: expected ", expected,
                    ", got ", static_cast<int>(output_int8[idx]));
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("Strided 2D quantization test PASSED!");
      log_info("Successfully read strided data [stride_M=", src_row_stride, 
               ", stride_N=", src_col_stride, "]");
    } else {
      log_error("Strided 2D quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// Test 12: Strided 3D Batched Matrix (BF16 -> S8)
//==============================================================================
int run_lowoha_reorder_bf16_to_s8_strided_3d_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Strided 3D Batched Matrix (BF16->S8)");
    log_info("========================================");

    // Test parameters: 3D matrix [batch=2, M=2, N=3] embedded in larger memory
    // Source layout: [batch=4, M=4, N=4] but we only use part of it with strides
    constexpr int64_t batch = 2;
    constexpr int64_t M = 2;
    constexpr int64_t N = 3;
    constexpr int64_t src_batch_stride = 32;  // Each batch separated by 32 elements (skip a batch)
    constexpr int64_t src_row_stride = 4;     // Each row has 4 elements
    constexpr int64_t src_col_stride = 1;     // Contiguous within row
    constexpr size_t nelems = batch * M * N;
    constexpr size_t src_total_size = 64;     // 4 batches * 4 rows * 4 cols

    // Per-tensor scale and zero_point
    float scale = 0.25f;
    int32_t zero_point = 5;

    // Create source BF16 data in strided layout
    // We'll read batches 0 and 2 (skip batch 1 and 3), rows 0-1, cols 0-2
    std::vector<float> src_f32_full(src_total_size);
    
    // Fill with recognizable pattern
    for (size_t i = 0; i < src_total_size; ++i) {
      src_f32_full[i] = static_cast<float>(i) * 0.1f;
    }

    // Expected values based on strided access:
    // Batch 0: [0][0-1][0-2], Batch 1 (skip), Batch 2: [2][0-1][0-2]
    std::vector<float> expected_f32 = {
      // Batch 0: indices 0,1,2 and 4,5,6
      0.0f, 0.1f, 0.2f,   // row 0, cols 0-2
      0.4f, 0.5f, 0.6f,   // row 1, cols 0-2
      // Batch 2 (offset 32): indices 32,33,34 and 36,37,38
      3.2f, 3.3f, 3.4f,   // row 0, cols 0-2
      3.6f, 3.7f, 3.8f    // row 1, cols 0-2
    };

    log_info("Source layout: [4, 4, 4] (contiguous, 64 elements)");
    log_info("Logical shape: [", batch, ", ", M, ", ", N, "] = ", nelems, " elements");
    log_info("Strides: [", src_batch_stride, ", ", src_row_stride, ", ", src_col_stride, "]");
    log_info("(Extracting batches 0,2 skipping 1,3; rows 0-1; cols 0-2)");

    std::vector<uint16_t> src_bf16(src_total_size);
    for (size_t i = 0; i < src_total_size; ++i) {
      src_bf16[i] = float_to_bf16(src_f32_full[i]);
    }

    // Output buffer (contiguous)
    std::vector<int8_t> output_int8(nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::bf16;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{batch, M, N};  // Logical shape
    params.dst_shape = std::vector<int64_t>{batch, M, N};  // Must match src_shape
    params.src_strides = std::vector<int64_t>{src_batch_stride, src_row_stride, src_col_stride};  // Strided access

    // Per-tensor quantization: dims = {1, 1, 1} for 3D
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = {1, 1, 1};  // per-tensor for 3D
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = {1, 1, 1};  // per-tensor for 3D

    log_info("scale=", scale, ", zero_point=", zero_point);

    // Execute reorder
    status_t status = reorder_direct(src_bf16.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results
    log_info("Verification:");
    bool all_correct = true;
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          size_t idx = b * (M * N) + i * N + j;
          int32_t expected = static_cast<int32_t>(std::round(expected_f32[idx] / scale) + zero_point);
          expected = std::max(-128, std::min(127, expected));
          if (output_int8[idx] != static_cast<int8_t>(expected)) {
            log_error("Mismatch at [batch=", b, ", ", i, ", ", j, "]: expected ", expected,
                      ", got ", static_cast<int>(output_int8[idx]),
                      " (input=", expected_f32[idx], ")");
            all_correct = false;
          }
        }
      }
    }

    if (all_correct) {
      log_info("Strided 3D quantization test PASSED!");
      log_info("Successfully read strided data [stride_batch=", src_batch_stride,
               ", stride_M=", src_row_stride, ", stride_N=", src_col_stride, "]");
    } else {
      log_error("Strided 3D quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// Test 13: Row-Padded Matrix (Common Alignment Use Case)
//==============================================================================
int run_lowoha_reorder_bf16_to_s8_strided_row_padding_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Row-Padded Matrix (BF16->S8)");
    log_info("========================================");

    // Real-world scenario: Matrix with row padding for memory alignment
    // Logical matrix: [M=4, N=6] (24 elements)
    // Physical layout: Each row padded to 8 elements for 64-byte alignment (bf16 = 2 bytes)
    // Total physical size: 4 rows × 8 elements = 32 elements
    constexpr int64_t M = 4;
    constexpr int64_t N = 6;
    constexpr int64_t padded_row_size = 8;  // Padded to 8 for alignment
    constexpr int64_t stride_M = padded_row_size;  // Elements between row starts
    constexpr int64_t stride_N = 1;  // Contiguous within row
    constexpr size_t logical_nelems = M * N;
    constexpr size_t physical_size = M * padded_row_size;

    // Per-tensor scale and zero_point
    float scale = 0.5f;
    int32_t zero_point = 0;

    // Create source BF16 data with row padding
    // Layout: [data0..data5, pad, pad] [data0..data5, pad, pad] ...
    std::vector<uint16_t> src_bf16(physical_size);
    std::vector<float> src_f32_physical(physical_size);
    std::vector<float> expected_f32(logical_nelems);

    log_info("Logical shape: [M=", M, ", N=", N, "] = ", logical_nelems, " elements");
    log_info("Physical layout: [", M, " rows × ", padded_row_size, " cols] = ", physical_size, " elements");
    log_info("Strides: [", stride_M, ", ", stride_N, "] (row padding for alignment)");
    log_info("");
    log_info("Memory Layout Visualization:");
    log_info("┌────────────────────────────────────────────────┐");

    // Fill source data: actual values in columns 0-5, padding value (99.0) in columns 6-7
    float val = 0.0f;
    size_t expected_idx = 0;
    for (int64_t row = 0; row < M; ++row) {
      std::string row_str = "│ Row " + std::to_string(row) + ": ";
      for (int64_t col = 0; col < padded_row_size; ++col) {
        size_t physical_idx = row * padded_row_size + col;
        if (col < N) {
          // Actual data
          src_f32_physical[physical_idx] = val;
          expected_f32[expected_idx++] = val;
          row_str += std::to_string(static_cast<int>(val)) + " ";
          val += 1.0f;
        } else {
          // Padding (will be skipped by strided access)
          src_f32_physical[physical_idx] = 99.0f;  // Padding value
          row_str += "[P] ";
        }
      }
      row_str += "│";
      log_info(row_str);
    }
    log_info("└────────────────────────────────────────────────┘");
    log_info("([P] = padding, skipped by strided access)");
    log_info("");

    // Convert to bf16
    for (size_t i = 0; i < physical_size; ++i) {
      src_bf16[i] = float_to_bf16(src_f32_physical[i]);
    }

    // Output buffer (contiguous, no padding)
    std::vector<int8_t> output_int8(logical_nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::bf16;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{M, N};  // Logical shape (without padding)
    params.dst_shape = std::vector<int64_t>{M, N};  // Must match src_shape
    params.src_strides = std::vector<int64_t>{stride_M, stride_N};  // Strided access to skip padding

    // Per-tensor quantization: dims = {1, 1} for 2D
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D

    log_info("Quantization: scale=", scale, ", zero_point=", zero_point);

    // Execute reorder
    status_t status = reorder_direct(src_bf16.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results - output should be contiguous [M × N]
    log_info("Verification:");
    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;  // Contiguous output index
        int32_t expected = static_cast<int32_t>(std::round(expected_f32[idx] / scale) + zero_point);
        expected = std::max(-128, std::min(127, expected));
        if (output_int8[idx] != static_cast<int8_t>(expected)) {
          log_error("Mismatch at [", i, ",", j, "]: expected ", expected,
                    ", got ", static_cast<int>(output_int8[idx]));
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("Row-Padded strided quantization test PASSED!");
      log_info("Successfully extracted [", M, "×", N, "] logical matrix from [", 
               M, "×", padded_row_size, "] physical layout");
      log_info("Output is contiguous: ", logical_nelems, " elements without padding");
    } else {
      log_error("Row-Padded strided quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// FP32 Basic Data Type Conversion Tests
//==============================================================================

int run_lowoha_reorder_f32_to_int8_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: FP32 to INT8 Quantization");
    log_info("========================================");

    // Test parameters
    constexpr size_t nelems = 16;
    float scale = 0.5f;
    int32_t zero_point = 0;

    // Create FP32 input data
    std::vector<float> input_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Input values (float32): ");

    // Output buffer
    std::vector<int8_t> output_int8(nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{4, 4};  // 4x4 matrix = 16 elements
    params.dst_shape = std::vector<int64_t>{4, 4};  // Must match src_shape
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D

    log_info("Quantization parameters: scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: int8_val = clamp(round(f32_val / scale) + zero_point, -128, 127)");

    // Execute reorder
    status_t status = reorder_direct(input_f32.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results
    log_info("Verification (expected vs actual):");
    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      int32_t expected = static_cast<int32_t>(std::round(input_f32[i] / scale) + zero_point);
      expected = std::max(-128, std::min(127, expected));
      if (output_int8[i] != static_cast<int8_t>(expected)) {
        log_error("Mismatch at index ", i, ": expected ", expected,
                  ", got ", static_cast<int>(output_int8[i]));
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("FP32 to INT8 quantization test PASSED!");
    } else {
      log_error("FP32 to INT8 quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_int8_to_f32_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: INT8 to FP32 Dequantization");
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

    // Output buffer
    std::vector<float> output_f32(nelems, 0.0f);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::s8;
    params.dst_dtype = data_type_t::f32;
    params.src_shape = std::vector<int64_t>{4, 4};  // 4x4 matrix = 16 elements
    params.dst_shape = std::vector<int64_t>{4, 4};  // Must match src_shape
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D

    log_info("Dequantization parameters: scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: f32_val = (int8_val - zero_point) * scale");

    // Execute reorder
    status_t status = reorder_direct(input_int8.data(), output_f32.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results
    log_info("Verification (expected vs actual):");
    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      float expected = (static_cast<float>(input_int8[i]) - zero_point) * scale;
      if (std::abs(output_f32[i] - expected) > 0.0001f) {
        log_error("Mismatch at index ", i, ": expected ", expected, ", got ", output_f32[i]);
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("INT8 to FP32 dequantization test PASSED!");
    } else {
      log_error("INT8 to FP32 dequantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_f32_to_uint8_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: FP32 to UINT8 Quantization");
    log_info("========================================");

    // Test parameters
    constexpr size_t nelems = 16;
    float scale = 0.5f;
    int32_t zero_point = 128;  // Typical zero-point for uint8

    // Create FP32 input data
    std::vector<float> input_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Input values (float32): ");

    // Output buffer
    std::vector<uint8_t> output_uint8(nelems, 0);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::u8;
    params.src_shape = std::vector<int64_t>{4, 4};  // 4x4 matrix = 16 elements
    params.dst_shape = std::vector<int64_t>{4, 4};  // Must match src_shape
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D

    log_info("Quantization parameters: scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: uint8_val = clamp(round(f32_val / scale) + zero_point, 0, 255)");

    // Execute reorder
    status_t status = reorder_direct(input_f32.data(), output_uint8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results
    log_info("Verification (expected vs actual):");
    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      int32_t expected = static_cast<int32_t>(std::round(input_f32[i] / scale) + zero_point);
      expected = std::max(0, std::min(255, expected));
      if (output_uint8[i] != static_cast<uint8_t>(expected)) {
        log_error("Mismatch at index ", i, ": expected ", expected,
                  ", got ", static_cast<int>(output_uint8[i]));
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("FP32 to UINT8 quantization test PASSED!");
    } else {
      log_error("FP32 to UINT8 quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_uint8_to_f32_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: UINT8 to FP32 Dequantization");
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
    std::vector<float> output_f32(nelems, 0.0f);

    // Setup LOWOHA reorder parameters
    reorder_params_t params;
    params.src_dtype = data_type_t::u8;
    params.dst_dtype = data_type_t::f32;
    params.src_shape = std::vector<int64_t>{4, 4};  // 4x4 matrix = 16 elements
    params.dst_shape = std::vector<int64_t>{4, 4};  // Must match src_shape
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};  // per-tensor for 2D

    log_info("Dequantization parameters: scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: f32_val = (uint8_val - zero_point) * scale");

    // Execute reorder
    status_t status = reorder_direct(input_uint8.data(), output_f32.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify results
    log_info("Verification (expected vs actual):");
    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      float expected = (static_cast<float>(input_uint8[i]) - zero_point) * scale;
      if (std::abs(output_f32[i] - expected) > 0.0001f) {
        log_error("Mismatch at index ", i, ": expected ", expected, ", got ", output_f32[i]);
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("UINT8 to FP32 dequantization test PASSED!");
    } else {
      log_error("UINT8 to FP32 dequantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// FP32 Granularity Tests
//==============================================================================

int run_lowoha_reorder_f32_to_s8_per_tensor_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: FP32 Per-Tensor Quantization");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    float scale = 0.5f;
    int32_t zero_point = 0;

    std::vector<float> input_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");

    std::vector<int8_t> output_int8(nelems, 0);

    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};

    log_info("Granularity: per-tensor (scale.dims={1,1}, zero_point.dims={1,1})");
    log_info("scale=", scale, ", zero_point=", zero_point);

    status_t status = reorder_direct(input_f32.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      int32_t expected = static_cast<int32_t>(std::round(input_f32[i] / scale) + zero_point);
      expected = std::max(-128, std::min(127, expected));
      if (output_int8[i] != static_cast<int8_t>(expected)) {
        log_error("Mismatch at index ", i, ": expected ", expected,
                  ", got ", static_cast<int>(output_int8[i]));
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("FP32 Per-Tensor quantization test PASSED!");
    } else {
      log_error("FP32 Per-Tensor quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_f32_to_s8_per_channel_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: FP32 Per-Channel Quantization");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    std::vector<float> scales = {0.25f, 0.5f, 0.75f, 1.0f};
    std::vector<int32_t> zero_points = {0, 10, -10, 5};

    std::vector<float> input_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Per-channel scales: [", scales[0], ", ", scales[1], ", ", scales[2], ", ", scales[3], "]");
    log_info("Per-channel zero_points: [", zero_points[0], ", ", zero_points[1], ", ", zero_points[2], ", ", zero_points[3], "]");

    std::vector<int8_t> output_int8(nelems, 0);

    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.quant_params.scale.buff = scales.data();
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, N};
    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, N};

    log_info("Granularity: per-channel (scale.dims={1,", N, "}, zero_point.dims={1,", N, "})");

    status_t status = reorder_direct(input_f32.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        float scale_j = scales[j];
        int32_t zp_j = zero_points[j];
        int32_t expected = static_cast<int32_t>(std::round(input_f32[idx] / scale_j) + zp_j);
        expected = std::max(-128, std::min(127, expected));
        if (output_int8[idx] != static_cast<int8_t>(expected)) {
          log_error("Mismatch at [", i, ",", j, "]: expected ", expected,
                    ", got ", static_cast<int>(output_int8[idx]),
                    " (scale=", scale_j, ", zp=", zp_j, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("FP32 Per-Channel quantization test PASSED!");
    } else {
      log_error("FP32 Per-Channel quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_f32_to_s8_per_group_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: FP32 Per-Group Quantization");
    log_info("========================================");

    constexpr int64_t M = 8;
    constexpr int64_t N = 4;
    constexpr int64_t G = 2;
    constexpr int64_t group_size = M / G;
    constexpr size_t nelems = M * N;

    std::vector<float> scales = {
      0.25f, 0.5f, 0.75f, 1.0f,
      0.5f, 1.0f, 1.5f, 2.0f
    };
    std::vector<int32_t> zero_points = {
      0, 5, -5, 10,
      -10, 0, 5, 15
    };

    std::vector<float> input_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f,
       1.0f,  1.5f,  2.0f,  2.5f,
       3.0f,  3.5f,  4.0f,  4.5f,
       0.5f,  1.0f,  1.5f,  2.0f,
       2.5f,  3.0f,  3.5f,  4.0f
    };

    log_info("Input shape: [M=", M, ", N=", N, "] = ", nelems, " elements");
    log_info("Groups: G=", G, " groups of ", group_size, " rows each");
    log_info("Per-group dims: {", G, ", ", N, "} = ", G * N, " total scale/zp values");

    std::vector<int8_t> output_int8(nelems, 0);

    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.quant_params.scale.buff = scales.data();
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{G, N};
    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{G, N};

    log_info("Granularity: per-group (dims={", G, ", ", N, "})");

    status_t status = reorder_direct(input_f32.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        int64_t group_idx = i / group_size;
        size_t scale_zp_idx = group_idx * N + j;
        float scale_g = scales[scale_zp_idx];
        int32_t zp_g = zero_points[scale_zp_idx];
        int32_t expected = static_cast<int32_t>(std::round(input_f32[idx] / scale_g) + zp_g);
        expected = std::max(-128, std::min(127, expected));
        if (output_int8[idx] != static_cast<int8_t>(expected)) {
          log_error("Mismatch at [", i, ",", j, "] (group ", group_idx, "): expected ", expected,
                    ", got ", static_cast<int>(output_int8[idx]),
                    " (scale=", scale_g, ", zp=", zp_g, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("FP32 Per-Group quantization test PASSED!");
    } else {
      log_error("FP32 Per-Group quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_f32_to_s8_mixed_granularity_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: FP32 Mixed Granularity");
    log_info("(Per-Tensor Scale + Per-Channel Zero-Point)");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    float scale = 0.5f;
    std::vector<int32_t> zero_points = {0, 5, -5, 10};

    std::vector<float> input_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Per-tensor scale: ", scale);
    log_info("Per-channel zero_points: [", zero_points[0], ", ", zero_points[1], ", ", zero_points[2], ", ", zero_points[3], "]");

    std::vector<int8_t> output_int8(nelems, 0);

    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};
    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, N};

    log_info("Granularity: mixed (scale.dims={1,1} per-tensor, zero_point.dims={1,", N, "} per-channel)");

    status_t status = reorder_direct(input_f32.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        int32_t zp_j = zero_points[j];
        int32_t expected = static_cast<int32_t>(std::round(input_f32[idx] / scale) + zp_j);
        expected = std::max(-128, std::min(127, expected));
        if (output_int8[idx] != static_cast<int8_t>(expected)) {
          log_error("Mismatch at [", i, ",", j, "]: expected ", expected,
                    ", got ", static_cast<int>(output_int8[idx]),
                    " (scale=", scale, ", zp=", zp_j, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("FP32 Mixed Granularity quantization test PASSED!");
    } else {
      log_error("FP32 Mixed Granularity quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_f32_to_s8_batched_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: FP32 Batched with Shared Scale/ZP");
    log_info("========================================");

    constexpr int64_t batch = 4;
    constexpr int64_t M = 2;
    constexpr int64_t N = 4;
    constexpr size_t nelems = batch * M * N;

    float scale = 0.5f;
    int32_t zero_point = 0;

    std::vector<float> input_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f,
      -1.0f, -0.5f,  0.0f,  0.5f,
       1.0f,  1.5f,  2.0f,  2.5f,
       3.0f,  3.5f,  4.0f,  4.5f,
       5.0f,  5.5f,  6.0f,  6.5f
    };

    log_info("Input shape: [", batch, ", ", M, ", ", N, "] = ", nelems, " elements");
    log_info("Single scale=", scale, " and zero_point=", zero_point, " for ALL batches");

    std::vector<int8_t> output_int8(nelems, 0);

    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{batch, M, N};
    params.dst_shape = std::vector<int64_t>{batch, M, N};
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1, 1};
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1, 1};

    log_info("Granularity: per-tensor (dims={1,1,1}, shared across all ", batch, " batches)");

    status_t status = reorder_direct(input_f32.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          size_t idx = b * (M * N) + i * N + j;
          int32_t expected = static_cast<int32_t>(std::round(input_f32[idx] / scale) + zero_point);
          expected = std::max(-128, std::min(127, expected));
          if (output_int8[idx] != static_cast<int8_t>(expected)) {
            log_error("Mismatch at [batch=", b, ", ", i, ", ", j, "]: expected ", expected,
                      ", got ", static_cast<int>(output_int8[idx]));
            all_correct = false;
          }
        }
      }
    }

    if (all_correct) {
      log_info("FP32 Batched quantization test PASSED!");
      log_info("Successfully applied single scale/zp to all ", batch, " batches");
    } else {
      log_error("FP32 Batched quantization test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// FP32 Dequantization Tests (INT8 -> FP32)
//==============================================================================

int run_lowoha_reorder_s8_to_f32_per_tensor_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Per-Tensor Dequantization (S8->FP32)");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    float scale = 0.5f;
    int32_t zero_point = 0;

    std::vector<int8_t> input_int8 = {
      -4, -3, -2, -1,
       0,  1,  2,  3,
       4,  5,  6,  7,
       8,  9, 10, 11
    };

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");

    std::vector<float> output_f32(nelems, 0.0f);

    reorder_params_t params;
    params.src_dtype = data_type_t::s8;
    params.dst_dtype = data_type_t::f32;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};

    log_info("Granularity: per-tensor (scale.dims={1,1}, zero_point.dims={1,1})");
    log_info("scale=", scale, ", zero_point=", zero_point);
    log_info("Formula: f32_val = (int8_val - zero_point) * scale");

    status_t status = reorder_direct(input_int8.data(), output_f32.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      float expected = (static_cast<float>(input_int8[i]) - zero_point) * scale;
      if (std::abs(output_f32[i] - expected) > 0.0001f) {
        log_error("Mismatch at index ", i, ": expected ", expected, ", got ", output_f32[i]);
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("Per-Tensor dequantization (S8->FP32) test PASSED!");
    } else {
      log_error("Per-Tensor dequantization (S8->FP32) test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_s8_to_f32_per_channel_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Per-Channel Dequantization (S8->FP32)");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    std::vector<float> scales = {0.25f, 0.5f, 0.75f, 1.0f};
    std::vector<int32_t> zero_points = {0, 10, -10, 5};

    std::vector<int8_t> input_int8 = {
      -8, -3,  0,  5,
       0,  1, -2,  6,
       4,  5,  4,  7,
       8, 15, 10, 10
    };

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Per-channel scales: [", scales[0], ", ", scales[1], ", ", scales[2], ", ", scales[3], "]");
    log_info("Per-channel zero_points: [", zero_points[0], ", ", zero_points[1], ", ", zero_points[2], ", ", zero_points[3], "]");

    std::vector<float> output_f32(nelems, 0.0f);

    reorder_params_t params;
    params.src_dtype = data_type_t::s8;
    params.dst_dtype = data_type_t::f32;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.quant_params.scale.buff = scales.data();
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, N};
    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, N};

    log_info("Granularity: per-channel (scale.dims={1,", N, "}, zero_point.dims={1,", N, "})");

    status_t status = reorder_direct(input_int8.data(), output_f32.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        float scale_j = scales[j];
        int32_t zp_j = zero_points[j];
        float expected = (static_cast<float>(input_int8[idx]) - zp_j) * scale_j;
        if (std::abs(output_f32[idx] - expected) > 0.0001f) {
          log_error("Mismatch at [", i, ",", j, "]: expected ", expected,
                    ", got ", output_f32[idx], " (scale=", scale_j, ", zp=", zp_j, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("Per-Channel dequantization (S8->FP32) test PASSED!");
    } else {
      log_error("Per-Channel dequantization (S8->FP32) test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_s8_to_f32_per_group_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Per-Group Dequantization (S8->FP32)");
    log_info("========================================");

    constexpr int64_t M = 8;
    constexpr int64_t N = 4;
    constexpr int64_t G = 2;
    constexpr int64_t group_size = M / G;
    constexpr size_t nelems = M * N;

    std::vector<float> scales = {
      0.25f, 0.5f, 0.75f, 1.0f,
      0.5f, 1.0f, 1.5f, 2.0f
    };
    std::vector<int32_t> zero_points = {
      0, 5, -5, 10,
      -10, 0, 5, 15
    };

    std::vector<int8_t> input_int8 = {
      -4, -3, -2, -1,
       4,  5,  6,  7,
       2,  3,  4,  5,
       1,  2,  3,  4,
      10, 11, 12, 13,
      14, 15, 16, 17,
      13, 14, 15, 16,
      12, 13, 14, 15
    };

    log_info("Input shape: [M=", M, ", N=", N, "] = ", nelems, " elements");
    log_info("Groups: G=", G, " groups of ", group_size, " rows each");
    log_info("Per-group dims: {", G, ", ", N, "} = ", G * N, " total scale/zp values");

    std::vector<float> output_f32(nelems, 0.0f);

    reorder_params_t params;
    params.src_dtype = data_type_t::s8;
    params.dst_dtype = data_type_t::f32;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.quant_params.scale.buff = scales.data();
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{G, N};
    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{G, N};

    log_info("Granularity: per-group (dims={", G, ", ", N, "})");

    status_t status = reorder_direct(input_int8.data(), output_f32.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        int64_t group_idx = i / group_size;
        size_t scale_zp_idx = group_idx * N + j;
        float scale_g = scales[scale_zp_idx];
        int32_t zp_g = zero_points[scale_zp_idx];
        float expected = (static_cast<float>(input_int8[idx]) - zp_g) * scale_g;
        if (std::abs(output_f32[idx] - expected) > 0.0001f) {
          log_error("Mismatch at [", i, ",", j, "] (group ", group_idx, "): expected ", expected,
                    ", got ", output_f32[idx], " (scale=", scale_g, ", zp=", zp_g, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("Per-Group dequantization (S8->FP32) test PASSED!");
    } else {
      log_error("Per-Group dequantization (S8->FP32) test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_s8_to_f32_mixed_granularity_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Mixed Granularity Dequantization (S8->FP32)");
    log_info("(Per-Tensor Scale + Per-Channel Zero-Point)");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    float scale = 0.5f;
    std::vector<int32_t> zero_points = {0, 5, -5, 10};

    std::vector<int8_t> input_int8 = {
      -4,  2, -7,  9,
       0,  6, -3, 13,
       4, 10,  1, 17,
       8, 14,  5, 21
    };

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Per-tensor scale: ", scale);
    log_info("Per-channel zero_points: [", zero_points[0], ", ", zero_points[1], ", ", zero_points[2], ", ", zero_points[3], "]");

    std::vector<float> output_f32(nelems, 0.0f);

    reorder_params_t params;
    params.src_dtype = data_type_t::s8;
    params.dst_dtype = data_type_t::f32;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};
    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, N};

    log_info("Granularity: mixed (scale.dims={1,1} per-tensor, zero_point.dims={1,", N, "} per-channel)");

    status_t status = reorder_direct(input_int8.data(), output_f32.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        int32_t zp_j = zero_points[j];
        float expected = (static_cast<float>(input_int8[idx]) - zp_j) * scale;
        if (std::abs(output_f32[idx] - expected) > 0.0001f) {
          log_error("Mismatch at [", i, ",", j, "]: expected ", expected,
                    ", got ", output_f32[idx], " (scale=", scale, ", zp=", zp_j, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("Mixed Granularity dequantization (S8->FP32) test PASSED!");
    } else {
      log_error("Mixed Granularity dequantization (S8->FP32) test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// FP32 Strided Memory Layout Tests
//==============================================================================

int run_lowoha_reorder_f32_to_s8_strided_2d_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Strided 2D Matrix (FP32->S8)");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr int64_t src_row_stride = 8;
    constexpr int64_t src_col_stride = 2;
    constexpr size_t nelems = M * N;

    float scale = 0.5f;
    int32_t zero_point = 0;

    // Source data in strided layout (4x8 matrix, we extract columns 0,2,4,6)
    std::vector<float> src_f32_full = {
      -2.0f, 99.0f, -1.5f, 99.0f, -1.0f, 99.0f, -0.5f, 99.0f,
       0.0f, 99.0f,  0.5f, 99.0f,  1.0f, 99.0f,  1.5f, 99.0f,
       2.0f, 99.0f,  2.5f, 99.0f,  3.0f, 99.0f,  3.5f, 99.0f,
       4.0f, 99.0f,  4.5f, 99.0f,  5.0f, 99.0f,  5.5f, 99.0f
    };

    std::vector<float> expected_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Source layout: [4, 8] (contiguous)");
    log_info("Logical shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Strides: [", src_row_stride, ", ", src_col_stride, "] (extract every 2nd column)");

    std::vector<int8_t> output_int8(nelems, 0);

    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.src_strides = std::vector<int64_t>{src_row_stride, src_col_stride};
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};

    log_info("scale=", scale, ", zero_point=", zero_point);

    status_t status = reorder_direct(src_f32_full.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        int32_t expected = static_cast<int32_t>(std::round(expected_f32[idx] / scale) + zero_point);
        expected = std::max(-128, std::min(127, expected));
        if (output_int8[idx] != static_cast<int8_t>(expected)) {
          log_error("Mismatch at [", i, ",", j, "]: expected ", expected,
                    ", got ", static_cast<int>(output_int8[idx]));
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("Strided 2D quantization (FP32->S8) test PASSED!");
      log_info("Successfully read strided data [stride_M=", src_row_stride, 
               ", stride_N=", src_col_stride, "]");
    } else {
      log_error("Strided 2D quantization (FP32->S8) test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_f32_to_s8_strided_3d_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Strided 3D Batched Matrix (FP32->S8)");
    log_info("========================================");

    constexpr int64_t batch = 2;
    constexpr int64_t M = 2;
    constexpr int64_t N = 3;
    constexpr int64_t src_batch_stride = 32;
    constexpr int64_t src_row_stride = 4;
    constexpr int64_t src_col_stride = 1;
    constexpr size_t nelems = batch * M * N;
    constexpr size_t src_total_size = 64;

    float scale = 0.25f;
    int32_t zero_point = 5;

    std::vector<float> src_f32_full(src_total_size);
    for (size_t i = 0; i < src_total_size; ++i) {
      src_f32_full[i] = static_cast<float>(i) * 0.1f;
    }

    std::vector<float> expected_f32 = {
      0.0f, 0.1f, 0.2f,
      0.4f, 0.5f, 0.6f,
      3.2f, 3.3f, 3.4f,
      3.6f, 3.7f, 3.8f
    };

    log_info("Source layout: [4, 4, 4] (contiguous, 64 elements)");
    log_info("Logical shape: [", batch, ", ", M, ", ", N, "] = ", nelems, " elements");
    log_info("Strides: [", src_batch_stride, ", ", src_row_stride, ", ", src_col_stride, "]");
    log_info("(Extracting batches 0,2 skipping 1,3; rows 0-1; cols 0-2)");

    std::vector<int8_t> output_int8(nelems, 0);

    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{batch, M, N};
    params.dst_shape = std::vector<int64_t>{batch, M, N};
    params.src_strides = std::vector<int64_t>{src_batch_stride, src_row_stride, src_col_stride};
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1, 1};
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1, 1};

    log_info("scale=", scale, ", zero_point=", zero_point);

    status_t status = reorder_direct(src_f32_full.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          size_t idx = b * (M * N) + i * N + j;
          int32_t expected = static_cast<int32_t>(std::round(expected_f32[idx] / scale) + zero_point);
          expected = std::max(-128, std::min(127, expected));
          if (output_int8[idx] != static_cast<int8_t>(expected)) {
            log_error("Mismatch at [batch=", b, ", ", i, ", ", j, "]: expected ", expected,
                      ", got ", static_cast<int>(output_int8[idx]),
                      " (input=", expected_f32[idx], ")");
            all_correct = false;
          }
        }
      }
    }

    if (all_correct) {
      log_info("Strided 3D quantization (FP32->S8) test PASSED!");
      log_info("Successfully read strided data [stride_batch=", src_batch_stride,
               ", stride_M=", src_row_stride, ", stride_N=", src_col_stride, "]");
    } else {
      log_error("Strided 3D quantization (FP32->S8) test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_f32_to_s8_strided_row_padding_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: Row-Padded Matrix (FP32->S8)");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 6;
    constexpr int64_t padded_row_size = 8;
    constexpr int64_t stride_M = padded_row_size;
    constexpr int64_t stride_N = 1;
    constexpr size_t logical_nelems = M * N;
    constexpr size_t physical_size = M * padded_row_size;

    float scale = 0.5f;
    int32_t zero_point = 0;

    std::vector<float> src_f32_physical(physical_size);
    std::vector<float> expected_f32(logical_nelems);

    log_info("Logical shape: [M=", M, ", N=", N, "] = ", logical_nelems, " elements");
    log_info("Physical layout: [", M, " rows x ", padded_row_size, " cols] = ", physical_size, " elements");
    log_info("Strides: [", stride_M, ", ", stride_N, "] (row padding for alignment)");

    float val = 0.0f;
    size_t expected_idx = 0;
    for (int64_t row = 0; row < M; ++row) {
      for (int64_t col = 0; col < padded_row_size; ++col) {
        size_t physical_idx = row * padded_row_size + col;
        if (col < N) {
          src_f32_physical[physical_idx] = val;
          expected_f32[expected_idx++] = val;
          val += 1.0f;
        } else {
          src_f32_physical[physical_idx] = 99.0f;
        }
      }
    }

    std::vector<int8_t> output_int8(logical_nelems, 0);

    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::s8;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.src_strides = std::vector<int64_t>{stride_M, stride_N};
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};

    log_info("Quantization: scale=", scale, ", zero_point=", zero_point);

    status_t status = reorder_direct(src_f32_physical.data(), output_int8.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        int32_t expected = static_cast<int32_t>(std::round(expected_f32[idx] / scale) + zero_point);
        expected = std::max(-128, std::min(127, expected));
        if (output_int8[idx] != static_cast<int8_t>(expected)) {
          log_error("Mismatch at [", i, ",", j, "]: expected ", expected,
                    ", got ", static_cast<int>(output_int8[idx]));
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("Row-Padded strided quantization (FP32->S8) test PASSED!");
      log_info("Successfully extracted [", M, "x", N, "] logical matrix from [", 
               M, "x", padded_row_size, "] physical layout");
      log_info("Output is contiguous: ", logical_nelems, " elements without padding");
    } else {
      log_error("Row-Padded strided quantization (FP32->S8) test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

//==============================================================================
// FP32 <-> BF16 Conversion Tests (with optional scale/zero-point)
//==============================================================================

int run_lowoha_reorder_f32_to_bf16_simple_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: FP32 to BF16 Simple Conversion (No Scale/ZP)");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    std::vector<float> input_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("No scale/zero-point - simple type conversion");

    std::vector<uint16_t> output_bf16(nelems, 0);

    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::bf16;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    // No scale/zp - leave as nullptr for simple type conversion

    status_t status = reorder_direct(input_f32.data(), output_bf16.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    // Verify by converting back to f32 and comparing
    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      // Convert bf16 back to float for comparison
      uint32_t bits = static_cast<uint32_t>(output_bf16[i]) << 16;
      float result;
      std::memcpy(&result, &bits, sizeof(result));
      
      // BF16 has limited precision, allow small error
      float expected = input_f32[i];
      float rel_error = std::abs(result - expected) / (std::abs(expected) + 1e-6f);
      if (rel_error > 0.01f) {  // Allow 1% relative error for bf16
        log_error("Mismatch at index ", i, ": expected ~", expected, ", got ", result);
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("FP32 to BF16 simple conversion test PASSED!");
    } else {
      log_error("FP32 to BF16 simple conversion test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_f32_to_bf16_with_scale_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: FP32 to BF16 with Scale/Zero-Point");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    float scale = 0.5f;
    int32_t zero_point = 2;

    std::vector<float> input_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Formula: bf16_val = bf16(f32_val / scale + zero_point)");
    log_info("scale=", scale, ", zero_point=", zero_point);

    std::vector<uint16_t> output_bf16(nelems, 0);

    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::bf16;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};

    status_t status = reorder_direct(input_f32.data(), output_bf16.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      // Expected: bf16(f32_val / scale + zp)
      float expected_f32 = input_f32[i] / scale + static_cast<float>(zero_point);
      
      // Convert output bf16 back to float
      uint32_t bits = static_cast<uint32_t>(output_bf16[i]) << 16;
      float result;
      std::memcpy(&result, &bits, sizeof(result));
      
      float rel_error = std::abs(result - expected_f32) / (std::abs(expected_f32) + 1e-6f);
      if (rel_error > 0.01f) {
        log_error("Mismatch at index ", i, ": expected ~", expected_f32, ", got ", result);
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("FP32 to BF16 with scale/zp test PASSED!");
    } else {
      log_error("FP32 to BF16 with scale/zp test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_bf16_to_f32_simple_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: BF16 to FP32 Simple Conversion (No Scale/ZP)");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    // Create BF16 values from known floats
    std::vector<float> reference_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };
    
    std::vector<uint16_t> input_bf16(nelems);
    for (size_t i = 0; i < nelems; ++i) {
      // Convert f32 to bf16
      uint32_t bits;
      std::memcpy(&bits, &reference_f32[i], sizeof(bits));
      input_bf16[i] = static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
    }

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("No scale/zero-point - simple type conversion");

    std::vector<float> output_f32(nelems, 0.0f);

    reorder_params_t params;
    params.src_dtype = data_type_t::bf16;
    params.dst_dtype = data_type_t::f32;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    // No scale/zp - leave as nullptr for simple type conversion

    status_t status = reorder_direct(input_bf16.data(), output_f32.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      float rel_error = std::abs(output_f32[i] - reference_f32[i]) / (std::abs(reference_f32[i]) + 1e-6f);
      if (rel_error > 0.01f) {
        log_error("Mismatch at index ", i, ": expected ~", reference_f32[i], ", got ", output_f32[i]);
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("BF16 to FP32 simple conversion test PASSED!");
    } else {
      log_error("BF16 to FP32 simple conversion test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_bf16_to_f32_with_scale_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: BF16 to FP32 with Scale/Zero-Point");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    float scale = 0.5f;
    int32_t zero_point = 2;

    // Create BF16 values
    std::vector<float> bf16_as_f32 = {
      -2.0f,  0.0f,  2.0f,  4.0f,
       6.0f,  8.0f, 10.0f, 12.0f,
      -1.0f,  1.0f,  3.0f,  5.0f,
       7.0f,  9.0f, 11.0f, 13.0f
    };
    
    std::vector<uint16_t> input_bf16(nelems);
    for (size_t i = 0; i < nelems; ++i) {
      uint32_t bits;
      std::memcpy(&bits, &bf16_as_f32[i], sizeof(bits));
      input_bf16[i] = static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
    }

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Formula: f32_val = (bf16_as_f32 - zero_point) * scale");
    log_info("scale=", scale, ", zero_point=", zero_point);

    std::vector<float> output_f32(nelems, 0.0f);

    reorder_params_t params;
    params.src_dtype = data_type_t::bf16;
    params.dst_dtype = data_type_t::f32;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1};
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1};

    status_t status = reorder_direct(input_bf16.data(), output_f32.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      // Expected: (bf16_val - zp) * scale
      float expected = (bf16_as_f32[i] - static_cast<float>(zero_point)) * scale;
      float rel_error = std::abs(output_f32[i] - expected) / (std::abs(expected) + 1e-6f);
      if (rel_error > 0.01f) {
        log_error("Mismatch at index ", i, ": expected ", expected, ", got ", output_f32[i]);
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("BF16 to FP32 with scale/zp test PASSED!");
    } else {
      log_error("BF16 to FP32 with scale/zp test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_f32_to_bf16_per_channel_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: FP32 to BF16 Per-Channel Conversion");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    std::vector<float> scales = {0.25f, 0.5f, 0.75f, 1.0f};
    std::vector<int32_t> zero_points = {0, 1, 2, 3};

    std::vector<float> input_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Per-channel scales: [", scales[0], ", ", scales[1], ", ", scales[2], ", ", scales[3], "]");
    log_info("Per-channel zero_points: [", zero_points[0], ", ", zero_points[1], ", ", zero_points[2], ", ", zero_points[3], "]");

    std::vector<uint16_t> output_bf16(nelems, 0);

    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::bf16;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.quant_params.scale.buff = scales.data();
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, N};
    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, N};

    status_t status = reorder_direct(input_f32.data(), output_bf16.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        float scale_j = scales[j];
        int32_t zp_j = zero_points[j];
        float expected_f32 = input_f32[idx] / scale_j + static_cast<float>(zp_j);
        
        uint32_t bits = static_cast<uint32_t>(output_bf16[idx]) << 16;
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        
        float rel_error = std::abs(result - expected_f32) / (std::abs(expected_f32) + 1e-6f);
        if (rel_error > 0.02f) {
          log_error("Mismatch at [", i, ",", j, "]: expected ~", expected_f32,
                    ", got ", result, " (scale=", scale_j, ", zp=", zp_j, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("FP32 to BF16 per-channel conversion test PASSED!");
    } else {
      log_error("FP32 to BF16 per-channel conversion test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_f32_to_bf16_per_group_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: FP32 to BF16 Per-Group Conversion");
    log_info("========================================");

    constexpr int64_t M = 8;
    constexpr int64_t N = 4;
    constexpr int64_t G = 2;
    constexpr int64_t group_size = M / G;
    constexpr size_t nelems = M * N;

    std::vector<float> scales = {
      0.25f, 0.5f, 0.75f, 1.0f,
      0.5f, 1.0f, 1.5f, 2.0f
    };
    std::vector<int32_t> zero_points = {
      0, 1, -1, 2,
      -2, 0, 1, 3
    };

    std::vector<float> input_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f,
       1.0f,  1.5f,  2.0f,  2.5f,
       3.0f,  3.5f,  4.0f,  4.5f,
       0.5f,  1.0f,  1.5f,  2.0f,
       2.5f,  3.0f,  3.5f,  4.0f
    };

    log_info("Input shape: [M=", M, ", N=", N, "] = ", nelems, " elements");
    log_info("Groups: G=", G, " groups of ", group_size, " rows each");

    std::vector<uint16_t> output_bf16(nelems, 0);

    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::bf16;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.quant_params.scale.buff = scales.data();
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{G, N};
    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{G, N};

    status_t status = reorder_direct(input_f32.data(), output_bf16.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        int64_t group_idx = i / group_size;
        size_t scale_zp_idx = group_idx * N + j;
        float scale_g = scales[scale_zp_idx];
        int32_t zp_g = zero_points[scale_zp_idx];
        float expected_f32 = input_f32[idx] / scale_g + static_cast<float>(zp_g);
        
        uint32_t bits = static_cast<uint32_t>(output_bf16[idx]) << 16;
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        
        float rel_error = std::abs(result - expected_f32) / (std::abs(expected_f32) + 1e-6f);
        if (rel_error > 0.02f) {
          log_error("Mismatch at [", i, ",", j, "] (group ", group_idx, "): expected ~", expected_f32,
                    ", got ", result, " (scale=", scale_g, ", zp=", zp_g, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("FP32 to BF16 per-group conversion test PASSED!");
    } else {
      log_error("FP32 to BF16 per-group conversion test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_bf16_to_f32_per_channel_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: BF16 to FP32 Per-Channel Conversion");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr size_t nelems = M * N;

    std::vector<float> scales = {0.25f, 0.5f, 0.75f, 1.0f};
    std::vector<int32_t> zero_points = {0, 2, -2, 4};

    // Create BF16 input values
    std::vector<float> bf16_as_f32 = {
      -4.0f,  6.0f, -2.0f,  8.0f,
       0.0f,  4.0f,  2.0f,  7.0f,
       4.0f,  5.0f,  4.0f, 10.0f,
       8.0f, 10.0f,  6.0f, 12.0f
    };
    
    std::vector<uint16_t> input_bf16(nelems);
    for (size_t i = 0; i < nelems; ++i) {
      uint32_t bits;
      std::memcpy(&bits, &bf16_as_f32[i], sizeof(bits));
      input_bf16[i] = static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
    }

    log_info("Input shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Per-channel scales: [", scales[0], ", ", scales[1], ", ", scales[2], ", ", scales[3], "]");
    log_info("Per-channel zero_points: [", zero_points[0], ", ", zero_points[1], ", ", zero_points[2], ", ", zero_points[3], "]");

    std::vector<float> output_f32(nelems, 0.0f);

    reorder_params_t params;
    params.src_dtype = data_type_t::bf16;
    params.dst_dtype = data_type_t::f32;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.quant_params.scale.buff = scales.data();
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, N};
    params.quant_params.zero_point.buff = zero_points.data();
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, N};

    status_t status = reorder_direct(input_bf16.data(), output_f32.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        float scale_j = scales[j];
        int32_t zp_j = zero_points[j];
        float expected = (bf16_as_f32[idx] - static_cast<float>(zp_j)) * scale_j;
        
        float rel_error = std::abs(output_f32[idx] - expected) / (std::abs(expected) + 1e-6f);
        if (rel_error > 0.02f) {
          log_error("Mismatch at [", i, ",", j, "]: expected ", expected,
                    ", got ", output_f32[idx], " (scale=", scale_j, ", zp=", zp_j, ")");
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("BF16 to FP32 per-channel conversion test PASSED!");
    } else {
      log_error("BF16 to FP32 per-channel conversion test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_f32_to_bf16_strided_2d_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: FP32 to BF16 Strided 2D (Simple Conversion)");
    log_info("========================================");

    constexpr int64_t M = 4;
    constexpr int64_t N = 4;
    constexpr int64_t src_row_stride = 8;
    constexpr int64_t src_col_stride = 2;
    constexpr size_t nelems = M * N;

    // Source data in strided layout (4x8 matrix, we extract columns 0,2,4,6)
    std::vector<float> src_f32_full = {
      -2.0f, 99.0f, -1.5f, 99.0f, -1.0f, 99.0f, -0.5f, 99.0f,
       0.0f, 99.0f,  0.5f, 99.0f,  1.0f, 99.0f,  1.5f, 99.0f,
       2.0f, 99.0f,  2.5f, 99.0f,  3.0f, 99.0f,  3.5f, 99.0f,
       4.0f, 99.0f,  4.5f, 99.0f,  5.0f, 99.0f,  5.5f, 99.0f
    };

    std::vector<float> expected_f32 = {
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Source layout: [4, 8] (contiguous)");
    log_info("Logical shape: [", M, ", ", N, "] = ", nelems, " elements");
    log_info("Strides: [", src_row_stride, ", ", src_col_stride, "] (extract every 2nd column)");
    log_info("No scale/zp - simple type conversion");

    std::vector<uint16_t> output_bf16(nelems, 0);

    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::bf16;
    params.src_shape = std::vector<int64_t>{M, N};
    params.dst_shape = std::vector<int64_t>{M, N};
    params.src_strides = std::vector<int64_t>{src_row_stride, src_col_stride};
    // No scale/zp - simple type conversion

    status_t status = reorder_direct(src_f32_full.data(), output_bf16.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        size_t idx = i * N + j;
        
        uint32_t bits = static_cast<uint32_t>(output_bf16[idx]) << 16;
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        
        float rel_error = std::abs(result - expected_f32[idx]) / (std::abs(expected_f32[idx]) + 1e-6f);
        if (rel_error > 0.01f) {
          log_error("Mismatch at [", i, ",", j, "]: expected ~", expected_f32[idx], ", got ", result);
          all_correct = false;
        }
      }
    }

    if (all_correct) {
      log_info("FP32 to BF16 strided 2D conversion test PASSED!");
    } else {
      log_error("FP32 to BF16 strided 2D conversion test FAILED!");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}

int run_lowoha_reorder_f32_to_bf16_batched_test() {
  try {
    log_info("========================================");
    log_info("LOWOHA Reorder: FP32 to BF16 Batched 3D with Scale/ZP");
    log_info("========================================");

    constexpr int64_t batch = 2;
    constexpr int64_t M = 2;
    constexpr int64_t N = 4;
    constexpr size_t nelems = batch * M * N;

    float scale = 0.5f;
    int32_t zero_point = 1;

    std::vector<float> input_f32 = {
      // Batch 0
      -2.0f, -1.5f, -1.0f, -0.5f,
       0.0f,  0.5f,  1.0f,  1.5f,
      // Batch 1
       2.0f,  2.5f,  3.0f,  3.5f,
       4.0f,  4.5f,  5.0f,  5.5f
    };

    log_info("Input shape: [", batch, ", ", M, ", ", N, "] = ", nelems, " elements");
    log_info("Per-tensor scale=", scale, ", zero_point=", zero_point);

    std::vector<uint16_t> output_bf16(nelems, 0);

    reorder_params_t params;
    params.src_dtype = data_type_t::f32;
    params.dst_dtype = data_type_t::bf16;
    params.src_shape = std::vector<int64_t>{batch, M, N};
    params.dst_shape = std::vector<int64_t>{batch, M, N};
    params.quant_params.scale.buff = &scale;
    params.quant_params.scale.dt = data_type_t::f32;
    params.quant_params.scale.dims = std::vector<int64_t>{1, 1, 1};
    params.quant_params.zero_point.buff = &zero_point;
    params.quant_params.zero_point.dt = data_type_t::s32;
    params.quant_params.zero_point.dims = std::vector<int64_t>{1, 1, 1};

    status_t status = reorder_direct(input_f32.data(), output_bf16.data(), params);

    if (status != status_t::success) {
      log_error("LOWOHA reorder failed!");
      return NOT_OK;
    }

    bool all_correct = true;
    for (size_t i = 0; i < nelems; ++i) {
      float expected_f32 = input_f32[i] / scale + static_cast<float>(zero_point);
      
      uint32_t bits = static_cast<uint32_t>(output_bf16[i]) << 16;
      float result;
      std::memcpy(&result, &bits, sizeof(result));
      
      float rel_error = std::abs(result - expected_f32) / (std::abs(expected_f32) + 1e-6f);
      if (rel_error > 0.02f) {
        log_error("Mismatch at index ", i, ": expected ~", expected_f32, ", got ", result);
        all_correct = false;
      }
    }

    if (all_correct) {
      log_info("FP32 to BF16 batched 3D conversion test PASSED!");
    } else {
      log_error("FP32 to BF16 batched 3D conversion test FAILED!");
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
