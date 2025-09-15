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

#include "gtest_utils_ai.hpp"
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <map>
#include <memory>
#include "operators/matmul/matmul_ref_kernel.hpp"
#include <set>

using namespace zendnnl::memory;
using namespace zendnnl::common;
using namespace zendnnl::ops;
using namespace zendnnl::error_handling;

namespace ai_gtests {
// Debug print utility
static bool ai_debug_enabled = [] {
  const char *env = std::getenv("AI_GTEST_DEBUG");
  return env && (std::string(env) == "1" || std::string(env) == "true");
}();
void AITestUtils::debug_print(const std::string &msg) {
  if (ai_debug_enabled) {
    std::cout << msg << std::endl;
  }
}

// Add tensor_map_type typedef for local use
using tensor_map_type = std::map<std::string, tensor_t>;

// Static member definitions for AITensorFactory
std::mt19937 AITensorFactory::rng(
  std::chrono::steady_clock::now().time_since_epoch().count());
//used to generate unique names for tensors
std::atomic<uint64_t> AITensorFactory::tensor_counter{0};

// AITensorFactory implementation
// -----------------------------------------------------------------------------
// fill_uniform_data
//
// Fills a raw data buffer with uniformly distributed random values according to
// the specified data type and scale. Used for initializing tensor data for tests.
//
// Parameters:
//   ptr   - pointer to the data buffer
//   nelem - number of elements to fill
//   dtype - data type of the buffer (f32, bf16, s8, etc.)
//   scale - range for uniform distribution (for floating point types)
// Usage:
//   Called by create_uniform_tensor to populate tensor data.
// -----------------------------------------------------------------------------
void AITensorFactory::fill_uniform_data(void *ptr, size_t nelem,
                                        data_type_t dtype) {
  if (nelem == 0 || ptr == nullptr) {
    return;
  }

  switch (dtype) {
  case data_type_t::f32: {
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    float *data = static_cast<float *>(ptr);
    for (size_t i = 0; i < nelem; ++i) {
      data[i] = dist(rng);
    }
    break;
  }
  case data_type_t::bf16: {
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    bfloat16_t *data = static_cast<bfloat16_t *>(ptr);
    for (size_t i = 0; i < nelem; ++i) {
      data[i] = bfloat16_t(dist(rng));
    }
    break;
  }
  case data_type_t::s8: {
    int8_t *data = static_cast<int8_t *>(ptr);
    std::uniform_int_distribution<int> int_dist(-127, 127);
    for (size_t i = 0; i < nelem; ++i) {
      data[i] = static_cast<int8_t>(int_dist(rng));
    }
    break;
  }
  case data_type_t::s4: {
    int8_t *data = static_cast<int8_t *>(ptr);
    std::uniform_int_distribution<int> int_dist(-8, 7);
    for (size_t i = 0; i < nelem; ++i) {
      data[i] = static_cast<int8_t>(int_dist(rng));
    }
    break;
  }
  case data_type_t::u8: {
    uint8_t *data = static_cast<uint8_t *>(ptr);
    std::uniform_int_distribution<int> int_dist(0, 255);
    for (size_t i = 0; i < nelem; ++i) {
      data[i] = static_cast<uint8_t>(int_dist(rng));
    }
    break;
  }
  case data_type_t::s32: {
    int32_t *data = static_cast<int32_t *>(ptr);
    std::uniform_int_distribution<int32_t> int_dist(-100000, 100000);
    for (size_t i = 0; i < nelem; ++i) {
      data[i] = int_dist(rng);
    }
    break;
  }
  default:
    std::memset(ptr, 0, nelem * sizeof(float));
    break;
  }
}

// -----------------------------------------------------------------------------
// create_uniform_tensor
//
// Creates a tensor with the given dimensions and data type, and fills it with
// uniformly distributed random values. Used for generating test tensors with
// random data for accuracy and general tests.
//
// Parameters:
//   dims  - tensor dimensions
//   dtype - data type
//   scale - range for uniform distribution
//   name  - optional tensor name
// Returns:
//   tensor_t object with allocated and filled data
// Usage:
//   Used by AITestUtils and test parameter generation functions.
// -----------------------------------------------------------------------------
tensor_t AITensorFactory::create_uniform_tensor(const std::vector<uint64_t>
    &dims,
    data_type_t dtype,
    const std::string &name) {
  if (dims.empty()) {
    throw std::invalid_argument("Tensor dimensions cannot be empty");
  }
  std::string tensor_name = name.empty() ?
                            "ai_tensor_" + std::to_string(tensor_counter.fetch_add(1)) : name;
  std::vector<tensor_t::index_type> size_vec(dims.begin(), dims.end());
  auto tensor = tensor_t()
                .set_name(tensor_name)
                .set_size(size_vec)
                .set_data_type(dtype)
                .set_storage()
                .create();
  if (!tensor.check()) {
    std::cerr << "[ERROR] Failed to create tensor: " << tensor_name << std::endl;
    throw std::runtime_error("Failed to create tensor: " + tensor_name);
  }
  size_t nelem = tensor.get_nelem();
  if (nelem == 0) {
    return tensor;
  }
  void *ptr = tensor.get_raw_handle_unsafe();
  if (!ptr) {
    std::cerr << "[ERROR] Null data pointer for tensor: " << tensor_name <<
              std::endl;
    throw std::runtime_error("Null data pointer for tensor: " + tensor_name);
  }
  fill_uniform_data(ptr, nelem, dtype);
  return tensor;
}

// -----------------------------------------------------------------------------
// create_zero_tensor
//
// Creates a tensor with the given dimensions and data type, and fills it with
// zeros. Used for generating zero-initialized tensors for tests.
//
// Parameters:
//   dims  - tensor dimensions
//   dtype - data type
//   name  - optional tensor name
// Returns:
//   tensor_t object with zero-filled data
// Usage:
//   Used by AITestUtils and test parameter generation functions.
// -----------------------------------------------------------------------------
tensor_t AITensorFactory::create_zero_tensor(const std::vector<uint64_t> &dims,
    data_type_t dtype,
    const std::string &name) {
  std::string tensor_name = name.empty() ?
                            "ai_zero_tensor_" + std::to_string(tensor_counter.fetch_add(1)) : name;
  std::vector<tensor_t::index_type> size_vec(dims.begin(), dims.end());
  auto tensor = tensor_t()
                .set_name(tensor_name)
                .set_size(size_vec)
                .set_data_type(dtype)
                .set_storage()
                .create();
  if (!tensor.check()) {
    std::cerr << "[ERROR] Failed to create tensor: " << tensor_name << std::endl;
    throw std::runtime_error("Failed to create tensor: " + tensor_name);
  }
  auto buf_size = tensor.get_buffer_sz_bytes();
  void *ptr = tensor.get_raw_handle_unsafe();
  if (buf_size > 0 && !ptr) {
    std::cerr << "[ERROR] Null data pointer for tensor: " << tensor_name <<
              std::endl;
    throw std::runtime_error("Null data pointer for tensor: " + tensor_name);
  }
  if (buf_size > 0) {
    std::memset(ptr, 0, buf_size);
  }
  return tensor;
}


// -----------------------------------------------------------------------------
// fill_boundary_data
//
// Fills a raw data buffer with boundary values for stress-testing matmul numerical stability.
// The values alternate between large, small, positive, and negative values for each supported type.
// Used by create_boundary_tensor to populate tensor data.
//
// Parameters:
//   ptr   - pointer to the data buffer
//   nelem - number of elements to fill
//   dtype - data type of the buffer (f32, bf16, s8, s4)
// Usage:
//   Called by create_boundary_tensor to fill tensor data with boundary values.
// -----------------------------------------------------------------------------
void AITensorFactory::fill_boundary_data(void *ptr, size_t nelem,
    data_type_t dtype) {
  if (nelem == 0 || ptr == nullptr) {
    return;
  }
  auto fill_pattern = [&](auto* data, const auto& pattern, size_t pattern_len) {
    for (size_t i = 0; i < nelem; ++i) {
      data[i] = pattern[i % pattern_len];
    }
  };
  if (dtype == data_type_t::f32) {
    static const float pattern[] = {1.0f, -1.0f, 1e-7f, -1e-7f};
    fill_pattern(static_cast<float *>(ptr), pattern,
                 sizeof(pattern)/sizeof(pattern[0]));
  }
  else if (dtype == data_type_t::bf16) {
    static const bfloat16_t pattern[] = {bfloat16_t(1.0f), bfloat16_t(-1.0f), bfloat16_t(1e-3f), bfloat16_t(-1e-3f)};
    fill_pattern(static_cast<bfloat16_t *>(ptr), pattern,
                 sizeof(pattern)/sizeof(pattern[0]));
  }
  else if (dtype == data_type_t::s8) {
    static const int8_t pattern[] = {127, -127, 1, -1};
    fill_pattern(static_cast<int8_t *>(ptr), pattern,
                 sizeof(pattern)/sizeof(pattern[0]));
  }
  else if (dtype == data_type_t::s4) {
    static const int8_t pattern[] = {7, -8, 1, -1};
    fill_pattern(static_cast<int8_t *>(ptr), pattern,
                 sizeof(pattern)/sizeof(pattern[0]));
  }
}

// -----------------------------------------------------------------------------
// create_boundary_tensor
//
// Creates a tensor with the given dimensions and data type, and fills it with
// special boundary values for stress-testing matmul numerical stability and correctness.
// Calls fill_boundary_data to populate the tensor buffer.
// -----------------------------------------------------------------------------
tensor_t AITensorFactory::create_boundary_tensor(const std::vector<uint64_t>
    &dims,
    data_type_t dtype,
    const std::string &name) {
  std::string tensor_name = name.empty() ?
                            "ai_boundary_tensor_" + std::to_string(tensor_counter.fetch_add(1)) : name;
  std::vector<tensor_t::index_type> size_vec(dims.begin(), dims.end());
  auto tensor = tensor_t()
                .set_name(tensor_name)
                .set_size(size_vec)
                .set_data_type(dtype)
                .set_storage()
                .create();
  if (!tensor.check()) {
    std::cerr << "[ERROR] Failed to create tensor: " << tensor_name << std::endl;
    throw std::runtime_error("Failed to create tensor: " + tensor_name);
  }
  size_t nelem = tensor.get_nelem();
  void *ptr = tensor.get_raw_handle_unsafe();
  if (nelem > 0 && !ptr) {
    std::cerr << "[ERROR] Null data pointer for tensor: " << tensor_name <<
              std::endl;
    throw std::runtime_error("Null data pointer for tensor: " + tensor_name);
  }
  if (nelem > 0) {
    fill_boundary_data(ptr, nelem, dtype);
  }
  return tensor;
}


// AITestUtils static member definitions
std::mt19937 AITestUtils::rng(
  std::chrono::steady_clock::now().time_since_epoch().count());

// AITestUtils implementation
// -----------------------------------------------------------------------------
// get_input_dtype
//
// Returns the input tensor data type for a given data type combination.
//
// Parameters:
//   combo - DataTypeCombination enum value
// Returns:
//   data_type_t for input tensor
// Usage:
//   Used in tensor creation and test parameter setup.
// -----------------------------------------------------------------------------
data_type_t AITestUtils::get_input_dtype(DataTypeCombination combo) {
  switch (combo) {
  case DataTypeCombination::F32_F32_F32:
  case DataTypeCombination::F32_BF16_F32:
    return data_type_t::f32;
  case DataTypeCombination::BF16_BF16_BF16:
  case DataTypeCombination::BF16_F32_BF16:
    return data_type_t::bf16;
  default:
    return data_type_t::f32;
  }
}

// -----------------------------------------------------------------------------
// get_weight_dtype
//
// Returns the weight tensor data type for a given data type combination.
//
// Parameters:
//   combo - DataTypeCombination enum value
// Returns:
//   data_type_t for weight tensor
// Usage:
//   Used in tensor creation and test parameter setup.
// -----------------------------------------------------------------------------
data_type_t AITestUtils::get_weight_dtype(DataTypeCombination combo) {
  switch (combo) {
  case DataTypeCombination::F32_F32_F32:
  case DataTypeCombination::BF16_F32_BF16:
    return data_type_t::f32;
  case DataTypeCombination::BF16_BF16_BF16:
  case DataTypeCombination::F32_BF16_F32:
    return data_type_t::bf16;
  default:
    return data_type_t::f32;
  }
}

// -----------------------------------------------------------------------------
// get_output_dtype
//
// Returns the output tensor data type for a given data type combination.
//
// Parameters:
//   combo - DataTypeCombination enum value
// Returns:
//   data_type_t for output tensor
// Usage:
//   Used in tensor creation and test parameter setup.
// -----------------------------------------------------------------------------
data_type_t AITestUtils::get_output_dtype(DataTypeCombination combo) {
  switch (combo) {
  case DataTypeCombination::F32_F32_F32:
  case DataTypeCombination::F32_BF16_F32:
    return data_type_t::f32;
  case DataTypeCombination::BF16_BF16_BF16:
  case DataTypeCombination::BF16_F32_BF16:
    return data_type_t::bf16;
  default:
    return data_type_t::f32;
  }
}


// -----------------------------------------------------------------------------
// validate_dimensions
//
// Checks if the provided matrix dimensions are within the valid range for tests.
//
// Parameters:
//   m, n, k - matrix dimensions
// Returns:
//   true if all dimensions are valid, false otherwise
// Usage:
//   Used in test parameter generation and validation logic.
// -----------------------------------------------------------------------------
bool AITestUtils::validate_dimensions(uint64_t m, uint64_t n, uint64_t k) {
  return (m >= AI_MIN_DIM && n >= AI_MIN_DIM && k >= AI_MIN_DIM &&
          m <= AI_MAX_DIM && n <= AI_MAX_DIM && k <= AI_MAX_DIM);
}

// -----------------------------------------------------------------------------
// generate_unique_name
//
// Generates a unique name string with the given prefix, for naming tensors or
// test cases to avoid collisions.
//
// Parameters:
//   prefix - string prefix for the name
// Returns:
//   unique name string
// Usage:
//   Used in tensor and test parameter creation.
// -----------------------------------------------------------------------------
std::string AITestUtils::generate_unique_name(const std::string &prefix) {
  static std::atomic<uint64_t> counter{0};
  return prefix + "_" + std::to_string(counter.fetch_add(1));
}

// -----------------------------------------------------------------------------
// log_tensor_info
//
// Logs the shape and data type of a tensor to stdout for debugging.
//
// Parameters:
//   tensor - tensor_t object
//   name   - name/label for the tensor
// Usage:
//   Used in test logic for debugging and validation.
// -----------------------------------------------------------------------------
void AITestUtils::log_tensor_info(const tensor_t &tensor,
                                  const std::string &name) {
  auto size_vec = tensor.get_size();
  std::cout << "[INFO] Tensor " << name << ": ";
  for (size_t i = 0; i < size_vec.size(); ++i) {
    std::cout << size_vec[i];
    if (i < size_vec.size() - 1) {
      std::cout << "x";
    }
  }
  std::cout << ", dtype: " << static_cast<int>(tensor.get_data_type()) <<
            std::endl;

  // Print values for 2D tensors (row-major)
  if (size_vec.size() == 2) {
    size_t rows = size_vec[0];
    size_t cols = size_vec[1];
    if (tensor.get_data_type() == data_type_t::f32) {
      const float *data = static_cast<const float *>(tensor.get_raw_handle_const());
      for (size_t i = 0; i < rows; ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < cols; ++j) {
          std::cout << data[i * cols + j];
          if (j < cols - 1) {
            std::cout << ", ";
          }
        }
        std::cout << "]" << std::endl;
      }
    }
    else if (tensor.get_data_type() == data_type_t::bf16) {
      const bfloat16_t *data = static_cast<const bfloat16_t *>
                               (tensor.get_raw_handle_const());
      for (size_t i = 0; i < rows; ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < cols; ++j) {
          std::cout << static_cast<float>(data[i * cols + j]);
          if (j < cols - 1) {
            std::cout << ", ";
          }
        }
        std::cout << "]" << std::endl;
      }
    }
    else {
      std::cout << "  [Tensor value printing not implemented for dtype " <<
                static_cast<int>(tensor.get_data_type()) << "]" << std::endl;
    }
  }
}

// -----------------------------------------------------------------------------
// is_aocl_kernel_supported
//
// Checks if AOCL BLIS kernel is supported for given data types and post-ops.
// Returns true if supported, false otherwise.
// -----------------------------------------------------------------------------
bool AITestUtils::is_aocl_kernel_supported(data_type_t input_dtype,
    data_type_t weight_dtype,
    data_type_t output_dtype,
    const std::vector<post_op_type_t> &post_ops) {

  static const std::set<post_op_type_t> supported_post_ops = {
    post_op_type_t::relu,
    post_op_type_t::leaky_relu,
    post_op_type_t::gelu_tanh,
    post_op_type_t::gelu_erf,
    post_op_type_t::tanh,
    post_op_type_t::swish,
    post_op_type_t::sigmoid,
    post_op_type_t::clip,
    post_op_type_t::binary_add,
    post_op_type_t::binary_mul
  };
  for (auto op : post_ops) {
    if (supported_post_ops.find(op) == supported_post_ops.end()) {
      return false;
    }
  }
  if (input_dtype == data_type_t::f32 &&
      weight_dtype == data_type_t::f32 &&
      output_dtype == data_type_t::f32) {
    return true;
  }
  if (input_dtype == data_type_t::bf16 &&
      weight_dtype == data_type_t::bf16 &&
      (output_dtype == data_type_t::f32 || output_dtype == data_type_t::bf16)) {
    return true;
  }
  return false;
}

// -----------------------------------------------------------------------------
// is_reference_implementation_supported
//
// Checks if reference implementation is supported for given data types and post-ops.
// Returns true if supported, false otherwise.
// -----------------------------------------------------------------------------
bool AITestUtils::is_reference_implementation_supported(data_type_t input_dtype,
    data_type_t weight_dtype,
    data_type_t output_dtype,
    const std::vector<post_op_type_t> &post_ops) {
  bool dtype_supported = false;
  if (input_dtype == data_type_t::f32 &&
      weight_dtype == data_type_t::f32 &&
      output_dtype == data_type_t::f32) {
    dtype_supported = true;
  }
  if (input_dtype == data_type_t::bf16 &&
      weight_dtype == data_type_t::bf16 &&
      (output_dtype == data_type_t::f32 || output_dtype == data_type_t::bf16)) {
    dtype_supported = true;
  }
  if (input_dtype == data_type_t::s8 &&
      weight_dtype == data_type_t::s8 &&
      (output_dtype == data_type_t::s8 || output_dtype == data_type_t::f32)) {
    dtype_supported = true;
  }
  if (input_dtype == data_type_t::f32 &&
      weight_dtype == data_type_t::bf16 &&
      output_dtype == data_type_t::f32) {
    dtype_supported = true;
  }
  if (input_dtype == data_type_t::bf16 &&
      weight_dtype == data_type_t::f32 &&
      output_dtype == data_type_t::f32) {
    dtype_supported = true;
  }
  static const std::set<post_op_type_t> ref_supported_post_ops = {
    post_op_type_t::elu,
    post_op_type_t::relu,
    post_op_type_t::leaky_relu,
    post_op_type_t::gelu_tanh,
    post_op_type_t::gelu_erf,
    post_op_type_t::swish,
    post_op_type_t::sigmoid,
    post_op_type_t::tanh,
    post_op_type_t::softmax,
    post_op_type_t::square,
    post_op_type_t::abs,
    post_op_type_t::sqrt,
    post_op_type_t::exp,
    post_op_type_t::log,
    post_op_type_t::clip,
    post_op_type_t::binary_add,
    post_op_type_t::binary_mul
  };
  for (auto op : post_ops) {
    if (ref_supported_post_ops.find(op) == ref_supported_post_ops.end()) {
      return false;
    }
  }
  return dtype_supported;
}

// -----------------------------------------------------------------------------
// get_sample_indices
//
// Selects a set of indices for sampling elements from a tensor for validation.
// If the tensor is small, returns all indices; otherwise, randomly samples up to
// max_samples unique indices. Used to efficiently validate large tensors without
// comparing every element, improving test performance while maintaining coverage.
//
// Parameters:
//   total_elements - total number of elements in the tensor
//   max_samples    - maximum number of indices to sample
// Returns:
//   Vector of sampled indices
// Usage:
//   Used by compare_sampled_tensors and validation functions to select elements
//   for comparison between tensors.
// -----------------------------------------------------------------------------
std::vector<size_t> AITestUtils::get_sample_indices(size_t total_elements,
    size_t max_samples) {
  std::vector<size_t> indices;

  if (total_elements <= max_samples) {
    // Return all indices if total is small
    for (size_t i = 0; i < total_elements; ++i) {
      indices.push_back(i);
    }
  }
  else {
    // Sample random indices
    std::uniform_int_distribution<size_t> dist(0, total_elements - 1);
    std::unordered_set<size_t> sampled;

    while (sampled.size() < max_samples) {
      sampled.insert(dist(rng));
    }

    indices.assign(sampled.begin(), sampled.end());
    std::sort(indices.begin(), indices.end());
  }

  return indices;
}

// -----------------------------------------------------------------------------
// compare_sampled_tensors
//
// Compares two tensors by sampling a subset of their elements and checking if
// the absolute difference for each sampled element is within the specified tolerance.
// This function is used to validate tensor outputs efficiently, especially for large
// tensors, by avoiding full element-wise comparison. Returns false if any sampled
// element pair exceeds the tolerance or if tensor shapes/types mismatch.
//
// Parameters:
//   tensor1    - first tensor to compare
//   tensor2    - second tensor to compare
//   tolerance  - allowed absolute difference for each element
// Returns:
//   true if all sampled elements are within tolerance, false otherwise
// Usage:
//   Used in output validation and test result checks for matmul tests.
// -----------------------------------------------------------------------------
bool AITestUtils::compare_sampled_tensors(const tensor_t &test_tensor,
    const tensor_t &ref_tensor,
    float abs_tolerance,
    float rel_tolerance) {
  if (test_tensor.get_nelem() != ref_tensor.get_nelem()) {
    return false;
  }

  if (test_tensor.get_data_type() != ref_tensor.get_data_type()) {
    return false;
  }

  size_t total_elements = test_tensor.get_nelem();
  auto sample_indices = get_sample_indices(total_elements,
                        AI_MAX_VALIDATION_ELEMENTS);

  // Generic comparison for all supported datatypes
  auto dtype = test_tensor.get_data_type();
  for (size_t idx : sample_indices) {
    float v1 = 0.0f, v2 = 0.0f;
    switch (dtype) {
    case data_type_t::f32: {
      const float *data1 = static_cast<const float *>
                           (test_tensor.get_raw_handle_const());
      const float *data2 = static_cast<const float *>
                           (ref_tensor.get_raw_handle_const());
      v1 = data1[idx];
      v2 = data2[idx];
      break;
    }
    case data_type_t::bf16: {
      const bfloat16_t *data1 = static_cast<const bfloat16_t *>
                                (test_tensor.get_raw_handle_const());
      const bfloat16_t *data2 = static_cast<const bfloat16_t *>
                                (ref_tensor.get_raw_handle_const());
      v1 = static_cast<float>(data1[idx]);
      v2 = static_cast<float>(data2[idx]);
      break;
    }
    case data_type_t::s8: {
      const int8_t *data1 = static_cast<const int8_t *>
                            (test_tensor.get_raw_handle_const());
      const int8_t *data2 = static_cast<const int8_t *>
                            (ref_tensor.get_raw_handle_const());
      v1 = static_cast<float>(data1[idx]);
      v2 = static_cast<float>(data2[idx]);
      break;
    }
    case data_type_t::s4: {
      const int8_t *data1 = static_cast<const int8_t *>
                            (test_tensor.get_raw_handle_const());
      const int8_t *data2 = static_cast<const int8_t *>
                            (ref_tensor.get_raw_handle_const());
      v1 = static_cast<float>(data1[idx]);
      v2 = static_cast<float>(data2[idx]);
      break;
    }
    case data_type_t::u8: {
      const uint8_t *data1 = static_cast<const uint8_t *>
                             (test_tensor.get_raw_handle_const());
      const uint8_t *data2 = static_cast<const uint8_t *>
                             (ref_tensor.get_raw_handle_const());
      v1 = static_cast<float>(data1[idx]);
      v2 = static_cast<float>(data2[idx]);
      break;
    }
    case data_type_t::s32: {
      const int32_t *data1 = static_cast<const int32_t *>
                             (test_tensor.get_raw_handle_const());
      const int32_t *data2 = static_cast<const int32_t *>
                             (ref_tensor.get_raw_handle_const());
      v1 = static_cast<float>(data1[idx]);
      v2 = static_cast<float>(data2[idx]);
      break;
    }
    default:
      // Unknown/unsupported type, treat as mismatch
      return false;
    }
    float diff = std::abs(v1 - v2);
    float tol = abs_tolerance + rel_tolerance * std::abs(
                  v2); // PyTorch formula: atol + rtol * |b|
    if (!(diff <= tol || (std::isnan(diff) && std::isnan(v2)))) {
      return false;
    }
  }

  return true;
}

// -----------------------------------------------------------------------------
// compare_sampled_tensors_matmul
//
// Compares two tensors by sampling a subset of their elements and checking if
// the absolute difference for each sampled element is within a calculated bound
// (abs_bound) plus a relative tolerance, using the logic from compare_tensor_2D_matrix.
// Returns false if any sampled element pair exceeds the tolerance or if tensor shapes/types mismatch.
// -----------------------------------------------------------------------------
bool AITestUtils::compare_sampled_tensors_matmul(const tensor_t &test_tensor,
    const tensor_t &ref_tensor,
    uint64_t k,
    float rel_tolerance,
    float epsilon) {
  if (test_tensor.get_nelem() != ref_tensor.get_nelem()) {
    return false;
  }

  if (test_tensor.get_data_type() != ref_tensor.get_data_type()) {
    return false;
  }

  size_t total_elements = test_tensor.get_nelem();
  auto sample_indices = get_sample_indices(total_elements,
                        AI_MAX_VALIDATION_ELEMENTS);

  constexpr int C = 20; // Margin for F32:: tolerance
  // ToDo: Add P value according to the postop currently, same value is used for all.
  constexpr int P = 15; // to handle postop accumulation error
  constexpr int scale_factor = 4; // scale factor
  float abs_bound = 0.0f;
  auto dtype = test_tensor.get_data_type();
  if (dtype == data_type_t::bf16) {
    abs_bound = k * epsilon;
  }
  else {
    abs_bound = ((C + std::log2(static_cast<float>(k))/scale_factor) * k + P) *
                epsilon;
  }

  for (size_t idx : sample_indices) {
    float v1 = 0.0f, v2 = 0.0f;
    switch (dtype) {
    case data_type_t::f32: {
      const float *data1 = static_cast<const float *>
                           (test_tensor.get_raw_handle_const());
      const float *data2 = static_cast<const float *>
                           (ref_tensor.get_raw_handle_const());
      v1 = data1[idx];
      v2 = data2[idx];
      break;
    }
    case data_type_t::bf16: {
      const bfloat16_t *data1 = static_cast<const bfloat16_t *>
                                (test_tensor.get_raw_handle_const());
      const bfloat16_t *data2 = static_cast<const bfloat16_t *>
                                (ref_tensor.get_raw_handle_const());
      v1 = static_cast<float>(data1[idx]);
      v2 = static_cast<float>(data2[idx]);
      break;
    }
    case data_type_t::s8: {
      const int8_t *data1 = static_cast<const int8_t *>
                            (test_tensor.get_raw_handle_const());
      const int8_t *data2 = static_cast<const int8_t *>
                            (ref_tensor.get_raw_handle_const());
      v1 = static_cast<float>(data1[idx]);
      v2 = static_cast<float>(data2[idx]);
      break;
    }
    case data_type_t::s4: {
      const int8_t *data1 = static_cast<const int8_t *>
                            (test_tensor.get_raw_handle_const());
      const int8_t *data2 = static_cast<const int8_t *>
                            (ref_tensor.get_raw_handle_const());
      v1 = static_cast<float>(data1[idx]);
      v2 = static_cast<float>(data2[idx]);
      break;
    }
    case data_type_t::u8: {
      const uint8_t *data1 = static_cast<const uint8_t *>
                             (test_tensor.get_raw_handle_const());
      const uint8_t *data2 = static_cast<const uint8_t *>
                             (ref_tensor.get_raw_handle_const());
      v1 = static_cast<float>(data1[idx]);
      v2 = static_cast<float>(data2[idx]);
      break;
    }
    case data_type_t::s32: {
      const int32_t *data1 = static_cast<const int32_t *>
                             (test_tensor.get_raw_handle_const());
      const int32_t *data2 = static_cast<const int32_t *>
                             (ref_tensor.get_raw_handle_const());
      v1 = static_cast<float>(data1[idx]);
      v2 = static_cast<float>(data2[idx]);
      break;
    }
    default:
      return false;
    }
    float abs_err = std::fabs(v2 - v1);
    float tol = abs_bound + rel_tolerance * std::fabs(v2);
    if (abs_err > tol) {
      return false;
    }
  }
  return true;
}

// -----------------------------------------------------------------------------
// run_reference_matmul
//
// Runs the reference matmul kernel with the given input, weights, bias, output,
// post-op configuration, and any binary post-op tensors. Used to validate
// correctness of the reference implementation.
//
// Parameters:
//   input, weights, bias, output - tensors for matmul
//   post_op_config              - post-op configuration
//   binary_postop_tensors       - tensors for binary post-ops
// Returns:
//   status_t indicating success or failure
// Usage:
//   Used in accuracy and reference kernel tests for validation.
// -----------------------------------------------------------------------------
status_t AITestUtils::run_reference_matmul(
  tensor_t &input, tensor_t &weights, tensor_t &bias,
  tensor_t &output, const PostOpConfig &post_op_config,
  std::vector<tensor_t> &binary_postop_tensors) {
  try {
    // Prepare input and output maps for the reference kernel
    tensor_map_type inputs;
    tensor_map_type outputs;
    inputs["matmul_input"] = input;
    outputs["matmul_output"] = output;

    // Create context and set parameters (weights/bias must be non-const)
    tensor_t weights_copy = weights;
    tensor_t bias_copy = bias;
    weights_copy.set_name("weights");
    bias_copy.set_name("bias");
    auto matmul_context = matmul_context_t()
                          .set_param("weights", weights_copy)
                          .set_param("bias", bias_copy);
    for (const auto &post_op_type : post_op_config.post_ops) {
      post_op_t post_op{post_op_type};
      matmul_context = matmul_context.set_post_op(post_op);
    }
    matmul_context = matmul_context.create();
    if (!matmul_context.check()) {
      return status_t::failure;
    }

    // Bind binary post-op tensors using the actual tensor names from the context
    size_t binary_tensor_idx = 0;
    for (size_t i = 0; i < post_op_config.post_ops.size(); ++i) {
      auto post_op_type = post_op_config.post_ops[i];
      if ((post_op_type == post_op_type_t::binary_add ||
           post_op_type == post_op_type_t::binary_mul)
          && binary_tensor_idx < binary_postop_tensors.size()) {
        std::string tensor_name;
        try {
          if (post_op_type == post_op_type_t::binary_add) {
            tensor_name = matmul_context.get_post_op(i).binary_add_params.tensor_name;
          }
          else {
            tensor_name = matmul_context.get_post_op(i).binary_mul_params.tensor_name;
          }
        }
        catch (...) {
          tensor_name = "binary_post_op_tensor";
        }
        inputs[tensor_name] = binary_postop_tensors[binary_tensor_idx];
        ++binary_tensor_idx;
      }
    }

    // Create the matmul operator and force reference kernel
    auto matmul_operator = matmul_operator_t()
                           .set_name("matmul_forced_ref_operator")
                           .set_context(matmul_context)
                           .create();
    if (!matmul_operator.check()) {
      return status_t::failure;
    }

    // Set all inputs
    matmul_operator = matmul_operator.set_input("matmul_input", input);
    // When binding tensors to the operator, ensure non-const references are used
    for (auto &kv : inputs) {
      matmul_operator = matmul_operator.set_input(kv.first, kv.second);
    }
    matmul_operator = matmul_operator.set_output("matmul_output", output);

    // Force reference kernel
    matmul_operator = matmul_operator.set_forced_kernel("reference");

    // Execute
    status_t status = matmul_operator.execute();
    return status;
  }
  catch (const std::exception &e) {
    std::cerr << "[AI_REF] Exception in run_reference_matmul: " << e.what() <<
              std::endl;
    return status_t::failure;
  }
  catch (...) {
    std::cerr << "[AI_REF] Unknown exception in run_reference_matmul" << std::endl;
    return status_t::failure;
  }
}

// -----------------------------------------------------------------------------
// is_valid_data_type_combination
//
// Checks if the given data type combination is supported for matmul tests.
//
// Parameters:
//   combo - DataTypeCombination enum value
// Returns:
//   true if supported, false otherwise
// Usage:
//   Used in parameter generation and validation.
// -----------------------------------------------------------------------------
bool AITestUtils::is_valid_data_type_combination(DataTypeCombination combo) {
  // Only allow supported combinations
  const auto &supported = ParameterGenerator::supported_combinations;
  return std::find(supported.begin(), supported.end(), combo) != supported.end();
}

// -----------------------------------------------------------------------------
// get_all_post_op_configs
//
// Returns a vector of all possible post-op configurations for matmul tests,
// including every single supported post-op as a separate config, and all
// multi-post-op configs used in the test suite.
//
// Returns:
//   Vector of PostOpConfig objects (one per post-op and multi-op chain)
// Usage:
//   Used for exhaustive parameter generation and coverage.
// -----------------------------------------------------------------------------
std::vector<PostOpConfig> AITestUtils::get_all_post_op_configs() {
  std::vector<PostOpConfig> configs;
  // Single post-op configs (all supported post-ops)
  configs.push_back(PostOpConfig{}); // No post-op
  configs.push_back(create_elu_config());
  configs.push_back(create_relu_config());
  configs.push_back(create_leaky_relu_config());
  configs.push_back(create_gelu_tanh_config());
  configs.push_back(create_gelu_erf_config());
  configs.push_back(create_silu_config()); // swish
  configs.push_back(create_sigmoid_config());
  configs.push_back(create_tanh_config());
  configs.push_back(create_softmax_config());
  configs.push_back(create_square_config());
  configs.push_back(create_abs_config());
  configs.push_back(create_sqrt_config());
  configs.push_back(create_exp_config());
  configs.push_back(create_log_config());
  configs.push_back(create_clip_config());
  configs.push_back(create_binary_add_config());
  configs.push_back(create_binary_mul_config());
  // Multi-post-op configs (chains)
  //configs.push_back(create_mixed_post_op_config());
  //configs.push_back(create_relu_clip_config());
  //configs.push_back(create_binary_add_mul_config());
  return configs;
}
// -----------------------------------------------------------------------------
// create_gelu_tanh_config
//
// Returns a PostOpConfig for gelu_tanh post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_gelu_tanh_config() {
  PostOpConfig cfg;
  cfg.config_name = "gelu_tanh";
  cfg.post_ops = {post_op_type_t::gelu_tanh};
  return cfg;
}

// -----------------------------------------------------------------------------
// create_gelu_erf_config
//
// Returns a PostOpConfig for gelu_erf post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_gelu_erf_config() {
  PostOpConfig cfg;
  cfg.config_name = "gelu_erf";
  cfg.post_ops = {post_op_type_t::gelu_erf};
  return cfg;
}

// -----------------------------------------------------------------------------
// create_sigmoid_config
//
// Returns a PostOpConfig for sigmoid post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_sigmoid_config() {
  PostOpConfig cfg;
  cfg.config_name = "sigmoid";
  cfg.post_ops = {post_op_type_t::sigmoid};
  return cfg;
}

// -----------------------------------------------------------------------------
// create_tanh_config
//
// Returns a PostOpConfig for tanh post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_tanh_config() {
  PostOpConfig cfg;
  cfg.config_name = "tanh";
  cfg.post_ops = {post_op_type_t::tanh};
  return cfg;
}

// -----------------------------------------------------------------------------
// create_clip_config
//
// Returns a PostOpConfig for clip post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_clip_config() {
  PostOpConfig cfg;
  cfg.config_name = "clip";
  cfg.post_ops = {post_op_type_t::clip};
  return cfg;
}

// -----------------------------------------------------------------------------
// create_relu_clip_config
//
// Returns a PostOpConfig for a chain of relu followed by clip.
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_relu_clip_config() {
  PostOpConfig cfg;
  cfg.config_name = "relu_clip";
  cfg.post_ops = {post_op_type_t::relu, post_op_type_t::clip};
  return cfg;
}

// -----------------------------------------------------------------------------
// create_binary_add_mul_config
//
// Returns a PostOpConfig for a chain of binary_add followed by binary_mul.
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_binary_add_mul_config() {
  PostOpConfig cfg;
  cfg.config_name = "binary_add_mul";
  cfg.post_ops = {post_op_type_t::binary_add, post_op_type_t::binary_mul};
  return cfg;
}

// -----------------------------------------------------------------------------
// create_binary_add_config
//
// Returns a PostOpConfig for binary add post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_binary_add_config() {
  PostOpConfig cfg;
  cfg.config_name = "binary_add";
  cfg.post_ops = {post_op_type_t::binary_add};
  return cfg;
}

// -----------------------------------------------------------------------------
// create_binary_mul_config
//
// Returns a PostOpConfig for binary multiply post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_binary_mul_config() {
  PostOpConfig cfg;
  cfg.config_name = "binary_mul";
  cfg.post_ops = {post_op_type_t::binary_mul};
  return cfg;
}

// -----------------------------------------------------------------------------
// create_relu_config
//
// Returns a PostOpConfig for ReLU post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_relu_config() {
  PostOpConfig cfg;
  cfg.config_name = "relu";
  cfg.post_ops = {post_op_type_t::relu};
  return cfg;
}

// -----------------------------------------------------------------------------
// create_silu_config
//
// Returns a PostOpConfig for SiLU/Swish post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_silu_config() {
  PostOpConfig cfg;
  cfg.config_name = "silu";
  // SiLU is equivalent to Swish, which is supported as post_op_type_t::swish
  cfg.post_ops = std::vector<post_op_type_t> {post_op_type_t::swish};
  return cfg;
}

// -----------------------------------------------------------------------------
// create_mixed_post_op_config
//
// Returns a PostOpConfig for a mixed chain of post-ops (binary add, relu, swish).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_mixed_post_op_config() {
  PostOpConfig cfg;
  cfg.config_name = "mixed";
  // Use swish instead of silu, as silu is not a valid enum value
  cfg.post_ops = std::vector<post_op_type_t> {post_op_type_t::binary_add, post_op_type_t::relu, post_op_type_t::swish};
  return cfg;
}

// -----------------------------------------------------------------------------
// create_softmax_config
//
// Returns a PostOpConfig for softmax post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_softmax_config() {
  PostOpConfig cfg;
  cfg.config_name = "softmax";
  cfg.post_ops = {post_op_type_t::softmax};
  return cfg;
}
// -----------------------------------------------------------------------------
// create_abs_config
//
// Returns a PostOpConfig for abs post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_abs_config() {
  PostOpConfig cfg;
  cfg.config_name = "abs";
  cfg.post_ops = {post_op_type_t::abs};
  return cfg;
}
// -----------------------------------------------------------------------------
// create_square_config
//
// Returns a PostOpConfig for square post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_square_config() {
  PostOpConfig cfg;
  cfg.config_name = "square";
  cfg.post_ops = {post_op_type_t::square};
  return cfg;
}
// -----------------------------------------------------------------------------
// create_sqrt_config
//
// Returns a PostOpConfig for sqrt post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_sqrt_config() {
  PostOpConfig cfg;
  cfg.config_name = "sqrt";
  cfg.post_ops = {post_op_type_t::sqrt};
  return cfg;
}
// -----------------------------------------------------------------------------
// create_exp_config
//
// Returns a PostOpConfig for exp post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_exp_config() {
  PostOpConfig cfg;
  cfg.config_name = "exp";
  cfg.post_ops = {post_op_type_t::exp};
  return cfg;
}
// -----------------------------------------------------------------------------
// create_log_config
//
// Returns a PostOpConfig for log post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_log_config() {
  PostOpConfig cfg;
  cfg.config_name = "log";
  cfg.post_ops = {post_op_type_t::log};
  return cfg;
}
// -----------------------------------------------------------------------------
// create_leaky_relu_config
//
// Returns a PostOpConfig for leaky ReLU post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_leaky_relu_config() {
  PostOpConfig cfg;
  cfg.config_name = "leaky_relu";
  cfg.post_ops = {post_op_type_t::leaky_relu};
  return cfg;
}
// -----------------------------------------------------------------------------
// create_elu_config
//
// Returns a PostOpConfig for ELU post-op (with bias).
// Usage:
//   Used in parameter generation and test setup.
// -----------------------------------------------------------------------------
PostOpConfig AITestUtils::create_elu_config() {
  PostOpConfig cfg;
  cfg.config_name = "elu";
  cfg.post_ops = {post_op_type_t::elu};
  return cfg;
}

// Random number generation for parameter generation
static std::mt19937 param_rng(
  std::chrono::steady_clock::now().time_since_epoch().count());

// Helper function to generate random dimensions within a range
static uint64_t generate_random_dim(uint64_t min_dim, uint64_t max_dim) {
  std::uniform_int_distribution<uint64_t> dist(min_dim, max_dim);
  return dist(param_rng);
}

MatmulParamsAI
ParameterGenerator::generate_random_params_for_accuracy_subcategory(
  const std::string &category,
  DataTypeCombination data_combo,
  const PostOpConfig &post_op_config,
  bool expect_success) {

  // Initialize with default values
  uint64_t m = 1, n = 1, k = 1;

  // Helper to ensure values are initialized
  auto generate_dims = [&](uint64_t min_m, uint64_t max_m,
                           uint64_t min_n, uint64_t max_n,
                           uint64_t min_k, uint64_t max_k,
  bool square = false) {
    m = generate_random_dim(min_m, max_m);
    n = square ? m : generate_random_dim(min_n, max_n);
    k = generate_random_dim(min_k, max_k);
  };

  if (category == "tiny_square") {
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      true);
  }
  else if (category == "tiny_rectangular") {
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::TINY_MAX),
      false);
  }
  else if (category == "small_square") {
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::SMALL_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      true);
  }
  else if (category == "medium_square") {
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_MIN),
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_MAX),
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_MIN),
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_MAX),
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_MIN),
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_MAX),
      true);
  }
  else if (category == "large_square") {
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::LARGE_MIN),
      static_cast<uint64_t>(MatrixDimensions::LARGE_MAX),
      static_cast<uint64_t>(MatrixDimensions::LARGE_MIN),
      static_cast<uint64_t>(MatrixDimensions::LARGE_MAX),
      static_cast<uint64_t>(MatrixDimensions::LARGE_MIN),
      static_cast<uint64_t>(MatrixDimensions::LARGE_MAX),
      true);
  }
  else if (category == "rectangular") {
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::RECT_MIN),
      static_cast<uint64_t>(MatrixDimensions::RECT_MAX),
      static_cast<uint64_t>(MatrixDimensions::RECT_MIN),
      static_cast<uint64_t>(MatrixDimensions::RECT_MAX),
      static_cast<uint64_t>(MatrixDimensions::RECT_MIN),
      static_cast<uint64_t>(MatrixDimensions::RECT_MAX),
      false);
  }
  else if (category == "skinny") {
    // Randomly choose between tall, wide, or deep
    int shape_type = generate_random_dim(0, 2);
    if (shape_type == 0) {  // tall
      generate_dims(
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_LARGE),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_LARGE),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MIN),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_SMALL),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MIN),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_SMALL),
        false);
    }
    else if (shape_type == 1) {  // wide
      generate_dims(
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MIN),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_SMALL),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_LARGE),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_LARGE),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MIN),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_SMALL),
        false);
    }
    else {  // deep
      generate_dims(
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MIN),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_SMALL),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MIN),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_SMALL),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_LARGE),
        static_cast<uint64_t>(MatrixDimensions::SKINNY_MAX_LARGE),
        false);
    }
  }
  else {
    // Default case - use small matrix dimensions from enum
    generate_dims(
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      static_cast<uint64_t>(MatrixDimensions::TINY_MIN),
      static_cast<uint64_t>(MatrixDimensions::SMALL_MAX),
      false);
  }

  return create_param(m, n, k,
                      data_combo,
                      TestCategory::ACCURACY,
                      post_op_config,
                      expect_success);
}

// Static member definitions for ParameterGenerator
std::vector<DataTypeCombination> ParameterGenerator::supported_combinations = {
  DataTypeCombination::F32_F32_F32,
  DataTypeCombination::F32_BF16_F32,
  DataTypeCombination::BF16_BF16_BF16,
  DataTypeCombination::BF16_F32_BF16,
  DataTypeCombination::S8_S8_S8
  // Add more as supported by your kernels
};

// -----------------------------------------------------------------------------
// generate_comprehensive_test_suite
//
// Generates a comprehensive suite of matmul test parameters, including all
// accuracy, boundary, edge case, and invalid tests.
//
// Returns:
//   Vector of MatmulParamsAI objects for all test categories
// Usage:
//   Used to instantiate the full test suite for ZenDNNL matmul.
// -----------------------------------------------------------------------------
std::vector<MatmulParamsAI>
ParameterGenerator::generate_comprehensive_test_suite() {
  std::vector<MatmulParamsAI> all_params;
  add_accuracy_params(all_params);
  add_boundary_params(all_params);
  add_edge_case_params(all_params);
  add_invalid_params(all_params);
  return all_params;
}

// -----------------------------------------------------------------------------
// generate_minimal_test_suite
//
// Generates a minimal set of matmul test parameters for smoke or CI tests.
//
// Returns:
//   Vector of MatmulParamsAI objects for a minimal test set
// Usage:
//   Used for quick validation or CI runs.
// -----------------------------------------------------------------------------
std::vector<MatmulParamsAI> ParameterGenerator::generate_minimal_test_suite() {
  std::vector<MatmulParamsAI> minimal_params;
  auto post_op_configs = AITestUtils::get_all_post_op_configs();

  // Add fixed dim accuracy params for minimal testing
  add_minimal_accuracy_params(minimal_params);

  // Add a minimal boundary test
  minimal_params.push_back(create_param(1, 1, 1, DataTypeCombination::F32_F32_F32,
                                        TestCategory::BOUNDARY, post_op_configs[0]));
  // Add a minimal invalid test
  minimal_params.push_back(create_param(0, 32, 32,
                                        DataTypeCombination::F32_F32_F32, TestCategory::INVALID, post_op_configs[0],
                                        false));
  return minimal_params;
}

// -----------------------------------------------------------------------------
// generate_category_specific_params
//
// Generates test parameters for a specific test category (accuracy, boundary,
// edge case, invalid, reference kernel).
//
// Parameters:
//   category - TestCategory enum value
// Returns:
//   Vector of MatmulParamsAI objects for the given category
// Usage:
//   Used to instantiate category-specific test suites.
// -----------------------------------------------------------------------------
std::vector<MatmulParamsAI>
ParameterGenerator::generate_category_specific_params(TestCategory category) {
  std::vector<MatmulParamsAI> params;
  switch (category) {
  case TestCategory::ACCURACY:
    add_accuracy_params(params);
    break;
  case TestCategory::BOUNDARY:
    add_boundary_params(params);
    break;
  case TestCategory::EDGE_CASE:
    add_edge_case_params(params);
    break;
  case TestCategory::INVALID:
    add_invalid_params(params);
    break;
  case TestCategory::REFERENCE_KERNEL:
    generate_reference_kernel_exhaustive_params(params);
    break;
  default:
    break;
  }
  return params;
}

// --- Add to ParameterGenerator::add_accuracy_params ---
// -----------------------------------------------------------------------------
// add_accuracy_params
//
// Populates the provided vector with accuracy test parameters for matmul.
// This function generates a comprehensive set of valid (expected to succeed)
// test cases covering:
//   - All supported data type combinations (see supported_combinations)
//   - All supported post-op configurations (see get_all_post_op_configs)
//   - A variety of matrix shapes, including square, rectangular, non-power-of-2,
//     and skinny (very tall, very wide, very deep) matrices
//   - Each (m, n, k) shape is tested for every data type and post-op config
//
// Usage:
//   - Used by ParameterGenerator::generate_comprehensive_test_suite and
//     generate_category_specific_params(TestCategory::ACCURACY) to build the
//     full set of accuracy tests for the ZenDNNL matmul test suite.
//   - Ensures that all supported kernel and post-op combinations are validated
//     for correctness on a wide range of input shapes.
// -----------------------------------------------------------------------------

// New function: add_minimal_accuracy_params
void ParameterGenerator::add_minimal_accuracy_params(std::vector<MatmulParamsAI>
    &params) {
  auto post_op_configs = AITestUtils::get_all_post_op_configs();
  std::vector<std::tuple<uint64_t, uint64_t, uint64_t, std::string>> fixed_dims
  = {
    {
      static_cast<uint64_t>(MatrixDimensions::TINY_SQUARE_FIXED),
      static_cast<uint64_t>(MatrixDimensions::TINY_SQUARE_FIXED),
      static_cast<uint64_t>(MatrixDimensions::TINY_SQUARE_FIXED),
      "tiny_square"
    },

    {
      static_cast<uint64_t>(MatrixDimensions::TINY_RECT_M_FIXED),
      static_cast<uint64_t>(MatrixDimensions::TINY_RECT_N_FIXED),
      static_cast<uint64_t>(MatrixDimensions::TINY_RECT_K_FIXED),
      "tiny_rectangular"
    },

    {
      static_cast<uint64_t>(MatrixDimensions::SMALL_SQUARE_FIXED),
      static_cast<uint64_t>(MatrixDimensions::SMALL_SQUARE_FIXED),
      static_cast<uint64_t>(MatrixDimensions::SMALL_SQUARE_FIXED),
      "small_square"
    },

    {
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_SQUARE_FIXED),
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_SQUARE_FIXED),
      static_cast<uint64_t>(MatrixDimensions::MEDIUM_SQUARE_FIXED),
      "medium_square"
    },

    {
      static_cast<uint64_t>(MatrixDimensions::LARGE_SQUARE_FIXED),
      static_cast<uint64_t>(MatrixDimensions::LARGE_SQUARE_FIXED),
      static_cast<uint64_t>(MatrixDimensions::LARGE_SQUARE_FIXED),
      "large_square"
    },

    {
      static_cast<uint64_t>(MatrixDimensions::RECT1_M_FIXED),
      static_cast<uint64_t>(MatrixDimensions::RECT1_N_FIXED),
      static_cast<uint64_t>(MatrixDimensions::RECT1_K_FIXED),
      "rectangular_1"
    },

    {
      static_cast<uint64_t>(MatrixDimensions::RECT2_M_FIXED),
      static_cast<uint64_t>(MatrixDimensions::RECT2_N_FIXED),
      static_cast<uint64_t>(MatrixDimensions::RECT2_K_FIXED),
      "rectangular_2"
    },

    {
      static_cast<uint64_t>(MatrixDimensions::NON_POW2_FIXED),
      static_cast<uint64_t>(MatrixDimensions::NON_POW2_FIXED),
      static_cast<uint64_t>(MatrixDimensions::NON_POW2_FIXED),
      "non_power_of_2"
    },

    {
      static_cast<uint64_t>(MatrixDimensions::SKINNY_LARGE_FIXED),
      static_cast<uint64_t>(MatrixDimensions::SKINNY_SMALL_FIXED),
      static_cast<uint64_t>(MatrixDimensions::SKINNY_SMALL_FIXED),
      "skinny_tall"
    },

    {
      static_cast<uint64_t>(MatrixDimensions::SKINNY_SMALL_FIXED),
      static_cast<uint64_t>(MatrixDimensions::SKINNY_LARGE_FIXED),
      static_cast<uint64_t>(MatrixDimensions::SKINNY_SMALL_FIXED),
      "skinny_wide"
    },

    {
      static_cast<uint64_t>(MatrixDimensions::SKINNY_SMALL_FIXED),
      static_cast<uint64_t>(MatrixDimensions::SKINNY_SMALL_FIXED),
      static_cast<uint64_t>(MatrixDimensions::SKINNY_LARGE_FIXED),
      "skinny_deep"
    }
  };
  for (auto data_combo : supported_combinations) {
    for (const auto &post_op_config : post_op_configs) {
      for (const auto& [m, n, k, desc] : fixed_dims) {
        if (AITestUtils::is_aocl_kernel_supported(AITestUtils::get_input_dtype(
              data_combo), AITestUtils::get_weight_dtype(data_combo),
            AITestUtils::get_output_dtype(data_combo), post_op_config.post_ops)) {
          params.push_back(create_param(m, n, k, data_combo, TestCategory::ACCURACY,
                                        post_op_config, true));
        }
      }
    }
  }
}

void ParameterGenerator::add_accuracy_params(std::vector<MatmulParamsAI>
    &params) {


  auto post_op_configs = AITestUtils::get_all_post_op_configs();
  // Define test categories and their counts
  const std::vector<std::string> categories = {
    "tiny_square",
    "tiny_rectangular",
    "small_square",
    "medium_square",
    "large_square",
    "rectangular",
    "skinny"
  };

  // Enum for maximum test cases per category
  enum class MaxTestCases {
    MAX_NUM_TINY_MATRIX = 10,       // Fewer cases for tiny matrices
    MAX_NUM_SMALL_MATRIX = 10,      // Medium number for small matrices
    MAX_NUM_MEDIUM_LARGE_MATRIX = 30, // Fewer cases for large matrices due to compute time
    MAX_NUM_RECTANGULAR_MATRIX = 30,  // Medium number for rectangular cases
    MAX_NUM_SKINNY_MATRIX = 20,      // Fewer cases for skinny matrices
    MAX_NUM_DEFAULT = 5            // Default case
  };

  // Helper function to get max test cases based on category
  auto get_max_cases_for_category = [](const std::string& category) -> int {
    if (category == "tiny_square" || category == "tiny_rectangular") {
      return static_cast<int>(MaxTestCases::MAX_NUM_TINY_MATRIX);
    }
    else if (category == "small_square") {
      return static_cast<int>(MaxTestCases::MAX_NUM_SMALL_MATRIX);
    }
    else if (category == "medium_square" || category == "large_square") {
      return static_cast<int>(MaxTestCases::MAX_NUM_MEDIUM_LARGE_MATRIX);
    }
    else if (category == "rectangular") {
      return static_cast<int>(MaxTestCases::MAX_NUM_RECTANGULAR_MATRIX);
    }
    else if (category == "skinny") {
      return static_cast<int>(MaxTestCases::MAX_NUM_SKINNY_MATRIX);
    }
    return static_cast<int>(MaxTestCases::MAX_NUM_DEFAULT);
  };

  // Add randomly generated test cases for each category
  for (const auto &category : categories) {
    const int max_cases = get_max_cases_for_category(category);
    for (auto data_combo : supported_combinations) {
      for (const auto &post_op_config : post_op_configs) {
        if (AITestUtils::is_aocl_kernel_supported(AITestUtils::get_input_dtype(
              data_combo), AITestUtils::get_weight_dtype(data_combo),
            AITestUtils::get_output_dtype(data_combo), post_op_config.post_ops)) {

          // Generate random test cases for this category/data_combo/post_op combination
          for (int i = 0; i < max_cases; i++) {
            params.push_back(generate_random_params_for_accuracy_subcategory(
                               category, data_combo, post_op_config, true));
          }
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// add_boundary_params
//
// Populates the provided vector with test parameters that target boundary cases
// for matmul. This function generates:
//   - Minimal and maximal supported dimensions (e.g., 1x1x1, large aligned sizes)
//   - Shapes that align with hardware or algorithmic boundaries (e.g., SIMD/AVX sizes)
//   - All supported data type combinations and post-op configurations
//   - Ensures that numerical stability and correctness are maintained at the edges
//
// Usage:
//   - Used by ParameterGenerator::generate_comprehensive_test_suite and
//     generate_category_specific_params(TestCategory::BOUNDARY) to build the
//     set of boundary tests for the ZenDNNL matmul test suite.
//   - Ensures correct handling at the edges of valid input space, such as
//     smallest/largest supported shapes and hardware-aligned sizes, for all
//     kernel and post-op combinations.
// -----------------------------------------------------------------------------
void ParameterGenerator::add_boundary_params(std::vector<MatmulParamsAI>
    &params) {
  auto post_op_configs = AITestUtils::get_all_post_op_configs();
  std::vector<std::tuple<uint64_t, uint64_t, uint64_t, std::string>> boundary_dims
  = {
    {1, 1, 1, "minimal_dims"},
    {1, 32, 32, "minimal_batch"},
    {32, 1, 32, "minimal_output"},
    {32, 32, 1, "minimal_inner"},
    {8, 8, 8, "simd_boundary"},
    {16, 16, 16, "avx_boundary"}
  };
  for (auto data_combo : supported_combinations) {
    for (const auto &post_op_config : post_op_configs) {
      for (const auto& [m, n, k, desc] : boundary_dims) {
        if (AITestUtils::is_aocl_kernel_supported(AITestUtils::get_input_dtype(
              data_combo), AITestUtils::get_weight_dtype(data_combo),
            AITestUtils::get_output_dtype(data_combo), post_op_config.post_ops)) {
          params.push_back(create_param(m, n, k, data_combo, TestCategory::BOUNDARY,
                                        post_op_config, true));
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// add_edge_case_params
//
// Populates the provided vector with test parameters for edge cases in matmul.
// This function generates:
//   - Unusual, rare, or extreme shapes (e.g., highly asymmetric, very large K,
//     degenerate cases) that are not necessarily at strict boundaries
//   - All supported data type combinations and post-op configurations
//
// Usage:
//   - Used by ParameterGenerator::generate_comprehensive_test_suite and
//     generate_category_specific_params(TestCategory::EDGE_CASE) to build the
//     set of edge case tests for the ZenDNNL matmul test suite.
//   - Designed to expose bugs in handling of pathological or less common input
//     shapes, ensuring robustness across all supported kernel and post-op combos.
// -----------------------------------------------------------------------------
void ParameterGenerator::add_edge_case_params(std::vector<MatmulParamsAI>
    &params) {
  auto post_op_configs = AITestUtils::get_all_post_op_configs();
  std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> edge_dims = {
    {1, 128, 128}, {128, 1, 128}, {128, 128, 1}, {1, 1, 1}, {1024, 1024, 1}, {1, 1024, 1024}, {1024, 1, 1024}
  };
  for (auto data_combo : supported_combinations) {
    for (const auto &post_op_config : post_op_configs) {
      for (const auto &tup : edge_dims) {
        uint64_t m = std::get<0>(tup);
        uint64_t n = std::get<1>(tup);
        uint64_t k = std::get<2>(tup);
        if (AITestUtils::is_aocl_kernel_supported(AITestUtils::get_input_dtype(
              data_combo), AITestUtils::get_weight_dtype(data_combo),
            AITestUtils::get_output_dtype(data_combo), post_op_config.post_ops)) {
          params.push_back(create_param(m, n, k, data_combo, TestCategory::EDGE_CASE,
                                        post_op_config, true));
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// add_invalid_params
//
// Populates the provided vector with invalid test parameters for matmul.
// This function generates a comprehensive set of invalid (expected to fail)
// test cases covering:
//   - All supported data type combinations (see supported_combinations)
//   - A variety of invalid dimension scenarios (zero, overflow, etc.)
//   - Invalid or missing binary post-op tensors
//   - Input/output tensor shape mismatches, non-2D tensors, and other
//     configuration errors
//   - Unsupported/unknown post-op types and forced kernel settings
//
// Usage:
//   - Used by ParameterGenerator::generate_comprehensive_test_suite and
//     generate_category_specific_params(TestCategory::INVALID) to build the
//     full set of negative tests for the ZenDNNL matmul test suite.
//   - Ensures that all error handling and validation logic is exercised for
//     each supported kernel and post-op combination, and for a wide range of
//     invalid input scenarios.
//   - Each test parameter is marked with expect_success = false and should
//     trigger a failure or error in the matmul implementation.
// -----------------------------------------------------------------------------
void ParameterGenerator::add_invalid_params(std::vector<MatmulParamsAI>
    &params) {
  auto post_op_configs = AITestUtils::get_all_post_op_configs();

  for (auto data_combo : supported_combinations) {
    for (auto post_op_config : post_op_configs) {
      if (AITestUtils::is_aocl_kernel_supported(AITestUtils::get_input_dtype(
            data_combo), AITestUtils::get_weight_dtype(data_combo),
          AITestUtils::get_output_dtype(data_combo), post_op_config.post_ops)) {
        // Add invalid dimension cases for all supported data type combinations
        params.push_back(create_param(0, 32, 32, data_combo, TestCategory::INVALID,
                                      post_op_config, false));
        params.push_back(create_param(32, 0, 32, data_combo, TestCategory::INVALID,
                                      post_op_config, false));
        params.push_back(create_param(32, 32, 0, data_combo, TestCategory::INVALID,
                                      post_op_config, false));
        params.push_back(create_param(0, 0, 0, data_combo, TestCategory::INVALID,
                                      post_op_config, false));
        params.push_back(create_param(AI_MAX_DIM + 1, 32, 32, data_combo,
                                      TestCategory::INVALID, post_op_config, false));
      }
    }
  }
  // --- New invalids for matmul_operator_t::validate() and validate_buffer_post_op ---
  // 1. Binary post-op buffer not passed
  {
    MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32,
                                    TestCategory::INVALID, AITestUtils::create_binary_add_config(), false);
    p.test_name = "invalid_binary_add_missing_tensor";
    // In test logic, do not bind the binary add tensor
    params.push_back(p);
  }
  // 2. Binary post-op buffer transposed
  {
    MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32,
                                    TestCategory::INVALID, AITestUtils::create_binary_add_config(), false);
    p.test_name = "invalid_binary_add_transposed";
    // In test logic, bind a tensor with order "ba"
    params.push_back(p);
  }
  // 3. Binary post-op buffer size mismatch
  {
    MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32,
                                    TestCategory::INVALID, AITestUtils::create_binary_add_config(), false);
    p.test_name = "invalid_binary_add_size_mismatch";
    // In test logic, bind a tensor with wrong shape
    params.push_back(p);
  }
  // 4. Input or output tensor is null (simulate by not binding input/output in test logic)
  // 5. Output tensor is transposed
  {
    MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32,
                                    TestCategory::INVALID, post_op_configs[0], false);
    p.test_name = "invalid_output_transposed";
    // In test logic, bind output tensor with order "ba"
    params.push_back(p);
  }
  // 6. Input/output not 2D
  {
    MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32,
                                    TestCategory::INVALID, post_op_configs[0], false);
    p.test_name = "invalid_input_not_2d";
    // In test logic, bind input tensor as 1D or 3D
    params.push_back(p);
  }
  // 7. Input/output/weights dimension mismatch
  {
    MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32,
                                    TestCategory::INVALID, post_op_configs[0], false);
    p.test_name = "invalid_dim_mismatch";
    // In test logic, set mismatched shapes
    params.push_back(p);
  }
  // 8. Forced kernel not supported
  {
    MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32,
                                    TestCategory::INVALID, post_op_configs[0], false);
    p.test_name = "invalid_forced_kernel_unsupported";
    // In test logic, set forced kernel to "onednn"
    params.push_back(p);
  }
  // 9. Kernel unimplemented (unsupported dtype combo)
  {
    MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::S4_S4_S4,
                                    TestCategory::INVALID, post_op_configs[0], false);
    p.test_name = "invalid_kernel_unimplemented";
    params.push_back(p);
  }

  // 10. Unknown/unsupported post-op type
  {
    PostOpConfig bad_postop;
    bad_postop.config_name = "bad_postop";
    bad_postop.post_ops = {static_cast<post_op_type_t>(999)};
    MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32,
                                    TestCategory::INVALID, bad_postop, false);
    p.test_name = "invalid_unknown_post_op";
    params.push_back(p);
  }
  // 11. Forced kernel set to unknown string
  {
    MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32,
                                    TestCategory::INVALID, post_op_configs[0], false);
    p.test_name = "invalid_forced_kernel_unknown";
    params.push_back(p);
  }
  // 12. Forced kernel set to empty string
  {
    MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32,
                                    TestCategory::INVALID, post_op_configs[0], false);
    p.test_name = "invalid_forced_kernel_empty";
    params.push_back(p);
  }
  // 13. Post-op config with a mix of valid and invalid post-ops
  {
    PostOpConfig mixed_postop;
    mixed_postop.config_name = "mixed_invalid";
    mixed_postop.post_ops = {post_op_type_t::relu, static_cast<post_op_type_t>(999)};
    MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32,
                                    TestCategory::INVALID, mixed_postop, false);
    p.test_name = "invalid_mixed_post_op";
    params.push_back(p);
  }
}

// -----------------------------------------------------------------------------
// generate_reference_kernel_exhaustive_params
//
// Populates the provided vector with an exhaustive set of test parameters
// specifically for validating the reference matmul kernel implementation.
// This function generates:
//   - All supported data type combinations for the reference kernel
//   - All supported post-op types (including single and multi-post-op chains)
//   - Negative cases with unknown/unsupported post-op types
//   - Coverage for all post-ops implemented in matmul_f32_ref_kernel_t::apply_post_op
//   - Each test parameter is marked with expect_success = true for valid
//     combinations, and false for negative/unsupported cases
//
// Usage:
//   - Used by ParameterGenerator::generate_category_specific_params(TestCategory::REFERENCE_KERNEL)
//     to build a comprehensive set of tests for the reference kernel path.
//   - Ensures that the reference kernel is validated for all supported data types,
//     post-op types, and error handling for unsupported post-ops.
//   - This is not intended for general kernel validation, but for deep coverage
//     of the reference implementation and its post-op logic.
// -----------------------------------------------------------------------------
void ParameterGenerator::generate_reference_kernel_exhaustive_params(
  std::vector<MatmulParamsAI> &params) {
  // Supported data type combinations for reference kernel
  auto post_op_configs = AITestUtils::get_all_post_op_configs();
  std::vector<std::tuple<uint64_t, uint64_t, uint64_t, std::string>> ref_dims = {
    {4, 4, 4, "tiny_square"},
    {4, 3, 2, "tiny_rectangular"},
    {32, 32, 32, "small_square"},
    {64, 64, 64, "medium_square"},
    {128, 128, 128, "large_square"},
    {32, 64, 32, "rectangular_1"},
    {64, 32, 64, "rectangular_2"},
    {96, 96, 96, "non_power_of_2"},
    // Skinny matrix cases (very tall or very wide)
    {256, 4, 4, "skinny_tall"},
    {4, 256, 4, "skinny_wide"},
    {4, 4, 256, "skinny_deep"},
    {512, 8, 8, "very_skinny_tall"},
    {8, 512, 8, "very_skinny_wide"},
    {8, 8, 512, "very_skinny_deep"}
  };
  for (auto data_combo : supported_combinations) {
    for (const auto &post_op_config : post_op_configs) {
      for (const auto& [m, n, k, desc] : ref_dims) {
        if (AITestUtils::is_reference_implementation_supported(
              AITestUtils::get_input_dtype(data_combo),
              AITestUtils::get_weight_dtype(data_combo),
              AITestUtils::get_output_dtype(data_combo), post_op_config.post_ops)) {
          params.push_back(create_param(m, n, k, data_combo,
                                        TestCategory::REFERENCE_KERNEL, post_op_config, true));
        }
      }
    }
  }
  for (const auto &combo : supported_combinations) {
    PostOpConfig bad_postop;
    bad_postop.config_name = "bad_postop";
    bad_postop.post_ops = {static_cast<post_op_type_t>(999)};
    params.push_back(ParameterGenerator::create_param(
                       8, 8, 8, combo, TestCategory::REFERENCE_KERNEL, bad_postop, false));
  }
}

MatmulParamsAI ai_gtests::ParameterGenerator::create_param(
  uint64_t m, uint64_t n, uint64_t k,
  DataTypeCombination combo,
  TestCategory category,
  const PostOpConfig &post_op_config,
  bool expect_success) {
  MatmulParamsAI param;
  param.m = m;
  param.n = n;
  param.k = k;
  param.data_types = combo;
  param.category = category;
  param.post_op_config = post_op_config;
  param.expect_success = expect_success;
  static std::atomic<uint64_t> param_counter{0};

  // Add data type info to param name
  auto dtype_to_str = [](data_type_t dt) {
    switch (dt) {
    case data_type_t::f32:
      return "f32";
    case data_type_t::bf16:
      return "bf16";
    case data_type_t::s8:
      return "s8";
    case data_type_t::s4:
      return "s4";
    case data_type_t::u8:
      return "u8";
    case data_type_t::s32:
      return "s32";
    default:
      return "unk";
    }
  };
  std::string input_dtype_str = dtype_to_str(AITestUtils::get_input_dtype(combo));
  std::string weight_dtype_str = dtype_to_str(AITestUtils::get_weight_dtype(
                                   combo));
  std::string output_dtype_str = dtype_to_str(AITestUtils::get_output_dtype(
                                   combo));
  // Always use the new format for test_name
  param.test_name = "m" + std::to_string(m) + "_n" + std::to_string(
                      n) + "_k" + std::to_string(k)
                    + "_in_" + input_dtype_str + "_wt_" + weight_dtype_str + "_out_" +
                    output_dtype_str
                    + "_" + post_op_config.config_name + "_" + std::to_string(
                      param_counter.fetch_add(1));
  return param;
}


} // namespace ai_gtests
