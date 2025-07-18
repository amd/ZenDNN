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
#include "operators/matmul/matmul_fp32_ref_kernel.hpp"

using namespace zendnnl::memory;
using namespace zendnnl::common;
using namespace zendnnl::ops;
using namespace zendnnl::error_handling;

namespace ai_gtests {

// Add tensor_map_type typedef for local use
using tensor_map_type = std::map<std::string, tensor_t>;

// Static member definitions for AITensorFactory
std::mt19937 AITensorFactory::rng(std::chrono::steady_clock::now().time_since_epoch().count());
std::atomic<uint64_t> AITensorFactory::tensor_counter{0};

// Static member definitions for PerformanceMeasurement
// Note: These static members don't exist in the header, so removing them

// Static member definitions for TensorSampler
std::mt19937 TensorSampler::rng(std::chrono::steady_clock::now().time_since_epoch().count());

// Static member definitions for ParameterGenerator
std::mt19937 ParameterGenerator::rng(std::chrono::steady_clock::now().time_since_epoch().count());
std::vector<std::pair<uint64_t, uint64_t>> ParameterGenerator::boundary_dims = {
    {1, 1}, {2, 2}, {16, 16}, {32, 32}, {64, 64}, {128, 128}, {256, 256}, {512, 512}, {1024, 1024}
};
std::vector<DataTypeCombination> ParameterGenerator::supported_combinations = {
    DataTypeCombination::F32_F32_F32,
    DataTypeCombination::BF16_BF16_BF16,
    DataTypeCombination::S8_S8_S8,
    DataTypeCombination::S4_S4_S4,
    DataTypeCombination::U8_U8_U8,
    DataTypeCombination::S32_S32_S32
    // Add more as supported by your kernels
};

// AITensorFactory implementation
void AITensorFactory::fill_uniform_data(void* ptr, size_t nelem, data_type_t dtype, float scale) {
    if (nelem == 0 || ptr == nullptr) return;
    
    std::uniform_real_distribution<float> dist(-scale, scale);
    
    switch (dtype) {
        case data_type_t::f32: {
            float* data = static_cast<float*>(ptr);
            for (size_t i = 0; i < nelem; ++i) {
                data[i] = dist(rng);
            }
            break;
        }
        case data_type_t::bf16: {
            bfloat16_t* data = static_cast<bfloat16_t*>(ptr);
            for (size_t i = 0; i < nelem; ++i) {
                data[i] = bfloat16_t(dist(rng));
            }
            break;
        }
        case data_type_t::s8: {
            int8_t* data = static_cast<int8_t*>(ptr);
            std::uniform_int_distribution<int> int_dist(-127, 127);
            for (size_t i = 0; i < nelem; ++i) {
                data[i] = static_cast<int8_t>(int_dist(rng));
            }
            break;
        }
        case data_type_t::s4: {
            int8_t* data = static_cast<int8_t*>(ptr);
            std::uniform_int_distribution<int> int_dist(-8, 7);
            for (size_t i = 0; i < nelem; ++i) {
                data[i] = static_cast<int8_t>(int_dist(rng));
            }
            break;
        }
        case data_type_t::u8: {
            uint8_t* data = static_cast<uint8_t*>(ptr);
            std::uniform_int_distribution<int> int_dist(0, 255);
            for (size_t i = 0; i < nelem; ++i) {
                data[i] = static_cast<uint8_t>(int_dist(rng));
            }
            break;
        }
        case data_type_t::s32: {
            int32_t* data = static_cast<int32_t*>(ptr);
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

tensor_t AITensorFactory::create_uniform_tensor(const std::vector<uint64_t>& dims,
                                               data_type_t dtype,
                                               float scale,
                                               const std::string& name) {
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
    void* ptr = tensor.get_raw_handle_unsafe();
    if (!ptr) {
        std::cerr << "[ERROR] Null data pointer for tensor: " << tensor_name << std::endl;
        throw std::runtime_error("Null data pointer for tensor: " + tensor_name);
    }
    fill_uniform_data(ptr, nelem, dtype, scale);
    return tensor;
}

tensor_t AITensorFactory::create_zero_tensor(const std::vector<uint64_t>& dims, 
                                            data_type_t dtype,
                                            const std::string& name) {
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
    void* ptr = tensor.get_raw_handle_unsafe();
    if (buf_size > 0 && !ptr) {
        std::cerr << "[ERROR] Null data pointer for tensor: " << tensor_name << std::endl;
        throw std::runtime_error("Null data pointer for tensor: " + tensor_name);
    }
    if (buf_size > 0) {
        std::memset(ptr, 0, buf_size);
    }
    return tensor;
}

tensor_t AITensorFactory::create_boundary_tensor(const std::vector<uint64_t>& dims, 
                                                data_type_t dtype,
                                                const std::string& name) {
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
    void* ptr = tensor.get_raw_handle_unsafe();
    if (nelem > 0 && !ptr) {
        std::cerr << "[ERROR] Null data pointer for tensor: " << tensor_name << std::endl;
        throw std::runtime_error("Null data pointer for tensor: " + tensor_name);
    }
    if (nelem > 0 && dtype == data_type_t::f32) {
        float* data = static_cast<float*>(ptr);
        for (size_t i = 0; i < nelem; ++i) {
            if (i % 4 == 0) {
                data[i] = 1.0f; // Positive one
            } else if (i % 4 == 1) {
                data[i] = -1.0f; // Negative one
            } else if (i % 4 == 2) {
                data[i] = 1e-7f; // Positive near-zero
            } else {
                data[i] = -1e-7f; // Negative near-zero
            }
        }
    }
    return tensor;
}

// Minimal implementation for other classes to get compilation working
void PerformanceMeasurement::start() {
    start_time = std::chrono::high_resolution_clock::now();
    measurement_started = true;
}

void PerformanceMeasurement::stop() {
    if (measurement_started) {
        end_time = std::chrono::high_resolution_clock::now();
        measurement_started = false;
    }
}

double PerformanceMeasurement::get_duration_ms() const {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0;
}

double PerformanceMeasurement::get_throughput_gflops(uint64_t m, uint64_t n, uint64_t k) const {
    if (!measurement_started) {
        double duration_s = get_duration_ms() / 1000.0;
        if (duration_s > 0) {
            double flops = 2.0 * m * n * k; // Matrix multiplication FLOPS
            return flops / (duration_s * 1e9);
        }
    }
    return 0.0;
}

std::vector<MatmulParamsAI> AITestUtils::generate_invalid_test_params() {
    std::vector<MatmulParamsAI> params;
    
    // Generate comprehensive invalid dimension tests
    PostOpConfig no_postop;
    no_postop.config_name = "no_postop";
    no_postop.has_bias = true;
    
    // Single zero dimension cases
    MatmulParamsAI param1;
    param1.m = 0; param1.n = 32; param1.k = 32;
    param1.data_types = DataTypeCombination::F32_F32_F32;
    param1.category = TestCategory::INVALID;
    param1.post_op_config = no_postop;
    param1.expect_success = false;
    param1.test_name = "invalid_m_zero";
    params.push_back(param1);
    
    MatmulParamsAI param2;
    param2.m = 32; param2.n = 0; param2.k = 32;
    param2.data_types = DataTypeCombination::F32_F32_F32;
    param2.category = TestCategory::INVALID;
    param2.post_op_config = no_postop;
    param2.expect_success = false;
    param2.test_name = "invalid_n_zero";
    params.push_back(param2);
    
    MatmulParamsAI param3;
    param3.m = 32; param3.n = 32; param3.k = 0;
    param3.data_types = DataTypeCombination::F32_F32_F32;
    param3.category = TestCategory::INVALID;
    param3.post_op_config = no_postop;
    param3.expect_success = false;
    param3.test_name = "invalid_k_zero";
    params.push_back(param3);
    
    // Double zero dimension cases
    MatmulParamsAI param4;
    param4.m = 0; param4.n = 0; param4.k = 32;
    param4.data_types = DataTypeCombination::F32_F32_F32;
    param4.category = TestCategory::INVALID;
    param4.post_op_config = no_postop;
    param4.expect_success = false;
    param4.test_name = "invalid_m_n_zero";
    params.push_back(param4);
    
    // Triple zero dimension case
    MatmulParamsAI param5;
    param5.m = 0; param5.n = 0; param5.k = 0;
    param5.data_types = DataTypeCombination::F32_F32_F32;
    param5.category = TestCategory::INVALID;
    param5.post_op_config = no_postop;
    param5.expect_success = false;
    param5.test_name = "invalid_all_zero";
    params.push_back(param5);
    
    return params;
}

// AITestUtils static member definitions
std::mt19937 AITestUtils::rng(std::chrono::steady_clock::now().time_since_epoch().count());
std::unordered_set<std::string> AITestUtils::generated_names;

// TensorSampler static member definitions
std::uniform_int_distribution<size_t> TensorSampler::dist;

// AITestUtils implementation
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

tensor_t AITestUtils::create_accuracy_tensor(const std::vector<uint64_t>& dims, 
                                            data_type_t dtype, 
                                            const std::string& name_prefix) {
    return AITensorFactory::create_uniform_tensor(dims, dtype, 1.0f, name_prefix + "_accuracy");
}

tensor_t AITestUtils::create_boundary_tensor(const std::vector<uint64_t>& dims, 
                                            data_type_t dtype, 
                                            const std::string& name_prefix) {
    return AITensorFactory::create_boundary_tensor(dims, dtype, name_prefix + "_boundary");
}

tensor_t AITestUtils::create_edge_case_tensor(const std::vector<uint64_t>& dims, 
                                             data_type_t dtype, 
                                             const std::string& name_prefix) {
    // For edge case testing, create tensors with special values
    return AITensorFactory::create_boundary_tensor(dims, dtype, name_prefix + "_edge");
}

tensor_t AITestUtils::create_zero_tensor(const std::vector<uint64_t>& dims, 
                                        data_type_t dtype, 
                                        const std::string& name_prefix) {
    return AITensorFactory::create_zero_tensor(dims, dtype, name_prefix + "_zero");
}

bool AITestUtils::validate_dimensions(uint64_t m, uint64_t n, uint64_t k) {
    return (m > 0 && n > 0 && k > 0 && 
            m <= AI_MAX_DIM && n <= AI_MAX_DIM && k <= AI_MAX_DIM);
}

std::string AITestUtils::generate_unique_name(const std::string& prefix) {
    static std::atomic<uint64_t> counter{0};
    return prefix + "_" + std::to_string(counter.fetch_add(1));
}

void AITestUtils::log_tensor_info(const tensor_t& tensor, const std::string& name) {
    auto size_vec = tensor.get_size();
    std::cout << "[INFO] Tensor " << name << ": ";
    for (size_t i = 0; i < size_vec.size(); ++i) {
        std::cout << size_vec[i];
        if (i < size_vec.size() - 1) std::cout << "x";
    }
    std::cout << ", dtype: " << static_cast<int>(tensor.get_data_type()) << std::endl;
}

void AITestUtils::log_validation_result(bool success, const std::string& test_name) {
    std::cout << "[" << (success ? "PASS" : "FAIL") << "] " << test_name << std::endl;
}

void AITestUtils::print_tensor_data(const tensor_t& tensor, const std::string& name, bool verbose) {
    std::cout << "[DATA] " << name << " tensor data:" << std::endl;
    if (!verbose && tensor.get_nelem() > 10) {
        std::cout << "  (Large tensor, showing first 10 elements only)" << std::endl;
    }
    
    size_t nelem = tensor.get_nelem();
    size_t show_count = verbose ? nelem : std::min(nelem, static_cast<size_t>(10));
    
    if (tensor.get_data_type() == data_type_t::f32) {
        const float* data = static_cast<const float*>(tensor.get_raw_handle_const());
        for (size_t i = 0; i < show_count; ++i) {
            std::cout << "  [" << i << "] = " << data[i] << std::endl;
        }
    }
}

// TensorSampler implementation
std::vector<size_t> TensorSampler::get_sample_indices(size_t total_elements, size_t max_samples) {
    std::vector<size_t> indices;
    
    if (total_elements <= max_samples) {
        // Return all indices if total is small
        for (size_t i = 0; i < total_elements; ++i) {
            indices.push_back(i);
        }
    } else {
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

bool TensorSampler::compare_sampled_tensors(const tensor_t& tensor1, 
                                           const tensor_t& tensor2,
                                           float tolerance) {
    if (tensor1.get_nelem() != tensor2.get_nelem()) {
        return false;
    }
    
    if (tensor1.get_data_type() != tensor2.get_data_type()) {
        return false;
    }
    
    size_t total_elements = tensor1.get_nelem();
    auto sample_indices = get_sample_indices(total_elements, AI_MAX_VALIDATION_ELEMENTS);
    
    if (tensor1.get_data_type() == data_type_t::f32) {
        const float* data1 = static_cast<const float*>(tensor1.get_raw_handle_const());
        const float* data2 = static_cast<const float*>(tensor2.get_raw_handle_const());
        
        for (size_t idx : sample_indices) {
            float diff = std::abs(data1[idx] - data2[idx]);
            if (diff > tolerance) {
                return false;
            }
        }
    }
    
    return true;
}

// --- MISSING FUNCTION IMPLEMENTATIONS (STUBS) ---

std::vector<MatmulParamsAI> ParameterGenerator::generate_comprehensive_test_suite() {
    std::vector<MatmulParamsAI> all_params;
    add_accuracy_params(all_params);
    add_boundary_params(all_params);
    add_edge_case_params(all_params);
    add_invalid_params(all_params);
    add_performance_params(all_params);
    return all_params;
}

std::vector<MatmulParamsAI> ParameterGenerator::generate_minimal_test_suite() {
    std::vector<MatmulParamsAI> minimal_params;
    auto post_op_configs = AITestUtils::get_supported_post_op_configs();
    minimal_params.push_back(create_param(32, 32, 32, DataTypeCombination::F32_F32_F32, TestCategory::ACCURACY, post_op_configs[0]));
    minimal_params.push_back(create_param(1, 1, 1, DataTypeCombination::F32_F32_F32, TestCategory::BOUNDARY, post_op_configs[0]));
    minimal_params.push_back(create_param(0, 32, 32, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, post_op_configs[0], false));
    return minimal_params;
}

std::vector<MatmulParamsAI> ParameterGenerator::generate_category_specific_params(TestCategory category) {
    if (supported_combinations.empty()) {
        initialize();
    }
    std::vector<MatmulParamsAI> params;
    switch (category) {
        case TestCategory::ACCURACY:   add_accuracy_params(params); break;
        case TestCategory::BOUNDARY:   add_boundary_params(params); break;
        case TestCategory::EDGE_CASE:  add_edge_case_params(params); break;
        case TestCategory::INVALID:    add_invalid_params(params); break;
        case TestCategory::PERFORMANCE:add_performance_params(params); break;
        default: break;
    }
    return params;
}

void ParameterGenerator::initialize() {}

void AITestUtils::log_test_params(const MatmulParamsAI&) {}

status_t AITestUtils::run_reference_matmul(
    tensor_t& input, tensor_t& weights, tensor_t& bias,
    tensor_t& output, const PostOpConfig& post_op_config,
    std::vector<tensor_t>& binary_postop_tensors)
{
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
        for (const auto& post_op_type : post_op_config.post_ops) {
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
            if ((post_op_type == post_op_type_t::binary_add || post_op_type == post_op_type_t::binary_mul)
                && binary_tensor_idx < binary_postop_tensors.size()) {
                std::string tensor_name;
                try {
                    if (post_op_type == post_op_type_t::binary_add) {
                        tensor_name = matmul_context.get_post_op(i).binary_add_params.tensor_name;
                    } else {
                        tensor_name = matmul_context.get_post_op(i).binary_mul_params.tensor_name;
                    }
                } catch (...) {
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
        for (auto& kv : inputs) {
            matmul_operator = matmul_operator.set_input(kv.first, kv.second);
        }
        matmul_operator = matmul_operator.set_output("matmul_output", output);

        // Force reference kernel
        matmul_operator = matmul_operator.set_forced_kernel("reference");

        // Execute
        status_t status = matmul_operator.execute();
        return status;
    } catch (const std::exception& e) {
        std::cerr << "[AI_REF] Exception in run_reference_matmul: " << e.what() << std::endl;
        return status_t::failure;
    } catch (...) {
        std::cerr << "[AI_REF] Unknown exception in run_reference_matmul" << std::endl;
        return status_t::failure;
    }
}

// Minimal implementations for other missing functions
bool AITestUtils::is_valid_data_type_combination(DataTypeCombination combo) {
    // Only allow supported combinations
    const auto& supported = ParameterGenerator::supported_combinations;
    return std::find(supported.begin(), supported.end(), combo) != supported.end();
}

std::vector<MatmulParamsAI> AITestUtils::generate_accuracy_test_params() {
    std::vector<MatmulParamsAI> params;
    // Typical accuracy test cases: small, medium, large, all supported dtypes, with/without bias, with/without post-ops
    for (auto combo : ParameterGenerator::supported_combinations) {
        for (auto m : {8, 32, 128}) {
            for (auto n : {8, 32, 128}) {
                for (auto k : {8, 32, 128}) {
                    MatmulParamsAI p;
                    p.m = m; p.n = n; p.k = k;
                    p.data_types = combo;
                    p.category = TestCategory::ACCURACY;
                    p.post_op_config = create_relu_config();
                    p.expect_success = true;
                    p.test_name = "accuracy_" + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
                    params.push_back(p);
                }
            }
        }
    }
    return params;
}

std::vector<MatmulParamsAI> AITestUtils::generate_boundary_test_params() {
    std::vector<MatmulParamsAI> params;
    // Use boundary_dims from ParameterGenerator
    for (auto combo : ParameterGenerator::supported_combinations) {
        for (const auto& dims : ParameterGenerator::boundary_dims) {
            MatmulParamsAI p;
            p.m = dims.first;
            p.n = dims.second;
            p.k = 1; // boundary on m/n, minimal k
            p.data_types = combo;
            p.category = TestCategory::BOUNDARY;
            p.post_op_config = create_binary_add_config();
            p.expect_success = true;
            p.test_name = "boundary_" + std::to_string(p.m) + "x" + std::to_string(p.n) + "x" + std::to_string(p.k);
            params.push_back(p);
        }
    }
    return params;
}

std::vector<MatmulParamsAI> AITestUtils::generate_edge_case_test_params() {
    std::vector<MatmulParamsAI> params;
    // Edge cases: 1xN, Mx1, 1x1, large K, etc.
    for (auto combo : ParameterGenerator::supported_combinations) {
        std::vector<std::tuple<uint64_t,uint64_t,uint64_t>> edge_dims = {
            {1, 128, 128}, {128, 1, 128}, {128, 128, 1}, {1, 1, 1}, {1024, 1024, 1}, {1, 1024, 1024}, {1024, 1, 1024}
        };
        for (const auto& tup : edge_dims) {
            MatmulParamsAI p;
            p.m = std::get<0>(tup);
            p.n = std::get<1>(tup);
            p.k = std::get<2>(tup);
            p.data_types = combo;
            p.category = TestCategory::EDGE_CASE;
            p.post_op_config = create_silu_config();
            p.expect_success = true;
            p.test_name = "edge_" + std::to_string(p.m) + "x" + std::to_string(p.n) + "x" + std::to_string(p.k);
            params.push_back(p);
        }
    }
    return params;
}

std::vector<MatmulParamsAI> AITestUtils::generate_performance_test_params() {
    std::vector<MatmulParamsAI> params;
    // Large matrices for performance
    for (auto combo : ParameterGenerator::supported_combinations) {
        std::vector<std::tuple<uint64_t,uint64_t,uint64_t>> perf_dims = {
            {512, 512, 512}, {1024, 1024, 1024}, {2048, 2048, 2048}
        };
        for (const auto& tup : perf_dims) {
            MatmulParamsAI p;
            p.m = std::get<0>(tup);
            p.n = std::get<1>(tup);
            p.k = std::get<2>(tup);
            p.data_types = combo;
            p.category = TestCategory::PERFORMANCE;
            p.post_op_config = create_mixed_post_op_config();
            p.expect_success = true;
            p.test_name = "perf_" + std::to_string(p.m) + "x" + std::to_string(p.n) + "x" + std::to_string(p.k);
            params.push_back(p);
        }
    }
    return params;
}

std::vector<PostOpConfig> AITestUtils::get_supported_post_op_configs() {
    // All supported post-op combinations
    return {
        PostOpConfig{},
        create_binary_add_config(),
        create_binary_mul_config(),
        create_relu_config(),
        create_silu_config(),
        create_mixed_post_op_config()
    };
}

PostOpConfig AITestUtils::create_binary_add_config() {
    PostOpConfig cfg;
    cfg.config_name = "binary_add";
    cfg.has_bias = true;
    cfg.post_ops = {post_op_type_t::binary_add};
    return cfg;
}

PostOpConfig AITestUtils::create_binary_mul_config() {
    PostOpConfig cfg;
    cfg.config_name = "binary_mul";
    cfg.has_bias = true;
    cfg.post_ops = {post_op_type_t::binary_mul};
    return cfg;
}

PostOpConfig AITestUtils::create_relu_config() {
    PostOpConfig cfg;
    cfg.config_name = "relu";
    cfg.has_bias = true;
    cfg.post_ops = {post_op_type_t::relu};
    return cfg;
}

PostOpConfig AITestUtils::create_silu_config() {
    PostOpConfig cfg;
    cfg.config_name = "silu";
    cfg.has_bias = true;
    // SiLU is equivalent to Swish, which is supported as post_op_type_t::swish
    cfg.post_ops = std::vector<post_op_type_t>{post_op_type_t::swish};
    return cfg;
}

PostOpConfig AITestUtils::create_mixed_post_op_config() {
    PostOpConfig cfg;
    cfg.config_name = "mixed";
    cfg.has_bias = true;
    // Use swish instead of silu, as silu is not a valid enum value
    cfg.post_ops = std::vector<post_op_type_t>{post_op_type_t::binary_add, post_op_type_t::relu, post_op_type_t::swish};
    return cfg;
}

// --- Add missing post-op config creators ---
PostOpConfig AITestUtils::create_softmax_config() {
    PostOpConfig cfg;
    cfg.config_name = "softmax";
    cfg.has_bias = true;
    cfg.post_ops = {post_op_type_t::softmax};
    return cfg;
}
PostOpConfig AITestUtils::create_abs_config() {
    PostOpConfig cfg;
    cfg.config_name = "abs";
    cfg.has_bias = true;
    cfg.post_ops = {post_op_type_t::abs};
    return cfg;
}
PostOpConfig AITestUtils::create_square_config() {
    PostOpConfig cfg;
    cfg.config_name = "square";
    cfg.has_bias = true;
    cfg.post_ops = {post_op_type_t::square};
    return cfg;
}
PostOpConfig AITestUtils::create_sqrt_config() {
    PostOpConfig cfg;
    cfg.config_name = "sqrt";
    cfg.has_bias = true;
    cfg.post_ops = {post_op_type_t::sqrt};
    return cfg;
}
PostOpConfig AITestUtils::create_exp_config() {
    PostOpConfig cfg;
    cfg.config_name = "exp";
    cfg.has_bias = true;
    cfg.post_ops = {post_op_type_t::exp};
    return cfg;
}
PostOpConfig AITestUtils::create_log_config() {
    PostOpConfig cfg;
    cfg.config_name = "log";
    cfg.has_bias = true;
    cfg.post_ops = {post_op_type_t::log};
    return cfg;
}
PostOpConfig AITestUtils::create_leaky_relu_config() {
    PostOpConfig cfg;
    cfg.config_name = "leaky_relu";
    cfg.has_bias = true;
    cfg.post_ops = {post_op_type_t::leaky_relu};
    return cfg;
}
PostOpConfig AITestUtils::create_elu_config() {
    PostOpConfig cfg;
    cfg.config_name = "elu";
    cfg.has_bias = true;
    cfg.post_ops = {post_op_type_t::elu};
    return cfg;
}

// --- Add to ParameterGenerator::add_accuracy_params ---
void ParameterGenerator::add_accuracy_params(std::vector<MatmulParamsAI>& params) {
    auto post_op_configs = AITestUtils::get_supported_post_op_configs();
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t, std::string>> accuracy_dims = {
        {4, 4, 4, "tiny_square"},
        {4, 3, 2, "tiny_rectangular"},
        {32, 32, 32, "small_square"},
        {64, 64, 64, "medium_square"},
        {128, 128, 128, "large_square"},
        {32, 64, 32, "rectangular_1"},
        {64, 32, 64, "rectangular_2"},
        {96, 96, 96, "non_power_of_2"}
    };
    for (auto data_combo : supported_combinations) {
        for (const auto& post_op_config : post_op_configs) {
            for (const auto& [m, n, k, desc] : accuracy_dims) {
                params.push_back(create_param(m, n, k, data_combo, TestCategory::ACCURACY, post_op_config, true));
            }
        }
    }
    // Add coverage for all post-ops in matmul_f32_ref_kernel_t::apply_post_op
    std::vector<PostOpConfig> extra_postops = {
        AITestUtils::create_softmax_config(),
        AITestUtils::create_abs_config(),
        AITestUtils::create_square_config(),
        AITestUtils::create_sqrt_config(),
        AITestUtils::create_exp_config(),
        AITestUtils::create_log_config(),
        AITestUtils::create_leaky_relu_config(),
        AITestUtils::create_elu_config()
    };
    for (auto data_combo : supported_combinations) {
        for (const auto& post_op_config : extra_postops) {
            params.push_back(create_param(8, 8, 8, data_combo, TestCategory::ACCURACY, post_op_config, true));
        }
    }
}

void ParameterGenerator::add_boundary_params(std::vector<MatmulParamsAI>& params) {
    auto post_op_configs = AITestUtils::get_supported_post_op_configs();
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t, std::string>> boundary_dims = {
        {1, 1, 1, "minimal_dims"},
        {1, 32, 32, "minimal_batch"},
        {32, 1, 32, "minimal_output"},
        {32, 32, 1, "minimal_inner"},
        {8, 8, 8, "simd_boundary"},
        {16, 16, 16, "avx_boundary"}
    };
    for (const auto& [m, n, k, desc] : boundary_dims) {
        params.push_back(create_param(m, n, k, DataTypeCombination::F32_F32_F32, TestCategory::BOUNDARY, post_op_configs[0]));
    }
}

void ParameterGenerator::add_edge_case_params(std::vector<MatmulParamsAI>& params) {
    auto post_op_configs = AITestUtils::get_supported_post_op_configs();
    params.push_back(create_param(AI_MAX_DIM/2, AI_MAX_DIM/2, AI_MAX_DIM/2, DataTypeCombination::F32_F32_F32, TestCategory::EDGE_CASE, post_op_configs[0]));
}

void ParameterGenerator::add_invalid_params(std::vector<MatmulParamsAI>& params) {
    auto post_op_configs = AITestUtils::get_supported_post_op_configs();
    params.push_back(create_param(0, 32, 32, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, post_op_configs[0], false));
    params.push_back(create_param(32, 0, 32, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, post_op_configs[0], false));
    params.push_back(create_param(32, 32, 0, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, post_op_configs[0], false));
    params.push_back(create_param(0, 0, 0, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, post_op_configs[0], false));
    params.push_back(create_param(AI_MAX_DIM + 1, 32, 32, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, post_op_configs[0], false));
    // --- New invalids for matmul_operator_t::validate() and validate_buffer_post_op ---
    // 1. Binary post-op buffer not passed
    {
        MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, AITestUtils::create_binary_add_config(), false);
        p.test_name = "invalid_binary_add_missing_tensor";
        // In test logic, do not bind the binary add tensor
        params.push_back(p);
    }
    // 2. Binary post-op buffer transposed
    {
        MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, AITestUtils::create_binary_add_config(), false);
        p.test_name = "invalid_binary_add_transposed";
        // In test logic, bind a tensor with order "ba"
        params.push_back(p);
    }
    // 3. Binary post-op buffer size mismatch
    {
        MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, AITestUtils::create_binary_add_config(), false);
        p.test_name = "invalid_binary_add_size_mismatch";
        // In test logic, bind a tensor with wrong shape
        params.push_back(p);
    }
    // 4. Input or output tensor is null (simulate by not binding input/output in test logic)
    // 5. Output tensor is transposed
    {
        MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, post_op_configs[0], false);
        p.test_name = "invalid_output_transposed";
        // In test logic, bind output tensor with order "ba"
        params.push_back(p);
    }
    // 6. Input/output not 2D
    {
        MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, post_op_configs[0], false);
        p.test_name = "invalid_input_not_2d";
        // In test logic, bind input tensor as 1D or 3D
        params.push_back(p);
    }
    // 7. Input/output/weights dimension mismatch
    {
        MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, post_op_configs[0], false);
        p.test_name = "invalid_dim_mismatch";
        // In test logic, set mismatched shapes
        params.push_back(p);
    }
    // 8. Forced kernel not supported
    {
        MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, post_op_configs[0], false);
        p.test_name = "invalid_forced_kernel_unsupported";
        // In test logic, set forced kernel to "onednn"
        params.push_back(p);
    }
    // 9. Kernel unimplemented (unsupported dtype combo)
    {
        MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::S4_S4_S4, TestCategory::INVALID, post_op_configs[0], false);
        p.test_name = "invalid_kernel_unimplemented";
        params.push_back(p);
    }

    // 10. Unknown/unsupported post-op type
    {
        PostOpConfig bad_postop;
        bad_postop.config_name = "bad_postop";
        bad_postop.has_bias = false;
        bad_postop.post_ops = {static_cast<post_op_type_t>(999)};
        MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, bad_postop, false);
        p.test_name = "invalid_unknown_post_op";
        params.push_back(p);
    }
    // 11. Forced kernel set to unknown string
    {
        MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, post_op_configs[0], false);
        p.test_name = "invalid_forced_kernel_unknown";
        params.push_back(p);
    }
    // 12. Forced kernel set to empty string
    {
        MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, post_op_configs[0], false);
        p.test_name = "invalid_forced_kernel_empty";
        params.push_back(p);
    }
    // 13. Post-op config with a mix of valid and invalid post-ops
    {
        PostOpConfig mixed_postop;
        mixed_postop.config_name = "mixed_invalid";
        mixed_postop.has_bias = true;
        mixed_postop.post_ops = {post_op_type_t::relu, static_cast<post_op_type_t>(999)};
        MatmulParamsAI p = create_param(8, 8, 8, DataTypeCombination::F32_F32_F32, TestCategory::INVALID, mixed_postop, false);
        p.test_name = "invalid_mixed_post_op";
        params.push_back(p);
    }
}

void ParameterGenerator::add_performance_params(std::vector<MatmulParamsAI>& params) {
    auto post_op_configs = AITestUtils::get_supported_post_op_configs();
    std::vector<uint64_t> perf_dims = {64, 128, 256, 512};
    for (auto dim : perf_dims) {
        params.push_back(create_param(dim, dim, dim, DataTypeCombination::F32_F32_F32, TestCategory::PERFORMANCE, post_op_configs[0]));
    }
}

} // namespace ai_gtests
