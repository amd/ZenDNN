/*******************************************************************************
* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
*
*******************************************************************************/

#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <random>
#include "zendnn.hpp"
#include "zendnn_quantize_dequantize.hpp"

using namespace zendnn;

/**
 * @brief Test harness for quantization and dequantization.
 */
void test_quant_dequant() {
    const size_t count = 1024;
    float scale = 0.1f;
    int zero_point = 0;

    std::vector<float> original(count);
    std::vector<uint16_t> bf16_input(count);
    std::vector<int8_t> int8_output(count);
    std::vector<uint16_t> bf16_reconstructed(count);
    std::vector<float> final_output(count);

    // Generate random float32 data
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto &val : original) {
        val = dist(rng);
    }

    // Convert to BF16
    float32_to_bf16(original.data(), bf16_input.data(), count);

    // Quantize and dequantize
    zendnn_custom_op::quantize_bf16_to_int8(bf16_input.data(), int8_output.data(),
                                            count, scale, zero_point);
    zendnn_custom_op::dequantize_int8_to_bf16(int8_output.data(),
            bf16_reconstructed.data(), count, scale, zero_point);

    // Convert back to float32
    bf16_to_float32(bf16_reconstructed.data(), final_output.data(), count);

    // Print results
    std::cout << "Original      Quantized       Reconstructed" << std::endl;
    for (size_t i = 0; i < 16; ++i) {
        std::cout << original[i] << "   " << static_cast<int>(int8_output[i]) <<
                  "              " << final_output[i] << std::endl;
    }
}

int main() {
    test_quant_dequant();
    return 0;
}

