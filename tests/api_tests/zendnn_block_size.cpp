/*******************************************************************************
* Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2020 Intel Corporation
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
*******************************************************************************/
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "zendnn.hpp"
#include "test_utils.hpp"
#include "zendnn_logging.hpp"


using namespace zendnn;
using tag = memory::format_tag;
using dt = memory::data_type;

size_t required_block_size(unsigned int K, unsigned int N, bool transB,
                           zendnn_data_type_t src_dt, int src_zp, zendnn_data_type_t wei_dt) {
    size_t req_s = zendnn_custom_op::zendnn_reorder_size(K, N,
                   transB, src_dt, src_zp, wei_dt);

    return req_s;
}

int main(int argc, char **argv) {
    //return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_block_size_test test starts");


    // Define list of M values
    std::vector<memory::dim> M_list = {10, 1024}; // Add more M values as needed

    // Define list of (K, N) pairs
    std::vector<std::pair<memory::dim, memory::dim>> KN_list = {
        {3456,512},
        {512, 3456},
        {512,256},
        {13,    512},
        {256,   128},
        {3456,  1024},
        {1024,  1024},
        {1024,  512},
        {256,   1},
        {13,    512},
        {256,   64},
        {415,   512},
        {512,   512},
        {256,   1},
        {100, 1},
        {1, 1},
        {64, 1}
        // Add more (K, N) pairs as needed
    };

    for (const auto &M : M_list) {
        for (const auto &KN : KN_list) {
            memory::dim K = KN.first;
            memory::dim N = KN.second;

            size_t actual_f32 = K*N*sizeof(float);
            size_t actual_bf16 = K*N*sizeof(int16_t);
            size_t actual_s8 = K*N;
            std::cout<<K<<":K| "<<N<<":N\n";

            int src_zp = 50;
            int no_src_zp = 0;
            size_t b_size;
            // src:u8, wei:s8, src_zp
            b_size = required_block_size(K, N, true, zendnn_u8, src_zp, zendnn_s8);
            std::cout<<b_size<<" required bytes for blocking| "<<actual_s8<<" actual| "<<b_size
                     -actual_s8<<" extra bytes| "<<"src:u8, wei:s8, With_SRC_ZP\n";
            // src:s8, wei:s8, src_zp
            b_size = required_block_size(K, N, true, zendnn_s8, src_zp, zendnn_s8);
            std::cout<<b_size<<" required bytes for blocking| "<<actual_s8<<" actual| "<<b_size
                     -actual_s8<<" extra bytes| "<<"src:s8, wei:s8, With_SRC_ZP\n";
            // src:u8, wei:s8, no_src_zp
            b_size = required_block_size(K, N, true, zendnn_u8, no_src_zp, zendnn_s8);
            std::cout<<b_size<<" required bytes for blocking| "<<actual_s8<<" actual| "<<b_size
                     -actual_s8<<" extra bytes| "<<"src:u8, wei:s8, NO_SRC_ZP\n";
            // src:s8, wei:s8, no_src_zp
            b_size = required_block_size(K, N, true, zendnn_s8, no_src_zp, zendnn_s8);
            std::cout<<b_size<<" required bytes for blocking| "<<actual_s8<<" actual| "<<b_size
                     -actual_s8<<" extra bytes| "<<"src:s8, wei:s8, NO_SRC_ZP\n";

            // src:s8, wei:s8, no_src_zp
            b_size = required_block_size(K, N, true, zendnn_bf16, no_src_zp, zendnn_bf16);
            std::cout<<b_size<<" required bytes for blocking| "<<actual_bf16<<" actual| "<<b_size
                     -actual_bf16<<" extra bytes| "<<"BF16\n";
            // src:s8, wei:s8, no_src_zp
            b_size = required_block_size(K, N, true, zendnn_f32, no_src_zp, zendnn_f32);
            std::cout<<b_size<<" required bytes for blocking| "<<actual_f32<<" actual| "<<b_size
                     -actual_f32<<" extra bytes| "<<"FP32\n\n\n";

        }
    }
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_block_size_test test ends");
    return 0;
}
