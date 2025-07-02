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
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <string>
#include "zendnn.hpp"
#include "test_utils.hpp"
#include "zendnn_logging.hpp"


#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#include <immintrin.h>
#include <stddef.h>
#include <algorithm>

using namespace zendnn;
using tag = memory::format_tag;
using dt = memory::data_type;

#define PERF    0
#define INPUT_FILE 0
#define ENABLE_BF16 0


void compare_float_arrays(const float *arr1, const float *arr2, int size,
                          float tolerance = 1e-3f) {
    bool mismatch_found = false;
    for (int i = 0; i < size; ++i) {
        float diff = std::fabs(arr1[i] - arr2[i]);
        if (diff > tolerance) {
            std::cout << "Mismatch at index " << i
                      << ": " << arr1[i] << " != " << arr2[i]
                      << " (diff = " << diff << ")\n";
            mismatch_found = true;
            break;
        }
    }

    if (!mismatch_found) {
        std::cout << "All values match within tolerance of " << tolerance << ".\n";
    }
}

float checksum(const std::vector<float> &mat) {
    float sum = 0.0f;
    for (float val : mat) {
        sum += val;
    }
    return sum;
}

void matmul_example_2D(zendnn::engine eng, zendnn::stream engine_stream,
                       int argc, char **argv, int M=1, int N=1, int K=1) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D starts");
    int batch_size=1;
#if ENABLE_BF16
    int16_t *dst1, *dst2, *src, *weight, *bias;
#else
    float *dst1, *dst2, *src, *weight, *bias;
#endif
#if INPUT_FILE
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file.txt>\n";
        return;
    }

    int m_array[10]= {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

    for (int m_count=0; m_count<4; m_count++) {
        std::ifstream infile(argv[1]);
        if (!infile) {
            std::cerr << "Failed to open input file.\n";
            return;
        }
        M = m_array[m_count];

        while (infile >> N >> K) {
#endif
            std::cout <<
                      "************************************************************Running test for M="
                      << M << ", K=" << K << ", N=" << N << "\n";
            //std::cout << "M=" << M << ", K=" << K << ", N=" << N << ", ";

            // Source (src), weights, bias, and destination (dst) tensors dimensions.
            memory::dims src_dims = {batch_size, M, K};
            memory::dims weights_dims = {batch_size, K, N};
            memory::dims bias_dims = {1, 1, N};
            memory::dims dst_dims = {batch_size, M, N};

            // Allocate buffers.
            std::vector<float> src_data(batch_size * M * K);
            std::vector<float> weights_data(batch_size * K * N);
            std::vector<float> bias_data(1 * 1 * N);
            std::vector<float> dst_data(batch_size * M * N);
            // Initialize src, weights, bias.
            std::generate(src_data.begin(), src_data.end(), []() {
                static int i = 0;
                //return i++%2;
                return std::cos(i++ / 10.f);
            });
            std::generate(weights_data.begin(), weights_data.end(), []() {
                static int i = 0;
                //return i++%2;
                return std::sin(i++ * 2.f);
            });
            std::generate(bias_data.begin(), bias_data.end(), []() {
                static int i = 0;
                //return i++%2;
                return std::tanh(i++);
            });
            std::generate(dst_data.begin(), dst_data.end(), []() {
                static int i = 0;
                //return i%7;
                return 0;
            });

            // Create memory descriptors and memory objects for src, weights, bias, and
            // dst.

            /***************************************************************************/
            int loop_count = 1000;
#if PERF
            //for (int loop=0; loop<loop_count; loop++) {
#endif

            auto src_md = memory::desc(src_dims, dt::f32, tag::abc);
            auto weights_md = memory::desc(weights_dims, dt::f32, tag::abc);
            auto bias_md = memory::desc(bias_dims, dt::f32, tag::abc);
#if ENABLE_BF16
            auto dst_md = memory::desc(dst_dims, dt::bf16, tag::abc);
#else
            auto dst_md = memory::desc(dst_dims, dt::f32, tag::abc);
#endif
            auto src_mem = memory(src_md, eng);
            auto weights_mem = memory(weights_md, eng);
            auto bias_mem = memory(bias_md, eng);
            auto dst_mem1 = memory(dst_md, eng);
            auto dst_mem2 = memory(dst_md, eng);

#if PERF
            {
                for (int i=0; i<100; i++) {
                    // Create operation descriptor
                    auto matmul_d = matmul::desc(src_md, weights_md, bias_md, dst_md);
                    // Create primitive post-ops (ReLU).
                    const float scale = 1.0f;
                    const float alpha = 0.f;
                    const float beta = 0.f;
                    //post_ops matmul_ops;
                    //matmul_ops.append_eltwise(scale, algorithm::eltwise_gelu, alpha, beta);
                    primitive_attr matmul_attr;
                    //matmul_attr.set_post_ops(matmul_ops);
                    // Create primitive descriptor.
                    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);
                    // Create the primitive.
                    auto matmul_prim = matmul(matmul_pd);
                    // Primitive arguments.
                    std::unordered_map<int, memory> matmul_args;
                    matmul_args.insert({ZENDNN_ARG_SRC, src_mem});
                    matmul_args.insert({ZENDNN_ARG_WEIGHTS, weights_mem});
                    matmul_args.insert({ZENDNN_ARG_BIAS, bias_mem});
                    matmul_args.insert({ZENDNN_ARG_DST, dst_mem1});
                    // Primitive execution: matrix multiplication with ReLU.
                    matmul_prim.execute(engine_stream, matmul_args);
                    // Wait for the computation to finalize.
                    engine_stream.wait();
                }

            }
#endif

            //First Kernel
            auto start_ms = std::chrono::high_resolution_clock::now();
#if PERF
            for (int loop=0; loop<loop_count; loop++) {
#endif
                // Write data to memory object's handles.i
#if !PERF
                write_to_zendnn_memory(src_data.data(), src_mem);
                write_to_zendnn_memory(weights_data.data(), weights_mem);
                write_to_zendnn_memory(bias_data.data(), bias_mem);
#endif
#if ENABLE_BF16
                //Create bf16 memory desc
                auto src_md_bf = memory::desc(src_dims, dt::bf16, tag::abc);
                auto weights_md_bf = memory::desc(weights_dims, dt::bf16, tag::abc);
                auto bias_md_bf = memory::desc(bias_dims, dt::bf16, tag::abc);

                //Create bf16 memory
                auto src_bf_mem = memory(src_md_bf, eng);
                auto weights_bf_mem = memory(weights_md_bf, eng);
                auto bias_bf_mem = memory(bias_md_bf, eng);

                //reorder the f32 to bf16 and execute
                reorder(src_mem, src_bf_mem).execute(engine_stream,src_mem,src_bf_mem);
                reorder(weights_mem, weights_bf_mem).execute(engine_stream,weights_mem,
                        weights_bf_mem);
                reorder(bias_mem, bias_bf_mem).execute(engine_stream, bias_mem, bias_bf_mem);
#endif

                // Create operation descriptor
#if ENABLE_BF16
                auto matmul_d = matmul::desc(src_md_bf, weights_md_bf, bias_md_bf, dst_md);
#else
                auto matmul_d = matmul::desc(src_md, weights_md, bias_md, dst_md);
#endif
                // Create primitive post-ops (ReLU).
                const float scale = 1.0f;
                const float alpha = 0.f;
                const float beta = 0.f;
                post_ops matmul_ops;
                //matmul_ops.append_eltwise(scale, algorithm::eltwise_tanh, alpha, beta);
                primitive_attr matmul_attr;
                matmul_attr.set_post_ops(matmul_ops);
                // Create primitive descriptor.
                auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);
                // Create the primitive.
                auto matmul_prim = matmul(matmul_pd);
                // Primitive arguments.
                std::unordered_map<int, memory> matmul_args;
#if ENABLE_BF16
                matmul_args.insert({ZENDNN_ARG_SRC, src_bf_mem});
                matmul_args.insert({ZENDNN_ARG_WEIGHTS, weights_bf_mem});
                matmul_args.insert({ZENDNN_ARG_BIAS, bias_bf_mem});
#else
                matmul_args.insert({ZENDNN_ARG_SRC, src_mem});
                matmul_args.insert({ZENDNN_ARG_WEIGHTS, weights_mem});
                matmul_args.insert({ZENDNN_ARG_BIAS, bias_mem});
#endif
                matmul_args.insert({ZENDNN_ARG_DST, dst_mem1});
                // Primitive execution: matrix multiplication with ReLU.
                matmul_prim.execute(engine_stream, matmul_args);
                // Wait for the computation to finalize.
                engine_stream.wait();
#if PERF
            }
#endif
            auto end_ms = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration<double, std::milli>
                               (end_ms - start_ms).count();
            //std::cout<<"ZenDNN Kernel *****************Time is "<<duration_ms<< "ms\n";
            std::cout<<duration_ms<< "ms, \n";
#if ENABLE_BF16
            dst1 = static_cast<int16_t *>(dst_mem1.get_data_handle());
            data_types dt(zendnn_bf16, zendnn_bf16, zendnn_bf16, zendnn_bf16);
            src = static_cast<int16_t *>(src_bf_mem.get_data_handle());
            weight = static_cast<int16_t *>(weights_bf_mem.get_data_handle());
            dst2 = static_cast<int16_t *>(dst_mem2.get_data_handle());
            bias = static_cast<int16_t *>(bias_bf_mem.get_data_handle());
#else
            dst1 = static_cast<float *>(dst_mem1.get_data_handle());
            data_types dt;
            src = static_cast<float *>(src_mem.get_data_handle());
            weight = static_cast<float *>(weights_mem.get_data_handle());
            dst2 = static_cast<float *>(dst_mem2.get_data_handle());
            bias = static_cast<float *>(bias_mem.get_data_handle());
#endif
            //Second Kernel
            start_ms = std::chrono::high_resolution_clock::now();
#if PERF
            for (int loop=0; loop<loop_count; loop++) {
#endif
                //auto dst_mem1 = memory(dst_md, eng);

                zendnn_custom_op::zendnn_matmul_direct(src, weight, dst2, bias, 1, 0, M, N,
                        K,
                        false, false, K, N, N, true, dt, ActivationPostOp::NONE, batch_size, batch_size);
#if PERF
            }
#endif
            end_ms = std::chrono::high_resolution_clock::now();
            duration_ms = std::chrono::duration<double, std::milli>(end_ms -
                          start_ms).count();

            // std::cout<<"Register Kernel *****************Time is "<<duration_ms<< "ms\n";
            std::cout<<duration_ms<< "ms, ";
            end_ms = std::chrono::high_resolution_clock::now();
            duration_ms = std::chrono::duration<double, std::milli>(end_ms -
                          start_ms).count();
            //std::cout<<"cache Blocking and Register Kernel *****************Time is "<<duration_ms<< "ms\n\n";
            std::cout<<duration_ms<< "ms \n\n";

#if ENABLE_BF16
            auto dst_md_f32 = memory::desc(dst_dims, dt::f32, tag::abc);
            auto dst_mem1_f32 = memory(dst_md_f32, eng);
            auto dst_mem2_f32 = memory(dst_md_f32, eng);
            reorder(dst_mem1, dst_mem1_f32).execute(engine_stream,dst_mem1,dst_mem1_f32);
            reorder(dst_mem2, dst_mem2_f32).execute(engine_stream,dst_mem2,dst_mem2_f32);
#endif
#if !PERF
#if ENABLE_BF16
            compare_float_arrays(static_cast<float *>(dst_mem1_f32.get_data_handle()),
                                 static_cast<float *>(dst_mem2_f32.get_data_handle()), M *N);
#else
            compare_float_arrays(dst1, dst2, batch_size * M *N);
#endif
#endif

#if PERF
            //}
#endif
#if INPUT_FILE
        }
    }
#endif

    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D ends");
}




int main(int argc, char **argv) {
    //return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test test starts");

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);
    // Define list of M values
    std::vector<memory::dim> M_list = {4,8,16,64,128}; // Add more M values as needed

    // Define list of (K, N) pairs
    std::vector<std::pair<memory::dim, memory::dim>> KN_list = {
        {3456,  1024},
        {3456,512},
        {512, 3456},
        {512,256},
        {13,    512},
        {256,   128},
        {1024,  1024},
        {1024,  512},
        {256,   1},
        {512,   256},
        {13,    512},
        {256,   64},
        {415,   512},
        {512,   512},
        {256,   1}
        // Add more (K, N) pairs as needed
    };

    std::vector<float> weights_data;
#if !INPUT_FILE
    for (const auto &M : M_list) {
        for (const auto &KN : KN_list) {
            memory::dim K = KN.first;
            memory::dim N = KN.second;
            std::cout<<"\nM="<<M<<", N="<<N<<", K="<<K<<std::endl;
            matmul_example_2D(eng, engine_stream, argc, argv, M, N, K);
#else
    matmul_example_2D(eng, engine_stream, argc, argv);
#endif
#if !INPUT_FILE
        }
    }
#endif
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test test ends");
    return 0;
}
