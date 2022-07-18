/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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

/// @example cnn_inference_f32.cpp
/// @copybrief cnn_inference_f32_cpp
/// > Annotated version: @ref cnn_inference_f32_cpp

/// @page cnn_inference_f32_cpp CNN f32 inference example
/// This C++ API example demonstrates how to build an AlexNet neural
/// network topology for forward-pass inference.
///
/// > Example code: @ref cnn_inference_f32.cpp
///
/// Some key take-aways include:
///
/// * How tensors are implemented and submitted to primitives.
/// * How primitives are created.
/// * How primitives are sequentially submitted to the network, where the output
///   from primitives is passed as input to the next primitive. The latter
///   specifies a dependency between the primitive input and output data.
/// * Specific 'inference-only' configurations.
/// * Limiting the number of reorders performed that are detrimental
///   to performance.
///
/// The example implements the AlexNet layers
/// as numbered primitives (for example, conv1, pool1, conv2).

// Sample Command to compile and run:
// gcc -std=c++20 -I/home/ZenDNN/inc zendnn_conv_lpgemm_u8s8s16os8.cpp -L/home/ZenDNN/_out/lib  -L/home/amd-blis/lib/ -L/home/amd-libm/lib/ -lamdZenDNN -lblis-mt -lamdlibm -fopenmp -lstdc++
// OMP_WAIT_POLICY=active OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=64 numactl --cpunodebind=0-3 --interleave=0-3 ./a.out

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>

#include <assert.h>

#include <chrono>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <cstdlib>

#include "test_utils.hpp"
#include "zendnn_logging.hpp"

#define ZENDNN_CMP_OUTPUT   1
#define GFLOP_SIZE 1000.0 // 10^9 / 10^6 = 10^3

using namespace zendnn;
using namespace std;

#define ITERATIONS 50
#define WARMUP 50

#define NUM_LAYERS 177
int num_test_cases = NUM_LAYERS;

int s32_downscaling_val = 1;
int all_direct_val = 0;

double time_taken = 0;
double time_taken_direct = 0;

memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
                           std::multiplies<memory::dim>());
}

//function to init data
void init_data_u8(memory &m, int8_t value) {
    size_t size = m.get_desc().get_size() /sizeof(uint8_t);
    srand(1111);
    uint8_t *data = (uint8_t *)m.get_data_handle();
    for (size_t i = 0; i < size; ++i) {
        if (value == -1) {
            data[i] = (uint8_t)(rand()%3);
        }
        else {
            data[i] = (uint8_t)(value);
        }
    }
}

//function to init data
void init_data_s8(memory &m, int8_t value) {
    size_t size = m.get_desc().get_size() /sizeof(int8_t);
    srand(1111);
    int8_t *data = (int8_t *)m.get_data_handle();
    for (size_t i = 0; i < size; ++i) {
        if (value == -1) {
            data[i] = (int8_t)(rand()%3);
        }
        else {
            data[i] = (int8_t)(value);
        }
    }
}

//function to init data
void init_data_s16(memory &m, int8_t value) {
    size_t size = m.get_desc().get_size() /sizeof(int16_t);
    srand(1111);
    int16_t *data = (int16_t *)m.get_data_handle();
    for (size_t i = 0; i < size; ++i) {
        if (value == -1) {
            data[i] = (int16_t)(rand()%3);
        }
        else {
            data[i] = (int16_t)(value);
        }
    }
}

//function to init data
void init_data_s32(memory &m, int8_t value) {
    size_t size = m.get_desc().get_size() /sizeof(int);
    srand(1111);
    int *data = (int *)m.get_data_handle();
    for (size_t i = 0; i < size; ++i) {
        if (value == -1) {
            data[i] = (int)(rand()%3);
        }
        else {
            data[i] = (int)(value);
        }
    }
}


void convolution_param(engine eng, zendnn::memory user_src_memory, int batch,
                       int channel, int height, int width,
                       zendnn::memory user_weights_memory, int no_of_filter, int kernel_h,
                       int kernel_w, int pad_h,
                       int pad_w, int stride_h, int stride_w, zendnn::memory conv1_user_bias_memory,
                       zendnn::memory conv1_dst_memory, int out_height, int out_width) {

    int times = 1;
    using tag = memory::format_tag;
    using dt = memory::data_type;

    /// Initialize an engine and stream. The last parameter in the call represents
    /// the index of the engine.
    /// @snippet cnn_inference_f32.cpp Initialize engine and stream
    //[Initialize engine and stream]
    //engine eng(engine_kind, 0);
    stream s(eng);
    //[Initialize engine and stream]

    /// Create a vector for the primitives and a vector to hold memory
    /// that will be used as arguments.
    /// @snippet cnn_inference_f32.cpp Create network
    //[Create network]
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;
    //[Create network]

    // AlexNet: conv1
    // {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 55, 55}
    // strides: {4, 4}
    memory::dims conv1_src_tz = {batch, channel, height, width};
    memory::dims conv1_weights_tz = {no_of_filter, channel, kernel_h, kernel_w};
    memory::dims conv1_bias_tz = {no_of_filter};
    memory::dims conv1_dst_tz = {batch, no_of_filter, out_height, out_width};
    memory::dims conv1_strides = {stride_h, stride_w};
    memory::dims conv1_padding = {pad_h, pad_w};


    //[Create user memory]
    /// Create memory descriptors with layout tag::any. The `any` format enables
    /// the convolution primitive to choose the data format that will result in
    /// best performance based on its input parameters (convolution kernel
    /// sizes, strides, padding, and so on). If the resulting format is different
    /// from `nchw`, the user data must be transformed to the format required for
    /// the convolution (as explained below).
    /// @snippet cnn_inference_f32.cpp Create convolution memory descriptors
    //[Create convolution memory descriptors]
    auto conv1_src_md = memory::desc({conv1_src_tz}, dt::bf16, tag::any);
    auto conv1_bias_md = memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::bf16, tag::any);
    auto conv1_dst_md = memory::desc({conv1_dst_tz}, dt::bf16, tag::any);
    if (s32_downscaling_val == 0)
        conv1_dst_md = memory::desc({conv1_dst_tz}, dt::f32, tag::any);
    //[Create convolution memory descriptors]

    /// Create a convolution descriptor by specifying propagation kind,
    /// [convolution algorithm](@ref dev_guide_convolution), shapes of input,
    /// weights, bias, output, convolution strides, padding, and kind of padding.
    /// Propagation kind is set to prop_kind::forward_inference to optimize for
    /// inference execution and omit computations that are necessary only for
    /// backward propagation.
    /// @snippet cnn_inference_f32.cpp Create convolution descriptor
    //[Create convolution descriptor]
    bool reluFused = true;
    auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
                      s32_downscaling_val!=0? algorithm::convolution_gemm_bf16bf16f32obf16:
                      algorithm::convolution_gemm_bf16bf16f32of32, conv1_src_md, conv1_weights_md,
                      conv1_bias_md, conv1_dst_md, conv1_strides, conv1_padding,
                      conv1_padding, reluFused);

    //[Create convolution descriptor]

    /// Create a convolution primitive descriptor. Once created, this
    /// descriptor has specific formats instead of the `any` format specified
    /// in the convolution descriptor.
    /// @snippet cnn_inference_f32.cpp Create convolution primitive descriptor
    //[Create convolution primitive descriptor]

    zendnn::post_ops post_ops;
    post_ops.append_eltwise(1.0f, zendnn::algorithm::eltwise_linear, 1.0f, 0.0f);
    zendnn::primitive_attr conv_attr;
    std::vector<float> scales(1);
    // Scaling value hardcoded. Add correct scale value depending on layer
    // for exact accuracy validation.
    float output_scale = 0.74f;
    if (s32_downscaling_val != 0) {
        scales[0] = output_scale;
        conv_attr.set_output_scales(0, scales);
    }
    conv_attr.set_post_ops(post_ops);

    auto conv1_prim_desc = convolution_forward::primitive_desc(conv1_desc,
                           conv_attr, eng);
    //[Create convolution primitive descriptor]

    /// Check whether data and weights formats required by convolution is different
    /// from the user format. In case it is different change the layout using
    /// reorder primitive.
    /// @snippet cnn_inference_f32.cpp Reorder data and weights
    //[Reorder data and weights]
    auto conv1_src_memory = user_src_memory;

    auto conv1_weights_memory = user_weights_memory;

    //[Reorder data and weights]

    /// Create a memory primitive for output.
    /// @snippet cnn_inference_f32.cpp Create memory for output
    //[Create memory for output]
    //auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);
    //[Create memory for output]

    /// Create a convolution primitive and add it to the net.
    /// @snippet cnn_inference_f32.cpp Create memory for output
    //[Create convolution primitive]
    net.push_back(convolution_forward(conv1_prim_desc));
#if ZENDNN_ENABLE
    net_args.push_back({{ZENDNN_ARG_SRC, user_src_memory},
        {ZENDNN_ARG_WEIGHTS, user_weights_memory},

#else
    net_args.push_back({{ZENDNN_ARG_SRC, conv1_src_memory},
        {ZENDNN_ARG_WEIGHTS, conv1_weights_memory},

#endif
        {ZENDNN_ARG_BIAS, conv1_user_bias_memory},
        {ZENDNN_ARG_DST, conv1_dst_memory}
    });
    //[Create convolution primitive]


    /// @page cnn_inference_f32_cpp
    /// Finally, execute the primitives. For this example, the net is executed
    /// multiple times and each execution is timed individually.
    /// @snippet cnn_inference_f32.cpp Execute model
    //[Execute model]

    for (int j = 0; j < times; ++j) {
        assert(net.size() == net_args.size() && "something is missing");
        for (size_t i = 0; i < net.size(); ++i) {
            net.at(i).execute(s, net_args.at(i));
        }
    }

    s.wait();
}

void convolution_ref_direct(engine eng, zendnn::memory user_src_memory,
                            int batch, int channel, int height, int width,
                            zendnn::memory user_weights_memory, int no_of_filter, int kernel_h,
                            int kernel_w, int pad_h,
                            int pad_w, int stride_h, int stride_w, zendnn::memory conv1_user_bias_memory,
                            zendnn::memory conv1_dst_memory, int out_height, int out_width) {

    int times = 1;
    using tag = memory::format_tag;
    using dt = memory::data_type;

    stream s(eng);

    memory::dims conv1_src_tz = {batch, channel, height, width};
    memory::dims conv1_weights_tz = {no_of_filter, channel, kernel_h, kernel_w};
    memory::dims conv1_bias_tz = {no_of_filter};
    memory::dims conv1_dst_tz = {batch, no_of_filter, out_height, out_width};
    memory::dims conv1_strides = {stride_h, stride_w};
    memory::dims conv1_padding = {pad_h, pad_w};


    auto conv1_src_md = memory::desc({conv1_src_tz}, dt::bf16, tag::acdb);
    auto conv1_bias_md = memory::desc({conv1_bias_tz}, dt::f32, tag::x);
    auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::bf16, tag::any);
    auto conv1_dst_md = memory::desc({conv1_dst_tz}, dt::bf16, tag::acdb);
    auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
                      algorithm::convolution_direct, conv1_src_md, conv1_weights_md,
                      conv1_bias_md, conv1_dst_md, conv1_strides, conv1_padding,
                      conv1_padding);

    auto conv1_prim_desc = convolution_forward::primitive_desc(conv1_desc, eng);
    auto conv1_src_memory = user_src_memory;
    auto conv1_weights_memory = user_weights_memory;


    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    net.push_back(convolution_forward(conv1_prim_desc));
    net_args.push_back({{ZENDNN_ARG_SRC, conv1_src_memory},
        {ZENDNN_ARG_WEIGHTS, conv1_weights_memory},
        {ZENDNN_ARG_BIAS, conv1_user_bias_memory},
        {ZENDNN_ARG_DST, conv1_dst_memory}
    });

    for (int j = 0; j < times; ++j) {
        assert(net.size() == net_args.size() && "something is missing");
        for (size_t i = 0; i < net.size(); ++i) {
            net.at(i).execute(s, net_args.at(i));
        }
    }
    s.wait();

}

int main(int argc, char **argv) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_avx_conv API test starts");

    char *filename = (char *)"input_sizes.txt";
    FILE *my_file;
    if (argc > 1) {
        my_file = fopen(argv[1], "r");
    }
    else {
        my_file = fopen(filename, "r");
    }
    int count_entry = 0;
    char chr = getc(my_file);
    while (chr != EOF) {
        if (chr == '\n') {
            count_entry = count_entry + 1;
        }
        chr = getc(my_file);
    }
    fclose(my_file); //close file

    std::cout << "Number of layers = " << count_entry << std::endl;

    int conv_test_dimension[190][13] = {}; //count_entry][13] = {};

    if (argc > 1) {
        my_file = fopen(argv[1], "r");
    }
    else {
        my_file = fopen(filename, "r");
    }
    size_t count = 0;
    char *line = NULL;
    size_t len = 0;
    size_t read;
    for (; count < count_entry; count++) {
        //std::cout << "read line: " << count << std::endl;
        read = getline(&line, &len, my_file);
        //std::cout << line << std::endl;
        int got = sscanf(line, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
                         &conv_test_dimension[count][0], &conv_test_dimension[count][1],
                         &conv_test_dimension[count][2], &conv_test_dimension[count][3],
                         &conv_test_dimension[count][4], &conv_test_dimension[count][5],
                         &conv_test_dimension[count][6], &conv_test_dimension[count][7],
                         &conv_test_dimension[count][8], &conv_test_dimension[count][9],
                         &conv_test_dimension[count][10], &conv_test_dimension[count][11],
                         &conv_test_dimension[count][12]);

        //fscanf(my_file, "\n");
        //if (got != 2) break; // wrong number of tokens - maybe end of file
    }
    fclose(my_file);
    num_test_cases = count_entry;

    const char *s32_downscaling = std::getenv("BF16_DOWNSCALING");
    //int s32_downscaling_val = 1;
    if (s32_downscaling) {
        s32_downscaling_val = atoi(s32_downscaling);
    }
    if (s32_downscaling_val != 0) {
        s32_downscaling_val = 1;
    }

    const char *all_direct = std::getenv("ALL_DIRECT");
    if (all_direct) {
        all_direct_val = atoi(all_direct);
    }
    if (all_direct_val != 1) {
        all_direct_val = 0;
    }

    if (all_direct_val == 1) {
        s32_downscaling_val = 1;
    }

    std::cout << "BF16_DOWNSCALING: " << s32_downscaling_val << std::endl;
    std::cout << "ALL_DIRECT: " << all_direct_val <<
              " (if set to 1, BF16_DOWNSCALING has no impact)" << std::endl;

    int count_elementwise_match = 0, count_sum_match = 0;
    double highest_gflops = 0, lowest_gflops = 100000;
    double total_gflops = 0;

    try {
        std::ofstream file;

        std::cout << "\n--------------------------------------------" << std::endl;
        std::cout << "C++ Benchmark of multiple convolution layers (in GFLOPs)" <<
                  std::endl;
        std::cout << "--------------------------------------------" << std::endl;

        //Input parameters to convolution
        int times = 1;
        int pad_type = 0;

        double operations = 0;
        double total_time_taken = 0;
        double total_operations = 0;

        int num_direct_layer = 0, num_lpgemm_layer = 0;

        char **cpu = NULL;
        engine::kind engine_kind = parse_engine_kind(1, cpu);
        engine eng(engine_kind, 0);
        stream s(eng);

        memory user_src_memory[NUM_LAYERS], user_weights_memory[NUM_LAYERS],
               conv1_user_bias_memory[NUM_LAYERS], conv1_dst_memory[NUM_LAYERS];

        int batch[num_test_cases], channel[num_test_cases], height[num_test_cases],
            width[num_test_cases],
            no_of_filter[num_test_cases], kernel_h[num_test_cases],
            kernel_w[num_test_cases],
            pad_h[num_test_cases], pad_w[num_test_cases], stride_h[num_test_cases],
            stride_w[num_test_cases],
            out_height[num_test_cases], out_width[num_test_cases];

        for (int test_num = 0; test_num < num_test_cases; ++test_num) {

            //std::cout << "Memory creation started: (" << (test_num+1) << "/" << num_test_cases << ")" << '\r' << flush;

            batch[test_num] = 64; //conv_test_dimension[test_num][0];
            out_height[test_num] = conv_test_dimension[test_num][1];
            out_width[test_num] = conv_test_dimension[test_num][2];
            no_of_filter[test_num] = conv_test_dimension[test_num][3];
            kernel_h[test_num] = conv_test_dimension[test_num][4];
            kernel_w[test_num] = conv_test_dimension[test_num][5];
            channel[test_num] = conv_test_dimension[test_num][6];
            // no_of_filter = 1x1conv_test_dimension[test_num][7]; already set
            stride_h[test_num] = conv_test_dimension[test_num][8]; // same as [9]
            stride_w[test_num] = conv_test_dimension[test_num][10]; // same as [11]
            pad_type = conv_test_dimension[test_num][12];

            //std::cout << "dimensions : " << conv_test_dimension[test_num][1] << ", " << conv_test_dimension[test_num][2] << std::endl;

            bool lpgemm_path = 0;
            if (kernel_h[test_num] == 1 && !all_direct_val) {
                lpgemm_path = 1;
                ++num_lpgemm_layer;
            }
            else {
                ++num_direct_layer;
            }

            // If pad_type = 0 => VALID
            if (pad_type == 0) {
                pad_w[test_num] = 0;
                pad_h[test_num] = 0;
                height[test_num] = (out_height[test_num] - 1) * stride_h[test_num] +
                                   kernel_h[test_num];
                width[test_num] = (out_width[test_num] - 1) * stride_w[test_num] +
                                  kernel_w[test_num];
            }
            else {
                // SAME padding
                pad_w[test_num] = (out_width[test_num] - 1) * stride_w[test_num] +
                                  kernel_w[test_num] - out_width[test_num];
                pad_h[test_num] = (out_height[test_num] - 1) * stride_h[test_num] +
                                  kernel_h[test_num] - out_height[test_num];
                height[test_num] = out_height[test_num];
                width[test_num] = out_width[test_num];
            }

            operations = ((out_height[test_num] * out_width[test_num] * kernel_h[test_num] *
                           kernel_w[test_num] * channel[test_num] *
                           no_of_filter[test_num])/GFLOP_SIZE) * 2.0 * batch[test_num];

            total_operations += operations;

            using tag = memory::format_tag;
            using dt = memory::data_type;
            memory::dims conv1_src_tz = {batch[test_num], channel[test_num], height[test_num], width[test_num]};
            memory::dims conv1_weights_tz = {no_of_filter[test_num], channel[test_num], kernel_h[test_num], kernel_w[test_num]};
            memory::dims conv1_bias_tz = {no_of_filter[test_num]};
            memory::dims conv1_dst_tz = {batch[test_num], no_of_filter[test_num], out_height[test_num], out_width[test_num]};

            //memory allocation
            if (lpgemm_path == 1)
                user_src_memory[test_num] = memory({{conv1_src_tz}, dt::bf16, tag::nhwc}, eng);
            else
                user_src_memory[test_num] = memory({{conv1_src_tz}, dt::bf16, tag::nhwc}, eng);
            if (lpgemm_path == 1)
                user_weights_memory[test_num] = memory({{conv1_weights_tz}, dt::bf16, tag::hwcn},
            eng); //cdba is hwcn for zendnn lib
            else
                user_weights_memory[test_num] = memory({{conv1_weights_tz}, dt::bf16, tag::hwcn},
            eng); //cdba is hwcn for zendnn lib
            conv1_user_bias_memory[test_num] = memory({{conv1_bias_tz}, dt::f32, tag::x},
            eng);
            if ((lpgemm_path == 1 && s32_downscaling_val != 0) || lpgemm_path == 0)
                conv1_dst_memory[test_num] = memory({{conv1_dst_tz}, dt::bf16, tag::aBcd8b },
            eng);
            else
                conv1_dst_memory[test_num] = memory({{conv1_dst_tz}, dt::f32, tag::aBcd8b },
            eng);

            //data initialization
            /*
            init_data_u8(user_src_memory[test_num], -1);
            init_data_s8(user_weights_memory[test_num], -1);
            if (lpgemm_path == 0) {
                init_data_s32(conv1_user_bias_memory[test_num], 0);
            }
            else {
                init_data_s32(conv1_user_bias_memory[test_num], 0);
            }
            init_data_s8(conv1_dst_memory[test_num], 0);*/

        }

        for (int num_iteration = 0; num_iteration < ITERATIONS + WARMUP;
                num_iteration++) {

            std::cout << "Inference Iterations started: (" << (num_iteration+1) << "/" <<
                      ITERATIONS + WARMUP << ")" << '\r' << flush;

            for (int test_num = 0; test_num < num_test_cases; ++test_num) {

                bool lpgemm_path = 0;
                if (kernel_h[test_num] == 1 && !all_direct_val) {
                    lpgemm_path = 1;
                }

                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

                if (lpgemm_path == 1) {
                    convolution_param(eng, user_src_memory[test_num], batch[test_num],
                                      channel[test_num], height[test_num], width[test_num],
                                      user_weights_memory[test_num],
                                      no_of_filter[test_num], kernel_h[test_num], kernel_w[test_num], pad_h[test_num],
                                      pad_w[test_num], stride_h[test_num], stride_w[test_num],
                                      conv1_user_bias_memory[test_num], conv1_dst_memory[test_num],
                                      out_height[test_num], out_width[test_num]);
                }
                else {
                    convolution_ref_direct(eng, user_src_memory[test_num], batch[test_num],
                                           channel[test_num], height[test_num], width[test_num],
                                           user_weights_memory[test_num],
                                           no_of_filter[test_num], kernel_h[test_num], kernel_w[test_num], pad_h[test_num],
                                           pad_w[test_num], stride_h[test_num], stride_w[test_num],
                                           conv1_user_bias_memory[test_num], conv1_dst_memory[test_num],
                                           out_height[test_num], out_width[test_num]);
                }

                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                time_taken = std::chrono::duration_cast<std::chrono::microseconds>
                             (end - begin).count();

                lpgemm_path = 0;
                if (num_iteration >= WARMUP) {
                    total_time_taken += time_taken;
                }
            }
        }

        // Calculate GFLOPs
        double gflops = total_operations * ITERATIONS / total_time_taken;
        std::cout << "\nAverage GFLOPs: " << gflops << std::endl;
        std::cout << "Average throughput: " << ((batch[0] * 1000000 / total_time_taken)
                                                * ITERATIONS) << " images/second\n" << std::endl;


        std::cout << "Total " << (num_direct_layer + num_lpgemm_layer) << " layers: " <<
                  num_direct_layer << " direct path, " << num_lpgemm_layer << " lpgemm path." <<
                  std::endl;
        std::cout << "Inference Iterations: " << ITERATIONS << " with " << WARMUP <<
                  " warmup Iterations" << std::endl;
        std::cout << "\n--------------------------------------------\n" << std::endl;
    }
    catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }

    zendnnInfo(ZENDNN_TESTLOG, "zendnn_avx_conv API test ends\n");
    return 0;
}
