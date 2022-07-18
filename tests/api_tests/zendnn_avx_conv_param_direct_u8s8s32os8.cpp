/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <assert.h>

#include <chrono>
#include <numeric>
#include <vector>
#include <unordered_map>

#include "test_utils.hpp"
#include "zendnn_logging.hpp"

#define ZENDNN_CMP_OUTPUT   1

using namespace zendnn;
using namespace std;

memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
                           std::multiplies<memory::dim>());
}

//function to init data
void init_data(memory &m) {
    size_t size = m.get_desc().get_size() /sizeof(float);
    //std::vector<float> data(size);
    srand(1111);
    int8_t *data= (int8_t *)m.get_data_handle();
    for (size_t i = 0; i < size; ++i) {
        data[i] = rand()%5;
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
    auto conv1_src_md = memory::desc({conv1_src_tz}, dt::u8, tag::acdb);
    auto conv1_bias_md = memory::desc({conv1_bias_tz}, dt::s32, tag::x);
    auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::s8, tag::any);
    auto conv1_dst_md = memory::desc({conv1_dst_tz}, dt::s8, tag::acdb);
    //[Create convolution memory descriptors]

    /// Create a convolution descriptor by specifying propagation kind,
    /// [convolution algorithm](@ref dev_guide_convolution), shapes of input,
    /// weights, bias, output, convolution strides, padding, and kind of padding.
    /// Propagation kind is set to prop_kind::forward_inference to optimize for
    /// inference execution and omit computations that are necessary only for
    /// backward propagation.
    /// @snippet cnn_inference_f32.cpp Create convolution descriptor
    //[Create convolution descriptor]
    auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
                      algorithm::convolution_direct, conv1_src_md, conv1_weights_md,
                      conv1_bias_md, conv1_dst_md, conv1_strides, conv1_padding,
                      conv1_padding);
    //[Create convolution descriptor]

    /// Create a convolution primitive descriptor. Once created, this
    /// descriptor has specific formats instead of the `any` format specified
    /// in the convolution descriptor.
    /// @snippet cnn_inference_f32.cpp Create convolution primitive descriptor
    //[Create convolution primitive descriptor]
    auto conv1_prim_desc = convolution_forward::primitive_desc(conv1_desc, eng);
    //[Create convolution primitive descriptor]

    /// Check whether data and weights formats required by convolution is different
    /// from the user format. In case it is different change the layout using
    /// reorder primitive.
    /// @snippet cnn_inference_f32.cpp Reorder data and weights
    //[Reorder data and weights]
    auto conv1_src_memory = user_src_memory;
    if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
        conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
        net.push_back(reorder(user_src_memory, conv1_src_memory));
        net_args.push_back({{ZENDNN_ARG_FROM, user_src_memory},
            {ZENDNN_ARG_TO, conv1_src_memory}});
    }

    auto conv1_weights_memory = user_weights_memory;
    if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
        reorder(user_weights_memory, conv1_weights_memory)
        .execute(s, user_weights_memory, conv1_weights_memory);
    }
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
    net_args.push_back({{ZENDNN_ARG_SRC, conv1_src_memory},
        {ZENDNN_ARG_WEIGHTS, conv1_weights_memory},
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
    //[Execute model]

    s.wait();

}

int main(int argc, char **argv) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_avx_conv API test starts");
    try {
        std::ofstream file;

        //Input parameters to convolution
        int times = 1; //100
        int batch = 640;
        int channel = 1024; //512; //256; //64;
        int height = 7; //14; //28; //56;
        int width = 7; //14; //28; //56;
        int no_of_filter = 512; //256; //128; //64;
        int kernel_h = 1;
        int kernel_w = 1;
        int pad_h = 0;
        int pad_w = 0;
        int stride_h = 1;
        int stride_w = 1;

        int out_height = (height + pad_h + pad_w - kernel_h) / stride_h + 1;
        int out_width = (width + pad_h + pad_w - kernel_w) / stride_w + 1;

        using tag = memory::format_tag;
        using dt = memory::data_type;
        memory::dims conv1_src_tz = {batch, channel, height, width};
        memory::dims conv1_weights_tz = {no_of_filter, channel, kernel_h, kernel_w};
        memory::dims conv1_bias_tz = {no_of_filter};
        memory::dims conv1_dst_tz = {batch, no_of_filter, out_height, out_width};

        engine::kind engine_kind = parse_engine_kind(argc, argv);
        engine eng(engine_kind, 0);
        stream s(eng);

        //memory allocation
        auto user_src_memory = memory({{conv1_src_tz}, dt::u8, tag::nhwc}, eng);
        auto user_weights_memory = memory({{conv1_weights_tz}, dt::s8, tag::hwcn},
        eng); //cdba is hwcn for zendnn lib
        auto conv1_user_bias_memory = memory({{conv1_bias_tz}, dt::s32, tag::x}, eng);
        auto conv1_dst_memory = memory({{conv1_dst_tz}, dt::s32, tag::aBcd8b }, eng);

        //data initialization
        init_data(user_src_memory);
        init_data(user_weights_memory);
        init_data(conv1_user_bias_memory);

        int count = 1;
        while (count < 100) {
            ++count;

            auto begin = chrono::duration_cast<chrono::microseconds>(
                             chrono::steady_clock::now().time_since_epoch())
                         .count();

            convolution_param(eng, user_src_memory, batch, channel, height, width,
                              user_weights_memory,
                              no_of_filter, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                              conv1_user_bias_memory, conv1_dst_memory, out_height, out_width);


            auto end = chrono::duration_cast<chrono::microseconds>(
                           chrono::steady_clock::now().time_since_epoch())
                       .count();
            zendnnInfo(ZENDNN_TESTLOG, "Use time ", (end - begin) / (times + 0.0));
            std::cout << "time = " << (end - begin) << std::endl;

        }

        //Output verification
        //Dump the output buffer to a file
        const char *zenDnnRootPath = getenv("ZENDNN_GIT_ROOT");
        auto conv1_dst_memory_new = memory({{conv1_dst_tz}, dt::f32, tag::nhwc}, eng);
        reorder(conv1_dst_memory, conv1_dst_memory_new).execute(s, conv1_dst_memory,
                conv1_dst_memory_new);
        float *dataHandle= (float *)conv1_dst_memory_new.get_data_handle();
        file.open(zenDnnRootPath +
                  std::string("/_out/tests/zendnn_avx_conv_param_output_direct"));

        double sum = 0;
        size_t size = conv1_dst_memory.get_desc().get_size() /sizeof(float);
        for (size_t i = 0; i < size; ++i) {
            sum += dataHandle[i];
        }

        //write the dataHandle to the file
        file.write(reinterpret_cast<char const *>(dataHandle),
                   conv1_dst_memory.get_desc().get_size());
        file.close();

        zendnnInfo(ZENDNN_TESTLOG, "ZenDNN API test: SUM: ", sum);
        string str = std::string("sha1sum ") + zenDnnRootPath +
                     std::string("/_out/tests/zendnn_avx_conv_param_output_direct > ") +
                     zenDnnRootPath +
                     std::string("/_out/tests/zendnn_avx_conv_param_output_direct.sha1");

        //Convert string to const char * as system requires
        //parameters of the type const char *
        const char *command = str.c_str();
        int status = system(command);

#if ZENDNN_CMP_OUTPUT //compare SHA1 value
        ifstream zenDnnSha1(zenDnnRootPath +
                            std::string("/_out/tests/zendnn_avx_conv_param_output_direct.sha1"));
        string firstWordZen;

        //zendnnInfo(ZENDNN_TESTLOG, "ZenDNN output SHA1 value: ", firstWordZen);

        while (zenDnnSha1 >> firstWordZen) {
            zendnnInfo(ZENDNN_TESTLOG, "ZenDNN output SHA1 value: ", firstWordZen);
            zenDnnSha1.ignore(numeric_limits<streamsize>::max(), '\n');
        }

        ifstream refSha1(zenDnnRootPath +
                         std::string("/tests/api_tests/sha_out_NHWC/ref_avx_conv_param_output.sha1"));
        string firstWordRef;

        while (refSha1 >> firstWordRef) {
            //zendnnInfo(ZENDNN_TESTLOG, "Ref output SHA1 value: ", firstWordRef);
            refSha1.ignore(numeric_limits<streamsize>::max(), '\n');
        }

        //ZENDNN_CHECK(firstWordZen == firstWordRef, ZENDNN_TESTLOG,
        //"sha1 /sha1sum value of ZenDNN output and Ref output do not matches");
#endif //compare SHA1 value


    }
    catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_avx_conv API test ends\n");
    return 0;
}
