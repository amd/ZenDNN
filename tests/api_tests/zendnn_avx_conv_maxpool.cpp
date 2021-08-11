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
#include <cstring>

#include "test_utils.hpp"
#include "zendnn_logging.hpp"

#define ZENDNN_CMP_OUTPUT   0

using namespace zendnn;
using namespace std;

memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
                           std::multiplies<memory::dim>());
}

//function to init data
void init_data(memory &m, float v) {
    size_t size = m.get_desc().get_size() /sizeof(float);
    //std::vector<float> data(size);
    srand(1111);
    float *data= (float *)m.get_data_handle();
    for (size_t i = 0; i < size; ++i) {
        data[i] = rand()%5;
    }
}


void simple_net(engine::kind engine_kind, int times = 100) {
    std::ofstream file;

    using tag = memory::format_tag;
    using dt = memory::data_type;

    /// Initialize an engine and stream. The last parameter in the call represents
    /// the index of the engine.
    /// @snippet cnn_inference_f32.cpp Initialize engine and stream
    //[Initialize engine and stream]
    engine eng(engine_kind, 0);
    stream s(eng);
    //[Initialize engine and stream]

    /// Create a vector for the primitives and a vector to hold memory
    /// that will be used as arguments.
    /// @snippet cnn_inference_f32.cpp Create network
    //[Create network]
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;
    //[Create network]

    const memory::dim batch = 1;

    // AlexNet: conv1
    // {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 55, 55}
    // strides: {4, 4}
    memory::dims conv1_src_tz = {batch, 3, 227, 227};
    memory::dims conv1_weights_tz = {96, 3, 11, 11};
    memory::dims conv1_bias_tz = {96};
    memory::dims conv1_dst_tz = {batch, 96, 55, 55};
    memory::dims conv1_strides = {4, 4};
    memory::dims conv1_padding = {0, 0};

    /// Allocate buffers for input and output data, weights, and bias.
    /// @snippet cnn_inference_f32.cpp Allocate buffers
    //[Allocate buffers]
    std::vector<float> user_src(batch * 3 * 227 * 227);
    std::vector<float> user_dst(batch * 1000);
    std::vector<float> conv1_weights(product(conv1_weights_tz));
    std::vector<float> conv1_bias(product(conv1_bias_tz));
    //[Allocate buffers]

    //Fill source, destination and weights with synthetic data

    /// Create memory that describes data layout in the buffers. This example uses
    /// tag::nchw (batch-channels-height-width) for input data and tag::oihw
    /// for weights.
    /// @snippet cnn_inference_f32.cpp Create user memory
    //[Create user memory]
    auto user_src_memory = memory({{conv1_src_tz}, dt::f32, tag::nhwc}, eng);
    init_data(user_src_memory, 1);
    write_to_zendnn_memory(user_src.data(), user_src_memory);
    auto user_weights_memory
    = memory({{conv1_weights_tz}, dt::f32, tag::hwcn}, eng); //cdba is hwcn for zendnn lib

    init_data(user_weights_memory, .5);
    write_to_zendnn_memory(conv1_weights.data(), user_weights_memory);
    auto conv1_user_bias_memory
    = memory({{conv1_bias_tz}, dt::f32, tag::x}, eng);
    init_data(conv1_user_bias_memory, .5);
    write_to_zendnn_memory(conv1_bias.data(), conv1_user_bias_memory);

    //Initialize buffers
    init_data(user_src_memory, 2);
    init_data(user_weights_memory, .4);
    init_data(conv1_user_bias_memory, .5);

    //[Create user memory]

    /// Create memory descriptors with layout tag::any. The `any` format enables
    /// the convolution primitive to choose the data format that will result in
    /// best performance based on its input parameters (convolution kernel
    /// sizes, strides, padding, and so on). If the resulting format is different
    /// from `nchw`, the user data must be transformed to the format required for
    /// the convolution (as explained below).
    /// @snippet cnn_inference_f32.cpp Create convolution memory descriptors
    //[Create convolution memory descriptors]
    auto conv1_src_md = memory::desc({conv1_src_tz}, dt::f32, tag::any);
    auto conv1_bias_md = memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::f32, tag::any);
    auto conv1_dst_md = memory::desc({conv1_dst_tz}, dt::f32, tag::any);
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
                      algorithm::convolution_gemm, conv1_src_md, conv1_weights_md,
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
#if !ZENDNN_ENABLE
        net.push_back(reorder(user_src_memory, conv1_src_memory));
        net_args.push_back({{ZENDNN_ARG_FROM, user_src_memory},
            {ZENDNN_ARG_TO, conv1_src_memory}});
#endif
    }

    auto conv1_weights_memory = user_weights_memory;
    if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
#if !ZENDNN_ENABLE
        reorder(user_weights_memory, conv1_weights_memory)
        .execute(s, user_weights_memory, conv1_weights_memory);
#endif
    }
    //[Reorder data and weights]

    /// Create a memory primitive for output.
    /// @snippet cnn_inference_f32.cpp Create memory for output
    //[Create memory for output]
    auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);
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


    // AlexNet: pool1
    // {batch, 96, 55, 55} -> {batch, 96, 27, 27}
    // kernel: {3, 3}
    // strides: {2, 2}
    memory::dims pool1_dst_tz = {batch, 96, 27, 27};
    memory::dims pool1_kernel = {3, 3};
    memory::dims pool1_strides = {2, 2};
    memory::dims pool_padding = {0, 0};

    auto pool1_dst_md = memory::desc({pool1_dst_tz}, dt::f32, tag::any);

    /// For training execution, pooling requires a private workspace memory
    /// to perform the backward pass. However, pooling should not use 'workspace'
    /// for inference, because this is detrimental to performance.
    /// @snippet cnn_inference_f32.cpp Create pooling primitive
    ///
    /// The example continues to create more layers according
    /// to the AlexNet topology.
    //[Create pooling primitive]
    auto pool1_desc = pooling_forward::desc(prop_kind::forward_inference,
                                            algorithm::pooling_max, conv1_dst_memory.get_desc(), pool1_dst_md,
                                            pool1_strides, pool1_kernel, pool_padding, pool_padding);
    auto pool1_pd = pooling_forward::primitive_desc(pool1_desc, eng);
    auto pool1_dst_memory = memory(pool1_pd.dst_desc(), eng);

    net.push_back(pooling_forward(pool1_pd));
    net_args.push_back({{ZENDNN_ARG_SRC, conv1_dst_memory},
        {ZENDNN_ARG_DST, pool1_dst_memory}});
    //[Create pooling primitive]


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

    //Output verification
    //Dump the output buffer to a file
    const char *zenDnnRootPath = getenv("ZENDNN_GIT_ROOT");
#if !ZENDNN_ENABLE
    auto pool1_dst_memory_new = memory({{pool1_dst_tz}, dt::f32, tag::nhwc}, eng);
    reorder(pool1_dst_memory, pool1_dst_memory_new).execute(s, pool1_dst_memory, pool1_dst_memory_new);
    float *dataHandle= (float *)pool1_dst_memory_new.get_data_handle();
    file.open(zenDnnRootPath + std::string("/_out/tests/ref_avx_conv_maxpool_output"));
#else
    float *dataHandle= (float *)pool1_dst_memory.get_data_handle();
    file.open(zenDnnRootPath + std::string("/_out/tests/zendnn_avx_conv_maxpool_output"));
#endif

    double sum = 0;
    size_t size = pool1_dst_memory.get_desc().get_size() /sizeof(float);
    for (size_t i = 0; i < size; ++i) {
        sum += dataHandle[i];
        //file << dataHandle[i] << endl;
    }

    //write the dataHandle to the file
    file.write(reinterpret_cast<char const *>(dataHandle), pool1_dst_memory.get_desc().get_size());
    file.close();

#if ZENDNN_ENABLE
    zendnnInfo(ZENDNN_TESTLOG, "ZenDNN API test: SUM: ", sum);
    string str = std::string("sha1sum ") + zenDnnRootPath +
        std::string("/_out/tests/zendnn_avx_conv_maxpool_output >") + zenDnnRootPath +
        std::string("/_out/tests/zendnn_avx_conv_maxpool_output.sha1");
#else
    zendnnInfo(ZENDNN_TESTLOG, "Ref API test: SUM: ", sum);
    string str = std::string("sha1sum ") + zenDnnRootPath +
        std::string("/_out/tests/ref_avx_conv_maxpool_output > ") + zenDnnRootPath +
        std::string("/tests/api_tests/sha_out_NHWC/ref_avx_conv_maxpool_output.sha1");
#endif

    //Convert string to const char * as system requires
    //parameters of the type const char *
    char *str_cpy = new char[str.size()+1] ;
    strcpy(str_cpy, str.c_str());
    const char *command = str_cpy;
    int status = system(command);

#if ZENDNN_CMP_OUTPUT //compare SHA1 value
    ifstream zenDnnSha1(zenDnnRootPath + std::string("/_out/tests/zendnn_avx_conv_maxpool_output.sha1"));
    string firstWordZen;

    while (zenDnnSha1 >> firstWordZen) {
        zendnnInfo(ZENDNN_TESTLOG, "ZenDNN output SHA1 value: ", firstWordZen);
        zenDnnSha1.ignore(numeric_limits<streamsize>::max(), '\n');
    }

    ifstream refSha1(zenDnnRootPath +
            std::string("/tests/api_tests/sha_out_NHWC/ref_avx_conv_maxpool_output.sha1"));
    string firstWordRef;

    while (refSha1 >> firstWordRef) {
        zendnnInfo(ZENDNN_TESTLOG, "Ref output SHA1 value: ", firstWordRef);
        refSha1.ignore(numeric_limits<streamsize>::max(), '\n');
    }

    ZENDNN_CHECK(firstWordZen == firstWordRef, ZENDNN_TESTLOG,
                 "sha1 /sha1sum value of ZenDNN output and Ref output do not matches");
#endif //compare SHA1 value

}

int main(int argc, char **argv) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_avx_conv_maxpool API test starts");
    try {
        auto begin = chrono::duration_cast<chrono::milliseconds>(
                         chrono::steady_clock::now().time_since_epoch())
                     .count();
        int times = 1; //100
        simple_net(parse_engine_kind(argc, argv), times);
        auto end = chrono::duration_cast<chrono::milliseconds>(
                       chrono::steady_clock::now().time_since_epoch())
                   .count();
        zendnnInfo(ZENDNN_TESTLOG, "Use time ", (end - begin) / (times + 0.0));
    }
    catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_avx_conv_maxpool API test ends\n");
    return 0;
}
