/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iomanip>
#include <random>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include "zendnn.hpp"
#include "test_utils.hpp"
#include "zendnn_logging.hpp"


using namespace zendnn;
using tag = memory::format_tag;
using dt = memory::data_type;

// Tensor dimensions.
const memory::dim MB = 3, // batch size
                  M = 128, K = 256, N = 512;

// Allocate buffers.
std::vector<float> src_data(M *K);
std::vector<float> weights_data(K *N);
std::vector<float> bias_data(1 * N);
std::vector<float> dst_data(M *N);
std::vector<float> dst_data_buff(M *N);
void matmul_example_2D(zendnn::engine eng, zendnn::stream engine_stream,
                       std::string fusion) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D starts");

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md = memory::desc(src_dims, dt::f32, tag::ab);
    auto weights_md = memory::desc(weights_dims, dt::f32, tag::ab);
    auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
    auto dst_md = memory::desc(dst_dims, dt::f32, tag::ab);
    auto src_mem = memory(src_md, eng);
    auto weights_mem = memory(weights_md, eng);
    auto bias_mem = memory(bias_md, eng);
    auto dst_mem = memory(dst_md, eng);
    // Write data to memory object's handles.
    write_to_zendnn_memory(src_data.data(), src_mem);
    write_to_zendnn_memory(weights_data.data(), weights_mem);
    if (fusion=="kBiasAdd"
            || fusion=="kBiasAddWithRelu"
            || fusion=="kBiasAddWithAdd"
            || fusion=="kBiasAddWithAddAndRelu"
            || fusion=="kBiasAddWithGeluApproximate"
            || fusion=="kBiasAddWithGeluExact") {
        printf("Bias Data Added in zendnn bias_mem\n");
        write_to_zendnn_memory(bias_data.data(), bias_mem);
    }
    if (fusion=="kBiasAddWithAdd"
            || fusion=="kBiasAddWithAddAndRelu") {
        printf("Dst Data Added in zendnn dst_mem\n");
        write_to_zendnn_memory(dst_data_buff.data(), dst_mem);
    }
    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, weights_md, bias_md, dst_md);
    if (fusion=="noFusion") {
        matmul_d = matmul::desc(src_md, weights_md, dst_md);
    }
    // Create primitive post-ops (ReLU).
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops matmul_ops;
    if (fusion=="kBiasAddWithRelu") {
        printf("Relu added as postop using zendnn api\n");
        matmul_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
    }
    if (fusion=="kBiasAddWithAdd") {
        printf("Add/Addv2 added as postop using zendnn api\n");
        matmul_ops.append_sum(scale);
    }
    if (fusion=="kBiasAddWithAddAndRelu") {
        printf("ADD/V2 and Relu added as postop using zendnn api\n");
        matmul_ops.append_sum(scale);
        matmul_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
    }
    if (fusion=="kBiasAddWithGeluApproximate") {
        printf("GeluApproximate added as postop using zendnn api\n");
        matmul_ops.append_eltwise(scale, algorithm::eltwise_gelu, alpha, beta);
    }
    if (fusion=="kBiasAddWithGeluExact") {
        printf("GeluExact added as postop using zendnn api\n");
        matmul_ops.append_eltwise(scale, algorithm::eltwise_gelu_erf, alpha, beta);
    }
    primitive_attr matmul_attr;
    matmul_attr.set_post_ops(matmul_ops);
    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);
    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);
    // Primitive arguments.
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({ZENDNN_ARG_SRC, src_mem});
    matmul_args.insert({ZENDNN_ARG_WEIGHTS, weights_mem});
    if (fusion=="kBiasAdd" || fusion=="kBiasAddWithRelu" ||
            fusion=="kBiasAddWithAdd"
            || fusion=="kBiasAddWithAddAndRelu"|| fusion=="kBiasAddWithGeluApproximate"
            ||fusion=="kBiasAddWithGeluExact")
        matmul_args.insert({ZENDNN_ARG_BIAS, bias_mem});
    matmul_args.insert({ZENDNN_ARG_DST, dst_mem});
    // Primitive execution: matrix multiplication With ReLU.
    matmul_prim.execute(engine_stream, matmul_args);
    // Wait for the computation to finalize.
    engine_stream.wait();
    // Read data from memory object's handle.
    read_from_zendnn_memory(dst_data.data(), dst_mem);
#if 0
    std::ofstream fout(fusion+"_dst.txt");
    fout<<std::setprecision(10);
    for (auto &x: dst_data) {
        fout<<x<<" ";
    }
    fout<<"\n";
    fout.close();
#endif
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D ends");
}

int main(int argc, char **argv) {
    //return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test test starts");

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);
    // Initialize src, weights, bias,dst.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        static int i = 0;
        return std::tanh(i++);
    });
    std::generate(dst_data_buff.begin(), dst_data_buff.end(), []() {
        static int i = 0;
        return std::tanh(i++);
    });

    std::vector<std::string> fusion_type{"noFusion","kBiasAdd","kBiasAddWithRelu",
                                         "kBiasAddWithAdd","kBiasAddWithAddAndRelu",
                                         "kBiasAddWithGeluApproximate","kBiasAddWithGeluExact"};

    for (auto &x: fusion_type) {
        printf("Fusion Type: %s\n",x.c_str());
        matmul_example_2D(eng, engine_stream,x);
        printf("----------------------------------------------------\n");
    }

    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test test ends");
    return 0;
}
