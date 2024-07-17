/*******************************************************************************
* Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <cmath>
#include <assert.h>
#include <random>
#include <chrono>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include "zendnn.hpp"
#include "zendnn_helper.hpp"

#include "test_utils.hpp"
#include "zendnn_logging.hpp"

using namespace zendnn;
using namespace std;
using tag = memory::format_tag;
using dt = memory::data_type;

template<typename T>
int compare_vectors(const std::vector<T> &v1, const std::vector<T> &v2,
                    int64_t K, const char *message) {
    double v1_l2 = 0, diff_l2 = 0;
    for (size_t n = 0; n < v1.size(); ++n) {
        float diff = (float)(v1[n] - v2[n]);
        v1_l2 += v1[n] * v1[n];
        diff_l2 += diff * diff;
    }
    v1_l2 = std::sqrt(v1_l2);
    diff_l2 = std::sqrt(diff_l2);
    // Finding the reasonable (tight and accurate) threshold is quite difficult
    // problem.
    // The implementation testing might also use special data filling to
    // alleviate issues related to the finite precision arithmetic.
    // However, in simple cases the machine epsilon multiplied by log(K) should
    // work reasonably well.
    const double threshold = std::numeric_limits<float>::epsilon()
                             * std::log(std::max(2., (double)K));
    bool ok = diff_l2 <= threshold * v1_l2;
    printf("%s\n\tL2 Norms"
           "\n\t\tReference matrix:%g\n\t\tError:%g\n\t\tRelative_error:%g\n",
           message, v1_l2, diff_l2, diff_l2 / v1_l2);
    return ok ? 0 : 1;
}
std::vector<float> matmul_example_2D_f32_ref(bool s4_weights, int post_op) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D FP32 starts");

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    // Tensor dimensions.
    const memory::dim M = 128, K = 256, N = 512;
    //const memory::dim M = 5, K = 5, N = 5;
    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};
    // Allocate buffers.
    std::vector<float> src_data(M * K);
    std::vector<float> weights_data(K * N);
    std::vector<float> bias_data(1 * N);
    std::vector<float> dst_data(M * N);
    // Initialize src, weights, bias.i
    for (int i=0; i<src_data.size(); i++) {
        src_data[i] = std::cos(i / 10.f);
    }
    for (int i=0; i<weights_data.size(); i++) {
        weights_data[i] = 2.0831*(i%5);
        //weights_data[i] = 1.0*(i%5);
    }
    for (int i=0; i<bias_data.size(); i++) {
        bias_data[i] = std::tanh(i);
    }

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
    write_to_zendnn_memory(bias_data.data(), bias_mem);


    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, weights_md,
                                 bias_md,
                                 dst_md);

    // Create primitive post-ops (ReLU).
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops matmul_ops;

    if (post_op == 0) {
        matmul_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
    }
    else if (post_op == 1) {
        matmul_ops.append_eltwise(scale, algorithm::eltwise_gelu, alpha, beta);
    }
    else if (post_op == 2) {
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
    matmul_args.insert({ZENDNN_ARG_BIAS, bias_mem});
    matmul_args.insert({ZENDNN_ARG_DST, dst_mem});
    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim.execute(engine_stream, matmul_args);
    // Wait for the computation to finalize.
    engine_stream.wait();
    // Read data from memory object's handle.
    read_from_zendnn_memory(dst_data.data(), dst_mem);
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D ends");

    return dst_data;
}

std::vector<float> matmul_example_2D_f32(bool s4_weights, int post_op) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D FP32 starts");

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    // Tensor dimensions.
    const memory::dim M = 128, K = 256, N = 512;
    //const memory::dim M = 5, K = 5, N = 5;
    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};
    // Allocate buffers.
    std::vector<float> src_data(M * K);
    std::vector<int8_t> weights_data(K * N);
    std::vector<float> bias_data(1 * N);
    std::vector<float> dst_data(M * N);
    // Initialize src, weights, bias.i
    for (int i=0; i<src_data.size(); i++) {
        src_data[i] = std::cos(i / 10.f);
    }
    int xx=0;
    if (s4_weights) {
        for (int i=0; i<ceil(weights_data.size()/2.0); i++) {
            weights_data[i] = xx++ %5 & 0x0F;
            weights_data[i] |= (xx++ %5 & 0x0F) << 4;
        }
    }
    else {
        for (int i=0; i<weights_data.size(); i++) {
            weights_data[i] = xx++ %5;
        }
    }
    for (int i=0; i<bias_data.size(); i++) {
        bias_data[i] = std::tanh(i);
    }

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md = memory::desc(src_dims, dt::f32, tag::ab);
    auto weights_md_s4 = memory::desc(weights_dims, dt::s4, tag::ab); //s4 md
    auto weights_md_s8 = memory::desc(weights_dims, dt::s8, tag::ab); //s8 md
    auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
    auto dst_md = memory::desc(dst_dims, dt::f32, tag::ab);

    auto src_mem = memory(src_md, eng);
    auto weights_mem = memory(s4_weights? weights_md_s4 : weights_md_s8, eng);
    auto bias_mem = memory(bias_md, eng);
    auto dst_mem = memory(dst_md, eng);

    // Write data to memory object's handles.
    write_to_zendnn_memory(src_data.data(), src_mem);
    write_to_zendnn_memory(weights_data.data(), weights_mem);
    write_to_zendnn_memory(bias_data.data(), bias_mem);


    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, s4_weights? weights_md_s4 : weights_md_s8,
                                 bias_md,
                                 dst_md);

    // Create primitive post-ops (ReLU).
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops matmul_ops;

    if (post_op == 0) {
        matmul_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
    }
    else if (post_op == 1) {
        matmul_ops.append_eltwise(scale, algorithm::eltwise_gelu, alpha, beta);
    }
    else if (post_op == 2) {
        matmul_ops.append_eltwise(scale, algorithm::eltwise_gelu_erf, alpha, beta);
    }

    primitive_attr matmul_attr;
    matmul_attr.set_post_ops(matmul_ops);
    //matmul_attr.set_output_scales(0, {2.0831});
    matmul_attr.set_woq_scale(0, {2.0831});
    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);
    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);
    // Primitive arguments.
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({ZENDNN_ARG_SRC, src_mem});
    matmul_args.insert({ZENDNN_ARG_WEIGHTS, weights_mem});
    matmul_args.insert({ZENDNN_ARG_BIAS, bias_mem});
    matmul_args.insert({ZENDNN_ARG_DST, dst_mem});
    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim.execute(engine_stream, matmul_args);
    // Wait for the computation to finalize.
    engine_stream.wait();
    // Read data from memory object's handle.
    read_from_zendnn_memory(dst_data.data(), dst_mem);
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D ends");

    return dst_data;
}

//Weights are passed as BF16
std::vector<float> matmul_example_2D_bf16_ref(bool s4_weights, bool dst_f32,
        unsigned int post_op) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D starts");

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    // Tensor dimensions.
    const memory::dim M = 128, K = 256, N = 512;
    //const memory::dim M = 5, K = 50, N = 5;
    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};
    // Allocate buffers.
    std::vector<float> src_data(M * K);
    std::vector<float> weights_data(K * N);
    std::vector<float> bias_data(1 * N);
    std::vector<float> dst_data(M * N);
    // Initialize src, weights, bias.i
    for (int i=0; i<src_data.size(); i++) {
        src_data[i] = std::cos(i / 10.f);
    }
    for (int i=0; i<weights_data.size(); i++) {
        weights_data[i] = 2.0831*(i%5);
    }
    for (int i=0; i<bias_data.size(); i++) {
        bias_data[i] = std::tanh(i);
    }
    for (int i=0; i<dst_data.size(); i++) {
        dst_data[i] = 0;//std::tanh(i);
    }

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md_f = memory::desc(src_dims, dt::f32, tag::ab);
    auto weights_md_f = memory::desc(weights_dims, dt::f32, tag::ab);
    auto bias_md_f = memory::desc(bias_dims, dt::f32, tag::ab);
    auto dst_md_f = memory::desc(dst_dims, dt::f32, tag::ab);

    auto src_mem = memory(src_md_f, eng);
    auto weights_mem = memory(weights_md_f, eng);
    auto bias_mem = memory(bias_md_f, eng);
    auto dst_mem = memory(dst_md_f, eng);

    // Write data to memory object's handles.
    write_to_zendnn_memory(src_data.data(), src_mem);
    write_to_zendnn_memory(weights_data.data(), weights_mem);
    write_to_zendnn_memory(bias_data.data(), bias_mem);
    write_to_zendnn_memory(dst_data.data(), dst_mem);

    //Create bf16 memory desc
    auto src_md = memory::desc(src_dims, dt::bf16, tag::ab);
    auto weights_md = memory::desc(weights_dims, dt::bf16, tag::ab);
    auto bias_md = memory::desc(bias_dims, dt::bf16, tag::ab);
    auto dst_md = memory::desc(dst_dims, dt::bf16, tag::ab);

    //Create bf16 memory
    auto src_bf_mem = memory(src_md, eng);
    auto weights_bf_mem = memory(weights_md, eng);
    auto bias_bf_mem = memory(bias_md, eng);
    auto dst_bf_mem = memory(dst_md, eng);

    //reorder the f32 to bf16 and execute
    reorder(src_mem, src_bf_mem).execute(engine_stream,src_mem,src_bf_mem);
    reorder(weights_mem, weights_bf_mem).execute(engine_stream,weights_mem,
            weights_bf_mem);
    if (!dst_f32) {
        reorder(bias_mem, bias_bf_mem).execute(engine_stream,bias_mem, bias_bf_mem);
        reorder(dst_mem, dst_bf_mem).execute(engine_stream,dst_mem, dst_bf_mem);
    }

    auto bias_arg_mem = dst_f32 ? bias_mem : bias_bf_mem;
    auto dst_arg_mem = dst_f32 ? dst_mem : dst_bf_mem;

    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, weights_md, dst_f32? bias_md_f:bias_md,
                                 dst_f32?dst_md_f:dst_md);

    // Create primitive post-ops (ReLU).
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops matmul_ops;

    if (post_op == 0) {
        matmul_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
    }
    else if (post_op == 1) {
        matmul_ops.append_eltwise(scale, algorithm::eltwise_gelu, alpha, beta);
    }
    else if (post_op == 2) {
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
    matmul_args.insert({ZENDNN_ARG_SRC, src_bf_mem});
    matmul_args.insert({ZENDNN_ARG_WEIGHTS, weights_bf_mem});
    matmul_args.insert({ZENDNN_ARG_BIAS, bias_arg_mem});
    matmul_args.insert({ZENDNN_ARG_DST, dst_arg_mem});
    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim.execute(engine_stream, matmul_args);
    // Wait for the computation to finalize.
    engine_stream.wait();
    // Read data from memory object's handle.
    read_from_zendnn_memory(dst_data.data(), dst_arg_mem);
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D ends");

    return dst_data;
}
std::vector<float> matmul_example_2D_bf16(bool s4_weights, bool dst_f32,
        unsigned int post_op) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D starts");

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    // Tensor dimensions.
    const memory::dim M = 128, K = 256, N = 512;
    //const memory::dim M = 5, K = 50, N = 5;
    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};
    // Allocate buffers.
    std::vector<float> src_data(M * K);
    std::vector<int8_t> weights_data(K * N);
    std::vector<float> bias_data(1 * N);
    std::vector<float> dst_data(M * N);
    // Initialize src, weights, bias.i
    for (int i=0; i<src_data.size(); i++) {
        src_data[i] = std::cos(i / 10.f);
    }
    int xx=0;
    if (s4_weights) {
        for (int i=0; i<ceil(weights_data.size()/2.0); i++) {
            weights_data[i] = (xx++ %5) & 0x0F;
            weights_data[i] |= (xx++ %5) << 4;
        }
    }
    else {
        for (int i=0; i<weights_data.size(); i++) {
            weights_data[i] = i%5;
        }
    }
    for (int i=0; i<bias_data.size(); i++) {
        bias_data[i] = std::tanh(i);
    }
    for (int i=0; i<dst_data.size(); i++) {
        dst_data[i] = 0;//std::tanh(i);
    }

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md_f = memory::desc(src_dims, dt::f32, tag::ab);
    auto weights_md_s4 = memory::desc(weights_dims, dt::s4, tag::ab); //s4 md
    auto weights_md_s8 = memory::desc(weights_dims, dt::s8, tag::ab); //s8 md
    auto bias_md_f = memory::desc(bias_dims, dt::f32, tag::ab);
    auto dst_md_f = memory::desc(dst_dims, dt::f32, tag::ab);

    auto src_mem = memory(src_md_f, eng);
    auto weights_mem = memory(s4_weights? weights_md_s4 : weights_md_s8, eng);
    auto bias_mem = memory(bias_md_f, eng);
    auto dst_mem = memory(dst_md_f, eng);

    // Write data to memory object's handles.
    write_to_zendnn_memory(src_data.data(), src_mem);
    write_to_zendnn_memory(weights_data.data(), weights_mem);
    write_to_zendnn_memory(bias_data.data(), bias_mem);
    write_to_zendnn_memory(dst_data.data(), dst_mem);

    //Create bf16 memory desc
    auto src_md = memory::desc(src_dims, dt::bf16, tag::ab);
    auto bias_md = memory::desc(bias_dims, dt::bf16, tag::ab);
    auto dst_md = memory::desc(dst_dims, dt::bf16, tag::ab);

    //Create bf16 memory
    auto src_bf_mem = memory(src_md, eng);
    auto bias_bf_mem = memory(bias_md, eng);
    auto dst_bf_mem = memory(dst_md, eng);

    //reorder the f32 to bf16 and execute
    reorder(src_mem, src_bf_mem).execute(engine_stream,src_mem,src_bf_mem);
    if (!dst_f32) {
        reorder(bias_mem, bias_bf_mem).execute(engine_stream,bias_mem, bias_bf_mem);
        reorder(dst_mem, dst_bf_mem).execute(engine_stream, dst_mem, dst_bf_mem);
    }

    auto bias_arg_mem = dst_f32 ? bias_mem : bias_bf_mem;
    auto dst_arg_mem = dst_f32 ? dst_mem : dst_bf_mem;

    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, s4_weights? weights_md_s4 : weights_md_s8,
                                 dst_f32? bias_md_f:bias_md,
                                 dst_f32?dst_md_f:dst_md);

    // Create primitive post-ops (ReLU).
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops matmul_ops;

    if (post_op == 0) {
        matmul_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
    }
    else if (post_op == 1) {
        matmul_ops.append_eltwise(scale, algorithm::eltwise_gelu, alpha, beta);
    }
    else if (post_op == 2) {
        matmul_ops.append_eltwise(scale, algorithm::eltwise_gelu_erf, alpha, beta);
    }

    primitive_attr matmul_attr;
    matmul_attr.set_post_ops(matmul_ops);
    //matmul_attr.set_output_scales(0, {2.0831});
    matmul_attr.set_woq_scale(0, {2.08});
    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);
    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);
    // Primitive arguments.
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({ZENDNN_ARG_SRC, src_bf_mem});
    matmul_args.insert({ZENDNN_ARG_WEIGHTS, weights_mem});
    matmul_args.insert({ZENDNN_ARG_BIAS, bias_arg_mem});
    matmul_args.insert({ZENDNN_ARG_DST, dst_arg_mem});
    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim.execute(engine_stream, matmul_args);
    // Wait for the computation to finalize.
    engine_stream.wait();
    // Read data from memory object's handle.
    read_from_zendnn_memory(dst_data.data(), dst_arg_mem);
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D ends");

    return dst_data;
}

void wrapper_woq(int val, int of32) {
    if (val == 0 || val == 1) {
        if (!of32) {
            std::vector<float> ref_dst = matmul_example_2D_bf16_ref(true, false, 1);
            auto woq_dst = matmul_example_2D_bf16(true, false, 1);
            compare_vectors(
                woq_dst, ref_dst, 256, "Compare s4 weights | comp BF16 output BF16");
            auto woq_dst2 = matmul_example_2D_bf16(false, false, 1);
            compare_vectors(
                woq_dst2, ref_dst, 256, "Compare s8 weights | comp BF16 output BF16");
        }
        if (of32) {
            auto ref_dst = matmul_example_2D_bf16_ref(true, true, 1);
            auto woq_dst = matmul_example_2D_bf16(true, true, 1);
            compare_vectors(
                woq_dst, ref_dst, 256, "Compare s4 weights | comp BF16 output FP32");
            woq_dst = matmul_example_2D_bf16(false, true, 1);
            compare_vectors(
                woq_dst, ref_dst, 256, "Compare s8 weights | comp BF16 output FP32");
        }
    }
    if (val == 0 || val == 2) {
        auto ref_dst = matmul_example_2D_f32_ref(true, 1);
        auto woq_dst = matmul_example_2D_f32(true, 1);
        compare_vectors(
            woq_dst, ref_dst, 256, "Compare s4 weights Compute F32");
        woq_dst = matmul_example_2D_f32(false, 1);
        compare_vectors(
            woq_dst, ref_dst, 256, "Compare s8 weights Compute FP32");
    }
}

int main(int argc, char **argv) {
    //return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test test starts");
    int test_call = 0, dst_f32 = 1;
    if (argc > 1) {
        //arg[1] 0:bf16 output| 1:f32 output
        test_call = std::stoi(std::string(argv[1]));
    }
    if (argc > 2) {
        //arg[1] 0:bf16 output| 1:f32 output
        dst_f32 = std::stoi(std::string(argv[2]));
    }

    wrapper_woq(test_call, dst_f32);

    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test test ends");
    return 0;
}

