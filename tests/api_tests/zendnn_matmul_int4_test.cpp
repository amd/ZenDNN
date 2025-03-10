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
std::vector<float> matmul_example_2D_f32_ref(bool s4_weights, int post_op,
        std::vector<float> woq_sc, int m, int n, int k,
        int group_size, int quant) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D FP32 starts");

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    // Tensor dimensions.
    const memory::dim M = m, K = k, N = n;
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
    // int idx=0;
    int idx_buff = 0;
    for (int i=0; i<weights_data.size(); i++) {
        if (quant == 0) {
            weights_data[i] = woq_sc[0] * (i%5);
        }
        else {
            weights_data[i] = woq_sc[(i % N) + idx_buff] * (i%5);
            if ((i % (group_size * N)) == (group_size * N) - 1) {
                idx_buff = idx_buff + N;
            }
        }
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

std::vector<float> matmul_example_2D_f32(bool s4_weights, int post_op,
        std::vector<float> woq_sc, int m, int n, int k,
        int group_size, int quant, int scale_type) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D FP32 starts");

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    // Tensor dimensions.
    const memory::dim M = m, K = k, N = n;
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
    memory::dims scale_dims;
    if (quant == 2) {
        matmul_attr.set_woq_scale(3, {group_size, 1}, scale_type == 1?dt::bf16:dt::f32);
        scale_dims = {N *(K/group_size)};
    }
    else if (quant == 1) {
        matmul_attr.set_woq_scale(2, {1, 1}, scale_type == 1?dt::bf16:dt::f32);
        scale_dims = {N};
    }
    else {
        matmul_attr.set_woq_scale(0, {}, scale_type == 1?dt::bf16:dt::f32);
        scale_dims = {1};
    }
    // Create scale memory
    auto scale_mem = memory({scale_dims, dt::f32, tag::a}, eng);
    write_to_zendnn_memory(woq_sc.data(), scale_mem);
    auto scale_bf_mem = memory({scale_dims, dt::bf16, tag::a}, eng);

    if (scale_type == 1) {
        reorder(scale_mem, scale_bf_mem).execute(engine_stream,scale_mem, scale_bf_mem);
    }

    auto scale_arg_mem = scale_type == 1 ? scale_bf_mem : scale_mem;

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
    matmul_args.insert({ZENDNN_ARG_ATTR_WOQ_SCALES, scale_arg_mem});
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
        unsigned int post_op, std::vector<float> woq_sc, int m, int n, int k,
        int group_size, int quant) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D starts");

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    // Tensor dimensions.
    const memory::dim M = m, K = k, N = n;
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
    int idx =0;
    int idx_buff = 0;
    for (int i=0; i<weights_data.size(); i++) {
        if (quant == 0) {
            weights_data[i] = woq_sc[0] * (i%5);
        }
        else {
            weights_data[i] = woq_sc[(i % N) + idx_buff] * (i%5);
            if ((i % (group_size * N)) == (group_size * N) - 1) {
                idx_buff = idx_buff + N;
            }
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
    matmul_attr.set_output_scales(0, {2.08});
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
        unsigned int post_op, std::vector<float> woq_sc, int m, int n, int k,
        int group_size, int quant, int scale_type) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D starts");

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    // Tensor dimensions.
    const memory::dim M = m, K = k, N = n;
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
    matmul_attr.set_output_scales(0, {2.08});
    //Passing WOQ scales as ARG
    //Mask values: per tensor(0), per channel(2) and per group(3)
    memory::dims scale_dims;
    if (quant == 2) {
        matmul_attr.set_woq_scale(3, {group_size, 1}, scale_type == 1?dt::bf16:dt::f32);
        scale_dims = {N *(K/group_size)};
    }
    else if (quant == 1) {
        matmul_attr.set_woq_scale(2, {1, 1}, scale_type == 1?dt::bf16:dt::f32);
        scale_dims = {N};
    }
    else {
        matmul_attr.set_woq_scale(0, {}, scale_type == 1?dt::bf16:dt::f32);
        scale_dims = {1};
    }
    // Create scale memory
    auto scale_mem = memory({scale_dims, dt::f32, tag::a}, eng);
    write_to_zendnn_memory(woq_sc.data(), scale_mem);
    auto scale_bf_mem = memory({scale_dims, dt::bf16, tag::a}, eng);

    if (scale_type == 1) {
        reorder(scale_mem, scale_bf_mem).execute(engine_stream,scale_mem, scale_bf_mem);
    }

    auto scale_arg_mem = scale_type == 1 ? scale_bf_mem : scale_mem;
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
    matmul_args.insert({ZENDNN_ARG_ATTR_WOQ_SCALES, scale_arg_mem});
    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim.execute(engine_stream, matmul_args);
    // Wait for the computation to finalize.
    engine_stream.wait();
    // Read data from memory object's handle.
    read_from_zendnn_memory(dst_data.data(), dst_arg_mem);
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D ends");

    return dst_data;
}

void wrapper_woq(int val, int of32, int quant, int m, int n, int k,
                 int group_size, int scale_type) {
    std::vector<float> woq_sc(1);
    string quant_type, sc_d;
    switch (quant) {
    case 0:
        quant_type = "Per Tensor";
        group_size = k;
        break;
    case 1:
        quant_type = "Per Channel";
        woq_sc.resize(n);
        group_size = k;
        break;
    case 2:
        quant_type = "Per Group";
        woq_sc.resize(n*(k/group_size));
        break;
    default:
        return;
    };
    switch (scale_type) {
    case 0:
        sc_d = "FP32";
        break;
    case 1:
        sc_d = "BF16";
        break;
    default:
        return;
    };

    default_random_engine gen;
    uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < woq_sc.size(); i++) {
        woq_sc[i] = distribution(gen);
    }
    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    if (val == 0 || val == 1) {
        obj.is_ref_gemm_bf16 = true;
        std::vector<float> ref_dst, woq_dst;
        if (!of32) {
            ref_dst = matmul_example_2D_bf16_ref(true, false, -1, woq_sc, m, n, k,
                                                 group_size, quant);
            obj.is_ref_gemm_bf16 = false;
            woq_dst = matmul_example_2D_bf16(true, false, -1, woq_sc, m, n, k,
                                             group_size, quant, scale_type);
            std::cout<<quant_type<<" Quant_type | "<<sc_d<<" Scale data type\n";
            compare_vectors(
                woq_dst, ref_dst, k, "Compare s4 weights | comp BF16 output BF16");

            woq_dst = matmul_example_2D_bf16(false, false, -1, woq_sc, m, n, k,
                                             group_size, quant, scale_type);
            std::cout<<quant_type<<" Quant_type | "<<sc_d<<" Scale data type\n";
            compare_vectors(
                woq_dst, ref_dst, k, "Compare s8 weights | comp BF16 output BF16");
        }
        if (of32) {
            ref_dst = matmul_example_2D_bf16_ref(true, true, -1, woq_sc, m, n, k,
                                                 group_size, quant);
            obj.is_ref_gemm_bf16 = false;
            woq_dst = matmul_example_2D_bf16(true, true, -1, woq_sc, m, n, k,
                                             group_size, quant, scale_type);
            std::cout<<quant_type<<" Quant_type | "<<sc_d<<" Scale data type\n";
            compare_vectors(
                woq_dst, ref_dst, k, "Compare s4 weights | comp BF16 output FP32");
            woq_dst = matmul_example_2D_bf16(false, true, -1, woq_sc, m, n, k,
                                             group_size, quant, scale_type);
            std::cout<<quant_type<<" Quant_type | "<<sc_d<<" Scale data type\n";
            compare_vectors(
                woq_dst, ref_dst, k, "Compare s8 weights | comp BF16 output FP32");
        }
    }
    if (val == 0 || val == 2) {
        auto ref_dst = matmul_example_2D_f32_ref(true, -1, woq_sc, m, n, k,
                       group_size, quant);
        auto woq_dst = matmul_example_2D_f32(true, -1, woq_sc, m, n, k,
                                             group_size, quant, scale_type);
        std::cout<<quant_type<<" Quant_type | "<<sc_d<<" Scale data type\n";
        compare_vectors(
            woq_dst, ref_dst, k, "Compare s4 weights Compute F32");
        woq_dst = matmul_example_2D_f32(false, -1, woq_sc, m, n, k,
                                        group_size, quant, scale_type);
        std::cout<<quant_type<<" Quant_type | "<<sc_d<<" Scale data type\n";
        compare_vectors(
            woq_dst, ref_dst, k, "Compare s8 weights Compute FP32");
    }
}

int main(int argc, char **argv) {
    //return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test test starts");
    int test_call = 0, dst_f32 = 0, woq_quant_type = 2;
    int m = 4096, n = 4096, k = 4096, group_size = 128, scale_type = 0;
    //Setting Primitive Cache capacity to 0.
#ifdef _WIN32
    _putenv_s("ZENDNN_WEIGHT_CACHE_CAPACITY","0");
#else
    setenv("ZENDNN_WEIGHT_CACHE_CAPACITY","0",1);
#endif
    //No arg passed
    if (argc < 2) {
        //Run all combinations
        for (int quant = 0; quant<3; quant++) {
            for (int sc=0; sc<2; sc++)
                wrapper_woq(test_call, dst_f32, quant, m, n, k, group_size,
                            sc);
        }
    }
    else {
        if (argc > 1) {
            //arg[1]
            //0:bf16 and fp32
            //1:bf16 compute
            //2:fp32 compute
            test_call = std::stoi(std::string(argv[1]));
        }
        if (argc > 2) {
            //arg[2] 0:bf16 output| 1:f32 output
            //only applicable for BF16 compute
            dst_f32 = std::stoi(std::string(argv[2]));
        }
        if (argc > 3) {
            //arg[3]
            //0:per tensor
            //1:per channel
            //2:per group
            woq_quant_type = std::stoi(std::string(argv[3]));
            if (woq_quant_type>2 || woq_quant_type<0) {
                woq_quant_type = 2;
            }
        }
        if (argc > 4) {
            //arg[4] M value
            m = std::stoi(std::string(argv[4]));
        }
        if (argc > 5) {
            //arg[5] N value
            n = std::stoi(std::string(argv[5]));
        }
        if (argc > 6) {
            //arg[6] K value
            k = std::stoi(std::string(argv[6]));
        }

        if (argc > 7) {
            //arg[7] Group size
            group_size = std::stoi(std::string(argv[7]));
        }
        if (argc > 8) {
            //arg[8] scale data type
            //0: FP32
            //1: BF16
            scale_type = std::stoi(std::string(argv[8]));
        }
        wrapper_woq(test_call, dst_f32, woq_quant_type, m, n, k, group_size,
                    scale_type);
    }
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test test ends");
    return 0;
}

