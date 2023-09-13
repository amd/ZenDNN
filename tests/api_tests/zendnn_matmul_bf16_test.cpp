/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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


using namespace zendnn;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace {
void init_vector(std::vector<float> &v) {
    std::mt19937 gen;
    std::uniform_real_distribution<float> u(-1, 1);
    for (auto &e : v) {
        e = u(gen);
    }
}
int compare_vectors(const std::vector<float> &v1, const std::vector<float> &v2,
                    int64_t K, const char *message) {
    double v1_l2 = 0, diff_l2 = 0;
    for (size_t n = 0; n < v1.size(); ++n) {
        float diff = v1[n] - v2[n];
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
} // namespace

int number_of_runs = 1;
float fixed_beta = 0.f;


void matmul_example_3D(unsigned int post_op) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_3D starts");

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);
    // Tensor dimensions.
    const memory::dim MB = 3, // batch size
                      M = 128, K = 256, N = 512;
    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {MB, M, K};
    memory::dims weights_dims = {MB, K, N};
    memory::dims bias_dims = {1, 1, N};
    memory::dims dst_dims = {MB, M, N};
    // Allocate buffers.
    std::vector<float> src_data(MB * M * K);
    std::vector<float> weights_data(MB * K * N);
    std::vector<float> bias_data(1 *1 * N);
    std::vector<float> dst_data(MB * M * N);
    // Initialize src, weights, bias.
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
    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md_f = memory::desc(src_dims, dt::f32, tag::abc);
    auto weights_md_f = memory::desc(weights_dims, dt::f32, tag::abc);
    auto bias_md_f = memory::desc(bias_dims, dt::f32, tag::abc);

    auto src_mem = memory(src_md_f, eng);
    auto weights_mem = memory(weights_md_f, eng);
    auto bias_mem = memory(bias_md_f, eng);

    // Write data to memory object's handles.
    write_to_zendnn_memory(src_data.data(), src_mem);
    write_to_zendnn_memory(weights_data.data(), weights_mem);
    write_to_zendnn_memory(bias_data.data(), bias_mem);

    //Create bf16 memory desc
    auto src_md = memory::desc(src_dims, dt::bf16, tag::abc);
    auto weights_md = memory::desc(weights_dims, dt::bf16, tag::abc);
    auto bias_md = memory::desc(bias_dims, dt::f32, tag::abc);
    auto dst_md = memory::desc(dst_dims, dt::bf16, tag::abc);
    //Create bf16 memory
    auto src_bf_mem = memory(src_md, eng);
    auto weights_bf_mem = memory(weights_md, eng);
    auto bias_bf_mem = memory(bias_md, eng);

    //reorder the f32 to bf16 and execute
    reorder(src_mem, src_bf_mem).execute(engine_stream,src_mem,src_bf_mem);
    reorder(weights_mem, weights_bf_mem).execute(engine_stream,weights_mem,
            weights_bf_mem);

    auto dst_bf_mem = memory(dst_md, eng);

    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, weights_md, bias_md_f, dst_md);

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
    matmul_args.insert({ZENDNN_ARG_BIAS, bias_mem});
    matmul_args.insert({ZENDNN_ARG_DST, dst_bf_mem});
    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim.execute(engine_stream, matmul_args);
    // Wait for the computation to finalize.
    engine_stream.wait();
    // Read data from memory object's handle.
    read_from_zendnn_memory(dst_data.data(), dst_bf_mem);
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_3D ends");
}

std::vector<float> matmul_example_2D(bool dst_f32, unsigned int post_op) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D starts");

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    // Tensor dimensions.
    const memory::dim MB = 3, // batch size
                      M = 128, K = 256, N = 512;
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
        weights_data[i] = std::sin(i * 2.f);
    }

    for (int i=0; i<bias_data.size(); i++) {
        bias_data[i] = std::tanh(i);
    }

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md_f = memory::desc(src_dims, dt::f32, tag::ab);
    auto weights_md_f = memory::desc(weights_dims, dt::f32, tag::ab);
    auto bias_md_f = memory::desc(bias_dims, dt::f32, tag::ab);
    auto src_mem = memory(src_md_f, eng);
    auto weights_mem = memory(weights_md_f, eng);
    auto bias_mem = memory(bias_md_f, eng);

    // Write data to memory object's handles.
    write_to_zendnn_memory(src_data.data(), src_mem);
    write_to_zendnn_memory(weights_data.data(), weights_mem);
    write_to_zendnn_memory(bias_data.data(), bias_mem);

    //Create bf16 memory desc
    auto src_md = memory::desc(src_dims, dt::bf16, tag::ab);
    auto weights_md = memory::desc(weights_dims, dt::bf16, tag::ab);
    auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);

    auto dst_md = memory::desc(dst_dims, dt::bf16, tag::ab);
    auto dst_md_f32 = memory::desc(dst_dims, dt::f32, tag::ab);

    //Create bf16 memory
    auto src_bf_mem = memory(src_md, eng);
    auto weights_bf_mem = memory(weights_md, eng);
    auto bias_bf_mem = memory(bias_md, eng);

    //reorder the f32 to bf16 and execute
    reorder(src_mem, src_bf_mem).execute(engine_stream,src_mem,src_bf_mem);
    reorder(weights_mem, weights_bf_mem).execute(engine_stream,weights_mem,
            weights_bf_mem);

    auto dst_bf_mem = memory(dst_f32?dst_md_f32:dst_md, eng);

    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, weights_md, bias_md_f,
                                 dst_f32?dst_md_f32:dst_md);

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
    matmul_args.insert({ZENDNN_ARG_BIAS, bias_mem});
    matmul_args.insert({ZENDNN_ARG_DST, dst_bf_mem});
    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim.execute(engine_stream, matmul_args);
    // Wait for the computation to finalize.
    engine_stream.wait();
    // Read data from memory object's handle.
    read_from_zendnn_memory(dst_data.data(), dst_bf_mem);
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D ends");

    return dst_data;
}

int main(int argc, char **argv) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test test starts");

    //output can be f32 or bf16
    bool f32_flag = 0;
    unsigned int post_op = 0;
    if (argc > 1) {
        //arg[1] 0:bf16 output| 1:f32 output
        f32_flag = std::stoi(std::string(argv[1]));
    }
    if (argc > 2) {
        //arg[2] -> post_op(0: relu, 1:gelu-tanh, 2:gelu-erf)
        post_op = std::stoi(std::string(argv[2]));
    }
    //Setting Primitive Cache capacity to 0.
    setenv("ZENDNN_PRIMITIVE_CACHE_CAPACITY","0",1);

    std::vector<float> brgemm, zen_aocl;
    matmul_example_3D(post_op);
    //Enable BLIS path
    setenv("ZENDNN_BLIS_MATMUL_BF16","1",1);
    zen_aocl = matmul_example_2D(f32_flag, post_op);

    //Disable BLIS path
    setenv("ZENDNN_BLIS_MATMUL_BF16","0",1);
    brgemm = matmul_example_2D(f32_flag, post_op);
    //Compare the brgemm and blis
    auto rc = compare_vectors(
                  brgemm, zen_aocl,256, "Compare BRGEMM MatMul vs BLIS Path");
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test test ends");
    return 0;
}
