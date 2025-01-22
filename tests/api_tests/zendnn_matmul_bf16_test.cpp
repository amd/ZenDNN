/*******************************************************************************
* Modifications Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "zendnn_helper.hpp"

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

std::vector<float> matmul_example_2D(bool dst_f32, unsigned int post_op,
                                     std::string binary_postop) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D starts");

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    // Tensor dimensions.
    const memory::dim M = 128, K = 256, N = 512;
    //const memory::dim M = 2, K = 2, N = 2;
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

    std::vector<float> bin_add_data1(M * N);
    std::vector<float> bin_add_data2(M * N);
    std::vector<float> bin_mul_data3(M * N);
    // Initialize src, weights, bias, add_mem
    for (int i=0; i<src_data.size(); i++) {
        src_data[i] = std::cos(i / 10.f);
    }

    for (int i=0; i<weights_data.size(); i++) {
        weights_data[i] = std::sin(i * 2.f);
    }

    for (int i=0; i<bias_data.size(); i++) {
        bias_data[i] = std::tanh(i);
    }
    for (int i=0; i<bin_add_data1.size(); i++) {
        bin_add_data1[i] = i*0.5116;// std::sin(i * 2.f);
    }
    for (int i=0; i<bin_add_data2.size(); i++) {
        bin_add_data2[i] = i*0.2031;// std::sin(i * 2.f);
    }
    for (int i=0; i<bin_mul_data3.size(); i++) {
        bin_mul_data3[i] = i*0.1015;// std::sin(i * 2.f);
    }

    // Create memory descriptors of memory objects for src, weights, bias, and
    // dst.
    auto src_md_f = memory::desc(src_dims, dt::f32, tag::ab);
    auto weights_md_f = memory::desc(weights_dims, dt::f32, tag::ab);
    auto bias_md_f = memory::desc(bias_dims, dt::f32, tag::ab);
    auto bin_md_f1 = memory::desc(dst_dims, dt::f32, tag::ab);
    auto bin_md_f2 = memory::desc(dst_dims, dt::f32, tag::ab);
    auto bin_md_f3 = memory::desc(dst_dims, dt::f32, tag::ab);

    //Create memory using desc
    auto src_mem = memory(src_md_f, eng);
    auto weights_mem = memory(weights_md_f, eng);
    auto bias_mem = memory(bias_md_f, eng);
    auto bin_mem1 = memory(bin_md_f1, eng);
    auto bin_mem2 = memory(bin_md_f2, eng);
    auto bin_mem3 = memory(bin_md_f3, eng);

    // Write data to memory object's handles.
    write_to_zendnn_memory(src_data.data(), src_mem);
    write_to_zendnn_memory(weights_data.data(), weights_mem);
    write_to_zendnn_memory(bias_data.data(), bias_mem);
    write_to_zendnn_memory(bin_add_data1.data(), bin_mem1);
    write_to_zendnn_memory(bin_add_data2.data(), bin_mem2);
    write_to_zendnn_memory(bin_mul_data3.data(), bin_mem3);

    //Create bf16 memory desc
    auto src_md = memory::desc(src_dims, dt::bf16, tag::ab);
    auto weights_md = memory::desc(weights_dims, dt::bf16, tag::ab);
    auto bias_md = memory::desc(bias_dims, dt::bf16, tag::ab);
    auto bin_md1 = memory::desc(dst_dims, dt::bf16, tag::ab);
    auto bin_md2 = memory::desc(dst_dims, dt::bf16, tag::ab);
    auto bin_md3 = memory::desc(dst_dims, dt::bf16, tag::ab);

    auto dst_md = memory::desc(dst_dims, dt::bf16, tag::ab);
    auto dst_md_f32 = memory::desc(dst_dims, dt::f32, tag::ab);

    //Create bf16 memory
    auto src_bf_mem = memory(src_md, eng);
    auto weights_bf_mem = memory(weights_md, eng);
    auto bias_bf_mem = memory(bias_md, eng);
    auto bin_bf_mem1 = memory(bin_md1, eng);
    auto bin_bf_mem2 = memory(bin_md2, eng);
    auto bin_bf_mem3 = memory(bin_md3, eng);

    //reorder the f32 to bf16 and execute
    reorder(src_mem, src_bf_mem).execute(engine_stream,src_mem,src_bf_mem);
    reorder(weights_mem, weights_bf_mem).execute(engine_stream,weights_mem,
            weights_bf_mem);
    if (!dst_f32) {
        reorder(bias_mem, bias_bf_mem).execute(engine_stream, bias_mem, bias_bf_mem);
        reorder(bin_mem1, bin_bf_mem1).execute(engine_stream, bin_mem1, bin_bf_mem1);
        reorder(bin_mem2, bin_bf_mem2).execute(engine_stream, bin_mem2, bin_bf_mem2);
        reorder(bin_mem3, bin_bf_mem3).execute(engine_stream, bin_mem3, bin_bf_mem3);
    }
    auto bias_arg_mem = dst_f32 ? bias_mem : bias_bf_mem;
    auto bin_arg_mem1 = dst_f32 ? bin_mem1 : bin_bf_mem1;
    auto bin_arg_mem2 = dst_f32 ? bin_mem2 : bin_bf_mem2;
    auto bin_arg_mem3 = dst_f32 ? bin_mem3 : bin_bf_mem3;

    auto dst_bf_mem = memory(dst_f32?dst_md_f32:dst_md, eng);

    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, weights_md, dst_f32?bias_md_f:bias_md,
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
        matmul_ops.append_eltwise(scale, algorithm::eltwise_gelu, 1.0, beta);
    }
    else if (post_op == 2) {
        matmul_ops.append_eltwise(scale, algorithm::eltwise_gelu_erf, 1.0, beta);
    }
    else if (post_op == 3) {
        matmul_ops.append_eltwise(scale, algorithm::eltwise_swish, 1.0, beta);
        //SiLU(alpha = 1.0)
    }
    else if (post_op == 4) {
        matmul_ops.append_eltwise(scale, algorithm::eltwise_logistic, alpha, beta);
    }
    if (binary_postop=="add") {
        matmul_ops.append_binary(algorithm::binary_add, dst_f32? bin_md_f1: bin_md1);
        matmul_ops.append_binary(algorithm::binary_add, dst_f32? bin_md_f2: bin_md2);
    }
    else if (binary_postop=="mul") {
        matmul_ops.append_binary(algorithm::binary_mul, dst_f32? bin_md_f3: bin_md3);
    }
    primitive_attr matmul_attr;

    //Set scale
    matmul_attr.set_output_scales(0, {2.08});
    matmul_attr.set_post_ops(matmul_ops);

    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);
    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);

    // Primitive execution: matrix multiplication with Add-Add.
    if (binary_postop == "add") {
        matmul_prim.execute(engine_stream, {
            {ZENDNN_ARG_SRC, src_bf_mem},
            {ZENDNN_ARG_WEIGHTS, weights_bf_mem},
            {ZENDNN_ARG_BIAS, bias_arg_mem},
            {ZENDNN_ARG_DST, dst_bf_mem},
            {ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(1) | ZENDNN_ARG_SRC_1, bin_arg_mem1},
            {ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(2) | ZENDNN_ARG_SRC_1, bin_arg_mem2}
        });
    }
    else if (binary_postop=="mul") {
        matmul_prim.execute(engine_stream, {
            {ZENDNN_ARG_SRC, src_bf_mem},
            {ZENDNN_ARG_WEIGHTS, weights_bf_mem},
            {ZENDNN_ARG_BIAS, bias_arg_mem},
            {ZENDNN_ARG_DST, dst_bf_mem},
            {ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(1) | ZENDNN_ARG_SRC_1, bin_arg_mem3},
        });
    }
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
    //silu
    unsigned int post_op = 0;
    //binary postops
    std::string binary_postop = "add";
    if (argc > 1) {
        //arg[1] 0:bf16 output| 1:f32 output
        f32_flag = std::stoi(std::string(argv[1]));
    }
    if (argc > 2) {
        //arg[2] -> post_op(0: relu, 1:gelu-tanh, 2:gelu-erf)
        post_op = std::stoi(std::string(argv[2]));
    }
    if (argc > 3) {
        //arg[3] -> post_op(add: binary_add, mul: binary_mul)
        binary_postop = std::string(argv[3]);
    }
    //Setting Primitive Cache capacity to 0.
#ifdef _WIN32
    _putenv_s("ZENDNN_PRIMITIVE_CACHE_CAPACITY","0");
#else
    setenv("ZENDNN_PRIMITIVE_CACHE_CAPACITY","0",1);
#endif

    std::vector<float> gemm_jit, zen;
    matmul_example_3D(post_op);
    //ZenDNN_Path: FP16:1-AOCL_BLIS, FP16:2-BLOCKED_BRGEMM, FP16:3-BRGEMM
    zen = matmul_example_2D(f32_flag, post_op, binary_postop);

    //Gemm-JIT Path
    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    obj.is_ref_gemm_bf16 = true;
    gemm_jit = matmul_example_2D(f32_flag, post_op, binary_postop);
    //Compare the ZENDNN_PATHS with GEMM_JIT Kernels
    auto rc = compare_vectors(
                  gemm_jit, zen, 256, "Compare GEMM_JIT MatMul vs ZenDNN Paths");
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_bf16 test ends");
    return 0;
}
