/*******************************************************************************
* Modifications Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cfloat>
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
    float max_diff=FLT_MIN, min_diff=FLT_MAX;
    for (size_t n = 0; n < v1.size(); ++n) {
        float diff = (float)(v1[n] - v2[n]);
        max_diff = std::max(max_diff, diff);
        min_diff = std::min(min_diff, diff);
        v1_l2 += v1[n] * v1[n];
        diff_l2 += diff * diff;
    }
    std::cout<<"max_diff: "<<max_diff<<std::endl;
    std::cout<<"min_diff: "<<min_diff<<std::endl;
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
           "\n\t\tReference matrix:%g\n\t\tError:%g\n\t\tRelative_error:%g\n"
           "\tAccuracy check: %s\n",
           message, v1_l2, diff_l2, diff_l2 / v1_l2, ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
} // namespace
int number_of_runs = 1;
float fixed_beta = 0.f;

// Create a _dynamic_ MatMul primitive that can work with arbitrary shapes
// and alpha parameters.
// Warning: current limitation is that beta parameter should be known in
// advance (use fixed_beta).
matmul dynamic_matmul_create(zendnn::engine eng) {
    // We assume that beta is known at the primitive creation time
    float beta = fixed_beta;
    memory::dims a_shape = {ZENDNN_RUNTIME_DIM_VAL, ZENDNN_RUNTIME_DIM_VAL};
    memory::dims b_shape = {ZENDNN_RUNTIME_DIM_VAL, ZENDNN_RUNTIME_DIM_VAL};
    memory::dims c_shape = {ZENDNN_RUNTIME_DIM_VAL, ZENDNN_RUNTIME_DIM_VAL};
    memory::dims a_strides = {ZENDNN_RUNTIME_DIM_VAL, ZENDNN_RUNTIME_DIM_VAL};
    memory::dims b_strides = {ZENDNN_RUNTIME_DIM_VAL, ZENDNN_RUNTIME_DIM_VAL};
    memory::dims c_strides = {ZENDNN_RUNTIME_DIM_VAL, 1};
    memory::desc a_md(a_shape, memory::data_type::f32, a_strides);
    memory::desc b_md(b_shape, memory::data_type::f32, b_strides);
    memory::desc c_md(c_shape, memory::data_type::f32, c_strides);
    // Create attributes (to handle alpha dynamically and beta if necessary)
    primitive_attr attr;
    attr.set_output_scales(/* mask */ 0, {ZENDNN_RUNTIME_F32_VAL});
    if (beta != 0.f) {
        post_ops po;
        po.append_sum(beta);
        attr.set_post_ops(po);
    }
    // Create a MatMul primitive
    matmul::desc matmul_d(a_md, b_md, c_md);
    matmul::primitive_desc matmul_pd(matmul_d, attr, eng);
    return matmul(matmul_pd);
}
// Execute a _dynamic_ MatMul primitive created earlier. All the parameters are
// passed at a run-time (except for beta which has to be specified at the
// primitive creation time due to the current limitation).
void dynamic_matmul_execute(matmul &matmul_p, char transA, char transB,
                            int64_t M, int64_t N, int64_t K, float alpha, const float *A,
                            int64_t lda, const float *B, int64_t ldb, float beta, float *C,
                            int64_t ldc, zendnn::engine eng, zendnn::stream engine_stream) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: dynamic_matmul_execute starts");
    using dims = memory::dims;
    if (beta != fixed_beta) {
        throw std::logic_error("Run-time beta is not yet supported.");
    }
    // Translate transA and transB
    dims a_strides = tolower(transA) == 'n' ? dims {lda, 1} :
                     dims {1, lda};
    dims b_strides = tolower(transB) == 'n' ? dims {ldb, 1} :
                     dims {1, ldb};
    // Wrap raw pointers into ZenDNN memories (with proper shapes)
    memory A_m({{M, K}, memory::data_type::f32, a_strides}, eng, (void *)A);
    memory B_m({{K, N}, memory::data_type::f32, b_strides}, eng, (void *)B);
    memory C_m({{M, N}, memory::data_type::f32, {ldc, 1}}, eng, (void *)C);
    // Prepare ZenDNN memory for alpha
    memory alpha_m({{1}, memory::data_type::f32, {1}}, eng, &alpha);
    // Execute the MatMul primitive
    matmul_p.execute(engine_stream, {
        {ZENDNN_ARG_SRC, A_m}, {ZENDNN_ARG_WEIGHTS, B_m}, {ZENDNN_ARG_DST, C_m},
        {ZENDNN_ARG_ATTR_OUTPUT_SCALES, alpha_m}
    });
    engine_stream.wait();
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: dynamic_matmul_execute ends");
}
// Create and execute a _static_ MatMul primitive. All shapes and parameters
// are hard-coded in the primitive and cannot be changed later.
void static_matmul_create_and_execute(char transA, char transB, int64_t M,
                                      int64_t N, int64_t K, float alpha, const float *A, int64_t lda,
                                      const float *B, int64_t ldb, float beta, float *C, int64_t ldc,
                                      zendnn::engine eng, zendnn::stream engine_stream) {
    zendnnInfo(ZENDNN_TESTLOG,
               "zendnn_matmul_test: static_matmul_create_and_execute starts");
    using dims = memory::dims;
    // Prepare strides based on the transA and transB flags: transposed
    // matrices have strides swapped
    dims a_strides = tolower(transA) == 'n' ? dims {lda, 1} :
                     dims {1, lda};
    dims b_strides = tolower(transB) == 'n' ? dims {ldb, 1} :
                     dims {1, ldb};
    // Prepare memory descriptors
    memory::desc a_md({M, K}, memory::data_type::f32, a_strides);
    memory::desc b_md({K, N}, memory::data_type::f32, b_strides);
    memory::desc c_md({M, N}, memory::data_type::f32, {ldc, 1});
    // Create attributes (to handle alpha and beta if necessary)
    primitive_attr attr;
    if (alpha != 1.f) attr.set_output_scales(/* mask */ 0, {alpha});
    if (beta != 0.f) {
        post_ops po;
        po.append_sum(beta);
        attr.set_post_ops(po);
    }
    // Create a MatMul primitive
    matmul::desc matmul_d(a_md, b_md, c_md);
    matmul::primitive_desc matmul_pd(matmul_d, attr, eng);
    matmul matmul_p(matmul_pd);
    // Wrap raw pointers into ZenDNN memory objects
    memory A_m(a_md, eng, (void *)A);
    memory B_m(b_md, eng, (void *)B);
    memory C_m(c_md, eng, (void *)C);
    // Execute the MatMul primitive.
    // Since here all shapes and parameters are static, please note that we
    // don't need to pass alpha (scales) again, as they are already hard-coded
    // in the primitive descriptor. Also, we are not allowed to change the
    // shapes of matrices A, B, and C -- they should exactly match
    // the memory descriptors passed to MatMul operation descriptor.
    matmul_p.execute(engine_stream, {
        {ZENDNN_ARG_SRC, A_m}, {ZENDNN_ARG_WEIGHTS, B_m},
        {ZENDNN_ARG_DST, C_m}
    });
    engine_stream.wait();
    zendnnInfo(ZENDNN_TESTLOG,
               "zendnn_matmul_test: static_matmul_create_and_execute ends");
}

// Create and execute a _static_ MatMul primitive with Bias. All shapes and
// parameters are hard-coded in the primitive and cannot be changed later.
void static_matmul_bias_create_and_execute(char transA, char transB, int64_t M,
        int64_t N, int64_t K, float alpha, const float *A, int64_t lda,
        const float *B, int64_t ldb, float beta, float *C, float *Bias,
        int64_t ldc, zendnn::engine eng, zendnn::stream engine_stream) {
    zendnnInfo(ZENDNN_TESTLOG,
               "zendnn_matmul_test: static_matmul_bias_create_and_execute starts");
    using dims = memory::dims;
    // Prepare strides based on the transA and transB flags: transposed
    // matrices have strides swapped
    dims a_strides = tolower(transA) == 'n' ? dims {lda, 1} :
                     dims {1, lda};
    dims b_strides = tolower(transB) == 'n' ? dims {ldb, 1} :
                     dims {1, ldb};
    dims bias_strides = dims {1, N};
    // Prepare memory descriptors
    memory::desc a_md({M, K}, memory::data_type::f32, a_strides);
    memory::desc b_md({K, N}, memory::data_type::f32, b_strides);
    memory::desc c_md({M, N}, memory::data_type::f32, {ldc, 1});
    memory::desc bias_md({N}, memory::data_type::f32, bias_strides);

    // Create attributes (to handle alpha and beta if necessary)
    primitive_attr attr;
    if (alpha != 1.f) attr.set_output_scales(/* mask */ 0, {alpha});
    if (beta != 0.f) {
        post_ops po;
        po.append_sum(beta);
        attr.set_post_ops(po);
    }
    // Create a MatMul primitive
    matmul::desc matmul_d(a_md, b_md, bias_md, c_md);
    matmul::primitive_desc matmul_pd(matmul_d, attr, eng);
    matmul matmul_p(matmul_pd);
    // Wrap raw pointers into ZenDNN memory objects
    memory A_m(a_md, eng, (void *)A);
    memory B_m(b_md, eng, (void *)B);
    memory C_m(c_md, eng, (void *)C);
    memory Bias_m(bias_md, eng, (void *)Bias);
    // Execute the MatMul primitive.
    // Since here all shapes and parameters are static, please note that we
    // don't need to pass alpha (scales) again, as they are already hard-coded
    // in the primitive descriptor. Also, we are not allowed to change the
    // shapes of matrices A, B, and C -- they should exactly match
    // the memory descriptors passed to MatMul operation descriptor.
    matmul_p.execute(engine_stream, {
        {ZENDNN_ARG_SRC, A_m}, {ZENDNN_ARG_WEIGHTS, B_m},
        {ZENDNN_ARG_BIAS, Bias_m},
        {ZENDNN_ARG_DST, C_m}
    });
    engine_stream.wait();
    zendnnInfo(ZENDNN_TESTLOG,
               "zendnn_matmul_test: static_matmul_bias_create_and_execute ends");
}

void sgemm_and_matmul_with_params(char transA, char transB, int64_t M,
                                  int64_t N, int64_t K, float alpha, float beta,
                                  zendnn::engine eng, zendnn::stream engine_stream) {
    zendnnInfo(ZENDNN_TESTLOG,
               "zendnn_matmul_test: sgemm_and_matmul_with_params starts");
    if (beta != fixed_beta) {
        throw std::logic_error("Run-time beta is not yet supported.");
    }
    // Allocate and initialize matrices
    std::vector<float> A(M * K);
    init_vector(A);
    std::vector<float> B(K * N);
    init_vector(B);
    std::vector<float> Bias(N);
    init_vector(Bias);
    std::vector<float> C_sgemm(M * N);
    init_vector(C_sgemm);
    std::vector<float> C_dynamic_matmul = C_sgemm;
    std::vector<float> C_static_matmul = C_sgemm;
    std::vector<float> C_static_matmul_bias = C_sgemm;
    // Prepare leading dimensions
    int64_t lda = tolower(transA) == 'n' ? K : M;
    int64_t ldb = tolower(transB) == 'n' ? N : K;
    int64_t ldc = N;
    // 1. Execute sgemm
    zendnnInfo(ZENDNN_TESTLOG, "sgemm_and_matmul_with_params: M: ", M, " N: ", N,
               " K:", K);
    zendnnInfo(ZENDNN_TESTLOG, "sgemm_and_matmul_with_params: lda: ", lda, " ldb: ",
               ldb);
    zendnnInfo(ZENDNN_TESTLOG, "sgemm_and_matmul_with_params: alpha: ", alpha,
               " beta: ", beta);

    for (int run = 0; run < number_of_runs; ++run)
        zendnn_sgemm(transA, transB, M, N, K, alpha, A.data(), lda, B.data(), ldb,
                     beta, C_sgemm.data(), ldc);
    // 2.a Create dynamic MatMul
    auto dynamic_matmul = dynamic_matmul_create(eng);
    // 2.b Execute
    for (int run = 0; run < number_of_runs; ++run)
        dynamic_matmul_execute(dynamic_matmul, transA, transB, M, N, K, alpha,
                               A.data(), lda, B.data(), ldb, beta, C_dynamic_matmul.data(),
                               ldc, eng, engine_stream);
    // 3. Execute static MatMul
    for (int run = 0; run < number_of_runs; ++run)
        static_matmul_create_and_execute(transA, transB, M, N, K, alpha,
                                         A.data(), lda, B.data(), ldb, beta, C_static_matmul.data(),
                                         ldc, eng, engine_stream);
    // 4. Execute static MatMul with Bias
    /*
    for (int run = 0; run < number_of_runs; ++run)
        static_matmul_bias_create_and_execute(transA, transB, M, N, K, alpha,
                A.data(), lda, B.data(), ldb, beta, C_static_matmul_bias.data(),
                Bias.data(), ldc);
    */

    int rc = 0;
    rc |= compare_vectors(
              C_sgemm, C_dynamic_matmul, K, "Compare SGEMM vs dynamic MatMul");
    if (rc) {
        //throw std::logic_error("The resulting matrices diverged too much.");
    }
    rc |= compare_vectors(
              C_sgemm, C_static_matmul, K, "Compare SGEMM vs static MatMul");
    if (rc) {
        //throw std::logic_error("The resulting matrices diverged too much.");
    }
    rc |= compare_vectors(
              C_dynamic_matmul, C_static_matmul, K, "Compare Dynamic vs static MatMul");
    if (rc) {
        //throw std::logic_error("The resulting matrices diverged too much.");
    }
    zendnnInfo(ZENDNN_TESTLOG,
               "zendnn_matmul_test: sgemm_and_matmul_with_params ends");
}

void matmul_example_3D(zendnn::engine eng, zendnn::stream engine_stream) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_3D starts");

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
    auto src_md = memory::desc(src_dims, dt::f32, tag::abc);
    auto weights_md = memory::desc(weights_dims, dt::f32, tag::abc);
    auto bias_md = memory::desc(bias_dims, dt::f32, tag::abc);
    auto dst_md = memory::desc(dst_dims, dt::f32, tag::abc);
    auto src_mem = memory(src_md, eng);
    auto weights_mem = memory(weights_md, eng);
    auto bias_mem = memory(bias_md, eng);
    auto dst_mem = memory(dst_md, eng);
    // Write data to memory object's handles.
    write_to_zendnn_memory(src_data.data(), src_mem);
    write_to_zendnn_memory(weights_data.data(), weights_mem);
    write_to_zendnn_memory(bias_data.data(), bias_mem);
    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, weights_md, bias_md, dst_md);
    // Create primitive post-ops (ReLU).
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops matmul_ops;
    matmul_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
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
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_3D ends");
}

std::vector<float> matmul_example_2D(zendnn::engine eng,
                                     zendnn::stream engine_stream, memory::dim M, memory::dim N, memory::dim K,
                                     std::vector<float> &weights_data) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D starts");
    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};
    // Allocate buffers.
    std::vector<float> src_data(M * K);
    weights_data.resize(K * N);
    std::cout<<"address:"<<(void *)weights_data.data()<<std::endl;
    std::vector<float> bias_data(1 * N);
    std::vector<float> dst_data(M * N);
    std::vector<float> bin_mul_data(M * N);
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
    for (int i = 0; i < bin_mul_data.size(); i++) {
        bin_mul_data[i] = i * 0.1015;
    }
    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md = memory::desc(src_dims, dt::f32, tag::ab);
    auto weights_md = memory::desc(weights_dims, dt::f32, tag::ab);
    auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
    auto dst_md = memory::desc(dst_dims, dt::f32, tag::ab);
    auto bin_md = memory::desc(dst_dims, dt::f32, tag::ab);

    auto src_mem = memory(src_md, eng);
    auto weights_mem = memory(weights_md, eng, weights_data.data());
    auto bias_mem = memory(bias_md, eng);
    auto dst_mem = memory(dst_md, eng);
    auto bin_mem = memory(bin_md, eng);
    // Write data to memory object's handles.
    write_to_zendnn_memory(src_data.data(), src_mem);
    write_to_zendnn_memory(bias_data.data(), bias_mem);
    write_to_zendnn_memory(bin_mul_data.data(), bin_mem);
    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, weights_md, bias_md, dst_md);
    // Create primitive post-ops (ReLU).
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops matmul_ops;
    //matmul_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
    //matmul_ops.append_eltwise(scale, algorithm::eltwise_swish, 1.0, beta);
    //matmul_ops.append_eltwise(scale, algorithm::eltwise_gelu, 1.0, beta);
    //matmul_ops.append_binary(zendnn::algorithm::binary_mul, bin_md);
    //matmul_ops.append_eltwise(scale, algorithm::eltwise_gelu, alpha, beta);
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
    //matmul_args.insert({ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1, bin_mem});
    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim.execute(engine_stream, matmul_args);
    // Wait for the computation to finalize.
    engine_stream.wait();
    // Read data from memory object's handle.
    read_from_zendnn_memory(dst_data.data(), dst_mem);
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D ends");
    return dst_data;
}

int main(int argc, char **argv) {
    //return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test test starts");

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

//Setting Primitive Cache capacity to 0.
#ifdef _WIN32
    _putenv_s("ZENDNN_PRIMITIVE_CACHE_CAPACITY","0");
#else
    setenv("ZENDNN_PRIMITIVE_CACHE_CAPACITY","0",1);
#endif
    matmul_example_3D(eng, engine_stream);

    std::vector<float> gemm_jit, zen;

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

    for (const auto &M : M_list) {
        for (const auto &KN : KN_list) {
            memory::dim K = KN.first;
            memory::dim N = KN.second;

            std::cout<<"\nM="<<M<<", N="<<N<<", K="<<K<<std::endl;

            zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
            obj.is_ref_gemm_bf16 = false;
            obj.is_brgemm = false;

            //ZenDNN_Path: FP16:1-AOCL_BLIS, FP16:2-BLOCKED_BRGEMM, FP16:3-BRGEMM
            zen = matmul_example_2D(eng, engine_stream, M, K, N, weights_data);


            //Gemm-JIT Path
            obj.is_ref_gemm_bf16 = true;
            obj.is_brgemm = true;
            gemm_jit = matmul_example_2D(eng, engine_stream, M, K, N, weights_data);
            //Compare the ZENDNN_PATHS with GEMM_JIT Kernels
            auto rc = compare_vectors(
                          gemm_jit, zen, 256, "Compare GEMM_JIT MatMul vs ZenDNN Paths");
        }
    }

    sgemm_and_matmul_with_params('N', 'T', 10, 20, 30, 1.1f, fixed_beta, eng,
                               engine_stream);

    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test test ends");
    return 0;
}
