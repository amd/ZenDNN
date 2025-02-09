/*******************************************************************************
* Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

std::vector<int8_t> matmul_example_2D_dst_s8(zendnn::engine eng,
        zendnn::stream engine_stream) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D starts");

    // Tensor dimensions.
    const memory::dim MB = 3, // batch size
                      M = 198, K = 256, N = 512;
    //M = 2, K = 5, N = 5;

    std::vector<int8_t> src_data(M * K);
    std::vector<int8_t> weights_data(K * N);
    std::vector<int8_t> bias_data(1 * N);
    std::vector<int8_t> dst_data(M * N);
    std::vector<int8_t> bin_data(M * N);

    int8_t x = 0;
    for (int i=0; i<src_data.size(); i++) {
        src_data[i] = x++ %4;
    }
    x = 0;
    for (int i=0; i<weights_data.size(); i++) {
        weights_data[i] = x++ %3;
    }

    x=0;
    for (int i=0; i<bias_data.size(); i++) {
        bias_data[i] = x++ %15;
    }
    x = 0;
    for (int i=0; i<dst_data.size(); i++) {
        dst_data[i] = x++ %5;
    }

    int8_t w=0;
    for (int i=0; i<bin_data.size(); i++) {
        bin_data[i] = w++ %256;
    }

    //Set zero_point vals for src and dst
    int32_t zp_A = 9, zp_C = 2;

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md = memory::desc(src_dims, dt::s8, tag::ab);
    auto weights_md = memory::desc(weights_dims, dt::s8, tag::ab, true);
    auto bias_md = memory::desc(bias_dims, dt::s8, tag::ab);
    auto dst_md = memory::desc(dst_dims, dt::s8, tag::ab);
    auto bin_md = memory::desc(dst_dims, dt::s8, tag::ab);

    auto src_mem = memory(src_md, eng);
    auto weights_mem = memory(weights_md, eng);
    auto bias_mem = memory(bias_md, eng);
    auto dst_mem = memory(dst_md, eng);
    auto bin_mem = memory(bin_md, eng);
    memory zp_A_mem({{1}, memory::data_type::s32, {1}}, eng);
    memory zp_C_mem({{1}, memory::data_type::s32, {1}}, eng);

    write_to_zendnn_memory(src_data.data(), src_mem);
    write_to_zendnn_memory(weights_data.data(), weights_mem);
    write_to_zendnn_memory(bias_data.data(), bias_mem);
    write_to_zendnn_memory(dst_data.data(), dst_mem);
    write_to_zendnn_memory((void *)&zp_A, zp_A_mem);
    write_to_zendnn_memory((void *)&zp_C, zp_C_mem);
    write_to_zendnn_memory(bin_data.data(), bin_mem);

    int src_scale_size = 1, wei_scale_size = N, dst_scale_size = 1;
    std::vector<float> src_scale(src_scale_size);
    std::vector<float> wei_scale(wei_scale_size);
    std::vector<float> dst_scale(dst_scale_size);

    default_random_engine gen;
    uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < src_scale.size(); i++) {
        src_scale[i] = distribution(gen);
    }
    for (int i = 0; i < wei_scale.size(); i++) {
        wei_scale[i] = distribution(gen);
    }
    for (int i = 0; i < dst_scale.size(); i++) {
        dst_scale[i] = distribution(gen);
    }

    memory src_scale_mem({{src_scale_size}, memory::data_type::f32, {1}}, eng);
    memory wei_scale_mem({{wei_scale_size}, memory::data_type::f32, {1}}, eng);
    memory dst_scale_mem({{dst_scale_size}, memory::data_type::f32, {1}}, eng);

    write_to_zendnn_memory(src_scale.data(), src_scale_mem);
    write_to_zendnn_memory(wei_scale.data(), wei_scale_mem);
    write_to_zendnn_memory(dst_scale.data(), dst_scale_mem);

    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, weights_md, bias_md, dst_md);
    // Create primitive post-ops (ReLU).
    const float scale = 3.4f;
    const float alpha = 1.f;
    const float beta = 0.f;
    std::vector<float> scales(N);
    for (int i =0; i<N; i++) {
        scales[i] = (float)((i%10)*0.1);
    }

    post_ops matmul_ops;
    //matmul_ops.append_sum(1.0);
    matmul_ops.append_eltwise(1, algorithm::eltwise_gelu, alpha, beta);
    matmul_ops.append_binary(algorithm::binary_add, bin_md);
    primitive_attr matmul_attr;
    matmul_attr.set_post_ops(matmul_ops);
    //Add scale
    // matmul_attr.set_output_scales((1<<1), scales);
    // matmul_attr.set_output_scales(0, {scale});
    //Add zero_point
    //ZENDNN_RUNTIME_S32_VAL used to create primitive without actual zero_point
    matmul_attr.set_zero_points(ZENDNN_ARG_SRC, 0, {ZENDNN_RUNTIME_S32_VAL});
    matmul_attr.set_zero_points(ZENDNN_ARG_DST, 0, {ZENDNN_RUNTIME_S32_VAL});
    matmul_attr.set_scales_mask(ZENDNN_ARG_SRC, 0, {}, dt::f32);
    matmul_attr.set_scales_mask(ZENDNN_ARG_WEIGHTS, 1, {1,1}, dt::f32);
    matmul_attr.set_scales_mask(ZENDNN_ARG_DST, 0, {}, dt::f32);
    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);
    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);

    auto t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(1) | ZENDNN_ARG_SRC_1;
    // Primitive execution: matrix multiplication with zero_points.
    matmul_prim.execute(engine_stream, {
        {ZENDNN_ARG_SRC, src_mem},
        {ZENDNN_ARG_WEIGHTS, weights_mem},
        {ZENDNN_ARG_BIAS, bias_mem},
        {ZENDNN_ARG_DST, dst_mem},
        {ZENDNN_ARG_ATTR_SCALES | ZENDNN_ARG_SRC, src_scale_mem},
        {ZENDNN_ARG_ATTR_SCALES | ZENDNN_ARG_WEIGHTS, wei_scale_mem},
        {ZENDNN_ARG_ATTR_SCALES | ZENDNN_ARG_DST, dst_scale_mem},
        {ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_SRC, zp_A_mem},
        {ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_DST, zp_C_mem},
        {t, bin_mem}
    });
    // Wait for the computation to finalize.
    engine_stream.wait();

    read_from_zendnn_memory(dst_data.data(), dst_mem);
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D ends");
    return dst_data;
}

std::vector<int32_t> matmul_example_2D_dst_s32(zendnn::engine eng,
        zendnn::stream engine_stream) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D starts");

    // Tensor dimensions.
    const memory::dim MB = 3, // batch size
                      M = 128, K = 256, N = 512;
    //M = 2, K = 5, N = 5;

    std::vector<int8_t> src_data(M * K);
    std::vector<int8_t> weights_data(K * N);
    std::vector<int32_t> bias_data(1 * N);
    std::vector<int32_t> dst_data(M * N);
    std::vector<int32_t> bin_data(M * N);

    int8_t x = 0;
    for (int i=0; i<src_data.size(); i++) {
        src_data[i] = x++ %5;
    }
    x = 0;
    for (int i=0; i<weights_data.size(); i++) {
        weights_data[i] = x++ %5;
    }

    int32_t y=0;
    for (int i=0; i<bias_data.size(); i++) {
        bias_data[i] = (y++ %5);
    }
    for (int i=0; i<dst_data.size(); i++) {
        dst_data[i] = y++ %5;
    }

    int32_t qu = 0;
    for (int i=0; i<bin_data.size(); i++) {
        bin_data[i] = qu++ %25;
    }

    //Set zero_point vals for src and dst
    int32_t zp_A = 5, zp_C = 5;

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md = memory::desc(src_dims, dt::s8, tag::ab);
    auto weights_md = memory::desc(weights_dims, dt::s8, tag::ab, true);
    auto bias_md = memory::desc(bias_dims, dt::s32, tag::ab);
    auto dst_md = memory::desc(dst_dims, dt::s32, tag::ab);
    auto bin_md = memory::desc(dst_dims, dt::s32, tag::ab);

    auto src_mem = memory(src_md, eng);
    auto weights_mem = memory(weights_md, eng);
    auto bias_mem = memory(bias_md, eng);
    auto dst_mem = memory(dst_md, eng);
    auto bin_mem = memory(bin_md, eng);
    memory zp_A_mem({{1}, memory::data_type::s32, {1}}, eng);
    memory zp_C_mem({{1}, memory::data_type::s32, {1}}, eng);

    write_to_zendnn_memory(src_data.data(), src_mem);
    write_to_zendnn_memory(weights_data.data(), weights_mem);
    write_to_zendnn_memory(bias_data.data(), bias_mem);
    write_to_zendnn_memory(bin_data.data(), bin_mem);
    write_to_zendnn_memory(dst_data.data(), dst_mem);
    write_to_zendnn_memory((void *)&zp_A, zp_A_mem);
    write_to_zendnn_memory((void *)&zp_C, zp_C_mem);

    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, weights_md, bias_md, dst_md);
    // Create primitive post-ops (ReLU).
    const float scale = 2.42f;
    const float alpha = 1.0f;
    const float beta = 0.f;
    std::vector<float> scales(N);
    for (int i =0; i<N; i++) {
        scales[i] = (float)((i%10)*1.1);
    }

    post_ops matmul_ops;
    //matmul_ops.append_sum(1.0);
    matmul_ops.append_eltwise(1, algorithm::eltwise_gelu, alpha, beta);
    matmul_ops.append_binary(algorithm::binary_add, bin_md);
    primitive_attr matmul_attr;
    matmul_attr.set_post_ops(matmul_ops);
    //Add scale
    //matmul_attr.set_output_scales((1<<1), scales);
    matmul_attr.set_output_scales(0, {scale});
    //Add zero_point
    //ZENDNN_RUNTIME_S32_VAL used to create primitive without actual zero_point
    matmul_attr.set_zero_points(ZENDNN_ARG_SRC, 0, {ZENDNN_RUNTIME_S32_VAL});
    matmul_attr.set_zero_points(ZENDNN_ARG_DST, 0, {ZENDNN_RUNTIME_S32_VAL});
    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);
    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);

    auto t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(1) | ZENDNN_ARG_SRC_1;
    // Primitive execution: matrix multiplication with zero_points.
    matmul_prim.execute(engine_stream, {
        {ZENDNN_ARG_SRC, src_mem},
        {ZENDNN_ARG_WEIGHTS, weights_mem},
        {ZENDNN_ARG_BIAS, bias_mem},
        {ZENDNN_ARG_DST, dst_mem},
        {ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_SRC, zp_A_mem},
        {ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_DST, zp_C_mem},
        {t, bin_mem}
    });
    // Wait for the computation to finalize.
    engine_stream.wait();

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
    std::vector<int32_t> aocl_s32,jit_s32;
    std::vector<int8_t> aocl_s8,jit_s8;

    std::cout<<"MatMul INT8 dst s32\n";
    aocl_s32 = matmul_example_2D_dst_s32(eng, engine_stream);

    std::cout<<"\nMatMul INT8 dst s8\n";
    aocl_s8 = matmul_example_2D_dst_s8(eng, engine_stream);

    //Set JIT kernel
    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    obj.is_brgemm = true;

    obj.is_ref_gemm_bf16 = true;
    std::cout<<"JIT MatMul INT8 dst s32\n";
    jit_s32 = matmul_example_2D_dst_s32(eng, engine_stream);

    std::cout<<"JIT MatMul INT8 dst s8\n";
    jit_s8 = matmul_example_2D_dst_s8(eng, engine_stream);

    auto rc = compare_vectors<int32_t>(
                  jit_s32, aocl_s32, 256, "Compare INT8 os32 JIT MatMul vs ZenDNN Paths");
    rc = compare_vectors<int8_t>(
             jit_s8, aocl_s8, 256, "Compare INT8 os8 JIT MatMul vs ZenDNN Paths");
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test test ends");
    return 0;
}

