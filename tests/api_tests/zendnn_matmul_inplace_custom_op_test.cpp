/*******************************************************************************
* Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cfloat>
#include <algorithm>

#include "test_utils.hpp"
#include "zendnn_logging.hpp"

using namespace zendnn;
using namespace std;
using tag = memory::format_tag;
using dt = memory::data_type;

// Global variables to control behavior
bool ENABLE_DST_U8 = false;
bool ENABLE_BF16 = false;
bool ENABLE_FP32 = false;
bool ENABLE_DST_S8 = false;
bool ENABLE_RELU = false;
bool ENABLE_MUL_ADD = false;
bool ENABLE_SIGMOID = false;
bool ENABLE_SRC_WEI_SCALES = false;
bool ENABLE_DST_SCALES = false;
bool ENABLE_SRC_ZP = false;
bool ENABLE_DST_ZP = false;
bool ENABLE_BF16_MUL_ADD = false;

// Function to join command-line arguments into a single string
std::string join_arguments(int argc, char **argv) {
    std::ostringstream oss;
    for (int i = 1; i < argc; ++i) {
        oss << argv[i];
        if (i < argc - 1) {
            oss << " ";
        }
    }
    return oss.str();
}

// Function to parse command-line arguments
void parse_arguments(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--enable-dst-u8") {
            ENABLE_DST_U8 = true;
        }
        else if (arg == "--enable-bf16") {
            ENABLE_BF16 = true;
        }
        else if (arg == "--enable-fp32") {
            ENABLE_FP32 = true;
        }
        else if (arg == "--enable-src-wei-scales") {
            ENABLE_SRC_WEI_SCALES = true;
        }
        else if (arg == "--enable-relu") {
            ENABLE_RELU = true;
        }
        else if (arg == "--enable-mul-add") {
            ENABLE_MUL_ADD = true;
        }
        else if (arg == "--enable-sigmoid") {
            ENABLE_SIGMOID = true;
        }
        else if (arg == "--enable-dst-scales") {
            ENABLE_DST_SCALES = true;
        }
        else if (arg == "--enable-src-zp") {
            ENABLE_SRC_ZP = true;
        }
        else if (arg == "--enable-dst-zp") {
            ENABLE_DST_ZP = true;
        }
        else if (arg == "--enable-bf16-mul-add") {
            ENABLE_BF16_MUL_ADD = true;
        }
        else if (arg == "--enable-dst-s8") {
            ENABLE_DST_S8 = true;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
        }
    }
}

template<typename T>
int compare_vectors(const std::vector<T> &v1, const std::vector<T> &v2,
                    int64_t K, const char *message, const std::string &args) {
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
           "\n\t\tReference matrix:%g\n\t\tError:%g\n\t\tRelative_error:%g\n",
           message, v1_l2, diff_l2, diff_l2 / v1_l2);

    // Check if the error is zero and print the result
    if (diff_l2 == 0) {
        std::cout << "PASSED" << std::endl;
    }
    else {
        std::cout << "FAILED" << std::endl;
        std::cout << "Command-line arguments: " << args << std::endl;
    }

    return ok ? 0 : 1;
}

template<typename T>
std::vector<T> matmul_example_2D_dst_actual(zendnn::engine eng,
        zendnn::stream engine_stream, memory::dim M, memory::dim K, memory::dim N,
        std::vector<float> &src_scale, std::vector<float> &wei_scale,
        std::vector<float> &dst_scale,
        bool enable_dst_u8, bool enable_bf16, bool enable_fp32,
        bool enable_relu, bool enable_mul_add, bool enable_sigmoid,
        bool enable_dst_scales,
        bool enable_src_wei_scales, bool enable_src_zp, bool enable_dst_zp,
        bool enable_bf16_mul_add, bool enable_dst_s8) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D starts");

    // Initialize data vectors for source, weights, bias, add, and multiply operations
    std::vector<uint8_t> src_data(M * K);
    std::vector<int8_t> weights_data(K * N);
    std::vector<float> bias_data(1 * N);
    std::vector<float> add_data(M * N);
    std::vector<float> mul_data(M * N);

    // Fill the data vectors with some values
    for (int i=0; i<src_data.size(); i++) {
        src_data[i] = i* 13 % 21;
    }
    for (int i=0; i<weights_data.size(); i++) {
        weights_data[i] = i* 13 % 21 - 10;
    }
    for (int i=0; i<bias_data.size(); i++) {
        bias_data[i] = 1.5*(i %21);
    }
    for (int i=0; i<add_data.size(); i++) {
        add_data[i] = 1.5*(i %21);
    }
    for (int i=0; i<mul_data.size(); i++) {
        mul_data[i] = 1.5*(i %21);
    }

    //Set zero_point vals for src and dst

    int32_t zp_A = 3, zp_C = 2;

    bool x = zendnn_custom_op::zendnn_reorder(weights_data.data(),
             weights_data.data(), K, N, false, zendnn_s8);

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md = memory::desc(src_dims, dt::u8, tag::ab);
    auto weights_md = memory::desc(weights_dims, dt::s8, tag::ab, true);
    auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
    memory::desc dst_md;
    if (enable_bf16) {
        dst_md = memory::desc(dst_dims, dt::bf16, tag::ab);
    }
    else if (enable_fp32) {
        dst_md = memory::desc(dst_dims, dt::f32, tag::ab);
    }
    else if (enable_dst_u8) {
        dst_md = memory::desc(dst_dims, dt::u8, tag::ab);
    }
    else if (enable_dst_s8) {
        dst_md = memory::desc(dst_dims, dt::s8, tag::ab);
    }

    auto add_md = memory::desc(dst_dims, dt::f32, tag::ab);
    auto mul_md = memory::desc(dst_dims, dt::f32, tag::ab);

    memory::desc add_md_bf16;
    memory::desc mul_md_bf16;
    if (enable_bf16_mul_add) {
        add_md_bf16 = memory::desc(dst_dims, dt::bf16, tag::ab);
        mul_md_bf16 = memory::desc(dst_dims, dt::bf16, tag::ab);
    }

    auto src_mem = memory(src_md, eng);
    auto weights_mem = memory(weights_md, eng);
    auto bias_mem = memory(bias_md, eng);
    auto dst_mem = memory(dst_md, eng);
    auto add_mem = memory(add_md, eng);
    auto mul_mem = memory(mul_md, eng);

    memory add_mem_bf16;
    memory mul_mem_bf16;
    if (enable_bf16_mul_add) {
        add_mem_bf16 = memory(add_md_bf16, eng);
        mul_mem_bf16 = memory(mul_md_bf16, eng);
    }
    memory zp_A_mem({{1}, memory::data_type::s32, {1}}, eng);
    memory zp_C_mem({{1}, memory::data_type::s32, {1}}, eng);

    write_to_zendnn_memory(src_data.data(), src_mem);
    write_to_zendnn_memory(weights_data.data(), weights_mem);
    write_to_zendnn_memory(bias_data.data(), bias_mem);
    write_to_zendnn_memory((void *)&zp_A, zp_A_mem);
    write_to_zendnn_memory((void *)&zp_C, zp_C_mem);
    write_to_zendnn_memory(add_data.data(), add_mem);
    write_to_zendnn_memory(mul_data.data(), mul_mem);
    if (enable_bf16_mul_add) {
        reorder(mul_mem, mul_mem_bf16).execute(engine_stream, mul_mem, mul_mem_bf16);
        reorder(add_mem, add_mem_bf16).execute(engine_stream, add_mem, add_mem_bf16);
    }


    memory src_scale_mem({{1}, memory::data_type::f32, {1}}, eng);
    memory wei_scale_mem({{wei_scale.size()}, memory::data_type::f32, {1}}, eng);
    memory dst_scale_mem({{1}, memory::data_type::f32, {1}}, eng);

    write_to_zendnn_memory(src_scale.data(), src_scale_mem);
    write_to_zendnn_memory(wei_scale.data(), wei_scale_mem);
    write_to_zendnn_memory(dst_scale.data(), dst_scale_mem);

    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, weights_md, bias_md, dst_md);
    // Create primitive post-ops (ReLU).
    const float scale = 3.4f;
    const float alpha = 0.f;
    const float beta = 0.f;

    primitive_attr matmul_attr;
    post_ops matmul_ops;
    int add_t;
    int mul_t;
    if (enable_relu) {
        matmul_ops.append_eltwise(1, algorithm::eltwise_relu, alpha, beta);
    }
    if (enable_mul_add) {
        if (enable_relu) {
            add_t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(2) | ZENDNN_ARG_SRC_1;
            mul_t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(1) | ZENDNN_ARG_SRC_1;
        }
        else {
            add_t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(1) | ZENDNN_ARG_SRC_1;
            mul_t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1;
        }
        if (enable_bf16_mul_add) {
            matmul_ops.append_binary(algorithm::binary_mul,mul_md_bf16);
            matmul_ops.append_binary(algorithm::binary_add,add_md_bf16);
        }
        else {
            matmul_ops.append_binary(algorithm::binary_mul,mul_md);
            matmul_ops.append_binary(algorithm::binary_add,add_md);
        }
    }
    if (enable_sigmoid) {
        matmul_ops.append_eltwise(1, algorithm::eltwise_logistic, 1.f, beta);
    }
    matmul_attr.set_post_ops(matmul_ops);

    //ZENDNN_RUNTIME_S32_VAL used to create primitive without actual zero_point
    if (enable_src_zp)
        matmul_attr.set_zero_points(ZENDNN_ARG_SRC, 0, {ZENDNN_RUNTIME_S32_VAL});
    if (enable_dst_zp)
        matmul_attr.set_zero_points(ZENDNN_ARG_DST, 0, {ZENDNN_RUNTIME_S32_VAL});
    if (enable_src_wei_scales) {
        matmul_attr.set_scales_mask(ZENDNN_ARG_SRC, 0, {}, dt::f32);
        matmul_attr.set_scales_mask(ZENDNN_ARG_WEIGHTS, 2, {1,1}, dt::f32);
    }
    if (enable_dst_scales)
        matmul_attr.set_scales_mask(ZENDNN_ARG_DST, 0, {}, dt::f32);
    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);
    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);

    // Primitive execution: matrix multiplication with zero_points.
    // Prepare the arguments for the execute call
    std::unordered_map<int, memory> args = {
        {ZENDNN_ARG_SRC, src_mem},
        {ZENDNN_ARG_WEIGHTS, weights_mem},
        {ZENDNN_ARG_BIAS, bias_mem},
        {ZENDNN_ARG_DST, dst_mem}
    };

    // Add optional arguments based on runtime conditions
    if (enable_mul_add) {
        if (!enable_bf16_mul_add) {
            args[add_t] = add_mem;
            args[mul_t] = mul_mem;
        }
        else {
            args[add_t] = add_mem_bf16;
            args[mul_t] = mul_mem_bf16;
        }
    }

    if (enable_src_wei_scales) {
        args[ZENDNN_ARG_ATTR_SCALES | ZENDNN_ARG_SRC] = src_scale_mem;
        args[ZENDNN_ARG_ATTR_SCALES | ZENDNN_ARG_WEIGHTS] = wei_scale_mem;
    }

    if (enable_dst_scales) {
        args[ZENDNN_ARG_ATTR_SCALES | ZENDNN_ARG_DST] = dst_scale_mem;
    }

    if (enable_src_zp) {
        args[ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_SRC] = zp_A_mem;
    }

    if (enable_dst_zp) {
        args[ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_DST] = zp_C_mem;
    }

    matmul_prim.execute(engine_stream, args);
    matmul_prim.execute(engine_stream, args);
    matmul_prim.execute(engine_stream, args);
    matmul_prim.execute(engine_stream, args);
    // Wait for the computation to finalize.

    if (enable_bf16) {
        std::vector<T> result(M*N);
        auto dst_md_f32 = memory::desc(dst_dims, dt::f32, tag::ab);
        auto dst_mem_f32 = memory(dst_md_f32, eng);

        reorder(dst_mem, dst_mem_f32).execute(engine_stream, dst_mem, dst_mem_f32);
        engine_stream.wait();

        read_from_zendnn_memory(result.data(), dst_mem_f32);
        zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D ends");
        return result;
    }
    else {
        std::vector<T> dst_data(M * N);
        engine_stream.wait();
        read_from_zendnn_memory(dst_data.data(), dst_mem);
        zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D ends");
        return dst_data;
    }
}

template<typename T>
std::vector<T> matmul_example_2D_dst_ref(zendnn::engine eng,
        zendnn::stream engine_stream, memory::dim M, memory::dim K, memory::dim N,
        std::vector<float> &src_scale, std::vector<float> &wei_scale,
        std::vector<float> &dst_scale,
        bool enable_dst_u8, bool enable_bf16, bool enable_fp32,
        bool enable_relu, bool enable_mul_add, bool enable_sigmoid,
        bool enable_dst_scales,
        bool enable_src_wei_scales, bool enable_src_zp, bool enable_dst_zp,
        bool enable_bf16_mul_add, bool enable_dst_s8) {
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D starts");

    // Initialize data vectors for source, weights, bias, add, and multiply operations
    std::vector<uint8_t> src_data(M * K);
    std::vector<int8_t> weights_data(K * N);
    std::vector<float> bias_data(1 * N);
    std::vector<float> add_data(M * N);
    std::vector<float> mul_data(M * N);

    // Fill the data vectors with some values
    for (int i=0; i<src_data.size(); i++) {
        src_data[i] = i* 13 % 21;
    }
    for (int i=0; i<weights_data.size(); i++) {
        weights_data[i] = i* 13 % 21 - 10;
    }
    for (int i=0; i<bias_data.size(); i++) {
        bias_data[i] = 1.5*(i %21);
    }
    for (int i=0; i<add_data.size(); i++) {
        add_data[i] = 1.5*(i %21);
    }
    for (int i=0; i<mul_data.size(); i++) {
        mul_data[i] = 1.5*(i %21);
    }

    //Set zero_point vals for src and dst
    int32_t zp_A = 3, zp_C = 2;

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md = memory::desc(src_dims, dt::u8, tag::ab);
    auto weights_md = memory::desc(weights_dims, dt::s8, tag::ab, true);
    auto bias_md = memory::desc(bias_dims, dt::f32, tag::ab);
    memory::desc dst_md;
    if (enable_bf16) {
        dst_md = memory::desc(dst_dims, dt::bf16, tag::ab);
    }
    else if (enable_fp32) {
        dst_md = memory::desc(dst_dims, dt::f32, tag::ab);
    }
    else if (enable_dst_u8) {
        dst_md = memory::desc(dst_dims, dt::u8, tag::ab);
    }
    else if (enable_dst_s8) {
        dst_md = memory::desc(dst_dims, dt::s8, tag::ab);
    }
    auto add_md = memory::desc(dst_dims, dt::f32, tag::ab);
    auto mul_md = memory::desc(dst_dims, dt::f32, tag::ab);
    memory::desc add_md_bf16;
    memory::desc mul_md_bf16;
    if (enable_bf16_mul_add) {
        add_md_bf16 = memory::desc(dst_dims, dt::bf16, tag::ab);
        mul_md_bf16 = memory::desc(dst_dims, dt::bf16, tag::ab);
    }

    // Create memory objects for source, weights, bias, destination, add, and multiply
    auto src_mem = memory(src_md, eng);
    auto weights_mem = memory(weights_md, eng);
    auto bias_mem = memory(bias_md, eng);
    auto dst_mem = memory(dst_md, eng);
    auto add_mem = memory(add_md, eng);
    auto mul_mem = memory(mul_md, eng);

    memory add_mem_bf16;
    memory mul_mem_bf16;
    if (enable_bf16_mul_add) {
        add_mem_bf16 = memory(add_md_bf16, eng);
        mul_mem_bf16 = memory(mul_md_bf16, eng);
    }

    // Create memory objects for zero points
    memory zp_A_mem({{1}, memory::data_type::s32, {1}}, eng);
    memory zp_C_mem({{1}, memory::data_type::s32, {1}}, eng);

    // Write data to memory objects
    write_to_zendnn_memory(src_data.data(), src_mem);
    write_to_zendnn_memory(weights_data.data(), weights_mem);
    write_to_zendnn_memory((void *)&zp_A, zp_A_mem);
    write_to_zendnn_memory((void *)&zp_C, zp_C_mem);
    write_to_zendnn_memory(add_data.data(), add_mem);
    write_to_zendnn_memory(mul_data.data(), mul_mem);
    if (enable_bf16_mul_add) {
        reorder(mul_mem, mul_mem_bf16).execute(engine_stream, mul_mem, mul_mem_bf16);
        reorder(add_mem, add_mem_bf16).execute(engine_stream, add_mem, add_mem_bf16);
    }

    // Adjust bias data if source and weight scales are enabled
    std::vector<float> bias_scale(N);
    if (enable_src_wei_scales) {
        for (int i = 0; i < bias_scale.size(); i++) {
            bias_scale[i] = wei_scale[i%wei_scale.size()] * src_scale[0];
            bias_data[i]/=bias_scale[i];
        }
    }
    write_to_zendnn_memory(bias_data.data(), bias_mem);

    // Prepare output scales if source and weight scales are enabled
    std::vector<float> output_scales;
    if (enable_src_wei_scales) {
        output_scales.resize(wei_scale.size());
        for (int i = 0; i < wei_scale.size(); i++) {
            output_scales[i] = wei_scale[i%wei_scale.size()] * src_scale[0];
        }
    }

    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, weights_md, bias_md, dst_md);
    // Create primitive post-ops (ReLU).
    const float scale = 3.4f;
    const float alpha = 0.f;
    const float beta = 0.f;

    primitive_attr matmul_attr;
    int count=0;
    post_ops matmul_ops;
    int add_t;
    int mul_t;
    if (enable_relu) {
        count++;
        matmul_ops.append_eltwise(1, algorithm::eltwise_relu, alpha, beta);
    }
    if (enable_mul_add) {
        if (enable_relu) {
            add_t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(2) | ZENDNN_ARG_SRC_1;
            mul_t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(1) | ZENDNN_ARG_SRC_1;
        }
        else {
            add_t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(1) | ZENDNN_ARG_SRC_1;
            mul_t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1;
        }
        count++;
        if (enable_bf16_mul_add) {
            matmul_ops.append_binary(algorithm::binary_mul,mul_md_bf16);
        }
        else {
            matmul_ops.append_binary(algorithm::binary_mul,mul_md);
        }
        count++;
        if (enable_bf16_mul_add) {
            matmul_ops.append_binary(algorithm::binary_add,mul_md_bf16);
        }
        else {
            matmul_ops.append_binary(algorithm::binary_add,add_md);
        }
    }
    if (enable_sigmoid) {
        count++;
        matmul_ops.append_eltwise(1, algorithm::eltwise_logistic, 1.f, beta);
    }
    memory dst_scale_mem;
    int scale_t;
    if (enable_dst_scales) {
        scale_t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(count) | ZENDNN_ARG_SRC_1;
        auto scale_md = memory::desc({1}, dt::f32, tag::a);
        //memory dst_scale_mem(scale_md, eng);
        dst_scale_mem = memory(scale_md, eng);
        write_to_zendnn_memory(dst_scale.data(), dst_scale_mem);
        matmul_ops.append_binary(algorithm::binary_mul,scale_md);
    }
    matmul_attr.set_post_ops(matmul_ops);

    //Add zero_point
    //ZENDNN_RUNTIME_S32_VAL used to create primitive without actual zero_point
    if (enable_src_zp)
        matmul_attr.set_zero_points(ZENDNN_ARG_SRC, 0, {ZENDNN_RUNTIME_S32_VAL});
    if (enable_dst_zp)
        matmul_attr.set_zero_points(ZENDNN_ARG_DST, 0, {ZENDNN_RUNTIME_S32_VAL});
    if (enable_src_wei_scales) {
        if (wei_scale.size()==N) {
            matmul_attr.set_output_scales(2,output_scales);
        }
        else {
            matmul_attr.set_output_scales(0,output_scales);
        }
    }
    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);
    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);

    // Primitive execution: matrix multiplication with zero_points.
    // Prepare the arguments for the execute call
    std::unordered_map<int, memory> args = {
        {ZENDNN_ARG_SRC, src_mem},
        {ZENDNN_ARG_WEIGHTS, weights_mem},
        {ZENDNN_ARG_BIAS, bias_mem},
        {ZENDNN_ARG_DST, dst_mem}
    };

    // Add optional arguments based on runtime conditions
    if (enable_mul_add) {
        if (!enable_bf16_mul_add) {
            args[add_t] = add_mem;
            args[mul_t] = mul_mem;
        }
        else {
            args[add_t] = add_mem_bf16;
            args[mul_t] = mul_mem_bf16;
        }
    }

    if (enable_dst_scales) {
        args[scale_t] = dst_scale_mem;
    }

    if (enable_src_zp) {
        args[ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_SRC] = zp_A_mem;
    }

    if (enable_dst_zp) {
        args[ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_DST] = zp_C_mem;
    }

    // Execute the primitive with the prepared arguments
    matmul_prim.execute(engine_stream, args);

    // Wait for the computation to finalize.
    if (enable_bf16) {
        std::vector<T> result(M*N);
        auto dst_md_f32 = memory::desc(dst_dims, dt::f32, tag::ab);
        auto dst_mem_f32 = memory(dst_md_f32, eng);

        reorder(dst_mem, dst_mem_f32).execute(engine_stream, dst_mem, dst_mem_f32);
        engine_stream.wait();

        read_from_zendnn_memory(result.data(), dst_mem_f32);
        zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D ends");
        return result;
    }
    else {
        std::vector<T> dst_data(M * N);
        engine_stream.wait();
        read_from_zendnn_memory(dst_data.data(), dst_mem);
        zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test: matmul_example_2D ends");
        return dst_data;
    }
}

int main(int argc, char **argv) {
    // Parse command-line arguments
    parse_arguments(argc, argv);

    // Join command-line arguments into a single string
    std::string args = join_arguments(argc, argv);

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

    // Define list of M values
    std::vector<memory::dim> M_list = {1024}; // Add more M values as needed

    // Define list of (K, N) pairs
    std::vector<std::pair<memory::dim, memory::dim>> KN_list = {
        {3456,512},
        {512, 3456},
        {512,256},
        {13,    512},
        {256,   128},
        {3456,  1024},
        {1024,  1024},
        {1024,  512},
        {256,   1},
        {13,    512},
        {256,   64},
        {415,   512},
        {512,   512},
        {256,   1}
        // Add more (K, N) pairs as needed
    };

    for (const auto &M : M_list) {
        for (const auto &KN : KN_list) {
            memory::dim K = KN.first;
            memory::dim N = KN.second;

            std::cout<<"\nM="<<M<<", N="<<N<<", K="<<K;

            // Create a seed based on M, K, and N
            size_t seed = std::hash<memory::dim>()(M) ^ std::hash<memory::dim>()(
                              K) ^ std::hash<memory::dim>()(N);
            std::default_random_engine gen(seed);
            std::uniform_real_distribution<double> distribution(0.0, 1.0);

            // Generate scales
            std::vector<float> src_scale(1);
            std::vector<float> wei_scale(N);
            std::vector<float> dst_scale(1);

            for (auto &scale : src_scale) {
                scale = distribution(gen);
            }
            for (auto &scale : wei_scale) {
                scale = distribution(gen);
            }
            for (auto &scale : dst_scale) {
                scale = distribution(gen);
            }

            zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();

            obj.is_brgemm = false;
            obj.is_ref_gemm_bf16 = false;

            std::cout<<"\n Actual MatMul\n";
            std::vector<float> aocl_float;
            std::vector<uint8_t> aocl_uint8;
            std::vector<int8_t> aocl_sint8;
            if (ENABLE_DST_U8) {
                aocl_uint8 = matmul_example_2D_dst_actual<uint8_t>(eng,
                             engine_stream, M, K, N, src_scale, wei_scale, dst_scale,
                             ENABLE_DST_U8, ENABLE_BF16, ENABLE_FP32,
                             ENABLE_RELU, ENABLE_MUL_ADD, ENABLE_SIGMOID, ENABLE_DST_SCALES,
                             ENABLE_SRC_WEI_SCALES, ENABLE_SRC_ZP, ENABLE_DST_ZP, ENABLE_BF16_MUL_ADD,
                             ENABLE_DST_S8);
            }
            else if (ENABLE_DST_S8) {
                aocl_sint8 = matmul_example_2D_dst_actual<int8_t>(eng,
                             engine_stream, M, K, N, src_scale, wei_scale, dst_scale,
                             ENABLE_DST_U8, ENABLE_BF16, ENABLE_FP32,
                             ENABLE_RELU, ENABLE_MUL_ADD, ENABLE_SIGMOID, ENABLE_DST_SCALES,
                             ENABLE_SRC_WEI_SCALES, ENABLE_SRC_ZP, ENABLE_DST_ZP, ENABLE_BF16_MUL_ADD,
                             ENABLE_DST_S8);
            }
            else {
                aocl_float = matmul_example_2D_dst_actual<float>(eng,
                             engine_stream, M, K, N, src_scale, wei_scale, dst_scale,
                             ENABLE_DST_U8, ENABLE_BF16, ENABLE_FP32,
                             ENABLE_RELU, ENABLE_MUL_ADD, ENABLE_SIGMOID, ENABLE_DST_SCALES,
                             ENABLE_SRC_WEI_SCALES, ENABLE_SRC_ZP, ENABLE_DST_ZP, ENABLE_BF16_MUL_ADD,
                             ENABLE_DST_S8);
            }
            //Set JIT kernel
            obj.is_brgemm = true;
            obj.is_ref_gemm_bf16 = true;

            std::cout<<"JIT MatMul\n";
            if (ENABLE_DST_U8) {
                std::vector<uint8_t> jit_uint8 = matmul_example_2D_dst_ref<uint8_t>(eng,
                                                 engine_stream, M,
                                                 K, N, src_scale, wei_scale, dst_scale,
                                                 ENABLE_DST_U8, ENABLE_BF16, ENABLE_FP32,
                                                 ENABLE_RELU, ENABLE_MUL_ADD, ENABLE_SIGMOID, ENABLE_DST_SCALES,
                                                 ENABLE_SRC_WEI_SCALES, ENABLE_SRC_ZP, ENABLE_DST_ZP, ENABLE_BF16_MUL_ADD,
                                                 ENABLE_DST_S8);

                auto rc = compare_vectors<uint8_t>(
                              jit_uint8, aocl_uint8, 256, "Compare INT8 os8 JIT MatMul vs ZenDNN Paths",
                              args);
            }
            else if (ENABLE_DST_S8) {
                std::vector<int8_t> jit_int8 = matmul_example_2D_dst_ref<int8_t>(eng,
                                               engine_stream, M,
                                               K, N, src_scale, wei_scale, dst_scale,
                                               ENABLE_DST_U8, ENABLE_BF16, ENABLE_FP32,
                                               ENABLE_RELU, ENABLE_MUL_ADD, ENABLE_SIGMOID, ENABLE_DST_SCALES,
                                               ENABLE_SRC_WEI_SCALES, ENABLE_SRC_ZP, ENABLE_DST_ZP, ENABLE_BF16_MUL_ADD,
                                               ENABLE_DST_S8);

                auto rc = compare_vectors<int8_t>(
                              jit_int8, aocl_sint8, 256, "Compare INT8 os8 JIT MatMul vs ZenDNN Paths",
                              args);
            }
            else {
                std::vector<float> jit_float = matmul_example_2D_dst_ref<float>(eng,
                                               engine_stream, M,
                                               K, N, src_scale, wei_scale, dst_scale,
                                               ENABLE_DST_U8, ENABLE_BF16, ENABLE_FP32,
                                               ENABLE_RELU, ENABLE_MUL_ADD, ENABLE_SIGMOID, ENABLE_DST_SCALES,
                                               ENABLE_SRC_WEI_SCALES, ENABLE_SRC_ZP, ENABLE_DST_ZP, ENABLE_BF16_MUL_ADD,
                                               ENABLE_DST_S8);

                auto rc = compare_vectors<float>(
                              jit_float, aocl_float, 256,
                              "Compare INT8 os8 JIT MatMul vs ZenDNN Paths", args);
            }
        }
    }
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_matmul_test test ends");
    return 0;
}
