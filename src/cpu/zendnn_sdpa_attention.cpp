/*******************************************************************************
* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <type_traits>

#define dim_t long int

// using namespace zendnn;
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <zendnn.hpp>
#include "zendnn_logging.hpp"

#ifndef _WIN32
    #include <sys/time.h>
#else
    #include <windows.h>
#endif

using namespace zendnn;


namespace zendnn {

void zenAttention_Matmul_Primitive(const memory& a ,
            zendnn::memory::desc a_md,
            const memory& b,
            zendnn::memory::desc b_md,
            float scale,
            const memory& o,
            zendnn::memory::desc o_md,
            zendnn::engine eng,
            zendnn::stream engine_stream){

    primitive_attr matmul_attr;
    zendnn::post_ops post_ops;
    bool post_attr = false;
    if (scale != 1.f) {
        post_attr = true;
        post_ops.append_eltwise(/* mask */ 1, algorithm::eltwise_linear, scale, 0);
    }
    if (post_attr) {
        matmul_attr.set_post_ops(post_ops);
    }

    // Create operation descriptor
    auto matmul_d = matmul::desc(a_md, b_md, o_md);

    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);

    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({ZENDNN_ARG_SRC, a});
    matmul_args.insert({ZENDNN_ARG_WEIGHTS, b});
    matmul_args.insert({ZENDNN_ARG_DST, o});

    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim.execute(engine_stream, matmul_args);

}

void zenAttention_Binary_Add_Primitive(memory& a,
            zendnn::memory::desc a_md,
            const memory& b,
            zendnn::engine eng,
            zendnn::stream engine_stream){

    auto binary_d = zendnn::binary::desc(zendnn::algorithm::binary_add,
                                         a_md,
                                         b.get_desc(),
                                         a_md
                                         );

    auto binary_pd = zendnn::binary::primitive_desc(binary_d, eng);

    // Create the primitive.
    auto binary_prim = zendnn::binary(binary_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> binary_add_args;
    binary_add_args.insert({ZENDNN_ARG_SRC_0, a});
    binary_add_args.insert({ZENDNN_ARG_SRC_1, b});
    binary_add_args.insert({ZENDNN_ARG_DST, a});

    // Primitive execution: binary with ReLU.
    binary_prim.execute(engine_stream, binary_add_args);

}


void zenAttention_Softmax(const memory& a,
            zendnn::memory::desc a_md,
            int axis,
            zendnn::engine eng,
            zendnn::stream engine_stream){

    auto softmax_d
            = softmax_forward::desc(prop_kind::forward_training, a_md, axis);

    // Create primitive descriptor.
    auto softmax_pd = softmax_forward::primitive_desc(softmax_d, eng);

    // Create the primitive.
    auto softmax_prim = softmax_forward(softmax_pd);

    // Primitive arguments. Set up in-place execution by assigning src as DST.
    std::unordered_map<int, memory> softmax_args;
    softmax_args.insert({ZENDNN_ARG_SRC, a});
    softmax_args.insert({ZENDNN_ARG_DST, a});

    // Primitive execution.
    softmax_prim.execute(engine_stream, softmax_args);

}


void zendnn_custom_op::zendnn_sdpa_attention(
                const memory& input_Q_mem, const memory& input_K_mem,
                const memory& input_V_mem,
                memory& input_mask_mem,
                memory& output_mem
                )
{
    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    float cur_algo_time; //current algorithm's execution time
    struct timeval start_n, end_n;

    //timer start
#ifdef _WIN32
        auto start_n = std::chrono::high_resolution_clock::now();
#else
        gettimeofday(&start_n, 0);
#endif

    auto dta = input_Q_mem.get_desc().data_type();
    auto dta_c = static_cast<zendnn_data_type_t>(dta);

    //get memory descriptors
    auto query_md = input_Q_mem.get_desc();
    auto key_md = input_K_mem.get_desc();
    auto value_md = input_V_mem.get_desc();
    auto mask_md = input_mask_mem.get_desc();
    auto output_md = output_mem.get_desc();

    const dim_t B = query_md.dims()[0];
    const dim_t N = query_md.dims()[1];
    const dim_t S = query_md.dims()[2];
    const dim_t H = query_md.dims()[3];

    //KV cache length L
    const dim_t L = key_md.dims()[2];

    float scale = 1/sqrt(H);

    memory::dims kt_dims = {B, N, H, L};

    //QK' matmul
    memory::dims qk_output_dims = {B, N, S, L};
    auto qk_output_md = memory::desc(qk_output_dims, dta, zendnn::memory::format_tag::abcd);

    auto qk_output_mem = memory(qk_output_md, eng);

    auto temp_md = memory::desc(kt_dims, dta, zendnn::memory::format_tag::abdc);

    zenAttention_Matmul_Primitive(input_Q_mem, query_md,
                                input_K_mem, temp_md,
                                scale, qk_output_mem, qk_output_md, eng, engine_stream);

    //Add mask
    if ( input_mask_mem.get_desc().get_size() && input_mask_mem.get_data_handle() != nullptr )
        zenAttention_Binary_Add_Primitive(qk_output_mem, qk_output_md, input_mask_mem, eng, engine_stream);

    //softmax
    zenAttention_Softmax(qk_output_mem, qk_output_md, 3, eng, engine_stream);

    //QKV final matmul
    zenAttention_Matmul_Primitive(qk_output_mem, qk_output_md,
                                input_V_mem, value_md,
                                1.0f, output_mem, output_md, eng, engine_stream);

    //timer end
#ifdef _WIN32
    auto end_n = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> difference = end_n - start_n;
    cur_algo_time = difference.count();
#else
    gettimeofday(&end_n, 0);
    cur_algo_time = (end_n.tv_sec - start_n.tv_sec) * 1000.0f +
        (end_n.tv_usec - start_n.tv_usec)/ 1000.0f; //time in milliseconds
#endif

    zendnnVerbose(ZENDNN_PROFLOG,"ZenDNN SDPA for :", " Batch: ",B,", Num heads: ",N,", Sequence length: ",S,
            ", Head size: ",H,", KV Cache length (Decoder): ",L, ", time : ",cur_algo_time);

}

}
