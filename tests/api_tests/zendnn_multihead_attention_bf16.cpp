/*******************************************************************************
* Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
*
*
*******************************************************************************/

/* Steps:
 *  1. create engine and stream
 *  2. create F32 user memory : input(q,k,v), weights(q,k,v), bias(q,k,v), mask
 *  3. initialize the F32 user memory
 *  4. create BF16 user memory : input(q,k,v), weights(q,k,v)
 *  5. create memory descriptor for F32 and BF16
 *  6. reorder F32 memory to BF16 memory
 *  7. create attention descriptor with BF16 memory
 *  8. create attention primitive descriptor
 *  9. create attentiom primitive
 *  10. execute the attention primitive

 Additional
 *  1. create attention descriptor with F32 memory
 *  2. create attention primitive descriptor
 *  3. execute the attention primitive
 *  4. compare the output from F32 and BF16 attention primitive
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <numeric>
#include <vector>
#include <cmath>

#include "test_utils.hpp"
#include "zendnn.hpp"

using namespace std;
using namespace zendnn;

using tag = memory::format_tag;
using dt = memory::data_type;

inline zendnn::memory::dim product(const zendnn::memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (zendnn::memory::dim)1,
            std::multiplies<zendnn::memory::dim>());
}

int attention_example(zendnn::engine::kind engine_kind, int argc, char **argv) {

    // Create execution zendnn::engine.
    zendnn::engine engine(engine_kind, 0);

    // Create zendnn::stream.
    zendnn::stream engine_stream(engine);

    int b,n,s,h,id;
    if(argc >= 2)
    {
        b = stoi(argv[1]);
        s = stoi(argv[2]);
        n = stoi(argv[3]);
        h = stoi(argv[4]);
        id = stoi(argv[3]) * stoi(argv[4]);
    }
    else
    {
        b = 64;
        s = 1024;
        n = 32;
        h = 128;
        id = n * h;
    }

    // Tensor dimensions.
    const memory::dim B = b, // batch size
                      S = s, // sequence length
                      Id = id, // Input dimension
                      N = n, // number of heads
                      H = h; // Hidden dimension/Size of each head

    memory::dim Od=Id; // Output dimension ; Id==Od
    const float scale = 1/sqrt(H);   //sqrt(H)
    const uint32_t num_threads = 64; // number of threads

    memory::dims src_query_dims = {B, S, Id};
    memory::dims src_key_dims = {B, S, Id};
    memory::dims src_value_dims = {B, S, Id};
    memory::dims weight_query_dims = {Id, N*H};
    memory::dims weight_key_dims = {Id, N*H};
    memory::dims weight_value_dims = {Id, N*H};
    memory::dims bias_query_dims = {N*H};
    memory::dims bias_key_dims = {N*H};
    memory::dims bias_value_dims = {N*H};
    memory::dims mask_dims = {B, S};
    memory::dims dst_dims = {B, S, Od};
    //We need BxNxSxS + 4*(BxNxSxH) number of element for scratchpad memory, for all computations.

    std::cout <<"\nSRC size: " <<product(src_query_dims);
    std::cout <<"\nQuery Weights size: " <<product(weight_query_dims);
    std::cout <<"\nKey Weights size: " <<product(weight_key_dims);
    std::cout <<"\nValue Weights size: " <<product(weight_value_dims);
    std::cout <<"\nQuery Bias size: " <<product(bias_query_dims);
    std::cout <<"\nKey Bias size: " <<product(bias_key_dims);
    std::cout <<"\nValue Bias size: " <<product(bias_value_dims);
    std::cout <<"\nMask size: " <<product(mask_dims);
    std::cout <<"\nNumber of heads: " <<N;
    std::cout <<"\nDST size: " <<product(dst_dims);

    // Allocate temporary float buffers to initialize the values.
    std::vector<float> src_query_data_f32(product(src_query_dims));
    std::vector<float> src_key_data_f32(product(src_key_dims));
    std::vector<float> src_value_data_f32(product(src_value_dims));
    std::vector<float> weight_query_data_f32(product(weight_query_dims));
    std::vector<float> weight_key_data_f32(product(weight_key_dims));
    std::vector<float> weight_value_data_f32(product(weight_value_dims));
    std::vector<float> bias_query_data_f32(product(bias_query_dims));
    std::vector<float> bias_key_data_f32(product(bias_key_dims));
    std::vector<float> bias_value_data_f32(product(bias_value_dims));

    std::vector<float> mask_data_f32(product(mask_dims));
    std::generate(mask_data_f32.begin(), mask_data_f32.end(), []() {
            static int i = 0;
            i ++ ;
            return (i % 5);
        });

    // Initialize src, weights, and bias tensors.
    std::generate(src_query_data_f32.begin(), src_query_data_f32.end(), []() {
        static int i = 0;
        i ++ ;
        return (i % 5);
    });
    std::generate(src_key_data_f32.begin(), src_key_data_f32.end(), []() {
        static int i = 0;
        i ++ ;
        return (i % 5);
    });
    std::generate(src_value_data_f32.begin(), src_value_data_f32.end(), []() {
        static int i = 0;
        i ++ ;
        return (i % 5);
    });

    std::generate(weight_query_data_f32.begin(), weight_query_data_f32.end(), []() {
        static int i = 0;
        i ++ ;
        return (i % 5);
    });
    std::generate(weight_key_data_f32.begin(), weight_key_data_f32.end(), []() {
        static int i = 0;
        i ++ ;
        return (i % 5);
    });
    std::generate(weight_value_data_f32.begin(), weight_value_data_f32.end(), []() {
        static int i = 0;
        i ++ ;
        return (i % 5);
    });

    std::generate(bias_query_data_f32.begin(), bias_query_data_f32.end(), []() {
        static int i = 0;
        i ++ ;
        return (i % 5);
    });
    std::generate(bias_key_data_f32.begin(), bias_key_data_f32.end(), []() {
        static int i = 0;
        i ++ ;
        return (i % 5);
    });
    std::generate(bias_value_data_f32.begin(), bias_value_data_f32.end(), []() {
        static int i = 0;
        i ++ ;
        return (i % 5);
    });

    std::vector<primitive> prim_vec;
    std::vector<std::unordered_map<int, memory>> prim_args_vec;

    std::cout <<"\n\n BF16 Attention_1 \n\n" ;
    // Create Temporary float memory objects for tensor data
    // (src, query_weights, key_weights, value_weights, query_bias, key_bias, value_bias, mask, dst)
    auto src_query_mem_f32 = memory({src_query_dims, dt::f32, tag::abc}, engine);
    auto src_key_mem_f32 = memory({src_key_dims, dt::f32, tag::abc}, engine);
    auto src_value_mem_f32 = memory({src_value_dims, dt::f32, tag::abc}, engine);
    auto query_weights_mem_f32 = memory({weight_query_dims, dt::f32, tag::ab}, engine);
    auto key_weights_mem_f32 = memory({weight_key_dims, dt::f32, tag::ab}, engine);
    auto value_weights_mem_f32 = memory({weight_value_dims, dt::f32, tag::ab}, engine);
    auto query_bias_mem_f32 = memory({bias_query_dims, dt::f32, tag::a}, engine);
    auto key_bias_mem_f32 = memory({bias_key_dims, dt::f32, tag::a}, engine);
    auto value_bias_mem_f32 = memory({bias_value_dims, dt::f32, tag::a}, engine);
    auto mask_mem_f32 = memory({mask_dims, dt::f32, tag::ab}, engine);
    auto dst_mem_f32 = memory({dst_dims, dt::f32, tag::abc}, engine);

    //To compare outputs from BF16 and F32 attention
    auto dst_mem_f32_temp = memory({dst_dims, dt::f32, tag::abc}, engine);

    // Create BF16 memory objects for tensor data (src, query_weights, key_weights, value_weights, dst)
    // Not required for mask and bias tensors as bf16 attention expects mask and bias tensors in float
    auto src_query_mem_bf16 = memory({src_query_dims, dt::bf16, tag::abc}, engine);
    auto src_key_mem_bf16 = memory({src_key_dims, dt::bf16, tag::abc}, engine);
    auto src_value_mem_bf16 = memory({src_value_dims, dt::bf16, tag::abc}, engine);
    auto query_weights_mem_bf16 = memory({weight_query_dims, dt::bf16, tag::ab}, engine);
    auto key_weights_mem_bf16 = memory({weight_key_dims, dt::bf16, tag::ab}, engine);
    auto value_weights_mem_bf16 = memory({weight_value_dims, dt::bf16, tag::ab}, engine);
    auto dst_mem_bf16 = memory({dst_dims, dt::bf16, tag::abc}, engine);

    // Write float data to float memory object's handle.
    write_to_zendnn_memory(src_query_data_f32.data(), src_query_mem_f32);
    write_to_zendnn_memory(src_key_data_f32.data(), src_key_mem_f32);
    write_to_zendnn_memory(src_value_data_f32.data(), src_value_mem_f32);
    write_to_zendnn_memory(weight_query_data_f32.data(), query_weights_mem_f32);
    write_to_zendnn_memory(weight_key_data_f32.data(), key_weights_mem_f32);
    write_to_zendnn_memory(weight_value_data_f32.data(), value_weights_mem_f32);
    write_to_zendnn_memory(bias_query_data_f32.data(), query_bias_mem_f32);
    write_to_zendnn_memory(bias_key_data_f32.data(), key_bias_mem_f32);
    write_to_zendnn_memory(bias_value_data_f32.data(), value_bias_mem_f32);
    write_to_zendnn_memory(mask_data_f32.data(), mask_mem_f32);

    zendnn::reorder(src_query_mem_f32, src_query_mem_bf16).execute(engine_stream, src_query_mem_f32, src_query_mem_bf16);
    zendnn::reorder(src_key_mem_f32, src_key_mem_bf16).execute(engine_stream, src_key_mem_f32, src_key_mem_bf16);
    zendnn::reorder(src_value_mem_f32, src_value_mem_bf16).execute(engine_stream, src_value_mem_f32, src_value_mem_bf16);
    zendnn::reorder(query_weights_mem_f32, query_weights_mem_bf16).execute(engine_stream, query_weights_mem_f32, query_weights_mem_bf16);
    zendnn::reorder(key_weights_mem_f32, key_weights_mem_bf16).execute(engine_stream, key_weights_mem_f32, key_weights_mem_bf16);
    zendnn::reorder(value_weights_mem_f32, value_weights_mem_bf16).execute(engine_stream, value_weights_mem_f32, value_weights_mem_bf16);

    /* Experimenting scratch-pad memory */
    zendnn::primitive_attr attn_attr;

    // Create operation descriptor.
    auto attn_desc = attention::desc(prop_kind::forward_inference,
                                     algorithm::multihead_attention,
                                     src_query_mem_bf16.get_desc(),
                                     src_key_mem_bf16.get_desc(),
                                     src_value_mem_bf16.get_desc(),
                                     query_weights_mem_bf16.get_desc(),
                                     key_weights_mem_bf16.get_desc(),
                                     value_weights_mem_bf16.get_desc(),
                                     query_bias_mem_f32.get_desc(),
                                     key_bias_mem_f32.get_desc(),
                                     value_bias_mem_f32.get_desc(),
                                     mask_mem_f32.get_desc(),
                                     dst_mem_bf16.get_desc(),
                                     scale,
                                     N,
                                     num_threads);

    // Create primitive descriptor.
    auto attn_pd = attention::primitive_desc(attn_desc, attn_attr, engine);

    // Create the primitive.
    prim_vec.push_back(attention(attn_pd));

    // Primitive arguments.
    std::unordered_map<int, memory> attn_args;
    attn_args.insert({ZENDNN_ARG_SRC_0, src_query_mem_bf16});
    attn_args.insert({ZENDNN_ARG_SRC_1, src_key_mem_bf16});
    attn_args.insert({ZENDNN_ARG_SRC_2, src_value_mem_bf16});
    attn_args.insert({ZENDNN_ARG_WEIGHTS, query_weights_mem_bf16});
    attn_args.insert({ZENDNN_ARG_WEIGHTS_1, key_weights_mem_bf16});
    attn_args.insert({ZENDNN_ARG_WEIGHTS_2, value_weights_mem_bf16});
    attn_args.insert({ZENDNN_ARG_BIAS_0, query_bias_mem_f32});
    attn_args.insert({ZENDNN_ARG_BIAS_1, key_bias_mem_f32});
    attn_args.insert({ZENDNN_ARG_BIAS_2, value_bias_mem_f32});
    attn_args.insert({ZENDNN_ARG_MASK, mask_mem_f32});
    attn_args.insert({ZENDNN_ARG_DST, dst_mem_bf16});

    // Primitive execution push_back: attention
    prim_args_vec.push_back(attn_args);

    // BF16 Primtive Execution and storing the result

    std::cout <<"\n\nPrimtive bf16 Execution and storing the result\n ";
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<prim_vec.size(); i++) {
        prim_vec.at(i).execute(engine_stream, prim_args_vec.at(i));
        engine_stream.wait();
    }

    zendnn::reorder(dst_mem_bf16, dst_mem_f32).execute(engine_stream, dst_mem_bf16, dst_mem_f32);

    float *float_result= static_cast<float *>(dst_mem_f32.get_data_handle());

    //primtive for f32 attention  ---------------------------

    std::vector<primitive> prim_vec2;
    std::vector<std::unordered_map<int, memory>> prim_args_vec2;

    /* Experimenting scratch-pad memory */
    zendnn::primitive_attr attn_attr_f32;

    // Create operation descriptor.
    auto attn_desc_f32 = attention::desc(prop_kind::forward_inference,
                                     algorithm::multihead_attention,
                                     src_query_mem_f32.get_desc(),
                                     src_key_mem_f32.get_desc(),
                                     src_value_mem_f32.get_desc(),
                                     query_weights_mem_f32.get_desc(),
                                     key_weights_mem_f32.get_desc(),
                                     value_weights_mem_f32.get_desc(),
                                     query_bias_mem_f32.get_desc(),
                                     key_bias_mem_f32.get_desc(),
                                     value_bias_mem_f32.get_desc(),
                                     mask_mem_f32.get_desc(),
                                     dst_mem_f32_temp.get_desc(),
                                     scale,
                                     N,
                                     num_threads);

    // Create primitive descriptor.
    auto attn_pd_f32 = attention::primitive_desc(attn_desc_f32, attn_attr_f32, engine);

    // Create the primitive.
    prim_vec2.push_back(attention(attn_pd_f32));

    // Primitive arguments.
    std::unordered_map<int, memory> attn_args_f32;
    attn_args_f32.insert({ZENDNN_ARG_SRC_0, src_query_mem_f32});
    attn_args_f32.insert({ZENDNN_ARG_SRC_1, src_key_mem_f32});
    attn_args_f32.insert({ZENDNN_ARG_SRC_2, src_value_mem_f32});
    attn_args_f32.insert({ZENDNN_ARG_WEIGHTS, query_weights_mem_f32});
    attn_args_f32.insert({ZENDNN_ARG_WEIGHTS_1, key_weights_mem_f32});
    attn_args_f32.insert({ZENDNN_ARG_WEIGHTS_2, value_weights_mem_f32});
    attn_args_f32.insert({ZENDNN_ARG_BIAS_0, query_bias_mem_f32});
    attn_args_f32.insert({ZENDNN_ARG_BIAS_1, key_bias_mem_f32});
    attn_args_f32.insert({ZENDNN_ARG_BIAS_2, value_bias_mem_f32});
    attn_args_f32.insert({ZENDNN_ARG_MASK, mask_mem_f32});
    attn_args_f32.insert({ZENDNN_ARG_DST, dst_mem_f32_temp});

    // Primitive execution push_back: attention
    prim_args_vec2.push_back(attn_args_f32);

    // F32 Primtive Execution and storing the result

    std::cout <<"\n\nPrimtive fp32 Execution and storing the result\n ";
    auto start2 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<prim_vec2.size(); i++) {
        prim_vec2.at(i).execute(engine_stream, prim_args_vec2.at(i));
        engine_stream.wait();
    }

    float *float_result_f32= static_cast<float *>(dst_mem_f32_temp.get_data_handle());

    // Printing some sample values from BF16 and F32 attention primitive
    std::cout <<"\nBF16 attention primitive computed values : ";
    for (int i=0; i< 20;i++)
        std::cout<<" "<<float_result[i];

    std::cout <<"\nF32 attention primitive computed values  : ";
    for (int i=0; i< 20;i++)
        std::cout<<" "<<float_result_f32[i];

    return 1;
}

int main(int argc, char **argv) {
    int status_code=0;
    if(attention_example(zendnn::engine::kind::cpu, argc, argv) == 1) {
        std::cout <<"\nExample passed on CPU" <<std::endl;
    }
}
