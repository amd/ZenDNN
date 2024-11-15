
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
    int b,n,s,h,id,l;
    if(argc >= 2)
    {
	b = stoi(argv[1]);
	s = stoi(argv[2]);
	n = stoi(argv[3]);
	h = stoi(argv[4]);
	id = stoi(argv[3]) * stoi(argv[4]);
    l = stoi(argv[5]);
    }
    else
    {
        b = 2;
	    s = 3;
	    n = 3;
	    h = 3;
	    id = n * h;
        l = 10;
    }
    // Tensor dimensions.
    const memory::dim B = b, // batch size
                      S = s, // sequence length
                      Id = id, // Input dimension
                      N = n, // number of heads
                      H = h, // Hidden dimension/Size of each head
                      L = l; // KV Cache length. L=S for encoder models

    const uint32_t num_threads = 64; // number of threads
    memory::dims src_query_dims = {B, N, S, H};
    memory::dims src_key_dims = {B, N, L, H};
    memory::dims src_value_dims = {B, N, L, H};
    memory::dims mask_dims = {B, 1, S, L};
    memory::dims dst_dims = {B, N, S, H};

    //We need BxNxSxS + 4*(BxNxSxH) number of element for scratchpad memory, for all computations.
    std::cout <<"\nSRC size: " <<product(src_query_dims);
    std::cout <<"\nMask size: " <<product(mask_dims);
    std::cout <<"\nNumber of heads: " <<N;
    std::cout <<"\nDST size: " <<product(dst_dims);
    // Allocate temporary float buffers to initialize the values.
    std::vector<float> src_query_data_f32(product(src_query_dims));
    std::vector<float> src_key_data_f32(product(src_key_dims));
    std::vector<float> src_value_data_f32(product(src_value_dims));
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
    std::vector<primitive> prim_vec;
    std::vector<std::unordered_map<int, memory>> prim_args_vec;

    std::cout <<"\n\n BF16 Attention_1 \n\n" ;
    // Create Temporary float memory objects for tensor data
    // (src, query_weights, key_weights, value_weights, query_bias, key_bias, value_bias, mask, dst)
    auto src_query_mem_f32 = memory({src_query_dims, dt::f32, tag::abcd}, engine);
    auto src_key_mem_f32 = memory({src_key_dims, dt::f32, tag::abcd}, engine);
    auto src_value_mem_f32 = memory({src_value_dims, dt::f32, tag::abcd}, engine);
    auto mask_mem_f32 = memory({mask_dims, dt::f32, tag::abcd}, engine);
    auto dst_mem_f32 = memory({dst_dims, dt::f32, tag::abcd}, engine);
    //To compare outputs from BF16 and F32 attention
    auto dst_mem_f32_temp = memory({dst_dims, dt::f32, tag::abcd}, engine);
    // Create BF16 memory objects for tensor data (src, query_weights, key_weights, value_weights, dst)
    // Not required for mask and bias tensors as bf16 attention expects mask and bias tensors in float
    auto src_query_mem_bf16 = memory({src_query_dims, dt::bf16, tag::abcd}, engine);
    auto src_key_mem_bf16 = memory({src_key_dims, dt::bf16, tag::abcd}, engine);
    auto src_value_mem_bf16 = memory({src_value_dims, dt::bf16, tag::abcd}, engine);

    auto dst_mem_bf16 = memory({dst_dims, dt::bf16, tag::abcd}, engine);
    auto mask_mem_bf16 = memory({mask_dims, dt::bf16, tag::abcd}, engine);

    auto dst_mem_bf16f32 = memory({dst_dims, dt::f32, tag::abcd}, engine);

    // Write float data to float memory object's handle.
    write_to_zendnn_memory(src_query_data_f32.data(), src_query_mem_f32);
    write_to_zendnn_memory(src_key_data_f32.data(), src_key_mem_f32);
    write_to_zendnn_memory(src_value_data_f32.data(), src_value_mem_f32);
    write_to_zendnn_memory(mask_data_f32.data(), mask_mem_f32);

    zendnn::reorder(src_query_mem_f32, src_query_mem_bf16).execute(engine_stream, src_query_mem_f32, src_query_mem_bf16);
    zendnn::reorder(src_key_mem_f32, src_key_mem_bf16).execute(engine_stream, src_key_mem_f32, src_key_mem_bf16);
    zendnn::reorder(src_value_mem_f32, src_value_mem_bf16).execute(engine_stream, src_value_mem_f32, src_value_mem_bf16);
    zendnn::reorder(mask_mem_f32, mask_mem_bf16).execute(engine_stream, mask_mem_f32, mask_mem_bf16);

    // BF16 SDPA execution -----------------------------


    // BF16 SDPA output BF16
    zendnn_custom_op::zendnn_sdpa_attention(
                src_query_mem_bf16, src_key_mem_bf16,
                src_value_mem_bf16,
                mask_mem_bf16,
                dst_mem_bf16
                );

    zendnn::reorder(dst_mem_bf16, dst_mem_f32).execute(engine_stream, dst_mem_bf16, dst_mem_f32);
    float *float_result_bf16= static_cast<float *>(dst_mem_f32.get_data_handle());

    //BF16 SDPA output F32
    zendnn_custom_op::zendnn_sdpa_attention(
                src_query_mem_bf16, src_key_mem_bf16,
                src_value_mem_bf16,
                mask_mem_bf16,
                dst_mem_bf16f32
                );

    float *float_result_bf16f32= static_cast<float *>(dst_mem_bf16f32.get_data_handle());

    // F32 SDPA execution  -----------------------------

    zendnn_custom_op::zendnn_sdpa_attention(
                src_query_mem_f32, src_key_mem_f32,
                src_value_mem_f32,
                mask_mem_f32,
                dst_mem_f32_temp
                );

    float *float_result_f32= static_cast<float *>(dst_mem_f32_temp.get_data_handle());
    // Printing some sample values from BF16 and F32 attention primitive
    std::cout <<"\nBF16 attention primitive computed values with BF16 output: ";
    for (int i=0; i< 10;i++)
        std::cout<<" "<<float_result_bf16[i];

    std::cout <<"\nBF16 attention primitive computed values with F32 output : ";
    for (int i=0; i< 10;i++)
        std::cout<<" "<<float_result_bf16f32[i];

    std::cout <<"\nF32 attention primitive computed values                  : ";
    for (int i=0; i< 10;i++)
        std::cout<<" "<<float_result_f32[i];
    return 1;
}
int main(int argc, char **argv) {
    int status_code=0;
    if(attention_example(zendnn::engine::kind::cpu, argc, argv) == 1) {
        std::cout <<"\nExample passed on CPU" <<std::endl;
    }
}
