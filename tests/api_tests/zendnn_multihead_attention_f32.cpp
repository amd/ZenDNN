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
 *  2. create user memory : input(q,k,v), weights(q,k,v), bias(q,k,v), mask
 *  3. create memory descriptor
 *  4. create attention descriptor
 *  5. create attention primitive descriptor
 *  6. create attentiom primitive
 *  7. execute the attention primitive
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

int attention_example(zendnn::engine::kind engine_kind) {

    // Create execution zendnn::engine.
    zendnn::engine engine(engine_kind, 0);

    // Create zendnn::stream.
    zendnn::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim B = 3, // batch size
                      S = 5, // sequence length
                      Id = 24, // Input dimension
                      N = 6, // number of heads
                      H = 4; // Hidden dimension/Size of each head
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

    // Allocate buffers.
    std::vector<float> src_query_data(product(src_query_dims));
    std::vector<float> src_key_data(product(src_key_dims));
    std::vector<float> src_value_data(product(src_value_dims));
    std::vector<float> weight_query_data(product(weight_query_dims));
    std::vector<float> weight_key_data(product(weight_key_dims));
    std::vector<float> weight_value_data(product(weight_value_dims));
    std::vector<float> bias_query_data(product(bias_query_dims));
    std::vector<float> bias_key_data(product(bias_key_dims));
    std::vector<float> bias_value_data(product(bias_value_dims));

    std::vector<float> mask_data(product(mask_dims));
    std::generate(mask_data.begin(), mask_data.end(), []() {
            static int i = 0;
            return 1.0f;
        });

    // Initialize src, weights, and bias tensors.
    std::generate(src_query_data.begin(), src_query_data.end(), []() {
        static int i = 0;
        return 1.0f;
    });
    std::generate(src_key_data.begin(), src_key_data.end(), []() {
        static int i = 0;
        return 1.0f;
    });
    std::generate(src_value_data.begin(), src_value_data.end(), []() {
        static int i = 0;
        return 1.0f;
    });

    std::generate(weight_query_data.begin(), weight_query_data.end(), []() {
        static int i = 0;
        return 1.0f;
    });
    std::generate(weight_key_data.begin(), weight_key_data.end(), []() {
        static int i = 0;
        return 1.0f;
    });
    std::generate(weight_value_data.begin(), weight_value_data.end(), []() {
        static int i = 0;
        return 1.0f;
    });

    std::generate(bias_query_data.begin(), bias_query_data.end(), []() {
        static int i = 0;
        return 1.0f;
    });
    std::generate(bias_key_data.begin(), bias_key_data.end(), []() {
        static int i = 0;
        return 1.0f;
    });
    std::generate(bias_value_data.begin(), bias_value_data.end(), []() {
        static int i = 0;
        return 1.0f;
    });

    std::vector<primitive> prim_vec;
    std::vector<std::unordered_map<int, memory>> prim_args_vec;

    /*####################################### Attention1 ###############################################################*/

    std::cout <<"\n\n Attention_1 \n\n" ;
    // Create memory objects for tensor data (src, query_weights, key_weights, value_weights, query_bias, key_bias, value_bias, mask, dst)
    auto src_query_mem = memory({src_query_dims, dt::f32, tag::abc}, engine);
    auto src_key_mem = memory({src_key_dims, dt::f32, tag::abc}, engine);
    auto src_value_mem = memory({src_value_dims, dt::f32, tag::abc}, engine);
    auto query_weights_mem = memory({weight_query_dims, dt::f32, tag::ab}, engine);
    auto key_weights_mem = memory({weight_key_dims, dt::f32, tag::ab}, engine);
    auto value_weights_mem = memory({weight_value_dims, dt::f32, tag::ab}, engine);
    auto query_bias_mem = memory({bias_query_dims, dt::f32, tag::a}, engine);
    auto key_bias_mem = memory({bias_key_dims, dt::f32, tag::a}, engine);
    auto value_bias_mem = memory({bias_value_dims, dt::f32, tag::a}, engine);
    auto mask_mem = memory({mask_dims, dt::f32, tag::ab}, engine);
    auto dst_mem = memory({dst_dims, dt::f32, tag::abc}, engine);

    // Write data to memory object's handle.
    write_to_zendnn_memory(src_query_data.data(), src_query_mem);
    write_to_zendnn_memory(src_key_data.data(), src_key_mem);
    write_to_zendnn_memory(src_value_data.data(), src_value_mem);
    write_to_zendnn_memory(weight_query_data.data(), query_weights_mem);
    write_to_zendnn_memory(weight_key_data.data(), key_weights_mem);
    write_to_zendnn_memory(weight_value_data.data(), value_weights_mem);
    write_to_zendnn_memory(bias_query_data.data(), query_bias_mem);
    write_to_zendnn_memory(bias_key_data.data(), key_bias_mem);
    write_to_zendnn_memory(bias_value_data.data(), value_bias_mem);
    write_to_zendnn_memory(mask_data.data(), mask_mem);

    /* Experimenting scratch-pad memory */
    zendnn::primitive_attr attn_attr;

    // Create operation descriptor.
    auto attn_desc = attention::desc(prop_kind::forward_inference,
                                     algorithm::multihead_attention,
                                     src_query_mem.get_desc(),
                                     src_key_mem.get_desc(),
                                     src_value_mem.get_desc(),
                                     query_weights_mem.get_desc(),
                                     key_weights_mem.get_desc(),
                                     value_weights_mem.get_desc(),
                                     query_bias_mem.get_desc(),
                                     key_bias_mem.get_desc(),
                                     value_bias_mem.get_desc(),
                                     mask_mem.get_desc(),
                                     dst_mem.get_desc(),
                                     scale,
                                     N,
                                     num_threads);

    // Create primitive descriptor.
    auto attn_pd = attention::primitive_desc(attn_desc, attn_attr, engine);

    // Create the primitive.
    prim_vec.push_back(attention(attn_pd));

    // Primitive arguments.
    std::unordered_map<int, memory> attn_args;
    attn_args.insert({ZENDNN_ARG_SRC_0, src_query_mem});
    attn_args.insert({ZENDNN_ARG_SRC_1, src_key_mem});
    attn_args.insert({ZENDNN_ARG_SRC_2, src_value_mem});
    attn_args.insert({ZENDNN_ARG_WEIGHTS, query_weights_mem});
    attn_args.insert({ZENDNN_ARG_WEIGHTS_1, key_weights_mem});
    attn_args.insert({ZENDNN_ARG_WEIGHTS_2, value_weights_mem});
    attn_args.insert({ZENDNN_ARG_BIAS_0, query_bias_mem});
    attn_args.insert({ZENDNN_ARG_BIAS_1, key_bias_mem});
    attn_args.insert({ZENDNN_ARG_BIAS_2, value_bias_mem});
    attn_args.insert({ZENDNN_ARG_MASK, mask_mem});
    attn_args.insert({ZENDNN_ARG_DST, dst_mem});

    // Primitive execution push_back: attention
    prim_args_vec.push_back(attn_args);

  /*######################################### Primtive Execution and storing the result #############################################################*/

  std::cout <<"\n\nPrimtive Execution and storing the result\n ";
  auto start = std::chrono::high_resolution_clock::now();
  for(int i=0; i<prim_vec.size(); i++) {
      prim_vec.at(i).execute(engine_stream, prim_args_vec.at(i));
      engine_stream.wait();
  }

  return 1;
}

int main(int argc, char **argv) {
    int status_code=0;
    if(attention_example(zendnn::engine::kind::cpu) == 1) {
        std::cout <<"Example passed on CPU" <<std::endl;
    }
}