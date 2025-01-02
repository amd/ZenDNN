/*******************************************************************************
* Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
*******************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include "zendnn.hpp"
#include "test_utils.hpp"
#include <cmath>
#include <fbgemm/Fbgemm.h>
#include <fbgemm/QuantUtils.h>
#include "fbgemm/FbgemmConvert.h"

#define BF16_ENABLE 0
using namespace zendnn;

// Function to generate random embedding table
std::vector<float> generateRandomEmbeddingTable(int num_rows,
        int embedding_dim) {

    std::vector<float> embedding_table(num_rows * embedding_dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 2.0f);
    for (auto &value : embedding_table) {
        //Generate random float between 1.0 and 2.0
        value = dis(gen);
    }
    return embedding_table;
}

void generateFusedEmbeddingTableInt4(
    const std::vector<float> &embedding_table,
    int num_rows,
    int embedding_dim,
    std::vector<uint8_t> &fused_embedding_table,
    bool scale_bias_last = true) {

    // For int4, 2 elements per byte
    int num_elements_per_byte = 2;
    int block_size = embedding_dim;

    // Calculate fused row size
    size_t row_size = (embedding_dim + num_elements_per_byte - 1) /
                      num_elements_per_byte +
                      2 * sizeof(fbgemm::float16);  // Scale and bias

    // Resize the fused embedding table
    fused_embedding_table.resize(num_rows * row_size);

    // Pointer to write fused rows
    uint8_t *fused_ptr = fused_embedding_table.data();

    for (int i = 0; i < num_rows; ++i) {
        const float *row = &embedding_table[ i * embedding_dim];

        // Calculate scale and bias for the row
        float min_val = *std::min_element(row, row + embedding_dim);
        float max_val = *std::max_element(row, row + embedding_dim);
        float scale = (max_val - min_val) / 15.0f;  // 15 = 2^4 - 1 (int4 range)
        float bias = min_val;

        // Quantize weights to int4
        int bit_position =0;
        std::vector<uint8_t> quantized_row((embedding_dim + 1) / num_elements_per_byte,
                                           0);

        //INT4 packing: 1st element = LSB, 2nd element = MSB
        for (int j = 0; j < embedding_dim; ++j) {
            int quantized_value = std::round((row[j] - bias) / scale);
            quantized_value = std::min(15, std::max(0,
                                                    quantized_value));  // Clamp to [0, 15]

            int value = quantized_value & 0xF;
            int byte_index = bit_position / 8;
            int bit_offset = bit_position % 8;

            quantized_row[byte_index] |= (value << bit_offset);
            if (bit_offset >4) {
                quantized_row[byte_index+1] |= (value >> (8- bit_offset));
            }
            bit_position += 4;
        }

        // Write to fused table
        if (scale_bias_last) {
            // Write quantized weights first
            std::memcpy(fused_ptr, quantized_row.data(), quantized_row.size());
            fused_ptr += quantized_row.size();

            // Write scale and bias (fp16)
            fbgemm::FloatToFloat16_ref(&scale,
                                       reinterpret_cast<fbgemm::float16 *>(fused_ptr), 1,true);
            fused_ptr += sizeof(fbgemm::float16);
            fbgemm::FloatToFloat16_ref(&bias,
                                       reinterpret_cast<fbgemm::float16 *>(fused_ptr), 1, true);
            fused_ptr += sizeof(fbgemm::float16);
        }
        else {
            // Write scale and bias (fp16) first
            fbgemm::FloatToFloat16_ref(&scale,
                                       reinterpret_cast<fbgemm::float16 *>(fused_ptr), 1, true);
            fused_ptr += sizeof(fbgemm::float16);
            fbgemm::FloatToFloat16_ref(&bias,
                                       reinterpret_cast<fbgemm::float16 *>(fused_ptr), 1, true);
            fused_ptr += sizeof(fbgemm::float16);

            // Write quantized weights
            std::memcpy(fused_ptr, quantized_row.data(), quantized_row.size());
            fused_ptr += quantized_row.size();
        }
    }
}

// Create memory object for a given memory descriptor
memory create_memory(const memory::desc &md, engine &eng) {
    return memory(md, eng);
}

void embedding_bag_exec(
    const memory &z_input, const memory &z_indices, const memory &z_offsets,
    const int32_t &scale_grad_by_freq,
    const algorithm &z_algorithm, const int32_t &sparse,
    const memory &z_per_sample_weights,
    const int32_t &z_per_sample_weights_defined,
    const int32_t &include_last_offset, const int32_t &padding_idx, memory &z_dst,
    unsigned int op_num_threads) {
    engine eng;
    stream s;
    eng=engine(engine::kind::cpu, 0);
    s=stream(eng);

    embedding_bag::desc pdesc;
    embedding_bag::primitive_desc pd;
    
    if (z_per_sample_weights_defined) {
        // declare embedding bag primitive
        pdesc = embedding_bag::desc(prop_kind::forward_inference,
                                    z_algorithm,
                                    op_num_threads,
                                    z_input.get_desc(),
                                    z_indices.get_desc(),
                                    z_offsets.get_desc(),
                                    z_per_sample_weights.get_desc(),
                                    z_dst.get_desc(),
                                    padding_idx);

        pd = embedding_bag::primitive_desc(pdesc, eng);
        embedding_bag(pd).execute(s, {{ZENDNN_ARG_SRC_0, z_input},
            {ZENDNN_ARG_SRC_1, z_indices},
            {ZENDNN_ARG_SRC_2, z_offsets},
            {ZENDNN_ARG_SRC_3, z_per_sample_weights},
            {ZENDNN_ARG_DST, z_dst}
        });
    }
    else {
        // declare embedding bag primitive
        pdesc = embedding_bag::desc(prop_kind::forward_inference,
                                    z_algorithm,
                                    op_num_threads,
                                    z_input.get_desc(),
                                    z_indices.get_desc(),
                                    z_offsets.get_desc(),
                                    z_dst.get_desc(),
                                    padding_idx);

        pd = embedding_bag::primitive_desc(pdesc, eng);
        embedding_bag(pd).execute(s, {{ZENDNN_ARG_SRC_0, z_input},
            {ZENDNN_ARG_SRC_1, z_indices},
            {ZENDNN_ARG_SRC_2, z_offsets},
            {ZENDNN_ARG_DST, z_dst}
        });
    }
}

int main(int argc, char **argv) {
    // Check if batch size is provided as a command line argument
    if (argc != 2) {
        std::cerr << "Usage: ./grp_embedding_bag_test <num_ops>" << std::endl;
        return 1;
    }

    // Get batch size from command line argument
    int num_ops  = std::stoi(argv[1]);

    // Define parameters
    int input_length = 100; //Input indices length
    int pool_size = 2; //Pooling size
    int batch_size = input_length/pool_size; // Number of batches
    int embedding_dim = 64;// Embedding dimension
    int num_embeddings = 4; // Number of embeddings per table

    //Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_input(0, num_embeddings-1);

    // Create engine and stream
    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    //Generate input indices
    std::vector<memory> input_mem(num_ops);
    for (int i = 0; i <num_ops; ++i) {
        input_mem[i] = create_memory(memory::desc({{input_length},
            memory::data_type::s32,
            memory::format_tag::a}), eng);
    }

    std::vector<int> input_data(input_length);
    for (int i = 0; i < num_ops; ++i) {
        for (auto &w : input_data) {
            w = dis_input(gen);
        }
        write_to_zendnn_memory(input_data.data(), input_mem[i]);
    }

    // Generate random offset
    std::vector<memory> offset_mem(num_ops);
    for (int i = 0; i <num_ops; ++i) {
        offset_mem[i] = create_memory(memory::desc({{batch_size},
            memory::data_type::s32,
            memory::format_tag::a}), eng);
    }

    std::vector<int> offset_data(batch_size);
    for (int i = 0; i < num_ops; ++i) {
        offset_data[0] = 0;
        for (int j = 1; j < batch_size; ++j) {
            offset_data[j]=offset_data[j-1]+pool_size;
        }
        write_to_zendnn_memory(offset_data.data(), offset_mem[i]);
    }

    //Generate embedding table
    std::vector<memory> embedding_mem(num_ops);
    for (int i = 0; i <num_ops; ++i) {
        embedding_mem[i] = create_memory(memory::desc({{num_embeddings, embedding_dim},
            memory::data_type::f32,
            memory::format_tag::ab}), eng);
    }
    size_t fused_embedding_dim = (embedding_dim + 2 - 1) / 2 + 2 * sizeof(
                                     fbgemm::float16);
    int num_int4_elem = 2*fused_embedding_dim;

    std::vector<memory> embedding_mem_int4(num_ops);
    for (int i = 0; i <num_ops; ++i) {
        embedding_mem_int4[i] = create_memory(memory::desc({{num_embeddings, num_int4_elem},
            memory::data_type::s4,
            memory::format_tag::ab}), eng);
    }

    // Create multiple embedding tables
    for (int t = 0; t < num_ops; ++t) {
        // Generate random embedding table
        std::vector<float> embedding_table = generateRandomEmbeddingTable(
                num_embeddings, embedding_dim);
        std::vector<uint8_t> fused_embedding_table;

        // Generate fused embedding table with int4 weights
        generateFusedEmbeddingTableInt4(embedding_table, num_embeddings, embedding_dim,
                                        fused_embedding_table);
        write_to_zendnn_memory(embedding_table.data(), embedding_mem[t]);
        write_to_zendnn_memory(fused_embedding_table.data(), embedding_mem_int4[t]);

    }

    //Create f32 output table
    std::vector<memory> out_mem(num_ops);
    for (int i = 0; i <num_ops; ++i) {
        out_mem[i] = create_memory(memory::desc({{batch_size, embedding_dim},
            memory::data_type::f32,
            memory::format_tag::ab}), eng);
    }

    std::vector<memory> out_mem_bf16(num_ops);
    for (int i = 0; i <num_ops; ++i) {
        out_mem_bf16[i] = create_memory(memory::desc({{batch_size, embedding_dim},
            memory::data_type::bf16,
            memory::format_tag::ab}), eng);
    }

    //Create f32 output table for custom grp embedding bag execution
    std::vector<memory> grp_out_mem(num_ops);
    for (int i = 0; i <num_ops; ++i) {
        grp_out_mem[i] = create_memory(memory::desc({{batch_size, embedding_dim},
            memory::data_type::f32,
            memory::format_tag::ab}), eng);
    }

    //Create f32 output table for custom grp embedding bag execution
    std::vector<memory> grp_out_mem_bf16(num_ops);
    for (int i = 0; i <num_ops; ++i) {
        grp_out_mem_bf16[i] = create_memory(memory::desc({{batch_size, embedding_dim},
            memory::data_type::bf16,
            memory::format_tag::ab}), eng);
    }

    std::vector <int32_t> scale_grad_by_freq(num_ops, 0);
    std::vector <int32_t> sparse(num_ops, 0);
    std::vector <int32_t> per_sample_weights_defined(num_ops, 0);
    std::vector <int32_t> include_last_offset(num_ops, 0);
    std::vector <int32_t> padding_idx(num_ops, -1);
    std::vector <memory> per_sample_weights_opt(num_ops);
    std::vector <algorithm> alg(num_ops, algorithm::embedding_bag_sum);

    //optional weights, can be null for non-weighted sum
    for (int i = 0; i <num_ops; ++i) {
        per_sample_weights_opt[i] = create_memory(memory::desc({{input_length, embedding_dim},
            memory::data_type::f32,
            memory::format_tag::ab}), eng);
    }
    std::vector<float> weights(input_length*embedding_dim);
    for (int i = 0; i < num_ops; ++i) {
        for (auto &w : weights) {
            w = 2.0f;
        }
        write_to_zendnn_memory(weights.data(), per_sample_weights_opt[i]);
    }

    //Execute Embedding bag
    for (int i = 0; i < num_ops; ++i) {
        embedding_bag_exec(embedding_mem[i],
                           input_mem[i], offset_mem[i],
                           scale_grad_by_freq[i], alg[i], sparse[i], per_sample_weights_opt[i],
                           per_sample_weights_defined[i], include_last_offset[i], padding_idx[i],
                           out_mem[i], 1);
    }

    //Execute Custom Group embedding bag
#if BF16_ENABLE
    zendnn_custom_op::zendnn_grp_embedding_bag(embedding_mem_int4,
            input_mem, offset_mem,
            scale_grad_by_freq, alg, sparse, per_sample_weights_opt,
            per_sample_weights_defined, include_last_offset, padding_idx, grp_out_mem_bf16,
            "zendnn_grp_embedding_bag");
#else
    zendnn_custom_op::zendnn_grp_embedding_bag(embedding_mem_int4,
            input_mem, offset_mem,
            scale_grad_by_freq, alg, sparse, per_sample_weights_opt,
            per_sample_weights_defined, include_last_offset, padding_idx, grp_out_mem,
            "zendnn_grp_embedding_bag");
#endif

    //Compare results
    for (int i = 0; i < num_ops; ++i) {
        std::vector<float> ebag_output(batch_size * embedding_dim);
        std::vector<float> grp_ebag_output(batch_size * embedding_dim);

#if BF16_ENABLE
        reorder(grp_out_mem_bf16[i], grp_out_mem[i]).execute(s,grp_out_mem_bf16[i],
                grp_out_mem[i]);
#endif
        read_from_zendnn_memory(ebag_output.data(), out_mem[i]);
        read_from_zendnn_memory(grp_ebag_output.data(), grp_out_mem[i]);

        for (int idx=0; idx < batch_size * embedding_dim; idx++) {
            assert((ebag_output[idx] - grp_ebag_output[idx])<= 0.1 ||
                   (ebag_output[idx] - grp_ebag_output[idx])<= -0.1);
        }
    }
    std::cout << " Custom Grp EBag test Comparison Successful " << std::endl;

    return 0;
}

