/*******************************************************************************
* Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "zendnn.hpp"
#include "test_utils.hpp"

using namespace zendnn;

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
    int pool_size = 25; //Pooling size
    int batch_size = input_length/pool_size; // Number of batches
    int embedding_dim = 128; // Embedding dimension
    int num_embeddings = 1000; // Number of embeddings per table

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_input(0, num_embeddings-1);
    std::uniform_real_distribution<float> dis_table(-1.0f,1.0f);

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
    std::vector<float> table_data(num_embeddings*embedding_dim);
    for (int i = 0; i < num_ops; ++i) {
        for (auto &w : table_data) {
            w = dis_table(gen);
        }
        write_to_zendnn_memory(table_data.data(), embedding_mem[i]);
    }

    //Create output table
    std::vector<memory> out_mem(num_ops);
    for (int i = 0; i <num_ops; ++i) {
        out_mem[i] = create_memory(memory::desc({{batch_size, embedding_dim},
            memory::data_type::f32,
            memory::format_tag::ab}), eng);
    }

    //Create output table for custom grp embedding bag execution
    std::vector<memory> grp_out_mem(num_ops);
    for (int i = 0; i <num_ops; ++i) {
        grp_out_mem[i] = create_memory(memory::desc({{batch_size, embedding_dim},
            memory::data_type::f32,
            memory::format_tag::ab}), eng);
    }

    std::vector <int32_t> scale_grad_by_freq(num_ops, 0);
    std::vector <int32_t> sparse(num_ops, 0);
    std::vector <int32_t> per_sample_weights_defined(num_ops, 0);
    std::vector <int32_t> include_last_offset(num_ops, 0);
    std::vector <int32_t> padding_idx(num_ops, -1);
    std::vector <memory> per_sample_weights_opt(num_ops);
    std::vector <algorithm> alg(num_ops, algorithm::embedding_bag_sum);

    //Execute Embedding bag
    for (int i = 0; i < num_ops; ++i) {
        embedding_bag_exec(embedding_mem[i],
                           input_mem[i], offset_mem[i],
                           scale_grad_by_freq[i], alg[i], sparse[i], per_sample_weights_opt[i],
                           per_sample_weights_defined[i], include_last_offset[i], padding_idx[i],
                           out_mem[i], 1);
    }

    //Execute Custom Group embedding bag
    zendnn_custom_op::zendnn_grp_embedding_bag(embedding_mem,
            input_mem, offset_mem,
            scale_grad_by_freq, alg, sparse, per_sample_weights_opt,
            per_sample_weights_defined, include_last_offset, padding_idx, grp_out_mem);

    //Compare results

    // Read data from memory object for the final output
    for (int i = 0; i < num_ops; ++i) {
        std::vector<float> ebag_output(batch_size * embedding_dim);
        std::vector<float> grp_ebag_output(batch_size * embedding_dim);

        read_from_zendnn_memory(ebag_output.data(), out_mem[i]);
        read_from_zendnn_memory(grp_ebag_output.data(), grp_out_mem[i]);

        for (int idx=0; idx < batch_size * embedding_dim; idx++) {
            assert(ebag_output[idx] == grp_ebag_output[idx]);
        }
    }
    std::cout << " Custom Grp EBag test Comparison Successful " << std::endl;

    return 0;
}
