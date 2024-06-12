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

// Embedding bag primitive execute
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

// Matmul primitive execute
void matmul_execute(engine &eng, stream &s, memory &src, memory &weights,
                    const float &alpha,
                    const float &beta,
                    const bool &bias_defined,
                    const int64_t &fuse,
                    memory &dst) {

    primitive_attr matmul_attr;
    post_ops matmul_ops;

    if (beta != 0.0f && !bias_defined) {
        // sets post_ops as add or sum
        matmul_ops.append_sum(beta);
    }
    if (alpha != 1.0f) {
        matmul_attr.set_output_scales(0, {alpha});
    }

    if (fuse == 1) {
        matmul_ops.append_eltwise(1.0f, algorithm::eltwise_relu, 0.f, 0.f);
    }
    if (fuse == 2) {
        matmul_ops.append_eltwise(1.0f, algorithm::eltwise_gelu_tanh, 1.f, 0.f);
    }
    if (fuse == 3) {
        matmul_ops.append_eltwise(1.0f, algorithm::eltwise_gelu_erf, 1.f, 0.f);
    }
    matmul_attr.set_post_ops(matmul_ops);

    matmul::desc matmul_desc(src.get_desc(), weights.get_desc(), dst.get_desc());
    matmul::primitive_desc matmul_pd(matmul_desc,matmul_attr, eng);
    matmul matmul_prim(matmul_pd);
    matmul_prim.execute(s, {{ZENDNN_ARG_SRC, src}, {ZENDNN_ARG_WEIGHTS, weights}, {ZENDNN_ARG_DST, dst}});
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_MLP_layers>"
                  <<" <num_embedding_ops> "<< std::endl;
        return 1;
    }
    int num_mlp_layers = std::stoi(argv[1]);
    int num_embedding_ops  = std::stoi(argv[2]);

    if (num_mlp_layers < 1) {
        std::cerr << "Number of layers must be at least 1" << std::endl;
        return 1;
    }

    // Create engine and stream
    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    //Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    //Embedding bag parameters
    int indices_length = 100; //Input indices length
    int pool_size = 25; //Pooling size
    int batch_size = indices_length/pool_size; // Number of batches
    int embedding_dim = 128; // Embedding dimension
    int num_embeddings = 1000; // Number of embeddings per table
    std::vector <int32_t> eb_scale_grad_by_freq(num_embedding_ops, 0);
    std::vector <int32_t> eb_sparse(num_embedding_ops, 0);
    std::vector <int32_t> eb_per_sample_weights_defined(num_embedding_ops, 0);
    std::vector <int32_t> eb_include_last_offset(num_embedding_ops, 0);
    std::vector <int32_t> eb_padding_idx(num_embedding_ops, -1);
    std::vector <memory>  eb_per_sample_weights_opt(num_embedding_ops);
    std::vector <algorithm> eb_alg(num_embedding_ops, algorithm::embedding_bag_sum);

    //Generate random indices and weights for embedding layers
    std::uniform_int_distribution<> indices_dis(0, num_embeddings-1);
    std::uniform_real_distribution<float> embedding_dis(-1.0f,1.0f);

    //MLP parameters
    std::vector<float> mlp_alpha(num_mlp_layers, 1.f);
    std::vector<float> mlp_beta(num_mlp_layers, 0.f);
    std::vector<memory> mlp_bias(num_mlp_layers);
    std::vector<bool> mlp_bias_defined(num_mlp_layers, 0);
    std::vector<int64_t> mlp_fuse(num_mlp_layers, 1);

    // Generate random input for MLP layers
    std::uniform_int_distribution<> mlp_dis(1, 10);
    std::uniform_real_distribution<float> weights_dis(0.0f, 1.0f);

    // Generate random dimensions for MLP layers
    std::vector<int> mlp_dims(num_mlp_layers + 1);
    for (int i = 0; i < num_mlp_layers + 1; ++i) {
        mlp_dims[i] = mlp_dis(gen);
    }

    // Create zen_memory for MLP input and generate random input
    std::vector<memory> mlp_input_mem;
    mlp_input_mem.push_back(create_memory(memory::desc({1, mlp_dims[0]},
                                          memory::data_type::f32,
                                          memory::format_tag::ab), eng));
    std::vector<float> mlp_input(1*mlp_dims[0]);
    for (int i = 0; i < mlp_dims[0]; ++i) {
        mlp_input[i] = mlp_dis(gen);
    }
    write_to_zendnn_memory(mlp_input.data(), mlp_input_mem[0]);

    // Create zen_memory for MLP weights generate random weights
    std::vector<memory> mlp_weight_mem(num_mlp_layers);

    for (int i = 0; i <num_mlp_layers; ++i) {
        mlp_weight_mem[i] = create_memory(memory::desc({mlp_dims[i], mlp_dims[i+1]},
                                          memory::data_type::f32,
                                          memory::format_tag::ab), eng);
    }
    std::vector<float> mlp_weights;
    for (int i = 0; i < num_mlp_layers; ++i) {
        mlp_weights.resize(mlp_dims[i] * mlp_dims[i + 1]);
        for (auto &w : mlp_weights) {
            w = weights_dis(gen);
        }
        write_to_zendnn_memory(mlp_weights.data(), mlp_weight_mem[i]);
    }

    // Create zen_memory for MLP output
    std::vector<memory> mlp_out_mem(num_mlp_layers);
    for (int i = 0; i < num_mlp_layers; ++i) {
        mlp_out_mem[i] = create_memory(memory::desc({1, mlp_dims[i+1]},
                                       memory::data_type::f32,
                                       memory::format_tag::ab), eng);
    }

    // Create zen_memory for GRP MLP output
    std::vector<memory> grp_mlp_out_mem(num_mlp_layers);
    for (int i = 0; i < num_mlp_layers; ++i) {
        grp_mlp_out_mem[i] = create_memory(memory::desc({1, mlp_dims[i+1]},
                                           memory::data_type::f32,
                                           memory::format_tag::ab), eng);
    }

    // Create zen_memory for Embedding input indices and generate random indices
    std::vector<memory> eb_indices_mem(num_embedding_ops);
    for (int i = 0; i <num_embedding_ops; ++i) {
        eb_indices_mem[i] = create_memory(memory::desc({{indices_length},
            memory::data_type::s32,
            memory::format_tag::a}), eng);
    }
    std::vector<int> indices_data(indices_length);
    for (int i = 0; i < num_embedding_ops; ++i) {
        for (auto &w : indices_data) {
            w = indices_dis(gen);
        }
        write_to_zendnn_memory(indices_data.data(), eb_indices_mem[i]);
    }

    // Create zen_memory for offsets
    std::vector<memory> eb_offsets_mem(num_embedding_ops);
    for (int i = 0; i <num_embedding_ops; ++i) {
        eb_offsets_mem[i] = create_memory(memory::desc({{batch_size},
            memory::data_type::s32,
            memory::format_tag::a}), eng);
    }
    std::vector<int> eb_offset_data(batch_size);
    for (int i = 0; i < num_embedding_ops; ++i) {
        eb_offset_data[0] = 0;
        for (int j = 1; j < batch_size; ++j) {
            eb_offset_data[j]=eb_offset_data[j-1]+pool_size;
        }
        write_to_zendnn_memory(eb_offset_data.data(), eb_offsets_mem[i]);
    }

    //Generate zen_memory for Embedding table and generate random weights
    std::vector<memory> embedding_mem(num_embedding_ops);
    for (int i = 0; i <num_embedding_ops; ++i) {
        embedding_mem[i] = create_memory(memory::desc({{num_embeddings, embedding_dim},
            memory::data_type::f32,
            memory::format_tag::ab}), eng);
    }
    std::vector<float> table_data(num_embeddings*embedding_dim);
    for (int i = 0; i < num_embedding_ops; ++i) {
        for (auto &w : table_data) {
            w = embedding_dis(gen);
        }
        write_to_zendnn_memory(table_data.data(), embedding_mem[i]);
    }

    //Create output table
    std::vector<memory> eb_out_mem(num_embedding_ops);
    for (int i = 0; i <num_embedding_ops; ++i) {
        eb_out_mem[i] = create_memory(memory::desc({{batch_size, embedding_dim},
            memory::data_type::f32,
            memory::format_tag::ab}), eng);
    }

    //Create grp output table
    std::vector<memory> grp_eb_out_mem(num_embedding_ops);
    for (int i = 0; i <num_embedding_ops; ++i) {
        grp_eb_out_mem[i] = create_memory(memory::desc({{batch_size, embedding_dim},
            memory::data_type::f32,
            memory::format_tag::ab}), eng);
    }

    //Execute Embedding bag
    for (int i = 0; i < num_embedding_ops; ++i) {
        embedding_bag_exec(embedding_mem[i],
                           eb_indices_mem[i], eb_offsets_mem[i],
                           eb_scale_grad_by_freq[i], eb_alg[i], eb_sparse[i], eb_per_sample_weights_opt[i],
                           eb_per_sample_weights_defined[i], eb_include_last_offset[i], eb_padding_idx[i],
                           eb_out_mem[i], 1);
    }


    // Execute MLP layers
    for (int i = 0; i < num_mlp_layers; ++i) {
        if (i==0) {
            matmul_execute(eng, s, mlp_input_mem[i], mlp_weight_mem[i], mlp_alpha[i],
                           mlp_beta[i],
                           mlp_bias_defined[i],
                           mlp_fuse[i], mlp_out_mem[i]);

        }
        else {
            matmul_execute(eng, s, mlp_out_mem[i-1], mlp_weight_mem[i], mlp_alpha[i],
                           mlp_beta[i],
                           mlp_bias_defined[i], mlp_fuse[i], mlp_out_mem[i]);
        }
    }

    // Read data from memory object for the final output
    std::vector<float> mlp_result(1*mlp_dims[num_mlp_layers]);
    read_from_zendnn_memory(mlp_result.data(), mlp_out_mem[num_mlp_layers-1]);

    // Execute Grp embedding + MLP layers
    zendnn_custom_op::zendnn_grp_ebag_mlp(embedding_mem, eb_indices_mem,
                                          eb_offsets_mem, eb_scale_grad_by_freq, eb_alg, eb_sparse,
                                          eb_per_sample_weights_opt, eb_per_sample_weights_defined,
                                          eb_include_last_offset, eb_padding_idx, grp_eb_out_mem, mlp_input_mem,
                                          mlp_weight_mem, mlp_bias, mlp_alpha, mlp_beta, mlp_bias_defined, mlp_fuse,
                                          grp_mlp_out_mem,"lib::zendnn_grp_ebag_mlp");

    //Compare embedding bag results
    // Read data from memory object for the final output
    for (int i = 0; i < num_embedding_ops; ++i) {
        std::vector<float> eb_out(batch_size * embedding_dim);
        std::vector<float> grp_eb_out(batch_size * embedding_dim);

        read_from_zendnn_memory(eb_out.data(), eb_out_mem[i]);
        read_from_zendnn_memory(grp_eb_out.data(), grp_eb_out_mem[i]);

        for (int idx=0; idx < batch_size * embedding_dim; idx++) {
            assert(eb_out[idx] == grp_eb_out[idx]);
        }
    }
    std::cout << " Group Embedding Bag Test : Successful " << std::endl;

    // Read data from memory object for the final output of grp Matmul
    std::vector<float> grp_mlp_result(1*mlp_dims[num_mlp_layers]);
    read_from_zendnn_memory(grp_mlp_result.data(),
                            grp_mlp_out_mem[num_mlp_layers-1]);

    //Compare MLP result
    for (int i=0; i<mlp_dims[num_mlp_layers]; i++) {
        assert(mlp_result[i] == grp_mlp_result[i]);
    }
    std::cout << " Group MLP Test : Successful " << std::endl;

    return 0;
}
