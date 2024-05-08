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
#include "zendnn.hpp"
#include "test_utils.hpp"
#include "zendnn_logging.hpp"

using namespace zendnn;

// Create memory object for a given memory descriptor
memory create_memory(const memory::desc &md, engine &eng) {
    return memory(md, eng);
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

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_layers>" << std::endl;
        return 1;
    }

    int num_layers = std::stoi(argv[1]);
    if (num_layers < 1) {
        std::cerr << "Number of layers must be at least 1" << std::endl;
        return 1;
    }

    // Create engine and stream
    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    // Generate random dimensions for layers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10); // Random dimension range: 1 to 10
    std::vector<int> dims(num_layers + 1);
    for (int i = 0; i < num_layers + 1; ++i) {
        dims[i] = dis(gen);
    }
    int batch_size=1;
    // Create memory descriptors for input matrices
    std::vector<memory> input_mem;
    input_mem.push_back(create_memory(memory::desc({batch_size, dims[0]},
                                      memory::data_type::f32,
                                      memory::format_tag::ab), eng));

    // Create memory descriptors for weight matrices
    std::vector<memory::dims> mdims(num_layers);
    std::vector<memory> weight_mem(num_layers);
    for (int i = 0; i <num_layers; ++i) {

        mdims[i] = {dims[i], dims[i+1]};
        weight_mem[i] = create_memory(memory::desc(mdims[i], memory::data_type::f32,
                                      memory::format_tag::ab), eng);
    }

    // Create memory descriptors for output matrices
    std::vector<memory::dims> out_mdims(num_layers);
    std::vector<memory> out_mem(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        out_mdims[i] = {batch_size, dims[i+1]};
        out_mem[i] = create_memory(memory::desc(out_mdims[i], memory::data_type::f32,
                                                memory::format_tag::ab), eng);
    }
    // Generate random input
    // Random weights range: 0.0 to 1.0
    std::uniform_real_distribution<float> dis_weights(0.0f,1.0f);
    std::vector<float> input_data;
    for (int i = 0; i < batch_size*dims[0]; ++i) {
        input_data.resize(batch_size*dims[0]);
        input_data[i] = dis_weights(gen);
    }
    write_to_zendnn_memory(input_data.data(), input_mem[0]);

    // Generate random weights for layers
    std::vector<float> weights_data;
    for (int i = 0; i < num_layers; ++i) {
        weights_data.resize(dims[i] * dims[i + 1]);
        for (auto &w : weights_data) {
            w = dis_weights(gen);
        }
        write_to_zendnn_memory(weights_data.data(), weight_mem[i]);
    }

    std::vector<float> alpha(num_layers, 1.f);
    std::vector<float> beta(num_layers, 0.f);
    std::vector<memory> bias(num_layers);
    std::vector<bool> bias_defined(num_layers, 0);
    std::vector<int64_t> fuse(num_layers, 1);
    // Execute the MLP layers
    for (int i = 0; i < num_layers; ++i) {
        if (i==0) {
            matmul_execute(eng, s, input_mem[i], weight_mem[i], alpha[i], beta[i],
                           bias_defined[i],
                           fuse[i], out_mem[i]);

        }
        else {
            matmul_execute(eng, s, out_mem[i-1], weight_mem[i], alpha[i], beta[i],
                           bias_defined[i], fuse[i], out_mem[i]);
        }
    }
    // Read data from memory object for the final output
    std::vector<float> output(batch_size*dims[num_layers]);
    read_from_zendnn_memory(output.data(), out_mem[num_layers-1]);


    // Create memory descriptors for output matrices for grp Matmul call
    std::vector<memory> out_grp_mem(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        out_mdims[i] = {batch_size, dims[i+1]};
        out_grp_mem[i] = create_memory(memory::desc(out_mdims[i],
                                       memory::data_type::f32,
                                       memory::format_tag::ab), eng);

    }
    /*****************Test for Group Linear MatMul********************/
    zendnn_custom_op::zendnn_grp_mlp(input_mem, weight_mem, bias, alpha, beta,
                                     bias_defined, fuse, out_grp_mem);

    // Read data from memory object for the final output of grp Matmul
    std::vector<float> output_grp(batch_size*dims[num_layers]);
    read_from_zendnn_memory(output_grp.data(), out_grp_mem[num_layers-1]);

    //Compare result
    for (int i=0; i<batch_size*dims[num_layers]; i++) {
        assert(output[i] == output_grp[i]);
    }
    std::cout << " Test Comparison for group Matmul Successful " << std::endl;

    /*****************Test for Group parallel MatMul********************/
    input_mem.push_back(create_memory(memory::desc({batch_size, dims[0]},
                                      memory::data_type::f32,
                                      memory::format_tag::ab), eng));
    input_mem.push_back(create_memory(memory::desc({batch_size, dims[0]},
                                      memory::data_type::f32,
                                      memory::format_tag::ab), eng));

    // Create memory descriptors for weight matrices
    std::vector<memory> weight_mem_parallel(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        weight_mem_parallel[i] = create_memory(memory::desc({dims[0], dims[1]},
                                               memory::data_type::f32,
                                               memory::format_tag::ab), eng);
    }

    // Generate random input
    // Random weights range: 0.0 to 1.0
    for (int j = 0; j< num_layers; ++j) {
        for (int i = 0; i < batch_size*dims[0]; ++i) {
            input_data.resize(batch_size*dims[0]);
            input_data[i] = dis_weights(gen);
        }
        write_to_zendnn_memory(input_data.data(), input_mem[j]);
    }

    // Generate random weights for layers
    for (int i = 0; i < num_layers; ++i) {
        weights_data.resize(dims[0] * dims[1]);
        for (auto &w : weights_data) {
            w = dis_weights(gen);
        }
        write_to_zendnn_memory(weights_data.data(), weight_mem_parallel[i]);
    }


    // Create memory descriptors for output matrices
    std::vector<memory> out_mem_parallel(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        out_mem_parallel[i] = create_memory(memory::desc({batch_size, dims[1]},
                                            memory::data_type::f32,
                                            memory::format_tag::ab), eng);
    }

    // Execute the MLP layers
    for (int i = 0; i < num_layers; ++i) {
        matmul_execute(eng, s, input_mem[i], weight_mem_parallel[i], alpha[i], beta[i],
                       bias_defined[i], fuse[i], out_mem_parallel[i]);
    }
    std::vector<float> output_parallel(batch_size*dims[1]);
    read_from_zendnn_memory(output_parallel.data(), out_mem_parallel[1]);


    zendnn_custom_op::zendnn_grp_mlp(input_mem, weight_mem_parallel, bias, alpha,
                                     beta,
                                     bias_defined, fuse, out_mem_parallel);

    std::vector<float> output_grp_parallel(batch_size*dims[1]);
    read_from_zendnn_memory(output_grp_parallel.data(), out_mem_parallel[1]);

    //Compare result
    for (int i=0; i<batch_size*dims[1]; i++) {
        assert(output_parallel[i] == output_grp_parallel[i]);
    }

    std::cout << " Test Comparison for group parallel Matmul Successful " <<
              std::endl;
    return 0;
}
