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

#define BF16_ENABLE 0
#define WEIGHT_CACHING 0

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
                    const std::vector<int64_t> &z_post_op_ids,
                    const std::vector<memory> &z_post_op_buffers,
                    memory &dst) {

    primitive_attr matmul_attr;
    post_ops po;

    int post_op_idx = 0;
    if (beta != 0.0f && !bias_defined) {
        // sets post_ops as add or sum
        post_op_idx++;
        po.append_sum(beta);
    }
    if (alpha != 1.0f) {
        matmul_attr.set_output_scales(0, {alpha});
    }

    int post_op_ids_size = z_post_op_ids.size();
    int post_op_buffers_size = z_post_op_buffers.size();
    std::unordered_map<int, memory> execute_args;
    int post_op_buffer_idx = 0;
    for (int i = 0; i < post_op_ids_size; i++) {
        int arg_position;
        // set the post-ops or fusion-ops;
        switch (z_post_op_ids[i]) {
        case 1:
            po.append_eltwise(1.0f, algorithm::eltwise_relu, 0.f, 0.f);
            break;
        case 2:
            po.append_eltwise(1.0f, algorithm::eltwise_gelu_tanh, 1.f, 0.f);
            break;
        case 3:
            po.append_eltwise(1.0f, algorithm::eltwise_gelu_erf, 1.f, 0.f);
            break;
        case 4:
            po.append_eltwise(1.0f, algorithm::eltwise_swish, 1.f, 0.f);
            break;
        case 5:
            po.append_binary(algorithm::binary_mul,
                             z_post_op_buffers[post_op_buffer_idx].get_desc());
            arg_position = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(post_op_idx) | ZENDNN_ARG_SRC_1;
            execute_args.insert(
            {arg_position, z_post_op_buffers[post_op_buffer_idx]});
            post_op_buffer_idx++;
            break;
        case 6:
            po.append_binary(algorithm::binary_add,
                             z_post_op_buffers[post_op_buffer_idx].get_desc());
            arg_position = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(post_op_idx) | ZENDNN_ARG_SRC_1;
            execute_args.insert(
            {arg_position, z_post_op_buffers[post_op_buffer_idx]});
            post_op_buffer_idx++;
            break;
        default:
            break;
        }
        post_op_idx++;
    }

    matmul_attr.set_post_ops(po);

    matmul::desc pdesc = matmul::desc(src.get_desc(), weights.get_desc(),
                                      dst.get_desc());

    matmul::primitive_desc pd =
        matmul::primitive_desc(pdesc, matmul_attr, eng);

    execute_args.insert({ZENDNN_ARG_SRC, src});
    execute_args.insert({ZENDNN_ARG_WEIGHTS, weights});
    execute_args.insert({ZENDNN_ARG_DST, dst});
    matmul(pd).execute(eng, execute_args);
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
#if WEIGHT_CACHING
    std::uniform_int_distribution<> dis(4096, 16384);
#else
    std::uniform_int_distribution<> dis(1, 10);
#endif
    std::vector<int> dims(num_layers + 1);
    for (int i = 0; i < num_layers + 1; ++i) {
        dims[i] = dis(gen);
    }
    int batch_size=dis(gen);

    // Create FP32 memory descriptors and memory objects for input, weight, output matrices
    std::vector<memory> input_mem;
    std::vector<memory> weight_mem(num_layers);
    std::vector<memory> out_mem(num_layers);
    std::vector<memory> add_mem(num_layers);

    input_mem.push_back(create_memory(memory::desc({batch_size, dims[0]},
                                      memory::data_type::f32,
                                      memory::format_tag::ab), eng));

    for (int i = 0; i <num_layers; ++i) {
        weight_mem[i] = create_memory(memory::desc({dims[i], dims[i+1]},
                                      memory::data_type::f32,
                                      memory::format_tag::ab), eng);
        out_mem[i] = create_memory(memory::desc({batch_size, dims[i+1]},
                                                memory::data_type::f32,
                                                memory::format_tag::ab), eng);
        add_mem[i] = create_memory(memory::desc({batch_size, dims[i+1]},
                                                memory::data_type::f32,
                                                memory::format_tag::ab), eng);
    }

    // Create BF16 memory descriptors and memory objects for input, weight, output matrices
    std::vector<memory> input_mem_bf16;
    std::vector<memory> weight_mem_bf16(num_layers);
    std::vector<memory> out_mem_bf16(num_layers);
    std::vector<memory> add_mem_bf16(num_layers);

    input_mem_bf16.push_back(create_memory(memory::desc({batch_size, dims[0]},
                                           memory::data_type::bf16,
                                           memory::format_tag::ab), eng));

    for (int i = 0; i <num_layers; ++i) {
        weight_mem_bf16[i] = create_memory(memory::desc({dims[i], dims[i+1]},
                                           memory::data_type::bf16,
                                           memory::format_tag::ab), eng);

        out_mem_bf16[i] = create_memory(memory::desc({batch_size, dims[i+1]},
                                        memory::data_type::bf16,
                                        memory::format_tag::ab), eng);
        add_mem_bf16[i] = create_memory(memory::desc({batch_size, dims[i+1]},
                                        memory::data_type::bf16,
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
    //reorder the f32 to bf16 and execute
    reorder(input_mem[0], input_mem_bf16[0]).execute(s,input_mem[0],
            input_mem_bf16[0]);

    // Generate random weights for layers
    std::vector<float> weights_data;
    for (int i = 0; i < num_layers; ++i) {
        weights_data.resize(dims[i] * dims[i + 1]);
        for (auto &w : weights_data) {
            w = dis_weights(gen);
        }
        write_to_zendnn_memory(weights_data.data(), weight_mem[i]);
        //reorder the f32 to bf16 and execute
        reorder(weight_mem[i], weight_mem_bf16[i]).execute(s,weight_mem[i],
                weight_mem_bf16[i]);
    }

    std::vector<float> add_data;
    for (int i = 0; i < num_layers; ++i) {
        add_data.resize(batch_size * dims[i + 1]);
        for (auto &w : add_data) {
            w = dis_weights(gen);
        }
        write_to_zendnn_memory(add_data.data(), add_mem[i]);
        //reorder the f32 to bf16 and execute
        reorder(add_mem[i], add_mem_bf16[i]).execute(s,add_mem[i],
                add_mem_bf16[i]);
    }

    std::vector<float> alpha(num_layers, 1.f);
    std::vector<float> beta(num_layers, 0.f);
    std::vector<memory> bias(num_layers);
    std::vector<bool> bias_defined(num_layers, 0);
    std::vector<int64_t> fuse(num_layers, 1);
    std::vector<std::vector<int64_t>> po_id(num_layers);
    std::vector<std::vector<memory>> po_mem_buff(num_layers);

    //ReLU -> ADD post-ops
    for (int i=0; i<num_layers; i++) {
        po_id[i] = {1,6};
        //Buffer for Add
#if BF16_ENABLE
        po_mem_buff[i] = {add_mem_bf16[i]};
#else
        po_mem_buff[i] = {add_mem[i]};
#endif
    }

#if !WEIGHT_CACHING
    // Execute the MLP layers
    for (int i = 0; i < num_layers; ++i) {
        if (i==0) {
#if BF16_ENABLE
            matmul_execute(eng, s, input_mem_bf16[i], weight_mem_bf16[i], alpha[i], beta[i],
                           bias_defined[i],
                           po_id[i], po_mem_buff[i], out_mem_bf16[i]);
#else
            matmul_execute(eng, s, input_mem[i], weight_mem[i], alpha[i], beta[i],
                           bias_defined[i],
                           po_id[i], po_mem_buff[i], out_mem[i]);

#endif
        }
        else {
#if BF16_ENABLE
            matmul_execute(eng, s, out_mem_bf16[i-1], weight_mem_bf16[i], alpha[i], beta[i],
                           bias_defined[i],
                           po_id[i], po_mem_buff[i], out_mem_bf16[i]);
#else
            matmul_execute(eng, s, out_mem[i-1], weight_mem[i], alpha[i], beta[i],
                           bias_defined[i], po_id[i], po_mem_buff[i], out_mem[i]);
#endif
        }
    }
#endif
    // Read data from memory object for the final output
    std::vector<float> output(batch_size*dims[num_layers]);
#if BF16_ENABLE
    read_from_zendnn_memory(output.data(), out_mem_bf16[num_layers-1]);
#else
    read_from_zendnn_memory(output.data(), out_mem[num_layers-1]);
#endif

    // Create memory descriptors for output matrices for grp Matmul call
    std::vector<memory> out_grp_mem(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        out_grp_mem[i] = create_memory(memory::desc({batch_size, dims[i+1]},
                                       memory::data_type::f32,
                                       memory::format_tag::ab), eng);

    }
    std::vector<memory> out_grp_mem_bf16(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        out_grp_mem_bf16[i] = create_memory(memory::desc({batch_size, dims[i+1]},
                                            memory::data_type::bf16,
                                            memory::format_tag::ab), eng);

    }

    /*****************Test for Group Linear MatMul********************/
#if BF16_ENABLE || WEIGHT_CACHING
#if WEIGHT_CACHING
    for (int i =0; i<50; i++)
#endif
        zendnn_custom_op::zendnn_grp_mlp(input_mem_bf16, weight_mem_bf16, bias, alpha,
                                         beta,
                                         bias_defined, po_id, po_mem_buff, out_grp_mem_bf16, "lib::zendnn_grp_mlp");
#else
    zendnn_custom_op::zendnn_grp_mlp(input_mem, weight_mem, bias, alpha, beta,
                                     bias_defined, po_id, po_mem_buff, out_grp_mem, "lib::zendnn_grp_mlp");
#endif
#if !WEIGHT_CACHING
    // Read data from memory object for the final output of grp Matmul
    std::vector<float> output_grp(batch_size*dims[num_layers]);
#if BF16_ENABLE
    read_from_zendnn_memory(output_grp.data(), out_grp_mem_bf16[num_layers-1]);
#else
    read_from_zendnn_memory(output_grp.data(), out_grp_mem[num_layers-1]);
#endif
    //Compare result
    for (int i=0; i<batch_size*dims[num_layers]; i++) {
        assert(output[i] == output_grp[i]);
    }
    std::cout << " Test Comparison for group Linear Matmul Successful " <<
              std::endl;

    /*****************Test for Group parallel MatMul********************/

    // Create FP32 memory descriptors and memory objects for input and weight matrices
    for (int i = 0; i < num_layers-1; ++i) {
        input_mem.push_back(create_memory(memory::desc({batch_size, dims[0]},
                                          memory::data_type::f32,
                                          memory::format_tag::ab), eng));
    }

    std::vector<memory> weight_mem_parallel(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        weight_mem_parallel[i] = create_memory(memory::desc({dims[0], dims[1]},
                                               memory::data_type::f32,
                                               memory::format_tag::ab), eng);
    }

    // Create BF16 memory descriptors and memory objects for input, weight, output matrices
    for (int i = 0; i < num_layers-1; ++i) {
        input_mem_bf16.push_back(create_memory(memory::desc({batch_size, dims[0]},
                                               memory::data_type::bf16,
                                               memory::format_tag::ab), eng));
    }

    std::vector<memory> weight_mem_parallel_bf16(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        weight_mem_parallel_bf16[i] = create_memory(memory::desc({dims[0], dims[1]},
                                      memory::data_type::bf16,
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
        //reorder the f32 to bf16 and execute
        reorder(input_mem[j], input_mem_bf16[j]).execute(s, input_mem[j],
                input_mem_bf16[j]);
    }

    // Generate random weights for layers
    for (int i = 0; i < num_layers; ++i) {
        weights_data.resize(dims[0] * dims[1]);
        for (auto &w : weights_data) {
            w = dis_weights(gen);
        }
        write_to_zendnn_memory(weights_data.data(), weight_mem_parallel[i]);
        //reorder the f32 to bf16 and execute
        reorder(weight_mem_parallel[i], weight_mem_parallel_bf16[i]).execute(s,
                weight_mem_parallel[i], weight_mem_parallel_bf16[i]);
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
#if BF16_ENABLE
        matmul_execute(eng, s, input_mem_bf16[i], weight_mem_parallel_bf16[i], alpha[i],
                       beta[i],
                       bias_defined[i], po_id[i], po_mem_buff[i], out_mem_parallel[i]);
#else
        matmul_execute(eng, s, input_mem[i], weight_mem_parallel[i], alpha[i], beta[i],
                       bias_defined[i], po_id[i], po_mem_buff[i], out_mem_parallel[i]);
#endif

    }
    std::vector<float> output_parallel(batch_size*dims[1]);
    read_from_zendnn_memory(output_parallel.data(), out_mem_parallel[1]);

#if BF16_ENABLE
    zendnn_custom_op::zendnn_grp_mlp(input_mem_bf16, weight_mem_parallel_bf16, bias,
                                     alpha,
                                     beta,
                                     bias_defined, po_id, po_mem_buff, out_mem_parallel, "lib::zendnn_grp_mlp");
#else
    zendnn_custom_op::zendnn_grp_mlp(input_mem, weight_mem_parallel, bias, alpha,
                                     beta,
                                     bias_defined, po_id, po_mem_buff, out_mem_parallel, "lib::zendnn_grp_mlp");
#endif
    std::vector<float> output_grp_parallel(batch_size*dims[1]);
    read_from_zendnn_memory(output_grp_parallel.data(), out_mem_parallel[1]);

    //Compare result
    for (int i=0; i<batch_size*dims[1]; i++) {
        assert(output_parallel[i] == output_grp_parallel[i]);
    }

    std::cout << " Test Comparison for group parallel Matmul Successful " <<
              std::endl;
    return 0;
#endif
}
