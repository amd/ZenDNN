/*******************************************************************************
* Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "zendnn.hpp"
#include "zendnn_helper.hpp"
#include <vector>
#include <iostream>
#include <cstring>
#include "zendnn_logging.hpp"
#include "verbose.hpp"
#include <string.h>
#include <omp.h>
#include <future>
#include <thread>
#include <blis.h>

namespace zendnn {

void zen_matmul_impl(
    const memory &z_input,
    const memory &z_weight,
    const memory &z_bias,
    const float &z_alpha,
    const float &z_beta,
    const bool &z_bias_defined,
    const std::vector<int64_t> &z_post_op_ids,
    const std::vector<memory> &z_post_op_buffers,
    const memory &z_result,
    engine eng,
    stream engine_stream, const char *plugin_op) {

    zendnn::primitive_attr op_attr;
    std::string op_name=plugin_op;
    op_attr.set_plugin_op_name(op_name);
    post_ops po;
    int post_op_idx = 0;
    if (z_beta != 0.0f && !z_bias_defined) {
        // sets post_ops as add or sum
        post_op_idx++;
        po.append_sum(z_beta);
    }
    if (z_alpha != 1.0f) {
        op_attr.set_output_scales(0, {z_alpha});
    }

    int post_op_ids_size = z_post_op_ids.size();
    std::unordered_map<int, memory> execute_args;
    int post_op_buffer_idx = 0;
    for (int i = 0; i < post_op_ids_size; i++) {
        int arg_position;
        // set the post-ops or fusion-ops;
        zendnnPostOp po_enum = static_cast<zendnnPostOp>(z_post_op_ids[i]);
        switch (po_enum) {
        case zendnnPostOp::RELU:
            po.append_eltwise(1.0f, algorithm::eltwise_relu, 0.f, 0.f);
            break;
        case zendnnPostOp::GELU_TANH:
            po.append_eltwise(1.0f, algorithm::eltwise_gelu_tanh, 1.f, 0.f);
            break;
        case zendnnPostOp::GELU_ERF:
            po.append_eltwise(1.0f, algorithm::eltwise_gelu_erf, 1.f, 0.f);
            break;
        case zendnnPostOp::SILU:
            po.append_eltwise(1.0f, algorithm::eltwise_swish, 1.f, 0.f);
            break;
        case zendnnPostOp::MUL:
            po.append_binary(algorithm::binary_mul,
                             z_post_op_buffers[post_op_buffer_idx].get_desc());
            arg_position = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(post_op_idx) | ZENDNN_ARG_SRC_1;
            execute_args.insert(
            {arg_position, z_post_op_buffers[post_op_buffer_idx]});
            post_op_buffer_idx++;
            break;
        case zendnnPostOp::ADD:
            po.append_binary(algorithm::binary_add,
                             z_post_op_buffers[post_op_buffer_idx].get_desc());
            arg_position = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(post_op_idx) | ZENDNN_ARG_SRC_1;
            execute_args.insert(
            {arg_position, z_post_op_buffers[post_op_buffer_idx]});
            post_op_buffer_idx++;
            break;
        case zendnnPostOp::NONE:
            break;
        default:
            ZENDNN_THROW_ERROR(zendnn_invalid_arguments,
                               "Unsupported post-op for group MLP");
            break;
        }
        post_op_idx++;
    }

    op_attr.set_post_ops(po);

    matmul::desc pdesc = z_bias_defined
                         ? matmul::desc(z_input.get_desc(), z_weight.get_desc(),
                                        z_bias.get_desc(),
                                        z_result.get_desc())
                         : matmul::desc(z_input.get_desc(), z_weight.get_desc(),
                                        z_result.get_desc());

    matmul::primitive_desc pd =
        matmul::primitive_desc(pdesc, op_attr, eng);

    execute_args.insert({ZENDNN_ARG_SRC, z_input});
    execute_args.insert({ZENDNN_ARG_WEIGHTS, z_weight});
    if (z_bias_defined) {
        execute_args.insert({ZENDNN_ARG_BIAS, z_bias});
    }
    execute_args.insert({ZENDNN_ARG_DST, z_result});
    matmul(pd).execute(engine_stream, execute_args);

}
void set_z_result(const float &alpha, const float &beta,
                  const bool &bias_defined, const memory &bias, const memory &result) {

    int result_size = result.get_desc().get_size()/sizeof(float);
    float *hndl = static_cast<float *>(result.get_data_handle());

    // If alpha = 0, does not need to actually do gemm computation
    if (beta == 0.0f) {
        memset(hndl, 0, result_size * sizeof(float));
        return;
    }
    else if (bias_defined) {
        // bias is already multiplied by beta
        float *bias_hndl = static_cast<float *>(bias.get_data_handle());
        memcpy(hndl, bias_hndl, result_size * sizeof(float));
        return;
    }
    else {
        for (int j = 0; j < result_size; j++) {
            hndl[j]*=beta;
        }
        return;
    }
}

void zendnn_custom_op::zendnn_grp_mlp(
    const std::vector<memory> &z_input,
    const std::vector<memory> &z_weight,
    const std::vector<memory> &z_bias,
    const std::vector<float> &z_alpha,
    const std::vector<float> &z_beta,
    const std::vector<bool> &z_bias_defined,
    const std::vector<std::vector<int64_t>> &z_post_op_ids,
    const std::vector<std::vector<memory>> &z_post_op_buffers,
    const std::vector<memory> &z_result, const char *plugin_op)

{
    double start_ms = impl::get_msec();
    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream stream(eng);
    int num_ops=z_result.size();
    std::string mlp_type;

    if (z_input.size()==1) {
        mlp_type="linear";
        for (int i = 0; i < num_ops; i++) {

            // If alpha = 0, does not need to actually do gemm computation
            if (z_alpha[i]==0) {
                set_z_result(z_alpha[i], z_beta[i], z_bias_defined[i], z_bias[i], z_result[i]);
                continue;
            }
            if (i==0) {
                zen_matmul_impl(z_input[i], z_weight[i], z_bias[i], z_alpha[i], z_beta[i],
                                z_bias_defined[i], z_post_op_ids[i], z_post_op_buffers[i], z_result[i], eng,
                                stream, plugin_op);
            }
            else {
                zen_matmul_impl(z_result[i-1], z_weight[i], z_bias[i], z_alpha[i], z_beta[i],
                                z_bias_defined[i], z_post_op_ids[i], z_post_op_buffers[i], z_result[i], eng,
                                stream, plugin_op);
            }
        }
    }

    else {
        mlp_type="parallel";
        const int num_outer_threads = z_input.size();
        uint  mlp_thread_type;
        mlp_thread_type = zendnn_getenv_int("ZENDNN_MLP_THREAD_TYPE", 0);

        //This is the experimental path to perform multi-level parallelism
        zendnnEnv zenEnvObj = readEnv();
        unsigned int mlp_thread_qty = zenEnvObj.omp_num_threads;
        unsigned int rem = mlp_thread_qty % num_outer_threads;

        if (mlp_thread_type==1) {
            omp_set_max_active_levels(2);
            #pragma omp parallel num_threads(num_outer_threads)
            {
                unsigned int inner_threads = mlp_thread_qty/num_outer_threads;
                unsigned int threadOffset = omp_get_thread_num();
                if (threadOffset < rem) {
                    inner_threads++;
                }
                bli_thread_set_num_threads(inner_threads);
                // If alpha = 0, does not need to actually do gemm computation
                if (z_alpha[threadOffset]==0) {
                    set_z_result(z_alpha[threadOffset], z_beta[threadOffset],
                                 z_bias_defined[threadOffset], z_bias[threadOffset], z_result[threadOffset]);
                }
                else {
                    zen_matmul_impl(z_input[threadOffset], z_weight[threadOffset],
                                    z_bias[threadOffset], z_alpha[threadOffset], z_beta[threadOffset],
                                    z_bias_defined[threadOffset], z_post_op_ids[threadOffset],
                                    z_post_op_buffers[threadOffset], z_result[threadOffset], eng,
                                    stream, plugin_op);
                }
            }
        }
        else if (mlp_thread_type==2) {
            std::vector<std::future<void>> futures;
            for (int i = 0; i < num_outer_threads; i++) {

                // If alpha = 0, does not need to actually do gemm computation
                if (z_alpha[i]==0) {
                    set_z_result(z_alpha[i], z_beta[i], z_bias_defined[i], z_bias[i], z_result[i]);
                    continue;
                }
                // Launch the top-level threads asynchronously
                futures.push_back(std::async(std::launch::async,zen_matmul_impl, z_input[i],
                                             z_weight[i], z_bias[i], z_alpha[i], z_beta[i],
                                             z_bias_defined[i], z_post_op_ids[i], z_post_op_buffers[i], z_result[i], eng,
                                             stream, plugin_op));
            }
            for (auto &fut : futures) {
                fut.get();
            }
        }
        else if (mlp_thread_type==3) {
            std::vector<std::thread> threads;
            for (int i = 0; i < num_outer_threads; i++) {

                // If alpha = 0, does not need to actually do gemm computation
                if (z_alpha[i]==0) {
                    set_z_result(z_alpha[i], z_beta[i], z_bias_defined[i], z_bias[i], z_result[i]);
                    continue;
                }
                threads.emplace_back(zen_matmul_impl, z_input[i], z_weight[i], z_bias[i],
                                     z_alpha[i], z_beta[i],
                                     z_bias_defined[i], z_post_op_ids[i], z_post_op_buffers[i], z_result[i], eng,
                                     stream, plugin_op);
            }
            for (auto &thread : threads) {
                thread.join();
            }

        }
        else {
            for (int i = 0; i < num_outer_threads; i++) {

                // If alpha = 0, does not need to actually do gemm computation
                if (z_alpha[i]==0) {
                    set_z_result(z_alpha[i], z_beta[i], z_bias_defined[i], z_bias[i], z_result[i]);
                    continue;
                }
                zen_matmul_impl(z_input[i], z_weight[i], z_bias[i], z_alpha[i], z_beta[i],
                                z_bias_defined[i], z_post_op_ids[i], z_post_op_buffers[i], z_result[i], eng,
                                stream, plugin_op);
            }
        }
    }
    double duration_ms = impl::get_msec() - start_ms;

    zendnnVerbose(ZENDNN_PROFLOG,
                  "zendnn_custom_op_execute,cpu,plugin_op:",plugin_op,",",
                  "num_ops:",num_ops,",","dims:",",","alg:mlp_",mlp_type,",",
                  duration_ms,",ms");
}

void zendnn_custom_op::zendnn_grp_ebag_mlp(
    std::vector <memory> &z_eb_input,
    std::vector <memory> &z_eb_indices, std::vector <memory> &z_eb_offsets,
    std::vector <int32_t> &z_eb_scale_grad_by_freq,
    std::vector <algorithm> &z_eb_modes,
    std::vector <int32_t> &z_eb_sparse,
    std::vector <memory> &z_eb_per_sample_weights_opt,
    std::vector <int32_t> &z_eb_per_sample_weights_defined,
    std::vector <int32_t> &z_eb_include_last_offset,
    std::vector <int32_t> &z_eb_padding_idx,
    std::vector <memory> &z_eb_destination,
    const std::vector<memory> &z_mm_input,
    const std::vector<memory> &z_mm_weight,
    const std::vector<memory> &z_mm_bias,
    const std::vector<float> &z_mm_alpha,
    const std::vector<float> &z_mm_beta,
    const std::vector<bool> &z_mm_bias_defined,
    const std::vector<std::vector<int64_t>> &z_post_op_ids,
    const std::vector<std::vector<memory>> &z_post_op_buffers,
    const std::vector<memory> &z_mm_result, const char *plugin_op)

{
    double start_ms = impl::get_msec();

    zendnn_custom_op::zendnn_grp_embedding_bag(
        z_eb_input, z_eb_indices, z_eb_offsets, z_eb_scale_grad_by_freq, z_eb_modes,
        z_eb_sparse, z_eb_per_sample_weights_opt, z_eb_per_sample_weights_defined,
        z_eb_include_last_offset, z_eb_padding_idx, z_eb_destination, plugin_op, 1);

    zendnn_custom_op::zendnn_grp_mlp(z_mm_input, z_mm_weight, z_mm_bias, z_mm_alpha,
                                     z_mm_beta, z_mm_bias_defined, z_post_op_ids, z_post_op_buffers, z_mm_result,
                                     plugin_op);

    double duration_ms = impl::get_msec() - start_ms;

    std::string mlp_type;
    std::string eb_type;

    if (z_mm_input.size()==1) {
        mlp_type="linear";
    }
    else {
        mlp_type="parallel";
    }
    zendnnEnv zenEnvObj = readEnv();
    switch (zenEnvObj.zenEBThreadAlgo) {
    case 1:
        eb_type="batch_threaded";
        break;
    case 2:
        eb_type="table_threaded";
        break;
    case 3:
        eb_type="hybrid_threaded";
        break;
    default:
        eb_type="ccd_threaded";
        break;
    }
    zendnnVerbose(ZENDNN_PROFLOG,
                  "zendnn_custom_op_execute,cpu,plugin_op:",plugin_op,",",
                  "num_ops:","matmuls=",z_mm_result.size()," ","eb=",z_eb_input.size(),",",
                  "dims:",
                  ",","alg:mlp=",mlp_type," ","eb=",
                  eb_type,",",
                  duration_ms,",ms");
}

void zendnn_custom_op::zendnn_grp_embedding_mlp(
    std::vector <memory> &z_embed_input,
    std::vector <memory> &z_embed_indices,
    std::vector <int32_t> &z_embed_scale_grad_by_freq,
    std::vector <int32_t> &z_embed_sparse,
    std::vector <int32_t> &z_embed_padding_idx,
    std::vector <memory> &z_embed_destination,
    const std::vector<memory> &z_mm_input,
    const std::vector<memory> &z_mm_weight,
    const std::vector<memory> &z_mm_bias,
    const std::vector<float> &z_mm_alpha,
    const std::vector<float> &z_mm_beta,
    const std::vector<bool> &z_mm_bias_defined,
    const std::vector<std::vector<int64_t>> &z_post_op_ids,
    const std::vector<std::vector<memory>> &z_post_op_buffers,
    const std::vector<memory> &z_mm_result)

{

    zendnn_custom_op::zendnn_grp_embedding(
        z_embed_input, z_embed_indices, z_embed_padding_idx, z_embed_scale_grad_by_freq,
        z_embed_sparse, z_embed_destination,"zendnn_grp_embedding_mlp",1);

    zendnn_custom_op::zendnn_grp_mlp(z_mm_input, z_mm_weight, z_mm_bias, z_mm_alpha,
                                     z_mm_beta, z_mm_bias_defined, z_post_op_ids, z_post_op_buffers, z_mm_result,
                                     "zendnn_grp_ebag_mlp");
}

}

