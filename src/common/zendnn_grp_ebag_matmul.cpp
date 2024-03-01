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
#include "zendnn.hpp"
#include <vector>
#include <iostream>
#include <cstring>

namespace zendnn {

void zen_matmul_impl(
    const memory &z_input,
    const memory &z_weight,
    const memory &z_bias,
    const float &z_alpha,
    const float &z_beta,
    const bool &z_bias_defined,
    const int64_t &z_fuse,
    const memory &z_result,
    engine eng,
    stream engine_stream) {

    zendnn::primitive_attr op_attr;
    post_ops po;
    if (z_beta != 0.0f && !z_bias_defined) {
        // sets post_ops as add or sum
        po.append_sum(z_beta);
    }
    if (z_alpha != 1.0f) {
        op_attr.set_output_scales(0, {z_alpha});
    }

    // set the post-ops or fusion-ops;
    // by default, fuse = 0,
    // fuse = 1 for relu op,
    // fuse = 2 for gelu approximate (tanh)
    // fuse = 3 for gelu exact (erf)
    if (z_fuse == 1) {
        po.append_eltwise(1.0f, algorithm::eltwise_relu, 0.f, 0.f);
    }
    if (z_fuse == 2) {
        po.append_eltwise(1.0f, algorithm::eltwise_gelu_tanh, 1.f, 0.f);
    }
    if (z_fuse == 3) {
        po.append_eltwise(1.0f, algorithm::eltwise_gelu_erf, 1.f, 0.f);
    }
    op_attr.set_post_ops(po);

    matmul::desc pdesc =
        z_bias_defined
        ? matmul::desc(z_input.get_desc(), z_weight.get_desc(),
                       z_bias.get_desc(),
                       z_result.get_desc())
        : matmul::desc(z_input.get_desc(), z_weight.get_desc(),
                       z_result.get_desc());

    matmul::primitive_desc pd =
        matmul::primitive_desc(pdesc, op_attr, eng);

    std::unordered_map<int, memory> execute_args =
        z_bias_defined
    ? std::unordered_map<int, memory>({
        {ZENDNN_ARG_SRC, z_input},
        {ZENDNN_ARG_WEIGHTS, z_weight},
        {ZENDNN_ARG_BIAS, z_bias},
        {ZENDNN_ARG_DST, z_result}})
        : std::unordered_map<int, memory>({
        {ZENDNN_ARG_SRC, z_input},
        {ZENDNN_ARG_WEIGHTS, z_weight},
        {ZENDNN_ARG_DST, z_result}});
    matmul(pd).execute(engine_stream, execute_args);

}


void zendnn_custom_op::zendnn_grp_mlp(
    const memory &z_input,
    const std::vector<memory> &z_weight,
    const std::vector<memory> &z_bias,
    const std::vector<float> &z_alpha,
    const std::vector<float> &z_beta,
    const std::vector<bool> &z_bias_defined,
    const std::vector<int64_t> &z_fuse,
    const std::vector<memory> &z_result)

{
    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream stream(eng);
    int num_ops=z_result.size();

    for (int i = 0; i < num_ops; i++) {

        int result_size = z_result[i].get_desc().get_size()/sizeof(float);
        float *hndl = static_cast<float *>(z_result[i].get_data_handle());

        // If alpha = 0, does not need to actually do gemm computation
        if (z_alpha[i] == 0) {
            if (z_beta[i] == 0.0f) {
                memset(hndl, 0, result_size * sizeof(float));
                continue;
            }
            else if (z_bias_defined[i]) {
                // bias is already multiplied by beta
                float *bias_hndl = static_cast<float *>(z_bias[i].get_data_handle());
                memcpy(hndl, bias_hndl, result_size * sizeof(float));
                continue;
            }
            else {
                for (int j = 0; j < result_size; j++) {
                    hndl[j]*=z_beta[i];

                }
                continue;
            }
        }

        if (i==0) {
            zen_matmul_impl(z_input, z_weight[i], z_bias[i], z_alpha[i], z_beta[i],
                            z_bias_defined[i], z_fuse[i], z_result[i], eng, stream);
        }
        else {
            zen_matmul_impl(z_result[i-1], z_weight[i], z_bias[i], z_alpha[i], z_beta[i],
                            z_bias_defined[i], z_fuse[i], z_result[i], eng, stream);
        }
    }
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
    const memory &z_mm_input,
    const std::vector<memory> &z_mm_weight,
    const std::vector<memory> &z_mm_bias,
    const std::vector<float> &z_mm_alpha,
    const std::vector<float> &z_mm_beta,
    const std::vector<bool> &z_mm_bias_defined,
    const std::vector<int64_t> &z_mm_fuse,
    const std::vector<memory> &z_mm_result)

{

    zendnn_custom_op::zendnn_grp_embedding_bag(
        z_eb_input, z_eb_indices, z_eb_offsets, z_eb_scale_grad_by_freq, z_eb_modes,
        z_eb_sparse, z_eb_per_sample_weights_opt, z_eb_per_sample_weights_defined,
        z_eb_include_last_offset, z_eb_padding_idx, z_eb_destination);

    zendnn_custom_op::zendnn_grp_mlp(z_mm_input, z_mm_weight, z_mm_bias, z_mm_alpha,
                                     z_mm_beta, z_mm_bias_defined, z_mm_fuse, z_mm_result);
}

}

