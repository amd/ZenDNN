/*****************************************************************************
* Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
* All rights reserved.
* Notified per clause 4(b) of the license.
******************************************************************************/
/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

#include <iostream>
#include <vector>

namespace zendnn {
namespace tpp {

void dense_sparse_add_(
    at::Tensor dense,
    at::Tensor sparse,
    /*torch::Scalar*/ float alpha);

void bf16_split_add_(
    at::Tensor hi_bits,
    at::Tensor lo_bits,
    at::Tensor grad,
    float lr);

void fused_adamw(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_exp_avg,
    at::Tensor& t_exp_avg_sq,
    float beta1,
    float beta2,
    float step_size,
    float lr,
    float weight_decay,
    float eps);

void fused_split_adamw(
    at::Tensor& t_data_hi,
    at::Tensor& t_data_lo,
    at::Tensor& t_grad,
    at::Tensor& t_exp_avg,
    at::Tensor& t_exp_avg_sq,
    float beta1,
    float beta2,
    float step_size,
    float lr,
    float weight_decay,
    float eps);

double clip_grad_norm(std::vector<at::Tensor>& grads, double max_norm);

float fused_lamb(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_exp_avg,
    at::Tensor& t_exp_avg_sq,
    float beta1,
    float beta2,
    float weight_norm,
    float lr,
    float weight_decay,
    float eps);

void fused_lamb_v2(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_exp_avg,
    at::Tensor& t_exp_avg_sq,
    at::Tensor& t_adam_step,
    at::Tensor& t_data_low,
    at::Tensor& t_offsets,
    at::Tensor& t_block_sizes,
    at::Tensor& t_block2param,
    at::Tensor& t_weight_norms,
    at::Tensor& t_update_norms,
    float weight_decay,
    float beta1,
    float beta2,
    float lr,
    float eps,
    int block_size,
    int step,
    bool fused_param_norm);

} // namespace tpp

} // namespace zendnn
