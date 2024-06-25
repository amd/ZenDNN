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

#pragma once
#ifdef USE_LIBXSMM
#include "../zen/zentpp_memory.hpp"
#include "../zen/zentpp_types.hpp"
#include "../dyndisp/DispatchStub.h"

namespace zendnn {
namespace tpp {
namespace cpu {

void tpp_linear_nobias_forward_cpu(
    ZenTensor& t_out,
    const ZenTensor& t_in,
    const ZenTensor& t_wt);

void tpp_linear_bias_forward_cpu(
    ZenTensor& t_out,
    const ZenTensor& t_in,
    const ZenTensor& t_wt,
    const ZenTensor& t_bias);

void tpp_linear_gelu_forward_cpu(
    ZenTensor& t_out,
    const ZenTensor& t_in,
    const ZenTensor& t_wt,
    const ZenTensor& t_bias);

void tpp_fused_gate_up_proj_forward_cpu(
    ZenTensor& t_out,
    const ZenTensor& t_in,
    const ZenTensor& t_wt_gate,
    const ZenTensor& t_bias_gate,
    const ZenTensor& t_wt_up,
    const ZenTensor& t_bias_up);

void tpp_linear_silu_forward_cpu(
    ZenTensor& t_out,
    const ZenTensor& t_in,
    const ZenTensor& t_wt,
    const ZenTensor& t_bias);

void tpp_linear_relu_forward_cpu(
    ZenTensor& t_out,
    const ZenTensor& t_in,
    const ZenTensor& t_wt,
    const ZenTensor& t_bias);

void tpp_linear_add_forward_cpu(
    ZenTensor& t_out,
    const ZenTensor& t_in,
    const ZenTensor& t_in1,
    const ZenTensor& t_wt,
    const ZenTensor& t_bias,
    double scale);

void tpp_linear_mul_forward_cpu(
    ZenTensor& t_out,
    const ZenTensor& t_in,
    const ZenTensor& t_in1,
    const ZenTensor& t_wt,
    const ZenTensor& t_bias);

void tpp_linear_add_add_forward_cpu(
    ZenTensor& t_out,
    const ZenTensor& t_in,
    const ZenTensor& t_in1,
    const ZenTensor& t_in2,
    const ZenTensor& t_wt,
    const ZenTensor& t_bias,
    double scale);

using tpp_linear_nobias_impl_fn =
    void (*)(ZenTensor&, const ZenTensor&,
             const ZenTensor&);

using tpp_linear_bias_kernel_impl_fn =
    void (*)(ZenTensor&, const ZenTensor&,
             const ZenTensor&, const ZenTensor&);

using tpp_linear_gelu_kernel_impl_fn =
    void (*)(ZenTensor&, const ZenTensor&,
             const ZenTensor&, const ZenTensor&);

using tpp_fused_gate_up_proj_kernel_impl_fn =
    void (*)(ZenTensor&,
             const ZenTensor&,
             const ZenTensor&,
             const ZenTensor&,
             const ZenTensor&,
             const ZenTensor&);

using tpp_linear_silu_kernel_impl_fn =
    void (*)(ZenTensor&, const ZenTensor&,
             const ZenTensor&, const ZenTensor&);

using tpp_linear_relu_kernel_impl_fn =
    void (*)(ZenTensor&, const ZenTensor&,
             const ZenTensor&, const ZenTensor&);

using tpp_linear_add_kernel_impl_fn =
    void (*)(ZenTensor&,
             const ZenTensor&,
             const ZenTensor&,
             const ZenTensor&,
             const ZenTensor&,
             double);

using tpp_linear_mul_kernel_impl_fn =
    void (*)(ZenTensor&,
    const ZenTensor&,
    const ZenTensor&,
    const ZenTensor&,
    const ZenTensor&);

using tpp_linear_add_add_kernel_impl_fn =
    void (*)(ZenTensor&,
    const ZenTensor&,
    const ZenTensor&,
    const ZenTensor&,
    const ZenTensor&,
    const ZenTensor&,
    double);

IPEX_DECLARE_DISPATCH(
    tpp_linear_nobias_impl_fn,
    tpp_linear_nobias_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_linear_bias_kernel_impl_fn,
    tpp_linear_bias_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_linear_gelu_kernel_impl_fn,
    tpp_linear_gelu_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_fused_gate_up_proj_kernel_impl_fn,
    tpp_fused_gate_up_proj_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_linear_silu_kernel_impl_fn,
    tpp_linear_silu_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_linear_relu_kernel_impl_fn,
    tpp_linear_relu_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_linear_add_kernel_impl_fn,
    tpp_linear_add_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_linear_mul_kernel_impl_fn,
    tpp_linear_mul_kernel_stub);
IPEX_DECLARE_DISPATCH(
    tpp_linear_add_add_kernel_impl_fn,
    tpp_linear_add_add_kernel_stub);

} // namespace cpu
} // namespace tpp
} // zendnn
#endif
