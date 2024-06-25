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

#include "zendnn_logging.hpp"
#include "zen/zentpp_defs.hpp"

#include <iostream>

#ifdef USE_LIBXSMM
#include "tpp/kernels/TPPGEMMKrnl.h"
#include <aten/TPPGEMM.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>

// using namespace zendnn;
// using namespace zendnn::tpp;

namespace zendnn {
namespace tpp {
namespace cpu {

namespace {

using ZenDType = zendnn::tpp::ZenDType;

void tpp_linear_bias_kernel_impl(
    zendnn::tpp::ZenTensor& t_out,
    const zendnn::tpp::ZenTensor& t_in,
    const zendnn::tpp::ZenTensor& t_wt,
    const zendnn::tpp::ZenTensor& t_bias) {

  auto dt = t_wt.dtype();
  if (dt == ZenDType::f32) {
    zendnn::tpp::tpp_linear_bias<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == ZenDType::bf16) {
      zendnn::tpp::tpp_linear_bias<zendnn::tpp::bfloat16>(t_in, t_wt,
                                                              t_bias, t_out);
  } else {
      zendnnError(ZENDNN_ALGOLOG, "tpp_linear_bias_kernel_impl:",
                  "datatype not supported.");
  }

}

void tpp_linear_nobias_kernel_impl(
    zendnn::tpp::ZenTensor& t_out,
    const zendnn::tpp::ZenTensor& t_in,
    const zendnn::tpp::ZenTensor& t_wt) {

  auto dt = t_wt.dtype();
  if (dt == ZenDType::f32) {
    zendnn::tpp::tpp_linear_no_bias<float>(t_in, t_wt, t_out);
  } else if (dt == ZenDType::bf16) {
      zendnn::tpp::tpp_linear_no_bias<zendnn::tpp::bfloat16>(t_in, t_wt,
                                                                 t_out);
  } else {
      zendnnError(ZENDNN_ALGOLOG, "tpp_linear_nobias_kernel_impl:",
                  "datatype not supported.");
  }

}

void tpp_linear_gelu_kernel_impl(
    zendnn::tpp::ZenTensor& t_out,
    const zendnn::tpp::ZenTensor& t_in,
    const zendnn::tpp::ZenTensor& t_wt,
    const zendnn::tpp::ZenTensor& t_bias) {

  auto dt = t_wt.dtype();
  if (dt == ZenDType::f32) {
    zendnn::tpp::tpp_linear_gelu<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == ZenDType::bf16) {
    zendnn::tpp::tpp_linear_gelu<zendnn::tpp::bfloat16>(t_in, t_wt, t_bias,
                                                            t_out);
  } else {
      zendnnError(ZENDNN_ALGOLOG, "tpp_linear_nobias_kernel_impl:",
                  "datatype not supported.");
  }

}

void tpp_fused_gate_up_proj_kernel_impl(
    zendnn::tpp::ZenTensor& t_out,
    const zendnn::tpp::ZenTensor& t_in,
    const zendnn::tpp::ZenTensor& t_wt_gate,
    const zendnn::tpp::ZenTensor& t_bias_gate,
    const zendnn::tpp::ZenTensor& t_wt_up,
    const zendnn::tpp::ZenTensor& t_bias_up) {

  auto dt = t_wt_gate.dtype();
  if (dt == ZenDType::f32) {
    zendnn::tpp::tpp_fused_gate_up_proj<float>(
        t_in, t_wt_gate, t_bias_gate, t_wt_up, t_bias_up, t_out);
  } else if (dt == ZenDType::bf16) {
    zendnn::tpp::tpp_fused_gate_up_proj<zendnn::tpp::bfloat16>(
        t_in, t_wt_gate, t_bias_gate, t_wt_up, t_bias_up, t_out);
  } else {
    zendnnError(ZENDNN_ALGOLOG, "tpp_fused_gate_up_proj_kernel_impl:",
                "datatype not supported.");
  }

}

void tpp_linear_silu_kernel_impl(
    zendnn::tpp::ZenTensor& t_out,
    const zendnn::tpp::ZenTensor& t_in,
    const zendnn::tpp::ZenTensor& t_wt,
    const zendnn::tpp::ZenTensor& t_bias) {

  auto dt = t_wt.dtype();
  if (dt == ZenDType::f32) {
    zendnn::tpp::tpp_linear_silu<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == ZenDType::bf16) {
    zendnn::tpp::tpp_linear_silu<zendnn::tpp::bfloat16>(t_in, t_wt, t_bias,
                                                            t_out);
  } else {
    zendnnError(ZENDNN_ALGOLOG, "tpp_linear_silu_kernel_impl:",
                "datatype not supported.");
  }

}

void tpp_linear_relu_kernel_impl(
    zendnn::tpp::ZenTensor& t_out,
    const zendnn::tpp::ZenTensor& t_in,
    const zendnn::tpp::ZenTensor& t_wt,
    const zendnn::tpp::ZenTensor& t_bias) {

  auto dt = t_wt.dtype();
  if (dt == ZenDType::f32) {
    zendnn::tpp::tpp_linear_relu<float>(t_in, t_wt, t_bias, t_out);
  } else if (dt == ZenDType::bf16) {
    zendnn::tpp::tpp_linear_relu<zendnn::tpp::bfloat16>(t_in, t_wt, t_bias,
                                                            t_out);
  } else {
    zendnnError(ZENDNN_ALGOLOG, "tpp_linear_relu_kernel_impl:",
                "datatype not supported.");
  }

}

void tpp_linear_add_add_kernel_impl(
    zendnn::tpp::ZenTensor& t_out,
    const zendnn::tpp::ZenTensor& t_in,
    const zendnn::tpp::ZenTensor& t_in1,
    const zendnn::tpp::ZenTensor& t_in2,
    const zendnn::tpp::ZenTensor& t_wt,
    const zendnn::tpp::ZenTensor& t_bias,
    double scale) {
  auto dt = t_wt.dtype();
  if (dt == ZenDType::f32) {
    zendnn::tpp::tpp_linear_add_add<float>(
        t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else if (dt == ZenDType::bf16) {
    zendnn::tpp::tpp_linear_add_add<zendnn::tpp::bfloat16>(
        t_in, t_in1, t_in2, t_wt, t_bias, t_out, scale);
  } else {
    zendnnError(ZENDNN_ALGOLOG, "tpp_linear_add_add_kernel_impl:",
                "datatype not supported.");
  }

}

void tpp_linear_add_kernel_impl(
    zendnn::tpp::ZenTensor& t_out,
    const zendnn::tpp::ZenTensor& t_in,
    const zendnn::tpp::ZenTensor& t_in1,
    const zendnn::tpp::ZenTensor& t_wt,
    const zendnn::tpp::ZenTensor& t_bias,
    double scale) {
  auto dt = t_wt.dtype();
  if (dt == ZenDType::f32) {
    zendnn::tpp::tpp_linear_add<float>(
        t_in, t_in1, t_wt, t_bias, t_out, scale);
  } else if (dt == ZenDType::bf16) {
    zendnn::tpp::tpp_linear_add<zendnn::tpp::bfloat16>(
        t_in, t_in1, t_wt, t_bias, t_out, scale);
  } else {
    zendnnError(ZENDNN_ALGOLOG, "tpp_linear_add_kernel_impl:",
                "datatype not supported.");
  }

}

void tpp_linear_mul_kernel_impl(
    zendnn::tpp::ZenTensor& t_out,
    const zendnn::tpp::ZenTensor& t_in,
    const zendnn::tpp::ZenTensor& t_in1,
    const zendnn::tpp::ZenTensor& t_wt,
    const zendnn::tpp::ZenTensor& t_bias) {
  auto dt = t_wt.dtype();
  if (dt == ZenDType::f32) {
    zendnn::tpp::tpp_linear_mul<float>(t_in, t_in1, t_wt, t_bias, t_out);
  } else if (dt == ZenDType::bf16) {
    zendnn::tpp::tpp_linear_mul<zendnn::tpp::bfloat16>(
        t_in, t_in1, t_wt, t_bias, t_out);
  } else {
    zendnnError(ZENDNN_ALGOLOG, "tpp_linear_mul_kernel_impl:",
                "datatype not supported.");
  }
}

} // namespace

IPEX_REGISTER_DISPATCH(
    tpp_linear_nobias_kernel_stub,
    &tpp_linear_nobias_kernel_impl);
IPEX_REGISTER_DISPATCH(
    tpp_linear_bias_kernel_stub,
    &tpp_linear_bias_kernel_impl);
IPEX_REGISTER_DISPATCH(
    tpp_linear_gelu_kernel_stub,
    &tpp_linear_gelu_kernel_impl);
IPEX_REGISTER_DISPATCH(
    tpp_fused_gate_up_proj_kernel_stub,
    &tpp_fused_gate_up_proj_kernel_impl);
IPEX_REGISTER_DISPATCH(
    tpp_linear_relu_kernel_stub,
    &tpp_linear_relu_kernel_impl);
IPEX_REGISTER_DISPATCH(
    tpp_linear_silu_kernel_stub,
    &tpp_linear_silu_kernel_impl);
IPEX_REGISTER_DISPATCH(tpp_linear_mul_kernel_stub, &tpp_linear_mul_kernel_impl);
IPEX_REGISTER_DISPATCH(tpp_linear_add_kernel_stub, &tpp_linear_add_kernel_impl);
IPEX_REGISTER_DISPATCH(
    tpp_linear_add_add_kernel_stub,
    &tpp_linear_add_add_kernel_impl);
} // namespace cpu
} // namespace tpp
} // namespace zendnn
#endif
