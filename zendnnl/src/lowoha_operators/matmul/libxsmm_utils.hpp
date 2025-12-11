/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

#ifndef _LIBXSMM_UTILS_HPP
#define _LIBXSMM_UTILS_HPP

#include "lowoha_matmul_utils.hpp"

#if ZENDNNL_DEPENDS_LIBXSMM
  #include "libxsmm.h"
#endif


namespace zendnnl {
namespace lowoha {


/**
 * @brief Check if LibXSMM can be used for the given matrix multiplication parameters
 *
 * @param transA Transpose flag for matrix A ('t' or 'n')
 * @param transB Transpose flag for matrix B ('t' or 'n')
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scaling factor for A*B (must be 1.0 for LibXSMM)
 * @param beta Scaling factor for C (must be 0.0 for LibXSMM)
 * @param dtypes Data types for the operands
 * @return true if LibXSMM can handle this operation, false otherwise
 */
static inline bool can_use_libxsmm(char transA, char transB, int M,
                                   int N, int K, float alpha, float beta,
                                   const data_types &dtypes, const lowoha_params &lowoha_param,
                                   const matmul_algo_t &kernel_name) {

#if ZENDNNL_DEPENDS_LIBXSMM

  // Early exit for unsupported scalar values
  if (alpha != 1.0f || (beta != 0.0f && beta != 1.0f)) {
    return false;
  }

  //LIBXSMM throws segfault for transA='t' cases
  if (transA == 't') {
    return false;
  }


  if (kernel_name == matmul_algo_t::libxsmm) {
    int64_t matrix_b_elements = static_cast<int64_t>(K) * N;
    if (matrix_b_elements > 1000000 || (M > 512 && matrix_b_elements > 1000000)) {
      return false;
    }
  }


  if (lowoha_param.postop_.size() > 0) {
    //ToDo: Silu is not supported currently, since there is no direct API call available(can use sigm+mul).
    for (const auto &postop : lowoha_param.postop_) {
      switch (postop.po_type) {
      case post_op_type_t::binary_add:
      case post_op_type_t::binary_mul:
      case post_op_type_t::gelu_erf:
      case post_op_type_t::relu:
      case post_op_type_t::tanh:
      case post_op_type_t::sigmoid:
        continue;
      default:
        return false;
      }
    }
  }


  return (dtypes.src == data_type_t::f32  && dtypes.dst == data_type_t::f32) ||
         (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::f32) ||
         (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::bf16);

#endif
  return false;
}


#if ZENDNNL_DEPENDS_LIBXSMM
/**
 * @brief Apply post-operations (e.g., ReLU, Sigmoid, Tanh, GELU, Binary Multiply, Binary Add)
 *        on the output matrix in F32/BF16 format using LIBXSMM kernels.
 *
 * @param M Number of rows in the output matrix.
 * @param N Number of columns in the output matrix.
 * @param ldc Leading dimension of the output matrix.
 * @param output Pointer to the output matrix where the post-operation will be applied.
 * @param po A `postop` object containing the type of post-operation and any additional buffers required.
 */
template<typename T>
inline static void libxsmm_postop(const int M, const int N, const int ldc,
                                  void *output, const postop &po) {
  constexpr libxsmm_datatype IN_TYPE =
    std::is_same<T, float>::value ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16;
  constexpr libxsmm_datatype OUT_TYPE = IN_TYPE;
  constexpr libxsmm_datatype COMP_TYPE = LIBXSMM_DATATYPE_F32;

  switch (po.po_type) {
  case post_op_type_t::relu:
  case post_op_type_t::sigmoid:
  case post_op_type_t::tanh:
  case post_op_type_t::gelu_erf: {
    libxsmm_meltw_unary_type unary_type;
    switch (po.po_type) {
    case post_op_type_t::relu:
      unary_type = LIBXSMM_MELTW_TYPE_UNARY_RELU;
      break;
    case post_op_type_t::sigmoid:
      unary_type = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID;
      break;
    case post_op_type_t::tanh:
      unary_type = LIBXSMM_MELTW_TYPE_UNARY_TANH;
      break;
    case post_op_type_t::gelu_erf:
      unary_type = LIBXSMM_MELTW_TYPE_UNARY_GELU;
      break;
    default:
      return;
    }

    libxsmm_meltw_unary_shape shape = libxsmm_create_meltw_unary_shape(
                                        N, M, ldc, ldc,
                                        IN_TYPE,
                                        OUT_TYPE,
                                        COMP_TYPE
                                      );

    libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary(
                                           unary_type, shape, LIBXSMM_MELTW_FLAG_UNARY_NONE
                                         );

    if (kernel) {
      libxsmm_meltw_unary_param param{};
      param.in.primary = output;
      param.out.primary = output;
      kernel(&param);
    }
    else {
      log_error("Failed to dispatch LIBXSMM unary post-op kernel");
    }
    break;
  }

  case post_op_type_t::binary_mul:
  case post_op_type_t::binary_add: {
    libxsmm_meltw_binary_type binary_type =
      (po.po_type == post_op_type_t::binary_mul)
      ? LIBXSMM_MELTW_TYPE_BINARY_MUL
      : LIBXSMM_MELTW_TYPE_BINARY_ADD;
    libxsmm_meltw_binary_shape s{};
    s.m = N;
    s.n = M;
    s.ldi = ldc;
    s.ldi2 = po.leading_dim;
    s.ldo = ldc;
    s.in0_type = IN_TYPE;
    s.in1_type = COMP_TYPE;
    s.comp_type = COMP_TYPE;
    s.out_type = OUT_TYPE;

    libxsmm_meltwfunction_binary kernel = libxsmm_dispatch_meltw_binary(
                                            binary_type, s, LIBXSMM_MELTW_FLAG_BINARY_NONE
                                          );

    if (kernel) {
      libxsmm_meltw_binary_param param{};
      param.in0.primary = output;
      param.in1.primary = po.buff;
      param.out.primary = output;
      kernel(&param);
    }
    else {
      log_error("Failed to dispatch LIBXSMM binary post-op kernel");
    }
    break;
  }

  default:
    log_error("Unsupported post-op type");
    break;
  }
}

/**
 * @brief Apply bias addition on the output matrix using LIBXSMM kernels.
 *
 * @param M Number of rows in the output matrix.
 * @param N Number of columns in the output matrix.
 * @param ldc Leading dimension of the output matrix.
 * @param output Pointer to the output matrix where bias will be added.
 * @param bias Pointer to the bias vector (1D array of length N).
 * @tparam TA Output datatype (float or bfloat16)
 * @tparam TB Bias datatype (float or bfloat16)
 */
template<typename TA, typename TB = TA>
inline static void libxsmm_bias(const int M, const int N, const int ldc,
                                void *output, const void *bias) {
  constexpr libxsmm_datatype OUT_TYPE =
    std::is_same<TA, float>::value ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16;
  constexpr libxsmm_datatype BIAS_TYPE =
    std::is_same<TB, float>::value ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16;
  constexpr libxsmm_datatype COMP_TYPE = LIBXSMM_DATATYPE_F32;

  libxsmm_meltw_binary_shape s{};
  s.m = N;
  s.n = M;
  s.ldi = N;
  s.ldi2 = ldc;
  s.ldo = ldc;
  s.in0_type = BIAS_TYPE;
  s.in1_type = OUT_TYPE;
  s.comp_type = COMP_TYPE;
  s.out_type = OUT_TYPE;


  libxsmm_meltwfunction_binary kernel = libxsmm_dispatch_meltw_binary(
                                          LIBXSMM_MELTW_TYPE_BINARY_ADD,
                                          s,
                                          LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0
                                        );

  if (kernel) {
    libxsmm_meltw_binary_param param{};
    param.in0.primary = const_cast<void *>(bias);
    param.in1.primary = output;
    param.out.primary = output;
    kernel(&param);
  }
  else {
    log_error("Failed to dispatch LIBXSMM bias add kernel");
  }
}
#endif
}
}
#endif