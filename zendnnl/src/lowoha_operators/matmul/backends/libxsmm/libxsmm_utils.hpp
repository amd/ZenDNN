/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

#if ZENDNNL_DEPENDS_LIBXSMM
  #include "libxsmm.h"
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {

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
 * @param set_sizes_limit Whether to set size limits for the operation
 * @return true if LibXSMM can handle this operation, false otherwise
 */
static inline bool can_use_libxsmm(char transA, char transB, int M,
                                   int N, int K, float alpha, float beta,
                                   const matmul_params &lowoha_param,
                                   bool set_sizes_limit = true) {

#if ZENDNNL_DEPENDS_LIBXSMM

  // Early exit for unsupported scalar values
  if (alpha != 1.0f || (beta != 0.0f && beta != 1.0f)) {
    return false;
  }

  //LIBXSMM throws segfault for transA='t' cases
  if (transA == 't') {
    return false;
  }

  if (set_sizes_limit) {
    int64_t matrix_b_elements = static_cast<int64_t>(K) * N;
    if (matrix_b_elements > 1000000 || (M > 512 && matrix_b_elements > 1000000)) {
      return false;
    }
  }

  if (lowoha_param.postop_.size() > 0) {
    for (const auto &postop : lowoha_param.postop_) {
      switch (postop.po_type) {
      case post_op_type_t::binary_add:
      case post_op_type_t::binary_mul:
      case post_op_type_t::gelu_erf:
      case post_op_type_t::relu:
      case post_op_type_t::tanh:
      case post_op_type_t::sigmoid:
        continue;
      case post_op_type_t::swish:
        // SiLU only: alpha == 1.0 (or 0.0, treated as default 1.0 by callers).
        // Anything else falls back to OneDNN/DLP.
        if (postop.alpha != 0.0f && postop.alpha != 1.0f) {
          return false;
        }
        continue;
      default:
        return false;
      }
    }
  }

  return (lowoha_param.dtypes.src == data_type_t::f32  &&
          lowoha_param.dtypes.dst == data_type_t::f32) ||
         (lowoha_param.dtypes.src == data_type_t::bf16 &&
          lowoha_param.dtypes.dst == data_type_t::f32) ||
         (lowoha_param.dtypes.src == data_type_t::bf16 &&
          lowoha_param.dtypes.dst == data_type_t::bf16);

#endif
  return false;
}

#if ZENDNNL_DEPENDS_LIBXSMM

/**
 * @brief Compute buffer size (bytes) needed for a blocked weight matrix.
 *
 * Total element count equals the original (K * N); only the layout changes.
 */
static inline size_t libxsmm_weight_block_size(int K, int N,
    data_type_t dtype) {
  size_t elem_size = (dtype == data_type_t::f32) ? sizeof(float)
                     : sizeof(libxsmm_bfloat16);
  return static_cast<size_t>(K) * N * elem_size;
}

/**
 * @brief Reblock a 2D weight matrix into blocked layout.
 *
 * F32:  [K, N] -> [num_n_blocks, num_k_blocks, k_block_size, n_block_size]          (4D, contiguous tiles)
 * BF16: [K, N] -> [num_n_blocks, num_k_blocks, k_block_size/2, n_block_size, 2]     (5D, VNNI-packed pairs)
 *
 * Only full blocks are copied. Remainder elements
 * are NOT blocked — the caller should use the original weight for tails.
 *
 * @param weight         Source 2D weight [K, N] (transB=false) or [N, K] (transB=true)
 * @param blocked_buf    Caller-allocated output (>= libxsmm_weight_block_size bytes)
 * @param K              Reduction dimension
 * @param N              Output dimension
 * @param ldb            Leading dimension of source weight
 * @param k_block_size   K-block size
 * @param n_block_size   N-block size
 * @param dtype          f32 or bf16
 * @param transB         If true, source is [N, K] row-major (transposed)
 */
static inline void libxsmm_weight_block(
  const void *weight, void *blocked_buf,
  int K, int N, int ldb,
  int k_block_size, int n_block_size,
  data_type_t dtype, bool transB = false) {

  int num_k_blocks = K / k_block_size;
  int num_n_blocks = N / n_block_size;

  log_info("libxsmm_weight_block: [", K, ",", N, "] -> [",
           num_n_blocks, ",", num_k_blocks, ",", n_block_size, ",", k_block_size,
           "] dtype=", (dtype == data_type_t::f32) ? "f32" : "bf16",
           " transB=", transB);

  if (dtype == data_type_t::f32) {
    const float *src = static_cast<const float *>(weight);
    float *dst = static_cast<float *>(blocked_buf);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int nblk = 0; nblk < num_n_blocks; ++nblk) {
      for (int kblk = 0; kblk < num_k_blocks; ++kblk) {
        float *tile = dst + (static_cast<size_t>(nblk) * num_k_blocks + kblk) *
                      n_block_size * k_block_size;
        for (int ki = 0; ki < k_block_size; ++ki) {
          for (int ni = 0; ni < n_block_size; ++ni) {
            int k_idx = kblk * k_block_size + ki;
            int n_idx = nblk * n_block_size + ni;
            tile[ki * n_block_size + ni] = transB
                                           ? src[n_idx * ldb + k_idx]
                                           : src[k_idx * ldb + n_idx];
          }
        }
      }
    }
  }
  else {
    constexpr int VNNI = 2;
    const libxsmm_bfloat16 *src = static_cast<const libxsmm_bfloat16 *>(weight);
    libxsmm_bfloat16 *dst = static_cast<libxsmm_bfloat16 *>(blocked_buf);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int nblk = 0; nblk < num_n_blocks; ++nblk) {
      for (int kblk = 0; kblk < num_k_blocks; ++kblk) {
        libxsmm_bfloat16 *tile = dst +
                                 (static_cast<size_t>(nblk) * num_k_blocks + kblk) * n_block_size * k_block_size;
        for (int k_pair = 0; k_pair < k_block_size / VNNI; ++k_pair) {
          for (int ni = 0; ni < n_block_size; ++ni) {
            for (int v = 0; v < VNNI; ++v) {
              int ki = k_pair * VNNI + v;
              int k_idx = kblk * k_block_size + ki;
              int n_idx = nblk * n_block_size + ni;
              tile[k_pair * n_block_size * VNNI + ni * VNNI + v] = transB
                  ? src[n_idx * ldb + k_idx]
                  : src[k_idx * ldb + n_idx];
            }
          }
        }
      }
    }
  }
}

/**
 * @brief Apply post-operations (e.g., ReLU, Sigmoid, Tanh, GELU, SiLU/Swish, Binary Multiply, Binary Add)
 *        on the output matrix in F32/BF16 format using LIBXSMM kernels.
 *
 * @param M Number of rows in the output matrix.
 * @param N Number of columns in the output matrix.
 * @param ldc Leading dimension of the output matrix.
 * @param output Pointer to the output matrix where the post-operation will be applied.
 * @param po A `matmul_post_op` object containing the type of post-operation and any additional buffers required.
 *
 * Note: the SiLU/Swish path uses a function-local stack scratch buffer of
 * exactly M*N elements of type T. Each thread already owns its own stack,
 * so no shared state, indexing, or thread-id math is needed.
 */
template<typename T>
inline static void libxsmm_postop(const int M, const int N, const int ldc,
                                  void *output, const matmul_post_op &po) {
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

    libxsmm_meltwfunction_unary kernel =
      libxsmm_dispatch_meltw_unary(unary_type, shape,
                                   LIBXSMM_MELTW_FLAG_UNARY_NONE);

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
    libxsmm_datatype po_buff_type =
      (po.dtype == data_type_t::bf16) ? LIBXSMM_DATATYPE_BF16 : LIBXSMM_DATATYPE_F32;
    libxsmm_meltw_binary_shape s{};
    s.m = N;
    s.n = M;
    s.ldi = ldc;
    s.ldi2 = po.leading_dim;
    s.ldo = ldc;
    s.in0_type = IN_TYPE;
    s.in1_type = po_buff_type;
    s.comp_type = COMP_TYPE;
    s.out_type = OUT_TYPE;

    // LOWOHA binary tensor {1, N}: same multiplier for every output row at column c.
    // LibXSMM meltw stores C as out[i + j*ldo] with j = row, i = col; BCAST_ROW_IN_1
    // indexes in1 by j (varies down rows). BCAST_COL_IN_1 indexes in1 by i only
    // (constant along rows) — that matches a logical row vector broadcast.
    const bool bcast_col_in1 =
      po.dims.size() == 2 && po.dims[0] == 1;
    const libxsmm_bitfield bflags = bcast_col_in1
                                    ? LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1
                                    : LIBXSMM_MELTW_FLAG_BINARY_NONE;

    libxsmm_meltwfunction_binary kernel = libxsmm_dispatch_meltw_binary(binary_type,
                                          s, bflags);

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

  case post_op_type_t::swish: {
    // SiLU = swish with alpha == 1.0:  out = out * sigmoid(out)
    //
    // Implemented as two LIBXSMM kernels with a tile-local stack scratch.
    // Each thread already owns its own stack, so the buffer is automatically
    // private — no shared pool, no thread-id math, no synchronization.
    // Sized exactly to the current tile (M*N elements, <= a few KB on the
    // partitioner's BRGEMM_M_BLOCK*BRGEMM_N_BLOCK upper bound).
    //
    //   Step 1 (unary sigmoid):  scratch = sigmoid(output)   ldi=ldc, ldo=N
    //   Step 2 (binary mul):     output  = output * scratch  ldi=ldc, ldi2=N, ldo=ldc
    T scratch[M * N];

    libxsmm_meltw_unary_shape sig_shape = libxsmm_create_meltw_unary_shape(
                                            N, M, ldc, N,
                                            IN_TYPE,
                                            OUT_TYPE,
                                            COMP_TYPE
                                          );

    libxsmm_meltwfunction_unary sig_kernel =
      libxsmm_dispatch_meltw_unary(LIBXSMM_MELTW_TYPE_UNARY_SIGMOID, sig_shape,
                                   LIBXSMM_MELTW_FLAG_UNARY_NONE);

    if (!sig_kernel) {
      log_error("Failed to dispatch LIBXSMM sigmoid kernel for SiLU post-op");
      break;
    }

    libxsmm_meltw_unary_param sig_param{};
    sig_param.in.primary = output;
    sig_param.out.primary = scratch;
    sig_kernel(&sig_param);

    libxsmm_meltw_binary_shape mul_shape{};
    mul_shape.m = N;
    mul_shape.n = M;
    mul_shape.ldi = ldc;
    mul_shape.ldi2 = N;
    mul_shape.ldo = ldc;
    mul_shape.in0_type = IN_TYPE;
    mul_shape.in1_type = IN_TYPE;
    mul_shape.comp_type = COMP_TYPE;
    mul_shape.out_type = OUT_TYPE;

    libxsmm_meltwfunction_binary mul_kernel =
      libxsmm_dispatch_meltw_binary(LIBXSMM_MELTW_TYPE_BINARY_MUL, mul_shape,
                                    LIBXSMM_MELTW_FLAG_BINARY_NONE);

    if (!mul_kernel) {
      log_error("Failed to dispatch LIBXSMM binary-mul kernel for SiLU post-op");
      break;
    }

    libxsmm_meltw_binary_param mul_param{};
    mul_param.in0.primary = output;
    mul_param.in1.primary = scratch;
    mul_param.out.primary = output;
    mul_kernel(&mul_param);
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

  libxsmm_meltwfunction_binary kernel =
    libxsmm_dispatch_meltw_binary(LIBXSMM_MELTW_TYPE_BINARY_ADD, s,
                                  LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0);

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
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
#endif
