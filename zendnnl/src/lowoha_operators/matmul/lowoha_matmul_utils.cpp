/*******************************************************************************
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

#include "lowoha_matmul_utils.hpp"
#include <cmath>
#include <cstring>
#if ZENDNNL_DEPENDS_AOCLDLP
#include "aocl_dlp.h"
#else
#include "blis.h"
#endif

namespace zendnnl {
namespace lowoha {

// Helper function to create post_op structure for bias and post-ops
#if ZENDNNL_DEPENDS_AOCLDLP
dlp_metadata_t* create_dlp_post_op(const lowoha_params &lowoha_po, const void *bias, const data_types &dtypes, int N) {
  // Count total operations (bias + post-ops)
  int total_ops = (bias ? 1 : 0) + lowoha_po.postop_.size();

  if (total_ops == 0) {
    return nullptr;
  }

  dlp_metadata_t *dlp_metadata = static_cast<dlp_metadata_t*>(calloc(1, sizeof(dlp_metadata_t)));
  if (!dlp_metadata) {
    return nullptr;
  }

  // Initialize all pointers to null
  dlp_metadata->eltwise = nullptr;
  dlp_metadata->bias = nullptr;
  dlp_metadata->scale = nullptr;
  dlp_metadata->matrix_add = nullptr;
  dlp_metadata->matrix_mul = nullptr;
  dlp_metadata->pre_ops = nullptr;
  dlp_metadata->post_op_grp = nullptr;

  // Count different types of operations
  int eltwise_count = 0;
  int matrix_add_count = 0;
  int matrix_mul_count = 0;
  int bias_count = bias ? 1 : 0;

  // Count post-ops by type
  for (const auto &po : lowoha_po.postop_) {
    switch (po.po_type) {
      case post_op_type_t::relu:
      case post_op_type_t::leaky_relu:
      case post_op_type_t::gelu_tanh:
      case post_op_type_t::gelu_erf:
      case post_op_type_t::sigmoid:
      case post_op_type_t::swish:
      case post_op_type_t::tanh:
        eltwise_count++;
        break;
      case post_op_type_t::binary_add:
        matrix_add_count++;
        break;
      case post_op_type_t::binary_mul:
        matrix_mul_count++;
        break;
      default:
        // Skip unsupported post-ops
        break;
    }
  }

  // Allocate seq_vector first
  dlp_metadata->seq_vector = static_cast<DLP_POST_OP_TYPE*>(calloc(total_ops, sizeof(DLP_POST_OP_TYPE)));
  if (!dlp_metadata->seq_vector) {
    free(dlp_metadata);
    return nullptr;
  }

  // Allocate memory for different post-op types
  if (bias_count > 0) {
    dlp_metadata->bias = static_cast<dlp_post_op_bias*>(calloc(bias_count, sizeof(dlp_post_op_bias)));
    if (!dlp_metadata->bias) {
      free(dlp_metadata->seq_vector);
      free(dlp_metadata);
      return nullptr;
    }
  }

  if (eltwise_count > 0) {
    dlp_metadata->eltwise = static_cast<dlp_post_op_eltwise*>(calloc(eltwise_count, sizeof(dlp_post_op_eltwise)));
    if (!dlp_metadata->eltwise) {
      if (dlp_metadata->bias) free(dlp_metadata->bias);
      free(dlp_metadata->seq_vector);
      free(dlp_metadata);
      return nullptr;
    }
  }

  if (matrix_add_count > 0) {
    dlp_metadata->matrix_add = static_cast<dlp_post_op_matrix_add*>(calloc(matrix_add_count, sizeof(dlp_post_op_matrix_add)));
    if (!dlp_metadata->matrix_add) {
      if (dlp_metadata->bias) free(dlp_metadata->bias);
      if (dlp_metadata->eltwise) free(dlp_metadata->eltwise);
      free(dlp_metadata->seq_vector);
      free(dlp_metadata);
      return nullptr;
    }
    // Allocate sf structures for matrix_add operations
    for (int i = 0; i < matrix_add_count; ++i) {
      dlp_metadata->matrix_add[i].sf = static_cast<dlp_sf_t*>(calloc(1, sizeof(dlp_sf_t)));
      if (!dlp_metadata->matrix_add[i].sf) {
        // Clean up partially allocated sf structures
        for (int j = 0; j < i; ++j) {
          free(dlp_metadata->matrix_add[j].sf);
        }
        if (dlp_metadata->bias) free(dlp_metadata->bias);
        if (dlp_metadata->eltwise) free(dlp_metadata->eltwise);
        free(dlp_metadata->matrix_add);
        free(dlp_metadata->seq_vector);
        free(dlp_metadata);
        return nullptr;
      }
    }
  }

  if (matrix_mul_count > 0) {
    dlp_metadata->matrix_mul = static_cast<dlp_post_op_matrix_mul*>(calloc(matrix_mul_count, sizeof(dlp_post_op_matrix_mul)));
    if (!dlp_metadata->matrix_mul) {
      if (dlp_metadata->bias) free(dlp_metadata->bias);
      if (dlp_metadata->eltwise) free(dlp_metadata->eltwise);
      if (dlp_metadata->matrix_add) {
        for (int i = 0; i < matrix_add_count; ++i) {
          if (dlp_metadata->matrix_add[i].sf) free(dlp_metadata->matrix_add[i].sf);
        }
        free(dlp_metadata->matrix_add);
      }
      free(dlp_metadata->seq_vector);
      free(dlp_metadata);
      return nullptr;
    }
    // Allocate sf structures for matrix_mul operations
    for (int i = 0; i < matrix_mul_count; ++i) {
      dlp_metadata->matrix_mul[i].sf = static_cast<dlp_sf_t*>(calloc(1, sizeof(dlp_sf_t)));
      if (!dlp_metadata->matrix_mul[i].sf) {
        // Clean up partially allocated sf structures
        for (int j = 0; j < i; ++j) {
          free(dlp_metadata->matrix_mul[j].sf);
        }
        if (dlp_metadata->bias) free(dlp_metadata->bias);
        if (dlp_metadata->eltwise) free(dlp_metadata->eltwise);
        if (dlp_metadata->matrix_add) {
          for (int k = 0; k < matrix_add_count; ++k) {
            if (dlp_metadata->matrix_add[k].sf) free(dlp_metadata->matrix_add[k].sf);
          }
          free(dlp_metadata->matrix_add);
        }
        free(dlp_metadata->matrix_mul);
        free(dlp_metadata->seq_vector);
        free(dlp_metadata);
        return nullptr;
      }
    }
  }

  int op_index = 0;
  int eltwise_index = 0;
  int matrix_add_index = 0;
  int matrix_mul_index = 0;

  // Add bias if present
  if (bias && bias_count > 0) {
    dlp_metadata->seq_vector[op_index++] = BIAS;
    dlp_metadata->bias[0].bias = const_cast<void*>(bias);

    // Set storage type based on bias data type
    switch (dtypes.bias) {
      case data_type_t::f32:
        dlp_metadata->bias[0].stor_type = DLP_F32;
        break;
      case data_type_t::bf16:
        dlp_metadata->bias[0].stor_type = DLP_BF16;
        break;
      default:
        dlp_metadata->bias[0].stor_type = DLP_F32;
        break;
    }
    dlp_metadata->bias[0].sf = nullptr; // No scale factor for bias
  }

  // Add post-ops
  for (const auto &po : lowoha_po.postop_) {
    switch (po.po_type) {
      case post_op_type_t::relu:
        if (eltwise_index >= eltwise_count) break;
        dlp_metadata->seq_vector[op_index++] = ELTWISE;
        dlp_metadata->eltwise[eltwise_index].algo.algo_type = RELU;
        dlp_metadata->eltwise[eltwise_index].algo.alpha = nullptr;
        dlp_metadata->eltwise[eltwise_index].algo.beta = nullptr;
        dlp_metadata->eltwise[eltwise_index].sf = nullptr;
        eltwise_index++;
        break;
      case post_op_type_t::leaky_relu:
        if (eltwise_index >= eltwise_count) break;
        dlp_metadata->seq_vector[op_index++] = ELTWISE;
        dlp_metadata->eltwise[eltwise_index].algo.algo_type = PRELU;
        dlp_metadata->eltwise[eltwise_index].algo.alpha = malloc(sizeof(float));
        *static_cast<float*>(dlp_metadata->eltwise[eltwise_index].algo.alpha) = 0.01f;
        dlp_metadata->eltwise[eltwise_index].algo.beta = nullptr;
        dlp_metadata->eltwise[eltwise_index].sf = nullptr;
        eltwise_index++;
        break;
      case post_op_type_t::gelu_tanh:
        if (eltwise_index >= eltwise_count) break;
        dlp_metadata->seq_vector[op_index++] = ELTWISE;
        dlp_metadata->eltwise[eltwise_index].algo.algo_type = GELU_TANH;
        dlp_metadata->eltwise[eltwise_index].algo.alpha = nullptr;
        dlp_metadata->eltwise[eltwise_index].algo.beta = nullptr;
        dlp_metadata->eltwise[eltwise_index].sf = nullptr;
        eltwise_index++;
        break;
      case post_op_type_t::gelu_erf:
        if (eltwise_index >= eltwise_count) break;
        dlp_metadata->seq_vector[op_index++] = ELTWISE;
        dlp_metadata->eltwise[eltwise_index].algo.algo_type = GELU_ERF;
        dlp_metadata->eltwise[eltwise_index].algo.alpha = nullptr;
        dlp_metadata->eltwise[eltwise_index].algo.beta = nullptr;
        dlp_metadata->eltwise[eltwise_index].sf = nullptr;
        eltwise_index++;
        break;
      case post_op_type_t::sigmoid:
        if (eltwise_index >= eltwise_count) break;
        dlp_metadata->seq_vector[op_index++] = ELTWISE;
        dlp_metadata->eltwise[eltwise_index].algo.algo_type = SIGMOID;
        dlp_metadata->eltwise[eltwise_index].algo.alpha = nullptr;
        dlp_metadata->eltwise[eltwise_index].algo.beta = nullptr;
        dlp_metadata->eltwise[eltwise_index].sf = nullptr;
        eltwise_index++;
        break;
      case post_op_type_t::swish:
        if (eltwise_index >= eltwise_count) break;
        dlp_metadata->seq_vector[op_index++] = ELTWISE;
        dlp_metadata->eltwise[eltwise_index].algo.algo_type = SWISH;
        dlp_metadata->eltwise[eltwise_index].algo.alpha = malloc(sizeof(float));
        *static_cast<float*>(dlp_metadata->eltwise[eltwise_index].algo.alpha) = 1.0f;
        dlp_metadata->eltwise[eltwise_index].algo.beta = nullptr;
        dlp_metadata->eltwise[eltwise_index].sf = nullptr;
        eltwise_index++;
        break;
      case post_op_type_t::tanh:
        if (eltwise_index >= eltwise_count) break;
        dlp_metadata->seq_vector[op_index++] = ELTWISE;
        dlp_metadata->eltwise[eltwise_index].algo.algo_type = TANH;
        dlp_metadata->eltwise[eltwise_index].algo.alpha = nullptr;
        dlp_metadata->eltwise[eltwise_index].algo.beta = nullptr;
        dlp_metadata->eltwise[eltwise_index].sf = nullptr;
        eltwise_index++;
        break;
      case post_op_type_t::binary_add:
        if (matrix_add_index >= matrix_add_count) break;
        dlp_metadata->seq_vector[op_index++] = MATRIX_ADD;
        dlp_metadata->matrix_add[matrix_add_index].matrix = po.buff;
        dlp_metadata->matrix_add[matrix_add_index].ldm = N;
        dlp_metadata->matrix_add[matrix_add_index].stor_type = po.dtype == data_type_t::bf16 ? DLP_BF16 : DLP_F32;
        // sf structure is already allocated, initialize with default values
        dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor = malloc(sizeof(float));
        if (dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor) {
          *static_cast<float*>(dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor) = 1.0f;
        }
        dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor_len = 1;
        matrix_add_index++;
        break;
      case post_op_type_t::binary_mul:
        if (matrix_mul_index >= matrix_mul_count) break;
        dlp_metadata->seq_vector[op_index++] = MATRIX_MUL;
        dlp_metadata->matrix_mul[matrix_mul_index].matrix = po.buff;
        dlp_metadata->matrix_mul[matrix_mul_index].ldm = N;
        dlp_metadata->matrix_mul[matrix_mul_index].stor_type = po.dtype == data_type_t::bf16 ? DLP_BF16 : DLP_F32;
        // sf structure is already allocated, initialize with default values
        dlp_metadata->matrix_mul[matrix_mul_index].sf->scale_factor = malloc(sizeof(float));
        if (dlp_metadata->matrix_mul[matrix_mul_index].sf->scale_factor) {
          *static_cast<float*>(dlp_metadata->matrix_mul[matrix_mul_index].sf->scale_factor) = 1.0f;
        }
        dlp_metadata->matrix_mul[matrix_mul_index].sf->scale_factor_len = 1;
        matrix_mul_index++;
        break;
      default:
        // Skip unsupported post-ops
        break;
    }
  }

  dlp_metadata->seq_length = op_index;
  dlp_metadata->num_eltwise = eltwise_count;

  return dlp_metadata;
}
#else
aocl_post_op* create_blis_post_op(const lowoha_params &lowoha_po, const void *bias, const data_types &dtypes, int N) {
  // Count total operations (bias + post-ops)
  int total_ops = (bias ? 1 : 0) + lowoha_po.postop_.size();

  if (total_ops == 0) {
    return nullptr;
  }

  aocl_post_op *aocl_po = static_cast<aocl_post_op*>(calloc(1, sizeof(aocl_post_op)));
  if (!aocl_po) {
    return nullptr;
  }

  // Count different types of operations
  int eltwise_count = 0;
  int matrix_add_count = 0;
  int matrix_mul_count = 0;
  int bias_count = bias ? 1 : 0;

  // Count post-ops by type
  for (const auto &po : lowoha_po.postop_) {
    switch (po.po_type) {
      case post_op_type_t::relu:
      case post_op_type_t::leaky_relu:
      case post_op_type_t::gelu_tanh:
      case post_op_type_t::gelu_erf:
      case post_op_type_t::sigmoid:
      case post_op_type_t::swish:
      case post_op_type_t::tanh:
        eltwise_count++;
        break;
      case post_op_type_t::binary_add:
        matrix_add_count++;
        break;
      case post_op_type_t::binary_mul:
        matrix_mul_count++;
        break;
      default:
        // Skip unsupported post-ops
        break;
    }
  }

  // Allocate memory for different post-op types
  if (bias_count > 0) {
    aocl_po->bias = static_cast<aocl_post_op_bias*>(calloc(bias_count, sizeof(aocl_post_op_bias)));
    if (!aocl_po->bias) {
      free(aocl_po);
      return nullptr;
    }
  }

  if (eltwise_count > 0) {
    aocl_po->eltwise = static_cast<aocl_post_op_eltwise*>(calloc(eltwise_count, sizeof(aocl_post_op_eltwise)));
    if (!aocl_po->eltwise) {
      if (aocl_po->bias) free(aocl_po->bias);
      free(aocl_po);
      return nullptr;
    }
  }

  if (matrix_add_count > 0) {
    aocl_po->matrix_add = static_cast<aocl_post_op_matrix_add*>(calloc(matrix_add_count, sizeof(aocl_post_op_matrix_add)));
    if (!aocl_po->matrix_add) {
      if (aocl_po->bias) free(aocl_po->bias);
      if (aocl_po->eltwise) free(aocl_po->eltwise);
      free(aocl_po);
      return nullptr;
    }
  }

  if (matrix_mul_count > 0) {
    aocl_po->matrix_mul = static_cast<aocl_post_op_matrix_mul*>(calloc(matrix_mul_count, sizeof(aocl_post_op_matrix_mul)));
    if (!aocl_po->matrix_mul) {
      if (aocl_po->bias) free(aocl_po->bias);
      if (aocl_po->eltwise) free(aocl_po->eltwise);
      if (aocl_po->matrix_add) free(aocl_po->matrix_add);
      free(aocl_po);
      return nullptr;
    }
  }

  // Set up sequence vector
  aocl_po->seq_vector = static_cast<AOCL_POST_OP_TYPE*>(calloc(total_ops, sizeof(AOCL_POST_OP_TYPE)));
  if (!aocl_po->seq_vector) {
    if (aocl_po->bias) free(aocl_po->bias);
    if (aocl_po->eltwise) free(aocl_po->eltwise);
    if (aocl_po->matrix_add) free(aocl_po->matrix_add);
    if (aocl_po->matrix_mul) free(aocl_po->matrix_mul);
    free(aocl_po);
    return nullptr;
  }

  int op_index = 0;
  int eltwise_index = 0;
  int matrix_add_index = 0;
  int matrix_mul_index = 0;

  // Add bias if present
  if (bias) {
    aocl_po->seq_vector[op_index++] = BIAS;
    aocl_po->bias[0].bias = const_cast<void*>(bias);

    // Set storage type based on bias data type
    switch (dtypes.bias) {
      case data_type_t::f32:
        aocl_po->bias[0].stor_type = AOCL_GEMM_F32;
        break;
      case data_type_t::bf16:
        aocl_po->bias[0].stor_type = AOCL_GEMM_BF16;
        break;
      default:
        aocl_po->bias[0].stor_type = AOCL_GEMM_F32;
        break;
    }
  }

  // Add post-ops
  for (const auto &po : lowoha_po.postop_) {
    switch (po.po_type) {
      case post_op_type_t::relu:
        aocl_po->seq_vector[op_index++] = ELTWISE;
        aocl_po->eltwise[eltwise_index].algo.algo_type = RELU;
        aocl_po->eltwise[eltwise_index].algo.alpha = nullptr;
        aocl_po->eltwise[eltwise_index].algo.beta = nullptr;
        aocl_po->eltwise[eltwise_index].is_power_of_2 = false;
        aocl_po->eltwise[eltwise_index].scale_factor = nullptr;
        aocl_po->eltwise[eltwise_index].scale_factor_len = 0;
        eltwise_index++;
        break;
      case post_op_type_t::leaky_relu:
        aocl_po->seq_vector[op_index++] = ELTWISE;
        aocl_po->eltwise[eltwise_index].algo.algo_type = PRELU;
        aocl_po->eltwise[eltwise_index].algo.alpha = malloc(sizeof(float));
        // Use default slope of 0.01 for leaky_relu
        *static_cast<float*>(aocl_po->eltwise[eltwise_index].algo.alpha) = 0.01f;
        aocl_po->eltwise[eltwise_index].algo.beta = nullptr;
        aocl_po->eltwise[eltwise_index].is_power_of_2 = false;
        aocl_po->eltwise[eltwise_index].scale_factor = nullptr;
        aocl_po->eltwise[eltwise_index].scale_factor_len = 0;
        eltwise_index++;
        break;
      case post_op_type_t::gelu_tanh:
        aocl_po->seq_vector[op_index++] = ELTWISE;
        aocl_po->eltwise[eltwise_index].algo.algo_type = GELU_TANH;
        aocl_po->eltwise[eltwise_index].algo.alpha = nullptr;
        aocl_po->eltwise[eltwise_index].algo.beta = nullptr;
        aocl_po->eltwise[eltwise_index].is_power_of_2 = false;
        aocl_po->eltwise[eltwise_index].scale_factor = nullptr;
        aocl_po->eltwise[eltwise_index].scale_factor_len = 0;
        eltwise_index++;
        break;
      case post_op_type_t::gelu_erf:
        aocl_po->seq_vector[op_index++] = ELTWISE;
        aocl_po->eltwise[eltwise_index].algo.algo_type = GELU_ERF;
        aocl_po->eltwise[eltwise_index].algo.alpha = nullptr;
        aocl_po->eltwise[eltwise_index].algo.beta = nullptr;
        aocl_po->eltwise[eltwise_index].is_power_of_2 = false;
        aocl_po->eltwise[eltwise_index].scale_factor = nullptr;
        aocl_po->eltwise[eltwise_index].scale_factor_len = 0;
        eltwise_index++;
        break;
      case post_op_type_t::sigmoid:
        aocl_po->seq_vector[op_index++] = ELTWISE;
        aocl_po->eltwise[eltwise_index].algo.algo_type = SIGMOID;
        aocl_po->eltwise[eltwise_index].algo.alpha = nullptr;
        aocl_po->eltwise[eltwise_index].algo.beta = nullptr;
        aocl_po->eltwise[eltwise_index].is_power_of_2 = false;
        aocl_po->eltwise[eltwise_index].scale_factor = nullptr;
        aocl_po->eltwise[eltwise_index].scale_factor_len = 0;
        eltwise_index++;
        break;
      case post_op_type_t::swish:
        aocl_po->seq_vector[op_index++] = ELTWISE;
        aocl_po->eltwise[eltwise_index].algo.algo_type = SWISH;
        aocl_po->eltwise[eltwise_index].algo.alpha = malloc(sizeof(float));
        // Use default scale of 1.0 for swish
        *static_cast<float*>(aocl_po->eltwise[eltwise_index].algo.alpha) = 1.0f;
        aocl_po->eltwise[eltwise_index].algo.beta = nullptr;
        aocl_po->eltwise[eltwise_index].is_power_of_2 = false;
        aocl_po->eltwise[eltwise_index].scale_factor = nullptr;
        aocl_po->eltwise[eltwise_index].scale_factor_len = 0;
        eltwise_index++;
        break;
      case post_op_type_t::tanh:
        aocl_po->seq_vector[op_index++] = ELTWISE;
        aocl_po->eltwise[eltwise_index].algo.algo_type = TANH;
        aocl_po->eltwise[eltwise_index].algo.alpha = nullptr;
        aocl_po->eltwise[eltwise_index].algo.beta = nullptr;
        aocl_po->eltwise[eltwise_index].is_power_of_2 = false;
        aocl_po->eltwise[eltwise_index].scale_factor = nullptr;
        aocl_po->eltwise[eltwise_index].scale_factor_len = 0;
        eltwise_index++;
        break;
      case post_op_type_t::binary_add:
        aocl_po->seq_vector[op_index++] = MATRIX_ADD;
        aocl_po->matrix_add[matrix_add_index].matrix = po.buff;
        aocl_po->matrix_add[matrix_add_index].scale_factor = malloc(sizeof(float));
        *static_cast<float*>(aocl_po->matrix_add[matrix_add_index].scale_factor) = 1.0f; // Default scale
        aocl_po->matrix_add[matrix_add_index].scale_factor_len = 1;
        aocl_po->matrix_add[matrix_add_index].ldm = N; // Set leading dimension to N
        aocl_po->matrix_add[matrix_add_index].stor_type = po.dtype == data_type_t::bf16 ? AOCL_GEMM_BF16 : AOCL_GEMM_F32;
        matrix_add_index++;
        break;
      case post_op_type_t::binary_mul:
        aocl_po->seq_vector[op_index++] = MATRIX_MUL;
        aocl_po->matrix_mul[matrix_mul_index].matrix = po.buff;
        aocl_po->matrix_mul[matrix_mul_index].scale_factor = malloc(sizeof(float));
        *static_cast<float*>(aocl_po->matrix_mul[matrix_mul_index].scale_factor) = 1.0f; // Default scale
        aocl_po->matrix_mul[matrix_mul_index].scale_factor_len = 1;
        aocl_po->matrix_mul[matrix_mul_index].ldm = N; // Set leading dimension to N
        aocl_po->matrix_mul[matrix_mul_index].stor_type = po.dtype == data_type_t::bf16 ? AOCL_GEMM_BF16 : AOCL_GEMM_F32;
        matrix_mul_index++;
        break;
      default:
        // Skip unsupported post-ops
        break;
    }
  }

  aocl_po->seq_length = op_index;
  aocl_po->num_eltwise = eltwise_count;

  return aocl_po;
}
#endif
// Cleanup functions for post-op structures
#if ZENDNNL_DEPENDS_AOCLDLP
void cleanup_dlp_post_op(dlp_metadata_t *aocl_po, const lowoha_params &post_op) {
  if (aocl_po) {
    // Count operations for proper cleanup
    int eltwise_count = 0;
    int matrix_add_count = 0;
    int matrix_mul_count = 0;

    // Count post-ops by type for cleanup
    for (const auto &po : post_op.postop_) {
      switch (po.po_type) {
        case post_op_type_t::relu:
        case post_op_type_t::leaky_relu:
        case post_op_type_t::gelu_tanh:
        case post_op_type_t::gelu_erf:
        case post_op_type_t::sigmoid:
        case post_op_type_t::swish:
        case post_op_type_t::tanh:
          eltwise_count++;
          break;
        case post_op_type_t::binary_add:
          matrix_add_count++;
          break;
        case post_op_type_t::binary_mul:
          matrix_mul_count++;
          break;
        default:
          break;
      }
    }

    // Clean up eltwise operations
    if (aocl_po->eltwise) {
      for (int i = 0; i < eltwise_count; i++) {
        if (aocl_po->eltwise[i].algo.alpha) {
          free(aocl_po->eltwise[i].algo.alpha);
        }
        if (aocl_po->eltwise[i].algo.beta) {
          free(aocl_po->eltwise[i].algo.beta);
        }
      }
      free(aocl_po->eltwise);
    }

    // Clean up bias operations
    if (aocl_po->bias) {
      free(aocl_po->bias);
    }

    // Clean up matrix operations
    if (aocl_po->matrix_add) {
      for (int i = 0; i < matrix_add_count; i++) {
        if (aocl_po->matrix_add[i].sf) {
          if (aocl_po->matrix_add[i].sf->scale_factor) {
            free(aocl_po->matrix_add[i].sf->scale_factor);
          }
          free(aocl_po->matrix_add[i].sf);
        }
      }
      free(aocl_po->matrix_add);
    }

    if (aocl_po->matrix_mul) {
      for (int i = 0; i < matrix_mul_count; i++) {
        if (aocl_po->matrix_mul[i].sf) {
          if (aocl_po->matrix_mul[i].sf->scale_factor) {
            free(aocl_po->matrix_mul[i].sf->scale_factor);
          }
          free(aocl_po->matrix_mul[i].sf);
        }
      }
      free(aocl_po->matrix_mul);
    }

    if (aocl_po->scale) {
      free(aocl_po->scale);
    }

    if (aocl_po->pre_ops) {
      free(aocl_po->pre_ops);
    }

    if (aocl_po->post_op_grp) {
      free(aocl_po->post_op_grp);
    }

    if (aocl_po->seq_vector) {
      free(aocl_po->seq_vector);
    }
    free(aocl_po);
  }
}
#else
void cleanup_blis_post_op(aocl_post_op *aocl_po, const lowoha_params &post_op) {
  if (aocl_po) {
    // Clean up eltwise operations
    if (aocl_po->eltwise) {
      for (int i = 0; i < aocl_po->num_eltwise; i++) {
        if (aocl_po->eltwise[i].algo.alpha) {
          free(aocl_po->eltwise[i].algo.alpha);
        }
        if (aocl_po->eltwise[i].algo.beta) {
          free(aocl_po->eltwise[i].algo.beta);
        }
      }
      free(aocl_po->eltwise);
    }

    // Clean up matrix operations
    if (aocl_po->matrix_add) {
      for (int i = 0; i < (post_op.postop_.size() > 0 ? 1 : 0); i++) {
        if (aocl_po->matrix_add[i].scale_factor) {
          free(aocl_po->matrix_add[i].scale_factor);
        }
      }
      free(aocl_po->matrix_add);
    }

    if (aocl_po->matrix_mul) {
      for (int i = 0; i < (post_op.postop_.size() > 0 ? 1 : 0); i++) {
        if (aocl_po->matrix_mul[i].scale_factor) {
          free(aocl_po->matrix_mul[i].scale_factor);
        }
      }
      free(aocl_po->matrix_mul);
    }

    if (aocl_po->bias) {
      free(aocl_po->bias);
    }
    if (aocl_po->seq_vector) {
      free(aocl_po->seq_vector);
    }
    free(aocl_po);
  }
}
#endif
} // namespace lowoha
} // namespace zendnnl