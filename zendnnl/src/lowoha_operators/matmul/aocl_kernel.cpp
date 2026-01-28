/*******************************************************************************
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

#include "lowoha_operators/matmul/aocl_kernel.hpp"
#include "lowoha_operators/matmul/lowoha_cache.hpp"
#include <cstdlib>
#include <cstring>

namespace zendnnl {
namespace lowoha {
namespace matmul {

// Helper function to compute number of elements from dimension vector
// Returns 1 for empty dims (per-tensor case) specific for DLP use case,
// or product of all dims
static inline size_t get_num_elements(const std::vector<int64_t> &dims) {
  if (dims.empty()) {
    return 1;
  }
  size_t count = 1;
  for (auto d : dims) {
    count *= static_cast<size_t>(d);
  }
  return count;
}

// Extract nibble from packed S4 byte (low nibble if is_low_nibble=true, else high nibble)
inline int8_t extract_s4_nibble(int8_t packed_byte, bool is_low_nibble) {
  int8_t s4_value = is_low_nibble ? (packed_byte & 0x0F) : ((
                      packed_byte >> 4) & 0x0F);
  // Sign extend from bit 3
  if (s4_value & 0x08) {
    s4_value |= 0xF0;
  }
  return s4_value;
}

void cvt_int4_to_bf16(const int8_t *weights, bfloat16_t *wei_bf16, int k, int n,
                      int ldb, bool is_transposed,
                      const void *scales, const std::vector<int64_t> &scale_dims,
                      data_type_t scale_dt,
                      const void *zp, const std::vector<int64_t> &zp_dims, data_type_t zp_dt) {
  const size_t scale_size = scale_dims.size() > 0 ? get_num_elements(
                              scale_dims) : 0;
  const size_t zp_size = zp_dims.size() > 0 ? get_num_elements(zp_dims) : 0;

  // Cache per-tensor scale/zp values for efficiency
  const float per_tensor_scale = (scale_size == 1) ? read_and_cast<float>(scales,
                                 scale_dt, 0) : 0.0f;
  const float per_tensor_zp = (zp_size == 1) ? read_and_cast<float>(zp, zp_dt,
                              0) : 0.0f;

  // Determine quantization granularity for group size calculation
  int num_groups = 1;
  int group_size = k;
  if (scale_size > 1 && scale_size != static_cast<size_t>(n)) {
    // Per-group quantization: scale_size = G * N
    num_groups = scale_size / n;
    if (num_groups > 0) {
      group_size = k / num_groups;
    }
  }

  // Helper lambda to compute scale/zp offset based on logical (row, col) position
  // For per-tensor:  offset = 0
  // For per-channel: offset = col (N dimension)
  // For per-group:   offset = group * n + col
  auto compute_quant_offset = [&](int row, int col, size_t qsize) -> size_t {
    if (qsize == 1) {
      return 0;  // Per-tensor
    }
    else if (qsize == static_cast<size_t>(n)) {
      return col;  // Per-channel
    }
    else {
      // Per-group
      int group = row / group_size;
      return group * n + col;
    }
  };

  // Iterate over logical K×N elements
  // Output is always in K×N layout for subsequent reorder
  #pragma omp parallel for collapse(2)
  for (int row = 0; row < k; ++row) {
    for (int col = 0; col < n; ++col) {
      // Calculate physical index in packed S4 buffer based on transpose
      // For non-transposed (ab): physical element at row*ldb + col
      // For transposed (ba):     physical element at col*ldb + row
      size_t physical_idx = is_transposed ?
                            (static_cast<size_t>(col) * ldb + row) :
                            (static_cast<size_t>(row) * ldb + col);
      size_t packed_byte_idx = physical_idx / 2;
      bool is_low_nibble = (physical_idx % 2) == 0;

      // Extract S4 value
      int8_t s4_value = extract_s4_nibble(weights[packed_byte_idx], is_low_nibble);

      // Apply zero-point
      float dequant_value;
      if (zp_size == 1) {
        dequant_value = static_cast<float>(s4_value) - per_tensor_zp;
      }
      else if (zp_size > 1) {
        size_t zp_offset = compute_quant_offset(row, col, zp_size);
        dequant_value = static_cast<float>(s4_value) - read_and_cast<float>(zp, zp_dt,
                        zp_offset);
      }
      else {
        dequant_value = static_cast<float>(s4_value);
      }

      // Apply scale
      if (scale_size == 1) {
        dequant_value *= per_tensor_scale;
      }
      else {
        size_t scale_offset = compute_quant_offset(row, col, scale_size);
        dequant_value *= read_and_cast<float>(scales, scale_dt, scale_offset);
      }

      // Store in K×N output layout (row-major, non-transposed)
      size_t out_idx = static_cast<size_t>(row) * n + col;
      wei_bf16[out_idx] = bfloat16_t(dequant_value);
    }
  }
}
// Helper function to setup post-ops (eltwise, binary_add, binary_mul)
#if ZENDNNL_DEPENDS_AOCLDLP
static void setup_dlp_postops(dlp_metadata_t *dlp_metadata,
                              const std::vector<matmul_post_op> &postops,
                              int &op_index, int &eltwise_index,
                              int &matrix_add_index, int &matrix_mul_index,
                              int eltwise_count, int matrix_add_count,
                              int matrix_mul_count) {
  for (const auto &po : postops) {
    switch (po.po_type) {
    case post_op_type_t::relu:
      if (eltwise_index >= eltwise_count) {
        break;
      }
      dlp_metadata->seq_vector[op_index++] = ELTWISE;
      dlp_metadata->eltwise[eltwise_index].algo.algo_type = RELU;
      dlp_metadata->eltwise[eltwise_index].algo.alpha = nullptr;
      dlp_metadata->eltwise[eltwise_index].algo.beta = nullptr;
      dlp_metadata->eltwise[eltwise_index].sf = nullptr;
      eltwise_index++;
      break;
    case post_op_type_t::leaky_relu:
      if (eltwise_index >= eltwise_count) {
        break;
      }
      dlp_metadata->seq_vector[op_index++] = ELTWISE;
      dlp_metadata->eltwise[eltwise_index].algo.algo_type = PRELU;
      dlp_metadata->eltwise[eltwise_index].algo.alpha = malloc(sizeof(float));
      *static_cast<float *>(dlp_metadata->eltwise[eltwise_index].algo.alpha) = 0.01f;
      dlp_metadata->eltwise[eltwise_index].algo.beta = nullptr;
      dlp_metadata->eltwise[eltwise_index].sf = nullptr;
      eltwise_index++;
      break;
    case post_op_type_t::gelu_tanh:
      if (eltwise_index >= eltwise_count) {
        break;
      }
      dlp_metadata->seq_vector[op_index++] = ELTWISE;
      dlp_metadata->eltwise[eltwise_index].algo.algo_type = GELU_TANH;
      dlp_metadata->eltwise[eltwise_index].algo.alpha = nullptr;
      dlp_metadata->eltwise[eltwise_index].algo.beta = nullptr;
      dlp_metadata->eltwise[eltwise_index].sf = nullptr;
      eltwise_index++;
      break;
    case post_op_type_t::gelu_erf:
      if (eltwise_index >= eltwise_count) {
        break;
      }
      dlp_metadata->seq_vector[op_index++] = ELTWISE;
      dlp_metadata->eltwise[eltwise_index].algo.algo_type = GELU_ERF;
      dlp_metadata->eltwise[eltwise_index].algo.alpha = nullptr;
      dlp_metadata->eltwise[eltwise_index].algo.beta = nullptr;
      dlp_metadata->eltwise[eltwise_index].sf = nullptr;
      eltwise_index++;
      break;
    case post_op_type_t::sigmoid:
      if (eltwise_index >= eltwise_count) {
        break;
      }
      dlp_metadata->seq_vector[op_index++] = ELTWISE;
      dlp_metadata->eltwise[eltwise_index].algo.algo_type = SIGMOID;
      dlp_metadata->eltwise[eltwise_index].algo.alpha = nullptr;
      dlp_metadata->eltwise[eltwise_index].algo.beta = nullptr;
      dlp_metadata->eltwise[eltwise_index].sf = nullptr;
      eltwise_index++;
      break;
    case post_op_type_t::swish:
      if (eltwise_index >= eltwise_count) {
        break;
      }
      dlp_metadata->seq_vector[op_index++] = ELTWISE;
      dlp_metadata->eltwise[eltwise_index].algo.algo_type = SWISH;
      dlp_metadata->eltwise[eltwise_index].algo.alpha = malloc(sizeof(float));
      *static_cast<float *>(dlp_metadata->eltwise[eltwise_index].algo.alpha) = 1.0f;
      dlp_metadata->eltwise[eltwise_index].algo.beta = nullptr;
      dlp_metadata->eltwise[eltwise_index].sf = nullptr;
      eltwise_index++;
      break;
    case post_op_type_t::tanh:
      if (eltwise_index >= eltwise_count) {
        break;
      }
      dlp_metadata->seq_vector[op_index++] = ELTWISE;
      dlp_metadata->eltwise[eltwise_index].algo.algo_type = TANH;
      dlp_metadata->eltwise[eltwise_index].algo.alpha = nullptr;
      dlp_metadata->eltwise[eltwise_index].algo.beta = nullptr;
      dlp_metadata->eltwise[eltwise_index].sf = nullptr;
      eltwise_index++;
      break;
    case post_op_type_t::binary_add:
      if (matrix_add_index >= matrix_add_count) {
        break;
      }
      dlp_metadata->seq_vector[op_index++] = MATRIX_ADD;
      dlp_metadata->matrix_add[matrix_add_index].matrix = po.buff;
      dlp_metadata->matrix_add[matrix_add_index].ldm = po.leading_dim;
      dlp_metadata->matrix_add[matrix_add_index].stor_type = po.dtype ==
          data_type_t::bf16 ? DLP_BF16 : DLP_F32;
      // sf structure is already allocated, initialize with default values
      dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor = malloc(sizeof(
            float));
      if (dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor) {
        *static_cast<float *>
        (dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor) = 1.0f;
      }
      dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor_len = 1;
      matrix_add_index++;
      break;
    case post_op_type_t::binary_mul:
      if (matrix_mul_index >= matrix_mul_count) {
        break;
      }
      dlp_metadata->seq_vector[op_index++] = MATRIX_MUL;
      dlp_metadata->matrix_mul[matrix_mul_index].matrix = po.buff;
      dlp_metadata->matrix_mul[matrix_mul_index].ldm = po.leading_dim;
      dlp_metadata->matrix_mul[matrix_mul_index].stor_type = po.dtype ==
          data_type_t::bf16 ? DLP_BF16 : DLP_F32;
      // sf structure is already allocated, initialize with default values
      dlp_metadata->matrix_mul[matrix_mul_index].sf->scale_factor = malloc(sizeof(
            float));
      if (dlp_metadata->matrix_mul[matrix_mul_index].sf->scale_factor) {
        *static_cast<float *>
        (dlp_metadata->matrix_mul[matrix_mul_index].sf->scale_factor) = 1.0f;
      }
      dlp_metadata->matrix_mul[matrix_mul_index].sf->scale_factor_len = 1;
      matrix_mul_index++;
      break;
    default:
      // Skip unsupported post-ops
      break;
    }
  }
}

// Helper function to setup pre-ops for WOQ (Weight-Only Quantization)
static void setup_woq_pre_ops(dlp_metadata_t *dlp_metadata,
                              const matmul_params &lowoha_param,
                              int64_t K, int64_t N) {
  dlp_metadata->pre_ops = static_cast<dlp_pre_op *>(malloc(sizeof(dlp_pre_op)));
  if (!dlp_metadata->pre_ops) {
    return;
  }

  const auto &wei_scale = lowoha_param.quant_params.wei_scale;
  const auto &wei_zp = lowoha_param.quant_params.wei_zp;

  // Setup weight scale factor
  dlp_metadata->pre_ops->b_scl = static_cast<dlp_sf_t *>(malloc(sizeof(
                                   dlp_sf_t)));
  if (dlp_metadata->pre_ops->b_scl) {
    size_t scale_len = get_num_elements(wei_scale.dims);
    dlp_metadata->pre_ops->b_scl->scale_factor = const_cast<void *>(wei_scale.buff);
    dlp_metadata->pre_ops->b_scl->scale_factor_len = scale_len;
    dlp_metadata->pre_ops->b_scl->scale_factor_type =
      (wei_scale.dt == data_type_t::bf16) ? DLP_BF16 : DLP_F32;
  }

  // Setup weight zero-point
  dlp_metadata->pre_ops->b_zp = static_cast<dlp_zp_t *>(malloc(sizeof(dlp_zp_t)));
  if (dlp_metadata->pre_ops->b_zp) {
    dlp_metadata->pre_ops->b_zp->zero_point = const_cast<void *>(wei_zp.buff);
    dlp_metadata->pre_ops->b_zp->zero_point_len = wei_zp.buff != nullptr ?
        get_num_elements(wei_zp.dims) : 0;
    dlp_metadata->pre_ops->b_zp->zero_point_type = DLP_S8;
  }

  dlp_metadata->pre_ops->seq_length = 1;

  // Determine group_size from scale dimensions
  // wei_scale.dims determines granularity:
  //   - Per-tensor:  dims = {} or {1}     → group_size = K
  //   - Per-channel: dims = {1, N}        → group_size = K
  //   - Per-group:   dims = {G, N}        → group_size = K / G
  int64_t group_size = K;  // Default per-tensor
  const auto &dims = wei_scale.dims;
  if (!dims.empty() && !(dims.size() == 1 && dims[0] == 1)) {
    // Not per-tensor, check for per-group
    if (dims.size() == 2 && dims[1] == N && dims[0] > 1) {
      group_size = K / dims[0];  // Per-group: dims = {G, N}
    }
    // Per-channel (dims={N} or {1,N}) keeps group_size = K
  }

  // Validation: group_size must divide K evenly
  if (K % group_size != 0) {
    log_error("WOQ: group_size (", group_size, ") must divide K (", K, ") evenly");
    group_size = K;  // Fallback to per-tensor
  }

  dlp_metadata->pre_ops->group_size = static_cast<int>(group_size);

  apilog_info("WOQ: scale_len=", get_num_elements(wei_scale.dims),
              ", group_size=",
              group_size);
}

// Helper function to create post_op structure for bias and post-ops
dlp_metadata_t *create_dlp_post_op(const matmul_params &lowoha_param,
                                   const void *bias, const matmul_data_types &dtypes, int N, int K,
                                   int M, int32_t *zp_comp_acc, int zp_comp_ndim,
                                   zendnnl::ops::matmul_algo_t kernel) {

  // Check if this is a WOQ case (need metadata even if no post-ops)
  bool is_woq = dtypes.wei == data_type_t::s4 &&
                dtypes.src == data_type_t::bf16 &&
                kernel == zendnnl::ops::matmul_algo_t::aocl_dlp_blocked;

  // Check if this is INT8 quantization case
  bool is_int8 = (dtypes.src == data_type_t::u8 ||
                  dtypes.src == data_type_t::s8) &&
                 dtypes.wei == data_type_t::s8;

  //Count INT8 scale post-ops
  int int8_scale_count = 0;
  if (is_int8) {
    if (lowoha_param.quant_params.src_scale.buff) {
      int8_scale_count++;
    }
    if (lowoha_param.quant_params.wei_scale.buff) {
      int8_scale_count++;
    }
    if (lowoha_param.quant_params.dst_scale.buff ||
        lowoha_param.quant_params.dst_zp.buff) {
      int8_scale_count++;
    }
  }

  // Count total operations (bias + post-ops + scales + zp_comp)
  int total_ops = (bias ? 1 : 0) + lowoha_param.postop_.size() + int8_scale_count;

  // Add zero-point compensation to total ops
  if (zp_comp_ndim > 0) {
    total_ops++;
  }

  if (total_ops == 0 && !is_woq && !is_int8) {
    return nullptr;
  }

  dlp_metadata_t *dlp_metadata = static_cast<dlp_metadata_t *>(calloc(1,
                                 sizeof(dlp_metadata_t)));
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
  int scale_count = 0;

  // For INT8, add scale count
  if (is_int8) {
    scale_count = int8_scale_count;
  }

  // Add zp_comp to appropriate count
  if (zp_comp_ndim == 1) {
    bias_count++;  // 1D compensation is added as bias
  }
  else if (zp_comp_ndim == 2) {
    matrix_add_count++;  // 2D compensation is added as matrix_add
  }

  // Count post-ops by type
  for (const auto &po : lowoha_param.postop_) {
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

  // Helper lambda to convert data_type_t to DLP_TYPE
  auto to_dlp_type = [](data_type_t dt) -> DLP_TYPE {
    switch (dt) {
    case data_type_t::f32:
      return DLP_F32;
    case data_type_t::bf16:
      return DLP_BF16;
    case data_type_t::s32:
      return DLP_S32;
    case data_type_t::s8:
      return DLP_S8;
    case data_type_t::u8:
      return DLP_U8;
    default:
      return DLP_F32;
    }
  };

  // Allocate seq_vector first (only if we have post-ops)
  if (total_ops > 0) {
    dlp_metadata->seq_vector = static_cast<DLP_POST_OP_TYPE *>(calloc(total_ops,
                               sizeof(DLP_POST_OP_TYPE)));
    if (!dlp_metadata->seq_vector) {
      free(dlp_metadata);
      return nullptr;
    }
  }
  else {
    dlp_metadata->seq_vector = nullptr;
  }

  // Allocate scale for INT8
  if (scale_count > 0) {
    dlp_metadata->scale = static_cast<dlp_scale_t *>(calloc(scale_count,
                          sizeof(dlp_scale_t)));
    if (!dlp_metadata->scale) {
      if (dlp_metadata->seq_vector) {
        free(dlp_metadata->seq_vector);
      }
      free(dlp_metadata);
      return nullptr;
    }
    // Allocate nested sf and zp structures for each scale
    for (int i = 0; i < scale_count; ++i) {
      dlp_metadata->scale[i].sf = static_cast<dlp_sf_t *>(calloc(1,
                                  sizeof(dlp_sf_t)));
      dlp_metadata->scale[i].zp = static_cast<dlp_zp_t *>(calloc(1,
                                  sizeof(dlp_zp_t)));
      if (!dlp_metadata->scale[i].sf || !dlp_metadata->scale[i].zp) {
        // Cleanup on failure
        for (int j = 0; j <= i; ++j) {
          if (dlp_metadata->scale[j].sf) {
            free(dlp_metadata->scale[j].sf);
          }
          if (dlp_metadata->scale[j].zp) {
            free(dlp_metadata->scale[j].zp);
          }
        }
        free(dlp_metadata->scale);
        if (dlp_metadata->seq_vector) {
          free(dlp_metadata->seq_vector);
        }
        free(dlp_metadata);
        return nullptr;
      }
    }
  }

  // Allocate memory for different post-op types
  if (bias_count > 0) {
    dlp_metadata->bias = static_cast<dlp_post_op_bias *>(calloc(bias_count,
                         sizeof(dlp_post_op_bias)));
    if (!dlp_metadata->bias) {
      if (dlp_metadata->scale) {
        for (int i = 0; i < scale_count; ++i) {
          if (dlp_metadata->scale[i].sf) {
            free(dlp_metadata->scale[i].sf);
          }
          if (dlp_metadata->scale[i].zp) {
            free(dlp_metadata->scale[i].zp);
          }
        }
        free(dlp_metadata->scale);
      }
      if (dlp_metadata->seq_vector) {
        free(dlp_metadata->seq_vector);
      }
      free(dlp_metadata);
      return nullptr;
    }
  }

  if (eltwise_count > 0) {
    dlp_metadata->eltwise = static_cast<dlp_post_op_eltwise *>(calloc(eltwise_count,
                            sizeof(dlp_post_op_eltwise)));
    if (!dlp_metadata->eltwise) {
      if (dlp_metadata->bias) {
        free(dlp_metadata->bias);
      }
      free(dlp_metadata->seq_vector);
      free(dlp_metadata);
      return nullptr;
    }
  }

  if (matrix_add_count > 0) {
    dlp_metadata->matrix_add = static_cast<dlp_post_op_matrix_add *>(calloc(
                                 matrix_add_count, sizeof(dlp_post_op_matrix_add)));
    if (!dlp_metadata->matrix_add) {
      if (dlp_metadata->bias) {
        free(dlp_metadata->bias);
      }
      if (dlp_metadata->eltwise) {
        free(dlp_metadata->eltwise);
      }
      free(dlp_metadata->seq_vector);
      free(dlp_metadata);
      return nullptr;
    }
    // Allocate sf structures for matrix_add operations
    for (int i = 0; i < matrix_add_count; ++i) {
      dlp_metadata->matrix_add[i].sf = static_cast<dlp_sf_t *>(calloc(1,
                                       sizeof(dlp_sf_t)));
      dlp_metadata->matrix_add[i].sf->scale_factor_type = to_dlp_type(
            data_type_t::f32);
      if (!dlp_metadata->matrix_add[i].sf) {
        // Clean up partially allocated sf structures
        for (int j = 0; j < i; ++j) {
          free(dlp_metadata->matrix_add[j].sf);
        }
        if (dlp_metadata->bias) {
          free(dlp_metadata->bias);
        }
        if (dlp_metadata->eltwise) {
          free(dlp_metadata->eltwise);
        }
        free(dlp_metadata->matrix_add);
        free(dlp_metadata->seq_vector);
        free(dlp_metadata);
        return nullptr;
      }
    }
  }

  if (matrix_mul_count > 0) {
    dlp_metadata->matrix_mul = static_cast<dlp_post_op_matrix_mul *>(calloc(
                                 matrix_mul_count, sizeof(dlp_post_op_matrix_mul)));
    if (!dlp_metadata->matrix_mul) {
      if (dlp_metadata->bias) {
        free(dlp_metadata->bias);
      }
      if (dlp_metadata->eltwise) {
        free(dlp_metadata->eltwise);
      }
      if (dlp_metadata->matrix_add) {
        for (int i = 0; i < matrix_add_count; ++i) {
          if (dlp_metadata->matrix_add[i].sf) {
            free(dlp_metadata->matrix_add[i].sf);
          }
        }
        free(dlp_metadata->matrix_add);
      }
      free(dlp_metadata->seq_vector);
      free(dlp_metadata);
      return nullptr;
    }
    // Allocate sf structures for matrix_mul operations
    for (int i = 0; i < matrix_mul_count; ++i) {
      dlp_metadata->matrix_mul[i].sf = static_cast<dlp_sf_t *>(calloc(1,
                                       sizeof(dlp_sf_t)));
      dlp_metadata->matrix_mul[i].sf->scale_factor_type = to_dlp_type(
            data_type_t::f32);
      if (!dlp_metadata->matrix_mul[i].sf) {
        // Clean up partially allocated sf structures
        for (int j = 0; j < i; ++j) {
          free(dlp_metadata->matrix_mul[j].sf);
        }
        if (dlp_metadata->bias) {
          free(dlp_metadata->bias);
        }
        if (dlp_metadata->eltwise) {
          free(dlp_metadata->eltwise);
        }
        if (dlp_metadata->matrix_add) {
          for (int k = 0; k < matrix_add_count; ++k) {
            if (dlp_metadata->matrix_add[k].sf) {
              free(dlp_metadata->matrix_add[k].sf);
            }
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
  int bias_index = 0;
  int scale_index = 0;

  // For INT8: Add zero-point compensation FIRST (before scales)
  if (zp_comp_ndim == 1 && zp_comp_acc) {
    dlp_metadata->seq_vector[op_index++] = BIAS;
    dlp_metadata->bias[bias_index].bias = zp_comp_acc;
    dlp_metadata->bias[bias_index].stor_type = DLP_S32;
    dlp_metadata->bias[bias_index].sf = nullptr;
    dlp_metadata->bias[bias_index].zp = nullptr;
    bias_index++;
  }
  else if (zp_comp_ndim == 2 && zp_comp_acc) {
    dlp_metadata->seq_vector[op_index++] = MATRIX_ADD;
    dlp_metadata->matrix_add[matrix_add_index].matrix = zp_comp_acc;
    dlp_metadata->matrix_add[matrix_add_index].stor_type = DLP_S32;
    dlp_metadata->matrix_add[matrix_add_index].ldm = N;
    // Allocate and set scale factor to 1.0
    dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor = malloc(sizeof(
          float));
    *static_cast<float *>
    (dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor) = 1.0f;
    dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor_len = 1;
    dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor_type = DLP_F32;
    matrix_add_index++;
  }

  // For INT8: Add source scale
  if (is_int8 && lowoha_param.quant_params.src_scale.buff) {
    dlp_metadata->seq_vector[op_index++] = SCALE;
    dlp_metadata->scale[scale_index].sf->scale_factor =
      const_cast<void *>(lowoha_param.quant_params.src_scale.buff);
    dlp_metadata->scale[scale_index].sf->scale_factor_type =
      to_dlp_type(lowoha_param.quant_params.src_scale.dt);
    dlp_metadata->scale[scale_index].sf->scale_factor_len =
      get_num_elements(lowoha_param.quant_params.src_scale.dims);
    // Set dummy zero point
    static int32_t dummy_zp = 0;
    dlp_metadata->scale[scale_index].zp->zero_point = &dummy_zp;
    dlp_metadata->scale[scale_index].zp->zero_point_type = DLP_S32;
    dlp_metadata->scale[scale_index].zp->zero_point_len = 1;
    scale_index++;
  }

  // For INT8: Add weight scale
  if (is_int8 && lowoha_param.quant_params.wei_scale.buff) {
    dlp_metadata->seq_vector[op_index++] = SCALE;
    dlp_metadata->scale[scale_index].sf->scale_factor =
      const_cast<void *>(lowoha_param.quant_params.wei_scale.buff);
    dlp_metadata->scale[scale_index].sf->scale_factor_type =
      to_dlp_type(lowoha_param.quant_params.wei_scale.dt);
    dlp_metadata->scale[scale_index].sf->scale_factor_len =
      get_num_elements(lowoha_param.quant_params.wei_scale.dims);
    // Set dummy zero point
    static int32_t dummy_zp_wei = 0;
    dlp_metadata->scale[scale_index].zp->zero_point = &dummy_zp_wei;
    dlp_metadata->scale[scale_index].zp->zero_point_type = DLP_S32;
    dlp_metadata->scale[scale_index].zp->zero_point_len = 1;
    scale_index++;
  }

  // Add bias if present
  if (bias && bias_count > 0) {
    dlp_metadata->seq_vector[op_index++] = BIAS;
    dlp_metadata->bias[bias_index].bias = const_cast<void *>(bias);

    // Set storage type based on bias data type
    dlp_metadata->bias[bias_index].stor_type = to_dlp_type(dtypes.bias);
    dlp_metadata->bias[bias_index].sf = nullptr; // No scale factor for bias
    bias_index++;
  }

  // Add post-ops
  setup_dlp_postops(dlp_metadata, lowoha_param.postop_,
                    op_index, eltwise_index, matrix_add_index, matrix_mul_index,
                    eltwise_count, matrix_add_count, matrix_mul_count);

  // For INT8: Add destination scale at the end (after eltwise post-ops)
  if (is_int8 && lowoha_param.quant_params.dst_scale.buff) {
    dlp_metadata->seq_vector[op_index++] = SCALE;
    dlp_metadata->scale[scale_index].sf->scale_factor =
      const_cast<void *>(lowoha_param.quant_params.dst_scale.buff);
    dlp_metadata->scale[scale_index].sf->scale_factor_type =
      to_dlp_type(lowoha_param.quant_params.dst_scale.dt);
    dlp_metadata->scale[scale_index].sf->scale_factor_len =
      get_num_elements(lowoha_param.quant_params.dst_scale.dims);
    // Set destination zero-point if present
    if (lowoha_param.quant_params.dst_zp.buff) {
      dlp_metadata->scale[scale_index].zp->zero_point =
        const_cast<void *>(lowoha_param.quant_params.dst_zp.buff);
      dlp_metadata->scale[scale_index].zp->zero_point_type =
        to_dlp_type(lowoha_param.quant_params.dst_zp.dt);
      dlp_metadata->scale[scale_index].zp->zero_point_len =
        get_num_elements(lowoha_param.quant_params.dst_zp.dims);
    }
    else {
      static int32_t dummy_dst_zp = 0;
      dlp_metadata->scale[scale_index].zp->zero_point = &dummy_dst_zp;
      dlp_metadata->scale[scale_index].zp->zero_point_type = DLP_S32;
      dlp_metadata->scale[scale_index].zp->zero_point_len = 1;
    }
    scale_index++;
  }

  dlp_metadata->seq_length = op_index;
  dlp_metadata->num_eltwise = eltwise_count;

  // Setup pre-ops for WOQ (Weight-Only Quantization)
  if (is_woq) {
    setup_woq_pre_ops(dlp_metadata, lowoha_param, K, N);
  }

  return dlp_metadata;
}
#else
aocl_post_op *create_blis_post_op(const matmul_params &lowoha_param,
                                  const void *bias, const matmul_data_types &dtypes, int N,
                                  int K, zendnnl::ops::matmul_algo_t kernel) {
  // Check if this is a WOQ case (S4/S8 weights with BF16 source)
  bool is_woq = dtypes.wei == data_type_t::s4 &&
                dtypes.src == data_type_t::bf16 &&
                kernel == zendnnl::ops::matmul_algo_t::aocl_dlp_blocked;

  // Count total operations (bias + post-ops)
  int total_ops = (bias ? 1 : 0) + lowoha_param.postop_.size();

  // For WOQ, we need aocl_post_op even if total_ops == 0 (for pre_ops)
  if (total_ops == 0 && !is_woq) {
    return nullptr;
  }

  aocl_post_op *aocl_po = static_cast<aocl_post_op *>(calloc(1,
                          sizeof(aocl_post_op)));
  if (!aocl_po) {
    return nullptr;
  }

  // Count different types of operations
  int eltwise_count = 0;
  int matrix_add_count = 0;
  int matrix_mul_count = 0;
  int bias_count = bias ? 1 : 0;

  // Count post-ops by type
  for (const auto &po : lowoha_param.postop_) {
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
    aocl_po->bias = static_cast<aocl_post_op_bias *>(calloc(bias_count,
                    sizeof(aocl_post_op_bias)));
    if (!aocl_po->bias) {
      free(aocl_po);
      return nullptr;
    }
  }

  if (eltwise_count > 0) {
    aocl_po->eltwise = static_cast<aocl_post_op_eltwise *>(calloc(eltwise_count,
                       sizeof(aocl_post_op_eltwise)));
    if (!aocl_po->eltwise) {
      if (aocl_po->bias) {
        free(aocl_po->bias);
      }
      free(aocl_po);
      return nullptr;
    }
  }

  if (matrix_add_count > 0) {
    aocl_po->matrix_add = static_cast<aocl_post_op_matrix_add *>(calloc(
                            matrix_add_count, sizeof(aocl_post_op_matrix_add)));
    if (!aocl_po->matrix_add) {
      if (aocl_po->bias) {
        free(aocl_po->bias);
      }
      if (aocl_po->eltwise) {
        free(aocl_po->eltwise);
      }
      free(aocl_po);
      return nullptr;
    }
  }

  if (matrix_mul_count > 0) {
    aocl_po->matrix_mul = static_cast<aocl_post_op_matrix_mul *>(calloc(
                            matrix_mul_count, sizeof(aocl_post_op_matrix_mul)));
    if (!aocl_po->matrix_mul) {
      if (aocl_po->bias) {
        free(aocl_po->bias);
      }
      if (aocl_po->eltwise) {
        free(aocl_po->eltwise);
      }
      if (aocl_po->matrix_add) {
        free(aocl_po->matrix_add);
      }
      free(aocl_po);
      return nullptr;
    }
  }

  // Set up sequence vector
  aocl_po->seq_vector = static_cast<AOCL_POST_OP_TYPE *>(calloc(total_ops,
                        sizeof(AOCL_POST_OP_TYPE)));
  if (!aocl_po->seq_vector) {
    if (aocl_po->bias) {
      free(aocl_po->bias);
    }
    if (aocl_po->eltwise) {
      free(aocl_po->eltwise);
    }
    if (aocl_po->matrix_add) {
      free(aocl_po->matrix_add);
    }
    if (aocl_po->matrix_mul) {
      free(aocl_po->matrix_mul);
    }
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
    aocl_po->bias[0].bias = const_cast<void *>(bias);

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
  for (const auto &po : lowoha_param.postop_) {
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
      *static_cast<float *>(aocl_po->eltwise[eltwise_index].algo.alpha) = 0.01f;
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
      *static_cast<float *>(aocl_po->eltwise[eltwise_index].algo.alpha) = 1.0f;
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
      *static_cast<float *>(aocl_po->matrix_add[matrix_add_index].scale_factor) =
        1.0f; // Default scale
      aocl_po->matrix_add[matrix_add_index].scale_factor_len = 1;
      aocl_po->matrix_add[matrix_add_index].ldm = po.leading_dim;
      aocl_po->matrix_add[matrix_add_index].stor_type = po.dtype == data_type_t::bf16
          ? AOCL_GEMM_BF16 : AOCL_GEMM_F32;
      matrix_add_index++;
      break;
    case post_op_type_t::binary_mul:
      aocl_po->seq_vector[op_index++] = MATRIX_MUL;
      aocl_po->matrix_mul[matrix_mul_index].matrix = po.buff;
      aocl_po->matrix_mul[matrix_mul_index].scale_factor = malloc(sizeof(float));
      *static_cast<float *>(aocl_po->matrix_mul[matrix_mul_index].scale_factor) =
        1.0f; // Default scale
      aocl_po->matrix_mul[matrix_mul_index].scale_factor_len = 1;
      aocl_po->matrix_mul[matrix_mul_index].ldm = N; // Set leading dimension to N
      aocl_po->matrix_mul[matrix_mul_index].stor_type = po.dtype == data_type_t::bf16
          ? AOCL_GEMM_BF16 : AOCL_GEMM_F32;
      matrix_mul_index++;
      break;
    default:
      // Skip unsupported post-ops
      break;
    }
  }

  aocl_po->seq_length = op_index;
  aocl_po->num_eltwise = eltwise_count;

  // Setup pre-ops for WOQ (Weight-Only Quantization)
  if (is_woq) {
    aocl_po->pre_ops = static_cast<aocl_pre_op *>(malloc(sizeof(aocl_pre_op)));
    if (aocl_po->pre_ops) {
      const auto &wei_scale = lowoha_param.quant_params.wei_scale;
      const auto &wei_zp = lowoha_param.quant_params.wei_zp;

      // Setup weight scale factor
      aocl_po->pre_ops->b_scl = static_cast<aocl_pre_op_sf *>(malloc(sizeof(
                                  aocl_pre_op_sf)));
      if (aocl_po->pre_ops->b_scl) {
        size_t scale_len = get_num_elements(wei_scale.dims);
        aocl_po->pre_ops->b_scl->scale_factor = const_cast<void *>(wei_scale.buff);
        aocl_po->pre_ops->b_scl->scale_factor_len = scale_len;
        aocl_po->pre_ops->b_scl->scale_factor_type =
          (wei_scale.dt == data_type_t::bf16) ? AOCL_GEMM_BF16 : AOCL_GEMM_F32;
      }

      // Setup weight zero-point
      aocl_po->pre_ops->b_zp = static_cast<aocl_pre_op_zp *>(malloc(sizeof(
                                 aocl_pre_op_zp)));
      if (aocl_po->pre_ops->b_zp) {
        if (wei_zp.buff) {
          aocl_po->pre_ops->b_zp->zero_point = const_cast<void *>(wei_zp.buff);
          aocl_po->pre_ops->b_zp->zero_point_len = get_num_elements(wei_zp.dims);
        }
        else {
          aocl_po->pre_ops->b_zp->zero_point = nullptr;
          aocl_po->pre_ops->b_zp->zero_point_len = 0;
        }
        aocl_po->pre_ops->b_zp->zero_point_type = AOCL_GEMM_INT8;
      }

      aocl_po->pre_ops->seq_length = 1;

      // Determine group_size from scale dimensions
      // wei_scale.dims determines granularity:
      //   - Per-tensor:  dims = {} or {1}     → group_size = K
      //   - Per-channel: dims = {1, N}        → group_size = K
      //   - Per-group:   dims = {G, N}        → group_size = K / G
      int64_t group_size = K;  // Default per-tensor
      const auto &dims = wei_scale.dims;
      if (!dims.empty() && !(dims.size() == 1 && dims[0] == 1)) {
        // Not per-tensor, check for per-group
        if (dims.size() == 2 && dims[1] == N && dims[0] > 1) {
          group_size = K / dims[0];  // Per-group: dims = {G, N}
        }
        // Per-channel (dims={N} or {1,N}) keeps group_size = K
      }

      // Validation: group_size must divide K evenly
      if (K % group_size != 0) {
        log_error("WOQ: group_size (", group_size, ") must divide K (", K, ") evenly");
        group_size = K;  // Fallback to per-tensor
      }
      aocl_po->pre_ops->group_size = static_cast<dim_t>(group_size);

      apilog_info("WOQ BLIS: scale_len=", get_num_elements(wei_scale.dims),
                  ", group_size=", group_size);
    }
  }

  return aocl_po;
}
#endif
// Cleanup functions for post-op structures
#if ZENDNNL_DEPENDS_AOCLDLP
void cleanup_dlp_post_op(dlp_metadata_t *aocl_po,
                         const matmul_params &lowoha_param) {
  if (aocl_po) {
    // Count operations from seq_vector for proper cleanup
    // This is more accurate than counting from lowoha_param.postop_ because
    // it includes zp_comp matrix_add entries added internally
    int eltwise_count = 0;
    int matrix_add_count = 0;
    int matrix_mul_count = 0;

    // Count from seq_vector to include all operations (including zp_comp)
    if (aocl_po->seq_vector) {
      for (int i = 0; i < aocl_po->seq_length; i++) {
        switch (aocl_po->seq_vector[i]) {
        case ELTWISE:
          eltwise_count++;
          break;
        case MATRIX_ADD:
          matrix_add_count++;
          break;
        case MATRIX_MUL:
          matrix_mul_count++;
          break;
        default:
          break;
        }
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
      // Count scale entries in seq_vector to properly free nested structures
      int scale_count = 0;
      if (aocl_po->seq_vector) {
        for (int i = 0; i < aocl_po->seq_length; i++) {
          if (aocl_po->seq_vector[i] == SCALE) {
            scale_count++;
          }
        }
      }
      // Free nested sf and zp structures for each scale
      for (int i = 0; i < scale_count; i++) {
        if (aocl_po->scale[i].sf) {
          // Note: scale_factor is user-provided buffer, don't free it
          free(aocl_po->scale[i].sf);
        }
        if (aocl_po->scale[i].zp) {
          // Note: zero_point is user-provided buffer, don't free it
          free(aocl_po->scale[i].zp);
        }
      }
      free(aocl_po->scale);
    }

    if (aocl_po->pre_ops) {
      if (aocl_po->pre_ops->b_scl) {
        free(aocl_po->pre_ops->b_scl);
      }
      if (aocl_po->pre_ops->b_zp) {
        free(aocl_po->pre_ops->b_zp);
      }
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
void cleanup_blis_post_op(aocl_post_op *aocl_po,
                          const matmul_params &lowoha_param) {
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
      for (int i = 0; i < (lowoha_param.postop_.size() > 0 ? 1 : 0); i++) {
        if (aocl_po->matrix_add[i].scale_factor) {
          free(aocl_po->matrix_add[i].scale_factor);
        }
      }
      free(aocl_po->matrix_add);
    }

    if (aocl_po->matrix_mul) {
      for (int i = 0; i < (lowoha_param.postop_.size() > 0 ? 1 : 0); i++) {
        if (aocl_po->matrix_mul[i].scale_factor) {
          free(aocl_po->matrix_mul[i].scale_factor);
        }
      }
      free(aocl_po->matrix_mul);
    }

    if (aocl_po->bias) {
      free(aocl_po->bias);
    }

    // Clean up pre-ops for WOQ
    if (aocl_po->pre_ops) {
      // Note: Don't free scale_factor and zero_point buffers - they're user-provided
      if (aocl_po->pre_ops->b_scl) {
        aocl_po->pre_ops->b_scl->scale_factor = nullptr;
        free(aocl_po->pre_ops->b_scl);
      }
      if (aocl_po->pre_ops->b_zp) {
        aocl_po->pre_ops->b_zp->zero_point = nullptr;
        free(aocl_po->pre_ops->b_zp);
      }
      free(aocl_po->pre_ops);
    }

    if (aocl_po->seq_vector) {
      free(aocl_po->seq_vector);
    }
    free(aocl_po);
  }
}
#endif

template <typename T>
bool reorderAndCacheWeights(Key_matmul key, const void *weights,
                            void *&reorder_weights, const int k, const int n, const int ldb,
                            const char order, const char trans, char mem_format_b,
                            get_reorder_buff_size_func_ptr get_reorder_buf_size,
                            reorder_func_ptr<T> reorder_func, int weight_cache_type) {
  // Weight caching
  static lru_cache_t<Key_matmul, void *> matmul_weight_cache;

  // Weights are already reordered and algo is aocl_dlp_blocked
  // Add the key into map and value as nullptr
  // Modify the reorder_weight as weight.
  if (mem_format_b == 'r') {
    matmul_weight_cache.add(key, nullptr);
    reorder_weights = const_cast<void *>(weights);
    return true;
  }

  if (weight_cache_type == 0) {
    apilog_info("AOCL reorder weights (WEIGHT_CACHE_DISABLE)");
    size_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, 'B',
                                   k, n
#if ZENDNNL_DEPENDS_AOCLDLP
                                   ,nullptr
#endif
                                                       );
    size_t alignment      = 64;
    size_t reorder_size   = (b_reorder_buf_siz_req + alignment - 1) & ~
                            (alignment - 1);
    reorder_weights       = (T *)aligned_alloc(alignment, reorder_size);
    reorder_func(order, trans, 'B', (T *)weights, (T *)reorder_weights, k, n, ldb
#if ZENDNNL_DEPENDS_AOCLDLP
                 ,nullptr
#endif
                );
  }
  // Out-of-place reordering
  else if (weight_cache_type == 1) {
    auto found_obj = matmul_weight_cache.find_key(key);
    if (!found_obj) {
      apilog_info("AOCL reorder weights WEIGHT_CACHE_OUT_OF_PLACE");
      size_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, 'B',
                                     k, n
#if ZENDNNL_DEPENDS_AOCLDLP
                                     ,nullptr
#endif
                                                         );
      size_t alignment      = 64;
      size_t reorder_size   = (b_reorder_buf_siz_req + alignment - 1) & ~
                              (alignment - 1);
      reorder_weights = (T *)aligned_alloc(alignment, reorder_size);
      reorder_func(order, trans, 'B', (T *)weights, (T *)reorder_weights, k, n, ldb
#if ZENDNNL_DEPENDS_AOCLDLP
                   ,nullptr
#endif
                  );
      // Create new entry
      matmul_weight_cache.add(key, reorder_weights);
    }
    else {
      apilog_info("Read AOCL cached weights WEIGHT_CACHE_OUT_OF_PLACE");
      reorder_weights = matmul_weight_cache.get(key);
    }
  }
  return true;
}

template bool reorderAndCacheWeights<short>(Key_matmul, const void *, void *&,
    int, int, int, char, char, char, get_reorder_buff_size_func_ptr,
    reorder_func_ptr<short>, int);
template bool reorderAndCacheWeights<float>(Key_matmul, const void *, void *&,
    int, int, int, char, char, char, get_reorder_buff_size_func_ptr,
    reorder_func_ptr<float>, int);
template bool reorderAndCacheWeights<int8_t>(Key_matmul, const void *, void *&,
    int, int, int, char, char, char, get_reorder_buff_size_func_ptr,
    reorder_func_ptr<int8_t>, int);

void woqReorderAndCacheWeightsAocl(Key_matmul key, const int8_t *weights,
                                   void *&reorder_weights, const int k, const int n, const int ldb,
                                   const bool is_weights_const, const char order, const char trans,
                                   char mem_format_b,
                                   const matmul_quantization_params_t &quant_params,
                                   int weight_cache_type) {
  // Weight caching inplace support cannot be added since buffer size is
  // always expanded.
  static lru_cache_t<Key_matmul, void *> matmul_weight_cache_woq;
  auto found_obj = matmul_weight_cache_woq.find_key(key);

  bool is_transposed = (trans == 't');
  if (!is_weights_const || !found_obj) {
    apilog_info("WOQ Simulated AOCL reorder weights (weight_cache_type=",
                weight_cache_type, ")");
    size_t alignment = 64;
    size_t cvt_weights_size = (sizeof(bfloat16_t)*k*n + alignment - 1) & ~
                              (alignment - 1);
    bfloat16_t *cvt_weights = (bfloat16_t *)aligned_alloc(alignment,
                              cvt_weights_size);
    cvt_int4_to_bf16(weights, cvt_weights, k, n, ldb, is_transposed,
                     quant_params.wei_scale.buff, quant_params.wei_scale.dims,
                     quant_params.wei_scale.dt,
                     quant_params.wei_zp.buff, quant_params.wei_zp.dims, quant_params.wei_zp.dt);
    // After cvt_int4_to_bf16, weights are in K×N layout (non-transposed), ldb_cvt = n
    int ldb_cvt = n;
    size_t b_reorder_buf_siz_req = aocl_get_reorder_buf_size_bf16bf16f32of32(order,
                                   'n', 'B', k, n
#if ZENDNNL_DEPENDS_AOCLDLP
                                   ,nullptr
#endif
                                                                            );
    size_t reorder_size   = (b_reorder_buf_siz_req + alignment - 1) & ~
                            (alignment - 1);
    reorder_weights = (int16_t *)aligned_alloc(alignment, reorder_size);
    aocl_reorder_bf16bf16f32of32(order, 'n', 'B', (int16_t *)cvt_weights,
                                 (int16_t *)reorder_weights, k, n, ldb_cvt
#if ZENDNNL_DEPENDS_AOCLDLP
                                 ,nullptr
#endif
                                );
    free(cvt_weights);
    if (is_weights_const && weight_cache_type == 1) {
      // Create new entry
      matmul_weight_cache_woq.add(key, reorder_weights);
    }
  }
  else {
    apilog_info("Read WOQ Simulated AOCL cached weights");
    reorder_weights = matmul_weight_cache_woq.get(key);
  }
}

void run_dlp(char layout, char transA, char transB, int M, int N,
             int K,
             float alpha, float beta, int lda, int ldb, int ldc,
             char mem_format_a, char mem_format_b, const void *A,
             const void *B, void *C, const matmul_data_types &dtypes,
             const matmul_params &lowoha_param, const void *bias,
             zendnnl::ops::matmul_algo_t kernel,
             bool is_weights_const) {

  bool is_weight_blocked = false;
  void *reordered_mem = nullptr;
  bool simulated_woq_free_buff = false;
  matmul_config_t &matmul_config = matmul_config_t::instance();
  int32_t weight_cache_type = matmul_config.get_weight_cache();

  // Create cache key once for both weight reordering and ZP compensation caching
  Key_matmul cache_key(transB == 't', K, N, ldb, B,
                       static_cast<uint32_t>(matmul_algo_t::aocl_dlp_blocked));

  // AOCL blocked kernel reordering for 2D MatMul
  if (kernel==zendnnl::ops::matmul_algo_t::aocl_dlp_blocked &&
      is_weights_const) {
    //call reorder and cache function
    bool blocked_flag = false;
    if (lowoha_param.dtypes.wei == data_type_t::f32) {
      blocked_flag = reorderAndCacheWeights<float>(cache_key, B, reordered_mem, K, N,
                     ldb,
                     'r', transB, mem_format_b,
                     aocl_get_reorder_buf_size_f32f32f32of32, aocl_reorder_f32f32f32of32,
                     weight_cache_type);
    }
    else if (lowoha_param.dtypes.wei == data_type_t::bf16) {
      blocked_flag = reorderAndCacheWeights<int16_t>(cache_key, B, reordered_mem, K,
                     N, ldb,
                     'r', transB, mem_format_b,
                     aocl_get_reorder_buf_size_bf16bf16f32of32, aocl_reorder_bf16bf16f32of32,
                     weight_cache_type);
    }
    else if (lowoha_param.dtypes.wei == data_type_t::s4) {
      blocked_flag = reorderAndCacheWeights<int8_t>(cache_key, B, reordered_mem, K,
                     N, ldb,
                     'r', transB, mem_format_b,
                     aocl_get_reorder_buf_size_bf16s4f32of32, aocl_reorder_bf16s4f32of32,
                     weight_cache_type);
    }
    else if (lowoha_param.dtypes.wei == data_type_t::s8) {
      // INT8 weight reordering - select based on source data type
      if (lowoha_param.dtypes.src == data_type_t::s8) {
        blocked_flag = reorderAndCacheWeights<int8_t>(cache_key, B, reordered_mem, K,
                       N, ldb,
                       'r', transB, mem_format_b,
                       aocl_get_reorder_buf_size_s8s8s32os32, aocl_reorder_s8s8s32os32,
                       weight_cache_type);
      }
      else if (lowoha_param.dtypes.src == data_type_t::u8) {
        blocked_flag = reorderAndCacheWeights<int8_t>(cache_key, B, reordered_mem, K,
                       N, ldb,
                       'r', transB, mem_format_b,
                       aocl_get_reorder_buf_size_u8s8s32os32, aocl_reorder_u8s8s32os32,
                       weight_cache_type);
      }
    }
    if (blocked_flag) {
      is_weight_blocked = true;
      mem_format_b = 'r';
    }
  }
  else if (kernel == zendnnl::ops::matmul_algo_t::aocl_dlp &&
           dtypes.wei == data_type_t::s4) {
    //call woq reorder and cache function
    woqReorderAndCacheWeightsAocl(cache_key, static_cast<const int8_t *>(B),
                                  reordered_mem, K,
                                  N, ldb, is_weights_const, 'r', transB, mem_format_b,
                                  lowoha_param.quant_params, weight_cache_type);
    is_weight_blocked = true;
    mem_format_b = 'r';
    simulated_woq_free_buff = !is_weights_const || weight_cache_type != 1;
  }

  // Compute zero-point compensation for INT8 (with caching for 1D case)
  int32_t *zp_comp_acc = nullptr;
  int zp_comp_ndim = 0;
  int32_t src_zp = 0;
  int32_t wei_zp = 0;
  bool is_int8 = (dtypes.src == data_type_t::u8 ||
                  dtypes.src == data_type_t::s8) &&
                 dtypes.wei == data_type_t::s8;
  if (is_int8) {
    // Extract zero-point values
    if (lowoha_param.quant_params.src_zp.buff) {
      src_zp = read_and_cast<int32_t>(lowoha_param.quant_params.src_zp.buff,
                                      lowoha_param.quant_params.src_zp.dt);
    }
    if (lowoha_param.quant_params.wei_zp.buff) {
      wei_zp = read_and_cast<int32_t>(lowoha_param.quant_params.wei_zp.buff,
                                      lowoha_param.quant_params.wei_zp.dt);
    }

    // Compute or retrieve cached zero-point compensation
    if (src_zp != 0 || wei_zp != 0) {
      zp_comp_acc = cache_or_compute_zp_compensation(
                      cache_key, M, N, K, A, B,
                      src_zp, wei_zp,
                      transA == 't', transB == 't',
                      lda, ldb,
                      dtypes.src,
                      is_weights_const,
                      zp_comp_ndim);

      if (zp_comp_acc) {
        bool is_cacheable = (wei_zp == 0 && is_weights_const &&
                             matmul_config.get_zp_comp_cache());
        apilog_info("INT8 ZP compensation: src_zp=", src_zp, ", wei_zp=", wei_zp,
                    ", ndim=", zp_comp_ndim, ", cached=", (is_cacheable ? "yes" : "no"));
      }
    }
  }

  // Create aocl_post_op structure for bias, post-ops, and WOQ pre-ops
#if ZENDNNL_DEPENDS_AOCLDLP
  dlp_metadata_t *aocl_po = create_dlp_post_op(lowoha_param, bias, dtypes, N, K,
                            M, zp_comp_acc, zp_comp_ndim, kernel);
#else
  aocl_post_op *aocl_po = create_blis_post_op(lowoha_param, bias, dtypes, N, K,
                          kernel);
#endif

  if (dtypes.src == data_type_t::f32 && dtypes.dst == data_type_t::f32) {
    aocl_gemm_f32f32f32of32(layout,transA,transB,M,N,K,alpha,
                            static_cast<const float *>(A),lda,mem_format_a,
                            is_weight_blocked ? (float *)reordered_mem : static_cast<const float *>(B),
                            ldb, mem_format_b, beta,static_cast<float *>(C),ldc,aocl_po);
  }
  // Skip this path for non-blocked AOCL-DLP kernels for S4 weights
  else if (dtypes.wei == data_type_t::s4 &&
           kernel == zendnnl::ops::matmul_algo_t::aocl_dlp_blocked) {
    if (dtypes.dst == data_type_t::bf16) {
      aocl_gemm_bf16s4f32obf16(layout,transA,transB,M,N,K,alpha,
                               static_cast<const int16_t *>(A),lda,mem_format_a,
                               is_weight_blocked ? (int8_t *)reordered_mem : static_cast<const int8_t *>(B),
                               ldb, mem_format_b, beta,static_cast<int16_t *>(C),ldc,aocl_po);
    }
    else if (dtypes.dst == data_type_t::f32) {
      aocl_gemm_bf16s4f32of32(layout,transA,transB,M,N,K,alpha,
                              static_cast<const int16_t *>(A),lda,mem_format_a,
                              is_weight_blocked ? (int8_t *)reordered_mem : static_cast<const int8_t *>(B),
                              ldb,mem_format_b, beta,static_cast<float *>(C),ldc,aocl_po);
    }
    else {
      log_error("Unsupported data type for matmul");
    }
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::bf16) {
    aocl_gemm_bf16bf16f32obf16(layout,transA,transB,M,N,K,alpha,
                               static_cast<const int16_t *>(A),lda,mem_format_a,
                               is_weight_blocked ? (int16_t *)reordered_mem : static_cast<const int16_t *>(B),
                               ldb, mem_format_b, beta,static_cast<int16_t *>(C),ldc,aocl_po);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::f32) {
    aocl_gemm_bf16bf16f32of32(layout,transA,transB,M,N,K,alpha,
                              static_cast<const int16_t *>(A),lda,mem_format_a,
                              is_weight_blocked ? (int16_t *)reordered_mem : static_cast<const int16_t *>(B),
                              ldb,mem_format_b, beta,static_cast<float *>(C),ldc,aocl_po);
  }
  // INT8 kernels: u8 source
  else if (dtypes.src == data_type_t::u8 && dtypes.wei == data_type_t::s8) {
    const int8_t *weight_ptr = is_weight_blocked ? static_cast<int8_t *>
                               (reordered_mem) : static_cast<const int8_t *>(B);
    switch (dtypes.dst) {
    case data_type_t::u8:
      aocl_gemm_u8s8s32ou8(layout, transA, transB, M, N, K, alpha,
                           static_cast<const uint8_t *>(A), lda, mem_format_a,
                           weight_ptr, ldb, mem_format_b, beta,
                           static_cast<uint8_t *>(C), ldc, aocl_po);
      break;
    case data_type_t::s8:
      aocl_gemm_u8s8s32os8(layout, transA, transB, M, N, K, alpha,
                           static_cast<const uint8_t *>(A), lda, mem_format_a,
                           weight_ptr, ldb, mem_format_b, beta,
                           static_cast<int8_t *>(C), ldc, aocl_po);
      break;
    case data_type_t::s32:
      aocl_gemm_u8s8s32os32(layout, transA, transB, M, N, K, alpha,
                            static_cast<const uint8_t *>(A), lda, mem_format_a,
                            weight_ptr, ldb, mem_format_b, beta,
                            static_cast<int32_t *>(C), ldc, aocl_po);
      break;
    case data_type_t::f32:
      aocl_gemm_u8s8s32of32(layout, transA, transB, M, N, K, alpha,
                            static_cast<const uint8_t *>(A), lda, mem_format_a,
                            weight_ptr, ldb, mem_format_b, beta,
                            static_cast<float *>(C), ldc, aocl_po);
      break;
    case data_type_t::bf16:
      aocl_gemm_u8s8s32obf16(layout, transA, transB, M, N, K, alpha,
                             static_cast<const uint8_t *>(A), lda, mem_format_a,
                             weight_ptr, ldb, mem_format_b, beta,
                             static_cast<int16_t *>(C), ldc, aocl_po);
      break;
    default:
      log_error("Unsupported output data type for u8 source");
      break;
    }
  }
  // INT8 kernels: s8 source
  else if (dtypes.src == data_type_t::s8 && dtypes.wei == data_type_t::s8) {
    const int8_t *weight_ptr = is_weight_blocked ? static_cast<int8_t *>
                               (reordered_mem) : static_cast<const int8_t *>(B);
    switch (dtypes.dst) {
    case data_type_t::u8:
      aocl_gemm_s8s8s32ou8(layout, transA, transB, M, N, K, alpha,
                           static_cast<const int8_t *>(A), lda, mem_format_a,
                           weight_ptr, ldb, mem_format_b, beta,
                           static_cast<uint8_t *>(C), ldc, aocl_po);
      break;
    case data_type_t::s8:
      aocl_gemm_s8s8s32os8(layout, transA, transB, M, N, K, alpha,
                           static_cast<const int8_t *>(A), lda, mem_format_a,
                           weight_ptr, ldb, mem_format_b, beta,
                           static_cast<int8_t *>(C), ldc, aocl_po);
      break;
    case data_type_t::s32:
      aocl_gemm_s8s8s32os32(layout, transA, transB, M, N, K, alpha,
                            static_cast<const int8_t *>(A), lda, mem_format_a,
                            weight_ptr, ldb, mem_format_b, beta,
                            static_cast<int32_t *>(C), ldc, aocl_po);
      break;
    case data_type_t::f32:
      aocl_gemm_s8s8s32of32(layout, transA, transB, M, N, K, alpha,
                            static_cast<const int8_t *>(A), lda, mem_format_a,
                            weight_ptr, ldb, mem_format_b, beta,
                            static_cast<float *>(C), ldc, aocl_po);
      break;
    case data_type_t::bf16:
      aocl_gemm_s8s8s32obf16(layout, transA, transB, M, N, K, alpha,
                             static_cast<const int8_t *>(A), lda, mem_format_a,
                             weight_ptr, ldb, mem_format_b, beta,
                             static_cast<int16_t *>(C), ldc, aocl_po);
      break;
    default:
      log_error("Unsupported output data type for s8 source");
      break;
    }
  }
  else {
    apilog_info("Data type not supported");
  }
  // Free reordered buffer for AOCL blocked non-cached
  bool weight_cache_disabled = (weight_cache_type == 0 &&
                                reordered_mem != nullptr &&
                                lowoha_param.mem_format_b != 'r'
                                && kernel==zendnnl::ops::matmul_algo_t::aocl_dlp_blocked);
  if (weight_cache_disabled || simulated_woq_free_buff)  {
    free(reordered_mem);
    reordered_mem = nullptr;
  }
  // Free zero-point compensation buffer (only if not cached)
  // 1D compensation (wei_zp == 0) is cached, 2D compensation is always freed
  bool zp_cache_enabled = matmul_config.get_zp_comp_cache();
  if (zp_comp_acc && (!zp_cache_enabled || wei_zp != 0)) {
    std::free(zp_comp_acc);
  }

  // Clean up aocl_post_op structure
#if ZENDNNL_DEPENDS_AOCLDLP
  cleanup_dlp_post_op(aocl_po, lowoha_param);
#else
  cleanup_blis_post_op(aocl_po, lowoha_param);
#endif
}

void matmul_batch_gemm_wrapper(char layout, char transA, char transB, int M,
                               int N, int K, float alpha, const void *A, int lda, const void *B, int ldb,
                               float beta, void *C, int ldc, matmul_data_types &dtypes, int batch_count,
                               int Batch_A, int Batch_B, char mem_format_a, char mem_format_b,
                               size_t src_stride, size_t weight_stride,
                               size_t dst_stride, const matmul_params &lowoha_param, const void *bias,
                               int num_threads) {


#if ZENDNNL_DEPENDS_AOCLDLP
  dlp_metadata_t *metadata_array = create_dlp_post_op(lowoha_param, bias, dtypes,
                                   N, K);
  md_t m_ = M;
  md_t n_ = N;
  md_t k_ = K;
  md_t lda_ = lda;
  md_t ldb_ = ldb;
  md_t ldc_ = ldc;
  md_t group_size = batch_count;
#else
  aocl_post_op *metadata_array = create_blis_post_op(lowoha_param, bias, dtypes,
                                 N, K);
  dim_t m_ = M;
  dim_t n_ = N;
  dim_t k_ = K;
  dim_t lda_ = lda;
  dim_t ldb_ = ldb;
  dim_t ldc_ = ldc;
  dim_t group_size = batch_count;
#endif

  // Helper lambda for batch index calculation (handles broadcasting)
  auto get_batch_idx = [](int b, int batch_size) {
    return (batch_size == 1) ? 0 : (b % batch_size);
  };

  // Prepare pointer arrays for matrices
  std::vector<const void *> a_ptrs(batch_count);
  std::vector<const void *> b_ptrs(batch_count);
  std::vector<void *> c_ptrs(batch_count);

  // Set up pointers for each batch (with broadcasting support)
  #pragma omp parallel for num_threads(num_threads)
  for (int b = 0; b < batch_count; ++b) {
    a_ptrs[b] = static_cast<const uint8_t *>(A) + get_batch_idx(b,
                Batch_A) * src_stride;
    b_ptrs[b] = static_cast<const uint8_t *>(B) + get_batch_idx(b,
                Batch_B) * weight_stride;
    c_ptrs[b] = static_cast<uint8_t *>(C) + b * dst_stride;
  }

  // Call appropriate batch GEMM based on data types
  if (dtypes.src == data_type_t::f32 && dtypes.dst == data_type_t::f32) {
    apilog_info("executing aocl_batch_gemm_f32f32f32of32");
    aocl_batch_gemm_f32f32f32of32(
      &layout, &transA, &transB,
      &m_, &n_, &k_,
      &alpha,
      reinterpret_cast<const float **>(a_ptrs.data()), &lda_,
      reinterpret_cast<const float **>(b_ptrs.data()), &ldb_,
      &beta,
      reinterpret_cast<float **>(c_ptrs.data()), &ldc_,
      1, // single group
      &group_size,
      &mem_format_a, &mem_format_b,
      &metadata_array);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::f32) {
    apilog_info("executing aocl_batch_gemm_bf16bf16f32of32");
    aocl_batch_gemm_bf16bf16f32of32(
      &layout, &transA, &transB,
      &m_, &n_, &k_,
      &alpha,
      reinterpret_cast<const bfloat16 **>(a_ptrs.data()), &lda_,
      reinterpret_cast<const bfloat16 **>(b_ptrs.data()), &ldb_,
      &beta,
      reinterpret_cast<float **>(c_ptrs.data()), &ldc_,
      1, // single group
      &group_size,
      &mem_format_a, &mem_format_b,
      &metadata_array);
  }
  else if (dtypes.src == data_type_t::bf16 && dtypes.dst == data_type_t::bf16) {
    apilog_info("executing aocl_batch_gemm_bf16bf16f32obf16");
    aocl_batch_gemm_bf16bf16f32obf16(
      &layout, &transA, &transB,
      &m_, &n_, &k_,
      &alpha,
      reinterpret_cast<const bfloat16 **>(a_ptrs.data()), &lda_,
      reinterpret_cast<const bfloat16 **>(b_ptrs.data()), &ldb_,
      &beta,
      reinterpret_cast<bfloat16 **>(c_ptrs.data()), &ldc_,
      1, // single group
      &group_size,
      &mem_format_a, &mem_format_b,
      &metadata_array);
  }
  else {
    log_error("Unsupported data type combination for batch GEMM");
  }

  // Clean up aocl_post_op structure
#if ZENDNNL_DEPENDS_AOCLDLP
  cleanup_dlp_post_op(metadata_array, lowoha_param);
#else
  cleanup_blis_post_op(metadata_array, lowoha_param);
#endif
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl