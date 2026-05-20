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

#include "lowoha_operators/matmul/backends/aocl/aocl_postop.hpp"
#include <cstdlib>

namespace zendnnl {
namespace lowoha {
namespace matmul {
// Helper function to convert zendnnl data_type_t to the AOCL DLP storage-type
// enum used by post-op slots (bias, scale/zp, matrix add/mul, etc.).
static DLP_TYPE to_dlp_type(data_type_t dt) {
  switch (dt) {
  case data_type_t::f32:
    return DLP_F32;
  case data_type_t::bf16:
    return DLP_BF16;
  case data_type_t::f16:
    return DLP_F16;
  case data_type_t::s32:
    return DLP_S32;
  case data_type_t::s8:
    return DLP_S8;
  case data_type_t::u8:
    return DLP_U8;
  default:
    return DLP_F32;
  }
}

// Fill DLP post-op slots (eltwise, binary_add, binary_mul) for one matmul.
//
// Precondition: the caller has already sized md->eltwise / matrix_add /
// matrix_mul to exactly the number of slots `postops` will produce, so this
// function does no further bounds checking on the slot indices.
static void setup_dlp_postops(dlp_metadata_t *md,
                              const std::vector<matmul_post_op> &postops,
                              int &op_index, int &eltwise_index,
                              int &matrix_add_index, int &matrix_mul_index) {
  // Write one ELTWISE slot. stor_type describes the storage of alpha/beta;
  // when the op carries neither, leave the field at its calloc-zero default
  // (DLP_INVALID) so behavior matches the pre-refactor code byte-for-byte.
  auto put_eltwise = [&](DLP_ELT_ALGO_TYPE algo,
  void *alpha = nullptr, void *beta = nullptr) {
    auto &e = md->eltwise[eltwise_index++];
    md->seq_vector[op_index++] = ELTWISE;
    e.algo.algo_type = algo;
    if (alpha || beta) {
      e.algo.stor_type = DLP_F32;
    }
    e.algo.alpha = alpha;
    e.algo.beta  = beta;
    e.sf = nullptr;
  };

  // Write one MATRIX_{ADD,MUL} slot.
  auto put_matrix = [&](auto *arr, int &idx, DLP_POST_OP_TYPE tag,
  const matmul_post_op &po) {
    auto &m = arr[idx++];
    md->seq_vector[op_index++] = tag;
    m.matrix    = po.buff;
    m.ldm       = po.leading_dim;
    m.stor_type = to_dlp_type(po.dtype);
    // sf is pre-allocated by the caller; matrix add/mul operands have no
    // per-element scale, so point at the shared ONE_F32 constant.
    m.sf->scale_factor     = get_void_ptr(ONE_F32);
    m.sf->scale_factor_len = 1;
  };

  for (const auto &po : postops) {
    switch (po.po_type) {
    case post_op_type_t::relu:
      put_eltwise(RELU);
      break;
    case post_op_type_t::leaky_relu:
      put_eltwise(PRELU, get_void_ptr(LEAKY_RELU_SLOPE_DEFAULT));
      break;
    case post_op_type_t::gelu_tanh:
      put_eltwise(GELU_TANH);
      break;
    case post_op_type_t::gelu_erf:
      put_eltwise(GELU_ERF);
      break;
    case post_op_type_t::sigmoid:
      put_eltwise(SIGMOID);
      break;
    case post_op_type_t::swish:
      put_eltwise(SWISH, get_void_ptr(ONE_F32));
      break;
    case post_op_type_t::tanh:
      put_eltwise(TANH);
      break;
    // clip(x; lo, hi): bounds from matmul_post_op::alpha (lower), ::beta (upper).
    case post_op_type_t::clip:
      put_eltwise(CLIP, get_void_ptr(po.alpha), get_void_ptr(po.beta));
      break;
    case post_op_type_t::binary_add:
      put_matrix(md->matrix_add, matrix_add_index, MATRIX_ADD, po);
      break;
    case post_op_type_t::binary_mul:
      put_matrix(md->matrix_mul, matrix_mul_index, MATRIX_MUL, po);
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
                              int64_t K, int64_t N, data_type_t wei_dt) {
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
  if (wei_dt == data_type_t::u4) {
    dlp_metadata->pre_ops->b_zp = static_cast<dlp_zp_t *>(malloc(sizeof(dlp_zp_t)));
    if (dlp_metadata->pre_ops->b_zp) {
      size_t zp_elements = get_num_elements(wei_zp.dims);
      dlp_metadata->pre_ops->b_zp->zero_point_len = zp_elements;
      dlp_metadata->pre_ops->b_zp->zero_point = const_cast<void *>(wei_zp.buff);
      dlp_metadata->pre_ops->b_zp->zero_point_type = wei_zp.dt == data_type_t::s8 ?
          DLP_S8 : DLP_BF16;
    }
  }
  else {
    dlp_metadata->pre_ops->b_zp = nullptr;
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
  bool is_woq = (dtypes.wei == data_type_t::s4 ||
                 dtypes.wei == data_type_t::u4) &&
                dtypes.src == data_type_t::bf16 &&
                kernel == zendnnl::ops::matmul_algo_t::aocl_dlp_blocked;

  // Check if this is INT8 quantization case
  bool is_int8 = dtypes.wei == data_type_t::s8;

  bool is_non_quant_src_int8 = (dtypes.src == data_type_t::bf16 ||
                                dtypes.src == data_type_t::f32) &&
                               is_int8;

  size_t src_scale_nelems = get_num_elements(
                              lowoha_param.quant_params.src_scale.dims);
  bool is_sym_quant = is_int8 && dtypes.src == data_type_t::s8 &&
                      !lowoha_param.quant_params.src_zp.buff &&
                      src_scale_nelems > 1 &&
                      (dtypes.dst == data_type_t::f32 || dtypes.dst == data_type_t::bf16);

  // Count INT8 scale post-ops (sym_quant scales go via post_op_grp, not SCALE post-ops)
  int int8_scale_count = 0;
  if (is_int8) {
    if (lowoha_param.quant_params.src_scale.buff && !is_non_quant_src_int8 &&
        !is_sym_quant) {
      int8_scale_count++;
    }
    if (lowoha_param.quant_params.wei_scale.buff && !is_sym_quant) {
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
  dlp_metadata->a_pre_quant = nullptr;
  dlp_metadata->a_post_quant = nullptr;

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
    case post_op_type_t::clip:
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
  // Allocate pre-quantization INT8 scale
  if (is_non_quant_src_int8) {
    dlp_metadata->a_pre_quant = static_cast<dlp_quant_op *>(calloc(1,
                                sizeof(dlp_quant_op)));
    if (!dlp_metadata->a_pre_quant) {
      cleanup_dlp_post_op(dlp_metadata);
      return nullptr;
    }
    dlp_metadata->a_pre_quant->scl = static_cast<dlp_sf_t *>(calloc(1,
                                     sizeof(dlp_sf_t)));
    if (!dlp_metadata->a_pre_quant->scl) {
      cleanup_dlp_post_op(dlp_metadata);
      return nullptr;
    }
    dlp_metadata->a_post_quant = static_cast<dlp_quant_op *>(calloc(1,
                                 sizeof(dlp_quant_op)));
    if (!dlp_metadata->a_post_quant) {
      cleanup_dlp_post_op(dlp_metadata);
      return nullptr;
    }
    dlp_metadata->a_post_quant->scl = static_cast<dlp_sf_t *>(calloc(1,
                                      sizeof(dlp_sf_t)));
    if (!dlp_metadata->a_post_quant->scl) {
      cleanup_dlp_post_op(dlp_metadata);
      return nullptr;
    }
    if (lowoha_param.quant_params.src_zp.buff) {
      dlp_metadata->a_pre_quant->zp = static_cast<dlp_zp_t *>(calloc(1,
                                      sizeof(dlp_zp_t)));
      if (!dlp_metadata->a_pre_quant->zp) {
        cleanup_dlp_post_op(dlp_metadata);
        return nullptr;
      }
      // Set zp for pre-quantization
      dlp_metadata->a_pre_quant->zp->zero_point = const_cast<void *>
          (lowoha_param.quant_params.src_zp.buff);
      dlp_metadata->a_pre_quant->zp->zero_point_len = 1;
      dlp_metadata->a_pre_quant->zp->zero_point_type = to_dlp_type(
            lowoha_param.quant_params.src_zp.dt);
      dlp_metadata->a_pre_quant->symmetric = false;
      // Set zp for post-quantization
      dlp_metadata->a_post_quant->symmetric = false;
      dlp_metadata->a_post_quant->zp = dlp_metadata->a_pre_quant->zp;
    }
    else {
      dlp_metadata->a_pre_quant->symmetric = true;
      dlp_metadata->a_pre_quant->zp = nullptr;
      dlp_metadata->a_post_quant->symmetric = true;
      dlp_metadata->a_post_quant->zp = nullptr;
    }
    // Set scale factor for pre-quantization
    // Validate that src_scale buffer exists
    if (!lowoha_param.quant_params.src_scale.buff) {
      log_error("BF16-INT8: src_scale buffer is null");
      cleanup_dlp_post_op(dlp_metadata);
      return nullptr;
    }
    dlp_metadata->a_pre_quant->group_size = 0;
    dlp_metadata->a_pre_quant->src_type = to_dlp_type(dtypes.src);
    dlp_metadata->a_pre_quant->dst_type = DLP_S8;
    dlp_metadata->a_pre_quant->scl->scale_factor = malloc(sizeof(float));
    if (!dlp_metadata->a_pre_quant->scl->scale_factor) {
      log_error("BF16-INT8: Failed to allocate pre_quant scale_factor");
      cleanup_dlp_post_op(dlp_metadata);
      return nullptr;
    }
    float inv_scale_factor = 1.0f / (static_cast<const float *>
                                     (lowoha_param.quant_params.src_scale.buff))[0];
    *static_cast<float *>(dlp_metadata->a_pre_quant->scl->scale_factor) =
      inv_scale_factor;
    dlp_metadata->a_pre_quant->scl->scale_factor_len = 1;
    dlp_metadata->a_pre_quant->scl->scale_factor_type = DLP_F32;
    // Set scale factor for post-quantization
    dlp_metadata->a_post_quant->group_size = 0;
    dlp_metadata->a_post_quant->src_type = to_dlp_type(dtypes.src);
    dlp_metadata->a_post_quant->dst_type = DLP_S8;
    dlp_metadata->a_post_quant->scl->scale_factor = const_cast<void *>
        (lowoha_param.quant_params.src_scale.buff);
    dlp_metadata->a_post_quant->scl->scale_factor_len = 1;
    dlp_metadata->a_post_quant->scl->scale_factor_type = DLP_F32;
  }
  if (is_sym_quant) {
    int64_t src_group_size = (src_scale_nelems == static_cast<size_t>(M))
                             ? K : K / (static_cast<int64_t>(src_scale_nelems) / M);

    dlp_metadata->post_op_grp = static_cast<dlp_group_post_op *>(calloc(1,
                                sizeof(dlp_group_post_op)));
    if (!dlp_metadata->post_op_grp) {
      cleanup_dlp_post_op(dlp_metadata);
      return nullptr;
    }
    dlp_metadata->post_op_grp->group_size = static_cast<int>(src_group_size);
    dlp_metadata->post_op_grp->seq_length = 1;

    dlp_metadata->post_op_grp->a_scl = static_cast<dlp_sf_t *>(calloc(1,
                                       sizeof(dlp_sf_t)));
    if (!dlp_metadata->post_op_grp->a_scl) {
      cleanup_dlp_post_op(dlp_metadata);
      return nullptr;
    }
    dlp_metadata->post_op_grp->a_scl->scale_factor =
      const_cast<void *>(lowoha_param.quant_params.src_scale.buff);
    dlp_metadata->post_op_grp->a_scl->scale_factor_len = src_scale_nelems;
    dlp_metadata->post_op_grp->a_scl->scale_factor_type =
      to_dlp_type(lowoha_param.quant_params.src_scale.dt);

    size_t wei_scale_nelems = get_num_elements(
                                lowoha_param.quant_params.wei_scale.dims);
    dlp_metadata->post_op_grp->b_scl = static_cast<dlp_sf_t *>(calloc(1,
                                       sizeof(dlp_sf_t)));
    if (!dlp_metadata->post_op_grp->b_scl) {
      cleanup_dlp_post_op(dlp_metadata);
      return nullptr;
    }
    dlp_metadata->post_op_grp->b_scl->scale_factor =
      const_cast<void *>(lowoha_param.quant_params.wei_scale.buff);
    dlp_metadata->post_op_grp->b_scl->scale_factor_len = wei_scale_nelems;
    dlp_metadata->post_op_grp->b_scl->scale_factor_type =
      to_dlp_type(lowoha_param.quant_params.wei_scale.dt);

    dlp_metadata->post_op_grp->a_zp = nullptr;
    dlp_metadata->post_op_grp->b_zp = nullptr;
  }
  // Allocate scale for INT8
  if (scale_count > 0) {
    dlp_metadata->scale = static_cast<dlp_scale_t *>(calloc(scale_count,
                          sizeof(dlp_scale_t)));
    if (!dlp_metadata->scale) {
      cleanup_dlp_post_op(dlp_metadata);
      return nullptr;
    }
    // Allocate nested sf and zp structures for each scale
    for (int i = 0; i < scale_count; ++i) {
      dlp_metadata->scale[i].sf = static_cast<dlp_sf_t *>(calloc(1,
                                  sizeof(dlp_sf_t)));
      dlp_metadata->scale[i].zp = static_cast<dlp_zp_t *>(calloc(1,
                                  sizeof(dlp_zp_t)));
      if (!dlp_metadata->scale[i].sf || !dlp_metadata->scale[i].zp) {
        // seq_vector not populated yet; free scale nested allocations manually
        for (int j = 0; j <= i; ++j) {
          if (dlp_metadata->scale[j].sf) {
            free(dlp_metadata->scale[j].sf);
          }
          if (dlp_metadata->scale[j].zp) {
            free(dlp_metadata->scale[j].zp);
          }
        }
        free(dlp_metadata->scale);
        dlp_metadata->scale = nullptr;
        cleanup_dlp_post_op(dlp_metadata);
        return nullptr;
      }
    }
  }

  // Allocate memory for different post-op types
  if (bias_count > 0) {
    dlp_metadata->bias = static_cast<dlp_post_op_bias *>(calloc(bias_count,
                         sizeof(dlp_post_op_bias)));
    if (!dlp_metadata->bias) {
      // seq_vector not populated yet; free scale nested allocations manually
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
        dlp_metadata->scale = nullptr;
      }
      cleanup_dlp_post_op(dlp_metadata);
      return nullptr;
    }
  }

  if (eltwise_count > 0) {
    dlp_metadata->eltwise = static_cast<dlp_post_op_eltwise *>(calloc(eltwise_count,
                            sizeof(dlp_post_op_eltwise)));
    if (!dlp_metadata->eltwise) {
      cleanup_dlp_post_op(dlp_metadata);
      return nullptr;
    }
  }

  if (matrix_add_count > 0) {
    dlp_metadata->matrix_add = static_cast<dlp_post_op_matrix_add *>(calloc(
                                 matrix_add_count, sizeof(dlp_post_op_matrix_add)));
    if (!dlp_metadata->matrix_add) {
      cleanup_dlp_post_op(dlp_metadata);
      return nullptr;
    }
    // Allocate sf structures for matrix_add operations
    for (int i = 0; i < matrix_add_count; ++i) {
      dlp_metadata->matrix_add[i].sf = static_cast<dlp_sf_t *>(calloc(1,
                                       sizeof(dlp_sf_t)));
      if (!dlp_metadata->matrix_add[i].sf) {
        // Clean up partially allocated sf structures, then full cleanup
        for (int j = 0; j < i; ++j) {
          free(dlp_metadata->matrix_add[j].sf);
        }
        free(dlp_metadata->matrix_add);
        dlp_metadata->matrix_add = nullptr;
        cleanup_dlp_post_op(dlp_metadata);
        return nullptr;
      }
      dlp_metadata->matrix_add[i].sf->scale_factor_type = to_dlp_type(
            data_type_t::f32);
    }
  }

  if (matrix_mul_count > 0) {
    dlp_metadata->matrix_mul = static_cast<dlp_post_op_matrix_mul *>(calloc(
                                 matrix_mul_count, sizeof(dlp_post_op_matrix_mul)));
    if (!dlp_metadata->matrix_mul) {
      // seq_vector not populated yet; free matrix_add[i].sf manually
      if (dlp_metadata->matrix_add) {
        for (int i = 0; i < matrix_add_count; ++i) {
          if (dlp_metadata->matrix_add[i].sf) {
            free(dlp_metadata->matrix_add[i].sf);
          }
        }
        free(dlp_metadata->matrix_add);
        dlp_metadata->matrix_add = nullptr;
      }
      cleanup_dlp_post_op(dlp_metadata);
      return nullptr;
    }
    // Allocate sf structures for matrix_mul operations
    for (int i = 0; i < matrix_mul_count; ++i) {
      dlp_metadata->matrix_mul[i].sf = static_cast<dlp_sf_t *>(calloc(1,
                                       sizeof(dlp_sf_t)));
      if (!dlp_metadata->matrix_mul[i].sf) {
        // Clean up partially allocated sf structures, then full cleanup
        for (int j = 0; j < i; ++j) {
          free(dlp_metadata->matrix_mul[j].sf);
        }
        free(dlp_metadata->matrix_mul);
        dlp_metadata->matrix_mul = nullptr;
        // seq_vector not populated yet; free matrix_add[i].sf manually
        if (dlp_metadata->matrix_add) {
          for (int k = 0; k < matrix_add_count; ++k) {
            if (dlp_metadata->matrix_add[k].sf) {
              free(dlp_metadata->matrix_add[k].sf);
            }
          }
          free(dlp_metadata->matrix_add);
          dlp_metadata->matrix_add = nullptr;
        }
        cleanup_dlp_post_op(dlp_metadata);
        return nullptr;
      }
      dlp_metadata->matrix_mul[i].sf->scale_factor_type = to_dlp_type(
            data_type_t::f32);
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
    dlp_metadata->bias[bias_index].bias_len = N;
    bias_index++;
  }
  else if (zp_comp_ndim == 2 && zp_comp_acc) {
    dlp_metadata->seq_vector[op_index++] = MATRIX_ADD;
    dlp_metadata->matrix_add[matrix_add_index].matrix = zp_comp_acc;
    dlp_metadata->matrix_add[matrix_add_index].stor_type = DLP_S32;
    dlp_metadata->matrix_add[matrix_add_index].ldm = N;
    // Point scale factor at the shared ONE_F32 constant (default 1.0, read-only).
    dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor = get_void_ptr(
          ONE_F32);
    dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor_len = 1;
    dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor_type = DLP_F32;
    matrix_add_index++;
  }

  // For INT8: Add source scale (skip for sym_quant, handled via post_op_grp)
  if (is_int8 && lowoha_param.quant_params.src_scale.buff &&
      !is_non_quant_src_int8 && !is_sym_quant) {
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

  // For INT8: Add weight scale (skip for sym_quant, handled via post_op_grp)
  if (is_int8 && lowoha_param.quant_params.wei_scale.buff && !is_sym_quant) {
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
    dlp_metadata->bias[bias_index].zp = nullptr; // No zero point for bias
    dlp_metadata->bias[bias_index].bias_len = N;
    bias_index++;
  }

  // Add post-ops
  setup_dlp_postops(dlp_metadata, lowoha_param.postop_,
                    op_index, eltwise_index, matrix_add_index, matrix_mul_index);

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
    setup_woq_pre_ops(dlp_metadata, lowoha_param, K, N, dtypes.wei);
  }

  return dlp_metadata;
}

// Cleanup functions for post-op structures
void cleanup_dlp_post_op(dlp_metadata_t *aocl_po) {
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

    // Clean up eltwise operations.
    if (aocl_po->eltwise) {
      free(aocl_po->eltwise);
    }

    // Clean up bias operations
    if (aocl_po->bias) {
      free(aocl_po->bias);
    }

    // Clean up matrix operations.
    if (aocl_po->matrix_add) {
      for (int i = 0; i < matrix_add_count; i++) {
        if (aocl_po->matrix_add[i].sf) {
          free(aocl_po->matrix_add[i].sf);
        }
      }
      free(aocl_po->matrix_add);
    }

    if (aocl_po->matrix_mul) {
      for (int i = 0; i < matrix_mul_count; i++) {
        if (aocl_po->matrix_mul[i].sf) {
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
        aocl_po->pre_ops->b_zp->zero_point = nullptr;
        free(aocl_po->pre_ops->b_zp);
      }
      free(aocl_po->pre_ops);
    }

    if (aocl_po->post_op_grp) {
      if (aocl_po->post_op_grp->a_scl) {
        free(aocl_po->post_op_grp->a_scl);
      }
      if (aocl_po->post_op_grp->b_scl) {
        free(aocl_po->post_op_grp->b_scl);
      }
      if (aocl_po->post_op_grp->a_zp) {
        free(aocl_po->post_op_grp->a_zp);
      }
      if (aocl_po->post_op_grp->b_zp) {
        free(aocl_po->post_op_grp->b_zp);
      }
      free(aocl_po->post_op_grp);
    }

    if (aocl_po->seq_vector) {
      free(aocl_po->seq_vector);
    }
    if (aocl_po->a_pre_quant) {
      if (aocl_po->a_pre_quant->scl) {
        free(aocl_po->a_pre_quant->scl->scale_factor);
        free(aocl_po->a_pre_quant->scl);
      }
      if (aocl_po->a_pre_quant->zp) {
        free(aocl_po->a_pre_quant->zp);
      }
      free(aocl_po->a_pre_quant);
    }
    if (aocl_po->a_post_quant) {
      if (aocl_po->a_post_quant->scl) {
        free(aocl_po->a_post_quant->scl);
      }
      free(aocl_po->a_post_quant);
    }
    free(aocl_po);
  }
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
