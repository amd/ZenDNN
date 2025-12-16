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
#include "operators/matmul/aocl_blis/matmul_aocl_dlp_utils.hpp"
#include "common/data_types.hpp"

namespace zendnnl {
namespace ops {

inline void eltwise_init(dlp_metadata_t *&aocl_dlp_po_ptr, int eltwise_count,
                         DLP_ELT_ALGO_TYPE algo_type) {
  (aocl_dlp_po_ptr->eltwise[eltwise_count]).algo.algo_type = algo_type;
}

//Returns AOCL data type
DLP_TYPE get_aocl_store_type(data_type_t dt) {
  switch (dt) {
  case data_type_t::f32:
    return DLP_TYPE::DLP_F32 ;
    break;
  case data_type_t::bf16:
    return DLP_TYPE::DLP_BF16 ;
    break;
  case data_type_t::s32:
    return DLP_TYPE::DLP_S32 ;
    break;
  case data_type_t::s8:
    return DLP_TYPE::DLP_S8 ;
    break;
  case data_type_t::u8:
    return DLP_TYPE::DLP_U8 ;
    break;
  case data_type_t::s4:
    return DLP_TYPE::DLP_S4 ;
    break;
  default:
    break;
  };
  return DLP_TYPE::DLP_INVALID;
}

template <typename T>
size_t aocl_dlp_utils_t::reorder_weights_execute(
  const void *weights,
  const int k,
  const int n,
  const int ldb,
  const char order,
  const char trans,
  get_reorder_buff_size_func_ptr get_reorder_buf_size,
  reorder_func_ptr<T> reorder_func) {
  LOG_DEBUG_INFO("Reodering weights aocl_dlp_utils_t");
  log_info("DLP reorder weights");
  size_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, 'B',
                                 k, n, nullptr);
  /*TODO: add support for tensor which will wrap the pointer instead of raw buffer*/
  size_t alignment = 64;
  size_t aligned_size = (b_reorder_buf_siz_req + alignment - 1) & ~(alignment - 1);
  reordered_weights_ptr = aligned_alloc(alignment, aligned_size);
  reorder_func(order, trans, 'B', (T *)weights, (T *)reordered_weights_ptr, k, n,
               ldb, nullptr);

  return b_reorder_buf_siz_req;
}

status_t aocl_dlp_utils_t::set_runtime_post_op_buffer(tensor_map_type
    &inputs_, bool is_bias, tensor_t &output_tensor) {
  uint32_t max_matrix_mul_po = post_op_size["binary_mul_2d"] +
                               post_op_size["binary_mul_1d"];
  uint32_t max_matrix_add_po = post_op_size["binary_add_2d"] +
                               post_op_size["binary_add_1d"] -
                               (zp_comp_ndim != 0 ? (uint32_t)1 : (uint32_t)0); //Remove zp comp

  bool is_dst_scale_zp = output_tensor.is_quantized();
  // Find 1d_mul idx for aocl post-op
  // (src, wei scales are applied first) and dst scale as last
  size_t mul_idx_1d = post_op_size["scales"] - is_dst_scale_zp;
  size_t mul_idx_2d = 0;
  // If bias is present, add one for bias
  // If zp_comp_ndim is 1d then increment
  size_t add_idx_1d = (is_bias ? 1 : 0) + (zp_comp_ndim == 1 ? 1 : 0);
  // If zp_comp_ndim is 2d then increment
  size_t add_idx_2d = (zp_comp_ndim == 2 ? 1 : 0);
  if (inputs_.size() > max_matrix_mul_po + max_matrix_add_po) {
    // Set Matrix Mul buffer
    for (size_t mul_idx=0; mul_idx < max_matrix_mul_po ; mul_idx++) {
      // name of tensor should be binary_mul_<num>
      std::string key_mul = "binary_mul_tensor_" + std::to_string(mul_idx);
      auto found_obj_mul = inputs_.find(key_mul);
      if (found_obj_mul != inputs_.end()) {
        auto mul_buff_tensor = inputs_[key_mul];
        if (found_obj_mul->second.get_size().size() == 1 &&
            aocl_dlp_po_ptr->bias != nullptr) {
          (aocl_dlp_po_ptr->scale + mul_idx_1d)->sf->scale_factor  =
            mul_buff_tensor.get_raw_handle_unsafe();
          (aocl_dlp_po_ptr->scale + mul_idx_1d)->zp->zero_point    = &dummy_zp;
          (aocl_dlp_po_ptr->scale + mul_idx_1d)->sf->scale_factor_type =
            get_aocl_store_type(
              mul_buff_tensor.get_data_type());
          (aocl_dlp_po_ptr->scale + mul_idx_1d)->zp->zero_point_type = DLP_TYPE::DLP_S32;
          mul_idx_1d++;
        }
        else if (found_obj_mul->second.get_size().size() == 2 &&
                 aocl_dlp_po_ptr->matrix_mul != nullptr) {
          (aocl_dlp_po_ptr->matrix_mul + mul_idx_2d)->matrix =
            mul_buff_tensor.get_raw_handle_unsafe();
          (aocl_dlp_po_ptr->matrix_mul + mul_idx_2d)->stor_type = get_aocl_store_type(
                mul_buff_tensor.get_data_type());
          (aocl_dlp_po_ptr->matrix_mul + mul_idx_2d)->ldm = mul_buff_tensor.get_size(1);
          mul_idx_2d++;
        }
        else {
          log_error("Improper input shape for matrix mul post-ops");
          return status_t::failure;
        }
      }
      else {
        log_error("Not enough inputs passed for matrix mul post-ops");
        return status_t::failure;
      }
    }
    // Set Matrix Add buffer
    for (size_t add_idx=0; add_idx < max_matrix_add_po ; add_idx++) {
      // name of tensor should be binary_add_<num>
      std::string key_add = "binary_add_tensor_" + std::to_string(add_idx);
      auto found_obj_add = inputs_.find(key_add);
      if (found_obj_add != inputs_.end()) {
        auto add_buff_tensor = inputs_[key_add];
        if (found_obj_add->second.get_size().size() == 1 &&
            aocl_dlp_po_ptr->bias != nullptr) {
          (aocl_dlp_po_ptr->bias + add_idx_1d)->bias = (void *)
              add_buff_tensor.get_raw_handle_unsafe();
          (aocl_dlp_po_ptr->bias + add_idx_1d)->stor_type = get_aocl_store_type(
                add_buff_tensor.get_data_type());
          add_idx_1d++;
        }
        else if (found_obj_add->second.get_size().size() == 2 &&
                 aocl_dlp_po_ptr->matrix_add != nullptr) {
          (aocl_dlp_po_ptr->matrix_add + add_idx_2d)->matrix
            = add_buff_tensor.get_raw_handle_unsafe();
          (aocl_dlp_po_ptr->matrix_add + add_idx_2d)->stor_type
            = get_aocl_store_type(add_buff_tensor.get_data_type());
          (aocl_dlp_po_ptr->matrix_add + add_idx_2d)->ldm
            = add_buff_tensor.get_size(1);
          add_idx_2d++;
        }
      }
      else {
        log_error("Not enough inputs passed for matrix add post-ops");
        return status_t::failure;
      }
    }
  }
  else {
    log_error("Not enough inputs passed for buffer based post-ops");
    return status_t::failure;
  }
  // Set Dst scale and dst zero-point buffer
  if (is_dst_scale_zp) {
    auto   dst_scale_ = output_tensor.get_quant_scale_raw_handle_const();
    auto   dst_zp_    = output_tensor.get_quant_subtype() ==
                        quant_subtype_t::asymmetric ? output_tensor.get_quant_zero_raw_handle_const() :
                        nullptr;
    if (dst_scale_ != nullptr || dst_zp_ != nullptr) {
      // Set dst scale
      (aocl_dlp_po_ptr->scale + mul_idx_1d)->sf->scale_factor  =
        dst_scale_ != nullptr ? const_cast<void *>(dst_scale_) : &dummy_scale;
      (aocl_dlp_po_ptr->scale + mul_idx_1d)->sf->scale_factor_type =
        dst_scale_ != nullptr ? get_aocl_store_type(
          output_tensor.get_quant_scale_data_type()) : DLP_TYPE::DLP_F32;
      (aocl_dlp_po_ptr->scale + mul_idx_1d)->sf->scale_factor_len =
        dst_scale_ != nullptr ? compute_product(output_tensor.get_quant_scale_size()) :
        1;
      // Set dst zero-point
      (aocl_dlp_po_ptr->scale + mul_idx_1d)->zp->zero_point =
        dst_zp_ != nullptr ? const_cast<void *>(dst_zp_) : &dummy_zp;
      (aocl_dlp_po_ptr->scale + mul_idx_1d)->zp->zero_point_type =
        dst_zp_ != nullptr ? get_aocl_store_type(
          output_tensor.get_quant_zero_data_type()) : DLP_TYPE::DLP_S32;
      (aocl_dlp_po_ptr->scale + mul_idx_1d)->zp->zero_point_len =
        dst_zp_ != nullptr ? compute_product(output_tensor.get_quant_zero_size()) : 1;
      mul_idx_1d++;
    }
  }
  return status_t::success;
}

status_t aocl_dlp_utils_t::aocl_post_op_memory_alloc(const
    std::vector<post_op_t> &post_op_vec_, bool is_bias,
    std::map<std::string, zendnnl::memory::tensor_t> &inputs_) {
  LOG_DEBUG_INFO("Allocating memory for post_ops in aocl_dlp_utils_t");
  //Allocate memory
  size_t max_post_ops = post_op_vec_.size();
  int num_post_ops_1d_add     = post_op_size["binary_add_1d"];
  int num_post_ops_binary_add = post_op_size["binary_add_2d"];
  int num_post_ops_1d_mul     = post_op_size["binary_mul_1d"];
  int num_post_ops_binary_mul = post_op_size["binary_mul_2d"];
  int num_post_ops_eltwise    = post_op_size["eltwise"];
  int num_post_ops_scale      = post_op_size["scales"];

  bool alloc_aocl_po = (is_bias || max_post_ops || num_post_ops_1d_add ||
                        num_post_ops_binary_add || num_post_ops_eltwise ||
                        num_post_ops_binary_mul || num_post_ops_1d_mul ||
                        num_post_ops_scale);

  if (alloc_aocl_po) {
    for (size_t i = 0; i < max_post_ops; ++ i) {
      post_op_t zen_po = post_op_vec_[i];
      switch (zen_po.type) {
      case post_op_type_t::relu:
        num_post_ops_eltwise++;
        break;
      case post_op_type_t::leaky_relu:
        num_post_ops_eltwise++;
        break;
      case post_op_type_t::gelu_tanh:
        num_post_ops_eltwise++;
        break;
      case post_op_type_t::gelu_erf:
        num_post_ops_eltwise++;
        break;
      case post_op_type_t::tanh:
        num_post_ops_eltwise++;
        break;
      case post_op_type_t::swish:
        num_post_ops_eltwise++;
        break;
      case post_op_type_t::sigmoid:
        num_post_ops_eltwise++;
        break;
      case post_op_type_t::clip:
        num_post_ops_eltwise++;
        break;
      case post_op_type_t::binary_add:
        {
          auto it = inputs_.find(zen_po.binary_add_params.tensor_name);
          if (it != inputs_.end() && it->second.get_size().size() == 1) {
            num_post_ops_1d_add++;
          }
          else {
            num_post_ops_binary_add++;
          }
        }
        break;
      case post_op_type_t::binary_mul:
        {
          auto it = inputs_.find(zen_po.binary_mul_params.tensor_name);
          if (it != inputs_.end() && it->second.get_size().size() == 1) {
            num_post_ops_1d_mul++;
          }
          else {
            num_post_ops_binary_mul++;
          }
        }
        break;
      default:
        log_error("This postop in aocl is not supported");
        return status_t::failure;
      }
    }
    int total_bias_mem            = num_post_ops_1d_add + (is_bias ? 1 : 0);
    aocl_dlp_po_ptr->bias        = (dlp_post_op_bias *) calloc(total_bias_mem,
                                   sizeof(dlp_post_op_bias));
    aocl_dlp_po_ptr->scale       = (dlp_scale_t *) calloc(
                                     num_post_ops_1d_mul + num_post_ops_scale,
                                     sizeof(dlp_scale_t));
    aocl_dlp_po_ptr->eltwise     = (dlp_post_op_eltwise *) calloc(
                                     num_post_ops_eltwise,
                                     sizeof(dlp_post_op_eltwise));
    aocl_dlp_po_ptr->matrix_add  = (dlp_post_op_matrix_add *) calloc(
                                     num_post_ops_binary_add,
                                     sizeof(dlp_post_op_matrix_add));
    aocl_dlp_po_ptr->matrix_mul  = (dlp_post_op_matrix_mul *) calloc(
                                     num_post_ops_binary_mul,
                                     sizeof(dlp_post_op_matrix_mul));
    post_op_size["eltwise"]       = num_post_ops_eltwise;
    post_op_size["binary_add_2d"] = num_post_ops_binary_add;
    post_op_size["binary_mul_2d"] = num_post_ops_binary_mul;
    post_op_size["binary_add_1d"] = num_post_ops_1d_add;
    post_op_size["binary_mul_1d"] = num_post_ops_1d_mul;

    // Allocate nested structures for scale and matrix ops to prevent null deref
    for (int i = 0; i < num_post_ops_1d_mul + num_post_ops_scale; ++i) {
      aocl_dlp_po_ptr->scale[i].sf = (dlp_sf_t *) calloc(1, sizeof(dlp_sf_t));
      aocl_dlp_po_ptr->scale[i].zp = (dlp_zp_t *) calloc(1, sizeof(dlp_zp_t));
    }
    for (int i = 0; i < num_post_ops_binary_add; ++i) {
      aocl_dlp_po_ptr->matrix_add[i].sf = (dlp_sf_t *) calloc(1, sizeof(dlp_sf_t));
    }
    for (int i = 0; i < num_post_ops_binary_mul; ++i) {
      aocl_dlp_po_ptr->matrix_mul[i].sf = (dlp_sf_t *) calloc(1, sizeof(dlp_sf_t));
    }
  }
  return status_t::success;
}

status_t aocl_dlp_utils_t::aocl_post_op_initialize(const std::vector<post_op_t>
    &post_op_vec_, int &post_op_count, bool is_bias,
    std::map<std::string, zendnnl::memory::tensor_t> &inputs_,
    tensor_t &output_tensor,
    size_t eltwise_index, size_t add_index_2d, size_t mul_index_1d,
    size_t mul_index_2d) {
  LOG_DEBUG_INFO("Initializing aocl post-op in aocl_dlp_utils_t");
  //add remaining post-ops
  size_t max_post_ops = post_op_vec_.size();

  for (size_t i = 0; i < max_post_ops; ++ i) {
    post_op_t zen_po = post_op_vec_[i];

    switch (zen_po.type) {
    case post_op_type_t::relu:
      log_info("Adding relu post-op");
      eltwise_init(aocl_dlp_po_ptr, eltwise_index, DLP_ELT_ALGO_TYPE::RELU);
      eltwise_index++;
      aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::ELTWISE;
      break;
    case post_op_type_t::leaky_relu:
      log_info("Adding leaky_relu post-op");
      eltwise_init(aocl_dlp_po_ptr, eltwise_index, DLP_ELT_ALGO_TYPE::PRELU);
      (aocl_dlp_po_ptr->eltwise[eltwise_index]).algo.alpha = malloc(sizeof(float));
      *((float *)(aocl_dlp_po_ptr->eltwise[eltwise_index]).algo.alpha)
        = zen_po.leaky_relu_params.nslope;
      eltwise_index++;
      aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::ELTWISE;
      break;
    case post_op_type_t::gelu_tanh:
      log_info("Adding gelu_tanh post-op");
      eltwise_init(aocl_dlp_po_ptr, eltwise_index, DLP_ELT_ALGO_TYPE::GELU_TANH);
      eltwise_index++;
      aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::ELTWISE;
      break;
    case post_op_type_t::gelu_erf:
      log_info("Adding gelu_erf post-op");
      eltwise_init(aocl_dlp_po_ptr, eltwise_index, DLP_ELT_ALGO_TYPE::GELU_ERF);
      eltwise_index++;
      aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::ELTWISE;
      break;
    case post_op_type_t::tanh:
      log_info("Adding tanh post-op");
      eltwise_init(aocl_dlp_po_ptr, eltwise_index, DLP_ELT_ALGO_TYPE::TANH);
      eltwise_index++;
      aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::ELTWISE;
      break;
    case post_op_type_t::swish:
      log_info("Adding swish post-op");
      eltwise_init(aocl_dlp_po_ptr, eltwise_index, DLP_ELT_ALGO_TYPE::SWISH);
      (aocl_dlp_po_ptr->eltwise[eltwise_index]).algo.alpha = malloc(sizeof(float));
      *((float *)(aocl_dlp_po_ptr->eltwise[eltwise_index]).algo.alpha) =
        zen_po.swish_params.scale;
      eltwise_index++;
      aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::ELTWISE;
      break;
    case post_op_type_t::sigmoid:
      log_info("Adding sigmoid post-op");
      eltwise_init(aocl_dlp_po_ptr, eltwise_index, DLP_ELT_ALGO_TYPE::SIGMOID);
      eltwise_index++;
      aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::ELTWISE;
      break;
    case post_op_type_t::clip:
      log_info("Adding clip post-op");
      eltwise_init(aocl_dlp_po_ptr, eltwise_index, DLP_ELT_ALGO_TYPE::CLIP);
      (aocl_dlp_po_ptr->eltwise[eltwise_index]).algo.alpha = malloc(sizeof(float));
      *((float *)(aocl_dlp_po_ptr->eltwise[eltwise_index]).algo.alpha) =
        zen_po.clip_params.lower;
      (aocl_dlp_po_ptr->eltwise[eltwise_index]).algo.beta = malloc(sizeof(float));
      *((float *)(aocl_dlp_po_ptr->eltwise[eltwise_index]).algo.beta) =
        zen_po.clip_params.upper;
      eltwise_index++;
      aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::ELTWISE;
      break;
    case post_op_type_t::binary_add:
      {
        log_info("Adding binary_add post-op");
        auto it = inputs_.find(zen_po.binary_add_params.tensor_name);
        if (it != inputs_.end() && it->second.get_size().size() == 1) {
          aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::BIAS;
        }
        else {
          (aocl_dlp_po_ptr->matrix_add + add_index_2d)->sf->scale_factor = malloc(sizeof(
                float));
          *((float *)(aocl_dlp_po_ptr->matrix_add[add_index_2d]).sf->scale_factor) =
            zen_po.binary_add_params.scale;
          (aocl_dlp_po_ptr->matrix_add + add_index_2d)->sf->scale_factor_len = 1;
          aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::MATRIX_ADD;
          add_index_2d++;
        }
      }
      break;
    case post_op_type_t::binary_mul:
      {
        log_info("Adding binary_mul post-op");
        auto it = inputs_.find(zen_po.binary_mul_params.tensor_name);
        if (it != inputs_.end() && it->second.get_size().size() == 1) {
          aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::SCALE;
          (aocl_dlp_po_ptr->scale + mul_index_1d)->sf->scale_factor = NULL;
          (aocl_dlp_po_ptr->scale + mul_index_1d)->zp->zero_point = NULL;
          (aocl_dlp_po_ptr->scale + mul_index_1d)->sf->scale_factor_len = it->second.get_size()[0];
          (aocl_dlp_po_ptr->scale + mul_index_1d)->zp->zero_point_len = 1;
          log_info("Adding done");
          mul_index_1d++;
        }
        else {
          (aocl_dlp_po_ptr->matrix_mul + mul_index_2d)->sf->scale_factor = malloc(sizeof(
                float));
          *((float *)(aocl_dlp_po_ptr->matrix_mul[mul_index_2d]).sf->scale_factor) =
            zen_po.binary_mul_params.scale;
          (aocl_dlp_po_ptr->matrix_mul + mul_index_2d)->sf->scale_factor_len = 1;
          aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::MATRIX_MUL;
          mul_index_2d++;
        }
      }
      break;
    default:
      log_error("This postop in aocl is not supported");
      return status_t::failure;
    }
  }
  return status_t::success;
}

void aocl_dlp_utils_t::zero_point_compensation(
  int M,
  int N,
  int K,
  tensor_t &src,
  tensor_t &wei,
  int32_t src_zero_point,
  int32_t wei_zero_point
) {
  LOG_DEBUG_INFO("Calculating zero-point compensation in zero_point_compensation");

  int src_s0;
  int src_s1;
  int wei_s0;
  int wei_s1;

  if (src.get_order() == "ba") {
    src_s0 = 1; // Stride along the second dimension
    src_s1 = src.get_stride(1);
  }
  else {
    src_s0 = src.get_stride(0);
    src_s1 = 1; // Stride along the second dimension
  }

  // Determine strides for wei tensor
  if (wei.get_order() == "ba") {
    wei_s0 = 1; // Stride along the second dimension
    wei_s1 = wei.get_stride(1);
  }
  else {
    wei_s0 = wei.get_stride(0);
    wei_s1 = 1; // Stride along the second dimension
  }
  char   *src_buff = (char *)src.get_raw_handle_unsafe();
  int8_t *wei_buff = (int8_t *)wei.get_raw_handle_unsafe();
  if (!wei_zero_point && !src_zero_point) {
    return;
  }
  else if (!wei_zero_point) {
    // acc is freed in post_op free function.
    size_t alignment = 64;
    size_t comp_size = (N*sizeof(int32_t) + alignment - 1) &
                       ~(alignment - 1);
    zp_comp_acc = (int32_t *)aligned_alloc(64, comp_size);
    std::vector<int32_t> wei_comp(N,0);

    for (auto k = 0; k < K; ++k) {
      for (auto n = 0; n < N; ++n) {
        if (k == 0) {
          wei_comp[n] = int32_t(0);
        }
        wei_comp[n] += wei_buff[wei_s0 * k + wei_s1 * n];
      }
    }
    for (auto n = 0; n < N; ++n) {
      zp_comp_acc[n] = 0 - src_zero_point * wei_comp[n];
    }
  }
  else if (!src_zero_point) {
    std::vector<int32_t> src_comp(M,0);
    // acc is freed in post_op free function.
    size_t alignment = 64;
    size_t comp_size = (M*N*sizeof(int32_t) + alignment - 1) &
                       ~(alignment - 1);
    zp_comp_acc = (int32_t *)aligned_alloc(64, comp_size);

    for (auto m = 0; m < M; ++m) {
      for (auto k = 0; k < K; ++k) {
        if (k == 0) {
          src_comp[m] = int32_t(0);
        }
        src_comp[m] += src_buff[src_s0 * m + src_s1 * k];
      }
    }

    for (auto m = 0; m < M; ++m) {
      for (auto n = 0; n < N; ++n) {
        zp_comp_acc[m * N + n] = 0 - wei_zero_point * src_comp[m];
      }
    }
  }
  else {
    std::vector<int32_t> src_comp(M,0);
    std::vector<int32_t> wei_comp(N,0);
    // acc is freed in post_op free function.
    size_t alignment = 64;
    size_t comp_size = (M*N*sizeof(int32_t) + alignment - 1) &
                       ~(alignment - 1);
    zp_comp_acc = (int32_t *)aligned_alloc(64, comp_size);
    //Src comp
    for (auto m = 0; m < M; ++m) {
      for (auto k = 0; k < K; ++k) {
        if (k == 0) {
          src_comp[m] = int32_t(0);
        }
        src_comp[m] += src_buff[src_s0 * m + src_s1 * k];
      }
    }

    for (auto k = 0; k < K; ++k) {
      for (auto n = 0; n < N; ++n) {
        if (k == 0) {
          wei_comp[n] = int32_t(0);
        }
        wei_comp[n] += wei_buff[wei_s0 * k + wei_s1 * n];
      }
    }

    for (auto m = 0; m < M; ++m) {
      for (auto n = 0; n < N; ++n) {
        zp_comp_acc[m * N + n] = 0 - src_zero_point * wei_comp[n]
                                 - wei_zero_point * src_comp[m]
                                 + src_zero_point * wei_zero_point * (int)K;
      }
    }
  }
}

status_t aocl_dlp_utils_t::alloc_post_op(const std::vector<post_op_t>
    &post_op_vec_, std::optional<tensor_t> optional_bias_tensor_,
    tensor_t &weight_tensor,
    std::map<std::string, zendnnl::memory::tensor_t> &inputs_,
    zendnnl::memory::tensor_t &output_tensor) {
  LOG_DEBUG_INFO("Allocating post-ops in aocl_dlp_utils_t");

  // Return if post-ops already set for context
  if (aocl_dlp_po_ptr != nullptr) {
    return status_t::success;
  }
  bool is_woq = false;
  auto weight_dtype = weight_tensor.get_data_type();
  if (weight_dtype == data_type_t::s4) {
    is_woq = true;
  }
  // Iterate through each postop, check and add it if needed.
  int post_op_count = 0;
  // Find total number of post-ops with bias and scales
  int total_po = post_op_vec_.size();
  auto src_it        = inputs_.find("matmul_input");
  if (src_it == inputs_.end()) {
    log_error("matmul_input tensor not found in inputs");
    return status_t::failure;
  }

  bool is_quant = (src_it->second.get_data_type() == data_type_t::u8 ||
                   src_it->second.get_data_type() == data_type_t::s8) &&
                  weight_tensor.get_data_type() == data_type_t::s8;

  [[maybe_unused]] uint32_t dim_M           = src_it->second.get_size(0);
  [[maybe_unused]] uint32_t dim_K           = weight_tensor.get_size(0);
  [[maybe_unused]] uint32_t dim_N           = weight_tensor.get_size(1);
  // Src scale
  [[maybe_unused]] void *src_scale_         = nullptr;
  [[maybe_unused]] int src_scale_size       = 0;
  [[maybe_unused]] data_type_t src_scale_dt = data_type_t::f32;
  // Wei scale
  [[maybe_unused]] void *wei_scale_         = nullptr;
  [[maybe_unused]] int wei_scale_size       = 0;
  [[maybe_unused]] data_type_t wei_scale_dt = data_type_t::f32;
  //dst scale
  bool is_dst_scale_                        = false;
  const void *dst_zp_                       = nullptr;
  if (is_quant) {
    auto src_tensor    = inputs_.find("matmul_input")->second;
    auto is_src_scale_ = src_tensor.is_quantized();
    auto is_wei_scale_ = weight_tensor.is_quantized();
    auto src_zp_       = is_src_scale_ &&
                         src_tensor.get_quant_subtype() == quant_subtype_t::asymmetric ?
                         src_tensor.get_quant_zero_raw_handle_const() : nullptr;
    auto wei_zp_       = is_wei_scale_ &&
                         weight_tensor.get_quant_subtype() == quant_subtype_t::asymmetric ?
                         weight_tensor.get_quant_zero_raw_handle_const() : nullptr;
    // Dst quant
    is_dst_scale_      = output_tensor.is_quantized();
    dst_zp_            = is_dst_scale_ &&
                         output_tensor.get_quant_subtype() == quant_subtype_t::asymmetric ?
                         output_tensor.get_quant_zero_raw_handle_const() : nullptr;

    if (is_src_scale_) {
      src_scale_      = const_cast<void *>
                        (src_tensor.get_quant_scale_raw_handle_const());
      // Compute the product of all elements in the vector
      src_scale_size = compute_product(src_tensor.get_quant_scale_size());
      src_scale_dt    = src_tensor.get_quant_scale_data_type();
      total_po++;
      post_op_size["scales"]++;
    }

    if (is_wei_scale_) {
      wei_scale_      = const_cast<void *>
                        (weight_tensor.get_quant_scale_raw_handle_const());
      // Compute the product of all elements in the vector
      wei_scale_size = compute_product(weight_tensor.get_quant_scale_size());
      wei_scale_dt    = weight_tensor.get_quant_scale_data_type();
      total_po++;
      post_op_size["scales"]++;
    }

    // dst scale and zp are applied as a single post-op
    if (is_dst_scale_ || dst_zp_ != nullptr) {
      total_po++;
      post_op_size["scales"]++;
    }
    // Compute zero_point compensation
    int32_t src_zp_val = src_zp_ != nullptr ? read_and_cast<int32_t>(src_zp_,
                         src_tensor.get_quant_zero_data_type()) : (int32_t)0;
    int32_t wei_zp_val = wei_zp_ != nullptr ? read_and_cast<int32_t>(wei_zp_,
                         weight_tensor.get_quant_zero_data_type()) : (int32_t)0;
    if (src_zp_val == 0 && wei_zp_val == 0) {
      zp_comp_ndim = 0;
    }
    else if (wei_zp_val == 0) {
      zero_point_compensation(dim_M, dim_N, dim_K, src_it->second,
                              weight_tensor, src_zp_val, wei_zp_val);
      zp_comp_ndim = 1;
      total_po++;
      post_op_size["binary_add_1d"]++;
    }
    else {
      zero_point_compensation(dim_M, dim_N, dim_K, src_it->second,
                              weight_tensor, src_zp_val, wei_zp_val);
      zp_comp_ndim = 2;
      total_po++;
      post_op_size["binary_add_2d"]++;
    }
  }
  // Add one for bias
  if (optional_bias_tensor_) {
    total_po++;
  }

  if (total_po > 0 || is_woq) {
    //Index for each post-op
    size_t eltwise_index = 0;
    size_t add_index_2d  = 0;
    size_t mul_index_2d  = 0;
    size_t mul_index_1d  = 0;
    size_t bias_index    = 0;

    aocl_dlp_po_ptr = (dlp_metadata_t *) calloc(1, sizeof(dlp_metadata_t));
    if (aocl_dlp_po_ptr == NULL) {
      return status_t::failure;
    }

    //Set all post-ops to NULL
    aocl_dlp_po_ptr->eltwise = NULL;
    aocl_dlp_po_ptr->bias = NULL;
    aocl_dlp_po_ptr->scale = NULL;
    aocl_dlp_po_ptr->matrix_add = NULL;
    aocl_dlp_po_ptr->matrix_mul = NULL;
    aocl_dlp_po_ptr->pre_ops = NULL;
  

    if (is_woq) {
      auto weight_size      = weight_tensor.get_size();
      auto scale_dt         = weight_tensor.get_quant_scale_data_type();
      const void *scale_ptr = weight_tensor.get_quant_scale_raw_handle_const();
      auto scale_size       = weight_tensor.get_quant_scale_size();
      bool is_zero_point    = weight_tensor.get_quant_subtype() == quant_subtype_t::asymmetric;
      auto scale_nelems     = compute_product(scale_size);
      // Calculate group_size: number of consecutive weights in K dimension sharing the same scale
      // AOCL DLP interprets group_size as: how many consecutive elements along K dimension share the same scale
      // For per-channel quantization (scale_nelems == N): group_size = K (all K weights in each column share the same scale)
      // For per-tensor quantization (scale_nelems == 1): group_size = K (all weights share the same scale)
      // For group-wise quantization: group_size = K / (scale_nelems / N)
      int group_size;
      if (static_cast<size_t>(scale_nelems) == weight_size[1]) {
        // Per-channel quantization: each output channel (column) has its own scale
        // All K weights in each column share the same scale
        group_size = static_cast<int>(weight_size[0]);  // K
      } else if (scale_nelems == 1) {
        // Per-tensor quantization: all weights share the same scale
        group_size = static_cast<int>(weight_size[0]);  // K
      } else {
        // Group-wise quantization
        group_size = static_cast<int>(weight_size[0]) / (scale_nelems / static_cast<int>(weight_size[1]));
      }
      aocl_dlp_po_ptr->pre_ops = (dlp_pre_op *)malloc(sizeof(dlp_pre_op));
      (aocl_dlp_po_ptr->pre_ops)->b_zp = (dlp_zp_t *)malloc(sizeof(dlp_zp_t));
      (aocl_dlp_po_ptr->pre_ops)->b_scl = (dlp_sf_t *)malloc(sizeof(dlp_sf_t));
      // Setup zero point for WOQ (asymmetric quantization)
      if (is_zero_point) {
        const void *zp_ptr = weight_tensor.get_quant_zero_raw_handle_const();
        auto zp_size = weight_tensor.get_quant_zero_size();
        auto zp_nelems = compute_product(zp_size);
        ((aocl_dlp_po_ptr->pre_ops)->b_zp)->zero_point = const_cast<void *>(zp_ptr);
        ((aocl_dlp_po_ptr->pre_ops)->b_zp)->zero_point_len = zp_nelems;
        ((aocl_dlp_po_ptr->pre_ops)->b_zp)->zero_point_type = DLP_TYPE::DLP_S8;
      } else {
        ((aocl_dlp_po_ptr->pre_ops)->b_zp)->zero_point = NULL;
        ((aocl_dlp_po_ptr->pre_ops)->b_zp)->zero_point_len = 0;
      }
      ((aocl_dlp_po_ptr->pre_ops)->b_scl)->scale_factor = (float *)scale_ptr;
      ((aocl_dlp_po_ptr->pre_ops)->b_scl)->scale_factor_len = scale_nelems;
      ((aocl_dlp_po_ptr->pre_ops)->b_scl)->scale_factor_type = get_aocl_store_type(scale_dt);
      (aocl_dlp_po_ptr->pre_ops)->seq_length = 1;
      (aocl_dlp_po_ptr->pre_ops)->group_size = group_size;
    }

    if (total_po > 0) {
      
      aocl_dlp_po_ptr->seq_vector = (DLP_POST_OP_TYPE *) calloc(total_po,
                                    sizeof(DLP_POST_OP_TYPE));
      if (aocl_dlp_po_ptr->seq_vector == NULL) {
        free(aocl_dlp_po_ptr);
        return status_t::failure;
      }
      // Allocate memory for post-ops
      if (aocl_post_op_memory_alloc(post_op_vec_, optional_bias_tensor_? true : false,
                                    inputs_)
          != status_t::success) {
        return status_t::failure;
      }
      if (is_quant) {
        // Add zero-point compensation
        if (zp_comp_ndim == 1) {
          aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::BIAS;
          if (aocl_dlp_po_ptr->bias == NULL) {
            return status_t::failure;
          }
          (aocl_dlp_po_ptr->bias + bias_index)->stor_type = DLP_TYPE::DLP_S32;
          (aocl_dlp_po_ptr->bias + bias_index)->bias      = zp_comp_acc;
          bias_index++;
        }
        else if (zp_comp_ndim == 2) {
          aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::MATRIX_ADD;
          if (aocl_dlp_po_ptr->matrix_add == NULL) {
            return status_t::failure;
          }
          (aocl_dlp_po_ptr->matrix_add + add_index_2d)->stor_type = DLP_TYPE::DLP_S32;
          (aocl_dlp_po_ptr->matrix_add + add_index_2d)->ldm = dim_N;
          (aocl_dlp_po_ptr->matrix_add + add_index_2d)->matrix = zp_comp_acc;
          (aocl_dlp_po_ptr->matrix_add + add_index_2d)->sf->scale_factor = malloc(sizeof(
                float));
          *((float *)(aocl_dlp_po_ptr->matrix_add[add_index_2d]).sf->scale_factor) = 1.0f;
          (aocl_dlp_po_ptr->matrix_add + add_index_2d)->sf->scale_factor_len = 1;
          add_index_2d++;
        }

        // Src Scale
        if (src_scale_) {
          aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::SCALE;
          (aocl_dlp_po_ptr->scale + mul_index_1d)->sf->scale_factor  = const_cast<void *>
              (src_scale_);
          (aocl_dlp_po_ptr->scale + mul_index_1d)->sf->scale_factor_type  =
            get_aocl_store_type(src_scale_dt);
          (aocl_dlp_po_ptr->scale + mul_index_1d)->sf->scale_factor_len = src_scale_size;
          (aocl_dlp_po_ptr->scale + mul_index_1d)->zp->zero_point = &dummy_zp;
          (aocl_dlp_po_ptr->scale + mul_index_1d)->zp->zero_point_type =
            DLP_TYPE::DLP_S32;
          (aocl_dlp_po_ptr->scale + mul_index_1d)->zp->zero_point_len = 1;
          mul_index_1d++;
        }
        // Wei Scale
        if (wei_scale_) {
          aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::SCALE;
          (aocl_dlp_po_ptr->scale + mul_index_1d)->sf->scale_factor  = const_cast<void *>
              (wei_scale_);
          (aocl_dlp_po_ptr->scale + mul_index_1d)->sf->scale_factor_type  =
            get_aocl_store_type(wei_scale_dt);
          (aocl_dlp_po_ptr->scale + mul_index_1d)->sf->scale_factor_len = wei_scale_size;
          (aocl_dlp_po_ptr->scale + mul_index_1d)->zp->zero_point = &dummy_zp;
          (aocl_dlp_po_ptr->scale + mul_index_1d)->zp->zero_point_type =
            DLP_TYPE::DLP_S32;
          (aocl_dlp_po_ptr->scale + mul_index_1d)->zp->zero_point_len = 1;
          mul_index_1d++;
        }
      }
      // Add bias postop
      if (optional_bias_tensor_) {
        auto bias_type = optional_bias_tensor_->get_data_type();
        aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::BIAS;
        if (aocl_dlp_po_ptr->bias == NULL) {
          return status_t::failure;
        }
        (aocl_dlp_po_ptr->bias + bias_index)->bias = (void *)
            optional_bias_tensor_->get_raw_handle_unsafe();
        (aocl_dlp_po_ptr->bias + bias_index)->stor_type = get_aocl_store_type(
              bias_type);
        bias_index++;
      }

      //Initialize other post-ops.
      if (aocl_post_op_initialize(post_op_vec_, post_op_count,
                                  optional_bias_tensor_? true : false, inputs_, output_tensor,
                                  eltwise_index, add_index_2d, mul_index_1d, mul_index_2d) != status_t::success) {
        return status_t::failure;
      }
      if (is_dst_scale_ || dst_zp_ != nullptr) {
        aocl_dlp_po_ptr->seq_vector[post_op_count++] = DLP_POST_OP_TYPE::SCALE;
      }
      aocl_dlp_po_ptr->seq_length = post_op_count;
    }
  }
  return status_t::success;
}

void aocl_dlp_utils_t::free_post_op() {
  LOG_DEBUG_INFO("Freeing aocl post-ops from aocl_dlp_utils_t");
  if (aocl_dlp_po_ptr == nullptr) {
    return;
  }

  if (aocl_dlp_po_ptr->pre_ops) {
    if (aocl_dlp_po_ptr->pre_ops->b_scl) {
      free(aocl_dlp_po_ptr->pre_ops->b_scl);
    }
    if (aocl_dlp_po_ptr->pre_ops->b_zp) {
      free(aocl_dlp_po_ptr->pre_ops->b_zp);
    }
    free(aocl_dlp_po_ptr->pre_ops);
  }

  if (aocl_dlp_po_ptr->bias) {
    free(aocl_dlp_po_ptr->bias);
  }

  int count_elt           = 0;
  int count_matrix_add_2d = 0;
  int count_matrix_mul_2d = 0;
  int count_matrix_mul_1d = 0;
  for (int idx = 0; idx < aocl_dlp_po_ptr->seq_length; idx++) {
    if (aocl_dlp_po_ptr->seq_vector[idx] == DLP_POST_OP_TYPE::ELTWISE) {
      if (aocl_dlp_po_ptr->eltwise[count_elt].algo.alpha) {
        free(aocl_dlp_po_ptr->eltwise[count_elt].algo.alpha);
      }
      if (aocl_dlp_po_ptr->eltwise[count_elt].algo.beta) {
        free(aocl_dlp_po_ptr->eltwise[count_elt].algo.beta);
      }
      count_elt++;
    }
    else if (aocl_dlp_po_ptr->seq_vector[idx] == DLP_POST_OP_TYPE::MATRIX_ADD) {
      if (aocl_dlp_po_ptr->matrix_add[count_matrix_add_2d].sf &&
          aocl_dlp_po_ptr->matrix_add[count_matrix_add_2d].sf->scale_factor) {
        free(aocl_dlp_po_ptr->matrix_add[count_matrix_add_2d].sf->scale_factor);
      }
      if (aocl_dlp_po_ptr->matrix_add[count_matrix_add_2d].sf) {
        free(aocl_dlp_po_ptr->matrix_add[count_matrix_add_2d].sf);
      }
      count_matrix_add_2d++;
    }
    else if (aocl_dlp_po_ptr->seq_vector[idx] == DLP_POST_OP_TYPE::MATRIX_MUL) {
      if (aocl_dlp_po_ptr->matrix_mul[count_matrix_mul_2d].sf &&
          aocl_dlp_po_ptr->matrix_mul[count_matrix_mul_2d].sf->scale_factor) {
        free(aocl_dlp_po_ptr->matrix_mul[count_matrix_mul_2d].sf->scale_factor);
      }
      if (aocl_dlp_po_ptr->matrix_mul[count_matrix_mul_2d].sf) {
        free(aocl_dlp_po_ptr->matrix_mul[count_matrix_mul_2d].sf);
      }
      count_matrix_mul_2d++;
    }
    else if (aocl_dlp_po_ptr->seq_vector[idx] == DLP_POST_OP_TYPE::SCALE) {
      if (aocl_dlp_po_ptr->scale[count_matrix_mul_1d].zp) {
        free(aocl_dlp_po_ptr->scale[count_matrix_mul_1d].zp);
      }
      if (aocl_dlp_po_ptr->scale[count_matrix_mul_1d].sf) {
        free(aocl_dlp_po_ptr->scale[count_matrix_mul_1d].sf);
      }
      count_matrix_mul_1d++;
    }
  }
  if (aocl_dlp_po_ptr->scale) {
    free(aocl_dlp_po_ptr->scale);
  }
  if (aocl_dlp_po_ptr->eltwise) {
    free(aocl_dlp_po_ptr->eltwise);
  }
  if (aocl_dlp_po_ptr->matrix_add) {
    free(aocl_dlp_po_ptr->matrix_add);
  }
  if (aocl_dlp_po_ptr->matrix_mul) {
    free(aocl_dlp_po_ptr->matrix_mul);
  }
  if (aocl_dlp_po_ptr->seq_vector) {
    free(aocl_dlp_po_ptr->seq_vector);
  }
  free(aocl_dlp_po_ptr);
}

dlp_metadata_t *aocl_dlp_utils_t::get_aocl_dlp_post_op_ptr_unsafe() const {
  LOG_DEBUG_INFO("Getting aocl dlp post-op ptr from aocl_dlp_utils_t");
  return aocl_dlp_po_ptr;
}

status_t aocl_dlp_utils_t::reorder_weights(std::optional<tensor_t> weights, data_type_t src_dt) {
  LOG_DEBUG_INFO("Selecting aocl_dlp reorder function based on data type");
  if (!weights) {
    log_error("Weights tensor is not set");
    return status_t::failure;
  }
  if (reordered_weights_ptr != nullptr ||
      (weights->get_layout() & uint16_t(tensor_layout_t::blocked))) {
    return status_t::success;
  }

  bool trans_weights = weights->get_order() == "ba";
  int k = weights->get_size(0);
  int n = weights->get_size(1);
  int ldb = trans_weights ? weights->get_stride(1) :
            weights->get_stride(0);
  auto weights_ptr = weights->get_raw_handle_const();
  data_type_t weight_data_type = weights->get_data_type();
  if (weight_data_type == data_type_t::f32) {
    log_info("Reordering f32 weights");
    reorder_weights_execute<float>(
      weights_ptr, // weights
      k, // k
      n, // n
      ldb, // ldb
      'r', // order
      trans_weights ? 't':'n', // trans
      aocl_get_reorder_buf_size_f32f32f32of32, // size function
      aocl_reorder_f32f32f32of32 // reorder_func
    );
  }
  else if (weight_data_type == data_type_t::bf16) {
    log_info("Reordering BF16 weights");
    reorder_weights_execute<int16_t>(
      weights_ptr, // weights
      k, // k
      n, // n
      ldb, // ldb
      'r', // order
      trans_weights ? 't':'n', // trans
      aocl_get_reorder_buf_size_bf16bf16f32of32, // size function
      aocl_reorder_bf16bf16f32of32 // reorder_func
    );
  }
  else if (weight_data_type == data_type_t::s8) {
    if (src_dt == data_type_t::s8) {
      log_info("Reordering INT8 weights and INT8 input");
      reorder_weights_execute<int8_t>(
        weights_ptr, // weights
        k, // k
        n, // n
        ldb, // ldb
        'r', // order
        trans_weights ? 't':'n', // trans
        aocl_get_reorder_buf_size_s8s8s32os32, // size function
        aocl_reorder_s8s8s32os32 // reorder_func
      );
    }
    else if (src_dt == data_type_t::u8) {
      log_info("Reordering INT8 weights and UINT8 input");
      reorder_weights_execute<int8_t>(
        weights_ptr, // weights
        k, // k
        n, // n
        ldb, // ldb
        'r', // order
        trans_weights ? 't':'n', // trans
        aocl_get_reorder_buf_size_u8s8s32os32, // size function
        aocl_reorder_u8s8s32os32 // reorder_func
      );
    }
  }
  else if (weight_data_type == data_type_t::s4) {
    log_info("Reordering S4 weights");
    reorder_weights_execute<int8_t>(
      weights_ptr, // weights
      k, // k
      n, // n
      ldb, // ldb
      'r', // order
      trans_weights ? 't':'n', // trans
      aocl_get_reorder_buf_size_bf16s4f32of32, // size function
      aocl_reorder_bf16s4f32of32 // reorder_func
    );
  }
  else {
    log_error("Unsupported data type for aocl reorder.");
    return status_t::failure;
  }
  return status_t::success;
}

void *aocl_dlp_utils_t::get_aocl_dlp_reordered_weights_ptr_unsafe() const {
  LOG_DEBUG_INFO("Getting aocl dlp reordered weights ptr from aocl_dlp_utils_t");
  return reordered_weights_ptr;
}

aocl_dlp_utils_t::aocl_dlp_utils_t()
  :zp_comp_acc{nullptr}, aocl_dlp_po_ptr{nullptr}, reordered_weights_ptr{nullptr} {
  post_op_size.insert({"eltwise",0});
  post_op_size.insert({"binary_add_2d",0});
  post_op_size.insert({"binary_mul_2d",0});
  post_op_size.insert({"binary_add_1d",0});
  post_op_size.insert({"binary_mul_1d",0});
  // Src, weight, dst scales
  post_op_size.insert({"scales",0});
  // Default zero-point compensation ndim
  zp_comp_ndim = 0;
  dummy_zp = (int32_t)0;
  dummy_scale = 1.0f;
}

aocl_dlp_utils_t::~aocl_dlp_utils_t() {
  LOG_DEBUG_INFO("Destroying aocl_dlp_utils_t");
  if (reordered_weights_ptr) {
    free(reordered_weights_ptr);
    reordered_weights_ptr = nullptr;
  }
  if (aocl_dlp_po_ptr) {
    free_post_op();
    aocl_dlp_po_ptr = nullptr;
  }
  if (zp_comp_acc) {
    free(zp_comp_acc);
    zp_comp_acc = nullptr;
  }
}


} // namespace ops
} // namespace zendnnl
