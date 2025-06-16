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
#include "operators/matmul/aocl_blis/matmul_aocl_blis_utils.hpp"
#include "common/data_types.hpp" // Ensure this header defines data_type_t

namespace zendnnl {
namespace ops {

inline void eltwise_init(aocl_post_op *&aocl_blis_po_ptr, int eltwise_count,
                         AOCL_ELT_ALGO_TYPE algo_type) {
  (aocl_blis_po_ptr->eltwise[eltwise_count]).is_power_of_2 = false;
  (aocl_blis_po_ptr->eltwise[eltwise_count]).scale_factor = nullptr;
  (aocl_blis_po_ptr->eltwise[eltwise_count]).scale_factor_len = 0;
  (aocl_blis_po_ptr->eltwise[eltwise_count]).algo.alpha = nullptr;
  (aocl_blis_po_ptr->eltwise[eltwise_count]).algo.beta = nullptr;
  (aocl_blis_po_ptr->eltwise[eltwise_count]).algo.algo_type = algo_type;
}

//Returns AOCL data type
AOCL_PARAMS_STORAGE_TYPES get_aocl_store_type(data_type_t dt) {
  switch (dt) {
  case data_type_t::f32:
    return AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_F32 ;
    break;
  case data_type_t::bf16:
    return AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_BF16 ;
    break;
  case data_type_t::s32:
    return AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_INT32 ;
    break;
  case data_type_t::s8:
    return AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_INT8 ;
    break;
  case data_type_t::u8:
    return AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_UINT8 ;
    break;
  case data_type_t::s4:
    return AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_INT4 ;
    break;
  default:
    break;
  };
  return AOCL_PARAMS_STORAGE_TYPES::NULLTYPE;
}

template <typename T>
size_t aocl_blis_utils_t::reorder_weights_execute(
  const void *weights,
  const int k,
  const int n,
  const int ldb,
  const char order,
  const char trans,
  get_reorder_buff_size_func_ptr get_reorder_buf_size,
  reorder_func_ptr<T> reorder_func) {
  LOG_DEBUG_INFO("Reodering weights aocl_blis_utils_t");
  log_info("BLIS reorder weights");
  siz_t b_reorder_buf_siz_req = get_reorder_buf_size(order, trans, 'B',
                                k, n);
  /*TODO: add support for tensor which will wrap the pointer instead of raw buffer*/
  reordered_weights_ptr = aligned_alloc(64, b_reorder_buf_siz_req);
  reorder_func(order, trans, 'B', (T *)weights, (T *)reordered_weights_ptr, k, n,
               ldb);

  return b_reorder_buf_siz_req;
}

status_t aocl_blis_utils_t::set_runtime_post_op_buffer(tensor_map_type
    &inputs_, bool is_bias) {
  uint32_t max_matrix_mul_po = post_op_size["binary_mul_2d"] +
                               post_op_size["binary_mul_1d"];
  uint32_t max_matrix_add_po = post_op_size["binary_add_2d"] +
                               post_op_size["binary_add_1d"];

  if (inputs_.size() > max_matrix_mul_po + max_matrix_add_po) {
    // Set Matrix Mul buffer
    size_t mul_idx_1d = 0;
    size_t mul_idx_2d = 0;
    size_t add_idx_1d = is_bias ? 1 : 0; // If bias is present, add one for bias
    size_t add_idx_2d = 0;
    for (size_t mul_idx=0; mul_idx < max_matrix_mul_po ; mul_idx++) {
      // name of tensor should be binary_mul_<num>
      std::string key_mul = "binary_mul_tensor_" + std::to_string(mul_idx);
      auto found_obj_mul = inputs_.find(key_mul);
      if (found_obj_mul != inputs_.end()) {
        auto mul_buff_tensor = inputs_[key_mul];
        if (found_obj_mul->second.get_size().size() == 1 &&
            aocl_blis_po_ptr->bias != nullptr) {
          (aocl_blis_po_ptr->sum + mul_idx_1d)->scale_factor  =
            mul_buff_tensor.get_raw_handle_unsafe();
          (aocl_blis_po_ptr->sum + mul_idx_1d)->zero_point    = malloc(sizeof(int32_t));
          int32_t *temp_dzero_point_ptr = (int32_t *)(aocl_blis_po_ptr->sum +
                                          mul_idx_1d)->zero_point;
          temp_dzero_point_ptr[0] = (int32_t)0;
          mul_idx_1d++;
        }
        else if (found_obj_mul->second.get_size().size() == 2 &&
                 aocl_blis_po_ptr->matrix_mul != nullptr) {
          (aocl_blis_po_ptr->matrix_mul + mul_idx_2d)->matrix =
            mul_buff_tensor.get_raw_handle_unsafe();
          (aocl_blis_po_ptr->matrix_mul + mul_idx_2d)->stor_type = get_aocl_store_type(
                mul_buff_tensor.get_data_type());
          (aocl_blis_po_ptr->matrix_mul + mul_idx_2d)->ldm = mul_buff_tensor.get_size(1);
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
            aocl_blis_po_ptr->bias != nullptr) {
          (aocl_blis_po_ptr->bias + add_idx_1d)->bias = (void *)
              add_buff_tensor.get_raw_handle_unsafe();
          (aocl_blis_po_ptr->bias + add_idx_1d)->stor_type = get_aocl_store_type(
                add_buff_tensor.get_data_type());
          add_idx_1d++;
        }
        else if (found_obj_add->second.get_size().size() == 2 &&
                 aocl_blis_po_ptr->matrix_add != nullptr) {
          (aocl_blis_po_ptr->matrix_add + add_idx_2d)->matrix
            = add_buff_tensor.get_raw_handle_unsafe();
          (aocl_blis_po_ptr->matrix_add + add_idx_2d)->stor_type
            = get_aocl_store_type(add_buff_tensor.get_data_type());
          (aocl_blis_po_ptr->matrix_add + add_idx_2d)->ldm
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
  return status_t::success;
}

status_t aocl_blis_utils_t::aocl_post_op_memory_alloc(const
    std::vector<post_op_t>
    post_op_vec_, bool is_bias,
    std::map<std::string, zendnnl::memory::tensor_t> inputs_) {
  LOG_DEBUG_INFO("Allocating memory for post_ops in aocl_blis_utils_t");
  //Allocate memory
  size_t max_post_ops = post_op_vec_.size();

  if (max_post_ops || is_bias) {
    int num_post_ops_1d_add     = is_bias ? 1 : 0;
    int num_post_ops_binary_add = 0;
    int num_post_ops_1d_mul     = 0;
    int num_post_ops_binary_mul = 0;
    int num_post_ops_eltwise    = 0;
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
        if (inputs_.find(zen_po.binary_add_params.tensor_name)->second.get_size().size()
            == 1) {
          num_post_ops_1d_add++;
        }
        else {
          num_post_ops_binary_add++;
        }
        break;
      case post_op_type_t::binary_mul:
        if (inputs_.find(zen_po.binary_mul_params.tensor_name)->second.get_size().size()
            == 1) {
          num_post_ops_1d_mul++;
        }
        else {
          num_post_ops_binary_mul++;
        }
        break;
      default:
        log_error("This postop in aocl is not supported");
        return status_t::failure;
      }
    }
    aocl_blis_po_ptr->bias        = (aocl_post_op_bias *) calloc(
                                      num_post_ops_1d_add,
                                      sizeof(aocl_post_op_bias));
    aocl_blis_po_ptr->sum         = (aocl_post_op_sum *) calloc(num_post_ops_1d_mul,
                                    sizeof(aocl_post_op_sum));
    aocl_blis_po_ptr->eltwise     = (aocl_post_op_eltwise *) calloc(
                                      num_post_ops_eltwise,
                                      sizeof(aocl_post_op_eltwise));
    aocl_blis_po_ptr->matrix_add  = (aocl_post_op_matrix_add *) calloc(
                                      num_post_ops_binary_add,
                                      sizeof(aocl_post_op_matrix_add));
    aocl_blis_po_ptr->matrix_mul  = (aocl_post_op_matrix_mul *) calloc(
                                      num_post_ops_binary_mul,
                                      sizeof(aocl_post_op_matrix_mul));
    post_op_size["eltwise"]       = num_post_ops_eltwise;
    post_op_size["binary_add_2d"] = num_post_ops_binary_add;
    post_op_size["binary_mul_2d"] = num_post_ops_binary_mul;
    post_op_size["binary_add_1d"] = num_post_ops_1d_add - (is_bias ? 1 :
                                    0); /*Don't count bias*/
    post_op_size["binary_mul_1d"] = num_post_ops_1d_mul;
  }
  return status_t::success;
}

status_t aocl_blis_utils_t::aocl_post_op_initialize(const std::vector<post_op_t>
    post_op_vec_, int &post_op_count, bool is_bias,
    std::map<std::string, zendnnl::memory::tensor_t> inputs_) {
  LOG_DEBUG_INFO("Initializing aocl post-op in aocl_blis_utils_t");
  //add remaining post-ops
  size_t max_post_ops = post_op_vec_.size();
  //Index for each post-op
  dim_t eltwise_index = 0;
  dim_t add_index_2d = 0;
  dim_t mul_index_2d = 0;
  dim_t mul_index_1d = 0;

  for (size_t i = 0; i < max_post_ops; ++ i) {
    post_op_t zen_po = post_op_vec_[i];

    switch (zen_po.type) {
    case post_op_type_t::relu:
      log_info("Adding relu post-op");
      eltwise_init(aocl_blis_po_ptr, eltwise_index, RELU);
      eltwise_index++;
      aocl_blis_po_ptr->seq_vector[post_op_count++] = ELTWISE;
      break;
    case post_op_type_t::leaky_relu:
      log_info("Adding leaky_relu post-op");
      eltwise_init(aocl_blis_po_ptr, eltwise_index, PRELU);
      (aocl_blis_po_ptr->eltwise[eltwise_index]).algo.alpha = malloc(sizeof(float));
      *((float *)(aocl_blis_po_ptr->eltwise[eltwise_index]).algo.alpha)
        = zen_po.leaky_relu_params.nslope;
      eltwise_index++;
      aocl_blis_po_ptr->seq_vector[post_op_count++] = ELTWISE;
      break;
    case post_op_type_t::gelu_tanh:
      log_info("Adding gelu_tanh post-op");
      eltwise_init(aocl_blis_po_ptr, eltwise_index, GELU_TANH);
      eltwise_index++;
      aocl_blis_po_ptr->seq_vector[post_op_count++] = ELTWISE;
      break;
    case post_op_type_t::gelu_erf:
      log_info("Adding gelu_erf post-op");
      eltwise_init(aocl_blis_po_ptr, eltwise_index, GELU_ERF);
      eltwise_index++;
      aocl_blis_po_ptr->seq_vector[post_op_count++] = ELTWISE;
      break;
    case post_op_type_t::tanh:
      log_info("Adding tanh post-op");
      eltwise_init(aocl_blis_po_ptr, eltwise_index, TANH);
      eltwise_index++;
      aocl_blis_po_ptr->seq_vector[post_op_count++] = ELTWISE;
      break;
    case post_op_type_t::swish:
      log_info("Adding swish post-op");
      eltwise_init(aocl_blis_po_ptr, eltwise_index, SWISH);
      (aocl_blis_po_ptr->eltwise[eltwise_index]).algo.alpha = malloc(sizeof(float));
      *((float *)(aocl_blis_po_ptr->eltwise[eltwise_index]).algo.alpha) =
        zen_po.swish_params.scale;
      eltwise_index++;
      aocl_blis_po_ptr->seq_vector[post_op_count++] = ELTWISE;
      break;
    case post_op_type_t::sigmoid:
      log_info("Adding sigmoid post-op");
      eltwise_init(aocl_blis_po_ptr, eltwise_index, SIGMOID);
      eltwise_index++;
      aocl_blis_po_ptr->seq_vector[post_op_count++] = ELTWISE;
      break;
    case post_op_type_t::clip:
      log_info("Adding clip post-op");
      eltwise_init(aocl_blis_po_ptr, eltwise_index, CLIP);
      (aocl_blis_po_ptr->eltwise[eltwise_index]).algo.alpha = malloc(sizeof(float));
      *((float *)(aocl_blis_po_ptr->eltwise[eltwise_index]).algo.alpha) =
        zen_po.clip_params.lower;
      (aocl_blis_po_ptr->eltwise[eltwise_index]).algo.beta = malloc(sizeof(float));
      *((float *)(aocl_blis_po_ptr->eltwise[eltwise_index]).algo.beta) =
        zen_po.clip_params.upper;
      eltwise_index++;
      aocl_blis_po_ptr->seq_vector[post_op_count++] = ELTWISE;
      break;
    case post_op_type_t::binary_add:
      log_info("Adding binary_add post-op");
      if (inputs_.find(zen_po.binary_add_params.tensor_name)->second.get_size().size()
          == 1) {
        aocl_blis_po_ptr->seq_vector[post_op_count++] = BIAS;
      }
      else {
        (aocl_blis_po_ptr->matrix_add + add_index_2d)->scale_factor = malloc(sizeof(
              float));
        *((float *)(aocl_blis_po_ptr->matrix_add[add_index_2d]).scale_factor) =
          zen_po.binary_add_params.scale;
        (aocl_blis_po_ptr->matrix_add + add_index_2d)->scale_factor_len = 1;
        aocl_blis_po_ptr->seq_vector[post_op_count++] = MATRIX_ADD;
        add_index_2d++;
      }
      break;
    case post_op_type_t::binary_mul:
      log_info("Adding binary_mul post-op");
      if (inputs_.find(zen_po.binary_mul_params.tensor_name)->second.get_size().size()
          == 1) {
        aocl_blis_po_ptr->seq_vector[post_op_count++] = SCALE;
        (aocl_blis_po_ptr->sum + mul_index_1d)->is_power_of_2 = FALSE;
        (aocl_blis_po_ptr->sum + mul_index_1d)->scale_factor = NULL;
        (aocl_blis_po_ptr->sum + mul_index_1d)->buff = NULL;
        (aocl_blis_po_ptr->sum + mul_index_1d)->zero_point = NULL;
        (aocl_blis_po_ptr->sum + mul_index_1d)->scale_factor_len = inputs_.find(
              zen_po.binary_mul_params.tensor_name)->second.get_size()[0];
        (aocl_blis_po_ptr->sum + mul_index_1d)->zero_point_len = 1;
        mul_index_1d++;
      }
      else {
        (aocl_blis_po_ptr->matrix_mul + mul_index_2d)->scale_factor = malloc(sizeof(
              float));
        *((float *)(aocl_blis_po_ptr->matrix_mul[mul_index_2d]).scale_factor) =
          zen_po.binary_mul_params.scale;
        (aocl_blis_po_ptr->matrix_mul + mul_index_2d)->scale_factor_len = 1;
        aocl_blis_po_ptr->seq_vector[post_op_count++] = MATRIX_MUL;
        mul_index_2d++;
      }
      break;
    default:
      log_error("This postop in aocl is not supported");
      return status_t::failure;
    }
  }
  return status_t::success;
}

status_t aocl_blis_utils_t::alloc_post_op(const std::vector<post_op_t>
    post_op_vec_,
    std::optional<tensor_t> optional_bias_tensor_,
    std::map<std::string, zendnnl::memory::tensor_t> inputs_) {
  LOG_DEBUG_INFO("Allocating post-ops in aocl_blis_utils_t");

  // Iterate through each postop, check and add it if needed.
  int post_op_count = 0;
  // Find total number of post-ops with bias and scales
  int total_po = post_op_vec_.size();
  if (optional_bias_tensor_) {
    total_po++;
  }
  if (total_po > 0) {
    aocl_blis_po_ptr = (aocl_post_op *) calloc(1, sizeof(aocl_post_op));
    if (aocl_blis_po_ptr == NULL) {
      return status_t::failure;
    }
    aocl_blis_po_ptr->seq_vector = (AOCL_POST_OP_TYPE *) calloc(total_po,
                                   sizeof(AOCL_POST_OP_TYPE));
    if (aocl_blis_po_ptr->seq_vector == NULL) {
      free(aocl_blis_po_ptr);
      return status_t::failure;
    }

    //Set all post-ops to NULL
    aocl_blis_po_ptr->eltwise = NULL;
    aocl_blis_po_ptr->bias = NULL;
    aocl_blis_po_ptr->sum = NULL;
    aocl_blis_po_ptr->matrix_add = NULL;
    aocl_blis_po_ptr->matrix_mul = NULL;
    aocl_blis_po_ptr->pre_ops = NULL;

    // Allocate memory for post-ops
    if (aocl_post_op_memory_alloc(post_op_vec_, optional_bias_tensor_? true : false,
                                  inputs_)
        != status_t::success) {
      return status_t::failure;
    }

    // Add bias postop
    if (optional_bias_tensor_) {
      auto bias_type = optional_bias_tensor_->get_data_type();
      aocl_blis_po_ptr->seq_vector[post_op_count++] = BIAS;
      if (aocl_blis_po_ptr->bias == NULL) {
        free(aocl_blis_po_ptr->seq_vector);
        free(aocl_blis_po_ptr);
        return status_t::failure;
      }
      (aocl_blis_po_ptr->bias)->bias = (void *)
                                       optional_bias_tensor_->get_raw_handle_unsafe();
      (aocl_blis_po_ptr->bias)->stor_type = get_aocl_store_type(bias_type);
    }

    //Initialize other post-ops.
    if (aocl_post_op_initialize(post_op_vec_, post_op_count,
                                optional_bias_tensor_? true : false, inputs_) != status_t::success) {
      return status_t::failure;
    }
    aocl_blis_po_ptr->seq_length = post_op_count;
  }

  return status_t::success;
}

void aocl_blis_utils_t::free_post_op() {
  LOG_DEBUG_INFO("Freeing aocl post-ops from aocl_blis_utils_t");
  if (aocl_blis_po_ptr == nullptr) {
    return;
  }

  if (aocl_blis_po_ptr->sum) {
    free(aocl_blis_po_ptr->sum);
  }
  if (aocl_blis_po_ptr->bias) {
    free(aocl_blis_po_ptr->bias);
  }

  int count_elt           = 0;
  int count_matrix_add_2d = 0;
  int count_matrix_mul_2d = 0;
  int count_matrix_mul_1d = 0;
  for (int idx = 0; idx < aocl_blis_po_ptr->seq_length; idx++) {
    if (aocl_blis_po_ptr->seq_vector[idx] == ELTWISE) {
      if (aocl_blis_po_ptr->eltwise[count_elt].algo.alpha) {
        free(aocl_blis_po_ptr->eltwise[count_elt].algo.alpha);
      }
      if (aocl_blis_po_ptr->eltwise[count_elt].algo.beta) {
        free(aocl_blis_po_ptr->eltwise[count_elt].algo.beta);
      }
      count_elt++;
    }
    else if (aocl_blis_po_ptr->seq_vector[idx] == MATRIX_ADD) {
      if (aocl_blis_po_ptr->matrix_add[count_matrix_add_2d].scale_factor) {
        free(aocl_blis_po_ptr->matrix_add[count_matrix_add_2d].scale_factor);
      }
      count_matrix_add_2d++;
    }
    else if (aocl_blis_po_ptr->seq_vector[idx] == MATRIX_MUL) {
      if (aocl_blis_po_ptr->matrix_mul[count_matrix_mul_2d].scale_factor) {
        free(aocl_blis_po_ptr->matrix_mul[count_matrix_mul_2d].scale_factor);
      }
      count_matrix_mul_2d++;
    }
    else if (aocl_blis_po_ptr->seq_vector[idx] == SCALE) {
      if (aocl_blis_po_ptr->sum[count_matrix_mul_1d].zero_point) {
        free(aocl_blis_po_ptr->sum[count_matrix_mul_1d].zero_point);
      }
      count_matrix_mul_1d++;
    }
  }
  if (aocl_blis_po_ptr->eltwise) {
    free(aocl_blis_po_ptr->eltwise);
  }
  if (aocl_blis_po_ptr->matrix_add) {
    free(aocl_blis_po_ptr->matrix_add);
  }
  if (aocl_blis_po_ptr->matrix_mul) {
    free(aocl_blis_po_ptr->matrix_mul);
  }
  if (aocl_blis_po_ptr->seq_vector) {
    free(aocl_blis_po_ptr->seq_vector);
  }
  if (aocl_blis_po_ptr->pre_ops) {
    free(aocl_blis_po_ptr->pre_ops);
  }
  free(aocl_blis_po_ptr);
}

aocl_post_op *aocl_blis_utils_t::get_aocl_blis_post_op_ptr_unsafe() const {
  LOG_DEBUG_INFO("Getting aocl blis post-op ptr from aocl_blis_utils_t");
  return aocl_blis_po_ptr;
}

status_t aocl_blis_utils_t::reorder_weights(std::optional<tensor_t> weights) {
  LOG_DEBUG_INFO("Selecting aocl_blis reorder function based on data type");
  if (!weights) {
    log_error("Weights tensor is not set");
    return status_t::failure;
  }
  if (reordered_weights_ptr != nullptr ||
      weights->get_layout() == tensor_layout_t::blocked) {
    return status_t::success;
  }

  bool trans_weights = weights->get_order() == "ba";
  int k = weights->get_size(0);
  int n = weights->get_size(1);
  int ldb = trans_weights ? weights->get_stride_size(0) :
            weights->get_stride_size(1);
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
  else {
    log_error("Unsupported data type for aocl reorder.");
    return status_t::failure;
  }
  return status_t::success;
}

void *aocl_blis_utils_t::get_aocl_blis_reordered_weights_ptr_unsafe() const {
  LOG_DEBUG_INFO("Getting aocl blis reordered weights ptr from aocl_blis_utils_t");
  return reordered_weights_ptr;
}

aocl_blis_utils_t::aocl_blis_utils_t()
  :aocl_blis_po_ptr{nullptr}, reordered_weights_ptr{nullptr} {
  post_op_size.insert({"eltwise",0});
  post_op_size.insert({"binary_add_2d",0});
  post_op_size.insert({"binary_mul_2d",0});
  post_op_size.insert({"binary_add_1d",0});
  post_op_size.insert({"binary_mul_1d",0});
}

aocl_blis_utils_t::~aocl_blis_utils_t() {
  LOG_DEBUG_INFO("Destroying aocl_blis_utils_t");
  if (reordered_weights_ptr) {
    free(reordered_weights_ptr);
    reordered_weights_ptr = nullptr;
  }
  if (aocl_blis_po_ptr) {
    free_post_op();
    aocl_blis_po_ptr = nullptr;
  }
}


} // namespace ops
} // namespace zendnnl
