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

#include "lowoha_operators/matmul/onednn_kernel.hpp"

namespace zendnnl {
namespace lowoha {

#if ZENDNNL_DEPENDS_ONEDNN
void matmul_onednn_wrapper(char transA, char transB, int M, int N,
                           int K, float alpha, const void *A, int lda, const void *B, int ldb, float beta,
                           void *C, int ldc, lowoha_params &lowoha_params, int batchA, int batchB,
                           const void *bias, zendnnl::ops::matmul_algo_t kernel, bool is_weights_const) {
  matmul_config_t &matmul_config = matmul_config_t::instance();
  int32_t weight_cache_type = matmul_config.get_weight_cache();
  onednn_utils_t::onednn_matmul_params dnnl_params;

  dnnl_params.src.buffer = const_cast<void *>(A);
  dnnl_params.weights.buffer = const_cast<void *>(B);
  dnnl_params.dst.buffer = C;

  dnnl_params.src.dtype = lowoha_params.dtypes.src;
  dnnl_params.weights.dtype = lowoha_params.dtypes.wei;
  dnnl_params.dst.dtype = lowoha_params.dtypes.dst;

  if (bias != nullptr)  {
    dnnl_params.bias.buffer = const_cast<void *>(bias);
    dnnl_params.bias.dtype = lowoha_params.dtypes.bias;
  }

  int batch_count = std::max(batchA, batchB);
  if (batch_count == 1) {
    dnnl_params.src.dims = {M, K};
    dnnl_params.weights.dims = {K, N};
    dnnl_params.dst.dims = {M, N};
    if (bias != nullptr) dnnl_params.bias.dims = {1, N};
  }
  else {
    dnnl_params.src.dims = {batchA, M, K};
    dnnl_params.weights.dims = {batchB, K, N};
    dnnl_params.dst.dims = {batch_count, M, N};
    if (bias != nullptr) dnnl_params.bias.dims = {1, 1, N};
  }

  dnnl_params.src.is_transposed = (transA == 'n') ? false : true;
  dnnl_params.weights.is_transposed = (transB == 'n') ? false : true;
  dnnl_params.algo = kernel;

  if (batch_count == 1) {
    dnnl_params.src.format_tag = (transA == 'n') ? "ab" : "ba";
    dnnl_params.weights.format_tag = (transB == 'n') ? "ab" : "ba";
    dnnl_params.dst.format_tag = "ab";
    if (bias != nullptr) {
      dnnl_params.bias.format_tag = "ab";
    }
  }
  else {
    dnnl_params.src.format_tag = (transA == 'n') ? "abc" : "acb";
    dnnl_params.weights.format_tag = (transB == 'n') ? "abc" : "acb";
    dnnl_params.dst.format_tag = "abc";
    if (bias != nullptr) {
      dnnl_params.bias.format_tag = "abc";
    }
  }

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  std::unordered_map<int, dnnl::memory> matmul_args;
  dnnl::primitive_attr matmul_attr;
  dnnl::post_ops matmul_pops;
  int post_op_index = 0;

  if (alpha != 1.0f) {
    matmul_attr.set_scales_mask(DNNL_ARG_SRC, 0);
    auto alpha_mem = dnnl::memory({{1}, dnnl::memory::data_type::f32, {1}}, eng,
    const_cast<float *>(&alpha));
    matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, alpha_mem});
  }

  if (beta != 0.0f) {
    matmul_pops.append_sum(beta);
    post_op_index++;
  }

  if (lowoha_params.postop_.size() > 0) {
    for (size_t po = 0; po < lowoha_params.postop_.size(); po++) {
      // float po_alpha = 0.0f, po_beta = 0.0f;
      switch (lowoha_params.postop_[po].po_type) {
      case post_op_type_t::elu: {
        log_info("Adding ELU post-op");
        lowoha_params.postop_[po].alpha = lowoha_params.postop_[po].alpha ?
                                          lowoha_params.postop_[po].alpha : 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_elu,
                                   lowoha_params.postop_[po].alpha, lowoha_params.postop_[po].beta);
        break;
      }
      case post_op_type_t::relu: {
        log_info("Adding ReLU post-op");
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_relu,
                                   lowoha_params.postop_[po].alpha, lowoha_params.postop_[po].beta);
        break;
      }
      case post_op_type_t::leaky_relu: {
        log_info("Adding Leaky ReLU post-op");
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_relu,
                                   lowoha_params.postop_[po].alpha, lowoha_params.postop_[po].beta);
        break;
      }
      case post_op_type_t::gelu_tanh: {
        log_info("Adding GELU-Tanh post-op");
        lowoha_params.postop_[po].alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh,
                                   lowoha_params.postop_[po].alpha, lowoha_params.postop_[po].beta);
        break;
      }
      case post_op_type_t::gelu_erf: {
        log_info("Adding GELU-Erf post-op");
        lowoha_params.postop_[po].alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_gelu_erf,
                                   lowoha_params.postop_[po].alpha, lowoha_params.postop_[po].beta);
        break;
      }
      case post_op_type_t::tanh: {
        log_info("Adding Tanh post-op");
        lowoha_params.postop_[po].alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_tanh,
                                   lowoha_params.postop_[po].alpha, lowoha_params.postop_[po].beta);
        break;
      }
      case post_op_type_t::square: {
        log_info("Adding Square post-op");
        lowoha_params.postop_[po].alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_square,
                                   lowoha_params.postop_[po].alpha, lowoha_params.postop_[po].beta);
        break;
      }
      case post_op_type_t::abs: {
        log_info("Adding Abs post-op");
        lowoha_params.postop_[po].alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_abs,
                                   lowoha_params.postop_[po].alpha, lowoha_params.postop_[po].beta);
        break;
      }
      case post_op_type_t::sqrt: {
        log_info("Adding Sqrt post-op");
        lowoha_params.postop_[po].alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_sqrt,
                                   lowoha_params.postop_[po].alpha, lowoha_params.postop_[po].beta);
        break;
      }
      case post_op_type_t::exp: {
        log_info("Adding Exp post-op");
        lowoha_params.postop_[po].alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_exp,
                                   lowoha_params.postop_[po].alpha, lowoha_params.postop_[po].beta);
        break;
      }
      case post_op_type_t::log: {
        log_info("Adding Log post-op");
        lowoha_params.postop_[po].alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_log,
                                   lowoha_params.postop_[po].alpha, lowoha_params.postop_[po].beta);
        break;
      }
      case post_op_type_t::sigmoid: {
        log_info("Adding Sigmoid post-op");
        lowoha_params.postop_[po].alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_logistic,
                                   lowoha_params.postop_[po].alpha, lowoha_params.postop_[po].beta);
        break;
      }
      case post_op_type_t::swish: {
        log_info("Adding Swish post-op");
        lowoha_params.postop_[po].alpha = lowoha_params.postop_[po].alpha ?
                                          lowoha_params.postop_[po].alpha : 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_swish,
                                   lowoha_params.postop_[po].alpha, lowoha_params.postop_[po].beta);
        break;
      }
      case post_op_type_t::clip: {
        log_info("Adding Clip post-op");
        lowoha_params.postop_[po].alpha = lowoha_params.postop_[po].alpha ?
                                          lowoha_params.postop_[po].alpha : -0.5f;
        lowoha_params.postop_[po].beta = lowoha_params.postop_[po].beta ?
                                         lowoha_params.postop_[po].beta : 0.5f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_clip,
                                   lowoha_params.postop_[po].alpha, lowoha_params.postop_[po].beta);
        break;
      }
      case post_op_type_t::binary_add: {
        log_info("Adding Binary Add post-op");
        std::vector<int64_t> binary_dims;
        if (lowoha_params.postop_[po].dims.size() == 2 && batch_count > 1) {
          binary_dims = {1, lowoha_params.postop_[po].dims[0], lowoha_params.postop_[po].dims[1]};
        }
        else {
          binary_dims = lowoha_params.postop_[po].dims;
        }
        onednn_utils_t::onednn_tensor_params binary_tensor;
        binary_tensor.dims = binary_dims;
        binary_tensor.buffer = lowoha_params.postop_[po].buff;
        binary_tensor.dtype = lowoha_params.postop_[po].dtype;
        binary_tensor.format_tag = binary_dims.size() == 3 ? "abc" :
                                   "ab";

        auto dnnl_buff_desc    = onednn_utils_t::to_dnnl_tensor(binary_tensor, eng);
        auto dnnl_buff_mem     = dnnl::memory(dnnl_buff_desc, eng,
                                              binary_tensor.buffer);
        matmul_pops.append_binary(dnnl::algorithm::binary_add, dnnl_buff_desc);
        matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index) | DNNL_ARG_SRC_1, dnnl_buff_mem});
        break;
      }
      case post_op_type_t::binary_mul: {
        log_info("Adding Binary Mul post-op");
        std::vector<int64_t> binary_dims;
        if (lowoha_params.postop_[po].dims.size() == 2 && batch_count > 1) {
          binary_dims = {1, lowoha_params.postop_[po].dims[0], lowoha_params.postop_[po].dims[1]};
        }
        else {
          binary_dims = lowoha_params.postop_[po].dims;
        }
        onednn_utils_t::onednn_tensor_params binary_tensor;
        binary_tensor.dims = binary_dims;
        binary_tensor.buffer = lowoha_params.postop_[po].buff;
        binary_tensor.dtype = lowoha_params.postop_[po].dtype;
        binary_tensor.format_tag = binary_dims.size() == 3 ? "abc" :
                                   "ab";

        auto dnnl_buff_desc    = onednn_utils_t::to_dnnl_tensor(binary_tensor, eng);
        auto dnnl_buff_mem     = dnnl::memory(dnnl_buff_desc, eng,
                                              binary_tensor.buffer);
        matmul_pops.append_binary(dnnl::algorithm::binary_mul, dnnl_buff_desc);
        matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index) | DNNL_ARG_SRC_1, dnnl_buff_mem});
        break;
      }
      default:
        break;
      }
      post_op_index++;
    }
  }

  if (matmul_pops.len() > 0) {
    matmul_attr.set_post_ops(matmul_pops);
  }

  bool is_blocked = dnnl_params.weights.dims.size() == 2 &&
                    dnnl_params.algo == matmul_algo_t::onednn_blocked &&
                    is_weights_const;
  if (is_blocked) {
    Key_matmul key_(transB, K, N, ldb, dnnl_params.weights.buffer,
                    static_cast<uint32_t>(matmul_algo_t::onednn_blocked));
    dnnl_params.is_blocked = reorderAndCacheWeights(key_, dnnl_params,
                             weight_cache_type, eng);
  }

  matmul_onednn_kernel_t::execute_matmul(dnnl_params, matmul_args, matmul_attr,
                                         eng);
  if (weight_cache_type == 0 && is_blocked) {
    free(dnnl_params.weights.buffer);
  }
}

void reorderWeights(onednn_utils_t::onednn_matmul_params &dnnl_params,
                    dnnl::engine &eng) {
  dnnl::stream eng_stream(eng);
  void *reordered_mem = nullptr;
  dnnl::memory::desc  dnnl_weight_desc   = onednn_utils_t::to_dnnl_tensor(
        dnnl_params.weights, eng);
  dnnl::memory        dnnl_weight_mem    = dnnl::memory(dnnl_weight_desc, eng,
      dnnl_params.weights.buffer);

  dnnl_params.weights.format_tag = (dnnl_params.weights.dtype == data_type_t::f32)
                                   ? "BA16a64b" : "BA16a64b2a";
  dnnl::memory::desc  dnnl_blocked_weight_desc   = onednn_utils_t::to_dnnl_tensor(
        dnnl_params.weights, eng);
  size_t reordered_size = dnnl_blocked_weight_desc.get_size();
  size_t alignment      = 64;
  size_t reorder_size   = (reordered_size + alignment - 1) & ~(alignment - 1);
  reordered_mem         = (void *)aligned_alloc(alignment, reorder_size);
  dnnl::memory        dnnl_blocked_weight_mem    = dnnl::memory(
        dnnl_blocked_weight_desc, eng, reordered_mem);

  reorder(dnnl_weight_mem, dnnl_blocked_weight_mem).execute(eng_stream,
      dnnl_weight_mem, dnnl_blocked_weight_mem);
  dnnl_params.weights.buffer = dnnl_blocked_weight_mem.get_data_handle();
}

bool reorderAndCacheWeights(Key_matmul key,
                            onednn_utils_t::onednn_matmul_params &dnnl_params, int weight_cache_type,
                            dnnl::engine &eng) {
  // Weight caching
  static lru_cache_t<Key_matmul, std::pair<void *, std::string>>
      matmul_weight_cache;

  if (weight_cache_type == 0) {
    apilog_info("onednn reorder weights (WEIGHT_CACHE_DISABLE)");
    reorderWeights(dnnl_params, eng);
  }
  else {
    auto found_obj = matmul_weight_cache.find_key(key);
    if (!found_obj) {
      apilog_info("onednn reorder weights WEIGHT_CACHE_OUT_OF_PLACE");
      reorderWeights(dnnl_params, eng);
      matmul_weight_cache.add(key, {dnnl_params.weights.buffer, dnnl_params.weights.format_tag});
    }
    else {
      apilog_info("Read onednn cached weights WEIGHT_CACHE_OUT_OF_PLACE");
      dnnl_params.weights.buffer = matmul_weight_cache.get(key).first;
      dnnl_params.weights.format_tag = matmul_weight_cache.get(key).second;
    }
  }
  return true;

}

#endif

} // lowoha namespace
} // zendnnl namespace