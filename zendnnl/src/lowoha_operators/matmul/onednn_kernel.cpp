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

#include "lowoha_operators/matmul/onednn_kernel.hpp"
#include <mutex>

namespace zendnnl {
namespace lowoha {
namespace matmul {

#if ZENDNNL_DEPENDS_ONEDNN

/**
 * @brief Creates matmul primitive descriptor with blocked weight format
 *
 * Creates memory descriptors for input, weights, output, and bias tensors,
 * then builds a matmul primitive descriptor using "any" format for weights
 * to allow oneDNN to choose optimal blocking.
 *
 * @param dnnl_params OneDNN parameters containing tensor info
 * @param eng OneDNN engine
 * @param matmul_attr Primitive attributes (post-ops, scales, etc.)
 * @return matmul primitive descriptor with optimal weight blocking
 */
dnnl::matmul::primitive_desc create_blocked_matmul_pd(
  onednn_utils_t::onednn_matmul_params &dnnl_params,
  const dnnl::engine &eng,
  const dnnl::primitive_attr &matmul_attr) {

  dnnl::memory::desc dnnl_input_desc = onednn_utils_t::to_dnnl_tensor(
                                         dnnl_params.src, eng);

  dnnl_params.weights.format_tag = "any";
  dnnl::memory::desc dnnl_blocked_weight_desc = onednn_utils_t::to_dnnl_tensor(
        dnnl_params.weights, eng);

  dnnl::memory::desc dnnl_output_desc = onednn_utils_t::to_dnnl_tensor(
                                          dnnl_params.dst, eng);

  dnnl::memory::desc dnnl_bias_desc = onednn_utils_t::to_dnnl_tensor(
                                        dnnl_params.bias, eng);

  if (dnnl_params.bias.buffer != nullptr) {
    return dnnl::matmul::primitive_desc(eng, dnnl_input_desc,
                                        dnnl_blocked_weight_desc, dnnl_bias_desc,
                                        dnnl_output_desc, matmul_attr);
  }
  else {
    return dnnl::matmul::primitive_desc(eng, dnnl_input_desc,
                                        dnnl_blocked_weight_desc,
                                        dnnl_output_desc, matmul_attr);
  }
}

/**
 * @brief Computes hash value for blocked memory descriptor
 *
 * Creates a hash from the memory descriptor's strides and blocking info
 * to uniquely identify the blocking format.
 *
 * @param mem_desc Memory descriptor to hash
 * @return Hash value representing the blocking format
 */
static size_t hashBlockingDesc(const dnnl::memory::desc &mem_desc) {
  size_t hash_value = 0;
  // Mersenne prime number to avoid collisions
  const size_t prime = 31;

  // Hash strides
  const auto strides = mem_desc.get_strides();
  for (const auto &stride : strides) {
    hash_value = hash_value * prime + std::hash<int64_t> {}(stride);
  }

  // Hash inner_nblks
  const int inner_nblks = mem_desc.get_inner_nblks();
  hash_value = hash_value * prime + std::hash<int> {}(inner_nblks);

  // Hash inner_blks and inner_idxs
  const auto inner_blks = mem_desc.get_inner_blks();
  const auto inner_idxs = mem_desc.get_inner_idxs();
  for (int i = 0; i < inner_nblks; ++i) {
    hash_value = hash_value * prime + std::hash<int64_t> {}(inner_blks[i]);
    hash_value = hash_value * prime + std::hash<int64_t> {}(inner_idxs[i]);
  }

  return hash_value;
}

void getOrCreateBlockedWeights(bool transA, bool transB, int M, int K, int N,
                               int lda, int ldb, onednn_utils_t::onednn_matmul_params &dnnl_params,
                               const dnnl::engine &eng, const dnnl::primitive_attr &matmul_attr,
                               int32_t weight_cache_type) {

  // Static containers with mutex for thread safety
  static std::unordered_map<Key_matmul, size_t> hash_values;
  static lru_cache_t<Key_matmul, dnnl::memory> matmul_weight_cache;
  static std::mutex blocked_weight_mutex;

  // Full key includes all parameters that affect blocking decision
  Key_matmul full_key(transA, transB, M, K, N, lda, ldb,
                      dnnl_params.weights.buffer,
                      static_cast<uint32_t>(matmul_algo_t::onednn_blocked));

  // Lock for thread-safe cache access
  std::lock_guard<std::mutex> lock(blocked_weight_mutex);

  // Check if we have a cached blocking hash for this configuration
  auto hash_it = hash_values.find(full_key);
  if (hash_it != hash_values.end()) {
    size_t blocking_hash = hash_it->second;
    Key_matmul cache_key(transB, K, N, ldb, dnnl_params.weights.buffer,
                         static_cast<uint32_t>(matmul_algo_t::onednn_blocked),
                         blocking_hash);

    // Check if the weight is still in the LRU cache (may have been evicted)
    if (matmul_weight_cache.find_key(cache_key)) {
      apilog_info("Read onednn cached weights (cache hit)");
      dnnl_params.weights.mem = matmul_weight_cache.get(cache_key);
      return;
    }
    // Entry was evicted from LRU cache, remove stale hash_values entry
    hash_values.erase(hash_it);
  }

  // Cache miss or stale entry - need to create primitive descriptor to get blocking format
  dnnl::memory::desc dnnl_weight_desc = onednn_utils_t::to_dnnl_tensor(
                                          dnnl_params.weights, eng);

  // Create blocked matmul primitive descriptor to determine optimal blocking
  dnnl::matmul::primitive_desc matmul_pd = create_blocked_matmul_pd(
        dnnl_params, eng, matmul_attr);

  // Compute blocking hash and check if already cached (by another full_key configuration)
  size_t blocking_hash = hashBlockingDesc(matmul_pd.weights_desc());
  Key_matmul cache_key(transB, K, N, ldb, dnnl_params.weights.buffer,
                       static_cast<uint32_t>(matmul_algo_t::onednn_blocked),
                       blocking_hash);

  // Check if blocked weights already exist in cache (from different full_key with same blocking)
  if (matmul_weight_cache.find_key(cache_key)) {
    apilog_info("Read onednn cached weights (blocking hash match)");
    dnnl_params.weights.mem = matmul_weight_cache.get(cache_key);
    // Update hash_values for faster lookup next time
    hash_values[full_key] = blocking_hash;
    return;
  }

  // Not in cache - perform reorder
  dnnl::memory dnnl_weight_mem = dnnl::memory(dnnl_weight_desc, eng,
                                 dnnl_params.weights.buffer);
  dnnl::memory dnnl_blocked_weight_mem = dnnl::memory(matmul_pd.weights_desc(),
                                         eng);

  dnnl::stream eng_stream(eng);
  reorder(dnnl_weight_mem, dnnl_blocked_weight_mem).execute(eng_stream,
      dnnl_weight_mem, dnnl_blocked_weight_mem);
  eng_stream.wait();  // Ensure reorder completes before using the memory

  dnnl_params.weights.mem = dnnl_blocked_weight_mem;

  if (weight_cache_type == 0) {
    apilog_info("onednn reorder weights (WEIGHT_CACHE_DISABLE)");
    return;
  }

  // Cache the blocked weights
  apilog_info("onednn reorder weights (adding to cache)");
  hash_values[full_key] = blocking_hash;
  matmul_weight_cache.add(cache_key, dnnl_params.weights.mem);
}

void matmul_onednn_wrapper(char transA, char transB, int M, int N,
                           int K, float alpha, const void *A, int lda, const void *B, int ldb, float beta,
                           void *C, int ldc, matmul_params &lowoha_params,
                           matmul_batch_params_t &batch_params,
                           const void *bias, zendnnl::ops::matmul_algo_t kernel, bool is_weights_const,
                           size_t src_batch_stride, size_t weight_batch_stride, size_t dst_batch_stride) {
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

  int batch_count = std::max(batch_params.Batch_A, batch_params.Batch_B);
  if (batch_count == 1) {
    dnnl_params.src.dims = {M, K};
    dnnl_params.weights.dims = {K, N};
    dnnl_params.dst.dims = {M, N};
    if (bias != nullptr) dnnl_params.bias.dims = {1, N};
  }
  else {
    dnnl_params.src.dims = {batch_params.Batch_A, M, K};
    dnnl_params.weights.dims = {batch_params.Batch_B, K, N};
    dnnl_params.dst.dims = {batch_count, M, N};
    if (bias != nullptr) dnnl_params.bias.dims = {1, 1, N};
  }

  dnnl_params.src.is_transposed = (transA == 'n') ? false : true;
  dnnl_params.weights.is_transposed = (transB == 'n') ? false : true;
  dnnl_params.algo = kernel;

  if (batch_count == 1) {
    dnnl_params.src.format_tag = (transA == 'n') ? "ab" : "ba";
    dnnl_params.src.strides    = (transA == 'n') ? std::vector<long int> {lda, 1} :
                                 std::vector<long int> {1, lda};
    dnnl_params.weights.format_tag = (transB == 'n') ? "ab" : "ba";
    dnnl_params.weights.strides = (transB == 'n') ? std::vector<long int> {ldb, 1} :
                                  std::vector<long int> {1, ldb};
    dnnl_params.dst.format_tag = "ab";
    dnnl_params.dst.strides    = std::vector<long int> {ldc, 1};
    if (bias != nullptr) {
      dnnl_params.bias.format_tag = "ab";
      dnnl_params.bias.strides    = std::vector<long int> {0, 1};
    }
  }
  else {
    // Cast size_t strides to long int to avoid narrowing conversion warnings
    long int src_stride = static_cast<long int>(src_batch_stride);
    long int wei_stride = static_cast<long int>(weight_batch_stride);
    long int dst_stride = static_cast<long int>(dst_batch_stride);

    dnnl_params.src.format_tag = (transA == 'n') ? "abc" : "acb";
    dnnl_params.src.strides = (transA == 'n') ?
                              std::vector<long int> {src_stride, lda, 1} :
                              std::vector<long int> {src_stride, 1, lda};

    dnnl_params.weights.format_tag = (transB == 'n') ? "abc" : "acb";
    dnnl_params.weights.strides = (transB == 'n') ?
                                  std::vector<long int> {wei_stride, ldb, 1} :
                                  std::vector<long int> {wei_stride, 1, ldb};

    dnnl_params.dst.format_tag = "abc";
    dnnl_params.dst.strides = std::vector<long int> {dst_stride, ldc, 1};

    if (bias != nullptr) {
      dnnl_params.bias.format_tag = "abc";
      dnnl_params.bias.strides = std::vector<long int> {0, 0, 1};
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

  if (lowoha_params.quant_params.src_scale.buff) {
    dnnl_params.src_quant.scales      = lowoha_params.quant_params.src_scale.buff;
    dnnl_params.src_quant.scale_dtype = lowoha_params.quant_params.src_scale.dt;
    dnnl_params.src_quant.scale_size  = lowoha_params.quant_params.src_scale.dims;
    matmul_attr.set_scales_mask(DNNL_ARG_SRC,
                                dnnl_params.src_quant.scale_size.back() == 1 ? 0 : 1 << 1);

    if (lowoha_params.quant_params.src_zp.buff) {
      dnnl_params.src_quant.zero_points      = lowoha_params.quant_params.src_zp.buff;
      dnnl_params.src_quant.zero_dtype       = lowoha_params.quant_params.src_zp.dt;
      dnnl_params.src_quant.zero_size        = lowoha_params.quant_params.src_zp.dims;
      matmul_attr.set_zero_points_mask(DNNL_ARG_SRC,
                                       dnnl_params.src_quant.zero_size.back() == 1 ? 0 : 1 << 1);
    }
  }

  if (lowoha_params.quant_params.wei_scale.buff) {
    dnnl_params.weights_quant.scales      =
      lowoha_params.quant_params.wei_scale.buff;
    dnnl_params.weights_quant.scale_dtype = lowoha_params.quant_params.wei_scale.dt;
    dnnl_params.weights_quant.scale_size  =
      lowoha_params.quant_params.wei_scale.dims;
    matmul_attr.set_scales_mask(DNNL_ARG_WEIGHTS,
                                dnnl_params.weights_quant.scale_size.back() == 1 ? 0 : 1 << 1);

    if (lowoha_params.quant_params.wei_zp.buff) {
      dnnl_params.weights_quant.zero_points      =
        lowoha_params.quant_params.wei_zp.buff;
      dnnl_params.weights_quant.zero_dtype       =
        lowoha_params.quant_params.wei_zp.dt;
      dnnl_params.weights_quant.zero_size        =
        lowoha_params.quant_params.wei_zp.dims;
      matmul_attr.set_zero_points_mask(DNNL_ARG_WEIGHTS,
                                       dnnl_params.weights_quant.zero_size.back() == 1 ? 0 : 1 << 1);
    }
  }

  if (lowoha_params.quant_params.dst_scale.buff) {
    dnnl_params.dst_quant.scales      = lowoha_params.quant_params.dst_scale.buff;
    dnnl_params.dst_quant.scale_dtype = lowoha_params.quant_params.dst_scale.dt;
    dnnl_params.dst_quant.scale_size  = lowoha_params.quant_params.dst_scale.dims;
    matmul_attr.set_scales_mask(DNNL_ARG_DST,
                                dnnl_params.dst_quant.scale_size.back() == 1 ? 0 : 1 << 1);

    if (lowoha_params.quant_params.dst_zp.buff) {
      dnnl_params.dst_quant.zero_points      = lowoha_params.quant_params.dst_zp.buff;
      dnnl_params.dst_quant.zero_dtype       = lowoha_params.quant_params.dst_zp.dt;
      dnnl_params.dst_quant.zero_size        = lowoha_params.quant_params.dst_zp.dims;
      matmul_attr.set_zero_points_mask(DNNL_ARG_DST,
                                       dnnl_params.dst_quant.zero_size.back() == 1 ? 0 : 1 << 1);
    }
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

  bool is_blocked = dnnl_params.algo == matmul_algo_t::onednn_blocked &&
                    is_weights_const;
  if (is_blocked) {
    getOrCreateBlockedWeights(transA == 't', transB == 't', M, K, N, lda, ldb,
                              dnnl_params, eng, matmul_attr, weight_cache_type);
    dnnl_params.is_blocked = true;
  }

  matmul_onednn_kernel_t::execute_matmul(dnnl_params, matmul_args, matmul_attr,
                                         eng);
}

#endif

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl