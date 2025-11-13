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
#include <sstream>
#include <string>
#if ZENDNNL_DEPENDS_AOCLDLP
  #include "aocl_dlp.h"
#else
  #include "blis.h"
#endif

namespace zendnnl {
namespace lowoha {

status_t validate_matmul_direct_inputs(const void *src, const void *weight,
                                       const void *dst,
                                       const int M, const int N, const int K,
                                       const int Batch_A, const int Batch_B,
                                       const lowoha_params &params) {
  // Check for null pointers
  if (!src || !weight || !dst) {
    log_error("Null pointer input to matmul_direct: src=",
              static_cast<const void *>(src),
              ", weight=", static_cast<const void *>(weight), ", dst=",
              static_cast<const void *>(dst));
    return status_t::failure;
  }

  // Validate matrix dimensions and batch sizes
  if (M <= 0 || N <= 0 || K <= 0) {
    log_error("Invalid matrix dimensions: M=", M, ", N=", N, ", K=", K);
    return status_t::failure;
  }

  if (Batch_A <= 0 || Batch_B <= 0) {
    log_error("Invalid batch sizes: Batch_A=", Batch_A, ", Batch_B=", Batch_B);
    return status_t::failure;
  }

  // Check quantization parameters (not supported yet)
  if (params.quant_params.src_scale.buff || params.quant_params.wei_scale.buff ||
      params.quant_params.dst_scale.buff ||
      params.quant_params.src_zp.buff || params.quant_params.wei_zp.buff ||
      params.quant_params.dst_zp.buff) {
    log_error("Quantization params are not supported in LOWOHA matmul_direct yet");
    return status_t::failure;
  }

  // Validate data types
  const bool is_f32_src  = (params.dtypes.src == data_type_t::f32);
  const bool is_bf16_src = (params.dtypes.src == data_type_t::bf16);
  const bool is_f32_out  = (params.dtypes.dst == data_type_t::f32);
  const bool is_bf16_out = (params.dtypes.dst == data_type_t::bf16);

  if ((!is_f32_src && !is_bf16_src)) {
    log_error("Unsupported source data type: ",
              data_type_to_string(params.dtypes.src));
    return status_t::failure;
  }

  if ((!is_f32_out && !is_bf16_out)) {
    log_error("Unsupported destination data type: ",
              data_type_to_string(params.dtypes.dst));
    return status_t::failure;
  }
  // F32 src and dst BF16 is not supported
  if (is_f32_src && is_bf16_out) {
    log_error("Unsupported GEMM configuration: F32 source with BF16 destination");
    return status_t::failure;
  }

  // Validate batch broadcasting compatibility
  if (std::max(Batch_A, Batch_B) % std::min(Batch_A, Batch_B) != 0) {
    log_error("Broadcasting is not compatible with given batch sizes: Batch_A=",
              Batch_A, ", Batch_B=", Batch_B);
    return status_t::failure;
  }

  return status_t::success;
}

template <typename T>
bool reorderAndCacheWeights(Key_matmul key, const void *weights,
                            void *&reorder_weights, const int k, const int n, const int ldb,
                            const char order, const char trans, char mem_format_b,
                            get_reorder_buff_size_func_ptr get_reorder_buf_size,
                            reorder_func_ptr<T> reorder_func, int weight_cache_type) {
  // Weight caching
  static lru_cache_t<Key_matmul, void *> matmul_weight_cache;

  // Weights are already reordered and algo is aocl_blis_blocked
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

// Helper function to convert post_op_type_t to string
inline const char *post_op_type_to_string(post_op_type_t type) {
  switch (type) {
  case post_op_type_t::none:
    return "none";
  case post_op_type_t::relu:
    return "relu";
  case post_op_type_t::leaky_relu:
    return "leaky_relu";
  case post_op_type_t::gelu_tanh:
    return "gelu_tanh";
  case post_op_type_t::gelu_erf:
    return "gelu_erf";
  case post_op_type_t::sigmoid:
    return "sigmoid";
  case post_op_type_t::swish:
    return "swish";
  case post_op_type_t::tanh:
    return "tanh";
  case post_op_type_t::binary_add:
    return "binary_add";
  case post_op_type_t::binary_mul:
    return "binary_mul";
  default:
    return "unknown";
  }
}

std::string post_op_names_to_string(const lowoha_params &params) {
  std::ostringstream post_op_names;
  if (params.postop_.empty()) {
    post_op_names << "none";
  }
  else {
    for (size_t i = 0; i < params.postop_.size(); ++i) {
      if (i > 0) {
        post_op_names << ",";
      }
      post_op_names << post_op_type_to_string(params.postop_[i].po_type);
    }
  }
  return post_op_names.str();
}

const char *kernel_to_string(matmul_algo_t kernel) {
  switch (kernel) {
#if ZENDNNL_DEPENDS_AOCLDLP
  case matmul_algo_t::aocl_blis:
    return "aocl_dlp";
  case matmul_algo_t::aocl_blis_blocked:
    return "aocl_dlp_blocked";
#else
  case matmul_algo_t::aocl_blis:
    return "aocl_blis";
  case matmul_algo_t::aocl_blis_blocked:
    return "aocl_blis_blocked";
#endif
  case matmul_algo_t::onednn:
    return "onednn";
  case matmul_algo_t::onednn_blocked:
    return "onednn_blocked";
  case matmul_algo_t::libxsmm:
    return "libxsmm";
  case matmul_algo_t::batched_sgemm:
    return "batched_sgemm";
  case matmul_algo_t::dynamic_dispatch:
    return "dynamic_dispatch";
  case matmul_algo_t::reference:
    return "reference";
  default:
    return "none";
  }
}

const char *data_type_to_string(data_type_t dtype) {
  switch (dtype) {
  case data_type_t::none:
    return "none";
  case data_type_t::f32:
    return "f32";
  case data_type_t::bf16:
    return "bf16";
  default:
    return "unknown";
  }
}

std::string post_op_data_types_to_string(const lowoha_params &params) {
  std::ostringstream post_op_dtypes;
  bool first = true;
  for (const auto &po : params.postop_) {
    if (po.po_type == post_op_type_t::binary_add ||
        po.po_type == post_op_type_t::binary_mul) {
      if (!first) {
        post_op_dtypes << ",";
      }
      post_op_dtypes << data_type_to_string(po.dtype);
      first = false;
    }
  }
  return post_op_dtypes.str();
}

inline bool may_i_use_blis_partition(int batch_count, int M, int N,
                                     int num_threads, data_type_t dtype) {

  // Set thresholds based on thread count and data type (powers of 2 only)
  int M_threshold = 0, N_threshold = 0, work_threshold = 0;

  /*BLIS performs better when M and N are large and thread count is moderate to high.
   It uses internal tiling and cache-aware scheduling,
   where each 8-core cluster shares a 32MB L3 cache. Manual OpenMP partitioning
   can disrupt BLIS's optimized workload distribution, leading to contention.
   Delegating to BLIS ensures better throughput and efficient hardware utilization.*/
  // TODO: Tune it more based on heuristics (threshold relies on problem size and data type)
  if (num_threads <= 16) {
    M_threshold    = 512;
    N_threshold    = 256;
    work_threshold = 128;
  }
  else if (num_threads <= 32) {
    M_threshold    = 1024;
    N_threshold    = 512;
    work_threshold = 256;
  }
  else {
    M_threshold    = 2048;
    N_threshold    = 1024;
    work_threshold = 512;
  }
  // Estimate effective workload per thread
  int work_per_thread = (batch_count * M) / num_threads;

  // Allow BLIS if batch size is small and M is reasonably large
  bool small_batch_override = (batch_count <= 8 && M >= 1024);

  return ((M >= M_threshold &&
           N >= N_threshold &&
           work_per_thread >= work_threshold)
          || small_batch_override);
}

// TODO: Further tune the heuristics based on num_threads and other params
inline matmul_algo_t select_algo_by_heuristics_bf16(int BS, int M, int N, int K,
    int num_threads) {
  if (BS <= 512) {
    if (N <= 512) {
      if (N <= 48) {
        if (M <= 196) {
          return matmul_algo_t::libxsmm;
        }
        else {
          return matmul_algo_t::aocl_blis;
        }
      }
      else {
        return matmul_algo_t::libxsmm;
      }
    }
    else {
      if (K <= 512) {
        return matmul_algo_t::libxsmm;
      }
      else {
        return matmul_algo_t::aocl_blis;
      }
    }
  }
  else {
    if (K <= 48) {
      return matmul_algo_t::libxsmm;
    }
    else {
      if (K < 50) {
        return matmul_algo_t::aocl_blis;
      }
      else {
        if (K <= 196) {
          return matmul_algo_t::libxsmm;
        }
        else {
          return matmul_algo_t::aocl_blis;
        }
      }
    }
  }
}

matmul_algo_t kernel_select(lowoha_params &params, int Batch_A, int Batch_B,
                            int batch_count, int M, int N, int K, int num_threads, const void *bias) {
  matmul_config_t &matmul_config = matmul_config_t::instance();
  int32_t algo = params.lowoha_algo == matmul_algo_t::none ?
                 matmul_config.get_algo() : static_cast<int>(params.lowoha_algo);

  matmul_algo_t kernel = (algo == static_cast<int>(matmul_algo_t::none)) ?
                         matmul_algo_t::dynamic_dispatch : static_cast<matmul_algo_t>(algo);

  // TODO: Fallback to reference/supported kernel
  if ((kernel == matmul_algo_t::onednn ||
       kernel == matmul_algo_t::onednn_blocked) && (Batch_A != 1 && Batch_B != 1 &&
           Batch_A != Batch_B)) {
    log_info("OneDNN kernel is not supported for the given batch sizes");
    kernel = matmul_algo_t::aocl_blis;
  }

  if (kernel==matmul_algo_t::dynamic_dispatch) {
    if (batch_count > 1) {
      kernel = select_algo_by_heuristics_bf16(batch_count, M, N, K, num_threads);
    }
    else {
      kernel = matmul_algo_t::aocl_blis;
    }
  }
  if ((!ZENDNNL_DEPENDS_ONEDNN && (kernel == matmul_algo_t::onednn ||
                                   kernel == matmul_algo_t::onednn_blocked)) ||
      (!ZENDNNL_DEPENDS_LIBXSMM && (kernel == matmul_algo_t::libxsmm ||
                                    kernel == matmul_algo_t::libxsmm_blocked)) ||
      (kernel >= matmul_algo_t::algo_count)) {
    kernel = matmul_algo_t::aocl_blis;
  }
  // TODO: Remove condition, when libxsmm supports bias and post_ops.
  if ((kernel == matmul_algo_t::libxsmm || kernel == matmul_algo_t::libxsmm_blocked) &&
      (params.postop_.size() > 0 || bias != nullptr)) {
    kernel = matmul_algo_t::aocl_blis;
  }

  // AOCL blocked kernel is not supported for batched matmul
  if ((Batch_A > 1 || Batch_B > 1) &&
      kernel == matmul_algo_t::aocl_blis_blocked) {
    kernel = matmul_algo_t::aocl_blis;
  }
  // TODO: Update the conditon once prepack supports other formats
  // Current prepack supports only AOCL blocked kernel
  if (params.mem_format_b == 'r') {
    kernel = matmul_algo_t::aocl_blis_blocked;
  }

  params.lowoha_algo = kernel;
  return kernel;
}

} // namespace lowoha
} // namespace zendnnl