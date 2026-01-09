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

// Global mutex for thread-safe lowoha operations (cache, auto-tuner map, etc.)
std::mutex& get_lowoha_mutex() {
  static std::mutex lowoha_mutex;
  return lowoha_mutex;
}

status_t validate_matmul_direct_inputs(const void *src, const void *weight,
                                       const void *dst,
                                       const int M, const int N, const int K,
                                       const int Batch_A, const int Batch_B,
                                       lowoha_params &params,
                                       const bool is_weights_const) {
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

  // Check quantization parameters
  // WOQ (Weight-Only Quantization) is supported for BF16 src with S4 weights
  // Only weight scale and weight zero point are allowed for WOQ
  const bool is_woq = (params.dtypes.src == data_type_t::bf16) &&
                      (params.dtypes.wei == data_type_t::s4);
  
  // INT8 quantization: u8/s8 src with s8 weights
  const bool is_int8 = (params.dtypes.src == data_type_t::u8 || 
                        params.dtypes.src == data_type_t::s8) &&
                       (params.dtypes.wei == data_type_t::s8);
  
  // WOQ and INT8 require constant weights for weight reordering/caching
  if (is_woq && !is_weights_const) {
    log_error("WOQ requires constant weights (is_weights_const=true)");
    return status_t::failure;
  }
  
  // Source and destination quantization params are only supported for INT8
  if ((params.quant_params.src_scale.buff || params.quant_params.dst_scale.buff ||
       params.quant_params.src_zp.buff || params.quant_params.dst_zp.buff) && !is_int8) {
    log_error("Source/destination quantization params are only supported for INT8 (u8/s8 src + s8 weights)");
    return status_t::failure;
  }
  
  // Weight quantization params only allowed for WOQ or INT8
  if ((params.quant_params.wei_scale.buff || params.quant_params.wei_zp.buff) && !is_woq && !is_int8) {
    log_error("Weight quantization params are only supported for WOQ (BF16 src + S4 weights) or INT8");
    return status_t::failure;
  }

  // Validate data types
  const bool is_f32_src  = (params.dtypes.src == data_type_t::f32);
  const bool is_bf16_src = (params.dtypes.src == data_type_t::bf16);
  const bool is_u8_src   = (params.dtypes.src == data_type_t::u8);
  const bool is_s8_src   = (params.dtypes.src == data_type_t::s8);
  const bool is_f32_out  = (params.dtypes.dst == data_type_t::f32);
  const bool is_bf16_out = (params.dtypes.dst == data_type_t::bf16);
  const bool is_u8_out   = (params.dtypes.dst == data_type_t::u8);
  const bool is_s8_out   = (params.dtypes.dst == data_type_t::s8);
  const bool is_s32_out  = (params.dtypes.dst == data_type_t::s32);

  if ((!is_f32_src && !is_bf16_src && !is_u8_src && !is_s8_src)) {
    log_error("Unsupported source data type: ",
              data_type_to_string(params.dtypes.src));
    return status_t::failure;
  }

  if ((!is_f32_out && !is_bf16_out && !is_u8_out && !is_s8_out && !is_s32_out)) {
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

  // Set leading dimension for binary buffers if not set
  for (auto &po : params.postop_) {
    if (po.po_type == post_op_type_t::binary_add ||
        po.po_type == post_op_type_t::binary_mul) {
      if (po.leading_dim == -1) {
        po.leading_dim = N;
      }
    }
  }

  return status_t::success;
}

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
  case matmul_algo_t::aocl_dlp:
    return "aocl_dlp";
  case matmul_algo_t::aocl_dlp_blocked:
    return "aocl_dlp_blocked";
#else
  case matmul_algo_t::aocl_dlp:
    return "aocl_blis";
  case matmul_algo_t::aocl_dlp_blocked:
    return "aocl_blis_blocked";
#endif
  case matmul_algo_t::onednn:
    return "onednn";
  case matmul_algo_t::onednn_blocked:
    return "onednn_blocked";
  case matmul_algo_t::libxsmm:
    return "libxsmm";
  case matmul_algo_t::libxsmm_blocked:
    return "libxsmm_blocked";
  case matmul_algo_t::batched_sgemm:
    return "batched_sgemm";
  case matmul_algo_t::dynamic_dispatch:
    return "dynamic_dispatch";
  case matmul_algo_t::reference:
    return "reference";
  case matmul_algo_t::auto_tuner:
    return "auto_tuner";
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
  case data_type_t::s4:
    return "s4";
  case data_type_t::s8:
    return "s8";
  case data_type_t::u8:
    return "u8";
  case data_type_t::s32:
    return "s32";
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

inline bool may_i_use_dlp_partition(int batch_count, int M, int N,
                                     int num_threads, data_type_t dtype) {

  // Set thresholds based on thread count and data type (powers of 2 only)
  int M_threshold = 0, N_threshold = 0, work_threshold = 0;

  /*DLP performs better when M and N are large and thread count is moderate to high.
   It uses internal tiling and cache-aware scheduling,
   where each 8-core cluster shares a 32MB L3 cache. Manual OpenMP partitioning
   can disrupt DLP's optimized workload distribution, leading to contention.
   Delegating to DLP ensures better throughput and efficient hardware utilization.*/
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

  // Allow DLP if batch size is small and M is reasonably large
  bool small_batch_override = (batch_count <= 8 && M >= 1024);

  return ((M >= M_threshold &&
           N >= N_threshold &&
           work_per_thread >= work_threshold)
          || small_batch_override);
}

// TODO: Further tune the heuristics based on num_threads and other params
inline matmul_algo_t select_algo_by_heuristics_bf16_bmm(int BS, int M, int N,
    int K, int num_threads) {
  if (BS <= 512) {
    if (N <= 512) {
      if (N <= 48) {
        if (M <= 196) {
          return matmul_algo_t::libxsmm;
        }
        else {
          return matmul_algo_t::aocl_dlp;
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
        return matmul_algo_t::aocl_dlp;
      }
    }
  }
  else {
    if (K <= 48) {
      return matmul_algo_t::libxsmm;
    }
    else {
      if (K < 50) {
        return matmul_algo_t::aocl_dlp;
      }
      else {
        if (K <= 196) {
          return matmul_algo_t::libxsmm;
        }
        else {
          return matmul_algo_t::aocl_dlp;
        }
      }
    }
  }
}

/* ML-generated heuristics using decision tree with instance-weighted training, cross-validation, and pruning.
   Optimized via grid search on stratified train-test split.*/
inline matmul_algo_t select_algo_by_heuristics_bf16_mm(int M, int N, int K) {
  if (M <= 493) {
      if (K <= 352) {
          if (K <= 124) {
              return matmul_algo_t::onednn_blocked;
          } else {
              return matmul_algo_t::aocl_dlp_blocked;
          }
      } else {
          if (M <= 352) {
              if (K <= 1344) {
                  return matmul_algo_t::onednn_blocked;
              } else {
                  if (N <= 2548) {
                      return matmul_algo_t::aocl_dlp_blocked;
                  } else {
                      return matmul_algo_t::onednn_blocked;
                  }
              }
          } else {
              return matmul_algo_t::onednn_blocked;
          }
      }
  } else {
      if (M <= 187680) {
          if (K <= 896) {
              if (K <= 248) {
                  if (N <= 2560) {
                      if (N <= 136) {
                          if (N <= 40) {
                              return matmul_algo_t::onednn_blocked;
                          } else {
                              return matmul_algo_t::aocl_dlp_blocked;
                          }
                      } else {
                          return matmul_algo_t::onednn_blocked;
                      }
                  } else {
                      return matmul_algo_t::aocl_dlp_blocked;
                  }
              } else {
                  if (N <= 96) {
                      return matmul_algo_t::aocl_dlp_blocked;
                  } else {
                      if (N <= 248) {
                          return matmul_algo_t::onednn_blocked;
                      } else {
                          if (K <= 448) {
                              return matmul_algo_t::aocl_dlp_blocked;
                          } else {
                              return matmul_algo_t::onednn_blocked;
                          }
                      }
                  }
              }
          } else {
              return matmul_algo_t::aocl_dlp_blocked;
          }
      } else {
          return matmul_algo_t::aocl_dlp_blocked;
      }
  }
}

unsigned int get_auto_tuner_ver() {
  char *skip_env_var = std::getenv("ZENDNNL_AUTO_TUNER_TYPE");
  if (skip_env_var) {
    unsigned int version = std::stoi(skip_env_var);
    // Current support is 2 versions.
    if (version == 0 || version > 2) {
      return 2;
    }
    return version;
  }
  // return version 2 as default
  return 2;
}

matmul_algo_t kernel_select(lowoha_params &params, int Batch_A, int Batch_B,
                            int batch_count, int M, int N, int K, int num_threads, const void *bias,
                            const bool is_weights_const) {
  matmul_config_t &matmul_config = matmul_config_t::instance();
  int32_t algo = params.lowoha_algo == matmul_algo_t::none ?
                 matmul_config.get_algo() : static_cast<int>(params.lowoha_algo);

  matmul_algo_t kernel = (algo == static_cast<int>(matmul_algo_t::none)) ?
                         ((batch_count == 1 && is_weights_const) ? matmul_algo_t::aocl_dlp
                          : matmul_algo_t::dynamic_dispatch) : static_cast<matmul_algo_t>(algo);

  // TODO: Fallback to reference/supported kernel
  if (kernel == matmul_algo_t::auto_tuner && (Batch_A != 1 || Batch_B != 1 ||
      !is_weights_const)) {
    kernel = matmul_algo_t::dynamic_dispatch;
  }
  if ((kernel == matmul_algo_t::onednn ||
       kernel == matmul_algo_t::onednn_blocked) && (Batch_A != 1 && Batch_B != 1 &&
           Batch_A != Batch_B)) {
    log_info("OneDNN kernel is not supported for the given batch sizes");
    kernel = matmul_algo_t::aocl_dlp;
  }

  if (kernel==matmul_algo_t::dynamic_dispatch) {
    if (batch_count > 1) {
      kernel = select_algo_by_heuristics_bf16_bmm(batch_count, M, N, K, num_threads);
    }
    else {
      if (is_weights_const == false && M >= 4096 && M <= 8192 && K == 1024 &&
          N == 1024) {
        kernel = matmul_algo_t::libxsmm_blocked;
      }
      else {
        kernel = select_algo_by_heuristics_bf16_mm(M, N, K);
      }
    }
  }
  if ((!ZENDNNL_DEPENDS_ONEDNN && (kernel == matmul_algo_t::onednn ||
                                   kernel == matmul_algo_t::onednn_blocked)) ||
      (!ZENDNNL_DEPENDS_LIBXSMM && (kernel == matmul_algo_t::libxsmm ||
                                    kernel == matmul_algo_t::libxsmm_blocked)) ||
      (kernel >= matmul_algo_t::algo_count)) {
    kernel = matmul_algo_t::aocl_dlp;
  }

  // Force aocl_dlp for WOQ (Weight-Only Quantization) cases
  // WOQ uses specialized AOCL kernels that don't support blocked format
  const bool is_woq = (params.dtypes.src == data_type_t::bf16) &&
                      (params.dtypes.wei == data_type_t::s4);
  if (is_woq) {
    log_info("WOQ detected, switching to aocl_dlp_blocked kernel");
    kernel = matmul_algo_t::aocl_dlp_blocked;
  }

  // AOCL blocked kernel is not supported for batched matmul
  if ((Batch_A > 1 || Batch_B > 1) &&
      kernel == matmul_algo_t::aocl_dlp_blocked) {
    kernel = matmul_algo_t::aocl_dlp;
  }
  // TODO: Update the conditon once prepack supports other formats
  // Current prepack supports only AOCL blocked kernel
  if (params.mem_format_b == 'r') {
    kernel = matmul_algo_t::aocl_dlp_blocked;
  }

  // TODO: Remove this once AOCL DLP has fix for parallel matmul
  // Currently diverting the call to LibXSMM for Batch MatMul.
  if ((Batch_A > 1 || Batch_B > 1) && kernel == matmul_algo_t::aocl_dlp) {
    kernel = matmul_algo_t::libxsmm;
  }

  params.lowoha_algo = kernel;
  return kernel;
}

int get_tile_size_from_env(const char *env_var, int default_value) {
  const char *env_value = std::getenv(env_var);
  if (env_value != nullptr) {
    int value = std::stoi(env_value);
    if (value > 0) {
      return value;
    }
    else {
      return default_value;
    }
  }
  return default_value;
}

// This function selects tile sizes for BF16 matmul based on heuristics
// TODO: Further tune the heuristics based on num_threads
std::tuple<int, int> selectTileBF16(int M, int N, int K, int num_threads) {

  if (M <= 2048 && N <= 128) {
    return {32, 32};
  }

  if (M <= 4096 && N > 768 && N <= 1024) {
    return {64, 64};
  }

  return {128, 64};
}

} // namespace lowoha
} // namespace zendnnl
