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
#include "common/zendnnl_global.hpp"
#include "matmul_native/native_matmul.hpp"
#include "matmul_native/common/cost_model.hpp"
#include "matmul_native/common/kernel_cache.hpp"
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
namespace matmul {

// Global mutex for thread-safe lowoha operations (cache, auto-tuner map, etc.)
std::mutex &get_lowoha_mutex() {
  static std::mutex lowoha_mutex;
  return lowoha_mutex;
}

size_t get_postop_batch_stride(const matmul_post_op &po) {
  if (po.dims.size() < 3 || po.dims[0] <= 1) {
    return 0;  // Not a 3D tensor or batch dimension is 1 (broadcast)
  }
  // Assuming dims layout is [Batch, M, N] with row-major storage
  size_t element_size = zendnnl::common::size_of(po.dtype);
  return po.dims[1] * po.dims[2] * element_size;
}

void apply_bmm_postop_offsets(matmul_params &params, int batch_idx,
                              int m_start, int N) {
  for (auto &po : params.postop_) {
    if ((po.po_type == post_op_type_t::binary_add ||
         po.po_type == post_op_type_t::binary_mul) &&
        po.buff != nullptr) {

      size_t element_size = zendnnl::common::size_of(po.dtype);
      size_t total_offset = 0;

      // Add batch offset for 3D post-ops
      if (is_3d_postop(po)) {
        total_offset += batch_idx * get_postop_batch_stride(po);
      }

      // Add row offset for partitioned execution
      total_offset += m_start * N * element_size;

      po.buff = static_cast<uint8_t *>(po.buff) + total_offset;
    }
  }
}

status_t validate_parallel_gemm_inputs(
  const std::vector<char> &layout,
  const std::vector<bool> &transA, const std::vector<bool> &transB,
  const std::vector<int> &M, const std::vector<int> &N,
  const std::vector<int> &K,
  const std::vector<float> &alpha,
  const std::vector<const void *> &src, const std::vector<int> &lda,
  const std::vector<const void *> &weight, const std::vector<int> &ldb,
  const std::vector<const void *> &bias, const std::vector<float> &beta,
  const std::vector<void *> &dst, const std::vector<int> &ldc,
  const std::vector<bool> &is_weights_const,
  const std::vector<matmul_params> &params) {

  const size_t num_ops = M.size();

  if (num_ops == 0) {
    log_error("group_matmul parallel: num_ops is 0");
    return status_t::failure;
  }

  // Validate all vector sizes match num_ops
  if (layout.size() != num_ops || transA.size() != num_ops ||
      transB.size() != num_ops || N.size() != num_ops ||
      K.size() != num_ops || alpha.size() != num_ops ||
      src.size() != num_ops || lda.size() != num_ops ||
      weight.size() != num_ops || ldb.size() != num_ops ||
      bias.size() != num_ops || beta.size() != num_ops ||
      dst.size() != num_ops || ldc.size() != num_ops ||
      is_weights_const.size() != num_ops || params.size() != num_ops) {
    log_error("group_matmul parallel: vector size mismatch, num_ops=", num_ops);
    return status_t::failure;
  }

  // Validate each operation's pointers and dimensions
  for (size_t i = 0; i < num_ops; ++i) {
    if (!src[i] || !weight[i] || !dst[i]) {
      log_error("group_matmul parallel: null pointer at operation ", i);
      return status_t::failure;
    }
    if (M[i] <= 0 || N[i] <= 0 || K[i] <= 0) {
      log_error("group_matmul parallel: invalid dimensions at operation ", i,
                ": M=", M[i], ", N=", N[i], ", K=", K[i]);
      return status_t::failure;
    }
  }

  return status_t::success;
}

status_t validate_sequential_gemm_inputs(
  const std::vector<char> &layout,
  const std::vector<bool> &transA, const std::vector<bool> &transB,
  const std::vector<int> &M, const std::vector<int> &N,
  const std::vector<int> &K,
  const std::vector<float> &alpha,
  const std::vector<const void *> &src, const std::vector<int> &lda,
  const std::vector<const void *> &weight, const std::vector<int> &ldb,
  const std::vector<const void *> &bias, const std::vector<float> &beta,
  const std::vector<void *> &dst, const std::vector<int> &ldc,
  const std::vector<bool> &is_weights_const,
  const std::vector<matmul_params> &params) {

  const size_t num_ops = M.size();

  if (num_ops == 0) {
    log_error("group_matmul sequential: num_ops is 0");
    return status_t::failure;
  }

  // In sequential mode, src.size() must be 1 (single input chained through ops)
  if (src.size() != 1) {
    log_error("group_matmul sequential: src.size() must be 1, got ", src.size());
    return status_t::failure;
  }

  // Validate all other vector sizes match num_ops
  if (layout.size() != num_ops || transA.size() != num_ops ||
      transB.size() != num_ops || N.size() != num_ops ||
      K.size() != num_ops || alpha.size() != num_ops ||
      lda.size() != num_ops || weight.size() != num_ops ||
      ldb.size() != num_ops || bias.size() != num_ops ||
      beta.size() != num_ops || dst.size() != num_ops ||
      ldc.size() != num_ops || is_weights_const.size() != num_ops ||
      params.size() != num_ops) {
    log_error("group_matmul sequential: vector size mismatch, num_ops=", num_ops);
    return status_t::failure;
  }

  // Validate src[0] is not null
  if (!src[0]) {
    log_error("group_matmul sequential: null src pointer");
    return status_t::failure;
  }

  // Validate each operation's pointers and dimensions
  for (size_t i = 0; i < num_ops; ++i) {
    if (!weight[i] || !dst[i]) {
      log_error("group_matmul sequential: null pointer at operation ", i);
      return status_t::failure;
    }
    if (M[i] <= 0 || N[i] <= 0 || K[i] <= 0) {
      log_error("group_matmul sequential: invalid dimensions at operation ", i,
                ": M=", M[i], ", N=", N[i], ", K=", K[i]);
      return status_t::failure;
    }
  }

  // Check M is constant across all operations (same batch size)
  for (size_t i = 1; i < num_ops; ++i) {
    if (M[i] != M[0]) {
      log_error("group_matmul sequential: M must be constant across layers, "
                "M[0]=", M[0], ", M[", i, "]=", M[i]);
      return status_t::failure;
    }
  }

  // Check dimension compatibility: K[i] must equal N[i-1] for chaining
  for (size_t i = 1; i < num_ops; ++i) {
    if (K[i] != N[i - 1]) {
      log_error("group_matmul sequential: dimension mismatch at layer ", i,
                ": K[", i, "]=", K[i], " != N[", i - 1, "]=", N[i - 1]);
      return status_t::failure;
    }
  }

  return status_t::success;
}

status_t validate_matmul_direct_inputs(const void *src, const void *weight,
                                       const void *dst,
                                       const int M, const int N, const int K,
                                       const int Batch_A, const int Batch_B,
                                       matmul_params &params,
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
  // WOQ (Weight-Only Quantization) is supported for BF16 src with S4/U4 weights
  // Only weight scale and weight zero point are allowed for WOQ
  const bool is_woq = (params.dtypes.src == data_type_t::bf16) &&
                      (params.dtypes.wei == data_type_t::s4 || params.dtypes.wei == data_type_t::u4);

  // INT8 quantization: s8 weights
  const bool is_int8 = params.dtypes.wei == data_type_t::s8;

  // WOQ and INT8 require constant weights for weight reordering/caching
  if (is_woq && !is_weights_const) {
    log_error("WOQ requires constant weights (is_weights_const=true)");
    return status_t::failure;
  }

  // Source and destination quantization params are only supported for INT8
  if ((params.quant_params.src_scale.buff || params.quant_params.dst_scale.buff ||
       params.quant_params.src_zp.buff || params.quant_params.dst_zp.buff) &&
      !is_int8) {
    log_error("Source/destination quantization params are only supported for INT8 (u8/s8 src + s8 weights)");
    return status_t::failure;
  }

  // TODO: Expand support for different granularities
  // Helper to validate per-tensor granularity for quant params
  auto validate_per_tensor = [](const void *buff,
  const std::vector<int64_t> &dims, const char *param_name) -> bool {
    if (!buff) {
      return true;
    }

    int64_t nelems = std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());

    if (nelems != 1) {
      log_error(param_name, " supports only per-tensor");
      return false;
    }
    return true;
  };

  if (params.quant_params.src_scale.buff) {
    int64_t nelems = std::accumulate(
                       params.quant_params.src_scale.dims.begin(),
                       params.quant_params.src_scale.dims.end(),
                       int64_t{1}, std::multiplies<int64_t>());
    if (nelems != 1) {
      bool is_supported_src = (params.dtypes.src == data_type_t::s8 ||
                               params.dtypes.src == data_type_t::bf16 ||
                               params.dtypes.src == data_type_t::f32);
      bool is_symmetric = (!params.quant_params.src_zp.buff);
      bool is_supported_dst = (params.dtypes.dst == data_type_t::f32 ||
                               params.dtypes.dst == data_type_t::bf16);
      bool is_per_token = (nelems == static_cast<int64_t>(M));
      bool is_per_group = (nelems > static_cast<int64_t>(M)) &&
                          (nelems % static_cast<int64_t>(M) == 0) &&
                          (static_cast<int64_t>(K) % (nelems / static_cast<int64_t>(M)) == 0);
      if (!(is_supported_src && is_symmetric && is_supported_dst &&
            (is_per_token || is_per_group))) {
        log_error("Source quant scale: per-token/per-group requires s8/bf16/f32 src, "
                  "symmetric quantization, and f32/bf16 output");
        return status_t::failure;
      }
      int64_t wei_scale_nelems = params.quant_params.wei_scale.buff
                                 ? std::accumulate(params.quant_params.wei_scale.dims.begin(),
                                     params.quant_params.wei_scale.dims.end(),
                                     int64_t{1}, std::multiplies<int64_t>())
                                 : 0;
      if (is_per_token && wei_scale_nelems != static_cast<int64_t>(N)) {
        log_error("Per-token source scale requires per-channel weight scale "
                  "(expected ", N, " elements, got ", wei_scale_nelems, ")");
        return status_t::failure;
      }
      if (is_per_group) {
        int64_t src_num_groups = nelems / static_cast<int64_t>(M);
        int64_t expected_wei_nelems = src_num_groups * static_cast<int64_t>(N);
        if (wei_scale_nelems != expected_wei_nelems) {
          log_error("Per-group source scale requires per-group weight scale "
                    "(expected ", expected_wei_nelems,
                    " elements, got ", wei_scale_nelems, ")");
          return status_t::failure;
        }
      }
    }
  }
  if (!validate_per_tensor(params.quant_params.src_zp.buff,
                           params.quant_params.src_zp.dims, "Source quant zero") ||
      !validate_per_tensor(params.quant_params.dst_scale.buff,
                           params.quant_params.dst_scale.dims, "Destination quant scale") ||
      !validate_per_tensor(params.quant_params.dst_zp.buff,
                           params.quant_params.dst_zp.dims, "Destination quant zero")) {
    return status_t::failure;
  }

  // Weight quantization params only allowed for WOQ or INT8
  if ((params.quant_params.wei_scale.buff || params.quant_params.wei_zp.buff) &&
      !is_woq && !is_int8) {
    log_error("Weight quantization params are only supported for WOQ (BF16 src + S4 weights) or INT8");
    return status_t::failure;
  }
  if (params.dtypes.wei == data_type_t::u4) {
    if (!params.quant_params.wei_zp.buff) {
      log_error("Weights quant zero is mandatory for u4 weights");
      return status_t::failure;
    }
    if (params.quant_params.wei_zp.dt != data_type_t::bf16 &&
        params.quant_params.wei_zp.dt != data_type_t::s8) {
      log_error("Weights quant zero supports only bf16 or s8 data type (mandatory) for u4 weights");
      return status_t::failure;
    }
  }

  // Validate data types
  const bool is_f32_src  = (params.dtypes.src == data_type_t::f32);
  const bool is_bf16_src = (params.dtypes.src == data_type_t::bf16);
  const bool is_u8_src   = (params.dtypes.src == data_type_t::u8);
  const bool is_s8_src   = (params.dtypes.src == data_type_t::s8);
  const bool is_f16_src  = (params.dtypes.src == data_type_t::f16);
  const bool is_f32_out  = (params.dtypes.dst == data_type_t::f32);
  const bool is_bf16_out = (params.dtypes.dst == data_type_t::bf16);
  const bool is_u8_out   = (params.dtypes.dst == data_type_t::u8);
  const bool is_s8_out   = (params.dtypes.dst == data_type_t::s8);
  const bool is_s32_out  = (params.dtypes.dst == data_type_t::s32);
  const bool is_f16_out  = (params.dtypes.dst == data_type_t::f16);

  if ((!is_f32_src && !is_bf16_src && !is_u8_src && !is_s8_src && !is_f16_src)) {
    log_error("Unsupported source data type: ",
              dtype_info(params.dtypes.src));
    return status_t::failure;
  }

  if ((!is_f32_out && !is_bf16_out && !is_u8_out && !is_s8_out && !is_s32_out &&
       !is_f16_out)) {
    log_error("Unsupported destination data type: ",
              dtype_info(params.dtypes.dst));
    return status_t::failure;
  }
  // F32 src with dst BF16/F16 is not supported
  if (is_f32_src && (is_bf16_out || is_f16_out)) {
    log_error("Unsupported GEMM configuration: F32 source with BF16/F16 destination");
    return status_t::failure;
  }

  if (is_bf16_src && is_f16_out) {
    log_error("Unsupported GEMM configuration: BF16 source with F16 destination");
    return status_t::failure;
  }

  if (is_f16_src && is_bf16_out) {
    log_error("Unsupported GEMM configuration: F16 source with BF16 destination");
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

std::string post_op_names_to_string(const matmul_params &params) {
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
  case matmul_algo_t::native_gemm:
    return "native_gemm";
  case matmul_algo_t::native_brgemm:
    return "native_brgemm";
  default:
    return "none";
  }
}

std::string post_op_data_types_to_string(const matmul_params &params) {
  std::ostringstream post_op_dtypes;
  bool first = true;
  for (const auto &po : params.postop_) {
    if (po.po_type == post_op_type_t::binary_add ||
        po.po_type == post_op_type_t::binary_mul) {
      if (!first) {
        post_op_dtypes << ",";
      }
      post_op_dtypes << dtype_info(po.dtype);
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
inline matmul_algo_t select_algo_by_heuristics_woq_int4_mm(int M, int N, int K,
    int num_threads) {
  // For Higher thread count(i.e >128) AOCL S4 Kernels gives optimal performance
  if (num_threads > 128) {
    return matmul_algo_t::aocl_dlp_blocked;
  }
  else {
    // If M <= 16 AOCL S4 Kernel gives Optimal Performance.
    // If M >= 128, N and K >=1024 AOCL BLIS kernels with Zen weights conversion
    // gives optimal performance.
    // This is based on heuristic with different models and difference BS
    if (M <= 16) {
      // AOCL S4 Kernel
      return matmul_algo_t::aocl_dlp_blocked;
    }
    else if (M >= 128 && N >= 1024 && K >= 1024) {
      // AOCL BF16 Kernel with Zen Weights Conversion
      return matmul_algo_t::aocl_dlp;
    }
    else if (M == 32) {
      if (N <= K) {
        // TODO: Implement Blocked BRGEMM BF16 with Zen Weights Conversion
        return matmul_algo_t::aocl_dlp;
      }
      else {
        // AOCL S4 Kernel
        return matmul_algo_t::aocl_dlp_blocked;
      }
    }
    else {
      // AOCL BF16 Kernel with Zen Weights Conversion
      // TODO: Implement Blocked BRGEMM BF16 with Zen Weights Conversion for N > K case
      return matmul_algo_t::aocl_dlp;
    }
  }
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
  if (K <= 864) {
    if (N <= 136) {
      if (K <= 448) {
        return matmul_algo_t::aocl_dlp_blocked;
      }
      else {
        if (K <= 640) {
          if (N <= 37) {
            return matmul_algo_t::onednn_blocked;
          }
          else {
            return matmul_algo_t::aocl_dlp_blocked;
          }
        }
        else {
          return matmul_algo_t::aocl_dlp_blocked;
        }
      }
    }
    else {
      if (N <= 884) {
        if (N <= 544) {
          if (N <= 248) {
            return matmul_algo_t::onednn_blocked;
          }
          else {
            return matmul_algo_t::aocl_dlp_blocked;
          }
        }
        else {
          // TODO: Consider adding K-based threshold check here (e.g., K <= 216)
          // if different kernels are needed for different K ranges in the future
          return matmul_algo_t::onednn_blocked;
        }
      }
      else {
        if (K <= 384) {
          if (K <= 160) {
            return matmul_algo_t::onednn_blocked;
          }
          else {
            return matmul_algo_t::aocl_dlp_blocked;
          }
        }
        else {
          if (K <= 576) {
            return matmul_algo_t::onednn_blocked;
          }
          else {
            return matmul_algo_t::aocl_dlp_blocked;
          }
        }
      }
    }
  }
  else {
    if (K <= 5120) {
      if (K <= 1152) {
        if (N <= 1012) {
          return matmul_algo_t::aocl_dlp_blocked;
        }
        else {
          if (N <= 1152) {
            return matmul_algo_t::onednn_blocked;
          }
          else {
            return matmul_algo_t::aocl_dlp_blocked;
          }
        }
      }
      else {
        if (K <= 1792) {
          if (N <= 67122) {
            return matmul_algo_t::onednn_blocked;
          }
          else {
            return matmul_algo_t::aocl_dlp_blocked;
          }
        }
        else {
          // TODO: Consider adding K-based threshold check here (e.g., K <= 2816)
          // if different kernels are needed for different K ranges in the future
          return matmul_algo_t::aocl_dlp_blocked;
        }
      }
    }
    else {
      return matmul_algo_t::aocl_dlp_blocked;
    }
  }
}

unsigned int get_auto_tuner_ver() {
  char *auto_tuner_version_env = std::getenv("ZENDNNL_AUTO_TUNER_TYPE");
  if (auto_tuner_version_env) {
    unsigned int version = std::stoi(auto_tuner_version_env);
    // Current support is with two versions.
    if (version == 0 || version > 2) {
      return 1;
    }
    return version;
  }
  // return version 1 as default
  return 1;
}

matmul_algo_t kernel_select(matmul_params &params, int Batch_A, int Batch_B,
                            int batch_count, int M, int N, int K, int num_threads, const void *bias,
                            const bool is_weights_const) {

  matmul_config_t &matmul_config = matmul_config_t::instance();
  int32_t algo;
  if (batch_count > 1) {
    // Get BMM algo for batched matrix multiplication
    algo = params.lowoha_algo == matmul_algo_t::none ?
           (matmul_config.get_bmm_algo() == static_cast<int>(matmul_algo_t::none) ?
            static_cast<int>(matmul_algo_t::aocl_dlp) : matmul_config.get_bmm_algo()) :
           static_cast<int>(params.lowoha_algo);
  }
  else {
    // Get regular matmul algo for single matrix multiplication
    algo = params.lowoha_algo == matmul_algo_t::none ?
           matmul_config.get_algo() : static_cast<int>(params.lowoha_algo);
  }

  // Default to AOCL DLP blocked kernel
  matmul_algo_t kernel = (algo == static_cast<int>(matmul_algo_t::none)) ?
                         matmul_algo_t::aocl_dlp_blocked : static_cast<matmul_algo_t>(algo);
  bool is_woq = (params.dtypes.src == data_type_t::bf16) &&
                (params.dtypes.wei == data_type_t::s4 || params.dtypes.wei == data_type_t::u4);
  bool is_f16 = (params.dtypes.src == data_type_t::f16) &&
                (params.dtypes.wei == data_type_t::f16) &&
                (params.dtypes.dst == data_type_t::f16 ||
                 params.dtypes.dst == data_type_t::f32);

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
      if (params.dtypes.wei == data_type_t::bf16) {
        kernel = select_algo_by_heuristics_bf16_bmm(batch_count, M, N, K, num_threads);
      }
      else {
        kernel = matmul_algo_t::aocl_dlp;
      }
    }
    else if (is_woq) {
      kernel = select_algo_by_heuristics_woq_int4_mm(M, N, K, num_threads);
    }
    else {
      if (is_weights_const == false && M >= 4096 && M <= 8192 && K == 1024 &&
          N == 1024) {
        kernel = matmul_algo_t::libxsmm_blocked;
      }
      else {
        if (params.dtypes.wei == data_type_t::bf16) {
          if (M <= 8) {
            // M=1: BKC GEMV with flat kernel + CCX scheduling.
            // M=2-8: BRGEMM with MR=M (decode) or MR=8, with
            // b_exceeds_l2 exemption. Both dominate DLP at high
            // thread counts (up to +50% at 128t on MoE shapes).
            kernel = native::bf16_gemv_best_algo(M, N, K, num_threads);
          }
          else {
            kernel = select_algo_by_heuristics_bf16_mm(M, N, K);
          }
        }
        else if (params.dtypes.wei == data_type_t::s8 && M == 1
                 && num_threads == 1) {
          // INT8 GEMV: BKC kernel for shapes where packed B fits in L2.
          // For large B near L2 capacity (>500KB) with N>256, DLP wins
          // due to two-block dispatch overhead. Route those to DLP.
          const int kp = (K + 3) & ~3;
          const int np = ((N + native::BKC_NR_PAD - 1) / native::BKC_NR_PAD) * native::BKC_NR_PAD;
          const size_t b_packed = static_cast<size_t>(kp) * np;
          static const size_t l2 =
              static_cast<size_t>(native::detect_uarch().l2_bytes);
          if (b_packed <= l2 && !(b_packed > 500*1024 && N > 256))
            kernel = matmul_algo_t::native_brgemm;
          else
            kernel = matmul_algo_t::aocl_dlp_blocked;
        }
        else {
          kernel = matmul_algo_t::aocl_dlp_blocked;
        }
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

  // Force aocl_dlp or aocl_dlp_blocked for WOQ (Weight-Only Quantization) cases
  if (is_woq && kernel != matmul_algo_t::aocl_dlp &&
      kernel != matmul_algo_t::aocl_dlp_blocked) {
    kernel = matmul_algo_t::aocl_dlp_blocked;
    log_info("WOQ detected, switching to DLP kernel");
  }

  size_t src_scale_nelems = 1;
  for (auto d : params.quant_params.src_scale.dims) {
    src_scale_nelems *= static_cast<size_t>(d);
  }
  const bool is_sym_quant = params.dtypes.src == data_type_t::s8 &&
                            params.dtypes.wei == data_type_t::s8 &&
                            !params.quant_params.src_zp.buff &&
                            src_scale_nelems > 1 &&
                            (params.dtypes.dst == data_type_t::f32 ||
                             params.dtypes.dst == data_type_t::bf16);

  if (is_sym_quant && kernel != matmul_algo_t::aocl_dlp_blocked) {
    kernel = matmul_algo_t::aocl_dlp_blocked;
  }

  const bool is_non_qunat_int8 = (params.dtypes.src == data_type_t::bf16 ||
                                  params.dtypes.src == data_type_t::f32) &&
                                 (params.dtypes.wei == data_type_t::s8);

  if (is_non_qunat_int8 && kernel != matmul_algo_t::aocl_dlp_blocked) {
    log_info("Non-quantized INT8 detected, switching to aocl_dlp kernel");
    kernel = matmul_algo_t::aocl_dlp;
  }

  // TODO: Remove this workaround once OneDNN fixes the GEMV M=1 + beta!=0 case.
  if (M == 1 && (kernel == matmul_algo_t::onednn ||
                 kernel == matmul_algo_t::onednn_blocked) &&
      params.dtypes.src == data_type_t::f32) {
    log_info("M=1 and src is F32, switching to aocl_dlp_blocked kernel");
    kernel = matmul_algo_t::aocl_dlp_blocked;
  }

  // TODO: Update the conditon once prepack supports other formats
  // Current prepack supports only AOCL blocked kernel
  if (params.mem_format_b == 'r') {
    kernel = matmul_algo_t::aocl_dlp_blocked;
  }

  // F16 kernel selection
  // oneDNN supports: src=f16, wei=f16, dst=f16 or f32
  // aocl supports: src=f16, wei=f16, dst=f16
  if (is_f16) {
    bool is_onednn_algo = (kernel == matmul_algo_t::onednn ||
                           kernel == matmul_algo_t::onednn_blocked);
    bool is_aocl_algo = (kernel == matmul_algo_t::aocl_dlp ||
                         kernel == matmul_algo_t::aocl_dlp_blocked);
    bool aocl_compatible = is_aocl_algo && params.dtypes.dst == data_type_t::f16 &&
                           bias == nullptr && params.postop_.empty();
    if (!is_onednn_algo && !aocl_compatible) {
      log_info("Switching to onednn_blocked kernel for F16 GEMM");
      kernel = matmul_algo_t::onednn_blocked;
    }
  }

  params.lowoha_algo = kernel;

  bool is_aocl = (kernel == matmul_algo_t::aocl_dlp ||
                  kernel == matmul_algo_t::aocl_dlp_blocked);
  matmul_config.set_accum_type(
    (is_aocl && is_f16) ? data_type_t::f16
    : data_type_t::f32);

  return kernel;
}

bool should_use_mm_partitioner(const matmul_algo_t &kernel) {
  matmul_config_t &matmul_config = matmul_config_t::instance();
  //Facing Segmentation Fault issue with libxsmm kernel, using partitioner for now.
  auto is_libxsmm = (kernel == matmul_algo_t::libxsmm ||
                     kernel == matmul_algo_t::libxsmm_blocked);
  return matmul_config.get_mm_partitioner_enabled() || is_libxsmm;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
