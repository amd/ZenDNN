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

#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_utils.hpp"
#include "lowoha_operators/reorder/reorder_data_type/reorder_dtype_dispatch.hpp"
#include "lowoha_operators/reorder/reorder_data_type/static_quant_dequant_impl/static_kernels.hpp"
#include "lowoha_operators/reorder/reorder_data_type/scalar_impl/scalar_kernels.hpp"
#include "lowoha_operators/reorder/reorder_data_type/dynamic_quant_impl/dynamic_kernels.hpp"
#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"

#include <algorithm>
#include <limits>
#include <sstream>

namespace zendnnl {
namespace lowoha {
namespace reorder {

using namespace zendnnl::error_handling;
using namespace zendnnl::profile;
using zendnnl::lowoha::matmul::zendnnl_parallel_for;
using zendnnl::common::op_instrumentation;

status_t reorder_direct(const void *src, void *dst,
                        reorder_params_t &params) {
  //============================================================================
  // Check if this is a weight-prepack request
  // When params.is_prepack is true, delegate to the prepack pipeline
  // (the prepack-specific fields live in params.prepack).
  //============================================================================
  if (params.is_prepack) {
    return weight_prepack_into(src, params, dst);
  }

  // Compute nelems from shape
  const size_t nelems = static_cast<size_t>(params.nelems());

  //============================================================================
  // Dynamic Quantization Mode
  //============================================================================
  if (params.dynamic_quant) {
    // Validate dynamic quantization parameters and shape
    status_t dq_status = op_instrumentation::validate([&]() {
      if (validate_dynamic_quant_params(src, params) != status_t::success) {
        return status_t::failure;
      }
      return validate_reorder_shape(params);
    });
    if (dq_status != status_t::success) {
      return dq_status;  // validation failed
    }

    // Create profiler instance for timing
    profiler_t profiler;
    bool is_profile = is_profile_enabled();

    // Determine quantization mode based on zp buffer presence
    const bool is_symmetric = (params.quant_params.zero_point.buff == nullptr);

    // Build log string for API and profile logging
    [[maybe_unused]] std::string dq_params_str;
    if (apilog_info_enabled() || is_profile) {
      std::ostringstream ss_tmp;
      ss_tmp << "M=" << params.M()
             << ", K=" << params.N()
             << ", mode=" << (is_symmetric ? "symmetric" : "asymmetric")
             << ", src_dtype=" << reorder_data_type_to_string(params.src_dtype)
             << ", dst_dtype=" << reorder_data_type_to_string(params.dst_dtype)
             << ", granularity=" << granularity_to_string(
               get_single_granularity(params.quant_params.scale.dims, params.src_shape))
             << ", dst=" << (dst == nullptr ? "nullptr (compute only)" : "valid");
      dq_params_str = ss_tmp.str();

      if (apilog_info_enabled()) {
        apilog_info("LOWOHA reorder_direct: " + dq_params_str);
      }
    }

    // Start profiling timer
    if (is_profile) {
      profiler.tbp_start();
    }

    //------------------------------------------------------------------
    // Fused per-token path: when granularity is per-channel-row (M,1)
    // and dst is provided, use fused kernel that computes scale/zp and
    // quantizes in one cache-friendly pass per row.
    //
    // ZENDNNL_DYNAMIC_QUANT_ALGO overrides:
    //   0 (or unset) = default behavior (respects API algo selection:
    //                   native -> vector fused, reference -> scalar unfused)
    //   1 = vector fused,   2 = vector unfused,
    //   3 = scalar fused,   4 = scalar unfused
    //------------------------------------------------------------------
    const auto &scale_dims_dq = params.quant_params.scale.dims;
    reorder_algo_t algo_dq = select_reorder_algo(params, nelems);

    static const int32_t dq_algo_override = get_dynamic_quant_algo_override();

    if (dst != nullptr && params.is_2d() &&
        (!params.has_src_strides() || params.is_src_contiguous()) &&
        is_per_channel_row_dims(scale_dims_dq, params.src_shape)) {

      if (((dq_algo_override == 0 && algo_dq == reorder_algo_t::native) ||
           dq_algo_override == 1) &&
          dispatch_fused_per_token(src, dst, params, params.M(), params.N())) {
        if (is_profile) {
          std::string dq_log = "LOWOHA reorder_direct (Dynamic_Quantize): " +
                               dq_params_str;
          profiler.tbp_stop();
          profilelog_verbose(dq_log,
                             ", kernel=native (fused per-token), time=",
                             profiler.tbp_elapsedtime(),
                             profiler.get_res_str());
        }
        return status_t::success;
      }
      else if (dq_algo_override == 2 &&
               dispatch_unfused_per_token(src, dst, params, params.M(), params.N())) {
        if (is_profile) {
          std::string dq_log = "LOWOHA reorder_direct (Dynamic_Quantize): " +
                               dq_params_str;
          profiler.tbp_stop();
          profilelog_verbose(dq_log,
                             ", kernel=native (unfused per-token), time=",
                             profiler.tbp_elapsedtime(),
                             profiler.get_res_str());
        }
        return status_t::success;
      }
      else if (dq_algo_override == 3 &&
               dispatch_fused_per_token_ref(src, dst, params, params.M(), params.N())) {
        if (is_profile) {
          std::string dq_log = "LOWOHA reorder_direct (Dynamic_Quantize): " +
                               dq_params_str;
          profiler.tbp_stop();
          profilelog_verbose(dq_log,
                             ", kernel=reference (fused per-token), time=",
                             profiler.tbp_elapsedtime(),
                             profiler.get_res_str());
        }
        return status_t::success;
      }
    }

    if (dst != nullptr && params.is_2d() &&
        (!params.has_src_strides() || params.is_src_contiguous()) &&
        is_per_group_col_dims(scale_dims_dq, params.src_shape)) {

      if (((dq_algo_override == 0 && algo_dq == reorder_algo_t::native) ||
           dq_algo_override == 1) &&
          dispatch_fused_per_group(src, dst, params, params.M(), params.N())) {
        if (is_profile) {
          std::string dq_log = "LOWOHA reorder_direct (Dynamic_Quantize): " +
                               dq_params_str;
          profiler.tbp_stop();
          profilelog_verbose(dq_log,
                             ", kernel=native (fused per-group), time=",
                             profiler.tbp_elapsedtime(),
                             profiler.get_res_str());
        }
        return status_t::success;
      }
    }

    // Compute dynamic quantization parameters (scale and zero_point)
    status_t status = compute_dynamic_quant_params(src, params);
    if (status != status_t::success) {
      return status;
    }

    if (is_profile) {
      std::string dq_log = "LOWOHA reorder_direct (Dynamic_Compute): " +
                           dq_params_str;
      profiler.tbp_stop();
      profilelog_verbose(dq_log, ", time=", profiler.tbp_elapsedtime(),
                         profiler.get_res_str());
    }
    // If dst is nullptr, only compute scale/zp and return
    if (dst == nullptr) {
      return status_t::success;
    }

    // Fall through to standard path with computed parameters
  }

  //============================================================================
  // Standard Reorder Mode (static quantization parameters)
  //============================================================================

  // Validate inputs and parameters
  status_t val_status = op_instrumentation::validate([&]() {
    return validate_reorder_inputs(src, dst, nelems, params);
  });
  if (val_status != status_t::success) {
    return val_status;
  }

  // Select algorithm
  reorder_algo_t algo = select_reorder_algo(params, nelems);

  // Create profiler instance for timing
  profiler_t profiler;
  bool is_profile = is_profile_enabled();

  // Build log string for API and profile logging
  [[maybe_unused]] std::string log_str;
  if (apilog_info_enabled() || is_profile) {
    std::ostringstream ss;
    float scale_val = get_scale_value(params.quant_params.scale);
    int zp_val = get_zero_point_value(params.quant_params.zero_point);
    ss << "LOWOHA reorder_direct (Static_Quantize): M=" << params.M()
       << ", K=" << params.N()
       << ", src_dtype=" << reorder_data_type_to_string(params.src_dtype)
       << ", dst_dtype=" << reorder_data_type_to_string(params.dst_dtype)
       << ", scale=" << scale_val
       << ", zero_point=" << zp_val
       << ", kernel=" << reorder_algo_to_string(algo)
       << ", granularity=" << granularity_to_string(
         (params.quant_params.zero_point.buff == nullptr)
         ? get_single_granularity(params.quant_params.scale.dims, params.src_shape)
         : get_granularity_type(params));

    // Add stride information to log
    if (params.has_src_strides()) {
      ss << ", strides=[";
      for (size_t i = 0; i < params.src_strides.size(); ++i) {
        if (i > 0) {
          ss << ", ";
        }
        ss << params.src_strides[i];
      }
      ss << "]";
    }

    log_str = ss.str();
    if (apilog_info_enabled()) {
      apilog_info(log_str);
    }
  }

  // Start profiling timer
  if (is_profile) {
    profiler.tbp_start();
  }

  // Execute reorder (handles all cases: contiguous, 1D/2D/3D strided)
  reorder_wrapper(src, dst, nelems, params, algo);

  // Stop profiling timer and log
  if (is_profile) {
    profiler.tbp_stop();
    profilelog_verbose(log_str, ", time=", profiler.tbp_elapsedtime(),
                       profiler.get_res_str());
  }

  return status_t::success;
}

status_t group_dynamic_quant(
    const std::vector<const void *> &src,
    const std::vector<int> &M,
    const std::vector<int> &K,
    const std::vector<std::vector<int64_t>> &src_strides,
    const std::vector<void *> &dst,
    const std::vector<std::vector<int64_t>> &dst_strides,
    const std::vector<void *> &scale,
    const group_dynamic_quant_params_t &params) {
  std::vector<int> src_row_stride;
  std::vector<int> dst_row_stride;

  const size_t num_ops = M.size();
  if (num_ops == 0) {
    log_error("group_dynamic_quant: M vector is empty");
    return status_t::failure;
  }
  if (src.size() != num_ops || K.size() != num_ops ||
      dst.size() != num_ops ||
      (!src_strides.empty() && src_strides.size() != num_ops) ||
      (!dst_strides.empty() && dst_strides.size() != num_ops) ||
      scale.size() != num_ops) {
    log_error("group_dynamic_quant: vector size mismatch");
    return status_t::failure;
  }
  if (params.dst_dtype != data_type_t::s8) {
    log_error("group_dynamic_quant: only s8 destination is supported");
    return status_t::failure;
  }
  if (params.src_dtype != data_type_t::bf16 &&
      params.src_dtype != data_type_t::f32) {
    log_error("group_dynamic_quant: only bf16/f32 source is supported");
    return status_t::failure;
  }
  if (params.scale_dtype != data_type_t::f32 &&
      params.scale_dtype != data_type_t::bf16) {
    log_error("group_dynamic_quant: scale dtype must be f32 or bf16");
    return status_t::failure;
  }
  src_row_stride.resize(num_ops);
  dst_row_stride.resize(num_ops);
  for (size_t i = 0; i < num_ops; ++i) {
    int64_t src_stride_m = K[i];
    if (!src_strides.empty() && !src_strides[i].empty()) {
      if (src_strides[i].size() != 2 || src_strides[i][1] != 1) {
        log_error("group_dynamic_quant: src_strides[", i,
                  "] must be empty or {row_stride, 1}");
        return status_t::failure;
      }
      src_stride_m = src_strides[i][0];
    }
    int64_t dst_stride_m = K[i];
    if (!dst_strides.empty() && !dst_strides[i].empty()) {
      if (dst_strides[i].size() != 2 || dst_strides[i][1] != 1) {
        log_error("group_dynamic_quant: dst_strides[", i,
                  "] must be empty or {row_stride, 1}");
        return status_t::failure;
      }
      dst_stride_m = dst_strides[i][0];
    }

    if (M[i] < 0 || K[i] <= 0 || src_stride_m < K[i] ||
        dst_stride_m < K[i] ||
        src_stride_m > std::numeric_limits<int>::max() ||
        dst_stride_m > std::numeric_limits<int>::max()) {
      log_error("group_dynamic_quant: invalid shape/stride at op ", i,
                " (M=", M[i], ", K=", K[i],
                ", src_row_stride=", src_stride_m,
                ", dst_row_stride=", dst_stride_m, ")");
      return status_t::failure;
    }
    src_row_stride[i] = static_cast<int>(src_stride_m);
    dst_row_stride[i] = static_cast<int>(dst_stride_m);
    if (M[i] == 0) continue;
    if (src[i] == nullptr || dst[i] == nullptr || scale[i] == nullptr) {
      log_error("group_dynamic_quant: null active buffer at op ", i);
      return status_t::failure;
    }
  }

  int64_t total_rows = 0;
  for (int m : M) total_rows += std::max(0, m);
  if (total_rows == 0) return status_t::success;

  if (apilog_info_enabled()) {
    apilog_info("LOWOHA group_dynamic_quant: num_ops=", M.size(),
                ", total_rows=", total_rows,
                ", src_dtype=", reorder_data_type_to_string(params.src_dtype),
                ", dst_dtype=", reorder_data_type_to_string(params.dst_dtype),
                ", scale_dtype=",
                reorder_data_type_to_string(params.scale_dtype),
                ", granularity=per_token");
  }

  if (!dispatch_group_dynamic_per_token(
          src, M, K, src_row_stride, dst, dst_row_stride, scale, params)) {
    log_error("group_dynamic_quant: no native grouped per-token kernel matched");
    return status_t::failure;
  }

  return status_t::success;
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
