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
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"

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
  // Compute nelems from shape
  const size_t nelems = static_cast<size_t>(params.nelems());

  const int32_t omp_mt = thread_guard::max_threads();
  params.num_threads = resolve_num_threads(params.num_threads, omp_mt);
  thread_guard tg(params.num_threads, omp_mt);

  //============================================================================
  // Dynamic Quantization Mode
  //============================================================================
  if (params.dynamic_quant) {
    // Validate dynamic quantization parameters and shape
    status_t dq_status = op_instrumentation::validate([&]() {
      if (validate_dynamic_quant_params(src, params) != status_t::success)
        return status_t::failure;
      return validate_reorder_shape(params);
    });
    if (dq_status != status_t::success) return dq_status; // validation failed

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
          std::string dq_log = "LOWOHA reorder_direct (Dynamic_Quantize): " + dq_params_str;
          profiler.tbp_stop();
          profilelog_verbose(dq_log,
                             ", kernel=native (fused per-token), time=",
                             profiler.tbp_elapsedtime(),
                             profiler.get_res_str());
        }
        return status_t::success;
      } else if (dq_algo_override == 2 &&
          dispatch_unfused_per_token(src, dst, params, params.M(), params.N())) {
        if (is_profile) {
          std::string dq_log = "LOWOHA reorder_direct (Dynamic_Quantize): " + dq_params_str;
          profiler.tbp_stop();
          profilelog_verbose(dq_log,
                             ", kernel=native (unfused per-token), time=",
                             profiler.tbp_elapsedtime(),
                             profiler.get_res_str());
        }
        return status_t::success;
      } else if (dq_algo_override == 3 &&
          dispatch_fused_per_token_ref(src, dst, params, params.M(), params.N())) {
        if (is_profile) {
          std::string dq_log = "LOWOHA reorder_direct (Dynamic_Quantize): " + dq_params_str;
          profiler.tbp_stop();
          profilelog_verbose(dq_log,
                             ", kernel=reference (fused per-token), time=",
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
      std::string dq_log = "LOWOHA reorder_direct (Dynamic_Compute): " + dq_params_str;
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
  if (val_status != status_t::success) return val_status;

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
        if (i > 0) ss << ", ";
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

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
