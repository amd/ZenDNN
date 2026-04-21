/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_normalization.hpp"
#include "lowoha_operators/normalization/kernel/reference_kernel.hpp"
#include "lowoha_operators/normalization/kernel/rmsnorm_avx512_kernel.hpp"
#include "lowoha_operators/normalization/kernel/layernorm_avx512_kernel.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"

namespace zendnnl {
namespace lowoha {
namespace normalization {

using namespace zendnnl::common;

status_t normalization_kernel_wrapper(
  const void *input,
  void *output,
  const void *gamma,
  const void *beta,
  const void *running_mean,
  const void *running_var,
  void       *residual,
  norm_params &params
) {

  const bool has_avx512f = zendnnl_platform_info().get_avx512f_status();

  if (has_avx512f && (params.norm_type == norm_type_t::RMS_NORM ||
                      params.norm_type == norm_type_t::FUSED_ADD_RMS_NORM)) {
    log_info("Using AVX512 kernel for ", norm_type_to_str(params.norm_type));

    status_t status = rms_norm_avx512(input, output, residual, gamma, params);
    if (status != status_t::success) {
      log_error(norm_type_to_str(params.norm_type), " kernel failed");
    }
    return status;
  }

  if (has_avx512f && params.norm_type == norm_type_t::LAYER_NORM) {
    log_info("Using AVX512 kernel for ", norm_type_to_str(params.norm_type));

    status_t status = layer_norm_avx512(input, output, gamma, beta, params);
    if (status != status_t::success) {
      log_error(norm_type_to_str(params.norm_type), " kernel failed");
    }
    return status;
  }

  log_info("Using reference kernel for ", norm_type_to_str(params.norm_type));
  status_t status = normalization_reference_wrapper(
                      input, output, gamma, beta,
                      running_mean, running_var, residual, params);

  if (status != status_t::success) {
    log_error("Normalization kernel failed for ",
              norm_type_to_str(params.norm_type));
  }

  return status;
}

status_t normalization_direct(
  const void *input,
  void *output,
  const void *gamma,
  const void *beta,
  const void *running_mean,
  const void *running_var,
  void       *residual,
  norm_params &params
) {
  // Create profiler instance for timing
  zendnnl::profile::profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile) {
    profiler.tbp_start();
  }

  // Validate inputs only when ZENDNNL_DIAGNOSTICS_ENABLE=1. In production this
  // resolves to a single predicted-not-taken branch, skipping the full
  // validation path (null-pointer checks, dimension checks, and
  // quantization-parameter validation).
  status_t status = zendnnl::common::op_instrumentation::validate([&]() {
    return validate_normalization_inputs(input, output, gamma, beta,
                                         running_mean, running_var, residual, params);
  });
  if (status != status_t::success) {
    return status;
  }

  // Execute normalization
  status_t kernel_status = normalization_kernel_wrapper(
                             input, output, gamma, beta,
                             running_mean, running_var, residual, params);

  if (is_profile) {
    profiler.tbp_stop();
  }

  if (kernel_status != status_t::success) {
    return kernel_status;
  }

  if (apilog_info_enabled() || is_profile) {
    [[maybe_unused]] std::ostringstream ss;
    ss << "LOWOHA " << norm_type_to_str(params.norm_type)
       << ": batch=" << params.batch
       << ", norm_size=" << params.norm_size
       << ", num_channels=" << params.num_channels
       << ", epsilon=" << params.epsilon
       << ", use_scale=" << (params.use_scale ? "true" : "false")
       << ", use_shift=" << (params.use_shift ? "true" : "false")
       << ", src_dt=" << dtype_info(params.src_dt)
       << ", dst_dt=" << dtype_info(params.dst_dt);

    apilog_info(ss.str());
    if (is_profile) {
      profilelog_verbose(ss.str(), ", time=",
                         profiler.tbp_elapsedtime(),
                         profiler.get_res_str());
    }
  }

  return status_t::success;
}

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

