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
#include "lowoha_operators/normalization/reference_kernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace normalization {

status_t setup_normalization_shape(norm_params &params);

status_t normalization_kernel_wrapper(
  const void *input,
  void *output,
  const void *gamma,
  const void *beta,
  const void *running_mean,
  const void *running_var,
  norm_params &params
) {
  // Select algorithm if not specified
  if (params.algorithm == norm_algo_t::none) {
    params.algorithm = norm_algo_t::reference;
  }

  // TODO: Add optimized (AVX512 / OneDNN) backends here.
  //       For now, always fall through to the reference implementation.

  log_info("Using reference kernel for ", norm_type_to_str(params.norm_type));
  status_t status = normalization_reference_wrapper(
                      input, output, gamma, beta,
                      running_mean, running_var, params);

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
  norm_params &params
) {
  // Create profiler instance for timing
  zendnnl::profile::profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile) {
    profiler.tbp_start();
  }

  // Derive batch, norm_size, num_channels from shape
  if (setup_normalization_shape(params) != status_t::success) {
    return status_t::failure;
  }

  // Validate inputs
  if (validate_normalization_inputs(input, output, gamma, beta,
                                    running_mean, running_var, params)
      != status_t::success) {
    return status_t::failure;
  }

  // Execute normalization
  status_t kernel_status = normalization_kernel_wrapper(
                             input, output, gamma, beta,
                             running_mean, running_var, params);

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
       << ", src_dt=" << static_cast<int>(params.src_dt)
       << ", dst_dt=" << static_cast<int>(params.dst_dt);

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

