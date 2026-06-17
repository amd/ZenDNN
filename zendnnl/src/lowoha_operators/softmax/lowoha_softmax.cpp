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

#include "lowoha_softmax.hpp"
#include "lowoha_operators/softmax/onednn_kernel.hpp"
#include "lowoha_operators/softmax/reference_kernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace softmax {

status_t softmax_kernel_wrapper(
  const void *input,
  void *output,
  softmax_params &params
) {
  // Select algorithm if not specified
  if (params.algorithm == softmax_algo_t::none) {
#if ZENDNNL_DEPENDS_ONEDNN
    params.algorithm = softmax_algo_t::onednn;
#else
    params.algorithm = softmax_algo_t::reference;
#endif
  }

#if ZENDNNL_DEPENDS_ONEDNN
  if (params.algorithm == softmax_algo_t::onednn) {
    log_info("Using OneDNN kernel for Softmax");
    return softmax_onednn_wrapper(input, output, params);
  }
#endif

  // Fallback to reference implementation (always reached if OneDNN not selected)
  log_info("Using reference kernel for Softmax");
  return softmax_reference_wrapper(input, output, params);
}

status_t softmax_direct(
  const void *input,
  void *output,
  softmax_params &params
) {
  // Create profiler instance for timing
  zendnnl::profile::profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile) {
    profiler.tbp_start();
  }

  // F16 requires AVX512-FP16; reject up-front on unsupported hosts to
  // avoid undefined behavior in kernels that touch F16 storage.
  const bool is_f16 = (params.src_dt == data_type_t::f16 ||
                       params.dst_dt == data_type_t::f16);
  if (is_f16 && !zendnnl_platform_info().get_avx512_f16_status()) {
    log_error("F16 data type is not supported on this platform "
              "(requires AVX512-FP16).");
    return status_t::isa_unsupported;
  }

  // Validate inputs
  if (validate_softmax_inputs(input, output, params)
      != status_t::success) {
    return status_t::failure;
  }

  // Log API call
  [[maybe_unused]] std::ostringstream ss;
  if (apilog_info_enabled() || is_profile) {
    ss << "LOWOHA softmax_direct: batch=" << params.batch
       << ", axis_dim=" << params.axis_dim
       << ", axis=" << params.axis
       << ", log_softmax=" << (params.log_softmax ? "true" : "false")
       << ", softmin=" << (params.softmin ? "true" : "false")
       << ", src_dt=" << static_cast<int>(params.src_dt)
       << ", dst_dt=" << static_cast<int>(params.dst_dt);
  }
  apilog_info(ss.str());

  // Execute softmax; propagate the backend outcome so callers are not
  // told the op succeeded when the kernel actually failed.
  status_t status = softmax_kernel_wrapper(input, output, params);

  if (is_profile) {
    profiler.tbp_stop();
    profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(),
                       profiler.get_res_str());
  }

  return status;
}

} // namespace softmax
} // namespace lowoha
} // namespace zendnnl
