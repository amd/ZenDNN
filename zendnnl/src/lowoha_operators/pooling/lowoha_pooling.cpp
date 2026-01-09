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

#include "lowoha_pooling.hpp"
#include "lowoha_operators/pooling/onednn_kernel.hpp"
#include "lowoha_operators/pooling/reference_kernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace pooling {

void pooling_kernel_wrapper(
    const void *input,
    void *output,
    pool_params &params
) {
#if ZENDNNL_DEPENDS_ONEDNN
    if (params.algo == pooling_algo_t::onednn) {
        log_info("Using OneDNN kernel for Pooling");
        pooling_onednn_wrapper(input, output, params);
        return;
    }
#endif

    // Fallback to reference implementation
    if (params.algo == pooling_algo_t::reference) {
        log_info("Using reference kernel for Pooling");
        pooling_reference_wrapper(input, output, params);
        return;
    }

    log_error("Pooling: No suitable backend available");
}

status_t pooling_direct(
    const void *input,
    void *output,
    pool_params &params
) {
    // Create profiler instance for timing
    zendnnl::profile::profiler_t profiler;
    bool is_profile = is_profile_enabled();
    if (is_profile) {
        profiler.tbp_start();
    }

    // Validate inputs
    if (validate_pooling_inputs(input, output, params)
        != status_t::success) {
        return status_t::failure;
    }

    // Log API call
    [[maybe_unused]] std::ostringstream ss;
    if (apilog_info_enabled() || is_profile) {
        ss << "LOWOHA pooling_direct: "
           << (params.is_max_pooling ? "max_pooling" : "avg_pooling")
           << ", batch=" << params.dims.batch
           << ", in_h=" << params.dims.in_height << ", in_w=" << params.dims.in_width
           << ", channels=" << params.dims.channels
           << ", out_h=" << params.dims.out_height << ", out_w=" << params.dims.out_width
           << ", kernel_h=" << params.dims.kernel_height << ", kernel_w=" << params.dims.kernel_width
           << ", stride_h=" << params.stride_h << ", stride_w=" << params.stride_w
           << ", pad_t=" << params.pad_top << ", pad_l=" << params.pad_left
           << ", pad_b=" << params.pad_bottom << ", pad_r=" << params.pad_right
           << ", data_format=" << params.data_format;
    }
    apilog_info(ss.str());
    
    // Execute pooling
    pooling_kernel_wrapper(input, output, params);

    if (is_profile) {
        profiler.tbp_stop();
        profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(), profiler.get_res_str());
    }

    return status_t::success;
}

} // namespace pooling
} // namespace lowoha
} // namespace zendnnl
