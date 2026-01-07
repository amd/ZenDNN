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

#include "lowoha_conv.hpp"
#include "lowoha_operators/conv/onednn_kernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace conv {

void conv_kernel_wrapper(
    const void *input,
    const void *filter,
    const void *bias,
    void *output,
    conv_params &params
) {
#if ZENDNNL_DEPENDS_ONEDNN
    if (params.algo == conv_algo_t::onednn ||
        params.algo == conv_algo_t::onednn_blocked) {
        log_info("Using OneDNN kernel for Conv");
        conv_onednn_wrapper(input, filter, bias, output,
                             params);
        return;
    }
#endif

    // Fallback to reference implementation (TODO)
    log_error("Conv: No suitable backend available");
}

status_t conv_direct(
    const void *input,
    const void *filter,
    const void *bias,
    void *output,
    conv_params &params
) {
    const conv_dims_t &dims = params.dims;

    // Create profiler instance for timing
    zendnnl::profile::profiler_t profiler;
    bool is_profile = is_profile_enabled();
    if (is_profile) {
        profiler.tbp_start();
    }

    // Validate inputs
    if (validate_conv_inputs(input, filter, output, params)
        != status_t::success) {
        return status_t::failure;
    }

    // Log API call
    [[maybe_unused]] std::ostringstream ss;
    if (apilog_info_enabled() || is_profile) {
        ss << "LOWOHA conv_direct: batch=" << dims.batch
           << ", in_h=" << dims.in_height << ", in_w=" << dims.in_width
           << ", in_c=" << dims.in_channels
           << ", out_h=" << dims.out_height << ", out_w=" << dims.out_width
           << ", out_c=" << dims.out_channels
           << ", filter_h=" << dims.filter_height << ", filter_w=" << dims.filter_width
           << ", stride_h=" << params.stride_h << ", stride_w=" << params.stride_w
           << ", pad_t=" << params.pad_top << ", pad_l=" << params.pad_left
           << ", pad_b=" << params.pad_bottom << ", pad_r=" << params.pad_right
           << ", dilation_h=" << params.dilation_h << ", dilation_w=" << params.dilation_w
           << ", bias=" << (bias != nullptr ? "true" : "false");
    }
    apilog_info(ss.str());

    // Select kernel algorithm
    // TODO: Remove this once we have a proper kernel selection logic
    params.algo = conv_algo_t::onednn;
    if (params.algo == conv_algo_t::none) {
        // Default to OneDNN if available
#if ZENDNNL_DEPENDS_ONEDNN
        params.algo = conv_algo_t::onednn;
#else
        params.algo = conv_algo_t::reference;
#endif
    }

    // Execute convolution
    conv_kernel_wrapper(input, filter, bias, output, params);

    if (is_profile) {
        profiler.tbp_stop();
        profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(), profiler.get_res_str());
    }

    return status_t::success;
}

} // namespace conv
} // namespace lowoha
} // namespace zendnnl
