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

status_t softmax_direct(
        const void *input, void *output, softmax_params &params) {
    // Create profiler instance for timing
    zendnnl::profile::profiler_t profiler;
    bool is_profile = is_profile_enabled();
    if (is_profile) { profiler.tbp_start(); }

    // Resolve the algorithm up front so the ISA gate below can be limited to
    // the OneDNN backend.
    const softmax_algo_t algo = algo_select(params);

    // F16 requires AVX512-FP16; reject up-front on unsupported hosts to avoid
    // undefined behavior in kernels that touch F16 storage. This gate is
    // specific to the OneDNN backend, which touches F16 storage directly. The
    // reference kernel computes in FP32 and converts f16 storage in software,
    // so it never needs AVX512-FP16 and is not gated here. No separate
    // ZENDNNL_DEPENDS_ONEDNN guard is needed: algo_select() has already
    // resolved algo against the build configuration, so algo == onednn can
    // only occur when OneDNN support is actually compiled in. Unsupported
    // algorithms fall through to the switch below and return failure.
    const bool is_f16 = (params.src_dt == data_type_t::f16
            || params.dst_dt == data_type_t::f16);
    if (is_f16 && algo == softmax_algo_t::onednn
            && !zendnnl_platform_info().get_avx512_f16_status()) {
        log_error(
                "F16 data type is not supported on this platform "
                "(requires AVX512-FP16).");
        return status_t::isa_unsupported;
    }

    // Validate inputs
    if (validate_softmax_inputs(input, output, params) != status_t::success) {
        return status_t::failure;
    }

    // Execute softmax; propagate the backend outcome so callers are not
    // told the op succeeded when the kernel actually failed. none was
    // normalized by algo_select, so it never reaches the switch below.
    // The onednn case needs no ZENDNNL_DEPENDS_ONEDNN guard either: algo_select()
    // already restricts algo == onednn to builds where OneDNN is compiled in.
    status_t status = status_t::failure;
    switch (algo) {
        case softmax_algo_t::onednn:
            log_info("Using OneDNN kernel for Softmax");
            status = softmax_onednn_wrapper(input, output, params);
            break;
        case softmax_algo_t::reference:
            status = softmax_reference_wrapper(input, output, params);
            break;
        default:
            log_error("softmax_direct: unsupported algorithm ",
                    static_cast<int32_t>(algo), " (", algo_to_string(algo),
                    ")");
            return status_t::failure;
    }

    if (is_profile) { profiler.tbp_stop(); }

    if (status != status_t::success) { return status; }

    if (apilog_info_enabled() || is_profile) {
        [[maybe_unused]] std::ostringstream ss;
        ss << "LOWOHA softmax_direct: batch=" << params.batch
           << ", axis_dim=" << params.axis_dim << ", axis=" << params.axis
           << ", log_softmax=" << (params.log_softmax ? "true" : "false")
           << ", softmin=" << (params.softmin ? "true" : "false")
           << ", src_dt=" << static_cast<int>(params.src_dt)
           << ", dst_dt=" << static_cast<int>(params.dst_dt)
           << ", algo=" << algo_to_string(algo);
        apilog_info(ss.str());
        if (is_profile) {
            profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(),
                    profiler.get_res_str());
        }
    }

    return status;
}

} // namespace softmax
} // namespace lowoha
} // namespace zendnnl
