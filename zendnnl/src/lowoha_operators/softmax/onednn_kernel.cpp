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

#include "onednn_kernel.hpp"
#include "common/logging.hpp"
#include "common/profiler.hpp"

#if ZENDNNL_DEPENDS_ONEDNN
#include "dnnl.hpp"
using namespace dnnl;
#endif

namespace zendnnl {
namespace lowoha {
namespace softmax {

#if ZENDNNL_DEPENDS_ONEDNN

status_t softmax_onednn_wrapper(
    const void *input,
    void *output,
    const softmax_params &params
) {
    try {
        // Create OneDNN engine and stream
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        dnnl::stream strm(eng);

        // Determine data type
        dnnl::memory::data_type dtype;
        if (params.src_dt == data_type_t::f32) {
            dtype = dnnl::memory::data_type::f32;
        } else if (params.src_dt == data_type_t::bf16) {
            dtype = dnnl::memory::data_type::bf16;
        } else {
            log_error("Softmax OneDNN: Unsupported data type");
            return status_t::failure;
        }

        // Validate that shape information is provided
        if (params.ndims <= 0 || params.ndims > SOFTMAX_MAX_NDIMS) {
            log_error("Softmax OneDNN: Invalid ndims: ", params.ndims,
                      " (must be 1-", SOFTMAX_MAX_NDIMS, ").");
            return status_t::failure;
        }

        // Build OneDNN memory dimensions from original N-D shape
        dnnl::memory::dims src_dims;
        src_dims.reserve(params.ndims);
        for (int i = 0; i < params.ndims; ++i) {
            src_dims.push_back(static_cast<dnnl::memory::dim>(params.shape[i]));
        }

        // Normalize axis to positive value
        int softmax_axis = params.axis >= 0 ? params.axis : params.ndims + params.axis;

        // Get appropriate format tag for this dimensionality
        dnnl::memory::format_tag format;
        switch (params.ndims) {
            case 1: format = dnnl::memory::format_tag::a; break;
            case 2: format = dnnl::memory::format_tag::ab; break;
            case 3: format = dnnl::memory::format_tag::abc; break;
            case 4: format = dnnl::memory::format_tag::abcd; break;
            case 5: format = dnnl::memory::format_tag::abcde; break;
            default:
                log_error("Softmax OneDNN: Unsupported number of dimensions: ", params.ndims);
                return status_t::failure;
        }

        log_info("Softmax OneDNN: ", params.ndims, "D tensor (shape=[", params.shape[0]);
        for (int i = 1; i < params.ndims; ++i) {
            log_info(",", params.shape[i]);
        }
        log_info("]), softmax on axis ", softmax_axis);

        // Create memory descriptors
        auto src_md = dnnl::memory::desc(src_dims, dtype, format);
        auto dst_md = dnnl::memory::desc(src_dims, dtype, format);

        // Create softmax primitive descriptor
        dnnl::softmax_forward::primitive_desc softmax_pd;

        if (params.log_softmax) {
            softmax_pd = dnnl::softmax_forward::primitive_desc(
                eng,
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::softmax_log,
                src_md,
                dst_md,
                softmax_axis
            );
        } else {
            softmax_pd = dnnl::softmax_forward::primitive_desc(
                eng,
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::softmax_accurate,
                src_md,
                dst_md,
                softmax_axis
            );
        }

        // Create memory objects
        auto src_mem = dnnl::memory(src_md, eng, const_cast<void*>(input));
        auto dst_mem = dnnl::memory(dst_md, eng, output);

        // Create softmax primitive
        auto softmax_prim = dnnl::softmax_forward(softmax_pd);

        // Setup arguments
        std::unordered_map<int, dnnl::memory> softmax_args;
        softmax_args.insert({DNNL_ARG_SRC, src_mem});
        softmax_args.insert({DNNL_ARG_DST, dst_mem});

        // Execute softmax
        softmax_prim.execute(strm, softmax_args);

        // Wait for completion
        strm.wait();

        log_info("Softmax OneDNN: Execution completed successfully");
        return status_t::success;

    } catch (const dnnl::error &e) {
        log_error("Softmax OneDNN error: ", e.what(), " (status: ", e.status, ")");
        return status_t::failure;
    } catch (const std::exception &e) {
        log_error("Softmax OneDNN exception: ", e.what());
        return status_t::failure;
    }
}

#else

status_t softmax_onednn_wrapper(
    const void *input,
    void *output,
    const softmax_params &params
) {
    log_error("Softmax OneDNN: OneDNN support not enabled");
    return status_t::failure;
}

#endif // ZENDNNL_DEPENDS_ONEDNN

} // namespace softmax
} // namespace lowoha
} // namespace zendnnl
