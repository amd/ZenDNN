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
namespace pooling {

#if ZENDNNL_DEPENDS_ONEDNN

status_t pooling_onednn_wrapper(
    const void *input,
    void *output,
    pool_params &params
) {
    try {
        // Create OneDNN engine and stream
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        dnnl::stream strm(eng);

        const pooling_dims_t &dims = params.dims;
        const pooling_data_types &dtypes = params.dtypes;

        // Determine data type
        dnnl::memory::data_type dtype;
        if (dtypes.src == data_type_t::f32) {
            dtype = dnnl::memory::data_type::f32;
        } else if (dtypes.src == data_type_t::bf16) {
            dtype = dnnl::memory::data_type::bf16;
        } else {
            log_error("Pooling OneDNN: Unsupported data type");
            return status_t::failure;
        }

        // Define memory dimensions
        // Input: NHWC -> need to describe as [N, C, H, W] for OneDNN
        dnnl::memory::dims src_dims = {
            static_cast<dnnl::memory::dim>(dims.batch),
            static_cast<dnnl::memory::dim>(dims.channels),
            static_cast<dnnl::memory::dim>(dims.in_height),
            static_cast<dnnl::memory::dim>(dims.in_width)
        };

        // Output: NHWC -> [N, C, H, W]
        dnnl::memory::dims dst_dims = {
            static_cast<dnnl::memory::dim>(dims.batch),
            static_cast<dnnl::memory::dim>(dims.channels),
            static_cast<dnnl::memory::dim>(dims.out_height),
            static_cast<dnnl::memory::dim>(dims.out_width)
        };

        // Kernel dimensions [KH, KW]
        dnnl::memory::dims kernel_dims = {
            static_cast<dnnl::memory::dim>(dims.kernel_height),
            static_cast<dnnl::memory::dim>(dims.kernel_width)
        };

        // Strides [stride_h, stride_w]
        dnnl::memory::dims strides = {
            static_cast<dnnl::memory::dim>(params.stride_h),
            static_cast<dnnl::memory::dim>(params.stride_w)
        };

        // Padding [top, left] and [bottom, right]
        dnnl::memory::dims padding_l = {
            static_cast<dnnl::memory::dim>(params.pad_top),
            static_cast<dnnl::memory::dim>(params.pad_left)
        };

        dnnl::memory::dims padding_r = {
            static_cast<dnnl::memory::dim>(params.pad_bottom),
            static_cast<dnnl::memory::dim>(params.pad_right)
        };
        

        // Dilation (pooling uses 0 for no dilation, unlike conv which uses dilation-1)
        dnnl::memory::dims dilations = {0, 0};

        // Create memory descriptors
        // Input in NHWC format
        auto src_md = dnnl::memory::desc(src_dims, dtype, dnnl::memory::format_tag::nhwc);

        // Output in NHWC format
        auto dst_md = dnnl::memory::desc(dst_dims, dtype, dnnl::memory::format_tag::nhwc);

        // Determine pooling algorithm
        dnnl::algorithm pooling_algo;
        if (params.is_max_pooling) {
            pooling_algo = dnnl::algorithm::pooling_max;
        } else {
            // Average pooling with padding mode
            if (params.avg_mode == avg_pooling_mode_t::include_padding) {
                pooling_algo = dnnl::algorithm::pooling_avg_include_padding;
                log_info("Pooling OneDNN: Using avg pooling with padding included");
            } else {
                pooling_algo = dnnl::algorithm::pooling_avg_exclude_padding;
                log_info("Pooling OneDNN: Using avg pooling with padding excluded");
            }
        }

        // Create pooling primitive descriptor
        // OneDNN pooling signature: (engine, prop_kind, algorithm, src_md, dst_md, 
        //                            strides, kernel, dilation, padding_l, padding_r)
        auto pooling_pd = dnnl::pooling_forward::primitive_desc(
            eng,
            dnnl::prop_kind::forward_inference,
            pooling_algo,
            src_md,
            dst_md,
            strides,
            kernel_dims,
            dilations,
            padding_l,
            padding_r
        );

        // Create memory objects
        auto src_mem = dnnl::memory(src_md, eng, const_cast<void*>(input));
        auto dst_mem = dnnl::memory(dst_md, eng, output);

        // Create pooling primitive
        auto pooling_prim = dnnl::pooling_forward(pooling_pd);

        // Setup arguments
        std::unordered_map<int, dnnl::memory> pooling_args;
        pooling_args.insert({DNNL_ARG_SRC, src_mem});
        pooling_args.insert({DNNL_ARG_DST, dst_mem});

        // Execute pooling
        pooling_prim.execute(strm, pooling_args);

        // Wait for completion
        strm.wait();

        log_info("Pooling OneDNN: Execution completed successfully");
        return status_t::success;

    } catch (const dnnl::error &e) {
        log_error("Pooling OneDNN error: ", e.what(), " (status: ", e.status, ")");
        return status_t::failure;
    } catch (const std::exception &e) {
        log_error("Pooling OneDNN exception: ", e.what());
        return status_t::failure;
    }
}

#else

status_t pooling_onednn_wrapper(
    const void *input,
    void *output,
    const pool_params &params
) {
    log_error("Pooling OneDNN: OneDNN support not enabled");
    return status_t::failure;
}

#endif // ZENDNNL_DEPENDS_ONEDNN

} // namespace pooling
} // namespace lowoha
} // namespace zendnnl

