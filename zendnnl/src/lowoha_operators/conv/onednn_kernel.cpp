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

#include "lowoha_operators/conv/onednn_kernel.hpp"

#if ZENDNNL_DEPENDS_ONEDNN
#include "dnnl.hpp"
using namespace dnnl;
#endif

namespace zendnnl {
namespace lowoha {
namespace conv {

#if ZENDNNL_DEPENDS_ONEDNN

status_t conv_onednn_wrapper(
    const void *input,
    const void *filter,
    const void *bias,
    void *output,
    conv_params &params
) {
    try {
        const conv_dims_t &dims = params.dims;
        const conv_data_types &dtypes = params.dtypes;

        // Create OneDNN engine and stream
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        dnnl::stream strm(eng);

        // Determine data type
        dnnl::memory::data_type dtype;
        if (dtypes.input == data_type_t::f32) {
            dtype = dnnl::memory::data_type::f32;
        } else if (dtypes.input == data_type_t::bf16) {
            dtype = dnnl::memory::data_type::bf16;
        } else {
            log_error("Conv OneDNN: Unsupported data type");
            return status_t::failure;
        }

        // Define memory dimensions
        // Input: NHWC -> need to describe as [N, C, H, W] for OneDNN
        dnnl::memory::dims src_dims = {
            static_cast<dnnl::memory::dim>(dims.batch),
            static_cast<dnnl::memory::dim>(dims.in_channels),
            static_cast<dnnl::memory::dim>(dims.in_height),
            static_cast<dnnl::memory::dim>(dims.in_width)
        };

        // Filter: [KH, KW, C_in, C_out] -> OneDNN expects [C_out, C_in, KH, KW]
        dnnl::memory::dims weights_dims = {
            static_cast<dnnl::memory::dim>(dims.out_channels),
            static_cast<dnnl::memory::dim>(dims.in_channels),
            static_cast<dnnl::memory::dim>(dims.filter_height),
            static_cast<dnnl::memory::dim>(dims.filter_width)
        };

        // Bias: [C_out]
        dnnl::memory::dims bias_dims = {
            static_cast<dnnl::memory::dim>(dims.out_channels)
        };

        // Output: NHWC -> [N, C, H, W]
        dnnl::memory::dims dst_dims = {
            static_cast<dnnl::memory::dim>(dims.batch),
            static_cast<dnnl::memory::dim>(dims.out_channels),
            static_cast<dnnl::memory::dim>(dims.out_height),
            static_cast<dnnl::memory::dim>(dims.out_width)
        };

        // Strides and padding
        dnnl::memory::dims strides = {
            static_cast<dnnl::memory::dim>(params.stride_h),
            static_cast<dnnl::memory::dim>(params.stride_w)
        };

        dnnl::memory::dims padding_l = {
            static_cast<dnnl::memory::dim>(params.pad_top),
            static_cast<dnnl::memory::dim>(params.pad_left)
        };

        dnnl::memory::dims padding_r = {
            static_cast<dnnl::memory::dim>(params.pad_bottom),
            static_cast<dnnl::memory::dim>(params.pad_right)
        };

        dnnl::memory::dims dilations = {
            static_cast<dnnl::memory::dim>(params.dilation_h - 1),
            static_cast<dnnl::memory::dim>(params.dilation_w - 1)
        };

        // Create memory descriptors
        // Input in NHWC format
        auto src_md = dnnl::memory::desc(src_dims, dtype, dnnl::memory::format_tag::nhwc);

        // Weights: need to reorder from HWCN to OIHW
        auto weights_md = dnnl::memory::desc(weights_dims, dtype, dnnl::memory::format_tag::any);
        auto weights_user_md = dnnl::memory::desc(weights_dims, dtype, dnnl::memory::format_tag::oihw);

        // Output in NHWC format
        auto dst_md = dnnl::memory::desc(dst_dims, dtype, dnnl::memory::format_tag::nhwc);

        // Bias memory descriptor (if present)
        std::unique_ptr<dnnl::memory::desc> bias_md_ptr;
        if (bias != nullptr) {
            bias_md_ptr.reset(new dnnl::memory::desc(bias_dims, dtype, dnnl::memory::format_tag::x));
        }

        // Setup post-ops
        dnnl::primitive_attr conv_attr;
        dnnl::post_ops pops;

        // Add post-operations from param
        using namespace zendnnl::ops;
        for (const auto &po : params.postop_) {
            switch (po.po_type) {
                case post_op_type_t::relu:
                    log_info("Conv2D OneDNN: Adding ReLU post-op");
                    pops.append_eltwise(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
                    break;
                case post_op_type_t::clip:
                    log_info("Conv2D OneDNN: Adding ReLU6 post-op (clipped relu)");
                    pops.append_eltwise(dnnl::algorithm::eltwise_clip, 0.0f, 6.0f);
                    break;
                case post_op_type_t::leaky_relu:
                    log_info("Conv2D OneDNN: Adding LeakyReLU post-op");
                    pops.append_eltwise(dnnl::algorithm::eltwise_relu, po.alpha, 0.0f);
                    break;
                case post_op_type_t::elu:
                    log_info("Conv2D OneDNN: Adding ELU post-op");
                    pops.append_eltwise(dnnl::algorithm::eltwise_elu, po.alpha, 0.0f);
                    break;
                case post_op_type_t::gelu_tanh:
                    log_info("Conv2D OneDNN: Adding GELU-Tanh post-op");
                    pops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 1.0f, 0.0f);
                    break;
                case post_op_type_t::gelu_erf:
                    log_info("Conv2D OneDNN: Adding GELU-Erf post-op");
                    pops.append_eltwise(dnnl::algorithm::eltwise_gelu_erf, 1.0f, 0.0f);
                    break;
                case post_op_type_t::tanh:
                    log_info("Conv2D OneDNN: Adding Tanh post-op");
                    pops.append_eltwise(dnnl::algorithm::eltwise_tanh, 1.0f, 0.0f);
                    break;
                case post_op_type_t::sigmoid:
                    log_info("Conv2D OneDNN: Adding Sigmoid post-op");
                    pops.append_eltwise(dnnl::algorithm::eltwise_logistic, 1.0f, 0.0f);
                    break;
                case post_op_type_t::binary_add:
                    log_info("Conv2D OneDNN: Adding Binary Add post-op (residual connection)");
                    // Binary add requires a scale (usually 1.0) for the residual tensor
                    pops.append_sum(1.0f);
                    break;
                default:
                    log_error("Conv2D OneDNN: Unsupported post-op type: ", static_cast<int>(po.po_type));
                    break;
            }
        }

        if (pops.len() > 0) {
            conv_attr.set_post_ops(pops);
        }

        // Create convolution primitive descriptor
        std::unique_ptr<dnnl::convolution_forward::primitive_desc> conv_pd;

        if (bias != nullptr) {
            conv_pd.reset(new dnnl::convolution_forward::primitive_desc(
                eng,
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::convolution_direct,
                src_md, weights_md, *bias_md_ptr, dst_md,
                strides, dilations, padding_l, padding_r,
                conv_attr
            ));
        } else {
            conv_pd.reset(new dnnl::convolution_forward::primitive_desc(
                eng,
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::convolution_direct,
                src_md, weights_md, dst_md,
                strides, dilations, padding_l, padding_r,
                conv_attr
            ));
        }

        // Create memory objects
        auto src_mem = dnnl::memory(src_md, eng, const_cast<void*>(input));
        auto dst_mem = dnnl::memory(dst_md, eng, output);

        // Handle weight reordering
        // Note: Filter format in TensorFlow is [KH, KW, C_in, C_out]
        // We need to reorder to OneDNN's expected format [C_out, C_in, KH, KW]
        // For now, we'll create a temporary reordered weight buffer
        // TODO: Add caching for reorder blocked format of constant filters

        auto weights_mem = dnnl::memory(conv_pd->weights_desc(), eng);

        // Create user weights memory with proper dimensions
        auto weights_user_mem = dnnl::memory(weights_user_md, eng, const_cast<void*>(filter));

        // Reorder weights if needed
        if (conv_pd->weights_desc() != weights_user_mem.get_desc()) {
            dnnl::reorder(weights_user_mem, weights_mem)
                .execute(strm, weights_user_mem, weights_mem);
        } else {
            weights_mem = weights_user_mem;
        }

        // Create convolution primitive
        auto conv_prim = dnnl::convolution_forward(*conv_pd);

        // Setup arguments
        std::unordered_map<int, dnnl::memory> conv_args;
        conv_args.insert({DNNL_ARG_SRC, src_mem});
        conv_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
        conv_args.insert({DNNL_ARG_DST, dst_mem});

        if (bias != nullptr) {
            auto bias_mem = dnnl::memory(*bias_md_ptr, eng, const_cast<void*>(bias));
            conv_args.insert({DNNL_ARG_BIAS, bias_mem});
        }

        // Execute convolution
        conv_prim.execute(strm, conv_args);

        log_info("Conv OneDNN: Execution completed successfully");
        return status_t::success;

    } catch (const dnnl::error &e) {
        log_error("Conv OneDNN error: ", e.what(), " (status: ", e.status, ")");
        return status_t::failure;
    } catch (const std::exception &e) {
        log_error("Conv OneDNN exception: ", e.what());
        return status_t::failure;
    }
}

#else

status_t conv_onednn_wrapper(
    const void *input,
    const void *filter,
    const void *bias,
    void *output,
    conv_params &params
) {
    log_error("Conv: OneDNN backend not available (ZENDNNL_DEPENDS_ONEDNN not defined)");
    return status_t::failure;
}

#endif

} // namespace conv
} // namespace lowoha
} // namespace zendnnl
