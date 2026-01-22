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

        // Filter dimensions depend on whether this is depthwise or standard convolution
        dnnl::memory::dims weights_dims;

        if (params.depthwise.is_depthwise) {
            // DepthwiseConv2D: Use grouped convolution
            // TF Filter: [KH, KW, C_in, depth_multiplier] (HWIO format)
            // OneDNN grouped: [groups, OC_per_group, IC_per_group, KH, KW]
            // For depthwise: groups = C_in, IC_per_group = 1, OC_per_group = depth_multiplier
            weights_dims = {
                static_cast<dnnl::memory::dim>(params.depthwise.groups),        // groups = C_in
                static_cast<dnnl::memory::dim>(params.depthwise.depth_multiplier), // OC per group
                static_cast<dnnl::memory::dim>(1),                              // IC per group = 1
                static_cast<dnnl::memory::dim>(dims.filter_height),
                static_cast<dnnl::memory::dim>(dims.filter_width)
            };
            log_info("Conv OneDNN: Depthwise convolution with groups=", params.depthwise.groups,
                     ", depth_multiplier=", params.depthwise.depth_multiplier);
        } else {
            // Standard Conv2D: Filter: [KH, KW, C_in, C_out] -> OneDNN expects [C_out, C_in, KH, KW]
            weights_dims = {
                static_cast<dnnl::memory::dim>(dims.out_channels),
                static_cast<dnnl::memory::dim>(dims.in_channels),
                static_cast<dnnl::memory::dim>(dims.filter_height),
                static_cast<dnnl::memory::dim>(dims.filter_width)
            };
        }

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

        // Weights: allow OneDNN to choose optimal format
        auto weights_md = dnnl::memory::desc(weights_dims, dtype, dnnl::memory::format_tag::any);

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
        std::unordered_map<int, dnnl::memory> conv_args;
        int post_op_index = 0;

        // Add post-operations from param
        using namespace zendnnl::ops;
        for (const auto &po : params.postop_) {
            switch (po.po_type) {
                case post_op_type_t::relu:
                    log_info("Conv2D OneDNN: Adding ReLU post-op");
                    pops.append_eltwise(dnnl::algorithm::eltwise_relu, po.alpha, po.beta);
                    break;
                case post_op_type_t::clip:
                    log_info("Conv2D OneDNN: Adding ReLU6 post-op (clipped relu)");
                    pops.append_eltwise(dnnl::algorithm::eltwise_clip, po.alpha, po.beta);
                    break;
                case post_op_type_t::leaky_relu:
                    log_info("Conv2D OneDNN: Adding LeakyReLU post-op");
                    pops.append_eltwise(dnnl::algorithm::eltwise_relu, po.alpha, po.beta);
                    break;
                case post_op_type_t::elu:
                    log_info("Conv2D OneDNN: Adding ELU post-op");
                    pops.append_eltwise(dnnl::algorithm::eltwise_elu, po.alpha, po.beta);
                    break;
                case post_op_type_t::gelu_tanh:
                    log_info("Conv2D OneDNN: Adding GELU-Tanh post-op");
                    pops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, po.alpha, po.beta);
                    break;
                case post_op_type_t::gelu_erf:
                    log_info("Conv2D OneDNN: Adding GELU-Erf post-op");
                    pops.append_eltwise(dnnl::algorithm::eltwise_gelu_erf, po.alpha, po.beta);
                    break;
                case post_op_type_t::tanh:
                    log_info("Conv2D OneDNN: Adding Tanh post-op");
                    pops.append_eltwise(dnnl::algorithm::eltwise_tanh, po.alpha, po.beta);
                    break;
                case post_op_type_t::sigmoid:
                    log_info("Conv2D OneDNN: Adding Sigmoid post-op");
                    pops.append_eltwise(dnnl::algorithm::eltwise_logistic, po.alpha, po.beta);
                    break;
                case post_op_type_t::binary_add: {
                    log_info("Conv2D OneDNN: Adding Binary Add post-op (residual connection)");
                    // Create memory descriptor for the binary add tensor
                    // Binary tensor should have same dims as output [N, C, H, W]
                    dnnl::memory::desc binary_md(dst_dims, dtype, dnnl::memory::format_tag::nhwc);

                    // Append binary add operation
                    pops.append_binary(dnnl::algorithm::binary_add, binary_md);

                    // Create memory object for the binary tensor (addend/residual)
                    auto binary_mem = dnnl::memory(binary_md, eng, po.buff);
                    conv_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index) | DNNL_ARG_SRC_1, binary_mem});
                    break;
                }
                default:
                    log_error("Conv2D OneDNN: Unsupported post-op type: ", static_cast<int>(po.po_type));
                    break;
            }
            post_op_index++;
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
        // Note: Filter format in TensorFlow is [H, W, I, O] (HWIO)
        // Standard Conv2D: Need to reorder to OneDNN's [O, I, H, W] (OIHW)
        // Depthwise Conv2D: Need to reorder to OneDNN's [G, O_per_group, I_per_group, H, W] (GOIHW)
        // TODO: Add caching for reorder blocked format of constant filters

        const uint64_t filt_h = dims.filter_height;
        const uint64_t filt_w = dims.filter_width;
        std::vector<char> weights_reordered_buffer;
        auto weights_mem = dnnl::memory(conv_pd->weights_desc(), eng);

        if (params.depthwise.is_depthwise) {
            // Depthwise convolution: HWIO -> GOIHW
            // TF: [H, W, C_in, depth_multiplier]
            // OneDNN: [groups, OC_per_group, IC_per_group, H, W] = [C_in, depth_multiplier, 1, H, W]
            const uint64_t groups = params.depthwise.groups;           // = C_in
            const uint64_t dm = params.depthwise.depth_multiplier;     // output channels per group
            const uint64_t ic_per_group = 1;                            // always 1 for depthwise

            size_t weights_size = groups * dm * ic_per_group * filt_h * filt_w;

            if (dtype == dnnl::memory::data_type::f32) {
                weights_reordered_buffer.resize(weights_size * sizeof(float));
                const float* filter_hwio = static_cast<const float*>(filter);
                float* filter_goihw = reinterpret_cast<float*>(weights_reordered_buffer.data());

                // Transpose from HWIO [H,W,I,O] to GOIHW [G,O,I,H,W] where G=I, I_per_group=1
                // In TF depthwise: I = C_in (groups), O = depth_multiplier
                for (uint64_t g = 0; g < groups; ++g) {              // group index (= input channel)
                    for (uint64_t o = 0; o < dm; ++o) {              // output index within group
                        for (uint64_t h = 0; h < filt_h; ++h) {
                            for (uint64_t w = 0; w < filt_w; ++w) {
                                // TF layout: [H, W, I, O] where I=groups, O=depth_multiplier
                                uint64_t hwio_idx = h * filt_w * groups * dm +
                                                   w * groups * dm +
                                                   g * dm +
                                                   o;
                                // OneDNN grouped layout: [G, O, 1, H, W]
                                uint64_t goihw_idx = g * dm * ic_per_group * filt_h * filt_w +
                                                    o * ic_per_group * filt_h * filt_w +
                                                    0 * filt_h * filt_w +  // IC_per_group=1
                                                    h * filt_w +
                                                    w;
                                filter_goihw[goihw_idx] = filter_hwio[hwio_idx];
                            }
                        }
                    }
                }
            } else {  // bf16
                weights_reordered_buffer.resize(weights_size * sizeof(uint16_t));
                const uint16_t* filter_hwio = static_cast<const uint16_t*>(filter);
                uint16_t* filter_goihw = reinterpret_cast<uint16_t*>(weights_reordered_buffer.data());

                for (uint64_t g = 0; g < groups; ++g) {
                    for (uint64_t o = 0; o < dm; ++o) {
                        for (uint64_t h = 0; h < filt_h; ++h) {
                            for (uint64_t w = 0; w < filt_w; ++w) {
                                uint64_t hwio_idx = h * filt_w * groups * dm +
                                                   w * groups * dm +
                                                   g * dm +
                                                   o;
                                uint64_t goihw_idx = g * dm * ic_per_group * filt_h * filt_w +
                                                    o * ic_per_group * filt_h * filt_w +
                                                    0 * filt_h * filt_w +
                                                    h * filt_w +
                                                    w;
                                filter_goihw[goihw_idx] = filter_hwio[hwio_idx];
                            }
                        }
                    }
                }
            }

            // Create memory descriptor for GOIHW layout
            // Note: OneDNN doesn't have a standard format tag for 2D grouped convolutions
            // Use custom strides to define the layout: [G, O_per_group, I_per_group, H, W]
            dnnl::memory::dims strides_goihw = {
                static_cast<dnnl::memory::dim>(dm * ic_per_group * filt_h * filt_w),
                static_cast<dnnl::memory::dim>(ic_per_group * filt_h * filt_w),
                static_cast<dnnl::memory::dim>(filt_h * filt_w),
                static_cast<dnnl::memory::dim>(filt_w),
                static_cast<dnnl::memory::dim>(1)
            };
            auto weights_goihw_md = dnnl::memory::desc(weights_dims, dtype, strides_goihw);
            auto weights_goihw_mem = dnnl::memory(weights_goihw_md, eng, weights_reordered_buffer.data());

            // Reorder to OneDNN's optimal format if needed
            if (conv_pd->weights_desc() != weights_goihw_mem.get_desc()) {
                dnnl::reorder(weights_goihw_mem, weights_mem)
                    .execute(strm, weights_goihw_mem, weights_mem);
            } else {
                weights_mem = weights_goihw_mem;
            }

            log_info("Conv OneDNN: Depthwise weights reordered from TF [H,W,I,dm] to OneDNN [G,O,I,H,W]");

        } else {
            // Standard Conv2D: HWIO -> OIHW
            const uint64_t out_ch = dims.out_channels;
            const uint64_t in_ch = dims.in_channels;
            size_t weights_size = out_ch * in_ch * filt_h * filt_w;

            if (dtype == dnnl::memory::data_type::f32) {
                weights_reordered_buffer.resize(weights_size * sizeof(float));
                const float* filter_hwio = static_cast<const float*>(filter);
                float* filter_oihw = reinterpret_cast<float*>(weights_reordered_buffer.data());

                // Manually transpose from HWIO to OIHW
                for (uint64_t o = 0; o < out_ch; ++o) {
                    for (uint64_t i = 0; i < in_ch; ++i) {
                        for (uint64_t h = 0; h < filt_h; ++h) {
                            for (uint64_t w = 0; w < filt_w; ++w) {
                                uint64_t hwio_idx = h * filt_w * in_ch * out_ch +
                                                   w * in_ch * out_ch +
                                                   i * out_ch +
                                                   o;
                                uint64_t oihw_idx = o * in_ch * filt_h * filt_w +
                                                   i * filt_h * filt_w +
                                                   h * filt_w +
                                                   w;
                                filter_oihw[oihw_idx] = filter_hwio[hwio_idx];
                            }
                        }
                    }
                }
            } else {  // bf16
                weights_reordered_buffer.resize(weights_size * sizeof(uint16_t));
                const uint16_t* filter_hwio = static_cast<const uint16_t*>(filter);
                uint16_t* filter_oihw = reinterpret_cast<uint16_t*>(weights_reordered_buffer.data());

                // Manually transpose from HWIO to OIHW
                for (uint64_t o = 0; o < out_ch; ++o) {
                    for (uint64_t i = 0; i < in_ch; ++i) {
                        for (uint64_t h = 0; h < filt_h; ++h) {
                            for (uint64_t w = 0; w < filt_w; ++w) {
                                uint64_t hwio_idx = h * filt_w * in_ch * out_ch +
                                                   w * in_ch * out_ch +
                                                   i * out_ch +
                                                   o;
                                uint64_t oihw_idx = o * in_ch * filt_h * filt_w +
                                                   i * filt_h * filt_w +
                                                   h * filt_w +
                                                   w;
                                filter_oihw[oihw_idx] = filter_hwio[hwio_idx];
                            }
                        }
                    }
                }
            }

            // Create memory descriptor for OIHW layout
            auto weights_oihw_md = dnnl::memory::desc(weights_dims, dtype, dnnl::memory::format_tag::oihw);
            auto weights_oihw_mem = dnnl::memory(weights_oihw_md, eng, weights_reordered_buffer.data());

            // Reorder from OIHW to OneDNN's optimal format (if needed)
            if (conv_pd->weights_desc() != weights_oihw_mem.get_desc()) {
                dnnl::reorder(weights_oihw_mem, weights_mem)
                    .execute(strm, weights_oihw_mem, weights_mem);
            } else {
                weights_mem = weights_oihw_mem;
            }

            log_info("Conv OneDNN: Standard weights reordered from TF [H,W,I,O] to OneDNN [O,I,H,W]");
        }

        // Create convolution primitive
        auto conv_prim = dnnl::convolution_forward(*conv_pd);

        // Setup basic arguments
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
