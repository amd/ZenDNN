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
#include "lowoha_operators/conv/conv_cache_key.hpp"
#include "../matmul/lru_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace conv {

#if ZENDNNL_DEPENDS_ONEDNN


/**
 * @brief Reorder and cache convolution weights
 *
 * Similar to matmul's reorderAndCacheWeights pattern.
 * Handles both cached and non-cached reordering based on is_weights_const flag.
 *
 * @param key               Cache key for weight identification
 * @param src_weights_mem   Source weights in HWIO format
 * @param dst_weights_mem   Destination weights in OneDNN blocked format
 * @param eng               OneDNN engine
 * @param is_weights_const  If true, enable caching; if false, reorder directly
 * @return true on success
 */
bool reorderAndCacheWeights(
    const Key_conv& key,
    dnnl::memory& src_weights_mem,
    dnnl::memory& dst_weights_mem,
    const dnnl::engine& eng,
    const bool is_weights_const) {

    // Static weight cache
    using namespace zendnnl::lowoha::matmul;
    static lru_cache_t<Key_conv, dnnl::memory> conv_weight_cache(std::numeric_limits<uint32_t>::max());
    static std::mutex weight_cache_mutex;  // Mutex to prevent TOCTOU race

    if (is_weights_const == 0) {
        apilog_info("onednn conv reorder weights (WEIGHT_CACHE_DISABLE)");
        dnnl::stream eng_stream(eng);
        dnnl::reorder(src_weights_mem, dst_weights_mem).execute(eng_stream, src_weights_mem, dst_weights_mem);
    }
    else {
        // Use lock guard to protect the entire check-compute-cache operation
        std::lock_guard<std::mutex> lock(weight_cache_mutex);
        auto found_obj = conv_weight_cache.find_key(key);
        if (!found_obj) {
            apilog_info("onednn conv reorder weights WEIGHT_CACHE_OUT_OF_PLACE");
            dnnl::stream eng_stream(eng);
            dnnl::reorder(src_weights_mem, dst_weights_mem).execute(eng_stream, src_weights_mem, dst_weights_mem);
            conv_weight_cache.add(key, dst_weights_mem);
        }
        else {
            apilog_info("Read onednn conv cached weights WEIGHT_CACHE_OUT_OF_PLACE");
            dst_weights_mem = conv_weight_cache.get(key);
        }
    }

    return true;
}

status_t conv_onednn_wrapper(
    const void *input,
    const void *filter,
    const void *bias,
    void *output,
    const bool is_weights_const,
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

        // Create memory objects
        auto src_mem = dnnl::memory(src_md, eng, const_cast<void*>(input));
        auto dst_mem = dnnl::memory(dst_md, eng, output);

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

        // Create hash for blocking format
        auto hash_blocking_desc = [](const dnnl::memory::desc &mem_desc) -> size_t {
            size_t hash_value = 0;
            const size_t prime = 31;

            const auto strides = mem_desc.get_strides();
            for (const auto &stride : strides) {
                hash_value = hash_value * prime + std::hash<int64_t>{}(stride);
            }

            const int inner_nblks = mem_desc.get_inner_nblks();
            hash_value = hash_value * prime + std::hash<int>{}(inner_nblks);

            const auto inner_blks = mem_desc.get_inner_blks();
            const auto inner_idxs = mem_desc.get_inner_idxs();
            for (int i = 0; i < inner_nblks; ++i) {
                hash_value = hash_value * prime + std::hash<int64_t>{}(inner_blks[i]);
                hash_value = hash_value * prime + std::hash<int64_t>{}(inner_idxs[i]);
            }

            return hash_value;
        };

        size_t blocking_hash = hash_blocking_desc(conv_pd->weights_desc());

        // Create cache key
        Key_conv cache_key;
        cache_key.filter_ptr = filter;
        cache_key.in_channels = dims.in_channels;
        cache_key.out_channels = dims.out_channels;
        cache_key.filter_height = dims.filter_height;
        cache_key.filter_width = dims.filter_width;
        cache_key.is_depthwise = params.depthwise.is_depthwise;
        cache_key.groups = params.depthwise.groups;
        cache_key.depth_multiplier = params.depthwise.depth_multiplier;
        cache_key.dtype = (dtype == dnnl::memory::data_type::f32) ? 0 : 1;
        cache_key.blocking_hash = blocking_hash;

        // Reorder weights using OneDNN reorder API
        const int64_t filt_w = static_cast<int64_t>(dims.filter_width);
        auto weights_mem = dnnl::memory(conv_pd->weights_desc(), eng);

        if (params.depthwise.is_depthwise) {
            // Depthwise convolution: HWIO -> GOIHW using OneDNN reorder
            const int64_t groups = static_cast<int64_t>(params.depthwise.groups);
            const int64_t dm = static_cast<int64_t>(params.depthwise.depth_multiplier);

            dnnl::memory::dims hwio_strides_5d = {
                dm,                                      // stride for G (maps to I in HWIO)
                static_cast<dnnl::memory::dim>(1),       // stride for O_per_group (maps to O in HWIO)
                static_cast<dnnl::memory::dim>(1),       // stride for I_per_group (trivial, size=1)
                filt_w * groups * dm,                    // stride for H
                groups * dm                              // stride for W
            };

            auto weights_hwio_md = dnnl::memory::desc(weights_dims, dtype, hwio_strides_5d);
            auto weights_hwio_mem = dnnl::memory(weights_hwio_md, eng, const_cast<void*>(filter));

            // Reorder and cache weights
            reorderAndCacheWeights(cache_key, weights_hwio_mem, weights_mem, eng, is_weights_const);

            apilog_info("Conv OneDNN: Depthwise weights reordered from TF [H,W,I,dm] to OneDNN blocked format");

        } else {
            // Standard Conv2D: HWIO -> OIHW using OneDNN reorder
            const int64_t out_ch = static_cast<int64_t>(dims.out_channels);
            const int64_t in_ch = static_cast<int64_t>(dims.in_channels);

            dnnl::memory::dims hwio_strides_4d = {
                static_cast<dnnl::memory::dim>(1),       // stride for O (innermost in HWIO)
                out_ch,                                  // stride for I
                filt_w * in_ch * out_ch,                 // stride for H
                in_ch * out_ch                           // stride for W
            };

            auto weights_hwio_md = dnnl::memory::desc(weights_dims, dtype, hwio_strides_4d);
            auto weights_hwio_mem = dnnl::memory(weights_hwio_md, eng, const_cast<void*>(filter));

            // Reorder and cache weights
            reorderAndCacheWeights(cache_key, weights_hwio_mem, weights_mem, eng, is_weights_const);

            apilog_info("Conv OneDNN: Standard weights reordered from TF [H,W,I,O] to OneDNN blocked format");
        }

        // Create convolution primitive (fresh every time - cheap to create)
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

        apilog_info("Conv OneDNN: Execution completed successfully");
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
