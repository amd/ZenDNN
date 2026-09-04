/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_operators/matmul/backends/onednn/onednn_execute.hpp"

#include <unordered_map>

#include "common/bfloat16.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using zendnnl::common::bfloat16_t;

#if ZENDNNL_DEPENDS_ONEDNN
void onednn_matmul_execute(const onednn_utils_t::onednn_matmul_params &params,
        std::unordered_map<int, dnnl::memory> &matmul_args,
        dnnl::primitive_attr &matmul_attr, dnnl::engine &eng) {
    dnnl::stream eng_stream(eng);
    dnnl::memory::desc dnnl_input_desc
            = onednn_utils_t::to_dnnl_tensor(params.src, eng);
    dnnl::memory::desc dnnl_weight_desc = params.is_blocked
            ? params.weights.mem.get_desc()
            : onednn_utils_t::to_dnnl_tensor(params.weights, eng);
    dnnl::memory::desc dnnl_output_desc
            = onednn_utils_t::to_dnnl_tensor(params.dst, eng);

    dnnl::memory dnnl_input_tensor
            = dnnl::memory(dnnl_input_desc, eng, params.src.buffer);
    dnnl::memory dnnl_weight_tensor = params.is_blocked
            ? params.weights.mem
            : dnnl::memory(dnnl_weight_desc, eng, params.weights.buffer);
    dnnl::memory dnnl_output_tensor
            = dnnl::memory(dnnl_output_desc, eng, params.dst.buffer);

    dnnl::memory::desc dnnl_bias_desc;
    dnnl::memory dnnl_bias_tensor;
    if (params.bias.buffer != nullptr) {
        dnnl_bias_desc = onednn_utils_t::to_dnnl_tensor(params.bias, eng);
        dnnl_bias_tensor
                = dnnl::memory(dnnl_bias_desc, eng, params.bias.buffer);
    }

    if (params.src_quant.scale_size.size()) {
        auto src_scale_format = (params.src_quant.scale_size.size() == 1)
                ? dnnl::memory::format_tag::a
                : dnnl::memory::format_tag::ab;
        dnnl::memory::desc src_scale_desc(params.src_quant.scale_size,
                onednn_utils_t::to_dnnl_datatype(params.src_quant.scale_dtype),
                src_scale_format);
        dnnl::memory src_scale_mem = dnnl::memory(src_scale_desc, eng,
                const_cast<void *>(params.src_quant.scales));
        matmul_args.insert(
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale_mem});

        if (params.src_quant.zero_size.size()) {
            auto src_zp_format = (params.src_quant.zero_size.size() == 1)
                    ? dnnl::memory::format_tag::a
                    : dnnl::memory::format_tag::ab;
            dnnl::memory::desc src_zero_desc(params.src_quant.zero_size,
                    onednn_utils_t::to_dnnl_datatype(
                            params.src_quant.zero_dtype),
                    src_zp_format);
            dnnl::memory src_zero_mem = dnnl::memory(src_zero_desc, eng,
                    const_cast<void *>(params.src_quant.zero_points));
            matmul_args.insert(
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_mem});
        }
    }

    if (params.weights_quant.scale_size.size()) {
        auto wei_scale_format = (params.weights_quant.scale_size.size() == 1)
                ? dnnl::memory::format_tag::a
                : dnnl::memory::format_tag::ab;
        dnnl::memory::desc wei_scale_desc(params.weights_quant.scale_size,
                onednn_utils_t::to_dnnl_datatype(
                        params.weights_quant.scale_dtype),
                wei_scale_format);
        dnnl::memory wei_scale_mem = dnnl::memory(wei_scale_desc, eng,
                const_cast<void *>(params.weights_quant.scales));
        matmul_args.insert(
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scale_mem});

        if (params.weights_quant.zero_size.size()) {
            auto wei_zp_format = (params.weights_quant.zero_size.size() == 1)
                    ? dnnl::memory::format_tag::a
                    : dnnl::memory::format_tag::ab;
            dnnl::memory::desc wei_zero_desc(params.weights_quant.zero_size,
                    onednn_utils_t::to_dnnl_datatype(
                            params.weights_quant.zero_dtype),
                    wei_zp_format);
            dnnl::memory wei_zero_mem = dnnl::memory(wei_zero_desc, eng,
                    const_cast<void *>(params.weights_quant.zero_points));
            matmul_args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
                    wei_zero_mem});
        }
    }

    // Inverse dst scale buffers must outlive matmul_prim.execute() below.
    bfloat16_t dst_inv_scale_bf16 {};
    float dst_inv_scale_f32 = 0.f;

    if (params.dst_quant.scale_size.size()) {
        auto dst_scale_format = (params.dst_quant.scale_size.size() == 1)
                ? dnnl::memory::format_tag::a
                : dnnl::memory::format_tag::ab;
        dnnl::memory::desc dst_scale_desc(params.dst_quant.scale_size,
                onednn_utils_t::to_dnnl_datatype(params.dst_quant.scale_dtype),
                dst_scale_format);
        float scale_val = (params.dst_quant.scale_dtype == data_type_t::bf16)
                ? static_cast<float>(*static_cast<const bfloat16_t *>(
                          params.dst_quant.scales))
                : *static_cast<const float *>(params.dst_quant.scales);
        const float inv_scale = (scale_val != 0.f) ? (1.0f / scale_val) : 0.f;
        void *dst_scale_buf = nullptr;
        if (params.dst_quant.scale_dtype == data_type_t::bf16) {
            dst_inv_scale_bf16 = bfloat16_t(inv_scale);
            dst_scale_buf = &dst_inv_scale_bf16;
        } else {
            dst_inv_scale_f32 = inv_scale;
            dst_scale_buf = &dst_inv_scale_f32;
        }
        dnnl::memory dst_scale_mem
                = dnnl::memory(dst_scale_desc, eng, dst_scale_buf);

        matmul_args.insert(
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scale_mem});

        if (params.dst_quant.zero_size.size()) {
            auto dst_zp_format = (params.dst_quant.zero_size.size() == 1)
                    ? dnnl::memory::format_tag::a
                    : dnnl::memory::format_tag::ab;
            dnnl::memory::desc dst_zero_desc(params.dst_quant.zero_size,
                    onednn_utils_t::to_dnnl_datatype(
                            params.dst_quant.zero_dtype),
                    dst_zp_format);
            dnnl::memory dst_zero_mem = dnnl::memory(dst_zero_desc, eng,
                    const_cast<void *>(params.dst_quant.zero_points));
            matmul_args.insert(
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_mem});
        }
    }

    [[maybe_unused]] dnnl::memory::desc dnnl_blocked_weight_desc;
    [[maybe_unused]] dnnl::memory dnnl_blocked_weight_tensor;

    bool is_reorder = !params.is_blocked
            && params.algo == matmul_algo_t::onednn_blocked;
    if (is_reorder) {
        onednn_utils_t::onednn_tensor_params blocked_weights_params
                = params.weights;
        blocked_weights_params.format_tag = "any";

        dnnl_blocked_weight_desc
                = onednn_utils_t::to_dnnl_tensor(blocked_weights_params, eng);
    }

    dnnl::matmul::primitive_desc matmul_pd;
    if (params.bias.buffer != nullptr) {
        matmul_pd = dnnl::matmul::primitive_desc(eng, dnnl_input_desc,
                (is_reorder) ? dnnl_blocked_weight_desc : dnnl_weight_desc,
                dnnl_bias_desc, dnnl_output_desc, matmul_attr);
    } else {
        matmul_pd = dnnl::matmul::primitive_desc(eng, dnnl_input_desc,
                (is_reorder) ? dnnl_blocked_weight_desc : dnnl_weight_desc,
                dnnl_output_desc, matmul_attr);
    }
    if (is_reorder) {
        dnnl_blocked_weight_tensor
                = dnnl::memory(matmul_pd.weights_desc(), eng);
        reorder(dnnl_weight_tensor, dnnl_blocked_weight_tensor)
                .execute(eng_stream, dnnl_weight_tensor,
                        dnnl_blocked_weight_tensor);
    }
    auto matmul_prim = dnnl::matmul(matmul_pd);
    matmul_args.insert({DNNL_ARG_SRC, dnnl_input_tensor});
    matmul_args.insert({DNNL_ARG_WEIGHTS,
            (is_reorder) ? dnnl_blocked_weight_tensor : dnnl_weight_tensor});
    if (params.bias.buffer != nullptr) {
        matmul_args.insert({DNNL_ARG_BIAS, dnnl_bias_tensor});
    }
    matmul_args.insert({DNNL_ARG_DST, dnnl_output_tensor});

    matmul_prim.execute(eng_stream, matmul_args);
    eng_stream.wait();
}
#endif

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
