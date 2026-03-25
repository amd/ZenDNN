/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef MATMUL_NATIVE_COMMON_NATIVE_UTILS_HPP
#define MATMUL_NATIVE_COMMON_NATIVE_UTILS_HPP

#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::ops::post_op_type_t;

/// Detect the first fuseable activation post-op from the post-op chain.
/// Returns fused_postop_t::none if no fuseable activation is found.
inline fused_postop_t detect_fused_postop(const matmul_params &params) {
    for (const auto &po : params.postop_) {
        if (po.po_type == post_op_type_t::relu && po.alpha == 0.0f)
            return fused_postop_t::relu;
        if (po.po_type == post_op_type_t::gelu_tanh)
            return fused_postop_t::gelu_tanh;
        if (po.po_type == post_op_type_t::gelu_erf)
            return fused_postop_t::gelu_erf;
        if (po.po_type == post_op_type_t::sigmoid)
            return fused_postop_t::sigmoid;
        if (po.po_type == post_op_type_t::tanh)
            return fused_postop_t::tanh_op;
        if (po.po_type == post_op_type_t::swish)
            return fused_postop_t::swish;
    }
    return fused_postop_t::none;
}

/// Extracted INT8 quantization parameters (per-tensor src, per-tensor/channel wei).
struct Int8QuantParams {
    float src_scale = 1.0f;
    int32_t src_zp = 0;
    const float *wei_scale = nullptr;  ///< nullptr if not provided
    int wei_scale_count = 1;
};

/// Extract INT8 quantization parameters from matmul_params.
/// If wei_scale is not provided, wei_scale is set to nullptr and
/// callers should use a local default (1.0f).
inline Int8QuantParams extract_int8_quant(const matmul_params &params) {
    Int8QuantParams q;
    if (params.quant_params.src_scale.buff)
        q.src_scale = *static_cast<const float *>(params.quant_params.src_scale.buff);

    if (params.quant_params.src_zp.buff) {
        auto dt = params.quant_params.src_zp.dt;
        if (dt == data_type_t::s32)
            q.src_zp = *static_cast<const int32_t *>(params.quant_params.src_zp.buff);
        else if (dt == data_type_t::s8)
            q.src_zp = static_cast<int32_t>(
                *static_cast<const int8_t *>(params.quant_params.src_zp.buff));
        else if (dt == data_type_t::u8)
            q.src_zp = static_cast<int32_t>(
                *static_cast<const uint8_t *>(params.quant_params.src_zp.buff));
    }

    if (params.quant_params.wei_scale.buff) {
        q.wei_scale = static_cast<const float *>(params.quant_params.wei_scale.buff);
        auto &dims = params.quant_params.wei_scale.dims;
        q.wei_scale_count = (!dims.empty() && dims.back() > 1)
            ? static_cast<int>(dims.back()) : 1;
    }
    return q;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_UTILS_HPP
