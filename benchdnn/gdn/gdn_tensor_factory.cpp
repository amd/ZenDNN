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

#include "gdn_tensor_factory.hpp"

namespace zendnnl {
namespace benchdnn {
namespace gdn {

using index_type = tensor_factory_t::index_type;

namespace {

bool is_gdn_float(data_type_t dt) {
    return dt == data_type_t::f32 || dt == data_type_t::bf16
            || dt == data_type_t::f16;
}

std::vector<index_type> dims(std::initializer_list<int64_t> d) {
    std::vector<index_type> out;
    for (int64_t x : d) {
        out.push_back(static_cast<index_type>(x));
    }
    return out;
}

} // anonymous namespace

int create_gdn_tensors(tensor_factory_t &tensor_factory,
        const gdn_tensor_sizes &s, gdn_tensors &out) {
    if (s.num_k_heads <= 0 || s.num_v_heads <= 0 || s.head_k_dim <= 0
            || s.head_v_dim <= 0 || s.conv_width < 0 || s.tokens <= 0
            || s.slots <= 0) {
        commonlog_error(
                "create_gdn_tensors: head counts, head dims, tokens and slots "
                "must be > 0 and conv_width >= 0 (got H=",
                s.num_k_heads, ", HV=", s.num_v_heads, ", K=", s.head_k_dim,
                ", V=", s.head_v_dim, ", W=", s.conv_width,
                ", tokens=", s.tokens, ", slots=", s.slots, ")");
        return NOT_OK;
    }
    if (!is_gdn_float(s.src_dt) || !is_gdn_float(s.state_dt)) {
        commonlog_error(
                "create_gdn_tensors: src_dt and state_dt must be f32, bf16 "
                "or f16, got ",
                datatypeToStr(s.src_dt), " / ", datatypeToStr(s.state_dt));
        return NOT_OK;
    }

    const int64_t D = s.conv_dim(), HV = s.num_v_heads, V = s.head_v_dim;
    const int64_t W = s.conv_width > 0 ? s.conv_width : 1;
    const data_type_t src = s.src_dt;
    // Unit-scale activations and states keep the recurrence bounded over
    // many iterations; small conv weights keep the conv output in range.
    out.mixed_qkv = tensor_factory.uniform_dist_tensor(
            dims({s.tokens, D}), src, 1.0f, "gdn_mixed_qkv");
    out.a = tensor_factory.uniform_dist_tensor(
            dims({s.tokens, HV}), src, 1.0f, "gdn_a");
    out.b = tensor_factory.uniform_dist_tensor(
            dims({s.tokens, HV}), src, 1.0f, "gdn_b");
    out.A_log = tensor_factory.uniform_dist_tensor(
            dims({HV}), data_type_t::f32, 1.0f, "gdn_A_log");
    out.dt_bias = tensor_factory.uniform_dist_tensor(
            dims({HV}), src, 0.5f, "gdn_dt_bias");
    out.conv_weight = tensor_factory.uniform_dist_tensor(
            dims({D, W}), src, 0.5f, "gdn_conv_weight");
    out.conv_bias = tensor_factory.uniform_dist_tensor(
            dims({D}), src, 0.1f, "gdn_conv_bias");
    out.conv_state = tensor_factory.uniform_dist_tensor(
            dims({s.slots, s.conv_taps(), D}), src, 1.0f, "gdn_conv_state");
    out.ssm_state = tensor_factory.uniform_dist_tensor(
            dims({s.slots, HV, s.head_k_dim * V}), s.state_dt, 0.5f,
            "gdn_ssm_state");
    out.output = tensor_factory.zero_tensor(
            dims({s.tokens, HV, V}), src, "gdn_output");
    if (s.fused_norm) {
        out.z = tensor_factory.uniform_dist_tensor(
                dims({s.tokens, HV, V}), src, 2.0f, "gdn_z");
        out.norm_weight = tensor_factory.uniform_dist_tensor(
                dims({V}), data_type_t::f32, 1.0f, "gdn_norm_weight");
    }
    return OK;
}

} // namespace gdn
} // namespace benchdnn
} // namespace zendnnl
