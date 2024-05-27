/*******************************************************************************
* Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
*******************************************************************************/

#include "cpu/cpu_engine.hpp"
#include "cpu/ref_attention.hpp"
//#include "cpu/avx2_attention.hpp"
//#include "cpu/avx512_attention.hpp"

namespace zendnn {
namespace impl {
namespace cpu {

namespace {
using namespace zendnn::impl::data_type;
using namespace zendnn::impl::prop_kind;

/* add new primitive */
// clang-format off
const std::map<pk_dt_impl_key_t, std::vector<impl_list_item_t>> &impl_list_map() {
    static const std::map<pk_dt_impl_key_t, std::vector<impl_list_item_t>> the_map = REG_ATTENTION_P({
        {{forward, f32, f32, f32}, {
            CPU_INSTANCE(ref_attention_t<f32>)
            /* eol */
            nullptr,
        }},
        {{forward, bf16, bf16, bf16}, {
            //Using zendnn type s16 as it returns int16_t type.
            //Attention primitive operations including blas matrix multiplications for bf16 type is
            //internally executed with int16_t type.
            CPU_INSTANCE(ref_attention_t<s16>)
            /* eol */
            nullptr,
        }},
    });
    return the_map;
}
// clang-format on
} // namespace


const impl_list_item_t *get_attention_impl_list(
        const attention_desc_t *desc) {
    static const impl_list_item_t empty_list[] = {nullptr};

    const bool is_fwd = utils::one_of(
            desc->prop_kind, forward_training, forward_inference);
    prop_kind_t prop_kind = is_fwd ? forward : backward;

    /* We can either define a new key struture like pk_dt_impl_key_t for attention, or use the simpler one defined for conv. */
    pk_dt_impl_key_t key {
            prop_kind,
            desc->query_desc.data_type,
            desc->weights_query_desc.data_type,
            desc->dst_desc.data_type,
    };

    const auto impl_list_it = impl_list_map().find(key);
    return impl_list_it != impl_list_map().cend() ? impl_list_it->second.data()
                                                  : empty_list;
}

} // namespace cpu
} // namespace impl
} // namespace zendnn

