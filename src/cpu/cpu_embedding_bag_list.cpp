/*******************************************************************************
* Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "cpu/ref_embedding_bag.hpp"
#include "cpu/avx2_embedding_bag.hpp"
#include "cpu/avx512_embedding_bag.hpp"

namespace zendnn {
namespace impl {
namespace cpu {

namespace {
using namespace zendnn::impl::data_type;
using namespace zendnn::impl::prop_kind;

/* add new primitive */
// clang-format off
const std::map<pk_dt_impl_key_t, std::vector<impl_list_item_t>>
&impl_list_map() {
    static const std::map<pk_dt_impl_key_t, std::vector<impl_list_item_t>> the_map =
    REG_EMBEDDING_BAG_P({
        {   {forward, f32, s32, f32}, {
                CPU_INSTANCE(avx512_embedding_bag_t<f32, f32>)
                CPU_INSTANCE(avx2_embedding_bag_t<f32>)
                CPU_INSTANCE(ref_embedding_bag_t<f32>)
                /* eol */
                nullptr,
            }
        },
        {   {forward, bf16, s32, bf16}, {
#if AVX512_BF16_EN
                CPU_INSTANCE(avx512_embedding_bag_t<bf16, bf16>)
#endif
                /* eol */
                nullptr,
            }
        },
        {   {forward, bf16, s32, f32}, {
#if AVX512_BF16_EN
                CPU_INSTANCE(avx512_embedding_bag_t<bf16, f32>)
#endif
                /* eol */
                nullptr,
            }
        },
    });
    return the_map;
}
// clang-format on
} // namespace

const impl_list_item_t *get_embedding_bag_impl_list(
    const embedding_bag_desc_t *desc) {
    static const impl_list_item_t empty_list[] = {nullptr};

    const bool is_fwd = utils::one_of(
                            desc->prop_kind, forward_training, forward_inference);
    prop_kind_t prop_kind = is_fwd ? forward : backward;

    const memory_desc_t *src_md = &desc->input_desc;
    const memory_desc_t *dst_md = &desc->dst_desc;

    pk_dt_impl_key_t key {prop_kind, src_md->data_type, s32, dst_md->data_type};

    const auto impl_list_it = impl_list_map().find(key);
    return impl_list_it != impl_list_map().cend() ? impl_list_it->second.data()
           : empty_list;
}

} // namespace cpu
} // namespace impl
} // namespace zendnn

