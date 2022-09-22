/*******************************************************************************
* Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
*
*******************************************************************************/

#include "cpu/cpu_engine.hpp"
#include "cpu/ref_embedding_bag.hpp"
#include "cpu/avx2_embedding_bag.hpp"

namespace zendnn {
namespace impl {
namespace cpu {

namespace {
using namespace zendnn::impl::data_type;
using namespace zendnn::impl::prop_kind;

/* add new primitive */
// clang-format off
const std::map<pk_impl_key_t, std::vector<impl_list_item_t>> &impl_list_map() {
    static const std::map<pk_impl_key_t, std::vector<impl_list_item_t>> the_map = REG_EMBEDDING_BAG_P({
        {{forward}, {
            CPU_INSTANCE(avx2_embedding_bag_t<f32>)
            CPU_INSTANCE(ref_embedding_bag_t<f32>)
            /* eol */
            nullptr,
        }},
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

    pk_impl_key_t key {prop_kind};

    const auto impl_list_it = impl_list_map().find(key);
    return impl_list_it != impl_list_map().cend() ? impl_list_it->second.data()
                                                  : empty_list;
}

} // namespace cpu
} // namespace impl
} // namespace zendnn

