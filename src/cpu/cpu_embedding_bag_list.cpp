/*******************************************************************************
* Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
*
*******************************************************************************/

#include "cpu/cpu_engine.hpp"
#include "cpu/ref_embedding_bag.hpp"
#include "cpu/avx2_embedding_bag.hpp"
#include "cpu/avx2_embedding_bag_v2.hpp"

namespace zendnn {
namespace impl {
namespace cpu {

/* add new primitive */
using pd_create_f = engine_t::primitive_desc_create_f;

namespace {
using namespace zendnn::impl::data_type;

// clang-format off
const pd_create_f impl_list[] = {
    CPU_INSTANCE(avx2_embedding_bag_t<f32>)
    CPU_INSTANCE(avx2_embedding_bag_v2_t<f32>)
    CPU_INSTANCE(ref_embedding_bag_t<f32>)
    /* eol */
    nullptr,
};
// clang-format on
} //namespace

const pd_create_f *
get_embedding_bag_impl_list(const embedding_bag_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
};

} // namespace cpu
} // namespace impl
} // namespace zendnn

