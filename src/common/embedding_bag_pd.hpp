/*******************************************************************************
* Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
*******************************************************************************/
#ifndef COMMON_EMBEDING_BAG_PD_HPP
#define COMMON_EMBEDING_BAG_PD_HPP

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

namespace zendnn {
namespace impl {

/* add new primitive */
struct embedding_bag_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::embedding_bag;

    typedef embedding_bag_pd_t hint_class;

    embedding_bag_pd_t(const embedding_bag_desc_t *adesc,
                       const primitive_attr_t *attr,
                       const hint_class *hint_fwd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , input_md_(desc_.input_desc)
        , indices_md_(desc_.indices_desc)
        , offsets_md_(desc_.offsets_desc)
        , weights_md_(desc_.weights_desc)
        , dst_md_(desc_.dst_desc) {}

    const embedding_bag_desc_t *desc() const {
        return &desc_;
    }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
        case query::embedding_bag_d:
            *(const embedding_bag_desc_t **)result = desc();
            break;
        default:
            return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    arg_usage_t arg_usage(int arg) const override {
        switch (arg) {
        case ZENDNN_ARG_SRC_0:
            return arg_usage_t::input;
            break;
        case ZENDNN_ARG_SRC_1:
            return arg_usage_t::input;
            break;
        case ZENDNN_ARG_SRC_2:
            return arg_usage_t::input;
            break;
        case ZENDNN_ARG_SRC_3:
            return arg_usage_t::input;
            break;
        case ZENDNN_ARG_DST:
            return arg_usage_t::output;
            break;
        default:
            return primitive_desc_t::arg_usage(arg);
        }
    }

    const memory_desc_t *arg_md(int arg) const override {
        switch (arg) {
        case ZENDNN_ARG_SRC_0:
            return &input_md_;
            break;
        case ZENDNN_ARG_SRC_1:
            return &indices_md_;
            break;
        case ZENDNN_ARG_SRC_2:
            return &offsets_md_;
            break;
        case ZENDNN_ARG_SRC_3:
            return &weights_md_;
            break;
        case ZENDNN_ARG_DST:
            return   &dst_md_;
            break;
        default:
            return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(int index) const override {
        switch (index) {
        case ZENDNN_ARG_SRC_0:
        case ZENDNN_ARG_SRC_1:
        case ZENDNN_ARG_SRC_2:
        case ZENDNN_ARG_SRC_3:
            return arg_md(index);
        }

        return &glob_zero_md;
    }

    const memory_desc_t *dst_md(int index = ZENDNN_ARG_DST) const override {
        return index == ZENDNN_ARG_DST ? arg_md(index) : &glob_zero_md;
    }

    int n_inputs() const override {
        return desc_.is_weights ? 4 : 3;
    }

    int n_outputs() const override {
        return 1;
    }

  protected:
    embedding_bag_desc_t desc_;

    memory_desc_t input_md_;
    memory_desc_t indices_md_;
    memory_desc_t offsets_md_;
    memory_desc_t weights_md_;
    memory_desc_t dst_md_;

    status_t set_default_params() {
        return status::success;
    }
};

} // namespace impl
} // namespace zendnn

#endif

