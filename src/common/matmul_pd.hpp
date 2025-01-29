/*******************************************************************************
* Modifications Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef COMMON_MATMUL_PD_HPP
#define COMMON_MATMUL_PD_HPP

#include <assert.h>

#include "zendnn.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

namespace zendnn {
namespace impl {

struct matmul_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::matmul;

    typedef matmul_pd_t base_class;
    typedef matmul_pd_t hint_class;

    const matmul_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::matmul_d:
                *(const matmul_desc_t **)result = desc();
                break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    arg_usage_t arg_usage(int arg) const override {
        const bool input = utils::one_of(
                arg, ZENDNN_ARG_SRC, ZENDNN_ARG_WEIGHTS, ZENDNN_ARG_BIAS);
        if (input) return arg_usage_t::input;

        if (arg == ZENDNN_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(int arg) const override {
        switch (arg) {
            case ZENDNN_ARG_SRC: return src_md(0);
            case ZENDNN_ARG_WEIGHTS: return weights_md(0);
            case ZENDNN_ARG_BIAS: return weights_md(1);
            case ZENDNN_ARG_DST: return dst_md(0);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(int index = 0) const override {
        return index == 0 ? &src_md_ : &glob_zero_md;
    }

    const memory_desc_t *weights_md(int index = 0) const override {
        return utils::pick(index, &weights_md_, &bias_md_, &glob_zero_md);
    }

    const memory_desc_t *dst_md(int index = 0) const override {
        return index == 0 ? &dst_md_ : &glob_zero_md;
    }

    int n_inputs() const override {
        return 2 + with_bias() + n_binary_po_inputs();
    }
    int n_outputs() const override { return 1; }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(dst_md(0)).has_zero_dim();
    }

    bool has_runtime_dims_or_strides() const {
        return memory_desc_wrapper(src_md_).has_runtime_dims_or_strides()
                || memory_desc_wrapper(weights_md_)
                           .has_runtime_dims_or_strides()
                || memory_desc_wrapper(dst_md_).has_runtime_dims_or_strides();
    };

    int ndims() const { return dst_md_.ndims; }

    dim_t ldc() const {
        return memory_desc_wrapper(dst_md(0))
                .blocking_desc()
                .strides[ndims() - 2];
    }

    bool with_bias() const { return bias_md_.ndims != 0; }
    bool batched() const { return ndims() > 2; }

    dim_t batch() const {
        return utils::array_product(dst_md_.dims, ndims() - 2);
    }
    dim_t M() const { return dst_md_.dims[ndims() - 2]; }
    dim_t N() const { return dst_md_.dims[ndims() - 1]; }
    dim_t K() const { return src_md_.dims[ndims() - 1]; }

    bool is_bias_1xN() const {
        if (!with_bias()) return false;

        const auto &dims = weights_md(1)->dims;
        const int n_dims = ndims();
        for (int i = 0; i < n_dims - 1; ++i) {
            if (dims[i] != 1) return false;
        }

        return dims[n_dims - 1] == N();
    }
    int src_qmask_M() const {
        const int src_ndims = src_md(0)->ndims;
        assert(src_ndims >= 2);
        return 1 << (src_ndims - 2);
    }

    int src_qmask_K() const {
        const int src_ndims = src_md(0)->ndims;
        assert(src_ndims >= 2);
        return 1 << (src_ndims - 1);
    }

    virtual bool attr_scales_ok(const std::vector<int> &supported_args
            = {ZENDNN_ARG_SRC, ZENDNN_ARG_WEIGHTS, ZENDNN_ARG_DST}) const {
        if (attr()->static_scales_.has_default_values()) return true;
        bool ok = attr()->static_scales_.has_default_values(supported_args);
        for (int arg : supported_args) {
            const auto &sc = attr()->static_scales_.get(arg);
            const auto &mask = sc.mask_;
            if (sc.has_default_values()) {
                continue;
            }
            if (arg == ZENDNN_ARG_WEIGHTS) {
                const bool wei_n_group_ok
                        = IMPLICATION(sc.ndims_ == 2 && sc.group_dims_[1] > 1,
                                N() % sc.group_dims_[1] == 0);

                // Any group is allowed to be greater than 1 but only one at a
                // time, not both.
                ok = ok && utils::one_of(sc.ndims_, 0, 2)
                        && IMPLICATION(sc.ndims_ == 2,
                                utils::one_of(
                                        1, sc.group_dims_[0]) && wei_n_group_ok);
            } else if (arg == ZENDNN_ARG_SRC) {
                ok = ok
                        && utils::one_of(mask, 0, src_qmask_K(),
                                src_qmask_M() + src_qmask_K());
                ok = ok && utils::one_of(sc.ndims_, 0, 2);
                ok = ok && IMPLICATION((mask & src_qmask_K()), sc.ndims_ == 2);
                ok = ok
                        && IMPLICATION(sc.ndims_ == 2,
                                sc.group_dims_[0] == 1);
            } else {
                ok = ok && (mask == 0);
            }
        }
        return ok;
    }

protected:
    matmul_desc_t desc_;

    memory_desc_t src_md_;
    memory_desc_t weights_md_;
    memory_desc_t bias_md_;
    memory_desc_t dst_md_;

    matmul_pd_t(const matmul_desc_t *adesc, const primitive_attr_t *attr,
            const matmul_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*adesc)
        , src_md_(desc_.src_desc)
        , weights_md_(desc_.weights_desc)
        , bias_md_(desc_.bias_desc)
        , dst_md_(desc_.dst_desc) {}

    // temporary solution to deal with format `any`
    bool set_default_formats() {
        for (auto md : {&src_md_, &weights_md_, &bias_md_, &dst_md_}) {
            memory_desc_wrapper mdw(md);
            if (mdw.format_any()) {
                if (mdw.has_runtime_dims_or_strides()) return false;
                status_t status = memory_desc_init_by_strides(*md, nullptr);
                if (status != status::success) return false;
            }
        }

        return true;
    }
};

} // namespace impl
} // namespace zendnn

#endif
