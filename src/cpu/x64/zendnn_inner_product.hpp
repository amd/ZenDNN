﻿/*******************************************************************************
* Modifications Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#ifndef CPU_ZENDNN_INNER_PRODUCT_HPP
#define CPU_ZENDNN_INNER_PRODUCT_HPP

#include <assert.h>

#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm_inner_product_utils.hpp"

#include "cpu/cpu_inner_product_pd.hpp"

namespace zendnn {
namespace impl {
namespace cpu {
namespace x64 {

template <impl::data_type_t data_type>
struct zendnn_inner_product_fwd_t : public primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T("zendnn", zendnn_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;

            const bool ok = true && is_fwd() && !has_zero_dim_memory()
                            && everyone_is(data_type, src_md()->data_type,
                                           weights_md()->data_type, dst_md()->data_type,
                                           with_bias() ? weights_md(1)->data_type : data_type)
                            && attr()->has_default_values(
                                primitive_attr_t::skip_mask_t::post_ops)
                            && set_default_params() == status::success
                            && dense_gemm_consitency_check(
                                src_md(), weights_md(), dst_md())
                            && inner_product_utils::post_ops_ok(
                                attr()->post_ops_, &dst_md_);
            return ok ? status::success : status::unimplemented;
        }
    };

    zendnn_inner_product_fwd_t(const pd_t *apd)
        : primitive_t(apd), postops_in_ip_(false), beta_(0) {}

    status_t init(engine_t *engine) override {
        const bool has_bias = pd()->with_bias();
        const bool has_eltwise
            = pd()->attr()->post_ops_.find(primitive_kind::eltwise) >= 0;
        const bool has_binary
            = pd()->attr()->post_ops_.find(primitive_kind::binary) >= 0;
        postops_in_ip_ = has_bias || has_eltwise || has_binary;

        CHECK(safe_ptr_assign(pp_kernel_, inner_product_utils::pp_kernel_t::create(pd(),
                              true)));

        auto sum_idx = pd()->attr()->post_ops_.find(primitive_kind::sum);
        beta_ = sum_idx >= 0 ? pd()->attr()->post_ops_.entry_[sum_idx].sum.scale
                : 0.0;

        return pp_kernel_->create_kernel();
    }

    typedef typename prec_traits<data_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

  private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }

    std::unique_ptr<inner_product_utils::pp_kernel_t> pp_kernel_;
    bool postops_in_ip_;
    float beta_;
};

template <impl::data_type_t data_type>
struct zendnn_inner_product_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_inner_product_bwd_data_pd_t {
        using cpu_inner_product_bwd_data_pd_t::cpu_inner_product_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("zendnn", zendnn_inner_product_bwd_data_t);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                      && !has_zero_dim_memory()
                      && utils::everyone_is(data_type, diff_src_md()->data_type,
                                            weights_md()->data_type, diff_dst_md()->data_type)
                      && attr()->has_default_values()
                      && set_default_params() == status::success
                      && dense_gemm_consitency_check(
                          diff_src_md(), weights_md(), diff_dst_md());
            return ok ? status::success : status::unimplemented;
        }
    };

    zendnn_inner_product_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}
    typedef typename prec_traits<data_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

  private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }
};

template <impl::data_type_t data_type>
struct zendnn_inner_product_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_inner_product_bwd_weights_pd_t {
        using cpu_inner_product_bwd_weights_pd_t::
        cpu_inner_product_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("zendnn", zendnn_inner_product_bwd_weights_t);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                      && !has_zero_dim_memory()
                      && utils::everyone_is(data_type, src_md()->data_type,
                                            diff_weights_md()->data_type,
                                            diff_dst_md()->data_type,
                                            with_bias() ? diff_weights_md(1)->data_type
                                            : data_type)
                      && attr()->has_default_values()
                      && set_default_params() == status::success
                      && dense_gemm_consitency_check(
                          src_md(), diff_weights_md(), diff_dst_md());

            return ok ? status::success : status::unimplemented;
        }
    };

    zendnn_inner_product_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}
    typedef typename prec_traits<data_type>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

  private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace zendnn

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
