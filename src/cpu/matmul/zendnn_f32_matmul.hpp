﻿/*******************************************************************************
* Modifications Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef ZENDNN_F32_MATMUL_HPP
#define ZENDNN_F32_MATMUL_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"

#include "cpu/gemm_inner_product_utils.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"
#include "cpu/matmul/gemm_based_common.hpp"

namespace zendnn {
namespace impl {
namespace cpu {
namespace matmul {

struct zendnn_f32_matmul_t : public primitive_t {
    struct pd_t : public cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("zendnn", zendnn_f32_matmul_t);

        status_t init(engine_t *engine);
        const gemm_based::params_t &params() const {
            return params_;
        }
        int nthr_ {1}; // To not exceed the limit in execute used for set up.
        bool set_default_formats();
      private:
        status_t check_and_configure_attributes();
        gemm_based::params_t params_;
    };

    zendnn_f32_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        if (pd()->params().has_pp_kernel_) {
            const int nthr = pd()->nthr_;
            CHECK(safe_ptr_assign(pp_kernel_,
                                  inner_product_utils::pp_kernel_t::create(pd()->N(),
                                          pd()->M(), pd()->ldc(), &pd()->params().pp_attr_,
                                          pd()->desc()->bias_desc.data_type,
                                          pd()->desc()->accum_data_type, pd()->dst_md(),
                                          false)));
            return pp_kernel_->create_kernel();
        }

        return status::success;
    }

    static constexpr data_type_t src_type = data_type::f32;
    static constexpr data_type_t weights_type = data_type::f32;
    static constexpr data_type_t dst_type = data_type::f32;
    static constexpr data_type_t acc_type = data_type::f32;

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<weights_type>::type weights_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

  private:
    const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }
    status_t execute_ref(const exec_ctx_t &ctx) const;

    std::unique_ptr<inner_product_utils::pp_kernel_t> pp_kernel_;
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace zendnn

#endif
