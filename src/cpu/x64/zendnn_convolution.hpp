/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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

#ifndef ZENDNN_CONVOLUTION_HPP
#define ZENDNN_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/zendnn_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/x64/cpu_reducer.hpp"

#include "cpu/x64/zendnn_conv_kernel_f32.hpp"

namespace zendnn {
namespace impl {
namespace cpu {
namespace x64 {

struct zendnn_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc,
             const primitive_attr_t *attr,
             const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T("zendnn", zendnn_convolution_fwd_t);

        status_t init(engine_t *engine) {
            zendnnVerbose(ZENDNN_CORELOG, "ZENDNN implementation path in zendnn_convolution_fwd_t::pd_t::init (before checks)");
            bool ok = true && is_fwd()
                      && (set_default_alg_kind(alg_kind::convolution_gemm)
                      ||  set_default_alg_kind(alg_kind::convolution_ref) )
                      && expect_data_types(data_type::f32, data_type::f32,
                                           data_type::f32, data_type::f32,
                                           data_type::f32)
                      && attr()->has_default_values(
                              primitive_attr_t::skip_mask_t::post_ops,
                              data_type::f32)
                      && !has_zero_dim_memory() && set_default_formats();
            if (!ok) return status::unimplemented;

            status_t status = zendnn_conv_fwd_kernel_f32::init_conf(
                    jcp_, *desc(), src_md(), weights_md(), dst_md(), *attr());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            zendnn_conv_fwd_kernel_f32::init_scratchpad(scratchpad, jcp_);

            zendnnVerbose(ZENDNN_CORELOG, "ZENDNN implementation path in zendnn_convolution_fwd_t::pd_t::init: status=status::success");
            return status::success;
        }

        jit_conv_conf_t jcp_;

      protected:
        bool set_default_formats() {
            using namespace format_tag;
	        auto src_tag = nhwc;
            auto dst_tag = nhwc;
            auto wei_tag = hwio;
            return set_default_formats_common(src_tag, wei_tag, dst_tag);
        }
    };

    zendnn_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                    new zendnn_conv_fwd_kernel_f32(
                        pd()->jcp_, *pd()->attr(), *pd()->dst_md(0))));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

  private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<zendnn_conv_fwd_kernel_f32> kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace zendnn

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
