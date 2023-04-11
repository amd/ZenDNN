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

#ifndef CK_CONVOLUTION_HPP
#define CK_CONVOLUTION_HPP

#ifdef ENABLE_CK

#include "common/c_types_map.hpp"
#include "common/zendnn_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/x64/cpu_reducer.hpp"

#include "cpu/x64/ck_conv_kernel_f32.hpp"

// CK headers
#include "ck/ck.hpp"
#include "ck/device_utility/kernel_launch.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/cpu/device/device_convnd_fwd_avx2_nhwc_kyxc_nhwk.hpp"
#include "ck/tensor_operation/cpu/element/element_wise_operation_cpu.hpp"

namespace zendnn {
namespace impl {
namespace cpu {
namespace x64 {

struct ck_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc,
             const primitive_attr_t *attr,
             const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T("ck", ck_convolution_fwd_t);

        status_t init(engine_t *engine) {
            zendnnVerbose(ZENDNN_CORELOG, "ZENDNN implementation path in ck_convolution_fwd_t::pd_t::init (before checks)");
            bool ok = true && is_fwd()
                      && set_default_alg_kind(alg_kind::convolution_ck)
                      && expect_data_types(data_type::f32, data_type::f32,
                                           data_type::f32, data_type::f32,
                                           data_type::f32)
                      && attr()->has_default_values(
                          primitive_attr_t::skip_mask_t::post_ops,
                          data_type::f32)
                      && !has_zero_dim_memory() && set_default_formats();
            if (!ok) {
                zendnnVerbose(ZENDNN_CORELOG, "ZENDNN implementation path in ck_convolution_fwd_t::pd_t::init: ok=false (after checks)");
                return status::unimplemented;
            }else{
                zendnnVerbose(ZENDNN_CORELOG, "ZENDNN implementation path in ck_convolution_fwd_t::pd_t::init: ok=true (after checks)");
            }

            status_t status = ck_conv_fwd_kernel_f32::init_conf(
                                  jcp_, *desc(), src_md(), weights_md(), dst_md(), *attr());
            if (status != status::success) {
                zendnnVerbose(ZENDNN_CORELOG, "status != status::success on exit from ck_conv_fwd_kernel_f32::init_conf");
                return status;
            } else {
                zendnnVerbose(ZENDNN_CORELOG, "status == status::success on exit from ck_conv_fwd_kernel_f32::init_conf");
            }

            status = ck_conv_fwd_kernel_f32::init_ck_idx(
                         jcp_, *desc(), src_md(), weights_md(), dst_md(), *attr());
            if (status != status::success) {
                zendnnVerbose(ZENDNN_CORELOG, "status != status::success on exit from ck_conv_fwd_kernel_f32::init_ck_idx");
                return status;
            } else {
                zendnnVerbose(ZENDNN_CORELOG, "status == status::success on exit from ck_conv_fwd_kernel_f32::init_ck_idx");
            }

            auto scratchpad = scratchpad_registry().registrar();
            ck_conv_fwd_kernel_f32::init_scratchpad(scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;

      protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto src_tag = nhwc;
            auto dst_tag = nhwc;
            auto wei_tag = Ohwi8o; // this should be KYXCK8 in CK convention
            return set_default_formats_common(src_tag, wei_tag, dst_tag);
        }
    };

    ck_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {
        // add device Conv instances
        using F32 = float;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t init(engine_t *engine) override {
        // add device Conv instances
        using F32 = float;
        add_device_conv_ptrs(conv_ptrs, F32(), F32(), F32());

        CHECK(safe_ptr_assign(kernel_,
                              new ck_conv_fwd_kernel_f32(
                                  pd()->jcp_, *pd()->attr(), *pd()->dst_md(0))));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        status_t status = execute_forward(ctx);
        return status;
    }

  private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }

    std::unique_ptr<ck_conv_fwd_kernel_f32> kernel_;

    // add device Conv instances
    using PassThrough = ck::tensor_operation::cpu::element_wise::PassThrough;
    using DeviceConvFwdNoOpPtr = ck::tensor_operation::cpu::device::
                                 DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>;
    std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
    status_t add_device_conv_ptrs(std::vector<DeviceConvFwdNoOpPtr> &conv_ptrs,
                                  float input_type, float wei_type, float out_type);

};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace zendnn

#endif // #ifdef ENABLE_CK

#endif // #ifndef CK_CONVOLUTION_HPP

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
