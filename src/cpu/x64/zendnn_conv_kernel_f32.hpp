/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ZENDNN_CONV_KERNEL_F32_HPP
#define ZENDNN_CONV_KERNEL_F32_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace zendnn {
namespace impl {
namespace cpu {
namespace x64 {

struct zendnn_conv_fwd_kernel_f32 : public jit_generator {
    zendnn_conv_fwd_kernel_f32(
        const jit_conv_conf_t &ajcp, const primitive_attr_t &attr,
		const memory_desc_t &dst_md);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(zendnn_conv_fwd_kernel_f32)

    static status_t init_conf(jit_conv_conf_t &jcp,
                              const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
                              const memory_desc_wrapper &weights_d,
                              const memory_desc_wrapper &dst_d, const primitive_attr_t &attr);
	static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
	void generate() override;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace zendnn

#endif
