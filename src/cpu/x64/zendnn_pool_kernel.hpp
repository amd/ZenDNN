/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
* Copyright 2018 YANDEX LLC
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

#ifndef ZENDNN_POOL_KERNEL_HPP
#define ZENDNN_POOL_KERNEL_HPP

#include <cfloat>
#include <functional>
#include <memory>

#include "common/memory_tracking.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace zendnn {
namespace impl {
namespace cpu {
namespace x64 {

struct bf16_emulation_t;

template <cpu_isa_t isa>
struct zendnn_pool_kernel : public jit_generator {

    zendnn_pool_kernel(
            const jit_pool_conf_t &ajpp,
            const memory_desc_t *dst_md);

    jit_pool_conf_t jpp;
    ~zendnn_pool_kernel();

    DECLARE_CPU_JIT_AUX_FUNCTIONS(zendnn_pool_kernel)

    static status_t init_conf(jit_pool_conf_t &jbp,
            memory_tracking::registrar_t &scratchpad, const pooling_pd_t *ppd,
            int nthreads);

private:
    void generate() override;
    std::unique_ptr<bf16_emulation_t> bf16_emu_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace zendnn

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
