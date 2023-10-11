/*******************************************************************************
* Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
*
*******************************************************************************/

#if AVX512_EB_EN

#ifndef CPU_AVX512_EMBEDDING_BAG_HPP
#define CPU_AVX512_EMBEDDING_BAG_HPP

#include <iostream>
#include <assert.h>
#include <cstdint>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "zendnn_helper.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

#include "cpu/cpu_embedding_bag_pd.hpp"
#include "cpu/avx2_embedding_bag.hpp"

#define  ALIGNED_AVX512_UNSAFE(addr)   ((uint64_t)(addr) & 0x3F)

namespace zendnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct avx512_embedding_bag_t : public primitive_t {
    struct pd_t : public cpu_embedding_bag_pd_t {
        using cpu_embedding_bag_pd_t::cpu_embedding_bag_pd_t;
        using input_type   = typename prec_traits<data_type>::type;
        using indices_type = int32_t;
        using offsets_type = int32_t;

        DECLARE_COMMON_PD_T("avx512:any", avx512_embedding_bag_t);

        status_t init(engine_t *engine) {
            if (! platform::has_data_type_support(data_type) ||
                    !x64::mayiuse(x64::avx512_core)) {
                return status::unimplemented;
            }
            bool eb_avx2 = zendnn_getenv_int("ZENDNN_EBAVX2_ENABLE", 0);
            if (eb_avx2) {
                return status::unimplemented;
            }
            return status::success;
        }
    };
    // constructor using pd_t
    avx512_embedding_bag_t(const pd_t *apd) : primitive_t(apd) {}

    // init() override from primitive_t
    status_t init(engine_t *engine) override {
        return status::success;
    }

    using input_type   = typename prec_traits<data_type>::type;
    using indices_type = int32_t;
    using offsets_type = int32_t;
    using dst_type     = input_type;

    // exec() override from primitive_t
    status_t execute(const exec_ctx_t &ctx) const override;

  private:
    const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }

    status_t pre_process(const exec_ctx_t &ctx,
                         emb_params_t &params) const;
    status_t avx512_sum(const emb_params_t &params) const;
    status_t avx512_sum_wt(const emb_params_t &params) const;

    status_t avx512_mean(const emb_params_t &params) const;
    status_t avx512_max(const emb_params_t &params) const;
    void ebvec_prefetch(float const *input, indices_type *indices,
                        const int32_t width, offsets_type *offsets, const int32_t index,
                        const int32_t offsz, const int32_t indsz) const;


};

} // namespace cpu
} // namespace impl
} // namespace zendnn

#endif
#endif
