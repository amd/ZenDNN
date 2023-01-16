/*******************************************************************************
* Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef CPU_AVX2_EMBEDDING_BAG_HPP
#define CPU_AVX2_EMBEDDING_BAG_HPP

#include <iostream>
#include <assert.h>
#include <cstdint>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_embedding_bag_pd.hpp"

#define  ALIGNED_AVX2_UNSAFE(addr)   ((uint64_t)(addr) & 0x3F)

namespace zendnn {
namespace impl {
namespace cpu {

/* adding for embedding_bag */
struct emb_params_t {
    int32_t         width;
    int32_t         indices_size;
    int32_t         offset_size;
    int32_t         dst_size;
    int32_t         padidx;
    uint32_t        nthr;
    void            *input;
    void            *indices;
    void            *offsets;
    void            *dst;
    void            *weights;
};

template <impl::data_type_t data_type>
struct avx2_embedding_bag_t : public primitive_t {
    struct pd_t : public cpu_embedding_bag_pd_t {
        using cpu_embedding_bag_pd_t::cpu_embedding_bag_pd_t;
        using input_type   = typename prec_traits<data_type>::type;
        using indices_type = int32_t;
        using offsets_type = int32_t;

        DECLARE_COMMON_PD_T("avx2:any", avx2_embedding_bag_t);

        status_t init(engine_t *engine) {
            if(! platform::has_data_type_support(data_type)) {
                return status::unimplemented;
            }

            return status::success;
        }
    };
    // constructor using pd_t
    avx2_embedding_bag_t(const pd_t *apd) : primitive_t(apd) {}

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
    status_t avx2_sum(const emb_params_t &params) const;
    status_t avx2_sum_wt(const emb_params_t &params) const;

    status_t avx2_mean(const emb_params_t &params) const;
    status_t avx2_max(const emb_params_t &params) const;

};

} // namespace cpu
} // namespace impl
} // namespace zendnn

#endif
