/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

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
#ifndef CPU_EMBEDDING_BAG_HPP
#define CPU_EMBEDDING_BAG_HPP

#include <iostream>
#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_embedding_bag_pd.hpp"
#include "common/zendnn_thread.hpp"

namespace zendnn {
namespace impl {
namespace cpu {

/* add new primitive */
template <impl::data_type_t data_type>
struct ref_embedding_bag_t : public primitive_t {
    struct pd_t : public cpu_embedding_bag_pd_t {
        using cpu_embedding_bag_pd_t::cpu_embedding_bag_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_embedding_bag_t);

        status_t init(engine_t *engine) {
            if (! platform::has_data_type_support(data_type)) {
                return status::unimplemented;
            }

            return status::success;
        }
    };
    // constructor using pd_t
    ref_embedding_bag_t(const pd_t *apd) : primitive_t(apd) {}

    // init() override from primitive_t
    status_t init(engine_t *engine) override {
        return status::success;
    }

    using input_type   = typename prec_traits<data_type>::type;
    using indices_type = int32_t;
    using offsets_type = int32_t;
    using dst_type     = input_type;

    // exec() override from primitive_t
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

  private:
    const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }
    status_t execute_ref(const exec_ctx_t &ctx) const;
};

template<data_type_t data_type>
status_t
ref_embedding_bag_t<data_type>::execute_ref(const exec_ctx_t &ctx) const {
    status_t status = status::success;

    // get algorithm params
    auto alg = pd()->desc()->alg_kind;
    auto is_weights = pd()->desc()->is_weights;
    auto padding_idx = pd()->desc()->padding_idx;

    // get the tensors
    auto input   = CTX_IN_MEM(const input_type *, ZENDNN_ARG_SRC_0);
    auto indices = CTX_IN_MEM(const indices_type *, ZENDNN_ARG_SRC_1);
    auto offsets = CTX_IN_MEM(const offsets_type *, ZENDNN_ARG_SRC_2);
    auto dst     = CTX_OUT_MEM(dst_type *, ZENDNN_ARG_DST);

    const input_type *weights = nullptr;
    if (is_weights) {
        weights = CTX_IN_MEM(const input_type *, ZENDNN_ARG_SRC_3);
    }

    // get memory descriptors
    memory_desc_wrapper input_mdw(pd()->src_md(ZENDNN_ARG_SRC_0));
    memory_desc_wrapper indices_mdw(pd()->src_md(ZENDNN_ARG_SRC_1));
    memory_desc_wrapper offsets_mdw(pd()->src_md(ZENDNN_ARG_SRC_2));
    memory_desc_wrapper dst_mdw(pd()->dst_md(ZENDNN_ARG_DST));

    const int  input_ndims  = input_mdw.ndims();
    const auto &input_dims  = input_mdw.dims();
    const auto offset_size  = offsets_mdw.nelems();
    const auto indices_size = indices_mdw.nelems();

    // initialize output to zero
    const auto dst_size     = dst_mdw.nelems();
    std::fill(dst, (dst+dst_size), 0);

    parallel_nd(offset_size,
    [=](dim_t thrd) {
        // compute start and end indices of the bag
        auto start = offsets[thrd];
        auto end   = (thrd < (offset_size-1)) ?
                     offsets[thrd +1] : indices_size;
        auto dst_offset   = thrd*input_dims[1];

        // get row corresponding to first index
        input_type wt_sum = 0;

        if (indices[start] != padding_idx) {
            auto input_offset = indices[start]*input_dims[1];
            auto wt = is_weights ? weights[start] : 1;

            wt_sum  = wt;
            for (auto j = 0; j < input_dims[1]; ++j) {
                dst[dst_offset +j] = wt*input[input_offset +j];
            }
        }

        // compute embedding bags as per the algorithm
        if (alg == alg_kind::embedding_bag_max) {
            for (auto i = start +1; i < end; ++i) {
                if (indices[i] != padding_idx) {
                    auto input_offset  = indices[i]*input_dims[1];
                    auto wt = is_weights ? weights[i] : 1;

                    for (auto j = 0; j < input_dims[1]; ++j)
                        if (dst[dst_offset +j] < wt*input[input_offset +j]) {
                            dst[dst_offset +j] = wt*input[input_offset +j];
                        }
                }
            }
        } else {
            for (auto i = start +1; i < end; ++i) {
                if (indices[i] != padding_idx) {
                    auto input_offset  = indices[i]*input_dims[1];
                    auto wt = is_weights ? weights[i] : 1;

                    wt_sum += wt;
                    for (auto j = 0; j < input_dims[1]; ++j) {
                        dst[dst_offset +j] += wt*input[input_offset +j];
                    }
                }
            }
            if (alg == alg_kind::embedding_bag_mean)
                for (auto j = 0; j < input_dims[1]; ++j) {
                    dst[dst_offset +j] /= wt_sum;
                }
        }
    });

    return status;
}

} // namespace cpu
} // namespace impl
} // namespace zendnn

#endif
