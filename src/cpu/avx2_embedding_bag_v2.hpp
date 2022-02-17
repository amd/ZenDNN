/*******************************************************************************
* Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*
*******************************************************************************/

#ifndef CPU_AVX2_EMBEDDING_BAG_V2_HPP
#define CPU_AVX2_EMBEDDING_BAG_V2_HPP

#include <iostream>
#include <assert.h>
#include <cstdint>
#include <immintrin.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_embedding_bag_pd.hpp"

#define  EMB_AVX2_PS_COUNT_V2             (8)
#define  EMB_SCRATCHPAD_LEN_V2            (2048)

namespace zendnn {
namespace impl {
namespace cpu {

/* adding for embedding_bag */
struct emb_params_v2_t {
    using indices_type = int32_t;
    using offsets_type = int32_t;

    bool            is_weights;
    indices_type    padding_idx;
    uint32_t        num_threads;
    void            *input;
    void            *indices;
    void            *offsets;
    void            *dst;
    void            *weights;
    void            *pad_indices;
    void            *pad_weights;
    int32_t         dim_embed;
    int32_t         indices_size;
    int32_t         offset_size;
    int32_t         dst_size;
};

template <impl::data_type_t data_type>
struct avx2_embedding_bag_v2_t : public primitive_t {
    struct pd_t : public cpu_embedding_bag_pd_t {
        using cpu_embedding_bag_pd_t::cpu_embedding_bag_pd_t;
        using input_type   = typename prec_traits<data_type>::type;
        using indices_type = int32_t;
        using offsets_type = int32_t;

        DECLARE_COMMON_PD_T("avx2_v2:any", avx2_embedding_bag_v2_t);

        status_t init(engine_t *engine) {
            if(! platform::has_data_type_support(data_type)) {
                return status::unimplemented;
            }

            //initialize scratchpad
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.template
            book<indices_type>(key_embed_bag_indices, EMB_SCRATCHPAD_LEN_V2);
            scratchpad.template
            book<input_type>(key_embed_bag_weights, EMB_SCRATCHPAD_LEN_V2);

            return status::success;
        }
    };
    // constructor using pd_t
    avx2_embedding_bag_v2_t(const pd_t *apd) : primitive_t(apd) {}

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
                         emb_params_v2_t &params) const;
    status_t avx2_sum(const emb_params_v2_t &params) const;
    status_t avx2_sum_wt(const emb_params_v2_t &params) const;
    status_t avx2_sum_pd(const emb_params_v2_t &params) const;
    status_t avx2_sum_wt_pd(const emb_params_v2_t &params) const;

    status_t avx2_mean(const emb_params_v2_t &params) const;
    status_t avx2_mean_wt(const emb_params_v2_t &params) const;
    status_t avx2_mean_pd(const emb_params_v2_t &params) const;
    status_t avx2_mean_wt_pd(const emb_params_v2_t &params) const;

    status_t avx2_max(const emb_params_v2_t &params) const;
    status_t avx2_max_wt(const emb_params_v2_t &params) const;
    status_t avx2_max_pd(const emb_params_v2_t &params) const;
    status_t avx2_max_wt_pd(const emb_params_v2_t &params) const;

};

} // namespace cpu
} // namespace impl
} // namespace zendnn

#endif
