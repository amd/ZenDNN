/*******************************************************************************
* Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*
*******************************************************************************/

#include "common/c_types_map.hpp"
#include "common/zendnn_thread.hpp"
#include "common/zendnn_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/avx2_embedding_bag_v2.hpp"

namespace zendnn {
namespace impl {
namespace cpu {

using namespace data_type;

template<data_type_t data_type>
status_t
avx2_embedding_bag_v2_t<data_type>::execute(const exec_ctx_t &ctx) const {

#if ZENDNN_CPU_THREADING_RUNTIME != ZENDNN_RUNTIME_OMP
    assert(!"threading env need to be omp for embedding_bag");
#endif

    status_t status;

    // initialize
    emb_params_v2_t  params;

    status = pre_process(ctx, params);
    if (status != status::success)
        return status;

    auto  algo                = pd()->desc()->alg_kind;
    bool &is_weights          = params.is_weights;

    switch(algo) {
    case alg_kind::embedding_bag_sum:
        return is_weights ? avx2_sum_wt(params) : avx2_sum(params);
    case alg_kind::embedding_bag_mean:
        return is_weights ? avx2_mean_wt(params) : avx2_mean(params);
    case alg_kind::embedding_bag_max:
        return is_weights ? avx2_max_wt(params) : avx2_max(params);
    }

    return status::unimplemented;
}

/*
 * extract embedding bag parameters
 */
template<data_type_t data_type>
status_t
avx2_embedding_bag_v2_t<data_type>::pre_process(const exec_ctx_t &ctx,
                                                emb_params_v2_t &params) const {

    status_t status = status::success;

    // get algorithm params
    params.is_weights   = pd()->desc()->is_weights;
    params.padding_idx  = pd()->desc()->padding_idx;
    params.num_threads  = pd()->desc()->num_threads;

    // get the tensors
    params.input =
        static_cast<void *>(ctx.host_ptr(ZENDNN_ARG_SRC_0));
    params.indices =
        static_cast<void *>(ctx.host_ptr(ZENDNN_ARG_SRC_1));
    params.offsets =
        static_cast<void *>(ctx.host_ptr(ZENDNN_ARG_SRC_2));

    params.weights = nullptr;
    if(params.is_weights) {
        params.weights
            = static_cast<void *>(ctx.host_ptr(ZENDNN_ARG_SRC_3));
    }

    params.dst = static_cast<void *>(ctx.host_ptr(ZENDNN_ARG_DST));

    // get memory descriptors
    memory_desc_wrapper input_mdw(pd()->src_md(ZENDNN_ARG_SRC_0));
    memory_desc_wrapper indices_mdw(pd()->src_md(ZENDNN_ARG_SRC_1));
    memory_desc_wrapper offsets_mdw(pd()->src_md(ZENDNN_ARG_SRC_2));
    memory_desc_wrapper dst_mdw(pd()->dst_md(ZENDNN_ARG_DST));

    const auto &input_dims   = input_mdw.dims();
    params.dim_embed         = input_dims[1];

    params.offset_size       = offsets_mdw.nelems();
    params.indices_size      = indices_mdw.nelems();

    // get scratchpad memory
    using namespace memory_tracking::names;
    using input_type   = typename prec_traits<data_type>::type;

    auto pad_grantr     = ctx.get_scratchpad_grantor();
    params.pad_indices  = pad_grantr.template get(key_embed_bag_indices);
    params.pad_weights  = pad_grantr.template get(key_embed_bag_weights);

    // initialize output to zero
    params.dst_size     = dst_mdw.nelems();

    if (params.offset_size < params.num_threads)
        params.num_threads = params.offset_size;

    if (params.indices_size > EMB_SCRATCHPAD_LEN_V2)
        return status::out_of_memory;

    return status;
}

/*
 * sum without weights
 */
template<>
status_t
avx2_embedding_bag_v2_t<f32>::avx2_sum(const emb_params_v2_t &params) const {

    const input_type   *input   = static_cast<input_type *>(params.input);
    const indices_type *indices = static_cast<indices_type *>(params.indices);
    const offsets_type *offsets = static_cast<offsets_type *>(params.offsets);
    dst_type     *dst           = static_cast<dst_type *>(params.dst);

    const int32_t &dim_embed        = params.dim_embed;
    const int32_t &indices_size     = params.indices_size;
    const int32_t &offset_size      = params.offset_size;
    const int32_t &dst_size         = params.dst_size;
    const indices_type &padding_idx = params.padding_idx;

    const int32_t dim_embed_div = params.dim_embed / EMB_AVX2_PS_COUNT_V2;
    const int32_t dim_embed_rem = params.dim_embed % EMB_AVX2_PS_COUNT_V2;

#pragma omp parallel num_threads(params.num_threads)
    {
        int nthr     = omp_get_num_threads();
        int ithr     = omp_get_thread_num();

        int sbuf_offset     = ithr*EMB_SCRATCHPAD_LEN_V2/nthr;
        indices_type* sidx  = static_cast<indices_type*>(params.pad_indices);
        sidx               += sbuf_offset;

        #pragma omp for
        for (auto oi = 0; oi < offset_size; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offset_size -1) ? offsets[oi+1] : indices_size;

            // preprocess indices
            int last = 0;
            if (padding_idx >= 0) {
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padding_idx) {
                        sidx[last++] = indices[i]*dim_embed;
                    }
                }
            }
            else {
                for (auto i = ofirst; i < olast; ++i) {
                    sidx[last++] = indices[i]*dim_embed;
                }
            }

            float *dst_base        = dst + (oi * dim_embed);
            float *in_base         = const_cast<float *>(input);
            float *dst_ptr         = dst_base;
            int    shift           = 0;

            for(auto j = 0; j < dim_embed_div; ++j) {
                __m256 sum     = _mm256_setzero_ps();

                for(auto i = 0; i < last; ++i) {
                    auto in_ptr = in_base + sidx[i];
                    __m256 aa   = _mm256_loadu_ps(in_ptr);
                    sum         = _mm256_add_ps(aa, sum);
                }

                _mm256_storeu_ps(dst_ptr, sum);

                shift   += EMB_AVX2_PS_COUNT_V2;
                in_base = const_cast<float *>(input)  + shift;
                dst_ptr = dst_base + shift;
            }

            // remaining vector
            if (dim_embed_rem) {
                auto in_ptr  = in_base + sidx[0];
                for(auto j = 0; j < dim_embed_rem; ++j) {
                    dst_ptr[j] = in_ptr[j];
                }

                for(auto i = 1; i < last; ++i) {
                    auto in_ptr  = in_base + sidx[i];
                    for(auto j = 0; j < dim_embed_rem; ++j) {
                        dst_ptr[j] += in_ptr[j];
                    }
                }
            }
          } //for oi
        } // omp parallel
    return status::success;
}

/*
 * sum with weights
 */
template<>
status_t
avx2_embedding_bag_v2_t<f32>::avx2_sum_wt(const emb_params_v2_t &params) const {

    const input_type   *input   = static_cast<input_type *>(params.input);
    const indices_type *indices = static_cast<indices_type *>(params.indices);
    const offsets_type *offsets = static_cast<offsets_type *>(params.offsets);
    const input_type   *weights = static_cast<input_type *>(params.weights);

    dst_type     *dst           = static_cast<dst_type *>(params.dst);

    const int32_t &dim_embed        = params.dim_embed;
    const int32_t &indices_size     = params.indices_size;
    const int32_t &offset_size      = params.offset_size;
    const int32_t &dst_size         = params.dst_size;
    const indices_type &padding_idx = params.padding_idx;

    const int32_t dim_embed_div = params.dim_embed / EMB_AVX2_PS_COUNT_V2;
    const int32_t dim_embed_rem = params.dim_embed % EMB_AVX2_PS_COUNT_V2;

#pragma omp parallel num_threads(params.num_threads)
    {
        int nthr     = omp_get_num_threads();
        int ithr     = omp_get_thread_num();

        int sbuf_offset     = ithr*EMB_SCRATCHPAD_LEN_V2/nthr;
        indices_type* sidx  = static_cast<indices_type*>(params.pad_indices);
        float* swt          = static_cast<float*>(params.pad_weights);
        sidx               += sbuf_offset;
        swt                += sbuf_offset;

        #pragma omp for
        for (auto oi = 0; oi < offset_size; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offset_size -1) ? offsets[oi+1] : indices_size;

            // preprocess indices
            int last = 0;
            if (padding_idx >= 0) {
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padding_idx) {
                        swt[last]    = static_cast<float>(weights[i]);
                        sidx[last++] = indices[i]*dim_embed;
                    }
                }
            }
            else {
                for (auto i = ofirst; i < olast; ++i) {
                    swt[last]    = static_cast<float>(weights[i]);
                    sidx[last++] = indices[i]*dim_embed;
                }
            }

            float *dst_base        = dst + (oi * dim_embed);
            float *in_base         = const_cast<float *>(input);
            float *dst_ptr         = dst_base;
            int    shift           = 0;

            for(auto j = 0; j < dim_embed_div; ++j) {
                __m256 sum     = _mm256_setzero_ps();

                for(auto i = 0; i < last; ++i) {
                    auto in_ptr = in_base + sidx[i];
                    __m256 aa   = _mm256_loadu_ps(in_ptr);
                    __m256 bb   = _mm256_set1_ps(swt[i]);
                    sum         = _mm256_fmadd_ps(aa, bb, sum);
                }

                _mm256_storeu_ps(dst_ptr, sum);

                shift   += EMB_AVX2_PS_COUNT_V2;
                in_base = const_cast<float *>(input)  + shift;
                dst_ptr = dst_base + shift;
            }

            // remaining vector
            if (dim_embed_rem) {
                auto in_ptr  = in_base + sidx[0];
                for(auto j = 0; j < dim_embed_rem; ++j) {
                    dst_ptr[j] = swt[0]*in_ptr[j];
                }

                for(auto i = 1; i < last; ++i) {
                    auto in_ptr  = in_base + sidx[i];
                    for(auto j = 0; j < dim_embed_rem; ++j) {
                        dst_ptr[j] += swt[i]*in_ptr[j];
                    }
                }
            }
        } //for oi
    } // omp parallel

    return status::success;
}

/*
 * mean without weights or padding index
 */
template<>
status_t
avx2_embedding_bag_v2_t<f32>::avx2_mean(const emb_params_v2_t &params) const {

    const input_type   *input   = static_cast<input_type *>(params.input);
    const indices_type *indices = static_cast<indices_type *>(params.indices);
    const offsets_type *offsets = static_cast<offsets_type *>(params.offsets);
    dst_type     *dst           = static_cast<dst_type *>(params.dst);

    const int32_t &dim_embed        = params.dim_embed;
    const int32_t &indices_size     = params.indices_size;
    const int32_t &offset_size      = params.offset_size;
    const int32_t &dst_size         = params.dst_size;
    const indices_type &padding_idx = params.padding_idx;

    const int32_t dim_embed_div = params.dim_embed / EMB_AVX2_PS_COUNT_V2;
    const int32_t dim_embed_rem = params.dim_embed % EMB_AVX2_PS_COUNT_V2;

#pragma omp parallel num_threads(params.num_threads)
    {
        int nthr     = omp_get_num_threads();
        int ithr     = omp_get_thread_num();

        int sbuf_offset     = ithr*EMB_SCRATCHPAD_LEN_V2/nthr;
        indices_type* sidx  = static_cast<indices_type*>(params.pad_indices);
        sidx               += sbuf_offset;

        #pragma omp for
        for (auto oi = 0; oi < offset_size; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offset_size -1) ? offsets[oi+1] : indices_size;

            // preprocess indices
            int last = 0;
            if (padding_idx >= 0) {
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padding_idx) {
                        sidx[last++] = indices[i]*dim_embed;
                    }
                }
            }
            else {
                for (auto i = ofirst; i < olast; ++i) {
                    sidx[last++] = indices[i]*dim_embed;
                }
            }

            float *dst_base        = dst + (oi * dim_embed);
            float *in_base         = const_cast<float *>(input);
            float *dst_ptr         = dst_base;
            int    shift           = 0;
            float  dn              = 1.0/(float)(last);


            for(auto j = 0; j < dim_embed_div; ++j) {
                __m256 sum     = _mm256_setzero_ps();

                for(auto i = 0; i < last; ++i) {
                    auto in_ptr = in_base + sidx[i];
                    __m256 aa   = _mm256_loadu_ps(in_ptr);
                    sum         = _mm256_add_ps(sum, aa);
                }

                __m256 ddn    = _mm256_set1_ps(dn);
                sum           = _mm256_mul_ps(sum, ddn);
                _mm256_storeu_ps(dst_ptr, sum);

                shift   += EMB_AVX2_PS_COUNT_V2;
                in_base = const_cast<float *>(input)  + shift;
                dst_ptr = dst_base + shift;
            }

            // remaining vector
            if (dim_embed_rem) {
                auto in_ptr  = in_base + sidx[0];
                for(auto j = 0; j < dim_embed_rem; ++j) {
                    dst_ptr[j] = in_ptr[j];
                }

                for(auto i = 1; i < last; ++i) {
                    auto in_ptr  = in_base + sidx[i];
                    for(auto j = 0; j < dim_embed_rem; ++j) {
                        dst_ptr[j] += in_ptr[j];
                    }
                }

                for(auto j = 0; j < dim_embed_rem; ++j) {
                    dst_ptr[j] *= dn;
                }
            }
        } //for oi
    } // omp parallel

    return status::success;
}

/*
 * mean with weights but without padding index
 */
template<>
status_t
avx2_embedding_bag_v2_t<f32>::avx2_mean_wt(const emb_params_v2_t &params) const {
    return status::unimplemented;
}

/*
 * max without weights or padding index
 */
template<>
status_t
avx2_embedding_bag_v2_t<f32>::avx2_max(const emb_params_v2_t &params) const {

    const input_type   *input   = static_cast<input_type *>(params.input);
    const indices_type *indices = static_cast<indices_type *>(params.indices);
    const offsets_type *offsets = static_cast<offsets_type *>(params.offsets);
    dst_type     *dst           = static_cast<dst_type *>(params.dst);

    const int32_t &dim_embed        = params.dim_embed;
    const int32_t &indices_size     = params.indices_size;
    const int32_t &offset_size      = params.offset_size;
    const int32_t &dst_size         = params.dst_size;
    const indices_type &padding_idx = params.padding_idx;

    const int32_t dim_embed_div = params.dim_embed / EMB_AVX2_PS_COUNT_V2;
    const int32_t dim_embed_rem = params.dim_embed % EMB_AVX2_PS_COUNT_V2;

#pragma omp parallel num_threads(params.num_threads)
    {
        int nthr     = omp_get_num_threads();
        int ithr     = omp_get_thread_num();

        int sbuf_offset     = ithr*EMB_SCRATCHPAD_LEN_V2/nthr;
        indices_type* sidx  = static_cast<indices_type*>(params.pad_indices);
        sidx               += sbuf_offset;

        #pragma omp for
        for (auto oi = 0; oi < offset_size; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offset_size -1) ? offsets[oi+1] : indices_size;

            // preprocess indices
            int last = 0;
            if (padding_idx >= 0) {
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padding_idx) {
                        sidx[last++] = indices[i]*dim_embed;
                    }
                }
            }
            else {
                for (auto i = ofirst; i < olast; ++i) {
                    sidx[last++] = indices[i]*dim_embed;
                }
            }

            float *dst_base        = dst + (oi * dim_embed);
            float *in_base         = const_cast<float *>(input);
            float *dst_ptr         = dst_base;
            int    shift           = 0;

            for(auto j = 0; j < dim_embed_div; ++j) {
                auto in_ptr = in_base + sidx[0];
                __m256 mx   = _mm256_loadu_ps(in_ptr);

                for(auto i = 1; i < last; ++i) {
                    auto in_ptr = in_base + sidx[i];
                    __m256 aa   = _mm256_loadu_ps(in_ptr);
                    mx          = _mm256_max_ps(mx, aa);
                }

                _mm256_storeu_ps(dst_ptr, mx);

                shift   += EMB_AVX2_PS_COUNT_V2;
                in_base = const_cast<float *>(input)  + shift;
                dst_ptr = dst_base + shift;
            }

            // remaining vector
            if (dim_embed_rem) {
                auto in_ptr  = in_base + sidx[0];
                for(auto j = 0; j < dim_embed_rem; ++j) {
                    dst_ptr[j] = in_ptr[j];
                }

                for(auto i = 1; i < last; ++i) {
                    auto in_ptr  = in_base + sidx[i];
                    for(auto j = 0; j < dim_embed_rem; ++j) {
                        dst_ptr[j] = dst_ptr[j] > in_ptr[j] ? dst_ptr[j]:
                                                              in_ptr[j];
                    }
                }
            }
        } //for oi
    } // omp parallel

    return status::success;
}

/*
 * max with weights
 */
template<>
status_t
avx2_embedding_bag_v2_t<f32>::avx2_max_wt(const emb_params_v2_t &params) const {
    return status::unimplemented;
}

template struct avx2_embedding_bag_v2_t<f32>;

} //namespace cpu
}
}
