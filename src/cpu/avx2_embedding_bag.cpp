/*******************************************************************************
* Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
*
*******************************************************************************/

#include "common/c_types_map.hpp"
#include "common/zendnn_thread.hpp"
#include "common/zendnn_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/avx2_embedding_bag.hpp"

namespace zendnn {
namespace impl {
namespace cpu {

using namespace data_type;

template<data_type_t data_type>
status_t
avx2_embedding_bag_t<data_type>::execute(const exec_ctx_t &ctx) const {

    // initialize
    emb_params_t  params;
    pre_process(ctx, params);

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
avx2_embedding_bag_t<data_type>::pre_process(const exec_ctx_t &ctx,
        emb_params_t &params) const {

    status_t status = status::success;

    // get algorithm params
    params.is_weights  = pd()->desc()->is_weights;
    params.padding_idx = pd()->desc()->padding_idx;

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

    auto scratchpad          = ctx.get_scratchpad_grantor();
    params.scratchpad_indices
      = scratchpad.template get(key_embed_bag_indices);
    params.scratchpad_weights
      = scratchpad.template get(key_embed_bag_weights);

    // initialize output to zero
    params.dst_size     = dst_mdw.nelems();

    return status;
}

/*
 * sum without weights
 */
template<>
status_t
avx2_embedding_bag_t<f32>::avx2_sum(const emb_params_t &params) const {

    const input_type   *input   = static_cast<input_type *>(params.input);
    const indices_type *indices = static_cast<indices_type *>(params.indices);
    const offsets_type *offsets = static_cast<offsets_type *>(params.offsets);
    dst_type     *dst           = static_cast<dst_type *>(params.dst);

    const int32_t &dim_embed        = params.dim_embed;
    const int32_t &indices_size     = params.indices_size;
    const int32_t &offset_size      = params.offset_size;
    const int32_t &dst_size         = params.dst_size;
    const indices_type &padding_idx = params.padding_idx;

    const int32_t dim_embed_div = params.dim_embed / AVX2_PS_COUNT;
    const int32_t dim_embed_rem = params.dim_embed % AVX2_PS_COUNT;

    // scratchpad buffers
    indices_type* scratchpad_indices
      = static_cast<indices_type *>(params.scratchpad_indices);

    // zero fill output tensor
    std::fill(dst, (dst+ dst_size), 0);

    parallel_nd(offset_size,
    [=](dim_t thrd) {
        // compute start and end indices of the bag
        auto first = offsets[thrd];
        auto last  = (thrd < (offset_size-1)) ?
                     offsets[thrd +1] : indices_size;

        // preprocess indices and weights
        if (padding_idx >= 0) {
            auto next = first;
            for (auto i = first; i < last; ++i) {
                if (indices[i] != padding_idx) {
                    scratchpad_indices[next++] = indices[i]*dim_embed;
                }
            }
            last = next;
        }
        else {
            for (auto i = first; i < last; ++i) {
                scratchpad_indices[i] = indices[i]*dim_embed;
            }
        }

        float *dst_base        = dst + (thrd * dim_embed);
        float *in_base         = const_cast<float *>(input);
        float *dst_ptr         = dst_base;
        int    shift           = 0;

        for(auto j = 0; j < dim_embed_div; ++j) {
            __m256 sum     = _mm256_setzero_ps();

            for(auto i = first; i < last; ++i) {
                auto in_ptr = in_base + scratchpad_indices[i];
                __m256 aa   = _mm256_loadu_ps(in_ptr);
                sum         = _mm256_add_ps(aa, sum);
            }

            _mm256_storeu_ps(dst_ptr, sum);

            shift   += AVX2_PS_COUNT;
            in_base = const_cast<float *>(input)  + shift;
            dst_ptr = dst_base + shift;
        }

        // remaining vector
        for(auto i = first; i < last; ++i) {
            auto in_ptr  = in_base + scratchpad_indices[i];
            for(auto j = 0; j < dim_embed_rem; ++j) {
                dst_ptr[j] += in_ptr[j];
            }
        }
    });

    return status::success;
}

/*
 * sum with weights
 */
template<>
status_t
avx2_embedding_bag_t<f32>::avx2_sum_wt(const emb_params_t &params) const {

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

    const int32_t dim_embed_div = params.dim_embed / AVX2_PS_COUNT;
    const int32_t dim_embed_rem = params.dim_embed % AVX2_PS_COUNT;

    // scratchpad buffers
    indices_type* scratchpad_indices
      = static_cast<indices_type *>(params.scratchpad_indices);
    input_type* scratchpad_weights
      = static_cast<input_type *>(params.scratchpad_weights);

    // zero fill output tensor
    std::fill(dst, (dst+ dst_size), 0);

    parallel_nd(offset_size,
    [=](dim_t thrd) {
        // compute start and end indices of the bag
        auto first = offsets[thrd];
        auto last  = (thrd < (offset_size-1)) ?
                     offsets[thrd +1] : indices_size;

        // preprocess indices and weights
        if (padding_idx >= 0) {
            auto next = first;
            for (auto i = first; i < last; ++i) {
                if (indices[i] != padding_idx) {
                    scratchpad_indices[next]   = indices[i]*dim_embed;
                    scratchpad_weights[next++] = weights[i];
                }
            }
            last = next;
        }
        else {
            for (auto i = first; i < last; ++i) {
                scratchpad_indices[i] = indices[i]*dim_embed;
            }
        }

        auto wts = (padding_idx >= 0) ? scratchpad_weights : weights;

        float *dst_base        = dst + (thrd * dim_embed);
        float *in_base         = const_cast<float *>(input);
        float *dst_ptr         = dst_base;
        int    shift           = 0;

        for(auto j = 0; j < dim_embed_div; ++j) {
            __m256 sum     = _mm256_setzero_ps();

            for(auto i = first; i < last; ++i) {
                auto in_ptr = in_base + scratchpad_indices[i];
                __m256 aa = _mm256_loadu_ps(in_ptr);
                __m256 bb = _mm256_set1_ps(wts[i]);
                sum       = _mm256_fmadd_ps(aa, bb, sum);
            }

            _mm256_storeu_ps(dst_ptr, sum);

            shift   += AVX2_PS_COUNT;
            in_base = const_cast<float *>(input)  + shift;
            dst_ptr = dst_base + shift;
        }

        // remaining vector
        for(auto i = first; i < last; ++i) {
            auto in_ptr  = in_base + scratchpad_indices[i];
            for(auto j = 0; j < dim_embed_rem; ++j) {
                dst_ptr[j] += wts[i]*in_ptr[j];
            }
        }
    });

    return status::success;
}

/*
 * mean without weights or padding index
 */
template<>
status_t
avx2_embedding_bag_t<f32>::avx2_mean(const emb_params_t &params) const {

    const input_type   *input   = static_cast<input_type *>(params.input);
    const indices_type *indices = static_cast<indices_type *>(params.indices);
    const offsets_type *offsets = static_cast<offsets_type *>(params.offsets);
    dst_type     *dst           = static_cast<dst_type *>(params.dst);

    const int32_t &dim_embed        = params.dim_embed;
    const int32_t &indices_size     = params.indices_size;
    const int32_t &offset_size      = params.offset_size;
    const int32_t &dst_size         = params.dst_size;
    const indices_type &padding_idx = params.padding_idx;

    const int32_t dim_embed_div = params.dim_embed / AVX2_PS_COUNT;
    const int32_t dim_embed_rem = params.dim_embed % AVX2_PS_COUNT;

    // scratchpad buffers
    indices_type* scratchpad_indices
      = static_cast<indices_type *>(params.scratchpad_indices);

    // zero fill output tensor
    std::fill(dst, (dst+ dst_size), 0);

    parallel_nd(offset_size,
    [=](dim_t thrd) {
        // compute start and end indices of the bag
        auto first = offsets[thrd];
        auto last  = (thrd < (offset_size-1)) ?
                     offsets[thrd +1] : indices_size;

        // preprocess indices and weights
        if (padding_idx >= 0) {
            auto next = first;
            for (auto i = first; i < last; ++i) {
                if (indices[i] != padding_idx) {
                    scratchpad_indices[next++] = indices[i]*dim_embed;
                }
            }
            last = next;
        }
        else {
            for (auto i = first; i < last; ++i) {
                scratchpad_indices[i] = indices[i]*dim_embed;
            }
        }

        float   *dst_base        = dst + (thrd * dim_embed);
        float   *in_base         = const_cast<float *>(input);
        float   *dst_ptr         = dst_base;
        int      shift           = 0;
        float    dn              = 1.0/(float)(last - first);

        for(auto j = 0; j < dim_embed_div; ++j) {
            __m256 sum     = _mm256_setzero_ps();

            for(auto i = first; i < last; ++i) {
	        auto in_ptr = in_base + scratchpad_indices[i];
                __m256 aa   = _mm256_loadu_ps(in_ptr);
                sum         = _mm256_add_ps(aa, sum);
            }

            __m256 ddn    = _mm256_set1_ps(dn);
            sum           = _mm256_mul_ps(sum, ddn);
            _mm256_storeu_ps(dst_ptr, sum);

            shift   += AVX2_PS_COUNT;
            in_base = const_cast<float *>(input)  + shift;
            dst_ptr = dst_base + shift;
        }

        // remaining vector
        for(auto i = first; i < last; ++i) {
            auto in_ptr  = in_base + scratchpad_indices[i];
            for(auto j = 0; j < dim_embed_rem; ++j) {
                dst_ptr[j] += in_ptr[j];
            }
        }
        for(auto j = 0; j < dim_embed_rem; ++j) {
            dst_ptr[j] *= dn;
        }
    });

    return status::success;
}

/*
 * mean with weights but without padding index
 */
template<>
status_t
avx2_embedding_bag_t<f32>::avx2_mean_wt(const emb_params_t &params) const {

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

    const int32_t dim_embed_div = params.dim_embed / AVX2_PS_COUNT;
    const int32_t dim_embed_rem = params.dim_embed % AVX2_PS_COUNT;

    // scratchpad buffers
    indices_type* scratchpad_indices
      = static_cast<indices_type *>(params.scratchpad_indices);
    input_type* scratchpad_weights
      = static_cast<input_type *>(params.scratchpad_weights);

    // zero fill output tensor
    std::fill(dst, (dst+ dst_size), 0);

    parallel_nd(offset_size,
    [=](dim_t thrd) {
        // compute start and end indices of the bag
        auto first = offsets[thrd];
        auto last  = (thrd < (offset_size-1)) ?
                     offsets[thrd +1] : indices_size;

        // preprocess indices and weights
        float dn = 0;
        if (padding_idx >= 0) {
            auto next = first;
            for (auto i = first; i < last; ++i) {
                if (indices[i] != padding_idx) {
                    scratchpad_indices[next]   = indices[i]*dim_embed;
                    scratchpad_weights[next++] = weights[i];
                    dn += weights[i];
                }
            }
            last = next;
        }
        else {
            for (auto i = first; i < last; ++i) {
                scratchpad_indices[i] = indices[i]*dim_embed;
                dn += weights[i];
            }
        }
        dn = 1/dn;

        auto wts = (padding_idx >= 0) ? scratchpad_weights : weights;

        float *dst_base        = dst + (thrd * dim_embed);
        float *in_base         = const_cast<float *>(input);
        float *dst_ptr         = dst_base;
        int    shift           = 0;

        for(auto j = 0; j < dim_embed_div; ++j) {
            __m256 sum     = _mm256_setzero_ps();

            for(auto i = first; i < last; ++i) {
                auto in_ptr = in_base + scratchpad_indices[i];
                __m256 aa = _mm256_loadu_ps(in_ptr);
                __m256 bb = _mm256_set1_ps(wts[i]);
                sum       = _mm256_fmadd_ps(aa, bb, sum);
            }

            __m256 ddn    = _mm256_set1_ps(dn);
            sum           = _mm256_mul_ps(sum, ddn);
            _mm256_storeu_ps(dst_ptr, sum);

            shift   += AVX2_PS_COUNT;
            in_base = const_cast<float *>(input)  + shift;
            dst_ptr = dst_base + shift;
        }

        // remaining vector
        for(auto i = first; i < last; ++i) {
            auto in_ptr  = in_base + scratchpad_indices[i];
            for(auto j = 0; j < dim_embed_rem; ++j) {
                dst_ptr[j] += wts[i]*in_ptr[j];
            }
        }
        for(auto j = 0; j < dim_embed_rem; ++j) {
            dst_ptr[j] *= dn;
        }
    });

    return status::success;
}

/*
 * max without weights or padding index
 */
template<>
status_t
avx2_embedding_bag_t<f32>::avx2_max(const emb_params_t &params) const {

    const input_type   *input   = static_cast<input_type *>(params.input);
    const indices_type *indices = static_cast<indices_type *>(params.indices);
    const offsets_type *offsets = static_cast<offsets_type *>(params.offsets);
    dst_type     *dst           = static_cast<dst_type *>(params.dst);

    const int32_t &dim_embed        = params.dim_embed;
    const int32_t &indices_size     = params.indices_size;
    const int32_t &offset_size      = params.offset_size;
    const int32_t &dst_size         = params.dst_size;
    const indices_type &padding_idx = params.padding_idx;

    const int32_t dim_embed_div = params.dim_embed / AVX2_PS_COUNT;
    const int32_t dim_embed_rem = params.dim_embed % AVX2_PS_COUNT;

    // scratchpad buffers
    indices_type* scratchpad_indices
      = static_cast<indices_type *>(params.scratchpad_indices);

    // zero fill output tensor
    std::fill(dst, (dst+ dst_size), 0);

    parallel_nd(offset_size,
    [=](dim_t thrd) {
        // compute start and end indices of the bag
        auto first = offsets[thrd];
        auto last  = (thrd < (offset_size-1)) ?
                     offsets[thrd +1] : indices_size;

        // preprocess indices and weights
        if (padding_idx >= 0) {
            auto next = first;
            for (auto i = first; i < last; ++i) {
                if (indices[i] != padding_idx) {
                    scratchpad_indices[next++]   = indices[i]*dim_embed;
                }
            }
            last = next;
        }
        else {
            for (auto i = first; i < last; ++i) {
                scratchpad_indices[i] = indices[i]*dim_embed;
            }
        }

        float *dst_base        = dst + (thrd * dim_embed);
        float *in_base         = const_cast<float *>(input);
        float *dst_ptr         = dst_base;
        int    shift           = 0;

        for(auto j = 0; j < dim_embed_div; ++j) {
            auto in_ptr    = in_base + scratchpad_indices[first];
            __m256 mx      = _mm256_loadu_ps(in_ptr);

            for(auto i = first + 1; i < last; ++i) {
                auto in_ptr = in_base + scratchpad_indices[i];
                __m256 aa   = _mm256_loadu_ps(in_ptr);
                mx          = _mm256_max_ps(mx, aa);
            }

            _mm256_storeu_ps(dst_ptr, mx);

            shift   += AVX2_PS_COUNT;
            in_base = const_cast<float *>(input)  + shift;
            dst_ptr = dst_base + shift;
        }

        // remaining vector
        auto in_ptr  = in_base + scratchpad_indices[first];
        for(auto j = 0; j < dim_embed_rem; ++j) {
            dst_ptr[j] = in_ptr[j];
        }

        for(auto i = first + 1; i < last; ++i) {
            auto in_ptr  = in_base + scratchpad_indices[i];
            for(auto j = 0; j < dim_embed_rem; ++j) {
                if(in_ptr[j] > dst_ptr[j])
                    dst_ptr[j] = in_ptr[j];
            }
        }
    });

    return status::success;
}

/*
 * max with weights
 */
template<>
status_t
avx2_embedding_bag_t<f32>::avx2_max_wt(const emb_params_t &params) const {

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

    const int32_t dim_embed_div = params.dim_embed / AVX2_PS_COUNT;
    const int32_t dim_embed_rem = params.dim_embed % AVX2_PS_COUNT;

    // scratchpad buffers
    indices_type* scratchpad_indices
      = static_cast<indices_type *>(params.scratchpad_indices);
    input_type* scratchpad_weights
      = static_cast<input_type *>(params.scratchpad_weights);

    // zero fill output tensor
    std::fill(dst, (dst+ dst_size), 0);

    parallel_nd(offset_size,
    [=](dim_t thrd) {
        // compute start and end indices of the bag
        auto first = offsets[thrd];
        auto last  = (thrd < (offset_size-1)) ?
                     offsets[thrd +1] : indices_size;

        // preprocess indices and weights
        if (padding_idx >= 0) {
            auto next = first;
            for (auto i = first; i < last; ++i) {
                if (indices[i] != padding_idx) {
                    scratchpad_indices[next]   = indices[i]*dim_embed;
                    scratchpad_weights[next++] = weights[i];
                }
            }
            last = next;
        }
        else {
            for (auto i = first; i < last; ++i) {
                scratchpad_indices[i] = indices[i]*dim_embed;
            }
        }

        auto wts = (padding_idx >= 0) ? scratchpad_weights : weights;

        float *dst_base        = dst + (thrd * dim_embed);
        float *in_base         = const_cast<float *>(input);
        float *dst_ptr         = dst_base;
        int    shift           = 0;

        for(auto j = 0; j < dim_embed_div; ++j) {
            auto in_ptr    = in_base + scratchpad_indices[first];
            __m256 mx      = _mm256_loadu_ps(in_ptr);
            __m256 bb      = _mm256_set1_ps(wts[first]);
            mx             = _mm256_mul_ps(mx,bb);

            for(auto i = first +1; i < last; ++i) {
                auto in_ptr = in_base + scratchpad_indices[i];
                __m256 aa = _mm256_loadu_ps(in_ptr);
                bb        = _mm256_set1_ps(wts[i]);
                __m256 cc = _mm256_mul_ps(aa,bb);
                mx        = _mm256_max_ps(mx, cc);
            }

            _mm256_storeu_ps(dst_ptr, mx);

            shift   += AVX2_PS_COUNT;
            in_base = const_cast<float *>(input)  + shift;
            dst_ptr = dst_base + shift;
        }

        // remaining vector
        auto in_ptr  = in_base + scratchpad_indices[first];
        for(auto j = 0; j < dim_embed_rem; ++j) {
            dst_ptr[j] = wts[first]*in_ptr[j];
        }

        for(auto i = first +1; i < last; ++i) {
            auto in_ptr  = in_base + scratchpad_indices[i];
            for(auto j = 0; j < dim_embed_rem; ++j) {
                if(wts[i]*in_ptr[j] > dst_ptr[j])
                    dst_ptr[j] = wts[i]*in_ptr[j];
            }
        }
    });

    return status::success;
}

template struct avx2_embedding_bag_t<f32>;

} //namespace cpu
}
}
