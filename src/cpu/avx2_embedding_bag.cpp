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
#include "cpu/zen_avx_utils.hpp"
#include "zendnn_logging.hpp"
#include "cpu/avx2_embedding_bag.hpp"

#include <vector>

namespace zendnn {
namespace impl {
namespace cpu {

using namespace data_type;

template<data_type_t data_type>
status_t
avx2_embedding_bag_t<data_type>::execute(const exec_ctx_t &ctx) const {

#if ZENDNN_CPU_THREADING_RUNTIME != ZENDNN_RUNTIME_OMP
    assert(!"threading env need to be omp for embedding_bag");
#endif

    status_t status;

    // initialize
    emb_params_t  params;

    status = pre_process(ctx, params);
    if (status != status::success)
        return status;

    auto  algo                = pd()->desc()->alg_kind;
    bool  is_weights          = pd()->desc()->is_weights;

    switch(algo) {
    case alg_kind::embedding_bag_sum:
        return is_weights ? avx2_sum_wt(params) : avx2_sum(params);
    case alg_kind::embedding_bag_mean:
        return avx2_mean(params);
    case alg_kind::embedding_bag_max:
        return avx2_max(params);
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
    params.padidx       = pd()->desc()->padding_idx;
    params.nthr         = pd()->desc()->num_threads;

    // get the tensors
    params.input =
        static_cast<void *>(ctx.host_ptr(ZENDNN_ARG_SRC_0));
    params.indices =
        static_cast<void *>(ctx.host_ptr(ZENDNN_ARG_SRC_1));
    params.offsets =
        static_cast<void *>(ctx.host_ptr(ZENDNN_ARG_SRC_2));

    params.weights = nullptr;
    if(pd()->desc()->is_weights) {
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
    params.width             = input_dims[1];

    params.offset_size       = offsets_mdw.nelems();
    params.indices_size      = indices_mdw.nelems();

    // check if aligned access for avx instructions will be safe
    if ((128 == params.width) || (64 == params.width)) {
        if (ALIGNED_AVX2_UNSAFE(params.input)) {
            zendnnError(ZENDNN_ALGOLOG, "embedding tables not aligned for avx instructions.");
            return status::runtime_error;
        }
    }

    // get rid of excess omp threads if any
    params.dst_size     = dst_mdw.nelems();

    if (params.offset_size < params.nthr)
        params.nthr = params.offset_size;

    return status;
}

/*
 * sum without weights
 */
template<>
status_t
avx2_embedding_bag_t<f32>::avx2_sum(const emb_params_t &params) const {

    float        const *input    = static_cast<float *>(params.input);
    indices_type       *indices  = static_cast<indices_type *>(params.indices);
    offsets_type       *offsets  = static_cast<offsets_type *>(params.offsets);
    dst_type           *dst      = static_cast<dst_type *>(params.dst);

    const int32_t      &width    = params.width;
    const int32_t      &indsz    = params.indices_size;
    const int32_t      &offsz    = params.offset_size;
    const int32_t      &dstsz    = params.dst_size;
    const indices_type &padidx   = params.padidx;
    const uint32_t     &nthr     = params.nthr;

    // fast path for common cases of width 128 and 64
    if (128 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps128 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx)
                        sum.fetch_add_ps(input + indices[i]*width);
                }
                sum.store_ps(dst + oi*width);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps128 sum;
                for (auto i = ofirst; i < olast; ++i)
                    sum.fetch_add_ps(input + indices[i]*width);
                sum.store_ps(dst + oi*width);
            }
        }

        return status::success;
    }

    if (64 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps64 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx)
                        sum.fetch_add_ps(input + indices[i]*width);
                }
                sum.store_ps(dst + oi*width);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps64 sum;
                for (auto i = ofirst; i < olast; ++i)
                    sum.fetch_add_ps(input + indices[i]*width);
                sum.store_ps(dst + oi*width);
            }
        }
        return status::success;
    }

    // slow path, no avx instructions
    if (padidx >= 0) {
        #pragma omp parallel for num_threads(nthr) //proc_bind(master)
        for (auto oi = 0; oi < offsz; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

            std::vector<dst_type> sum(width,0.0);
            for (auto i = ofirst; i < olast; ++i) {
                if (indices[i] != padidx) {
                    for (auto j = 0; j < width; ++j)
                        sum[j] += input[j + indices[i]*width];
                }
            }
            for (auto j = 0; j < width; ++j)
                dst[j + oi*width] = sum[j];
        }
    }
    else {
        #pragma omp parallel for num_threads(nthr) //proc_bind(master)
        for (auto oi = 0; oi < offsz; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

            std::vector<dst_type> sum(width,0.0);
            for (auto i = ofirst; i < olast; ++i)
                for (auto j = 0; j < width; ++j)
                    sum[j] += input[j + indices[i]*width];
            for (auto j = 0; j < width; ++j)
                dst[j + oi*width] = sum[j];
        }
    }

    return status::success;
}

/*
 * sum with weights
 */

template<>
status_t
avx2_embedding_bag_t<f32>::avx2_sum_wt(const emb_params_t &params) const {

    float        const *input    = static_cast<float *>(params.input);
    float        const *wts      = static_cast<float *>(params.weights);
    indices_type       *indices  = static_cast<indices_type *>(params.indices);
    offsets_type       *offsets  = static_cast<offsets_type *>(params.offsets);
    dst_type           *dst      = static_cast<dst_type *>(params.dst);

    const int32_t      &width    = params.width;
    const int32_t      &indsz    = params.indices_size;
    const int32_t      &offsz    = params.offset_size;
    const int32_t      &dstsz    = params.dst_size;
    const indices_type &padidx   = params.padidx;
    const uint32_t     &nthr     = params.nthr;

    // fast path for common cases of width 128 and 64
    if (128 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps128 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx)
                        sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                }
                sum.store_ps(dst + oi*width);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps128 sum;
                for (auto i = ofirst; i < olast; ++i)
                    sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                sum.store_ps(dst + oi*width);
            }
        }

        return status::success;
    }

    if (64 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps64 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx)
                        sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                }
                sum.store_ps(dst + oi*width);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps64 sum;
                for (auto i = ofirst; i < olast; ++i)
                    sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                sum.store_ps(dst + oi*width);
            }
        }
        return status::success;
    }

    // slow path, no avx instructions
    if (padidx >= 0) {
        #pragma omp parallel for num_threads(nthr) //proc_bind(master)
        for (auto oi = 0; oi < offsz; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

            std::vector<dst_type> sum(width,0.0);
            for (auto i = ofirst; i < olast; ++i) {
                if (indices[i] != padidx) {
                    for (auto j = 0; j < width; ++j)
                        sum[j] += wts[i]*input[j + indices[i]*width];
                }
            }
            for (auto j = 0; j < width; ++j)
                dst[j + oi*width] = sum[j];
        }
    }
    else {
        #pragma omp parallel for num_threads(nthr) //proc_bind(master)
        for (auto oi = 0; oi < offsz; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

            std::vector<dst_type> sum(width,0.0);
            for (auto i = ofirst; i < olast; ++i)
                for (auto j = 0; j < width; ++j)
                    sum[j] += wts[i]*input[j + indices[i]*width];
            for (auto j = 0; j < width; ++j)
                dst[j + oi*width] = sum[j];
        }
    }

    return status::success;
}

/*
 * mean without weights
 */
template<>
status_t
avx2_embedding_bag_t<f32>::avx2_mean(const emb_params_t &params) const {

    float        const *input    = static_cast<float *>(params.input);
    indices_type       *indices  = static_cast<indices_type *>(params.indices);
    offsets_type       *offsets  = static_cast<offsets_type *>(params.offsets);
    dst_type           *dst      = static_cast<dst_type *>(params.dst);

    const int32_t      &width    = params.width;
    const int32_t      &indsz    = params.indices_size;
    const int32_t      &offsz    = params.offset_size;
    const int32_t      &dstsz    = params.dst_size;
    const indices_type &padidx   = params.padidx;
    const uint32_t     &nthr     = params.nthr;

    // fast path for common cases of width 128 and 64
    if (128 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps128 sum;
                int32_t         count = 0;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        count++;
                        sum.fetch_add_ps(input + indices[i]*width);
                    }
                }
                sum.scale_store_ps(dst + oi*width, (1.0/float(count)));
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps128 sum;
                for (auto i = ofirst; i < olast; ++i)
                    sum.fetch_add_ps(input + indices[i]*width);

                sum.scale_store_ps(dst + oi*width, (1.0/float(olast - ofirst)));
            }
        }

        return status::success;
    }

    if (64 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps64  sum;
                int32_t         count = 0;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        count++;
                        sum.fetch_add_ps(input + indices[i]*width);
                    }
                }
                sum.scale_store_ps(dst + oi*width, (1.0/float(count)));
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps64 sum;
                for (auto i = ofirst; i < olast; ++i)
                    sum.fetch_add_ps(input + indices[i]*width);

                sum.scale_store_ps(dst + oi*width, (1.0/float(olast - ofirst)));
            }
        }

        return status::success;
    }

    // slow path, no avx instructions
    if (padidx >= 0) {
        #pragma omp parallel for num_threads(nthr) //proc_bind(master)
        for (auto oi = 0; oi < offsz; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

            std::vector<dst_type> sum(width,0.0);
            int32_t               count = 0;
            for (auto i = ofirst; i < olast; ++i) {
                if (indices[i] != padidx) {
                    count++;
                    for (auto j = 0; j < width; ++j)
                        sum[j] += input[j + indices[i]*width];
                }
            }

            float dn = 1.0/float(count);
            for (auto j = 0; j < width; ++j)
                dst[j + oi*width] = dn*sum[j];
        }
    }
    else {
        #pragma omp parallel for num_threads(nthr) //proc_bind(master)
        for (auto oi = 0; oi < offsz; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

            std::vector<dst_type> sum(width,0.0);
            for (auto i = ofirst; i < olast; ++i)
                for (auto j = 0; j < width; ++j)
                    sum[j] += input[j + indices[i]*width];

            float dn = 1.0/float(olast - ofirst);
            for (auto j = 0; j < width; ++j)
                dst[j + oi*width] = dn*sum[j];
        }
    }

    return status::success;
}

/*
 * max without weights
 */
template<>
status_t
avx2_embedding_bag_t<f32>::avx2_max(const emb_params_t &params) const {

    float        const *input    = static_cast<float *>(params.input);
    indices_type       *indices  = static_cast<indices_type *>(params.indices);
    offsets_type       *offsets  = static_cast<offsets_type *>(params.offsets);
    dst_type           *dst      = static_cast<dst_type *>(params.dst);

    const int32_t      &width    = params.width;
    const int32_t      &indsz    = params.indices_size;
    const int32_t      &offsz    = params.offset_size;
    const int32_t      &dstsz    = params.dst_size;
    const indices_type &padidx   = params.padidx;
    const uint32_t     &nthr     = params.nthr;

    // fast path for common cases of width 128 and 64
    if (128 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps128 sum;
                int32_t         nfirst = ofirst;
                while (nfirst < olast) {
                    if (nfirst != padidx) {
                        sum.load_ps(input + indices[nfirst]*width);
                        break;
                    }
                    nfirst++;
                }

                for (auto i = nfirst +1; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        sum.fetch_max_ps(input + indices[i]*width);
                    }
                }
                sum.store_ps(dst + oi*width);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps128 sum;
                sum.load_ps(input + indices[ofirst]*width);
                for (auto i = ofirst+1; i < olast; ++i)
                    sum.fetch_max_ps(input + indices[i]*width);

                sum.store_ps(dst + oi*width);
            }
        }

        return status::success;
    }

    if (64 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps64 sum;
                int32_t        nfirst = ofirst;
                while (nfirst < olast) {
                    if (nfirst != padidx) {
                        sum.load_ps(input + indices[nfirst]*width);
                        break;
                    }
                    nfirst++;
                }

                for (auto i = nfirst +1; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        sum.fetch_max_ps(input + indices[i]*width);
                    }
                }
                sum.store_ps(dst + oi*width);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

                zenmm_ext_ps64 sum;
                sum.load_ps(input + indices[ofirst]*width);
                for (auto i = ofirst+1; i < olast; ++i)
                    sum.fetch_max_ps(input + indices[i]*width);

                sum.store_ps(dst + oi*width);
            }
        }

        return status::success;
    }

    // slow path, no avx instructions
    if (padidx >= 0) {
        #pragma omp parallel for num_threads(nthr) //proc_bind(master)
        for (auto oi = 0; oi < offsz; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

            std::vector<dst_type> sum(width,0.0);
            int32_t               nfirst = ofirst;
            while (nfirst < olast) {
                if (nfirst != padidx) {
                    for (auto j = 0; j < width; ++j)
                        sum[j]  = input[j + indices[nfirst]*width];
                    break;
                }
                nfirst++;
            }

            for (auto i = nfirst+1; i < olast; ++i) {
                if (indices[i] != padidx) {
                    for (auto j = 0; j < width; ++j)
                        if (sum[j]  < input[j + indices[i]*width])
                            sum[j] = input[j + indices[i]*width];
                }
            }

            for (auto j = 0; j < width; ++j)
                dst[j + oi*width] = sum[j];
        }
    }
    else {
        #pragma omp parallel for num_threads(nthr) //proc_bind(master)
        for (auto oi = 0; oi < offsz; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;

            std::vector<dst_type> sum(width,0.0);
            for (auto j = 0; j < width; ++j)
                sum[j]  = input[j + indices[ofirst]*width];

            for (auto i = ofirst+1; i < olast; ++i)
                for (auto j = 0; j < width; ++j)
                    if (sum[j]  < input[j + indices[i]*width])
                        sum[j] = input[j + indices[i]*width];

            for (auto j = 0; j < width; ++j)
                dst[j + oi*width] = sum[j];
        }
    }

    return status::success;
}


template struct avx2_embedding_bag_t<f32>;

} //namespace cpu
}
}
