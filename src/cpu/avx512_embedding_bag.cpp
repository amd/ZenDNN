
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

#include "common/c_types_map.hpp"
#include "common/zendnn_thread.hpp"
#include "common/zendnn_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/simple_q10n.hpp"
#include "cpu/zen_avx512_utils.hpp"
#include "zendnn_logging.hpp"
#include "cpu/avx512_embedding_bag.hpp"
#include <vector>

#define PREFETCH_EN         1
#define PREFETCH_DISTANCE   0

namespace zendnn {
namespace impl {
namespace cpu {
using namespace data_type;
template<>
inline void avx512_embedding_bag_t<f32>::ebvec_prefetch(float const *input,
        indices_type *indices,
        const int32_t width, offsets_type *offsets, const int32_t index,
        const int32_t offsz, const int32_t indsz) const {

    auto prefetch_distance = PREFETCH_DISTANCE;
    //if((index+PREFETCH_DISTANCE)>=offsz)
    //  prefetch_distance = 0;

    auto ofirst = offsets[index+prefetch_distance];
    auto olast  = index < (offsz -1) ? offsets[index+prefetch_distance+1] : indsz;

    for (auto i = ofirst; i < olast; ++i) {
        float const *prefetch_addr = input + (indices[i]*width);
        _mm_prefetch(prefetch_addr, _MM_HINT_T0);
    }
}

template<data_type_t data_type>
status_t
avx512_embedding_bag_t<data_type>::execute(const exec_ctx_t &ctx) const {
#if ZENDNN_CPU_THREADING_RUNTIME != ZENDNN_RUNTIME_OMP
    assert(!"threading env need to be omp for embedding_bag");
#endif
    status_t status;
    // initialize
    emb_params_t  params;
    status = pre_process(ctx, params);
    if (status != status::success) {
        return status;
    }
    auto  algo                = pd()->desc()->alg_kind;
    bool  is_weights          = pd()->desc()->is_weights;
    switch (algo) {
    case alg_kind::embedding_bag_sum:
        return is_weights ? avx512_sum_wt(params) : avx512_sum(params);
    case alg_kind::embedding_bag_mean:
        return avx512_mean(params);
    case alg_kind::embedding_bag_max:
        return avx512_max(params);
    }
    return status::unimplemented;
}
/*
 * extract embedding bag parameters
 */
template<data_type_t data_type>
status_t
avx512_embedding_bag_t<data_type>::pre_process(const exec_ctx_t &ctx,
        emb_params_t &params) const {
    status_t status = status::success;
    // get algorithm params
    params.padidx         = pd()->desc()->padding_idx;
    params.nthr           = pd()->desc()->num_threads;
    params.scatter_stride = pd()->desc()->scatter_stride;
    params.scatter_offset = pd()->desc()->scatter_offset;

    //Overriding threads with OMP_NUM_THREADS
    zendnnEnv zenEnvObj = readEnv();
    params.nthr = zenEnvObj.zen_num_threads;

    // get the tensors
    params.input =
        static_cast<void *>(ctx.host_ptr(ZENDNN_ARG_SRC_0));
    params.indices =
        static_cast<void *>(ctx.host_ptr(ZENDNN_ARG_SRC_1));
    params.offsets =
        static_cast<void *>(ctx.host_ptr(ZENDNN_ARG_SRC_2));
    params.weights = nullptr;
    if (pd()->desc()->is_weights) {
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

    // get rid of excess omp threads if any
    params.dst_size     = dst_mdw.nelems();
    if (params.offset_size < params.nthr) {
        params.nthr = params.offset_size;
    }

    return status;
}
/*
 * sum without weights
 */
template<>
status_t
avx512_embedding_bag_t<f32>::avx512_sum(const emb_params_t &params) const {
    float        const *input    = static_cast<float *>(params.input);
    indices_type       *indices  = static_cast<indices_type *>(params.indices);
    offsets_type       *offsets  = static_cast<offsets_type *>(params.offsets);
    dst_type           *dst      = static_cast<dst_type *>(params.dst);
    const int32_t      &width          = params.width;
    const int32_t      &indsz          = params.indices_size;
    const int32_t      &offsz          = params.offset_size;
    const int32_t      &dstsz          = params.dst_size;
    const indices_type &padidx         = params.padidx;
    const uint32_t     &nthr           = params.nthr;
    const uint32_t     &scatter_offset = params.scatter_offset;
    const uint32_t     &scatter_stride = params.scatter_stride;
    // add scatter_offset
    uint32_t stride  = scatter_stride*width;
    dst             += scatter_offset*width;
    // fast path for common cases of width 512, 256, 128, 64, 32 and 16
    if (512 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps512 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        sum.fetch_add_ps(input + indices[i]*width);
                    }
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps512 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_add_ps(input + indices[i]*width);
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        return status::success;
    }
    if (256 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps256 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        sum.fetch_add_ps(input + indices[i]*width);
                    }
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps256 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_add_ps(input + indices[i]*width);
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        return status::success;
    }
    if (128 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps128 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        sum.fetch_add_ps(input + indices[i]*width);
                    }
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps128 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_add_ps(input + indices[i]*width);
                }
                sum.store_ps(dst + oi*stride);
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
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps64 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        sum.fetch_add_ps(input + indices[i]*width);
                    }
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps64 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_add_ps(input + indices[i]*width);
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        return status::success;
    }
    if (32 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps32 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        sum.fetch_add_ps(input + indices[i]*width);
                    }
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps32 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_add_ps(input + indices[i]*width);
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        return status::success;
    }
    if (16 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps16 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        sum.fetch_add_ps(input + indices[i]*width);
                    }
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps16 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_add_ps(input + indices[i]*width);
                }
                sum.store_ps(dst + oi*stride);
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
#if PREFETCH_EN
            ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
            std::vector<dst_type> sum(width,0.0);
            for (auto i = ofirst; i < olast; ++i) {
                if (indices[i] != padidx) {
                    for (auto j = 0; j < width; ++j) {
                        sum[j] += input[j + indices[i]*width];
                    }
                }
            }
            for (auto j = 0; j < width; ++j) {
                dst[j + oi*stride] = sum[j];
            }
        }
    }
    else {
        #pragma omp parallel for num_threads(nthr) //proc_bind(master)
        for (auto oi = 0; oi < offsz; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
            ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
            std::vector<dst_type> sum(width,0.0);
            for (auto i = ofirst; i < olast; ++i)
                for (auto j = 0; j < width; ++j) {
                    sum[j] += input[j + indices[i]*width];
                }
            for (auto j = 0; j < width; ++j) {
                dst[j + oi*stride] = sum[j];
            }
        }
    }
    return status::success;
}
/*
 * sum with weights
 */
template<>
status_t
avx512_embedding_bag_t<f32>::avx512_sum_wt(const emb_params_t &params) const {
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
    const uint32_t     &scatter_offset = params.scatter_offset;
    const uint32_t     &scatter_stride = params.scatter_stride;
    // add scatter_offset
    uint32_t stride  = scatter_stride*width;
    dst             += scatter_offset*width;
    // fast path for common cases of width 512, 256, 128, 64, 32 and 16
    if (512 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps512 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                    }
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps512 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        return status::success;
    }
    if (256 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps256 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                    }
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps256 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        return status::success;
    }
    if (128 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps128 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                    }
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps128 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                }
                sum.store_ps(dst + oi*stride);
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
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps64 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                    }
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps64 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        return status::success;
    }
    if (32 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps32 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                    }
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps32 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        return status::success;
    }
    if (16 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps16 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                    }
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps16 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_fmadd_ps(input + indices[i]*width, wts[i]);
                }
                sum.store_ps(dst + oi*stride);
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
#if PREFETCH_EN
            ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
            std::vector<dst_type> sum(width,0.0);
            for (auto i = ofirst; i < olast; ++i) {
                if (indices[i] != padidx) {
                    for (auto j = 0; j < width; ++j) {
                        sum[j] += wts[i]*input[j + indices[i]*width];
                    }
                }
            }
            for (auto j = 0; j < width; ++j) {
                dst[j + oi*stride] = sum[j];
            }
        }
    }
    else {
        #pragma omp parallel for num_threads(nthr) //proc_bind(master)
        for (auto oi = 0; oi < offsz; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
            ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
            std::vector<dst_type> sum(width,0.0);
            for (auto i = ofirst; i < olast; ++i)
                for (auto j = 0; j < width; ++j) {
                    sum[j] += wts[i]*input[j + indices[i]*width];
                }
            for (auto j = 0; j < width; ++j) {
                dst[j + oi*stride] = sum[j];
            }
        }
    }
    return status::success;
}
/*
 * mean without weights
 */
template<>
status_t
avx512_embedding_bag_t<f32>::avx512_mean(const emb_params_t &params) const {
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
    const uint32_t     &scatter_offset = params.scatter_offset;
    const uint32_t     &scatter_stride = params.scatter_stride;
    // add scatter_offset
    uint32_t stride  = scatter_stride*width;
    dst             += scatter_offset*width;
    // fast path for common cases of width 512, 256, 128, 64, 32 and 16
    if (512 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps512 sum;
                int32_t         count = 0;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        count++;
                        sum.fetch_add_ps(input + indices[i]*width);
                    }
                }
                sum.scale_store_ps(dst + oi*stride, (1.0/float(count)));
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps512 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_add_ps(input + indices[i]*width);
                }
                float dn = (ofirst!=indsz) ? (1.0/float(olast - ofirst)) : 1.0;
                sum.scale_store_ps(dst + oi*stride, dn);
            }
        }
        return status::success;
    }
    if (256 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps256 sum;
                int32_t         count = 0;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        count++;
                        sum.fetch_add_ps(input + indices[i]*width);
                    }
                }
                sum.scale_store_ps(dst + oi*stride, (1.0/float(count)));
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps256 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_add_ps(input + indices[i]*width);
                }
                float dn = (ofirst!=indsz) ? (1.0/float(olast - ofirst)) : 1.0;
                sum.scale_store_ps(dst + oi*stride, dn);
            }
        }
        return status::success;
    }
    if (128 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps128 sum;
                int32_t         count = 0;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        count++;
                        sum.fetch_add_ps(input + indices[i]*width);
                    }
                }
                sum.scale_store_ps(dst + oi*stride, (1.0/float(count)));
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps128 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_add_ps(input + indices[i]*width);
                }
                float dn = (ofirst!=indsz) ? (1.0/float(olast - ofirst)) : 1.0;
                sum.scale_store_ps(dst + oi*stride, dn);
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
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps64 sum;
                int32_t         count = 0;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        count++;
                        sum.fetch_add_ps(input + indices[i]*width);
                    }
                }
                sum.scale_store_ps(dst + oi*stride, (1.0/float(count)));
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps64 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_add_ps(input + indices[i]*width);
                }
                float dn = (ofirst!=indsz) ? (1.0/float(olast - ofirst)) : 1.0;
                sum.scale_store_ps(dst + oi*stride, dn);
            }
        }
        return status::success;
    }
    if (32 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps32 sum;
                int32_t         count = 0;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        count++;
                        sum.fetch_add_ps(input + indices[i]*width);
                    }
                }
                sum.scale_store_ps(dst + oi*stride, (1.0/float(count)));
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps32 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_add_ps(input + indices[i]*width);
                }
                float dn = (ofirst!=indsz) ? (1.0/float(olast - ofirst)) : 1.0;
                sum.scale_store_ps(dst + oi*stride, dn);
            }
        }
        return status::success;
    }
    if (16 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps16  sum;
                int32_t         count = 0;
                for (auto i = ofirst; i < olast; ++i) {
                    if (indices[i] != padidx) {
                        count++;
                        sum.fetch_add_ps(input + indices[i]*width);
                    }
                }
                sum.scale_store_ps(dst + oi*stride, (1.0/float(count)));
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps16 sum;
                for (auto i = ofirst; i < olast; ++i) {
                    sum.fetch_add_ps(input + indices[i]*width);
                }
                float dn = (ofirst!=indsz) ? (1.0/float(olast - ofirst)) : 1.0;
                sum.scale_store_ps(dst + oi*stride, dn);
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
#if PREFETCH_EN
            ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
            std::vector<dst_type> sum(width,0.0);
            int32_t               count = 0;
            for (auto i = ofirst; i < olast; ++i) {
                if (indices[i] != padidx) {
                    count++;
                    for (auto j = 0; j < width; ++j) {
                        sum[j] += input[j + indices[i]*width];
                    }
                }
            }
            float dn = 1.0/float(count);
            for (auto j = 0; j < width; ++j) {
                dst[j + oi*stride] = dn*sum[j];
            }
        }
    }
    else {
        #pragma omp parallel for num_threads(nthr) //proc_bind(master)
        for (auto oi = 0; oi < offsz; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
            ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
            std::vector<dst_type> sum(width,0.0);
            for (auto i = ofirst; i < olast; ++i)
                for (auto j = 0; j < width; ++j) {
                    sum[j] += input[j + indices[i]*width];
                }
            float dn = (ofirst!=indsz) ? (1.0/float(olast - ofirst)) : 1.0;
            for (auto j = 0; j < width; ++j) {
                dst[j + oi*stride] = dn*sum[j];
            }
        }
    }
    return status::success;
}
/*
 * max without weights
 */
template<>
status_t
avx512_embedding_bag_t<f32>::avx512_max(const emb_params_t &params) const {
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
    const uint32_t     &scatter_offset = params.scatter_offset;
    const uint32_t     &scatter_stride = params.scatter_stride;
    // add scatter_offset
    uint32_t stride  = scatter_stride*width;
    dst             += scatter_offset*width;
    // fast path for common cases of width 512, 256, 128, 64, 32 and 16
    if (512 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps512 sum;
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
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps512 sum;
                if (ofirst!=indsz) {
                    sum.load_ps(input + indices[ofirst]*width);
                }
                for (auto i = ofirst+1; i < olast; ++i) {
                    sum.fetch_max_ps(input + indices[i]*width);
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        return status::success;
    }
    if (256 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps256 sum;
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
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps256 sum;
                if (ofirst!=indsz) {
                    sum.load_ps(input + indices[ofirst]*width);
                }
                for (auto i = ofirst+1; i < olast; ++i) {
                    sum.fetch_max_ps(input + indices[i]*width);
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        return status::success;
    }
    if (128 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps128 sum;
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
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps128 sum;
                if (ofirst!=indsz) {
                    sum.load_ps(input + indices[ofirst]*width);
                }
                for (auto i = ofirst+1; i < olast; ++i) {
                    sum.fetch_max_ps(input + indices[i]*width);
                }
                sum.store_ps(dst + oi*stride);
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
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps64 sum;
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
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps64 sum;
                if (ofirst!=indsz) {
                    sum.load_ps(input + indices[ofirst]*width);
                }
                for (auto i = ofirst+1; i < olast; ++i) {
                    sum.fetch_max_ps(input + indices[i]*width);
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        return status::success;
    }
    if (32 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps32 sum;
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
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps32 sum;
                if (ofirst!=indsz) {
                    sum.load_ps(input + indices[ofirst]*width);
                }
                for (auto i = ofirst+1; i < olast; ++i) {
                    sum.fetch_max_ps(input + indices[i]*width);
                }
                sum.store_ps(dst + oi*stride);
            }
        }
        return status::success;
    }
    if (16 == width) {
        if (padidx >= 0) {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps16 sum;
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
                sum.store_ps(dst + oi*stride);
            }
        }
        else {
            #pragma omp parallel for num_threads(nthr) //proc_bind(master)
            for (auto oi = 0; oi < offsz; ++oi) {
                auto ofirst = offsets[oi];
                auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
                ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
                zenmmAVX512_ext_ps16 sum;
                if (ofirst!=indsz) {
                    sum.load_ps(input + indices[ofirst]*width);
                }
                for (auto i = ofirst+1; i < olast; ++i) {
                    sum.fetch_max_ps(input + indices[i]*width);
                }
                sum.store_ps(dst + oi*stride);
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
#if PREFETCH_EN
            ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
            std::vector<dst_type> sum(width,0.0);
            int32_t               nfirst = ofirst;
            while (nfirst < olast) {
                if (nfirst != padidx) {
                    for (auto j = 0; j < width; ++j) {
                        sum[j]  = input[j + indices[nfirst]*width];
                    }
                    break;
                }
                nfirst++;
            }
            for (auto i = nfirst+1; i < olast; ++i) {
                if (indices[i] != padidx) {
                    for (auto j = 0; j < width; ++j)
                        if (sum[j]  < input[j + indices[i]*width]) {
                            sum[j] = input[j + indices[i]*width];
                        }
                }
            }
            for (auto j = 0; j < width; ++j) {
                dst[j + oi*stride] = sum[j];
            }
        }
    }
    else {
        #pragma omp parallel for num_threads(nthr) //proc_bind(master)
        for (auto oi = 0; oi < offsz; ++oi) {
            auto ofirst = offsets[oi];
            auto olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
#if PREFETCH_EN
            ebvec_prefetch(input, indices, width, offsets, oi, offsz, indsz);
#endif
            std::vector<dst_type> sum(width,0.0);
            for (auto j = 0; j < width; ++j) {
                if (ofirst!=indsz) {
                    sum[j]  = input[j + indices[ofirst]*width];
                }
            }
            for (auto i = ofirst+1; i < olast; ++i)
                for (auto j = 0; j < width; ++j)
                    if (sum[j]  < input[j + indices[i]*width]) {
                        sum[j] = input[j + indices[i]*width];
                    }
            for (auto j = 0; j < width; ++j) {
                dst[j + oi*stride] = sum[j];
            }
        }
    }
    return status::success;
}
template struct avx512_embedding_bag_t<f32>;
} //namespace cpu
}
}

#endif
