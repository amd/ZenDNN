/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include <assert.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/zendnn_thread.hpp"
#include "common/type_helpers.hpp"
#include "cpu/ref_layer_normalization.hpp"

namespace zendnn {
namespace impl {
namespace cpu {

namespace {

template <typename T>
inline float maybe_up_convert(T x) {
    return x;
}

template <>
inline float maybe_up_convert<bfloat16_t>(bfloat16_t x) {
    return (float)x;
}

} // namespace

using namespace data_type;

template <impl::data_type_t d_type>
status_t ref_layer_normalization_fwd_t<d_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    const auto use_ss = pd()->use_scaleshift();
    const auto use_scale = pd()->use_scale();
    const auto use_shift = pd()->use_shift();

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper stat_d(pd()->stat_md());
    const memory_desc_wrapper ss_d(pd()->weights_md());

    const size_t shift_off
            = use_ss && !ss_d.has_zero_dim() ? ss_d.off(1, 0) : 0;

    auto src = CTX_IN_MEM(const data_t *, ZENDNN_ARG_SRC);
    auto scale = CTX_IN_MEM(
            const float *, use_scale ? ZENDNN_ARG_SCALE : ZENDNN_ARG_SCALE_SHIFT);
    auto shift = use_shift ? CTX_IN_MEM(const float *, ZENDNN_ARG_SHIFT)
                           : use_ss ? &scale[shift_off] : nullptr;

    auto mean = pd()->stats_are_src()
            ? const_cast<float *>(CTX_IN_MEM(const float *, ZENDNN_ARG_MEAN))
            : CTX_OUT_MEM(float *, ZENDNN_ARG_MEAN);
    auto variance = pd()->stats_are_src()
            ? const_cast<float *>(CTX_IN_MEM(const float *, ZENDNN_ARG_VARIANCE))
            : CTX_OUT_MEM(float *, ZENDNN_ARG_VARIANCE);

    auto dst = CTX_OUT_MEM(data_t *, ZENDNN_ARG_DST);

    const dim_t N = pd()->across_axis();
    const dim_t C = pd()->norm_axis();

    const float eps = pd()->desc()->layer_norm_epsilon;
    const bool save_stats = pd()->is_training();
    const bool calculate_stats = !pd()->stats_are_src();

    const auto ss_off = [&use_scale, &use_shift, &use_ss](
                                const memory_desc_wrapper &md, dim_t c) {
        dim_t offset = 0;
        if (use_ss) offset = md.off(0, c);
        if (use_scale || use_shift) offset = md.off(c);
        return offset;
    };

    /* fast return */
    if (this->pd()->has_zero_dim_memory()) {
        if (calculate_stats && save_stats) {
            for (dim_t n = 0; n < N; n++) {
                mean[n] = 0;
                variance[n] = 0;
            }
        }
        return status::success;
    }

    parallel_nd(N, [&](dim_t n) {
        const size_t s_off = stat_d.off_l(n);
        auto v_mean = calculate_stats ? 0 : mean[s_off];
        auto v_variance = calculate_stats ? 0 : variance[s_off];

        if (calculate_stats) {
            for (dim_t c = 0; c < C; ++c)
                v_mean += maybe_up_convert(src[src_d.off_l(n * C + c)]);
            v_mean /= C;

            for (dim_t c = 0; c < C; ++c) {
                float m = src[src_d.off_l(n * C + c)] - v_mean;
                v_variance += m * m;
            }
            v_variance /= C;
        }

        float sqrt_variance = sqrtf(v_variance + eps);
        for (dim_t c = 0; c < C; ++c) {
            const float sm
                    = (scale ? scale[ss_off(ss_d, c)] : 1.0f) / sqrt_variance;
            const float sv = shift ? shift[ss_off(ss_d, c)] : 0;
            const size_t dst_off = dst_d.off_l(n * C + c),
                         src_off = src_d.off_l(n * C + c);

            dst[dst_off] = sm * (maybe_up_convert(src[src_off]) - v_mean) + sv;
        }

        if (calculate_stats) {
            if (save_stats) {
                mean[s_off] = v_mean;
                variance[s_off] = v_variance;
            }
        }
    });
    return status::success;
}

template struct ref_layer_normalization_fwd_t<f32>;
template struct ref_layer_normalization_fwd_t<bf16>;

template <impl::data_type_t d_type>
status_t ref_layer_normalization_bwd_t<d_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper stat_d(pd()->stat_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper ss_d(pd()->weights_md());
    const memory_desc_wrapper diff_ss_d(pd()->diff_weights_md());

    const auto use_ss = pd()->use_scaleshift();
    const auto use_scale = pd()->use_scale();
    const auto use_shift = pd()->use_shift();

    auto src = CTX_IN_MEM(const data_t *, ZENDNN_ARG_SRC);
    auto mean = CTX_IN_MEM(const float *, ZENDNN_ARG_MEAN);
    auto variance = CTX_IN_MEM(const float *, ZENDNN_ARG_VARIANCE);
    auto diff_dst = CTX_IN_MEM(const data_t *, ZENDNN_ARG_DIFF_DST);
    auto scale = CTX_IN_MEM(
            float *, use_scale ? ZENDNN_ARG_SCALE : ZENDNN_ARG_SCALE_SHIFT);
    auto diff_src = CTX_OUT_CLEAN_MEM(data_t *, ZENDNN_ARG_DIFF_SRC, status);
    CHECK(status);

    const size_t diff_shift_off
            = use_ss && !diff_ss_d.has_zero_dim() ? diff_ss_d.off(1, 0) : 0;

    auto diff_scale = use_scale
            ? CTX_OUT_CLEAN_MEM(float *, ZENDNN_ARG_DIFF_SCALE, status)
            : use_ss ? CTX_OUT_CLEAN_MEM(
                      float *, ZENDNN_ARG_DIFF_SCALE_SHIFT, status)
                     : nullptr;
    CHECK(status);
    auto diff_shift = use_shift
            ? CTX_OUT_CLEAN_MEM(float *, ZENDNN_ARG_DIFF_SHIFT, status)
            : use_ss ? &diff_scale[diff_shift_off] : nullptr;
    CHECK(status);

    const dim_t N = pd()->across_axis();
    const dim_t C = pd()->norm_axis();

    const auto ss_off = [&use_scale, &use_shift, &use_ss](
                                const memory_desc_wrapper &md, dim_t c) {
        dim_t offset = 0;
        if (use_ss) offset = md.off(0, c);
        if (use_scale || use_shift) offset = md.off(c);
        return offset;
    };

    /* fast return */
    if (this->pd()->has_zero_dim_memory()) {
        if (diff_scale) {
            for (dim_t c = 0; c < C; ++c) {
                diff_scale[ss_off(diff_ss_d, c)] = 0;
            }
        }
        if (diff_shift) {
            for (dim_t c = 0; c < C; ++c) {
                diff_shift[ss_off(diff_ss_d, c)] = 0;
            }
        }
        return status::success;
    }

    const float eps = pd()->desc()->layer_norm_epsilon;
    const bool calculate_diff_stats = !pd()->use_global_stats();

    if (diff_scale || diff_shift) {
        parallel_nd(C, [&](dim_t c) {
            float diff_gamma = float(0);
            float diff_beta = float(0);

            for (dim_t n = 0; n < N; ++n) {
                const size_t src_off = src_d.off_l(n * C + c),
                             diff_dst_off = diff_dst_d.off_l(n * C + c),
                             s_off = stat_d.off_l(n);
                float inv_sqrt_variance = static_cast<float>(
                        1.0f / sqrtf(variance[s_off] + eps));
                data_t dd = maybe_up_convert(diff_dst[diff_dst_off]);
                diff_gamma += (maybe_up_convert(src[src_off]) - mean[s_off])
                        * dd * inv_sqrt_variance;
                diff_beta += dd;
            }

            if (diff_scale) diff_scale[ss_off(diff_ss_d, c)] = diff_gamma;
            if (diff_shift) diff_shift[ss_off(diff_ss_d, c)] = diff_beta;
        });
    }

    parallel_nd(N, [&](dim_t n) {
        const size_t s_off = stat_d.off_l(n);
        float inv_sqrt_variance
                = static_cast<float>(1.0f / sqrtf(variance[s_off] + eps));
        float dd_gamma = float(0), dd_gamma_x = float(0);
        if (calculate_diff_stats) {
            for (dim_t c = 0; c < C; ++c) {
                float gamma = scale ? scale[ss_off(ss_d, c)] : 1;
                const size_t src_off = src_d.off_l(n * C + c),
                             diff_dst_off = diff_dst_d.off_l(n * C + c);
                data_t dd = maybe_up_convert(diff_dst[diff_dst_off]);
                dd_gamma += dd * gamma;
                dd_gamma_x += dd * gamma
                        * (maybe_up_convert(src[src_off]) - mean[s_off]);
            }
            dd_gamma_x *= inv_sqrt_variance;
        }

        for (dim_t c = 0; c < C; ++c) {
            float gamma = scale ? scale[ss_off(ss_d, c)] : 1;
            const size_t src_off = src_d.off_l(n * C + c),
                         diff_src_off = diff_src_d.off_l(n * C + c),
                         diff_dst_off = diff_dst_d.off_l(n * C + c);
            float v_diff_src = maybe_up_convert(diff_dst[diff_dst_off]) * gamma;
            if (calculate_diff_stats)
                v_diff_src -= dd_gamma / C
                        + (maybe_up_convert(src[src_off]) - mean[s_off])
                                * dd_gamma_x * inv_sqrt_variance / C;
            v_diff_src *= inv_sqrt_variance;
            diff_src[diff_src_off] = v_diff_src;
        }
    });
    return status::success;
}

template struct ref_layer_normalization_bwd_t<f32>;
template struct ref_layer_normalization_bwd_t<bf16>;

} // namespace cpu
} // namespace impl
} // namespace zendnn

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
