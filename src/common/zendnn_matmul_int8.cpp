/*******************************************************************************
* Modifications Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#include <atomic>

#include <assert.h>
#include <float.h>
#include <math.h>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <tuple>

#ifndef ZENDNN_USE_AOCL_BLIS_API
    #include <cblas.h>
#else // ZENDNN_USE_AOCL_BLIS_API
    #include "cblas_with_blis_api.hpp"
#endif // ZENDNN_USE_AOCL_BLIS_API

#include "common/primitive.hpp"
#include "zendnn_logging.hpp"
#include "zendnn_private.hpp"
#include "zendnn_helper.hpp"
#include "zendnn.hpp"
#include "zendnn_reorder_cache.hpp"

#define ZENDNN_INT8_MATMUL_VERSION1 1
#define ZENDNN_INT8_MATMUL_VERSION2 2
#define ZENDNN_INT8_MATMUL_VERSION3 3

using namespace zendnn;
using tag = memory::format_tag;
using dt = memory::data_type;
extern std::mutex map_mutex;

template <typename T>
void mul_add_quantize_matrix(float *dst_data, char *C_Array, float dst_scale,
                             int M,
                             int N, const int32_t zero_point_dst, int dst_type, T *mul_buf,
                             T *add_buf, int buffer_type) {
    float result = 0.0f;
    if (mul_buf == nullptr || add_buf == nullptr) {
        #pragma omp parallel for
        for (int idx = 0; idx < M * N; idx++) {
            result = dst_data[idx] * dst_scale + zero_point_dst;
            zendnn::impl::cpu::io::store_float_value((zendnn::impl::data_type_t)dst_type,
                    result, C_Array, idx);
        }
    }
    else {
        #pragma omp parallel for
        for (int idx = 0; idx < M * N; idx++) {
            result = (((dst_data[idx] * zendnn::impl::cpu::io::load_float_value((
                            zendnn::impl::data_type_t)buffer_type, (void *)mul_buf,
                        idx)) + zendnn::impl::cpu::io::load_float_value((zendnn::impl::data_type_t)
                                buffer_type, (void *)add_buf, idx))* dst_scale) +
                     zero_point_dst;
            zendnn::impl::cpu::io::store_float_value((zendnn::impl::data_type_t)dst_type,
                    result, C_Array, idx);
        }
    }
}

void zenMatMulPrimitiveINT8(zendnn::zendnnEnv zenEnvObj,
                            const impl::exec_ctx_t &ctx, int thread_qty,
                            int src_type, int dst_type,
                            int bias_type, const bool Layout, const bool TransA, const bool TransB,
                            const int M, const int N, const int K, const char *A_Array,
                            const int8_t *B_Array, const char *bias, char *C_Array,
                            const float alpha, const float beta, const int lda, const int ldb,
                            const int ldc, const impl::post_ops_t &po_ops, bool blocked_format,
                            const int32_t zero_point_src,
                            const int32_t zero_point_wei, const int32_t zero_point_dst,
                            float do_sum, bool is_weights_const, bool is_inplace,
                            float *src_scale, int src_scale_size, bool default_src_scales,
                            float *wei_scale, int wei_scale_size, bool default_wei_scales,
                            float *dst_scale, int dst_scale_size, bool default_dst_scales,
                            int scale_type) {
    // TODO: This version control is for internal use only
    unsigned int zen_matmul_version =
        zendnn::zendnn_getenv_int("ZENDNN_INT8_MATMUL_VER",
                                  ZENDNN_INT8_MATMUL_VERSION1);

    // In-place will use fixed tag.
    unsigned int weight_cache_type = zenEnvObj.zenWeightCache;
    if (!is_weights_const) {
        blocked_format = false;
    }
    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    Key_matmul key_obj(TransA, TransB, M, K, N, lda, ldb, ldc, B_Array,
                       thread_qty, false);

    std::vector<primitive> net;
    std::unordered_map<int, memory> net_args;

    //Prepare Data
    char *in_arr = const_cast<char *>(A_Array);
    int8_t *filt_arr = const_cast<int8_t *>(B_Array);
    char *bias_arr = const_cast<char *>(bias);

    memory::dims src_dims = {M, K};
    memory::dims weight_dims = {K, N};
    memory::dims bias_dims = {1, N};
    memory::dims dst_dims = {M, N};

    memory::dims a_strides = TransA ? memory::dims {1, lda} :
                             memory::dims {lda, 1};
    memory::dims b_strides = TransB ? memory::dims {1, ldb} :
                             memory::dims {ldb, 1};

    memory::desc src_md = memory::desc({src_dims}, src_type == zendnn_s8? dt::s8 :
                                       dt::u8, a_strides);
    memory::desc matmul_weights_md = memory::desc({weight_dims}, dt::s8,
                                     b_strides);
    memory::desc blocked_matmul_weights_md = memory::desc({weight_dims}, dt::s8,
            weight_cache_type > zendnnWeightCacheType::WEIGHT_CACHE_OUT_OF_PLACE ?
            tag::BA16a64b4a : tag::any);

    bool is_algo2 = zenEnvObj.zenINT8GEMMalgo ==
                    zenINT8MatMulAlgoType::MATMUL_BLOCKED_JIT_INT8;
    bool default_src_zp = zero_point_src == 0;
    bool default_wei_zp = zero_point_wei == 0;
    if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_INPLACE ||
            weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_AOT_RESIZED_INPLACE) {
        memory::desc want_B_md = blocked_matmul_weights_md;
        if (src_type == zendnn_s8) {
            want_B_md.data.extra.flags |= zendnn_memory_extra_flag_compensation_conv_s8s8;
            want_B_md.data.extra.compensation_mask = (1 << 1);
        }
        if (!default_src_zp) {
            want_B_md.data.extra.flags
            |= zendnn_memory_extra_flag_compensation_conv_asymmetric_src;
            want_B_md.data.extra.asymm_compensation_mask = (1 << 1);
        }
        blocked_matmul_weights_md = want_B_md;
    }
    // If size doesn't match with reordered_memory don't do blocking
    if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_AOT_INPLACE &&
            (matmul_weights_md.get_size() != blocked_matmul_weights_md.get_size())) {
        blocked_format = false;
    }

    if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_AOT_INPLACE &&
            is_algo2 &&
            (!default_src_zp || !default_wei_zp) && blocked_format) {
        zen_matmul_version = ZENDNN_INT8_MATMUL_VERSION3;
    }

    memory::desc bias_md;
    zendnn::post_ops post_ops;
    zendnn::memory bias_memory;
    int postop_index = 0;
    bool post_attr = false;

    if (bias_type) {
        bias_md = memory::desc({bias_dims}, (zendnn::memory::data_type)bias_type,
                               tag::ab);
    }

    bool relu_po = (po_ops.len() == 1 &&
                    po_ops.entry_[0].eltwise.alg == impl::alg_kind::eltwise_relu);

    //new memory for cached scales
    float *n_scale = NULL;
    // Ver 1 --> Relu/no-post ops (scales = src*wei*dst), Other (scales = src * wei)
    // Ver 2 --> Relu/no-post ops (scales = src*wei*dst), Other (scales = src * wei)
    // Ver 3 --> scales = src * wei
    if (zen_matmul_version != ZENDNN_INT8_MATMUL_VERSION3 && (po_ops.len() == 0 ||
            relu_po)) {
        cacheStaticScales(zenEnvObj, key_obj, n_scale, src_scale, wei_scale,
                          dst_scale, src_scale_size, wei_scale_size, dst_scale_size, scale_type);
    }
    else {
        cacheStaticScales(zenEnvObj, key_obj, n_scale, src_scale, wei_scale,
                          NULL, src_scale_size, wei_scale_size, dst_scale_size, scale_type);
    }
    std::vector<float> scale_vector(n_scale, n_scale + wei_scale_size);

    // Acc for compensation
    int32_t *acc = NULL;

    // Bias is executed as postop add for Ver 3
    // Computed scales (src * wei) are executed as postop mul for Ver 3
    if (zen_matmul_version == ZENDNN_INT8_MATMUL_VERSION3) {
        int src_0 = TransA ? 1 : lda;
        int src_1 = TransA ? lda : 1;
        int wei_0 = TransB ? 1 : ldb;
        int wei_1 = TransB ? ldb : 1;
        cacheZeroPointCompensation(zenEnvObj, key_obj, M, N, K, (char *)A_Array,
                                   src_0, src_1,
                                   filt_arr, wei_0, wei_1, acc, ldc, zero_point_src, zero_point_wei,
                                   blocked_format, is_weights_const, is_inplace,
                                   zenEnvObj.zenINT8GEMMalgo, weight_cache_type, eng, engine_stream);

        // Compensation matrix as add postop
        if (acc != NULL) {
            post_attr = true;
            zendnn::memory acc_memory;
            // TODO: Chnage the Dimension to 2D when we have Src and Wei Zeropoint
            auto acc_md = memory::desc({1, N}, dt::s32, tag::ab);
            acc_memory = memory(acc_md, eng, acc);
            int acc_t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(postop_index) | ZENDNN_ARG_SRC_1;
            postop_index++;
            post_ops.append_binary(algorithm::binary_add, acc_md);
            net_args.insert({acc_t, acc_memory});
        }
        // Computed scales as Mul Postop
        if (!(default_src_scales && default_wei_scales)) {
            post_attr = true;
            zendnn::memory src_wei_scale_memory;
            auto src_wei_scale_md = memory::desc({1, wei_scale_size}, dt::f32, tag::ab);
            src_wei_scale_memory = memory(src_wei_scale_md, eng, n_scale);
            auto src_wei_scale_t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(postop_index) |
                                   ZENDNN_ARG_SRC_1;
            postop_index++;
            post_ops.append_binary(algorithm::binary_mul, src_wei_scale_md);
            net_args.insert({src_wei_scale_t, src_wei_scale_memory});
        }

        // Bias as add postop
        if (bias_type) {
            post_attr = true;
            bias_memory = memory(bias_md, eng, bias_arr);
            auto bias_t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(postop_index) | ZENDNN_ARG_SRC_1;
            postop_index++;
            post_ops.append_binary(algorithm::binary_add, bias_md);
            net_args.insert({bias_t, bias_memory});
        }
    }

    // Mul and Add buffers are of same datatype (Either BF16 or F32)
    // TODO: Add support for different datatypes for Mul and Add (u8, s8, etc.,) and mixed datatypes
    bool mul_add_ver2 = po_ops.len() == 2 &&
                        po_ops.entry_[0].binary.alg == impl::alg_kind::binary_mul &&
                        po_ops.entry_[1].binary.alg == impl::alg_kind::binary_add &&
                        (po_ops.entry_[0].binary.src1_desc.data_type == zendnn_f32 ||
                         po_ops.entry_[0].binary.src1_desc.data_type == zendnn_bf16) &&
                        zen_matmul_version == ZENDNN_INT8_MATMUL_VERSION2;
    primitive_attr matmul_attr;
    const float scale_po = 1.0f;
    void *mul_buff = NULL, *add_buff = NULL;
    int buffer_type;
    for (auto idx = 0; idx < po_ops.len(); ++idx) {
        const auto &e = po_ops.entry_[idx];
        if (e.eltwise.alg == impl::alg_kind::eltwise_relu) {
            // Relu
            post_attr = true;
            post_ops.append_eltwise(scale_po, algorithm::eltwise_relu, 0.f, 0.f);
        }
        else if (e.eltwise.alg == impl::alg_kind::eltwise_swish) {
            // SiLU
            post_attr = true;
            post_ops.append_eltwise(scale_po, algorithm::eltwise_swish, 1.f, 0.f);
        }
        else if (e.eltwise.alg == impl::alg_kind::eltwise_gelu) {
            // Gelu tanH
            post_attr = true;
            post_ops.append_eltwise(scale_po, algorithm::eltwise_gelu, 1.f, 0.f);
        }
        else if (e.eltwise.alg == impl::alg_kind::eltwise_gelu_erf) {
            // Gelu ERF
            post_attr = true;
            post_ops.append_eltwise(scale_po, algorithm::eltwise_gelu_erf, 1.f, 0.f);
        }
        else if (e.eltwise.alg == impl::alg_kind::eltwise_logistic) {
            // Sigmoid
            post_attr = true;
            post_ops.append_eltwise(scale_po, algorithm::eltwise_logistic, 0.f, 0.f);
        }
        else if (e.eltwise.alg == impl::alg_kind::eltwise_tanh) {
            // Tanh
            post_attr = true;
            post_ops.append_eltwise(scale_po, algorithm::eltwise_tanh, 0.f, 0.f);
        }
        else if (e.kind == impl::primitive_kind::sum) {
            post_attr = true;
            post_ops.append_sum(e.sum.scale);
        }
        else if (e.binary.alg == impl::alg_kind::binary_add) {
            if (mul_add_ver2) {
                buffer_type = e.binary.src1_desc.data_type;
                if (buffer_type == zendnn_f32) {
                    add_buff = const_cast<float *>(CTX_IN_MEM(const float *,
                                                   (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1)));
                }
                else {
                    add_buff = const_cast<int16_t *>(CTX_IN_MEM(const int16_t *,
                                                     (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1)));

                }
            }
            else {
                post_attr = true;
                const auto &src1_desc = e.binary.src1_desc;
                post_ops.append_binary(algorithm::binary_add, src1_desc);
                auto add_raw = const_cast<void *>(CTX_IN_MEM(const void *,
                                                  (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1)));
                auto po_mem = memory(src1_desc,eng,add_raw);
                int t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(postop_index) | ZENDNN_ARG_SRC_1;
                net_args.insert({t,po_mem});
            }
        }
        else if (e.binary.alg == impl::alg_kind::binary_mul) {
            if (mul_add_ver2) {
                buffer_type = e.binary.src1_desc.data_type;
                if (buffer_type == zendnn_f32) {
                    mul_buff = const_cast<float *>(CTX_IN_MEM(const float *,
                                                   (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1)));
                }
                else {
                    mul_buff = const_cast<int16_t *>(CTX_IN_MEM(const int16_t *,
                                                     (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1)));
                }
            }
            else {
                post_attr = true;
                const auto &src1_desc = e.binary.src1_desc;
                post_ops.append_binary(algorithm::binary_mul, src1_desc);
                auto mul_raw = const_cast<void *>(CTX_IN_MEM(const void *,
                                                  (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1)));
                auto po_mem = memory(src1_desc,eng,mul_raw);
                int t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(postop_index) | ZENDNN_ARG_SRC_1;
                net_args.insert({t,po_mem});
            }
        }
        postop_index++;
    }

    // Dest scales are applied
    if (!((mul_add_ver2) || (zen_matmul_version != ZENDNN_INT8_MATMUL_VERSION3 &&
                             (po_ops.len() == 0 || relu_po)))) {
        if (!default_dst_scales) {
            post_attr = true;
            // Dst scale
            zendnn::memory dst_scale_memory;
            auto dst_scale_md = memory::desc({1}, dt::f32, tag::a);
            dst_scale_memory = memory(dst_scale_md, eng, dst_scale);

            auto dst_scale_t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(postop_index) |
                               ZENDNN_ARG_SRC_1;
            post_ops.append_binary(algorithm::binary_mul, dst_scale_md);
            net_args.insert({dst_scale_t, dst_scale_memory});
        }
    }

    if (post_attr) {
        matmul_attr.set_post_ops(post_ops);
    }

    memory::desc dst_md;
    if (mul_add_ver2) {
        dst_md = memory::desc({dst_dims}, dt::f32, {ldc, 1});
    }
    else {
        dst_md = memory::desc({dst_dims}, (zendnn::memory::data_type)dst_type, {ldc, 1});
    }

    // TODO: update the name of variable
    matmul_attr.set_autoTunerEnable(true);
    if (zen_matmul_version != ZENDNN_INT8_MATMUL_VERSION3) {
        matmul_attr.set_output_scales(wei_scale_size == 1? 0: (1<<1), scale_vector);
        if (!default_src_zp) {
            matmul_attr.set_zero_points(ZENDNN_ARG_SRC, 0, {ZENDNN_RUNTIME_S32_VAL});
        }
        if (!default_wei_zp) {
            matmul_attr.set_zero_points(ZENDNN_ARG_WEIGHTS, 0, {ZENDNN_RUNTIME_S32_VAL});
        }
    }
    if (!mul_add_ver2 && (zero_point_dst != 0)) {
        matmul_attr.set_zero_points(ZENDNN_ARG_DST, 0, {ZENDNN_RUNTIME_S32_VAL});
    }

    zendnn::matmul::primitive_desc matmul_prim_disc;
    //MatMul desc
    if (zen_matmul_version == ZENDNN_INT8_MATMUL_VERSION3) {
        // Bias computation takes place as a post-op
        auto matmul_disc = blocked_format ? zendnn::matmul::desc(src_md,
                           blocked_matmul_weights_md, dst_md) :
                           zendnn::matmul::desc(src_md, matmul_weights_md, dst_md);
        matmul_prim_disc = zendnn::matmul::primitive_desc(matmul_disc, matmul_attr,
                           eng);
    }
    else {
        auto matmul_disc = blocked_format ? bias_type? zendnn::matmul::desc(src_md,
                           blocked_matmul_weights_md, bias_md, dst_md): zendnn::matmul::desc(src_md,
                                   blocked_matmul_weights_md, dst_md) : bias_type? zendnn::matmul::desc(src_md,
                                           matmul_weights_md, bias_md, dst_md): zendnn::matmul::desc(src_md,
                                                   matmul_weights_md, dst_md);
        matmul_prim_disc = zendnn::matmul::primitive_desc(matmul_disc, matmul_attr,
                           eng);
    }
    //Memory creation
    zendnn::memory user_weights_memory, src_memory, dst_memory;
    zendnn::memory reordered_bias_memory;
    src_memory = memory(src_md, eng, in_arr);
    user_weights_memory = memory(matmul_weights_md, eng, filt_arr);

    char *new_bias = NULL;
    // Ver 1 --> Reordered Bias (Bias / Src scale * Wei scale)
    // Ver 2 --> Reordered Bias (Bias / Src scale * Wei scale)
    // Ver 3 --> Bias (Applies as a postop)
    if (bias_type && zen_matmul_version != ZENDNN_INT8_MATMUL_VERSION3) {
        // Reordered Bias values to Quantize the Bias
        auto bias_desc = matmul_prim_disc.bias_desc();
        cacheScaledBias(zenEnvObj, key_obj, engine_stream, eng,
                        bias_desc, new_bias, bias_arr, N, src_scale, wei_scale, src_scale_size,
                        wei_scale_size);
        reordered_bias_memory = memory(bias_md, eng, new_bias);
    }

    float *new_dst = NULL;
    if (mul_add_ver2) {
        // New Memory is created to handle Explicit Mul+Add computation.
        new_dst = (float *)zendnn_aligned_alloc(64, sizeof(float)*M*N);
        dst_memory = memory(dst_md, eng, new_dst);
    }
    else {
        dst_memory = memory(dst_md, eng, C_Array);
    }

    //Weight reordering
    zendnn::memory reordered_weights_memory;
    auto block_info = matmul_prim_disc.weights_desc().data.format_desc.blocking;
    Key_matmul key_obj_reorder(TransA, TransB, M, K, N, lda, ldb, ldc, B_Array,
                               thread_qty, false, block_info);

    if (blocked_format) {
        reorderAndCacheWeightsBrgemm(
            key_obj_reorder, matmul_prim_disc.weights_desc(), user_weights_memory,
            reordered_weights_memory, eng, engine_stream, is_weights_const,
            is_inplace, weight_cache_type);
    }
    zendnn::matmul matmul_prim = zendnn::matmul(matmul_prim_disc);
    net_args.insert({ZENDNN_ARG_SRC, src_memory});
    net_args.insert({ZENDNN_ARG_WEIGHTS, blocked_format?reordered_weights_memory:user_weights_memory});
    if (zen_matmul_version != ZENDNN_INT8_MATMUL_VERSION3) {
        if (bias_type) net_args.insert({ZENDNN_ARG_BIAS, reordered_bias_memory});
    }
    net_args.insert({ZENDNN_ARG_DST,dst_memory});
    if (zen_matmul_version != ZENDNN_INT8_MATMUL_VERSION3) {
        if (!default_src_zp) {
            int32_t *zero_point_src_nc = const_cast<int32_t *>(&zero_point_src);
            memory zp_A_mem({{1}, memory::data_type::s32, {1}}, eng, zero_point_src_nc);
            net_args.insert({ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_SRC, zp_A_mem});
        }
        if (!default_wei_zp) {
            int32_t *zero_point_wei_nc = const_cast<int32_t *>(&zero_point_wei);
            memory zp_B_mem({{1}, memory::data_type::s32, {1}}, eng, zero_point_wei_nc);
            net_args.insert({ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_WEIGHTS, zp_B_mem});
        }
    }
    if (!mul_add_ver2 && (zero_point_dst != 0)) {
        int32_t *zero_point_dst_nc = const_cast<int32_t *>(&zero_point_dst);
        memory zp_C_mem({{1}, memory::data_type::s32, {1}}, eng, zero_point_dst_nc);
        net_args.insert({ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_DST, zp_C_mem});
    }
    matmul_prim.execute(engine_stream, net_args);

    if (mul_add_ver2) {
        if (buffer_type == zendnn_f32) {
            mul_add_quantize_matrix<float>(new_dst, C_Array, dst_scale[0], M, N,
                                           zero_point_dst,
                                           dst_type, (float *)mul_buff, (float *)add_buff, buffer_type);
        }
        else {
            mul_add_quantize_matrix<int16_t>(new_dst, C_Array, dst_scale[0], M, N,
                                             zero_point_dst,
                                             dst_type, (int16_t *)mul_buff, (int16_t *)add_buff, buffer_type);
        }

        free(new_dst);
    }

    if (!zenEnvObj.zenStaticScaleCache && n_scale != NULL) {
        free(n_scale);
    }
    if (!zenEnvObj.zenBiasCache && new_bias != NULL) {
        free(new_bias);
    }
    if (!zenEnvObj.zenZpCompCache && acc != NULL) {
        free(acc);
    }
}

template<typename T>
aocl_post_op *create_aocl_post_ops_int8(
    const impl::exec_ctx_t &ctx,
    const zendnn_post_ops &po,
    int n,
    char *bias,
    int bias_type,
    float *dq_scale,
    int dq_scale_size,
    float *dst_scale,
    int dst_scale_size,
    int32_t *src_wei_comp_zp,
    bool is_1d_comp,
    const int8_t *zero_point_dst,
    float do_sum,
    T *sum_buffer,
    float *dummy_scale,
    int8_t *dummy_zp,
    int zp_dt = zendnn_s8
) {
    aocl_post_op *post_ops = NULL;
    // By default, scale postop is always enabled.
    // Check if Bias and zero_point_dst postops are required.
    bool apply_dst_scale_or_dst_zp = *zero_point_dst != 0 || (dst_scale != NULL &&
                                     !(dst_scale[0] == 1.0 && dst_scale_size == 1));
    int postop_count = 1;
    int bias_cnt = 0;
    if (bias != NULL) {
        ++postop_count;
        ++bias_cnt;
    }
    if (src_wei_comp_zp != NULL) {
        if (is_1d_comp) {
            ++bias_cnt;
        }
        ++postop_count;
    }
    if (apply_dst_scale_or_dst_zp) {
        ++postop_count;
    }
    // Create postop for LPGEMM
    // Order of postops: BIAS -> scale -> other po
    if (po.len() + postop_count > 0) {
        post_ops = (aocl_post_op *) malloc(sizeof(aocl_post_op));
        dim_t max_post_ops_seq_length = po.len() + postop_count;
        post_ops->seq_vector = (AOCL_POST_OP_TYPE *) malloc(max_post_ops_seq_length *
                               sizeof(AOCL_POST_OP_TYPE));

        // Iterate through each postop, check and add it if needed.
        int post_op_i = 0;
        //Set all post-ops to NULL
        post_ops->eltwise = NULL;
        post_ops->bias = NULL;
        post_ops->sum = NULL;
        post_ops->matrix_add = NULL;
        post_ops->matrix_mul = NULL;

        dim_t eltwise_index = 0;
        dim_t bias_index = 0;
        dim_t scale_index = 0;
        dim_t add_index = 0;
        dim_t mul_index = 0;

        //Get count of eltwise and binary post-ops
        int mem_count[3] = {0};
        //Check if src_wei_compensation is required.
        if (src_wei_comp_zp != NULL && !is_1d_comp) {
            mem_count[1]++;
        }
        for (auto idx = 0; idx < po.len(); ++idx) {
            const auto po_type = po.entry_[idx];
            switch (po_type.kind) {
            case impl::primitive_kind::eltwise:
                mem_count[0]++;
                break;
            case impl::primitive_kind::binary:
                if (po_type.binary.alg == impl::alg_kind::binary_add) {
                    mem_count[1]++;
                }
                else if (po_type.binary.alg == impl::alg_kind::binary_mul) {
                    mem_count[2]++;
                }
                break;
            case impl::primitive_kind::sum:
                mem_count[1]++;
                break;
            default:
                break;
            }
        }

        if (mem_count[0] > 0) {
            post_ops->eltwise = (aocl_post_op_eltwise *) malloc(sizeof(
                                    aocl_post_op_eltwise)*mem_count[0]);
        }
        if (mem_count[1] > 0) {
            post_ops->matrix_add = (aocl_post_op_matrix_add *) malloc(sizeof(
                                       aocl_post_op_matrix_add)*mem_count[1]);
        }
        if (mem_count[2] > 0) {
            post_ops->matrix_mul = (aocl_post_op_matrix_mul *) malloc(sizeof(
                                       aocl_post_op_matrix_mul)*mem_count[2]);
        }
        if (bias_cnt) {
            post_ops->bias = (aocl_post_op_bias *) malloc(sizeof(
                                 aocl_post_op_bias)*bias_cnt);
        }
        // Src wei zero-point compensation
        if (is_1d_comp && src_wei_comp_zp != NULL) {
            // Add compensation as bias postop
            post_ops->seq_vector[post_op_i++] = BIAS;
            (post_ops->bias + bias_index)->stor_type = getAOCLstoreType(zendnn_s32);
            (post_ops->bias + bias_index)->bias = (int32_t *)src_wei_comp_zp;
            bias_index++;
        }
        else if (src_wei_comp_zp != NULL) {
            post_ops->seq_vector[post_op_i++] = MATRIX_ADD;
            (post_ops->matrix_add + add_index)->ldm = n;
            (post_ops->matrix_add + add_index)->scale_factor = (float *)dummy_scale;
            (post_ops->matrix_add + add_index)->scale_factor_len = 1;
            (post_ops->matrix_add + add_index)->stor_type = getAOCLstoreType(zendnn_s32);
            (post_ops->matrix_add + add_index)->matrix = (int32_t *)src_wei_comp_zp;
            add_index++;
        }
        // Create zero-point and scale size
        // Output scale is applied before post-ops.
        // Dst zero-point is applied at end.
        size_t scale_zp_size = sizeof(aocl_post_op_sum);
        if (apply_dst_scale_or_dst_zp) {
            scale_zp_size = 2*sizeof(aocl_post_op_sum);
        }
        post_ops->sum = (aocl_post_op_sum *) malloc(scale_zp_size);
        //Scale post-op

        if (dq_scale) {
            //Apply scales
            post_ops->seq_vector[post_op_i++] = SCALE;
            (post_ops->sum + scale_index)->is_power_of_2 = FALSE;
            (post_ops->sum + scale_index)->scale_factor = NULL;
            (post_ops->sum + scale_index)->buff = NULL;
            (post_ops->sum + scale_index)->zero_point = NULL;

            (post_ops->sum + scale_index)->scale_factor = (float *)dq_scale;
            (post_ops->sum + scale_index)->zero_point = (int8_t *)dummy_zp;
            (post_ops->sum + scale_index)->scale_factor_len = dq_scale_size;
            (post_ops->sum + scale_index)->zero_point_len = 1;
            (post_ops->sum + scale_index)->sf_stor_type = AOCL_GEMM_F32;
            (post_ops->sum + scale_index)->zp_stor_type = AOCL_GEMM_INT8;
            scale_index++;
        }
        //Add bias post-op
        if (bias != NULL && (post_ops->bias + bias_index) != NULL) {
            // Add bias postop
            post_ops->seq_vector[post_op_i++] = BIAS;
            (post_ops->bias + bias_index)->stor_type = getAOCLstoreType((
                        zendnn_data_type_t)bias_type);
            (post_ops->bias + bias_index)->bias = bias;
            bias_index++;
        }
        //Add eltwise and binary post-ops in given sequence
        for (auto idx = 0; idx < po.len(); ++idx) {
            const auto po_type = po.entry_[idx];
            if (po_type.kind == impl::primitive_kind::eltwise &&
                    (post_ops->eltwise + eltwise_index) != NULL) {

                if (po_type.eltwise.alg == impl::alg_kind::eltwise_relu) {
                    // Add ReLU postop
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = RELU;
                    eltwise_index++;
                }
                else if (po_type.eltwise.alg == impl::alg_kind::eltwise_gelu) {
                    // Gelu tanh.
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = GELU_TANH;
                    eltwise_index++;
                }
                else if (po_type.eltwise.alg == impl::alg_kind::eltwise_gelu_erf) {
                    // Gelu erf.
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = GELU_ERF;
                    eltwise_index++;
                }
                else if (po_type.eltwise.alg == impl::alg_kind::eltwise_swish) {
                    // SiLU
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    float alpha_val = (float)po_type.eltwise.alpha;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = (float *)dummy_scale;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = SWISH;
                    eltwise_index++;
                }
                else if (po_type.eltwise.alg == impl::alg_kind::eltwise_tanh) {
                    // tanh
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    post_ops->eltwise[eltwise_index].is_power_of_2 = FALSE;
                    post_ops->eltwise[eltwise_index].scale_factor = NULL;
                    post_ops->eltwise[eltwise_index].algo.alpha = NULL;
                    post_ops->eltwise[eltwise_index].algo.beta = NULL;
                    post_ops->eltwise[eltwise_index].algo.algo_type = TANH;
                    eltwise_index+=1;
                }
                else if (po_type.eltwise.alg == impl::alg_kind::eltwise_logistic) {
                    // Sigmoid.
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = SIGMOID;
                    eltwise_index++;
                }
            }
            else if (po_type.kind == impl::primitive_kind::binary) {
                if (po_type.binary.alg == impl::alg_kind::binary_add &&
                        (post_ops->matrix_add + add_index) != NULL) {
                    post_ops->seq_vector[post_op_i++] = MATRIX_ADD;
                    (post_ops->matrix_add + add_index)->ldm = n;
                    auto b_dt = po_type.binary.src1_desc.data_type;
                    (post_ops->matrix_add + add_index)->scale_factor = (float *)dummy_scale;
                    (post_ops->matrix_add + add_index)->scale_factor_len = 1;
                    (post_ops->matrix_add + add_index)->stor_type = getAOCLstoreType(b_dt);
                    auto binary_po = CTX_IN_MEM(const void *,
                                                (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1));
                    if (b_dt == zendnn_f32) {
                        auto addA = reinterpret_cast<float *>(const_cast<void *>(binary_po));
                        (post_ops->matrix_add + add_index)->matrix = (float *)addA;
                    }
                    else if (b_dt == zendnn_bf16) {
                        auto addA = reinterpret_cast<int16_t *>(const_cast<void *>(binary_po));
                        (post_ops->matrix_add + add_index)->matrix = (int16_t *)addA;
                    }
                    add_index++;
                }
                else if (po_type.binary.alg == impl::alg_kind::binary_mul &&
                         (post_ops->matrix_mul + mul_index) != NULL) {
                    post_ops->seq_vector[post_op_i++] = MATRIX_MUL;
                    (post_ops->matrix_mul + mul_index)->ldm = n;
                    (post_ops->matrix_mul + mul_index)->scale_factor = (float *)dummy_scale;
                    (post_ops->matrix_mul + mul_index)->scale_factor_len = 1;

                    auto b_dt = po_type.binary.src1_desc.data_type;
                    (post_ops->matrix_mul + mul_index)->stor_type = getAOCLstoreType(b_dt);
                    auto binary_po = CTX_IN_MEM(const void *,
                                                (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1));
                    if (b_dt == zendnn_f32) {
                        auto mulA = reinterpret_cast<float *>(const_cast<void *>(binary_po));
                        (post_ops->matrix_mul + mul_index)->matrix = (float *)mulA;
                    }
                    if (b_dt == zendnn_bf16) {
                        auto mulA = reinterpret_cast<int16_t *>(const_cast<void *>(binary_po));
                        (post_ops->matrix_mul + mul_index)->matrix = (int16_t *)mulA;
                    }
                    mul_index++;
                }
            }
            // Using sum post-op with GEMM_beta
            else if (po_type.kind == impl::primitive_kind::sum &&
                     (post_ops->matrix_add + add_index) != NULL) {
                post_ops->seq_vector[post_op_i++] = MATRIX_ADD;
                (post_ops->matrix_add + add_index)->ldm = n;
                (post_ops->matrix_add + add_index)->scale_factor = (float *)dummy_scale;
                (post_ops->matrix_add + add_index)->scale_factor_len = 1;
                (post_ops->matrix_add + add_index)->stor_type =
                    AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_INT8;
                (post_ops->matrix_add + add_index)->matrix = (T *)sum_buffer;
                add_index++;
            }
        }
        // Dst zero point
        if (apply_dst_scale_or_dst_zp && (post_ops->sum + scale_index) != NULL) {
            post_ops->seq_vector[post_op_i++] = SCALE;
            (post_ops->sum + scale_index)->is_power_of_2 = FALSE;
            (post_ops->sum + scale_index)->scale_factor = NULL;
            (post_ops->sum + scale_index)->buff = NULL;
            (post_ops->sum + scale_index)->zero_point = NULL;
            (post_ops->sum + scale_index)->scale_factor = (float *)dst_scale;
            (post_ops->sum + scale_index)->zero_point = (int8_t *)zero_point_dst;
            (post_ops->sum + scale_index)->scale_factor_len = dst_scale_size;
            (post_ops->sum + scale_index)->zero_point_len = 1;
            (post_ops->sum + scale_index)->sf_stor_type = AOCL_GEMM_F32;
            (post_ops->sum + scale_index)->zp_stor_type = getAOCLstoreType((zendnn_data_type_t)zp_dt);
            scale_index++;
        }
        post_ops->seq_length = po.len() + postop_count;
    }
    return post_ops;
}

// Free AOCL post-ops allocated memory
void free_aocl_po_memory_int8(
    aocl_post_op *post_ops
) {
    //Free memory

    if (post_ops != NULL) {
        if (post_ops->bias != NULL) {
            post_ops->bias->bias=NULL;
            free(post_ops->bias);
        }
        if (post_ops->eltwise != NULL) {
            free(post_ops->eltwise);
        }
        if (post_ops->sum != NULL) {
            free(post_ops->sum);
        }
        if (post_ops->matrix_add != NULL) {
            free(post_ops->matrix_add);
        }
        if (post_ops->matrix_mul != NULL) {
            free(post_ops->matrix_mul);
        }
        free(post_ops->seq_vector);
        free(post_ops);
    }

}

void free_cached_memory(
    zendnn::zendnnEnv zenEnvObj,
    int32_t *comp_acc,
    int32_t wei_zp,
    float *new_scales
) {
    bool cache_comp_acc = zenEnvObj.zenZpCompCache;
    bool is_zp_comp_allocated = zenEnvObj.zenWeightCache !=
                                zendnnWeightCacheType::WEIGHT_CACHE_AOT_RESIZED_INPLACE;
    bool cache_scale = zenEnvObj.zenStaticScaleCache;

    if ((!cache_comp_acc || wei_zp) && comp_acc != NULL && is_zp_comp_allocated) {
        free(comp_acc);
    }
    if (!cache_scale && new_scales != NULL) {
        free(new_scales);
    }
}

void zenMatMul_gemm_u8s8s32ofloat(
    zendnn::zendnnEnv zenEnvObj,
    const impl::exec_ctx_t &ctx,
    int thread_qty,
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const uint8_t *input,
    const int lda,
    const int8_t *filter,
    const int ldb,
    char *bias,
    int bias_type,
    const impl::post_ops_t &po_ops,
    const float beta,
    char *output,
    const int ldc,
    int dst_type,
    const int32_t zero_point_src,
    const int32_t zero_point_wei,
    const int8_t zero_point_dst,
    float do_sum,
    bool is_weights_const,
    bool is_inplace,
    bool blocked_format,
    float *src_scale,
    int src_scale_size,
    float *wei_scale,
    int wei_scale_size,
    float *dst_scale,
    int dst_scale_size
) {
    int weight_cache_type = zenEnvObj.zenWeightCache;
    Key_matmul key_obj(transpose_input, transpose_filter, m, k, n, lda, ldb, ldc,
                       filter, thread_qty, false);

    // New scale for src*wei
    float *new_scale = NULL;
    // Acc for compensation
    int32_t *acc = NULL;
    bool is_allocated = true;
    // Reordered weights
    int8_t *reorder_filter = NULL;
    bool reorder_status = false;

    int src_0 = transpose_input ? 1 : lda;
    int src_1 = transpose_input ? lda : 1;
    int wei_0 = transpose_filter ? 1 : ldb;
    int wei_1 = transpose_filter ? ldb : 1;

    if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_AOT_RESIZED_INPLACE
            && zero_point_src) {
        size_t wei_size = zendnn_custom_op::zendnn_reorder_size(k, n, transpose_filter,
                          zendnn_u8, 0, zendnn_s8);
        acc = (int32_t *)(filter + wei_size);
        is_allocated = false;
    }
    else {
        cacheZeroPointCompensation(zenEnvObj, key_obj, m, n, k, (char *)input, src_0,
                                   src_1,
                                   filter, wei_0, wei_1, acc, ldc, zero_point_src, zero_point_wei,
                                   blocked_format, is_weights_const, is_inplace,
                                   zenEnvObj.zenINT8GEMMalgo, weight_cache_type);
    }
    // Passing dst scale as NULL (Applied as aocl post-op).
    cacheStaticScales(zenEnvObj, key_obj, new_scale, src_scale, wei_scale, NULL,
                      src_scale_size, wei_scale_size, 0, zendnn_f32);

    char transA = 'n', transB = 'n', order = 'r';
    char mem_format_a = 'n', mem_format_b = 'r';
    if (transpose_filter) {
        transB = 't';
    }
    if (transpose_input) {
        transA = 't';
    }

    if (blocked_format) {
        const char reorder_param0 = 'B';
        const dim_t reorder_param1 = k;
        const dim_t reorder_param2 = n;
        reorder_status = reorderAndCacheWeights<int8_t>(key_obj, filter, reorder_filter,
                         k, n, ldb, is_weights_const, is_inplace, order, transB,
                         reorder_param0, reorder_param1, reorder_param2,
                         aocl_get_reorder_buf_size_u8s8s32os32, aocl_reorder_u8s8s32os32,
                         weight_cache_type
                                                       );
        if (!reorder_status) {
            mem_format_b = 'n';
            blocked_format = false;
        }
    }
    else {
        mem_format_b = 'n';
    }

    aocl_post_op *post_ops = NULL;
    float dummy_scale = (float)1.0;
    int8_t dummy_zp = (int8_t)0;
    if (dst_type == zendnn_bf16) {
        //Create post_ops
        // If zp_wei exists then can't apply 1d compensation.
        post_ops = create_aocl_post_ops_int8<int16_t>(ctx, po_ops, n,
                   bias, bias_type,
                   new_scale, wei_scale_size, dst_scale, dst_scale_size,
                   acc, !zero_point_wei, &zero_point_dst, do_sum, (int16_t *)output, &dummy_scale,
                   &dummy_zp);
        zendnnVerbose(ZENDNN_PROFLOG,"Using AOCL GEMM API: aocl_gemm_u8s8s32obf16");
        aocl_gemm_u8s8s32obf16(order, transA, transB, m,
                               n,
                               k, alpha, input,
                               lda, mem_format_a, blocked_format ? reorder_filter : filter,
                               ldb, mem_format_b, beta, (int16_t *)output,
                               ldc, post_ops);
    }
    else if (dst_type == zendnn_f32) {
        //Create post_ops
        // If zp_wei exists then can't apply 1d compensation.
        post_ops = create_aocl_post_ops_int8<float>(ctx, po_ops, n,
                   bias, bias_type,
                   new_scale, wei_scale_size, dst_scale, dst_scale_size,
                   acc, !zero_point_wei, &zero_point_dst, do_sum, (float *)output, &dummy_scale,
                   &dummy_zp);
        zendnnVerbose(ZENDNN_PROFLOG,"Using AOCL GEMM API: aocl_gemm_u8s8s32of32");
        aocl_gemm_u8s8s32of32(order, transA, transB, m,
                              n,
                              k, alpha, input,
                              lda, mem_format_a, blocked_format ? reorder_filter : filter,
                              ldb, mem_format_b, beta, (float *)output,
                              ldc, post_ops);
    }
    // Free memory for postops.
    if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_DISABLE && reorder_filter != NULL) {
        free(reorder_filter);
    }
    // Free memory for postops.
    free_aocl_po_memory_int8(post_ops);
    free_cached_memory(zenEnvObj, is_allocated ? acc : NULL, zero_point_wei, new_scale);
}
void zenMatMul_gemm_s8s8s32ofloat(
    zendnn::zendnnEnv zenEnvObj,
    const impl::exec_ctx_t &ctx,
    int thread_qty,
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const int8_t *input,
    const int lda,
    const int8_t *filter,
    const int ldb,
    char *bias,
    int bias_type,
    const impl::post_ops_t &po_ops,
    const float beta,
    char *output,
    const int ldc,
    int dst_type,
    const int32_t zero_point_src,
    const int32_t zero_point_wei,
    const int8_t zero_point_dst,
    float do_sum,
    bool is_weights_const,
    bool is_inplace,
    bool blocked_format,
    float *src_scale,
    int src_scale_size,
    float *wei_scale,
    int wei_scale_size,
    float *dst_scale,
    int dst_scale_size
) {
    int weight_cache_type = zenEnvObj.zenWeightCache;
    Key_matmul key_obj(transpose_input, transpose_filter, m, k, n, lda, ldb, ldc,
                       filter, thread_qty, false);

    // New scale for src*wei
    float *new_scale = NULL;
    // Acc for compensation
    int32_t *acc = NULL;
    bool is_allocated = true;
    // Reordered weights
    int8_t *reorder_filter = NULL;
    bool reorder_status = false;

    int src_0 = transpose_input ? 1 : lda;
    int src_1 = transpose_input ? lda : 1;
    int wei_0 = transpose_filter ? 1 : ldb;
    int wei_1 = transpose_filter ? ldb : 1;

    if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_AOT_RESIZED_INPLACE
            && zero_point_src) {
        size_t wei_size = zendnn_custom_op::zendnn_reorder_size(k, n, transpose_filter,
                          zendnn_s8, 0, zendnn_s8);
        acc = (int32_t *)(filter + wei_size);
        is_allocated = false;
    }
    else {
        cacheZeroPointCompensation(zenEnvObj, key_obj, m, n, k, (char *)input, src_0,
                                   src_1,
                                   filter, wei_0, wei_1, acc, ldc, zero_point_src, zero_point_wei,
                                   blocked_format, is_weights_const, is_inplace,
                                   zenEnvObj.zenINT8GEMMalgo, weight_cache_type);
    }

    // Passing dst scale as NULL (Applied as aocl post-op).
    cacheStaticScales(zenEnvObj, key_obj, new_scale, src_scale, wei_scale, NULL,
                      src_scale_size, wei_scale_size, 0, zendnn_f32);

    char transA = 'n', transB = 'n', order = 'r';
    char mem_format_a = 'n', mem_format_b = 'r';
    if (transpose_filter) {
        transB = 't';
    }
    if (transpose_input) {
        transA = 't';
    }

    if (blocked_format) {
        const char reorder_param0 = 'B';
        const dim_t reorder_param1 = k;
        const dim_t reorder_param2 = n;
        reorder_status = reorderAndCacheWeights<int8_t>(key_obj, filter, reorder_filter,
                         k, n, ldb, is_weights_const, is_inplace, order, transB,
                         reorder_param0, reorder_param1, reorder_param2,
                         aocl_get_reorder_buf_size_s8s8s32os32, aocl_reorder_s8s8s32os32,
                         weight_cache_type);
        if (!reorder_status) {
            mem_format_b = 'n';
            blocked_format = false;
        }
    }
    else {
        mem_format_b = 'n';
    }

    aocl_post_op *post_ops = NULL;
    float dummy_scale = (float)1.0;
    int8_t dummy_zp = (int8_t)0;
    if (dst_type == zendnn_bf16) {
        //Create post_ops
        // If zp_wei exists then can't apply 1d compensation.
        post_ops = create_aocl_post_ops_int8<int16_t>(ctx, po_ops, n,
                   bias, bias_type,
                   new_scale, wei_scale_size, dst_scale, dst_scale_size,
                   acc, !zero_point_wei, &zero_point_dst, do_sum, (int16_t *)output, &dummy_scale,
                   &dummy_zp);
        zendnnVerbose(ZENDNN_PROFLOG,"Using AOCL GEMM API: aocl_gemm_s8s8s32obf16");
        aocl_gemm_s8s8s32obf16(order, transA, transB, m,
                               n,
                               k, alpha, input,
                               lda, mem_format_a, blocked_format ? reorder_filter : filter,
                               ldb, mem_format_b, beta, (int16_t *)output,
                               ldc, post_ops);
    }
    else if (dst_type == zendnn_f32) {
        //Create post_ops
        // If zp_wei exists then can't apply 1d compensation.
        post_ops = create_aocl_post_ops_int8<float>(ctx, po_ops, n,
                   bias, bias_type,
                   new_scale, wei_scale_size, dst_scale, dst_scale_size,
                   acc, !zero_point_wei, &zero_point_dst, do_sum, (float *)output, &dummy_scale,
                   &dummy_zp);
        zendnnVerbose(ZENDNN_PROFLOG,"Using AOCL GEMM API: aocl_gemm_s8s8s32of32");
        aocl_gemm_s8s8s32of32(order, transA, transB, m,
                              n,
                              k, alpha, input,
                              lda, mem_format_a, blocked_format ? reorder_filter : filter,
                              ldb, mem_format_b, beta, (float *)output,
                              ldc, post_ops);
    }
    // Free memory for postops.
    if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_DISABLE && reorder_filter != NULL) {
        free(reorder_filter);
    }
    // Free memory for postops.
    free_aocl_po_memory_int8(post_ops);
    free_cached_memory(zenEnvObj, is_allocated ? acc : NULL, zero_point_wei, new_scale);
}
void zenMatMul_gemm_s8s8s32oInt(
    zendnn::zendnnEnv zenEnvObj,
    const impl::exec_ctx_t &ctx,
    int thread_qty,
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const int8_t *input,
    const int lda,
    const int8_t *filter,
    const int ldb,
    char *bias,
    int bias_type,
    const impl::post_ops_t &po_ops,
    const float beta,
    char *output,
    const int ldc,
    int dst_type,
    const int32_t zero_point_src,
    const int32_t zero_point_wei,
    const int8_t zero_point_dst,
    float do_sum,
    bool is_weights_const,
    bool is_inplace,
    bool blocked_format,
    float *src_scale,
    int src_scale_size,
    float *wei_scale,
    int wei_scale_size,
    float *dst_scale,
    int dst_scale_size
) {
    int weight_cache_type = zenEnvObj.zenWeightCache;
    Key_matmul key_obj(transpose_input, transpose_filter, m, k, n, lda, ldb, ldc,
                       filter, thread_qty, false);

    // New scale for src*wei
    float *new_scale = NULL;
    // Acc for compensation
    int32_t *acc = NULL;
    bool is_allocated = true;
    // Reordered weights
    int8_t *reorder_filter = NULL;
    bool reorder_status = false;

    int src_0 = transpose_input ? 1 : lda;
    int src_1 = transpose_input ? lda : 1;
    int wei_0 = transpose_filter ? 1 : ldb;
    int wei_1 = transpose_filter ? ldb : 1;

    if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_AOT_RESIZED_INPLACE
            && zero_point_src) {
        size_t wei_size = zendnn_custom_op::zendnn_reorder_size(k, n, transpose_filter,
                          zendnn_s8, 0, zendnn_s8);
        acc = (int32_t *)(filter + wei_size);
        is_allocated = false;
    }
    else {
        cacheZeroPointCompensation(zenEnvObj, key_obj, m, n, k, (char *)input, src_0,
                                   src_1,
                                   filter, wei_0, wei_1, acc, ldc, zero_point_src, zero_point_wei,
                                   blocked_format, is_weights_const, is_inplace,
                                   zenEnvObj.zenINT8GEMMalgo, weight_cache_type);
    }

    // Passing dst scale as NULL (Applied as aocl post-op).
    cacheStaticScales(zenEnvObj, key_obj, new_scale, src_scale, wei_scale, NULL,
                      src_scale_size, wei_scale_size, 0, zendnn_f32);

    char transA = 'n', transB = 'n', order = 'r';
    char mem_format_a = 'n', mem_format_b = 'r';
    if (transpose_filter) {
        transB = 't';
    }
    if (transpose_input) {
        transA = 't';
    }

    if (blocked_format) {
        const char reorder_param0 = 'B';
        const dim_t reorder_param1 = k;
        const dim_t reorder_param2 = n;
        reorder_status = reorderAndCacheWeights<int8_t>(key_obj, filter, reorder_filter,
                         k, n, ldb, is_weights_const, is_inplace, order, transB,
                         reorder_param0, reorder_param1, reorder_param2,
                         aocl_get_reorder_buf_size_s8s8s32os32, aocl_reorder_s8s8s32os32,
                         weight_cache_type);
        if (!reorder_status) {
            mem_format_b = 'n';
            blocked_format = false;
        }
    }
    else {
        mem_format_b = 'n';
    }

    aocl_post_op *post_ops = NULL;
    float dummy_scale = (float)1.0;
    int8_t dummy_zp = (int8_t)0;

    if (dst_type == zendnn_s8) {
        //Create post_ops
        // If zp_wei exists then can't apply 1d compensation.
        post_ops = create_aocl_post_ops_int8<int8_t>(ctx, po_ops, n,
                   bias, bias_type,
                   new_scale, wei_scale_size, dst_scale, dst_scale_size,
                   acc, !zero_point_wei, &zero_point_dst, do_sum, (int8_t *)output, &dummy_scale,
                   &dummy_zp);
        zendnnVerbose(ZENDNN_PROFLOG,"Using AOCL GEMM API: aocl_gemm_s8s8s32os8");
        aocl_gemm_s8s8s32os8(order, transA, transB, m,
                             n,
                             k, alpha, input,
                             lda, mem_format_a, blocked_format ? reorder_filter : filter,
                             ldb, mem_format_b, beta, (int8_t *)output,
                             ldc, post_ops);
    }
    else if(dst_type == zendnn_u8) {
    //Create post_ops
    // If zp_wei exists then can't apply 1d compensation.
    post_ops = create_aocl_post_ops_int8<int8_t>(ctx, po_ops, n,
               bias, bias_type,
               new_scale, wei_scale_size, dst_scale, dst_scale_size,
               acc, !zero_point_wei, &zero_point_dst, do_sum, (int8_t*)output,
               &dummy_scale, &dummy_zp, zendnn_u8);

    aocl_gemm_s8s8s32ou8(order, transA, transB, m,
                         n,
                         k, alpha, input,
                         lda, mem_format_a, blocked_format ? reorder_filter : filter,
                         ldb, mem_format_b, beta, (uint8_t*)output,
                         ldc, post_ops);
    }
    else if (dst_type == zendnn_s32) {
        //Create post_ops
        // If zp_wei exists then can't apply 1d compensation.
        post_ops = create_aocl_post_ops_int8<int32_t>(ctx, po_ops, n,
                   bias, bias_type,
                   new_scale, wei_scale_size, dst_scale, dst_scale_size,
                   acc, !zero_point_wei, &zero_point_dst, do_sum, (int32_t *)output, &dummy_scale,
                   &dummy_zp);
        zendnnVerbose(ZENDNN_PROFLOG,"Using AOCL GEMM API: aocl_gemm_s8s8s32os32");
        aocl_gemm_s8s8s32os32(order, transA, transB, m,
                              n,
                              k, alpha, input,
                              lda, mem_format_a, blocked_format ? reorder_filter : filter,
                              ldb, mem_format_b, beta, (int32_t *)output,
                              ldc, post_ops);
    }
    // Free memory for postops.
    if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_DISABLE && reorder_filter != NULL) {
        free(reorder_filter);
    }
    // Free memory for postops.
    free_aocl_po_memory_int8(post_ops);
    free_cached_memory(zenEnvObj, is_allocated ? acc : NULL, zero_point_wei, new_scale);
}

void zenMatMul_gemm_u8s8s32oInt(
    zendnn::zendnnEnv zenEnvObj,
    const impl::exec_ctx_t &ctx,
    int thread_qty,
    const bool Layout,
    const bool transpose_input,
    const bool transpose_filter,
    const int m,
    const int k,
    const int n,
    const float alpha,
    const uint8_t *input,
    const int lda,
    const int8_t *filter,
    const int ldb,
    char *bias,
    int bias_type,
    const impl::post_ops_t &po_ops,
    const float beta,
    char *output,
    const int ldc,
    int dst_type,
    const int32_t zero_point_src,
    const int32_t zero_point_wei,
    const int8_t zero_point_dst,
    float do_sum,
    bool is_weights_const,
    bool is_inplace,
    bool blocked_format,
    float *src_scale,
    int src_scale_size,
    float *wei_scale,
    int wei_scale_size,
    float *dst_scale,
    int dst_scale_size
) {

    int weight_cache_type = zenEnvObj.zenWeightCache;
    Key_matmul key_obj(transpose_input, transpose_filter, m, k, n, lda, ldb, ldc,
                       filter, thread_qty, false);

    // New scale for src*wei
    float *new_scale = NULL;
    // Acc for compensation
    int32_t *acc = NULL;
    bool is_allocated = true;
    // Reordered weights
    int8_t *reorder_filter = NULL;
    bool reorder_status = false;

    int src_0 = transpose_input ? 1 : lda;
    int src_1 = transpose_input ? lda : 1;
    int wei_0 = transpose_filter ? 1 : ldb;
    int wei_1 = transpose_filter ? ldb : 1;

    if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_AOT_RESIZED_INPLACE
            && zero_point_src) {
        size_t wei_size = zendnn_custom_op::zendnn_reorder_size(k, n, transpose_filter,
                          zendnn_u8, 0, zendnn_s8);
        acc = (int32_t *)(filter + wei_size);
        is_allocated = false;
    }
    else {
        cacheZeroPointCompensation(zenEnvObj, key_obj, m, n, k, (char *)input, src_0,
                                   src_1,
                                   filter, wei_0, wei_1, acc, ldc, zero_point_src, zero_point_wei,
                                   blocked_format, is_weights_const, is_inplace,
                                   zenEnvObj.zenINT8GEMMalgo, weight_cache_type);
    }

    // Passing dst scale as NULL (Applied as aocl post-op).
    cacheStaticScales(zenEnvObj, key_obj, new_scale, src_scale, wei_scale, NULL,
                      src_scale_size, wei_scale_size, 0, zendnn_f32);

    char transA = 'n', transB = 'n', order = 'r';
    char mem_format_a = 'n', mem_format_b = 'r';
    if (transpose_filter) {
        transB = 't';
    }
    if (transpose_input) {
        transA = 't';
    }

    if (blocked_format) {
        const char reorder_param0 = 'B';
        const dim_t reorder_param1 = k;
        const dim_t reorder_param2 = n;

        reorder_status = reorderAndCacheWeights<int8_t>(key_obj, filter, reorder_filter,
                         k, n, ldb, is_weights_const, is_inplace, order, transB,
                         reorder_param0, reorder_param1, reorder_param2,
                         aocl_get_reorder_buf_size_u8s8s32os32, aocl_reorder_u8s8s32os32,
                         weight_cache_type);
        if (!reorder_status) {
            mem_format_b = 'n';
            blocked_format = false;
        }
    }
    else {
        mem_format_b = 'n';
    }
    aocl_post_op *post_ops = NULL;
    float dummy_scale = (float)1.0;
    int8_t dummy_zp = (int8_t)0;
    if (dst_type == zendnn_s8) {
        //Create post_ops
        post_ops = create_aocl_post_ops_int8<int8_t>(ctx, po_ops, n,
                   bias, bias_type,
                   new_scale, wei_scale_size, dst_scale, dst_scale_size,
                   acc, !zero_point_wei, &zero_point_dst, do_sum, (int8_t *)output, &dummy_scale,
                   &dummy_zp);
        zendnnVerbose(ZENDNN_PROFLOG,"Using AOCL GEMM API: aocl_gemm_u8s8s32os8");
        aocl_gemm_u8s8s32os8(order, transA, transB, m,
                             n, k, alpha, input,
                             lda, mem_format_a, blocked_format ? reorder_filter : filter,
                             ldb, mem_format_b, beta, (int8_t *)output,
                             ldc, post_ops);
    }
    else if (dst_type == zendnn_u8) {
        //Create post_ops
        post_ops = create_aocl_post_ops_int8<int8_t>(ctx, po_ops, n,
                   bias, bias_type,
                   new_scale, wei_scale_size, dst_scale, dst_scale_size,
                   acc, !zero_point_wei, &zero_point_dst, do_sum, (int8_t *)output, &dummy_scale,
                   &dummy_zp, zendnn_u8);
        zendnnVerbose(ZENDNN_PROFLOG,"Using AOCL GEMM API: aocl_gemm_u8s8s32ou8");
        aocl_gemm_u8s8s32ou8(order, transA, transB, m,
                             n, k, alpha, input,
                             lda, mem_format_a, blocked_format ? reorder_filter : filter,
                             ldb, mem_format_b, beta, (uint8_t *)output,
                             ldc, post_ops);
    }
    else if (dst_type == zendnn_s32) {
        //Create post_ops
        post_ops = create_aocl_post_ops_int8<int32_t>(ctx, po_ops, n,
                   bias, bias_type,
                   new_scale, wei_scale_size, dst_scale, dst_scale_size,
                   acc, !zero_point_wei, &zero_point_dst, do_sum, (int32_t *)output, &dummy_scale,
                   &dummy_zp);
        zendnnVerbose(ZENDNN_PROFLOG,"Using AOCL GEMM API: aocl_gemm_u8s8s32os32");
        aocl_gemm_u8s8s32os32(order, transA, transB, m,
                              n, k, alpha, input,
                              lda, mem_format_a, blocked_format ? reorder_filter : filter,
                              ldb, mem_format_b, beta, (int32_t *)output,
                              ldc, post_ops);
    }
    // Free memory for postops.
    if (weight_cache_type == zendnnWeightCacheType::WEIGHT_CACHE_DISABLE && reorder_filter != NULL) {
        free(reorder_filter);
    }
    free_aocl_po_memory_int8(post_ops);
    free_cached_memory(zenEnvObj, is_allocated ? acc : NULL, zero_point_wei, new_scale);
}

//Temporary checks for post-ops and data type for AOCL
bool check_dt_po_int8(
    float do_sum
) {
    //Check sum post-op scale
    if (do_sum != 0.0 && do_sum != 1.0) {
        return false;
    }

    return true;
}
int matmul_int8_wrapper(
    const impl::exec_ctx_t &ctx,
    zendnn::zendnnEnv zenEnvObj,
    int src_type,
    int dst_type,
    int bias_type,
    const bool Layout,
    const bool transA,
    const bool transB,
    const int M,
    const int K,
    const int N,
    const float alpha,
    const char *src,
    const int lda,
    const int8_t *weights,
    const int ldb,
    const char *bias,
    const impl::post_ops_t &po_ops,
    const float beta,
    char *dst,
    const int ldc,
    const int32_t zero_point_src,
    const int32_t zero_point_wei,
    const int32_t zero_point_dst,
    float do_sum,
    bool is_weights_const,
    bool is_inplace,
    float *src_scale,
    int src_scale_size,
    bool default_src_scales,
    float *wei_scale,
    int wei_scale_size,
    bool default_wei_scales,
    float *dst_scales,
    int dst_scale_size,
    bool default_dst_scales,
    int scale_type
) {
    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    int thread_qty = zenEnvObj.omp_num_threads;
    // DT for INT8
    if (zenEnvObj.zenINT8GEMMalgo == zenINT8MatMulAlgoType::MATMUL_DT_INT8) {
        // If in-place then use JIT non-blocked.
        if ((M <= 256) || (N < K)) {
            zenEnvObj.zenINT8GEMMalgo = zenINT8MatMulAlgoType::MATMUL_BLOCKED_AOCL_INT8;
        }
        else {
            zenEnvObj.zenINT8GEMMalgo = zenINT8MatMulAlgoType::MATMUL_BLOCKED_JIT_INT8;
        }
    }
    // AOCL Support Check
    if ((zenEnvObj.zenINT8GEMMalgo ==
            zenINT8MatMulAlgoType::MATMUL_BLOCKED_AOCL_INT8 ||
            zenEnvObj.zenINT8GEMMalgo == zenINT8MatMulAlgoType::MATMUL_AOCL_INT8) &&
            !check_dt_po_int8(do_sum)) {
        zenEnvObj.zenINT8GEMMalgo = zenINT8MatMulAlgoType::MATMUL_BLOCKED_JIT_INT8;
    }
    // TODO: Remove this condition once AOT_RESIZED_INPLACE enabled.
    size_t aocl_reorder_size = aocl_get_reorder_buf_size_u8s8s32os32('r',
                               transB ? 't': 'n', 'B', K, N);
    if (zenEnvObj.zenWeightCache ==
            zendnnWeightCacheType::WEIGHT_CACHE_AOT_INPLACE &&
            src_type == zendnn_u8 &&
            (zenEnvObj.zenINT8GEMMalgo == zenINT8MatMulAlgoType::MATMUL_BLOCKED_AOCL_INT8 &&
             (K*N != aocl_reorder_size))) {
        zendnnVerbose(ZENDNN_PROFLOG,"Not running AOCL BLOCKED format");
        zenEnvObj.zenINT8GEMMalgo = zenINT8MatMulAlgoType::MATMUL_JIT_INT8;
    }

    // is_weights_const is set to false then use ZENDNN_JIT_INT8 for blocked algo paths
    if (is_weights_const == false) {
        if (zenEnvObj.zenINT8GEMMalgo == zenINT8MatMulAlgoType::MATMUL_BLOCKED_JIT_INT8
                ||
                zenEnvObj.zenINT8GEMMalgo == zenINT8MatMulAlgoType::MATMUL_BLOCKED_AOCL_INT8) {
            zenEnvObj.zenINT8GEMMalgo = zenINT8MatMulAlgoType::MATMUL_JIT_INT8;
        }
    }
    // If inplace is false and weight cache is AOT inplace then use JIT for blocked algo paths
    if (is_inplace == false &&
            zenEnvObj.zenWeightCache > zendnnWeightCacheType::WEIGHT_CACHE_INPLACE) {
        if (zenEnvObj.zenINT8GEMMalgo == zenINT8MatMulAlgoType::MATMUL_BLOCKED_JIT_INT8
                ||
                zenEnvObj.zenINT8GEMMalgo == zenINT8MatMulAlgoType::MATMUL_BLOCKED_AOCL_INT8) {
            zenEnvObj.zenINT8GEMMalgo = zenINT8MatMulAlgoType::MATMUL_JIT_INT8;
        }
    }

    if (zenEnvObj.zenINT8GEMMalgo ==
            zenINT8MatMulAlgoType::MATMUL_BLOCKED_AOCL_INT8) {
        int8_t zero_point_dst_8 = (int8_t)zero_point_dst;
        if (src_type == zendnn_s8) {
            if (dst_type == zendnn_s8 || dst_type == zendnn_u8 || dst_type == zendnn_s32) {
                zenMatMul_gemm_s8s8s32oInt(zenEnvObj, ctx, thread_qty, Layout, transA, transB,
                                           M, K, N, alpha, (const int8_t *)src, lda, (const int8_t *)weights, ldb,
                                           (char *)bias, bias_type, po_ops, beta, (char *)dst, ldc, dst_type,
                                           zero_point_src, zero_point_wei, zero_point_dst_8, do_sum, is_weights_const,
                                           is_inplace, true, src_scale, src_scale_size, wei_scale, wei_scale_size,
                                           dst_scales, dst_scale_size);
            }
            else if (dst_type == zendnn_f32 || dst_type == zendnn_bf16) {
                zenMatMul_gemm_s8s8s32ofloat(zenEnvObj, ctx, thread_qty, Layout, transA, transB,
                                             M, K, N, alpha, (const int8_t *)src, lda, (const int8_t *)weights, ldb,
                                             (char *)bias, bias_type, po_ops, beta, (char *)dst, ldc, dst_type,
                                             zero_point_src, zero_point_wei, zero_point_dst_8, do_sum, is_weights_const,
                                             is_inplace, true, src_scale, src_scale_size, wei_scale, wei_scale_size,
                                             dst_scales, dst_scale_size);
            }
            else {
                //dst src:s8 and dst:u8
                //CALL BRGEMM Primitive
                map_mutex.lock();
                obj.is_brgemm = true;
                map_mutex.unlock();
                zenMatMulPrimitiveINT8(zenEnvObj, ctx, thread_qty, src_type, dst_type,
                                       bias_type, Layout, transA, transB, M, N, K, src,
                                       weights, bias, dst, alpha, beta, lda, ldb, ldc, po_ops, true, zero_point_src,
                                       zero_point_wei, zero_point_dst, do_sum, is_weights_const, is_inplace,
                                       src_scale, src_scale_size, default_src_scales, wei_scale, wei_scale_size,
                                       default_wei_scales,
                                       dst_scales, dst_scale_size, default_dst_scales,
                                       scale_type);
                map_mutex.lock();
                obj.is_brgemm = false;
                map_mutex.unlock();
            }
        }
        else {
            if (dst_type == zendnn_s8 || dst_type == zendnn_u8 || dst_type == zendnn_s32) {
                zenMatMul_gemm_u8s8s32oInt(zenEnvObj, ctx, thread_qty, Layout, transA, transB,
                                           M, K, N, alpha,
                                           (const uint8_t *)src, lda, (const int8_t *)weights, ldb,
                                           (char *)bias, bias_type, po_ops, beta, (char *)dst, ldc, dst_type,
                                           zero_point_src, zero_point_wei, zero_point_dst_8, do_sum, is_weights_const,
                                           is_inplace, true, src_scale, src_scale_size, wei_scale, wei_scale_size,
                                           dst_scales, dst_scale_size);
            }
            else if (dst_type == zendnn_f32 || dst_type == zendnn_bf16) {
                zenMatMul_gemm_u8s8s32ofloat(zenEnvObj, ctx, thread_qty, Layout, transA, transB,
                                             M, K, N, alpha,
                                             (const uint8_t *)src, lda, (const int8_t *)weights, ldb,
                                             (char *)bias, bias_type, po_ops, beta, (char *)dst, ldc, dst_type,
                                             zero_point_src, zero_point_wei, zero_point_dst_8, do_sum, is_weights_const,
                                             is_inplace, true, src_scale, src_scale_size, wei_scale, wei_scale_size,
                                             dst_scales, dst_scale_size);
            }
            else {
                //dst u8
                //CALL BRGEMM Primitive
                map_mutex.lock();
                obj.is_brgemm = true;
                map_mutex.unlock();
                zenMatMulPrimitiveINT8(zenEnvObj, ctx, thread_qty, src_type, dst_type,
                                       bias_type, Layout, transA, transB, M, N, K, src,
                                       weights, bias, dst, alpha, beta, lda, ldb, ldc, po_ops, true, zero_point_src,
                                       zero_point_wei, zero_point_dst, do_sum, is_weights_const, is_inplace,
                                       src_scale, src_scale_size, default_src_scales, wei_scale, wei_scale_size,
                                       default_wei_scales,
                                       dst_scales, dst_scale_size, default_dst_scales,
                                       scale_type);
                map_mutex.lock();
                obj.is_brgemm = false;
                map_mutex.unlock();
            }
        }
    }
    else if (zenEnvObj.zenINT8GEMMalgo == zenINT8MatMulAlgoType::MATMUL_AOCL_INT8) {
        //Disbale all caching
        zenEnvObj.zenStaticScaleCache = 0;
        zenEnvObj.zenZpCompCache = 0;
        int8_t zero_point_dst_8 = (int8_t)zero_point_dst;
        if (src_type == zendnn_s8) {
            if (dst_type == zendnn_s8 || dst_type == zendnn_s32) {
                zenMatMul_gemm_s8s8s32oInt(zenEnvObj, ctx, thread_qty, Layout, transA, transB,
                                           M, K, N, alpha,
                                           (const int8_t *)src, lda, (const int8_t *)weights, ldb,
                                           (char *)bias, bias_type, po_ops, beta, (char *)dst, ldc, dst_type,
                                           zero_point_src, zero_point_wei, zero_point_dst_8, do_sum, is_weights_const,
                                           is_inplace, false, src_scale, src_scale_size, wei_scale, wei_scale_size,
                                           dst_scales, dst_scale_size);
            }
            else if (dst_type == zendnn_f32 || dst_type == zendnn_bf16) {
                zenMatMul_gemm_s8s8s32ofloat(zenEnvObj, ctx, thread_qty, Layout, transA, transB,
                                             M, K, N, alpha,
                                             (const int8_t *)src, lda, (const int8_t *)weights, ldb,
                                             (char *)bias, bias_type, po_ops, beta, (char *)dst, ldc, dst_type,
                                             zero_point_src, zero_point_wei, zero_point_dst_8, do_sum, is_weights_const,
                                             is_inplace, false, src_scale, src_scale_size, wei_scale, wei_scale_size,
                                             dst_scales, dst_scale_size);
            }
            else {
                //dst u8
                //CALL BRGEMM Primitive
                map_mutex.lock();
                obj.is_brgemm = true;
                map_mutex.unlock();
                zenMatMulPrimitiveINT8(zenEnvObj, ctx, thread_qty, src_type, dst_type,
                                       bias_type, Layout,
                                       transA, transB, M, N, K, src, weights, bias, dst, alpha, beta,
                                       lda, ldb, ldc, po_ops, false, zero_point_src,
                                       zero_point_wei, zero_point_dst, do_sum, is_weights_const, is_inplace,
                                       src_scale, src_scale_size, default_src_scales, wei_scale, wei_scale_size,
                                       default_wei_scales,
                                       dst_scales, dst_scale_size, default_dst_scales,
                                       scale_type);
                map_mutex.lock();
                obj.is_brgemm = false;
                map_mutex.unlock();
            }
        }
        else { // make function for src:u8
            if (dst_type == zendnn_s8 || dst_type == zendnn_u8 || dst_type == zendnn_s32) {
                zenMatMul_gemm_u8s8s32oInt(zenEnvObj, ctx, thread_qty, Layout, transA, transB,
                                           M, K, N, alpha,
                                           (const uint8_t *)src, lda, (const int8_t *)weights, ldb,
                                           (char *)bias, bias_type, po_ops, beta, (char *)dst, ldc, dst_type,
                                           zero_point_src, zero_point_wei, zero_point_dst_8, do_sum, is_weights_const,
                                           is_inplace, false, src_scale, src_scale_size, wei_scale, wei_scale_size,
                                           dst_scales, dst_scale_size);
            }
            else if (dst_type == zendnn_f32 || dst_type == zendnn_bf16) {
                zenMatMul_gemm_u8s8s32ofloat(zenEnvObj, ctx, thread_qty, Layout, transA, transB,
                                             M, K, N, alpha,
                                             (const uint8_t *)src, lda, (const int8_t *)weights, ldb,
                                             (char *)bias, bias_type, po_ops, beta, (char *)dst, ldc, dst_type,
                                             zero_point_src, zero_point_wei, zero_point_dst_8, do_sum, is_weights_const,
                                             is_inplace, false, src_scale, src_scale_size, wei_scale, wei_scale_size,
                                             dst_scales, dst_scale_size);
            }
            else {
                //dst u8
                //CALL BRGEMM Primitive
                map_mutex.lock();
                obj.is_brgemm = true;
                map_mutex.unlock();
                zenMatMulPrimitiveINT8(zenEnvObj, ctx, thread_qty, src_type, dst_type,
                                       bias_type, Layout,
                                       transA, transB, M, N, K, src, weights, bias, dst, alpha, beta,
                                       lda, ldb, ldc, po_ops, false, zero_point_src,
                                       zero_point_wei, zero_point_dst, do_sum, is_weights_const, is_inplace,
                                       src_scale, src_scale_size, default_src_scales, wei_scale, wei_scale_size,
                                       default_wei_scales,
                                       dst_scales, dst_scale_size, default_dst_scales,
                                       scale_type);
                map_mutex.lock();
                obj.is_brgemm = false;
                map_mutex.unlock();
            }
        }
    }
    else if (zenEnvObj.zenINT8GEMMalgo ==
             zenINT8MatMulAlgoType::MATMUL_BLOCKED_JIT_INT8) {
        //CALL blocked BRGEMM Primitive
        map_mutex.lock();
        obj.is_brgemm = true;
        map_mutex.unlock();
        zenMatMulPrimitiveINT8(zenEnvObj, ctx, thread_qty, src_type, dst_type,
                               bias_type, Layout,
                               transA, transB, M, N, K, src, weights, bias, dst, alpha, beta,
                               lda, ldb, ldc, po_ops, true, zero_point_src,
                               zero_point_wei, zero_point_dst, do_sum, is_weights_const, is_inplace,
                               src_scale, src_scale_size, default_src_scales, wei_scale, wei_scale_size,
                               default_wei_scales,
                               dst_scales, dst_scale_size, default_dst_scales,
                               scale_type);
        map_mutex.lock();
        obj.is_brgemm = false;
        map_mutex.unlock();
    }
    else {
        //CALL BRGEMM Primitive
        map_mutex.lock();
        obj.is_brgemm = true;
        map_mutex.unlock();
        // Disable all caching for non-blocked BRGEMM
        zenEnvObj.zenBiasCache = 0;
        zenEnvObj.zenStaticScaleCache = 0;

        zenMatMulPrimitiveINT8(zenEnvObj, ctx, thread_qty, src_type, dst_type,
                               bias_type, Layout,
                               transA, transB, M, N, K, src, weights, bias, dst, alpha, beta,
                               lda, ldb, ldc, po_ops, false, zero_point_src,
                               zero_point_wei, zero_point_dst, do_sum, is_weights_const, is_inplace,
                               src_scale, src_scale_size, default_src_scales, wei_scale, wei_scale_size,
                               default_wei_scales,
                               dst_scales, dst_scale_size, default_dst_scales,
                               scale_type);
        map_mutex.lock();
        obj.is_brgemm = false;
        map_mutex.unlock();
    }
    map_mutex.lock();
    obj.is_log = true;
    map_mutex.unlock();
    return zenEnvObj.zenINT8GEMMalgo ;
}
