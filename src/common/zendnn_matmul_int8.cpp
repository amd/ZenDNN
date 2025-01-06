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
#include "common/weight_cache.hpp"
#include "zendnn_helper.hpp"
#include "zendnn.hpp"

using namespace zendnn;
using tag = memory::format_tag;
using dt = memory::data_type;
extern std::mutex map_mutex;

void zenMatMulPrimitiveINT8(const impl::exec_ctx_t &ctx, int thread_qty,
                            int src_type, int dst_type,
                            int bias_type, const bool Layout, const bool TransA, const bool TransB,
                            const int M, const int N, const int K, const char *A_Array,
                            const int8_t *B_Array, const char *bias, char *C_Array,
                            const float alpha, const float beta, const int lda, const int ldb,
                            const int ldc, const impl::post_ops_t &po_ops, bool blocked_format,
                            float *scale, const int32_t zero_point_src,
                            const int32_t zero_point_wei, const int32_t zero_point_dst,
                            int out_scale_size, float do_sum, bool is_weights_const) {
    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);

    Key_matmul key_obj_reorder(TransA, TransB, M, K, N, lda, ldb, ldc, B_Array,
                               thread_qty, false);

    std::vector<primitive> net;
    std::unordered_map<int, memory> net_args;

    //Prepare Data
    char *in_arr = const_cast<char *>(A_Array);
    int8_t *filt_arr = const_cast<int8_t *>(B_Array);
    char *bias_arr = const_cast<char *>(bias);
    std::vector<float> scale_vector;
    scale_vector.insert(scale_vector.end(), scale, scale + out_scale_size);
    int32_t *zero_point_dst_nc = const_cast<int32_t *>(&zero_point_dst);
    int32_t *zero_point_src_nc = const_cast<int32_t *>(&zero_point_src);
    int32_t *zero_point_wei_nc = const_cast<int32_t *>(&zero_point_wei);

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
            tag::any);

    memory::desc bias_md;
    //Bias type bf16 or f32
    if (bias_type == zendnn_s32)
        bias_md = memory::desc({bias_dims}, dt::s32, tag::ab);
    else if (bias_type == zendnn_s8)
        bias_md = memory::desc({bias_dims}, dt::s8, tag::ab);
    else if (bias_type == zendnn_f32)
        bias_md = memory::desc({bias_dims}, dt::f32, tag::ab);
    else if (bias_type == zendnn_bf16)
        bias_md = memory::desc({bias_dims}, dt::bf16, tag::ab);

    memory::desc dst_md = memory::desc({dst_dims}, (zendnn::memory::data_type)
                                       dst_type, {ldc, 1});
    primitive_attr matmul_attr;
    zendnn::post_ops post_ops;
    bool post_attr = false;
    const float scale_po = 1.0f;
    for (auto idx = 0; idx < po_ops.len(); ++idx) {
        const auto &e = po_ops.entry_[idx];
        if (e.eltwise.alg == impl::alg_kind::eltwise_relu) {
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
        else if (e.kind == impl::primitive_kind::sum) {
            post_attr = true;
            post_ops.append_sum(e.sum.scale);
        }
        else if (e.binary.alg == impl::alg_kind::binary_add) {
            post_attr = true;
            const auto &src1_desc = e.binary.src1_desc;
            post_ops.append_binary(algorithm::binary_add, src1_desc);
            auto add_raw = const_cast<void *>(CTX_IN_MEM(const void *,
                                              (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1)));
            auto po_mem = memory(src1_desc,eng,add_raw);
            int t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1;
            net_args.insert({t,po_mem});
        }
        else if (e.binary.alg == impl::alg_kind::binary_mul) {
            post_attr = true;
            const auto &src1_desc = e.binary.src1_desc;
            post_ops.append_binary(algorithm::binary_mul, src1_desc);
            auto mul_raw = const_cast<void *>(CTX_IN_MEM(const void *,
                                              (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1)));
            auto po_mem = memory(src1_desc,eng,mul_raw);
            int t = ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1;
            net_args.insert({t,po_mem});
        }
    }
    if (post_attr) {
        matmul_attr.set_post_ops(post_ops);
    }
    matmul_attr.set_autoTunerEnable(true);
    matmul_attr.set_output_scales(out_scale_size == 1? 0: (1<<1), scale_vector);
    matmul_attr.set_zero_points(ZENDNN_ARG_SRC, 0, {ZENDNN_RUNTIME_S32_VAL});
    matmul_attr.set_zero_points(ZENDNN_ARG_WEIGHTS, 0, {ZENDNN_RUNTIME_S32_VAL});
    matmul_attr.set_zero_points(ZENDNN_ARG_DST, 0, {ZENDNN_RUNTIME_S32_VAL});
    //MatMul desc
    auto matmul_disc = blocked_format ? bias_type? zendnn::matmul::desc(src_md,
                       blocked_matmul_weights_md, bias_md, dst_md): zendnn::matmul::desc(src_md,
                               blocked_matmul_weights_md, dst_md) : bias_type? zendnn::matmul::desc(src_md,
                                       matmul_weights_md, bias_md, dst_md): zendnn::matmul::desc(src_md,
                                               matmul_weights_md, dst_md);

    //MatMul primitive desc
    auto matmul_prim_disc =
        zendnn::matmul::primitive_desc(matmul_disc, matmul_attr, eng);

    //Memory creation
    zendnn::memory user_weights_memory, src_memory, bias_memory, dst_memory;
    src_memory = memory(src_md, eng, in_arr);
    user_weights_memory = memory(matmul_weights_md, eng, filt_arr);
    if (bias_type) {
        bias_memory = memory(bias_md, eng, bias_arr);
    }
    dst_memory = memory(dst_md, eng, C_Array);
    memory zp_A_mem({{1}, memory::data_type::s32, {1}}, eng, zero_point_src_nc);
    memory zp_B_mem({{1}, memory::data_type::s32, {1}}, eng, zero_point_wei_nc);
    memory zp_C_mem({{1}, memory::data_type::s32, {1}}, eng, zero_point_dst_nc);

    //Weight reordering
    zendnn::memory reordered_weights_memory;

    //weight caching
    static zendnn::impl::lru_weight_cache_t<Key_matmul, zendnn::memory>
    matmul_weight_cache;
    auto found_obj_reorder = matmul_weight_cache.find_key(key_obj_reorder);

    if (blocked_format) {
        if (!is_weights_const || !found_obj_reorder) {
            reordered_weights_memory = memory(matmul_prim_disc.weights_desc(), eng);
            reorder(user_weights_memory, reordered_weights_memory).execute(engine_stream,
                    user_weights_memory, reordered_weights_memory);
            if (is_weights_const) {
                //Save in map
                map_mutex.lock();
                matmul_weight_cache.add(key_obj_reorder, reordered_weights_memory);
                map_mutex.unlock();
            }
        }
        else {
            reordered_weights_memory = matmul_weight_cache.get(key_obj_reorder);
        }
    }

    zendnn::matmul matmul_prim = zendnn::matmul(matmul_prim_disc);
    net_args.insert({ZENDNN_ARG_SRC, src_memory});
    net_args.insert({ZENDNN_ARG_WEIGHTS, blocked_format?reordered_weights_memory:user_weights_memory});
    if (bias_type) net_args.insert({ZENDNN_ARG_BIAS,bias_memory});
    net_args.insert({ZENDNN_ARG_DST,dst_memory});
    net_args.insert({ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_SRC, zp_A_mem});
    net_args.insert({ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_WEIGHTS, zp_B_mem});
    net_args.insert({ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_DST, zp_C_mem});
    matmul_prim.execute(engine_stream, net_args);
}

template<typename T>
aocl_post_op *create_aocl_post_ops_int8(
    const impl::exec_ctx_t &ctx,
    const zendnn_post_ops &po,
    int n,
    char *bias,
    int bias_type,
    const float *scale,
    const int8_t zero_point_dst,
    const int out_scale_size,
    float do_sum,
    T *sum_buffer,
    float *dummy_scale,
    int8_t *dummy_zp
) {
    aocl_post_op *post_ops = NULL;
    // By default, scale postop is always enabled.
    // Check if Bias and zero_point_dst postops are required.
    int postop_count = 1;
    if (bias != NULL) {
        ++postop_count;
    }
    if (zero_point_dst != 0) {
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

        //Add bias post-op
        if (bias != NULL) {
            // Add bias postop
            post_ops->seq_vector[post_op_i++] = BIAS;
            post_ops->bias = (aocl_post_op_bias *) malloc(sizeof(
                                 aocl_post_op_bias));
            if ((post_ops->bias + bias_index) != NULL) {
                (post_ops->bias + bias_index)->stor_type = getAOCLstoreType((
                            zendnn_data_type_t)bias_type);
                (post_ops->bias + bias_index)->bias = bias;
                bias_index++;
            }
        }

        // Create zero-point and scale size
        // Output scale is applied before post-ops.
        // Dst zero-point is applied at end.
        size_t scale_zp_size = sizeof(aocl_post_op_sum);
        if (zero_point_dst != 0) {
            scale_zp_size = 2*sizeof(aocl_post_op_sum);
        }
        post_ops->sum = (aocl_post_op_sum *) malloc(scale_zp_size);
        //Scale post-op
        if (scale && (post_ops->sum + scale_index) != NULL) {
            //Apply scales
            post_ops->seq_vector[post_op_i++] = SCALE;
            (post_ops->sum + scale_index)->is_power_of_2 = FALSE;
            (post_ops->sum + scale_index)->scale_factor = NULL;
            (post_ops->sum + scale_index)->buff = NULL;
            (post_ops->sum + scale_index)->zero_point = NULL;

            (post_ops->sum + scale_index)->scale_factor = (float *)scale;
            (post_ops->sum + scale_index)->zero_point = (int8_t *)dummy_zp;
            (post_ops->sum + scale_index)->scale_factor_len = out_scale_size;
            (post_ops->sum + scale_index)->zero_point_len = 1;
            scale_index++;
        }
        //Get count of eltwise and binary post-ops
        int mem_count[3] = {0};
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
                    dim_t eltwise_index = 0;
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
                    dim_t eltwise_index = 0;
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
                    dim_t eltwise_index = 0;
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    float alpha_val = (float)po_type.eltwise.alpha;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = (float *)dummy_scale;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = SWISH;
                    eltwise_index++;
                }
                else if (po_type.eltwise.alg == impl::alg_kind::eltwise_logistic) {
                    // Sigmoid.
                    dim_t eltwise_index = 0;
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
                    auto addA = reinterpret_cast<T *>(const_cast<void *>(binary_po));
                    (post_ops->matrix_add + add_index)->matrix = (int32_t *)addA;
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
                    auto mulA = reinterpret_cast<T *>(const_cast<void *>(binary_po));
                    (post_ops->matrix_mul + mul_index)->matrix = (T *)mulA;
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
        if (zero_point_dst != 0 && (post_ops->sum + scale_index) != NULL) {
            post_ops->seq_vector[post_op_i++] = SCALE;
            (post_ops->sum + scale_index)->is_power_of_2 = FALSE;
            (post_ops->sum + scale_index)->scale_factor = NULL;
            (post_ops->sum + scale_index)->buff = NULL;
            (post_ops->sum + scale_index)->zero_point = NULL;

            int8_t *zp = (int8_t *)malloc(sizeof(int8_t));
            zp[0] = zero_point_dst;
            (post_ops->sum + scale_index)->scale_factor = (float *)dummy_scale;
            (post_ops->sum + scale_index)->zero_point = (int8_t *)zp;
            (post_ops->sum + scale_index)->scale_factor_len = 1;
            (post_ops->sum + scale_index)->zero_point_len = 1;
            scale_index++;
        }
        post_ops->seq_length = po.len() + postop_count;
    }
    return post_ops;
}

// Free AOCL post-ops allocated memory
void free_aocl_po_memory_int8(
    aocl_post_op *post_ops,
    int8_t zp
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
            // If dst zero point
            // Using index 1
            if (zp != 0) {
                free((post_ops->sum + 1)->zero_point);
            }
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

void zenMatMul_gemm_s8s8s32os8(
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
    int8_t *output,
    const int ldc,
    const float *scale,
    const int8_t zero_point_dst,
    const int out_scale_size,
    float do_sum,
    bool is_weights_const,
    bool blocked_format
) {
    Key_matmul key_obj(transpose_input, transpose_filter, m, k, n, lda, ldb, ldc,
                       filter, thread_qty, true);

    char transA = 'n', transB = 'n', order = 'r';
    char mem_format_a = 'n', mem_format_b = 'r';
    if (transpose_filter) {
        transB = 't';
    }
    if (transpose_input) {
        transA = 't';
    }
    int8_t *reorder_filter = NULL;

    if (blocked_format) {
        const char reorder_param0 = 'B';
        const dim_t reorder_param1 = k;
        const dim_t reorder_param2 = n;

        reorderAndCacheWeights<int8_t>(key_obj, filter, reorder_filter, k, n,
                                       ldb, is_weights_const, order, transB, reorder_param0, reorder_param1,
                                       reorder_param2,
                                       aocl_get_reorder_buf_size_s8s8s32os32, aocl_reorder_s8s8s32os32
                                      );
    }
    else {
        mem_format_b = 'n';
    }

    aocl_post_op *post_ops = NULL;
    float dummy_scale = (float)1.0;
    int8_t dummy_zp = (int8_t)0;
    //Create post_ops
    post_ops = create_aocl_post_ops_int8<int8_t>(ctx, po_ops, n, bias, bias_type,
               scale,
               zero_point_dst, out_scale_size, do_sum, output, &dummy_scale, &dummy_zp);

    aocl_gemm_s8s8s32os8(order, transA, transB, m,
                         n,
                         k, alpha, input,
                         lda, mem_format_a, blocked_format ? reorder_filter : filter,
                         ldb, mem_format_b, beta, output,
                         ldc, post_ops);

    // Free memory for postops.
    free_aocl_po_memory_int8(post_ops, zero_point_dst);
    // Free reordered weights if weights not const
    if (!is_weights_const && blocked_format) {
        free(reorder_filter);
    }
}

void zenMatMul_gemm_s8s8s32os32(
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
    int32_t *output,
    const int ldc,
    const float *scale,
    const int8_t zero_point_dst,
    const int out_scale_size,
    float do_sum,
    bool is_weights_const,
    bool blocked_format
) {
    Key_matmul key_obj(transpose_input, transpose_filter, m, k, n, lda, ldb, ldc,
                       filter, thread_qty, true);

    char transA = 'n', transB = 'n', order = 'r';
    char mem_format_a = 'n', mem_format_b = 'r';
    if (transpose_filter) {
        transB = 't';
    }
    if (transpose_input) {
        transA = 't';
    }
    int8_t *reorder_filter = NULL;

    if (blocked_format) {
        const char reorder_param0 = 'B';
        const dim_t reorder_param1 = k;
        const dim_t reorder_param2 = n;

        reorderAndCacheWeights<int8_t>(key_obj, filter, reorder_filter, k, n,
                                       ldb, is_weights_const, order, transB, reorder_param0, reorder_param1,
                                       reorder_param2,
                                       aocl_get_reorder_buf_size_s8s8s32os32, aocl_reorder_s8s8s32os32
                                      );
    }
    else {
        mem_format_b = 'n';
    }

    aocl_post_op *post_ops = NULL;
    float dummy_scale = (float)1.0;
    int8_t dummy_zp = (int8_t)0;
    //Create post_ops
    post_ops = create_aocl_post_ops_int8<int32_t>(ctx, po_ops, n, bias, bias_type,
               scale,
               zero_point_dst, out_scale_size, do_sum, output, &dummy_scale, &dummy_zp);

    aocl_gemm_s8s8s32os32(order, transA, transB, m,
                          n, k, alpha, input,
                          lda, mem_format_a, blocked_format ? reorder_filter : filter,
                          ldb, mem_format_b, beta, output,
                          ldc, post_ops);
    // Free memory for postops.
    free_aocl_po_memory_int8(post_ops, zero_point_dst);
    // Free reordered weights if weights not const
    if (!is_weights_const && blocked_format) {
        free(reorder_filter);
    }
}

void zenMatMul_gemm_u8s8s32os8(
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
    int8_t *output,
    const int ldc,
    const float *scale,
    const int8_t zero_point_dst,
    const int out_scale_size,
    float do_sum,
    bool is_weights_const,
    bool blocked_format
) {
    Key_matmul key_obj(transpose_input, transpose_filter, m, k, n, lda, ldb, ldc,
                       filter, thread_qty, true);

    char transA = 'n', transB = 'n', order = 'r';
    char mem_format_a = 'n', mem_format_b = 'r';
    if (transpose_filter) {
        transB = 't';
    }
    if (transpose_input) {
        transA = 't';
    }
    int8_t *reorder_filter = NULL;

    if (blocked_format) {
        const char reorder_param0 = 'B';
        const dim_t reorder_param1 = k;
        const dim_t reorder_param2 = n;

        reorderAndCacheWeights<int8_t>(key_obj, filter, reorder_filter, k, n,
                                       ldb, is_weights_const, order, transB, reorder_param0, reorder_param1,
                                       reorder_param2,
                                       aocl_get_reorder_buf_size_u8s8s32os32, aocl_reorder_u8s8s32os32
                                      );
    }
    else {
        mem_format_b = 'n';
    }
    aocl_post_op *post_ops = NULL;
    float dummy_scale = (float)1.0;
    int8_t dummy_zp = (int8_t)0;
    //Create post_ops
    post_ops = create_aocl_post_ops_int8<int8_t>(ctx, po_ops, n, bias, bias_type,
               scale,
               zero_point_dst, out_scale_size, do_sum, output, &dummy_scale, &dummy_zp);

    aocl_gemm_u8s8s32os8(order, transA, transB, m,
                         n, k, alpha, input,
                         lda, mem_format_a, blocked_format ? reorder_filter : filter,
                         ldb, mem_format_b, beta, output,
                         ldc, post_ops);

    // Free memory for postops.
    free_aocl_po_memory_int8(post_ops, zero_point_dst);
    // Free reordered weights if weights not const
    if (!is_weights_const && blocked_format) {
        free(reorder_filter);
    }
}

void zenMatMul_gemm_u8s8s32os32(
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
    int32_t *output,
    const int ldc,
    const float *scale,
    const int8_t zero_point_dst,
    const int out_scale_size,
    float do_sum,
    bool is_weights_const,
    bool blocked_format
) {
    Key_matmul key_obj(transpose_input, transpose_filter, m, k, n, lda, ldb, ldc,
                       filter, thread_qty, true);

    char transA = 'n', transB = 'n', order = 'r';
    char mem_format_a = 'n', mem_format_b = 'r';
    if (transpose_filter) {
        transB = 't';
    }
    if (transpose_input) {
        transA = 't';
    }
    int8_t *reorder_filter = NULL;

    if (blocked_format) {
        const char reorder_param0 = 'B';
        const dim_t reorder_param1 = k;
        const dim_t reorder_param2 = n;

        reorderAndCacheWeights<int8_t>(key_obj, filter, reorder_filter, k, n,
                                       ldb, is_weights_const, order, transB, reorder_param0, reorder_param1,
                                       reorder_param2,
                                       aocl_get_reorder_buf_size_u8s8s32os32, aocl_reorder_u8s8s32os32
                                      );
    }
    else {
        mem_format_b = 'n';
    }

    aocl_post_op *post_ops = NULL;
    float dummy_scale = (float)1.0;
    int8_t dummy_zp = (int8_t)0;
    //Create post_ops
    post_ops = create_aocl_post_ops_int8<int32_t>(ctx, po_ops, n, bias, bias_type,
               scale,
               zero_point_dst, out_scale_size, do_sum, output, &dummy_scale, &dummy_zp);

    aocl_gemm_u8s8s32os32(order, transA, transB, m,
                          n,
                          k, alpha, input,
                          lda, mem_format_a, blocked_format ? reorder_filter : filter,
                          ldb, mem_format_b, beta, output,
                          ldc, post_ops);
    // Free memory for postops.
    free_aocl_po_memory_int8(post_ops, zero_point_dst);
    // Free reordered weights if weights not const
    if (!is_weights_const && blocked_format) {
        free(reorder_filter);
    }
}

//Temporary checks for post-ops and data type for AOCL
bool check_dt_po_int8(
    int dst_type,
    int bias_type,
    float do_sum,
    const int32_t src_zp,
    const int32_t wei_zp
) {
    //Check sum post-op scale
    if (do_sum != 0.0 && do_sum != 1.0) {
        return false;
    }
    //check bias data type
    //Current support for bias is limited to BF16/S32/S8
    if (bias_type == zendnn_f32) {
        return false;
    }
    //Check src, weights zp
    if (src_zp != 0 || wei_zp != 0) {
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
    float *scale,
    const int32_t zero_point_src,
    const int32_t zero_point_wei,
    const int32_t zero_point_dst,
    int out_scale_size,
    float do_sum,
    bool is_weights_const
) {
    zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
    int thread_qty = zenEnvObj.omp_num_threads;
    if (!check_dt_po_int8(dst_type, bias_type, do_sum, zero_point_src,
                          zero_point_wei)) {
        zenEnvObj.zenINT8GEMMalgo = zenINT8MatMulAlgoType::MATMUL_BLOCKED_JIT_INT8;
    }
    if (zenEnvObj.zenINT8GEMMalgo ==
            zenINT8MatMulAlgoType::MATMUL_BLOCKED_AOCL_INT8) {
        int8_t zero_point_dst_8 = (int8_t)zero_point_dst;
        if (src_type == zendnn_s8) {
            if (dst_type == zendnn_s8) {
                zenMatMul_gemm_s8s8s32os8(ctx, thread_qty, Layout, transA, transB, M, K, N,
                                          alpha,
                                          (const int8_t *)src, lda, (const int8_t *)weights, ldb,
                                          (char *)bias, bias_type,
                                          po_ops, beta, (int8_t *)dst, ldc, scale,
                                          zero_point_dst_8, out_scale_size, do_sum, is_weights_const, true);
            }
            else if (dst_type == zendnn_s32) {
                zenMatMul_gemm_s8s8s32os32(ctx, thread_qty, Layout, transA, transB, M, K, N,
                                           alpha,
                                           (const int8_t *)src, lda, (const int8_t *)weights, ldb,
                                           (char *)bias, bias_type,
                                           po_ops, beta, (int32_t *)dst, ldc, scale,
                                           zero_point_dst_8, out_scale_size, do_sum, is_weights_const, true);
            }
            else {
                //dst float32
                //CALL BRGEMM Primitive
                obj.is_brgemm = true;
                zenMatMulPrimitiveINT8(ctx, thread_qty, src_type, dst_type, bias_type, Layout,
                                       transA, transB, M, N, K, src, weights, bias, dst, alpha, beta,
                                       lda, ldb, ldc, po_ops, true, scale, zero_point_src,
                                       zero_point_wei, zero_point_dst, out_scale_size, do_sum,
                                       is_weights_const);
                obj.is_brgemm = false;
            }
        }
        else { // make function for src:u8
            if (dst_type == zendnn_s8) {
                zenMatMul_gemm_u8s8s32os8(ctx, thread_qty, Layout, transA, transB, M, K, N,
                                          alpha,
                                          (const uint8_t *)src, lda, (const int8_t *)weights, ldb,
                                          (char *)bias, bias_type,
                                          po_ops, beta, (int8_t *)dst, ldc, scale,
                                          zero_point_dst_8, out_scale_size, do_sum, is_weights_const, true);
            }
            else if (dst_type == zendnn_s32) {
                zenMatMul_gemm_u8s8s32os32(ctx, thread_qty, Layout, transA, transB, M, K, N,
                                           alpha,
                                           (const uint8_t *)src, lda, (const int8_t *)weights, ldb,
                                           (char *)bias, bias_type,
                                           po_ops, beta, (int32_t *)dst, ldc, scale,
                                           zero_point_dst_8, out_scale_size, do_sum, is_weights_const, true);
            }
            else {
                //dst float32
                //CALL BRGEMM Primitive
                obj.is_brgemm = true;
                zenMatMulPrimitiveINT8(ctx, thread_qty, src_type, dst_type, bias_type, Layout,
                                       transA, transB, M, N, K, src, weights, bias, dst, alpha, beta,
                                       lda, ldb, ldc, po_ops, true, scale, zero_point_src,
                                       zero_point_wei, zero_point_dst, out_scale_size, do_sum,
                                       is_weights_const);
                obj.is_brgemm = false;
            }
        }
    }
    else if (zenEnvObj.zenINT8GEMMalgo == zenINT8MatMulAlgoType::MATMUL_AOCL_INT8) {
        int8_t zero_point_dst_8 = (int8_t)zero_point_dst;
        if (src_type == zendnn_s8) {
            if (dst_type == zendnn_s8) {
                zenMatMul_gemm_s8s8s32os8(ctx, thread_qty, Layout, transA, transB, M, K, N,
                                          alpha,
                                          (const int8_t *)src, lda, (const int8_t *)weights, ldb,
                                          (char *)bias, bias_type,
                                          po_ops, beta, (int8_t *)dst, ldc, scale,
                                          zero_point_dst_8, out_scale_size, do_sum, is_weights_const, false);
            }
            else if (dst_type == zendnn_s32) {
                zenMatMul_gemm_s8s8s32os32(ctx, thread_qty, Layout, transA, transB, M, K, N,
                                           alpha,
                                           (const int8_t *)src, lda, (const int8_t *)weights, ldb,
                                           (char *)bias, bias_type,
                                           po_ops, beta, (int32_t *)dst, ldc, scale,
                                           zero_point_dst_8, out_scale_size, do_sum, is_weights_const, false);
            }
            else {
                //dst float32
                //CALL BRGEMM Primitive
                obj.is_brgemm = true;
                zenMatMulPrimitiveINT8(ctx, thread_qty, src_type, dst_type, bias_type, Layout,
                                       transA, transB, M, N, K, src, weights, bias, dst, alpha, beta,
                                       lda, ldb, ldc, po_ops, true, scale, zero_point_src,
                                       zero_point_wei, zero_point_dst, out_scale_size, do_sum,
                                       is_weights_const);
                obj.is_brgemm = false;
            }
        }
        else { // make function for src:u8
            if (dst_type == zendnn_s8) {
                zenMatMul_gemm_u8s8s32os8(ctx, thread_qty, Layout, transA, transB, M, K, N,
                                          alpha,
                                          (const uint8_t *)src, lda, (const int8_t *)weights, ldb,
                                          (char *)bias, bias_type,
                                          po_ops, beta, (int8_t *)dst, ldc, scale,
                                          zero_point_dst_8, out_scale_size, do_sum, is_weights_const, false);
            }
            else if (dst_type == zendnn_s32) {
                zenMatMul_gemm_u8s8s32os32(ctx, thread_qty, Layout, transA, transB, M, K, N,
                                           alpha,
                                           (const uint8_t *)src, lda, (const int8_t *)weights, ldb,
                                           (char *)bias, bias_type,
                                           po_ops, beta, (int32_t *)dst, ldc, scale,
                                           zero_point_dst_8, out_scale_size, do_sum, is_weights_const, false);
            }
            else {
                //dst float32
                //CALL BRGEMM Primitive
                obj.is_brgemm = true;
                zenMatMulPrimitiveINT8(ctx, thread_qty, src_type, dst_type, bias_type, Layout,
                                       transA, transB, M, N, K, src, weights, bias, dst, alpha, beta,
                                       lda, ldb, ldc, po_ops, true, scale, zero_point_src,
                                       zero_point_wei, zero_point_dst, out_scale_size, do_sum,
                                       is_weights_const);
                obj.is_brgemm = false;
            }
        }
    }
    else if (zenEnvObj.zenINT8GEMMalgo ==
             zenINT8MatMulAlgoType::MATMUL_BLOCKED_JIT_INT8) {
        //CALL blocked BRGEMM Primitive
        obj.is_brgemm = true;
        zenMatMulPrimitiveINT8(ctx, thread_qty, src_type, dst_type, bias_type, Layout,
                               transA, transB, M, N, K, src, weights, bias, dst, alpha, beta, lda, ldb, ldc,
                               po_ops, true, scale, zero_point_src, zero_point_wei, zero_point_dst,
                               out_scale_size, do_sum, is_weights_const);
        obj.is_brgemm = false;
    }
    else {
        //CALL BRGEMM Primitive
        obj.is_brgemm = true;
        zenMatMulPrimitiveINT8(ctx, thread_qty, src_type, dst_type, bias_type, Layout,
                               transA, transB, M, N, K, src, weights, bias, dst, alpha, beta, lda, ldb, ldc,
                               po_ops, false, scale, zero_point_src, zero_point_wei, zero_point_dst,
                               out_scale_size, do_sum, is_weights_const);
        obj.is_brgemm = false;
    }
    obj.is_log = true;
    return zenEnvObj.zenINT8GEMMalgo ;
}
