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
#include <vector>
#include <unordered_map>
#include <tuple>

#ifndef _WIN32
    #include <sys/time.h>
#else
    #include <windows.h>
#endif

#ifndef ZENDNN_USE_AOCL_BLIS_API
    #include <cblas.h>
#else // ZENDNN_USE_AOCL_BLIS_API
    #include "cblas_with_blis_api.hpp"
#endif // ZENDNN_USE_AOCL_BLIS_API

#include "common/c_types_map.hpp"
#include "common/zendnn_thread.hpp"
#include "common/type_helpers.hpp"
#include "zendnn_helper.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/platform.hpp"
#include "cpu/ref_io_helper.hpp"

#include "cpu/gemm/gemm.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/matmul/zendnn_bf16_matmul.hpp"
#include "cpu/matmul/matmul_utils.hpp"

#include "zendnn_logging.hpp"
#include "common/zendnn_private.hpp"
#include "zendnn.hpp"
#include "zendnn_reorder_cache.hpp"

using namespace ::zendnn::impl;

template<typename T>
aocl_post_op *create_aocl_post_ops(const exec_ctx_t &ctx,
                                   const post_ops_t &po,
                                   int n, const float alpha, const char *bias,
                                   int bias_type, const bool relu, const int gelu,
                                   T *sum_buff, int &postop_count,
                                   const float *scale, float *dummy_scale) {
    aocl_post_op *post_ops = NULL;
    if (bias != NULL) {
        ++postop_count;
    }

    // Create postop for LPGEMM
    // Order of postops: BIAS -> RELU
    if (po.len() + postop_count > 0) {
        post_ops = (aocl_post_op *) malloc(sizeof(aocl_post_op));
        if (post_ops == NULL) {
            return NULL;
        }
        dim_t max_post_ops_seq_length = po.len() + postop_count;
        post_ops->seq_vector = (AOCL_POST_OP_TYPE *) malloc(max_post_ops_seq_length *
                               sizeof(AOCL_POST_OP_TYPE));
        if (post_ops->seq_vector == NULL) {
            std::free(post_ops);
            return NULL;
        }

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
            if (post_ops->bias == NULL) {
                std::free(post_ops->seq_vector);
                std::free(post_ops);
                return NULL;
            }
            if (bias_type == zendnn_bf16) {
                (post_ops->bias + bias_index)->bias = (int16_t *)bias;
                (post_ops->bias)->stor_type = AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_BF16;
            }
            else if (bias_type == zendnn_f32) {
                (post_ops->bias + bias_index)->bias = (float *)bias;
                (post_ops->bias)->stor_type = AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_F32;
            }
            else {
                zendnnError(ZENDNN_ALGOLOG,
                            "Check Bias data type, only f32 and bf16 are supported");
            }
        }

        //Scale post-op
        if (scale) {
            post_ops->sum = (aocl_post_op_sum *) malloc(sizeof(
                                aocl_post_op_sum));

            if (post_ops->sum == NULL) {
                if (post_ops->bias != NULL) {
                    std::free(post_ops->bias);
                }
                std::free(post_ops->seq_vector);
                std::free(post_ops);
                return NULL;
            }
            post_ops->seq_vector[post_op_i++] = SCALE;
            (post_ops->sum + scale_index)->is_power_of_2 = FALSE;
            (post_ops->sum + scale_index)->scale_factor = NULL;
            (post_ops->sum + scale_index)->buff = NULL;
            (post_ops->sum + scale_index)->zero_point = NULL;

            (post_ops->sum + scale_index)->scale_factor = malloc(sizeof(float));
            (post_ops->sum + scale_index)->zero_point = malloc(sizeof(int16_t));

            //SCALE
            float *temp_dscale_ptr = (float *)(post_ops->sum + scale_index)->scale_factor;
            int16_t *temp_dzero_point_ptr = (int16_t *)(post_ops->sum +
                                            scale_index)->zero_point;
            temp_dscale_ptr[0] = (float)(scale[0]);

            temp_dzero_point_ptr[0] = (int16_t)0;

            (post_ops->sum + scale_index)->scale_factor_len = 1;
            (post_ops->sum + scale_index)->zero_point_len = 1;
            scale_index++;
        }
        //Get count of eltwise and binary post-ops
        int mem_count[3] = {0};
        for (auto idx = 0; idx < po.len(); ++idx) {
            const auto po_type = po.entry_[idx];
            switch (po_type.kind) {
            case primitive_kind::eltwise:
                mem_count[0]++;
                break;
            case primitive_kind::binary:
                if (po_type.binary.alg == alg_kind::binary_add) {
                    mem_count[1]++;
                }
                else if (po_type.binary.alg == alg_kind::binary_mul) {
                    mem_count[2]++;
                }
                break;
            case primitive_kind::sum:
                //condition gemm_applies beta for alpha = 1.0
                if (scale != NULL && scale[0] != 1.0) {
                    mem_count[1]++;
                }
                else {
                    postop_count -= 1; //Since sum post-op is not applied.
                }
                break;
            default:
                break;
            }
        }

        if (mem_count[0] > 0) {
            post_ops->eltwise = (aocl_post_op_eltwise *) malloc(sizeof(
                                    aocl_post_op_eltwise)*mem_count[0]);
            if (post_ops->eltwise == NULL) {
                if (post_ops->sum != NULL) {
                    std::free(post_ops->sum);
                }
                if (post_ops->bias != NULL) {
                    std::free(post_ops->bias);
                }
                std::free(post_ops->seq_vector);
                std::free(post_ops);
                return NULL;
            }
        }
        if (mem_count[1] > 0) {
            post_ops->matrix_add = (aocl_post_op_matrix_add *) malloc(sizeof(
                                       aocl_post_op_matrix_add)*mem_count[1]);
            if (post_ops->matrix_add == NULL) {
                if (post_ops->eltwise != NULL) {
                    std::free(post_ops->eltwise);
                }
                if (post_ops->sum != NULL) {
                    std::free(post_ops->sum);
                }
                if (post_ops->bias != NULL) {
                    std::free(post_ops->bias);
                }
                std::free(post_ops->seq_vector);
                std::free(post_ops);
                return NULL;
            }
        }
        if (mem_count[2] > 0) {
            post_ops->matrix_mul = (aocl_post_op_matrix_mul *) malloc(sizeof(
                                       aocl_post_op_matrix_mul)*mem_count[2]);
            if (post_ops->matrix_mul == NULL) {
                if (post_ops->matrix_add != NULL) {
                    std::free(post_ops->matrix_add);
                }
                if (post_ops->eltwise != NULL) {
                    std::free(post_ops->eltwise);
                }
                if (post_ops->sum != NULL) {
                    std::free(post_ops->sum);
                }
                if (post_ops->bias != NULL) {
                    std::free(post_ops->bias);
                }
                std::free(post_ops->seq_vector);
                std::free(post_ops);
                return NULL;
            }
        }

        //Add eltwise and binary post-ops in given sequence
        for (auto idx = 0; idx < po.len(); ++idx) {

            const auto po_type = po.entry_[idx];
            if (po_type.kind == primitive_kind::eltwise &&
                    post_ops->eltwise != NULL) {

                if (po_type.eltwise.alg == alg_kind::eltwise_relu) {
                    // Add ReLU postop
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = RELU;
                    eltwise_index++;
                }
                else if (po_type.eltwise.alg == alg_kind::eltwise_gelu) {
                    // Gelu tanh.
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = GELU_TANH;
                    eltwise_index++;
                }
                else if (po_type.eltwise.alg == alg_kind::eltwise_gelu_erf) {
                    // Gelu erf.
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = GELU_ERF;
                    eltwise_index++;
                }
                else if (po_type.eltwise.alg == alg_kind::eltwise_logistic) {
                    // Sigmoid
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = SIGMOID;
                    eltwise_index++;
                }
                else if (po_type.eltwise.alg == alg_kind::eltwise_swish) {
                    // Silu.
                    post_ops->seq_vector[post_op_i++] = ELTWISE;
                    (post_ops->eltwise + eltwise_index)->is_power_of_2 = FALSE;
                    (post_ops->eltwise + eltwise_index)->scale_factor = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.alpha = malloc(sizeof(float));
                    *((float *)(post_ops->eltwise + eltwise_index)->algo.alpha) = (float)1;
                    (post_ops->eltwise + eltwise_index)->algo.beta = NULL;
                    (post_ops->eltwise + eltwise_index)->algo.algo_type = SWISH;
                    eltwise_index++;
                }
            }
            else if (po_type.kind == primitive_kind::binary) {
                if (po_type.binary.alg == alg_kind::binary_add &&
                        post_ops->matrix_add != NULL) {
                    post_ops->seq_vector[post_op_i++] = MATRIX_ADD;
                    (post_ops->matrix_add + add_index)->ldm = n;
                    auto binary_po = CTX_IN_MEM(const void *,
                                                (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1));
                    //Currently putting 1.0 as scale
                    (post_ops->matrix_add + add_index)->scale_factor = (float *)dummy_scale;
                    (post_ops->matrix_add + add_index)->scale_factor_len = 1;
                    auto b_dt = po_type.binary.src1_desc.data_type;
                    if (b_dt == zendnn_bf16) {
                        (post_ops->matrix_add + add_index)->stor_type =
                            AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_BF16;
                        auto addA = reinterpret_cast<int16_t *>(const_cast<void *>(binary_po));
                        (post_ops->matrix_add + add_index)->matrix = (int16_t *)addA;
                    }
                    else if (b_dt == zendnn_f32) {
                        (post_ops->matrix_add + add_index)->stor_type =
                            AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_F32;
                        auto addA = reinterpret_cast<float *>(const_cast<void *>(binary_po));
                        (post_ops->matrix_add + add_index)->matrix = (float *)addA;
                    }
                    add_index++;
                }
                else if (po_type.binary.alg == alg_kind::binary_mul &&
                         post_ops->matrix_mul != NULL) {
                    post_ops->seq_vector[post_op_i++] = MATRIX_MUL;
                    (post_ops->matrix_mul + mul_index)->ldm = n;
                    auto binary_po = CTX_IN_MEM(const void *,
                                                (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1));
                    //Currently putting 1.0 as scale
                    (post_ops->matrix_mul + mul_index)->scale_factor = (float *)dummy_scale;
                    (post_ops->matrix_mul + mul_index)->scale_factor_len = 1;
                    auto b_dt = po_type.binary.src1_desc.data_type;
                    if (b_dt == zendnn_bf16) {
                        (post_ops->matrix_mul + mul_index)->stor_type =
                            AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_BF16;
                        auto mulA = reinterpret_cast<int16_t *>(const_cast<void *>(binary_po));
                        (post_ops->matrix_mul + mul_index)->matrix = (int16_t *)mulA;
                    }
                    else if (b_dt == zendnn_f32) {
                        (post_ops->matrix_mul + mul_index)->stor_type =
                            AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_F32;
                        auto mulA = reinterpret_cast<float *>(const_cast<void *>(binary_po));
                        (post_ops->matrix_mul + mul_index)->matrix = (float *)mulA;
                    }
                    mul_index++;
                }
            }
            else if (po_type.kind == primitive_kind::sum &&
                     post_ops->matrix_add != NULL) {
                if (scale[0] != 1.0) {
                    post_ops->seq_vector[post_op_i++] = MATRIX_ADD;
                    (post_ops->matrix_add + add_index)->ldm = n;
                    //Currently putting 1.0 as scale
                    (post_ops->matrix_add + add_index)->scale_factor = (float *)dummy_scale;
                    (post_ops->matrix_add + add_index)->scale_factor_len = 1;

                    (post_ops->matrix_add + add_index)->matrix = (T *)sum_buff;
                    add_index++;
                }
            }
        }
        post_ops->seq_length = po.len() + postop_count;
    }
    return (post_ops);
}

template aocl_post_op *create_aocl_post_ops<float>(const exec_ctx_t &ctx,
        const post_ops_t &po,
        int n, const float alpha, const char *bias,
        int bias_type, const bool relu, const int gelu,
        float *sum_buff, int &postop_count,
        const float *scale, float *dummy_scale);

template aocl_post_op *create_aocl_post_ops<int16_t>(const exec_ctx_t
        &ctx,
        const post_ops_t &po,
        int n, const float alpha, const char *bias,
        int bias_type, const bool relu, const int gelu,
        int16_t *sum_buff, int &postop_count,
        const float *scale, float *dummy_scale);

void create_post_ops_fp32(aocl_post_op *&post_ops, const exec_ctx_t &ctx,
                          const post_ops_t &po_ops,
                          const float *bias, float alpha, int n, int thread_qty, dim_t &eltwise_index,
                          float &dummy_scale, size_t index_offset) {

    int postop_count = 0;

    if (bias != NULL) {
        ++postop_count;
    }

    dim_t bias_index = 0;
    dim_t add_index = 0;
    dim_t mul_index = 0;

    if (po_ops.len() + postop_count > 0) {
        post_ops = (aocl_post_op *) malloc(sizeof(aocl_post_op));
        if (post_ops == NULL) {
            return ;
        }

        dim_t max_post_ops_seq_length = po_ops.len() + postop_count;
        post_ops->seq_vector = (AOCL_POST_OP_TYPE *) malloc(max_post_ops_seq_length *
                               sizeof(AOCL_POST_OP_TYPE));
        if (post_ops->seq_vector == NULL) {
            std::free(post_ops);
            return ;
        }

        int post_op_i = 0;
        post_ops->eltwise = NULL;
        post_ops->bias = NULL;
        post_ops->sum = NULL;
        post_ops->matrix_add = NULL;
        post_ops->matrix_mul = NULL;

        if (bias != NULL) {
            // Add bias postop
            float *bias_ = NULL;
            if (alpha != 1.0f) {
                bias_ = new float[n]();
                #pragma omp parallel for num_threads(thread_qty)
                for (int i=0; i<n; ++i) {
                    bias_[i] = alpha * bias[i];
                }
            }
            post_ops->seq_vector[post_op_i++] = BIAS;
            post_ops->bias = (aocl_post_op_bias *) malloc(sizeof(
                                 aocl_post_op_bias));

            (post_ops->bias)->bias = (alpha!=1.0f) ? bias_ : (float *)bias;
        }

        int mem_count[3] = {0};
        for (auto idx = 0; idx < po_ops.len(); ++idx) {
            const auto &po_type = po_ops.entry_[idx];
            switch (po_type.kind) {
            case primitive_kind::eltwise:
                mem_count[0]++;
                break;
            case primitive_kind::binary:
                if (po_type.binary.alg == alg_kind::binary_add) {
                    mem_count[1]++;
                }
                else if (po_type.binary.alg == alg_kind::binary_mul) {
                    mem_count[2]++;
                }
                break;
            default:
                break;
            }
        }

        if (mem_count[0] > 0) {
            post_ops->eltwise = (aocl_post_op_eltwise *) malloc(sizeof(
                                    aocl_post_op_eltwise) * mem_count[0]);
            if (post_ops->eltwise == NULL) {
                if (post_ops->bias != NULL) {
                    std::free(post_ops->bias);
                }
                std::free(post_ops->seq_vector);
                std::free(post_ops);
                return ;
            }
        }
        if (mem_count[1] > 0) {
            post_ops->matrix_add = (aocl_post_op_matrix_add *) malloc(sizeof(
                                       aocl_post_op_matrix_add) * mem_count[1]);
            if (post_ops->matrix_add == NULL) {
                if (post_ops->eltwise != NULL) {
                    std::free(post_ops->eltwise);
                }
                if (post_ops->bias != NULL) {
                    std::free(post_ops->bias);
                }
                std::free(post_ops->seq_vector);
                std::free(post_ops);
                return ;
            }
        }
        if (mem_count[2] > 0) {
            post_ops->matrix_mul = (aocl_post_op_matrix_mul *) malloc(sizeof(
                                       aocl_post_op_matrix_mul) * mem_count[2]);
            if (post_ops->matrix_mul == NULL) {
                if (post_ops->matrix_add != NULL) {
                    std::free(post_ops->matrix_add);
                }
                if (post_ops->eltwise != NULL) {
                    std::free(post_ops->eltwise);
                }
                if (post_ops->bias != NULL) {
                    std::free(post_ops->bias);
                }
                std::free(post_ops->seq_vector);
                std::free(post_ops);
                return ;
            }
        }

        for (auto idx = 0; idx < po_ops.len(); ++idx) {
            const auto &po_type = po_ops.entry_[idx];
            if (po_type.kind == primitive_kind::eltwise &&
                    post_ops->eltwise != NULL) {
                if (po_type.eltwise.alg == alg_kind::eltwise_relu) {
                    post_ops->seq_vector[post_op_i++] = AOCL_POST_OP_TYPE::ELTWISE;
                    post_ops->eltwise[eltwise_index].is_power_of_2 = FALSE;
                    post_ops->eltwise[eltwise_index].scale_factor = NULL;
                    post_ops->eltwise[eltwise_index].algo.alpha = NULL;
                    post_ops->eltwise[eltwise_index].algo.beta = NULL;
                    post_ops->eltwise[eltwise_index].algo.algo_type = AOCL_ELT_ALGO_TYPE::RELU;
                    eltwise_index+=1;
                }
                else if (po_type.eltwise.alg == alg_kind::eltwise_gelu) {
                    post_ops->seq_vector[post_op_i++] = AOCL_POST_OP_TYPE::ELTWISE;
                    post_ops->eltwise[eltwise_index].is_power_of_2 = FALSE;
                    post_ops->eltwise[eltwise_index].scale_factor = NULL;
                    post_ops->eltwise[eltwise_index].algo.alpha = NULL;
                    post_ops->eltwise[eltwise_index].algo.beta = NULL;
                    post_ops->eltwise[eltwise_index].algo.algo_type = AOCL_ELT_ALGO_TYPE::GELU_TANH;
                    eltwise_index+=1;
                }
                else if (po_type.eltwise.alg == alg_kind::eltwise_gelu_erf) {
                    post_ops->seq_vector[post_op_i++] = AOCL_POST_OP_TYPE::ELTWISE;
                    post_ops->eltwise[eltwise_index].is_power_of_2 = FALSE;
                    post_ops->eltwise[eltwise_index].scale_factor = NULL;
                    post_ops->eltwise[eltwise_index].algo.alpha = NULL;
                    post_ops->eltwise[eltwise_index].algo.beta = NULL;
                    post_ops->eltwise[eltwise_index].algo.algo_type = AOCL_ELT_ALGO_TYPE::GELU_ERF;
                    eltwise_index+=1;
                }
                else if (po_type.eltwise.alg == alg_kind::eltwise_logistic) {
                    post_ops->seq_vector[post_op_i++] = AOCL_POST_OP_TYPE::ELTWISE;
                    post_ops->eltwise[eltwise_index].is_power_of_2 = FALSE;
                    post_ops->eltwise[eltwise_index].scale_factor = NULL;
                    post_ops->eltwise[eltwise_index].algo.alpha = NULL;
                    post_ops->eltwise[eltwise_index].algo.beta = NULL;
                    post_ops->eltwise[eltwise_index].algo.algo_type = AOCL_ELT_ALGO_TYPE::SIGMOID;
                    eltwise_index+=1;
                }
                else if (po_type.eltwise.alg == alg_kind::eltwise_swish) {
                    post_ops->seq_vector[post_op_i++] = AOCL_POST_OP_TYPE::ELTWISE;
                    post_ops->eltwise[eltwise_index].is_power_of_2 = FALSE;
                    post_ops->eltwise[eltwise_index].scale_factor = NULL;
                    post_ops->eltwise[eltwise_index].algo.alpha = malloc(sizeof(float));
                    *((float *)(post_ops->eltwise[eltwise_index].algo.alpha)) = 1.0f;
                    post_ops->eltwise[eltwise_index].algo.beta = NULL;
                    post_ops->eltwise[eltwise_index].algo.algo_type = AOCL_ELT_ALGO_TYPE::SWISH;
                    eltwise_index+=1;
                }
            }
            else if (po_type.kind == primitive_kind::binary) {
                if (po_type.binary.alg == alg_kind::binary_add &&
                        post_ops->matrix_add != NULL) {
                    post_ops->seq_vector[post_op_i++] = AOCL_POST_OP_TYPE::MATRIX_ADD;
                    post_ops->matrix_add[add_index].ldm = n;
                    auto binary_po = CTX_IN_MEM(const void *,
                                                (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1));
                    post_ops->matrix_add[add_index].scale_factor = &dummy_scale;
                    post_ops->matrix_add[add_index].scale_factor_len = 1;
                    auto b_dt = po_type.binary.src1_desc.data_type;
                    const zendnn_dim_t *dims = po_type.binary.src1_desc.dims;
                    int ndims = po_type.binary.src1_desc.ndims;
                    size_t binary_offset = (ndims == 3) ? index_offset * dims[1] * dims[2]: 0;
                    post_ops->matrix_add[add_index].stor_type =
                        AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_F32;
                    auto addA = reinterpret_cast<float *>(const_cast<void *>
                                                          (binary_po)) + binary_offset;
                    post_ops->matrix_add[add_index].matrix = (float *)addA;
                    add_index++;
                }
                else if (po_type.binary.alg == alg_kind::binary_mul &&
                         post_ops->matrix_mul != NULL) {
                    post_ops->seq_vector[post_op_i++] = AOCL_POST_OP_TYPE::MATRIX_MUL;
                    post_ops->matrix_mul[mul_index].ldm = n;
                    auto binary_po = CTX_IN_MEM(const void *,
                                                (ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(idx) | ZENDNN_ARG_SRC_1));
                    post_ops->matrix_mul[mul_index].scale_factor = &dummy_scale;
                    post_ops->matrix_mul[mul_index].scale_factor_len = 1;
                    auto b_dt = po_type.binary.src1_desc.data_type;
                    const zendnn_dim_t *dims = po_type.binary.src1_desc.dims;
                    int ndims = po_type.binary.src1_desc.ndims;
                    size_t binary_offset = (ndims == 3) ? index_offset * dims[1] * dims[2] : 0;
                    post_ops->matrix_mul[mul_index].stor_type =
                        AOCL_PARAMS_STORAGE_TYPES::AOCL_GEMM_F32;
                    auto mulA = reinterpret_cast<float *>(const_cast<void *>
                                                          (binary_po)) + binary_offset;
                    post_ops->matrix_mul[mul_index].matrix = (float *)mulA;
                    mul_index++;
                }
            }
        }
        post_ops->seq_length = po_ops.len() + postop_count;
    }
    return ;
}

void clear_post_ops_memory(aocl_post_op *post_ops, float alpha,
                           dim_t eltwise_index) {
    if (post_ops != NULL) {
        if (post_ops->bias != NULL) {
            if (alpha != 1.0) {
                delete ((float *)post_ops->bias->bias);
            }
            else {
                post_ops->bias->bias = NULL;
            }
            std::free(post_ops->bias);
        }
        if (post_ops->matrix_add != NULL) {
            std::free(post_ops->matrix_add);
        }
        if (post_ops->matrix_mul != NULL) {
            std::free(post_ops->matrix_mul);
        }
        if (post_ops->eltwise != NULL) {
            for (int i = 0; i < eltwise_index; ++i) {
                if (post_ops->eltwise[i].algo.alpha != NULL) {
                    std::free(post_ops->eltwise[i].algo.alpha);
                }
            }
            std::free(post_ops->eltwise);
        }
        std::free(post_ops->seq_vector);
        std::free(post_ops);
    }
}
