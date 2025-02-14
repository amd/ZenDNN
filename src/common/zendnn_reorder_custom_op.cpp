/*******************************************************************************
* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "zendnn.hpp"
#include "zendnn_helper.hpp"
#include <vector>
#include <iostream>
#include "zendnn_logging.hpp"
#include "verbose.hpp"
#include <string.h>
#include <blis.h>
#include "common/zendnn_reorder_cache.hpp"

#define TAG_F32 tag::BA16a64b
#define TAG_BF16 tag::BA16a64b2a
#define TAG_INT8 tag::BA16a64b4a

using tag = memory::format_tag;
namespace zendnn {
// Currently supporting inplace only BRGEMM
bool reorder_brgemm_inplace(void *src, void *dst, uint k, uint n,
                            bool trans_mem, zendnn_data_type_t dt, zendnnEnv zenEnvObj) {
    // TODO:Remove key dependency
    Key_matmul key_obj(false, trans_mem, 1, k, n, k, trans_mem ? k : n, n, src,
                       zenEnvObj.omp_num_threads,
                       false);

    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);
    zendnn::memory user_weights_memory, reordered_weights_memory;

    memory::desc matmul_weights_md = memory::desc({k,n}, (memory::data_type)dt,
                                     trans_mem ? tag::ba : tag::ab);
    user_weights_memory = memory(matmul_weights_md, eng, src);
    if (dt == zendnn_f32) {
        memory::desc blocked_matmul_weights_md = memory::desc({k,n}, (
                    memory::data_type)dt, TAG_F32);
        if (matmul_weights_md.get_size() != blocked_matmul_weights_md.get_size()) {
            return false;
        }

        reorderAndCacheWeightsBrgemm(
            key_obj, blocked_matmul_weights_md, user_weights_memory,
            reordered_weights_memory, eng, engine_stream, true/*is_weights_const*/,
            zendnnWeightCacheType::WEIGHT_CACHE_AOT_REORDER);
    }
    else if (dt == zendnn_bf16) {
        memory::desc blocked_matmul_weights_md = memory::desc({k,n}, (
                    memory::data_type)dt, TAG_BF16);
        if (matmul_weights_md.get_size() != blocked_matmul_weights_md.get_size()) {
            return false;
        }
        reorderAndCacheWeightsBrgemm(
            key_obj, blocked_matmul_weights_md, user_weights_memory,
            reordered_weights_memory, eng, engine_stream, true/*is_weights_const*/,
            zendnnWeightCacheType::WEIGHT_CACHE_AOT_REORDER);
    }
    else if (dt == zendnn_s8) {
        memory::desc blocked_matmul_weights_md = memory::desc({k,n}, (
                    memory::data_type)dt, TAG_INT8);
        if (matmul_weights_md.get_size() != blocked_matmul_weights_md.get_size()) {
            return false;
        }
        reorderAndCacheWeightsBrgemm(
            key_obj, blocked_matmul_weights_md, user_weights_memory,
            reordered_weights_memory, eng, engine_stream, true/*is_weights_const*/,
            zendnnWeightCacheType::WEIGHT_CACHE_AOT_REORDER);
    }

    return true;
}
// Currently supporting inplace only
bool reorder_aocl_inplace(void *src, void *dst, uint k, uint n, bool trans_mem,
                          zendnn_data_type_t dt, zendnnEnv zenEnvObj) {
    const char reorder_param0 = 'B';
    const dim_t reorder_param1 = k;
    const dim_t reorder_param2 = n;
    const char order = 'r';
    char trans = 'n';
    if (trans_mem) {
        trans = 't';
    }

    // TODO:Remove key dependency
    Key_matmul key_obj(false, trans_mem, 1, k, n, k, trans_mem ? k : n, n, src,
                       zenEnvObj.omp_num_threads,
                       false);
    if (dt == zendnn_f32) {
        int siz_req = aocl_get_reorder_buf_size_f32f32f32of32(order, trans,
                      reorder_param0, reorder_param1, reorder_param2);
        // TODO: get size of data type using some function
        if (siz_req != 4*k*n) {
            return false;
        }
        float *temp = NULL;
        reorderAndCacheWeights<float>(key_obj, (float *)src, temp, k, n,
                                      trans_mem ? k : n, true/*weights const*/, order, trans, reorder_param0,
                                      reorder_param1, reorder_param2,
                                      aocl_get_reorder_buf_size_f32f32f32of32, aocl_reorder_f32f32f32of32,
                                      zendnnWeightCacheType::WEIGHT_CACHE_AOT_REORDER);
    }
    else if (dt == zendnn_bf16) {
        int siz_req = aocl_get_reorder_buf_size_bf16bf16f32of32(order, trans,
                      reorder_param0, reorder_param1, reorder_param2);
        // TODO: get size of data type using some function
        if (siz_req != 2*k*n) {
            return false;
        }
        int16_t *temp = NULL;
        reorderAndCacheWeights<int16_t>(key_obj, (int16_t *)src, temp, k, n,
                                        trans_mem ? k : n, true/*weights const*/, order, trans, reorder_param0,
                                        reorder_param1, reorder_param2,
                                        aocl_get_reorder_buf_size_bf16bf16f32of32, aocl_reorder_bf16bf16f32of32,
                                        zendnnWeightCacheType::WEIGHT_CACHE_AOT_REORDER);
    }
    // Current support is for activation U8, Weights S8 only.
    else if (dt == zendnn_s8) {
        int siz_req = aocl_get_reorder_buf_size_u8s8s32os32(order, trans,
                      reorder_param0, reorder_param1, reorder_param2);
        if (siz_req != k*n) {
            return false;
        }
        int8_t *temp = NULL;
        reorderAndCacheWeights<int8_t>(key_obj, (int8_t *)src, temp, k, n,
                                       trans_mem ? k : n, true/*weights const*/, order, trans, reorder_param0,
                                       reorder_param1, reorder_param2,
                                       aocl_get_reorder_buf_size_u8s8s32os32, aocl_reorder_u8s8s32os32,
                                       zendnnWeightCacheType::WEIGHT_CACHE_AOT_REORDER);
    }
    return true;
}

// returns size required by AOCL blocked format weights
size_t get_aocl_size(uint k, uint n, bool trans_mem, zendnn_data_type_t src_dt,
                     int src_zp, zendnn_data_type_t dt) {

    size_t req_buff_size = 0;
    const char reorder_param0 = 'B';
    const dim_t reorder_param1 = k;
    const dim_t reorder_param2 = n;
    const char order = 'r';
    char trans = 'n';
    if (trans_mem) {
        trans = 't';
    }
    if (dt == zendnn_f32) {
        req_buff_size = aocl_get_reorder_buf_size_f32f32f32of32(order, trans,
                        reorder_param0, reorder_param1, reorder_param2);
    }
    else if (dt == zendnn_bf16) {
        req_buff_size = aocl_get_reorder_buf_size_bf16bf16f32of32(order, trans,
                        reorder_param0, reorder_param1, reorder_param2);
    }
    // Current support is for activation U8, Weights S8 only.
    else if (dt == zendnn_s8) {
        if (src_dt == zendnn_u8) {
            req_buff_size = aocl_get_reorder_buf_size_u8s8s32os32(order, trans,
                            reorder_param0, reorder_param1, reorder_param2);
        }
        else if (src_dt == zendnn_s8) {
            req_buff_size = aocl_get_reorder_buf_size_s8s8s32os32(order, trans,
                            reorder_param0, reorder_param1, reorder_param2);
        }
        if (src_zp) {
            req_buff_size += n*sizeof(int32_t);
        }
    }

    return req_buff_size;
}

// returns size required by BRGEMM blocked format weights
size_t get_brgemm_size(uint k, uint n, bool trans, zendnn_data_type_t src_dt,
                       int src_zp, zendnn_data_type_t dt) {
    zendnn::engine eng(engine::kind::cpu, 0);
    zendnn::stream engine_stream(eng);
    memory::desc blocked_matmul_weights_md;
    if (dt == zendnn_f32) {
        blocked_matmul_weights_md = memory::desc({k,n}, (
                                        memory::data_type)dt, TAG_F32);
    }
    else if (dt == zendnn_bf16) {
        blocked_matmul_weights_md = memory::desc({k,n}, (
                                        memory::data_type)dt, TAG_BF16);
    }
    // Current support is for activation U8, Weights S8 only.
    else if (dt == zendnn_s8) {
        blocked_matmul_weights_md = memory::desc({k,n}, (
                                        memory::data_type)dt, TAG_INT8);
    }
    memory::desc want_B_md = blocked_matmul_weights_md;
    if (src_dt == zendnn_s8) {
        want_B_md.data.extra.flags |= zendnn_memory_extra_flag_compensation_conv_s8s8;
        want_B_md.data.extra.compensation_mask = (1 << 1);
    }
    if (src_zp) {
        want_B_md.data.extra.flags
        |= zendnn_memory_extra_flag_compensation_conv_asymmetric_src;
        want_B_md.data.extra.asymm_compensation_mask = (1 << 1);
    }
    blocked_matmul_weights_md = want_B_md;
    return blocked_matmul_weights_md.get_size();
}


// Returns backend/ALGO for given data type
unsigned int fetch_backend(zendnn_data_type_t dt, zendnnEnv zenEnvObj) {
    // Return 0 if AUTO is set
    if (dt == zendnn_s8) {
        return zenEnvObj.zenINT8GEMMalgo == zenINT8MatMulAlgoType::MATMUL_AUTO_INT8 ? 0:
               zenEnvObj.zenINT8GEMMalgo;
    }
    else if (dt == zendnn_bf16) {
        return zenEnvObj.zenBF16GEMMalgo == zenBF16MatMulAlgoType::MATMUL_AUTO_BF16 ? 0:
               zenEnvObj.zenBF16GEMMalgo;
    }
    else if (dt == zendnn_f32) {
        return zenEnvObj.zenGEMMalgo == zenMatMulAlgoType::MATMUL_AUTO_FP32 ? 0 :
               zenEnvObj.zenGEMMalgo;
    }
    else {
        // return 401 for unsupported data_type.
        return 401;
    }
}

// ZenDNN reorder API for AOCL and BRGEMM
bool zendnn_custom_op::zendnn_reorder(void *src, void *dst, uint k, uint n,
                                      bool trans, zendnn_data_type_t dt) {
    bool status = false;
    zendnnEnv zenEnvObj = readEnv();
    unsigned int backend = fetch_backend(dt, zenEnvObj);
    unsigned int weight_cache_type = zenEnvObj.zenWeightCache;
    // TODO: Support AOT_RESIZED_INPLACE
    if (weight_cache_type != zendnnWeightCacheType::WEIGHT_CACHE_AOT_INPLACE) {
        return false;
    }
    if (backend == 1/*aocl*/) {
        status = reorder_aocl_inplace(src, dst, k, n, trans, dt, zenEnvObj);
        zendnnVerbose(ZENDNN_PROFLOG,"AOCL reorder custom op,", " status ",
                      status ? "True" : "False");
    }
    else if (backend == 2/*brgemm*/ || backend == 0) {
        status = reorder_brgemm_inplace(src, dst, k, n, trans, dt, zenEnvObj);
        zendnnVerbose(ZENDNN_PROFLOG,"BRGEMM reorder custom op,", " status ",
                      status ? "True" : "False");
    }
    else if (backend == 3 || backend == 4/*non-blocked*/) {
        status = true;
        zendnnVerbose(ZENDNN_PROFLOG,"No Blocking reorder custom op,", " status ",
                      status ? "True" : "False");
    }
    else {
        ZENDNN_THROW_ERROR(zendnn_invalid_arguments,
                           "Unsupported backend argument passed for zendnn_reorder");
    }
    return status;
}
//Returns size for reorder
size_t zendnn_custom_op::zendnn_reorder_size(uint k, uint n, bool trans,
        zendnn_data_type_t src_dt, int src_zp, zendnn_data_type_t weight_dt) {

    size_t req_bytes = 0;
    zendnnEnv zenEnvObj = readEnv();
    unsigned int backend = fetch_backend(weight_dt, zenEnvObj);
    // Check Backend
    if (backend == 1/*aocl*/) {
        req_bytes = get_aocl_size(k, n, trans, src_dt, src_zp, weight_dt);
    }
    else if (backend == 2/*brgemm*/ || backend == 0) {
        req_bytes = get_brgemm_size(k, n, trans, src_dt, src_zp, weight_dt);
    }
    else if (backend == 3 || backend == 4/*non-blocked*/) {
        req_bytes = k*n;
        if (weight_dt == zendnn_f32) {
            req_bytes *= sizeof(float);
        }
        else if (weight_dt == zendnn_bf16) {
            req_bytes *= sizeof(int16_t);
        }
    }
    else {
        ZENDNN_THROW_ERROR(zendnn_invalid_arguments,
                           "Unsupported backend argument passed for zendnn_reorder_size");
    }
    return req_bytes;
}
}
