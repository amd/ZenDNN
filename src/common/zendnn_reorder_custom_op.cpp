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
                            bool trans_mem,
                            zendnn_data_type_t dt) {
    // TODO:Remove key dependency
    Key_matmul key_obj(false, trans_mem, 1, k, n, k, n, n, NULL, 1, true);

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
            reordered_weights_memory, eng, engine_stream, 0/*is_weights_const*/,
            1/*inplace_reorder_wei*/);
    }
    else if (dt == zendnn_bf16) {
        memory::desc blocked_matmul_weights_md = memory::desc({k,n}, (
                    memory::data_type)dt, TAG_BF16);
        if (matmul_weights_md.get_size() != blocked_matmul_weights_md.get_size()) {
            return false;
        }
        reorderAndCacheWeightsBrgemm(
            key_obj, blocked_matmul_weights_md, user_weights_memory,
            reordered_weights_memory, eng, engine_stream, 0/*is_weights_const*/,
            1/*inplace_reorder_wei*/);
    }
    else if (dt == zendnn_s8) {
        memory::desc blocked_matmul_weights_md = memory::desc({k,n}, (
                    memory::data_type)dt, TAG_INT8);
        if (matmul_weights_md.get_size() != blocked_matmul_weights_md.get_size()) {
            return false;
        }
        reorderAndCacheWeightsBrgemm(
            key_obj, blocked_matmul_weights_md, user_weights_memory,
            reordered_weights_memory, eng, engine_stream, 0/*is_weights_const*/,
            1/*inplace_reorder_wei*/);
    }

    return true;
}
// Currently supporting inplace only
bool reorder_aocl_inplace(void *src, void *dst, uint k, uint n, bool trans_mem,
                          zendnn_data_type_t dt) {
    const char reorder_param0 = 'B';
    const dim_t reorder_param1 = k;
    const dim_t reorder_param2 = n;
    const char order = 'r';
    char trans = 'n';
    if (trans_mem) {
        trans = 't';
    }

    // TODO:Remove key dependency
    Key_matmul key_obj(false, trans_mem, 1, k, n, k, n, n, NULL, 1, true);
    if (dt == zendnn_f32) {
        int siz_req = aocl_get_reorder_buf_size_f32f32f32of32(order, trans,
                      reorder_param0, reorder_param1, reorder_param2);
        // TODO: get size of data type using some function
        if (siz_req != 4*k*n) {
            return false;
        }
        float *temp = NULL;
        reorderAndCacheWeights<float>(key_obj, (float *)src, temp, k, n,
                                      trans_mem ? k : n, 0, 1/*inplace*/, order, trans, reorder_param0,
                                      reorder_param1,
                                      reorder_param2,
                                      aocl_get_reorder_buf_size_f32f32f32of32, aocl_reorder_f32f32f32of32
                                     );
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
                                        trans_mem ? k : n, 0, 1/*inplace*/, order, trans, reorder_param0,
                                        reorder_param1,
                                        reorder_param2,
                                        aocl_get_reorder_buf_size_bf16bf16f32of32, aocl_reorder_bf16bf16f32of32
                                       );
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
                                       trans_mem ? k : n, 0, 1/*inplace*/, order, trans, reorder_param0,
                                       reorder_param1,
                                       reorder_param2,
                                       aocl_get_reorder_buf_size_u8s8s32os32, aocl_reorder_u8s8s32os32
                                      );
    }
    return true;
}

// Returns backend/ALGO for given data type
unsigned int fetch_backend(zendnn_data_type_t dt) {
    zendnnEnv zenEnvObj = readEnv();
    if (dt == zendnn_s8) {
        return zenEnvObj.zenINT8GEMMalgo;
    }
    else if (dt == zendnn_bf16) {
        return zenEnvObj.zenBF16GEMMalgo;
    }
    else if (dt == zendnn_f32) {
        return zenEnvObj.zenGEMMalgo;
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

    unsigned int backend = fetch_backend(dt);
    if (backend == 1/*aocl*/) {
        status = reorder_aocl_inplace(src, dst, k, n, trans, dt);
        zendnnVerbose(ZENDNN_PROFLOG,"AOCL reorder custom op,", " status ",
                      status ? "True" : "False");
    }
    else if (backend == 2/*brgemm*/) {
        status = reorder_brgemm_inplace(src, dst, k, n, trans, dt);
        zendnnVerbose(ZENDNN_PROFLOG,"BRGEMM reorder custom op,", " status ",
                      status ? "True" : "False");
    }
    else if (backend == 3 || backend == 4 || backend == 0/*non-blocked*/) {
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
}
