/*******************************************************************************
* Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>
#include "zendnn.h"

#include "zendnn_helper.hpp"
#include "c_types_map.hpp"
#include "utils.hpp"

using namespace zendnn::impl;
using namespace zendnn::impl::utils;
using namespace zendnn::impl::status;
using namespace zendnn::impl::prop_kind;
using namespace zendnn::impl::alg_kind;

/* add new primitive */
zendnn_status_t
zendnn_attention_desc_init(attention_desc_t *desc,
                           prop_kind_t prop_kind,
                           alg_kind_t alg_kind,
                           const zendnn_memory_desc_t *query_desc,
                           const zendnn_memory_desc_t *key_desc,
                           const zendnn_memory_desc_t *value_desc,
                           const zendnn_memory_desc_t *weights_query_desc,
                           const zendnn_memory_desc_t *weights_key_desc,
                           const zendnn_memory_desc_t *weights_value_desc,
                           const zendnn_memory_desc_t *bias_query_desc,
                           const zendnn_memory_desc_t *bias_key_desc,
                           const zendnn_memory_desc_t *bias_value_desc,
                           const zendnn_memory_desc_t *mask_desc,
                           const zendnn_memory_desc_t *dst_desc,
                           float scale,
                           uint32_t num_heads,
                           uint32_t num_threads){

    // run sanity check on parameters
    bool args_ok = !any_null(desc, query_desc, key_desc,
                             value_desc, weights_query_desc,
                             weights_key_desc, weights_value_desc,
                             bias_query_desc, bias_key_desc,
                             bias_value_desc, mask_desc, dst_desc)
                   && (prop_kind == forward_inference)
                   && one_of(alg_kind, multihead_attention);
    if (!args_ok) {
        return invalid_arguments;
    }

    auto qHidden_size = query_desc->dims[2];
    auto kHidden_size = key_desc->dims[2];
    auto vHidden_size = value_desc->dims[2];

    if (qHidden_size % num_heads != 0 ||
        kHidden_size % num_heads != 0 ||
        vHidden_size % num_heads != 0) {
            return invalid_arguments;
    }

    auto attn = attention_desc_t();
    attn.primitive_kind     = primitive_kind::attention;
    attn.alg_kind           = alg_kind;
    attn.prop_kind          = prop_kind;
    attn.query_desc         = *query_desc;
    attn.key_desc           = *key_desc;
    attn.value_desc         = *value_desc;
    attn.weights_query_desc = *weights_query_desc;
    attn.weights_key_desc   = *weights_key_desc;
    attn.weights_value_desc = *weights_value_desc;
    attn.bias_query_desc    = *bias_query_desc;
    attn.bias_key_desc      = *bias_key_desc;
    attn.bias_value_desc    = *bias_value_desc;
    attn.mask_desc          = *mask_desc;
    attn.dst_desc           = *dst_desc;
    attn.scale              =  scale;
    attn.num_heads          =  num_heads;
    attn.num_threads        =  num_threads;

    *desc = attn;
    return success;
}
