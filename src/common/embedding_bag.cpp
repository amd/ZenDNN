/*******************************************************************************
* Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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
zendnn_embedding_bag_desc_init(embedding_bag_desc_t *desc,
                               prop_kind_t prop_kind,
                               alg_kind_t alg_kind,
                               uint32_t num_threads,
                               const memory_desc_t *input_desc,
                               const memory_desc_t *indices_desc,
                               const memory_desc_t *offsets_desc,
                               const memory_desc_t *weights_desc,
                               const memory_desc_t *dst_desc,
                               int32_t  padding_idx,
                               uint32_t scatter_stride,
                               uint32_t scatter_offset) {

    // run sanity check on parameters
    bool args_ok = !any_null(desc, input_desc, indices_desc,
                             offsets_desc, dst_desc)
                   && (prop_kind == forward_inference)
                   && one_of(alg_kind, embedding_bag_max, embedding_bag_sum,
                             embedding_bag_mean);
    if (!args_ok) {
        return invalid_arguments;
    }

    if (input_desc->ndims != dst_desc->ndims) {
        return invalid_arguments;
    }

    // currently accept only s32 type indices and offsets
    // TODO: add support for 64bit integers if zendnn supports them
    if (indices_desc->data_type != data_type::s32) {
        return invalid_arguments;
    }

    if (offsets_desc->data_type != data_type::s32) {
        return invalid_arguments;
    }

    // check the tensor sizes
    // output size should at least be bags*scatter_offset. please see documentation
    // of embedding_bag_desc_t for scatter_offset
    auto bags           = offsets_desc->dims[0];
    auto embedding_dim  = input_desc->dims[1];

    if ((dst_desc->dims[0] < bags*scatter_stride) || (dst_desc->dims[1] != embedding_dim)) {
        return invalid_arguments;
    }

    // number of weights should be same as number of indices
    if ((nullptr != weights_desc)
            && (weights_desc->dims[0] != indices_desc->dims[0])) {
        return invalid_arguments;
    }

    auto emd = embedding_bag_desc_t();
    emd.primitive_kind   = primitive_kind::embedding_bag;
    emd.alg_kind         = alg_kind;
    emd.prop_kind        = prop_kind;
    emd.input_desc       = *input_desc;
    emd.indices_desc     = *indices_desc;
    emd.offsets_desc     = *offsets_desc;
    emd.dst_desc         = *dst_desc;
    emd.padding_idx      = padding_idx;
    emd.scatter_stride   = scatter_stride;
    emd.scatter_offset   = scatter_offset;

    // weights tensor may or may not be present.
    emd.is_weights = false;
    if (nullptr != weights_desc) {
        emd.is_weights = true;
        emd.weights_desc = *weights_desc;
    } else {
        emd.weights_desc = memory_desc_t();
    }

    // get parallel threads
    zendnn::zendnnEnv zenEnvObj = readEnv();
    if (num_threads) {
        emd.num_threads = num_threads < zenEnvObj.omp_num_threads ?
                            num_threads : zenEnvObj.omp_num_threads;
    }
    else {
        emd.num_threads = zenEnvObj.omp_num_threads;
    }

    *desc = emd;
    return success;
}
