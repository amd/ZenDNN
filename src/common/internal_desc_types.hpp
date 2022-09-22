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

#ifndef COMMON_INTERNAL_DESC_TYPES_HPP
#define COMMON_INTERNAL_DESC_TYPES_HPP

#include <vector>
#include "zendnn_types.h"

namespace zendnn {
namespace impl {

// The types are not exposed
struct zendnn_reorder_desc_t {
    zendnn_primitive_kind_t primitive_kind;
    const zendnn_memory_desc_t *src_md;
    const zendnn_memory_desc_t *dst_md;
    zendnn_engine_kind_t src_engine_kind;
    zendnn_engine_kind_t dst_engine_kind;
    bool is_cross_engine;
};

struct zendnn_concat_desc_t {
    zendnn_primitive_kind_t primitive_kind;
    const zendnn_memory_desc_t *dst_md;
    zendnn_dim_t n;
    zendnn_dim_t concat_dimension;
    const zendnn_memory_desc_t *src_mds;
};

struct zendnn_sum_desc_t {
    zendnn_primitive_kind_t primitive_kind;
    const zendnn_memory_desc_t *dst_md;
    zendnn_dim_t n;
    const float *scales;
    const zendnn_memory_desc_t *src_mds;
};

struct zendnn_zero_pad_desc_t {
    zendnn_primitive_kind_t primitive_kind;
};

} // namespace impl
} // namespace zendnn

#endif // INTERNAL_DESC_TYPES
