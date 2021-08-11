/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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

// DO NOT EDIT, AUTO-GENERATED

// clang-format off

#ifndef ZENDNN_DEBUG_H
#define ZENDNN_DEBUG_H

/// @file
/// Debug capabilities

#include "zendnn_config.h"
#include "zendnn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

const char ZENDNN_API *zendnn_status2str(zendnn_status_t v);
const char ZENDNN_API *zendnn_dt2str(zendnn_data_type_t v);
const char ZENDNN_API *zendnn_fmt_kind2str(zendnn_format_kind_t v);
const char ZENDNN_API *zendnn_fmt_tag2str(zendnn_format_tag_t v);
const char ZENDNN_API *zendnn_prop_kind2str(zendnn_prop_kind_t v);
const char ZENDNN_API *zendnn_prim_kind2str(zendnn_primitive_kind_t v);
const char ZENDNN_API *zendnn_alg_kind2str(zendnn_alg_kind_t v);
const char ZENDNN_API *zendnn_rnn_flags2str(zendnn_rnn_flags_t v);
const char ZENDNN_API *zendnn_rnn_direction2str(zendnn_rnn_direction_t v);
const char ZENDNN_API *zendnn_engine_kind2str(zendnn_engine_kind_t v);
const char ZENDNN_API *zendnn_scratchpad_mode2str(zendnn_scratchpad_mode_t v);
const char ZENDNN_API *zendnn_cpu_isa2str(zendnn_cpu_isa_t v);
const char ZENDNN_API *zendnn_cpu_isa_hints2str(zendnn_cpu_isa_hints_t v);

const char ZENDNN_API *zendnn_runtime2str(unsigned v);

/// Forms a format string for a given memory descriptor.
///
/// The format is defined as: 'dt:[p|o|0]:fmt_kind:fmt:extra'.
/// Here:
///  - dt       -- data type
///  - p        -- indicates there is non-trivial padding
///  - o        -- indicates there is non-trivial padding offset
///  - 0        -- indicates there is non-trivial offset0
///  - fmt_kind -- format kind (blocked, wino, etc...)
///  - fmt      -- extended format string (format_kind specific)
///  - extra    -- shows extra fields (underspecified)
int ZENDNN_API zendnn_md2fmt_str(char *fmt_str, size_t fmt_str_len,
        const zendnn_memory_desc_t *md);

/// Forms a dimension string for a given memory descriptor.
///
/// The format is defined as: 'dim0xdim1x...xdimN
int ZENDNN_API zendnn_md2dim_str(char *dim_str, size_t dim_str_len,
        const zendnn_memory_desc_t *md);

#ifdef __cplusplus
}
#endif

#endif
