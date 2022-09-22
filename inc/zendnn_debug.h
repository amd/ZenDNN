/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2018-2022 Intel Corporation
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
// Use this script to update the file: scripts/generate_zendnn_debug.py

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
const char ZENDNN_API *zendnn_fpmath_mode2str(zendnn_fpmath_mode_t v);
const char ZENDNN_API *zendnn_scratchpad_mode2str(zendnn_scratchpad_mode_t v);
const char ZENDNN_API *zendnn_cpu_isa2str(zendnn_cpu_isa_t v);
const char ZENDNN_API *zendnn_cpu_isa_hints2str(zendnn_cpu_isa_hints_t v);

const char ZENDNN_API *zendnn_runtime2str(unsigned v);

#ifdef __cplusplus
}
#endif

#endif
