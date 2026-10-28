/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

#ifndef LOWOHA_HSTU_ATTENTION_HPP
#define LOWOHA_HSTU_ATTENTION_HPP

#include "common/error_status.hpp"
#include "common/zendnnl_api.hpp"
#include "lowoha_operators/hstu_attention/lowoha_hstu_attention_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

using zendnnl::error_handling::status_t;

/**
 * @brief HSTU attention over jagged sequences (see hstu_attention_params).
 *
 * @param query   Q rows (full or delta mode), params.qkv_dt
 * @param key     K, params.qkv_dt
 * @param value   V, params.qkv_dt
 * @param output  out rows, params.out_dt
 * @param params  shapes, mask, scaling and strides
 *
 * @return status_t::success, status_t::failure on invalid arguments, or
 *         status_t::unimplemented when the requested kernel does not support
 *         the call.
 */
ZENDNNL_API status_t hstu_attention_direct(const void *query, const void *key,
        const void *value, void *output, const hstu_attention_params &params);

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_ATTENTION_HPP
