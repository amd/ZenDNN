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

#ifndef _LOWOHA_EMBAG_REF_KERNEL_HPP_
#define _LOWOHA_EMBAG_REF_KERNEL_HPP_

#include "common/op_config.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_embag_common.hpp"
#include "operators/embag/native_kernels/embag_avx512_int8_int4_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace embag {

status_t embedding_bag_ref_direct(const void *table, const void *indices,
        const void *offsets, const void *weights, void *dst,
        embag_params_t params);

status_t embedding_ref_direct(const void *table, const void *indices,
        const void *weights, void *dst, embag_params_t params);

} // namespace embag
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_EMBAG_REF_KERNEL_HPP_
