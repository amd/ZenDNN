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

/// The output store of the tiled HSTU attention kernels.

#ifndef LOWOHA_HSTU_TILED_STORE_HPP
#define LOWOHA_HSTU_TILED_STORE_HPP

#include <cstdint>
#include "common/bfloat16.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

/// dst[0, n) = o[0, n) in the output data type (BF16: AVX512-BF16).
void store_row(const float *o, int64_t n, zendnnl::common::bfloat16_t *dst);
void store_row(const float *o, int64_t n, float *dst);

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_TILED_STORE_HPP
