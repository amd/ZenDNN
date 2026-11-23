/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "lowoha_operators/matmul/matmul_native/common/native_format.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

const native_format_desc_t *native_format_of(data_type_t wei_dt) {
    for (const native_format_desc_t &f : kNativeFormats)
        if (f.wei_dt == wei_dt) return &f;
    return nullptr;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
