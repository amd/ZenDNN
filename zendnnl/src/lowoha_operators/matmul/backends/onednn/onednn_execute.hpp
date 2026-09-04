/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _LOWOHA_MATMUL_ONEDNN_EXECUTE_HPP_
#define _LOWOHA_MATMUL_ONEDNN_EXECUTE_HPP_

#include <unordered_map>

#include "lowoha_operators/matmul/backends/onednn/onednn_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

#if ZENDNNL_DEPENDS_ONEDNN
void onednn_matmul_execute(const onednn_utils_t::onednn_matmul_params &params,
        std::unordered_map<int, dnnl::memory> &matmul_args,
        dnnl::primitive_attr &matmul_attr, dnnl::engine &eng);
#endif

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif
