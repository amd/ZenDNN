/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef GRP_MATMUL_BENCHDNN_HPP
#define GRP_MATMUL_BENCHDNN_HPP

#include <string>
#include "utils/benchdnn_utils.hpp"

namespace zendnnl {
namespace benchdnn {
namespace grp_matmul {

int bench(const std::string &in_filename, const std::string &out_filename,
          size_t cache_size);

} // namespace grp_matmul
} // namespace benchdnn
} // namespace zendnnl

#endif
