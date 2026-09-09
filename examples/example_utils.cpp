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

#include "example_utils.hpp"

#include <iostream>

namespace zendnnl {
namespace examples {

using namespace zendnnl::interface;

void tensor_functions_t::tensor_pretty_print(const tensor_t &tensor_) {
    //works only for 3D as of now
    auto tensor_size = tensor_.get_size();

    auto depths = tensor_size[0];
    auto rows = tensor_size[1];
    auto cols = tensor_size[2];

    for (uint64_t d = 0; d < depths; ++d) {
        std::cout << "depth = " << d << std::endl;
        for (uint64_t r = 0; r < rows; ++r) {
            std::cout << "r" << r << " : ";
            for (uint64_t c = 0; c < cols; ++c) {
                std::cout << tensor_.at({d, r, c}) << ", ";
            }
            std::cout << std::endl;
        }
    }
}

size_t get_aligned_size(size_t alignment, size_t size_) {
    return ((size_ + alignment - 1) & ~(alignment - 1));
}

} // namespace examples
} // namespace zendnnl
