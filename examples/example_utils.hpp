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
#ifndef _EXAMPLE_UTILS_HPP_
#define _EXAMPLE_UTILS_HPP_

#include <cstdlib>
#if defined(_WIN32)
#include <malloc.h>
#endif

#include "tensor_helper/tensor_factory.hpp"
#include "zendnnl.hpp"

#define MATMUL_M 10
#define MATMUL_K 6
#define MATMUL_N 4

#define ROWS 32
#define COLS 64

namespace zendnnl {
/** @namespace zendnnl::examples
 *  @brief A namespace that contains examples of how to use ZenDNNL.
 */
namespace examples {
using namespace zendnnl::interface;
using tensor_helper::StorageParam;
using tensor_helper::tensor_factory_t;

/** @class tensor_functions
 * @brief Quick generation of predefined tensors.
 */
class tensor_functions_t {
public:
    void tensor_pretty_print(const tensor_t &tensor_);
};

/** @fn get_aligned_size
 *  @brief Function to align the given size_ according to the alignment
 */
size_t get_aligned_size(size_t alignment, size_t size_);

/** @fn example_aligned_alloc / example_aligned_free
 *  @brief Portable aligned allocation for the examples. Windows/MSVC has no C11
 *  aligned_alloc; it uses _aligned_malloc (note the swapped argument order) and
 *  requires the matching _aligned_free. Other platforms use aligned_alloc/free.
 */
inline void *example_aligned_alloc(size_t alignment, size_t size_) {
#if defined(_WIN32)
    return _aligned_malloc(size_, alignment);
#else
    return aligned_alloc(alignment, size_);
#endif
}

inline void example_aligned_free(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

} // namespace examples
} // namespace zendnnl

#endif
