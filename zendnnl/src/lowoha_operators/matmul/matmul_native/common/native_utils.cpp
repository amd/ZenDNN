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

#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"

#include <algorithm>

#include <omp.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

void native_for_parallel(
        int n, int threads, void (*range)(void *, int, int), void *fn) {
#pragma omp parallel num_threads(threads)
    {
        const int t = omp_get_thread_num(), nt = omp_get_num_threads();
        const int q = n / nt, r = n % nt;
        const int i0 = t * q + std::min(t, r);
        range(fn, i0, i0 + q + (t < r ? 1 : 0));
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
