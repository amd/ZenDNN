/*******************************************************************************
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

// Error-out stubs for the AOCL-DLP backend, compiled only when ZenDNNL is
// built without AOCL-DLP (ZENDNNL_DEPENDS_AOCLDLP=0). They satisfy the link
// dependencies of the always-compiled LOWOHA matmul dispatch (run_dlp,
// matmul_batch_gemm_wrapper, weight-cache helpers and reorder caching) while
// making it explicit at runtime that the AOCL-DLP path is unavailable.
//
// The selecting layers (e.g. lowoha::matmul::matmul_direct) already reject
// AOCL-DLP kernels up front and return status_t::unimplemented. The compute
// entry points below (run_dlp / matmul_batch_gemm_wrapper) are the final
// backstop for any residual fall-through path (e.g. a deeper native/onednn
// decline that mutates the kernel to aocl_dlp). They are void, so returning
// normally would leave the caller's output buffer uninitialized (a silent
// wrong result); instead they throw so an unsupported call fails loudly.

#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "common/zendnnl_exceptions.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::error_handling;

void run_dlp(char, char, char, int, int, int,
             float, float, int, int, int,
             char, char, const void *,
             const void *, void *, const matmul_data_types &,
             const matmul_params &, const void *,
             zendnnl::ops::matmul_algo_t, bool) {
  apilog_error("AOCL-DLP matmul kernel (run_dlp) invoked but ZenDNNL was built "
               "without AOCL-DLP support (ZENDNNL_DEPENDS_AOCLDLP=0).");
  EXCEPTION_WITH_LOC("AOCL-DLP matmul kernel (run_dlp) invoked but ZenDNNL was "
                     "built without AOCL-DLP support "
                     "(ZENDNNL_DEPENDS_AOCLDLP=0).");
}

void matmul_batch_gemm_wrapper(char, char, char, int,
                               int, int, float, const void *, int, const void *, int,
                               float, void *, int, matmul_data_types &, int,
                               int, int, char,
                               char, size_t, size_t,
                               size_t, const matmul_params &, const void *,
                               int) {
  apilog_error("AOCL-DLP batch matmul kernel (matmul_batch_gemm_wrapper) "
               "invoked but ZenDNNL was built without AOCL-DLP support "
               "(ZENDNNL_DEPENDS_AOCLDLP=0).");
  EXCEPTION_WITH_LOC("AOCL-DLP batch matmul kernel (matmul_batch_gemm_wrapper) "
                     "invoked but ZenDNNL was built without AOCL-DLP support "
                     "(ZENDNNL_DEPENDS_AOCLDLP=0).");
}

void clear_aocl_matmul_weight_caches() {
  // No AOCL weight caches exist in this build; nothing to clear.
}

template <typename T>
bool reorderAndCacheWeights(Key_matmul, const void *,
                            void *&, const int, const int, const int,
                            const char, const char, char,
                            get_reorder_buff_size_func_ptr,
                            reorder_func_ptr<T>, int) {
  apilog_error("AOCL-DLP weight reorder requested but ZenDNNL was built "
               "without AOCL-DLP support (ZENDNNL_DEPENDS_AOCLDLP=0).");
  return false;
}

template bool reorderAndCacheWeights<int16_t>(Key_matmul, const void *,
    void *&, const int, const int, const int,
    const char, const char, char,
    get_reorder_buff_size_func_ptr,
    reorder_func_ptr<int16_t>, int);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
