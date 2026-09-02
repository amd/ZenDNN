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

// Stubs for the AOCL-DLP backend, compiled only when ZenDNNL is built without
// AOCL-DLP (ZENDNNL_DEPENDS_AOCLDLP=0). They satisfy the link dependencies of
// the always-compiled LOWOHA matmul dispatch while making the missing backend
// explicit at runtime.
//
// The two groups below fail differently, on purpose:
//
//   * Compute entry points (run_dlp, matmul_batch_gemm_wrapper) throw. Every
//     call site either sits behind #if ZENDNNL_DEPENDS_AOCLDLP or redirects
//     AOCL-DLP kernels to the reference kernel (kernel_select() and the
//     !ZENDNNL_DEPENDS_AOCLDLP branches of matmul_kernel_wrapper, bmm_execute
//     and execute_partitioned_matmul), so both should be unreachable. They
//     return void, so a silent return would leave the caller's output buffer
//     uninitialized -- a wrong result rather than an error. Throwing makes any
//     residual fall-through fail loudly instead.
//
//   * Cache, prepack and quantization helpers return benign "nothing cached /
//     not available" values. Some of their callers are compiled
//     unconditionally -- the W4A8 plain-s8 side table in
//     group_matmul_dispatch.cpp, broadcast_w4a8_src_scale() in the N-tile
//     group path, and the gtest weight-cache reset -- and must be able to
//     carry on down the reference path, so aborting is wrong.

#include "common/zendnnl_exceptions.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"

#include <atomic>

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::error_handling;

void run_dlp(char, char, char, int, int, int, float, float, int, int, int, char,
        char, const void *, const void *, void *, const matmul_data_types &,
        const matmul_params &, const void *, zendnnl::ops::matmul_algo_t,
        bool) {
    apilog_error(
            "AOCL-DLP matmul kernel (run_dlp) invoked but ZenDNNL was built "
            "without AOCL-DLP support (ZENDNNL_DEPENDS_AOCLDLP=0).");
    EXCEPTION_WITH_LOC(
            "AOCL-DLP matmul kernel (run_dlp) invoked but ZenDNNL was "
            "built without AOCL-DLP support "
            "(ZENDNNL_DEPENDS_AOCLDLP=0).");
}

void matmul_batch_gemm_wrapper(char, char, char, int, int, int, float,
        const void *, int, const void *, int, float, void *, int,
        matmul_data_types &, int, int, int, char, char, size_t, size_t, size_t,
        const matmul_params &, const void *, int) {
    apilog_error(
            "AOCL-DLP batch matmul kernel (matmul_batch_gemm_wrapper) "
            "invoked but ZenDNNL was built without AOCL-DLP support "
            "(ZENDNNL_DEPENDS_AOCLDLP=0).");
    EXCEPTION_WITH_LOC(
            "AOCL-DLP batch matmul kernel (matmul_batch_gemm_wrapper) "
            "invoked but ZenDNNL was built without AOCL-DLP support "
            "(ZENDNNL_DEPENDS_AOCLDLP=0).");
}

void clear_aocl_matmul_weight_caches() {
    // No AOCL weight caches exist in this build; nothing to clear.
}

template <typename T>
bool reorderAndCacheWeights(Key_matmul, const void *, void *&, const int,
        const int, const int, const char, const char, char,
        get_reorder_buff_size_func_ptr, reorder_func_ptr<T>, int) {
    apilog_error(
            "AOCL-DLP weight reorder requested but ZenDNNL was built "
            "without AOCL-DLP support (ZENDNNL_DEPENDS_AOCLDLP=0).");
    return false;
}

template bool reorderAndCacheWeights<int16_t>(Key_matmul, const void *, void *&,
        const int, const int, const int, const char, const char, char,
        get_reorder_buff_size_func_ptr, reorder_func_ptr<int16_t>, int);

void w4a8_cvt_and_cache_plain_s8(
        Key_matmul, const int8_t *, void *&s8_plain, int, int, int, bool) {
    s8_plain = nullptr;
}

void w4a8_populate_plain_s8_cache(const std::vector<const void *> &,
        const std::vector<int> &, const std::vector<int> &,
        const std::vector<int> &, const std::vector<bool> &,
        const std::vector<matmul_params> &, int num_ops,
        std::vector<void *> &w4a8_s8_out, bool &any_w4a8) {
    // No AOCL plain-s8 LRU in this build; leave the side table empty so
    // downstream W4A8 AOCL paths are skipped (reference W4A8 still works).
    static std::atomic<bool> s_w4a8_plain_stub_announced {false};
    if (!s_w4a8_plain_stub_announced.exchange(
                true, std::memory_order_relaxed)) {
        apilog_verbose(
                "[W4A8.PLAIN] w4a8_populate_plain_s8_cache skipped: ZenDNNL "
                "was built without AOCL-DLP (ZENDNNL_DEPENDS_AOCLDLP=0); "
                "W4A8 AOCL plain-s8 prep and ALGO-3 N-tile sym-quant path "
                "are unavailable (reference W4A8 unaffected).");
    }
    w4a8_s8_out.assign(num_ops, nullptr);
    any_w4a8 = false;
}

void w4a8ReorderAndCacheWeightsAocl(Key_matmul, const int8_t *,
        void *&reorder_weights, const int, const int, const int, const bool,
        const char, const char, data_type_t, data_type_t, int, int) {
    reorder_weights = nullptr;
}

status_t broadcast_w4a8_src_scale(
        matmul_params &, int, std::vector<uint8_t> &) {
    apilog_error(
            "W4A8 source-scale broadcast requested but ZenDNNL was built "
            "without AOCL-DLP support (ZENDNNL_DEPENDS_AOCLDLP=0).");
    return status_t::failure;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
