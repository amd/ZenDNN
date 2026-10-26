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

#include "lowoha_operators/matmul/matmul_native/common/native_aocl_unreorder.hpp"

#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

reorder::reorder_params_t aocl_prepack_params(
        int K, int N, int ldb, bool transB, bool int8_u8s8) {
    reorder::reorder_params_t params;
    params.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    params.prepack.wei_dtype = int8_u8s8 ? data_type_t::s8 : data_type_t::bf16;
    params.prepack.src_dtype = int8_u8s8 ? data_type_t::u8 : data_type_t::bf16;
    params.prepack.K = K;
    params.prepack.N = N;
    params.prepack.ldb = ldb;
    params.prepack.transposed = transB;
    return params;
}

} // namespace

bool aocl_prepack_unreorderable(const matmul_params &params) {
#if !ZENDNNL_DEPENDS_AOCLDLP
    (void)params;
    return false;
#else
    if (zendnnl::common::matmul_config_t::instance().get_dlp_m_hint() > 0)
        return false;
    const bool bf16 = params.dtypes.src == data_type_t::bf16
            && params.dtypes.wei == data_type_t::bf16 && !params.dynamic_quant;
    return bf16 || is_static_u8s8_quant_config(params);
#endif
}

status_t aocl_prepack_unreorder(const void *aocl_packed, int K, int N, int ldb,
        bool transB, bool int8_u8s8, int num_threads,
        std::unique_ptr<void, native_aligned_free_t> &raw) {
    raw.reset();
    reorder::reorder_params_t params
            = aocl_prepack_params(K, N, ldb, transB, int8_u8s8);
    params.is_unreorder = true;
    params.num_threads = num_threads;
    const size_t raw_bytes = reorder::weight_unreorder_size(params);
    if (raw_bytes == 0) return status_t::unimplemented;
    std::unique_ptr<void, native_aligned_free_t> buf(
            zendnnl_aligned_alloc(64, (raw_bytes + 63) & ~size_t {63}));
    if (!buf) return status_t::failure;
    const status_t st = reorder::reorder_direct(aocl_packed, buf.get(), params);
    if (st != status_t::success) return st;
    raw = std::move(buf);
    return status_t::success;
}

size_t aocl_prepack_bytes(int K, int N, int ldb, bool transB, bool int8_u8s8) {
    // Asked for on every call on a prepacked weight; the size depends on the
    // shape only.
    struct entry_t {
        int K = 0, N = 0, ldb = 0;
        bool transB = false, int8_u8s8 = false;
        size_t bytes = 0;
    };
    constexpr int kEntries = 16;
    thread_local entry_t entries[kEntries];
    thread_local int next = 0;
    for (const entry_t &e : entries)
        if (e.bytes != 0 && e.K == K && e.N == N && e.ldb == ldb
                && e.transB == transB && e.int8_u8s8 == int8_u8s8)
            return e.bytes;
    reorder::reorder_params_t params
            = aocl_prepack_params(K, N, ldb, transB, int8_u8s8);
    params.is_prepack = true;
    const size_t bytes = reorder::weight_prepack_size(params);
    if (bytes != 0) {
        entry_t &e = entries[next];
        e.K = K;
        e.N = N;
        e.ldb = ldb;
        e.transB = transB;
        e.int8_u8s8 = int8_u8s8;
        e.bytes = bytes;
        next = (next + 1) % kEntries;
    }
    return bytes;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
