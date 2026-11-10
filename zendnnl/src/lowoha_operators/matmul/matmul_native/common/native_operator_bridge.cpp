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

#include "lowoha_operators/matmul/matmul_native/common/native_operator_bridge.hpp"

#include <utility>

#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::size_of;

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

bool aocl_prepacked_weight(const matmul_params &params) {
    return is_aocl_prepacked_weight(params);
}

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

size_t aocl_prepack_bytes(data_type_t src, data_type_t wei, int group, int K,
        int N, int ldb, bool transB) {
#if !ZENDNNL_DEPENDS_AOCLDLP
    (void)src;
    (void)wei;
    (void)group;
    (void)K;
    (void)N;
    (void)ldb;
    (void)transB;
    return 0;
#else
    reorder::reorder_params_t params
            = aocl_prepack_params(K, N, ldb, transB, false);
    params.prepack.wei_dtype = wei;
    params.prepack.src_dtype = src;
    params.prepack.sym_group_size = group;
    params.is_prepack = true;
    return reorder::weight_prepack_size(params);
#endif
}

bool custom_kernel_packed_in_place(const void *weight) {
    return custom_kernel::weight_packed_in_place(weight);
}

namespace {

/// Hands the buffers back to the quantizer's own type, whose destructor frees
/// them as the quantizer allocated them.
void release_quant_buffers(void *rows, void *scales, void *zero_points) {
    reorder_quant_buffers_t owner;
    owner.src_buf = static_cast<uint8_t *>(rows);
    owner.scale_buf = static_cast<uint8_t *>(scales);
    owner.zp_buf = static_cast<uint8_t *>(zero_points);
}

} // namespace

native_quant_buffers_t::~native_quant_buffers_t() {
    release_quant_buffers(rows, scales, zero_points);
}

native_quant_buffers_t::native_quant_buffers_t(
        native_quant_buffers_t &&o) noexcept
    : rows(std::exchange(o.rows, nullptr))
    , scales(std::exchange(o.scales, nullptr))
    , zero_points(std::exchange(o.zero_points, nullptr)) {}

native_quant_buffers_t &native_quant_buffers_t::operator=(
        native_quant_buffers_t &&o) noexcept {
    if (this != &o) {
        release_quant_buffers(rows, scales, zero_points);
        rows = std::exchange(o.rows, nullptr);
        scales = std::exchange(o.scales, nullptr);
        zero_points = std::exchange(o.zero_points, nullptr);
    }
    return *this;
}

status_t reorder_quantize_source(const matmul_params &params, const void *rows,
        int rows_ld, int M, int K, int num_threads,
        native_quant_buffers_t &buffers, const void *&quantized,
        int &quantized_lda, const matmul_params *&q, bool transA) {
    // The quantization kernels size their thread teams from the nthreads ICV.
    const thread_guard team(num_threads);
    // What the quantizer reads, and rewrites for the quantized source.  One
    // per thread: a call assigns into the storage of the last one instead of
    // allocating copies of the dims.  Every field the quantizer reads is
    // assigned whole: between calls the object keeps the last call's
    // src_scale.buff / src_zp.buff, freed with that call's buffers, and the
    // quantizer writes into a non-null buff as the caller's.
    thread_local matmul_params quantizer;
    quantizer.dtypes = params.dtypes;
    quantizer.quant_params = params.quant_params;
    quantizer.dynamic_quant = params.dynamic_quant;
    quantizer.packing = params.packing;
    quantized = rows;
    quantized_lda = rows_ld;
    size_t elem_size = size_of(params.dtypes.src);
    matmul_batch_params_t batch;
    reorder_quant_buffers_t held;
    const status_t st = reorder_quantization_wrapper(quantized, rows_ld,
            quantized_lda, elem_size, quantizer, batch, transA, M, K,
            num_threads, held);
    // They live as long as the call.
    native_quant_buffers_t made;
    made.rows = std::exchange(held.src_buf, nullptr);
    made.scales = std::exchange(held.scale_buf, nullptr);
    made.zero_points = std::exchange(held.zp_buf, nullptr);
    held.src_cap = held.scale_cap = held.zp_cap = 0;
    buffers = std::move(made);
    q = &quantizer;
    return st;
}

std::string post_op_names_text(const matmul_params &params) {
    return post_op_names_to_string(params);
}

std::string post_op_dtypes_text(const matmul_params &params) {
    return post_op_data_types_to_string(params);
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
