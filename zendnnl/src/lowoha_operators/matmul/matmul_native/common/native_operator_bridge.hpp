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

#ifndef MATMUL_NATIVE_COMMON_NATIVE_OPERATOR_BRIDGE_HPP
#define MATMUL_NATIVE_COMMON_NATIVE_OPERATOR_BRIDGE_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/error_status.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"

/// What the native BRGEMM uses of the rest of the matmul operator: the AOCL
/// reorder and prepack API, the reorder quantization of a source, the custom
/// kernel's record of the buffers it packed in place, and the log text of
/// lowoha_matmul_utils.hpp.  The .cpp is the only native file that includes
/// their headers.

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::status_t;

// ── Weights held as an AOCL prepack ─────────────────────────────────────

/// The weight is an AOCL prepack: mem_format_b 'r' with no lowoha_algo or
/// aocl_dlp_blocked.
bool aocl_prepacked_weight(const matmul_params &params);

/// The aocl_dlp_blocked prepack of the call's weight has an inverse: a bf16
/// weight, or a static u8 x s8 one, prepacked without a DLP M hint.
bool aocl_prepack_unreorderable(const matmul_params &params);

/// Unreorders an aocl_dlp_blocked prepack of a K x N weight into raw, which
/// then holds the plain weight with rows ldb elements apart.
status_t aocl_prepack_unreorder(const void *aocl_packed, int K, int N, int ldb,
        bool transB, bool int8_u8s8, int num_threads,
        std::unique_ptr<void, native_aligned_free_t> &raw);

/// Bytes of the aocl_dlp_blocked prepack of a K x N weight.
size_t aocl_prepack_bytes(int K, int N, int ldb, bool transB, bool int8_u8s8);

/// Bytes of the aocl_dlp_blocked prepack of a K x N weight of wei for a src
/// source and weight scale groups of group values along K (0: one group),
/// with the column sums the prepack of a u8 source carries after its image;
/// 0 when no prepack of that format is made.
size_t aocl_prepack_bytes(data_type_t src, data_type_t wei, int group, int K,
        int N, int ldb, bool transB);

// ── Buffers the custom kernel packed in place ───────────────────────────

/// A custom-kernel pack rewrote the buffer at weight in place: its reader is
/// that route.
bool custom_kernel_packed_in_place(const void *weight);

// ── Source quantization ─────────────────────────────────────────────────

/// The quantized rows, scales and zero points the reorder quantization
/// allocated for a call; freed with it.
struct native_quant_buffers_t {
    void *rows = nullptr;
    void *scales = nullptr;
    void *zero_points = nullptr;

    native_quant_buffers_t() = default;
    ~native_quant_buffers_t();
    native_quant_buffers_t(const native_quant_buffers_t &) = delete;
    native_quant_buffers_t &operator=(const native_quant_buffers_t &) = delete;
    native_quant_buffers_t(native_quant_buffers_t &&o) noexcept;
    native_quant_buffers_t &operator=(native_quant_buffers_t &&o) noexcept;
};

/// Quantizes the M x K source rows (rows_ld elements apart; with transA the
/// source stored transposed, K rows of M elements rows_ld apart) of a call
/// with params by matmul's reorder quantization on num_threads threads (it
/// sets the nthreads ICV for the quantizer's call), whose buffers then belong
/// to buffers.  quantized and quantized_lda are the rows it yields, in the
/// source's layout (the source rows when it does not quantize them), and q
/// the parameters that describe them: dtypes.src their type,
/// quant_params.src_scale and src_zp their scales and zero points, [M][G]
/// either way.  q is the calling thread's, valid until its next call.
/// Returns the quantization's status.
status_t reorder_quantize_source(const matmul_params &params, const void *rows,
        int rows_ld, int M, int K, int num_threads,
        native_quant_buffers_t &buffers, const void *&quantized,
        int &quantized_lda, const matmul_params *&q, bool transA = false);

// ── Log text ────────────────────────────────────────────────────────────

/// The post-ops of a call as matmul_direct logs them: their names, and the
/// data types of their tensors.
std::string post_op_names_text(const matmul_params &params);
std::string post_op_dtypes_text(const matmul_params &params);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_OPERATOR_BRIDGE_HPP
