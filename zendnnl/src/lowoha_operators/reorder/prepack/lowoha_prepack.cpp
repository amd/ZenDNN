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

#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp"
#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"

#include <cstring>
#include <limits>
#include <sstream>
#include <string>

namespace zendnnl {
namespace lowoha {
namespace reorder {

using namespace zendnnl::error_handling;
using namespace zendnnl::profile;
using zendnnl::common::is_profile_enabled;
using zendnnl::common::op_instrumentation;
using zendnnl::lowoha::matmul::kernel_to_string;
using zendnnl::profile::profiler_t;

namespace {

namespace ck = zendnnl::lowoha::matmul::custom_kernel;
using zendnnl::common::dtype_info;

constexpr size_t kPrepackAlign
        = zendnnl::lowoha::matmul::kStaticQuantColsumAlign;

inline size_t round_up_align(size_t bytes, size_t align) {
    return (bytes + align - 1) & ~(align - 1);
}

// Resolve the custom-kernel pack width: honour an explicit
// `params.pack_nr` (must be 32 or 64 and divide N) else auto-select
// via the same `plan_pack_nr(K, N)` the dispatcher uses. Returns 0
// when no supported NR divides N (caller treats as unsupported).
int ck_resolve_pack_nr(const prepack_params_t &params) {
    const int K = static_cast<int>(params.K);
    const int N = static_cast<int>(params.N);
    if (params.pack_nr != 0) {
        const bool ok
                = (params.pack_nr == ck::kNRMin || params.pack_nr == ck::kNRMax)
                && (N % params.pack_nr) == 0;
        return ok ? params.pack_nr : 0;
    }
    return ck::plan_pack_nr(K, N);
}

inline bool wants_colsum(const prepack_params_t &params) {
    return params.wei_dtype == data_type_t::s8 && params.sym_group_size <= 0
            && params.src_dtype == data_type_t::u8;
}

inline size_t colsum_buffer_bytes(const prepack_params_t &params) {
    if (!wants_colsum(params)) return 0;
    return round_up_align(
            static_cast<size_t>(params.N) * sizeof(int32_t), kPrepackAlign);
}

#if ZENDNNL_DEPENDS_AOCLDLP
void write_weight_colsum(
        const void *weights, const prepack_params_t &params, void *dst) {
    using namespace zendnnl::lowoha::matmul;
    const int k = static_cast<int>(params.K);
    const int n = static_cast<int>(params.N);
    const int64_t ldb = params.ldb;
    const char order = 'r';
    const char trans = params.transposed ? 't' : 'n';
    const size_t offset
            = static_quant_colsum_offset(order, trans, params.K, params.N);
    int32_t *colsum
            = reinterpret_cast<int32_t *>(static_cast<char *>(dst) + offset);
    const int8_t *wei = static_cast<const int8_t *>(weights);
    const int64_t wei_s0 = params.transposed ? 1 : ldb;
    const int64_t wei_s1 = params.transposed ? ldb : 1;

#pragma omp parallel for
    for (int col = 0; col < n; ++col) {
        int32_t acc = 0;
        for (int row = 0; row < k; ++row) {
            const int64_t wei_idx = wei_s0 * row + wei_s1 * col;
            acc += wei[wei_idx];
        }
        colsum[col] = acc;
    }
}
#endif
#if !ZENDNNL_DEPENDS_AOCLDLP
inline size_t dense_weight_bytes(const prepack_params_t &params) {
    return static_cast<size_t>(params.K) * static_cast<size_t>(params.N)
            * zendnnl::common::size_of(params.wei_dtype);
}

// Identity prepack when AOCL-DLP is unavailable: pack the logical KxN
// weight matrix into a dense row-major buffer, honoring source ldb.
void reference_prepack_weights(
        const void *weights, const prepack_params_t &params, void *dst) {
    const auto *src = static_cast<const uint8_t *>(weights);
    auto *out = static_cast<uint8_t *>(dst);
    const size_t elem_size = zendnnl::common::size_of(params.wei_dtype);
    const int64_t K = params.K;
    const int64_t N = params.N;
    const int64_t ldb = params.ldb;

    // Fast path only when the source is already dense row-major (order="ab").
    // transposed=true with ldb==K is dense column-major and still needs
    // element-wise reorder into row-major KxN.
    if (!params.transposed && ldb == N) {
        std::memcpy(out, src, dense_weight_bytes(params));
        return;
    }

    if (!params.transposed) {
        for (int64_t k = 0; k < K; ++k) {
            const size_t row_src = static_cast<size_t>(k)
                    * static_cast<size_t>(ldb) * elem_size;
            const size_t row_dst = static_cast<size_t>(k)
                    * static_cast<size_t>(N) * elem_size;
            std::memcpy(out + row_dst, src + row_src,
                    static_cast<size_t>(N) * elem_size);
        }
        return;
    }

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            const size_t src_off
                    = (static_cast<size_t>(n) * static_cast<size_t>(ldb)
                              + static_cast<size_t>(k))
                    * elem_size;
            const size_t dst_off
                    = (static_cast<size_t>(k) * static_cast<size_t>(N)
                              + static_cast<size_t>(n))
                    * elem_size;
            std::memcpy(out + dst_off, src + src_off, elem_size);
        }
    }
}
#endif

// Params-only validation (no weights pointer needed). Used by both the
// size-query path and the data-movement paths. `caller` is the name of
// the public function being validated, so log lines name the actual
// entry point a user invoked (easier to triage from logs).
status_t validate_prepack_params(
        const char *caller, const prepack_params_t &params) {
    if (params.K <= 0 || params.N <= 0) {
        apilog_error(caller, ": invalid K or N (K=", params.K, ", N=", params.N,
                ")");
        return status_t::failure;
    }
    if (params.ldb <= 0) {
        apilog_error(caller, ": invalid ldb (", params.ldb, ")");
        return status_t::failure;
    }

    const int64_t required_ldb = params.transposed ? params.K : params.N;
    if (params.ldb < required_ldb) {
        apilog_error(caller, ": invalid ldb (", params.ldb,
                "), expected at least ", required_ldb, " for ",
                (params.transposed ? "transposed" : "non-transposed"),
                " weights");
        return status_t::failure;
    }
    if (wants_colsum(params)
            && (params.K > std::numeric_limits<int>::max()
                    || params.N > std::numeric_limits<int>::max()
                    || params.ldb > std::numeric_limits<int>::max())) {
        apilog_error(caller,
                ": static-quant colsum requires K, N, and ldb <= INT_MAX "
                "(K=",
                params.K, ", N=", params.N, ", ldb=", params.ldb, ")");
        return status_t::failure;
    }

    // ── Custom-kernel (group_matmul) path ──────────────────────────
    // Selected by `algo == moe_custom_kernel`; the pack family is chosen
    // from `wei_dtype`. Validate the constraints the CK pack imposes,
    // mirroring custom_kernel/dispatch.cpp::prepare_for_call.
    if (params.algo == matmul_algo_t::moe_custom_kernel) {
        if (params.wei_dtype != data_type_t::bf16
                && params.wei_dtype != data_type_t::s8
                && params.wei_dtype != data_type_t::f16) {
            apilog_error(caller,
                    ": custom_kernel prepack supports wei_dtype bf16, f16 "
                    "or s8 (got ",
                    dtype_info(params.wei_dtype), ")");
            return status_t::failure;
        }
        if (ck_resolve_pack_nr(params) == 0) {
            apilog_error(caller,
                    ": custom_kernel prepack requires pack_nr in {32, 64} "
                    "dividing N (N=",
                    params.N, ", requested pack_nr=", params.pack_nr, ")");
            return status_t::failure;
        }
        // DQ-INT8 VNNI broadcasts 4 src bytes per K-quad — K must be a
        // multiple of 4 (the dispatcher refuses otherwise).
        if (params.wei_dtype == data_type_t::s8 && (params.K % 4) != 0) {
            apilog_error(caller,
                    ": custom_kernel int8 prepack requires K divisible by "
                    "4 (got K=",
                    params.K, ")");
            return status_t::failure;
        }
        if (params.interleave_split_halves && (params.N & 1)) {
            apilog_error(caller,
                    ": custom_kernel interleave_split_halves requires even "
                    "N (got N=",
                    params.N, ")");
            return status_t::failure;
        }
        return status_t::success;
    }

    if (params.algo == matmul_algo_t::aocl_dlp_blocked
            && params.src_dtype == data_type_t::u8
            && params.sym_group_size > 0) {
        apilog_error(caller,
                ": src_dtype u8 with sym_group_size > 0 is unsupported");
        return status_t::failure;
    }

    // Prepack only supports the AOCL DLP blocked layout. (libxsmm_blocked
    // and onednn_blocked were intentionally dropped -- see lowoha_prepack.hpp
    // for the rationale.)
    if (params.algo != matmul_algo_t::aocl_dlp_blocked) {
        apilog_error(caller, ": prepack.algo must be aocl_dlp_blocked (got ",
                kernel_to_string(params.algo), ")");
        return status_t::failure;
    }
    return status_t::success;
}

// Full input validation for path the weight buffer
// (weight_prepack_into).
status_t validate_prepack_inputs(const char *caller, const void *weights,
        const prepack_params_t &params) {
    if (!weights) {
        apilog_error(caller, ": weights pointer is null");
        return status_t::failure;
    }
    return validate_prepack_params(caller, params);
}

// True when prepack attaches gemm_hints: bf16, or s8 with an s8/u8 source.
bool aocl_pack_uses_m_hint(const prepack_params_t &params) {
    if (params.wei_dtype == data_type_t::bf16) { return true; }
    return params.wei_dtype == data_type_t::s8
            && (params.src_dtype == data_type_t::s8
                    || params.src_dtype == data_type_t::u8);
}

// Not diagnostic-gated. failure: bad K/N/ldb.
// unimplemented: unsupported algo, dtype, or sym-quant.
status_t validate_unreorder_params(
        const char *caller, const prepack_params_t &params) {
    if (params.K <= 0 || params.N <= 0) {
        apilog_error(caller, ": invalid K or N (K=", params.K, ", N=", params.N,
                ")");
        return status_t::failure;
    }
    if (params.ldb <= 0) {
        apilog_error(caller, ": invalid ldb (", params.ldb, ")");
        return status_t::failure;
    }
    const int64_t required_ldb = params.transposed ? params.K : params.N;
    if (params.ldb < required_ldb) {
        apilog_error(caller, ": invalid ldb (", params.ldb,
                "), expected at least ", required_ldb, " for ",
                (params.transposed ? "transposed" : "non-transposed"),
                " weights");
        return status_t::failure;
    }

    if (params.algo != matmul_algo_t::aocl_dlp_blocked) {
        apilog_error(caller, ": unreorder.algo must be aocl_dlp_blocked (got ",
                kernel_to_string(params.algo),
                "); custom-kernel / onednn / libxsmm layouts have no "
                "unreorder API");
        return status_t::unimplemented;
    }

    if (params.sym_group_size > 0) {
        apilog_error(caller,
                ": sym_group_size > 0 (s8 sym-quant) has no AOCL unreorder "
                "API (reorder and GEMM sym-quant symbols exist; unreorder "
                "does not)");
        return status_t::unimplemented;
    }

    switch (params.wei_dtype) {
        case data_type_t::f32:
        case data_type_t::bf16:
        case data_type_t::f16:
        case data_type_t::s8: break;
        default:
            apilog_error(caller,
                    ": unsupported wei_dtype=", dtype_info(params.wei_dtype),
                    " for unreorder (supported: f32 / bf16 / f16 / s8)");
            return status_t::unimplemented;
    }
    return status_t::success;
}

// ===========================================================================
// Split into two single-purpose functions sharing the same per-dtype
// dispatch table:
//   aocl_compute_size : returns prepacked size in bytes (0 on failure).
//   aocl_prepack      : writes the prepacked layout into the caller's
//                       buffer.
// ===========================================================================
size_t aocl_compute_size(
        const prepack_params_t &params, int32_t num_threads = 0) {
    using namespace zendnnl::lowoha::matmul;

#if !ZENDNNL_DEPENDS_AOCLDLP
    (void)params;
    apilog_error(
            "weight_prepack(aocl_dlp): ZenDNNL was built without AOCL-DLP "
            "support (ZENDNNL_DEPENDS_AOCLDLP=0); prepack is unavailable.");
    return 0;
#else
    const char order = 'r';
    const char trans = params.transposed ? 't' : 'n';
    const md_t k = static_cast<md_t>(params.K);
    const md_t n = static_cast<md_t>(params.N);

    if (params.wei_dtype == data_type_t::f32) {
        const size_t req = aocl_get_reorder_buf_size_f32f32f32of32(
                order, trans, 'B', k, n, nullptr);
        return round_up_align(req, kPrepackAlign);
    }

    if (params.wei_dtype == data_type_t::bf16) {
        dlp_gemm_hints_t bf16_hints {};
        dlp_metadata_t bf16_md {};
        dlp_metadata_t *reorder_meta = matmul::aocl_dlp_hints_metadata(
                nullptr, bf16_hints, bf16_md, num_threads);
        const size_t req = aocl_get_reorder_buf_size_bf16bf16f32of32(
                order, trans, 'B', k, n, reorder_meta);
        return round_up_align(req, kPrepackAlign);
    }

    if (params.wei_dtype == data_type_t::f16) {
        const size_t req = aocl_get_reorder_buf_size_f16f16f16of16(
                order, trans, 'B', k, n, nullptr);
        return round_up_align(req, kPrepackAlign);
    }

    if (params.wei_dtype == data_type_t::s4
            && params.src_dtype == data_type_t::s8) {
        // Native W4A8: packed s4, group size in b_quant_op.
        dlp_metadata_t symq_meta = {};
        dlp_quant_op_t symq_b_quant_op = {};
        symq_b_quant_op.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
        symq_b_quant_op.group_size = params.sym_group_size > 0
                ? params.sym_group_size
                : static_cast<int>(params.K);
        symq_meta.b_quant_op = &symq_b_quant_op;
        const size_t req = aocl_get_reorder_buf_size_s8s4s32os32(
                order, trans, 'B', k, n, &symq_meta);
        if (req == 0) {
            apilog_error(
                    "weight_prepack(aocl_dlp): "
                    "aocl_get_reorder_buf_size_s8s4s32os32 returned 0 "
                    "(unsupported ISA or group size); group_size=",
                    symq_b_quant_op.group_size, ", K=", params.K);
            return 0;
        }
        return round_up_align(req, kPrepackAlign);
    }

    if (params.wei_dtype == data_type_t::s4
            || params.wei_dtype == data_type_t::u4) {
        const size_t req = aocl_get_reorder_buf_size_bf16s4f32of32(
                order, trans, 'B', k, n, nullptr);
        return round_up_align(req, kPrepackAlign);
    }

    if (params.wei_dtype == data_type_t::s8) {
        if (params.sym_group_size > 0
                && (params.src_dtype == data_type_t::bf16
                        || params.src_dtype == data_type_t::s8)) {
            // B-side group size now travels inside dlp_metadata_t->b_quant_op
            // (new AOCL DLP reorder API); DLP_SYMM_STAT_QUANT was removed.
            dlp_metadata_t symq_meta = {};
            dlp_quant_op_t symq_b_quant_op = {};
            symq_b_quant_op.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
            symq_b_quant_op.group_size = params.sym_group_size;
            symq_meta.b_quant_op = &symq_b_quant_op;
            const size_t req = aocl_get_reorder_buf_size_s8s8s32os32_sym_quant(
                    order, trans, 'B', k, n, &symq_meta);
            return round_up_align(req, kPrepackAlign);
        }

        if (params.src_dtype == data_type_t::u8) {
            dlp_gemm_hints_t int8_hints {};
            dlp_metadata_t int8_md {};
            dlp_metadata_t *reorder_meta = matmul::aocl_dlp_hints_metadata(
                    nullptr, int8_hints, int8_md, num_threads);
            const size_t req = aocl_get_reorder_buf_size_u8s8s32os32(
                    order, trans, 'B', k, n, reorder_meta);
            // AOCL returns zero when the reorder is unsupported or its
            // parameters are invalid. Preserve that failure sentinel: adding
            // the ZenDNN column-sum tail to zero would otherwise make the size
            // query appear successful.
            if (req == 0) {
                apilog_error(
                        "weight_prepack(aocl_dlp): AOCL u8s8 reorder size "
                        "query failed");
                return 0;
            }
            return round_up_align(req, kPrepackAlign)
                    + colsum_buffer_bytes(params);
        }

        // src = s8 / bf16 / f32 / unspecified -> s8s8s32os32
        // Attach m_hint only for INT8-pure src (s8). Mixed bf16/f32 src uses
        // the same reorder API but GEMM does not take gemm_hints, so a
        // hinted pack would mismatch the runtime kernel.
        dlp_metadata_t *reorder_meta = nullptr;
        dlp_gemm_hints_t int8_hints {};
        dlp_metadata_t int8_md {};
        if (params.src_dtype == data_type_t::s8) {
            reorder_meta = matmul::aocl_dlp_hints_metadata(
                    nullptr, int8_hints, int8_md, num_threads);
        }
        const size_t req = aocl_get_reorder_buf_size_s8s8s32os32(
                order, trans, 'B', k, n, reorder_meta);
        return round_up_align(req, kPrepackAlign);
    }

    apilog_error("weight_prepack(aocl_dlp): unsupported wei_dtype=",
            dtype_info(params.wei_dtype));
    return 0;
#endif
}

status_t aocl_prepack(const void *weights, const prepack_params_t &params,
        void *dst, int32_t num_threads = 0) {
    using namespace zendnnl::lowoha::matmul;

#if !ZENDNNL_DEPENDS_AOCLDLP
    (void)weights;
    (void)params;
    (void)dst;
    apilog_error(
            "weight_prepack(aocl_dlp): ZenDNNL was built without AOCL-DLP "
            "support (ZENDNNL_DEPENDS_AOCLDLP=0); prepack is unavailable.");
    return status_t::unimplemented;
#else
    const char order = 'r';
    const char trans = params.transposed ? 't' : 'n';
    const md_t k = static_cast<md_t>(params.K);
    const md_t n = static_cast<md_t>(params.N);
    const md_t ldb = static_cast<md_t>(params.ldb);

    if (params.wei_dtype == data_type_t::f32) {
        aocl_reorder_f32f32f32of32(order, trans, 'B',
                static_cast<const float *>(weights), static_cast<float *>(dst),
                k, n, ldb, nullptr);
        return status_t::success;
    }

    if (params.wei_dtype == data_type_t::bf16) {
        dlp_gemm_hints_t bf16_hints {};
        dlp_metadata_t bf16_md {};
        dlp_metadata_t *reorder_meta = matmul::aocl_dlp_hints_metadata(
                nullptr, bf16_hints, bf16_md, num_threads);
        aocl_reorder_bf16bf16f32of32(order, trans, 'B',
                static_cast<const int16_t *>(weights),
                static_cast<int16_t *>(dst), k, n, ldb, reorder_meta);
        return status_t::success;
    }

    if (params.wei_dtype == data_type_t::f16) {
        aocl_reorder_f16f16f16of16(order, trans, 'B',
                static_cast<const uint16_t *>(weights),
                static_cast<uint16_t *>(dst), k, n, ldb, nullptr);
        return status_t::success;
    }

    if (params.wei_dtype == data_type_t::s4
            && params.src_dtype == data_type_t::s8) {
        // Native W4A8 reorder consumes packed s4 in place.
        dlp_metadata_t symq_meta = {};
        dlp_quant_op_t symq_b_quant_op = {};
        symq_b_quant_op.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
        symq_b_quant_op.group_size = params.sym_group_size > 0
                ? params.sym_group_size
                : static_cast<int>(params.K);
        symq_meta.b_quant_op = &symq_b_quant_op;
        aocl_reorder_s8s4s32os32(order, trans, 'B',
                static_cast<const int8_t *>(weights),
                static_cast<int8_t *>(dst), k, n, ldb, &symq_meta);
        return status_t::success;
    }

    if (params.wei_dtype == data_type_t::s4
            || params.wei_dtype == data_type_t::u4) {
        aocl_reorder_bf16s4f32of32(order, trans, 'B',
                static_cast<const int8_t *>(weights),
                static_cast<int8_t *>(dst), k, n, ldb, nullptr);
        return status_t::success;
    }

    if (params.wei_dtype == data_type_t::s8) {
        if (params.sym_group_size > 0
                && (params.src_dtype == data_type_t::bf16
                        || params.src_dtype == data_type_t::s8)) {
            // B-side group size now travels inside dlp_metadata_t->b_quant_op
            // (new AOCL DLP reorder API); DLP_SYMM_STAT_QUANT was removed.
            dlp_metadata_t symq_meta = {};
            dlp_quant_op_t symq_b_quant_op = {};
            symq_b_quant_op.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
            symq_b_quant_op.group_size = params.sym_group_size;
            symq_meta.b_quant_op = &symq_b_quant_op;
            aocl_reorder_s8s8s32os32_sym_quant(order, trans, 'B',
                    static_cast<const int8_t *>(weights),
                    static_cast<int8_t *>(dst), k, n, ldb, &symq_meta);
            return status_t::success;
        }

        if (params.src_dtype == data_type_t::u8) {
            dlp_gemm_hints_t int8_hints {};
            dlp_metadata_t int8_md {};
            dlp_metadata_t *reorder_meta = matmul::aocl_dlp_hints_metadata(
                    nullptr, int8_hints, int8_md, num_threads);
            aocl_reorder_u8s8s32os32(order, trans, 'B',
                    static_cast<const int8_t *>(weights),
                    static_cast<int8_t *>(dst), k, n, ldb, reorder_meta);
            if (wants_colsum(params)) {
                write_weight_colsum(weights, params, dst);
            }
            return status_t::success;
        }

        dlp_metadata_t *reorder_meta = nullptr;
        dlp_gemm_hints_t int8_hints {};
        dlp_metadata_t int8_md {};
        if (params.src_dtype == data_type_t::s8) {
            reorder_meta = matmul::aocl_dlp_hints_metadata(
                    nullptr, int8_hints, int8_md, num_threads);
        }
        aocl_reorder_s8s8s32os32(order, trans, 'B',
                static_cast<const int8_t *>(weights),
                static_cast<int8_t *>(dst), k, n, ldb, reorder_meta);
        return status_t::success;
    }

    apilog_error("weight_prepack(aocl_dlp): unsupported wei_dtype=",
            dtype_info(params.wei_dtype));
    return status_t::unimplemented;
#endif
}

// Unpack an AOCL blocked weight into dst. Size is rows * ldb * elem.
size_t aocl_unreorder_compute_size(const prepack_params_t &params) {
#if !ZENDNNL_DEPENDS_AOCLDLP
    (void)params;
    apilog_error(
            "weight_unreorder(aocl_dlp): ZenDNNL was built without AOCL-DLP "
            "support (ZENDNNL_DEPENDS_AOCLDLP=0); unreorder is unavailable.");
    return 0;
#else
    const size_t elem = zendnnl::common::size_of(params.wei_dtype);
    if (elem == 0) {
        apilog_error("weight_unreorder(aocl_dlp): unsupported wei_dtype=",
                dtype_info(params.wei_dtype));
        return 0;
    }
    const uint64_t rows
            = static_cast<uint64_t>(params.transposed ? params.N : params.K);
    const uint64_t ldb = static_cast<uint64_t>(params.ldb);
    if (ldb != 0 && rows > std::numeric_limits<uint64_t>::max() / ldb) {
        apilog_error("weight_unreorder(aocl_dlp): rows*ldb overflows");
        return 0;
    }
    const uint64_t n_elem = rows * ldb;
    if (n_elem > std::numeric_limits<uint64_t>::max() / elem) {
        apilog_error("weight_unreorder(aocl_dlp): output byte size overflows");
        return 0;
    }
    return static_cast<size_t>(n_elem * elem);
#endif
}

status_t aocl_unreorder(const void *packed, const prepack_params_t &params,
        void *dst, int32_t num_threads = 0) {
#if !ZENDNNL_DEPENDS_AOCLDLP
    (void)packed;
    (void)params;
    (void)dst;
    (void)num_threads;
    apilog_error(
            "weight_unreorder(aocl_dlp): ZenDNNL was built without AOCL-DLP "
            "support (ZENDNNL_DEPENDS_AOCLDLP=0); unreorder is unavailable.");
    return status_t::unimplemented;
#else
    using namespace zendnnl::lowoha::matmul;

    // 0 means an unsupported dtype or an overflow. Validation already ran.
    if (aocl_unreorder_compute_size(params) == 0) {
        return status_t::unimplemented;
    }

    // AOCL reports failure only through metadata.
    const char order = 'r';
    const char trans = params.transposed ? 't' : 'n';
    const char mat_type = 'B';
    const md_t k = static_cast<md_t>(params.K);
    const md_t n = static_cast<md_t>(params.N);
    const md_t ldb = static_cast<md_t>(params.ldb);
    dlp_gemm_hints_t hints {};
    dlp_metadata_t md {};
    // Same gemm_hints as prepack. AOCL derives NR from them.
    dlp_metadata_t *unreorder_meta = &md;
    if (params.consider_m_hint && aocl_pack_uses_m_hint(params)) {
        unreorder_meta
                = matmul::aocl_dlp_hints_metadata(&md, hints, md, num_threads);
    }

    if (params.wei_dtype == data_type_t::f32) {
        aocl_unreorder_f32f32f32of32_reference(order, trans, mat_type,
                static_cast<const float *>(packed), static_cast<float *>(dst),
                k, n, ldb, unreorder_meta);
    } else if (params.wei_dtype == data_type_t::bf16) {
        aocl_unreorder_bf16bf16f32of32(order, trans, mat_type,
                static_cast<const int16_t *>(packed),
                static_cast<int16_t *>(dst), k, n, ldb, unreorder_meta);
    } else if (params.wei_dtype == data_type_t::f16) {
        // Tuned kernel needs AVX512-FP16. Reference is portable C.
        if (zendnnl::common::zendnnl_platform_info().get_avx512_f16_status()) {
            aocl_unreorder_f16f16f16of16(order, trans, mat_type,
                    static_cast<const uint16_t *>(packed),
                    static_cast<uint16_t *>(dst), k, n, ldb, unreorder_meta);
        } else {
            aocl_unreorder_f16f16f16of16_reference(order, trans, mat_type,
                    static_cast<const uint16_t *>(packed),
                    static_cast<uint16_t *>(dst), k, n, ldb, unreorder_meta);
        }
    } else if (params.wei_dtype == data_type_t::s8) {
        // u8 source: recover the s8 weights. The column-sum tail is not written.
        if (params.src_dtype == data_type_t::u8) {
            aocl_unreorder_u8s8s32os32_reference(order, trans, mat_type,
                    static_cast<const int8_t *>(packed),
                    static_cast<int8_t *>(dst), k, n, ldb, unreorder_meta);
        } else {
            aocl_unreorder_s8s8s32os32_reference(order, trans, mat_type,
                    static_cast<const int8_t *>(packed),
                    static_cast<int8_t *>(dst), k, n, ldb, unreorder_meta);
        }
    } else {
        apilog_error("weight_unreorder(aocl_dlp): unsupported wei_dtype=",
                dtype_info(params.wei_dtype));
        return status_t::unimplemented;
    }

    if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        apilog_error("weight_unreorder(aocl_dlp): AOCL error_code=",
                static_cast<int>(md.error_hndl.error_code));
        return md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED
                ? status_t::unimplemented
                : status_t::failure;
    }
    return status_t::success;
#endif
}

// =====================================================================
// Custom-kernel (group_matmul) prepack — MEMORY-FORMAT CHANGE ONLY.
// Packs the weight into the caller's `dst` buffer in the VNNI layout
// the custom kernel consumes; it does NOT touch the per-process LRU
// pack cache (that is the matmul side's job when it consumes the
// already-reordered weight). Family is chosen by wei_dtype
// (bf16 -> VDPBF16PS pack; f16 -> plain native-FP16 slab; s8 ->
// DQ-INT8 VPDPBUSD pack + comp row).
// =====================================================================
size_t ck_compute_size(const prepack_params_t &params) {
    const int pack_nr = ck_resolve_pack_nr(params);
    if (pack_nr == 0) return 0; // validation already logged the cause
    const int K = static_cast<int>(params.K);
    const int N = static_cast<int>(params.N);
    if (params.wei_dtype == data_type_t::s8)
        return ck::packed_weight_size_int8(K, N, pack_nr);
    if (params.wei_dtype == data_type_t::f16)
        return ck::packed_weight_size_f16(K, N, pack_nr);
    return ck::packed_weight_size_bf16(K, N, pack_nr);
}

status_t ck_prepack(
        const void *weights, const prepack_params_t &params, void *dst) {
    const int pack_nr = ck_resolve_pack_nr(params);
    if (pack_nr == 0) {
        apilog_error(
                "weight_prepack_into(moe_custom_kernel): no pack_nr in "
                "{32, 64} divides N=",
                params.N);
        return status_t::failure;
    }
    const int K = static_cast<int>(params.K);
    const int N = static_cast<int>(params.N);
    const int ldb = static_cast<int>(params.ldb);

    if (params.wei_dtype == data_type_t::s8) {
        return ck::prepack_weight_into_int8(
                static_cast<const int8_t *>(weights), K, N, ldb, pack_nr,
                params.transposed, params.interleave_split_halves, dst);
    }
    if (params.wei_dtype == data_type_t::f16) {
        return ck::prepack_weight_into_f16(
                static_cast<const zendnnl::common::float16_t *>(weights), K, N,
                ldb, pack_nr, params.transposed, params.interleave_split_halves,
                dst);
    }
    return ck::prepack_weight_into_bf16(
            static_cast<const zendnnl::common::bfloat16_t *>(weights), K, N,
            ldb, pack_nr, params.transposed, params.interleave_split_halves,
            dst);
}

// =====================================================================
// Backend dispatchers: AOCL DLP blocked layout, or the group_matmul
// custom-kernel VNNI pack-into-dst (selected by algo == moe_custom_kernel).
// =====================================================================
size_t backend_size_by_algo(
        const prepack_params_t &params, int32_t num_threads = 0) {
    if (params.algo == matmul_algo_t::moe_custom_kernel) {
        return ck_compute_size(params);
    }
#if !ZENDNNL_DEPENDS_AOCLDLP
    apilog_warning(
            "backend_size_by_algo: ZenDNNL was built without AOCL-DLP "
            "support (ZENDNNL_DEPENDS_AOCLDLP=0); sizing reference (dense "
            "row-major) prepack fallback instead of aocl_dlp_blocked layout.");
    return round_up_align(dense_weight_bytes(params), kPrepackAlign);
#endif
    if (params.algo == matmul_algo_t::aocl_dlp_blocked) {
        return aocl_compute_size(params, num_threads);
    }
    apilog_error("weight_prepack_size: algo not supported by prepack API (",
            kernel_to_string(params.algo),
            "); only aocl_dlp_blocked / moe_custom_kernel are supported");
    return 0;
}

status_t backend_prepack_by_algo(const void *weights,
        const prepack_params_t &params, void *dst, int32_t num_threads = 0) {
    if (params.algo == matmul_algo_t::moe_custom_kernel) {
        return ck_prepack(weights, params, dst);
    }
#if !ZENDNNL_DEPENDS_AOCLDLP
    apilog_warning(
            "backend_prepack_by_algo: ZenDNNL was built without AOCL-DLP "
            "support (ZENDNNL_DEPENDS_AOCLDLP=0); using reference (dense "
            "row-major) prepack fallback instead of aocl_dlp_blocked layout.");
    reference_prepack_weights(weights, params, dst);
    return status_t::success;
#endif
    if (params.algo == matmul_algo_t::aocl_dlp_blocked) {
        return aocl_prepack(weights, params, dst, num_threads);
    }
    apilog_error("weight_prepack_into: algo not supported by prepack API (",
            kernel_to_string(params.algo),
            "); only aocl_dlp_blocked / moe_custom_kernel are supported");
    return status_t::unimplemented;
}

} // anonymous namespace

// ===========================================================================
// Public API: weight_prepack_size
// ===========================================================================

size_t weight_prepack_size(const reorder_params_t &params) {
    const prepack_params_t &pp = params.prepack;

    // Always reset the cache and recompute.
    pp.cached_size = 0;

    // Diagnostic-gated params validation (no weights pointer needed here).
    status_t val_status = op_instrumentation::validate([&]() {
        return validate_prepack_params("weight_prepack_size", pp);
    });
    if (val_status != status_t::success) { return 0; }

    if (apilog_info_enabled()) {
        std::ostringstream ss;
        ss << "LOWOHA weight_prepack_size: algo=" << kernel_to_string(pp.algo)
           << ", K=" << pp.K << ", N=" << pp.N
           << ", wei_dtype=" << dtype_info(pp.wei_dtype);
        apilog_info(ss.str());
    }

    const size_t size = backend_size_by_algo(pp, params.num_threads);
    if (size > 0) { pp.cached_size = size; }
    return size;
}

status_t weight_prepack_into(
        const void *weights, const reorder_params_t &params, void *dst) {
    const prepack_params_t &pp = params.prepack;

    // Forwarded as nt_hint, not used as the OpenMP team size.

    // Diagnostic-gated input validation.
    status_t val_status = op_instrumentation::validate([&]() {
        return validate_prepack_inputs("weight_prepack_into", weights, pp);
    });
    if (val_status != status_t::success) { return val_status; }
    // Every path (AOCL blocked and moe_custom_kernel) now writes the
    // reordered weight into the caller's `dst` buffer.
    if (!dst) {
        apilog_error("weight_prepack_into: dst pointer is null");
        return status_t::failure;
    }

    // The caller is contractually required to have allocated at least
    // weight_prepack_size(params) bytes at `dst`. No alignment check
    // here: none of the supported backends require a specific alignment
    // for correctness.

    profiler_t profiler;
    const bool is_profile = is_profile_enabled();

    [[maybe_unused]] std::string log_str;
    if (apilog_info_enabled() || is_profile) {
        std::ostringstream ss;
        ss << "LOWOHA weight_prepack_into: algo=" << kernel_to_string(pp.algo)
           << ", K=" << pp.K << ", N=" << pp.N << ", ldb=" << pp.ldb
           << ", trans=" << (pp.transposed ? 't' : 'n')
           << ", wei_dtype=" << dtype_info(pp.wei_dtype);
        log_str = ss.str();
        if (apilog_info_enabled()) { apilog_info(log_str); }
    }

    if (is_profile) { profiler.tbp_start(); }

    const status_t st
            = backend_prepack_by_algo(weights, pp, dst, params.num_threads);

    if (is_profile) {
        profiler.tbp_stop();
        profilelog_verbose(log_str, ", time=", profiler.tbp_elapsedtime(),
                profiler.get_res_str());
    }

    return st;
}

// ===========================================================================
// Public API: weight_unreorder_size / weight_unreorder_into.
// ===========================================================================

size_t weight_unreorder_size(const reorder_params_t &params) {
    const prepack_params_t &pp = params.prepack;

    // Leave cached_size alone: it stores the last prepack size.
    if (validate_unreorder_params("weight_unreorder_size", pp)
            != status_t::success) {
        return 0;
    }

    if (apilog_info_enabled()) {
        std::ostringstream ss;
        ss << "LOWOHA weight_unreorder_size: algo=" << kernel_to_string(pp.algo)
           << ", K=" << pp.K << ", N=" << pp.N
           << ", wei_dtype=" << dtype_info(pp.wei_dtype);
        apilog_info(ss.str());
    }

    return aocl_unreorder_compute_size(pp);
}

status_t weight_unreorder_into(
        const void *packed, const reorder_params_t &params, void *dst) {
    const prepack_params_t &pp = params.prepack;

    if (!packed) {
        apilog_error("weight_unreorder_into: packed pointer is null");
        return status_t::failure;
    }
    if (!dst) {
        apilog_error("weight_unreorder_into: dst pointer is null");
        return status_t::failure;
    }

    const status_t val_status
            = validate_unreorder_params("weight_unreorder_into", pp);
    if (val_status != status_t::success) { return val_status; }

    profiler_t profiler;
    const bool is_profile = is_profile_enabled();

    [[maybe_unused]] std::string log_str;
    if (apilog_info_enabled() || is_profile) {
        std::ostringstream ss;
        ss << "LOWOHA weight_unreorder_into: algo=" << kernel_to_string(pp.algo)
           << ", K=" << pp.K << ", N=" << pp.N << ", ldb=" << pp.ldb
           << ", trans=" << (pp.transposed ? 't' : 'n')
           << ", wei_dtype=" << dtype_info(pp.wei_dtype);
        log_str = ss.str();
        if (apilog_info_enabled()) { apilog_info(log_str); }
    }

    if (is_profile) { profiler.tbp_start(); }

    const status_t st = aocl_unreorder(packed, pp, dst, params.num_threads);

    if (is_profile) {
        profiler.tbp_stop();
        profilelog_verbose(log_str, ", time=", profiler.tbp_elapsedtime(),
                profiler.get_res_str());
    }

    return st;
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
