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

#include "lowoha_operators/matmul/matmul_native/native_matmul.hpp"

#include <algorithm>
#include <exception>
#include <new>
#include <sstream>
#include <string>
#include <type_traits>

#include "common/data_types.hpp"
#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/lowoha_quant_layout.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_gemv.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_operator_bridge.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::dtype_info;
using namespace zendnnl::error_handling;

// The descriptor of a call.
static GemmDescriptor make_desc(bool transA, bool transB, int M, int N, int K,
        float alpha, float beta, int lda, int ldb, int ldc,
        bool is_weights_const, int num_threads, const matmul_params &params) {
    GemmDescriptor desc;
    desc.M = M;
    desc.N = N;
    desc.K = K;
    desc.lda = lda;
    desc.ldb = ldb;
    desc.ldc = ldc;
    desc.transA = transA;
    desc.transB = transB;
    desc.alpha = alpha;
    desc.beta = beta;
    desc.src_dt = params.dtypes.src;
    desc.wei_dt = params.dtypes.wei;
    desc.dst_dt = params.dtypes.dst;
    desc.bias_dt = params.dtypes.bias;
    desc.bias = nullptr;
    desc.is_weights_const = is_weights_const;
    desc.num_threads = num_threads;
    return desc;
}

native_layout_query_t native_layout_query_for(
        const matmul_params &params, int K, int N, int ldb, bool transB) {
    native_layout_query_t q;
    q.wei_dt = params.dtypes.wei;
    q.src_dt = params.dtypes.src;
    const data_type_t compute = params.dtypes.compute;
    if (params.dynamic_quant
            && (q.src_dt == data_type_t::bf16 || q.src_dt == data_type_t::f32)
            && (q.wei_dt == data_type_t::s8 || q.wei_dt == data_type_t::s4)
            && (compute == data_type_t::s8 || compute == data_type_t::u8)) {
        q.src_dt = compute;
    }
    q.K = K;
    q.N = N;
    q.ldb = ldb;
    q.transB = transB;
    if (has_per_group_wei_scale(params.quant_params, N)) {
        const int64_t groups = params.quant_params.wei_scale.dims[0];
        q.wei_group_size = quant_groups_divide(K, groups)
                ? static_cast<int>(K / groups)
                : -1;
    }
    return q;
}

namespace {

/// Why the kernel did not serve a call it was given.
const char *kernel_status_text(status_t st) {
    switch (st) {
        case status_t::unimplemented: return "the kernel does not implement it";
        case status_t::isa_unsupported:
            return "the CPU lacks an instruction set the kernel needs";
        default: return "the kernel failed";
    }
}

const char *gated_act_name(matmul_gated_act_t act) {
    switch (act) {
        case matmul_gated_act_t::silu_and_mul: return "silu_and_mul";
        case matmul_gated_act_t::gelu_and_mul: return "gelu_and_mul";
        case matmul_gated_act_t::swiglu_oai_mul: return "swiglu_oai_mul";
        default: return "none";
    }
}

/// Where the weight of a call the native BRGEMM takes comes from.
enum class native_source_t : uint8_t {
    none, ///< not for the native BRGEMM: the other routes serve the call
    /// A call that asks for the native BRGEMM, which does not serve it.
    refused,
    raw, ///< a raw weight the layout rule places on the layout
    /// An AOCL prepack whose raw weight the layout rule places on the layout:
    /// unreordered, then packed per the weight cache mode.  AOCL still reads
    /// it until a weight cache mode 2 pack rewrites it in place.
    aocl_prepacked,
    in_place, ///< a buffer an earlier native call rewrote in place
};

/// The bytes at weight a call vouches for as its weight buffer: what it
/// reads from a bf16 / s8 / s4 / f32 weight (an AOCL prepack this route
/// unreorders, or the span of a plain weight), or the capacity it states
/// when larger; 0 when neither is known.
ZENDNNL_ALWAYS_INLINE inline size_t weight_buffer_bytes(
        const matmul_params &params, int K, int N, int ldb, bool transB) {
    size_t reads = 0;
    const data_type_t wdt = params.dtypes.wei;
    if ((wdt == data_type_t::bf16 || wdt == data_type_t::s8
                || wdt == data_type_t::s4 || wdt == data_type_t::f32)
            && K > 0 && N > 0 && ldb > 0 && params.packing.pack_format_b == 0) {
        if (params.mem_format_b == 'r') {
            if (aocl_prepacked_weight(params)
                    && aocl_prepack_unreorderable(params))
                reads = aocl_prepack_bytes(
                        K, N, ldb, transB, wdt == data_type_t::s8);
        } else if (params.mem_format_b == 'n') {
            reads = native_raw_weight_bytes(wdt, K, N, ldb, transB);
        }
    }
    return std::max(reads, params.wei_buffer_capacity_bytes);
}

/// The source of the weight of a call the native BRGEMM looks at.  A call
/// that asks for it (native_brgemm_requested) is the native BRGEMM's
/// whatever it carries: refused, with reason and status (what matmul_direct
/// returns for it), when the native BRGEMM does not serve its weight.  So is
/// a call on a buffer an earlier native call rewrote in place, which no
/// other route reads: its reason is set when it cannot read the image.
/// requested is native_brgemm_requested of the call, owned its
/// weight_buffer_bytes; image is what its check for an in-place image found.  With gemv (the call runs on a GEMV kernel over its
/// raw weight) the layout rule does not apply and pack_nr stays 0.
ZENDNNL_ALWAYS_INLINE inline native_source_t classify_source(
        const matmul_params &params, const void *weight, int K, int N, int ldb,
        bool transB, int batch_count, bool requested, bool process_native,
        bool gemv, size_t owned, int &pack_nr, const char *&reason,
        status_t &status, native_image_check_t &image) {
    // A native in-place image has this route as its only reader, whatever
    // algorithm the call selects.
    if (native_weight_find_image(weight, owned, image)) {
        const native_layout_decision_t rule = decide_native_layout(
                native_layout_query_for(params, K, N, ldb, transB));
        pack_nr = kNativeLayoutPackNr;
        if (!rule.native)
            reason = rule.reason;
        else if (batch_count != 1)
            reason = "a batched call";
        if (reason != nullptr) status = status_t::failure;
        return native_source_t::in_place;
    }
    const auto refuse = [&](const char *why, status_t st) {
        if (!requested) return native_source_t::none;
        reason = why;
        status = st;
        return native_source_t::refused;
    };
    bool aocl = false;
    if (params.mem_format_b == 'r') {
        if (!process_native)
            return refuse("a reordered weight without ZENDNNL_MATMUL_ALGO=10",
                    status_t::unimplemented);
        if (!aocl_prepacked_weight(params))
            return refuse(
                    "a reordered weight other than an AOCL-DLP blocked one "
                    "under ZENDNNL_MATMUL_ALGO=10",
                    status_t::unimplemented);
        aocl = true;
    } else if (!requested) {
        return native_source_t::none;
    } else if (params.mem_format_b != 'n') {
        return refuse("a weight format other than raw or AOCL-DLP blocked",
                status_t::unimplemented);
    }

    if (batch_count != 1)
        return refuse("a batched call", status_t::unimplemented);
    if (params.packing.pack_format_b != 0)
        return refuse("a GGML pack format", status_t::unimplemented);
    if (!gemv) {
        const native_layout_decision_t rule = decide_native_layout(
                native_layout_query_for(params, K, N, ldb, transB));
        if (!rule.native)
            return refuse(rule.reason,
                    rule.isa ? status_t::isa_unsupported
                             : status_t::unimplemented);
        pack_nr = rule.pack_nr;
    }
    if (aocl)
        return aocl_prepack_unreorderable(params)
                ? native_source_t::aocl_prepacked
                : refuse("an AOCL-DLP blocked weight without an unreorder",
                          status_t::unimplemented);
    // A buffer a custom-kernel pack rewrote in place has its reader on that
    // route.
    if (custom_kernel_packed_in_place(weight))
        return refuse("a buffer the custom kernel rewrote in place",
                status_t::failure);
    return native_source_t::raw;
}

/// native_layout_matmul, or with CheckOnly native_layout_check: reason names
/// why a call the native BRGEMM takes is not served.
template <bool CheckOnly>
ZENDNNL_ALWAYS_INLINE inline bool take_call(char layout, bool transA,
        bool transB, int M, int N, int K, float alpha, const void *src, int lda,
        const void *weight, int ldb, const void *bias, float beta, void *dst,
        int ldc, bool is_weights_const, int batch_count, int num_threads,
        matmul_gated_act_t gated_act, const matmul_params &params,
        zendnnl::profile::profiler_t *profiler, status_t &status,
        const char *&reason, const matmul_batch_params_t *batch) {
    // The call's descriptor, made once for the GEMV choice and the run, in
    // place: it is never destroyed.
    static_assert(std::is_trivially_destructible<GemmDescriptor>::value,
            "the call's descriptor is never destroyed");
    alignas(GemmDescriptor) unsigned char call_storage[sizeof(GemmDescriptor)];
    GemmDescriptor *call = nullptr;
    const auto call_desc = [&]() -> const GemmDescriptor & {
        if (call == nullptr) {
            call = ::new (call_storage) GemmDescriptor(
                    make_desc(transA, transB, M, N, K, alpha, beta, lda, ldb,
                            ldc, is_weights_const, num_threads, params));
            call->bias = bias;
            call->gated_act = gated_act;
        }
        return *call;
    };
    const bool requested = native_brgemm_requested(params, batch_count);
    // An M = 1 call on its raw weight may run on a GEMV kernel instead of
    // the tiles.
    native_gemv_plan_t gemv;
    if (M == 1 && batch_count == 1 && params.mem_format_b == 'n' && requested)
        gemv = decide_native_gemv(call_desc(), params);
    const bool process_native = matmul_config_t::is_native_brgemm_algo();
    int pack_nr = 0;
    reason = nullptr;
    status = status_t::success;
    const size_t owned = weight_buffer_bytes(params, K, N, ldb, transB);
    native_weight_ref_t ref;
    const native_source_t source = classify_source(params, weight, K, N, ldb,
            transB, batch_count, requested, process_native,
            gemv.kernel != native_gemv_t::none, owned, pack_nr, reason, status,
            ref.image);
    if (source == native_source_t::none) return false;
    bool in_place = source == native_source_t::in_place;
    // An image holds the layout, not the raw weight the GEMV kernels read.
    if (in_place) gemv = native_gemv_plan_t {};
    status_t refusal = in_place ? status_t::failure : status_t::unimplemented;
    if (reason == nullptr && layout != 'r' && layout != 'R') {
        reason = "a column-major call";
        status = refusal;
    }
    if (reason == nullptr && params.mem_format_a != 'n') {
        reason = "a reordered source";
        status = refusal;
    }
    // beta of a gated call reads the dst's N pre-activation columns, as on
    // the one-op group route.
    if (reason == nullptr && gated_act != matmul_gated_act_t::none
            && beta != 0.0f && ldc < N) {
        reason = "beta with a gated activation over a dst of fewer than N "
                 "columns";
        status = refusal;
    }

    ref.weight = weight;
    ref.pack_nr = pack_nr;
    ref.aocl_blocked = source == native_source_t::aocl_prepacked
            || (in_place && aocl_prepacked_weight(params));
    ref.interleave_split_halves = gated_act == matmul_gated_act_t::silu_and_mul
            || gated_act == matmul_gated_act_t::gelu_and_mul;
    ref.weight_cache_mode = is_weights_const
            ? effective_weight_cache_type(params.weight_cache_type)
            : 0;
    // A buffer rewritten in place must have no other reader, which only the
    // process-wide ALGO 10 guarantees (every call then reaches the image
    // check of classify_source); a call that selects the native BRGEMM by
    // itself caches the layout out of place instead.
    if (ref.weight_cache_mode == 2 && !process_native)
        ref.weight_cache_mode = 1;
    ref.capacity_bytes = params.wei_buffer_capacity_bytes;

    // A concurrent first call may have rewritten the buffer since it was
    // classified; then no other route may read it either.  A call that does
    // not ask for the native BRGEMM (an AOCL prepack under
    // ZENDNNL_MATMUL_ALGO=10 that names its backend) keeps its own route
    // when the native BRGEMM does not serve it.
    const auto image_now = [&]() {
        return in_place || native_weight_packed_in_place(weight, owned);
    };
    if (reason != nullptr && !requested && !image_now()) return false;
    std::string error_text;
    if (reason == nullptr) {
        const GemmDescriptor &desc = call_desc();
        const thread_guard tg(num_threads, thread_guard::max_threads());
        const char *why = nullptr;
        // Nothing may throw out of matmul_direct (allocations of the call's
        // buffers, an executor).
        try {
            status = CheckOnly ? native_brgemm_check(desc, ref, src, dst, bias,
                                         params, gemv, &why)
                               : native_brgemm_execute(desc, ref, src, dst,
                                         bias, params, gemv, &why);
        } catch (const std::exception &e) {
            // The text of the exception lives until the line names it.
            if (!CheckOnly) error_text = e.what();
            why = CheckOnly ? "an exception of the native BRGEMM"
                            : error_text.c_str();
            status = status_t::failure;
        }
        if (status != status_t::success) {
            const bool image = image_now();
            if (!requested && !image
                    && (status == status_t::unimplemented
                            || status == status_t::isa_unsupported))
                return false;
            reason = why != nullptr ? why : kernel_status_text(status);
            if (image) {
                in_place = true;
                status = status_t::failure;
            }
        }
    } else if (status != status_t::failure && image_now()) {
        in_place = true;
        status = status_t::failure;
    }
    if (CheckOnly) return true;

    // The line matmul_direct logs for every other route, with the fields of
    // this one.
    if (apilog_info_enabled() || profiler != nullptr) {
        const std::string po_dtypes = post_op_dtypes_text(params);
        std::ostringstream ss;
        ss << "LOWOHA matmul_direct: M=" << M << ", N=" << N << ", K=" << K
           << ", alpha=" << alpha << ", beta=" << beta << ", lda=" << lda
           << ", ldb=" << ldb << ", ldc=" << ldc
           << ", transA=" << (transA ? "true" : "false")
           << ", transB=" << (transB ? "true" : "false")
           << ", input_dtype=" << dtype_info(params.dtypes.src)
           << ", weight_dtype=" << dtype_info(params.dtypes.wei)
           << ", output_dtype=" << dtype_info(params.dtypes.dst)
           << ", bias_dtype=" << dtype_info(params.dtypes.bias)
           << ", bias=" << (bias != nullptr ? "true" : "false")
           << ", is_weights_const=" << (is_weights_const ? "true" : "false")
           << ", post_op=[" << post_op_names_text(params) << "]"
           << ", post_op_dtype=[" << (po_dtypes.empty() ? "none" : po_dtypes)
           << "], Batch_A=" << (batch ? batch->Batch_A : batch_count)
           << ", Batch_B=" << (batch ? batch->Batch_B : batch_count)
           << ", plugin_op=" << params.plugin_op
           << ", dynamic_quant=" << (params.dynamic_quant ? "true" : "false")
           << ", kernel=native_brgemm, layout=";
        if (gemv.kernel != native_gemv_t::none)
            ss << "raw";
        else
            ss << "native_nr" << pack_nr;
        ss << ", weight_cache=";
        if (in_place)
            ss << "inplace";
        else
            ss << ref.weight_cache_mode;
        ss << ", gated_act=" << gated_act_name(gated_act);
        const std::string line = ss.str();
        if (apilog_info_enabled()) apilog_info(line);
        if (profiler != nullptr) {
            profiler->tbp_stop();
            profilelog_verbose(line, ", weight_address=", weight,
                    ", time=", profiler->tbp_elapsedtime(),
                    profiler->get_res_str());
        }
    }
    if (status != status_t::success) {
        apilog_error("matmul_direct: native BRGEMM did not serve the call (",
                reason, ")",
                in_place ? "; no other route reads a buffer it rewrote" : "");
    }
    return true;
}

} // namespace

bool native_layout_matmul(char layout, bool transA, bool transB, int M, int N,
        int K, float alpha, const void *src, int lda, const void *weight,
        int ldb, const void *bias, float beta, void *dst, int ldc,
        bool is_weights_const, int batch_count, int num_threads,
        matmul_gated_act_t gated_act, matmul_params &params,
        zendnnl::profile::profiler_t *profiler, status_t &status,
        const matmul_batch_params_t *batch) {
    const char *reason = nullptr;
    return take_call<false>(layout, transA, transB, M, N, K, alpha, src, lda,
            weight, ldb, bias, beta, dst, ldc, is_weights_const, batch_count,
            num_threads, gated_act, params, profiler, status, reason, batch);
}

bool native_layout_check(char layout, bool transA, bool transB, int M, int N,
        int K, float alpha, const void *src, int lda, const void *weight,
        int ldb, const void *bias, float beta, void *dst, int ldc,
        bool is_weights_const, int batch_count, int num_threads,
        matmul_gated_act_t gated_act, const matmul_params &params,
        status_t &status, const char *&reason) {
    return take_call<true>(layout, transA, transB, M, N, K, alpha, src, lda,
            weight, ldb, bias, beta, dst, ldc, is_weights_const, batch_count,
            num_threads, gated_act, params, nullptr, status, reason, nullptr);
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
