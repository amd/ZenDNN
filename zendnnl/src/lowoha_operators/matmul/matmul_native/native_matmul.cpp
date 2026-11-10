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
#include <atomic>
#include <cstring>
#include <exception>
#include <mutex>
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
        case matmul_gated_act_t::gelu_tanh_and_mul: return "gelu_tanh_and_mul";
        case matmul_gated_act_t::swiglu_oai_mul: return "swiglu_oai_mul";
        default: return "none";
    }
}

/// The reason note_other_default_route added to its table last.
std::atomic<const char *> last_default_route {nullptr};

/// note_default_route for a reason other than the one logged last: the
/// reasons are literals, so a call finds a reason's address in the table
/// without a lock or a write.  An address the table lacks takes the lock to
/// join it, and logs unless its text is there at another address; past the
/// table's size a reason logs on each call.
ZENDNNL_NOINLINE void note_other_default_route(const char *reason) {
    constexpr int kNoted = 128;
    static std::atomic<const char *> noted[kNoted];
    for (const std::atomic<const char *> &slot : noted) {
        const char *seen = slot.load(std::memory_order_acquire);
        if (seen == reason) return;
        if (seen == nullptr) break;
    }
    {
        static std::mutex mutex;
        const std::lock_guard<std::mutex> lock(mutex);
        bool known = false;
        int at = 0;
        for (; at < kNoted; ++at) {
            const char *seen = noted[at].load(std::memory_order_relaxed);
            if (seen == nullptr) break;
            if (seen == reason) return;
            known = known || std::strcmp(seen, reason) == 0;
        }
        if (at < kNoted) noted[at].store(reason, std::memory_order_release);
        last_default_route.store(reason, std::memory_order_relaxed);
        if (known) return;
    }
    apilog_warning(
            "matmul_direct: under ZENDNNL_MATMUL_ALGO=10 a call the native "
            "BRGEMM does not serve (",
            reason, ") takes the default route (logged once per reason)");
}

/// One warning line per process for each reason of the calls without
/// lowoha_algo that ZENDNNL_MATMUL_ALGO=10 brings to the native BRGEMM and
/// that it hands to the default route.
ZENDNNL_ALWAYS_INLINE inline void note_default_route(const char *reason) {
    if (last_default_route.load(std::memory_order_relaxed) != reason)
        note_other_default_route(reason);
}

/// The default route has a kernel for a call of a src source (the type it
/// quantizes the source to, for a call it quantizes), a wei weight and a
/// dst destination.  For other types it returns success with dst not
/// written.
bool default_route_has_kernel(
        data_type_t src, data_type_t wei, data_type_t dst) {
    const bool f32_or_bf16
            = dst == data_type_t::f32 || dst == data_type_t::bf16;
    switch (wei) {
        case data_type_t::f32:
            return src == data_type_t::f32 && dst == data_type_t::f32;
        case data_type_t::bf16: return src == data_type_t::bf16 && f32_or_bf16;
        case data_type_t::f16:
            return src == data_type_t::f16
                    && (dst == data_type_t::f16 || dst == data_type_t::f32);
        case data_type_t::s8:
            return src == data_type_t::u8 || src == data_type_t::s8
                    || ((src == data_type_t::bf16 || src == data_type_t::f32)
                            && dst != data_type_t::s32
                            && dst != data_type_t::f16);
        case data_type_t::s4:
            return (src == data_type_t::bf16 || src == data_type_t::s8)
                    && f32_or_bf16;
        case data_type_t::u4: return src == data_type_t::bf16 && f32_or_bf16;
        default: return false;
    }
}

/// Why a call of ZENDNNL_MATMUL_ALGO=10 the native BRGEMM does not serve
/// keeps its error instead of taking the default route: for these the
/// default route returns success with a dst that is wrong or not written,
/// or it does not implement the activation.  Null for any other call.
const char *kept_from_default_route(char layout, int M, int N, float alpha,
        bool is_weights_const, matmul_gated_act_t gated_act,
        const matmul_params &params) {
    if (gated_act == matmul_gated_act_t::gelu_tanh_and_mul)
        return "the default route does not implement gelu_tanh_and_mul";
    if (params.mem_format_a != 'n')
        return "the default route does not compute a call with a reordered "
               "source correctly";
    const data_type_t src = params.dtypes.src;
    const data_type_t wei = params.dtypes.wei;
    const auto &qp = params.quant_params;
    const bool int_wei = wei == data_type_t::s8 || wei == data_type_t::s4
            || wei == data_type_t::u4;
    const bool int_src = src == data_type_t::u8 || src == data_type_t::s8;
    if (layout != 'r' && layout != 'R'
            && (is_weights_const || int_wei || int_src))
        return "the default route does not compute a column-major call on a "
               "constant weight or with integer operands correctly";
    const data_type_t compute = params.dtypes.compute;
    // The default route quantizes a bf16 or f32 source only.
    const bool quantizes = params.dynamic_quant
            && (src == data_type_t::bf16 || src == data_type_t::f32)
            && (compute == data_type_t::s8 || compute == data_type_t::u8);
    const data_type_t dst = params.dtypes.dst;
    if (!int_src && !quantizes) {
        if (wei == data_type_t::s8)
            return "the default route does not compute W8A16 (an s8 weight "
                   "with a source it does not quantize) correctly";
        if ((wei == data_type_t::s4 || wei == data_type_t::u4)
                && qp.wei_scale.buff != nullptr
                && (qp.wei_scale.dt == data_type_t::f16
                        || has_per_group_wei_scale(qp, N)))
            return "the default route does not compute W4A16 with "
                   "per-group or f16 weight scales correctly";
        if (wei == data_type_t::s4 && qp.wei_zp.buff != nullptr)
            return "the default route does not compute W4A16 with a zero "
                   "point of its s4 weight correctly";
        if (dst == data_type_t::s32 || dst == data_type_t::s8
                || dst == data_type_t::u8)
            return "the default route does not compute a call of float "
                   "operands into an integer dst correctly";
    } else if (wei == data_type_t::s8 || wei == data_type_t::s4) {
        if (quantizes && qp.src_zp.buff != nullptr
                && compute == data_type_t::s8)
            return "the default route does not compute a source quantized to "
                   "s8 with a source zero point correctly";
        if (quantizes && compute == data_type_t::u8
                && quant_param_tensor_or_row_values(qp.src_zp.dims, M) > 1)
            return "the default route does not compute a source quantized to "
                   "u8 with a zero point per row correctly";
        if (wei == data_type_t::s8 && qp.wei_zp.buff != nullptr
                && (quantizes
                        || !quant_param_dims_are_per_tensor(qp.wei_zp.dims)))
            return "the default route does not compute a weight zero point "
                   "of more than one value, or over a source it quantizes, "
                   "correctly";
        if (wei == data_type_t::s8 && alpha != 1.0f
                && (qp.wei_zp.buff != nullptr || qp.src_zp.buff != nullptr
                        || (quantizes && compute == data_type_t::u8)))
            return "the default route does not compute zero points with "
                   "alpha correctly";
        const data_type_t ws = qp.wei_scale.buff != nullptr ? qp.wei_scale.dt
                                                            : data_type_t::none;
        const data_type_t ss = qp.src_scale.buff != nullptr || quantizes
                ? qp.src_scale.dt
                : data_type_t::none;
        if ((wei == data_type_t::s8
                    && (ws == data_type_t::f16 || ss == data_type_t::f16))
                || (ws == data_type_t::f32 && ss == data_type_t::bf16)
                || (wei == data_type_t::s4 && ws == data_type_t::bf16
                        && ss == data_type_t::f32))
            return "the default route does not compute these weight and "
                   "source scale types correctly";
    }
    // The default route quantizes the source of an s8 / s4 weight only.
    const bool quantized
            = quantizes && (wei == data_type_t::s8 || wei == data_type_t::s4);
    if (!default_route_has_kernel(quantized ? compute : src, wei, dst))
        return "the default route does not compute a call of these source, "
               "weight and dst types";
    return nullptr;
}

/// kept_from_default_route, and in a build without AOCL-DLP what its default
/// route, the reference kernel, does not compute: it reads a row-major raw
/// weight, per-group weight scales of an s8 weight with source scales of
/// the same groups only, a binary operand as {1, N} or as rows of N values,
/// and no pooling.
const char *kept_in_this_build(char layout, int M, int N, float alpha,
        bool is_weights_const, matmul_gated_act_t gated_act,
        const matmul_params &params) {
    const char *why = kept_from_default_route(
            layout, M, N, alpha, is_weights_const, gated_act, params);
#if !ZENDNNL_DEPENDS_AOCLDLP
    if (why != nullptr) return why;
    if (layout != 'r' && layout != 'R')
        return "the default route of a build without AOCL-DLP does not "
               "compute a column-major call";
    if (params.mem_format_b != 'n' || params.packing.pack_format_b != 0)
        return "the default route of a build without AOCL-DLP does not "
               "compute a call on a reordered or GGML packed weight";
    const auto &qp = params.quant_params;
    if (params.dtypes.wei == data_type_t::s8 && qp.wei_scale.buff != nullptr) {
        const int64_t w = checked_quant_param_num_elements(qp.wei_scale.dims);
        const int64_t s = qp.src_scale.buff != nullptr || params.dynamic_quant
                ? checked_quant_param_num_elements(qp.src_scale.dims)
                : 0;
        if (w > N && w % N == 0 && !(s > M && s % M == 0 && s / M == w / N))
            return "the default route of a build without AOCL-DLP does not "
                   "compute per-group weight scales without source scales of "
                   "the same groups";
    }
    for (const matmul_post_op &po : params.postop_)
        if (po.po_type == post_op_type_t::pooling
                || ((po.po_type == post_op_type_t::binary_add
                            || po.po_type == post_op_type_t::binary_mul)
                        && !(po.dims.size() == 2 && po.dims[0] == 1)
                        && po.leading_dim != N))
            return "the default route of a build without AOCL-DLP does not "
                   "compute a pooling post-op or a binary operand whose rows "
                   "are not N values apart";
#endif
    return why;
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

/// The bytes at weight a call of layout vouches for as its weight buffer:
/// what it reads from a bf16 / s8 / s4 / f32 weight (an AOCL prepack this
/// route unreorders, or the span of a plain weight as the caller lays it
/// out), or the capacity it states when larger; 0 when neither is known.
ZENDNNL_ALWAYS_INLINE inline size_t weight_buffer_bytes(char layout,
        const matmul_params &params, int K, int N, int ldb, bool transB) {
    size_t reads = 0;
    const data_type_t wdt = params.dtypes.wei;
    if ((wdt == data_type_t::bf16 || wdt == data_type_t::s8
                || wdt == data_type_t::s4 || wdt == data_type_t::f32)
            && K > 0 && N > 0 && ldb > 0 && params.packing.pack_format_b == 0) {
        // A column-major weight spans as the row-major one of the other
        // transB.
        const bool row_trans = transB != (layout != 'r' && layout != 'R');
        if (params.mem_format_b == 'r') {
            if (aocl_prepacked_weight(params)
                    && aocl_prepack_unreorderable(params))
                reads = aocl_prepack_bytes(
                        K, N, ldb, row_trans, wdt == data_type_t::s8);
        } else if (params.mem_format_b == 'n') {
            reads = native_raw_weight_bytes(wdt, K, N, ldb, row_trans);
        }
    }
    return std::max(reads, params.wei_buffer_capacity_bytes);
}

/// The bytes at weight the default route reads of a call of layout handed
/// to it: the blocks of a GGML pack, the aocl_dlp_blocked prepack of its
/// weight's format, or the raw elements of a weight of any type as the
/// caller lays them out (two u4 elements a byte, as for s4); at least
/// weight_buffer_bytes.
ZENDNNL_NOINLINE size_t handed_back_weight_bytes(char layout,
        const matmul_params &params, int K, int N, int ldb, bool transB) {
    const size_t owned = weight_buffer_bytes(layout, params, K, N, ldb, transB);
    if (K < 1 || N < 1 || ldb < 1) return owned;
    const data_type_t wdt = params.dtypes.wei;
    const bool row_trans = transB != (layout != 'r' && layout != 'R');
    size_t reads = 0;
    if (params.packing.pack_format_b == 1) {
        // Each row of N holds a block per 32 values of K: an f16 scale and
        // 32 s8 values (Q8_0) or 16 bytes of s4 values (Q4_0, an s4 weight).
        reads = static_cast<size_t>(N) * ceil_div(K, 32)
                * (wdt == data_type_t::s4 ? 18 : 34);
    } else if (params.mem_format_b == 'r') {
        // The default route reads the column sums after the prepack of a u8
        // source at any weight scale groups.
        const native_layout_query_t q
                = native_layout_query_for(params, K, N, ldb, row_trans);
        reads = aocl_prepack_bytes(q.src_dt, wdt,
                q.src_dt == data_type_t::u8 ? 0 : std::max(0, q.wei_group_size),
                K, N, ldb, row_trans);
    }
    if (reads == 0)
        reads = native_raw_weight_bytes(
                wdt == data_type_t::u4 ? data_type_t::s4 : wdt, K, N, ldb,
                row_trans);
    return std::max(reads, owned);
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
    const bool process_native = matmul_config_t::is_native_brgemm_algo();
    // A buffer rewritten in place must have no other reader, which only the
    // process-wide ALGO 10 guarantees (every call then reaches the image
    // check of classify_source); a call that selects the native BRGEMM by
    // itself caches the layout out of place instead.
    const auto weight_cache_mode = [&]() -> int32_t {
        if (!is_weights_const) return 0;
        const int32_t mode
                = effective_weight_cache_type(params.weight_cache_type);
        return mode == 2 && !process_native ? 1 : mode;
    };
    // An M = 1 call on its raw weight may run on a GEMV kernel instead of
    // the tiles, except at weight cache mode 2, where the layout of the
    // tiles is the weight's only one and serves its M = 1 calls too.
    native_gemv_plan_t gemv;
    if (M == 1 && batch_count == 1 && params.mem_format_b == 'n' && requested
            && weight_cache_mode() != 2)
        gemv = decide_native_gemv(call_desc(), params);
    // A call that ZENDNNL_MATMUL_ALGO=10 alone brings here goes to the
    // default route when the native BRGEMM does not implement what it
    // carries, unless its shape or ldb describe no matmul or the default
    // route does not compute it correctly (kept says why); a call with
    // lowoha_algo native_brgemm gets the error.
    const char *kept = nullptr;
    const bool row_major = layout == 'r' || layout == 'R';
    // ldb spans a weight row as the caller lays the weight out: K elements
    // for a row-major [N, K] or a column-major [K, N], else N.
    const auto row_spanned
            = [&]() { return ldb >= (transB == row_major ? K : N); };
    const auto hands_back = [&]() {
        if (!requested || params.lowoha_algo != matmul_algo_t::none || N < 1
                || K < 1 || !row_spanned())
            return false;
        // lda and ldc span a row of the source and of dst as the caller
        // lays them out: a call refused before the native BRGEMM's operand
        // checks would reach a default route whose checks do not see them.
        const int dst_cols = gated_act != matmul_gated_act_t::none ? N / 2 : N;
        if (lda < (transA == row_major ? M : K)
                || ldc < (row_major ? dst_cols : M)) {
            reason = "a bad shape or leading dimension";
            status = status_t::failure;
            return false;
        }
        kept = kept_in_this_build(
                layout, M, N, alpha, is_weights_const, gated_act, params);
        return kept == nullptr;
    };
    const auto out_of_scope = [](status_t st) {
        return st == status_t::unimplemented || st == status_t::isa_unsupported;
    };
    int pack_nr = 0;
    reason = nullptr;
    status = status_t::success;
    const size_t owned = weight_buffer_bytes(layout, params, K, N, ldb, transB);
    native_weight_ref_t ref;
    const native_source_t source = classify_source(params, weight, K, N, ldb,
            transB, batch_count, requested, process_native,
            gemv.kernel != native_gemv_t::none, owned, pack_nr, reason, status,
            ref.image);
    if (source == native_source_t::none) {
        // ZENDNNL_MATMUL_ALGO=10 does not select the kernel of a batched call.
        if (!CheckOnly && process_native
                && params.lowoha_algo == matmul_algo_t::none)
            note_default_route("a batched call");
        return false;
    }
    bool in_place = source == native_source_t::in_place;
    // An image holds the layout, not the raw weight the GEMV kernels read.
    if (in_place) gemv = native_gemv_plan_t {};
    status_t refusal = in_place ? status_t::failure : status_t::unimplemented;
    // The layout rule bounds ldb for a row-major weight: a column-major call
    // it refuses, of an ldb that spans a row of the caller's layout, is
    // refused as column-major.
    if (!row_major
            && (reason == nullptr
                    || (source == native_source_t::refused
                            && status == status_t::unimplemented
                            && row_spanned()))) {
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
    ref.weight_cache_mode = weight_cache_mode();
    ref.capacity_bytes = params.wei_buffer_capacity_bytes;

    // A concurrent first call may have rewritten the buffer since it was
    // classified; then no other route may read it either.  A call that does
    // not ask for the native BRGEMM (an AOCL prepack under
    // ZENDNNL_MATMUL_ALGO=10 that names its backend) keeps its own route
    // when the native BRGEMM does not serve it.
    const auto image_now = [&]() {
        return in_place
                || native_weight_packed_in_place(weight,
                        handed_back_weight_bytes(
                                layout, params, K, N, ldb, transB));
    };
    if (reason != nullptr
            && (!requested || (out_of_scope(status) && hands_back()))
            && !image_now()) {
        if (requested && !CheckOnly) note_default_route(reason);
        return false;
    }
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
            reason = why != nullptr ? why : kernel_status_text(status);
            if (out_of_scope(status) && !image
                    && (!requested || hands_back())) {
                if (requested && !CheckOnly) note_default_route(reason);
                return false;
            }
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
        if (in_place) kept = nullptr;
        apilog_error("matmul_direct: native BRGEMM did not serve the call (",
                reason, ")",
                in_place ? "; no other route reads a buffer it rewrote"
                        : params.lowoha_algo == matmul_algo_t::native_brgemm
                        ? "; lowoha_algo native_brgemm takes no other route"
                        : kept != nullptr ? "; not handed back: "
                                          : "",
                kept != nullptr ? kept : "");
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

status_t native_hold_handed_back_weight(native_weight_raw_hold_t &hold,
        char layout, const void *weight, int K, int N, int ldb, bool transB,
        const matmul_params &params, const matmul_batch_params_t *batch) {
    // Whatever the process's weight cache mode is now: an image written in
    // place, and a native call that took mode 2, outlive a downgrade of it.
    size_t reads = handed_back_weight_bytes(layout, params, K, N, ldb, transB);
    if (batch != nullptr && batch->Batch_B > 1) {
        // Matrix b of the batch at b times the stride the batched kernels
        // take (bmm_execute).
        const size_t stride = batch->batch_stride_wei != static_cast<size_t>(-1)
                ? batch->batch_stride_wei
                : static_cast<size_t>(transB ? N : K) * ldb;
        reads += static_cast<size_t>(batch->Batch_B - 1) * stride
                * zendnnl::common::size_of(params.dtypes.wei);
    }
    switch (hold.hold(weight, std::max<size_t>(reads, 1), reads)) {
        case native_raw_read_t::done: return status_t::success;
        case native_raw_read_t::image:
            apilog_error(
                    "matmul_direct: the weight of a call handed to the default "
                    "route holds an image the native BRGEMM rewrote in place; "
                    "no other route reads a buffer the native BRGEMM rewrote");
            return status_t::failure;
        default:
            apilog_error(
                    "matmul_direct: no memory to hold the weight of a call "
                    "handed to the default route");
            return status_t::failure;
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
