/*******************************************************************************
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

#include "common/zendnnl_compat.hpp"
#include "ggml_weight_unpack.hpp"
#include "lowoha_matmul_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"
#include "lowoha_operators/matmul/aocl_unreorder_cache.hpp"
#include "lowoha_operators/matmul/auto_tuner/auto_tuner.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#if ZENDNNL_DEPENDS_AOCLDLP
#include "lowoha_operators/matmul/backends/aocl/aocl_postop.hpp"
#endif
#include "lowoha_operators/matmul/backends/libxsmm/libxsmm_kernel.hpp"
#include "lowoha_operators/matmul/backends/onednn/onednn_kernel.hpp"
#include "lowoha_operators/matmul/backends/reference/reference_kernel.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/group_matmul_n_tile.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack.hpp"
#include "lowoha_operators/matmul/group_matmul/single_expert/single_expert_route.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"
#include "matmul_native/native_matmul.hpp"
#include "partitioning/bmm/looper/bmm_looper.hpp"
#include "partitioning/matmul/matmul_partitioner.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::size_of;

void matmul_kernel_wrapper(char layout, char transA, char transB, int M, int N,
        int K, float alpha, const void *A, int lda, const void *B, int ldb,
        float beta, void *C, int ldc, matmul_data_types &dtypes,
        zendnnl::common::matmul_algo_t &kernel, char mem_format_a,
        char mem_format_b, matmul_params &lowoha_param,
        matmul_batch_params_t &batch_params, const void *bias,
        bool is_weights_const, int num_threads) {
#if ZENDNNL_DEPENDS_LIBXSMM
    if (kernel == matmul_algo_t::libxsmm) {
        log_info("Using libxsmm kernel");
        if (run_libxsmm_std(transA, transB, M, N, K, beta, lda, ldb, ldc, A, B,
                    C, dtypes, lowoha_param, bias)) {
            return;
        }
    }
#endif
#if ZENDNNL_DEPENDS_ONEDNN
    if (kernel == matmul_algo_t::onednn
            || kernel == matmul_algo_t::onednn_blocked) {
        log_info("Using onednn kernel");
        matmul_onednn_wrapper(transA, transB, M, N, K, alpha, A, lda, B, ldb,
                beta, C, ldc, lowoha_param, batch_params, bias, kernel,
                is_weights_const);
        return;
    }
#endif

#if !ZENDNNL_DEPENDS_AOCLDLP
    log_info("Using reference kernel (AOCL-DLP fallback unavailable, was ",
            kernel_to_string(kernel), ")");
    kernel = matmul_algo_t::reference;
    reference_matmul_execute(layout, transA == 't', transB == 't', M, N, K,
            alpha, A, lda, B, ldb, bias, beta, C, ldc, is_weights_const,
            batch_params, lowoha_param);
    return;
#else
    log_info("Using AOCL DLP kernel");
    run_dlp(layout, transA, transB, M, N, K, alpha, beta, lda, ldb, ldc,
            mem_format_a, mem_format_b, A, B, C, dtypes, lowoha_param, bias,
            kernel, is_weights_const);
    return;
#endif
}

void matmul_execute(const char layout, const bool transA, const bool transB,
        const int M, const int N, const int K, const float alpha,
        const void *src, const int lda, const void *weight, const int ldb,
        const void *bias, const float beta, void *dst, const int ldc,
        const bool is_weights_const, const size_t src_type_size,
        const size_t out_type_size, const int num_threads,
        matmul_algo_t &kernel, matmul_params &params,
        matmul_batch_params_t &batch_params, unsigned int auto_version) {

    const char trans_input = transA ? 't' : 'n';
    const char trans_weight = transB ? 't' : 'n';

    // Auto-tuner kernel selection for single batch.
    if (kernel == matmul_algo_t::auto_tuner) {
        if (auto_version == 1) {
            kernel = auto_compute_matmul_v1(layout, trans_input, trans_weight,
                    M, N, K, alpha, src, lda, weight, ldb, beta, dst, ldc,
                    params.dtypes, kernel, params.mem_format_a,
                    params.mem_format_b, params, batch_params, bias,
                    is_weights_const, num_threads);
        } else {
            kernel = auto_compute_matmul_v2(layout, trans_input, trans_weight,
                    M, N, K, alpha, src, lda, weight, ldb, beta, dst, ldc,
                    params.dtypes, kernel, params.mem_format_a,
                    params.mem_format_b, params, batch_params, bias,
                    is_weights_const, num_threads);
        }
        return;
    }

    // Currently supported only for LIBXSMM BACKEND
    if (should_use_mm_partitioner(kernel)) {
        // Setup partition configuration
        matmul_partition_config_t part_config;
        part_config.M = M;
        part_config.N = N;
        part_config.K = K;
        part_config.num_threads = num_threads;
        part_config.kernel = kernel;
        part_config.src_type_size = src_type_size;
        part_config.out_type_size = out_type_size;
        part_config.lda = lda;
        part_config.ldb = ldb;
        part_config.ldc = ldc;
        part_config.transA = transA;
        part_config.transB = transB;
        part_config.dtypes = params.dtypes;

        execute_partitioned_matmul(layout, trans_input, trans_weight, src,
                weight, dst, bias, part_config, params, batch_params,
                is_weights_const, alpha, beta);
        kernel = part_config.kernel;
        return;
    }

#if ZENDNNL_DEPENDS_ONEDNN
    if (kernel == matmul_algo_t::onednn
            || kernel == matmul_algo_t::onednn_blocked) {
        apilog_info("Executing matmul LOWOHA kernel with oneDNN, algo: ",
                static_cast<int>(kernel));

        matmul_onednn_wrapper(trans_input, trans_weight, M, N, K, alpha, src,
                lda, weight, ldb, beta, dst, ldc, params, batch_params, bias,
                kernel, is_weights_const);
        return;
    }

#endif

    apilog_info(
            "Executing matmul LOWOHA kernel without zendnnl-partitioner, "
            "algo: ",
            static_cast<int>(kernel));
    matmul_kernel_wrapper(layout, trans_input, trans_weight, M, N, K, alpha,
            src, lda, weight, ldb, beta, dst, ldc, params.dtypes, kernel,
            params.mem_format_a, params.mem_format_b, params, batch_params,
            bias, is_weights_const, num_threads);
}

// The kernel runs a call off AOCL-DLP: the reference kernel, or oneDNN in a
// build that has it (every other kernel passes the calls it does not serve
// to AOCL-DLP).
static constexpr bool runs_off_aocl_dlp(matmul_algo_t kernel) noexcept {
    return kernel == matmul_algo_t::reference
            || (ZENDNNL_DEPENDS_ONEDNN
                    && (kernel == matmul_algo_t::onednn
                            || kernel == matmul_algo_t::onednn_blocked));
}

// What the reference kernel does not serve of a call it takes for a post-op
// the AOCL-DLP chain does not take, or nullptr: it reads row-major operands
// as given (not a weight AOCL-DLP prepacked, nor the GGML unpack it rewrites
// in place), applies per-group weight scales of an integer weight per group
// only with a source scale of the same groups, needs a source scale for a
// bf16 / f32 source with s8 weights, and reads a binary operand as one value
// per column ({1, N}) or as rows of N values.  caller is the call's params as
// given, params the call's copy.
static const char *reference_declines(char layout, int M, int N,
        const matmul_params &params, const matmul_params &caller) {
    if (layout != 'r' && layout != 'R') { return "a column-major call"; }
    if (params.mem_format_b == 'r') { return "AOCL-prepacked weights"; }
    if (caller.packing.pack_format_b == 1) { return "GGML packed weights"; }
    const auto &qp = params.quant_params;
    const data_type_t src = params.dtypes.src;
    const data_type_t wei = params.dtypes.wei;
    if (wei == data_type_t::s8
            && (src == data_type_t::bf16 || src == data_type_t::f32)
            && qp.src_scale.buff == nullptr) {
        return "a bf16 / f32 source with s8 weights and no source scale";
    }
    const auto count = [](const std::vector<int64_t> &dims) {
        int64_t n = 1;
        for (const int64_t d : dims)
            n *= d;
        return n;
    };
    const bool int_weight = wei == data_type_t::s8
            || (wei == data_type_t::s4 && src != data_type_t::bf16
                    && !is_w4a8_config(params));
    if (int_weight && qp.wei_scale.buff != nullptr) {
        const int64_t wei_count = count(qp.wei_scale.dims);
        const int64_t src_count
                = qp.src_scale.buff != nullptr ? count(qp.src_scale.dims) : 0;
        if (wei_count > N && wei_count % N == 0
                && !(src_count > M && src_count % M == 0
                        && src_count / M == wei_count / N)) {
            return "per-group weight scales without a source scale of the "
                   "same groups";
        }
    }
    for (const auto &po : params.postop_) {
        if ((po.po_type == post_op_type_t::binary_add
                    || po.po_type == post_op_type_t::binary_mul)
                && !(po.dims.size() == 2 && po.dims[0] == 1)
                && po.leading_dim != N) {
            return "a binary operand whose rows are not N values apart";
        }
    }
    return nullptr;
}

// A gated activation with a post-op the AOCL-DLP chain does not take
// (pooling among them), which has no defined place against the activation:
// logged, the call fails before a kernel writes dst.
static bool aocl_dlp_refuses_gated_post_ops(bool unmapped_post_op) {
#if ZENDNNL_DEPENDS_AOCLDLP
    if (!unmapped_post_op) { return false; }
    log_error(
            "matmul_direct: a gated activation takes only post-ops the "
            "AOCL-DLP chain applies, not elu, square, abs, sqrt, exp, log, "
            "softmax or pooling");
    return true;
#else
    (void)unmapped_post_op;
    return false;
#endif
}

// A weight scale AOCL-DLP cannot apply (aocl_dlp_applies_wei_scale) on a call
// that reaches it: logged when report, the call fails before a kernel writes
// dst.
static bool aocl_dlp_refuses_wei_scale(const matmul_params &params,
        data_type_t src, int M, int N, int K, bool report = true) {
#if ZENDNNL_DEPENDS_AOCLDLP
    if (aocl_dlp_applies_wei_scale(params, src, M, N, K)) { return false; }
    if (report)
        log_error(
                "matmul_direct: AOCL-DLP has no kernel for a weight scale of ",
                get_num_elements(params.quant_params.wei_scale.dims),
                " values with a ", dtype_info(src), " source and a ",
                dtype_info(params.dtypes.dst), " destination");
    return true;
#else
    (void)report;
    return false;
#endif
}

// A weight-scale type AOCL-DLP does not read (aocl_dlp_applies_wei_scale_type)
// on a call that reaches it: logged when report, the call fails before a
// kernel writes dst.
static bool aocl_dlp_refuses_wei_scale_type(const matmul_params &params,
        data_type_t src, int M, int N, bool report) {
#if ZENDNNL_DEPENDS_AOCLDLP
    if (aocl_dlp_applies_wei_scale_type(params, src, M, N)) { return false; }
    if (report)
        log_error(
                "matmul_direct: AOCL-DLP's sym-quant kernel reads f32 weight "
                "scales or ones of the source scale's type, not ",
                dtype_info(params.quant_params.wei_scale.dt), " with ",
                dtype_info(params.quant_params.src_scale.dt), " source scales");
    return true;
#else
    (void)params;
    (void)src;
    (void)M;
    (void)N;
    (void)report;
    return false;
#endif
}

// An alpha or beta AOCL-DLP does not take (aocl_dlp_takes_alpha_beta) or,
// with applied, does not apply (aocl_dlp_applies_alpha_beta) on a call that
// reaches it: logged when report, the call fails before a kernel writes dst.
static bool aocl_dlp_refuses_alpha_beta(const matmul_params &params,
        data_type_t src, int M, int N, float alpha, float beta, bool applied,
        bool report = true) {
#if ZENDNNL_DEPENDS_AOCLDLP
    if (applied ? aocl_dlp_applies_alpha_beta(params, src, M, N, alpha, beta)
                : aocl_dlp_takes_alpha_beta(params, src, alpha, beta)) {
        return false;
    }
    if (report)
        log_error("matmul_direct: AOCL-DLP has no kernel for alpha=", alpha,
                ", beta=", beta, " with a ", dtype_info(src), " source, a ",
                dtype_info(params.dtypes.wei), " weight and a ",
                dtype_info(params.dtypes.dst), " destination");
    return true;
#else
    (void)params;
    (void)src;
    (void)M;
    (void)N;
    (void)alpha;
    (void)beta;
    (void)applied;
    (void)report;
    return false;
#endif
}

static bool to_group_gated_act(
        matmul_gated_act_t act, grp_matmul_gated_act_t &group_act) noexcept {
    switch (act) {
        case matmul_gated_act_t::none:
            group_act = grp_matmul_gated_act_t::none;
            return true;
        case matmul_gated_act_t::silu_and_mul:
            group_act = grp_matmul_gated_act_t::silu_and_mul;
            return true;
        case matmul_gated_act_t::gelu_and_mul:
            group_act = grp_matmul_gated_act_t::gelu_and_mul;
            return true;
        case matmul_gated_act_t::swiglu_oai_mul:
            group_act = grp_matmul_gated_act_t::swiglu_oai_mul;
            return true;
    }
    return false;
}

namespace {

/// The working state of a matmul_direct call after its input checks.
struct direct_call_t {
    explicit direct_call_t(const matmul_params &p) : params(p) {}
    /// The call's copy of params: reorder, kernel_select, and post-op
    /// normalization mutate effective execution state only; caller params
    /// are left unchanged so the same struct can be reused across
    /// iterations.
    matmul_params params;
    grp_matmul_gated_act_params group_gated_act;
    bool has_gated_act = false; ///< a gated_act argument was given
    bool run_gated_act = false; ///< and it is an activation
    int batch_count = 1;
    /// A post-op the AOCL-DLP chain does not take (aocl_dlp_maps_post_op).
    bool aocl_unmapped_post_op = false;
    bool pooling_post_op = false; ///< and one of them is pooling
};

/// matmul_direct's checks of a call and its normalization of call.params,
/// the call's copy of params: the status it returns before any route runs,
/// success when the call goes on.  Inlined into its callers: matmul_direct
/// runs it on every call, whatever route the call takes.
ZENDNNL_ALWAYS_INLINE inline status_t prepare_direct_call(const int M,
        const int N, const int K, const void *src, const void *weight,
        void *dst, const bool is_weights_const,
        const matmul_batch_params_t &batch_params, const matmul_params &params,
        const matmul_gated_act_params *gated_act, direct_call_t &call) {
    // F16 ISA check must always run — not gated behind diagnostics.
    // Prevents undefined behavior on platforms without AVX512-FP16 support.
    // Covers every F16-bearing operand: src / wei / dst, bias, and any binary
    // post-op buffer. Anything else slips through to a kernel that touches
    // F16 storage and produces wrong results on non-FP16 hardware.
    const bool is_f16 = (params.dtypes.src == data_type_t::f16
            || params.dtypes.wei == data_type_t::f16
            || params.dtypes.dst == data_type_t::f16
            || params.dtypes.bias == data_type_t::f16);
    if (is_f16 && !zendnnl_platform_info().get_avx512_f16_status()) {
        log_error(
                "F16 data type is not supported on this platform "
                "(requires AVX512-FP16).");
        return status_t::isa_unsupported;
    }

    // Validate inputs unless ZENDNNL_DIAGNOSTICS_ENABLE=0 is set (the gate
    // defaults to enabled). In production hot paths where the variable is
    // explicitly disabled, this resolves to a single predicted-taken branch,
    // skipping the full validation path (null-pointer checks, dimension
    // checks, and quantization-parameter validation).
    // Must judge params, not the call's copy: diagnostics report the
    // caller-supplied configuration, not the library's internal state.
    status_t status = zendnnl::common::op_instrumentation::validate([&]() {
        return validate_matmul_direct_inputs(src, weight, dst, M, N, K,
                batch_params.Batch_A, batch_params.Batch_B, params,
                is_weights_const);
    });
    if (status != status_t::success) { return status; }

    // Defaults N for binary post-ops when leading_dim is -1. Mutates the
    // call's copy only. Must always execute regardless of
    // ZENDNNL_DIAGNOSTICS_ENABLE.
    // Notes the post-ops the AOCL-DLP chain does not take.
    for (auto &po : call.params.postop_) {
        if (!aocl_dlp_maps_post_op(po.po_type)) {
            call.aocl_unmapped_post_op = true;
            call.pooling_post_op |= po.po_type == post_op_type_t::pooling;
        }
        // Materialize operation-specific defaults on the per-call copy.
        // In particular, an omitted Swish/ELU alpha becomes 1 while an
        // explicitly supplied zero remains zero.
        po.apply_alpha_default();
        if (po.po_type == post_op_type_t::binary_add
                || po.po_type == post_op_type_t::binary_mul) {
            if (po.leading_dim == -1) { po.leading_dim = N; }
            if (po.dtype == data_type_t::f16
                    && !zendnnl_platform_info().get_avx512_f16_status()) {
                log_error(
                        "F16 binary post-op tensor is not supported on this "
                        "platform "
                        "(requires AVX512-FP16).");
                return status_t::isa_unsupported;
            }
        } else if (po.po_type == post_op_type_t::clip) {
            // matmul_post_op encodes clip bounds as (alpha = lower, beta = upper).
            if (po.alpha > po.beta) {
                log_info("Clip post-op: alpha (", po.alpha, ") > beta (",
                        po.beta, "), swapping to keep lower <= upper.");
                std::swap(po.alpha, po.beta);
            }
        }
    }

    call.batch_count = std::max(batch_params.Batch_A, batch_params.Batch_B);
    if (gated_act != nullptr) {
        if (!to_group_gated_act(gated_act->act, call.group_gated_act.act)) {
            log_error("matmul_direct: unsupported gated activation value ",
                    static_cast<int>(gated_act->act));
            return status_t::failure;
        }
        call.has_gated_act = true;
    }
    call.run_gated_act = call.has_gated_act
            && call.group_gated_act.act != grp_matmul_gated_act_t::none;
    if (call.run_gated_act && call.batch_count != 1) {
        log_error(
                "matmul_direct: gated activation is supported only for a "
                "single batch");
        return status_t::failure;
    }
    return status_t::success;
}

/// Points q at held, a copy of the elements its dims name; a null buffer,
/// or dims that name no element, stay as they are.
void point_at_copy(matmul_quantization_params_t::matmul_quant_t &q,
        std::vector<uint8_t> &held) {
    const int64_t n = checked_quant_param_num_elements(q.dims);
    const size_t bytes = n > 0 ? static_cast<size_t>(n) * size_of(q.dt) : 0;
    if (q.buff == nullptr || bytes == 0) return;
    const uint8_t *from = static_cast<const uint8_t *>(q.buff);
    held.assign(from, from + bytes);
    q.buff = held.data();
}

} // namespace

status_t native_brgemm_refusal(const char layout, const bool transA,
        const bool transB, const int M, const int N, const int K,
        const float alpha, const void *src, const int lda, const void *weight,
        const int ldb, const void *bias, const float beta, void *dst,
        const int ldc, const bool is_weights_const,
        const matmul_batch_params_t &batch_params, const matmul_params &params,
        const matmul_gated_act_params *gated_act, const char **reason) {
    direct_call_t call(params);
    if (prepare_direct_call(M, N, K, src, weight, dst, is_weights_const,
                batch_params, params, gated_act, call)
                    != status_t::success
            || !native::native_brgemm_selected(call.params))
        return status_t::success;
    // A dynamic quantization writes the scales and zero points it makes
    // into the buffers of the call's params: the check writes into copies.
    std::vector<uint8_t> scale_copy, zp_copy;
    if (call.params.dynamic_quant) {
        point_at_copy(call.params.quant_params.src_scale, scale_copy);
        point_at_copy(call.params.quant_params.src_zp, zp_copy);
    }
    const int32_t num_threads = resolve_num_threads(
            call.params.num_threads, thread_guard::max_threads());
    status_t status = status_t::success;
    const char *why = nullptr;
    if (!native::native_layout_check(layout, transA, transB, M, N, K, alpha,
                src, lda, weight, ldb, bias, beta, dst, ldc, is_weights_const,
                call.batch_count, num_threads,
                call.run_gated_act ? gated_act->act : matmul_gated_act_t::none,
                call.params, status, why))
        return status_t::success;
    if (reason != nullptr) *reason = why;
    return status;
}

status_t matmul_direct(const char layout, const bool transA, const bool transB,
        const int M, const int N, const int K, const float alpha,
        const void *src, const int lda, const void *weight, const int ldb,
        const void *bias, const float beta, void *dst, const int ldc,
        const bool is_weights_const, const matmul_batch_params_t &batch_params,
        const matmul_params &params, const matmul_gated_act_params *gated_act) {
    // Profiler overhead in production (ZENDNNL_ENABLE_PROFILER unset):
    //  - profiler_t constructor: eliminated by dead store elimination at -O3
    //    when profiler is never used.
    //  - is_profile_enabled(): negligible (cached static const bool, shared
    //    across all translation units via inline linkage).
    //  - if (is_profile) branch: negligible (always false, branch predictor
    //    learns not-taken quickly).
    profiler_t profiler;
    bool is_profile = is_profile_enabled();
    if (is_profile) { profiler.tbp_start(); }

    direct_call_t call(params);
    const status_t checked = prepare_direct_call(M, N, K, src, weight, dst,
            is_weights_const, batch_params, params, gated_act, call);
    if (checked != status_t::success) { return checked; }
    matmul_params &exec_params = call.params;
    matmul_batch_params_t exec_batch_params = batch_params;

    size_t src_type_size = size_of(exec_params.dtypes.src);
    const size_t weight_type_size = size_of(exec_params.dtypes.wei);
    const size_t out_type_size = size_of(exec_params.dtypes.dst);

    const int batch_count = call.batch_count;
    const grp_matmul_gated_act_params &group_gated_act = call.group_gated_act;
    const grp_matmul_gated_act_params *group_gated_act_ptr
            = call.has_gated_act ? &call.group_gated_act : nullptr;
    const bool run_gated_act = call.run_gated_act;
    const int32_t omp_mt = thread_guard::max_threads();
    const int32_t num_threads
            = resolve_num_threads(exec_params.num_threads, omp_mt);
    // The native BRGEMM takes every call that asks for it: it serves it or
    // returns the call's error; the others continue below as without it.
    if (native::native_brgemm_selected(exec_params)) {
        status_t native_status = status_t::success;
        if (native::native_layout_matmul(layout, transA, transB, M, N, K, alpha,
                    src, lda, weight, ldb, bias, beta, dst, ldc,
                    is_weights_const, batch_count, num_threads,
                    run_gated_act ? gated_act->act : matmul_gated_act_t::none,
                    exec_params, is_profile ? &profiler : nullptr,
                    native_status, &exec_batch_params)) {
            return native_status;
        }
    }

    // Unpack GGML before the group route. The route rejects pack_format_b==1,
    // so a still-packed weight can never enter group_matmul_direct. After
    // unpack the weight is raw per-group s8 in a buffer sized for the AOCL
    // blocked image, and the same route that accepts a caller-provided s8
    // weight can hand that buffer to group_matmul_direct.
    const void *exec_weight = weight;
    if (exec_params.packing.pack_format_b == 1) {
        status_t ggml_val_status
                = zendnnl::common::op_instrumentation::validate([&]() {
            return validate_ggml_packed_inputs(exec_params, is_weights_const,
                    exec_batch_params.Batch_B, transB);
        });
        if (ggml_val_status != status_t::success) { return ggml_val_status; }
        // Unpack reads only the packed weight. The GEMM contract is checked
        // here, before that unpack: an f32/bf16 activation with dynamic_quant
        // is accepted, and source quantization runs later. A scalar source
        // scale is not a sym-quant per-group matmul.
        if (!ggml_packed_gemm_supported(exec_params)) {
            log_error(
                    "GGML packed weights require sym-quant per-group int8: "
                    "an s8 source, or an f32/bf16 source with dynamic_quant, "
                    "and a non-scalar source scale");
            return status_t::failure;
        }
        status_t unpack_status = unpack_ggml_weights_and_cache(exec_weight, N,
                K, ldb, transB ? 't' : 'n', exec_params, is_weights_const);
        if (unpack_status != status_t::success) { return unpack_status; }
    }

    // AOCL records a reorder refusal against the buffer it was asked to
    // reorder. After unpack that is exec_weight, not the caller's packed
    // pointer. Snapshot here so a miss on the materialized buffer fails this
    // call instead of returning success or falling through to scalar.
    const uint64_t scalar_mutated_conflicts_before
            = aocl_mutated_source_conflict_count(exec_weight);
    auto scalar_mutation_conflict = [&]() {
        return aocl_mutated_source_conflict_count(exec_weight)
                != scalar_mutated_conflicts_before;
    };

    // Wide-N projections and gated activations run as a one-op group.  Wide-N
    // calls use this route to reach the single-expert custom kernel.  Gated
    // activation has no equivalent in the normal matmul_direct continuation,
    // so an active request always enters the grouped path and propagates its
    // status. Routed INT8 and W4A8 calls are quantized by
    // group_matmul_direct's own pre-pass. GGML weights have already been
    // unpacked above.
    if (run_gated_act
            || single_matmul_group_routable(layout, M, N, K, num_threads,
                    batch_count, transA, transB, alpha, beta, bias, ldb,
                    is_weights_const, exec_params)) {
        // The group route runs per-group weight scales, alpha and beta on
        // AOCL-DLP, over the source its pre-pass quantizes, and the post-ops
        // of a gated activation there.
        const data_type_t grp_src_dt = is_dynamic_quant_config(exec_params)
                ? exec_params.dtypes.compute
                : exec_params.dtypes.src;
        if (aocl_dlp_refuses_wei_scale(exec_params, grp_src_dt, M, N, K)
                || aocl_dlp_refuses_alpha_beta(
                        exec_params, grp_src_dt, M, N, alpha, beta, false)
                || (run_gated_act
                        && aocl_dlp_refuses_gated_post_ops(
                                call.aocl_unmapped_post_op))) {
            return status_t::unimplemented;
        }
        matmul_params grp_param = exec_params;
        grp_param.num_threads = num_threads;
        const bool aocl_prepacked = is_aocl_prepacked_weight(exec_params);
        const std::vector<char> grp_layout {layout};
        const std::vector<bool> grp_trans_a {transA};
        const std::vector<bool> grp_trans_b {transB};
        const std::vector<int> grp_m {M};
        const std::vector<int> grp_n {N};
        const std::vector<int> grp_k {K};
        const std::vector<float> grp_alpha {alpha};
        const std::vector<const void *> grp_src {src};
        const std::vector<int> grp_lda {lda};
        // Preserve the caller's weight representation through AUTO. AOCL
        // consumes its blocked buffer directly; only a selected CK path may
        // unreorder it into a CK-local raw lease.
        const std::vector<const void *> grp_weight {exec_weight};
        const std::vector<int> grp_ldb {ldb};
        const std::vector<const void *> grp_bias {bias};
        const std::vector<float> grp_beta {beta};
        const std::vector<void *> grp_dst {dst};
        const std::vector<int> grp_ldc {ldc};
        const std::vector<bool> grp_weights_const {is_weights_const};
        const std::vector<matmul_params> grp_params {grp_param};

        const status_t grp_status = group_matmul_direct(grp_layout, grp_trans_a,
                grp_trans_b, grp_m, grp_n, grp_k, grp_alpha, grp_src, grp_lda,
                grp_weight, grp_ldb, grp_bias, grp_beta, grp_dst, grp_ldc,
                grp_weights_const, grp_params,
                /*moe_postop=*/nullptr, group_gated_act_ptr,
                /*fused_moe=*/nullptr);
        if (grp_status == status_t::success) {
            if (scalar_mutation_conflict()) {
                if (is_profile) { profiler.tbp_stop(); }
                log_error(
                        "matmul_direct: grouped scalar route refused a reorder "
                        "from an in-place-mutated weight; failing the call");
                return status_t::failure;
            }
            if (is_profile) { profiler.tbp_stop(); }
            apilog_info(
                    "LOWOHA matmul_direct: routed to group_matmul_direct "
                    "(num_ops=1, act=",
                    static_cast<int>(run_gated_act
                                    ? group_gated_act.act
                                    : grp_matmul_gated_act_t::none),
                    "), M=", M, ", K=", K, ", N=", N,
                    ", source=", (aocl_prepacked ? "aocl_prepacked" : "raw"));
            return status_t::success;
        }
        if (scalar_mutation_conflict()) {
            if (is_profile) { profiler.tbp_stop(); }
            log_error(
                    "matmul_direct: grouped scalar route hit a mutated-weight "
                    "cache conflict; refusing standard scalar fallback");
            return status_t::failure;
        }
        if (run_gated_act) {
            if (is_profile) { profiler.tbp_stop(); }
            log_error(
                    "matmul_direct: group_matmul_direct could not execute the "
                    "requested gated activation");
            return grp_status;
        }
        log_info(
                "group_matmul_direct route declined; continuing on the "
                "standard single-matmul path");
    }

    if (is_foreign_prepacked_weight(exec_params)) {
        log_error(
                "matmul_direct: pre-reordered weight has unsupported backend "
                "provenance; only AOCL-DLP and custom-kernel packed layouts "
                "are accepted");
        return status_t::failure;
    }

    // The current call may be prompt-shaped, under-threaded, or carry an
    // execution-only option that prevents grouped routing while a later call
    // reusing the SAME raw weight is decode-eligible. Do not let this scalar
    // AOCL fallback consume WC=2 and destroy the raw bytes that future CK
    // packing needs. The candidate predicate intentionally ignores those
    // per-call fields and only identifies a CK-compatible weight family.
    if (custom_kernel_cross_phase_weight_candidate(exec_params)) {
        exec_params.weight_cache_type
                = std::min<int32_t>(exec_params.weight_cache_type, 1);
        apilog_verbose(
                "[MATMUL.CK CROSS_PHASE] scalar fallback capped at WC=1 "
                "to preserve raw weight for a future grouped CK call");
    }

    // A custom-kernel image -- a CK-VNNI caller-prepacked weight, or one a
    // grouped call packed in place -- has no scalar consumer. The grouped
    // route above is the only legal hand-off; if it did not engage, fail
    // closed before kernel_select reads the image as the weight.
    if ((exec_params.mem_format_b == 'r'
                && exec_params.lowoha_algo == matmul_algo_t::moe_custom_kernel)
            || custom_kernel::weight_packed_in_place(weight)) {
        log_error(
                "matmul_direct: CK-packed weight requires the grouped "
                "custom-kernel route; no scalar fallback can consume it");
        return status_t::failure;
    }

    // Reorder quant mutates exec_params / exec_batch_params and may redirect
    // exec_src to an internal contiguous buffer (quant_buffers).
    const void *exec_src = src;
    int reordered_lda = lda;
    reorder_quant_buffers_t quant_buffers;
    const bool scalar_dynamic_w4a8 = exec_params.packing.pack_format_b == 0
            && exec_params.dynamic_quant && is_w4a8_config(exec_params);
    if (reorder_quantization_wrapper(exec_src, lda, reordered_lda,
                src_type_size, exec_params, exec_batch_params, transA, M, K,
                num_threads, quant_buffers)
            != status_t::success) {
        return status_t::failure;
    }
    // The scalar reorder has materialized the dynamic BF16/F32 source as S8.
    // From this point onward it is the same representation as
    // caller-prequantized W4A8 and must not retain the entry-state dynamic flag.
    const bool w4a8_source_reordered = scalar_dynamic_w4a8 && exec_src != src
            && exec_params.dtypes.src == data_type_t::s8;
    if (w4a8_source_reordered) { exec_params.dynamic_quant = false; }

    matmul_algo_t kernel = kernel_select(exec_params, exec_batch_params.Batch_A,
            exec_batch_params.Batch_B, batch_count, M, N, K, num_threads, bias,
            is_weights_const);
    // The auto tuner runs each call on one of its candidates: with every
    // candidate off AOCL-DLP the call is not AOCL-DLP's, and a call AOCL-DLP
    // refuses runs on the first candidate off AOCL-DLP, or fails without one.
    bool on_aocl_dlp = !runs_off_aocl_dlp(kernel);
    matmul_algo_t off_aocl_dlp = matmul_algo_t::none;
    if (kernel == matmul_algo_t::auto_tuner) {
        const std::vector<matmul_algo_t> &candidates = get_algo_candidates();
        const auto off = std::find_if(
                candidates.begin(), candidates.end(), runs_off_aocl_dlp);
        if (off != candidates.end()) { off_aocl_dlp = *off; }
        on_aocl_dlp = !std::all_of(
                candidates.begin(), candidates.end(), runs_off_aocl_dlp);
    }
    // A call with a post-op the AOCL-DLP chain does not take runs on the
    // reference kernel, which applies it; pooling runs on no kernel.
    if (ZENDNNL_DEPENDS_AOCLDLP && on_aocl_dlp && call.aocl_unmapped_post_op) {
        if (call.pooling_post_op) {
            log_error("matmul_direct: no kernel applies a pooling post-op");
            return status_t::unimplemented;
        }
        const char *declined
                = reference_declines(layout, M, N, exec_params, params);
        if (declined != nullptr) {
            log_error(
                    "matmul_direct: the AOCL-DLP chain does not apply elu, "
                    "square, abs, sqrt, exp, log or softmax, and the "
                    "reference kernel does not serve ",
                    declined);
            return status_t::unimplemented;
        }
        log_info(
                "matmul_direct: the AOCL-DLP chain does not apply elu, "
                "square, abs, sqrt, exp, log or softmax; the call runs on the "
                "reference kernel");
        kernel = matmul_algo_t::reference;
        on_aocl_dlp = false;
    }
    const bool report = off_aocl_dlp == matmul_algo_t::none;
    if (on_aocl_dlp
            && (aocl_dlp_refuses_wei_scale(
                        exec_params, exec_params.dtypes.src, M, N, K, report)
                    || aocl_dlp_refuses_wei_scale_type(
                            exec_params, exec_params.dtypes.src, M, N, report)
                    || aocl_dlp_refuses_alpha_beta(exec_params,
                            exec_params.dtypes.src, M, N, alpha, beta, true,
                            report))) {
        if (report) { return status_t::unimplemented; }
        log_info(
                "matmul_direct: the auto tuner runs a call AOCL-DLP does not "
                "serve on ",
                kernel_to_string(off_aocl_dlp));
        kernel = off_aocl_dlp;
    }
    matmul_algo_t api_log_kernel = kernel;
    static unsigned int auto_version = get_auto_tuner_ver();

    // TODO: Add memory unreordering logic
    // Unreorder if onednn/ libxsmm is used
    // Implement the necessary logic for memory reordering here
    // if (params.mem_format_b) {}
    if (kernel == matmul_algo_t::reference && exec_params.mem_format_b == 'r') {
        log_error(
                "Reference kernel does not support AOCL-reordered weights "
                "(mem_format_b='r')");
        return status_t::unimplemented;
    }

    // Dispatch to reference, BMM, or single matmul based on kernel / batch_count
    thread_guard tg(num_threads, omp_mt);
    const int exec_lda = (exec_params.dynamic_quant || w4a8_source_reordered)
            ? reordered_lda
            : lda;

    if (kernel == matmul_algo_t::reference) {
        reference_matmul_execute(layout, transA, transB, M, N, K, alpha,
                exec_src, exec_lda, exec_weight, ldb, bias, beta, dst, ldc,
                is_weights_const, exec_batch_params, exec_params);
    } else if (batch_count > 1) {
        // Batch Matrix Multiplication (BMM)
        bmm::bmm_execute(layout, transA, transB, M, N, K, alpha, exec_src,
                exec_lda, exec_weight, ldb, bias, beta, dst, ldc,
                is_weights_const, exec_batch_params, src_type_size,
                weight_type_size, out_type_size, num_threads, kernel,
                exec_params);
    } else {
        // Single Matrix Multiplication (Matmul)
        matmul_execute(layout, transA, transB, M, N, K, alpha, exec_src,
                exec_lda, exec_weight, ldb, bias, beta, dst, ldc,
                is_weights_const, src_type_size, out_type_size, num_threads,
                kernel, exec_params, exec_batch_params, auto_version);
    }

    if (scalar_mutation_conflict()) {
        if (is_profile) { profiler.tbp_stop(); }
        log_error(
                "matmul_direct: scalar AOCL reorder was refused because the "
                "weight was already mutated under a different cache key");
        return status_t::failure;
    }

    if (is_profile) { profiler.tbp_stop(); }

    if (apilog_info_enabled() || is_profile) {
        [[maybe_unused]] std::ostringstream ss;
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
           << ", post_op=[" << post_op_names_to_string(params) << "]"
           << ", post_op_dtype=[" << ([&]() {
            std::string dtypes = post_op_data_types_to_string(params);
            return dtypes.empty() ? "none" : dtypes;
        })()
           << "]"
           << ", Batch_A=" << batch_params.Batch_A
           << ", Batch_B=" << batch_params.Batch_B
           << ", plugin_op=" << params.plugin_op
           << ", dynamic_quant=" << (params.dynamic_quant ? "true" : "false");

        if (api_log_kernel == matmul_algo_t::auto_tuner) {
            apilog_info(ss.str(), ", kernel=", kernel_to_string(api_log_kernel),
                    ", auto_tuner version=", auto_version);
        } else {
            apilog_info(
                    ss.str(), ", kernel=", kernel_to_string(api_log_kernel));
        }
        if (is_profile) {
            profilelog_verbose(ss.str(), ", kernel=", kernel_to_string(kernel),
                    ", weight_address=", static_cast<const void *>(weight),
                    ", time=", profiler.tbp_elapsedtime(),
                    profiler.get_res_str());
        }
    }

    return status_t::success;
}

ZENDNNL_API void clear_matmul_aocl_weight_caches() {
    zendnnl::lowoha::matmul::clear_aocl_matmul_weight_caches();
}

ZENDNNL_API void clear_matmul_aocl_postop_metadata_cache() {
#if ZENDNNL_DEPENDS_AOCLDLP
    zendnnl::lowoha::matmul::clear_aocl_postop_metadata_cache();
#endif
}

ZENDNNL_API void clear_matmul_native_weight_caches() {
    zendnnl::lowoha::matmul::native::clear_all_weight_caches();
}

ZENDNNL_API void clear_matmul_onednn_weight_caches() {
#if ZENDNNL_DEPENDS_ONEDNN
    zendnnl::lowoha::matmul::clear_onednn_matmul_weight_cache();
#endif
}

ZENDNNL_API void clear_grp_matmul_weight_caches() {
    clear_aocl_ck_weight_cache();
    custom_kernel::clear_custom_kernel_pack_cache_bf16();
    custom_kernel::clear_custom_kernel_pack_cache_int8();
    custom_kernel::clear_custom_kernel_pack_cache_f16();
    custom_kernel::clear_custom_kernel_pack_cache_s4();
    group_matmul_prepack::clear_fingerprint_cache_for_test();
    clear_weight_scale_f32_cache();
}

ZENDNNL_API void clear_matmul_weight_caches() {
    clear_matmul_aocl_weight_caches();
    clear_matmul_aocl_postop_metadata_cache();
    clear_matmul_onednn_weight_caches();
    clear_matmul_native_weight_caches();
    clear_grp_matmul_weight_caches();
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
