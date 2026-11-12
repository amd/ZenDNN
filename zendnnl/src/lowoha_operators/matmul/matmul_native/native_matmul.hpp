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

#ifndef MATMUL_NATIVE_NATIVE_MATMUL_HPP
#define MATMUL_NATIVE_NATIVE_MATMUL_HPP

#include "common/profiler.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// The layout-rule query of a matmul call: a bf16 or f32 source the call
/// quantizes dynamically to u8 / s8 (over an s8 or s4 weight) counts as that
/// type, and a {G > 1, N} weight scale as per-group.
native_layout_query_t native_layout_query_for(
        const matmul_params &params, int K, int N, int ldb, bool transB);

/// The native BRGEMM looks at the call: under ZENDNNL_MATMUL_ALGO=10 every
/// call, whatever it selects (a buffer it rewrote in place has no other
/// reader), else a call with lowoha_algo == native_brgemm.
inline bool native_brgemm_selected(const matmul_params &params) {
    return params.lowoha_algo == matmul_algo_t::native_brgemm
            || matmul_config_t::is_native_brgemm_algo();
}

/// The call asks for the native BRGEMM as its kernel: lowoha_algo ==
/// native_brgemm, or a single call (batch_count 1) with no lowoha_algo
/// under ZENDNNL_MATMUL_ALGO=10, which selects the kernel of single calls
/// (a batched call takes its kernel from ZENDNNL_BMM_ALGO).
inline bool native_brgemm_requested(
        const matmul_params &params, int batch_count) {
    return params.lowoha_algo == matmul_algo_t::native_brgemm
            || (params.lowoha_algo == matmul_algo_t::none && batch_count == 1
                    && matmul_config_t::is_native_brgemm_algo());
}

/// Runs a matmul_direct call that selects the native BRGEMM
/// (native_brgemm_selected) on it when the call is the native BRGEMM's:
///   * a call with lowoha_algo == native_brgemm, whatever it carries;
///   * a single call without lowoha_algo under ZENDNNL_MATMUL_ALGO=10
///     (native_brgemm_requested), unless the native BRGEMM does not
///     implement what it carries or the CPU lacks the instruction set it
///     needs (unimplemented, isa_unsupported), with N and K of at least 1
///     and ldb at least the weight row: that call goes back to the caller
///     for the default route, with one warning line per process for each
///     reason (a batched call without lowoha_algo too, which
///     ZENDNNL_MATMUL_ALGO=10 does not select), unless the default route
///     returns success for its form with a wrong or unwritten dst or does
///     not implement its activation (gelu_tanh_and_mul): that call keeps
///     its error, whose line says why it is not handed back;
///   * a call on a buffer an earlier native call rewrote in place (whatever
///     algorithm the call selects);
///   * a call on an aocl_dlp_blocked prepack under ZENDNNL_MATMUL_ALGO=10
///     whose pack has an inverse, when the native BRGEMM serves it.
/// Returns false for any other call, with the weight and dst untouched: the
/// caller continues on its other routes.  Returns true otherwise, with
/// status the call's status.  The native BRGEMM serves a call whose weight
/// decide_native_layout places on the layout (a raw ('n') weight, the
/// prepack's raw weight, or the image), single and row-major, with a plain
/// source, that uses only what the native BRGEMM implements, on the native
/// tiles over the weight's layout at every M.  Any other call it takes is
/// not served: status is unimplemented
/// (isa_unsupported on a CPU without the instruction set its data types
/// need; failure for a buffer rewritten in place, which no other route
/// reads, or for bad arguments), checked before any work, with dst and the
/// weight untouched, and one line at the error level names the reason (and,
/// with lowoha_algo == native_brgemm, that no other route takes the call).
/// Weight cache mode 2 rewrites a buffer in place only under
/// ZENDNNL_MATMUL_ALGO=10; with lowoha_algo == native_brgemm alone it caches
/// out of place.  A call it takes logs the line matmul_direct logs, batch
/// giving its batch fields (batch_count both when null).
bool native_layout_matmul(char layout, bool transA, bool transB, int M, int N,
        int K, float alpha, const void *src, int lda, const void *weight,
        int ldb, const void *bias, float beta, void *dst, int ldc,
        bool is_weights_const, int batch_count, int num_threads,
        matmul_gated_act_t gated_act, matmul_params &params,
        zendnnl::profile::profiler_t *profiler, status_t &status,
        const matmul_batch_params_t *batch = nullptr);

/// What native_layout_matmul does with the call, without running it and
/// without a log line: false when the call is not the native BRGEMM's (or
/// goes back to the caller for the default route); true
/// otherwise, with status success when the native BRGEMM serves it, or the
/// status native_layout_matmul returns and reason naming why.  Runs every
/// check of the native BRGEMM up to the point where it would change the
/// weight: dst and the weight stay untouched; a bf16 / f32 source it
/// quantizes goes into the call's own buffers, and the scales and zero
/// points it makes into the buffers of params.
bool native_layout_check(char layout, bool transA, bool transB, int M, int N,
        int K, float alpha, const void *src, int lda, const void *weight,
        int ldb, const void *bias, float beta, void *dst, int ldc,
        bool is_weights_const, int batch_count, int num_threads,
        matmul_gated_act_t gated_act, const matmul_params &params,
        status_t &status, const char *&reason);

class native_weight_raw_hold_t;

/// For a call of layout native_layout_matmul handed back, or a batched call
/// it does not take (batch, null for a single call): holds the bytes the
/// default route reads of its weight (a GGML pack's blocks, an AOCL-DLP
/// prepack, or the raw weight as the call lays it out; of a batched call
/// its Batch_B matrices, batch_stride_wei elements apart or as the batched
/// kernels lay them out) in hold, which the caller keeps while the default
/// route reads them, as no in-place rewrite of them may run meanwhile,
/// whatever the process's weight cache mode is now.  Returns success;
/// failure, with one error line, when the bytes hold an image an in-place
/// rewrite wrote (no other route reads it), or when no memory holds them.
status_t native_hold_handed_back_weight(native_weight_raw_hold_t &hold,
        char layout, const void *weight, int K, int N, int ldb, bool transB,
        const matmul_params &params,
        const matmul_batch_params_t *batch = nullptr);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_NATIVE_MATMUL_HPP
