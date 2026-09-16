/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/// ALGO 5 — expert-parallel grouped GEMM, public interface header.
///
/// Library-internal: not part of the public ZenDNN API and not meant for
/// inclusion outside `src/lowoha_operators/matmul/group_matmul/` (plus the
/// `gtests/group_matmul/` files that exercise the env override atoms).
///
/// Strategy: one `omp parallel for` over experts, each executed end-to-end by
/// a single thread with no intra-expert split.  That makes it the only
/// executor with zero cross-thread coordination per expert, at the cost of
/// needing `num_ops >= num_threads` to fill the machine.  Unlike ALGO 2
/// (row-split) and ALGO 3 (column-split) it needs no planner and no
/// per-thread scratch, so none of their planner env knobs apply here.

#ifndef ZENDNNL_GROUP_MATMUL_EXPERT_PARALLEL_HPP
#define ZENDNNL_GROUP_MATMUL_EXPERT_PARALLEL_HPP

#include <vector>

#include "../group_matmul_parallel_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

/// ALGO 5 executor.  Parallel-for over experts, each executed by a single
/// thread.  NOT a 1:1 expert<->thread mapping: with `schedule(dynamic, 1)`
/// and `num_ops > num_threads`, a thread processes several in sequence.
///
/// `fused_act != none` applies a gated activation over `dst[i]` after that
/// expert's GEMM, inside the same loop iteration, so it stays on the thread
/// that owns the rows and in the cache level the GEMM just left them.
void parallel_per_expert(const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        std::vector<matmul_params> &params, int num_threads,
        grp_matmul_gated_act_t fused_act, data_type_t act_dtype);

/// One matmul half (W13 or W2) of the fused pipeline below.
///
/// The halves differ only in these per-expert vectors; `layout`, `transB`,
/// `M`, `is_weights_const`, the activation and the thread count are shared.
/// Grouping them keeps the pipeline entry at a readable arity.
///
/// All members are references into the caller's vectors — an argument bundle
/// with the lifetime of the call, not a value type to store.
struct pipeline_half_t {
    const std::vector<bool> &transA;
    const std::vector<int> &N;
    const std::vector<int> &K;
    const std::vector<float> &alpha;
    const std::vector<const void *> &weight;
    const std::vector<int> &ldb;
    const std::vector<const void *> &bias;
    const std::vector<float> &beta;
    const std::vector<void *> &dst;
    const std::vector<int> &ldc;
    std::vector<matmul_params> &params;
};

/// Outcome of `try_expert_parallel_pipeline`.
///
/// Three states, not two, because "did not complete" is NOT one situation.
/// A decline happens before anything is written and the two-pass can serve the
/// call; a mid-region failure cannot be retried at all, and the difference is
/// not recoverable by the caller:
///
///   * `reorder_quantization_wrapper` takes `params` by NON-const reference and
///     rebinds it as a side effect — `dtypes.src` becomes the quantized dtype
///     and `src_scale.buff` / `src_zp.buff` are repointed into the worker's
///     `thread_local` pooled storage.  `pipeline_half_t::params` is itself a
///     reference into the caller's vector, so those edits are visible after we
///     return.  A two-pass rerun would then read `dtypes.src == s8` against the
///     caller's ORIGINAL float source and reinterpret it, while the scale
///     pointers dangle into another thread's pool.  It also leaves
///     `dtypes.src == s8` with `dynamic_quant` still set — the "half-applied
///     pre-pass" state `classify_pipeline_half` explicitly refuses.
///   * Earlier slices may already have activated `w13.dst` in place, so a
///     rerun would activate twice, and a non-zero Op1 `beta` would accumulate
///     onto a destination that already holds a result.
///
/// Restoring all of that is not practical — the mutation is per-expert, spread
/// across threads, and points at pooled storage — so a failure is terminal.
enum class expert_parallel_result {
    declined, ///< Nothing written; run the two-pass.
    completed, ///< Both destinations complete; do NOT run the two-pass.
    failed, ///< May have written part; `params` may be mutated; fail the call.
};

/// ALGO 5 — fused per-expert MoE FFN pipeline: W13 -> gated act -> W2 in ONE
/// OMP region, replacing two `parallel_per_expert` passes.
///
/// Each expert's three stages are a chain private to that expert, so the
/// fork/join between the two passes separated work that was never dependent.
/// Removing it saves, per MoE layer: one fork/join, one dispatcher prologue
/// (ALGO selection, prepack prelude, the fingerprint hash and its lock), and
/// the intermediate's round trip to DRAM — W2's source is read by the thread
/// that just wrote it, while still cache-resident.
///
/// Stages through `w13.dst`, the same buffer and stride the two-pass writes,
/// so it needs no scratch and no scratch budget.  That is the structural
/// difference from the ALGO 2 pipeline, which slices experts and must size,
/// allocate and cap a per-thread staging tile.  It also means it inherits the
/// two-pass's buffer and stride handling rather than re-deriving it — though
/// it does NOT accept everything the two-pass does; see the declines below.
///
/// Returns `completed` when the pipeline ran and BOTH `w13.dst` and `w2.dst`
/// are complete, in which case the caller MUST NOT run a two-pass over the
/// same buffers.
///
/// Returns `declined` having written NOTHING.  Every eligibility gate is
/// evaluated before the region opens, since once threads begin writing there
/// is no way to unwind.  A decline is a routing choice, never an error, and
/// the two-pass serves the call.
///
/// Returns `failed` when a slice failed INSIDE the region, having potentially
/// written part of `w13.dst` / `w2.dst`.  Only the per-expert source
/// quantization can do
/// this — a runtime allocation failure, which no gate can screen.  The caller
/// MUST surface this as a hard failure and MUST NOT fall back to the two-pass;
/// see `expert_parallel_result` for why that rerun is unsafe rather than
/// merely wasteful.
///
/// Serves two weight-dtype regimes, which both halves must share:
///
///   * FLOAT end-to-end — `src == wei == dst`, one of f32 / bf16 / f16, with
///     `dynamic_quant == false`.  Bit-identical to the two-pass; only the
///     barrier position differs.
///   * DA8W8 — `wei == s8` with `compute == s8` (symmetric) and a float
///     destination.  The source may arrive already `s8`, or float with
///     `dynamic_quant`, in which case `execute_expert_slice` quantizes it per
///     expert.  W2 always takes the float form, so its source quantization IS
///     the re-quant stage, done by the thread that produced the intermediate.
///
///     Fusing bypasses the grouped dynamic-quant pre-pass (inherently a
///     barrier across all experts, since the intermediate does not exist
///     until every W13 has finished), so W2 quantizes its own source through
///     `dispatch_fused_per_token` instead — a row at a time when the
///     post-activation source is strided, or as one block when `w13.ldc`
///     leaves it contiguous.  Granularity, row set, scale formula and
///     rounding all match the grouped kernel, and the parity gtest asserts
///     EXACT equality against the two-pass for every shape it covers.
///
/// Anything else declines to the two-pass, so enabling the pipeline cannot
/// change an unsupported dtype's behaviour.
///
/// Declines, leaving both destinations untouched, unless ALL of the following
/// hold.  Anything here that the two-pass would have served falls back to it,
/// so a decline is a routing choice and never an error:
///
///   * the env knob is on, and `fused_act` is one of the four served kinds;
///   * every per-expert vector is at least `M.size()` long;
///   * every ACTIVE expert classifies into the SAME regime, both halves;
///   * under DA8W8: `compute == s8` (symmetric only — asymmetric u8 would
///     need its zero point carried through the re-quant, which is not
///     plumbed), `quant_params.wei_scale.buff` is non-null, and a float
///     source carries `dynamic_quant` with PER-TOKEN scale metadata present
///     (`src_scale.dims` non-empty, `.dt` set, innermost extent 1).  An
///     already-s8 source must NOT still carry `dynamic_quant`;
///   * BOTH halves carry a plain row-major, unpacked weight
///     (`mem_format_b == 'n'` and `packing.pack_format_b == 0`).  This
///     pipeline runs the regular AOCL / BRGEMM kernel and has no
///     caller-prepacked-weight consumption path, so a caller-prepacked CK
///     VNNI weight (or an AOCL-blocked / GGML-reordered one) must decline
///     to the two-pass, where the CK-only-or-fail guard rejects it.  Only
///     ALGO 3 + the custom kernel can consume those layouts;
///   * `w2.transA` is false (W2 reads the row-major intermediate);
///   * `w2`'s source dtype is what `w13` produces, and `act_dtype` is that
///     same dtype;
///   * `w13.N` is even when activating, `w2.K` equals the post-activation
///     W13 width, and `w13.ldc >= w13.N`;
///   * `w13.dst` and `w2.dst` are non-null on every active expert.
///
/// Two of these are narrower than the two-pass: a transposed `w2` source and
/// a per-group source scale are both served there and declined here.
///
/// The caller owns one further precondition: a `w2.dst` that aliases the W13
/// SOURCE is unsafe here even though the two-pass serves it, because fusion
/// drops the barrier that ordered every W13 read ahead of every W2 write.
/// `group_matmul_fused_moe.cpp` screens it with
/// `fused_moe_src_op2dst_hazard()` before calling.
expert_parallel_result try_expert_parallel_pipeline(
        const std::vector<char> &layout, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<const void *> &src,
        const std::vector<int> &lda, const std::vector<bool> &is_weights_const,
        grp_matmul_gated_act_t fused_act, data_type_t act_dtype,
        int num_threads, const pipeline_half_t &w13, const pipeline_half_t &w2);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_EXPERT_PARALLEL_HPP
