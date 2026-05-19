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

/**
 * @file group_matmul_direct.hpp
 * @brief Group matmul: MoE post-op, parallel dispatch, and direct-path helpers.
 *
 * Stability contract:
 *
 *   - PUBLIC (stable) user-facing API surface is declared in
 *     `lowoha_matmul.hpp`: the @c group_matmul_direct entry point plus
 *     the three parameter structs it needs (@c group_matmul_moe_postop_params,
 *     @c grp_matmul_gated_act_params, @c grp_matmul_fused_moe_params) and
 *     the associated enum `grp_matmul_gated_act_t`.  Those are guaranteed
 *     API/ABI across patch versions.
 *
 *   - The symbols declared in THIS header (`group_matmul_moe_act_execute`,
 *     `apply_gated_act_inplace`, `apply_swiglu_oai_tile_rows`,
 *     `apply_swiglu_oai_tile_rows_oop`, `group_matmul_fused_moe_execute`
 *     [both overloads], `validate_group_matmul_moe_postop`,
 *     `group_matmul_moe_postop_execute`, and the internal dispatch /
 *     planner plumbing) are LIBRARY-INTERNAL: they are split across
 *     multiple translation units (group_matmul_direct.cpp,
 *     group_matmul_parallel.cpp, group_matmul_moe_postop.cpp,
 *     group_matmul_fused_moe.cpp, group_matmul_moe_act.cpp) and the
 *     declarations here exist so those TUs can link against a common
 *     signature.  They are NOT part of the stable external API and are
 *     not covered by the compatibility guarantee.
 *
 * Implementation note: the entire `include/` tree (including this file)
 * is installed into the package's public include directory because the
 * build system installs by prefix.  External callers CAN observe these
 * symbols, but SHOULD treat everything except the
 * `group_matmul_direct`-related interface above as implementation
 * detail that may change without notice.
 */

#ifndef LOWOHA_GROUP_MATMUL_DIRECT_HPP
#define LOWOHA_GROUP_MATMUL_DIRECT_HPP

#include <cstdint>
#include <vector>

#include "common/error_status.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// --- MoE post-op: weighted-reduce after parallel expert GEMMs ---

/**
 * @brief Optional MoE post-op parameters for parallel group matmul.
 *
 * When supplied to @c group_matmul_direct (parallel mode only), the post-op
 * performs a weighted-reduce over pre-gathered expert output rows:
 *
 *   For each token t and hidden dim d:
 *     output[t, d] = Σ_k  topk_weights[t, k] * row_ptrs[t*topk+k][d]
 *
 * When @c skip_weighted is true, all weights are implicitly 1.0.
 *
 * The caller builds @c row_ptrs during the token-to-expert scatter step,
 * fusing scatter + gather in one pass on the frontend side.  Each entry
 * is a pointer to the start of a D-wide row in an expert dst buffer.
 * This design allows future fusion with the expert GEMM kernels.
 *
 * Constraints checked by validation:
 *   - @c row_ptrs and @c output must be non-null.
 *   - @c num_tokens > 0 and @c topk > 0.
 *   - @c ldc_output >= D (hidden dimension).
 *   - dst dtype must be FP32 or BF16.
 *   - @c topk_weights required unless @c skip_weighted is true.
 */
struct group_matmul_moe_postop_params {
  /// Number of input tokens (rows in the output buffer).
  int num_tokens = 0;

  /// Number of experts selected per token.
  int topk = 0;

  /// Output buffer: row-major [num_tokens, ldc_output].
  /// First D columns of each row are written (FP32 or BF16).
  void *output = nullptr;

  /// Leading dimension of the output buffer (>= D).
  int ldc_output = 0;

  /// Routing weights: tightly packed [num_tokens, topk] (row-major).
  /// Entry [t * topk + k] is the weight for token t's k-th expert.
  /// Required unless @c skip_weighted is true.
  const float *topk_weights = nullptr;

  /// When true, every routing weight is implicitly 1.0 and
  /// @c topk_weights may be nullptr (plain gather-sum, no weighting).
  bool skip_weighted = false;

  /// Pre-gathered row pointers: flat array of size num_tokens * topk.
  /// Entry row_ptrs[t * topk + k] points to the start of a D-wide row
  /// (FP32 or BF16) in an expert dst buffer — the row that contributes
  /// to token t's k-th expert slot.
  ///
  /// The caller builds this during token-to-expert scatter:
  ///   row_ptrs[t * topk + k] = dst[expert_id] + row_j * ldc[expert_id]
  /// (with appropriate element-size scaling for the dst dtype).
  const void **row_ptrs = nullptr;
};

status_t validate_group_matmul_moe_postop(
    const group_matmul_moe_postop_params *postop,
    int D,
    data_type_t dst_elem);

status_t group_matmul_moe_postop_execute(
    const group_matmul_moe_postop_params *postop,
    int D,
    int num_threads,
    data_type_t dst_elem);

// --- MoE gated activation: fused act(gate) * up after GEMM ---

/**
 * @brief Gated activation type for MoE fused gate+up GEMM.
 *
 * Applied after the GEMM that uses fused [gate_W | up_W] weights.
 * The GEMM output dst[M, 2*dim] is split into gate[:, 0:dim] and
 * up[:, dim:2*dim].  The activation computes in-place:
 *   dst[:, 0:dim] = act(gate) * up
 * The second half (up columns) becomes garbage after activation.
 * The caller passes ldc=2*dim to the subsequent down_proj GEMM as lda.
 */
enum class grp_matmul_gated_act_t : int {
  none = 0,           ///< No gated activation (down_proj or unfused).
  silu_and_mul = 1,   ///< SiLU(gate) * up — split-halves [gate | up] layout.
  gelu_and_mul = 2,   ///< GELU(gate) * up — split-halves [gate | up] layout.
  swiglu_oai_mul = 3  ///< SwigluOAI — interleaved [g0,u0,g1,u1,...] layout.
};

/**
 * @brief Parameters for MoE gated activation post-op.
 *
 * Single struct (not per-expert) — all experts in a group_matmul call
 * use the same activation type.  The activation operates on each expert's
 * dst buffer in-place.
 */
struct grp_matmul_gated_act_params {
  grp_matmul_gated_act_t act;  ///< Activation type (or none).

  grp_matmul_gated_act_params() : act(grp_matmul_gated_act_t::none) {}
};

/**
 * @brief Apply gated activation to all experts' dst buffers.
 *
 * For each expert e: dst[e][:, 0:dim] = act(dst[e][:, 0:dim]) * dst[e][:, dim:2*dim]
 * where dim = N[e] / 2.
 *
 * @param act_params  Activation type (nullptr or act==none → no-op).
 * @param dst         Per-expert output buffers from GEMM [M_e, N_e].
 * @param M           Per-expert row counts.
 * @param N           Per-expert column counts (must be even: N = 2*dim).
 * @param ldc         Per-expert leading dimensions of dst.
 * @param dst_dtype   Data type of dst buffers (FP32 or BF16).
 * @param num_threads OMP thread count for parallel execution.
 */
status_t group_matmul_moe_act_execute(
    const grp_matmul_gated_act_params *act_params,
    const std::vector<void *> &dst,
    const std::vector<int> &M,
    const std::vector<int> &N,
    const std::vector<int> &ldc,
    data_type_t dst_dtype,
    int num_threads);

/**
 * @brief Apply gated activation in-place on a row range of a single expert.
 *
 * Single-threaded — designed to be called from within OMP parallel regions
 * (ALGO 2 M-tile, ALGO 1/4/5 per-expert) for fused activation.
 *
 * @param act       Activation type (none is a no-op).
 * @param dst       Expert output buffer [M, ldc].
 * @param row_start First row to process (inclusive).
 * @param row_end   Last row to process (exclusive).
 * @param N         Total columns (must be even: N = 2*dim).
 * @param ldc       Leading dimension of dst.
 * @param dst_dtype Data type of dst (f32 or bf16).
 */
void apply_gated_act_inplace(
    grp_matmul_gated_act_t act,
    void *dst, int row_start, int row_end,
    int N, int ldc, data_type_t dst_dtype);

/**
 * @brief Apply swiglu_oai_mul to an M x pairs interleaved tile, in-place.
 *
 * Used by the ALGO 3 N-tile fused-swiglu-oai path.  Each OMP thread owns a
 * pair-aligned column range [col_start, col_start + 2*pairs) of the matmul
 * output.  After an OMP barrier separates matmul from activation, this
 * function reads the thread's `pairs` interleaved (g, u) pairs from those
 * columns of every row and writes `pairs` activated values into columns
 * [col_start/2, col_start/2 + pairs) of the same rows.
 *
 * In-place safety: at col_start == 0 the read-ahead property (read 2n, 2n+1
 * before writing n) prevents self-corruption; at col_start > 0 the write
 * range is strictly to the left of the read range and cannot stomp it.
 * Cross-thread correctness relies on the caller placing a barrier between
 * the matmul writes and the first call into this function.
 *
 * Only swiglu_oai_mul is supported (interleaved gate/up layout).  Legacy
 * split-halves activations (silu_and_mul / gelu_and_mul) are still handled
 * via the separate-pass path.
 *
 * @param dst_buf    Expert output buffer (base pointer to row 0, col 0).
 * @param M          Number of rows in the expert's output.
 * @param col_start  First column in the interleaved buffer (must be even).
 * @param pairs      Number of (g, u) pairs this thread owns.
 * @param ldc        Leading dimension of dst_buf (elements, not bytes).
 * @param dtype      Buffer element type (f32 or bf16).
 */
void apply_swiglu_oai_tile_rows(
    void *dst_buf, int M, int col_start, int pairs,
    int ldc, data_type_t dtype);

/**
 * @brief Out-of-place per-thread tile activation for swiglu_oai_mul.
 *
 * Reads pairs interleaved (g, u) elements per row from src_buf
 * starting at column src_col_start (must be even), applies swiglu_oai
 * (clamp + α-swish + (1 + u) factor), and writes `pairs` activated
 * values to dst_buf starting at column dst_col_start.  src_buf and
 * dst_buf can be different buffers with different strides — used by
 * the I-only fused MoE path where the matmul writes a wide tile to
 * per-thread scratch and the activation packs the half-width
 * activated result directly into a tight output arena, skipping the
 * round-trip through a global wide intermediate buffer.
 *
 * @param src_buf       Source buffer (per-thread scratch from wide matmul).
 * @param src_ldc       Leading dimension of src_buf (elements, not bytes).
 * @param src_col_start First column to read from src (must be even).
 * @param dst_buf       Destination buffer (tight output arena).
 * @param dst_ldc       Leading dimension of dst_buf (elements, not bytes).
 * @param dst_col_start First column to write in dst.
 * @param M             Row count.
 * @param pairs         Number of (g, u) pairs to process (= dst cols written).
 * @param dtype         Buffer element type (f32 or bf16).
 */
void apply_swiglu_oai_tile_rows_oop(
    const void *src_buf, int src_ldc, int src_col_start,
    void *dst_buf, int dst_ldc, int dst_col_start,
    int M, int pairs, data_type_t dtype);

// --- Fused MoE: Op1(gate+up) → activation → Op2(down_proj) in one pass ---

/**
 * @brief Parameters for fused MoE execution (Op1 → Act → Op2).
 *
 * When provided to group_matmul_direct, the entire MoE block is executed
 * as a single API call.
 *
 * Current implementation — always two-pass for all GRP_ALGO values:
 *     Pass 1: Op1 (gate+up) + gated activation via parallel dispatch.
 *     Pass 2: Op2 (down_proj)                  via parallel dispatch.
 *   Both passes honour ZENDNNL_GRP_MATMUL_ALGO.  The dispatcher may
 *   fuse the gated activation into Pass 1's epilogue (all activations
 *   on ALGO 1/2/4/5, and swiglu_oai_mul on ALGO 3); otherwise a
 *   separate activation sub-pass is applied to Pass 1's output before
 *   Pass 2.  Per-expert deep fusion of Op1 → activation → Op2 chained
 *   at L1/L2/L3 boundaries is a possible future extension.
 *
 * The weighted-reduce (moe_postop) always runs in a separate pass after
 * Op2 since it requires all experts' outputs to be complete.
 *
 * All vectors must have size num_ops (one entry per expert).
 *
 * Constraints:
 *   - N must be even when a gated activation (swiglu/silu/gelu_and_mul)
 *     is selected — those activations collapse pairs of cols.  N may be
 *     any positive value for act=none.
 *   - dst_down[i] must be at least [M[i], N_down[i]].
 *   - Op2's K-dimension (the row count of `down_weight[i]`) follows
 *     the activation:
 *         act gated   →  K_down = N[i] / 2  (down_weight is [N/2, N_down]).
 *         act = none  →  K_down = N[i]      (down_weight is [N,   N_down]).
 *     The caller's `down_weight[i]` and `ldb_down[i]` MUST be sized
 *     accordingly.  See `op2_k_for_act` in group_matmul_fused_moe.cpp
 *     for the formal contract.
 *   - Op2 inherits the caller-provided per-expert layout from Op1;
 *     transA=false is hardcoded for Op2, transB is inherited from Op1
 *     (typically both false for MoE row-major weights).
 *   - Op2's alpha/beta are hardcoded to 1.0/0.0 (down_proj is a clean
 *     overwrite-style GEMM; caller-provided alpha/beta apply to Op1
 *     only).
 */
struct grp_matmul_fused_moe_params {
  /// Per-expert down_proj weights — shape depends on activation
  /// (see the contract above): [N/2, N_down[i]] for gated acts,
  /// [N, N_down[i]] for act=none.
  /// Must use same dtype as Op1 weights (params[i].dtypes.wei).
  std::vector<const void *> down_weight;
  std::vector<int> N_down;                ///< Per-expert output columns of down_proj.
  std::vector<int> ldb_down;              ///< Leading dimension of down_weight per expert.
  std::vector<const void *> bias_down;    ///< Per-expert bias for down_proj (nullptr OK).
  data_type_t bias_dt_down = data_type_t::none;  ///< Bias dtype for Op2 (none = no bias).

  // ─── Op2 (down_proj) weight quantization (optional) ──────────────────
  //
  // The fused-MoE dispatcher inherits every quant *scheme* knob from
  // the caller's `params[i]` for Op2 (so Op1 and Op2 always use the
  // same scheme — same `dtypes.wei`, same `dynamic_quant` flag, same
  // `dtypes.compute`, same per-token `src_scale.dims`).  The ONLY
  // thing that has to be carried separately is the down_weight scale
  // (and optional zero-point) tensor itself, because `down_weight[i]`
  // is a different tensor from Op1's `weight[i]` and therefore has
  // its own per-channel / per-group / per-tensor scale buffer.
  //
  // The two fields below carry exactly that:
  //   * `down_scale[i]` — per-expert weight scale for `down_weight[i]`.
  //   * `down_zp[i]`    — per-expert weight zero-point (asymmetric only;
  //                     leave default-constructed / `dims.empty()`
  //                     for symmetric quant).
  //
  // Both vectors are OPTIONAL and default-empty.  Behaviour matrix:
  //
  //   `params[i].dynamic_quant` | `params[i].dtypes.wei` | `down_scale` | Op2 scheme
  //   ─────────────────────────── ───────────────────────  ─────────── ────────────
  //   false                     | bf16 / f32             | empty      | Un-quantized (legacy default)
  //   false                     | s4 / u4                | populated  | WOQ S4 / U4 on Op2
  //   false                     | s8                     | populated  | (limited — see note below)
  //   true                      | s8                     | populated  | Dynamic INT8 on Op2 (runtime BF16→S8 reorder)
  //
  // The Op2-side runtime reorder for dynamic INT8 inherits its
  // `src_scale.dims` (and `dt`) from `params[i].quant_params.src_scale`
  // and lets the kernel allocate the runtime scratch internally —
  // the caller never sees nor manages an Op2-side `src_scale.buff`.
  //
  // **Per-token source granularity ONLY**: dynamic source quant in
  // the fused MoE path supports `src_scale.dims = {M, 1}` (and the
  // trivial per-tensor `{1, 1}` / `{1}` forms).  Per-group
  // (`{M, ngroups}` with `ngroups > 1`) is rejected up front by
  // the dispatcher because Op1 reduces over `K[i]` (= K_in) and
  // Op2 reduces over `op2_k_for_act(N[i], act)` (= K_down), and
  // K_in != K_down in general — so Op1's ngroups cannot transfer
  // to Op2 verbatim (the documented invariant at
  // `docs/operator/lowoha_matmul_operator.md:227` requires
  // source-side and weight-side ngroups to match along K *per
  // pass*).  Use per-token granularity instead — it is K-
  // independent and works on both passes.
  //
  // Note on pure WOQ-S8: AOCL DLP's WOQ fast path is gated to s4/u4
  // only (see `aocl_postop.cpp::is_woq`); a bf16-src + s8-wei combo
  // without a caller-provided src_scale falls into the BF16-INT8
  // pre-quant path and is rejected.  Prefer S4 for WOQ or pair S8
  // weights with `dynamic_quant = true` on `params[i]`.
  //
  // Each per-expert quantization vector is validated independently:
  // if `down_scale` is non-empty it MUST hold at least `num_ops`
  // entries, and if `down_zp` is non-empty it MUST likewise hold at
  // least `num_ops` entries. Partial vectors are rejected up front to
  // avoid silently leaving tail experts un-quantized.
  struct down_weight_quant_t {
    const void *buff;              ///< Pointer to quantization data buffer
    data_type_t dt;                ///< Data type of the buffer
    std::vector<int64_t> dims;     ///< Dimensions of the quantization tensor

    down_weight_quant_t() : buff(nullptr), dt(data_type_t::none), dims() {}
  };

  std::vector<down_weight_quant_t> down_scale;  ///< Per-expert Op2 weight scale (down_weight[i]'s scale tensor).
  std::vector<down_weight_quant_t> down_zp;     ///< Per-expert Op2 weight zero-point (asymmetric quant only; empty for sym).

  // ─── Op2 output mode selection (dst_down behaviour) ──────────────────
  //
  // The fused MoE op supports TWO modes for the down_proj output:
  //
  //   (1) Legacy / caller-allocated mode (BACKWARD COMPATIBLE):
  //       Caller allocates per-expert dst_down[i] buffers and supplies
  //       ldc_down[i] strides.  The library writes Op2 output there and
  //       returns.  Caller (or moe_postop) reads from dst_down[i].
  //       Engaged when dst_down is non-empty.
  //
  //   (2) Internal-alloc + src-reuse mode:
  //       Caller leaves dst_down empty.  The library:
  //         (a) obtains Op1 output scratch sized [M[i], N[i]] (wide)
  //             or [M[i], N[i]/2] (tight / swiglu_oai-compact) per
  //             expert in dst dtype;
  //         (b) runs Op1 + activation into the scratch;
  //         (c) runs Op2 reading from the scratch and writing BACK
  //             INTO the caller's src[] buffer (in-place reuse);
  //         (d) releases the scratch back to the library's internal
  //             per-thread arena for reuse.  The arena keeps its
  //             high-water capacity for subsequent calls.
  //       Caller then reads Op2 output from the same src[] buffer.
  //       Caller's `dst` parameter is ignored in this mode (typically
  //       passed as a vector of nullptrs or an empty vector).
  //
  //       Memory-lifetime note for mode (2):
  //         - The library does NOT call `free()` on the scratch at
  //           end-of-call.  Instead, Op1 scratch is allocated from a
  //           `static thread_local` arena (see `FusedMoEArena` in
  //           group_matmul_fused_moe.cpp) that grows on demand to
  //           the largest sizeof(M * N * dst_elem) seen on THIS
  //           thread across all calls, and is retained for the
  //           lifetime of the thread (freed in the thread-local
  //           destructor when the thread exits).
  //         - Similarly, the Op1 setup-side scratch (dst-ptr arrays,
  //           per-expert ldc vectors, etc.) lives in a thread-local
  //           `FusedMoEScratch` whose std::vectors keep their
  //           allocated capacity across calls.
  //         - Net per-call allocator traffic in steady state is
  //           O(num_ops) field writes — no malloc/free on the hot
  //           path.
  //         - Per-thread resident-set footprint reflects the largest
  //           fused-MoE shape that thread has ever executed; across
  //           a pool of N worker threads the total resident footprint
  //           is bounded above by
  //             N × max_seen(M_total × N_max × sizeof(dst_elem)).
  //           Frameworks that briefly run an outsized MoE shape on
  //           many worker threads and do not want the footprint
  //           persisted should either (a) use mode (1) (caller-
  //           allocated dst_down) where the framework owns buffer
  //           lifetime directly, or (b) execute such shapes on a
  //           dedicated thread that can be torn down afterwards.
  //
  //       Caller-side preconditions for mode (2):
  //         - src[i] must point to WRITABLE memory.  The const_cast
  //           inside the library is well-defined because the caller
  //           opted in by clearing dst_down and dst[].
  //         - lda[i] >= N_down[i] so the Op2 row stride (which equals
  //           lda[i] in mode 2) fits within the original src row
  //           stride.  Naturally holds for MoE layers with
  //           hidden_dim = K_input = N_down.
  //         - **MATCHED PRECISION REQUIRED**: params[i].dtypes.src
  //           MUST equal params[i].dtypes.dst.  Op2 writes dst-typed
  //           elements at row stride lda[i] (in dst-element units)
  //           into the caller's src[i] buffer.  When dst element
  //           size > src element size (e.g. bf16 src + f32 dst) the
  //           per-row write footprint (lda[i] * sizeof(dst_elem))
  //           exceeds the per-row allocation footprint
  //           (lda[i] * sizeof(src_elem)) and corrupts memory.
  //           This is enforced as an always-on guard inside
  //           group_matmul_fused_moe_execute() — mixed-precision
  //           callers must use mode (1) (caller-allocated dst_down)
  //           where the destination buffer is sized for dst dtype
  //           independently of src.
  //         - src[i] buffer size must be at least
  //               M[i] * lda[i] * sizeof(dtypes.dst) bytes
  //           (== M[i] * lda[i] * sizeof(dtypes.src) under matched
  //           precision).  This is the Op2 row-pitched write footprint.
  //
  // Mode (2) is targeted at frameworks that do their own token-grouping
  // scatter on src (so src is already a writable scratch), and their
  // own weighted-reduce gather on the Op2 output.  ZenDNN allocating
  // Op1 scratch internally (and freeing it at end-of-call) avoids
  // forcing the framework to size and own the W13 intermediate.
  std::vector<void *> dst_down;           ///< Empty → mode (2); non-empty → mode (1).
  std::vector<int> ldc_down;              ///< Stride for dst_down (mode (1) only).
};

/**
 * @brief Execute fused MoE: Op1(gate+up) → activation → Op2(down_proj)
 *        → optional MoE post-op (weighted reduce).
 *
 * The flow runs as two passes of group_matmul_run_parallel_dispatch;
 * both passes honour ZENDNNL_GRP_MATMUL_ALGO.  See
 * grp_matmul_fused_moe_params (above) for the full design note.
 *
 * When `moe_postop` is non-null, the weighted-reduce post-op is invoked
 * automatically after Op2 with `D = fused.N_down[0]` (the planner has
 * already validated that N_down is uniform across experts in this
 * combination).  Passing `moe_postop = nullptr` skips the post-op so the
 * caller can read the per-expert Op2 outputs from `fused.dst_down[]`
 * and run their own gather elsewhere.
 *
 * @param gemm_mode_out If non-null, receives a string literal for logging.
 */
// Primary signature: takes optional moe_postop for the integrated
// weighted-reduce post-op.  No default on moe_postop so the older
// 5-trailing-arg overload below remains unambiguous.
status_t group_matmul_fused_moe_execute(
    const grp_matmul_fused_moe_params &fused,
    grp_matmul_gated_act_t act, data_type_t act_dtype,
    const std::vector<char> &layout,
    const std::vector<bool> &transA, const std::vector<bool> &transB,
    const std::vector<int> &M, const std::vector<int> &N,
    const std::vector<int> &K, const std::vector<float> &alpha,
    const std::vector<const void *> &src, const std::vector<int> &lda,
    const std::vector<const void *> &weight, const std::vector<int> &ldb,
    const std::vector<const void *> &bias, const std::vector<float> &beta,
    const std::vector<void *> &dst, const std::vector<int> &ldc,
    const std::vector<bool> &is_weights_const,
    std::vector<matmul_params> &params,
    int num_threads,
    const char **gemm_mode_out,
    const group_matmul_moe_postop_params *moe_postop);

// Legacy ABI-preserving overload: same as the original (pre-postop)
// signature.  Kept as a separate exported symbol so binaries linked
// against the older library version continue to find their mangled
// name; internally forwards to the primary overload with
// moe_postop = nullptr.  Source-level callers that previously relied
// on `gemm_mode_out`'s default get the same default here.
status_t group_matmul_fused_moe_execute(
    const grp_matmul_fused_moe_params &fused,
    grp_matmul_gated_act_t act, data_type_t act_dtype,
    const std::vector<char> &layout,
    const std::vector<bool> &transA, const std::vector<bool> &transB,
    const std::vector<int> &M, const std::vector<int> &N,
    const std::vector<int> &K, const std::vector<float> &alpha,
    const std::vector<const void *> &src, const std::vector<int> &lda,
    const std::vector<const void *> &weight, const std::vector<int> &ldb,
    const std::vector<const void *> &bias, const std::vector<float> &beta,
    const std::vector<void *> &dst, const std::vector<int> &ldc,
    const std::vector<bool> &is_weights_const,
    std::vector<matmul_params> &params,
    int num_threads,
    const char **gemm_mode_out = nullptr);

// --- Parallel expert dispatch ---

/**
 * @brief Run independent expert GEMMs (parallel group matmul path).
 *
 * @param gemm_mode_out  If non-null, receives a static string literal for logging
 *                       ("sequential_experts", "flat_m_tile", "flat_n_tile",
 *                       "multilevel", or "per_expert").
 */
/**
 * @return true if the gated activation was fused into the ALGO's epilogue
 *         (caller should skip the separate activation pass).  false if
 *         the caller must apply activation separately.  Today:
 *           ALGO 1 / 2 / 4 / 5 — always fuse any supported activation.
 *           ALGO 3 N-tile      — fuses swiglu_oai_mul only (interleaved
 *                                layout); silu_and_mul / gelu_and_mul are
 *                                returned unfused because their split-
 *                                halves layout does not colocate (g, u)
 *                                pairs on the same thread's N-tile.
 */
bool group_matmul_run_parallel_dispatch(
    const std::vector<char> &layout,
    const std::vector<bool> &transA,
    const std::vector<bool> &transB,
    const std::vector<int> &M,
    const std::vector<int> &N,
    const std::vector<int> &K,
    const std::vector<float> &alpha,
    const std::vector<const void *> &src,
    const std::vector<int> &lda,
    const std::vector<const void *> &weight,
    const std::vector<int> &ldb,
    const std::vector<const void *> &bias,
    const std::vector<float> &beta,
    const std::vector<void *> &dst,
    const std::vector<int> &ldc,
    const std::vector<bool> &is_weights_const,
    std::vector<matmul_params> &params,
    int num_threads,
    const char **gemm_mode_out,
    grp_matmul_gated_act_t fused_act = grp_matmul_gated_act_t::none,
    data_type_t act_dtype = data_type_t::none);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif
