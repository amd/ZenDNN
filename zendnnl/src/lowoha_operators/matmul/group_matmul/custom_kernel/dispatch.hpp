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

/// BF16 custom kernel — public dispatcher.
///
/// Consumed through a single caller: `flat_n_tile` (ALGO 3).  That
/// executor serves both non-fused group_matmul (act=none) and the
/// fused-MoE Op1 (act=swiglu_oai_mul) / Op2 (act=none) paths — the
/// fused-MoE entry routes through the parallel dispatcher, which
/// forwards to flat_n_tile when ALGO 3 is selected, so the dispatcher
/// below sees only one consumer.
///
/// The caller sees just two entry points and one opaque context:
///
///   1.  `prepare_for_call()` — runs once in the caller's single-
///       threaded entry section.  Internally:
///         * runs CPUID + dtype + activation + per-expert contract
///           validation,
///         * picks the pack-NR (`plan_pack_nr`),
///         * builds the per-MR microkernel function-pointer table,
///         * pre-packs every active expert's weight via the LRU pack
///           cache (so the per-tile path never touches the mutex),
///         * computes the L2-friendly sub-tile width.
///       Returns success ⇒ `out.enabled` is true.  Returns failure ⇒
///       `out.enabled` is false; the caller must take the standard
///       path (e.g. `execute_expert_slice` or the standard fused-MoE
///       two-pass path).
///
///   2.  `dispatch_tile()` — runs once per (expert, per-thread N
///       range) inside the caller's OMP region.  Internally chunks the
///       N range into L2-friendly sub-tiles and drives the microkernel
///       through the pre-resolved function-pointer table.  No
///       validation, no mutex, no kernel-table switch in this path.
///
/// Callers only read `ctx.enabled` and `ctx.pack_nr` from the context
/// (the latter for column-split alignment); everything else is owned
/// by the dispatcher and the microkernel.

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_DISPATCH_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_DISPATCH_HPP

#include <array>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/data_types.hpp"
#include "common/error_status.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"
#include "ukernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

using zendnnl::common::bfloat16_t;
using zendnnl::common::data_type_t;
using zendnnl::error_handling::status_t;
using zendnnl::lowoha::matmul::grp_matmul_gated_act_t;

/// Quick CPUID gate (cached after first call).  Cheap; callers can
/// use this to early-out before building any inputs to
/// `prepare_for_call`.
bool dispatch_supported();

/// Pick the pack/microkernel NR for one (K, N) shape.  Returns either
/// `kNRMin` (32), `kNRMax` (64), or `0` when no supported NR divides N.
/// Honours the `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR` env override
/// (parsed via `get_grp_matmul_custom_kernel_nr()` in
/// `group_matmul_parallel_common.hpp`).  Public so `prepack/` and any
/// other auxiliary code can compute the same NR the dispatcher will
/// later pack/dispatch under, ensuring the cache key matches.
int plan_pack_nr(int K, int N);

/// Per-call invariants the per-tile path needs, resolved once.
/// The caller stack-allocates one of these and passes it to every
/// dispatch inside its OMP region.
struct CallContext {
  /// True only if `prepare_for_call()` succeeded — the caller reads
  /// this to decide whether to call `dispatch_tile()` or fall back.
  bool enabled = false;

  /// Pack/microkernel NR (32 or 64).  The caller reads this for the
  /// `aligned_n_split()` column-split alignment so each per-thread
  /// N-tile is a whole number of NR-blocks.
  int pack_nr = 0;

  // ── Internal — fields below are written by `prepare_for_call()`
  // and read only by `dispatch_tile()`.  Callers should not touch.
  int            NV            = 0;          // = pack_nr / 16
  int            max_mr        = 0;          // = max_mr_for_nv(NV)
  // Representative L2-friendly N-chunk width (worst case, sized from
  // the call's m_max).  Kept as a single value for APILOG / debug
  // output; the actual per-expert values live in `subtile_cols_per_expert`
  // below.  Dispatch reads per-expert, not this field.
  int            subtile_cols  = 0;
  ActKind        act_kind      = ActKind::none;
  BiasKind       bias_kind     = BiasKind::none;  // resolved from bias_dtype
  // Per-MR microkernel function pointers.  Slot 0 is unused (MR=0
  // would be a no-op); slots 1..kMaxMR hold the selected specializations.
  // Sized via `kMaxMR` so future max_mr bumps (e.g. int8 wider register
  // budget) only need a constant update in ukernel.hpp.
  ukernel_fn_t   kfn_table[kMaxMR + 1] = {};

  /// Maximum experts per call we cache packed pointers for.  Must
  /// match (or exceed) each caller's own expert-count cap.
  static constexpr int kMaxExperts = 256;
  std::array<const bfloat16_t *, kMaxExperts> packed_ptrs{};

  /// Per-expert L2-friendly N-chunk width (cols).  Sized individually
  /// so small-M experts (low A footprint) get a wider subtile with
  /// better B reuse, while large-M experts (higher A footprint) stay
  /// conservative.  Formula: same L2-budget arithmetic as the global
  /// `subtile_cols`, but using the expert's own M.  Populated in
  /// `prepare_for_call`; read directly in `dispatch_tile` via
  /// `packed_ptrs`-indexed `expert_idx`.  Zero for inactive experts.
  std::array<int, kMaxExperts> subtile_cols_per_expert{};
};

/// One-shot per-call prep (single-threaded).  See header doc for what
/// this function owns.  Caller passes the per-expert vectors it
/// already has on hand; the dispatcher reads what it needs.
///
/// `src_dtype` / `wei_dtype` / `dst_dtype` are the matmul A / B / C
/// dtypes (typically read from `params[0].dtypes.{src,wei,dst}`).  The
/// caller is expected to have already validated cross-expert uniformity;
/// the dispatcher trusts that.  The custom microkernel only implements
/// the `bf16 x bf16 -> bf16` math path (VDPBF16PS), so ALL THREE must be
/// bf16 for `out.enabled` to become true.  If any of them differs (e.g.
/// fp32 weights with bf16 dst on a mixed-precision call), the dispatcher
/// refuses and the caller falls back to its standard execution path —
/// without this gate the microkernel would silently reinterpret the
/// buffers as bf16 and produce corrupt output.
///
/// `act_dtype` is consulted only when `act != ActKind::none`; for
/// `act = none` the dispatcher skips the bf16 check so callers that
/// pass `act_dtype = none` (plain GEMM, no activation) are accepted
/// transparently.
///
/// `bias_dtype` must be one of `{none, bf16, f32}`.  `none` means no
/// bias at all (per-expert bias pointers are expected to be null).
/// Any other value causes `out.enabled` to be left false and the
/// caller falls back to its standard path.  The dispatcher resolves
/// this into `out.bias_kind` which the microkernel branches on once
/// per tile (no specialisation explosion).
///
/// `is_weights_const` is the framework's per-expert constancy hint
/// (matches the public API contract on `group_matmul_direct`).  The
/// custom kernel uses a process-wide pack cache keyed on the source
/// weight pointer; if a caller flags expert `i` as variable
/// (`is_weights_const[i] == false`) the cache may serve a stale pack
/// when the underlying weight buffer is mutated in-place between
/// calls.  The CK runtime has no per-expert "skip cache" branch
/// (unlike AOCL DLP's `run_dlp(...)`), so we conservatively REFUSE
/// the entire call when any active expert is non-const, falling
/// back to the standard AOCL DLP path which honours the flag at
/// runtime.  Empty `is_weights_const` means "treat every entry as
/// const" (legacy behaviour for callers that don't pass it).
///
/// On failure `out.enabled` is left false and the caller takes its
/// standard path (e.g. `execute_expert_slice` + separate activation).
status_t prepare_for_call(
    grp_matmul_gated_act_t act,
    data_type_t src_dtype,
    data_type_t wei_dtype,
    data_type_t dst_dtype,
    data_type_t act_dtype,
    data_type_t bias_dtype,
    const std::vector<bool>          &transA,
    const std::vector<bool>          &transB,
    const std::vector<int>           &M,
    const std::vector<int>           &N,
    const std::vector<int>           &K,
    const std::vector<int>           &ldb,
    const std::vector<float>         &alpha,
    const std::vector<float>         &beta,
    const std::vector<const void *>  &weight,
    const std::vector<bool>          &is_weights_const,
    CallContext &out);

// (`PackProbeStats` and `warm_pack_all_custom_kernel_experts` moved
// to `group_matmul/prepack/prepack_custom_kernel.{hpp,cpp}` so the
// dispatcher header keeps only the per-call public surface.  See
// that header for the warm-pack contract; the per-ALGO entries in
// `prepack/prepack.{hpp,cpp}` decide when to call it.)

/// Per-tile dispatch.  Runs inside the caller's OMP region.  Trusts
/// every input (the caller already vetted them through
/// `prepare_for_call()`), pulls the pre-resolved packed weight pointer
/// from `ctx.packed_ptrs[expert_idx]` and chunks the per-thread N
/// range into `ctx.subtile_cols` blocks before calling the microkernel.
///
/// Arguments mirror the per-tile call site in the caller:
///   * `expert_idx` — raw expert index (caller's `e`; per-expert order
///                    is independent of this API as long as
///                    `packed_ptrs[expert_idx]` was set during prep).
///   * `n_tile`     — width of this thread's N range (multiple of
///                    `ctx.pack_nr`, from `aligned_n_split`).
///   * `col_start`  — wide-N column offset of this thread's range
///                    within the expert's full N (multiple of
///                    `ctx.pack_nr`).
///   * `tight_dst`  — caller's output base for this expert.  When the
///                    context was prepared with act=swiglu_oai_mul
///                    this is the halved [M, N/2] tight arena;
///                    otherwise it is the wide [M, N] destination.
void dispatch_tile(
    const CallContext &ctx,
    int   expert_idx,
    int   M, int K,
    int   n_tile, int col_start,
    const void *src,  int lda,
    const void *bias,           // per `ctx.bias_kind`; nullptr when BiasKind::none
    void       *tight_dst, int tight_ldc);


} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_DISPATCH_HPP
