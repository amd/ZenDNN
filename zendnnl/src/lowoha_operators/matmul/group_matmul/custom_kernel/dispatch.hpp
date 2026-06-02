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
///
/// The dispatcher surface is intentionally narrow so the underlying
/// kernel implementation (MR/NV specialisations, pack format, ISA
/// targeting) can evolve without breaking callers.
///
/// ── Support matrix (act × dst_dtype) ───────────────────────────────
/// `prepare_for_call` accepts four activation values directly:
/// `none`, `swiglu_oai_mul`, `silu_and_mul`, and `gelu_and_mul`.
/// All three gated kinds use an in-register fused epilogue that
/// halves the output to `[M, N/2]`.  The split-halves variants
/// (silu_and_mul, gelu_and_mul) take caller-side canonical
/// `[gate_cols | up_cols]` W13; the prepack module permutes source
/// columns at pack time so the CK arena physically matches the
/// swiglu_oai_mul interleaved layout — only the kernel-side
/// activation math (silu vs gelu_tanh vs swiglu_oai) differs.
///
/// For BF16 src + BF16 wei (the only src/wei tuple the kernel serves
/// today; see `resolve_variant` for the full truth table) the
/// directly-served cells are:
///
///   * (act = none,           dst = bf16) — CK serves via
///       `kBF16_BF16_BF16` matmul-only; epilogue stores BF16.
///   * (act = none,           dst = f32 ) — CK serves via
///       `kBF16_BF16_F32` matmul-only; epilogue stores FP32.
///   * (act = swiglu_oai_mul, dst = bf16) — CK serves via
///       `kBF16_BF16_BF16` with the activation fused in the per-tile
///       epilogue.  Output is the half-width [M, N/2] BF16 tile.
///   * (act = silu_and_mul,   dst = bf16) — CK serves via
///       `kBF16_BF16_BF16` with `silu_and_mul_store_pair` fused in
///       the per-tile epilogue.  Half-width output.  No bias.
///   * (act = gelu_and_mul,   dst = bf16) — CK serves via
///       `kBF16_BF16_BF16` with `gelu_and_mul_store_pair` fused
///       (gelu_tanh polynomial form, within BF16 tolerance of
///       `gelu_erf`).  Half-width output.  No bias.
///   * (gated-act, dst = f32 ) — CK refuses (every gated-act
///       pair-pack store helper writes BF16 only).  Caller falls
///       back to AOCL DLP + a separate FP32 activation pass.
///   * (silu_and_mul / gelu_and_mul, +bias) — CK refuses
///       (`split_halves_act_with_bias_not_fused`).  Bias-into-init
///       under the prepack-permuted layout is a planned follow-up.
///
/// For cells the CK fused path doesn't serve, the production call
/// sequence falls back to a two-pass route (matmul then separate
/// activation):
///
///   * (act = silu_and_mul,   dst = f32) — `dst_dtype = f32`
///       refuses CK fusion (gated-store helpers write BF16 only).
///       Caller passes `act = none` to `prepare_for_call` (CK
///       serves the matmul via `kBF16_BF16_F32`), then runs
///       `group_matmul_moe_act_execute` over the wide
///       `[gate_cols | up_cols]` output.
///   * (act = gelu_and_mul,   dst = f32) — same as `silu_and_mul`.
///   * (silu/gelu, dst = bf16, +bias) — CK refuses
///       (`split_halves_act_with_bias_not_fused`).  Caller falls
///       back through `flat_n_tile`'s tight_split_halves Sequential
///       path (wide [M, N] scratch + apply_gated_act_inplace +
///       memcpy into the tight [M, I] dst).
///
/// The bf16-dst, no-bias silu/gelu cells DO take the direct CK
/// fused path — see the support matrix above for the cells the
/// kernel serves natively.
///
/// Cells the kernel does not serve fall back to AOCL DLP via the
/// caller's standard path (`execute_expert_slice` for non-fused,
/// the two-pass route for fused MoE).  The fallback is always
/// behaviourally equivalent — the distinction is only which
/// implementation runs the matmul and (for swiglu+f32) the
/// activation.
///
/// `bias_dtype` ∈ {none, bf16, f32} and `act_dtype` (when
/// `act != none`) must equal `bf16` for fused activation; the
/// non-fused activation pass handles its own dtype routing.

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_DISPATCH_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_DISPATCH_HPP

#include <array>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/data_types.hpp"
#include "common/error_status.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"
#include "ukernel/bf16_microkernel.hpp"
#include "ukernel/int8_microkernel.hpp"

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

// ────────────────────────────────────────────────────────────────────
// Kernel variant — which (src, wei, dst) tuple the dispatcher routes to
// ────────────────────────────────────────────────────────────────────
//
// Single per-call discriminator that selects the concrete microkernel
// instantiation `dispatch_tile()` calls.  Set by `prepare_for_call()`
// from the dtype tuple via `resolve_variant()`; read by
// `dispatch_tile()` to switch between kernel paths.
//
// `kBF16_BF16_BF16` and `kBF16_BF16_F32` are the bf16-family
// variants; `kS8_S8_BF16_SYM` / `kU8_S8_BF16_ASYM` (bf16 dst) and
// `kS8_S8_F32_SYM` / `kU8_S8_F32_ASYM` (f32 dst, act=none only) are
// the DQ-INT8 variants.  Any tuple outside these six resolves to
// `kUnsupported` and the caller falls back to DLP.
enum class KernelVariant : uint8_t {
  kUnsupported       = 0,  ///< Not supported by the custom kernel; caller falls back to DLP.
  kBF16_BF16_BF16    = 1,  ///< `bf16:bf16:bf16`.
  kBF16_BF16_F32     = 2,  ///< `bf16:bf16:f32`.
  kS8_S8_BF16_SYM    = 3,  ///< DQ-INT8 symmetric: src(hoisted s8) × wei(s8) → bf16.
  kU8_S8_BF16_ASYM   = 4,  ///< DQ-INT8 asymmetric: src(hoisted u8) × wei(s8) → bf16.
  kS8_S8_F32_SYM     = 5,  ///< DQ-INT8 symmetric: src(hoisted s8) × wei(s8) → f32.
  kU8_S8_F32_ASYM    = 6,  ///< DQ-INT8 asymmetric: src(hoisted u8) × wei(s8) → f32.
};

/// Predicate: is this variant in the DQ-INT8 family?
///
/// Useful at call sites that branch on bf16 vs DQ-INT8 ahead of the
/// `dispatch_tile()` call (e.g. the N-tile `do_tile()` body in
/// `group_matmul_n_tile.cpp` that builds the per-tile `src_scale` /
/// `src_zp` / `wei_scale` pointer triple only for the int8 path,
/// leaving them `nullptr` on the bf16 path).  Kept `noexcept` so the
/// optimiser folds it into a single `cmp + or` at every callsite.
inline bool is_int8_variant(KernelVariant v) noexcept {
  return v == KernelVariant::kS8_S8_BF16_SYM
      || v == KernelVariant::kU8_S8_BF16_ASYM
      || v == KernelVariant::kS8_S8_F32_SYM
      || v == KernelVariant::kU8_S8_F32_ASYM;
}

/// Map a (src, wei, dst, dynamic_quant, compute_dtype) tuple to a
/// `KernelVariant`.  Returns `kUnsupported` for any combination the
/// custom kernel cannot handle (caller then falls back to DLP).
///
/// Truth table:
///
///   * BF16 family (`dynamic_quant == false`):
///     `(bf16, bf16, bf16, _, _)`            → `kBF16_BF16_BF16`
///     `(bf16, bf16, f32 , _, _)`            → `kBF16_BF16_F32`
///
///   * DQ-INT8 family — accepted in TWO equivalent src forms:
///       1. Runtime hoist (legacy): `src == bf16 && dynamic_quant ==
///          true`; the N-tile executor reorder-quantises bf16 → s8/u8
///          (per-token src_scale) before `dispatch_tile`.
///       2. Grouped pre-quant (`ZENDNNL_ENABLE_GROUP_DQ`, default on):
///          the `group_dynamic_quant` pre-pass already produced an s8
///          src and CLEARED `dynamic_quant`, so `src == s8` arrives
///          directly (dynamic_quant may be false here).
///     `wei` is always `s8`; `compute_dtype` discriminates sym (s8, no
///     src_zp) vs asym (u8, with src_zp); `dst` selects the store dtype.
///     Writing `s8*` for "s8 (hoist) or bf16+dynamic_quant (runtime)":
///       bf16 dst:
///         `(s8*, s8, bf16, _, s8)`           → `kS8_S8_BF16_SYM`
///         `(s8*, s8, bf16, _, u8)`           → `kU8_S8_BF16_ASYM`
///       f32 dst (Act = none only):
///         `(s8*, s8, f32 , _, s8)`           → `kS8_S8_F32_SYM`
///         `(s8*, s8, f32 , _, u8)`           → `kU8_S8_F32_ASYM`
///
///   * any other tuple                        → `kUnsupported`
///
/// `noexcept` because it's a pure switch over POD enums — no
/// allocation, no I/O.
KernelVariant resolve_variant(data_type_t src, data_type_t wei,
                              data_type_t dst,
                              bool        dynamic_quant,
                              data_type_t compute_dtype) noexcept;

/// BF16-only legacy overload — kept for callers that have not been
/// updated to thread the `dynamic_quant` / `compute_dtype`
/// discriminators yet.  Internally calls the 5-arg form with
/// `dynamic_quant = false` and `compute_dtype = none`, which
/// reduces to the original two-row truth table.
inline KernelVariant resolve_variant(data_type_t src, data_type_t wei,
                                     data_type_t dst) noexcept {
  return resolve_variant(src, wei, dst,
                         /*dynamic_quant=*/false,
                         /*compute_dtype=*/data_type_t::none);
}

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

  /// Resolved kernel variant for this call.  Set by
  /// `prepare_for_call()` via `resolve_variant()`.  On the success
  /// path it is one of the six served variants (bf16 bf16/f32 dst, or
  /// DQ-INT8 sym/asym × bf16/f32 dst).  `dispatch_tile()` reads this
  /// to route to the correct kernel instantiation.
  KernelVariant variant = KernelVariant::kUnsupported;

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
  /// DQ-INT8 quant flavour — set by `prepare_for_call()` to
  /// `kU8_Asym` for the asymmetric variants (bf16- or f32-dst) and
  /// `kS8_Sym` otherwise; ignored on the bf16 path.  Baked into the
  /// single `kfn_table_int8` per-MR table by `fill_kfn_table_int8`
  /// (there is one table, not separate sym/asym tables), so the
  /// inner loop does not re-touch the variant discriminator.
  IntCompute     compute_int   = IntCompute::kS8_Sym;
  /// Dtype of the src/wei scale buffers the microkernel will read.
  /// Set by the N-tile hoist (the dispatcher cannot see the scale
  /// dtype): `kBf16` when the raw bf16 scales are passed straight
  /// through (swiglu_oai_mul / none — the common gpt-oss path, kernel
  /// converts on load), `kF32` when the scales are already f32 OR were
  /// converted+permuted to f32 by the silu/gelu interleave pre-pass.
  /// Ignored on the bf16 (non-quant) path.
  ScaleKind      scale_kind    = ScaleKind::kF32;
  // Per-MR microkernel function pointers.  Slot 0 is unused (MR=0
  // would be a no-op); slots 1..kMaxMR hold the selected specializations.
  // Sized via `kMaxMR` so any future max_mr bump only needs a constant
  // update in `ukernel/bf16_microkernel.hpp`.
  //
  // BF16 family uses `kfn_table`; DQ-INT8 family uses
  // `kfn_table_int8` (one entry per MR, already specialised on
  // `compute_int` + `act_kind` at `prepare_for_call` time).  Only
  // one of the two tables is populated per call (the other stays
  // zero-initialised); `dispatch_tile()` reads the right table
  // off `variant`.
  ukernel_fn_t       kfn_table[kMaxMR + 1]      = {};
  int8_ukernel_fn_t  kfn_table_int8[kMaxMR + 1] = {};

  /// Maximum experts per call we cache packed pointers for.  Must
  /// match (or exceed) each caller's own expert-count cap.
  static constexpr int kMaxExperts = 256;
  std::array<const bfloat16_t *, kMaxExperts> packed_ptrs{};
  /// DQ-INT8 packed-weight pointers (signed `int8_t *`).  Populated
  /// when the variant is `kS8_S8_BF16_SYM` / `kU8_S8_BF16_ASYM`;
  /// stays all-null on the bf16 path.  Layout per o-block is
  /// `[K_pad/4][pack_nr][4]` weight bytes followed by `[pack_nr]
  /// int32` per-column compensation (see pack.hpp); `dispatch_tile()`
  /// passes the raw `int8_t *` to the microkernel which reads the
  /// compensation row by byte arithmetic.
  std::array<const int8_t *, kMaxExperts> packed_ptrs_int8{};

  /// Per-expert L2-friendly N-chunk width (cols).  Sized individually
  /// so small-M experts (low A footprint) get a wider subtile with
  /// better B reuse, while large-M experts (higher A footprint) stay
  /// conservative.  Formula: same L2-budget arithmetic as the global
  /// `subtile_cols`, but using the expert's own M.  Populated in
  /// `prepare_for_call`; read directly in `dispatch_tile` via
  /// `packed_ptrs`-indexed `expert_idx`.  Zero for inactive experts.
  std::array<int, kMaxExperts> subtile_cols_per_expert{};

  /// Caller-owned packed-weight pointers — populated only when the
  /// library-wide weight-cache toggle is off
  /// (`matmul_config_t::get_weight_cache() != 1`).  In that mode
  /// `prepare_for_call()` routes each per-expert pack through
  /// `get_or_pack_weight_bf16(..., disable_cache=true)`, which
  /// allocates a fresh aligned buffer per call and skips the LRU
  /// singleton; the resulting raw pointer is stored here AND
  /// aliased into `packed_ptrs[i]` so `dispatch_tile()` reads it
  /// transparently (the dispatcher cannot distinguish a cache-
  /// served pointer from a caller-owned one — that's by design).
  ///
  /// Lifetime: owned by this `CallContext` instance.
  /// `release_owned_buffers()` frees every non-null slot via
  /// `free_owned_packed_weight()` and zeroes the array; the
  /// destructor calls it unconditionally, and `prepare_for_call()`
  /// calls it before its `out = CallContext{}` reset so a context
  /// reused across calls does not leak the previous call's
  /// buffers.  In the cache-enabled mode (the default) every slot
  /// stays `nullptr` and `release_owned_buffers()` is a cheap no-op.
  std::array<const bfloat16_t *, kMaxExperts> owned_packed_ptrs{};
  /// DQ-INT8 sibling of `owned_packed_ptrs` — caller-owned int8
  /// packed-weight pointers, used in the `weight_cache_type != 1`
  /// branch.  Freed via `free_owned_packed_weight_int8()`.  Same
  /// lifetime contract as the bf16 array; the destructor /
  /// `release_owned_buffers()` zero both on exit.
  std::array<const int8_t *, kMaxExperts> owned_packed_ptrs_int8{};

  /// Free every caller-owned packed buffer this context holds and
  /// zero the `owned_packed_ptrs` array.  Idempotent and safe to
  /// call at any time (frees only non-null slots).  Does NOT touch
  /// `packed_ptrs` — when a slot in `owned_packed_ptrs` is non-null
  /// the matching `packed_ptrs[i]` aliases it and becomes dangling
  /// after this call, which is the expected post-condition: the
  /// dispatcher must not read `packed_ptrs` after a successful
  /// `release_owned_buffers()` unless `prepare_for_call()` has
  /// repopulated the context.
  void release_owned_buffers();

  /// Reset every field to its post-construction default WITHOUT
  /// leaking caller-owned packed buffers from a previous use of
  /// the same context.  Order is important: free first (via
  /// `release_owned_buffers()`), THEN zero the rest of the state.
  /// Used by `prepare_for_call()` to start each call from a clean
  /// slate when a single `CallContext` is reused across multiple
  /// `group_matmul_direct(...)` invocations.  Replaces the
  /// previous `out = CallContext{}` idiom that relied on a default
  /// move-assignment operator — that operator is now deleted to
  /// prevent silent double-frees on the owning pointer array.
  void reset();

  /// Non-copyable / non-movable.  `release_owned_buffers()` runs in
  /// the destructor and freeing the same pointer twice would crash
  /// the process; defaulting copy/move would silently duplicate
  /// the owning pointers.  Callers stack-allocate a single
  /// `CallContext` per `prepare_for_call() + dispatch_tile()`
  /// scope (the typical idiom for both `group_matmul_direct` and
  /// `group_matmul_fused_moe`); state reuse across calls goes
  /// through `reset()` above.
  CallContext() = default;
  ~CallContext() { release_owned_buffers(); }
  CallContext(const CallContext &)            = delete;
  CallContext &operator=(const CallContext &) = delete;
  CallContext(CallContext &&)                 = delete;
  CallContext &operator=(CallContext &&)      = delete;
};

/// One-shot per-call prep (single-threaded).  See header doc for what
/// this function owns.  Caller passes the per-expert vectors it
/// already has on hand; the dispatcher reads what it needs.
///
/// SCOPE NOTE — the dtype contract (single source of truth).
///   `src_dtype` / `wei_dtype` / `dst_dtype` are the matmul A / B / C
///   dtypes (typically read from `params[0].dtypes.{src,wei,dst}`).
///   The dispatcher trusts the caller has already validated cross-
///   expert uniformity (every expert in the call shares these dtypes).
///
///   The (src, wei, dst) tuple is routed through `resolve_variant()`,
///   which is the SINGLE source of truth for what the custom kernel
///   can serve.  Its current truth table:
///
///     SUPPORTED (returns `out.enabled = true`):
///       (bf16, bf16, bf16)  -> kBF16_BF16_BF16
///       (bf16, bf16, f32 )  -> kBF16_BF16_F32
///
///     REJECTED (returns failure -> caller falls back to DLP):
///       Anything else, including:
///         (f32 , f32 , *   )  -- no FP32 GEMM in this kernel
///         (f32 , bf16, *   )  -- mixed-precision src
///         (bf16, bf16, f16 )  -- F16 dst not implemented
///         any tuple involving `data_type_t::u8` / `f16` / `s8`
///
///   On a rejected tuple the caller is expected to take its standard
///   path (e.g., AOCL DLP via `execute_expert_slice`) — failure to
///   do so will trip the `ctx.variant != kUnsupported` debug assert
///   in `dispatch_tile()`.
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
/// Activation contract:
///   * `none` — plain matmul.  Both BF16 and F32 dst admitted.
///   * `swiglu_oai_mul` — fused in the per-tile epilogue (caller-
///     side interleaved [g0,u0,g1,u1,...] layout puts (gate, up)
///     pairs on every 32-col tile).  Halved output width.
///     `dst_dtype = bf16` ONLY (the swiglu store helper writes
///     BF16; (swiglu, f32-dst) is rejected inside `select_ukernel`).
///   * `silu_and_mul` / `gelu_and_mul` — ALSO fused in the per-tile
///     epilogue (via `silu_and_mul_store_pair` and
///     `gelu_and_mul_store_pair` respectively).  Caller passes
///     canonical split-halves `[gate_cols | up_cols]` W13; the
///     prepack module re-permutes source columns so the CK arena
///     physically matches the swiglu_oai_mul interleaved layout —
///     silu and gelu only differ in the kernel-side activation
///     math, not in the pack layout.  Halved output width.
///     Refusal conditions, both routed back through the caller's
///     fallback path:
///       * `dst_dtype = f32` (gated-store helpers write BF16
///         only) → caller takes the two-pass route described in
///         the file header (act = none + `group_matmul_moe_act_
///         execute`).
///       * `bias_dtype != none` (bias-into-init under the
///         prepack-permuted layout is a planned follow-up) →
///         `flat_n_tile` routes to the Sequential tight_split_
///         halves fallback.
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
/// `dynamic_quant` + `compute_dtype` discriminate the DQ-INT8
/// family from the BF16 family.  Both default to "off / none" so
/// existing callers that haven't been ported to the int8 path
/// continue to engage the bf16 microkernel unchanged.
///
/// DQ-INT8 acceptance (mirrored from `resolve_variant`):
///   * `(src=bf16, wei=s8, dst=bf16, dynamic_quant=true,
///      compute_dtype=s8)` → `kS8_S8_BF16_SYM` (per-token
///      symmetric — no src_zp expected at dispatch time).
///   * `(src=bf16, wei=s8, dst=bf16, dynamic_quant=true,
///      compute_dtype=u8)` → `kU8_S8_BF16_ASYM` (per-token
///      asymmetric — caller will pass a non-null `src_zp` to
///      `dispatch_tile()`).
/// Any other (dynamic_quant=true) tuple falls through to
/// `kUnsupported` and the caller's standard path takes over.
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
    CallContext &out,
    bool         dynamic_quant   = false,
    data_type_t  compute_dtype   = data_type_t::none);

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
/// `src_scale` / `src_zp` / `wei_scale` are consumed only on the
/// DQ-INT8 path (`ctx.variant ∈ {kS8_S8_BF16_SYM, kU8_S8_BF16_ASYM}`).
/// On the BF16 path they must be `nullptr` (the caller can either
/// omit the args entirely — they default to `nullptr` — or pass
/// `nullptr` explicitly).  Layout contract (DQ-INT8 only):
///   * `src_scale` — `M` floats (one per row of the expert's A).
///                   The dispatcher slices to the per-call MR
///                   window before invoking the microkernel.
///   * `src_zp`    — `M` int32s, OR nullptr for symmetric calls.
///                   Required when variant is `kU8_S8_BF16_ASYM`,
///                   ignored when variant is `kS8_S8_BF16_SYM`.
///   * `wei_scale` — `N` floats (one per output column).  The
///                   dispatcher slices to the per-(o-block) NR
///                   window before invoking the microkernel.
void dispatch_tile(
    const CallContext &ctx,
    int   expert_idx,
    int   M, int K,
    int   n_tile, int col_start,
    const void *src,  int lda,
    const void *bias,           // per `ctx.bias_kind`; nullptr when BiasKind::none
    void       *tight_dst, int tight_ldc,
    const void    *src_scale  = nullptr,
    const int32_t *src_zp     = nullptr,
    const void    *wei_scale  = nullptr);


} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_DISPATCH_HPP
