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

#include "dispatch.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "../group_matmul_parallel_common.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "pack.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

using zendnnl::error_handling::apilog_verbose;
using zendnnl::error_handling::apilog_verbose_enabled;

bool dispatch_supported() {
  return avx512bf16_available();
}

// ─────────────────────────────────────────────────────────────────────
// Convenience predicate — variant is in the DQ-INT8 family:
//   {kS8_S8_BF16_SYM, kU8_S8_BF16_ASYM,   // bf16 dst
//    kS8_S8_F32_SYM,  kU8_S8_F32_ASYM}.   // f32 dst (act=none only)
// Hoisted to one place so dispatch.cpp, the warmer, and the
// debug-assert in `dispatch_tile()` agree on the int8 family
// membership.  Keeping it `inline` (no namespace qualifier) lets the
// optimiser fold the call into a single `cmp + or` at every callsite.
// `is_int8_variant` now lives in the public header (`dispatch.hpp`)
// so call sites in the N-tile executor can branch on it without
// touching dispatch internals; this TU keeps using the header-level
// definition.
// ─────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────
// CallContext lifecycle — `release_owned_buffers()` + `reset()`.
//
// The destructor + the in-place reset are the only places the
// caller-owned packed buffers can be freed (the `disable_cache=true`
// branch of `get_or_pack_weight_bf16()` skips the LRU singleton, so
// no other code path knows these pointers exist).  Implementing both
// in dispatch.cpp keeps the `<array>` member's destructor trivial
// while routing the actual free through `free_owned_packed_weight()`
// (pack.cpp's matched companion to `aligned_alloc` from the
// disable-cache branch).
// ─────────────────────────────────────────────────────────────────────
void CallContext::release_owned_buffers() {
  // Iterate every slot — both `owned_packed_ptrs` and `packed_ptrs`
  // are zero-initialised on construction, so the bulk of slots cost
  // one branch each on a typical (active_count << kMaxExperts) call.
  // The `free_owned_packed_weight` helper itself is null-safe so the
  // inner check is only an optimisation to skip the call entirely
  // when the slot is empty.
  for (auto &ptr : owned_packed_ptrs) {
    if (ptr != nullptr) {
      free_owned_packed_weight(ptr);
      ptr = nullptr;
    }
  }
  // DQ-INT8 sibling — symmetric semantics with a different free
  // sink (the int8 cache and the bf16 cache are disjoint singletons
  // and their disable-cache buffers come from different
  // aligned_alloc'ed regions tagged by the per-pack family in
  // `pack.cpp`).
  for (auto &ptr : owned_packed_ptrs_int8) {
    if (ptr != nullptr) {
      free_owned_packed_weight_int8(ptr);
      ptr = nullptr;
    }
  }
}

void CallContext::reset() {
  // Free caller-owned packs FIRST so subsequent field zeroing does
  // not strand the pointers.  `release_owned_buffers()` clears
  // `owned_packed_ptrs` itself, so the loops below only need to
  // touch the remaining state.
  release_owned_buffers();
  enabled       = false;
  variant       = KernelVariant::kUnsupported;
  pack_nr       = 0;
  NV            = 0;
  max_mr        = 0;
  subtile_cols  = 0;
  act_kind      = ActKind::none;
  bias_kind     = BiasKind::none;
  compute_int   = IntCompute::kS8_Sym;
  // Restore the same safe default as the member initializer: f32 scales.
  // `dispatch_tile_int8()` derives src/wei scale byte strides from
  // `scale_kind`, so a stale value across a CallContext reuse would read
  // scales at the wrong element width if the next caller forgot to set it.
  scale_kind    = ScaleKind::kF32;
  for (auto &fn : kfn_table)      fn = nullptr;
  for (auto &fn : kfn_table_int8) fn = nullptr;
  packed_ptrs.fill(nullptr);
  packed_ptrs_int8.fill(nullptr);
  subtile_cols_per_expert.fill(0);
}

// ────────────────────────────────────────────────────────────────────
// Variant resolution — pure switch over the (src, wei, dst) dtype
// tuple.  No I/O, no allocation, no logging — `prepare_for_call()`
// owns the logging (this helper is also called from gtests that
// want to assert the routing without any side effect).
//
// The BF16-BF16-BF16 and BF16-BF16-F32 rows are wired up.  Any other
// tuple falls through to `kUnsupported`, which the caller routes to
// DLP.
// ────────────────────────────────────────────────────────────────────
KernelVariant resolve_variant(data_type_t src, data_type_t wei,
                              data_type_t dst,
                              bool        dynamic_quant,
                              data_type_t compute_dtype) noexcept {
  // BF16 family — `dynamic_quant` MUST be false (any
  // dynamic_quant=true call must go through the DQ-INT8 branch
  // below; we keep these mutually exclusive to avoid silent
  // mis-routing).
  if (!dynamic_quant
      && src == data_type_t::bf16
      && wei == data_type_t::bf16) {
    if (dst == data_type_t::bf16) return KernelVariant::kBF16_BF16_BF16;
    if (dst == data_type_t::f32 ) return KernelVariant::kBF16_BF16_F32;
  }
  // DQ-INT8 family — two ways the per-token-quantized s8/u8 src reaches
  // the CK microkernel, both with `wei == s8` and the same dequant math:
  //
  //   1. Runtime hoist (legacy): caller passes bf16 src with
  //      `dynamic_quant = true`; the N-tile executor hoists bf16 → s8/u8
  //      (per-token src_scale) before `dispatch_tile`.
  //   2. Grouped pre-quant (ZENDNNL_ENABLE_GROUP_DQ, default on): the
  //      `group_dynamic_quant` pre-pass already quantized the src to s8,
  //      set `dtypes.src = s8`, cleared `dynamic_quant`, and left a
  //      per-token `src_scale` buffer in params.  The src arrives here
  //      ALREADY s8, so accept `src == s8` directly — `check_n_tile_extra`
  //      has already validated the grouped per-token src_scale + per-
  //      channel wei_scale shape, so this only fires on the DQ-INT8
  //      grouped form (a static s8s8 GEMM would have dst = s8, not
  //      bf16/f32, and is rejected by the dst checks below).
  //
  // Discriminator on the compute dtype (identical for both):
  //   * `compute_dtype = s8` → kS8_S8_BF16_SYM  (sym, no src_zp).
  //   * `compute_dtype = u8` → kU8_S8_BF16_ASYM (asym, with src_zp).
  const bool dq_int8_src =
      (dynamic_quant && src == data_type_t::bf16)   // (1) runtime hoist
      || (src == data_type_t::s8);                  // (2) grouped pre-quant
  if (dq_int8_src
      && wei == data_type_t::s8) {
    if (dst == data_type_t::bf16) {
      if (compute_dtype == data_type_t::s8)
        return KernelVariant::kS8_S8_BF16_SYM;
      if (compute_dtype == data_type_t::u8)
        return KernelVariant::kU8_S8_BF16_ASYM;
    }
    // FP32 dst — direct store of the dequantised accumulator (Act=none
    // only; gated kinds stay BF16 and are refused in fill_kfn_table_int8
    // / the microkernel selector).  Mirror of the bf16 family's
    // bf16/f32 dst split.
    if (dst == data_type_t::f32) {
      if (compute_dtype == data_type_t::s8)
        return KernelVariant::kS8_S8_F32_SYM;
      if (compute_dtype == data_type_t::u8)
        return KernelVariant::kU8_S8_F32_ASYM;
    }
  }
  return KernelVariant::kUnsupported;
}

// Helper to pick the kernel-store dtype (DstDt) from a resolved
// variant.  Lives in this TU so dispatch.cpp's `fill_kfn_table` and
// `dispatch_tile` agree with `resolve_variant` on the variant→DstDt
// mapping (single source of truth: the table here).
namespace {
DstDt dst_dt_for_variant(KernelVariant v) noexcept {
  switch (v) {
    case KernelVariant::kBF16_BF16_BF16: return DstDt::kBf16;
    case KernelVariant::kBF16_BF16_F32:  return DstDt::kF32;
    case KernelVariant::kS8_S8_BF16_SYM:  return DstDt::kBf16;
    case KernelVariant::kU8_S8_BF16_ASYM: return DstDt::kBf16;
    case KernelVariant::kS8_S8_F32_SYM:   return DstDt::kF32;
    case KernelVariant::kU8_S8_F32_ASYM:  return DstDt::kF32;
    default: return DstDt::kBf16;  // unreachable on the success path
  }
}
}  // namespace

// Pick the pack/microkernel NR for one (K, N) shape.  Env override
// (from `get_grp_matmul_custom_kernel_nr()` in parallel_common.hpp)
// wins when it divides N; otherwise default to NR=32 — it gives a
// clean register budget against the 32-zmm budget on AVX-512 and
// keeps M chunking small on small-M decode shapes.  NR=64 stays
// available via env override for shapes that benefit.
//
// Now a public symbol (declared in dispatch.hpp) so the warm-pack
// helpers in `group_matmul/prepack/` can compute the same NR the
// dispatcher will pack under, keeping cache keys aligned.
int plan_pack_nr(int /*K*/, int N) {
  if (N <= 0) return 0;
  const int env_nr = get_grp_matmul_custom_kernel_nr();
  if (env_nr != 0) return ((N % env_nr) == 0) ? env_nr : 0;
  if ((N % kNRMin) == 0) return kNRMin;
  if ((N % kNRMax) == 0) return kNRMax;
  return 0;
}

// ─────────────────────────────────────────────────────────────────────
// Internal helpers — owned entirely by this file.  The callers never call
// these directly; the only public entries are `prepare_for_call()`
// and `dispatch_tile()` below.
// ─────────────────────────────────────────────────────────────────────
namespace {

// L2-friendly sub-tile width (in cols) for (M_max, K, pack_nr).
// Sized so each microkernel call's working set — input + the B
// strip this thread streams — fits in this CPU's per-core L2 with
// some headroom (`l2_budget` uses a 4/5 fraction of detected L2 to
// leave room for caller-side spills).  Accumulators live in zmm
// registers, so they don't enter the budget.  Returned value is a
// multiple of pack_nr, minimum one o-block.
//
// `src_elem_bytes` and `wei_elem_bytes` describe the per-element
// byte volume the kernel consumes:
//   * BF16 family       → src=2, wei=2, comp_bytes_per_col=0.
//   * DQ-INT8 family    → src=1 (the hoisted s8/u8 stripe the int8
//     microkernel reads), wei=1, comp_bytes_per_col=4 (the int32
//     compensation row appended after each o-block — adds a
//     constant per-column overhead independent of K).
// Without dtype-awareness the bf16 byte model (2,2,0) was used for
// the int8 family too, over-budgeting the working set by 2x and
// shrinking subtiles below their byte-budget optimum.  See R2 in
// `analysis_scratch/int8_ntile_audit.md`.
int pick_l2_subtile_cols(int M_max, int K, int pack_nr,
                         int src_elem_bytes, int wei_elem_bytes,
                         int comp_bytes_per_col) {
  if (M_max <= 0 || K <= 0 || pack_nr <= 0
      || src_elem_bytes <= 0 || wei_elem_bytes <= 0) return pack_nr;
  static const int64_t l2_budget = []() {
    const int64_t l2 =
        zendnnl::lowoha::matmul::native::detect_uarch().l2_bytes;
    const int64_t safe = (l2 >= 64 * 1024) ? l2 : (256 * 1024);
    return (safe * 4) / 5;
  }();

  const int64_t input_bytes =
      static_cast<int64_t>(M_max) * K * src_elem_bytes;
  const int64_t weight_bytes_per_col =
      static_cast<int64_t>(K) * wei_elem_bytes
      + static_cast<int64_t>(comp_bytes_per_col);
  if (l2_budget <= input_bytes || weight_bytes_per_col <= 0) {
    return pack_nr;
  }
  const int64_t cols_raw =
      (l2_budget - input_bytes) / weight_bytes_per_col;
  const int cols_aligned =
      static_cast<int>((cols_raw / pack_nr) * pack_nr);
  return std::max(cols_aligned, pack_nr);
}

// Per-variant byte model used by `pick_l2_subtile_cols`.  The
// switch is constexpr-friendly so the subtile selection turns
// into two branchless multiplies + one constant load on the
// hot prepare-for-call path; the runtime overhead vs the previous
// hard-coded `sizeof(uint16_t)` is microscopic.
struct SubtileBytes {
  int src_bytes_per_elem;
  int wei_bytes_per_elem;
  int comp_bytes_per_col;
};
inline SubtileBytes subtile_bytes_for_variant(KernelVariant v) {
  if (is_int8_variant(v)) {
    // The hoisted src is s8/u8 (1 byte) — the int8 microkernel
    // reads it directly; the bf16 caller src is replaced by the
    // pre-OMP hoist before the kernel runs.
    // Comp row is int32 per-column appended after each o-block.
    return {1, 1, 4};
  }
  return {2, 2, 0};
}

// Resolve the per-MR microkernel function-pointer table for one
// (NV, act_kind, dst_dt) tuple.  Called once by `prepare_for_call()`.
// Per-tile lookup is then a direct `kfn_table[mr_now]` — no switch.
status_t fill_kfn_table(
    int NV, ActKind act_kind, DstDt dst_dt,
    ukernel_fn_t (&kfn_table)[kMaxMR + 1]) {
  const int max_mr = max_mr_for_nv(NV);
  for (int mr = 1; mr <= max_mr; ++mr) {
    kfn_table[mr] = select_ukernel(mr, NV, act_kind, dst_dt);
    if (kfn_table[mr] == nullptr) return status_t::failure;
  }
  return status_t::success;
}

// DQ-INT8 sibling of `fill_kfn_table` — same loop shape, different
// selector (`select_int8_ukernel`).  Carries both the `compute`
// (kS8_Sym / kU8_Asym) axis and the `dst_dt` (kBf16 / kF32) axis;
// the selector returns nullptr for any (gated_act, kF32) tuple so a
// mis-resolved f32 gated variant refuses here rather than at runtime.
status_t fill_kfn_table_int8(
    int NV, IntCompute compute, ActKind act_kind, DstDt dst_dt,
    int8_ukernel_fn_t (&kfn_table)[kMaxMR + 1]) {
  const int max_mr = max_mr_for_nv(NV);
  for (int mr = 1; mr <= max_mr; ++mr) {
    kfn_table[mr] = select_int8_ukernel(mr, NV, compute, act_kind, dst_dt);
    if (kfn_table[mr] == nullptr) return status_t::failure;
  }
  return status_t::success;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────
// Public entries
// ─────────────────────────────────────────────────────────────────────

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
    bool         dynamic_quant,
    data_type_t  compute_dtype) {

  // Reset the full CallContext to defaults on entry.  Callers may
  // reuse a single context across calls (e.g. an OMP region that
  // dispatches multiple group_matmul_direct calls back-to-back), so
  // we cannot rely on the destructor to clear stale state from a
  // previous successful prepare.  If we only cleared `enabled` and
  // a refusal exited before assigning `variant` / `pack_nr` /
  // `kfn_table[...]` / `packed_ptrs[...]` / `subtile_cols_per_expert[...]`,
  // those fields would carry over from the prior call — making any
  // future read that bypasses the `enabled` gate (a debug assert,
  // an APILOG line, or a refactor that misses the gate) silently
  // observe a stale-but-plausible config.  `reset()` value-init's
  // every field via the in-class defaults declared in
  // `CallContext` (dispatch.hpp): `enabled = false`, `variant =
  // kUnsupported`, scalar fields = 0 / sentinel, fn-pointer table
  // and the std::array members zero-filled.  Critically `reset()`
  // also calls `release_owned_buffers()` BEFORE zeroing
  // `owned_packed_ptrs` so a previous call's caller-owned packed
  // weights (disable-cache mode) are freed rather than stranded.
  // Cost: ~2 KB of zeroing once per call on the cold prep path,
  // negligible vs the surrounding kernel work.
  out.reset();

  // ── Refusal logging helper ───────────────────────────────────────
  // Every early return below represents a concrete dispatch-contract
  // violation that forces the caller to fall back to the standard
  // AOCL / BRGEMM path.  Log the reason so a model-level read makes
  // it immediately obvious why the custom kernel was skipped — a
  // silent refusal just produced `kernel=standard` with no clue
  // whether the env was off, a dtype mismatched, or the pack-NR
  // check failed.
  //
  // Gated on the verbose level (`ZENDNNL_API_LOG_LEVEL=4`): these
  // lines fire per-expert per-call from inside the ALGO 3 OMP region,
  // so they belong on the per-thread / per-event channel.  Info
  // level (3) stays clean and shows only the consolidated planner
  // and prepack summary lines.  Single cached `apilog_verbose_enabled()`
  // check.
  //
  // Lambda overhead note: `dt_name` is captureless (`[]`) — a
  // stateless functor with zero runtime construction cost.  `refuse`
  // is `[&]` cosmetically (see capture-note paragraph below for why):
  // `s_refuse_log` has STATIC storage duration (function-local
  // `static const`), so a captureless `[]` for `refuse` would also
  // compile and reference it correctly per
  // [expr.prim.lambda.capture] — captures are only required for
  // variables with AUTOMATIC storage duration.  `[&]` produces the
  // same closure body in this case (no actual captures are emitted
  // because `s_refuse_log` is still static-duration), so the
  // construction cost is still zero.  The `[&]` form is used purely
  // to silence static-analysis tooling that mis-reads the
  // captureless variant as a capture omission.
  //
  // Net impact on a successful `prepare_for_call` with logging
  // disabled: zero — neither lambda object is ever invoked on the
  // success path, and `dt_name` only gets called from the failure
  // paths that also read `s_refuse_log`.
  static const bool s_refuse_log = apilog_verbose_enabled();
  auto refuse = [&](const char *reason_tag,
                    const char *detail = nullptr) -> status_t {
    if (s_refuse_log) {
      if (detail != nullptr) {
        apilog_verbose("[GRP_MATMUL.CK REFUSED] reason=",
                       reason_tag, " (", detail, ")");
      } else {
        apilog_verbose("[GRP_MATMUL.CK REFUSED] reason=",
                       reason_tag);
      }
    }
    return status_t::failure;
  };
  auto dt_name = [](data_type_t dt) -> const char * {
    switch (dt) {
    case data_type_t::none: return "none";
    case data_type_t::f32:  return "f32";
    case data_type_t::bf16: return "bf16";
    case data_type_t::f16:  return "f16";
    case data_type_t::s8:   return "s8";
    case data_type_t::u8:   return "u8";
    default:                return "?";
    }
  };

  // ── Run-once invariants (CPU + dtypes + activation) ──────────────
  // The dispatcher serves two ISA families: bf16 variants need
  // AVX-512 BF16 (VDPBF16PS); DQ-INT8 variants need AVX-512 VNNI
  // (VPDPBUSD).  On the Zen 4/5 targets VNNI is a strict superset of
  // BF16, but on broader x86 (e.g. Cascade Lake / Ice Lake) VNNI
  // exists WITHOUT BF16.  Refuse early ONLY when NEITHER family's ISA
  // is present; the precise per-variant ISA gate runs after
  // `resolve_variant()` below (bf16 -> BF16, int8 -> VNNI) so a
  // VNNI-only host can still serve the DQ-INT8 fast path.
  if (!avx512bf16_available() && !avx512vnni_available())
    return refuse("no_avx512_bf16_or_vnni");

  // ── Library-wide weight-cache toggle ─────────────────────────────
  // `ZENDNNL_MATMUL_WEIGHT_CACHE` (read via
  // `matmul_config_t::get_weight_cache()`) is the process-wide
  // contract for "the caller is allowed to cache weight-derived
  // packed buffers across calls".  The CK pack arena
  // (`custom_kernel/pack.cpp::pack_cache_singleton`) is keyed on the
  // raw user weight pointer and has eviction disabled
  // (UINT32_MAX capacity), so when the framework mutates weight
  // addresses between calls — exactly the scenario the env knob is
  // designed for — a stale pointer key would collide with a fresh
  // upload and the dispatcher would read freed memory in
  // `dispatch_tile`.
  //
  // Resolution: when the toggle is non-1 we DO NOT refuse CK.
  // Instead each per-expert pack below is routed through
  // `get_or_pack_weight_bf16(..., disable_cache=true)`, which
  // allocates a fresh aligned buffer, packs into it, and returns
  // the raw pointer without touching the LRU singleton.  The raw
  // pointers land in `out.owned_packed_ptrs[i]` AND
  // `out.packed_ptrs[i]` (alias), so `dispatch_tile()` reads them
  // transparently and the `CallContext` destructor /
  // `release_owned_buffers()` frees them once the OMP region
  // unwinds.  Cost: per-call pack work, no inter-call amortisation,
  // no stale pointer hits — exactly the semantic the env knob
  // promises.
  //
  // The matching prepack-side gate lives in
  // `prepack/prepack_custom_kernel.cpp::warm_pack_all_custom_kernel_experts`
  // (skip warming when the toggle is non-1 since the warmer can
  // only populate the LRU cache, which the runtime will not
  // consult); the prepack fingerprint
  // (`prepack/prepack.cpp::fingerprint`) folds the toggle into its
  // hash so flipping the env mid-process re-arms both code paths
  // on the next call.
  const bool cache_off =
      (zendnnl::ops::matmul_config_t::instance().get_weight_cache() != 1);
  if (cache_off && s_refuse_log) {
    apilog_verbose("[GRP_MATMUL.CK NOCACHE] "
                   "weight_cache_type != 1 — packing into per-call "
                   "caller-owned buffers (LRU singleton bypassed)");
  }

  // A / B / C dtype gate: route through `resolve_variant()` which is
  // the single source of truth for "which (src, wei, dst, dynamic_quant,
  // compute_dtype) tuples the custom kernel can serve".  `kUnsupported`
  // here means the caller falls back to DLP.  Routing through the
  // variant keeps the dispatcher's gate aligned with the gtest matrix
  // in `gtests/group_matmul/custom_kernel/`, which calls
  // `resolve_variant()` directly to assert the table.
  //
  // DQ-INT8 family additionally requires the AVX-512 VNNI feature
  // (VPDPBUSD), which is a strict superset of AVX-512 BF16 on the
  // Zen 4/5 microarchitectures we target but on broader x86 silicon
  // is an independent feature flag.  Refuse cleanly with a specific
  // tag so the user can distinguish "no AVX-512 VNNI" from "wrong
  // dtype" in apilog.
  const KernelVariant variant =
      resolve_variant(src_dtype, wei_dtype, dst_dtype,
                      dynamic_quant, compute_dtype);
  if (variant == KernelVariant::kUnsupported) {
    if (s_refuse_log) {
      apilog_verbose("[GRP_MATMUL.CK REFUSED] reason="
                     "unsupported_dtype (src=", dt_name(src_dtype),
                     " wei=", dt_name(wei_dtype),
                     " dst=", dt_name(dst_dtype),
                     " dynamic_quant=", (dynamic_quant ? 1 : 0),
                     " compute=", dt_name(compute_dtype),
                     " — see resolve_variant() in custom_kernel/dispatch.cpp"
                     " for the supported table)");
    }
    return status_t::failure;
  }
  // Per-variant ISA gate (the early run-once check above only confirms
  // at least ONE family's ISA exists).  bf16 variants require AVX-512
  // BF16; int8 variants require AVX-512 VNNI.  Splitting the gate here
  // lets a VNNI-only host (no BF16) still serve DQ-INT8.
  if (!is_int8_variant(variant) && !avx512bf16_available()) {
    return refuse("avx512bf16_not_available",
                  "BF16 custom kernel requires AVX-512 BF16 (VDPBF16PS)"
                  " — runtime CPU detection failed");
  }
  if (is_int8_variant(variant) && !avx512vnni_available()) {
    return refuse("avx512vnni_not_available",
                  "DQ-INT8 custom kernel requires AVX-512 VNNI "
                  "(VPDPBUSD) — runtime CPU detection failed");
  }
  // Cache the variant on the per-call context so `dispatch_tile()`
  // can route to the right kernel instantiation without re-running
  // the dtype switch per tile.
  out.variant = variant;
  // Asymmetric (u8 src + per-row src_zp) for BOTH the bf16-dst and
  // f32-dst asym variants; symmetric otherwise.  Keying on the
  // resolved variant (not just the bf16 one) ensures the f32-dst asym
  // path selects the kU8_Asym microkernels and honours src_zp.
  out.compute_int = (variant == KernelVariant::kU8_S8_BF16_ASYM
                     || variant == KernelVariant::kU8_S8_F32_ASYM)
      ? IntCompute::kU8_Asym
      : IntCompute::kS8_Sym;
  // Activation gate.
  //
  // The kernel handles four activation states inline:
  //
  //   * `none` — plain matmul; the epilogue stores the wide output.
  //   * `swiglu_oai_mul` — fused-in-kernel activation.  Caller
  //     provides W13 already interleaved as `[g0, u0, g1, u1, ...]`;
  //     the pair-pack store helper deinterleaves and applies the
  //     activation in registers (one fused matmul + activation pass,
  //     halved-width output).
  //   * `silu_and_mul` — fused-in-kernel activation.  Caller's W13
  //     is in canonical split-halves `[gate_cols | up_cols]`; the
  //     prepack permutes source columns so the CK arena physically
  //     matches the swiglu_oai_mul layout.  Same in-register
  //     epilogue contract.
  //   * `gelu_and_mul` — fused-in-kernel activation.  Caller-side
  //     contract is identical to `silu_and_mul`; the prepack uses
  //     the same column-interleave permutation, and the in-register
  //     epilogue uses a `gelu_tanh` polynomial approximation
  //     (max delta ≤ 1.5e-3 vs the reference's `gelu_erf`,
  //     comfortably inside the BF16 tolerance band).
  //
  // Bias is NOT supported on `silu_and_mul` / `gelu_and_mul` yet —
  // the bias-into-init epilogue would need to read
  // `[gate_bias | up_bias]` in permuted order; planned follow-up.
  if (act != grp_matmul_gated_act_t::swiglu_oai_mul
      && act != grp_matmul_gated_act_t::silu_and_mul
      && act != grp_matmul_gated_act_t::gelu_and_mul
      && act != grp_matmul_gated_act_t::none) {
    return refuse("unsupported_activation",
                  "custom kernel admits only none / swiglu_oai_mul / "
                  "silu_and_mul / gelu_and_mul (fused).  Other "
                  "activations require the caller to translate to "
                  "act=none and apply the activation as a separate "
                  "post-pass.");
  }
  // silu_and_mul / gelu_and_mul + bias is not yet supported by the
  // fused epilogue (split-halves bias would need to be read in
  // permuted order to match the interleaved layout).  Refuse the
  // combination cleanly so the caller falls back to the
  // separate-pass path with canonical bias-into-init.  Production
  // envelope (Qwen3, Mixtral, DBRX, DeepSeek) is bias-free on W13,
  // so this refusal does not affect typical decode.
  if ((act == grp_matmul_gated_act_t::silu_and_mul
       || act == grp_matmul_gated_act_t::gelu_and_mul)
      && bias_dtype != data_type_t::none) {
    return refuse("split_halves_act_with_bias_not_fused",
                  "silu_and_mul / gelu_and_mul fused-CK path is "
                  "bias-free in this release; fall back to "
                  "separate-pass for biased calls (planned "
                  "follow-up).");
  }
  // `act_dtype` only matters when an activation is actually applied —
  // for `act = none` (plain GEMM) we accept any act_dtype (including
  // `none`) so callers that have no activation at all don't have to
  // fabricate a dummy value.
  if (act != grp_matmul_gated_act_t::none
      && act_dtype != data_type_t::bf16) {
    return refuse("unsupported_act_dtype",
                  "fused activation requires act_dtype=bf16");
  }
  // Bias dtype resolution — the ukernel handles three cases:
  //   * no bias buffer at all (caller passes nullptr per-expert).
  //   * bf16 bias — load 16 bf16 lanes, convert to fp32 in-register.
  //   * fp32 bias — load 16 fp32 lanes directly.
  // Anything else (e.g. f16, s8) falls back to the standard path.
  BiasKind bias_kind = BiasKind::none;
  if (bias_dtype == data_type_t::none) {
    bias_kind = BiasKind::none;
  } else if (bias_dtype == data_type_t::bf16) {
    bias_kind = BiasKind::bf16;
  } else if (bias_dtype == data_type_t::f32) {
    bias_kind = BiasKind::fp32;
  } else {
    return refuse("unsupported_bias_dtype",
                  "custom kernel supports none/bf16/fp32 bias only");
  }

  const int num_ops = static_cast<int>(M.size());
  if (num_ops <= 0 || num_ops > CallContext::kMaxExperts) {
    return refuse("num_ops_out_of_range",
                  "num_ops must be in [1, kMaxExperts]");
  }

  // ── Pack-NR planner — uniform across experts in any call we
  // accept (the caller has already asserted N uniformity diagnostically). ──
  int pack_nr = 0;
  for (int i = 0; i < num_ops; ++i) {
    if (M[i] <= 0) continue;
    pack_nr = plan_pack_nr(K[i], N[i]);
    break;  // first active expert is representative
  }
  if (pack_nr != kNRMin && pack_nr != kNRMax) {
    return refuse("N_not_multiple_of_pack_nr",
                  "pack_nr must be 32 or 64 and divide N");
  }

  // ── Per-expert contract gate ─────────────────────────────────────
  // Any failing expert disables the custom kernel for the whole
  // call — we don't want a mixed dispatch where some tiles take the
  // custom path and others fall back inside the OMP loop.
  int m_max = 0;
  int K_for_subtile = 0;
  // Per-expert ldb sanity (also catches a missized `ldb` vector
  // before we index it inside the pack).  Relaxed from strict
  // equality so callers that pass a longer `ldb[]` (e.g.
  // `group_matmul_direct` keeps ldb at the framework's prepack-
  // extras size while M is sliced to the active count) still pass
  // — only the first `num_ops` entries are read by the dispatch
  // loop below.  Undersized vectors still fail.
  if (ldb.size() < static_cast<size_t>(num_ops))
    return refuse("ldb_size_mismatch",
                  "ldb vector size below num_ops");
  for (int i = 0; i < num_ops; ++i) {
    if (M[i] <= 0) continue;
    if (transA[i])
      return refuse("transA_not_supported",
                    "custom kernel reads src in row-major only");
    // Note: transB[i]=true is now SUPPORTED (PyTorch [N,K] layout).
    // The pack reads with the appropriate addressing and the cache
    // key folds transB into the discriminator so the two layouts
    // never alias.
    if (weight[i] == nullptr)
      return refuse("null_weight_in_active_expert");
    if ((N[i] % pack_nr) != 0)
      return refuse("N_not_multiple_of_pack_nr",
                    "an active expert's N is not a multiple of pack_nr");
    // The int8 microkernel reads the per-row src in 4-byte (K-quad)
    // broadcasts; the packed weight is zero-padded to a K-quad but
    // the hoisted src buffer is exactly K bytes per row, so a
    // K % 4 != 0 tail would over-read the src row.  Refuse the int8
    // CK path for unaligned K (the AOCL DLP sym-quant fallback
    // handles any K).  bf16 (K-pair) is unaffected.
    if (is_int8_variant(variant) && (K[i] % kVNNIInt8Quad) != 0)
      return refuse("int8_K_not_multiple_of_4",
                    "DQ-INT8 custom kernel requires K divisible by 4 "
                    "(VNNI K-quad); falling back to AOCL DLP");
    if (alpha[i] != 1.0f || beta[i] != 0.0f)
      return refuse("alpha_beta_not_supported",
                    "custom kernel requires alpha=1, beta=0 per expert");
    // ldb must accommodate the inner row of the chosen layout.
    const int min_ldb = transB[i] ? K[i] : N[i];
    if (ldb[i] < min_ldb)
      return refuse("ldb_below_min_row_stride",
                    "an active expert's ldb is smaller than min row stride");
    // is_weights_const = false means the caller may mutate this
    // expert's weight buffer in-place between calls.  The CK pack
    // cache is keyed on the source weight pointer and has no per-
    // expert "skip cache" branch (unlike AOCL DLP's `run_dlp(...)`
    // at aocl_kernel.cpp:1700-1702), so a cached packed copy from a
    // previous call could silently shadow a mutated weight and
    // produce wrong results.  Refuse the whole call so the caller
    // falls back to the standard AOCL DLP path, which honours the
    // flag at runtime.  Empty `is_weights_const` is treated as "all
    // const" (legacy callers that don't pass the field).
    if (!is_weights_const.empty()
        && static_cast<size_t>(i) < is_weights_const.size()
        && !is_weights_const[i])
      return refuse("non_const_weight_in_active_expert",
                    "custom kernel pack cache cannot honour an active "
                    "expert's is_weights_const=false");
    if (M[i] > m_max) m_max = M[i];
    if (K_for_subtile == 0) K_for_subtile = K[i];
  }
  if (m_max == 0) {
    // No active experts — mark disabled but treat as a no-op so the
    // caller can still complete its own bookkeeping.  Not strictly a
    // "refusal" (nothing to dispatch), but we log so all_zero_M
    // problems are visible too.
    return refuse("all_experts_have_M_zero");
  }

  // ── Build the run context ─────────────────────────────────────────
  out.pack_nr      = pack_nr;
  out.NV           = pack_nr / 16;
  out.max_mr       = max_mr_for_nv(out.NV);
  // Map framework activation → CK ActKind.  Order matters: refusal
  // gate above guarantees `act` is one of {none, swiglu_oai_mul,
  // silu_and_mul, gelu_and_mul} at this point.
  out.act_kind     = (act == grp_matmul_gated_act_t::swiglu_oai_mul)
      ? ActKind::swiglu_oai_mul
      : (act == grp_matmul_gated_act_t::silu_and_mul)
          ? ActKind::silu_and_mul
          : (act == grp_matmul_gated_act_t::gelu_and_mul)
              ? ActKind::gelu_and_mul
              : ActKind::none;
  out.bias_kind    = bias_kind;
  // Representative subtile_cols (worst-case m_max) — used by APILOG /
  // debug.  Dispatch reads per-expert values below.  Byte model is
  // dtype-aware: bf16 family uses (2,2,0); int8 family uses (1,1,4)
  // — see `subtile_bytes_for_variant` for the rationale.
  const SubtileBytes sb = subtile_bytes_for_variant(out.variant);
  out.subtile_cols = pick_l2_subtile_cols(
      m_max, K_for_subtile, pack_nr,
      sb.src_bytes_per_elem, sb.wei_bytes_per_elem, sb.comp_bytes_per_col);

  // Fill the per-MR kernel-pointer table from the resolved variant:
  // BF16 family → `kfn_table` (template axis is `DstDt`); DQ-INT8
  // family → `kfn_table_int8` (template axes are `IntCompute` and
  // `DstDt` — bf16 or f32 dst).  Only the populated table is read
  // by `dispatch_tile`; the other stays zero-initialised.
  if (is_int8_variant(out.variant)) {
    const DstDt dst_dt = dst_dt_for_variant(out.variant);
    if (fill_kfn_table_int8(out.NV, out.compute_int, out.act_kind, dst_dt,
                            out.kfn_table_int8) != status_t::success) {
      return refuse("kfn_table_int8_fill_failed",
                    "no DQ-INT8 microkernel for this (NV, compute, "
                    "act_kind, dst_dt) tuple — note gated act + FP32-dst "
                    "is intentionally rejected (BF16 dst only)");
    }
  } else {
    const DstDt dst_dt = dst_dt_for_variant(out.variant);
    if (fill_kfn_table(out.NV, out.act_kind, dst_dt, out.kfn_table)
        != status_t::success) {
      return refuse("kfn_table_fill_failed",
                    "no microkernel for this (NV, act_kind, dst_dt) tuple "
                    "— note swiglu_oai_mul + FP32-dst is intentionally "
                    "rejected (BF16 dst only)");
    }
  }

  // ── Pre-pack every active expert's weight (single-threaded; the
  // LRU mutex never contends in the per-tile path).  When the env
  // knob `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_SUBTILE_PER_EXPERT=1`
  // is set, also size each expert's L2-friendly subtile individually:
  // small-M experts (low A panel footprint) get a wider subtile for
  // better B reuse.  Default (knob=0): leave `subtile_cols_per_expert`
  // zero-filled so `dispatch_tile` falls back to the global m_max-
  // sized `ctx.subtile_cols` — the production baseline.
  const bool per_expert_subtile =
      get_grp_matmul_custom_kernel_subtile_per_expert();
  // For `silu_and_mul` and `gelu_and_mul` the caller's W13 is in
  // canonical split-halves layout; the prepack permutes source
  // columns into the interleaved layout the in-register fused
  // epilogue expects.  Both split-halves variants share the SAME
  // permutation — they differ only in the kernel-side activation
  // math, not in the pack layout, so the cache key bit
  // (`kInterleaveSplitMarker` in pack.cpp) is shared and a packed
  // arena warmed for silu can be reused for gelu on the same
  // weight pointer.
  // `swiglu_oai_mul` callers already provide interleaved weight at
  // the API boundary, so the flag stays false for them.
  const bool interleave_split_halves =
      (act == grp_matmul_gated_act_t::silu_and_mul)
      || (act == grp_matmul_gated_act_t::gelu_and_mul);
  out.packed_ptrs.fill(nullptr);
  out.packed_ptrs_int8.fill(nullptr);
  out.subtile_cols_per_expert.fill(0);
  // `out.owned_packed_ptrs` / `owned_packed_ptrs_int8` were already
  // zeroed by `out.reset()` at entry; only the `cache_off` branch
  // below stores into them.
  const bool variant_is_int8 = is_int8_variant(out.variant);
  for (int i = 0; i < num_ops; ++i) {
    if (M[i] <= 0) continue;
    bool was_hit_unused = false;
    if (variant_is_int8) {
      // DQ-INT8 path — caller's weight is signed s8 (`is_weights_const`
      // / pack_nr / ldb / transB contracts are the same as bf16).
      // The int8 pack writes both the VNNI-quad weight slab and the
      // per-column compensation row; the microkernel reads both
      // through a single `int8_t *`.
      status_t pst = get_or_pack_weight_int8(
          static_cast<const int8_t *>(weight[i]),
          K[i], N[i], ldb[i], pack_nr,
          /*transB=*/transB[i],
          /*interleave_split_halves=*/interleave_split_halves,
          &out.packed_ptrs_int8[i],
          /*was_hit_out=*/&was_hit_unused,
          /*disable_cache=*/cache_off);
      if (pst != status_t::success) {
        return refuse("weight_pack_failed",
                      "get_or_pack_weight_int8 returned failure — "
                      "see preceding log_error for OOM/arg detail");
      }
      if (cache_off) {
        out.owned_packed_ptrs_int8[i] = out.packed_ptrs_int8[i];
      }
    } else {
      // BF16 path — unchanged.
      status_t pst = get_or_pack_weight_bf16(
          static_cast<const bfloat16_t *>(weight[i]),
          K[i], N[i], ldb[i], pack_nr,
          /*transB=*/transB[i],
          /*interleave_split_halves=*/interleave_split_halves,
          &out.packed_ptrs[i],
          /*was_hit_out=*/&was_hit_unused,
          /*disable_cache=*/cache_off);
      if (pst != status_t::success) {
        return refuse("weight_pack_failed",
                      "get_or_pack_weight_bf16 returned failure — "
                      "see preceding log_error for OOM/arg detail");
      }
      if (cache_off) {
        out.owned_packed_ptrs[i] = out.packed_ptrs[i];
      }
    }
    if (per_expert_subtile) {
      out.subtile_cols_per_expert[i] =
          pick_l2_subtile_cols(M[i], K[i], pack_nr,
                               sb.src_bytes_per_elem,
                               sb.wei_bytes_per_elem,
                               sb.comp_bytes_per_col);
    }
  }

  out.enabled = true;
  if (s_refuse_log) {
    const char *variant_name =
        (out.variant == KernelVariant::kBF16_BF16_BF16) ? "bf16_bf16_bf16"
        : (out.variant == KernelVariant::kBF16_BF16_F32) ? "bf16_bf16_f32"
        : (out.variant == KernelVariant::kS8_S8_BF16_SYM) ? "s8_s8_bf16_sym"
        : (out.variant == KernelVariant::kU8_S8_BF16_ASYM) ? "u8_s8_bf16_asym"
        : (out.variant == KernelVariant::kS8_S8_F32_SYM) ? "s8_s8_f32_sym"
        : (out.variant == KernelVariant::kU8_S8_F32_ASYM) ? "u8_s8_f32_asym"
        : "unsupported";
    apilog_verbose("[GRP_MATMUL.CK ENGAGED] variant=", variant_name,
                " pack_nr=", out.pack_nr,
                " NV=", out.NV,
                " max_mr=", out.max_mr,
                " subtile_cols=", out.subtile_cols,
                " act_kind=", (out.act_kind == ActKind::swiglu_oai_mul
                               ? "swiglu_oai_mul"
                               : out.act_kind == ActKind::silu_and_mul
                                 ? "silu_and_mul"
                                 : out.act_kind == ActKind::gelu_and_mul
                                   ? "gelu_and_mul" : "none"),
                " bias_kind=", (out.bias_kind == BiasKind::none ? "none"
                                : out.bias_kind == BiasKind::bf16 ? "bf16"
                                : "fp32"),
                " num_ops=", num_ops,
                " per_expert_subtile=", (per_expert_subtile ? 1 : 0));
  }
  return status_t::success;
}

// (The ahead-of-time warm-pack — `warm_pack_all_custom_kernel_experts`
// + `PackProbeStats` — moved to
// `group_matmul/prepack/prepack_custom_kernel.{hpp,cpp}`.  This file
// keeps the per-call dispatcher path; `plan_pack_nr` was promoted to
// the public dispatch.hpp surface so the prepack module can compute
// the same NR the dispatcher will pack under.)

// DQ-INT8 dispatch tile — owns the per-tile sub-tile + per-MR loop
// for the `kS8_S8_BF16_SYM` / `kU8_S8_BF16_ASYM` variants.  Splits
// out of the public `dispatch_tile` so the BF16 loop body stays
// straight-line and any future Phase-2 changes to the int8 hot
// path land here without disturbing the bf16 sibling.
//
// Pointer convention:
//   * `A`         — caller's `src` cast to `uint8_t *`.  For
//     `kS8_S8_BF16_SYM` the caller passes the s8 buffer
//     (`int8_t *`) but the microkernel re-interprets it as
//     `uint8_t *` and XORs each broadcast with `0x80808080` to
//     run VPDPBUSD; the compensation row precomputed at pack time
//     undoes the resulting `+128 × sum_wei` bias.  Casting once
//     here keeps the signature uniform across compute flavours.
//   * `Bpacked`   — `int8_t *` to this expert's packed weight slab
//     (one o-block of weight bytes followed by `pack_nr` int32
//     compensation lanes; the microkernel walks both regions via
//     byte arithmetic, see int8_microkernel.cpp).
//   * `src_scale_full` / `src_zp_full` / `wei_scale_full` — full
//     M / M / N vectors as the caller passed to the public
//     `dispatch_tile`.  The dispatcher slices to the per-(m_off,
//     mr_now) and per-(sub_col_base, n_blocks) windows.  `src_zp`
//     is passed straight through (nullptr for sym).
//
// Tight / wide destination handling is identical to the bf16
// sibling — gated activation epilogues store to the half-width tight
// arena; act=none stores to the wide [M, N] dst.  The dst element
// width is variant-dependent (`dst_elem_bytes_int8` below): bf16 for
// the *_BF16_* variants, f32 for the *_F32_* (act=none) variants.
namespace {
inline void dispatch_tile_int8(
    const CallContext &ctx,
    int   expert_idx,
    int   M, int K,
    int   n_tile, int col_start,
    const void   *src,  int lda,
    const void   *bias,
    void         *tight_dst, int tight_ldc,
    const void    *src_scale_full,
    const int32_t *src_zp_full,
    const void    *wei_scale_full) {

  // Per-tile contract: src_scale + wei_scale are required; src_zp
  // is required iff the variant is asym.  The caller (N-tile
  // executor) populates both from the hoisted dynamic-quant output;
  // mis-routing here would silently dequant to zero, which the
  // microkernel's epilogue would propagate.  Assert in debug.
  assert(src_scale_full != nullptr
         && "dispatch_tile_int8: src_scale is null — N-tile hoist "
            "must populate this for DQ-INT8");
  assert(wei_scale_full != nullptr
         && "dispatch_tile_int8: wei_scale is null — caller must "
            "thread the per-expert wei_scale");
  // Asym requires src_zp — true for both the bf16-dst and f32-dst
  // asym variants.  Key off the resolved compute family rather than a
  // single variant so the f32-dst asym path is covered too.
  if (ctx.compute_int == IntCompute::kU8_Asym) {
    assert(src_zp_full != nullptr
           && "dispatch_tile_int8: src_zp is null on the asym path "
              "— hoist must populate it for compute=u8");
  }

  const int8_t  *Bpacked_full =
      ctx.packed_ptrs_int8[expert_idx];
  const auto    *A_bytes      = static_cast<const uint8_t *>(src);
  std::byte     *Tight_bytes  = static_cast<std::byte *>(tight_dst);
  // Dst element width tracks the resolved variant: BF16 (2 bytes) for
  // the kS8/U8_*_BF16_* variants, FP32 (4 bytes) for the kS8/U8_*_F32_*
  // variants.  The kernel itself reinterprets the `void *` dst to the
  // right type via its `DstDt` template; the dispatcher only needs the
  // element stride here for pointer arithmetic.  FP32 dst is act=none
  // only (gated kinds are BF16-dst), so the gated path below always
  // sees the BF16 width.
  const size_t dst_elem_bytes_int8 =
      (dst_dt_for_variant(ctx.variant) == DstDt::kF32)
          ? sizeof(float)
          : sizeof(bfloat16_t);

  const int    K_quad = (K + kVNNIInt8Quad - 1) / kVNNIInt8Quad;
  // Per-o-block stride in BYTES: weight slab + per-column int32
  // compensation row.  Mirror of the pack-time
  // `bytes_per_oblock` expression in pack.cpp; the two MUST stay
  // in lockstep.
  const size_t o_blk_stride_bytes =
      static_cast<size_t>(K_quad) * ctx.pack_nr * kVNNIInt8Quad
      + static_cast<size_t>(ctx.pack_nr) * sizeof(int32_t);

  const int subtile_cols = (ctx.subtile_cols_per_expert[expert_idx] > 0)
      ? ctx.subtile_cols_per_expert[expert_idx]
      : ctx.subtile_cols;

  // Balanced MR partition — identical scheme to the bf16 sibling.
  // See dispatch_tile (bf16) for the rationale.
  const int n_calls = (M + ctx.max_mr - 1) / ctx.max_mr;
  const int mr_base = M / n_calls;
  const int n_big   = M - mr_base * n_calls;

  const size_t bias_elem_bytes = (ctx.bias_kind == BiasKind::fp32)
      ? sizeof(float)
      : sizeof(bfloat16_t);

  // Scale element width — the microkernel reads src/wei scales as bf16
  // (converting on load) or f32.  The dispatcher slices both scale
  // buffers by byte offset using this stride; the kernel re-casts the
  // `const void *` per `ctx.scale_kind`.
  const size_t scale_elem_bytes = (ctx.scale_kind == ScaleKind::kBf16)
      ? sizeof(bfloat16_t)
      : sizeof(float);

  const bool is_gated_act =
      (ctx.act_kind == ActKind::swiglu_oai_mul)
      || (ctx.act_kind == ActKind::silu_and_mul)
      || (ctx.act_kind == ActKind::gelu_and_mul);

  if (is_gated_act) {
    for (int sub_off = 0; sub_off < n_tile; sub_off += subtile_cols) {
      const int sub_n        = std::min(subtile_cols, n_tile - sub_off);
      const int sub_col_base = col_start + sub_off;
      const int n_blocks     = sub_n / ctx.pack_nr;

      const int8_t *Bpacked_blk_base = Bpacked_full
          + static_cast<size_t>(sub_col_base / ctx.pack_nr) * o_blk_stride_bytes;
      const char *bias_blk_base = (bias != nullptr)
          ? static_cast<const char *>(bias)
              + static_cast<size_t>(sub_col_base) * bias_elem_bytes
          : nullptr;
      const char *wei_scale_blk_base = static_cast<const char *>(wei_scale_full)
          + static_cast<size_t>(sub_col_base) * scale_elem_bytes;

      int m_off = 0;
      for (int c = 0; c < n_calls; ++c) {
        const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
        const int8_ukernel_fn_t kfn = ctx.kfn_table_int8[mr_now];

        const uint8_t *A_chunk =
            A_bytes + static_cast<size_t>(m_off) * lda;
        const void    *src_scale_chunk = static_cast<const char *>(src_scale_full)
            + static_cast<size_t>(m_off) * scale_elem_bytes;
        const int32_t *src_zp_chunk =
            (src_zp_full != nullptr) ? src_zp_full + m_off : nullptr;

        std::byte *Tight_row_base = Tight_bytes
            + (static_cast<size_t>(m_off) * tight_ldc
               + (sub_col_base / 2)) * dst_elem_bytes_int8;

        for (int b = 0; b < n_blocks; ++b) {
          const int8_t *Bpacked_blk =
              Bpacked_blk_base + static_cast<size_t>(b) * o_blk_stride_bytes;
          const void *bias_blk = (bias_blk_base != nullptr)
              ? static_cast<const void *>(bias_blk_base
                  + static_cast<size_t>(b) * ctx.pack_nr * bias_elem_bytes)
              : nullptr;
          const void *wei_scale_blk = static_cast<const void *>(
              wei_scale_blk_base
              + static_cast<size_t>(b) * ctx.pack_nr * scale_elem_bytes);
          // 16 cols per (g, u) pair × NV/2 pairs per kernel call —
          // mirror of the bf16 sibling's tight stride.
          std::byte *Tight_row = Tight_row_base
              + static_cast<size_t>(b) * (ctx.pack_nr / 2) * dst_elem_bytes_int8;

          kfn(A_chunk, lda, Bpacked_blk,
              src_scale_chunk, src_zp_chunk, wei_scale_blk, ctx.scale_kind,
              bias_blk, ctx.bias_kind,
              /*Cout=*/nullptr, /*ldc=*/0,
              Tight_row, tight_ldc, K);
        }
        m_off += mr_now;
      }
    }
    return;
  }

  // Act = none — wide output to the [M, N] dst arena.
  for (int sub_off = 0; sub_off < n_tile; sub_off += subtile_cols) {
    const int sub_n        = std::min(subtile_cols, n_tile - sub_off);
    const int sub_col_base = col_start + sub_off;
    const int n_blocks     = sub_n / ctx.pack_nr;

    const int8_t *Bpacked_blk_base = Bpacked_full
        + static_cast<size_t>(sub_col_base / ctx.pack_nr) * o_blk_stride_bytes;
    const char *bias_blk_base = (bias != nullptr)
        ? static_cast<const char *>(bias)
            + static_cast<size_t>(sub_col_base) * bias_elem_bytes
        : nullptr;
    const char *wei_scale_blk_base = static_cast<const char *>(wei_scale_full)
        + static_cast<size_t>(sub_col_base) * scale_elem_bytes;

    int m_off = 0;
    for (int c = 0; c < n_calls; ++c) {
      const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
      const int8_ukernel_fn_t kfn = ctx.kfn_table_int8[mr_now];

      const uint8_t *A_chunk =
          A_bytes + static_cast<size_t>(m_off) * lda;
      const void    *src_scale_chunk = static_cast<const char *>(src_scale_full)
          + static_cast<size_t>(m_off) * scale_elem_bytes;
      const int32_t *src_zp_chunk =
          (src_zp_full != nullptr) ? src_zp_full + m_off : nullptr;

      std::byte *Wide_row_base = Tight_bytes
          + (static_cast<size_t>(m_off) * tight_ldc
             + sub_col_base) * dst_elem_bytes_int8;

      for (int b = 0; b < n_blocks; ++b) {
        const int8_t *Bpacked_blk =
            Bpacked_blk_base + static_cast<size_t>(b) * o_blk_stride_bytes;
        const void *bias_blk = (bias_blk_base != nullptr)
            ? static_cast<const void *>(bias_blk_base
                + static_cast<size_t>(b) * ctx.pack_nr * bias_elem_bytes)
            : nullptr;
        const void *wei_scale_blk = static_cast<const void *>(
            wei_scale_blk_base
            + static_cast<size_t>(b) * ctx.pack_nr * scale_elem_bytes);
        std::byte *Wide_row = Wide_row_base
            + static_cast<size_t>(b) * ctx.pack_nr * dst_elem_bytes_int8;

        kfn(A_chunk, lda, Bpacked_blk,
            src_scale_chunk, src_zp_chunk, wei_scale_blk, ctx.scale_kind,
            bias_blk, ctx.bias_kind,
            Wide_row, tight_ldc,
            /*Cout_tight=*/nullptr, /*ldc_tight=*/0, K);
      }
      m_off += mr_now;
    }
  }
}
}  // namespace

void dispatch_tile(
    const CallContext &ctx,
    int   expert_idx,
    int   M, int K,
    int   n_tile, int col_start,
    const void *src,  int lda,
    const void *bias,
    void       *tight_dst, int tight_ldc,
    const void    *src_scale,
    const int32_t *src_zp,
    const void    *wei_scale) {

  // ── Per-tile contract assertions (debug-only) ────────────────────
  //
  // Every assert below mirrors a contract that `prepare_for_call`
  // already validated.  Caller bug = call `dispatch_tile` without a
  // successful `prepare_for_call` first; these asserts turn silent
  // wrong output (zeros / garbage) into a loud debug-build failure.
  //
  // Asserts are debug-only; in release the contracts are upheld by
  // the upstream gates documented next to each assert.
  //
  // 1. ctx.enabled — set true by `prepare_for_call` on success only.
  //    A `dispatch_tile` call with `ctx.enabled == false` means the
  //    caller skipped the post-prepare check and is asking the
  //    dispatcher to use a CallContext that was either never
  //    populated (default-constructed) or rejected.
  assert(ctx.enabled
         && "dispatch_tile: ctx.enabled is false — caller must check "
            "prepare_for_call's status (or kctx.enabled) and route to "
            "DLP on failure");
  // 2. ctx.variant — set by `prepare_for_call` via `resolve_variant`.
  //    `kUnsupported` would mean the routing table couldn't classify
  //    the (src, wei, dst) tuple but the caller proceeded anyway.
  assert(ctx.variant != KernelVariant::kUnsupported
         && "dispatch_tile: ctx.variant is kUnsupported — "
            "prepare_for_call should have returned failure on this "
            "dtype tuple and the caller should have routed to DLP");
  // 3. Tile-shape contract: every per-thread tile produced by
  //    `aligned_n_split` must be a multiple of `pack_nr` (the kernel
  //    processes B in `pack_nr`-wide blocks via truncating
  //    `n_blocks = sub_n / pack_nr`; any non-multiple tail would be
  //    SILENTLY DROPPED, leaving dst columns uninitialised).
  //
  //    This holds for the current production envelope because:
  //      a. `prepare_for_call` refuses (N % pack_nr != 0);
  //      b. for any `n_thr ≤ N / pack_nr`, `aligned_n_split` always
  //         finds an aligned candidate that satisfies the 2× imbalance
  //         bound, never falling back to its unaligned even-split path;
  //      c. `participating_n_thr` upstream caps `n_thr` by
  //         `N / min_n_tile` (= `N / kDecodeNTile` = `N / 256`), and
  //         `min_n_tile >= pack_nr`, so condition (b) holds.
  assert(ctx.pack_nr > 0
         && "dispatch_tile: pack_nr is zero — call prepare_for_call");
  assert((n_tile % ctx.pack_nr) == 0
         && "dispatch_tile: n_tile not a multiple of pack_nr — tail "
            "cols would be silently dropped by truncating n_blocks");
  assert((col_start % ctx.pack_nr) == 0
         && "dispatch_tile: col_start not a multiple of pack_nr — "
            "B-pack offset (col_start / pack_nr) would mis-align");

  // ── DQ-INT8 fast path — hand off to the int8 sibling and return.
  // The bf16 loop body below is straight-line; keeping the int8
  // branch up front avoids loading the bf16-specific stride
  // constants we don't need on this path.
  if (is_int8_variant(ctx.variant)) {
    dispatch_tile_int8(ctx, expert_idx, M, K, n_tile, col_start,
                       src, lda, bias, tight_dst, tight_ldc,
                       src_scale, src_zp, wei_scale);
    return;
  }
  // BF16 callers must NOT pass src_scale / src_zp / wei_scale —
  // those are DQ-INT8-only.  If they did, fail loudly in debug so
  // the contract violation is visible; in release the BF16 body
  // simply ignores them.
  assert(src_scale == nullptr
         && "dispatch_tile: src_scale must be null on the BF16 path");
  assert(src_zp == nullptr
         && "dispatch_tile: src_zp must be null on the BF16 path");
  assert(wei_scale == nullptr
         && "dispatch_tile: wei_scale must be null on the BF16 path");
  (void)src_scale; (void)src_zp; (void)wei_scale;

  const bfloat16_t *Bpacked_full = ctx.packed_ptrs[expert_idx];
  const auto       *A            = static_cast<const bfloat16_t *>(src);
  // Use byte-arithmetic for the dst since the element width depends on
  // the kernel variant (BF16=2B for kBF16_BF16_BF16; FP32=4B for
  // kBF16_BF16_F32).  The kernel internally re-casts the void* to its
  // template `DstT`; ldc / ldc_tight stay in element units (not bytes).
  std::byte *Tight_bytes        = static_cast<std::byte *>(tight_dst);
  const size_t dst_elem_bytes   =
      (ctx.variant == KernelVariant::kBF16_BF16_F32)
          ? sizeof(float) : sizeof(bfloat16_t);

  // Per-tile constants (hoisted once outside the sub-tile / m-chunk /
  // n-block loops below).
  const int    K_pair       = (K + 1) / 2;
  const size_t o_blk_stride =
      static_cast<size_t>(K_pair) * ctx.pack_nr * kVNNIPair;

  // Per-expert L2-friendly subtile width (see `subtile_cols_per_expert`
  // doc in dispatch.hpp).  Falls back to the representative global
  // value if the per-expert slot wasn't populated (M <= 0 at prep
  // time, which the caller's sweep over num_ops upstream should have
  // already filtered out).
  const int subtile_cols = (ctx.subtile_cols_per_expert[expert_idx] > 0)
      ? ctx.subtile_cols_per_expert[expert_idx]
      : ctx.subtile_cols;

  // Balanced MR partition: split M into `n_calls = ceil(M / max_mr)`
  // kernel calls, spreading the rows so each call's MR is either
  // `mr_base` or `mr_base + 1`.  First `n_big` calls take mr_base+1.
  // Avoids thin-tail MR=1 / MR=2 calls that would otherwise cap FMA
  // ILP (MR<4 is latency-limited on Zen4/5 with 4-cycle FMA dep).
  //
  // Worked examples (max_mr=8):
  //   M= 9 → 2 calls [5, 4]   (naive would be [8, 1] — MR=1 tail)
  //   M=17 → 3 calls [6, 6, 5] (naive [8, 8, 1])
  //   M=24 → 3 calls [8, 8, 8] (no tail either way)
  //   M= 5 → 1 call  [5]
  const int n_calls = (M + ctx.max_mr - 1) / ctx.max_mr;
  const int mr_base = M / n_calls;
  const int n_big   = M - mr_base * n_calls;

  // Per-col byte stride for bias pointer arithmetic — the dispatcher
  // stores the bias dtype in `ctx.bias_kind` so we advance the pointer
  // by the right element width when slicing into the per-sub-tile /
  // per-block bias window.
  const size_t bias_elem_bytes = (ctx.bias_kind == BiasKind::fp32)
      ? sizeof(float)
      : sizeof(bfloat16_t);  // bf16 (and none — unused when bias==null)

  // Hoist the act-kind branch out of the loops — straight-line code
  // with one indirect call per microkernel invocation.  All three
  // gated-activation kinds (swiglu_oai_mul, silu_and_mul,
  // gelu_and_mul) share the same tight-arena contract: halved-width
  // output, BF16-only, 16 cols per (g, u) pair × NV/2 pairs per
  // kernel call.  They differ only in which in-register pair-store
  // helper the microkernel calls (resolved at `select_ukernel` time
  // and baked into `ctx.kfn_table[]`), so the dispatcher loop body
  // is identical.
  const bool is_gated_act =
      (ctx.act_kind == ActKind::swiglu_oai_mul)
      || (ctx.act_kind == ActKind::silu_and_mul)
      || (ctx.act_kind == ActKind::gelu_and_mul);
  if (is_gated_act) {
    // Walk the per-thread N range in L2-friendly sub-tiles.  Each
    // sub-tile's B strip (= K × subtile_cols × 2 bytes) fits in L2
    // alongside the input slice, so the microkernel never hits L3 /
    // DRAM mid-call for weight.
    for (int sub_off = 0; sub_off < n_tile; sub_off += subtile_cols) {
      const int sub_n        = std::min(subtile_cols, n_tile - sub_off);
      const int sub_col_base = col_start + sub_off;
      const int n_blocks     = sub_n / ctx.pack_nr;

      const bfloat16_t *Bpacked_blk_base = Bpacked_full
          + static_cast<size_t>(sub_col_base / ctx.pack_nr) * o_blk_stride;
      const char *bias_blk_base = (bias != nullptr)
          ? static_cast<const char *>(bias)
              + static_cast<size_t>(sub_col_base) * bias_elem_bytes
          : nullptr;

      int m_off = 0;
      for (int c = 0; c < n_calls; ++c) {
        const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
        const ukernel_fn_t kfn = ctx.kfn_table[mr_now];

        const bfloat16_t *A_chunk =
            A + static_cast<size_t>(m_off) * lda;
        // Gated-act epilogues are BF16-dst-only (gated by
        // select_ukernel for swiglu_oai_mul, silu_and_mul, and
        // gelu_and_mul — see `is_gated_act` above), so the half-width
        // tight arena's element stride is fixed at `sizeof(bfloat16_t)`
        // here regardless of `dst_elem_bytes`.
        std::byte *Tight_row_base = Tight_bytes
            + (static_cast<size_t>(m_off) * tight_ldc
               + (sub_col_base / 2)) * sizeof(bfloat16_t);

        for (int b = 0; b < n_blocks; ++b) {
          const bfloat16_t *Bpacked_blk =
              Bpacked_blk_base + static_cast<size_t>(b) * o_blk_stride;
          const void *bias_blk = (bias_blk_base != nullptr)
              ? static_cast<const void *>(bias_blk_base
                  + static_cast<size_t>(b) * ctx.pack_nr * bias_elem_bytes)
              : nullptr;
          // 16 cols per (g, u) pair × NV/2 pairs per kernel call.
          std::byte *Tight_row = Tight_row_base
              + static_cast<size_t>(b) * (ctx.pack_nr / 2) * sizeof(bfloat16_t);

          kfn(A_chunk, lda, Bpacked_blk, bias_blk, ctx.bias_kind,
              /*Cout=*/nullptr, /*ldc=*/0,
              Tight_row, tight_ldc, K);
        }
        m_off += mr_now;
      }
    }
    return;
  }

  // Act = none — write the full pack_nr cols straight to the wide
  // [M, N] arena (Tight is the wide destination in this mode).
  for (int sub_off = 0; sub_off < n_tile; sub_off += subtile_cols) {
    const int sub_n        = std::min(subtile_cols, n_tile - sub_off);
    const int sub_col_base = col_start + sub_off;
    const int n_blocks     = sub_n / ctx.pack_nr;

    const bfloat16_t *Bpacked_blk_base = Bpacked_full
        + static_cast<size_t>(sub_col_base / ctx.pack_nr) * o_blk_stride;
    const char *bias_blk_base = (bias != nullptr)
        ? static_cast<const char *>(bias)
            + static_cast<size_t>(sub_col_base) * bias_elem_bytes
        : nullptr;

    int m_off = 0;
    for (int c = 0; c < n_calls; ++c) {
      const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
      const ukernel_fn_t kfn = ctx.kfn_table[mr_now];

      const bfloat16_t *A_chunk =
          A + static_cast<size_t>(m_off) * lda;
      // Wide / act=none path: dst element width depends on the
      // resolved variant (BF16=2B, FP32=4B).  Use byte arithmetic
      // here so the same loop body serves both store epilogues; the
      // kernel re-casts the void* row to its template `DstT`.
      std::byte *Wide_row_base = Tight_bytes
          + (static_cast<size_t>(m_off) * tight_ldc
             + sub_col_base) * dst_elem_bytes;

      for (int b = 0; b < n_blocks; ++b) {
        const bfloat16_t *Bpacked_blk =
            Bpacked_blk_base + static_cast<size_t>(b) * o_blk_stride;
        const void *bias_blk = (bias_blk_base != nullptr)
            ? static_cast<const void *>(bias_blk_base
                + static_cast<size_t>(b) * ctx.pack_nr * bias_elem_bytes)
            : nullptr;
        std::byte *Wide_row = Wide_row_base
            + static_cast<size_t>(b) * ctx.pack_nr * dst_elem_bytes;

        kfn(A_chunk, lda, Bpacked_blk, bias_blk, ctx.bias_kind,
            Wide_row, tight_ldc,
            /*Cout_tight=*/nullptr, /*ldc_tight=*/0, K);
      }
      m_off += mr_now;
    }
  }
}

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
