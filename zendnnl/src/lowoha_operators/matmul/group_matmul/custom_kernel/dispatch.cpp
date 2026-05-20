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

using zendnnl::error_handling::apilog_info;
using zendnnl::error_handling::apilog_info_enabled;

bool dispatch_supported() {
  return avx512bf16_available();
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
                              data_type_t dst) noexcept {
  if (src == data_type_t::bf16 && wei == data_type_t::bf16) {
    if (dst == data_type_t::bf16) return KernelVariant::kBF16_BF16_BF16;
    if (dst == data_type_t::f32 ) return KernelVariant::kBF16_BF16_F32;
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
int pick_l2_subtile_cols(int M_max, int K, int pack_nr) {
  if (M_max <= 0 || K <= 0 || pack_nr <= 0) return pack_nr;
  static const int64_t l2_budget = []() {
    const int64_t l2 =
        zendnnl::lowoha::matmul::native::detect_uarch().l2_bytes;
    const int64_t safe = (l2 >= 64 * 1024) ? l2 : (256 * 1024);
    return (safe * 4) / 5;
  }();

  const int64_t input_bytes =
      static_cast<int64_t>(M_max) * K * sizeof(uint16_t);
  const int64_t weight_bytes_per_col =
      static_cast<int64_t>(K) * sizeof(uint16_t);
  if (l2_budget <= input_bytes || weight_bytes_per_col <= 0) {
    return pack_nr;
  }
  const int64_t cols_raw =
      (l2_budget - input_bytes) / weight_bytes_per_col;
  const int cols_aligned =
      static_cast<int>((cols_raw / pack_nr) * pack_nr);
  return std::max(cols_aligned, pack_nr);
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
    CallContext &out) {

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
  // observe a stale-but-plausible config.  `out = {}` value-init's
  // every field via the in-class defaults declared in
  // `CallContext` (dispatch.hpp): `enabled = false`, `variant =
  // kUnsupported`, scalar fields = 0 / sentinel, fn-pointer table
  // and the two `std::array` members zero-filled.  Cost: ~2 KB of
  // zeroing once per call on the cold prep path, negligible vs
  // the surrounding kernel work.
  out = CallContext{};

  // ── Refusal logging helper ───────────────────────────────────────
  // Every early return below represents a concrete dispatch-contract
  // violation that forces the caller to fall back to the standard
  // AOCL / BRGEMM path.  Log the reason so a model-level read makes
  // it immediately obvious why the custom kernel was skipped — a
  // silent refusal just produced `kernel=standard` with no clue
  // whether the env was off, a dtype mismatched, or the pack-NR
  // check failed.  Single cached apilog_info_enabled() check.
  //
  // Lambda overhead note: both helpers are captureless (`[]`, not
  // `[&]`) so they are stateless functors — zero construction cost
  // at runtime.  `s_refuse_log` is a function-scope `static const`
  // which C++ lambdas can reference without capture, so the `if
  // (s_refuse_log)` guard inside `refuse()` still compiles and still
  // short-circuits to the plain `return status_t::failure` when API
  // info logging is off.  Net impact on a successful
  // `prepare_for_call` with logging disabled: zero — neither lambda
  // object is ever instantiated nor invoked on the success path,
  // and `dt_name` only gets called from the failure paths that also
  // read `s_refuse_log`.
  static const bool s_refuse_log = apilog_info_enabled();
  auto refuse = [](const char *reason_tag,
                   const char *detail = nullptr) -> status_t {
    if (s_refuse_log) {
      if (detail != nullptr) {
        apilog_info("[GRP_MATMUL Level4 custom_kernel REFUSED] reason=",
                    reason_tag, " (", detail, ")");
      } else {
        apilog_info("[GRP_MATMUL Level4 custom_kernel REFUSED] reason=",
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
  if (!dispatch_supported())
    return refuse("avx512bf16_not_available");
  // A / B / C dtype gate: route through `resolve_variant()` which is
  // the single source of truth for "which (src, wei, dst) tuples the
  // custom kernel can serve".  `kUnsupported` here means the caller
  // falls back to DLP.  Routing through the variant keeps the
  // dispatcher's gate aligned with the gtest matrix in
  // `gtests/group_matmul/custom_kernel/`, which calls
  // `resolve_variant()` directly to assert the table.
  const KernelVariant variant =
      resolve_variant(src_dtype, wei_dtype, dst_dtype);
  if (variant == KernelVariant::kUnsupported) {
    if (s_refuse_log) {
      apilog_info("[GRP_MATMUL Level4 custom_kernel REFUSED] reason="
                  "unsupported_dtype (src=", dt_name(src_dtype),
                  " wei=", dt_name(wei_dtype),
                  " dst=", dt_name(dst_dtype),
                  " — see resolve_variant() in custom_kernel/dispatch.cpp"
                  " for the supported table)");
    }
    return status_t::failure;
  }
  // Cache the variant on the per-call context so `dispatch_tile()`
  // can route to the right kernel instantiation without re-running
  // the dtype switch per tile.
  out.variant = variant;
  // Activation gate.
  //
  // The kernel handles two activation states inline:
  //
  //   * `none` — plain matmul; the epilogue stores the wide output.
  //   * `swiglu_oai_mul` — fused-in-kernel activation.  The
  //     interleaved `[g0, u0, g1, u1, ...]` layout puts gate / up
  //     pairs on every 32-col tile, so the pair-pack store helper
  //     can deinterleave and apply the activation in registers (one
  //     fused matmul + activation pass, halved-width output).
  //
  // `silu_and_mul` and `gelu_and_mul` are NOT accepted here even
  // though the kernel could serve them via the matmul-only path.
  // Their split-halves layout `[gate_cols | up_cols]` puts gate /
  // up of the same output position in DIFFERENT N-tiles, so the
  // per-tile microkernel cannot deinterleave them — the activation
  // must run as a separate post-pass on the wide [M, N] output.
  // Because the dispatcher cannot enforce that a direct caller
  // actually runs that post-pass, accepting these activations here
  // would silently leave a caller with only matmul output and no
  // activation applied.  Production callers translate silu/gelu →
  // act = none BEFORE invoking `prepare_for_call` and run the
  // activation themselves; `flat_n_tile` does this for the
  // group_matmul_direct path (it passes `custom_act = none` for any
  // act other than swiglu_oai_mul) and `group_matmul_direct` then
  // invokes `group_matmul_moe_act_execute` on the wide output.
  if (act != grp_matmul_gated_act_t::swiglu_oai_mul
      && act != grp_matmul_gated_act_t::none) {
    return refuse("unsupported_activation",
                  "custom kernel admits only none / swiglu_oai_mul "
                  "(fused).  Split-halves activations "
                  "(silu_and_mul, gelu_and_mul) require the caller "
                  "to translate to act=none and apply the "
                  "activation as a separate post-pass.");
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
  out.act_kind     = (act == grp_matmul_gated_act_t::swiglu_oai_mul)
      ? ActKind::swiglu_oai_mul : ActKind::none;
  out.bias_kind    = bias_kind;
  // Representative subtile_cols (worst-case m_max) — used by APILOG /
  // debug.  Dispatch reads per-expert values below.
  out.subtile_cols = pick_l2_subtile_cols(m_max, K_for_subtile, pack_nr);

  // Resolve the dst dtype for kernel-table fill from the cached
  // variant (single source of truth — keeps fill_kfn_table aligned
  // with `resolve_variant`).  `select_ukernel` rejects the
  // structurally-impossible (swiglu, FP32) combination by returning
  // nullptr, which `fill_kfn_table` then surfaces as failure.
  const DstDt dst_dt = dst_dt_for_variant(out.variant);
  if (fill_kfn_table(out.NV, out.act_kind, dst_dt, out.kfn_table)
      != status_t::success) {
    return refuse("kfn_table_fill_failed",
                  "no microkernel for this (NV, act_kind, dst_dt) tuple "
                  "— note swiglu_oai_mul + FP32-dst is intentionally "
                  "rejected (BF16 dst only)");
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
  out.packed_ptrs.fill(nullptr);
  out.subtile_cols_per_expert.fill(0);
  for (int i = 0; i < num_ops; ++i) {
    if (M[i] <= 0) continue;
    status_t pst = get_or_pack_weight_bf16(
        static_cast<const bfloat16_t *>(weight[i]),
        K[i], N[i], ldb[i], pack_nr,
        /*transB=*/transB[i],
        &out.packed_ptrs[i]);
    if (pst != status_t::success) {
      // `get_or_pack_weight_bf16` already logged the concrete reason
      // (OOM / transB mismatch) via log_error.  Attribute the
      // refusal here so the L4 chain is self-contained.
      return refuse("weight_pack_failed",
                    "get_or_pack_weight_bf16 returned failure — "
                    "see preceding log_error for OOM/arg detail");
    }
    if (per_expert_subtile) {
      out.subtile_cols_per_expert[i] =
          pick_l2_subtile_cols(M[i], K[i], pack_nr);
    }
  }

  out.enabled = true;
  if (s_refuse_log) {
    apilog_info("[GRP_MATMUL Level4 custom_kernel] ENGAGED pack_nr=",
                out.pack_nr,
                " NV=", out.NV,
                " max_mr=", out.max_mr,
                " subtile_cols=", out.subtile_cols,
                " act_kind=", (out.act_kind == ActKind::swiglu_oai_mul
                               ? "swiglu_oai_mul" : "none"),
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

void dispatch_tile(
    const CallContext &ctx,
    int   expert_idx,
    int   M, int K,
    int   n_tile, int col_start,
    const void *src,  int lda,
    const void *bias,
    void       *tight_dst, int tight_ldc) {

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
  // with one indirect call per microkernel invocation.
  if (ctx.act_kind == ActKind::swiglu_oai_mul) {
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
        // swiglu epilogue is BF16-dst-only (gated by select_ukernel),
        // so the half-width tight arena's element stride is fixed at
        // `sizeof(bfloat16_t)` here regardless of `dst_elem_bytes`.
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
