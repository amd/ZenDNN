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

/// Group MatMul direct — public entry point and input validation.
///
/// Implementation details:
///   - group_matmul/group_matmul_dispatch.cpp — parallel expert dispatch (OMP).
///   - group_matmul/group_matmul_moe_postop.cpp — optional MoE weighted-reduce post-op.

#include <sstream>
#include <vector>

#include "group_matmul/detect_internal_alloc.hpp"
#include "group_matmul/group_matmul_direct.hpp"
#include "group_matmul/group_matmul_parallel_common.hpp"
#include "lowoha_matmul_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::size_of;
using zendnnl::common::op_instrumentation;

namespace {

// ───────────────────────────────────────────────────────────────────────
// File-local helpers for the size + prepack-extras predicates shared
// between the diagnostic validator and the always-on inline guard.
//
// `size_check_ctx::fail(s)`:
//   Returns true iff the vector size `s` violates the per-call sizing
//   contract.  In legacy mode (no framework opt-in via
//   `params[0].active_matmul > 0`) it returns `s != num_ops` —
//   strict equality, preserves the original reject-oversized-vectors
//   behaviour bit-for-bit.  In opt-in mode it returns `s < num_ops`,
//   accepting the prepack-extras tail (s ≥ num_ops).
//
//   Lifted from two open-coded copies of `size_bad` /
//   `inline_size_bad` lambdas in `validate_group_matmul_direct_inputs`
//   (Phase A diagnostic) and `group_matmul_direct(...)` (always-on
//   inline guard).  Behaviour is identical; this struct just gives
//   one source of truth.
struct size_check_ctx {
  size_t num_ops;
  bool   relaxed;
  constexpr bool fail(size_t s) const noexcept {
    return relaxed ? (s < num_ops) : (s != num_ops);
  }
};

// `prepack_extras_metadata_undersized(...)`:
//   Returns true iff at least one weight-side metadata vector is
//   shorter than `total_matmul` (only checked when the framework
//   opted in via `total_matmul > active_matmul`).  The prepack
//   module iterates `[0, total_matmul)` over six vectors (weight,
//   K, N, ldb, transB, is_weights_const) and silently truncates the
//   warm to `min(total_matmul, vec.size())` if any is shorter.
//
//   `out_first_failure` (nullable) is populated with `{name, got,
//   need}` for the first vector that failed so the diagnostic site
//   can `log_error` with a precise message.  Pass nullptr from the
//   inline always-on path; both paths return the same boolean.
struct undersized_info {
  const char *name;
  size_t got;
  size_t need;
};

inline bool prepack_extras_metadata_undersized(
  const std::vector<const void *> &weight,
  const std::vector<int>          &K,
  const std::vector<int>          &N,
  const std::vector<int>          &ldb,
  const std::vector<bool>         &transB,
  const std::vector<bool>         &is_weights_const,
  size_t                           total_matmul,
  undersized_info                 *out_first_failure /* nullable */) {
  auto under = [&](const char *name, size_t got) -> bool {
    if (got < total_matmul) {
      if (out_first_failure != nullptr) {
        *out_first_failure = {name, got, total_matmul};
      }
      return true;
    }
    return false;
  };
  if (under("weight",          weight.size())) {
    return true;
  }
  if (under("K",               K.size())) {
    return true;
  }
  if (under("N",               N.size())) {
    return true;
  }
  if (under("ldb",             ldb.size())) {
    return true;
  }
  if (under("transB",          transB.size())) {
    return true;
  }
  if (under("is_weights_const",is_weights_const.size())) {
    return true;
  }
  return false;
}

// ───────────────────────────────────────────────────────────────────────
// Diagnostic-mode unified input validation for group_matmul_direct.
//
// Two-tier validation contract:
//
//   (i)  This function — `validate_group_matmul_direct_inputs` —
//        owns the FULL Phase A-G input contract with rich
//        `log_error` diagnostics.  Wrapped in
//        `op_instrumentation::validate(...)`, whose gate defaults
//        to ENABLED; the body therefore runs on first call unless
//        `ZENDNNL_DIAGNOSTICS_ENABLE=0` was captured at process
//        start.  Production deployments that disable diagnostics
//        explicitly skip the entire body.
//
//   (ii) The dispatch body (`group_matmul_direct(...)` below) carries
//        an always-on inline guard that runs the SUBSET of Phase A
//        rules required for production memory-safety even when
//        diagnostics are off: empty-vector reject, parallel-only-
//        feature reject, per-vector size consistency, prepack-extras
//        weight-side metadata length, and gated-act `ldc` uniformity.
//        It returns `status_t::failure` without log_error so the
//        production CPU cost stays at ~20 O(1) checks per call.  The
//        full per-element / fused-MoE / moe-postop contracts (Phases
//        B-G) live ONLY here, behind the diagnostics gate.
//
//        See `group_matmul_direct(...)` lines around "Always-on
//        safety guards" for the inline subset.
//
// Phases (each short-circuits on failure so the diagnostic message
// is closest to the failing rule):
//
//   A  Input shapes & mode-feature compatibility
//        empty / size mismatch / src.size() ∈ {1, num_ops} /
//        moe_postop ⋅ gated_act ⋅ fused_moe → parallel-mode-only
//
//   B  Sequential mode (src.size() == 1)
//        per-element non-null + positive dims, uniform M, K[i]==N[i-1]
//
//   C  Parallel mode (src.size() == num_ops)
//        per-element dim sanity (M ≥ 0, N/K > 0); active-expert
//        non-null pointers (skip null-check on M[i]==0 inactive slots
//        — fused MoE workloads legitimately pass nullptr there)
//
//   D  Gated activation (parallel mode only)
//        f32/bf16/f16 dst, uniform dst dtype, even N
//
//   E  Row-major layout when fusing (parallel mode only)
//        layout[i] ∈ {'r','R'} when gated_act or fused_moe is active
//
//   F  Fused MoE (parallel mode only)
//        per-expert vector sizes; active-expert non-null pointers;
//        N_down/ldb_down/ldc_down range; bias_dt_down vs bias_down;
//        uniform N_down when also paired with moe_postop
//
//   G  MoE post-op
//        uniform N (or N_down if fused) and uniform dst dtype across
//        experts; validate_group_matmul_moe_postop with the correct D
// ───────────────────────────────────────────────────────────────────────
status_t validate_group_matmul_direct_inputs(
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
  const std::vector<matmul_params> &params,
  const group_matmul_moe_postop_params *moe_postop,
  const grp_matmul_gated_act_params *gated_act,
  const grp_matmul_fused_moe_params *fused_moe) {

  // ── Phase A ─ input shapes & mode-feature compatibility ────────────
  if (M.empty() || params.empty() || src.empty()) {
    log_error("group_matmul_direct: required input vector is empty");
    return status_t::failure;
  }

  // Active-set count.  When the framework signals via
  // `params[0].active_matmul` we honour that as the matmul-processing
  // count (the K fired experts are stored contiguously at the start
  // of every per-expert vector).  Legacy callers that don't fill the
  // field see `active_matmul == 0` and fall back to the original
  // convention of deriving from `M.size()`, so behaviour is
  // bit-identical for any caller that hasn't opted in.  See
  // `matmul_params` in `lowoha_common.hpp` for the field semantics.
  //
  // Contract checks for opt-in callers (active_matmul > 0):
  //   - `active_matmul <= M.size()` — the firing experts must fit
  //     inside the supplied input-side metadata.  Two sizing
  //     patterns are both legitimate:
  //         M.size() == active_matmul                  (compact;
  //                                                     benchdnn,
  //                                                     compact-form
  //                                                     frameworks),
  //         M.size() == total_matmul with M[active..total)=0 placeholders
  //                                                    (padded; gtests).
  //     Reject `active_matmul > M.size()` to avoid silently dropping
  //     work — that direction is unambiguously a caller bug.
  //   - `total_matmul == 0 || total_matmul >= active_matmul` —
  //     specifying fewer total experts than firing experts is
  //     logically impossible.  Reject early so the size validator
  //     and prepack module aren't asked to reconcile a contract
  //     they cannot satisfy.
  const bool framework_opt_in =
    (!params.empty() && params[0].active_matmul > 0);
  if (framework_opt_in) {
    const uint32_t am = params[0].active_matmul;
    const uint32_t tm = params[0].total_matmul;
    if (am > M.size()) {
      log_error("group_matmul_direct: params[0].active_matmul=", am,
                " exceeds M.size()=", M.size(),
                " (firing experts must have input-side metadata)");
      return status_t::failure;
    }
    if (tm > 0 && tm < am) {
      log_error("group_matmul_direct: params[0].total_matmul=", tm,
                " must be >= active_matmul=", am,
                " (total experts cannot be less than firing experts)");
      return status_t::failure;
    }
  }
  const size_t num_ops =
    framework_opt_in ? static_cast<size_t>(params[0].active_matmul)
    : M.size();

  // Fused-MoE internal-alloc detection — INDEPENDENT per side.
  //
  // Op1 (dst[])         and Op2 (fused_moe->dst_down[]) are detected
  // SEPARATELY so all four caller patterns work:
  //
  //   dst[]     | dst_down[] | Op1 destination     | Op2 destination
  //   ───────── | ────────── | ─────────────────── | ───────────────────
  //   no        | no         | library Op1 arena   | in-place src reuse
  //   no        | yes        | library Op1 arena   | caller's dst_down
  //   yes       | no         | caller's dst        | in-place src reuse
  //   yes       | yes        | caller's dst        | caller's dst_down
  //
  // Each side rejects the mixed-null state (some entries null,
  // others non-null) — that's an unambiguous contract break we
  // cannot disambiguate.  The active-range sweep skips a prepack-
  // extras tail of trailing nullptr placeholders that is otherwise
  // legitimate; the caller-side detection in
  // `group_matmul_fused_moe_execute` runs the same predicate via
  // the shared `detect_internal_alloc` helper.
  using zendnnl::lowoha::matmul::group_matmul_internal::
  detect_internal_alloc;
  using zendnnl::lowoha::matmul::group_matmul_internal::
  internal_alloc_mode;
  auto run_detect = [&](const std::vector<void *> &v,
                        const char *name,
  bool *out_internal) -> status_t {
    const status_t st = detect_internal_alloc(
      v, num_ops, /*fused_moe_present=*/(fused_moe != nullptr),
      internal_alloc_mode::sweep_active, out_internal);
    if (st != status_t::success) {
      log_error("group_matmul_direct: fused_moe ", name, " has a mixed "
                "null/non-null state — every active entry must be "
                "either nullptr (library-managed) or non-null "
                "(caller-allocated).  Mixing the two would silently "
                "route some experts to library-managed and others "
                "to caller-allocated, ignoring the non-null entries.");
    }
    return st;
  };
  bool fused_op1_internal = false;
  bool fused_op2_internal = false;
  if (run_detect(dst, "dst", &fused_op1_internal) != status_t::success) {
    return status_t::failure;
  }
  if (fused_moe != nullptr) {
    if (run_detect(fused_moe->dst_down, "dst_down",
                   &fused_op2_internal) != status_t::success) {
      return status_t::failure;
    }
  }

  // Vector sizes — strict equality for legacy callers (preserves the
  // original behaviour exactly, including rejection of accidentally
  // oversized vectors), relaxed to "≥ num_ops" only when the framework
  // opted in via `params[0].active_matmul > 0`.  In the relaxed mode
  // the framework can pass weight-side metadata at total_matmul (full
  // expert table for prepack) while input-side vectors stay at
  // active_matmul (or vice-versa); the dispatch still iterates
  // `[0, num_ops)` only.
  const size_check_ctx size_ctx{
    /*num_ops=*/num_ops,
    /*relaxed=*/(!params.empty() &&params[0].active_matmul > 0)};
  auto size_bad = [&](size_t s) {
    return size_ctx.fail(s);
  };
  if (size_bad(N.size()) || size_bad(K.size()) || size_bad(weight.size())
      || size_bad(lda.size()) || size_bad(ldb.size())
      || size_bad(layout.size()) || size_bad(transA.size())
      || size_bad(transB.size()) || size_bad(alpha.size())
      || size_bad(beta.size()) || size_bad(bias.size())
      || size_bad(is_weights_const.size()) || size_bad(params.size())
      || (!fused_op1_internal
          && (size_bad(dst.size()) || size_bad(ldc.size())))) {
    log_error("group_matmul_direct: vector size mismatch, num_ops=", num_ops);
    return status_t::failure;
  }

  // Prepack-extras contract (mirrors the always-on inline guard in
  // `group_matmul_direct(...)`): when the framework opts in with
  // `total_matmul > active_matmul` the prepack module iterates
  // `[0, total_matmul)` over the weight-side metadata vectors.  If
  // any of these is shorter than `total_matmul`, the warmer's
  // `bound = std::min({total_count, weight.size(), K.size(), N.size(),
  // ldb.size(), transB.size()})` silently truncates the warm — un-
  // warmed experts then trigger a runtime reorder spike when they
  // first become active, defeating eager prepack.  Reject up front
  // with a precise diagnostic so framework integrators can tell at a
  // glance which vector was undersized.  Diagnostic-only path
  // (gated by ZENDNNL_DIAGNOSTICS_ENABLE, default ON; bypassed only
  // when explicitly set to "0"); the always-on inline guard returns
  // the same failure without the log_error when diagnostics are
  // disabled.
  if (size_ctx.relaxed
      && params[0].total_matmul > params[0].active_matmul) {
    undersized_info first_failure{};
    if (prepack_extras_metadata_undersized(
          weight, K, N, ldb, transB, is_weights_const,
          /*total_matmul=*/params[0].total_matmul, &first_failure)) {
      log_error("group_matmul_direct: ", first_failure.name,
                ".size()=", first_failure.got,
                " < total_matmul=", first_failure.need,
                " — prepack-extras contract requires every weight-side "
                "metadata vector (weight, K, N, ldb, transB, "
                "is_weights_const) to be sized to at least total_matmul "
                "so the prepack module can warm every advertised "
                "expert without silent truncation");
      return status_t::failure;
    }
  }

  // Internal-alloc tightening: dst / ldc / ldc_down must be EITHER
  // empty (library owns the buffer) OR sized to at least num_ops
  // (caller passes all-null placeholders, possibly with prepack-
  // extras tail).  Op1 side gates on `fused_op1_internal`; Op2's
  // ldc_down on `fused_op2_internal`.  Reject in-between sizes that
  // suggest a malformed caller intent.
  if (fused_op1_internal) {
    if (!dst.empty() && size_bad(dst.size())) {
      log_error("group_matmul_direct: fused_moe Op1 internal-alloc requires "
                "dst to be empty or sized to num_ops; got dst.size()=",
                dst.size(), ", num_ops=", num_ops);
      return status_t::failure;
    }
    if (!ldc.empty() && size_bad(ldc.size())) {
      log_error("group_matmul_direct: fused_moe Op1 internal-alloc requires "
                "ldc to be empty or sized to num_ops; got ldc.size()=",
                ldc.size(), ", num_ops=", num_ops);
      return status_t::failure;
    }
  }
  if (fused_op2_internal) {
    if (!fused_moe->ldc_down.empty()
        && size_bad(fused_moe->ldc_down.size())) {
      log_error("group_matmul_direct: fused_moe internal-alloc requires "
                "ldc_down to be empty or sized to num_ops; got ldc_down.size()=",
                fused_moe->ldc_down.size(), ", num_ops=", num_ops);
      return status_t::failure;
    }
  }
  if (src.size() != 1 && size_bad(src.size())) {
    log_error("group_matmul_direct: src.size() must be 1 or num_ops");
    return status_t::failure;
  }
  if (moe_postop != nullptr && src.size() == 1) {
    log_error("group_matmul_direct: moe_postop is only supported in parallel mode");
    return status_t::failure;
  }
  if (gated_act != nullptr
      && gated_act->act != grp_matmul_gated_act_t::none
      && src.size() == 1) {
    log_error("group_matmul_direct: gated_act is only supported in parallel mode");
    return status_t::failure;
  }
  if (fused_moe != nullptr && src.size() == 1) {
    log_error("group_matmul_direct: fused_moe is only supported in parallel mode");
    return status_t::failure;
  }
  if (fused_moe != nullptr && fused_moe->down_weight.empty()) {
    log_error("group_matmul_direct: fused_moe is non-null but down_weight is "
              "empty; pass nullptr to disable");
    return status_t::failure;
  }

  // ── Phase B ─ sequential mode per-element rules ────────────────────
  if (src.size() == 1) {
    if (src[0] == nullptr) {
      log_error("group_matmul sequential: null src pointer");
      return status_t::failure;
    }
    for (size_t i = 0; i < num_ops; ++i) {
      if (weight[i] == nullptr || dst[i] == nullptr) {
        log_error("group_matmul sequential: null pointer at operation ", i);
        return status_t::failure;
      }
      if (M[i] <= 0 || N[i] <= 0 || K[i] <= 0) {
        log_error("group_matmul sequential: invalid dimensions at operation ",
                  i, ": M=", M[i], ", N=", N[i], ", K=", K[i]);
        return status_t::failure;
      }
    }
    for (size_t i = 1; i < num_ops; ++i) {
      if (M[i] != M[0]) {
        log_error("group_matmul sequential: M must be constant across layers, "
                  "M[0]=", M[0], ", M[", i, "]=", M[i]);
        return status_t::failure;
      }
      if (K[i] != N[i - 1]) {
        log_error("group_matmul sequential: dimension mismatch at layer ", i,
                  ": K[", i, "]=", K[i], " != N[", i - 1, "]=", N[i - 1]);
        return status_t::failure;
      }
    }
    // Sequential path has no fused-MoE / gated-act / moe_postop variants
    // (Phase A already rejected those combos), so we are done.
    return status_t::success;
  }

  // ── Phase C ─ parallel mode per-element rules ──────────────────────
  // Inactive experts (M[i] == 0) are valid in MoE workloads — dispatch
  // short-circuits empty rows, so null pointers are harmless there.
  // dst[] is enforced non-null only in legacy mode; in fused-MoE
  // internal-alloc mode the library owns Op1 dst.
  for (size_t i = 0; i < num_ops; ++i) {
    if (M[i] < 0 || N[i] <= 0 || K[i] <= 0) {
      log_error("group_matmul parallel: invalid dimensions at operation ", i,
                ": M=", M[i], ", N=", N[i], ", K=", K[i]);
      return status_t::failure;
    }
    if (M[i] > 0) {
      if (src[i] == nullptr || weight[i] == nullptr) {
        log_error("group_matmul parallel: null src/weight at active operation ",
                  i);
        return status_t::failure;
      }
      if (!fused_op1_internal && dst[i] == nullptr) {
        log_error("group_matmul parallel: null dst at active operation ", i);
        return status_t::failure;
      }
    }
  }

  const bool run_gated_act = (gated_act != nullptr
                              && gated_act->act != grp_matmul_gated_act_t::none);

  // ── Phase D ─ gated activation rules ───────────────────────────────
  if (run_gated_act) {
    const data_type_t act_dtype = params[0].dtypes.dst;
    if (act_dtype != data_type_t::f32
        && act_dtype != data_type_t::bf16
        && act_dtype != data_type_t::f16) {
      log_error("group_matmul_direct: gated_act requires f32, bf16, or f16 dst");
      return status_t::failure;
    }
    for (size_t i = 1; i < num_ops; ++i) {
      if (params[i].dtypes.dst != act_dtype) {
        log_error("group_matmul_direct: gated_act requires uniform dst dtype");
        return status_t::failure;
      }
    }
    for (size_t i = 0; i < num_ops; ++i) {
      if (N[i] % 2 != 0) {
        log_error("group_matmul_direct: gated_act requires even N, N[",
                  i, "]=", N[i]);
        return status_t::failure;
      }
    }
  }

  // ── Phase E ─ row-major required when fusing ───────────────────────
  if (run_gated_act || fused_moe != nullptr) {
    for (size_t i = 0; i < num_ops; ++i) {
      if (layout[i] != 'r' && layout[i] != 'R') {
        log_error("group_matmul_direct: gated_act/fused_moe requires "
                  "row-major layout, layout[", i, "]='", layout[i], "'");
        return status_t::failure;
      }
    }
  }

  // ── Phase F ─ fused MoE deep checks ────────────────────────────────
  if (fused_moe != nullptr) {
    // Mode (2) (internal-alloc) was detected up front by Phase A.  In
    // that mode dst_down / ldc_down are unused so their size and per-
    // element checks are skipped; instead we add a per-expert
    // lda[i] >= N_down[i] check (Op2 reuses src as its output).
    //
    // Mixed-state detection (defence-in-depth): Phase A's
    // `detect_internal` lambda already swept the active range for
    // each side and rejected mixed null/non-null.  This block re-
    // confirms the per-side invariant under DIAGNOSTICS, iterating
    // the active range only so a prepack-extras tail of trailing
    // nullptrs doesn't false-flag the active subset.
    if (fused_op1_internal) {
      const size_t dst_sweep = std::min<size_t>(num_ops, dst.size());
      for (size_t i = 0; i < dst_sweep; ++i) {
        if (dst[i] != nullptr) {
          log_error("group_matmul_direct: fused_moe Op1 internal-alloc "
                    "requires every active dst[i] to be nullptr; "
                    "dst[", i, "] is non-null.");
          return status_t::failure;
        }
      }
    }
    if (fused_op2_internal) {
      const size_t dd_sweep =
        std::min<size_t>(num_ops, fused_moe->dst_down.size());
      for (size_t i = 0; i < dd_sweep; ++i) {
        if (fused_moe->dst_down[i] != nullptr) {
          log_error("group_matmul_direct: fused_moe Op2 internal-alloc "
                    "requires every active dst_down[i] to be nullptr; "
                    "dst_down[", i, "] is non-null.");
          return status_t::failure;
        }
      }
    }

    // fused_moe per-expert vectors — required regardless of which
    // side is internal-alloc (the dispatcher always needs the
    // down-side weight metadata).  Output-side vectors gated per
    // side: dst_down/ldc_down need only reach num_ops when Op2 is
    // caller-allocated.
    if (size_bad(fused_moe->down_weight.size())
        || size_bad(fused_moe->N_down.size())
        || size_bad(fused_moe->ldb_down.size())
        || size_bad(fused_moe->bias_down.size())
        || (!fused_op2_internal
            && (size_bad(fused_moe->dst_down.size())
                || size_bad(fused_moe->ldc_down.size())))) {
      log_error("group_matmul_direct: fused_moe vector size mismatch");
      return status_t::failure;
    }
    for (size_t i = 0; i < num_ops; ++i) {
      if (N[i] % 2 != 0) {
        log_error("group_matmul_direct: fused_moe requires even N, N[",
                  i, "]=", N[i]);
        return status_t::failure;
      }
      if (M[i] > 0) {
        if (fused_moe->down_weight[i] == nullptr) {
          log_error("group_matmul_direct: fused_moe down_weight[", i,
                    "] is null at active expert");
          return status_t::failure;
        }
        if (!fused_op2_internal && fused_moe->dst_down[i] == nullptr) {
          log_error("group_matmul_direct: fused_moe dst_down[", i,
                    "] is null at active expert");
          return status_t::failure;
        }
      }
      if (fused_moe->N_down[i] <= 0) {
        log_error("group_matmul_direct: fused_moe N_down[", i, "]=",
                  fused_moe->N_down[i], " must be positive");
        return status_t::failure;
      }
      // `op2_k_for_act` returns N for `act == none` (full passthrough)
      // and N/2 for any gated act (gate+up collapse).  Using the helper
      // keeps the validator in sync with `group_matmul_fused_moe.cpp`'s
      // execute path; the previous unconditional `N[i] / 2` here under-
      // restricted `ldb_down` for `act == none` callers (validator
      // accepted ldb_down = N/2 but the execute path needs ldb_down = N).
      const grp_matmul_gated_act_t act_for_op2 =
        run_gated_act ? gated_act->act : grp_matmul_gated_act_t::none;
      const int K_down_i = op2_k_for_act(N[i], act_for_op2);
      const int min_ldb = transB[i] ? K_down_i : fused_moe->N_down[i];
      if (fused_moe->ldb_down[i] < min_ldb) {
        log_error("group_matmul_direct: fused_moe ldb_down[", i, "]=",
                  fused_moe->ldb_down[i], " < required=", min_ldb);
        return status_t::failure;
      }
      // Op2 output stride invariant — gated on which side owns the
      // Op2 destination:
      //   * caller-allocated (op2_internal == false) — ldc_down[i]
      //     must accommodate the N_down write per row.
      //   * library-managed (op2_internal == true) — Op2 writes
      //     BACK into src[i] with stride lda[i], so lda[i] must
      //     accommodate N_down[i] instead.  Typical MoE has
      //     hidden_dim = K_input = N_down so this is naturally
      //     satisfied; we still check.
      if (!fused_op2_internal) {
        if (fused_moe->ldc_down[i] < fused_moe->N_down[i]) {
          log_error("group_matmul_direct: fused_moe ldc_down[", i, "]=",
                    fused_moe->ldc_down[i],
                    " < N_down=", fused_moe->N_down[i]);
          return status_t::failure;
        }
      }
      else {
        if (M[i] > 0 && lda[i] < fused_moe->N_down[i]) {
          log_error("group_matmul_direct: fused_moe Op2 internal-alloc "
                    "requires lda[", i, "]=", lda[i],
                    " >= N_down[", i, "]=", fused_moe->N_down[i],
                    " (Op2 reuses src as output)");
          return status_t::failure;
        }
        // Op2 writes dst-typed elements into the caller's src[i].
        // When dst_elem > src_elem (e.g. bf16 src + f32 dst), every
        // Op2 row overruns the original src row stride and corrupts
        // memory.  Require matched precision; mixed-precision callers
        // must use legacy mode (caller-allocated dst_down).  This is
        // also enforced as an always-on guard inside
        // group_matmul_fused_moe_execute() — the diagnostic message
        // here is the richer one for caller debugging.
        if (M[i] > 0
            && params[i].dtypes.src != params[i].dtypes.dst) {
          log_error("group_matmul_direct: fused_moe internal-alloc requires "
                    "matching src/dst dtypes when Op2 reuses src as output; "
                    "params[", i, "].dtypes.src=",
                    static_cast<int>(params[i].dtypes.src),
                    ", params[", i, "].dtypes.dst=",
                    static_cast<int>(params[i].dtypes.dst));
          return status_t::failure;
        }
      }
      if (fused_moe->bias_down[i] != nullptr
          && fused_moe->bias_dt_down == data_type_t::none) {
        log_error("group_matmul_direct: fused_moe bias_down[", i,
                  "] is non-null but bias_dt_down is none");
        return status_t::failure;
      }
    }
    if (moe_postop != nullptr) {
      for (size_t i = 1; i < num_ops; ++i) {
        if (fused_moe->N_down[i] != fused_moe->N_down[0]) {
          log_error("group_matmul_direct: fused_moe + moe_postop requires "
                    "uniform N_down, N_down[0]=", fused_moe->N_down[0],
                    " N_down[", i, "]=", fused_moe->N_down[i]);
          return status_t::failure;
        }
      }
    }
  }

  // ── Phase G ─ MoE post-op rules ────────────────────────────────────
  if (moe_postop != nullptr) {
    for (size_t i = 1; i < num_ops; ++i) {
      if (N[i] != N[0]) {
        log_error("group_matmul_direct: moe_postop requires identical N "
                  "across experts");
        return status_t::failure;
      }
      if (params[i].dtypes.dst != params[0].dtypes.dst) {
        log_error("group_matmul_direct: moe_postop requires identical dst "
                  "dtype across experts");
        return status_t::failure;
      }
    }
  }
  // Validate moe_postop with the correct D: N_down[0] when fused, N[0]
  // otherwise.  The helper accepts moe_postop == nullptr (returns
  // success) so it is safe to call unconditionally.
  const int moe_D = (fused_moe != nullptr && !fused_moe->N_down.empty())
                    ? fused_moe->N_down[0] : N[0];
  if (validate_group_matmul_moe_postop(moe_postop, moe_D,
                                       params[0].dtypes.dst) != status_t::success) {
    return status_t::failure;
  }

  return status_t::success;
}

// ───────────────────────────────────────────────────────────────────────
// L1 APILOG helpers
//
// One log line per group_matmul_direct() call summarising every input
// the dispatch saw (shape + dtype + leading dims + per-expert
// uniformity flags) plus the dispatch decision (`mode`).  The line
// runs after the inner executor returns, so a model-level read top-to-
// bottom shows:
//   [L2 dispatch]  → [L3 executor]  → [L4 kernel]  → [L1]
// for every call.
//
// First-expert-only dumps (lda[0], ldb[0], …, wconst[0]) keep the line
// bounded for MoE workloads with many experts; when a value varies
// across experts we append a trailing "(*)" so readers see mixed state
// at a glance.  M is printed in full because it is the primary shape
// variance axis.
// ───────────────────────────────────────────────────────────────────────

inline const char *dt_name(data_type_t dt) {
  switch (dt) {
  case data_type_t::f32:
    return "f32";
  case data_type_t::bf16:
    return "bf16";
  case data_type_t::f16:
    return "f16";
  case data_type_t::s8:
    return "s8";
  case data_type_t::u8:
    return "u8";
  default:
    return "?";
  }
}

template <typename T>
inline const char *uniformity_marker(const std::vector<T> &v) {
  for (size_t i = 1; i < v.size(); ++i)
    if (v[i] != v[0]) {
      return "(*)";
    }
  return "";
}

} // namespace

status_t group_matmul_direct(const std::vector<char> &layout,
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
                             const group_matmul_moe_postop_params *moe_postop,
                             const grp_matmul_gated_act_params *gated_act,
                             const grp_matmul_fused_moe_params *fused_moe) {

  // ── Always-on safety guards (O(1), production-live) ───────────────
  // The dispatch below indexes every input vector up to num_ops, so
  // an OOB on any of them in production (DIAGNOSTICS off) would be
  // undefined behaviour.  This guard runs the Phase-A subset of the
  // diagnostic validator (`validate_group_matmul_direct_inputs`)
  // unconditionally so production builds stay memory-safe even with
  // diagnostics off.  Five check classes below; each returns
  // `status_t::failure` without log_error so the production CPU
  // cost stays at ~20 O(1) checks per call (well below 1 µs,
  // negligible vs the GEMM):
  //
  //   (1) primary-vector emptiness — params[0] is dereferenced
  //       unconditionally for thread / dtype resolution.
  //   (2) parallel-mode-only feature rejection — moe_postop /
  //       gated_act / fused_moe are only supported when src.size()
  //       == num_ops (one src per expert).  Allowing any of them
  //       through in sequential mode (src.size() == 1) and then
  //       failing late is unsafe: e.g., fused_moe with empty dst[]
  //       would mark Op1 as library-managed, bypass the dst sizing
  //       check, then segfault in the sequential dispatch's dst[i]
  //       writes.  Reject up front before the per-side
  //       internal-alloc detection runs.
  //   (3) per-vector size consistency — every required vector must
  //       be sized to num_ops (with the dst / ldc exception for
  //       fused-MoE Op1-internal-alloc, and the dst_down / ldc_down
  //       exception for Op2-internal-alloc, where the caller
  //       legitimately leaves them empty and the library owns the
  //       respective destination).
  //   (4) prepack-extras weight-side metadata length — when the
  //       framework opts in via `total_matmul > active_matmul`, the
  //       six weight-side vectors (weight, K, N, ldb, transB,
  //       is_weights_const) must be sized to at least total_matmul
  //       so the prepack module's `min({...})` clamp cannot silently
  //       truncate the warm.
  //   (5) gated-act `ldc` uniformity — ALGO 3 flat_n_tile auto-
  //       engages from `ldc[0] < N[0]` and applies the tight/wide
  //       decision uniformly across experts; mixed states would
  //       OOB-write or leak undefined output.
  //
  // Phase B-G (per-element null / dimension / fused-MoE /
  // moe-postop) plus the rich `log_error` messages remain behind
  // the `ZENDNNL_DIAGNOSTICS_ENABLE` gate (default ON; disabled
  // only when explicitly set to `0`) and live in
  // `validate_group_matmul_direct_inputs()`.
  if (M.empty() || params.empty() || src.empty()) {
    return status_t::failure;
  }

  // ── F16 ISA gate + reference-accum-type setup ────────────────────
  // The single-op path runs `kernel_select` per call, which both
  // ISA-gates F16 and publishes the AOCL-DLP F16 accumulator type to
  // the singleton config; the parallel group-matmul dispatch below
  // bypasses `kernel_select` entirely and goes straight to per-expert
  // execution, so without the same setup here:
  //
  //   1) F16 inputs would silently slip through to a kernel that
  //      touches F16 storage and produces undefined results on hosts
  //      without AVX-512-FP16.
  //   2) Tests (and any user) validating the F16 group-matmul output
  //      against the per-op reference would see systematic drift: the
  //      F16 AOCL kernels here accumulate in native F16, but the
  //      reference kernel reads `accum_type` from the singleton and
  //      defaults to F32 accumulation, yielding a different rounding
  //      profile.
  //
  // Run this scan once over `params[0..num_ops_input)` — every expert
  // shares the same F16 / non-F16 classification in practice (uniform
  // dtype per group is the documented contract for the F16 path), so
  // checking the first expert suffices for the ISA gate, and the
  // accum_type knob is a per-process singleton anyway. Iterate to be
  // safe against mixed-dtype callers and reject early on any F16
  // operand when the ISA is missing.
  const size_t f16_scan_n = std::min<size_t>(M.size(), params.size());

  // Per-expert predicates lifted out of the scan loop so the contract
  // ("any F16 operand?" + "AOCL-DLP F16 kernel?") reads as named
  // properties.  AOCL-DLP F16 GEMM accumulates in native F16; the
  // group qualifies for ref-accum=F16 publication only when *every*
  // F16-bearing expert routes through aocl_dlp / aocl_dlp_blocked.
  // Anything else (mixed algos, non-AOCL F16 kernels) falls back to
  // the F32 accum default so a stray non-AOCL expert can't poison the
  // reference comparison for the whole group.
  auto is_f16_op = [](const matmul_params &p) -> bool {
    return p.dtypes.src  == data_type_t::f16 ||
    p.dtypes.wei  == data_type_t::f16 ||
    p.dtypes.dst  == data_type_t::f16 ||
    p.dtypes.bias == data_type_t::f16;
  };
  auto is_aocl_dlp = [](const matmul_params &p) -> bool {
    return p.lowoha_algo == matmul_algo_t::aocl_dlp ||
    p.lowoha_algo == matmul_algo_t::aocl_dlp_blocked;
  };

  bool any_f16_operand   = false;
  // `true` is the AND identity; only consumed below when
  // `any_f16_operand` is also true, so the seed is never observed
  // for non-F16 groups.
  bool group_is_aocl_f16 = true;
  for (size_t i = 0; i < f16_scan_n; ++i) {
    const auto &p = params[i];
    if (!is_f16_op(p)) {
      continue;
    }
    any_f16_operand   = true;
    group_is_aocl_f16 = group_is_aocl_f16 && is_aocl_dlp(p);
  }
  if (any_f16_operand &&
      !zendnnl::common::zendnnl_platform_info().get_avx512_f16_status()) {
    log_error("group_matmul_direct: F16 data type is not supported on "
              "this platform (requires AVX-512-FP16).");
    return status_t::isa_unsupported;
  }
  if (any_f16_operand) {
    // Match the single-op behaviour: AOCL-DLP F16 → ref accum=F16,
    // everything else (incl. non-F16 paths) → F32. Restore F32
    // afterwards to avoid leaking F16 accum into a subsequent
    // non-F16 caller that reads the singleton.
    zendnnl::ops::matmul_config_t::instance().set_accum_type(
      group_is_aocl_f16 ? data_type_t::f16 : data_type_t::f32);
  }

  // (2) Parallel-only features must not appear in sequential mode.
  // The same rejection lives in validate_group_matmul_direct_inputs
  // Phase A with richer log_error messages; promoting to always-on
  // closes a production-time crash window when diagnostics are off.
  // Emit the contract name in the error so rollout debugging from a
  // production log (no DIAGNOSTICS) still pinpoints which gate the
  // caller violated without needing to enable diagnostics and
  // reproduce.
  if (src.size() == 1) {
    if (moe_postop != nullptr) {
      log_error("group_matmul_direct: moe_postop is only supported in "
                "parallel mode (src.size() == num_ops); got src.size() "
                "== 1 (sequential chain dispatch).  Either enable "
                "parallel mode by supplying a per-expert src[] vector, "
                "or drop the moe_postop argument.");
      return status_t::failure;
    }
    if (gated_act != nullptr
        && gated_act->act != grp_matmul_gated_act_t::none) {
      log_error("group_matmul_direct: gated_act is only supported in "
                "parallel mode (src.size() == num_ops); got src.size() "
                "== 1 (sequential chain dispatch).  Either enable "
                "parallel mode or set gated_act->act = none.");
      return status_t::failure;
    }
    if (fused_moe != nullptr) {
      log_error("group_matmul_direct: fused_moe is only supported in "
                "parallel mode (src.size() == num_ops); got src.size() "
                "== 1 (sequential chain dispatch).  Pass fused_moe = "
                "nullptr for sequential chains, or switch to parallel "
                "mode for MoE workloads.");
      return status_t::failure;
    }
  }

  {
    // Active-set count for the inline strict guard.  Mirrors the
    // diagnostic validator's logic so production and DIAGNOSTICS
    // builds agree on what "size mismatch" means.  Legacy callers
    // that don't set `active_matmul` see `no == M.size()` and the
    // checks below use strict equality (preserves the original
    // reject-oversized-vectors behaviour bit-for-bit).  Only when
    // the framework opts in via `active_matmul > 0` do we relax to
    // `≥ no`, accepting the prepack-extras tail.
    const bool inline_relaxed =
      (!params.empty() && params[0].active_matmul > 0);
    const size_t no = inline_relaxed
                      ? std::min<size_t>(params[0].active_matmul, M.size())
                      : M.size();
    const size_check_ctx inline_ctx{/*num_ops=*/no,
        /*relaxed=*/inline_relaxed};
    auto inline_size_bad = [&](size_t s) {
      return inline_ctx.fail(s);
    };
    // Per-side internal-alloc detection — independent for Op1
    // (dst[]) and Op2 (fused_moe->dst_down[]).  O(1) inference from
    // [0] is sufficient for the inline guard; the diagnostic
    // validator already runs the active-range mixed-state sweep
    // via the same `detect_internal_alloc` helper in sweep mode.
    // Only valid in parallel mode (the sequential rejection above
    // already returned for src.size()==1 with fused_moe set, so we
    // never compute these in the unsafe case).
    using group_matmul_internal::detect_internal_alloc;
    using group_matmul_internal::internal_alloc_mode;
    bool inline_op1_internal = false;
    bool inline_op2_internal = false;
    detect_internal_alloc(
      dst, /*num_ops=*/no, /*fused_moe_present=*/(fused_moe != nullptr),
      internal_alloc_mode::quick_o1, &inline_op1_internal);
    if (fused_moe != nullptr) {
      detect_internal_alloc(
        fused_moe->dst_down, /*num_ops=*/no,
        /*fused_moe_present=*/true, internal_alloc_mode::quick_o1,
        &inline_op2_internal);
    }
    // else: inline_op2_internal stays false (no fused-MoE, no Op2).
    if (inline_size_bad(N.size()) || inline_size_bad(K.size())
        || inline_size_bad(weight.size())
        || inline_size_bad(lda.size()) || inline_size_bad(ldb.size())
        || inline_size_bad(layout.size()) || inline_size_bad(transA.size())
        || inline_size_bad(transB.size()) || inline_size_bad(alpha.size())
        || inline_size_bad(beta.size()) || inline_size_bad(bias.size())
        || inline_size_bad(is_weights_const.size())
        || inline_size_bad(params.size())) {
      return status_t::failure;
    }
    // dst/ldc only required (sized to >= num_ops) when Op1 is
    // caller-allocated; same for dst_down/ldc_down vs Op2.
    if (!inline_op1_internal
        && (inline_size_bad(dst.size()) || inline_size_bad(ldc.size()))) {
      return status_t::failure;
    }
    if (fused_moe != nullptr && !inline_op2_internal
        && (inline_size_bad(fused_moe->dst_down.size())
            || inline_size_bad(fused_moe->ldc_down.size()))) {
      return status_t::failure;
    }
    if (src.size() != 1 && inline_size_bad(src.size())) {
      return status_t::failure;
    }

    // Prepack-extras contract: when the framework opts in with
    // `total_matmul > active_matmul`, the prepack module (see
    // `group_matmul/prepack/prepack_aocl_dlp.cpp::warm_pack_all_aocl_
    // dlp_experts`) iterates `[0, total_matmul)` over the weight-side
    // metadata vectors.  If any of these vectors is shorter than
    // `total_matmul`, the warmer's `bound = std::min({total_count,
    // weight.size(), K.size(), N.size(), ldb.size(), transB.size()})`
    // SILENTLY truncates the warm to the shorter length — the un-
    // warmed experts then trigger a runtime reorder spike when they
    // first become active, defeating the purpose of eager prepack.
    // Reject up front so callers learn at integration time rather
    // than discovering the latency surprise in production.
    //
    // Only the SIX warm-pack-iterated vectors are checked here:
    // weight, K, N, ldb, transB, is_weights_const.  Input-side
    // vectors (M, src, lda) and per-expert metadata (alpha, beta,
    // bias, layout, transA, params) stay at active_matmul (compact)
    // or total_matmul (padded) — both are legitimate per the
    // dispatcher contract.  `lda` is also not consumed by the
    // warmers (only `ldb` is), so we don't require it at total.
    if (inline_relaxed
        && params[0].total_matmul > params[0].active_matmul) {
      // Always-on path passes nullptr for `out_first_failure` — no
      // log_error in production builds.  The diagnostic validator
      // populates and logs the precise undersized-vector name.
      if (prepack_extras_metadata_undersized(
            weight, K, N, ldb, transB, is_weights_const,
            /*total_matmul=*/params[0].total_matmul,
            /*out_first_failure=*/nullptr)) {
        return status_t::failure;
      }
    }

    // (4) Cross-expert caller-layout uniformity for the gated-activation
    // path (ALGO 3 flat_n_tile auto-engage).  The planner infers the
    // tight vs wide swiglu writer from `ldc[0] < N[0]` and applies
    // that decision to EVERY expert — the tight branch writes
    // compacted [M, :I] output at each expert's stride `ldc[e]`.
    // A mixed state (expert 0 tight, expert i wide, or vice versa)
    // would either leave the wide expert's second half undefined
    // (garbage / stale bytes → silent wrong result downstream) or
    // OOB-write into the tight expert's undersized buffer (hard
    // memory-safety bug).  Catch up front so the guard is
    // independent of which algo env the user picks.  Only relevant
    // in parallel gated_act mode; fused_moe has its own uniform
    // Op1-arena layout; sequential mode doesn't use ALGO 3.
    // Op1-internal-alloc callers also bypass this (dst/ldc empty);
    // the gate `fused_moe == nullptr` already excludes any
    // fused-MoE caller (which is the only path that can be
    // op1_internal anyway), so the redundant op1 check is kept
    // explicit here only for symmetry with the rest of the file.
    if (!inline_op1_internal && src.size() != 1 && fused_moe == nullptr
        && gated_act != nullptr
        && gated_act->act != grp_matmul_gated_act_t::none) {
      bool seen_tight = false, seen_wide = false;
      // Iterate the active range only; non-fired tail slots (when
      // present) carry placeholder dimensions that wouldn't reflect
      // the caller's ldc-vs-N invariant.
      for (size_t e = 0; e < no; ++e) {
        if (M[e] <= 0) {
          continue;
        }
        if (ldc[e] < N[e]) {
          seen_tight = true;
        }
        else {
          seen_wide = true;
        }
        if (seen_tight && seen_wide) {
          return status_t::failure;
        }
      }
    }
  }

  // Diagnostic-only full input validation.  Runs by default (the
  // gate is ON when ZENDNNL_DIAGNOSTICS_ENABLE is unset or set to
  // anything other than "0"); collapses to a single predicted-
  // taken branch when explicitly disabled via
  // ZENDNNL_DIAGNOSTICS_ENABLE=0.  See the doc-block above
  // validate_group_matmul_direct_inputs() for the seven phases.
  status_t val = op_instrumentation::validate([&]() {
    return validate_group_matmul_direct_inputs(
             layout, transA, transB, M, N, K, alpha,
             src, lda, weight, ldb, bias, beta, dst, ldc,
             is_weights_const, params,
             moe_postop, gated_act, fused_moe);
  });
  if (val != status_t::success) {
    return val;
  }

  // ── Active-set + prepack accounting ───────────────────────────────
  // Two counts drive the rest of this function:
  //
  //   * `num_ops`        — matmul-processing count; every dispatcher
  //                        downstream of this point iterates `[0,
  //                        num_ops)`.  Honours the framework's new
  //                        `params[0].active_matmul` hint when set;
  //                        otherwise falls back to `M.size()` so
  //                        legacy callers see no change.
  //
  //   * `num_ops_total`  — prepack iteration count; consumed by the
  //                        per-ALGO prepack functions in
  //                        group_matmul/prepack/, NOT by this
  //                        dispatcher.  Each scheduling ALGO body
  //                        reads `params[0].total_matmul` itself when
  //                        building its `PrepackParams`, so the value
  //                        is no longer materialised here.  When the
  //                        field is unset (legacy callers),
  //                        `build_prepack_params` resolves
  //                        `num_ops_total = M.size()` so the prepack
  //                        module still warms the firing experts up
  //                        front under the uniform-eager semantic
  //                        (`ZENDNNL_GRP_MATMUL_PREPACK=1`, default).
  //                        Set the env to `0` to restore the strict
  //                        pre-PR / lazy-only behaviour.
  const size_t num_ops_input = M.size();
  const size_t num_ops =
    (params[0].active_matmul > 0)
    ? std::min<size_t>(params[0].active_matmul, num_ops_input)
    : num_ops_input;

  profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile) {
    profiler.tbp_start();
  }

  const char *gemm_mode = nullptr;

  const int32_t omp_mt = thread_guard::max_threads();
  const int32_t num_threads = resolve_num_threads(params[0].num_threads, omp_mt);
  thread_guard tg(num_threads, omp_mt);

  // Cache the apilog gate; the post-dispatch L1 summary at the bottom
  // of this function reads it through a single predicted-not-taken
  // branch when API logging is below info level.
  static const bool s_l1_log = apilog_info_enabled();

  if (src.size() == 1) {
    // ── Sequential chain dispatch ─────────────────────────────────────
    static unsigned int auto_version = get_auto_tuner_ver();
    for (size_t i = 0; i < num_ops; ++i) {
      matmul_batch_params_t bp;
      bp.Batch_A = 1;
      bp.Batch_B = 1;
      const void *cur_src = (i == 0) ? src[i] : dst[i - 1];
      int cur_lda = (i == 0) ? lda[i] : ldc[i - 1];

      matmul_algo_t kernel = kernel_select(
                               params[i], bp.Batch_A, bp.Batch_B,
                               1, M[i], N[i], K[i], num_threads, bias[i], is_weights_const[i],
                               transB[i]);

      params[i].num_threads = num_threads;
      matmul_execute(
        layout[i], transA[i], transB[i],
        M[i], N[i], K[i], alpha[i],
        cur_src, cur_lda, weight[i], ldb[i],
        bias[i], beta[i], dst[i], ldc[i],
        is_weights_const[i],
        size_of(params[i].dtypes.src), size_of(params[i].dtypes.dst),
        num_threads, kernel, params[i], bp, auto_version);
    }
    gemm_mode = "sequential";
  }
  else {
    // ── Parallel grouped dispatch ─────────────────────────────────────
    const bool run_gated_act = (gated_act != nullptr
                                && gated_act->act != grp_matmul_gated_act_t::none);
    const data_type_t act_dtype = run_gated_act
                                  ? params[0].dtypes.dst : data_type_t::none;

    // ── Ahead-of-time weight pre-pack ─────────────────────────────────
    // Moved out of this dispatcher entirely: each scheduling ALGO's
    // body in group_matmul_dispatch.cpp / group_matmul_m_tile.cpp /
    // group_matmul_n_tile.cpp now invokes its matching
    // `group_matmul_prepack::prepack_for_algo_X(...)` as the first
    // action.  Under the uniform-eager semantic, the per-ALGO call
    // warms `max(M.size(), params[0].total_matmul)` experts whenever
    // `ZENDNNL_GRP_MATMUL_PREPACK=1` (the default) — both the
    // framework-hint regime (`total_matmul > active_matmul`,
    // production MoE rotating-experts) and the legacy regime
    // (`active = total = 0` → both fall back to `M.size()`,
    // first-iter serial reorder cost).  The new module owns the
    // env-knob (`ZENDNNL_GRP_MATMUL_PREPACK`), the thread-local
    // fingerprint cache, the `resolve_kernel()` gate, and the AOCL /
    // custom-kernel branch logic.  See group_matmul/prepack/prepack.hpp
    // for the public contract.

    // ── Optional M-slice for the parallel dispatch path ──────────────
    // When `num_ops < M.size()` the framework appended prepack-extras
    // to the tail of the per-expert vectors; the dispatchers downstream
    // derive their iteration count from `M.size()`, so we need to
    // present them with an M vector trimmed to the active set.  The
    // other vectors don't need slicing — the dispatcher only indexes
    // them at `[0, num_ops)`, which is valid for any vector that
    // satisfied the size guard above.  When `num_ops == M.size()`
    // (legacy callers, OR new callers whose framework happens to size
    // M to the active count exactly), `M_eff` aliases the caller's
    // vector with no copy.
    std::vector<int> M_active_local;
    const std::vector<int> *M_eff_ptr = &M;
    if (num_ops < M.size()) {
      M_active_local.assign(
        M.begin(),
        M.begin()
        + static_cast<std::vector<int>::difference_type>(num_ops));
      M_eff_ptr = &M_active_local;
    }
    const std::vector<int> &M_eff = *M_eff_ptr;

    if (fused_moe != nullptr) {
      // Note on act=none + fused_moe: Op2 consumes the raw first-half
      // columns of Op1's [M, 2*dim] output as its input.  This is NOT
      // MoE-semantically equivalent to a gated workflow and is a
      // potential footgun for framework integrators — the grp_matmul
      // gtests use this path deliberately as a two-call reference.
      // Frameworks that want "Op1 → activation → Op2" MUST also pass
      // gated_act with a non-none activation.
      //
      // When moe_postop is also supplied the fused MoE op runs the
      // weighted-reduce internally as the natural Stage 4 of the
      // pipeline, so the dispatch body has nothing more to do here.
      status_t fused_st = group_matmul_fused_moe_execute(
                            *fused_moe,
                            run_gated_act ? gated_act->act : grp_matmul_gated_act_t::none,
                            act_dtype,
                            layout, transA, transB, M_eff, N, K, alpha,
                            src, lda, weight, ldb, bias, beta, dst, ldc,
                            is_weights_const, params, num_threads, &gemm_mode, moe_postop);
      if (fused_st != status_t::success) {
        return fused_st;
      }
    }
    else {
      // Non-fused path: Op1 + activation (fused where possible) followed
      // by separate Op2 / moe_postop as needed.
      //
      // Source dynamic quantization is gated by ZENDNNL_ENABLE_GROUP_DQ
      // (default on).  When on, the grouped pre-pass quantizes all expert
      // sources up front (rewriting `params_dispatch` to s8 + clearing
      // dynamic_quant) so `execute_expert_slice` does not re-quant.  When
      // off, the pre-pass is skipped and dynamic quant flows through the
      // per-expert `reorder_quantization_wrapper` inside
      // `execute_expert_slice` (legacy behaviour).
      std::vector<const void *> quantized_src;
      std::vector<int> quantized_lda;
      std::vector<matmul_params> params_dispatch = params;
      group_reorder_quant_buffers_t group_quant_buffers;
      bool group_quantized = false;
      if (get_grp_matmul_enable_group_dq()) {
        status_t group_quant_st = group_reorder_quantization_wrapper(
                                    src, lda, transA, M_eff, K, num_threads, params_dispatch,
                                    quantized_src, quantized_lda, group_quant_buffers,
                                    group_quantized);
        if (group_quant_st != status_t::success) {
          return group_quant_st;
        }
      }

      const bool act_fused = group_matmul_run_parallel_dispatch(
                               layout, transA, transB, M_eff, N, K, alpha,
                               group_quantized ? quantized_src : src,
                               group_quantized ? quantized_lda : lda,
                               weight, ldb, bias, beta, dst, ldc,
                               is_weights_const,
                               group_quantized ? params_dispatch : params,
                               num_threads, &gemm_mode,
                               run_gated_act ? gated_act->act : grp_matmul_gated_act_t::none,
                               act_dtype);

      if (run_gated_act && !act_fused) {
        status_t act_st = group_matmul_moe_act_execute(
                            gated_act, dst, M_eff, N, ldc, act_dtype, num_threads);
        if (act_st != status_t::success) {
          return act_st;
        }
      }

      if (moe_postop != nullptr) {
        status_t moe_st = group_matmul_moe_postop_execute(moe_postop, N[0],
                          num_threads, params[0].dtypes.dst);
        if (moe_st != status_t::success) {
          return moe_st;
        }
      }
    }
  }

  if (is_profile) {
    profiler.tbp_stop();
  }

  // ── L1 APILOG (single per-call summary) ───────────────────────────
  // Built once; consumed by both apilog (full structured line) and
  // profilelog (same line + timing breakdown).  Skipped entirely
  // when both gates are off.  See the helper-comment block above
  // `dt_name` for the format contract.
  if (s_l1_log || is_profile) {
    std::ostringstream ss;
    ss << "[GRP_MATMUL.CALL] num_ops=" << num_ops
       << " mode=" << (gemm_mode != nullptr ? gemm_mode : "null")
       << " exec_algo=" << executed_algo_from_gemm_mode(gemm_mode)
       << " threads=" << num_threads
       << " dtype=" << dt_name(params[0].dtypes.src)
       << ">"       << dt_name(params[0].dtypes.wei)
       << ">"       << dt_name(params[0].dtypes.dst)
       << " layout=" << layout[0] << uniformity_marker(layout)
       << " transA=" << (transA[0] ? 'T' : 'N')
       << uniformity_marker(transA)
       << " transB=" << (transB[0] ? 'T' : 'N')
       << uniformity_marker(transB)
       << " alpha[0]=" << alpha[0] << uniformity_marker(alpha)
       << " beta[0]="  << beta[0]  << uniformity_marker(beta)
       << " wconst[0]=" << (is_weights_const[0] ? 1 : 0)
       << uniformity_marker(is_weights_const)
       << " lda[0]=" << lda[0]
       << " ldb[0]=" << ldb[0]
       << " ldc[0]=" << (ldc.empty() ? -1 : ldc[0])
       << " N[0]="   << N[0]
       << " K[0]="   << K[0]
       << " M=[";

    int64_t m_sum = 0;
    for (size_t i = 0; i < num_ops; ++i) {
      if (i > 0) {
        ss << ',';
      }
      ss << M[i];
      m_sum += M[i];
    }
    ss << "](sum=" << m_sum << ")";

    // Fused-operation summary — empty list when this is a plain GEMM,
    // otherwise records activation kind, fused down-projection (with
    // N_down for cross-checking), and weighted-reduce post-op
    // (with token count + topk).
    const bool has_act = (gated_act != nullptr
                          && gated_act->act != grp_matmul_gated_act_t::none);
    const bool has_fused = (fused_moe  != nullptr);
    const bool has_moe   = (moe_postop != nullptr);
    ss << " fused=[";
    bool need_comma = false;
    if (has_act) {
      ss << "act=" << act_name(gated_act->act);
      need_comma = true;
    }
    if (has_fused) {
      if (need_comma) {
        ss << ',';
      }
      ss << "down_proj=N_down[0]=" << fused_moe->N_down[0];
      need_comma = true;
    }
    if (has_moe) {
      if (need_comma) {
        ss << ',';
      }
      ss << "moe_postop(tokens=" << moe_postop->num_tokens
         << ",topk=" << moe_postop->topk << ')';
      need_comma = true;
    }
    if (!need_comma) {
      ss << "none";
    }
    ss << ']';
    ss << " sequential_chain=" << (src.size() == 1 ? 1 : 0);

    if (s_l1_log) {
      apilog_info(ss.str());
    }
    if (is_profile)
      profilelog_verbose(ss.str(), " time=", profiler.tbp_elapsedtime(),
                         profiler.get_res_str());
  }

  // Test-only inspection hook: publish the resolved `gemm_mode`
  // (a static literal owned by `flat_n_tile`'s `gemm_mode_label`
  // or one of the per-algo executors) so gtests can assert which
  // executor path actually ran without a public-API change.
  //
  // Gated on `s_capture_gemm_mode`:
  //   * Production builds never arm it → branch-not-taken on a
  //     relaxed atomic load whose cache line is in Shared state
  //     across cores.  No coherence traffic, ~1 cycle total.
  //   * Tests arm via `GemmModeCaptureGuard` (RAII in
  //     `moe_test_utils.hpp`) for the scope of the assertion.
  // Without the gate the unconditional store marks its cache
  // line Modified on every dispatcher call, ping-ponging the
  // line across cores under concurrent traffic — a hidden tax
  // for multi-rank serving deployments that have no use for
  // the hook.  See the doc-block on `s_capture_gemm_mode` in
  // `group_matmul/group_matmul_parallel_common.hpp`.
  if (zendnnl::lowoha::matmul::test_api::s_capture_gemm_mode.load(
        std::memory_order_relaxed)) {
    zendnnl::lowoha::matmul::test_api
    ::s_last_group_matmul_direct_gemm_mode.store(
      gemm_mode, std::memory_order_relaxed);
  }

  return status_t::success;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
