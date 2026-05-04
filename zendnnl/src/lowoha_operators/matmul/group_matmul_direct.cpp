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
///   - group_matmul/group_matmul_parallel.cpp — parallel expert dispatch (OMP).
///   - group_matmul/group_matmul_moe_postop.cpp — optional MoE weighted-reduce post-op.

#include <sstream>
#include <vector>

#include "group_matmul/group_matmul_direct.hpp"
#include "lowoha_matmul_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::size_of;
using zendnnl::common::op_instrumentation;

namespace {

// ───────────────────────────────────────────────────────────────────────
// Unified input validation for group_matmul_direct.
//
// This is the single source of truth for what valid inputs to
// group_matmul_direct() look like.  The dispatch body (below) does
// NOT redo any of these checks inline; it assumes inputs are
// well-formed and runs straight to the kernel calls.
//
// Invocation contract:
//   * Always called from inside `op_instrumentation::validate(...)`,
//     so the entire body is gated on `ZENDNNL_DIAGNOSTICS_ENABLE=1`.
//     Production runs (env unset / =0) skip every check below — the
//     framework integrating this library is contractually responsible
//     for passing valid inputs.
//   * The dispatch body keeps a one-line empty-vector guard so that
//     param[0] / src / M can be dereferenced for thread / mode
//     resolution without segfaulting on adversarial empty inputs;
//     everything richer than that is exclusively here.
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
//        f32/bf16 dst, uniform dst dtype, even N
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

  const size_t num_ops = M.size();

  // Fused-MoE internal-alloc + src-reuse mode detection.
  //
  // Engagement contract: the caller clears BOTH Op1 and Op2 output
  // handles — `fused_moe->dst_down` is empty AND dst[] is either
  // empty (size 0) or EVERY element is null.  In that state the
  // library owns the Op1 arena and reuses src as the Op2 output
  // buffer; `dst` and `ldc` are ignored.
  //
  // Mode inference is ALWAYS-ON and a full O(num_ops) sweep to
  // reject mixed states (e.g. `dst[0]=null, dst[1]=ptr`) in
  // production too.  A previous two-level design inferred the mode
  // from `dst[0]` alone and ran the consistency sweep only behind
  // ZENDNNL_DIAGNOSTICS_ENABLE; that silently routed expert 1's
  // output back into src[1] while ignoring the caller-provided
  // dst[1] — a contract break.  The always-on sweep costs one
  // predicted-not-taken branch per expert (< 50 ns for 32 experts)
  // and is far cheaper than the GEMM that follows.
  bool fused_internal_alloc = false;
  if (fused_moe != nullptr && fused_moe->dst_down.empty()) {
    if (dst.empty()) {
      fused_internal_alloc = true;
    } else {
      bool any_null = false, any_nonnull = false;
      for (size_t i = 0; i < dst.size(); ++i) {
        if (dst[i] == nullptr) any_null = true;
        else any_nonnull = true;
      }
      if (any_null && any_nonnull) {
        log_error("group_matmul_direct: fused_moe dst[] has a mixed "
                  "null/non-null state — internal-alloc mode requires "
                  "EVERY dst[i] to be nullptr (or dst[] to be empty); "
                  "legacy mode requires every dst[i] to be non-null.  "
                  "Mixing the two would silently route some experts "
                  "to internal-alloc and others to caller-allocated, "
                  "ignoring the non-null entries.");
        return status_t::failure;
      }
      fused_internal_alloc = any_null;  // all null → internal-alloc
    }
  }

  if (N.size() != num_ops || K.size() != num_ops || weight.size() != num_ops
      || lda.size() != num_ops || ldb.size() != num_ops
      || layout.size() != num_ops || transA.size() != num_ops
      || transB.size() != num_ops || alpha.size() != num_ops
      || beta.size() != num_ops || bias.size() != num_ops
      || is_weights_const.size() != num_ops || params.size() != num_ops
      || (!fused_internal_alloc
          && (dst.size() != num_ops || ldc.size() != num_ops))) {
    log_error("group_matmul_direct: vector size mismatch, num_ops=", num_ops);
    return status_t::failure;
  }

  // Internal-alloc tightening: dst and ldc must be EITHER empty (size 0,
  // library owns both) OR sized exactly to num_ops (caller passes
  // all-null dst placeholders / zero ldc placeholders).  Reject in-
  // between sizes (e.g. dst.size() == 1 with num_ops == 32) which
  // suggest a malformed caller intent rather than legitimate use.
  if (fused_internal_alloc) {
    if (!dst.empty() && dst.size() != num_ops) {
      log_error("group_matmul_direct: fused_moe internal-alloc requires dst "
                "to be empty or sized to num_ops; got dst.size()=",
                dst.size(), ", num_ops=", num_ops);
      return status_t::failure;
    }
    if (!ldc.empty() && ldc.size() != num_ops) {
      log_error("group_matmul_direct: fused_moe internal-alloc requires ldc "
                "to be empty or sized to num_ops; got ldc.size()=",
                ldc.size(), ", num_ops=", num_ops);
      return status_t::failure;
    }
    if (!fused_moe->ldc_down.empty()
        && fused_moe->ldc_down.size() != num_ops) {
      log_error("group_matmul_direct: fused_moe internal-alloc requires "
                "ldc_down to be empty or sized to num_ops; got ldc_down.size()=",
                fused_moe->ldc_down.size(), ", num_ops=", num_ops);
      return status_t::failure;
    }
  }
  if (src.size() != 1 && src.size() != num_ops) {
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
      if (!fused_internal_alloc && dst[i] == nullptr) {
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
    if (act_dtype != data_type_t::f32 && act_dtype != data_type_t::bf16) {
      log_error("group_matmul_direct: gated_act requires f32 or bf16 dst");
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
    // Mixed-state detection: Phase A keyed on dst[0] == nullptr (O(1)).
    // Now do a full pass over dst[] to confirm it's actually all-null
    // when internal-alloc is engaged — and confirm it's all-non-null
    // when legacy mode is engaged.  Catches `dst[0]=null,dst[1]=ptr`
    // and the inverse.
    const bool internal_alloc = fused_internal_alloc;
    if (internal_alloc) {
      for (size_t i = 0; i < dst.size(); ++i) {
        if (dst[i] != nullptr) {
          log_error("group_matmul_direct: fused_moe internal-alloc requires "
                    "every dst[i] to be nullptr; dst[", i, "] is non-null "
                    "(caller appears to have a mixed state).");
          return status_t::failure;
        }
      }
    } else {
      // Legacy mode: dst_down must be present and sized to num_ops
      // (already checked by Phase F's size block below).  No per-
      // element dst[] check here — Phase C already required dst[i]
      // non-null for active experts.
    }

    if (fused_moe->down_weight.size() != num_ops
        || fused_moe->N_down.size() != num_ops
        || fused_moe->ldb_down.size() != num_ops
        || fused_moe->bias_down.size() != num_ops
        || (!internal_alloc
            && (fused_moe->dst_down.size() != num_ops
                || fused_moe->ldc_down.size() != num_ops))) {
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
        if (!internal_alloc && fused_moe->dst_down[i] == nullptr) {
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
      const int K_down_i = N[i] / 2;
      const int min_ldb = transB[i] ? K_down_i : fused_moe->N_down[i];
      if (fused_moe->ldb_down[i] < min_ldb) {
        log_error("group_matmul_direct: fused_moe ldb_down[", i, "]=",
                  fused_moe->ldb_down[i], " < required=", min_ldb);
        return status_t::failure;
      }
      if (!internal_alloc) {
        if (fused_moe->ldc_down[i] < fused_moe->N_down[i]) {
          log_error("group_matmul_direct: fused_moe ldc_down[", i, "]=",
                    fused_moe->ldc_down[i],
                    " < N_down=", fused_moe->N_down[i]);
          return status_t::failure;
        }
      } else {
        // Internal-alloc reuses src[i] as Op2 output with stride
        // lda[i].  Each Op2 row writes N_down[i] columns; for the
        // write to fit within the original src row stride we need
        // lda[i] >= N_down[i].  Typical MoE has hidden_dim = K_input
        // = N_down so this is naturally satisfied.
        if (M[i] > 0 && lda[i] < fused_moe->N_down[i]) {
          log_error("group_matmul_direct: fused_moe internal-alloc requires "
                    "lda[", i, "]=", lda[i], " >= N_down[", i, "]=",
                    fused_moe->N_down[i],
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
          params[0].dtypes.dst) != status_t::success)
    return status_t::failure;

  return status_t::success;
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
  // undefined behaviour.  Keep three cheap always-on checks here:
  //
  //   (1) primary-vector emptiness — params[0] is dereferenced
  //       unconditionally for thread / dtype resolution.
  //   (2) parallel-mode-only feature rejection — moe_postop /
  //       gated_act / fused_moe are only supported when src.size()
  //       == num_ops (one src per expert).  Allowing any of them
  //       through in sequential mode (src.size() == 1) and then
  //       failing late is unsafe: e.g., fused_moe with empty dst[]
  //       would set fused_ialloc=true, bypass the dst sizing check,
  //       then segfault in the sequential dispatch's dst[i] writes.
  //       Reject up front before fused_ialloc is even computed.
  //   (3) per-vector size consistency — every required vector must
  //       be sized to num_ops (with the dst / ldc exception for
  //       fused-MoE internal-alloc, where the caller legitimately
  //       leaves them empty and the library owns Op1 dst).
  //
  // Per-element null / dimension / fused-MoE / moe-postop contracts
  // remain behind ZENDNNL_DIAGNOSTICS_ENABLE and live in
  // validate_group_matmul_direct_inputs().  Total cost here is ~20
  // O(1) checks — well below 1 µs, negligible vs the GEMM.
  if (M.empty() || params.empty() || src.empty())
    return status_t::failure;

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
    const size_t no = M.size();
    // fused_ialloc detection is only valid in parallel mode (the
    // sequential rejection above already returned for src.size()==1
    // with fused_moe set, so we never compute this in the unsafe
    // case).
    const bool fused_ialloc = (fused_moe != nullptr
        && fused_moe->dst_down.empty()
        && (dst.empty() || dst[0] == nullptr));
    if (N.size() != no || K.size() != no || weight.size() != no
        || lda.size() != no || ldb.size() != no
        || layout.size() != no || transA.size() != no
        || transB.size() != no || alpha.size() != no
        || beta.size() != no || bias.size() != no
        || is_weights_const.size() != no || params.size() != no)
      return status_t::failure;
    if (!fused_ialloc && (dst.size() != no || ldc.size() != no))
      return status_t::failure;
    if (src.size() != 1 && src.size() != no)
      return status_t::failure;

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
    // `fused_ialloc` callers also bypass this (dst/ldc empty).
    if (!fused_ialloc && src.size() != 1 && fused_moe == nullptr
        && gated_act != nullptr
        && gated_act->act != grp_matmul_gated_act_t::none) {
      bool seen_tight = false, seen_wide = false;
      for (size_t e = 0; e < no; ++e) {
        if (M[e] <= 0) continue;
        if (ldc[e] < N[e]) seen_tight = true;
        else seen_wide = true;
        if (seen_tight && seen_wide) return status_t::failure;
      }
    }
  }

  // Diagnostic-only full input validation.  No-op at near-zero cost
  // when ZENDNNL_DIAGNOSTICS_ENABLE is unset (single branch), full
  // contract enforcement when set.  See the doc-block above
  // validate_group_matmul_direct_inputs() for the seven phases.
  status_t val = op_instrumentation::validate([&]() {
    return validate_group_matmul_direct_inputs(
        layout, transA, transB, M, N, K, alpha,
        src, lda, weight, ldb, bias, beta, dst, ldc,
        is_weights_const, params,
        moe_postop, gated_act, fused_moe);
  });
  if (val != status_t::success)
    return val;

  const size_t num_ops = M.size();

  profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile)
    profiler.tbp_start();

  const char *gemm_mode = nullptr;

  const int32_t omp_mt = thread_guard::max_threads();
  const int32_t num_threads = resolve_num_threads(params[0].num_threads, omp_mt);
  thread_guard tg(num_threads, omp_mt);

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
          1, M[i], N[i], K[i], num_threads, bias[i], is_weights_const[i]);

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
  } else {
    // ── Parallel grouped dispatch ─────────────────────────────────────
    const bool run_gated_act = (gated_act != nullptr
        && gated_act->act != grp_matmul_gated_act_t::none);
    const data_type_t act_dtype = run_gated_act
        ? params[0].dtypes.dst : data_type_t::none;

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
          layout, transA, transB, M, N, K, alpha,
          src, lda, weight, ldb, bias, beta, dst, ldc,
          is_weights_const, params, num_threads, &gemm_mode, moe_postop);
      if (fused_st != status_t::success)
        return fused_st;
    } else {
      // Non-fused path: Op1 + activation (fused where possible) followed
      // by separate Op2 / moe_postop as needed.
      const bool act_fused = group_matmul_run_parallel_dispatch(
          layout, transA, transB, M, N, K, alpha,
          src, lda, weight, ldb, bias, beta, dst, ldc,
          is_weights_const, params, num_threads, &gemm_mode,
          run_gated_act ? gated_act->act : grp_matmul_gated_act_t::none,
          act_dtype);

      if (run_gated_act && !act_fused) {
        status_t act_st = group_matmul_moe_act_execute(
            gated_act, dst, M, N, ldc, act_dtype, num_threads);
        if (act_st != status_t::success)
          return act_st;
      }

      if (moe_postop != nullptr) {
        status_t moe_st = group_matmul_moe_postop_execute(moe_postop, N[0],
            num_threads, params[0].dtypes.dst);
        if (moe_st != status_t::success)
          return moe_st;
      }
    }
  }

  if (is_profile)
    profiler.tbp_stop();

  if (apilog_info_enabled() || is_profile) {
    auto dt_str = [](data_type_t dt) -> const char * {
      switch (dt) {
      case data_type_t::f32:  return "f32";
      case data_type_t::bf16: return "bf16";
      case data_type_t::f16:  return "f16";
      case data_type_t::s8:   return "s8";
      case data_type_t::u8:   return "u8";
      default:                return "?";
      }
    };

    std::ostringstream ss;
    ss << "LOWOHA group_matmul_direct: "
       << "num_ops=" << num_ops
       << ", mode=" << (gemm_mode != nullptr ? gemm_mode : "null")
       << ", threads=" << num_threads
       << ", dtype=" << dt_str(params[0].dtypes.src)
       << ">" << dt_str(params[0].dtypes.dst);

    // M values (vary per expert) + sum
    int64_t m_sum = 0;
    ss << ", M=[";
    for (size_t i = 0; i < num_ops; ++i) {
      if (i > 0) ss << ",";
      ss << M[i];
      m_sum += M[i];
    }
    ss << "](sum=" << m_sum << ")";

    ss << ", N[0]=" << N[0] << ", K[0]=" << K[0];

    // Fusion flags: compact summary of what was enabled
    const bool has_act = (gated_act != nullptr
        && gated_act->act != grp_matmul_gated_act_t::none);
    const bool has_fused = (fused_moe != nullptr);
    const bool has_moe = (moe_postop != nullptr);

    if (has_act || has_fused || has_moe) {
      ss << ", fused=[";
      bool need_comma = false;
      if (has_act) {
        static const char *act_names[] = {
            "none", "silu_and_mul", "gelu_and_mul", "swiglu_oai_mul"};
        int ai = static_cast<int>(gated_act->act);
        ss << "act=" << ((ai >= 0 && ai <= 3) ? act_names[ai] : "?");
        need_comma = true;
      }
      if (has_fused) {
        if (need_comma) ss << ",";
        ss << "down_proj=N_down[0]=" << fused_moe->N_down[0];
        need_comma = true;
      }
      if (has_moe) {
        if (need_comma) ss << ",";
        ss << "moe_postop(tokens=" << moe_postop->num_tokens
           << ",topk=" << moe_postop->topk << ")";
      }
      ss << "]";
    }

    apilog_info(ss.str());
    if (is_profile)
      profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(),
                         profiler.get_res_str());
  }

  return status_t::success;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
