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

// ─────────────────────────────────────────────────────────────────────
// Internal helpers — owned entirely by this file.  The callers never call
// these directly; the only public entries are `prepare_for_call()`
// and `dispatch_tile()` below.
// ─────────────────────────────────────────────────────────────────────
namespace {

// Pick the pack/microkernel NR for one (K, N) shape.  Env override
// (from `get_grp_matmul_custom_kernel_nr()` in parallel_common.hpp)
// wins when it divides N; otherwise default to NR=32 (matches vLLM's
// `2 × NSize` inner-loop step) since it gives the cleanest register
// budget and the smallest M chunking on small-M decode shapes.
// NR=64 stays available via env override for shapes that benefit.
int plan_pack_nr(int /*K*/, int N) {
  if (N <= 0) return 0;
  const int env_nr = get_grp_matmul_custom_kernel_nr();
  if (env_nr != 0) return ((N % env_nr) == 0) ? env_nr : 0;
  if ((N % kNRMin) == 0) return kNRMin;
  if ((N % kNRMax) == 0) return kNRMax;
  return 0;
}

// L2-friendly sub-tile width (in cols) for (M_max, K, pack_nr).
// Sized so each microkernel call's working set — input + the B
// strip this thread streams — fits in this CPU's per-core L2 with
// 20 % headroom.  Accumulators live in zmm registers, so they
// don't enter the budget.  Returned value is a multiple of pack_nr,
// minimum one o-block.
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
// (NV, act_kind) pair.  Called once by `prepare_for_call()`.  Per-
// tile lookup is then a direct `kfn_table[mr_now]` — no switch.
status_t fill_kfn_table(
    int NV, ActKind act_kind, ukernel_fn_t (&kfn_table)[kMaxMR + 1]) {
  const int max_mr = max_mr_for_nv(NV);
  for (int mr = 1; mr <= max_mr; ++mr) {
    kfn_table[mr] = select_ukernel(mr, NV, act_kind);
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
    CallContext &out) {

  out.enabled = false;

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
  // A / B / C dtype gate: the custom microkernel implements only the
  // bf16 × bf16 → bf16 VDPBF16PS math path.  Accepting any dtype other
  // than bf16 for src / wei / dst would cause the microkernel to
  // reinterpret the caller's buffers as bf16 and produce corrupt
  // output (no runtime type conversion inside the kernel).  Callers
  // on mixed-precision paths (e.g. fp32 weights + bf16 dst) fall back
  // to the standard path on dispatcher refusal.
  if (src_dtype != data_type_t::bf16
      || wei_dtype != data_type_t::bf16
      || dst_dtype != data_type_t::bf16) {
    if (s_refuse_log) {
      apilog_info("[GRP_MATMUL Level4 custom_kernel REFUSED] reason="
                  "unsupported_dtype (src=", dt_name(src_dtype),
                  " wei=", dt_name(wei_dtype),
                  " dst=", dt_name(dst_dtype),
                  " — custom kernel is bf16×bf16→bf16 only)");
    }
    return status_t::failure;
  }
  if (act != grp_matmul_gated_act_t::swiglu_oai_mul
      && act != grp_matmul_gated_act_t::none) {
    return refuse("unsupported_activation",
                  "custom kernel fuses only swiglu_oai_mul or none");
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
  // before we index it inside the pack).
  if (ldb.size() != static_cast<size_t>(num_ops))
    return refuse("ldb_size_mismatch",
                  "ldb vector size != num_ops");
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

  if (fill_kfn_table(out.NV, out.act_kind, out.kfn_table)
      != status_t::success) {
    return refuse("kfn_table_fill_failed",
                  "no microkernel for this (NV, act_kind) pair");
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

void dispatch_tile(
    const CallContext &ctx,
    int   expert_idx,
    int   M, int K,
    int   n_tile, int col_start,
    const void *src,  int lda,
    const void *bias,
    void       *tight_dst, int tight_ldc) {

  // Tile-shape contract for the custom microkernel: every per-thread
  // tile produced by `aligned_n_split` must be a multiple of
  // `pack_nr` (the kernel processes B in `pack_nr`-wide blocks via
  // truncating `n_blocks = sub_n / pack_nr`; any non-multiple tail
  // would be SILENTLY DROPPED, leaving dst columns uninitialised).
  //
  // This holds for the current production envelope because:
  //   1. `prepare_for_call` refuses (N % pack_nr != 0);
  //   2. for any `n_thr ≤ N / pack_nr`, `aligned_n_split` always
  //      finds an aligned candidate that satisfies the 2× imbalance
  //      bound, never falling back to its unaligned even-split path;
  //   3. `participating_n_thr` upstream caps `n_thr` by
  //      `N / min_n_tile` (= `N / kDecodeNTile` = `N / 256`), and
  //      `min_n_tile >= pack_nr`, so condition 2 holds.
  //
  // Asserts are debug-only; in release the contract is upheld by the
  // upstream gates above.  If a future change introduces a per-tile
  // shape that violates either contract, this assert turns the
  // silent drop into a loud debug failure rather than wrong output.
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
  auto             *Tight        = static_cast<bfloat16_t *>(tight_dst);

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
        bfloat16_t *Tight_row_base = Tight
            + static_cast<size_t>(m_off) * tight_ldc + (sub_col_base / 2);

        for (int b = 0; b < n_blocks; ++b) {
          const bfloat16_t *Bpacked_blk =
              Bpacked_blk_base + static_cast<size_t>(b) * o_blk_stride;
          const void *bias_blk = (bias_blk_base != nullptr)
              ? static_cast<const void *>(bias_blk_base
                  + static_cast<size_t>(b) * ctx.pack_nr * bias_elem_bytes)
              : nullptr;
          // 16 cols per (g, u) pair × NV/2 pairs per kernel call.
          bfloat16_t *Tight_row = Tight_row_base + b * (ctx.pack_nr / 2);

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
      bfloat16_t *Wide_row_base = Tight
          + static_cast<size_t>(m_off) * tight_ldc + sub_col_base;

      for (int b = 0; b < n_blocks; ++b) {
        const bfloat16_t *Bpacked_blk =
            Bpacked_blk_base + static_cast<size_t>(b) * o_blk_stride;
        const void *bias_blk = (bias_blk_base != nullptr)
            ? static_cast<const void *>(bias_blk_base
                + static_cast<size_t>(b) * ctx.pack_nr * bias_elem_bytes)
            : nullptr;
        bfloat16_t *Wide_row = Wide_row_base + b * ctx.pack_nr;

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
