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

/// BF16 custom microkernel — per-tile (M × NR) GEMM with optional
/// gated-activation epilogue applied directly in accumulator registers.
///
/// Engaged through a single call site: `flat_n_tile` (ALGO 3) in
/// group_matmul_n_tile.cpp.  That path serves both plain group_matmul
/// (ActKind::none) and the fused-MoE entry (which routes Op1 +
/// swiglu_oai_mul and Op2 + none through the parallel dispatcher,
/// landing in flat_n_tile when ALGO 3 is selected).  The microkernel
/// honours the caller's ldc so the same code covers both the wide
/// (ldc = 2I) and tight (ldc = I) destination layouts without a
/// second implementation.
///
/// One microkernel call computes the FP32 accumulator
///
///   C[0..MR, 0..NR] = A[0..MR, 0..K] @ B_packed[0..K, 0..NR]
///
/// where NR = NV * 16, then either
///   * stores raw BF16 output (Act = none), or
///   * applies the gated activation in registers and stores the
///     halved BF16 output to a tight destination (Act = swiglu_oai_mul).
///
/// Compile-time template parameters:
///   * MR ∈ {1..8}  — row count handled per call.
///   * NV ∈ {2, 4}  — accumulator zmms per row (NR = NV × 16).
///   * Act          — none or swiglu_oai_mul.
///
/// Inner-loop pattern follows `bf16_brgemm_ukernel.cpp`:
///   * K-pair unroll-by-2 to expose ILP between two B-load batches
///   * VDPBF16PS (AVX512_BF16) for one FMA per 2 K-elements
///   * `_mm512_cvtneps_pbh` for the FP32→BF16 store conversion
///
/// Register-pressure caps (Zen4/5: 32 zmm):
///   * NV=2 (NR=32): max MR=8 — 16 acc + 2 b + 1 a ≈ 19 zmms (clean)
///   * NV=4 (NR=64): max MR=6 — 24 acc + 4 b + 1 a ≈ 29 zmms (clean);
///                   MR=8 spills 5 zmms (matches BRGEMM).

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_BF16_MICROKERNEL_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_BF16_MICROKERNEL_HPP

#include "common/bfloat16.hpp"
#include "../pack.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

/// Activation kinds the microkernel epilogue supports.  `swiglu_oai_mul`
/// is used by the fused-MoE Op1 tight-arena path (halved output);
/// `none` is used by the non-fused flat_n_tile path and by Op2 of the
/// fused-MoE flow.  `silu_and_mul` / `gelu_and_mul` use a split layout
/// that does not fit the in-register epilogue and stay on the standard
/// dispatcher.
enum class ActKind {
  none,            ///< Store full NR BF16 cols.
  swiglu_oai_mul,  ///< Apply swiglu in registers, store NR/2 BF16 cols.
};

/// Bias data-type — resolved at `prepare_for_call()` time and stored in
/// `CallContext::bias_kind`.  The microkernel branches on this once per
/// tile (outside the K loop) to pick the correct load width:
///   * `none` → skip the bias add entirely (bias pointer ignored).
///   * `bf16` → load 16 BF16 cols per NV step, convert to FP32.
///   * `fp32` → load 16 FP32 cols per NV step directly.
/// Keeping `bias_kind` as a runtime argument (not a template parameter)
/// avoids doubling the specialisation count — the bias-add block runs
/// once per (M, NR) tile and the branch cost is negligible vs the FMA
/// chain.
enum class BiasKind {
  none,
  bf16,
  fp32,
};

/// Destination data-type — selects the kernel's store-epilogue branch.
/// The internal accumulators stay in FP32 zmms regardless; the only
/// difference is whether the final store narrows back to BF16 (manual
/// RNE conversion via `fp32_to_bf16x16_rne`) or stores the FP32 lanes
/// straight via `_mm512_storeu_ps`.  Resolved at `prepare_for_call()`
/// time from the `KernelVariant` and threaded through `select_ukernel`
/// so each (MR, NV, Act, DstDt) pair has its own ukernel instantiation.
///
/// Constraint: `Act == ActKind::swiglu_oai_mul` is only valid with
/// `DstDt::kBf16` — the swiglu fused epilogue's store path uses the
/// half-width BF16 pair-pack helper (`swiglu_oai_store_pair`), and the
/// downstream consumers (Op2 in fused MoE) read BF16.  `select_ukernel`
/// returns `nullptr` for the (swiglu, kF32) tuple so the dispatcher's
/// `fill_kfn_table` refuses the call cleanly.
enum class DstDt {
  kBf16,
  kF32,
};

/// True when the running CPU supports AVX512_BF16 (VDPBF16PS).
/// Cached after first call.  When false the dispatcher must fall back.
bool avx512bf16_available();

/// Function-pointer type for one (MR, NV, Act, DstDt) microkernel
/// specialization.  Whichever of `Cout` / `Cout_tight` is unused is
/// passed nullptr / 0 — see the dispatcher for argument routing.
/// `bias` may be BF16 or FP32; `bias_kind` tells the kernel how to
/// load it.  Pass `bias=nullptr` / `bias_kind=BiasKind::none` when no
/// bias is applied.
///
/// `Cout` / `Cout_tight` are typed `void *` because the destination
/// element width depends on the kernel's `DstDt` template parameter
/// (BF16 = 2 bytes, FP32 = 4 bytes).  The dispatcher passes the same
/// caller-owned buffer cast to `void *`; the kernel reinterprets it
/// as `DstT *` internally.  `ldc` / `ldc_tight` stay in element units
/// (not bytes) — the kernel knows the element width via `DstT`.
using ukernel_fn_t = void (*)(
    const bfloat16_t *A, int lda,
    const bfloat16_t *Bpacked,
    const void       *bias, BiasKind bias_kind,
    void       *Cout, int ldc,
    void       *Cout_tight, int ldc_tight,
    int K);

/// Upper bound on `max_mr_for_nv(NV)` across all supported NV values.
/// Used to size the per-MR function-pointer table in `CallContext`
/// (`kfn_table[kMaxMR + 1]`; slot 0 is unused).  Keeping it as a
/// single constant makes the table sizing independent of future
/// additions (larger MR for int8, or another NV value).  Current
/// upper bound is 8, from NV=2 (NR=32) where register budget allows
/// MR=8; NV=4 caps at MR=6 (register pressure).
inline constexpr int kMaxMR = 8;

/// Maximum MR safe for this NV under the 32-zmm AVX-512 register
/// budget (matches the BRGEMM ukernel instantiation set).  Must
/// return a value ≤ `kMaxMR` above.
inline int max_mr_for_nv(int NV) {
  return NV == 4 ? 6 : 8;
}

/// Runtime selector — returns the function pointer for the requested
/// (MR ∈ 1..max_mr_for_nv(NV), NV ∈ {2, 4}, Act, DstDt) tuple, or
/// nullptr if the combination is not instantiated.
///
/// (swiglu_oai_mul, DstDt::kF32) is intentionally NOT instantiated
/// (returns nullptr) — see the `DstDt` doc-block above.
ukernel_fn_t select_ukernel(int MR, int NV, ActKind act, DstDt dst_dt);

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_BF16_MICROKERNEL_HPP
