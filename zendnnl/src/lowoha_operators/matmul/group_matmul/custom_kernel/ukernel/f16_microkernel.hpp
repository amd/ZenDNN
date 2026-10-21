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

/// FP16 custom microkernel — per-tile (M × NR) GEMM with optional
/// gated-activation epilogue applied directly in accumulator
/// registers.  Sibling of the BF16 microkernel in
/// `bf16_microkernel.{hpp,cpp}` — both files follow the same layout,
/// templating discipline, and dispatch contract.  The only
/// fundamental difference is the inner-loop instruction
/// (`_mm512_fmadd_ph` (AVX-512-FP16) vs `VDPBF16PS` (AVX-512-BF16))
/// and the accumulator element type (`__m512h` native FP16 lanes vs
/// the bf16 path's FP32 zmm accumulators).
///
/// Engaged through a single call site: `flat_n_tile` (ALGO 3) in
/// group_matmul_n_tile.cpp, exactly like the bf16 sibling.  That path
/// serves both plain group_matmul (`ActKind::none`) and the fused-MoE
/// entry (Op1 + a gated activation / Op2 + none).  The microkernel
/// honours the caller's ldc so the same code covers both the wide
/// (ldc = N) and tight (ldc = N/2) destination layouts without a
/// second implementation.
///
/// One microkernel call computes the accumulator
///
///   C[0..MR, 0..NR] = A[0..MR, 0..K] @ B_packed[0..K, 0..NR]
///
/// where NR = NV * 16, then either
///   * stores raw F16 / FP32 output (Act = none), or
///   * applies the gated activation in registers and stores the
///     halved F16 output to a tight destination (Act =
///     swiglu_oai_mul / silu_and_mul / gelu_and_mul).
///
/// ── Compute strategy (native AVX-512-FP16) ──────────────────────────
/// The inner loop multiplies and accumulates in NATIVE FP16 (`__m512h`,
/// 32 half lanes per zmm) via `_mm512_fmadd_ph`, for blocks of
/// `kF16AccumBlockK` K steps.  After each block the FP16 accumulators are
/// widened (`float16_t::cvt_f16_to_f32_vec` in `common/float16.hpp`) and
/// added into FP32 sums, and the bias, the activation math and the store
/// work from those sums, reusing the SAME FP32 primitives
/// (`group_matmul_act_avx512.hpp`) the bf16 sibling and the separate-pass
/// reference consume.
///
/// PRECISION NOTE — blocked FP16 accumulation:
///   FP16 has a 10-bit mantissa, so a running sum kept in FP16 over the
///   whole K loses the contributions that fall below its last bit as it
///   grows: over a long K the result degrades to noise.  Accumulating
///   each block in FP16 and the blocks in FP32 bounds that error by one
///   block, while keeping the 2x lane throughput of the native FP16
///   FMA.
///
/// Compile-time template parameters:
///   * MR ∈ {1..8}  — row count handled per call.
///   * NV ∈ {2, 4}  — FP32-equivalent N-lane groups (NR = NV × 16);
///                    realised as NV/2 native `__m512h` accumulators
///                    per row (NV=2 → 1 zmm of 32 cols, NV=4 → 2 zmms).
///   * Act          — none / swiglu_oai_mul / silu_and_mul / gelu_and_mul.
///   * DstDt        — kF16 or kF32 store (gated kinds are kF16-only).
///
/// Register-pressure caps (32 zmm available on AVX-512-FP16 CPUs):
///   FP16 accumulators pack 32 cols/zmm (vs the bf16 path's 16
///   FP32 cols/zmm), so the accumulator footprint is HALF the bf16
///   sibling's — every (MR, NV) tuple the bf16 selector instantiates
///   fits comfortably.  `max_mr_for_nv` / `kMaxMR` are reused
///   unchanged from `bf16_microkernel.hpp` so the dispatcher's
///   per-MR table sizing is shared across all three families.

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_F16_MICROKERNEL_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_F16_MICROKERNEL_HPP

#include "../pack.hpp"
#include "bf16_microkernel.hpp" // ActKind, BiasKind, DstDt, kMaxMR, max_mr_for_nv
#include "common/float16.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

using zendnnl::common::float16_t;

/// True when the running CPU supports native AVX-512-FP16
/// (VFMADD*PH etc.).  Cached after first call.  Sourced from the
/// shared platform-info ISA probe
/// (`platform_info_t::get_avx512_f16_status()`, CPUID leaf 7 / sub 0 /
/// EDX bit 23) — the SAME gate the embag / normalization AVX-512-FP16
/// kernels and the `group_matmul_direct` F16 entry check use, rather
/// than a private CPUID path.  When false the dispatcher must fall
/// back to the standard AOCL DLP / BRGEMM F16 path.
bool avx512f16_available();

/// Function-pointer type for one (MR, NV, Act, DstDt) F16 microkernel
/// specialization.  Mirrors the bf16-side `ukernel_fn_t` exactly: the
/// A panel and packed weight are FP16 (`float16_t`); whichever of
/// `Cout` / `Cout_tight` is unused is passed nullptr / 0.  `bias` may
/// be bf16, fp32, or f16 — the FP16 microkernel widens it to FP32 and
/// seeds its FP32 sums with it.  (`BiasKind::f16` is also accepted by the
/// bf16 / DQ-INT8 kernels, which widen it to fp32 via `_mm512_cvtph_ps`.)
/// `bias_kind` tells the kernel how to load it.  Pass `bias=nullptr` /
/// `bias_kind=BiasKind::none` when no bias is applied.
///
/// `Cout` / `Cout_tight` are typed `void *` because the destination
/// element width depends on the kernel's `DstDt` template parameter
/// (F16 = 2 bytes, FP32 = 4 bytes).  The dispatcher passes the same
/// caller-owned buffer cast to `void *`; the kernel reinterprets it
/// as `DstT *` internally.  `ldc` / `ldc_tight` stay in element units
/// (not bytes) — the kernel knows the element width via `DstT`.
using f16_ukernel_fn_t = void (*)(const float16_t *A, int lda,
        const float16_t *Bpacked, const void *bias, BiasKind bias_kind,
        void *Cout, int ldc, void *Cout_tight, int ldc_tight, int K);

/// Runtime selector — returns the function pointer for the requested
/// (MR ∈ 1..max_mr_for_nv(NV), NV ∈ {2, 4}, Act, DstDt) tuple, or
/// nullptr if the combination is not instantiated.
///
/// Mirrors `select_ukernel` on the bf16 side, including the
/// (gated_act, DstDt::kF32) refusal: the in-register pair-store
/// helpers write the half-width F16 output only, so any gated
/// activation with a non-F16 dst returns nullptr and the dispatcher's
/// `fill_kfn_table_f16` refuses the call cleanly.
///
/// On a toolchain WITHOUT AVX-512-FP16 intrinsics the microkernel
/// bodies are `#ifdef`-compiled out, so this returns nullptr for
/// EVERY (MR, NV, Act, DstDt) slot — not just the gated+f32-dst
/// combination above.  `fill_kfn_table_f16` then fails on the first
/// slot and `prepare_for_call` refuses, routing the call to AOCL DLP.
f16_ukernel_fn_t select_f16_ukernel(int MR, int NV, ActKind act, DstDt dst_dt);

/// K steps each FP16 accumulator covers before it is added into the FP32
/// sums (see the PRECISION NOTE above).
inline constexpr int kF16AccumBlockK = 512;

/// Shape of the single-expert FP16 prompt microkernel: up to 12 rows over
/// up to `kF16PromptNb` o-blocks (24 accumulators).
inline constexpr int kMaxF16PromptMr = 12;
inline constexpr int kF16PromptNb = 2;

/// Single-expert prompt microkernel: `MR` rows over `NB` adjacent NR=32
/// o-blocks, block `j` of B at `Bpacked + j * b_stride` and of the output
/// at column `j * 32` (`j * 16` in the tight output of a gated `act`).
/// With `load_c` (act = none only) the sums start from `Cout` rather than
/// the bias, continuing a K-chunked reduction: exactly for an FP32 `Cout`,
/// with one FP16 rounding per chunk for an F16 `Cout`.
using f16_prompt_ukernel_fn_t = void (*)(const float16_t *A, int lda,
        const float16_t *Bpacked, size_t b_stride, const void *bias,
        BiasKind bias_kind, void *Cout, int ldc, void *Cout_tight,
        int ldc_tight, int K, bool load_c);

/// MR ∈ 1..kMaxF16PromptMr, NB ∈ 1..kF16PromptNb; nullptr otherwise, for
/// a gated `act` with an FP32 dst, and on a toolchain without AVX-512-FP16.
f16_prompt_ukernel_fn_t select_f16_prompt_ukernel(
        int MR, int NB, ActKind act, DstDt dst_dt);

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_F16_MICROKERNEL_HPP
