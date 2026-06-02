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

/// DQ-INT8 custom microkernel — per-tile (M × NR) GEMM with optional
/// gated-activation epilogue applied directly in accumulator
/// registers.  Sibling of the BF16 microkernel in
/// `bf16_microkernel.{hpp,cpp}` — both files follow the same layout,
/// templating discipline, and dispatch contract.  The only
/// fundamental difference is the inner-loop instruction
/// (`VPDPBUSD` vs `VDPBF16PS`) and the K-interleave stride (4 int8
/// elements per dot vs 2 bf16 elements per dot).
///
/// Engaged through `flat_n_tile` (ALGO 3) when the call dispatcher
/// resolves the kernel variant to `kS8_S8_BF16_SYM` (per-token
/// symmetric DQ-INT8) or `kU8_S8_BF16_ASYM` (per-token asymmetric
/// DQ-INT8).  The src buffer is the per-row reorder-quantised src
/// (s8 for sym, u8 for asym) produced by
/// `dynamic_per_token_quant_bf16_{s8,u8}_native` — the same kernels
/// the M-tile DQ-INT8 path consumes.  The weight buffer is per-
/// channel-quantised s8 with a per-channel f32 scale; a non-null
/// weight zero-point (wei_zp) is rejected upstream by
/// `check_n_tile_extra` (the N-tile DQ-INT8 path assumes symmetric
/// weights — folding a weight zp into the compensation row is a
/// possible future extension).
///
/// One microkernel call computes the FP32 dequantised accumulator
///
///   C[0..MR, 0..NR] =
///       (A[0..MR, 0..K] @ B_packed[0..K, 0..NR]   // s32 GEMM
///        - K_m * sum_wei[0..NR])                  // per-col compensation
///       * src_scale[m] * wei_scale[v]             // per-token × per-channel
///       + bias[v]                                 // optional
///
/// where `K_m` is `128` for symmetric (compute=s8) and `src_zp[m]`
/// for asymmetric (compute=u8).  The compensation `sum_wei[v]` is
/// precomputed at pack time and stored as an `int32_t` row appended
/// after each o-block of the packed weight (see pack.hpp).
///
/// The kernel then either
///   * stores raw BF16 output (Act = none), or
///   * applies the gated activation in registers and stores the
///     halved BF16 output to a tight destination (Act =
///     swiglu_oai_mul / silu_and_mul / gelu_and_mul).
///
/// Compile-time template parameters:
///   * MR ∈ {1..8}      — row count handled per call (max_mr depends on NV).
///   * NV ∈ {2, 4}      — accumulator zmms per row (NR = NV × 16).
///   * Compute ∈
///       {kS8_Sym, kU8_Asym}  — quant flavour (see `IntCompute` below).
///   * Act ∈
///       {none, swiglu_oai_mul, silu_and_mul, gelu_and_mul}.
///   * DstDt ∈
///       {kBf16, kF32}  — store dtype.  kF32 is valid for Act=none
///       only; gated kinds are BF16-dst only (see `select_int8_ukernel`).
///
/// Inner-loop pattern follows the bf16 sibling:
///   * K-quad unroll-by-2 to expose ILP between two B-load batches.
///   * `VPDPBUSD` (AVX-512 VNNI) for one FMA per 4 K-elements.
///   * For `kS8_Sym`: src bytes are XOR-ed with `0x80808080` at
///     broadcast time so VPDPBUSD's signed × unsigned operand
///     ordering holds — the compensation row is precomputed at pack
///     time to undo the resulting `+128 × sum_wei` bias.
///
/// Register-pressure caps (Zen4/5: 32 zmm):
///   * NV=2 (NR=32): max MR=8 — 16 s32 acc + 2 b + 1 a ≈ 19 zmms.
///   * NV=4 (NR=64): max MR=6 — 24 s32 acc + 4 b + 1 a ≈ 29 zmms.
/// Double-buffering on MR ≤ 3 (kBuffers = 2) IS enabled, mirroring the
/// bf16 sibling: VPDPBUSD has 5-cycle latency on Zen4 / 4-cycle on
/// Zen5 with 2/cycle throughput, so small-MR specialisations whose
/// independent accumulator chain count (MR × NV) is below the
/// latency × throughput product are FMA-latency bound.  Splitting the
/// s32 accumulators into two sets fed from even / odd K-quads (summed
/// with `_mm512_add_epi32` before the compensation correction)
/// doubles the chain count and brings MR=2, MR=3 from latency-bound to
/// issue-bound.  MR ≥ 4 keeps kBuffers = 1 (single buffering already
/// saturates issue; avoids spilling the MR=8 × NV=2 spec's 16 acc
/// zmms).
///
/// Destination dtype is a template axis (`DstDt`, shared with the bf16
/// sibling): BF16 store for the standard path, FP32 store (direct
/// `_mm512_storeu_ps` of the dequantised accumulator) for `Act = none`.
/// Gated activations are BF16-dst only.

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_INT8_MICROKERNEL_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_INT8_MICROKERNEL_HPP

#include <cstdint>

#include "common/bfloat16.hpp"
#include "../pack.hpp"
#include "bf16_microkernel.hpp"  // ActKind, BiasKind, kMaxMR, max_mr_for_nv

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

/// Quant flavour the int8 microkernel handles.  Selected by
/// `resolve_variant()` on the (src, wei, dst, dynamic_quant,
/// compute) tuple and threaded into the kernel template parameter.
///
///   * `kS8_Sym`  — per-token symmetric DQ-INT8.  Src is `int8_t`,
///                  no `src_zp`.  The K-loop XORs every src broadcast
///                  with `0x80808080` so VPDPBUSD's `(unsigned ×
///                  signed)` operand ordering holds; the resulting
///                  `+128 × sum_wei[v]` bias is undone by the
///                  per-column compensation row stored at pack time.
///   * `kU8_Asym` — per-token asymmetric DQ-INT8.  Src is `uint8_t`
///                  with a per-row `src_zp` (int32).  VPDPBUSD
///                  consumes the u8 src directly; the bias is
///                  `src_zp[m] × sum_wei[v]` (per row × per col),
///                  also undone in the epilogue from the
///                  compensation row.
enum class IntCompute : uint8_t {
  kS8_Sym = 0,
  kU8_Asym = 1,
};

/// Dtype of the per-row `src_scale` and per-channel `wei_scale` the
/// microkernel consumes.  The kernel dequantises in f32, so f32 scales
/// load directly while bf16 scales are converted on load in the
/// epilogue (one cheap widening per (m,v) tile — the load is already
/// off the hot VPDPBUSD path, mirroring how `BiasKind` handles a bf16
/// bias).  A single `ScaleKind` covers BOTH scales: the dispatcher
/// requires `src_scale.dt == wei_scale.dt` (refusing the CK path to
/// AOCL otherwise), and the silu/gelu interleave pre-pass emits f32 for
/// both, so the two scales always share one dtype at the kernel.
enum class ScaleKind : uint8_t {
  kF32  = 0,
  kBf16 = 1,
};

// `kVNNIInt8Quad` (= 4) is declared once in pack.hpp alongside
// the bf16-side `kVNNIPair` so the two pack families share a
// single source of truth for the VNNI multiplicity constants.
// `pack.hpp` is included above (through "../pack.hpp"); both the
// microkernel and the pack consumer see the same constant value.

/// True when the running CPU supports AVX-512 VNNI (VPDPBUSD).
/// Cached after first call.  When false the dispatcher must fall
/// back to AOCL DLP `aocl_gemm_s8s8s32obf16_sym_quant`.
bool avx512vnni_available();

/// Function-pointer type for one (MR, NV, Compute, Act) int8
/// microkernel specialization.  Whichever of `Cout` / `Cout_tight`
/// is unused is passed nullptr / 0 — see the dispatcher for
/// argument routing.  `bias` / `bias_kind` follow the bf16-side
/// contract (none / bf16 / fp32).
///
/// Arguments:
///   * `A` — per-row int byte stream of MR rows of K K-elements at
///     `lda` element-stride.  For `Compute = kS8_Sym` the caller
///     passes the s8 buffer cast to `uint8_t *` (the kernel
///     internally XORs with `0x80808080` per K-broadcast).  For
///     `Compute = kU8_Asym` the caller passes the u8 buffer
///     directly.
///   * `Bpacked` — pointer to this o-block's packed weight slab
///     (laid out `[O/pack_nr, K_pad/4, pack_nr, 4]` for the weight
///     bytes followed by `[pack_nr] int32_t` compensation row; see
///     pack.hpp for the full layout).  The kernel reads
///     `K_quad * pack_nr * 4` bytes of weight then reads `pack_nr`
///     int32 lanes of compensation.
///   * `src_scale` — per-row scale vector, `MR` floats.  The
///     dispatcher slices the expert's full src_scale to this MR
///     window before the call.
///   * `src_zp` — per-row zero-point vector for `kU8_Asym`,
///     `MR` int32s.  `nullptr` for `kS8_Sym`.
///   * `wei_scale` — per-channel scale vector, `NR` floats (one
///     per output column).  The dispatcher slices the expert's
///     full wei_scale to this o-block window before the call.
///   * `bias` — optional bias pointer (see `bias_kind` for dtype).
///   * `Cout` / `Cout_tight` — destination pointers in element
///     units (the kernel re-casts to `bfloat16_t *` or `float *`
///     internally per its `DstDt` template axis).  Exactly one of
///     the two is non-null per call (matches the bf16 pattern); the
///     unused one is passed `nullptr` with `ldc = 0`.  The gated-act
///     epilogue always writes BF16 to `Cout_tight`; `Act = none`
///     writes `Cout` as BF16 (DstDt::kBf16) or FP32 (DstDt::kF32).
///   * `K` — K dimension of this call (in K-elements).  MUST be a
///     multiple of 4 for the DQ-INT8 CK path: the microkernel
///     broadcasts 4 source bytes per step (the VNNI K-quad) directly
///     from the UNPADDED hoisted src row, so a `K % 4 != 0` tail would
///     over-read that row.  The dispatcher enforces this and falls back
///     to AOCL DLP for unaligned K (see `prepare_for_call`'s
///     `int8_K_not_multiple_of_4` gate and `ck_eligible_int8` in
///     prepack.cpp).  The WEIGHT pack zero-pads its trailing K-quad
///     INDEPENDENTLY, so the packed-weight 4-byte load is always safe;
///     it is only the src side that requires the alignment.
using int8_ukernel_fn_t = void (*)(
    const uint8_t      *A, int lda,
    const int8_t       *Bpacked,
    const void         *src_scale,
    const int32_t      *src_zp,
    const void         *wei_scale,
    ScaleKind           scale_kind,
    const void         *bias, BiasKind bias_kind,
    void               *Cout, int ldc,
    void               *Cout_tight, int ldc_tight,
    int                 K);

/// Runtime selector — returns the function pointer for the
/// requested `(MR ∈ 1..max_mr_for_nv(NV), NV ∈ {2, 4}, Compute,
/// Act, DstDt)` tuple, or `nullptr` if the combination is not
/// instantiated.
///
/// `dst_dt` mirrors the bf16 sibling's `DstDt` axis: `kBf16` for the
/// standard half/full-width BF16 store, `kF32` for the direct FP32
/// store of the dequantised accumulator.  FP32 dst is only valid for
/// `Act = none` — gated activations are BF16-dst only and the
/// selector returns `nullptr` for any (gated, kF32) tuple, mirroring
/// `select_ukernel` on the bf16 side.
int8_ukernel_fn_t select_int8_ukernel(int MR, int NV,
                                      IntCompute compute,
                                      ActKind    act,
                                      DstDt      dst_dt);

}  // namespace custom_kernel
}  // namespace matmul
}  // namespace lowoha
}  // namespace zendnnl

#endif  // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_INT8_MICROKERNEL_HPP
