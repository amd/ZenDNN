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

/// INT8 custom microkernel — per-tile (M × NR) GEMM with optional
/// gated-activation epilogue applied directly in accumulator
/// registers.  Sibling of the BF16 microkernel in
/// `bf16_microkernel.{hpp,cpp}` — both files follow the same layout,
/// templating discipline, and dispatch contract.  The only
/// fundamental difference is the inner-loop instruction
/// (`VPDPBUSD` vs `VDPBF16PS`) and the K-interleave stride (4 int8
/// elements per dot vs 2 bf16 elements per dot).
///
/// Engaged through `flat_n_tile` (ALGO 3) when the call dispatcher
/// resolves one of the symmetric/asymmetric S8-weight variants. The source is
/// either dynamically quantized with row-local metadata or statically
/// quantized U8 with scalar metadata. Weight scales are scalar or per-channel;
/// weight zero-points are rejected upstream.
///
/// One microkernel call computes the FP32 dequantised accumulator
///
///   C[0..MR, 0..NR] =
///       (A[0..MR, 0..K] @ B_packed[0..K, 0..NR]   // s32 GEMM
///        - K_m * sum_wei[0..NR])                  // per-col compensation
///       * src_scale[m or 0] * wei_scale[v or 0]   // row/scalar × col/scalar
///       + bias[v]                                 // optional
///
/// where `K_m` is `128` for symmetric (compute=s8) and `src_zp[m]`
/// for asymmetric (compute=u8).  The compensation `sum_wei[v]` is
/// precomputed at pack time and stored as an `int32_t` row appended
/// after each o-block of the packed weight (see pack.hpp).
///
/// The kernel then either
///   * stores BF16, F32, or requantized U8 output (Act = none), or
///   * applies the gated activation in registers and stores the
///     halved BF16 output to a tight destination (Act =
///     swiglu_oai_mul / silu_and_mul / gelu_and_mul).
///
/// Compile-time template parameters:
///   * MR ∈ {1..max_mr_for_nv_int8(NV, Compute)} — row count handled per call.
///   * NV ∈ {2, 4}      — accumulator zmms per row (NR = NV × 16).
///   * Compute ∈
///       {kS8_Sym, kU8_Asym}  — quant flavour (see `IntCompute` below).
///   * Act ∈
///       {none, swiglu_oai_mul, silu_and_mul, gelu_and_mul}.
///   * DstDt ∈ {kBf16, kF32, kU8}. F32/U8 are valid for Act=none only;
///     gated kinds are BF16-dst only (see `select_int8_ukernel`).
///
/// Inner-loop pattern follows the bf16 sibling:
///   * K-quad unroll to expose ILP between independent B-load batches.
///   * `VPDPBUSD` (AVX-512 VNNI) for one FMA per 4 K-elements.
///   * For `kS8_Sym`: src bytes are XOR-ed with `0x80808080` at
///     broadcast time so VPDPBUSD's signed × unsigned operand
///     ordering holds — the compensation row is precomputed at pack
///     time to undo the resulting `+128 × sum_wei` bias.
///
/// Register-pressure budget (32 zmm available on AVX-512-VNNI CPUs).  The K-loop footprint is
///
///   kBuffers*MR*NV + (kPipelineB ? 2*NV : 0) + MR + 1
///
/// i.e. accumulators, the two live B staging sets when B is pipelined,
/// one broadcast per row (the m-loop is fully unrolled, so all MR are
/// live at once), and the s8→u8 bias vector on `kS8_Sym`.  Both the
/// `MR` and the `2*NV` terms are load-bearing: an earlier version of
/// this note omitted them, and so declared NV=4 safe up to MR=6 when
/// the real footprint spills from MR=3 up.
///
/// VPDPBUSD has 4-5 cycle latency at 2/cycle
/// throughput, so a specialisation needs 8 independent accumulator
/// chains to be issue- rather than latency-bound.  `kBuffers` supplies
/// them by splitting the s32 accumulators into sets fed from
/// consecutive K-quads (summed with `_mm512_add_epi32` before the
/// compensation correction), sized to the largest value that still
/// fits the budget and capped at the 8 chains that saturate issue.
/// Whether B is staged into registers or left as a VPDPBUSD memory
/// operand follows from whether the plain form can reach 8 chains
/// within budget, i.e. `MR*NV <= 8`; see the register-budget block in
/// the .cpp for the per-specialisation measurements behind that
/// boundary.
///
/// Destination dtype is a template axis (`DstDt`, shared with the bf16
/// sibling): BF16, FP32, or U8 store for `Act = none`. Gated activations are
/// BF16-dst only.

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_INT8_MICROKERNEL_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_INT8_MICROKERNEL_HPP

#include <cstdint>
#include <limits>

#include "../pack.hpp"
#include "bf16_microkernel.hpp" // ActKind, BiasKind, kMaxMR, max_mr_for_nv
#include "common/bfloat16.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

/// Quant flavour the int8 microkernel handles.  Selected by
/// `resolve_variant()` on the (src, wei, dst, dynamic_quant,
/// compute) tuple and threaded into the kernel template parameter.
///
///   * `kS8_Sym`  — symmetric dynamic INT8. Src is `int8_t`,
///                  no `src_zp`.  The K-loop XORs every src broadcast
///                  with `0x80808080` so VPDPBUSD's `(unsigned ×
///                  signed)` operand ordering holds; the resulting
///                  `+128 × sum_wei[v]` bias is undone by the
///                  per-column compensation row stored at pack time.
///   * `kU8_Asym` — asymmetric INT8. Src is `uint8_t` with a per-row or
///                  scalar `src_zp` (int32). VPDPBUSD
///                  consumes the u8 src directly; the bias is
///                  `src_zp[m] × sum_wei[v]` (per row × per col),
///                  also undone in the epilogue from the
///                  compensation row.
enum class IntCompute : uint8_t {
    kS8_Sym = 0,
    kU8_Asym = 1,
};

/// Conservative bound that keeps both the raw `u8*s8` dot product and the
/// source-zero-point compensation in signed int32 for every legal byte value.
inline constexpr int kMaxSafeInt8AccumK
        = std::numeric_limits<int32_t>::max() / (255 * 128);

/// Dtype of the row/scalar `src_scale` and channel/scalar `wei_scale` the
/// microkernel consumes.  The kernel dequantises in f32, so f32 scales
/// load directly while bf16 scales are converted on load in the
/// epilogue (one cheap widening per (m,v) tile — the load is already
/// off the hot VPDPBUSD path, mirroring how `BiasKind` handles a bf16
/// bias).  A single `ScaleKind` covers BOTH scales: the dispatcher
/// requires `src_scale.dt == wei_scale.dt` (refusing the CK path to AOCL
/// otherwise), and the silu/gelu interleave pre-pass emits f32 for
/// both, so the two scales always share one dtype at the kernel.
enum class ScaleKind : uint8_t {
    kF32 = 0,
    kBf16 = 1,
};

enum class Int8UnaryPostOp : uint8_t {
    kNone = 0,
    kRelu,
    kSigmoid,
};

/// Per-call quantization details that are orthogonal to the selected
/// `(Compute, DstDt)` microkernel specialization.
///
/// Dynamic per-token INT8 uses per-row source and per-channel weight scales,
/// so both broadcast flags remain false. Static U8xS8 uses one source scale/zp
/// and either a scalar or per-channel weight scale; therefore
/// `src_quant_per_tensor` is true while `wei_scale_per_tensor` reflects the
/// weight metadata. `dst_scale` follows the existing matmul post-op convention:
/// it is a multiplier, not a dequantization scale. The u8 store computes
/// `saturate_u8(round_nearest_even(real_value * dst_scale + dst_zp))`. The
/// multiply remains a separate FP32 operation from the zero-point add.
///
/// Static U8xS8 may additionally fuse the exact post-op chain
/// `binary_mul(F32[M,N]) -> binary_add(F32[M,N])`. The dispatcher offsets
/// each pointer to the current MR/o-block before invoking the microkernel;
/// the retained leading dimension advances between local rows. A single
/// buffer-free ReLU or sigmoid may instead be applied before the final store.
struct Int8EpilogueConfig {
    bool src_quant_per_tensor = false;
    bool wei_scale_per_tensor = false;
    float dst_scale = 1.0f;
    int32_t dst_zp = 0;
    const float *binary_mul = nullptr;
    const float *binary_add = nullptr;
    int binary_mul_ld = 0;
    int binary_add_ld = 0;
    Int8UnaryPostOp unary_postop = Int8UnaryPostOp::kNone;
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

/// INT8-specific MR limit.  The generic BF16 bound remains 8 rows for
/// NV=2; the S8/NV=2 scalar INT8 family instantiates up to 12 rows.
/// Dispatchers should query this helper rather than assuming the BF16
/// row bound applies to every dtype/kernel family.
inline int max_mr_for_nv_int8(int NV, IntCompute compute) {
    return (NV == 2 && compute == IntCompute::kS8_Sym) ? 12 : max_mr_for_nv(NV);
}

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
///   * `src_scale` — per-row scale vector, or one scalar for static U8xS8.
///   * `src_zp` — per-row zero-point vector for `kU8_Asym`, or one scalar
///     for static U8xS8. `nullptr` for `kS8_Sym`.
///   * `wei_scale` — per-channel scale vector, or one scalar for static U8xS8.
///   * `epilogue` — selects scalar broadcasts and carries the destination
///     multiplier/zero-point used by the u8 store.
///   * `bias` — optional bias pointer (see `bias_kind` for dtype).
///   * `Cout` / `Cout_tight` — destination pointers in element
///     units (the kernel re-casts to `bfloat16_t *` or `float *`
///     internally per its `DstDt` template axis).  Exactly one of
///     the two is non-null per call (matches the bf16 pattern); the
///     unused one is passed `nullptr` with `ldc = 0`.  The gated-act
///     epilogue always writes BF16 to `Cout_tight`; `Act = none`
///     writes `Cout` as BF16 (DstDt::kBf16), FP32 (DstDt::kF32), or
///     requantized U8 (DstDt::kU8).
///   * `K` — K dimension of this call (in K-elements).  MUST be a
///     multiple of 4 for the INT8 CK path: the microkernel
///     broadcasts 4 source bytes per step (the VNNI K-quad) directly
///     from the UNPADDED hoisted src row, so a `K % 4 != 0` tail would
///     over-read that row.  The dispatcher enforces this and falls back
///     to AOCL DLP for unaligned K (see `prepare_for_call`'s
///     `int8_K_not_multiple_of_4` gate and `ck_eligible_int8` in
///     prepack.cpp).  The WEIGHT pack zero-pads its trailing K-quad
///     INDEPENDENTLY, so the packed-weight 4-byte load is always safe;
///     it is only the src side that requires the alignment.
using int8_ukernel_fn_t = void (*)(const uint8_t *A, int lda,
        const int8_t *Bpacked, const void *src_scale, const int32_t *src_zp,
        const void *wei_scale, ScaleKind scale_kind,
        const Int8EpilogueConfig &epilogue, const void *bias,
        BiasKind bias_kind, void *Cout, int ldc, void *Cout_tight,
        int ldc_tight, int K);

/// Accumulation-only symmetric INT8 kernel used by the deep-K blocked path.
/// It computes one NR=32 block into an s32 scratch tile without applying
/// compensation, scales, bias, or destination conversion.  `add=true`
/// accumulates onto an existing scratch tile; otherwise it overwrites it.
using int8_accum_ukernel_fn_t = void (*)(const uint8_t *A, int lda,
        const int8_t *Bpacked, int K, int32_t *C, int ldc, bool add);

/// Select the NR=32 symmetric accumulation-only specialization for `MR`.
/// Returns nullptr outside MR 1..12.
int8_accum_ukernel_fn_t select_int8_accum_ukernel(int MR);

/// Apply full-K compensation and dynamic per-token/per-channel dequantization
/// to one NR=32 s32 scratch block, then store BF16 output.
void finish_int8_accum_nv2_sym_bf16(const int32_t *C, int ldc_c, int M,
        const int32_t *comp, const void *src_scale, const void *wei_scale,
        ScaleKind scale_kind, bfloat16_t *dst, int ldc_dst);

/// Runtime selector — returns the function pointer for the
/// requested `(MR ∈ 1..max_mr_for_nv_int8(NV, Compute), NV ∈ {2, 4},
/// Compute, Act, DstDt)` tuple, or `nullptr` if the combination is not
/// instantiated.
///
/// `dst_dt` mirrors the bf16 sibling's `DstDt` axis and adds `kU8` for
/// static per-tensor requantization. F32/U8 destinations are valid only for
/// `Act = none`; gated activations remain BF16-only.
int8_ukernel_fn_t select_int8_ukernel(
        int MR, int NV, IntCompute compute, ActKind act, DstDt dst_dt);

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_INT8_MICROKERNEL_HPP
