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

/// W4A8 custom microkernel — per-group s4 weights × per-token s8 source.
/// Sibling of `bf16_microkernel` / `int8_microkernel`, engaged from
/// `n_tile_execute` for the `kS8_S4_BF16_SYM` variant.
///
/// Per-token dequant math:
///
///   comp[g][v] = −128 · Σ_{k∈g} W_s4[k,v]
///   facc[m][v] = Σ_g ( comp[g][v]
///                      + Σ_{k∈g} A_u8[m,k]·W_s4[k,v] )
///                    · wei_scale[g][v]
///   C[m][v]    = facc[m][v] · src_scale[m] + bias[v]
///
/// Per-group source scales move the source factor inside the group sum:
///
///   C[m][v] = Σ_g dot[m,g,v] · src_scale[m,g] · wei_scale[g,v] + bias[v]
///
/// The pre-scaled `comp[g][v]` term removes the +128 activation zero point.
/// Grouped MatMul supplies signed bytes and the kernel applies
/// `XOR 0x80808080`; routed MoE can supply the already-biased U8 bytes
/// directly.  Both modes therefore consume the same packed compensation.
///
/// Dispatcher-enforced shape contract: `group_size % 8 == 0` and
/// `K % group_size == 0`, so every group is a whole number of K-octets
/// and the kernel has no tail handling; `N % pack_nr == 0` with
/// `pack_nr ∈ {32, 64}`, `NV = pack_nr/16`.

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_S4_MICROKERNEL_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_S4_MICROKERNEL_HPP

#include <cstdint>

#include "../pack.hpp"
#include "bf16_microkernel.hpp" // ActivationKind, BiasKind, kMaxMR
#include "common/bfloat16.hpp"
#include "int8_microkernel.hpp" // ScaleKind, avx512vnni_available

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

/// Encoding of the activation bytes consumed by the S4 dot-product loop.
enum class S4SourceKind : uint8_t {
    /// Signed symmetric quants; recenter each broadcast with XOR 0x80.
    kS8 = 0,
    /// Unsigned bytes already equal to signed_quant + 128.
    kU8Zp128 = 1
};

/// Activation-scale layout. The existing per-token specialization remains the
/// default; per-group uses the same G as the weight scale `{G, N}`.
enum class S4SrcScaleGranularity : uint8_t { kPerToken = 0, kPerGroup = 1 };

/// Storage selected for the non-gated epilogue.
enum class S4OutputKind : uint8_t { kBf16 = 0, kF32 = 1 };

/// Function-pointer type for one (MR, NV, Act) s4 specialization.
///
///   * `Bpacked` — `[K/8][pack_nr][4]` biased-nibble bytes followed by
///     `[G][pack_nr]` int32 compensation rows (see `pack.hpp`).
///   * `wei_scale` — read as `[g * wei_scale_grp_stride + v*16 + lane]`.
///   * `Cout` / `Cout_tight` — exactly one is non-null, the other
///     nullptr with ld = 0.  Gated acts write the halved tile to
///     `Cout_tight`.
///   * `wei_scale_grp_stride` — the CALLER'S FULL N, not the tile
///     width: an N-tile slices columns out of a `{G, N}` scale buffer
///     without repacking it.
///   * `act_group_sum` — optional `{MR, G}` signed activation sums,
///     index `m * G + g`. Non-null only with a Q4_K offset. The caller
///     computes them once and every output-column tile reuses them.
using w4a8_ukernel_fn_t = void (*)(const uint8_t *A, int lda,
        const int8_t *Bpacked, const void *src_scale, const void *wei_scale,
        ScaleKind scale_kind, const void *bias, BiasKind bias_kind, void *Cout,
        int ldc, void *Cout_tight, int ldc_tight, int K, int group_size,
        int wei_scale_grp_stride, const void *wei_zp,
        const float *act_group_sum);

/// Maximum MR under the 32-zmm budget: the K-loop holds `2*MR*NV`
/// accumulators (f32 and s32 per group) plus 8 working registers.
inline int max_mr_for_nv_s4(int NV) {
    return NV == 4 ? 3 : 6;
}

/// Returns the specialization for `(MR ∈ 1..max_mr_for_nv_s4(NV),
/// NV ∈ {2, 4}, Act, source, output)`, or nullptr when not instantiated.
/// `has_offset` selects the Q4_K epilogue. The false specialization is the
/// Q4_0 / DA8W4 kernel and does not read `wei_zp`.
/// Gated activations intentionally support BF16 output only.
w4a8_ukernel_fn_t select_w4a8_ukernel(int MR, int NV, ActivationKind act,
        S4SourceKind source = S4SourceKind::kS8,
        S4OutputKind output = S4OutputKind::kBf16,
        S4SrcScaleGranularity src_scale_granularity
        = S4SrcScaleGranularity::kPerToken,
        bool has_offset = false);

/// Writes the weights of `K` rows (a multiple of 8) of one `pack_nr == 32`
/// block (`[K/8][32][4]` biased nibbles) to `dst` as signed bytes in the
/// DQ-INT8 VNNI layout `[K/4][32][4]`, the B operand of
/// `select_int8_accum_ukernel` kernels.  Both pointers are 64-byte aligned.
void unpack_s4_block_nr32(const int8_t *Bpacked, int K, int8_t *dst);

/// The `select_w4a8_ukernel` kernels at NV=2, split so one unpacked chunk of
/// groups serves every MR call of a tile: `groups` folds groups
/// `[g0, g0 + ng)` of `MR` rows (1..12) into the f32 sums `facc` (`[MR][32]`,
/// 64-byte aligned, zeroed before group 0), each group summed by `accum` (the
/// `select_int8_accum_ukernel` kernel for `MR`) into `Cs32` (`[MR][32]`,
/// 64-byte aligned) from `Bunpacked`, the chunk unpacked from group `g0`;
/// `store` then writes `facc` out.  `A`, `comp` (the block's `[G][32]`
/// compensation rows), the scales and the outputs index from row 0, group 0
/// and column 0 of the block as in those kernels, whose bits both match.
using w4a8_unpacked_groups_fn_t = void (*)(int MR, const uint8_t *A, int lda,
        const int8_t *Bunpacked, const int32_t *comp,
        int8_accum_ukernel_fn_t accum, const void *src_scale,
        const void *wei_scale, ScaleKind scale_kind, int g0, int ng, int G,
        int group_size, int wei_scale_grp_stride, float *facc, int32_t *Cs32);
using w4a8_unpacked_store_fn_t
        = void (*)(int MR, const float *facc, const void *src_scale,
                ScaleKind scale_kind, const void *bias, BiasKind bias_kind,
                void *Cout, int ldc, void *Cout_tight, int ldc_tight);

w4a8_unpacked_groups_fn_t select_w4a8_unpacked_groups(
        S4SrcScaleGranularity src_scale_granularity);
/// nullptr where `select_w4a8_ukernel` has no kernel for the tuple.
w4a8_unpacked_store_fn_t select_w4a8_unpacked_store(ActivationKind act,
        S4OutputKind output, S4SrcScaleGranularity src_scale_granularity);

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_S4_MICROKERNEL_HPP
