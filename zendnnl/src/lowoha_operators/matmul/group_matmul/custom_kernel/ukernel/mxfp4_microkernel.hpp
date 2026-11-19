/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// MXFP4 custom microkernel: f4_e2m1 weights with f8_e8m0 block scales ×
/// s8 source with per-token or per-block scales.  Sibling of
/// `s4_microkernel`, engaged from `n_tile_execute` for the MXFP4 variants
/// (`kS8_MXFP4_BF16_SYM`, `kS8_MXFP4_F32_SYM`); it reads the pack
/// `prepack_weight_into_w4` builds for `kW4FormatMxfp4`.
///
/// Dequant math, with v the E2M1 value and e the block's E8M0 byte:
///
///   u[k][v]    = 2·v[k][v] + 12 ∈ [0, 24]          (`kE2M1ToBiasedU8`)
///   dot[m,g,v] = −12 · Σ_{k∈g} A_s8[m,k] + Σ_{k∈g} u[k][v] · A_s8[m,k]
///   facc[m][v] = Σ_g dot[m,g,v] · 2^(e[g][v] − 128)
///   C[m][v]    = facc[m][v] · src_scale[m] + bias[v]
///
/// Per-block source scales (`{M, K/32}`, one per E8M0 block) move the source
/// factor inside the block sum:
///
///   C[m][v]    = Σ_g dot[m,g,v] · src_scale[m,g] · 2^(e[g][v] − 128) + bias[v]
///
/// Twice an E2M1 value is an integer, so the dot product is exact in int32
/// and the halved scale undoes the doubling.  VPDPBUSD multiplies an
/// unsigned byte by a signed one: the decoded weight is the unsigned
/// operand, offset by 12, and the −12·Σ seed removes the offset.
///
/// Dispatcher-enforced shape contract, as for the s4 kernel: `group_size %
/// 8 == 0` and `K % group_size == 0`, so every group is a whole number of
/// K-octets; `N % pack_nr == 0` with `pack_nr ∈ {32, 64}`, `NV = pack_nr/16`.

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_MXFP4_MICROKERNEL_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_MXFP4_MICROKERNEL_HPP

#include <cstdint>

#include "../pack.hpp"
#include "bf16_microkernel.hpp" // ActivationKind, BiasKind, kMaxMR
#include "int8_microkernel.hpp" // ScaleKind

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

/// Function-pointer type for one (MR, NV, Act) MXFP4 specialization.
///
///   * `Bpacked` — `[K/8][pack_nr][4]` code bytes followed by `[G][pack_nr]`
///     scale bytes (e − 1) and the o-block's shift byte
///     (`w4_tail_bytes_per_oblock` in `pack.hpp`).
///   * `src_scale` — f32 or bf16 as `src_scale_kind` says: one per row, or
///     for a per-block specialization `[MR][K / group_size]`, one per row
///     and block.
///   * `Cout` / `Cout_tight` — exactly one is non-null, the other nullptr
///     with ld = 0.  Gated acts write the halved bf16 tile to `Cout_tight`;
///     `Cout` holds `output`, bf16 or (act = none only) f32.
///   * `src_group_sums` — `s8_group_sums` of the MR source rows,
///     `[MR][K / group_size]`.
using mxfp4_ukernel_fn_t = void (*)(const uint8_t *A, int lda,
        const int8_t *Bpacked, const void *src_scale, ScaleKind src_scale_kind,
        const void *bias, BiasKind bias_kind, void *Cout, OutputType output,
        int ldc, void *Cout_tight, int ldc_tight, int K, int group_size,
        const int32_t *src_group_sums);

/// Maximum MR under the 32-zmm budget: the K-loop holds `2*MR*NV`
/// accumulators (f32 and s32 per group) plus 8 working registers.
inline int max_mr_for_nv_mxfp4(int NV) {
    return NV == 4 ? 3 : 6;
}

/// Returns the specialization for `(MR ∈ 1..max_mr_for_nv_mxfp4(NV),
/// NV ∈ {2, 4}, Act)`, with an s8 source and one source scale per row, or
/// per row and block with `per_block_src_scale`; nullptr when not
/// instantiated.  For act = none each one takes both outputs.
mxfp4_ukernel_fn_t select_mxfp4_ukernel(
        int MR, int NV, ActivationKind act, bool per_block_src_scale = false);

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_MXFP4_MICROKERNEL_HPP
