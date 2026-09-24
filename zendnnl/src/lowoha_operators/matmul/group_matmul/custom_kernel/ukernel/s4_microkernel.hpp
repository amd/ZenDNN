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
/// `flat_n_tile` for the `kS8_S4_BF16_SYM` variant.
///
/// Dequant math:
///
///   facc[m][v] = Σ_g ( Σ_{k∈g} A_s8[m,k]·W_s4[k,v]
///                      − 128 · comp[g][v] ) · wei_scale[g][v]
///   C[m][v]    = facc[m][v] · src_scale[m] + bias[v]
///
/// The `128 · comp[g][v]` term undoes the `XOR 0x80808080` applied to
/// each source broadcast, which is what makes VPDPBUSD's
/// `unsigned × signed` ordering valid for an s8 source.  `comp` is
/// summed over sign-recovered weights at pack time (see `pack.hpp`).
///
/// Dispatcher-enforced shape contract: `group_size % 8 == 0` and
/// `K % group_size == 0`, so every group is a whole number of K-octets
/// and the kernel has no tail handling; `N % pack_nr == 0` with
/// `pack_nr ∈ {32, 64}`, `NV = pack_nr/16`.

#ifndef ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_S4_MICROKERNEL_HPP
#define ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_S4_MICROKERNEL_HPP

#include <cstdint>

#include "../pack.hpp"
#include "bf16_microkernel.hpp" // ActKind, BiasKind, kMaxMR
#include "common/bfloat16.hpp"
#include "int8_microkernel.hpp" // ScaleKind, avx512vnni_available

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

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
using s4_ukernel_fn_t = void (*)(const uint8_t *A, int lda,
        const int8_t *Bpacked, const void *src_scale, const void *wei_scale,
        ScaleKind scale_kind, const void *bias, BiasKind bias_kind, void *Cout,
        int ldc, void *Cout_tight, int ldc_tight, int K, int group_size,
        int wei_scale_grp_stride);

/// Maximum MR under the 32-zmm budget: the K-loop holds `2*MR*NV`
/// accumulators (f32 and s32 per group) plus 8 working registers.
inline int max_mr_for_nv_s4(int NV) {
    return NV == 4 ? 3 : 6;
}

/// Returns the specialization for `(MR ∈ 1..max_mr_for_nv_s4(NV),
/// NV ∈ {2, 4}, Act)`, or nullptr when not instantiated.
s4_ukernel_fn_t select_s4_ukernel(int MR, int NV, ActKind act);

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CUSTOM_KERNEL_UKERNEL_S4_MICROKERNEL_HPP
