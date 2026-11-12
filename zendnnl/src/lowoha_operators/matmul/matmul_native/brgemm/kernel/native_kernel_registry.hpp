/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef MATMUL_NATIVE_BRGEMM_KERNEL_NATIVE_KERNEL_REGISTRY_HPP
#define MATMUL_NATIVE_BRGEMM_KERNEL_NATIVE_KERNEL_REGISTRY_HPP

#include <cstdint>
#include <vector>

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_brgemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/f16/native_f16_ph_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/f16/native_f16_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/fp32/native_fp32_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_int8_scheme.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// int4: the group kernels of an s8 source and an s4 weight; f16: the
/// kernels of an f16 weight over a source widened to f32; f16_ph: the
/// kernels of an f16 weight and an f16 source with AVX512-FP16
/// multiply-adds, which accumulate in f16 over blocks of
/// kNativeF16AccumBlockK steps and add the blocks in f32.
enum class native_kernel_kind_t : uint8_t {
    bf16,
    int8,
    f32,
    int4,
    f16,
    f16_ph
};

/// Columns of a slab: one 2-vector BF16 kernel call of a wider tile.
inline constexpr int kNativeBf16SlabCols = 32;

/// Widest BF16 tile: slabs of kNativeBf16SlabCols, at most this many columns.
inline constexpr int kNativeBf16MaxTileCols = 128;
static_assert(kNativeBf16MaxTileCols % kNativeBf16SlabCols == 0,
        "the widest BF16 tile is whole slabs");

/// Columns of one 2-vector BF16 kernel call in a tile nr wide: the tile, or
/// a slab for a tile of several slabs.
constexpr int native_bf16_slab_cols(int nr) {
    return nr < kNativeBf16SlabCols ? nr : kNativeBf16SlabCols;
}

/// What the micro-kernels of one dtype offer at one tile width.
struct native_kernel_caps_t {
    native_kernel_kind_t kind = native_kernel_kind_t::bf16;
    int nr = 0; ///< 16 * nv
    int max_mr = 0; ///< every row count 1 .. max_mr has a kernel
    /// INT8: the weight-scale schemes the kernel serves, a mask of
    /// native_int8_scheme_bit: the per-channel kernels serve per_channel, the
    /// group kernels per_group (as do the INT4 kernels).
    /// validate_native_plan checks the call's scheme against the mask, so a
    /// kernel is never planned for the other.
    uint8_t int8_schemes = 0;

    bool serves(native_int8_scheme_t scheme) const noexcept {
        return (int8_schemes & native_int8_scheme_bit(scheme)) != 0;
    }
};

class native_kernel_registry_t {
public:
    /// nullptr when kind has no kernel at nr on this CPU; for INT8, the
    /// per-channel kernels.
    const native_kernel_caps_t *caps(
            native_kernel_kind_t kind, int nr) const noexcept;
    /// The INT8 kernels that serve scheme at nr; nullptr when none.
    const native_kernel_caps_t *int8_caps(
            int nr, native_int8_scheme_t scheme) const noexcept;
    /// A kernel for every row count 1 .. mr at nr.
    bool covers(native_kernel_kind_t kind, int mr, int nr) const noexcept;
    /// An INT8 kernel that serves scheme, for every row count 1 .. mr at nr.
    bool covers_int8(
            int mr, int nr, native_int8_scheme_t scheme) const noexcept;
    /// BF16 kernels by row count (index 0 unused, size 13) of the calls of a
    /// tile nr wide, native_bf16_slab_cols(nr) columns each; nullptr when
    /// caps(bf16, nr) is nullptr.
    const bf16_brgemm_fn_t *bf16_table(int nr) const noexcept;
    /// FP32 kernels by row count (index 0 unused, size 13) of a tile nr wide
    /// (16 or 32); nullptr when caps(f32, nr) is nullptr.
    const native_fp32_ukernel_fn_t *f32_table(int nr) const noexcept;
    /// F16 kernels by row count (index 0 unused, size 13) of a tile nr wide
    /// (16 or 32); nullptr when caps(f16, nr) is nullptr.
    const native_f16_ukernel_fn_t *f16_table(int nr) const noexcept;
    /// The AVX512-FP16 kernels as f16_table, for caps(f16_ph, nr), nr 16, 32
    /// or 64.
    const native_f16_ph_ukernel_fn_t *f16_ph_table(int nr) const noexcept;
    /// The kernels of an f16 call: f16_ph when the registry has them, else
    /// f16.
    native_kernel_kind_t f16_kind() const noexcept;

private:
    friend const native_kernel_registry_t &native_kernel_registry();
    native_kernel_registry_t(
            bool bf16_isa, bool int8_isa, bool f32_isa, bool f16_ph_isa);

    static constexpr int kMaxRows = 12;
    static constexpr int kWidths = 2; ///< nr 16, 32
    static constexpr int kF16PhWidths = 3; ///< nr 16, 32, 64

    std::vector<native_kernel_caps_t> all_;
    bf16_brgemm_fn_t bf16_[kWidths][kMaxRows + 1] = {};
    native_fp32_ukernel_fn_t f32_[kWidths][kMaxRows + 1] = {};
    native_f16_ukernel_fn_t f16_[kWidths][kMaxRows + 1] = {};
    native_f16_ph_ukernel_fn_t f16_ph_[kF16PhWidths][kMaxRows + 1] = {};
};

/// Built once by probing select_bf16_brgemm_tall_kernel (mr 1 .. 12 at nr
/// 16 / 32), select_native_int8_ukernel, select_native_int8_group_ukernel,
/// select_native_int4_group_ukernel (both orders),
/// select_native_fp32_ukernel and select_native_f16_ukernel (mr 1 .. 12 at
/// nv 1 / 2), and select_native_f16_ph_ukernel (mr 1 .. 12 at nv 1 / 2 / 4).
/// BF16 tiles of 64 and 128 columns take the row counts of nr 32.  A dtype is probed only
/// when the CPU has its ISA (AVX512-BF16, AVX512-VNNI for INT8 and INT4,
/// AVX-512 F for FP32 and F16, AVX512-FP16 for F16_PH); otherwise it has no
/// entries.  F16_PH is probed only while ZENDNNL_MATMUL_NATIVE_F16_ACCUM,
/// read here, is unset or f16; at f32 an f16 call takes the F16 kernels,
/// and another value keeps the default with one warning.
const native_kernel_registry_t &native_kernel_registry();

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_KERNEL_NATIVE_KERNEL_REGISTRY_HPP
