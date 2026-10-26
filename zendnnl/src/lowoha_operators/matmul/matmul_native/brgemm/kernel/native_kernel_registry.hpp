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
#include "lowoha_operators/matmul/matmul_native/common/native_int8_scheme.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

enum class native_kernel_kind_t : uint8_t { bf16, int8 };

/// What the micro-kernels of one dtype offer at one tile width.
struct native_kernel_caps_t {
    native_kernel_kind_t kind = native_kernel_kind_t::bf16;
    int nr = 0; ///< 16 * nv
    int max_mr = 0; ///< every row count 1 .. max_mr has a kernel
    /// INT8: the weight-scale schemes the kernel serves, a mask of
    /// native_int8_scheme_bit.  Every INT8 kernel serves every scheme;
    /// validate_native_plan checks the call's scheme against the mask, so a
    /// kernel that serves fewer schemes is never planned for the others.
    uint8_t int8_schemes = 0;

    bool serves(native_int8_scheme_t scheme) const noexcept {
        return (int8_schemes & native_int8_scheme_bit(scheme)) != 0;
    }
};

class native_kernel_registry_t {
public:
    /// nullptr when kind has no kernel at nr on this CPU.
    const native_kernel_caps_t *caps(
            native_kernel_kind_t kind, int nr) const noexcept;
    /// A kernel for every row count 1 .. mr at nr.
    bool covers(native_kernel_kind_t kind, int mr, int nr) const noexcept;
    /// An INT8 kernel that serves scheme, for every row count 1 .. mr at nr.
    bool covers_int8(
            int mr, int nr, native_int8_scheme_t scheme) const noexcept;
    /// BF16 kernels by row count (index 0 unused, size 13); nullptr when
    /// caps(bf16, nr) is nullptr.
    const bf16_brgemm_fn_t *bf16_table(int nr) const noexcept;

private:
    friend const native_kernel_registry_t &native_kernel_registry();
    native_kernel_registry_t(bool bf16_isa, bool int8_isa);

    static constexpr int kMaxRows = 12;
    static constexpr int kWidths = 2; ///< nr 16, 32

    std::vector<native_kernel_caps_t> all_;
    bf16_brgemm_fn_t bf16_[kWidths][kMaxRows + 1] = {};
};

/// Built once by probing select_bf16_brgemm_tall_kernel (mr 1 .. 12 at nr
/// 16 / 32) and select_native_int8_ukernel (mr 1 .. 12 at nv 1 / 2).
/// A dtype is probed only when the CPU has its ISA; otherwise it has no
/// entries.
const native_kernel_registry_t &native_kernel_registry();

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_KERNEL_NATIVE_KERNEL_REGISTRY_HPP
