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

/// The per-weight custom-kernel family classifier shared by the single- and
/// multi-expert policies.

#include "ck_enable.hpp"

#include "pack.hpp"
#include "ukernel/f16_microkernel.hpp"
#include "ukernel/int8_microkernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

custom_kernel::KernelVariant grp_matmul_ck_weight_variant(data_type_t src,
        data_type_t wei, data_type_t dst, bool dynamic_quant,
        data_type_t compute, bool static_u8s8, bool aocl_prepacked, int K,
        int N) {
    using custom_kernel::KernelVariant;
    constexpr KernelVariant kNone = KernelVariant::kUnsupported;
    const KernelVariant v = custom_kernel::resolve_variant(
            src, wei, dst, dynamic_quant, compute);
    const bool bf16 = v == KernelVariant::kBF16_BF16_BF16
            || v == KernelVariant::kBF16_BF16_F32;
    const bool int8 = custom_kernel::is_int8_variant(v);
    const bool s4 = custom_kernel::is_w4a8_variant(v);
    const bool f16 = custom_kernel::is_f16_variant(v);
    if (bf16) {
        if (!custom_kernel::bf16_isa_supported()) return kNone;
    } else if (int8) {
        if (!get_grp_matmul_custom_kernel_int8()
                || !custom_kernel::avx512vnni_available())
            return kNone;
    } else if (s4) {
        if (!custom_kernel::avx512vnni_available()) return kNone;
    } else if (f16) {
        if (!get_grp_matmul_custom_kernel_f16()
                || !custom_kernel::avx512f16_available())
            return kNone;
    } else {
        return kNone;
    }
    if (K > 0 && N > 0 && custom_kernel::plan_pack_nr(K, N) <= 0) return kNone;
    // Shapes `prepare_for_call` always refuses.
    if (K > 0 && int8
            && (K % custom_kernel::kVNNIInt8Quad != 0
                    || K > custom_kernel::kMaxSafeInt8AccumK))
        return kNone;
    if (K > 0 && s4 && K % custom_kernel::kS4Octet != 0) return kNone;
    // AOCL-prepacked dynamic asymmetric INT8 has no CK inverse route. Keep it
    // on the scalar/full-N AOCL path instead of entering grouped preparation
    // only to be refused there.
    if (aocl_prepacked && int8 && !static_u8s8 && compute == data_type_t::u8) {
        return kNone;
    }
    // Every currently supported CK weight family has an AOCL inverse.
    (void)aocl_prepacked;
    (void)static_u8s8;
    return v;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
