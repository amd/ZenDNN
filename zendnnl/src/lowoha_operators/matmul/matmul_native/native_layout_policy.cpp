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

#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"

#include "lowoha_operators/matmul/lowoha_quant_layout.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_format.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

native_layout_decision_t declined(const char *reason, bool isa = false) {
    native_layout_decision_t d;
    d.reason = reason;
    d.isa = isa;
    return d;
}

/// Why the CPU of uarch cannot run kernels of isa besides native_base_isa;
/// nullptr when it can.
const char *isa_refusal(native_isa_t isa, const UarchParams &uarch) {
    // No default: an instruction set added to native_isa_t fails to compile
    // here until it is checked.
    switch (isa) {
        case native_isa_t::base: return nullptr;
        case native_isa_t::avx512_bf16:
            return uarch.avx512bf16 ? nullptr : "no AVX512-BF16";
        case native_isa_t::avx512_vnni:
            return uarch.avx512vnni ? nullptr : "no AVX512-VNNI";
        case native_isa_t::avx512_fp16:
            return uarch.avx512fp16 ? nullptr : "no AVX512-FP16";
    }
    return nullptr;
}

} // namespace

bool native_base_isa() {
    return native_hw().base_isa;
}

native_layout_decision_t decide_native_layout(const native_layout_query_t &q) {
    if (!native_base_isa())
        return declined("no AVX-512 F / BW / VL / DQ or FMA", true);
    const native_format_desc_t *f = native_format_of(q.wei_dt);
    if (f == nullptr)
        return declined("weight type other than bf16 / s8 / s4 / f32 / f16");
    const native_contract_t &c = f->contract;
    for (const native_refusal_t &r : c.src_refusals)
        if ((r.types & native_type_bit(q.src_dt)) != 0) return declined(r.why);
    if (const char *why = isa_refusal(c.isa, detect_uarch()))
        return declined(why, true);
    if (q.N < 1) return declined("N < 1");
    if (q.N > kMaxNativeN)
        return declined(
                "a weight of N whose whole blocks of 32 columns exceed int");
    if (q.K < 1) return declined("K < 1");
    if (q.K > c.k.max) return declined(c.k.why);
    if ((c.scale_grains & native_grain_bit(native_grain_t::group)) != 0) {
        // No per-group weight scales: one group over K.
        const int group = q.wei_group_size != 0 ? q.wei_group_size : q.K;
        const bool per_channel = group == q.K
                && (c.scale_grains & native_grain_bit(native_grain_t::channel))
                        != 0;
        if (!per_channel && !quant_group_size_is_whole_quads(q.K, group))
            return declined(c.group_refusal);
        if (group > c.group.max) return declined(c.group.why);
    }
    if (q.K > native_max_layout_k(*f)) return declined(c.layout_k_refusal);
    if (q.ldb < (q.transB ? q.K : q.N))
        return declined("ldb below the weight row length");

    native_layout_decision_t d;
    d.native = true;
    d.pack_nr = kNativeLayoutPackNr;
    return d;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
