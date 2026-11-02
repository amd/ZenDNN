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
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
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

} // namespace

bool native_base_isa() {
    return native_hw().base_isa;
}

native_layout_decision_t decide_native_layout(const native_layout_query_t &q) {
    const UarchParams &uarch = detect_uarch();
    const bool int8 = q.wei_dt == data_type_t::s8;
    if (!native_base_isa())
        return declined("no AVX-512 F / BW / VL / DQ or FMA", true);
    if (q.wei_dt == data_type_t::bf16) {
        if (q.src_dt != data_type_t::bf16)
            return declined("bf16 weight with a non-bf16 source");
        if (!uarch.avx512bf16) return declined("no AVX512-BF16", true);
    } else if (int8) {
        if (q.src_dt != data_type_t::u8 && q.src_dt != data_type_t::s8)
            return declined("s8 weight with a source other than u8 / s8");
        if (!uarch.avx512vnni) return declined("no AVX512-VNNI", true);
    } else if (q.wei_dt == data_type_t::f32) {
        if (q.src_dt != data_type_t::f32)
            return declined("f32 weight with a non-f32 source");
    } else {
        return declined("weight type other than bf16 / s8 / f32");
    }
    if (q.N < 1) return declined("N < 1");
    if (q.K < 1) return declined("K < 1");
    if (int8 && q.K > kMaxSafeInt8AccumK)
        return declined("K exceeds the int32 accumulation bound");
    if (int8 && q.wei_group_size != 0 && q.wei_group_size != q.K
            && !quant_group_size_is_whole_quads(q.K, q.wei_group_size))
        return declined(
                "per-group weight scales over groups of other than whole K "
                "quads");
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
