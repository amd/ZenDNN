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

/// W4A8: symmetric s4 weights, with one scale per (group, column), against
/// s8 rows the call quantizes from BF16, one scale per row or per (row,
/// group).  Packs of K octets ([K/8][32][4], each nibble XOR-ed with 8) with
/// -128 times each column's sum per group; kernels that unpack each octet
/// into two VPDPBUSD quads and store BF16.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_FORMATS_S4_S4_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_FORMATS_S4_S4_HPP

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/format.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {
namespace s4 {

void pack(const void *weight, const weight_desc &d, const pack_out &out);
ukernel_fn narrow(
        int mr, epilogue ep, bool src_u8, bool unaligned, scale_groups groups);
ukernel_fn wide(
        int mr, epilogue ep, bool src_u8, bool two_chains, scale_groups groups);

inline constexpr format_desc kFormat = {data_type_t::s4, "s4",
        /*elem_bits=*/4, /*word_bytes=*/4, /*k_per_word=*/8,
        encode::nibble_octet, scale_enc::real, /*always_grouped=*/true,
        /*sums=*/true,
        /*src=*/data_type_t::s8, /*dst=*/data_type_t::bf16, /*s8_rows=*/false,
        isa_need::vnni, /*max_mr=*/6, /*max_mr_wide=*/3, /*wide_rows=*/3,
        /*two_chain_rows=*/0, /*u8_src=*/false, &pack, &narrow, &wide,
        /*built=*/nullptr};

} // namespace s4
} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_FORMATS_S4_S4_HPP
