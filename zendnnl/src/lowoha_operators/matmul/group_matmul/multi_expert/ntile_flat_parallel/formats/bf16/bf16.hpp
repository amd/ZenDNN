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

/// BF16 weights against BF16 rows, which the call passes as they are: packs
/// of VDPBF16PS pairs ([ceil(K/2)][32][2]), and kernels that store BF16.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_FORMATS_BF16_BF16_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_FORMATS_BF16_BF16_HPP

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/format.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {
namespace bf16 {

void pack(const void *weight, const weight_desc &d, const pack_out &out);
ukernel_fn narrow(
        int mr, epilogue ep, bool src_u8, bool unaligned, scale_groups groups);
ukernel_fn wide(
        int mr, epilogue ep, bool src_u8, bool two_chains, scale_groups groups);

inline constexpr format_desc kFormat = {data_type_t::bf16, "bf16",
        /*elem_bits=*/16, /*word_bytes=*/4, /*k_per_word=*/2, encode::copy,
        scale_enc::none, /*always_grouped=*/false, /*sums=*/false,
        /*src=*/data_type_t::bf16, /*dst=*/data_type_t::bf16,
        /*s8_rows=*/false, isa_need::bf16, /*max_mr=*/8, /*max_mr_wide=*/6,
        /*wide_rows=*/0, /*two_chain_rows=*/3, /*u8_src=*/false, &pack, &narrow,
        &wide, /*built=*/nullptr};

} // namespace bf16
} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_FORMATS_BF16_BF16_HPP
