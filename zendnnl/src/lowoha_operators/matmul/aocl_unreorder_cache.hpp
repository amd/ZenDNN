/*******************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

#ifndef ZENDNNL_LOWOHA_MATMUL_AOCL_UNREORDER_CACHE_HPP
#define ZENDNNL_LOWOHA_MATMUL_AOCL_UNREORDER_CACHE_HPP

#include <cstdint>
#include <memory>

#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

/**
 * Convert an AOCL-DLP-blocked BF16 weight into the CK VNNI layout and retain
 * one final derived arena. The INT8 sibling below provides the same ownership
 * contract for pure static U8xS8 packs.
 *
 * On a miss, AOCL unreorders into temporary strided raw storage and CK packs
 * directly into a right-sized final arena. The temporary raw representation
 * is never published or retained, and no full-size CK memcpy is required.
 *
 * The returned shared pointer is a call-scoped lease, so eviction or an
 * explicit cache clear cannot release the CK buffer while its caller uses it.
 */
status_t get_or_cache_aocl_ck_weight(const void *aocl_packed, int K, int N,
        int ldb, bool transB, int32_t num_threads, int pack_nr,
        bool interleave_split_halves, std::shared_ptr<void> &ck_weight,
        bool *was_hit = nullptr, int32_t weight_cache_mode = -1);

/**
 * Convert a pure static U8xS8 AOCL-DLP-blocked weight into the CK INT8
 * VNNI-quad layout (including CK's per-column compensation row).
 *
 * AOCL symmetric-quant packs are deliberately excluded: AOCL exposes no
 * inverse for `s8s8s32os32_sym_quant`. The source dtype is fixed to U8 so the
 * inverse exactly matches `aocl_reorder_u8s8s32os32`.
 */
status_t get_or_cache_aocl_ck_weight_int8(const void *aocl_packed, int K, int N,
        int ldb, bool transB, int32_t num_threads, int pack_nr,
        bool interleave_split_halves, std::shared_ptr<void> &ck_weight,
        bool *was_hit = nullptr, int32_t weight_cache_mode = -1);

/** Clear every cached AOCL-derived CK weight. */
void clear_aocl_ck_weight_cache();

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_LOWOHA_MATMUL_AOCL_UNREORDER_CACHE_HPP
