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

#include "aocl_unreorder_cache.hpp"

#include <array>
#include <limits>
#include <memory>
#include <mutex>

#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/lru_cache/lru_cache.hpp"
#include "lowoha_operators/matmul/lru_cache/zendnnl_key.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace {

constexpr size_t kWeightAlignment = 64;
using cache_value_t = std::shared_ptr<void>;

lru_cache_t<Key_matmul, cache_value_t> &aocl_ck_weight_cache() {
    static lru_cache_t<Key_matmul, cache_value_t> cache;
    return cache;
}

std::array<std::mutex, 64> &unreorder_mutexes() {
    static std::array<std::mutex, 64> mutexes;
    return mutexes;
}

struct aligned_deleter {
    void operator()(void *ptr) const noexcept {
        if (ptr != nullptr) zendnnl_aligned_free(ptr);
    }
};

using aligned_ptr_t = std::unique_ptr<void, aligned_deleter>;

aligned_ptr_t allocate_buffer(size_t bytes) {
    if (bytes == 0
            || bytes > std::numeric_limits<size_t>::max()
                            - (kWeightAlignment - 1)) {
        return {};
    }
    const size_t aligned_bytes
            = (bytes + kWeightAlignment - 1) & ~(kWeightAlignment - 1);
    return aligned_ptr_t(
            zendnnl_aligned_alloc(kWeightAlignment, aligned_bytes));
}

} // namespace

static status_t get_or_cache_aocl_ck_weight_impl(const void *aocl_packed, int K,
        int N, int ldb, bool transB, int32_t num_threads, int pack_nr,
        bool interleave_split_halves, std::shared_ptr<void> &ck_weight,
        bool *was_hit, int32_t weight_cache_mode, bool int8_u8s8) {
    ck_weight.reset();
    if (was_hit != nullptr) *was_hit = false;
    if (aocl_packed == nullptr || K <= 0 || N <= 0 || ldb <= 0) {
        return status_t::failure;
    }
    auto &config = zendnnl::common::matmul_config_t::instance();
    const int32_t effective_cache_mode = weight_cache_mode >= 0
            ? effective_weight_cache_type(weight_cache_mode)
            : config.get_weight_cache();
    // Weight-cache mode 0 promises no persistent pointer-keyed state. A
    // derived-cache hit in that mode could outlive/rebind the caller's packed
    // address and return stale bytes, so decline before consulting the cache.
    if (effective_cache_mode == 0 || config.get_lru_cache_capacity() == 0) {
        return status_t::unimplemented;
    }
    // AOCL derives the blocked panel width from the pair (m_hint, nt_hint).
    // Scalar weight prepack currently exposes only mem_format_b='r', so the
    // original prepack thread hint is not available here. The CK call's
    // current num_threads may differ and cannot safely stand in for it.
    // Decline the derived route when M-hint packing is active; AOCL can still
    // consume its original blocked buffer directly.
    if (config.get_dlp_m_hint() > 0) { return status_t::unimplemented; }

    size_t cache_extra_hash = 0;
    cache_extra_hash = zendnnl::common::hash_combine(
            cache_extra_hash, static_cast<size_t>(pack_nr));
    cache_extra_hash = zendnnl::common::hash_combine(
            cache_extra_hash, interleave_split_halves);
    cache_extra_hash = zendnnl::common::hash_combine(
            cache_extra_hash, static_cast<size_t>(int8_u8s8));
    cache_extra_hash = zendnnl::common::hash_combine(cache_extra_hash,
            static_cast<size_t>(
                    int8_u8s8 ? data_type_t::u8 : data_type_t::bf16));
    cache_extra_hash = zendnnl::common::hash_combine(cache_extra_hash,
            static_cast<size_t>(
                    int8_u8s8 ? data_type_t::s8 : data_type_t::bf16));

    const Key_matmul key(transB, static_cast<unsigned int>(K),
            static_cast<unsigned int>(N), static_cast<unsigned int>(ldb),
            aocl_packed,
            static_cast<uint32_t>(matmul_algo_t::moe_custom_kernel),
            cache_extra_hash);

    auto &cache = aocl_ck_weight_cache();
    if (cache.try_get(key, ck_weight)) {
        if (was_hit != nullptr) *was_hit = true;
        return status_t::success;
    }

    auto &stripes = unreorder_mutexes();
    auto &stripe = stripes[std::hash<Key_matmul> {}(key) % stripes.size()];
    std::lock_guard<std::mutex> lock(stripe);
    if (cache.try_get(key, ck_weight)) {
        if (was_hit != nullptr) *was_hit = true;
        return status_t::success;
    }

    reorder::reorder_params_t params;
    params.is_unreorder = true;
    params.num_threads = num_threads;
    params.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    params.prepack.wei_dtype = int8_u8s8 ? data_type_t::s8 : data_type_t::bf16;
    params.prepack.src_dtype = int8_u8s8 ? data_type_t::u8 : data_type_t::bf16;
    params.prepack.K = K;
    params.prepack.N = N;
    params.prepack.ldb = ldb;
    params.prepack.transposed = transB;

    const size_t raw_bytes = reorder::weight_unreorder_size(params);
    const size_t ck_bytes = int8_u8s8
            ? custom_kernel::packed_weight_size_int8(K, N, pack_nr)
            : custom_kernel::packed_weight_size_bf16(K, N, pack_nr);
    if (raw_bytes == 0 || ck_bytes == 0) { return status_t::unimplemented; }

    aligned_ptr_t raw_temporary = allocate_buffer(raw_bytes);
    aligned_ptr_t ck_arena = allocate_buffer(ck_bytes);
    if (!raw_temporary || !ck_arena) { return status_t::failure; }

    const status_t unreorder_status
            = reorder::reorder_direct(aocl_packed, raw_temporary.get(), params);
    if (unreorder_status != status_t::success) return unreorder_status;

    const status_t pack_status = int8_u8s8
            ? custom_kernel::prepack_weight_into_int8(
                      static_cast<const int8_t *>(raw_temporary.get()), K, N,
                      ldb, pack_nr, transB, interleave_split_halves,
                      ck_arena.get())
            : custom_kernel::prepack_weight_into_bf16(
                      static_cast<const bfloat16_t *>(raw_temporary.get()), K,
                      N, ldb, pack_nr, transB, interleave_split_halves,
                      ck_arena.get());
    if (pack_status != status_t::success) return pack_status;

    cache_value_t candidate(ck_arena.release(), aligned_deleter {});
    ck_weight = cache.get_or_add(key, candidate);
    return ck_weight ? status_t::success : status_t::failure;
}

status_t get_or_cache_aocl_ck_weight(const void *aocl_packed, int K, int N,
        int ldb, bool transB, int32_t num_threads, int pack_nr,
        bool interleave_split_halves, std::shared_ptr<void> &ck_weight,
        bool *was_hit, int32_t weight_cache_mode) {
    return get_or_cache_aocl_ck_weight_impl(aocl_packed, K, N, ldb, transB,
            num_threads, pack_nr, interleave_split_halves, ck_weight, was_hit,
            weight_cache_mode, /*int8_u8s8=*/false);
}

status_t get_or_cache_aocl_ck_weight_int8(const void *aocl_packed, int K, int N,
        int ldb, bool transB, int32_t num_threads, int pack_nr,
        bool interleave_split_halves, std::shared_ptr<void> &ck_weight,
        bool *was_hit, int32_t weight_cache_mode) {
    return get_or_cache_aocl_ck_weight_impl(aocl_packed, K, N, ldb, transB,
            num_threads, pack_nr, interleave_split_halves, ck_weight, was_hit,
            weight_cache_mode, /*int8_u8s8=*/true);
}

void clear_aocl_ck_weight_cache() {
    aocl_ck_weight_cache().clear();
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
