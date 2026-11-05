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

#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include <algorithm>
#include <cstring>
#include "common/op_config.hpp"
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_gemv_bkc.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/int8_gemv_bkc.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// LRU capacity (in entries) shared across every native weight cache.
// Sourced from matmul_config_t (env: ZENDNNL_LRU_CACHE_CAPACITY).
// Default UINT32_MAX → eviction disabled.
uint32_t get_weight_cache_capacity() {
    return common::matmul_config_t::instance().get_lru_cache_capacity();
}

// ============================================================================
// BF16 Blocked K-contiguous (BKC) VNNI packed weight cache
// ============================================================================

BF16BKCWeightCache &BF16BKCWeightCache::instance() {
    static BF16BKCWeightCache inst;
    return inst;
}

const BF16BKCWeight *BF16BKCWeightCache::find(
        const PrepackedWeightKey &key, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    const BF16BKCWeight *cached = cache_.find_and_touch(key);
    return cached != nullptr
                    && weight_end_words_hold(
                            cached->source_ends, key.weight_ptr, bytes)
            ? cached
            : nullptr;
}

const BF16BKCWeight *BF16BKCWeightCache::get_or_pack(
        const PrepackedWeightKey &key, const uint16_t *weight, size_t bytes) {

    // Fast path: check cache under lock.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto *cached = cache_.find_and_touch(key);
        if (cached != nullptr
                && weight_end_words_hold(cached->source_ends, weight, bytes))
            return cached;
    }

    // Cache miss: pack outside the lock to avoid serializing threads.
    const int K = key.K, N = key.N, ldb = key.ldb;
    const bool transB = key.transB;
    if (K <= 0 || N <= 0) return nullptr;
    const int K_padded = (K + 1) & ~1;
    const int N_padded = ((N + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
    const size_t total = static_cast<size_t>(K_padded) * N_padded;

    uint16_t *buf = static_cast<uint16_t *>(zendnnl_aligned_alloc(
            64, ((total * sizeof(uint16_t) + 63) & ~size_t(63))));
    if (!buf) return nullptr;

    const weight_end_words_t source_ends = weight_end_words_of(weight, bytes);
    pack_b_bkc_ext(weight, ldb, K, N, transB, buf);

    auto pw = std::make_unique<BF16BKCWeight>();
    pw->buf.reset(buf);
    pw->data = buf;
    pw->K = K;
    pw->K_padded = K_padded;
    pw->N = N;
    pw->N_padded = N_padded;
    pw->total = total;
    pw->source_ends = source_ends;

    // Re-acquire lock and insert (another thread may have raced); a pack of
    // other bytes under the key is replaced.
    std::lock_guard<std::mutex> lock(mutex_);
    auto *cached = cache_.find_and_touch(key);
    if (cached != nullptr
            && weight_end_words_hold(cached->source_ends, weight, bytes))
        return cached; // race loser — discard our pack, use winner's

    return cache_.insert(key, std::move(pw));
}

// ============================================================================
// INT8 K-contiguous VNNI packed weight cache
// ============================================================================

INT8KContiguousWeightCache &INT8KContiguousWeightCache::instance() {
    static INT8KContiguousWeightCache inst;
    return inst;
}

const INT8KContiguousWeight *INT8KContiguousWeightCache::get_or_pack(
        const PrepackedWeightKey &key, const int8_t *weight, size_t bytes) {

    // Fast path: check cache under lock.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto *entry = cache_.find_and_touch(key);
        if (entry != nullptr
                && weight_end_words_hold(entry->source_ends, weight, bytes))
            return entry;
    }

    // Cache miss: pack outside the lock.
    const int K = key.K, N = key.N, ldb = key.ldb;
    const bool transB = key.transB;
    if (K <= 0 || N <= 0) return nullptr;
    const int K_padded = (K + 3) & ~3;
    const int N_padded = ((N + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
    const size_t packed_total = static_cast<size_t>(K_padded) * N_padded;
    // int8_gemv_bkc_nr64_core<1> loads a full 64-column panel for sub-64
    // tails. When the last block's nb_padded < 64, those unmasked loads
    // extend past packed_total. The guard ensures valid (zero) memory so
    // the accumulator picks up zeros; the epilogue masks prevent output
    // corruption. No effect on performance — allocation-time only.
    constexpr size_t bkc_guard = NR_PACK * INT8_VNNI_GRP;

    int8_t *pbuf = static_cast<int8_t *>(zendnnl_aligned_alloc(
            64, ((packed_total + bkc_guard + 63) & ~size_t(63))));
    if (!pbuf) return nullptr;

    int32_t *cs_buf = static_cast<int32_t *>(zendnnl_aligned_alloc(
            64, ((N_padded * sizeof(int32_t) + 63) & ~size_t(63))));
    if (!cs_buf) {
        zendnnl_aligned_free(pbuf);
        return nullptr;
    }

    const weight_end_words_t source_ends = weight_end_words_of(weight, bytes);
    pack_b_int8_bkc(weight, ldb, K, N, transB, pbuf, cs_buf);
    std::memset(pbuf + packed_total, 0, bkc_guard);

    auto pw = std::make_unique<INT8KContiguousWeight>();
    pw->packed_buf.reset(pbuf);
    pw->col_sum_buf.reset(cs_buf);
    pw->data = pbuf;
    pw->col_sum = cs_buf;
    pw->K = K;
    pw->K_padded = K_padded;
    pw->N = N;
    pw->N_padded = N_padded;
    pw->total = packed_total;
    pw->source_ends = source_ends;

    // Re-acquire lock and insert; a pack of other bytes under the key is
    // replaced.
    std::lock_guard<std::mutex> lock(mutex_);
    auto *cached = cache_.find_and_touch(key);
    if (cached != nullptr
            && weight_end_words_hold(cached->source_ends, weight, bytes))
        return cached;

    return cache_.insert(key, std::move(pw));
}

std::atomic<uint64_t> &weight_cache_generation() {
    static std::atomic<uint64_t> gen {0};
    return gen;
}

void clear_all_weight_caches() {
    BF16BKCWeightCache::instance().clear();
    INT8KContiguousWeightCache::instance().clear();
    clear_native_weight_caches();
    weight_cache_generation().fetch_add(1, std::memory_order_relaxed);
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
