/********************************************************************************
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

// =============================================================================
// Unit tests for lru_cache_t::try_get().
//
// try_get() is the single-lookup replacement for the find_key()+get() pair
// used across the matmul / group-matmul / conv weight, pack, reorder and
// zero-point-compensation caches. It performs one mutex acquisition and one
// hash lookup, bumps the LRU timestamp on hit, reports presence via its bool
// return, and copies the stored value into the out-parameter.
//
// These tests pin that contract directly — in particular the nullptr-value
// case, which the integration suites (test_postop_cache.cpp, test_matmul.cpp,
// group_matmul/*) do not exercise: those caches only ever store non-null
// holders, whereas the AOCL in-place reorder path legitimately stores nullptr
// as "reorder already done, reuse the caller's buffer". The bool/out split
// exists precisely so a nullptr value reads as a hit, not a miss.
//
// Implementation note: lru_cache_t::evict() std::free()s pointer-typed values.
// The pointer-valued test therefore stores ONLY nullptr (which evict skips);
// the remaining tests use int values (no free path). All caches are built with
// an explicit capacity so the tests never depend on matmul_config defaults
// =============================================================================

#include <gtest/gtest.h>

#include "lowoha_operators/matmul/lru_cache/lru_cache.hpp"
#include "lowoha_operators/matmul/lru_cache/zendnnl_key.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace {

// A populated key reads back as a hit with the stored value.
TEST(LruCacheTryGet, HitReturnsTrueAndValue) {
  lru_cache_t<int, int> cache(8);
  cache.add(42, 100);

  int out = -1;
  EXPECT_TRUE(cache.try_get(42, out));
  EXPECT_EQ(out, 100);
}

// A missing key reports false and must leave the out-parameter untouched —
// the ggml / oneDNN / AOCL miss paths rely on the out-param being undisturbed
// so their fall-through compute/reorder logic runs against a known value.
TEST(LruCacheTryGet, MissReturnsFalseAndLeavesOutUntouched) {
  lru_cache_t<int, int> cache(8);
  cache.add(1, 10);

  constexpr int kSentinel = 0x5A5A;
  int out = kSentinel;
  EXPECT_FALSE(cache.try_get(999, out));
  EXPECT_EQ(out, kSentinel);
}

// The critical case: a key whose stored value is nullptr is a HIT, not a miss.
// This is the AOCL in-place-reorder convention (add(key, nullptr) == "reorder
// done, reuse the user buffer"). A naive "return null on miss" design would
// be indistinguishable here; the bool/out split makes it unambiguous.
TEST(LruCacheTryGet, NullptrValueIsHitNotMiss) {
  lru_cache_t<int, void *> cache(8);
  cache.add(7, nullptr);

  int sentinel = 0;
  void *out = &sentinel;                // non-null sentinel
  EXPECT_TRUE(cache.try_get(7, out));   // present...
  EXPECT_EQ(out, nullptr);              // ...and the value is nullptr
  // evict() skips std::free on nullptr, so destruction here is safe.
}

// try_get() must bump the LRU timestamp exactly like get() did, so a bumped
// entry survives eviction while the un-bumped one is evicted.
TEST(LruCacheTryGet, BumpsRecencyLikeGet) {
  lru_cache_t<int, int> cache(2);
  cache.add(1, 10);
  cache.add(2, 20);

  int out = -1;
  ASSERT_TRUE(cache.try_get(1, out));  // bump key 1 -> key 2 is now LRU

  cache.add(3, 30);                    // capacity 2 -> evict the LRU (key 2)

  EXPECT_TRUE(cache.find_key(1));
  EXPECT_FALSE(cache.find_key(2));
  EXPECT_TRUE(cache.find_key(3));
}

// For a populated key, try_get() is behaviorally equivalent to the old
// find_key()+get() pair: presence agrees and the returned value agrees.
TEST(LruCacheTryGet, EquivalentToFindKeyPlusGet) {
  lru_cache_t<int, int> cache(8);
  cache.add(5, 55);

  ASSERT_TRUE(cache.find_key(5));
  EXPECT_EQ(cache.get(5), 55);

  int out = -1;
  EXPECT_TRUE(cache.try_get(5, out));
  EXPECT_EQ(out, 55);
}

// An evicted key must read back as a miss through try_get() itself (not only
// via find_key) — this is exactly the "entry gone -> recompute/reorder" path
// the weight/pack caches rely on after eviction under capacity pressure.
TEST(LruCacheTryGet, EvictedKeyMissesViaTryGet) {
  lru_cache_t<int, int> cache(1);
  cache.add(1, 10);
  cache.add(2, 20);  // capacity 1 -> key 1 is evicted

  int out = -1;
  EXPECT_FALSE(cache.try_get(1, out));  // evicted -> miss
  EXPECT_EQ(out, -1);                   // out untouched on miss

  out = -1;
  EXPECT_TRUE(cache.try_get(2, out));   // survivor still present
  EXPECT_EQ(out, 20);
}

// try_get() on a never-populated cache is a clean miss.
TEST(LruCacheTryGet, EmptyCacheMisses) {
  lru_cache_t<int, int> cache(8);

  int out = 123;
  EXPECT_FALSE(cache.try_get(0, out));
  EXPECT_EQ(out, 123);
  EXPECT_EQ(cache.get_size(), 0);
}

// Fidelity: exercise try_get() with the real production key type Key_matmul
// (custom std::hash + operator==) rather than a trivial int key, so the
// container's hashing/equality path is covered the way every call site uses it.
TEST(LruCacheTryGet, WorksWithKeyMatmul) {
  lru_cache_t<Key_matmul, int> cache(8);

  int weight_a = 0;
  int weight_b = 0;
  const Key_matmul key_a(/*TransB=*/false, /*K=*/64, /*N=*/128, /*ldb=*/64,
                         &weight_a,
                         static_cast<uint32_t>(matmul_algo_t::aocl_dlp_blocked));
  const Key_matmul key_b(/*TransB=*/false, /*K=*/64, /*N=*/128, /*ldb=*/64,
                         &weight_b,
                         static_cast<uint32_t>(matmul_algo_t::aocl_dlp_blocked));

  cache.add(key_a, 7);

  int out = -1;
  EXPECT_TRUE(cache.try_get(key_a, out));  // present
  EXPECT_EQ(out, 7);

  out = -1;
  EXPECT_FALSE(cache.try_get(key_b, out));  // distinct weight ptr -> distinct key
  EXPECT_EQ(out, -1);
}

}  // namespace
}  // namespace matmul
}  // namespace lowoha
}  // namespace zendnnl
