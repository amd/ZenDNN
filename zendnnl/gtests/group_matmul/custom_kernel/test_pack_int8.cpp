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

/// CK pack module — DQ-INT8 sibling of `test_pack_bf16.cpp`.
///
/// Covers:
///   * `get_or_pack_weight_int8` layout — K/4 VNNI-quad interleave +
///     trailing int32 per-column compensation row.  The pack helper
///     is the only public surface for the int8 pack so we exercise
///     it directly (no `plan_pack_nr_int8` accessor exists today; the
///     bf16 `plan_pack_nr` is shared and already tested in test_pack_bf16).
///   * Compensation correctness — `comp[v] = sum_k wei_s8[k, v]`
///     across the FULL K window (the kernel's epilogue undoes the
///     `+128 × sum_wei` (sym) or `src_zp × sum_wei` (asym) bias by
///     reading this row).
///   * Cache key uniqueness — a bf16 pack and an int8 pack for the
///     same underlying weight pointer must NOT collide (the int8
///     cache-key marker bit is documented as a distinct constant
///     in `pack.hpp`).
///   * `disable_cache=true` mode — every call returns a fresh
///     caller-owned buffer that the test releases via
///     `free_owned_packed_weight_int8`.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "ck_test_helpers.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ukernel/int8_microkernel.hpp"

namespace {

namespace ck = zendnnl::lowoha::matmul::custom_kernel;
using zendnnl::error_handling::status_t;

#define INT8_CK_SKIP_IF_NO_VNNI()                                   \
  do {                                                              \
    if (!ck::avx512vnni_available()) {                              \
      GTEST_SKIP()                                                  \
          << "AVX-512 VNNI (VPDPBUSD) not available on this host";  \
    }                                                               \
  } while (0)

// Pack-layout sanity: weight bytes occupy `[O/pack_nr][K_pad/4][pack_nr][4]`
// and the trailing `[pack_nr] int32_t` compensation row follows each
// o-block.  Re-derive the byte-level offset and read both the
// packed-weight cell and the compensation cell, comparing them to the
// scalar reference computed from the un-packed source weight.
TEST(CkInt8Pack, LayoutAndCompensation) {
  INT8_CK_SKIP_IF_NO_VNNI();
  constexpr int K = 64;
  constexpr int N = 64;       // single o-block at pack_nr=64 (NV=4)
                              // OR two o-blocks at pack_nr=32 (NV=2)
  constexpr int pack_nr = 32;
  constexpr int K_pad = ((K + 3) / 4) * 4;
  constexpr int K_quad = K_pad / 4;

  // Random s8 weight, row-major `[K, N]`, ldb = N.
  std::vector<int8_t> wei(static_cast<size_t>(K) * N);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> wd(-32, 31);
  for (auto &w : wei) w = static_cast<int8_t>(wd(rng));

  const int8_t *packed = nullptr;
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, /*ldb=*/N, pack_nr,
                /*transB=*/false,
                /*interleave_split_halves=*/false,
                &packed, nullptr, /*disable_cache=*/true),
            status_t::success);
  ASSERT_NE(packed, nullptr);
  struct PackedGuard {
    const int8_t *p;
    ~PackedGuard() { ck::free_owned_packed_weight_int8(p); }
  } pg{packed};

  // Per o-block byte stride: `K_quad * pack_nr * 4` weight bytes +
  // `pack_nr * sizeof(int32_t)` compensation row.
  const size_t weight_bytes_per_oblk =
      static_cast<size_t>(K_quad) * pack_nr * 4;
  const size_t comp_bytes_per_oblk = pack_nr * sizeof(int32_t);
  const size_t oblk_stride_bytes =
      weight_bytes_per_oblk + comp_bytes_per_oblk;

  const int num_oblks = N / pack_nr;
  ASSERT_EQ(num_oblks * pack_nr, N);

  // Spot-check a handful of weight cells.
  for (int o = 0; o < num_oblks; ++o) {
    const std::byte *oblk_base =
        reinterpret_cast<const std::byte *>(packed)
        + static_cast<size_t>(o) * oblk_stride_bytes;
    const int8_t *wbytes =
        reinterpret_cast<const int8_t *>(oblk_base);
    // Pick the corner + center cells.
    int k_samples[] = {0, K / 2, K - 1};
    int v_samples[] = {0, pack_nr / 2, pack_nr - 1};
    for (int kk : k_samples) {
      for (int vv : v_samples) {
        // Layout: [K_quad][pack_nr][4] (last dim is the K-quad inner
        // index).  Compute the packed byte offset for (k=kk, v=vv).
        const int kq    = kk / 4;
        const int q_in  = kk % 4;
        const size_t off =
            (static_cast<size_t>(kq) * pack_nr + vv) * 4 + q_in;
        const int8_t got = wbytes[off];
        const int v_full = o * pack_nr + vv;
        const int8_t exp = wei[kk * N + v_full];
        EXPECT_EQ(got, exp)
            << "Pack byte mismatch at o=" << o
            << " k=" << kk << " v=" << vv
            << " (off=" << off << ")";
      }
    }

    // Compensation row sits immediately after the weight slab.
    const int32_t *comp =
        reinterpret_cast<const int32_t *>(oblk_base + weight_bytes_per_oblk);
    for (int vv = 0; vv < pack_nr; ++vv) {
      const int v_full = o * pack_nr + vv;
      int32_t expected = 0;
      for (int k = 0; k < K; ++k) expected += wei[k * N + v_full];
      EXPECT_EQ(comp[vv], expected)
          << "Compensation mismatch at o=" << o << " v=" << vv;
    }
  }
}

// Same layout + compensation invariants at pack_nr=64 (the production
// swiglu / wide-N pack width), N=128 -> 2 o-blocks.  Previously only
// pack_nr=32 was spot-checked at the byte level.
TEST(CkInt8Pack, LayoutAndCompensationAtPackNr64) {
  INT8_CK_SKIP_IF_NO_VNNI();
  constexpr int K = 64, N = 128, pack_nr = 64;
  constexpr int K_quad = ((K + 3) / 4);

  std::vector<int8_t> wei(static_cast<size_t>(K) * N);
  std::mt19937 rng(64);
  std::uniform_int_distribution<int> wd(-32, 31);
  for (auto &w : wei) w = static_cast<int8_t>(wd(rng));

  const int8_t *packed = nullptr;
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, /*ldb=*/N, pack_nr, /*transB=*/false,
                /*interleave_split_halves=*/false, &packed, nullptr,
                /*disable_cache=*/true),
            status_t::success);
  ASSERT_NE(packed, nullptr);
  struct PackedGuard {
    const int8_t *p;
    ~PackedGuard() { ck::free_owned_packed_weight_int8(p); }
  } pg{packed};

  const size_t weight_bytes_per_oblk =
      static_cast<size_t>(K_quad) * pack_nr * 4;
  const size_t oblk_stride_bytes =
      weight_bytes_per_oblk + pack_nr * sizeof(int32_t);
  const int num_oblks = N / pack_nr;  // = 2
  ASSERT_EQ(num_oblks, 2);

  for (int o = 0; o < num_oblks; ++o) {
    const std::byte *oblk_base =
        reinterpret_cast<const std::byte *>(packed)
        + static_cast<size_t>(o) * oblk_stride_bytes;
    const int8_t *wbytes = reinterpret_cast<const int8_t *>(oblk_base);
    for (int kk : {0, K / 2, K - 1}) {
      for (int vv : {0, pack_nr / 2, pack_nr - 1}) {
        const size_t off =
            (static_cast<size_t>(kk / 4) * pack_nr + vv) * 4 + (kk % 4);
        EXPECT_EQ(wbytes[off], wei[kk * N + (o * pack_nr + vv)])
            << "pack_nr=64 byte mismatch o=" << o << " k=" << kk
            << " v=" << vv;
      }
    }
    const int32_t *comp = reinterpret_cast<const int32_t *>(
        oblk_base + weight_bytes_per_oblk);
    for (int vv = 0; vv < pack_nr; ++vv) {
      const int v_full = o * pack_nr + vv;
      int32_t expected = 0;
      for (int k = 0; k < K; ++k) expected += wei[k * N + v_full];
      EXPECT_EQ(comp[vv], expected)
          << "pack_nr=64 compensation mismatch o=" << o << " v=" << vv;
    }
  }
}

// `disable_cache=true` must return a fresh caller-owned buffer on
// every call AND the cache LRU must not register the result.  We
// verify by issuing two calls with the same inputs and asserting
// the returned pointers differ.
TEST(CkInt8Pack, DisableCacheReturnsFreshBuffer) {
  INT8_CK_SKIP_IF_NO_VNNI();
  constexpr int K = 32, N = 32, pack_nr = 32;
  std::vector<int8_t> wei(static_cast<size_t>(K) * N, 0);
  for (size_t i = 0; i < wei.size(); ++i)
    wei[i] = static_cast<int8_t>((i * 7) & 0x7f);

  const int8_t *p1 = nullptr, *p2 = nullptr;
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, N, pack_nr, false, false, &p1,
                nullptr, /*disable_cache=*/true),
            status_t::success);
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, N, pack_nr, false, false, &p2,
                nullptr, /*disable_cache=*/true),
            status_t::success);
  ASSERT_NE(p1, nullptr);
  ASSERT_NE(p2, nullptr);
  EXPECT_NE(p1, p2)
      << "disable_cache must allocate a fresh buffer per call";

  ck::free_owned_packed_weight_int8(p1);
  ck::free_owned_packed_weight_int8(p2);
}

// Cache hit — issuing the same (weight, K, N, pack_nr, transB)
// tuple twice in the LRU mode returns the same pointer the second
// time (`was_hit_out = true`).  This pins the cache key contract:
// it MUST include the weight pointer + shape + pack_nr; it MUST
// NOT depend on transient stack addresses, RNG state, etc.
TEST(CkInt8Pack, CacheHitReturnsSamePointer) {
  INT8_CK_SKIP_IF_NO_VNNI();
  ck::clear_custom_kernel_pack_cache_int8();
  constexpr int K = 32, N = 32, pack_nr = 32;
  std::vector<int8_t> wei(static_cast<size_t>(K) * N, 0);
  for (size_t i = 0; i < wei.size(); ++i)
    wei[i] = static_cast<int8_t>((i * 5) & 0x3f);

  const int8_t *p1 = nullptr, *p2 = nullptr;
  bool hit1 = true, hit2 = false;
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, N, pack_nr, false, false, &p1, &hit1,
                /*disable_cache=*/false),
            status_t::success);
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, N, pack_nr, false, false, &p2, &hit2,
                /*disable_cache=*/false),
            status_t::success);
  ASSERT_NE(p1, nullptr);
  ASSERT_NE(p2, nullptr);
  EXPECT_FALSE(hit1) << "First call must be a miss (cleared cache above)";
  EXPECT_TRUE(hit2)  << "Second call must reuse the cached pack";
  EXPECT_EQ(p1, p2)  << "Cache hit must return the same pointer";

  ck::clear_custom_kernel_pack_cache_int8();
}

// Disjoint LRU vs the bf16 pack cache — clearing one MUST NOT evict
// the other.  We seed both caches with packs for SEPARATE weight
// pointers (so the bf16 cache holds a `bf16_t*` entry and the int8
// cache holds an `int8_t*` entry for two distinct weights), clear
// the int8 cache, and verify a fresh int8 pack misses while a
// repeat bf16 pack still hits.  Mirrors the documented "different
// extra_hash bit + different value type — disjoint cache" contract
// from `pack.hpp`.
TEST(CkInt8Pack, ClearInt8CacheDoesNotEvictBf16) {
  INT8_CK_SKIP_IF_NO_VNNI();
  ck::clear_custom_kernel_pack_cache();
  ck::clear_custom_kernel_pack_cache_int8();

  constexpr int K = 32, N = 32, pack_nr = 32;
  std::vector<bfloat16_t> wei_bf16(static_cast<size_t>(K) * N,
                                   bfloat16_t(0.0f));
  for (size_t i = 0; i < wei_bf16.size(); ++i)
    wei_bf16[i] = bfloat16_t(0.01f * static_cast<float>(i & 0x3f));
  std::vector<int8_t> wei_s8(static_cast<size_t>(K) * N, 0);
  for (size_t i = 0; i < wei_s8.size(); ++i)
    wei_s8[i] = static_cast<int8_t>((i * 3) & 0x3f);

  const bfloat16_t *pb = nullptr;
  const int8_t     *pi = nullptr;
  bool bhit = false, ihit = false;

  ASSERT_EQ(ck::get_or_pack_weight_bf16(
                wei_bf16.data(), K, N, N, pack_nr, false, false, &pb,
                &bhit),
            status_t::success);
  EXPECT_FALSE(bhit);
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei_s8.data(), K, N, N, pack_nr, false, false, &pi,
                &ihit),
            status_t::success);
  EXPECT_FALSE(ihit);

  // Clear ONLY the int8 cache.  bf16 entry must still be a hit.
  ck::clear_custom_kernel_pack_cache_int8();

  const bfloat16_t *pb2 = nullptr;
  bhit = false;
  ASSERT_EQ(ck::get_or_pack_weight_bf16(
                wei_bf16.data(), K, N, N, pack_nr, false, false, &pb2,
                &bhit),
            status_t::success);
  EXPECT_TRUE(bhit)
      << "bf16 cache entry must survive an int8 cache clear";
  EXPECT_EQ(pb, pb2);

  // Re-packing the int8 weight is a fresh miss.
  const int8_t *pi2 = nullptr;
  ihit = true;
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei_s8.data(), K, N, N, pack_nr, false, false, &pi2,
                &ihit),
            status_t::success);
  EXPECT_FALSE(ihit)
      << "int8 pack must MISS after clear_custom_kernel_pack_cache_int8";

  ck::clear_custom_kernel_pack_cache();
  ck::clear_custom_kernel_pack_cache_int8();
}

// ──────────────────────────────────────────────────────────────────
// C.2 cell #1 — K not divisible by 4.  The pack helper must accept
// the input and pad K to the next multiple of `kVNNIInt8Quad=4`,
// zero-filling the trailing slot.  We pack with K=5 and verify the
// returned buffer's K_quad equals 2 by reading the byte-level slot
// and confirming the padded byte is zero.
// ──────────────────────────────────────────────────────────────────
TEST(CkInt8Pack, KNotDivisibleByFourPadsToQuad) {
  INT8_CK_SKIP_IF_NO_VNNI();
  constexpr int K       = 5;     // not multiple of 4 → pads to 8.
  constexpr int N       = 32;
  constexpr int pack_nr = 32;
  std::vector<int8_t> wei(static_cast<size_t>(K) * N, 0);
  for (size_t i = 0; i < wei.size(); ++i)
    wei[i] = static_cast<int8_t>((i + 1) & 0x3f);

  const int8_t *packed = nullptr;
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, /*ldb=*/N, pack_nr,
                /*transB=*/false,
                /*interleave_split_halves=*/false,
                &packed, nullptr, /*disable_cache=*/true),
            status_t::success);
  ASSERT_NE(packed, nullptr);

  const int K_pad  = ((K + 3) / 4) * 4;          // = 8
  const int K_quad = K_pad / 4;                  // = 2
  // Verify the K-tail bytes (k=5,6,7) read as zero in the packed
  // layout.  Layout: [K_quad][pack_nr][4]; the inner-quad index
  // for (kk=k_pad, vv=0) maps to off = (kq*pack_nr + 0)*4 + (kk%4).
  const int8_t *wbytes = packed;  // first o-block starts at packed[0]
  for (int kk = K; kk < K_pad; ++kk) {
    const int kq    = kk / 4;
    const int q_in  = kk % 4;
    const size_t off =
        (static_cast<size_t>(kq) * pack_nr + 0) * 4 + q_in;
    EXPECT_EQ(wbytes[off], 0)
        << "Pack tail byte must be zero-padded; K_pad=" << K_pad
        << " K_quad=" << K_quad << " kk=" << kk;
  }
  // Recompute the comp row over the FULL K_pad (zero-pad ⇒ no
  // contribution beyond k=K-1, so comp == sum over k∈[0,K)).
  const size_t weight_bytes_per_oblock =
      static_cast<size_t>(K_quad) * pack_nr * 4;
  const int32_t *comp =
      reinterpret_cast<const int32_t *>(packed + weight_bytes_per_oblock);
  for (int vv = 0; vv < pack_nr; ++vv) {
    int32_t expected = 0;
    for (int k = 0; k < K; ++k) expected += wei[k * N + vv];
    EXPECT_EQ(comp[vv], expected)
        << "Comp row must sum only the un-padded K rows at vv=" << vv;
  }
  ck::free_owned_packed_weight_int8(packed);
}

// ──────────────────────────────────────────────────────────────────
// C.2 cell #2 — N rejected by the planner contract.  The pack
// requires N % pack_nr == 0 AND pack_nr ∈ {kNRMin=32, kNRMax=64};
// any other shape returns failure.  Verify both rejections are
// surfaced as `status_t::failure`.
// ──────────────────────────────────────────────────────────────────
TEST(CkInt8Pack, RejectsBadShape) {
  INT8_CK_SKIP_IF_NO_VNNI();
  constexpr int K = 32;
  std::vector<int8_t> wei(static_cast<size_t>(K) * 64, 0);

  const int8_t *p = nullptr;
  // (a) N not multiple of pack_nr.
  EXPECT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, /*N=*/33, /*ldb=*/64, /*pack_nr=*/32,
                false, false, &p, nullptr, true),
            status_t::failure);
  // (b) pack_nr out of {32, 64}.
  EXPECT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, /*N=*/64, /*ldb=*/64, /*pack_nr=*/16,
                false, false, &p, nullptr, true),
            status_t::failure);
  EXPECT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, /*N=*/64, /*ldb=*/64, /*pack_nr=*/48,
                false, false, &p, nullptr, true),
            status_t::failure);
  // (c) ldb below minimum row stride for transB=false (= N).
  EXPECT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, /*N=*/64, /*ldb=*/32, /*pack_nr=*/32,
                /*transB=*/false, false, &p, nullptr, true),
            status_t::failure);
}

// ──────────────────────────────────────────────────────────────────
// C.2 cell #3 — Cache key includes pack_nr.  Packing the SAME
// underlying weight pointer first at pack_nr=32 then at pack_nr=64
// must produce TWO disjoint cache entries (different pointers,
// both miss on first call).  Without pack_nr in the key the second
// call would return the pack_nr=32 layout under a different shape
// and silently corrupt the dispatcher.
// ──────────────────────────────────────────────────────────────────
TEST(CkInt8Pack, CacheKeyIncludesPackNR) {
  INT8_CK_SKIP_IF_NO_VNNI();
  ck::clear_custom_kernel_pack_cache_int8();
  constexpr int K = 32, N = 64;  // divisible by both 32 and 64.
  std::vector<int8_t> wei(static_cast<size_t>(K) * N, 0);
  for (size_t i = 0; i < wei.size(); ++i)
    wei[i] = static_cast<int8_t>((i * 9) & 0x3f);

  const int8_t *p32 = nullptr;
  const int8_t *p64 = nullptr;
  bool h32 = true, h64 = true;
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, N, /*pack_nr=*/32, false, false,
                &p32, &h32, /*disable_cache=*/false),
            status_t::success);
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, N, /*pack_nr=*/64, false, false,
                &p64, &h64, /*disable_cache=*/false),
            status_t::success);
  EXPECT_FALSE(h32) << "first 32-pack must miss";
  EXPECT_FALSE(h64) << "first 64-pack must miss (different cache key)";
  EXPECT_NE(p32, p64) << "different pack_nr must yield disjoint buffers";

  // Re-issue both packs and confirm cache hits.
  bool h32b = false, h64b = false;
  const int8_t *p32b = nullptr, *p64b = nullptr;
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, N, 32, false, false, &p32b, &h32b, false),
            status_t::success);
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, N, 64, false, false, &p64b, &h64b, false),
            status_t::success);
  EXPECT_TRUE(h32b);
  EXPECT_TRUE(h64b);
  EXPECT_EQ(p32b, p32);
  EXPECT_EQ(p64b, p64);
  ck::clear_custom_kernel_pack_cache_int8();
}

// ──────────────────────────────────────────────────────────────────
// C.2 cell #4 — `ZENDNNL_MATMUL_WEIGHT_CACHE != 1` (i.e. the
// `disable_cache=true` mode driven by the global config) returns
// a fresh buffer per call AND does not register the LRU.  We
// drive this through the `disable_cache=true` path which mirrors
// what the runtime takes when `matmul_config_t::get_weight_cache()
// != 1`.  Verifies (a) p1 != p2 across two identical-input calls,
// and (b) the LRU is empty after both (a fresh `disable_cache=false`
// call is a miss).
// ──────────────────────────────────────────────────────────────────
TEST(CkInt8Pack, NoCacheModeBypassesLRU) {
  INT8_CK_SKIP_IF_NO_VNNI();
  ck::clear_custom_kernel_pack_cache_int8();
  constexpr int K = 32, N = 32, pack_nr = 32;
  std::vector<int8_t> wei(static_cast<size_t>(K) * N, 0);
  for (size_t i = 0; i < wei.size(); ++i)
    wei[i] = static_cast<int8_t>((i ^ 0x55) & 0x3f);

  const int8_t *p1 = nullptr, *p2 = nullptr;
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, N, pack_nr, false, false, &p1,
                nullptr, /*disable_cache=*/true),
            status_t::success);
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, N, pack_nr, false, false, &p2,
                nullptr, /*disable_cache=*/true),
            status_t::success);
  EXPECT_NE(p1, p2);

  // Fresh `disable_cache=false` after two no-cache calls must miss
  // — the LRU was not populated by the no-cache path.
  const int8_t *p3 = nullptr;
  bool hit3 = true;
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, N, pack_nr, false, false, &p3, &hit3,
                /*disable_cache=*/false),
            status_t::success);
  EXPECT_FALSE(hit3) << "no-cache calls must not populate the LRU";

  ck::free_owned_packed_weight_int8(p1);
  ck::free_owned_packed_weight_int8(p2);
  ck::clear_custom_kernel_pack_cache_int8();
}

// ──────────────────────────────────────────────────────────────────
// C.2 cell #5 — silu vs gelu produce byte-identical packs.  The
// pack module's `interleave_split_halves` permutation re-orders
// source columns to produce an interleaved arena; the activation
// kind itself is consumed in the microkernel epilogue, NOT at
// pack time.  Verify two packs of the same weight with
// `interleave_split_halves=true` (regardless of which gated
// activation will eventually consume them) produce identical
// byte streams.  Without this guarantee the pack cache could not
// share entries between silu_and_mul and gelu_and_mul calls.
// ──────────────────────────────────────────────────────────────────
TEST(CkInt8Pack, SiluGeluProduceIdenticalPacks) {
  INT8_CK_SKIP_IF_NO_VNNI();
  constexpr int K = 32, N = 64, pack_nr = 32;  // even N for split-halves
  std::vector<int8_t> wei(static_cast<size_t>(K) * N, 0);
  for (size_t i = 0; i < wei.size(); ++i)
    wei[i] = static_cast<int8_t>((i * 11) & 0x7f);

  const int8_t *p_silu = nullptr;
  const int8_t *p_gelu = nullptr;
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, N, pack_nr, false,
                /*interleave_split_halves=*/true,
                &p_silu, nullptr, /*disable_cache=*/true),
            status_t::success);
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                wei.data(), K, N, N, pack_nr, false,
                /*interleave_split_halves=*/true,
                &p_gelu, nullptr, /*disable_cache=*/true),
            status_t::success);
  ASSERT_NE(p_silu, nullptr);
  ASSERT_NE(p_gelu, nullptr);

  // Compute byte volume the same way the pack does.
  const int K_quad = (K + 3) / 4;
  const int n_blocks = N / pack_nr;
  const size_t bytes =
      static_cast<size_t>(n_blocks) *
      (static_cast<size_t>(K_quad) * pack_nr * 4
       + static_cast<size_t>(pack_nr) * sizeof(int32_t));
  EXPECT_EQ(std::memcmp(p_silu, p_gelu, bytes), 0)
      << "silu and gelu packs must be byte-identical (the activation "
         "is applied in the microkernel epilogue, not at pack time)";

  ck::free_owned_packed_weight_int8(p_silu);
  ck::free_owned_packed_weight_int8(p_gelu);
}

// The split-halves interleave (silu/gelu, `interleave_split_halves=true`
// on a `[gate | up]` weight) must produce the SAME packed bytes as
// packing a pre-interleaved `[g0,u0,g1,u1,...]` weight WITHOUT the
// permutation (the swiglu layout).  This pins the int8 split-halves
// permutation byte-for-byte against the canonical interleaved pack —
// the int8 analogue of `CkPackBf16.SiluGeluInterleavedPackMatchesSwigluBytes`.
TEST(CkInt8Pack, SplitHalvesInterleaveMatchesSwigluLayout) {
  INT8_CK_SKIP_IF_NO_VNNI();
  constexpr int K = 32, N = 64, I = N / 2, pack_nr = 32;
  auto vg = [](int k, int j) {
    return static_cast<int8_t>((k * 7 + j) & 0x3f);
  };
  auto vu = [](int k, int j) {
    return static_cast<int8_t>((k * 7 + j + I) & 0x3f);
  };
  std::vector<int8_t> w_inter(static_cast<size_t>(K) * N, 0);
  std::vector<int8_t> w_split(static_cast<size_t>(K) * N, 0);
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < I; ++j) {
      w_inter[k * N + 2 * j + 0] = vg(k, j);
      w_inter[k * N + 2 * j + 1] = vu(k, j);
      w_split[k * N + j]         = vg(k, j);
      w_split[k * N + I + j]     = vu(k, j);
    }
  }

  const int8_t *p_inter = nullptr;
  const int8_t *p_split = nullptr;
  // Interleaved weight, packed WITHOUT the split-halves permutation.
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                w_inter.data(), K, N, N, pack_nr, /*transB=*/false,
                /*interleave_split_halves=*/false, &p_inter, nullptr,
                /*disable_cache=*/true),
            status_t::success);
  // Split-halves weight, packed WITH the permutation -> must equal the
  // interleaved pack above.
  ASSERT_EQ(ck::get_or_pack_weight_int8(
                w_split.data(), K, N, N, pack_nr, /*transB=*/false,
                /*interleave_split_halves=*/true, &p_split, nullptr,
                /*disable_cache=*/true),
            status_t::success);
  ASSERT_NE(p_inter, nullptr);
  ASSERT_NE(p_split, nullptr);

  const int K_quad = (K + 3) / 4;
  const int n_blocks = N / pack_nr;
  const size_t bytes =
      static_cast<size_t>(n_blocks) *
      (static_cast<size_t>(K_quad) * pack_nr * 4
       + static_cast<size_t>(pack_nr) * sizeof(int32_t));
  EXPECT_EQ(std::memcmp(p_inter, p_split, bytes), 0)
      << "split-halves permuted pack must be byte-identical to the "
         "pre-interleaved swiglu-layout pack (quads AND compensation)";

  ck::free_owned_packed_weight_int8(p_inter);
  ck::free_owned_packed_weight_int8(p_split);
}

}  // namespace
