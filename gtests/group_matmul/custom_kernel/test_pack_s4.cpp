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

/// CK pack module — W4A8 (s4) sibling of `test_pack_int8.cpp`.
///
/// Covers the pack half of the contract in `pack.hpp`: (k, k+4) nibble
/// placement, XOR-8 bias, pre-scaled per-group compensation over
/// sign-recovered values, both caller layouts, the `group_size` shape refusals, and
/// cache-key independence across group sizes.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/pack.hpp"

namespace ck = ::zendnnl::lowoha::matmul::custom_kernel;
using zendnnl::error_handling::status_t;

namespace {

// Sign-recovered value, matching the AOCL `extract_4bit_nibble`.
int s4_true(uint8_t raw) {
    return (raw & 0x08u) ? (static_cast<int>(raw) - 16) : static_cast<int>(raw);
}

// Deterministic so expected values are recomputable without a buffer.
uint8_t nib_at(int k, int n) {
    uint32_t h = static_cast<uint32_t>(k) * 2654435761u
            + static_cast<uint32_t>(n) * 40503u;
    return static_cast<uint8_t>((h >> 13) & 0x0Fu);
}

// Linear nibble index `transB ? n*ldb + k : k*ldb + n`, even index in
// the low nibble — the `cvt_s4_to_s8` convention.
std::vector<int8_t> build_packed(int K, int N, int ldb, bool transB) {
    std::vector<int8_t> buf((static_cast<size_t>(N) * K + 1) / 2, 0);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            const uint8_t raw = nib_at(k, n);
            const size_t idx = transB ? static_cast<size_t>(n) * ldb + k
                                      : static_cast<size_t>(k) * ldb + n;
            auto &byte = reinterpret_cast<uint8_t &>(buf[idx >> 1]);
            if ((idx & 1u) == 0u) {
                byte = static_cast<uint8_t>((byte & 0xF0u) | raw);
            } else {
                byte = static_cast<uint8_t>(
                        (byte & 0x0Fu) | static_cast<uint8_t>(raw << 4));
            }
        }
    }
    return buf;
}

struct PackedView {
    const int8_t *base;
    int K, N, pack_nr, group_size;

    size_t oblock_bytes() const {
        return static_cast<size_t>(K / ck::kS4Octet) * pack_nr
                * ck::kS4BytesPerOctetCol
                + static_cast<size_t>(K / group_size) * pack_nr
                * sizeof(int32_t);
    }
    // Stored (biased) nibble for packed column `col`, K index `k`.
    uint8_t nibble(int col, int k) const {
        const int o_blk = col / pack_nr, n = col % pack_nr;
        const int ko = k / ck::kS4Octet, rem = k % ck::kS4Octet;
        const int q = rem % ck::kS4BytesPerOctetCol;
        const bool high = rem >= ck::kS4BytesPerOctetCol;
        const int8_t *blk = base + static_cast<size_t>(o_blk) * oblock_bytes();
        const uint8_t byte = static_cast<uint8_t>(blk[static_cast<size_t>(ko)
                        * pack_nr * ck::kS4BytesPerOctetCol
                + static_cast<size_t>(n) * ck::kS4BytesPerOctetCol + q]);
        return high ? static_cast<uint8_t>((byte >> 4) & 0x0Fu)
                    : static_cast<uint8_t>(byte & 0x0Fu);
    }
    int32_t comp(int col, int g) const {
        const int o_blk = col / pack_nr, n = col % pack_nr;
        const int8_t *blk = base + static_cast<size_t>(o_blk) * oblock_bytes();
        const int32_t *comp_base = reinterpret_cast<const int32_t *>(blk
                + static_cast<size_t>(K / ck::kS4Octet) * pack_nr
                        * ck::kS4BytesPerOctetCol);
        return comp_base[static_cast<size_t>(g) * pack_nr + n];
    }
};

// Owns an aligned prepack destination so a failing EXPECT cannot leak.
struct OwnedSlab {
    void *p = nullptr;
    explicit OwnedSlab(size_t bytes) : p(zendnnl_aligned_alloc(64, bytes)) {
        if (p != nullptr) std::memset(p, 0, bytes);
    }
    ~OwnedSlab() { zendnnl_aligned_free(p); }
    OwnedSlab(const OwnedSlab &) = delete;
    OwnedSlab &operator=(const OwnedSlab &) = delete;
};

} // namespace

// Layout: nibble placement, XOR-8 bias, per-group compensation.
class CkPackS4Layout
    : public ::testing::TestWithParam<std::tuple<int, int, int, int, bool>> {};

TEST_P(CkPackS4Layout, MatchesDocumentedLayout) {
    const auto [K, N, group_size, pack_nr, transB] = GetParam();
    const int ldb = transB ? K : N;
    const int G = K / group_size;

    const std::vector<int8_t> src = build_packed(K, N, ldb, transB);
    const size_t bytes = ck::packed_weight_size_s4(K, N, pack_nr, group_size);
    ASSERT_GT(bytes, 0u);
    OwnedSlab slab(bytes);
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_s4(src.data(), K, N, ldb, pack_nr, transB,
                      /*interleave_split_halves=*/false, group_size, slab.p),
            status_t::success);

    const PackedView v {
            static_cast<const int8_t *>(slab.p), K, N, pack_nr, group_size};

    // Every (k, col) nibble lands where the microkernel will look for
    // it, biased by XOR 8.
    for (int col = 0; col < N; ++col) {
        for (int k = 0; k < K; ++k) {
            const uint8_t expect = static_cast<uint8_t>(nib_at(k, col) ^ 0x08u);
            ASSERT_EQ(v.nibble(col, k), expect)
                    << "nibble mismatch at col=" << col << " k=" << k
                    << " (transB=" << transB << ")";
        }
    }

    // Complete per-group correction over SIGN-RECOVERED values.
    for (int col = 0; col < N; ++col) {
        for (int g = 0; g < G; ++g) {
            int32_t want = 0;
            for (int k = g * group_size; k < (g + 1) * group_size; ++k) {
                want += s4_true(nib_at(k, col));
            }
            want *= -128;
            ASSERT_EQ(v.comp(col, g), want)
                    << "comp mismatch at col=" << col << " g=" << g;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Shapes, CkPackS4Layout,
        ::testing::Values(
                // K, N, group_size, pack_nr, transB
                std::make_tuple(128, 64, 32, 32, true),
                std::make_tuple(128, 64, 32, 32, false),
                std::make_tuple(256, 128, 64, 64, true),
                std::make_tuple(256, 128, 64, 64, false),
                // group_size == 8: the octet floor, one octet per group.
                std::make_tuple(64, 32, 8, 32, true),
                // group_size == K: a single group.
                std::make_tuple(256, 64, 256, 32, true),
                // Qwen3-style decode shape.
                std::make_tuple(2048, 1024, 128, 32, true)));

// Zero encodes as a stored nibble of 8, not 0.
TEST(CkPackS4, ZeroEncodesAsBiasedEight) {
    const int K = 64, N = 32, gs = 32, nr = 32, ldb = K;
    // All-zero nibbles -> every stored nibble must be 8 (0x88 bytes),
    // and every compensation entry must be 0.
    const std::vector<int8_t> src((static_cast<size_t>(N) * K + 1) / 2, 0);
    const size_t bytes = ck::packed_weight_size_s4(K, N, nr, gs);
    OwnedSlab slab(bytes);
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_s4(src.data(), K, N, ldb, nr,
                      /*transB=*/true, false, gs, slab.p),
            status_t::success);

    const PackedView v {static_cast<const int8_t *>(slab.p), K, N, nr, gs};
    for (int col = 0; col < N; ++col) {
        for (int k = 0; k < K; ++k) {
            ASSERT_EQ(v.nibble(col, k), 0x08u)
                    << "value 0 must store as biased nibble 8";
        }
        for (int g = 0; g < K / gs; ++g) {
            ASSERT_EQ(v.comp(col, g), 0);
        }
    }
}

TEST(CkPackS4, CompensationIsPreScaledNegative) {
    const int K = 64, N = 32, gs = 32, nr = 32, ldb = K;
    // Every logical S4 value is +1, so each group sums to 32 and the
    // accumulator seed must be -128 * 32.
    const std::vector<int8_t> src(
            (static_cast<size_t>(N) * K + 1) / 2, static_cast<int8_t>(0x11));
    OwnedSlab slab(ck::packed_weight_size_s4(K, N, nr, gs));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_s4(src.data(), K, N, ldb, nr,
                      /*transB=*/true, /*interleave_split_halves=*/false, gs,
                      slab.p),
            status_t::success);

    PackedView v {static_cast<const int8_t *>(slab.p), K, N, nr, gs};
    for (int col = 0; col < N; ++col) {
        for (int g = 0; g < K / gs; ++g) {
            EXPECT_EQ(v.comp(col, g), -128 * gs);
        }
    }
}

// Shape refusals — group_size must be a positive multiple of 8 that
// divides K, the constraint that removes kernel tail handling.
TEST(CkPackS4, RejectsInvalidGroupSize) {
    const int K = 128, N = 64, nr = 32, ldb = K;
    const std::vector<int8_t> src = build_packed(K, N, ldb, true);
    OwnedSlab slab(ck::packed_weight_size_s4(K, N, nr, 32));
    ASSERT_NE(slab.p, nullptr);

    for (const int bad : {0, -8, 4, 12, 20, 48}) {
        EXPECT_EQ(ck::packed_weight_size_s4(K, N, nr, bad), 0u)
                << "group_size=" << bad << " must size to 0";
        EXPECT_EQ(ck::prepack_weight_into_s4(
                          src.data(), K, N, ldb, nr, true, false, bad, slab.p),
                status_t::failure)
                << "group_size=" << bad << " must be refused";
    }
}

// Cache-key independence — the slab LENGTH depends on group_size, so
// two group sizes over one weight pointer must not alias.
TEST(CkPackS4, GroupSizeParticipatesInCacheKey) {
    const int K = 256, N = 64, nr = 32, ldb = K;
    const std::vector<int8_t> src = build_packed(K, N, ldb, true);

    ck::clear_custom_kernel_pack_cache_s4();

    const int8_t *p32 = nullptr;
    const int8_t *p64 = nullptr;
    bool hit32 = true, hit64 = true;
    ASSERT_EQ(ck::get_or_pack_weight_s4(src.data(), K, N, ldb, nr, true, false,
                      /*group_size=*/32, &p32, &hit32),
            status_t::success);
    EXPECT_FALSE(hit32) << "first pack must MISS";
    ASSERT_EQ(ck::get_or_pack_weight_s4(src.data(), K, N, ldb, nr, true, false,
                      /*group_size=*/64, &p64, &hit64),
            status_t::success);
    EXPECT_FALSE(hit64) << "a different group_size must MISS, not alias";
    EXPECT_NE(p32, p64) << "distinct group sizes must occupy distinct slabs";

    // Re-asking for the first group size now HITS the original entry.
    const int8_t *again = nullptr;
    bool hit_again = false;
    ASSERT_EQ(ck::get_or_pack_weight_s4(src.data(), K, N, ldb, nr, true, false,
                      32, &again, &hit_again),
            status_t::success);
    EXPECT_TRUE(hit_again);
    EXPECT_EQ(again, p32);

    // Sizes differ: G=8 vs G=4 compensation rows.
    EXPECT_NE(ck::packed_weight_size_s4(K, N, nr, 32),
            ck::packed_weight_size_s4(K, N, nr, 64));

    ck::clear_custom_kernel_pack_cache_s4();
}

// disable_cache — fresh caller-owned buffer, never in the LRU.
TEST(CkPackS4, DisableCacheReturnsOwnedBuffer) {
    const int K = 128, N = 64, gs = 32, nr = 32, ldb = K;
    const std::vector<int8_t> src = build_packed(K, N, ldb, true);
    ck::clear_custom_kernel_pack_cache_s4();

    const int8_t *a = nullptr;
    const int8_t *b = nullptr;
    bool hit_a = true, hit_b = true;
    ASSERT_EQ(ck::get_or_pack_weight_s4(src.data(), K, N, ldb, nr, true, false,
                      gs, &a, &hit_a, /*disable_cache=*/true),
            status_t::success);
    ASSERT_EQ(ck::get_or_pack_weight_s4(src.data(), K, N, ldb, nr, true, false,
                      gs, &b, &hit_b, /*disable_cache=*/true),
            status_t::success);
    EXPECT_FALSE(hit_a);
    EXPECT_FALSE(hit_b);
    EXPECT_NE(a, b) << "disable_cache must allocate a fresh buffer per call";
    EXPECT_EQ(std::memcmp(a, b, ck::packed_weight_size_s4(K, N, nr, gs)), 0)
            << "two uncached packs of one weight must be byte-identical";
    ck::free_owned_packed_weight_s4(a);
    ck::free_owned_packed_weight_s4(b);
}
