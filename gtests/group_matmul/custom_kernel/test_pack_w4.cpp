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

/// CK pack module — four-bit sibling of `test_pack_int8.cpp`, covering both
/// `W4Format`s: W4A8 (s4, external scales) and MXFP4 (f4_e2m1, f8_e8m0
/// scales in the pack).
///
/// Covers the pack half of the contract in `pack.hpp`: (k, k+4) nibble
/// placement, the per-format store bias, s4's pre-scaled per-group
/// compensation, MXFP4's halved in-pack scale bytes, both caller layouts,
/// the shape and scale refusals, and cache-key independence.

#include <gtest/gtest.h>

#include <cmath>
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

bool is_mxfp4(ck::W4Format fmt) {
    return fmt == ck::kW4FormatMxfp4;
}

// s4 recovers the sign with a subtract, so it is stored biased; E2M1 goes
// through a LUT that needs no bias to undo, so it is stored verbatim.
uint8_t store_bias(ck::W4Format fmt) {
    return is_mxfp4(fmt) ? 0x00u : 0x08u;
}

const char *fmt_name(ck::W4Format fmt) {
    return is_mxfp4(fmt) ? "mxfp4" : "s4";
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

// f8_e8m0 `{G, N}` scales spanning 2^-20..2^19.
std::vector<uint8_t> e8m0_scales(int G, int N) {
    std::vector<uint8_t> s(static_cast<size_t>(G) * N);
    for (int g = 0; g < G; ++g) {
        for (int n = 0; n < N; ++n) {
            s[static_cast<size_t>(g) * N + n]
                    = static_cast<uint8_t>(127 + (g * 7 + n * 3) % 40 - 20);
        }
    }
    return s;
}

// The byte the pack stores for an E8M0 scale: the f32 exponent field of half
// its value, flushed to 0 below the f32 normal range.
uint8_t halved_exponent(uint8_t e8m0) {
    return e8m0 > 1 ? static_cast<uint8_t>(e8m0 - 1) : 0;
}

struct PackedView {
    const int8_t *base;
    int K, N, pack_nr, group_size;
    ck::W4Format fmt = ck::kW4FormatS4;

    size_t slab_bytes() const {
        return static_cast<size_t>(K / ck::kW4Octet) * pack_nr
                * ck::kW4BytesPerOctetCol;
    }
    size_t oblock_bytes() const {
        return ck::w4_oblock_bytes(K, pack_nr, group_size, fmt);
    }
    // Stored (biased) nibble for packed column `col`, K index `k`.
    uint8_t nibble(int col, int k) const {
        const int o_blk = col / pack_nr, n = col % pack_nr;
        const int ko = k / ck::kW4Octet, rem = k % ck::kW4Octet;
        const int q = rem % ck::kW4BytesPerOctetCol;
        const bool high = rem >= ck::kW4BytesPerOctetCol;
        const int8_t *blk = base + static_cast<size_t>(o_blk) * oblock_bytes();
        const uint8_t byte = static_cast<uint8_t>(blk[static_cast<size_t>(ko)
                        * pack_nr * ck::kW4BytesPerOctetCol
                + static_cast<size_t>(n) * ck::kW4BytesPerOctetCol + q]);
        return high ? static_cast<uint8_t>((byte >> 4) & 0x0Fu)
                    : static_cast<uint8_t>(byte & 0x0Fu);
    }
    int32_t comp(int col, int g) const {
        const int o_blk = col / pack_nr, n = col % pack_nr;
        const int8_t *blk = base + static_cast<size_t>(o_blk) * oblock_bytes();
        const int32_t *comp_base
                = reinterpret_cast<const int32_t *>(blk + slab_bytes());
        return comp_base[static_cast<size_t>(g) * pack_nr + n];
    }
    uint8_t scale_byte(int col, int g) const {
        const int o_blk = col / pack_nr, n = col % pack_nr;
        const int8_t *blk = base + static_cast<size_t>(o_blk) * oblock_bytes();
        return static_cast<uint8_t>(
                blk[slab_bytes() + static_cast<size_t>(g) * pack_nr + n]);
    }
    // MXFP4: by how much the pack lowered o-block `o_blk`'s exponents.
    uint8_t exp_shift(int o_blk) const {
        const int8_t *blk = base + static_cast<size_t>(o_blk) * oblock_bytes();
        return static_cast<uint8_t>(blk[slab_bytes()
                + static_cast<size_t>(K / group_size) * pack_nr]);
    }
};

// What follows each nibble slab: s4's compensation over decoded values, or
// MXFP4's halved scale bytes.
void expect_tail(const PackedView &v, int col, int canon,
        const std::vector<uint8_t> &scales) {
    const int G = v.K / v.group_size;
    if (!is_mxfp4(v.fmt)) {
        for (int g = 0; g < G; ++g) {
            int32_t want = 0;
            for (int k = g * v.group_size; k < (g + 1) * v.group_size; ++k) {
                want += s4_true(nib_at(k, canon));
            }
            ASSERT_EQ(v.comp(col, g), want * -128)
                    << "comp mismatch at col=" << col << " g=" << g;
        }
    } else {
        for (int g = 0; g < G; ++g) {
            ASSERT_EQ(v.scale_byte(col, g),
                    halved_exponent(
                            scales[static_cast<size_t>(g) * v.N + canon]))
                    << "scale byte mismatch at col=" << col << " g=" << g;
        }
    }
}

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

// Layout: nibble placement, per-format store bias, the per-group tail.
using PackShape = std::tuple<int, int, int, int, bool>; // K,N,gs,pack_nr,transB

class CkPackW4Layout
    : public ::testing::TestWithParam<std::tuple<PackShape, ck::W4Format>> {};

TEST_P(CkPackW4Layout, MatchesDocumentedLayout) {
    const auto [shape, fmt] = GetParam();
    const auto [K, N, group_size, pack_nr, transB] = shape;
    const int ldb = transB ? K : N;
    const int G = K / group_size;

    const std::vector<int8_t> src = build_packed(K, N, ldb, transB);
    const std::vector<uint8_t> scales = e8m0_scales(G, N);
    const size_t bytes
            = ck::packed_weight_size_w4(K, N, pack_nr, group_size, fmt);
    ASSERT_GT(bytes, 0u);
    OwnedSlab slab(bytes);
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(src.data(), K, N, ldb, pack_nr, transB,
                      /*interleave_split_halves=*/false, group_size, slab.p,
                      fmt, scales.data(), N),
            status_t::success);

    const PackedView v {static_cast<const int8_t *>(slab.p), K, N, pack_nr,
            group_size, fmt};

    // Every (k, col) nibble lands where the microkernel will look for it,
    // carrying the store bias its decode undoes.
    for (int col = 0; col < N; ++col) {
        for (int k = 0; k < K; ++k) {
            const uint8_t expect
                    = static_cast<uint8_t>(nib_at(k, col) ^ store_bias(fmt));
            ASSERT_EQ(v.nibble(col, k), expect)
                    << "nibble mismatch at col=" << col << " k=" << k
                    << " (transB=" << transB << " fmt=" << fmt_name(fmt) << ")";
        }
    }

    for (int col = 0; col < N; ++col) {
        SCOPED_TRACE(fmt_name(fmt));
        expect_tail(v, col, col, scales);
    }
}

INSTANTIATE_TEST_SUITE_P(Shapes, CkPackW4Layout,
        ::testing::Combine(
                ::testing::Values(
                        // K, N, group_size, pack_nr, transB
                        std::make_tuple(128, 64, 32, 32, true),
                        std::make_tuple(128, 64, 32, 32, false),
                        std::make_tuple(256, 128, 64, 64, true),
                        std::make_tuple(256, 128, 64, 64, false),
                        // group_size == 8: the octet floor, one octet per
                        // group.
                        std::make_tuple(64, 32, 8, 32, true),
                        // group_size == K: a single group.
                        std::make_tuple(256, 64, 256, 32, true),
                        // Qwen3-style decode shape.
                        std::make_tuple(2048, 1024, 128, 32, true),
                        // MXFP4's spec group size (gpt-oss K), and a
                        // narrower one the format does not depend on.
                        std::make_tuple(2880, 64, 32, 32, true),
                        std::make_tuple(2880, 64, 16, 64, true)),
                ::testing::Values(ck::kW4FormatS4, ck::kW4FormatMxfp4)));

// Split-halves interleave: pack column `2i` reads gate column `i`, `2i+1` up
// column `N/2 + i`.  A swapped permutation stays in range and only skews the
// output, so assert the mapping rather than infer it from a GEMM result.
class CkPackW4Interleave
    : public ::testing::TestWithParam<std::tuple<PackShape, ck::W4Format>> {};

TEST_P(CkPackW4Interleave, PermutesGateAndUpColumns) {
    const auto [shape, fmt] = GetParam();
    const auto [K, N, group_size, pack_nr, transB] = shape;
    const int ldb = transB ? K : N;
    const int G = K / group_size;
    const int I = N / 2;
    ASSERT_EQ(N % 2, 0) << "interleave needs an even N to split";

    const std::vector<int8_t> src = build_packed(K, N, ldb, transB);
    const std::vector<uint8_t> scales = e8m0_scales(G, N);
    const size_t bytes
            = ck::packed_weight_size_w4(K, N, pack_nr, group_size, fmt);
    ASSERT_GT(bytes, 0u);
    OwnedSlab slab(bytes);
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(src.data(), K, N, ldb, pack_nr, transB,
                      /*interleave_split_halves=*/true, group_size, slab.p, fmt,
                      scales.data(), N),
            status_t::success);

    const PackedView v {static_cast<const int8_t *>(slab.p), K, N, pack_nr,
            group_size, fmt};

    for (int col = 0; col < N; ++col) {
        const int canon = (col & 1) ? (I + (col >> 1)) : (col >> 1);
        for (int k = 0; k < K; ++k) {
            const uint8_t expect
                    = static_cast<uint8_t>(nib_at(k, canon) ^ store_bias(fmt));
            ASSERT_EQ(v.nibble(col, k), expect)
                    << "nibble mismatch at pack col=" << col
                    << " canon=" << canon << " k=" << k << " (transB=" << transB
                    << " fmt=" << fmt_name(fmt) << ")";
        }
        // Compensation and in-pack scales are written after the
        // permutation, so they have to follow the same column the weight
        // bytes did.
        SCOPED_TRACE(fmt_name(fmt));
        expect_tail(v, col, canon, scales);
    }
}

INSTANTIATE_TEST_SUITE_P(Shapes, CkPackW4Interleave,
        ::testing::Combine(::testing::Values(
                                   // K, N, group_size, pack_nr, transB
                                   std::make_tuple(128, 64, 32, 32, true),
                                   std::make_tuple(128, 64, 32, 32, false),
                                   std::make_tuple(256, 128, 64, 64, true),
                                   // gpt-oss's w13 at MXFP4's group size.
                                   std::make_tuple(2880, 64, 32, 32, true),
                                   std::make_tuple(768, 128, 32, 64, true)),
                ::testing::Values(ck::kW4FormatS4, ck::kW4FormatMxfp4)));

// Zero encodes as a stored nibble of 8, not 0.
TEST(CkPackW4, ZeroEncodesAsBiasedEight) {
    const int K = 64, N = 32, gs = 32, nr = 32, ldb = K;
    // All-zero nibbles -> every stored nibble must be 8 (0x88 bytes),
    // and every compensation entry must be 0.
    const std::vector<int8_t> src((static_cast<size_t>(N) * K + 1) / 2, 0);
    const size_t bytes = ck::packed_weight_size_w4(K, N, nr, gs);
    OwnedSlab slab(bytes);
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(src.data(), K, N, ldb, nr,
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

TEST(CkPackW4, CompensationIsPreScaledNegative) {
    const int K = 64, N = 32, gs = 32, nr = 32, ldb = K;
    // Every logical S4 value is +1, so each group sums to 32 and the
    // accumulator seed must be -128 * 32.
    const std::vector<int8_t> src(
            (static_cast<size_t>(N) * K + 1) / 2, static_cast<int8_t>(0x11));
    OwnedSlab slab(ck::packed_weight_size_w4(K, N, nr, gs));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(src.data(), K, N, ldb, nr,
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
TEST(CkPackW4, RejectsInvalidGroupSize) {
    const int K = 128, N = 64, nr = 32, ldb = K;
    const std::vector<int8_t> src = build_packed(K, N, ldb, true);
    OwnedSlab slab(ck::packed_weight_size_w4(K, N, nr, 32));
    ASSERT_NE(slab.p, nullptr);

    for (const int bad : {0, -8, 4, 12, 20, 48}) {
        EXPECT_EQ(ck::packed_weight_size_w4(K, N, nr, bad), 0u)
                << "group_size=" << bad << " must size to 0";
        EXPECT_EQ(ck::prepack_weight_into_w4(
                          src.data(), K, N, ldb, nr, true, false, bad, slab.p),
                status_t::failure)
                << "group_size=" << bad << " must be refused";
    }
}

// Cache-key independence — the slab LENGTH depends on group_size, so
// two group sizes over one weight pointer must not alias.
TEST(CkPackW4, GroupSizeParticipatesInCacheKey) {
    const int K = 256, N = 64, nr = 32, ldb = K;
    const std::vector<int8_t> src = build_packed(K, N, ldb, true);

    ck::clear_custom_kernel_pack_cache_w4();

    const int8_t *p32 = nullptr;
    const int8_t *p64 = nullptr;
    bool hit32 = true, hit64 = true;
    ASSERT_EQ(ck::get_or_pack_weight_w4(src.data(), K, N, ldb, nr, true, false,
                      /*group_size=*/32, &p32, &hit32),
            status_t::success);
    EXPECT_FALSE(hit32) << "first pack must MISS";
    ASSERT_EQ(ck::get_or_pack_weight_w4(src.data(), K, N, ldb, nr, true, false,
                      /*group_size=*/64, &p64, &hit64),
            status_t::success);
    EXPECT_FALSE(hit64) << "a different group_size must MISS, not alias";
    EXPECT_NE(p32, p64) << "distinct group sizes must occupy distinct slabs";

    // Re-asking for the first group size now HITS the original entry.
    const int8_t *again = nullptr;
    bool hit_again = false;
    ASSERT_EQ(ck::get_or_pack_weight_w4(src.data(), K, N, ldb, nr, true, false,
                      32, &again, &hit_again),
            status_t::success);
    EXPECT_TRUE(hit_again);
    EXPECT_EQ(again, p32);

    // Sizes differ: G=8 vs G=4 compensation rows.
    EXPECT_NE(ck::packed_weight_size_w4(K, N, nr, 32),
            ck::packed_weight_size_w4(K, N, nr, 64));

    ck::clear_custom_kernel_pack_cache_w4();
}

// Cache-key independence across formats.  The two formats pack the SAME
// pointer to different bytes, so an aliasing key would serve an s4 slab to
// an MXFP4 lookup and silently corrupt output.
TEST(CkPackW4, FormatParticipatesInCacheKey) {
    const int K = 256, N = 64, gs = 32, nr = 32, ldb = K;
    const std::vector<int8_t> src = build_packed(K, N, ldb, true);
    const std::vector<uint8_t> scales = e8m0_scales(K / gs, N);
    ck::clear_custom_kernel_pack_cache_w4();

    const int8_t *ps4 = nullptr;
    const int8_t *pmx = nullptr;
    bool hit_s4 = true, hit_mx = true;
    ASSERT_EQ(ck::get_or_pack_weight_w4(src.data(), K, N, ldb, nr, true, false,
                      gs, &ps4, &hit_s4, /*disable_cache=*/false,
                      ck::kW4FormatS4),
            status_t::success);
    EXPECT_FALSE(hit_s4);
    ASSERT_EQ(ck::get_or_pack_weight_w4(src.data(), K, N, ldb, nr, true, false,
                      gs, &pmx, &hit_mx, /*disable_cache=*/false,
                      ck::kW4FormatMxfp4, scales.data(), N),
            status_t::success);
    EXPECT_FALSE(hit_mx) << "a different format must MISS, not alias";
    ASSERT_NE(ps4, pmx);
    const size_t slab = static_cast<size_t>(K / ck::kW4Octet) * nr
            * ck::kW4BytesPerOctetCol;
    EXPECT_NE(std::memcmp(ps4, pmx, slab), 0)
            << "the two formats must store the same nibbles differently";

    // Each format still hits its own entry afterwards.
    for (const auto fmt : {ck::kW4FormatS4, ck::kW4FormatMxfp4}) {
        const int8_t *again = nullptr;
        bool hit = false;
        ASSERT_EQ(
                ck::get_or_pack_weight_w4(src.data(), K, N, ldb, nr, true,
                        false, gs, &again, &hit, false, fmt, scales.data(), N),
                status_t::success);
        EXPECT_TRUE(hit) << "fmt=" << fmt_name(fmt);
        EXPECT_EQ(again, is_mxfp4(fmt) ? pmx : ps4);
    }

    ck::clear_custom_kernel_pack_cache_w4();
}

// s4 keeps its compensation layout byte for byte; MXFP4 replaces it with one
// scale byte per (group, column) and one shift byte, padded to 64 per o-block.
TEST(CkPackW4, Mxfp4TailIsOneScaleBytePerGroup) {
    const int K = 2880, N = 64, nr = 32;
    const size_t slab = static_cast<size_t>(N / nr) * (K / ck::kW4Octet) * nr
            * ck::kW4BytesPerOctetCol;
    for (const int gs : {16, 32, 96}) {
        const size_t comp = static_cast<size_t>(N) * (K / gs) * sizeof(int32_t);
        EXPECT_EQ(ck::packed_weight_size_w4(K, N, nr, gs, ck::kW4FormatS4),
                (slab + comp + 63) & ~size_t(63))
                << "gs=" << gs;
    }
    EXPECT_EQ(ck::packed_weight_size_w4(K, N, nr, 32, ck::kW4FormatMxfp4),
            slab
                    + static_cast<size_t>(N / nr)
                            * (((K / 32) * nr + 1 + 63) & ~63));
}

// Every E8M0 byte but NaN is a valid scale; MXFP4 cannot pack without its
// scales at all.
TEST(CkPackW4, Mxfp4RefusesNaNOrMissingScales) {
    const int K = 128, N = 64, gs = 32, nr = 32, ldb = K;
    const std::vector<int8_t> src = build_packed(K, N, ldb, true);
    std::vector<uint8_t> scales = e8m0_scales(K / gs, N);
    OwnedSlab slab(ck::packed_weight_size_w4(K, N, nr, gs, ck::kW4FormatMxfp4));
    ASSERT_NE(slab.p, nullptr);
    const auto pack = [&](const uint8_t *s, int ld) {
        return ck::prepack_weight_into_w4(src.data(), K, N, ldb, nr, true,
                false, gs, slab.p, ck::kW4FormatMxfp4, s, ld);
    };
    EXPECT_EQ(pack(scales.data(), N), status_t::success);
    EXPECT_EQ(pack(nullptr, 0), status_t::failure);
    EXPECT_EQ(pack(scales.data(), N - 1), status_t::failure);
    for (const uint8_t ok : {0, 1, 127, 254}) {
        scales[5] = ok;
        EXPECT_EQ(pack(scales.data(), N), status_t::success) << int(ok);
    }
    scales[5] = 0xFF;
    EXPECT_EQ(pack(scales.data(), N), status_t::failure) << "NaN scale";
}

// The pack halves each scale for the doubled E2M1 decode: byte e becomes
// e - 1, and halves below the f32 normal range (e of 0 or 1) flush to 0,
// which the kernel reads as 0.0f.  Up to `kE8M0InPackMaxExp` nothing else
// changes.
TEST(CkPackW4, Mxfp4HalvesScalesAndFlushesSubnormals) {
    const int K = 64, N = 32, gs = 32, nr = 32, ldb = K;
    const std::vector<int8_t> src = build_packed(K, N, ldb, true);
    std::vector<uint8_t> scales = e8m0_scales(K / gs, N);
    scales[0] = 0;
    scales[1] = 1;
    scales[2] = 2;
    scales[3] = 127;
    scales[4] = ck::kE8M0InPackMaxExp + 1;
    OwnedSlab slab(ck::packed_weight_size_w4(K, N, nr, gs, ck::kW4FormatMxfp4));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(src.data(), K, N, ldb, nr, true, false,
                      gs, slab.p, ck::kW4FormatMxfp4, scales.data(), N),
            status_t::success);
    const PackedView v {static_cast<const int8_t *>(slab.p), K, N, nr, gs,
            ck::kW4FormatMxfp4};
    EXPECT_EQ(v.scale_byte(0, 0), 0u);
    EXPECT_EQ(v.scale_byte(1, 0), 0u);
    EXPECT_EQ(v.scale_byte(2, 0), 1u) << "2^-126 is the smallest normal";
    EXPECT_EQ(v.scale_byte(3, 0), 126u) << "1.0 halves to 0.5";
    EXPECT_EQ(v.scale_byte(4, 0), ck::kE8M0InPackMaxExp);
    EXPECT_EQ(v.exp_shift(0), 0u);
}

// Above `kE8M0InPackMaxExp` the f32 sums could overflow, so the pack lowers
// that o-block's exponents by the excess (flushing those it would take
// below one) and records the shift; other o-blocks are untouched.
TEST(CkPackW4, Mxfp4LowersAnOblockThatWouldOverflow) {
    const int K = 64, N = 64, gs = 32, nr = 32, ldb = K;
    const std::vector<int8_t> src = build_packed(K, N, ldb, true);
    std::vector<uint8_t> scales(static_cast<size_t>(K / gs) * N, 127);
    scales[0] = 254; // halves to 253: 26 over the cap
    scales[1] = 40; // halves to 39, lowered to 13
    scales[2] = 27; // halves to 26, lowered to the flush
    scales[32] = 254 - 30; // second o-block: below the cap
    OwnedSlab slab(ck::packed_weight_size_w4(K, N, nr, gs, ck::kW4FormatMxfp4));
    ASSERT_NE(slab.p, nullptr);
    ASSERT_EQ(ck::prepack_weight_into_w4(src.data(), K, N, ldb, nr, true, false,
                      gs, slab.p, ck::kW4FormatMxfp4, scales.data(), N),
            status_t::success);
    const PackedView v {static_cast<const int8_t *>(slab.p), K, N, nr, gs,
            ck::kW4FormatMxfp4};
    const int shift = 253 - ck::kE8M0InPackMaxExp;
    EXPECT_EQ(v.exp_shift(0), shift);
    EXPECT_EQ(v.scale_byte(0, 0), ck::kE8M0InPackMaxExp);
    EXPECT_EQ(v.scale_byte(1, 0), 39 - shift);
    EXPECT_EQ(v.scale_byte(2, 0), 0u);
    EXPECT_EQ(v.scale_byte(3, 1), 126 - shift) << "every group of the o-block";
    EXPECT_EQ(v.exp_shift(1), 0u);
    EXPECT_EQ(v.scale_byte(32, 0), 254 - 30 - 1);
    EXPECT_EQ(v.scale_byte(33, 1), 126u);
}

// The pack reads row g of the scales at `g * scale_ld`, so the same buffer
// under two strides is two different packs and must not share a cache entry.
TEST(CkPackW4, Mxfp4ScaleStrideParticipatesInCacheKey) {
    const int K = 64, N = 32, gs = 32, nr = 32, ldb = K;
    const std::vector<int8_t> src = build_packed(K, N, ldb, true);
    std::vector<uint8_t> scales(static_cast<size_t>(K / gs) * (N + 1));
    for (size_t i = 0; i < scales.size(); ++i)
        scales[i] = static_cast<uint8_t>(120 + i % 13);
    ck::clear_custom_kernel_pack_cache_w4();

    const int8_t *p32 = nullptr;
    const int8_t *p33 = nullptr;
    bool hit32 = true, hit33 = true;
    ASSERT_EQ(ck::get_or_pack_weight_w4(src.data(), K, N, ldb, nr, true, false,
                      gs, &p32, &hit32, false, ck::kW4FormatMxfp4,
                      scales.data(), N),
            status_t::success);
    ASSERT_EQ(ck::get_or_pack_weight_w4(src.data(), K, N, ldb, nr, true, false,
                      gs, &p33, &hit33, false, ck::kW4FormatMxfp4,
                      scales.data(), N + 1),
            status_t::success);
    EXPECT_FALSE(hit32);
    EXPECT_FALSE(hit33) << "a different scale stride must MISS";
    ASSERT_NE(p32, p33);
    const PackedView v32 {p32, K, N, nr, gs, ck::kW4FormatMxfp4};
    const PackedView v33 {p33, K, N, nr, gs, ck::kW4FormatMxfp4};
    EXPECT_EQ(v32.scale_byte(0, 1), scales[N] - 1);
    EXPECT_EQ(v33.scale_byte(0, 1), scales[N + 1] - 1);
    ck::clear_custom_kernel_pack_cache_w4();
}

// An MXFP4 pack embeds its scales, so new scales over the same weight must
// MISS rather than reuse bytes built from the old ones.
TEST(CkPackW4, Mxfp4ScaleBufferParticipatesInCacheKey) {
    const int K = 128, N = 64, gs = 32, nr = 32, ldb = K;
    const std::vector<int8_t> src = build_packed(K, N, ldb, true);
    const std::vector<uint8_t> a = e8m0_scales(K / gs, N);
    std::vector<uint8_t> b = a;
    b[0] = static_cast<uint8_t>(b[0] + 1);
    ck::clear_custom_kernel_pack_cache_w4();

    const int8_t *pa = nullptr;
    const int8_t *pb = nullptr;
    bool hit_a = true, hit_b = true;
    ASSERT_EQ(ck::get_or_pack_weight_w4(src.data(), K, N, ldb, nr, true, false,
                      gs, &pa, &hit_a, false, ck::kW4FormatMxfp4, a.data(), N),
            status_t::success);
    ASSERT_EQ(ck::get_or_pack_weight_w4(src.data(), K, N, ldb, nr, true, false,
                      gs, &pb, &hit_b, false, ck::kW4FormatMxfp4, b.data(), N),
            status_t::success);
    EXPECT_FALSE(hit_a);
    EXPECT_FALSE(hit_b) << "different scale buffers must not share a pack";
    EXPECT_NE(pa, pb);
    ck::clear_custom_kernel_pack_cache_w4();
}

// disable_cache — fresh caller-owned buffer, never in the LRU.
TEST(CkPackW4, DisableCacheReturnsOwnedBuffer) {
    const int K = 128, N = 64, gs = 32, nr = 32, ldb = K;
    const std::vector<int8_t> src = build_packed(K, N, ldb, true);
    ck::clear_custom_kernel_pack_cache_w4();

    const int8_t *a = nullptr;
    const int8_t *b = nullptr;
    bool hit_a = true, hit_b = true;
    ASSERT_EQ(ck::get_or_pack_weight_w4(src.data(), K, N, ldb, nr, true, false,
                      gs, &a, &hit_a, /*disable_cache=*/true),
            status_t::success);
    ASSERT_EQ(ck::get_or_pack_weight_w4(src.data(), K, N, ldb, nr, true, false,
                      gs, &b, &hit_b, /*disable_cache=*/true),
            status_t::success);
    EXPECT_FALSE(hit_a);
    EXPECT_FALSE(hit_b);
    EXPECT_NE(a, b) << "disable_cache must allocate a fresh buffer per call";
    EXPECT_EQ(std::memcmp(a, b, ck::packed_weight_size_w4(K, N, nr, gs)), 0)
            << "two uncached packs of one weight must be byte-identical";
    ck::free_owned_packed_weight_w4(a);
    ck::free_owned_packed_weight_w4(b);
}
