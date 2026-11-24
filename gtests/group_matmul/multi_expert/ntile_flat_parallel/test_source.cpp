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

/**
 * @file test_source.cpp
 * @brief Tests for N-tile flat parallel's source rows.
 *
 * The rows a call makes for its kernels are checked against the quantizer
 * they come from, run on each scale group, with and without the u8 flip; the
 * copies and views of a row against the row; and the scratch's sizing
 * against the rows and the bytes the kernels read past them.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/source.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

namespace {

namespace nfp = zendnnl::lowoha::matmul::ntile_flat_parallel;
namespace detail = nfp::detail;

detail::source_spec s8_spec(const int groups, const bool u8) {
    detail::source_spec s;
    s.s8 = true;
    s.groups = groups;
    s.u8 = u8;
    s.elem = 1;
    return s;
}

detail::source_spec bf16_spec() {
    detail::source_spec s;
    s.elem = 2;
    return s;
}

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

std::vector<uint16_t> bf16_values(const int n, const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-3.0f, 3.0f);
    std::vector<uint16_t> v(static_cast<size_t>(n));
    for (uint16_t &h : v)
        h = to_bf16(unit(rng));
    return v;
}

std::vector<int8_t> s8_values(const int n, const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code(-128, 127);
    std::vector<int8_t> v(static_cast<size_t>(n));
    for (int8_t &c : v)
        c = static_cast<int8_t>(code(rng));
    return v;
}

bool all_zero(const uint8_t *bytes, const size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (bytes[i] != 0) return false;
    }
    return true;
}

// Each scale group of a quantized row is the quantizer's row for that group,
// XOR-ed with 0x80 for the u8-source kernels, and no other row changes.
TEST(Nr32Source, AQuantizedRowIsTheQuantizersRowPerGroup) {
    constexpr int kLen = 96;
    const std::vector<uint16_t> in = bf16_values(kLen, 7);
    for (const int groups : {1, 3, 4}) {
        for (const bool u8 : {false, true}) {
            SCOPED_TRACE(::testing::Message()
                    << "groups " << groups << ", u8 " << u8);
            const detail::source_spec spec = s8_spec(groups, u8);
            detail::source_rows out;
            out.grow(spec, 3, kLen);
            detail::quantize_row(spec, in.data(), kLen, out, 1);

            const int run = kLen / groups;
            std::vector<int8_t> want(kLen);
            std::vector<float> want_scales(groups);
            for (int g = 0; g < groups; ++g) {
                zendnnl::lowoha::reorder::
                        dynamic_per_token_quant_bf16_s8_native(
                                in.data() + g * run, want.data() + g * run,
                                want_scales.data() + g, 1, run);
            }
            const uint8_t *const row = out.row(spec, 1, kLen);
            for (int c = 0; c < kLen; ++c) {
                const uint8_t code = static_cast<uint8_t>(want[c]);
                ASSERT_EQ(
                        row[c], u8 ? static_cast<uint8_t>(code ^ 0x80u) : code)
                        << "column " << c;
            }
            const float *const scales = out.row_scales(spec, 1);
            for (int g = 0; g < groups; ++g)
                EXPECT_EQ(scales[g], want_scales[g]) << "group " << g;
            EXPECT_TRUE(all_zero(out.row(spec, 0, kLen), kLen));
            EXPECT_TRUE(all_zero(out.row(spec, 2, kLen), kLen));
        }
    }
}

// A caller's s8 row is copied as it is, and flipped only for u8 sources,
// including the bytes past the last whole word.
TEST(Nr32Source, ACopiedS8RowFlipsOnlyForU8) {
    constexpr int kLen = 37;
    const std::vector<int8_t> in = s8_values(kLen, 11);
    for (const bool u8 : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "u8 " << u8);
        const detail::source_spec spec = s8_spec(1, u8);
        detail::source_rows out;
        out.grow(spec, 3, kLen);
        detail::copy_s8_row(spec, in.data(), kLen, out, 2);
        const uint8_t *const row = out.row(spec, 2, kLen);
        for (int c = 0; c < kLen; ++c) {
            const uint8_t code = static_cast<uint8_t>(in[c]);
            ASSERT_EQ(row[c], u8 ? static_cast<uint8_t>(code ^ 0x80u) : code)
                    << "column " << c;
        }
        EXPECT_TRUE(all_zero(out.rows.data(), 2 * kLen));
    }
}

// A copied row has the row's elements and, for s8 rows, its scales.
TEST(Nr32Source, ACopiedRowKeepsItsScales) {
    constexpr int kLen = 64;
    const detail::source_spec spec = s8_spec(2, false);
    detail::source_rows from, to;
    from.grow(spec, 2, kLen);
    to.grow(spec, 3, kLen);
    const std::vector<int8_t> in = s8_values(kLen, 13);
    detail::copy_s8_row(spec, in.data(), kLen, from, 1);
    from.row_scales(spec, 1)[0] = 0.5f;
    from.row_scales(spec, 1)[1] = 0.25f;

    detail::copy_row(spec, from, 1, to, 2, kLen);
    EXPECT_EQ(std::memcmp(to.row(spec, 2, kLen), in.data(), kLen), 0);
    EXPECT_EQ(to.row_scales(spec, 2)[0], 0.5f);
    EXPECT_EQ(to.row_scales(spec, 2)[1], 0.25f);
    EXPECT_TRUE(all_zero(to.rows.data(), 2 * kLen));

    const detail::source_spec bf16 = bf16_spec();
    detail::source_rows bf16_from, bf16_to;
    bf16_from.grow(bf16, 2, kLen);
    bf16_to.grow(bf16, 2, kLen);
    const std::vector<uint16_t> values = bf16_values(kLen, 17);
    detail::copy_in(bf16, values.data(), kLen, bf16_from, 1);
    EXPECT_EQ(std::memcmp(bf16_from.row(bf16, 1, kLen), values.data(),
                      values.size() * sizeof(uint16_t)),
            0);
    detail::copy_row(bf16, bf16_from, 1, bf16_to, 0, kLen);
    EXPECT_EQ(std::memcmp(bf16_to.row(bf16, 0, kLen), values.data(),
                      values.size() * sizeof(uint16_t)),
            0);
    EXPECT_TRUE(bf16_to.scales.empty());
}

// A view starts at its row's elements and, for s8 rows, its row's scales.
TEST(Nr32Source, AViewStartsAtItsRow) {
    constexpr int kLen = 64;
    const detail::source_spec spec = s8_spec(3, true);
    detail::source_rows s8_rows;
    s8_rows.grow(spec, 4, kLen);
    const nfp::nr32::tile_source t = s8_rows.view(spec, 2, kLen);
    EXPECT_EQ(
            t.rows, static_cast<const void *>(s8_rows.rows.data() + 2 * kLen));
    EXPECT_EQ(t.lda, kLen);
    EXPECT_EQ(t.scales, s8_rows.scales.data() + 2 * 3);

    const detail::source_spec bf16 = bf16_spec();
    detail::source_rows bf16_rows;
    bf16_rows.grow(bf16, 4, kLen);
    const nfp::nr32::tile_source b = bf16_rows.view(bf16, 2, kLen);
    EXPECT_EQ(b.rows,
            static_cast<const void *>(bf16_rows.rows.data() + 2 * kLen * 2));
    EXPECT_EQ(b.lda, kLen);
    EXPECT_EQ(b.scales, nullptr);
}

// The scratch has room for its rows and the bytes the kernels read past the
// last, scales only for s8 rows, and never shrinks.
TEST(Nr32Source, TheScratchHasRoomPastTheLastRow) {
    constexpr int kLen = 40;
    const detail::source_spec spec = s8_spec(4, false);
    detail::source_rows s8_rows;
    s8_rows.grow(spec, 5, kLen);
    EXPECT_GE(s8_rows.rows.size(), 5u * kLen + detail::kRowSlack);
    EXPECT_GE(s8_rows.scales.size(), 5u * 4);
    const size_t rows = s8_rows.rows.size();
    s8_rows.grow(spec, 1, kLen);
    EXPECT_EQ(s8_rows.rows.size(), rows);
    EXPECT_GE(s8_rows.bytes(), rows + 5u * 4 * sizeof(float));

    const detail::source_spec bf16 = bf16_spec();
    detail::source_rows bf16_rows;
    bf16_rows.grow(bf16, 5, kLen);
    EXPECT_GE(bf16_rows.rows.size(), 5u * kLen * 2 + detail::kRowSlack);
    EXPECT_TRUE(bf16_rows.scales.empty());
}

} // namespace
