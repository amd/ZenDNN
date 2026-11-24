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
 * @file test_nr32.cpp
 * @brief Tests for ALGO 4's NR32 packs and tiles.
 *
 * The pack is checked against the weight it came from, element by element,
 * with its padding, sums and scales.  Tiles are checked against a scalar
 * reference for every weight type and epilogue, on shapes whose columns end
 * inside a block, whose up columns start inside a kernel vector, and whose K
 * ends inside a dot-product group, at every row count up to a few kernel
 * calls, with each bias type.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "common/float16.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/nr32_pack.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/nr32_tile.hpp"

namespace {

namespace nr32 = zendnnl::lowoha::matmul::ntile_flat_parallel::nr32;
using data_type_t = zendnnl::common::data_type_t;
using status_t = zendnnl::error_handling::status_t;
using zendnnl::common::float16_t;

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

float from_bf16(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

const char *type_name(data_type_t dt) {
    switch (dt) {
        case data_type_t::bf16: return "bf16";
        case data_type_t::f16: return "f16";
        case data_type_t::s8: return "s8";
        case data_type_t::s4: return "s4";
        default: return "?";
    }
}

/// One weight of `E` experts in the caller's layout, with what a pack of it
/// should hold.
struct weight_set {
    nr32::weight_desc desc;
    int E = 0;
    /// Per expert: the raw bytes, and the element values as numbers.
    std::vector<std::vector<uint8_t>> raw;
    std::vector<std::vector<float>> value; // [N][K]
    /// s8 and s4: [G][N] per expert, in `scale_dt`.
    std::vector<std::vector<float>> scale;
    std::vector<std::vector<uint16_t>> scale_bf16;
    data_type_t scale_dt = data_type_t::f32;

    int groups() const {
        return desc.group_size > 0 ? desc.K / desc.group_size : 1;
    }

    void build(data_type_t wei, int K, int N, bool trans, int ld_pad,
            int group_size, int experts, uint32_t seed,
            data_type_t sdt = data_type_t::f32) {
        desc = {};
        desc.wei = wei;
        desc.K = K;
        desc.N = N;
        desc.trans = trans;
        desc.ldb = (trans ? K : N) + ld_pad;
        desc.group_size = wei == data_type_t::s4 || wei == data_type_t::s8
                ? group_size
                : 0;
        E = experts;
        scale_dt = sdt;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> unit(-1.f, 1.f);
        std::uniform_int_distribution<int> s8v(-127, 127);
        std::uniform_int_distribution<int> s4v(-8, 7);
        std::uniform_real_distribution<float> sc(0.002f, 0.02f);
        const size_t rows = trans ? N : K;
        const size_t elems = rows * desc.ldb;
        raw.assign(E, {});
        value.assign(E, std::vector<float>(static_cast<size_t>(N) * K));
        scale.assign(E, {});
        scale_bf16.assign(E, {});
        const float a = 1.7f / std::sqrt(static_cast<float>(K));
        for (int e = 0; e < E; ++e) {
            auto &bytes = raw[e];
            const auto index = [&](int k, int n) {
                return trans ? static_cast<size_t>(n) * desc.ldb + k
                             : static_cast<size_t>(k) * desc.ldb + n;
            };
            if (wei == data_type_t::s4) {
                bytes.assign((elems + 1) / 2, 0);
            } else {
                bytes.assign(elems * (wei == data_type_t::s8 ? 1 : 2), 0);
            }
            for (int n = 0; n < N; ++n) {
                for (int k = 0; k < K; ++k) {
                    const size_t i = index(k, n);
                    float v = 0.f;
                    if (wei == data_type_t::bf16) {
                        const uint16_t h = to_bf16(a * unit(rng));
                        std::memcpy(bytes.data() + 2 * i, &h, 2);
                        v = from_bf16(h);
                    } else if (wei == data_type_t::f16) {
                        const uint16_t h
                                = float16_t::f32_to_f16_val(a * unit(rng));
                        std::memcpy(bytes.data() + 2 * i, &h, 2);
                        v = float16_t::f16_to_f32_val(h);
                    } else if (wei == data_type_t::s8) {
                        const int q = s8v(rng);
                        bytes[i] = static_cast<uint8_t>(static_cast<int8_t>(q));
                        v = static_cast<float>(q);
                    } else {
                        const int q = s4v(rng);
                        const uint8_t nib = static_cast<uint8_t>(q) & 0x0Fu;
                        bytes[i >> 1] |= (i & 1u)
                                ? static_cast<uint8_t>(nib << 4)
                                : nib;
                        v = static_cast<float>(q);
                    }
                    value[e][static_cast<size_t>(n) * K + k] = v;
                }
            }
            if (wei == data_type_t::s8 || wei == data_type_t::s4) {
                scale[e].resize(static_cast<size_t>(groups()) * N);
                for (auto &s : scale[e]) {
                    s = sc(rng);
                    if (sdt == data_type_t::bf16) {
                        scale_bf16[e].push_back(to_bf16(s));
                        s = from_bf16(scale_bf16[e].back());
                    }
                }
            }
        }
    }

    status_t pack(nr32::packed_weights &pw) const {
        std::vector<const void *> w, s;
        for (int e = 0; e < E; ++e) {
            w.push_back(raw[e].data());
            if (desc.wei == data_type_t::s8 || desc.wei == data_type_t::s4) {
                s.push_back(scale_dt == data_type_t::bf16
                                ? static_cast<const void *>(
                                          scale_bf16[e].data())
                                : static_cast<const void *>(scale[e].data()));
            }
        }
        return nr32::prepack(desc, w, s, scale_dt, 4, pw);
    }
};

// ---------------------------------------------------------------------------
// Pack
// ---------------------------------------------------------------------------

TEST(Nr32Pack, RoundTripsEveryTypeLayoutAndPadding) {
    struct shape_t {
        int K, N, group;
    };
    uint32_t seed = 0x6100u;
    for (const data_type_t dt : {data_type_t::bf16, data_type_t::f16,
                 data_type_t::s8, data_type_t::s4}) {
        const bool s4 = dt == data_type_t::s4;
        std::vector<shape_t> shapes = s4
                ? std::vector<shape_t> {{64, 32, 32}, {136, 70, 8},
                          {256, 96, 64}}
                : std::vector<shape_t> {
                          {1, 32, 0}, {63, 40, 0}, {130, 70, 0}, {256, 96, 0}};
        if (dt == data_type_t::s8) {
            shapes.push_back({128, 70, 32});
            shapes.push_back({96, 40, 4});
        }
        for (const shape_t sh : shapes) {
            // A padding of 0 or 2 keeps an s4 row on a byte, as the packs'
            // row fast paths read it; 3 does not.
            for (const auto &[trans, ld_pad] :
                    {std::pair {true, 0}, std::pair {true, 2},
                            std::pair {true, 3}, std::pair {false, 3}}) {
                for (const data_type_t sdt :
                        {data_type_t::f32, data_type_t::bf16}) {
                    SCOPED_TRACE(::testing::Message()
                            << type_name(dt) << " K=" << sh.K << " N=" << sh.N
                            << " trans=" << trans << " ld_pad=" << ld_pad
                            << " scales="
                            << (sdt == data_type_t::bf16 ? "bf16" : "f32"));
                    weight_set ws;
                    ws.build(dt, sh.K, sh.N, trans, ld_pad, sh.group, 3, seed++,
                            sdt);
                    nr32::packed_weights pw;
                    ASSERT_EQ(ws.pack(pw), status_t::success);
                    const int n_pad = (sh.N + 31) / 32 * 32;
                    ASSERT_EQ(pw.n_pad, n_pad);
                    ASSERT_EQ(pw.groups, ws.groups());
                    ASSERT_EQ(pw.block_bytes, nr32::block_bytes(ws.desc));
                    for (int e = 0; e < ws.E; ++e) {
                        // Every weight comes back in place.
                        const size_t row_bytes = s4
                                ? static_cast<size_t>(sh.K) / 2
                                : static_cast<size_t>(sh.K)
                                        * (dt == data_type_t::s8 ? 1 : 2);
                        std::vector<uint8_t> back(row_bytes * sh.N, 0);
                        nr32::unpack(pw, e, back.data(), sh.K);
                        for (int n = 0; n < sh.N; ++n) {
                            for (int k = 0; k < sh.K; ++k) {
                                const size_t i
                                        = static_cast<size_t>(n) * sh.K + k;
                                float got = 0.f;
                                if (dt == data_type_t::bf16) {
                                    uint16_t h;
                                    std::memcpy(&h, back.data() + 2 * i, 2);
                                    got = from_bf16(h);
                                } else if (dt == data_type_t::f16) {
                                    uint16_t h;
                                    std::memcpy(&h, back.data() + 2 * i, 2);
                                    got = float16_t::f16_to_f32_val(h);
                                } else if (dt == data_type_t::s8) {
                                    got = static_cast<int8_t>(back[i]);
                                } else {
                                    const uint8_t b = back[i >> 1];
                                    const int nib
                                            = (i & 1u) ? b >> 4 : b & 0x0F;
                                    got = static_cast<float>(
                                            nib >= 8 ? nib - 16 : nib);
                                }
                                ASSERT_EQ(got, ws.value[e][i])
                                        << "expert " << e << " n=" << n
                                        << " k=" << k;
                            }
                        }
                        // Padded columns hold zero weights: zero bytes, and
                        // for s4 nibbles that read as zero.
                        const uint8_t pad = s4 ? 0x88 : 0x00;
                        const uint8_t *blk = pw.blocks(e);
                        const int rows = static_cast<int>(pw.block_bytes
                                / (32 * (dt == data_type_t::f16 ? 2 : 4)));
                        const int per_col = dt == data_type_t::f16 ? 2 : 4;
                        for (int n = sh.N; n < n_pad; ++n) {
                            const uint8_t *b = blk + (n / 32) * pw.block_bytes;
                            for (int r = 0; r < rows; ++r) {
                                for (int q = 0; q < per_col; ++q) {
                                    ASSERT_EQ(
                                            b[(r * 32 + n % 32) * per_col + q],
                                            pad)
                                            << "padded column " << n;
                                }
                            }
                        }
                        // Sums and scales by group, in column order, zero
                        // past N: s8 sums as they are, s4 times -128.
                        if (dt == data_type_t::s8 || s4) {
                            const int gs = sh.K / pw.groups;
                            const int32_t times = s4 ? -128 : 1;
                            for (int g = 0; g < pw.groups; ++g) {
                                for (int n = 0; n < n_pad; ++n) {
                                    int32_t sum = 0;
                                    for (int k = g * gs;
                                            n < sh.N && k < (g + 1) * gs; ++k)
                                        sum += static_cast<int32_t>(
                                                ws.value[e]
                                                        [static_cast<size_t>(n)
                                                                        * sh.K
                                                                + k]);
                                    const size_t at
                                            = static_cast<size_t>(g) * n_pad
                                            + n;
                                    EXPECT_EQ((*pw.experts[e].comp)[at],
                                            times * sum);
                                    EXPECT_EQ(pw.experts[e].scale[at],
                                            n < sh.N
                                                    ? ws.scale[e]
                                                              [static_cast<
                                                                       size_t>(
                                                                       g) * sh.N
                                                                      + n]
                                                    : 0.f);
                                }
                            }
                        } else {
                            EXPECT_EQ(pw.experts[e].comp, nullptr);
                            EXPECT_TRUE(pw.experts[e].scale.empty());
                        }
                    }
                }
            }
        }
    }
}

TEST(Nr32Pack, RefusesWhatItCannotPack) {
    nr32::weight_desc d;
    d.wei = data_type_t::s4;
    d.K = 60;
    d.N = 32;
    d.ldb = 60;
    d.group_size = 4;
    EXPECT_FALSE(nr32::packable(d)) << "s4 K not a multiple of 8";
    d.K = 64;
    d.ldb = 64;
    d.group_size = 0;
    EXPECT_FALSE(nr32::packable(d)) << "s4 without groups";
    d.group_size = 24;
    EXPECT_FALSE(nr32::packable(d)) << "group not dividing K";
    d.group_size = 32;
    EXPECT_TRUE(nr32::packable(d));
    d.ldb = 63;
    EXPECT_FALSE(nr32::packable(d)) << "leading dimension shorter than a row";

    d = {};
    d.wei = data_type_t::s8;
    d.K = 64;
    d.N = 32;
    d.ldb = 64;
    EXPECT_TRUE(nr32::packable(d)) << "s8 with one scale per column";
    d.group_size = 6;
    EXPECT_FALSE(nr32::packable(d)) << "s8 group not a multiple of 4";
    d.group_size = 24;
    EXPECT_FALSE(nr32::packable(d)) << "s8 group not dividing K";
    d.group_size = 16;
    EXPECT_TRUE(nr32::packable(d));

    d = {};
    d.wei = data_type_t::bf16;
    d.K = 64;
    d.N = 32;
    d.ldb = 64;
    EXPECT_TRUE(nr32::packable(d));
    d.group_size = 32;
    EXPECT_FALSE(nr32::packable(d)) << "groups on a 16-bit weight";
    d.group_size = 0;
    d.wei = data_type_t::u8;
    EXPECT_FALSE(nr32::packable(d)) << "unsupported type";

    // Missing or mistyped scales.
    weight_set ws;
    ws.build(data_type_t::s8, 64, 32, true, 0, 0, 2, 0x6200u);
    nr32::packed_weights pw;
    std::vector<const void *> w {ws.raw[0].data(), ws.raw[1].data()};
    EXPECT_EQ(nr32::prepack(ws.desc, w, {ws.scale[0].data()}, data_type_t::f32,
                      1, pw),
            status_t::unimplemented);
    EXPECT_EQ(
            nr32::prepack(ws.desc, w, {ws.scale[0].data(), ws.scale[1].data()},
                    data_type_t::s8, 1, pw),
            status_t::unimplemented);
    EXPECT_EQ(nr32::prepack(ws.desc, {ws.raw[0].data(), nullptr},
                      {ws.scale[0].data(), ws.scale[1].data()},
                      data_type_t::f32, 1, pw),
            status_t::unimplemented);
    EXPECT_TRUE(pw.experts.empty());
}

// A quantized pack made without scales and given them later holds what a
// pack made with them holds, on the first pack's blocks.
TEST(Nr32Pack, ScalesJoinAPackMadeWithoutThem) {
    struct case_t {
        data_type_t wei, scale_dt;
        int K, N, group;
    };
    const case_t cases[] = {{data_type_t::s8, data_type_t::f32, 128, 70, 0},
            {data_type_t::s8, data_type_t::bf16, 128, 96, 32},
            {data_type_t::s4, data_type_t::f32, 128, 70, 32},
            {data_type_t::s4, data_type_t::bf16, 256, 64, 128}};
    uint32_t seed = 0x6300u;
    for (const case_t &c : cases) {
        SCOPED_TRACE(::testing::Message()
                << "wei=" << static_cast<int>(c.wei) << " N=" << c.N
                << " group=" << c.group);
        weight_set ws;
        ws.build(c.wei, c.K, c.N, true, 0, c.group, 2, seed++, c.scale_dt);
        nr32::packed_weights scaled, bare;
        ASSERT_EQ(ws.pack(scaled), status_t::success);
        ASSERT_EQ(nr32::prepack(ws.desc, {ws.raw[0].data(), ws.raw[1].data()},
                          {}, data_type_t::none, 1, bare),
                status_t::success);
        const size_t bytes = bare.block_bytes * (bare.n_pad / nr32::kNr);
        for (int e = 0; e < 2; ++e) {
            EXPECT_TRUE(bare.experts[e].scale.empty());
            const void *s = c.scale_dt == data_type_t::bf16
                    ? static_cast<const void *>(ws.scale_bf16[e].data())
                    : static_cast<const void *>(ws.scale[e].data());
            nr32::packed_expert joined;
            ASSERT_EQ(nr32::with_scales(
                              bare, bare.experts[e], s, c.scale_dt, joined),
                    status_t::success);
            EXPECT_EQ(joined.blocks.get(), bare.experts[e].blocks.get());
            EXPECT_EQ(std::memcmp(joined.blocks.get(), scaled.blocks(e), bytes),
                    0);
            EXPECT_EQ(joined.comp, bare.experts[e].comp)
                    << "a scaled pack shares its weight's sums";
            EXPECT_EQ(*joined.comp, *scaled.experts[e].comp);
            EXPECT_EQ(joined.scale, scaled.experts[e].scale);
        }
        nr32::packed_expert refused;
        EXPECT_EQ(nr32::with_scales(bare, bare.experts[0], ws.scale[0].data(),
                          data_type_t::s8, refused),
                status_t::unimplemented);
    }
}

// ---------------------------------------------------------------------------
// Tiles
// ---------------------------------------------------------------------------

float silu_ref(float x) {
    return x / (1.f + std::exp(-x));
}

float gelu_ref(float x) {
    return 0.5f * x * (1.f + std::erf(x / std::sqrt(2.f)));
}

float swiglu_oai_ref(float g, float u) {
    g = std::clamp(g, -7.f, 7.f);
    u = std::clamp(u, -7.f, 7.f);
    return (1.f + u) * g / (1.f + std::exp(-1.702f * g));
}

const char *epilogue_name(nr32::epilogue ep) {
    switch (ep) {
        case nr32::epilogue::plain: return "plain";
        case nr32::epilogue::swiglu_pairs: return "swiglu_pairs";
        case nr32::epilogue::silu_split: return "silu_split";
        default: return "gelu_split";
    }
}

/// Source rows for a tile of `ws`: values, and their bytes at `lda`, with
/// the slack the kernels may read past the last row.
struct rows_t {
    int M = 0, lda = 0;
    std::vector<float> value; // [M][K]
    std::vector<uint8_t> bytes;
    /// Quantized: [M][groups], one scale per row or per (row, weight group).
    std::vector<float> scale;
    int groups = 1;
};

rows_t make_rows(const weight_set &ws, int M, uint32_t seed, int groups = 1) {
    const int K = ws.desc.K;
    const data_type_t dt = ws.desc.wei;
    rows_t r;
    r.M = M;
    r.lda = K + 3;
    r.groups = groups;
    r.value.resize(static_cast<size_t>(M) * K);
    const size_t es
            = (dt == data_type_t::bf16 || dt == data_type_t::f16) ? 2 : 1;
    r.bytes.assign(
            static_cast<size_t>(M) * r.lda * es + nr32::kRowReadSlack, 0x5a);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-1.f, 1.f);
    std::uniform_int_distribution<int> s8v(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.02f);
    for (int m = 0; m < M; ++m) {
        for (int g = 0; es == 1 && g < groups; ++g)
            r.scale.push_back(sc(rng));
        for (int k = 0; k < K; ++k) {
            const size_t at = static_cast<size_t>(m) * r.lda + k;
            float v;
            if (dt == data_type_t::bf16) {
                const uint16_t h = to_bf16(unit(rng));
                std::memcpy(r.bytes.data() + 2 * at, &h, 2);
                v = from_bf16(h);
            } else if (dt == data_type_t::f16) {
                const uint16_t h = float16_t::f32_to_f16_val(unit(rng));
                std::memcpy(r.bytes.data() + 2 * at, &h, 2);
                v = float16_t::f16_to_f32_val(h);
            } else {
                const int q = s8v(rng);
                r.bytes[at] = static_cast<uint8_t>(static_cast<int8_t>(q));
                v = static_cast<float>(q);
            }
            r.value[static_cast<size_t>(m) * K + k] = v;
        }
    }
    return r;
}

/// Row m's dot product with weight column n of expert e, in its scales.
double dot_ref(const weight_set &ws, const rows_t &r, int e, int m, int n) {
    const int K = ws.desc.K;
    const float *a = r.value.data() + static_cast<size_t>(m) * K;
    const float *w = ws.value[e].data() + static_cast<size_t>(n) * K;
    if (ws.desc.wei != data_type_t::s8 && ws.desc.wei != data_type_t::s4) {
        double s = 0.0;
        for (int k = 0; k < K; ++k)
            s += static_cast<double>(a[k]) * w[k];
        return s;
    }
    const int G = ws.groups();
    const int gs = K / G;
    double total = 0.0;
    for (int g = 0; g < G; ++g) {
        double s = 0.0;
        for (int k = g * gs; k < (g + 1) * gs; ++k)
            s += static_cast<double>(a[k]) * w[k];
        s *= ws.scale[e][static_cast<size_t>(g) * ws.desc.N + n];
        if (r.groups > 1) s *= r.scale[static_cast<size_t>(m) * r.groups + g];
        total += s;
    }
    return r.groups > 1 ? total : total * r.scale[m];
}

/// A bias row of `N` values in `kind`, with the values it holds.
struct bias_t {
    nr32::bias_kind kind = nr32::bias_kind::none;
    std::vector<uint8_t> bytes;
    std::vector<float> value;
};

bias_t make_bias(nr32::bias_kind kind, int N, uint32_t seed) {
    bias_t b;
    b.kind = kind;
    if (kind == nr32::bias_kind::none) return b;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-0.5f, 0.5f);
    // Exactly N values: a read past them is out of the row.
    b.bytes.resize(
            static_cast<size_t>(N) * (kind == nr32::bias_kind::f32 ? 4 : 2));
    for (int n = 0; n < N; ++n) {
        float v = unit(rng);
        if (kind == nr32::bias_kind::f32) {
            std::memcpy(b.bytes.data() + 4 * n, &v, 4);
        } else {
            const uint16_t h = kind == nr32::bias_kind::bf16
                    ? to_bf16(v)
                    : float16_t::f32_to_f16_val(v);
            std::memcpy(b.bytes.data() + 2 * n, &h, 2);
            v = kind == nr32::bias_kind::bf16 ? from_bf16(h)
                                              : float16_t::f16_to_f32_val(h);
        }
        b.value.push_back(v);
    }
    return b;
}

/// Runs plan columns [0, plan.cols) of expert `e` in two tiles split at a
/// step and checks every output against the reference, and that no column
/// past the output's end is written.
void check_tile(const weight_set &ws, const nr32::packed_weights &pw,
        const nr32::tile_plan &plan, int e, const rows_t &r, const bias_t &b,
        bool src_u8) {
    const data_type_t dt = ws.desc.wei;
    const int M = r.M;
    // Gated epilogues: gate columns [0, N / 2), up columns [N / 2, N).
    const int I = ws.desc.N / 2;
    const int out_cols
            = plan.ep == nr32::epilogue::plain ? ws.desc.N : ws.desc.N / 2;
    EXPECT_EQ(plan.out_cols, out_cols);
    if (plan.ep == nr32::epilogue::silu_split
            || plan.ep == nr32::epilogue::gelu_split) {
        EXPECT_EQ(plan.up, I);
    }
    const int ldc = out_cols + 5;
    const uint16_t poison = 0x7FC1;
    std::vector<uint16_t> dst(static_cast<size_t>(M) * ldc, poison);
    std::vector<uint8_t> src = r.bytes;
    if (src_u8) {
        for (auto &v : src)
            v ^= 0x80u;
    }
    const int split = plan.cols / plan.step / 2 * plan.step;
    const void *bias
            = b.kind == nr32::bias_kind::none ? nullptr : b.bytes.data();
    const float *scale = r.scale.empty() ? nullptr : r.scale.data();
    if (split > 0) {
        nr32::run_tile(plan, pw, pw.experts[e], M, src.data(), r.lda, scale,
                bias, b.kind, dst.data(), ldc, 0, split);
    }
    nr32::run_tile(plan, pw, pw.experts[e], M, src.data(), r.lda, scale, bias,
            b.kind, dst.data(), ldc, split, plan.cols - split);

    const auto biased = [&](int m, int n) {
        return static_cast<float>(dot_ref(ws, r, e, m, n))
                + (b.value.empty() ? 0.f : b.value[n]);
    };
    const bool f16 = dt == data_type_t::f16;
    // FP16 accumulates within K blocks in FP16; every type rounds its output.
    const double rtol = f16 ? 1.5e-2 : 1.2e-2;
    const double atol = f16 ? 2e-2 : 1e-2;
    int bad = 0;
    for (int m = 0; m < M && bad < 5; ++m) {
        for (int j = 0; j < ldc; ++j) {
            const uint16_t got_bits = dst[static_cast<size_t>(m) * ldc + j];
            if (j >= out_cols) {
                if (got_bits != poison) {
                    ADD_FAILURE() << "row " << m << " column " << j
                                  << " past the output was written";
                    ++bad;
                }
                continue;
            }
            float want;
            switch (plan.ep) {
                case nr32::epilogue::plain: want = biased(m, j); break;
                case nr32::epilogue::swiglu_pairs:
                    want = swiglu_oai_ref(
                            biased(m, 2 * j), biased(m, 2 * j + 1));
                    break;
                case nr32::epilogue::silu_split:
                    want = silu_ref(biased(m, j)) * biased(m, I + j);
                    break;
                default:
                    want = gelu_ref(biased(m, j)) * biased(m, I + j);
                    break;
            }
            const float got = f16 ? float16_t::f16_to_f32_val(got_bits)
                                  : from_bf16(got_bits);
            if (!(std::fabs(got - want) <= atol + rtol * std::fabs(want))) {
                ADD_FAILURE() << "row " << m << " column " << j << ": got "
                              << got << ", want " << want;
                ++bad;
            }
        }
    }
}

TEST(Nr32Tile, MatchesTheReferenceForEveryTypeAndEpilogue) {
    struct shape_t {
        int K, N, group;
    };
    const nr32::epilogue eps[]
            = {nr32::epilogue::plain, nr32::epilogue::swiglu_pairs,
                    nr32::epilogue::silu_split, nr32::epilogue::gelu_split};
    uint32_t seed = 0x6300u;
    int served = 0;
    for (const data_type_t dt : {data_type_t::bf16, data_type_t::f16,
                 data_type_t::s8, data_type_t::s4}) {
        if (!nr32::isa_supported(dt)) continue;
        const bool s4 = dt == data_type_t::s4;
        // N: whole blocks, a block and a half, columns ending inside a
        // vector; as 2 * I: up columns on a vector, inside one (I % 16), and
        // inside an f16 vector (I % 32 == 16).
        const std::vector<shape_t> shapes = s4
                ? std::vector<shape_t> {{64, 64, 32}, {136, 96, 8},
                          {256, 80, 64}, {128, 74, 32}, {64, 200, 16}}
                : std::vector<shape_t> {{64, 64, 0}, {63, 96, 0}, {130, 80, 0},
                          {256, 74, 0}, {32, 200, 0}, {520, 128, 0}};
        for (const shape_t sh : shapes) {
            weight_set ws;
            ws.build(dt, sh.K, sh.N, true, 2, sh.group, 2, seed++);
            nr32::packed_weights pw;
            ASSERT_EQ(ws.pack(pw), status_t::success);
            for (const nr32::epilogue ep : eps) {
                for (const bool src_u8 : {false, true}) {
                    if (src_u8 && dt != data_type_t::s8) continue;
                    nr32::tile_plan plan;
                    ASSERT_TRUE(nr32::make_plan(pw, ep, src_u8, plan));
                    // Every kernel row count, and tiles of two and three
                    // kernel calls.
                    const int max_mr = std::max(plan.max_mr, plan.wide_max_mr);
                    for (int M = 1; M <= 2 * max_mr + 1; ++M) {
                        const rows_t r = make_rows(ws, M, seed++);
                        for (const nr32::bias_kind bk :
                                {nr32::bias_kind::none, nr32::bias_kind::bf16,
                                        nr32::bias_kind::f16,
                                        nr32::bias_kind::f32}) {
                            SCOPED_TRACE(::testing::Message()
                                    << type_name(dt) << " " << epilogue_name(ep)
                                    << " K=" << sh.K << " N=" << sh.N << " M="
                                    << M << " bias=" << static_cast<int>(bk)
                                    << " u8=" << src_u8);
                            const bias_t b = make_bias(bk, sh.N, seed++);
                            check_tile(ws, pw, plan, 1, r, b, src_u8);
                            ++served;
                        }
                    }
                }
            }
        }
    }
    if (served == 0) GTEST_SKIP() << "no NR32 kernel runs on this CPU";
}

// s8 and s4 weights with per-group scales, against one source scale per row
// and one per (row, group).
TEST(Nr32Tile, MatchesTheReferenceWithScaleGroups) {
    struct shape_t {
        int K, N, group;
    };
    const nr32::epilogue eps[]
            = {nr32::epilogue::plain, nr32::epilogue::swiglu_pairs,
                    nr32::epilogue::silu_split, nr32::epilogue::gelu_split};
    uint32_t seed = 0x6600u;
    int served = 0;
    for (const data_type_t dt : {data_type_t::s8, data_type_t::s4}) {
        if (!nr32::isa_supported(dt)) continue;
        // As 2 * I: up columns on a block (wide kernels), inside a vector,
        // and inside a vector of a block and a half.
        const std::vector<shape_t> shapes = dt == data_type_t::s8
                ? std::vector<shape_t> {{128, 64, 32}, {96, 80, 4},
                          {256, 74, 64}, {64, 200, 16}}
                : std::vector<shape_t> {
                          {128, 64, 32}, {256, 80, 64}, {64, 200, 16}};
        for (const shape_t sh : shapes) {
            weight_set ws;
            ws.build(dt, sh.K, sh.N, true, 2, sh.group, 2, seed++);
            nr32::packed_weights pw;
            ASSERT_EQ(ws.pack(pw), status_t::success);
            ASSERT_GT(pw.groups, 1);
            for (const nr32::epilogue ep : eps) {
                for (const bool src_groups : {false, true}) {
                    for (const bool src_u8 : {false, true}) {
                        if (src_u8 && dt != data_type_t::s8) continue;
                        nr32::tile_plan plan;
                        ASSERT_TRUE(nr32::make_plan(
                                pw, ep, src_u8, plan, src_groups));
                        EXPECT_EQ(plan.src_groups, src_groups);
                        const int max_mr
                                = std::max(plan.max_mr, plan.wide_max_mr);
                        for (int M = 1; M <= 2 * max_mr + 1; ++M) {
                            const rows_t r = make_rows(
                                    ws, M, seed++, src_groups ? pw.groups : 1);
                            for (const nr32::bias_kind bk :
                                    {nr32::bias_kind::none,
                                            nr32::bias_kind::bf16}) {
                                SCOPED_TRACE(::testing::Message()
                                        << type_name(dt) << " "
                                        << epilogue_name(ep) << " K=" << sh.K
                                        << " N=" << sh.N
                                        << " group=" << sh.group << " M=" << M
                                        << " src_groups=" << src_groups
                                        << " u8=" << src_u8
                                        << " bias=" << static_cast<int>(bk));
                                const bias_t b = make_bias(bk, sh.N, seed++);
                                check_tile(ws, pw, plan, 1, r, b, src_u8);
                                ++served;
                            }
                        }
                    }
                }
            }
        }
    }
    if (served == 0) GTEST_SKIP() << "no NR32 kernel runs on this CPU";
}

// The two source forms of the s8 kernels write the same bits.
TEST(Nr32Tile, U8SourceKernelsWriteWhatS8SourceKernelsWrite) {
    if (!nr32::isa_supported(data_type_t::s8)) GTEST_SKIP();
    struct form_t {
        nr32::epilogue ep;
        int K, group;
        bool src_groups;
    };
    for (const form_t f : {form_t {nr32::epilogue::plain, 130, 0, false},
                 form_t {nr32::epilogue::silu_split, 130, 0, false},
                 form_t {nr32::epilogue::plain, 128, 32, false},
                 form_t {nr32::epilogue::silu_split, 128, 32, true}}) {
        SCOPED_TRACE(::testing::Message()
                << epilogue_name(f.ep) << " group=" << f.group
                << " src_groups=" << f.src_groups);
        const nr32::epilogue ep = f.ep;
        weight_set ws;
        ws.build(data_type_t::s8, f.K, 74, true, 0, f.group, 1, 0x6400u);
        nr32::packed_weights pw;
        ASSERT_EQ(ws.pack(pw), status_t::success);
        const rows_t r
                = make_rows(ws, 13, 0x6401u, f.src_groups ? pw.groups : 1);
        std::vector<uint16_t> out[2];
        for (const bool u8 : {false, true}) {
            nr32::tile_plan plan;
            ASSERT_TRUE(nr32::make_plan(pw, ep, u8, plan, f.src_groups));
            std::vector<uint8_t> src = r.bytes;
            if (u8) {
                for (auto &v : src)
                    v ^= 0x80u;
            }
            out[u8].assign(static_cast<size_t>(r.M) * plan.out_cols, 0);
            nr32::run_tile(plan, pw, pw.experts[0], r.M, src.data(), r.lda,
                    r.scale.data(), nullptr, nr32::bias_kind::none,
                    out[u8].data(), plan.out_cols, 0, plan.cols);
        }
        EXPECT_EQ(out[0], out[1]) << epilogue_name(ep);
    }
}

// Scales whose product overflows, or is NaN, meet a zero dot product: the
// output is 0, as the custom kernels write, not 0 x Inf, whichever row of the
// tile the scales are on.  A row of ordinary scales in the same tile writes
// what it writes alone.
TEST(Nr32Tile, AZeroDotProductStaysZeroWhereTheScalesOverflow) {
    int served = 0;
    for (const data_type_t dt : {data_type_t::s8, data_type_t::s4}) {
        if (!nr32::isa_supported(dt)) continue;
        SCOPED_TRACE(type_name(dt));
        constexpr int K = 64, N = 32, group = 32;
        weight_set ws;
        ws.build(dt, K, N, true, 0, group, 1, 0x6700u);
        // Column 0's weights are zero.
        for (int k = 0; k < K; ++k) {
            if (dt == data_type_t::s4) {
                ws.raw[0][k >> 1] &= (k & 1) ? 0x0Fu : 0xF0u;
            } else {
                ws.raw[0][k] = 0;
            }
            ws.value[0][k] = 0.f;
        }
        for (int g = 0; g < K / group; ++g) {
            for (int n = 0; n < N; ++n) {
                ws.scale[0][static_cast<size_t>(g) * N + n]
                        = std::ldexp(1.f, n == 0 ? 20 : -10);
            }
        }
        nr32::packed_weights pw;
        ASSERT_EQ(ws.pack(pw), status_t::success);
        nr32::tile_plan plan;
        ASSERT_TRUE(nr32::make_plan(pw, nr32::epilogue::plain, false, plan,
                /*src_groups=*/true));
        rows_t r = make_rows(ws, 2, 0x6701u, pw.groups);
        for (int g = 0; g < pw.groups; ++g)
            r.scale[g] = std::ldexp(1.f, 110);
        const int ldc = plan.out_cols;
        std::vector<uint16_t> dst(static_cast<size_t>(2) * ldc, 0x7FC1);
        nr32::run_tile(plan, pw, pw.experts[0], 2, r.bytes.data(), r.lda,
                r.scale.data(), nullptr, nr32::bias_kind::none, dst.data(), ldc,
                0, plan.cols);
        EXPECT_EQ(from_bf16(dst[0]), 0.f);
        for (int n = 1; n < N; ++n) {
            EXPECT_TRUE(std::isfinite(from_bf16(dst[n]))) << "column " << n;
        }
        std::vector<uint16_t> alone(static_cast<size_t>(ldc), 0x7FC1);
        nr32::run_tile(plan, pw, pw.experts[0], 1, r.bytes.data() + r.lda,
                r.lda, r.scale.data() + pw.groups, nullptr,
                nr32::bias_kind::none, alone.data(), ldc, 0, plan.cols);
        EXPECT_TRUE(std::equal(alone.begin(), alone.end(), dst.begin() + ldc))
                << "the second row's outputs differ from its own tile's";
        // The same with the overflowing row second, and with NaN scales.
        for (const float big : {std::ldexp(1.f, 110),
                     std::numeric_limits<float>::quiet_NaN()}) {
            rows_t r2 = make_rows(ws, 2, 0x6704u, pw.groups);
            for (int g = 0; g < pw.groups; ++g)
                r2.scale[pw.groups + g] = big;
            std::vector<uint16_t> out(static_cast<size_t>(2) * ldc, 0x7FC1);
            nr32::run_tile(plan, pw, pw.experts[0], 2, r2.bytes.data(), r2.lda,
                    r2.scale.data(), nullptr, nr32::bias_kind::none, out.data(),
                    ldc, 0, plan.cols);
            EXPECT_EQ(from_bf16(out[ldc]), 0.f) << "second row, scales " << big;
        }
        ++served;
    }
    // S4's wide kernels: a split epilogue whose up columns start on a block,
    // at 1 to 3 rows, gate and up column 0 zero.
    if (nr32::isa_supported(data_type_t::s4)) {
        constexpr int K = 64, N = 64, group = 32;
        weight_set ws;
        ws.build(data_type_t::s4, K, N, true, 0, group, 1, 0x6702u);
        for (const int col : {0, N / 2}) {
            for (int k = 0; k < K; ++k) {
                const size_t i = static_cast<size_t>(col) * ws.desc.ldb + k;
                ws.raw[0][i >> 1] &= (i & 1) ? 0x0Fu : 0xF0u;
                ws.value[0][static_cast<size_t>(col) * K + k] = 0.f;
            }
        }
        for (int g = 0; g < K / group; ++g) {
            for (int n = 0; n < N; ++n) {
                ws.scale[0][static_cast<size_t>(g) * N + n]
                        = std::ldexp(1.f, n == 0 || n == N / 2 ? 20 : -10);
            }
        }
        nr32::packed_weights pw;
        ASSERT_EQ(ws.pack(pw), status_t::success);
        nr32::tile_plan plan;
        ASSERT_TRUE(nr32::make_plan(pw, nr32::epilogue::silu_split, false, plan,
                /*src_groups=*/true));
        ASSERT_TRUE(plan.wide);
        for (int M = 1; M <= plan.wide_rows; ++M) {
            SCOPED_TRACE("s4 wide, M=" + std::to_string(M));
            rows_t r = make_rows(ws, M, 0x6703u, pw.groups);
            for (float &v : r.scale)
                v = std::ldexp(1.f, 110);
            const int ldc = plan.out_cols;
            std::vector<uint16_t> dst(static_cast<size_t>(M) * ldc, 0x7FC1);
            nr32::run_tile(plan, pw, pw.experts[0], M, r.bytes.data(), r.lda,
                    r.scale.data(), nullptr, nr32::bias_kind::none, dst.data(),
                    ldc, 0, plan.cols);
            for (int m = 0; m < M; ++m) {
                EXPECT_EQ(from_bf16(dst[static_cast<size_t>(m) * ldc]), 0.f)
                        << "row " << m;
            }
        }
        // Only the last row's scales overflow, or are NaN.
        for (const float big : {std::ldexp(1.f, 110),
                     std::numeric_limits<float>::quiet_NaN()}) {
            const int M = plan.wide_rows;
            SCOPED_TRACE("s4 wide, last of " + std::to_string(M) + " rows");
            rows_t r = make_rows(ws, M, 0x6704u, pw.groups);
            for (int g = 0; g < pw.groups; ++g)
                r.scale[static_cast<size_t>(M - 1) * pw.groups + g] = big;
            const int ldc = plan.out_cols;
            std::vector<uint16_t> dst(static_cast<size_t>(M) * ldc, 0x7FC1);
            nr32::run_tile(plan, pw, pw.experts[0], M, r.bytes.data(), r.lda,
                    r.scale.data(), nullptr, nr32::bias_kind::none, dst.data(),
                    ldc, 0, plan.cols);
            EXPECT_EQ(from_bf16(dst[static_cast<size_t>(M - 1) * ldc]), 0.f)
                    << "scales " << big;
        }
        ++served;
    }
    if (served == 0) GTEST_SKIP() << "no NR32 kernel runs on this CPU";
}

// A plan refuses a gated epilogue over an odd column count.
TEST(Nr32Tile, PlansNoGatedEpilogueOverOddColumns) {
    if (!nr32::isa_supported(data_type_t::bf16)) GTEST_SKIP();
    weight_set ws;
    ws.build(data_type_t::bf16, 64, 33, true, 0, 0, 1, 0x6500u);
    nr32::packed_weights pw;
    ASSERT_EQ(ws.pack(pw), status_t::success);
    nr32::tile_plan plan;
    EXPECT_TRUE(nr32::make_plan(pw, nr32::epilogue::plain, false, plan));
    EXPECT_FALSE(nr32::make_plan(pw, nr32::epilogue::silu_split, false, plan));
    EXPECT_FALSE(
            nr32::make_plan(pw, nr32::epilogue::swiglu_pairs, false, plan));
}

} // namespace
