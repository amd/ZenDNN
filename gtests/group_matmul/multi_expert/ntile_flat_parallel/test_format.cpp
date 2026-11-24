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

/// N-tile flat parallel's format rows: what the packs, the plans, the kernels'
/// selection and the config read from them, how a call finds its row, and the
/// checks that each row holds together.

#include <gtest/gtest.h>

#include <string>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/source.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/formats.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/kernel/nr32_ukernel.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel_config.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/nr32_pack.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/tile/nr32_tile.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nfp = zendnnl::lowoha::matmul::ntile_flat_parallel;
namespace nr32 = nfp::nr32;
using mm::grp_matmul_phase;
using zendnnl::common::data_type_t;

/// Reads the environment's config again when it goes.
struct config_restore {
    ~config_restore() { mm::test_api::set_ntile_flat_parallel_config(nullptr); }
};

nr32::weight_desc desc_of(
        const nr32::format_desc &f, const int K, const int N) {
    nr32::weight_desc d;
    d.fmt = &f;
    d.K = K;
    d.N = N;
    d.trans = true;
    d.ldb = K;
    d.group_size = f.always_grouped ? K / 2 : 0;
    return d;
}

// A row's words size its packs, and its kernels' rows and vectors its plans:
// a split step is a gate and an up vector of one 64-byte line each, or one
// wide call of 64 columns.
TEST(Nr32Format, EachRowSizesItsPacksAndPlans) {
    for (const nr32::format_desc &f : nr32::kFormats) {
        SCOPED_TRACE(f.name);
        const nr32::weight_desc d = desc_of(f, 96, 192);
        ASSERT_TRUE(nr32::packable(d));
        EXPECT_EQ(nr32::block_bytes(d),
                static_cast<size_t>((96 + f.k_per_word - 1) / f.k_per_word)
                        * nr32::kNr * f.word_bytes);
        if (!nr32::isa_supported(f)) continue;
        const nr32::pack_layout layout = nr32::layout_of(d);
        EXPECT_EQ(&layout.format(), &f);
        nr32::tile_plan plain, split;
        ASSERT_TRUE(nr32::make_plan(
                layout, nr32::epilogue::plain, false, plain, false));
        EXPECT_EQ(plain.step, nr32::kNr);
        EXPECT_EQ(plain.max_mr, f.max_mr);
        EXPECT_FALSE(plain.wide);
        ASSERT_TRUE(nr32::make_plan(
                layout, nr32::epilogue::silu_split, false, split, false));
        EXPECT_EQ(split.wide, f.max_mr_wide > 0);
        EXPECT_EQ(split.step, split.wide ? 2 * nr32::kNr : 2 * f.vec_cols());
        EXPECT_EQ(split.wide_max_mr, f.max_mr_wide);
        nr32::tile_plan u8;
        EXPECT_EQ(nr32::make_plan(layout, nr32::epilogue::plain, true, u8),
                f.u8_src);
    }
}

constexpr nr32::epilogue kEpilogues[]
        = {nr32::epilogue::plain, nr32::epilogue::swiglu_pairs,
                nr32::epilogue::silu_split, nr32::epilogue::gelu_split};

// A row's kernels take every combination of the variants a plan asks for,
// at every row count up to the row's limits: u8 sources where the row has
// them, unaligned up columns for a split epilogue, and weight and source
// scale groups for a scaled format.  Its own selectors refuse more rows.
// Kernel selection reads no ISA, so every row whose kernels the build has is
// checked.
TEST(Nr32Format, EachRowsKernelsTakeEveryVariantAPlanAsksFor) {
    const nr32::scale_groups groupings[]
            = {{false, false}, {true, false}, {true, true}};
    int rows = 0;
    for (const nr32::format_desc &f : nr32::kFormats) {
        SCOPED_TRACE(f.name);
        if (f.built != nullptr && !f.built()) continue;
        ++rows;
        for (const nr32::epilogue ep : kEpilogues) {
            const bool split = nr32::split(ep);
            for (const bool u8 : {false, true}) {
                for (const nr32::scale_groups &g : groupings) {
                    // Plans of a format without scales ask for no groups.
                    if (g.any() && !f.scaled()) continue;
                    SCOPED_TRACE(::testing::Message()
                            << "ep=" << static_cast<int>(ep) << " u8=" << u8
                            << " weight_groups=" << g.weight
                            << " source_groups=" << g.source);
                    const bool takes = !u8 || f.u8_src;
                    for (const bool unaligned : {false, true}) {
                        for (int mr = 1; mr <= f.max_mr; ++mr) {
                            EXPECT_EQ(nr32::select(f, mr, ep, u8, unaligned, g)
                                            != nullptr,
                                    takes && (!unaligned || split))
                                    << "mr=" << mr
                                    << " unaligned=" << unaligned;
                        }
                        EXPECT_EQ(f.narrow(f.max_mr + 1, ep, u8, unaligned, g),
                                nullptr);
                    }
                    for (int mr = 1; mr <= f.max_mr_wide; ++mr) {
                        EXPECT_EQ(nr32::select_wide(f, mr, ep, u8, false, g)
                                        != nullptr,
                                takes && split)
                                << "mr=" << mr;
                        EXPECT_EQ(nr32::select_wide(f, mr, ep, u8, true, g)
                                        != nullptr,
                                takes && split && mr <= f.two_chain_rows)
                                << "mr=" << mr << " two_chains";
                    }
                    if (f.wide != nullptr) {
                        EXPECT_EQ(f.wide(f.max_mr_wide + 1, ep, u8, false, g),
                                nullptr);
                    }
                }
            }
        }
    }
    if (rows == 0) GTEST_SKIP() << "this build has no format's kernels";
}

// A projection reads what its plan's kernels read: s8 rows the call makes,
// XOR-ed with 0x80 exactly when the plan runs the u8-source kernels, with one
// scale per row or, where the plan's kernels take them, one per weight scale
// group; or rows of the format's source type as they are.
TEST(Nr32Format, EachRowsSourceIsWhatItsPlansKernelsRead) {
    int rows = 0;
    for (const nr32::format_desc &f : nr32::kFormats) {
        SCOPED_TRACE(f.name);
        // Plans need the ISA.
        if (!nr32::isa_supported(f)) continue;
        ++rows;
        nr32::weight_desc d = desc_of(f, 128, 64);
        if (f.scaled()) d.group_size = 32;
        const nr32::pack_layout layout = nr32::layout_of(d);
        for (const bool groups : {false, true}) {
            for (const bool u8 : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                        << "groups=" << groups << " u8=" << u8);
                nr32::tile_plan plan;
                if (!nr32::make_plan(layout, nr32::epilogue::silu_split, u8,
                            plan, groups)) {
                    EXPECT_TRUE(u8 && !f.u8_src);
                    continue;
                }
                const nfp::detail::source_spec s
                        = nfp::detail::source_of(layout, plan);
                EXPECT_EQ(s.s8, f.src == data_type_t::s8);
                EXPECT_EQ(s.elem_bytes(), static_cast<size_t>(f.src_bytes()));
                EXPECT_EQ(s.u8, plan.src_u8);
                EXPECT_EQ(s.u8, u8);
                // 128 / 32: four scales per row where the kernels take them.
                const bool per_group = s.s8 && groups;
                EXPECT_EQ(plan.src_groups, per_group);
                EXPECT_EQ(s.groups, per_group ? 4 : 1);
                EXPECT_EQ(s.groups, plan.src_scales(layout));
            }
        }
    }
    if (rows == 0) GTEST_SKIP() << "this CPU runs no format's kernels";
}

// A call finds its row by its weight type, the type of the rows its kernels
// read and its scales' encoding.  The single row of a weight and source type
// is the call's whatever its scales, which the APIs then check.
TEST(Nr32Format, ACallFindsItsRowByItsTypesAndScales) {
    for (int i = 0; i < nr32::kFormatCount; ++i) {
        const nr32::format_desc &f = nr32::kFormats[i];
        SCOPED_TRACE(f.name);
        EXPECT_EQ(nr32::format_index(f), i);
        for (const nr32::scale_enc e :
                {f.scales, nr32::scale_enc::none, nr32::scale_enc::real}) {
            const nr32::format_desc *r = nr32::format_for(f.wei, f.src, e);
            ASSERT_NE(r, nullptr);
            EXPECT_STREQ(r->name, f.name);
        }
    }
    EXPECT_EQ(nr32::format_for(data_type_t::s8, data_type_t::bf16,
                      nr32::scale_enc::real),
            nullptr);
    EXPECT_EQ(nr32::format_for(data_type_t::bf16, data_type_t::s8,
                      nr32::scale_enc::none),
            nullptr);
    EXPECT_EQ(nr32::format_for(data_type_t::bf16, data_type_t::f16,
                      nr32::scale_enc::none),
            nullptr);
    const float one = 1.0f;
    EXPECT_EQ(nr32::scale_enc_of(nullptr, data_type_t::f32),
            nr32::scale_enc::none);
    EXPECT_EQ(
            nr32::scale_enc_of(&one, data_type_t::f32), nr32::scale_enc::real);
}

// The row checks refuse a row that does not hold together, as a new format's
// might, and take one that shares a weight type with another format, as a
// W4A16 row beside W4A8's would.
TEST(Nr32Format, BrokenRowsDoNotHold) {
    using nr32::format_desc;
    const auto holds = [](format_desc f, const auto &change) {
        change(f);
        return nr32::format_holds(f);
    };
    for (const format_desc &f : nr32::kFormats)
        EXPECT_TRUE(nr32::format_holds(f)) << f.name;
    const format_desc &s8 = nr32::s8::kFormat;
    const format_desc &bf16 = nr32::bf16::kFormat;
    EXPECT_FALSE(holds(s8, [](format_desc &f) { f.pack = nullptr; }));
    EXPECT_FALSE(holds(s8, [](format_desc &f) { f.narrow = nullptr; }));
    // Kernels that need AVX-512 FP16 say whether the build has them.
    EXPECT_FALSE(holds(
            nr32::f16::kFormat, [](format_desc &f) { f.built = nullptr; }));
    // Wide kernels without wide rows, and wide rows past the wide kernels'.
    EXPECT_FALSE(holds(s8, [](format_desc &f) { f.max_mr_wide = 0; }));
    EXPECT_FALSE(
            holds(s8, [](format_desc &f) { f.wide_rows = f.max_mr_wide + 1; }));
    EXPECT_FALSE(holds(s8, [](format_desc &f) { f.word_bytes = 3; }));
    EXPECT_FALSE(holds(bf16, [](format_desc &f) { f.elem_bits = 12; }));
    // s8 sources come with weight scales and a bf16 output.
    EXPECT_FALSE(holds(
            s8, [](format_desc &f) { f.scales = nr32::scale_enc::none; }));
    EXPECT_FALSE(holds(s8, [](format_desc &f) { f.dst = data_type_t::f32; }));
    // Sums, u8 sources and the caller's s8 rows are s8 sources' only.
    EXPECT_FALSE(holds(bf16, [](format_desc &f) { f.sums = true; }));
    EXPECT_FALSE(holds(bf16, [](format_desc &f) { f.u8_src = true; }));
    EXPECT_FALSE(holds(bf16, [](format_desc &f) { f.s8_rows = true; }));
    EXPECT_FALSE(holds(bf16, [](format_desc &f) { f.always_grouped = true; }));
    // A format without scales reads and stores its own type, of its
    // elements' size.
    EXPECT_FALSE(holds(bf16, [](format_desc &f) { f.dst = data_type_t::f16; }));
    EXPECT_FALSE(holds(nr32::f16::kFormat, [](format_desc &f) {
        f.elem_bits = 8;
        f.k_per_word = 2;
    }));
    // An FP32 row waits for the shared code to take FP32 rows and outputs.
    EXPECT_FALSE(holds(bf16, [](format_desc &f) {
        f.wei = f.src = f.dst = data_type_t::f32;
        f.elem_bits = 32;
        f.word_bytes = 4;
        f.k_per_word = 1;
    }));
    EXPECT_TRUE(holds(nr32::s4::kFormat, [](format_desc &f) {
        f.name = "w4a16";
        f.src = data_type_t::bf16;
        f.sums = false;
    }));
}

// A type without a row has no format, no packs and no config filter.
TEST(Nr32Format, TypesWithoutARowAreRefused) {
    for (const data_type_t t :
            {data_type_t::f32, data_type_t::u8, data_type_t::f4_e2m1}) {
        SCOPED_TRACE(static_cast<int>(t));
        EXPECT_FALSE(nr32::has_format(t));
        EXPECT_EQ(nr32::format_for(t, t, nr32::scale_enc::none), nullptr);
        EXPECT_EQ(nr32::format_for(t, data_type_t::s8, nr32::scale_enc::real),
                nullptr);
    }
    nr32::weight_desc d;
    d.K = 64;
    d.N = 64;
    d.ldb = 64;
    EXPECT_FALSE(nr32::packable(d));
    EXPECT_EQ(nr32::block_bytes(d), 0u);
    const config_restore restore;
    mm::test_api::set_ntile_flat_parallel_config("dtype=f32:units=3");
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_config_warnings().size(), 1u);
}

// A routed layer whose key has no format is declined before the call reads
// anything or packs the layer.
TEST(Nr32Format, ALayerWithoutAFormatIsDeclined) {
    nfp::layer_key key;
    key.act = mm::grp_matmul_gated_act_t::silu_and_mul;
    key.num_experts = 1;
    key.H = 64;
    key.I = 32;
    const int expert = 0, count = 1;
    nfp::moe_call call;
    call.num_tokens = 1;
    call.topk = 1;
    call.local_experts = &expert;
    call.counts = &count;
    call.num_threads = 1;
    bool described = false;
    const nfp::describe_layer describe
            = [&](nfp::projection &, nfp::projection &) -> const char * {
        described = true;
        return nullptr;
    };
    EXPECT_EQ(nfp::execute(key, {}, describe, call),
            zendnnl::error_handling::status_t::unimplemented);
    EXPECT_FALSE(described);
}

// The config's dtype filter takes every row's name.
TEST(Nr32Format, TheConfigFiltersByEachRowsName) {
    const config_restore restore;
    std::string text;
    for (int i = 0; i < nr32::kFormatCount; ++i)
        text += std::string("dtype=") + nr32::kFormats[i].name
                + ":units=" + std::to_string(i + 3) + ";";
    mm::test_api::set_ntile_flat_parallel_config(text.c_str());
    EXPECT_TRUE(mm::test_api::ntile_flat_parallel_config_warnings().empty());
    for (int i = 0; i < nr32::kFormatCount; ++i) {
        EXPECT_EQ(nfp::call_knobs_for(grp_matmul_phase::decode,
                          nr32::kFormats[i], nfp::api_kind::routed)
                          .units,
                i + 3)
                << nr32::kFormats[i].name;
    }
}

} // namespace
