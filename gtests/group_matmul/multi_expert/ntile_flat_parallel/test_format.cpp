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
/// selection and the config read from them.

#include <gtest/gtest.h>

#include <string>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/nr32_format.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/nr32_pack.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/nr32_tile.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel_config.hpp"

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
    d.wei = f.wei;
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
        EXPECT_EQ(nr32::max_mr(f.wei), f.max_mr);
        EXPECT_EQ(nr32::max_mr_wide(f.wei), f.max_mr_wide);
        if (!nr32::isa_supported(f.wei)) continue;
        const nr32::pack_layout layout = nr32::layout_of(d);
        EXPECT_EQ(layout.fmt->wei, f.wei);
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

// A type without a row has no packs, no kernels and no config filter.
TEST(Nr32Format, TypesWithoutARowAreRefused) {
    for (const data_type_t t :
            {data_type_t::f32, data_type_t::u8, data_type_t::f4_e2m1}) {
        SCOPED_TRACE(static_cast<int>(t));
        EXPECT_EQ(nr32::format_of(t), nullptr);
        EXPECT_EQ(nr32::format_index(t), nr32::kFormatCount);
        nr32::weight_desc d;
        d.wei = t;
        d.K = 64;
        d.N = 64;
        d.ldb = 64;
        EXPECT_FALSE(nr32::packable(d));
        EXPECT_EQ(nr32::block_bytes(d), 0u);
        EXPECT_FALSE(nr32::isa_supported(t));
        EXPECT_EQ(nr32::select(t, 1, nr32::epilogue::plain, false, false),
                nullptr);
    }
    const config_restore restore;
    mm::test_api::set_ntile_flat_parallel_config("dtype=f32:units=3");
    EXPECT_EQ(mm::test_api::ntile_flat_parallel_config_warnings().size(), 1u);
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
                          nr32::kFormats[i].wei, nfp::api_kind::routed)
                          .units,
                i + 3)
                << nr32::kFormats[i].name;
    }
}

} // namespace
