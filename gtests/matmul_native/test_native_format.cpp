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

// The format, family and route tables of the native BRGEMM
// (native_format.hpp) against what they describe: the layout's geometry,
// the layout rule, the names of the tuning file and the plan line, and the
// kernels the registry holds.

#include <iterator>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_file.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_format.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_view.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
using nat::native_format_t;
using nat::native_kernel_kind_t;
using zendnnl::memory::data_type_t;

constexpr int kNr = nat::kNativeLayoutPackNr;

// K groups, block bytes, K group stride and raw bytes of each weight type
// at K 9: K pairs of bf16, K quads of s8 and their column sums, K octets of
// s4 nibbles, K rows of f32 and f16.
TEST(NativeFormat, GeometryIsTheLayouts) {
    struct case_t {
        data_type_t dt;
        int groups;
        size_t block_bytes;
        int stride;
        size_t raw_32, raw_1; ///< a [32, 9] and a [1, 9] weight
    };
    const case_t cases[] = {
            {data_type_t::bf16, 5, 5 * 4 * kNr, 2 * kNr, 576, 18},
            {data_type_t::s8, 3, (3 * 4 + 4) * kNr, 4 * kNr, 288, 9},
            {data_type_t::s4, 2, 2 * 4 * kNr, 4 * kNr, 144, 5},
            {data_type_t::f32, 9, 9 * 4 * kNr, kNr, 1152, 36},
            {data_type_t::f16, 9, 9 * 2 * kNr, kNr, 576, 18},
    };
    const int K = 9;
    for (const case_t &c : cases) {
        SCOPED_TRACE(nat::native_format(c.dt).name);
        EXPECT_EQ(nat::native_format_of(c.dt)->wei_dt, c.dt);
        EXPECT_EQ(nat::native_k_groups(c.dt, K), c.groups);
        EXPECT_EQ(nat::native_block_bytes(c.dt, K, kNr), c.block_bytes);
        nat::native_weight_view_t v;
        v.wei_dt = c.dt;
        v.pack_nr = kNr;
        EXPECT_EQ(v.group_stride(), c.stride);
        EXPECT_EQ(nat::native_raw_weight_bytes(c.dt, K, 32, K, true), c.raw_32);
        EXPECT_EQ(nat::native_raw_weight_bytes(c.dt, K, 1, K, true), c.raw_1);
        EXPECT_TRUE(nat::native_layout_fits(c.dt, K, 1, kNr));
    }
    EXPECT_EQ(nat::kNativeS4Octet, 8);
    for (const data_type_t dt : {data_type_t::none, data_type_t::u8,
                 data_type_t::s32, data_type_t::u4, data_type_t::f4_e2m1}) {
        EXPECT_EQ(nat::native_format_of(dt), nullptr);
        EXPECT_FALSE(nat::native_layout_fits(dt, K, 1, kNr));
    }
}

/// The CPU runs kernels of isa.
bool has_isa(nat::native_isa_t isa) {
    const nat::UarchParams &u = nat::detect_uarch();
    switch (isa) {
        case nat::native_isa_t::base: return true;
        case nat::native_isa_t::avx512_bf16: return u.avx512bf16;
        case nat::native_isa_t::avx512_vnni: return u.avx512vnni;
        case nat::native_isa_t::avx512_fp16: return u.avx512fp16;
    }
    return false;
}

/// The first data type of a native_type_bit mask.
data_type_t first_type(uint32_t types) {
    for (int t = 0; t < 32; ++t)
        if ((types & (1u << t)) != 0) return static_cast<data_type_t>(t);
    return data_type_t::none;
}

// The layout rule takes what each format's contract takes and refuses the
// rest with the contract's reasons: each refused source, K past k, groups
// of other than whole K quads (one group over K too for a weight without
// per-column scales), groups past group, and K past the whole K groups an
// int holds.
TEST(NativeFormat, LayoutRuleReadsTheContract) {
    if (!nat::native_base_isa()) GTEST_SKIP() << "no AVX-512 base ISA";
    for (const nat::native_format_desc_t &f : nat::kNativeFormats) {
        SCOPED_TRACE(f.name);
        const nat::native_contract_t &c = f.contract;
        uint32_t refused = 0;
        for (const nat::native_refusal_t &r : c.src_refusals)
            refused |= r.types;
        nat::native_layout_query_t q;
        q.wei_dt = f.wei_dt;
        q.src_dt = first_type(
                ~refused & ~nat::native_type_bit(data_type_t::none));
        q.K = 64;
        q.N = 48;
        q.ldb = 64;
        q.transB = true;
        if (!has_isa(c.isa)) {
            const nat::native_layout_decision_t d
                    = nat::decide_native_layout(q);
            EXPECT_FALSE(d.native);
            EXPECT_TRUE(d.isa);
            continue;
        }
        EXPECT_TRUE(nat::decide_native_layout(q).native);
        for (const nat::native_refusal_t &r : c.src_refusals) {
            if (r.why == nullptr) continue;
            nat::native_layout_query_t s = q;
            s.src_dt = first_type(r.types);
            EXPECT_STREQ(nat::decide_native_layout(s).reason, r.why);
        }
        if (c.k.max < nat::kNativeNoLimit.max) {
            nat::native_layout_query_t k = q;
            k.K = c.k.max + 1;
            k.ldb = k.K;
            EXPECT_STREQ(nat::decide_native_layout(k).reason, c.k.why);
        }
        if (!f.quantized()) continue;
        nat::native_layout_query_t g = q;
        g.wei_group_size = 6;
        EXPECT_STREQ(nat::decide_native_layout(g).reason, c.group_refusal);
        g.wei_group_size = 0;
        g.K = 66;
        g.ldb = 66;
        const bool per_channel
                = (c.scale_grains
                          & nat::native_grain_bit(nat::native_grain_t::channel))
                != 0;
        if (per_channel) {
            EXPECT_TRUE(nat::decide_native_layout(g).native);
        } else {
            EXPECT_STREQ(nat::decide_native_layout(g).reason, c.group_refusal);
        }
        if (c.group.max < nat::kNativeNoLimit.max) {
            g.K = (c.group.max / 4 + 1) * 4;
            g.ldb = g.K;
            EXPECT_STREQ(nat::decide_native_layout(g).reason, c.group.why);
        }
        if (c.layout_k_refusal != nullptr) {
            g.K = nat::native_max_layout_k(f) + 4;
            g.ldb = g.K;
            g.wei_group_size = 4;
            EXPECT_STREQ(
                    nat::decide_native_layout(g).reason, c.layout_k_refusal);
        }
    }
}

// Every refusal of a contract has its reason: each entry's, and a copy of an
// entry without one fails the check (a refused source, a K past k, groups of
// other than whole K quads, groups past group, a K past the layout's bound).
TEST(NativeFormat, EveryRefusalHasItsReason) {
    for (const nat::native_format_desc_t &f : nat::kNativeFormats)
        EXPECT_TRUE(nat::native_refusals_have_reasons(f)) << f.name;
    const nat::native_format_desc_t &f16
            = nat::native_format(native_format_t::f16);
    const nat::native_format_desc_t &s8
            = nat::native_format(native_format_t::s8);
    const nat::native_format_desc_t &s4
            = nat::native_format(native_format_t::s4);
    nat::native_format_desc_t f = f16;
    f.contract.k = {1000, nullptr};
    EXPECT_FALSE(nat::native_refusals_have_reasons(f));
    f = s8;
    f.contract.src_refusals[1]
            = {nat::native_type_bit(data_type_t::bf16), nullptr};
    EXPECT_FALSE(nat::native_refusals_have_reasons(f));
    f = s8;
    f.contract.group_refusal = nullptr;
    EXPECT_FALSE(nat::native_refusals_have_reasons(f));
    f = f16;
    f.contract.scale_grains = nat::native_grain_bit(nat::native_grain_t::group);
    EXPECT_FALSE(nat::native_refusals_have_reasons(f));
    f = s4;
    f.contract.group.why = nullptr;
    EXPECT_FALSE(nat::native_refusals_have_reasons(f));
    f = s4;
    f.contract.layout_k_refusal = nullptr;
    EXPECT_FALSE(nat::native_refusals_have_reasons(f));
}

// The formats are the tuning file's weight types, and the families the
// kernel kinds of the plan line (kernels=), of the plan dump and of the
// schema.
TEST(NativeFormat, NamesAreThoseOfTheTuningFileAndThePlanLine) {
    std::set<std::string> formats, dtypes;
    for (const nat::native_format_desc_t &f : nat::kNativeFormats)
        formats.insert(f.name);
    for (const char *name : nat::kNativeDtypeNames)
        dtypes.insert(name);
    EXPECT_EQ(formats, dtypes);
    const char *const kinds[]
            = {"bf16", "int8", "f32", "int4", "f16", "f16_ph"};
    ASSERT_EQ(std::size(nat::kNativeFamilies), std::size(kinds));
    for (size_t i = 0; i < std::size(kinds); ++i)
        EXPECT_STREQ(nat::kNativeFamilies[i].name, kinds[i]);
}

// The registry holds kernels for every route whose family's instruction set
// the CPU has, at nr 16 and 32, and none outside the routes.
TEST(NativeFormat, RoutesAreTheRegistrysKernels) {
    if (!nat::native_base_isa()) GTEST_SKIP() << "no AVX-512 base ISA";
    const nat::native_kernel_registry_t &reg = nat::native_kernel_registry();
    const nat::native_int8_scheme_t schemes[]
            = {nat::native_int8_scheme_t::per_channel,
                    nat::native_int8_scheme_t::per_group};
    for (const nat::native_route_desc_t &r : nat::kNativeRoutes) {
        const nat::native_family_desc_t &fam = nat::native_family(r.family);
        SCOPED_TRACE(std::string(fam.name) + " for "
                + nat::native_format(r.format).name);
        const bool runs = r.family == native_kernel_kind_t::f16_ph
                ? nt::f16_ph_kernels_registered()
                : has_isa(fam.isa);
        for (const int nr : {16, 32}) {
            const nat::native_kernel_caps_t *caps = reg.caps(r.family, nr);
            for (const nat::native_int8_scheme_t s : schemes)
                if (r.schemes == nat::native_int8_scheme_bit(s))
                    caps = reg.caps(r.family, nr, s);
            EXPECT_EQ(caps != nullptr, runs) << "nr " << nr;
        }
    }
    for (const nat::native_family_desc_t &fam : nat::kNativeFamilies)
        for (const int nr : {16, 32}) {
            SCOPED_TRACE(std::string(fam.name) + " nr " + std::to_string(nr));
            const nat::native_kernel_caps_t *caps = reg.caps(fam.id, nr);
            if (caps == nullptr) continue;
            uint8_t routed = 0;
            bool float_route = false;
            for (const nat::native_route_desc_t &r : nat::kNativeRoutes)
                if (r.family == fam.id) {
                    routed |= r.schemes;
                    float_route = float_route || r.schemes == 0;
                }
            if (caps->int8_schemes == 0) { EXPECT_TRUE(float_route); }
            for (const nat::native_int8_scheme_t s : schemes) {
                if (reg.caps(fam.id, nr, s) != nullptr) {
                    EXPECT_NE(routed & nat::native_int8_scheme_bit(s), 0);
                }
            }
        }
}

} // namespace
