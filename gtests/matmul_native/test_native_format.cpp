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
#include <vector>

#include <gtest/gtest.h>

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/gemv/native_gemv_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int4_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_call.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_file.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_format.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_view.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
using nat::native_format_t;
using nat::native_kernel_kind_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;

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
    f = s8;
    f.contract.wei_zp_refusal = nullptr;
    EXPECT_FALSE(nat::native_refusals_have_reasons(f));
    f = s4;
    f.contract.ungrouped_refusal = nullptr;
    EXPECT_FALSE(nat::native_refusals_have_reasons(f));
}

// An INT8 call takes the route of its plan (prepare_int8_call is given it):
// its format's route whose kernels serve its scheme, one group over K of a
// format of group scales only the kernels of one group; the tiles take the
// group or the channel tile by the route's class.  The group shifts cancel
// each format's decode offset, and the tiles read every format's scales in
// place.
TEST(NativeFormat, Int8CallsTakeTheRouteOfTheirScheme) {
    using nat::native_route_t;
    nat::native_problem_t p;
    p.M = 4;
    p.N = 64;
    p.K = 256;
    p.pack_nr = 32;
    p.threads = 1;
    p.src_dt = data_type_t::s8;
    p.dst_dt = data_type_t::f32;
    p.wei_dt = data_type_t::s8;
    EXPECT_EQ(p.route(), native_route_t::s8_channel);
    p.int8_scheme = nat::native_int8_scheme_t::per_group;
    EXPECT_EQ(p.route(), native_route_t::s8_group);
    p.wei_dt = data_type_t::s4;
    for (const int gs : {64, 128, 0, 256}) {
        p.wei_group_size = gs;
        EXPECT_EQ(p.route(),
                gs % 256 != 0 ? native_route_t::s4_group
                              : native_route_t::s4_one_group)
                << "group " << gs;
    }
    for (const native_route_t r : {native_route_t::s8_group,
                 native_route_t::s4_group, native_route_t::s4_one_group})
        EXPECT_TRUE(nat::native_of_class<nat::native_route_class_t::group>(r));
    EXPECT_FALSE(nat::native_of_class<nat::native_route_class_t::group>(
            native_route_t::s8_channel));
    EXPECT_EQ(nat::kNativeGroupedTypes, nat::native_type_bit(data_type_t::s4));
    EXPECT_EQ(nat::native_format(native_format_t::s8).offset_bits, 7);
    EXPECT_EQ(nat::native_format(native_format_t::s4).offset_bits, 3);
    EXPECT_TRUE(nat::native_scales_at(nat::native_place_t::caller));
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

// The registry lists, by route and then by width, the kernels of every route
// whose family's instruction set the CPU has, and nothing else: at nr 16 and
// 32 (F16_PH also 64) the route table's rows, which are the row counts the
// selectors serve, and tiles of several slabs up to 128 columns with the rows
// of the slab; each entry carries its route, the route's family and its INT8
// scheme.
TEST(NativeFormat, RoutesAreTheRegistrysKernels) {
    if (!nat::native_base_isa()) GTEST_SKIP() << "no AVX-512 base ISA";
    std::vector<nat::native_kernel_caps_t> want;
    for (const nat::native_route_desc_t &r : nat::kNativeRoutes) {
        const bool runs = r.family == native_kernel_kind_t::f16_ph
                ? nt::f16_ph_kernels_registered()
                : has_isa(nat::native_family(r.family).isa);
        for (const int nr : {16, 32, 64, 128}) {
            const uint32_t rows = nt::route_kernel_rows(r.id, nr);
            nat::native_kernel_caps_t c;
            c.route = r.id;
            c.kind = r.family;
            c.nr = nr;
            while (c.max_mr < 12 && ((rows >> c.max_mr) & 1u) != 0)
                ++c.max_mr;
            c.int8_schemes = r.schemes;
            if (runs && c.max_mr > 0) want.push_back(c);
        }
    }
    const std::vector<nat::native_kernel_caps_t> &all
            = nat::native_kernel_registry().all();
    ASSERT_EQ(all.size(), want.size());
    for (size_t i = 0; i < all.size(); ++i) {
        SCOPED_TRACE(std::string(nat::native_family(want[i].kind).name)
                + " for "
                + nat::native_format(nat::native_route(want[i].route).format)
                          .name
                + " nr " + std::to_string(want[i].nr));
        EXPECT_EQ(all[i].route, want[i].route);
        EXPECT_EQ(all[i].kind, want[i].kind);
        EXPECT_EQ(all[i].nr, want[i].nr);
        EXPECT_EQ(all[i].max_mr, want[i].max_mr);
        EXPECT_EQ(all[i].max_mr,
                nat::native_route_rows(nat::native_route(want[i].route),
                        nat::native_tile_slab_cols(want[i].kind, want[i].nr)
                                / 16));
        EXPECT_EQ(all[i].int8_schemes, want[i].int8_schemes);
    }
}

// The kernels of the INT8 tile and the GEMV take their format: each selector
// has kernels for a format of a route on its family and none for another,
// the per-channel kernels (which read the bytes as they are against a source
// plus 128) for a format of the byte_offset decode, the INT4 group GEMV for
// an s8 source only.  The weight scales' types are today's three.
TEST(NativeFormat, KernelsTakeTheirFormat) {
    using nat::native_kernel_kind_t;
    using nat::native_scale_t;
    using order_t = nat::native_int8_order_t;
    for (const nat::native_format_desc_t &f : nat::kNativeFormats) {
        SCOPED_TRACE(f.name);
        const bool int8
                = nat::native_family_takes<native_kernel_kind_t::int8>(f.id);
        const bool int4
                = nat::native_family_takes<native_kernel_kind_t::int4>(f.id);
        const bool bytes
                = int8 && f.decode == nat::native_decode_t::byte_offset;
        EXPECT_EQ(nat::select_native_int8_ukernel(f.id, 1, 1, true) != nullptr,
                bytes);
        EXPECT_EQ(nat::select_native_int8_group_ukernel(f.id, 1, 2, false)
                        != nullptr,
                int8);
        EXPECT_EQ(nat::select_native_int4_group_ukernel(
                          f.id, 1, 2, order_t::fused, true)
                        != nullptr,
                int4);
        EXPECT_EQ(nat::select_native_int4_one_group_ukernel(
                          f.id, 12, 2, order_t::reference)
                        != nullptr,
                int4);
        EXPECT_EQ(nat::select_native_s8_gemv(1, false, f.id, true) != nullptr,
                bytes);
        EXPECT_EQ(nat::select_native_s8_fused_gemv(2, true, f.id, false)
                        != nullptr,
                bytes);
        EXPECT_EQ(
                nat::select_native_group_gemv(1, false, f.id, false) != nullptr,
                int8 || int4);
        EXPECT_EQ(nat::select_native_group_gemv(2, true, f.id, true) != nullptr,
                int8);
        for (const nat::native_int8_scheme_t s :
                {nat::native_int8_scheme_t::per_channel,
                        nat::native_int8_scheme_t::per_group}) {
            EXPECT_EQ(nat::native_int8_kernels_serve(f.id, s, 6, 2), int8);
            EXPECT_EQ(nat::native_int4_kernels_serve(f.id, s, 6, 2), int4);
        }
        EXPECT_EQ(f.contract.scale_types != 0, f.quantized());
    }
    EXPECT_EQ(nat::native_scale_of(data_type_t::f32), native_scale_t::f32);
    EXPECT_EQ(nat::native_scale_of(data_type_t::bf16), native_scale_t::bf16);
    EXPECT_EQ(nat::native_scale_of(data_type_t::f16), native_scale_t::f16);
    EXPECT_EQ(nat::native_scale_of(data_type_t::s8), native_scale_t::none);
    EXPECT_EQ(nat::native_scale_bytes(native_scale_t::f32), 4u);
    EXPECT_EQ(nat::native_scale_bytes(native_scale_t::bf16), 2u);
    EXPECT_EQ(nat::native_scale_bytes(native_scale_t::f16), 2u);
    EXPECT_EQ(nat::kNativeFloatScales,
            nat::native_scale_bit(native_scale_t::f32)
                    | nat::native_scale_bit(native_scale_t::bf16)
                    | nat::native_scale_bit(native_scale_t::f16));
}

// The INT8 call checks take weight scales of a data type exactly when it is
// a scale type of the weight's format's contract (native_contract_t::
// scale_types): an s8 weight's per column, per group and under an {M, G}
// source scale, an s4 weight's per group.
TEST(NativeFormat, CallChecksTakeTheContractsScaleTypes) {
    const int M = 4, N = 32, K = 256, G = 2;
    struct scheme_t {
        const char *name;
        data_type_t wei;
        std::vector<int64_t> wei_dims, src_dims;
    };
    const scheme_t schemes[] = {
            {"s8 per column", data_type_t::s8, {1, N}, {1, 1}},
            {"s8 per group", data_type_t::s8, {G, N}, {1, 1}},
            {"s8 under source groups", data_type_t::s8, {1, N}, {M, G}},
            {"s4 per group", data_type_t::s4, {G, N}, {1, 1}},
    };
    // The checks read no buffer.
    const float any = 1.0f;
    for (const scheme_t &s : schemes) {
        const nat::native_format_desc_t &f = nat::native_format(s.wei);
        for (int t = 0; t <= static_cast<int>(data_type_t::f8_e8m0); ++t) {
            const data_type_t dt = static_cast<data_type_t>(t);
            SCOPED_TRACE(std::string(s.name) + ", weight scales of data type "
                    + std::to_string(t));
            nat::GemmDescriptor d;
            d.M = M;
            d.N = N;
            d.K = K;
            d.src_dt = data_type_t::s8;
            d.wei_dt = s.wei;
            d.dst_dt = data_type_t::f32;
            mm::matmul_params p;
            p.dtypes.src = data_type_t::s8;
            p.dtypes.wei = s.wei;
            p.dtypes.dst = data_type_t::f32;
            p.quant_params.src_scale.buff = &any;
            p.quant_params.src_scale.dt = data_type_t::f32;
            p.quant_params.src_scale.dims = s.src_dims;
            p.quant_params.wei_scale.buff = &any;
            p.quant_params.wei_scale.dt = dt;
            p.quant_params.wei_scale.dims = s.wei_dims;
            const bool takes
                    = (f.contract.scale_types
                              & nat::native_scale_bit(nat::native_scale_of(dt)))
                    != 0;
            EXPECT_EQ(nat::check_int8_call(
                              d, p, nat::decide_int8_call(d, p, nullptr))
                            == status_t::success,
                    takes);
        }
    }
}

// The data type of each scale type of a format's contract (native_scale_dt,
// which plan_dump gives a call's weight scales) gives it back and has its
// bytes.
TEST(NativeFormat, ScaleTypesHaveTheirDataTypes) {
    for (const nat::native_format_desc_t &f : nat::kNativeFormats) {
        for (unsigned i = 0; i < 8 * sizeof(f.contract.scale_types); ++i) {
            if ((f.contract.scale_types & (1u << i)) == 0) continue;
            const auto s = static_cast<nat::native_scale_t>(i);
            const data_type_t dt = nat::native_scale_dt(s);
            SCOPED_TRACE(
                    std::string(f.name) + ", scale type " + std::to_string(i));
            EXPECT_EQ(nat::native_scale_of(dt), s);
            EXPECT_EQ(nat::native_scale_bytes(s), zendnnl::common::size_of(dt));
        }
    }
    EXPECT_EQ(
            nat::native_scale_dt(nat::native_scale_t::none), data_type_t::none);
}

} // namespace
