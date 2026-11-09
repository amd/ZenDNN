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

// Native BRGEMM planner: config text, plan validation, the override and
// tuning-file sources, precedence and fallback, the plan cache, the process
// planner's settings, the kernel registry, the hardware probe, and the
// schema dump.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_brgemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int4_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_config.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_file.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_keys.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"
#include "nlohmann/json.hpp"

#if !defined(_WIN32)
#include <sys/stat.h>
#endif

namespace {

using namespace zendnnl::lowoha::matmul::native;
using json = nlohmann::json;
using source_kind = native_plan_source_t;

bool has_isa() {
    const UarchParams &u = detect_uarch();
    return u.avx512bf16 && u.avx512vnni;
}

native_problem_t problem(
        int M, int N, int K, bool bf16 = true, int threads = 24) {
    native_problem_t p;
    p.M = M;
    p.N = N;
    p.K = K;
    p.pack_nr = 32;
    p.threads = threads;
    p.src_dt = bf16 ? data_type_t::bf16 : data_type_t::u8;
    p.wei_dt = bf16 ? data_type_t::bf16 : data_type_t::s8;
    p.dst_dt = data_type_t::bf16;
    return p;
}

/// The problem of an s4 weight in groups of gs over a bf16 source the call
/// quantizes.
native_problem_t int4_problem(int M, int N, int K, int gs, int threads = 24) {
    native_problem_t p = problem(M, N, K, false, threads);
    p.src_dt = data_type_t::bf16;
    p.wei_dt = data_type_t::s4;
    p.dynamic_quant = true;
    p.int8_scheme = native_int8_scheme_t::per_group;
    p.wei_group_size = gs;
    return p;
}

/// The hardware of the tuning-file examples: every ISA, 48 KiB L1d, 1 MiB
/// L2, 32 MiB L3 shared by 16 logical CPUs, SMT 2, 128 cores.
native_hw_t test_hw() {
    native_hw_t hw;
    hw.uarch.l1d_bytes = 48 << 10;
    hw.uarch.l2_bytes = 1 << 20;
    hw.uarch.l3_domain_bytes = 32 << 20;
    hw.uarch.avx512f = true;
    hw.uarch.avx512bf16 = true;
    hw.uarch.avx512vnni = true;
    hw.logical_cpus = 256;
    hw.smt = 2;
    hw.physical_cores = 128;
    hw.l3_domain_threads = 16;
    return hw;
}

native_config_t config(const std::string &text) {
    native_config_t c;
    std::string why;
    EXPECT_TRUE(parse_native_config(text, c, why)) << text << ": " << why;
    return c;
}

bool parses(const std::string &text) {
    native_config_t c;
    std::string why;
    return parse_native_config(text, c, why);
}

/// The config the source gives p, as text.
std::string config_of(
        const native_config_source_t *source, const native_problem_t &p) {
    if (source == nullptr) return "<no source>";
    const native_config_entry_t *e = source->lookup(p);
    return e == nullptr ? "<none>" : format_native_config(e->config);
}

/// Lines f writes that are planner or loader warnings.
template <typename F>
int warnings_of(F &&f) {
    testing::internal::CaptureStdout();
    f();
    std::cout.flush();
    const std::string out = testing::internal::GetCapturedStdout();
    int n = 0;
    for (size_t at = out.find("native BRGEMM:"); at != std::string::npos;
            at = out.find("native BRGEMM:", at + 1))
        ++n;
    return n;
}

/// n when warnings are written at all.
int shown(int n) {
    return zendnnl::error_handling::apilog_warning_enabled() ? n : 0;
}

/// A tuning file of the format and version this build reads.
std::string tuning(const std::string &rest) {
    return std::string(
                   R"({"format": "zendnnl-native-brgemm-tuning", "version": 2)")
            + (rest.empty() ? "" : ", " + rest) + "}";
}

/// The sources of a tuning file of the given text, read on hw.
native_tuning_sources_t load(
        const std::string &text, const native_hw_t &hw = test_hw()) {
    const native_brgemm_test::temp_file_t file("test.json", text);
    return load_native_tuning_file(file.path(), hw);
}

bool same_grid(const native_plan_t &a, const native_plan_t &b) {
    return a.nr == b.nr && a.mr == b.mr && a.mb == b.mb
            && a.m_blocks == b.m_blocks && a.n_tiles == b.n_tiles
            && a.kc == b.kc && a.threads == b.threads
            && a.col_groups == b.col_groups && a.unit_tiles == b.unit_tiles
            && a.domain_threads == b.domain_threads && a.row_unit == b.row_unit
            && a.traversal == b.traversal
            && a.scratch_floats == b.scratch_floats
            && a.int8_rows_nv1 == b.int8_rows_nv1
            && a.int8_rows_nv2 == b.int8_rows_nv2
            && a.int8_cw_mask == b.int8_cw_mask
            && a.int8_pass_budget == b.int8_pass_budget
            && a.int8_pass_quads == b.int8_pass_quads
            && a.int8_direct_store == b.int8_direct_store
            && a.int8_bias_in_dequant == b.int8_bias_in_dequant
            && a.copy_block == b.copy_block && a.copy_par_min == b.copy_par_min;
}

bool same_plan(const native_plan_t &a, const native_plan_t &b) {
    return same_grid(a, b) && a.sources == b.sources
            && a.rules_entry == b.rules_entry && a.table_entry == b.table_entry
            && a.override_entry == b.override_entry;
}

/// A source that gives its one entry to every problem.
class one_entry_source_t final : public native_config_source_t {
public:
    one_entry_source_t(const char *text, const char *label) {
        entry_.config = config(text);
        entry_.label = label;
    }
    const native_config_entry_t *lookup(
            const native_problem_t &) const noexcept override {
        return &entry_;
    }
    const native_config_entry_t &entry() const { return entry_; }

private:
    native_config_entry_t entry_;
};

std::unique_ptr<one_entry_source_t> one_entry(
        const char *text, const char *label) {
    return std::make_unique<one_entry_source_t>(text, label);
}

/// The highest layer the plan applied, heuristic when none.
source_kind top_source(const native_plan_t &q) {
    return native_brgemm_test::top_source(q);
}

/// heuristic_params_t's value of field, as an integer.
template <typename T, T heuristic_params_t::*field>
int64_t field_value(const heuristic_params_t &p) {
    return static_cast<int64_t>(p.*field);
}

/// What the schema dump says of a key besides its type and range: the
/// problems it applies to, whether every value keeps the output bits of the
/// built-in plan, whether it pins a decision (else it is a constant of a
/// rule), and its built-in value as text and in heuristic_params_t.
struct key_doc_t {
    const char *name;
    const char *applies; ///< both, bf16 or int8
    bool same_bits;
    bool pin;
    const char *builtin;
    int64_t (*value)(const heuristic_params_t &);
};

using hp_t = heuristic_params_t;

/// In the order of native_config_keys().
const key_doc_t kKeyDocs[] = {
        {"nr", "both", true, true, "auto", &field_value<int, &hp_t::nr>},
        {"mr", "both", true, true, "auto", &field_value<int, &hp_t::mr>},
        {"mb", "both", true, true, "auto", &field_value<int, &hp_t::mb>},
        {"kc", "bf16", true, true, "auto", &field_value<int, &hp_t::kc>},
        {"col_groups", "both", true, true, "auto",
                &field_value<int, &hp_t::col_groups>},
        {"schedule", "both", true, true, "auto",
                &field_value<native_schedule_t, &hp_t::schedule>},
        {"traversal", "both", true, true, "m_inner",
                &field_value<native_traversal_t, &hp_t::traversal>},
        {"unit_tiles", "both", true, true, "auto",
                &field_value<int, &hp_t::unit_tiles>},
        {"threads_cap", "both", true, false, "none",
                &field_value<int, &hp_t::threads_cap>},
        {"min_tiles_per_thread", "both", true, false, "1",
                &field_value<int, &hp_t::min_tiles_per_thread>},
        {"row_unit", "both", true, true, "12",
                &field_value<int, &hp_t::row_unit>},
        {"tile_rows", "both", true, false, "6",
                &field_value<int, &hp_t::tile_rows>},
        {"tall_rows", "bf16", true, false, "12",
                &field_value<int, &hp_t::tall_rows>},
        {"block_rows", "both", true, false, "192",
                &field_value<int, &hp_t::block_rows>},
        {"min_tile_cols", "both", true, false, "16",
                &field_value<int, &hp_t::min_tile_cols>},
        {"row_split", "both", true, false, "1",
                &field_value<bool, &hp_t::row_split>},
        {"min_pass_k", "both", true, false, "256",
                &field_value<int, &hp_t::min_pass_k>},
        {"bf16_l2_pct", "bf16", true, false, "50",
                &field_value<int, &hp_t::bf16_l2_pct>},
        {"kc_split", "bf16", true, false, "budget",
                &field_value<native_kc_split_t, &hp_t::kc_split>},
        {"l3_pct", "both", true, false, "100",
                &field_value<int, &hp_t::l3_pct>},
        {"domain_threads", "both", true, false, "l3_domain_cores",
                &field_value<int, &hp_t::domain_threads>},
        {"int8_mr", "int8", true, false, "12",
                &field_value<int, &hp_t::int8_mr>},
        {"int8_cw", "int8", true, false, "32/16",
                &field_value<uint8_t, &hp_t::int8_cw_mask>},
        {"int8_l2_pct", "int8", true, false, "50",
                &field_value<int, &hp_t::int8_l2_pct>},
        {"int8_kc", "int8", true, true, "auto",
                &field_value<int, &hp_t::int8_kc>},
        {"int4_l1_pct", "int8", true, false, "125",
                &field_value<int, &hp_t::int4_l1_pct>},
        {"direct_store", "int8", true, false, "1",
                &field_value<bool, &hp_t::direct_store>},
        // The bias then follows the dequantization instead of fusing with
        // it in one FMA.
        {"bias_in_dequant", "int8", false, false, "1",
                &field_value<bool, &hp_t::bias_in_dequant>},
        {"copy_block", "both", true, false, "64",
                &field_value<int, &hp_t::copy_block>},
        {"bf16_copy_par_min", "bf16", true, false, "0",
                &field_value<int64_t, &hp_t::bf16_copy_par_min>},
        {"int8_copy_par_min", "int8", true, false, "65536",
                &field_value<int64_t, &hp_t::int8_copy_par_min>},
        {"makespan", "both", true, false, "1",
                &field_value<int, &hp_t::makespan>},
        {"min_gain_pct", "both", true, false, "5",
                &field_value<int, &hp_t::min_gain_pct>},
        {"idle_pct", "both", true, false, "10",
                &field_value<int, &hp_t::idle_pct>},
        {"decode_m", "both", true, false, "128",
                &field_value<int, &hp_t::decode_m>},
        {"l1_pass", "both", true, false, "1",
                &field_value<int, &hp_t::l1_pass>},
        {"l1_pct", "both", true, false, "100",
                &field_value<int, &hp_t::l1_pct>},
        {"l1_pass_max_mb", "both", true, false, "64",
                &field_value<int, &hp_t::l1_pass_max_mb>},
        {"traffic_groups", "both", true, false, "1",
                &field_value<int, &hp_t::traffic_groups>},
        {"bf16_alias", "bf16", true, false, "1",
                &field_value<int, &hp_t::bf16_alias>},
        {"bf16_alias_mr", "bf16", true, false, "8",
                &field_value<int, &hp_t::bf16_alias_mr>},
        {"bf16_narrow_n", "bf16", true, false, "1",
                &field_value<int, &hp_t::bf16_narrow_n>},
        {"bf16_narrow_n_min_m", "bf16", true, false, "1024",
                &field_value<int, &hp_t::bf16_narrow_n_min_m>},
        {"blocks", "both", true, false, "1", &field_value<int, &hp_t::blocks>},
        {"bf16_blocks_min_m", "bf16", true, false, "1024",
                &field_value<int, &hp_t::bf16_blocks_min_m>},
        {"int8_blocks_min_m", "int8", true, false, "3072",
                &field_value<int, &hp_t::int8_blocks_min_m>},
        {"cost_model", "both", true, false, "1",
                &field_value<int, &hp_t::cost_model>},
        {"cm_gain_pct", "both", true, false, "5",
                &field_value<int, &hp_t::cm_gain_pct>},
        {"cm_short_rows", "both", true, false, "8",
                &field_value<int, &hp_t::cm_short_rows>},
        {"cm_fma_pipes", "both", true, false, "fma_ports",
                &field_value<int, &hp_t::cm_fma_pipes>},
        {"cm_fma_lat", "both", true, false, "6",
                &field_value<int, &hp_t::cm_fma_lat>},
        {"cm_call_cycles", "both", true, false, "800",
                &field_value<int, &hp_t::cm_call_cycles>},
        {"cm_tall_pct", "both", true, false, "domain_threads",
                &field_value<int, &hp_t::cm_tall_pct>},
        {"cm_l2_bytes", "both", true, false, "16",
                &field_value<int, &hp_t::cm_l2_bytes>},
        {"cm_l3_cycles", "both", true, false, "10",
                &field_value<int, &hp_t::cm_l3_cycles>},
        {"cm_mem_cycles", "both", true, false, "domain_threads",
                &field_value<int, &hp_t::cm_mem_cycles>},
        {"cm_prefetch_kib", "both", true, false, "4",
                &field_value<int, &hp_t::cm_prefetch_kib>},
        {"cm_overlap_pct", "both", true, false, "20",
                &field_value<int, &hp_t::cm_overlap_pct>},
        {"cm_reuse_pct", "both", true, false, "50",
                &field_value<int, &hp_t::cm_reuse_pct>},
        {"cm_stream_cycles", "both", true, false, "600",
                &field_value<int, &hp_t::cm_stream_cycles>},
        {"cm_sync_tiles", "both", true, false, "5",
                &field_value<int, &hp_t::cm_sync_tiles>},
        {"cm_group_slab_pct", "both", true, false, "75",
                &field_value<int, &hp_t::cm_group_slab_pct>},
        {"cm_group_l3_pct", "both", true, false, "75",
                &field_value<int, &hp_t::cm_group_l3_pct>},
        {"cm_group_skew_pct", "both", true, false, "110",
                &field_value<int, &hp_t::cm_group_skew_pct>},
        {"cm_group_rows_pct", "both", true, false, "115",
                &field_value<int, &hp_t::cm_group_rows_pct>},
        {"cm_drift_kib", "bf16", true, false, "4096",
                &field_value<int, &hp_t::cm_drift_kib>},
        {"cm_pass_kib", "bf16", true, false, "128",
                &field_value<int, &hp_t::cm_pass_kib>},
        {"cm_pass_calls", "bf16", true, false, "4",
                &field_value<int, &hp_t::cm_pass_calls>},
        {"cm_block_kib", "bf16", true, false, "16384",
                &field_value<int, &hp_t::cm_block_kib>},
        {"cm_int8_l1_pct", "int8", true, false, "0",
                &field_value<int, &hp_t::cm_int8_l1_pct>},
        {"cm_int4_min_m", "int8", true, false, "48",
                &field_value<int, &hp_t::cm_int4_min_m>},
};

/// The doc of key, nullptr for a key it does not list.
const key_doc_t *key_doc(const native_config_key_t &k) {
    for (const key_doc_t &d : kKeyDocs)
        if (std::string(d.name) == k.name) return &d;
    return nullptr;
}

/// A legal value of key, other than its built-in one.
std::string sample_value(const native_config_key_t &k) {
    const key_doc_t *doc = key_doc(k);
    const int64_t builtin
            = doc != nullptr ? doc->value(heuristic_params_t {}) : 0;
    switch (k.type) {
        case native_key_type_t::int32:
        case native_key_type_t::int64:
            return std::to_string(k.max != builtin ? k.max : k.min);
        case native_key_type_t::boolean: return builtin != 0 ? "0" : "1";
        case native_key_type_t::cw_list: return "16";
        default: {
            // The sample of col_groups pins it, which only the row-group
            // schedule allows.
            if (k.type == native_key_type_t::schedule) return "row_groups";
            const std::vector<std::string> names
                    = native_brgemm_test::native_enum_values(k.type);
            return names[builtin == 0 ? names.size() - 1 : 0];
        }
    }
}

} // namespace

TEST(NativeConfigText, RoundTripsEveryKey) {
    size_t count = 0;
    const native_config_key_t *keys = native_config_keys(count);
    ASSERT_GT(count, 0u);
    std::string all;
    for (size_t i = 0; i < count; ++i) {
        const std::string text
                = std::string(keys[i].name) + "=" + sample_value(keys[i]);
        native_config_t c;
        std::string why;
        ASSERT_TRUE(parse_native_config(text, c, why)) << text << ": " << why;
        EXPECT_FALSE(c.empty()) << text;
        EXPECT_EQ(format_native_config(c), text);
        native_config_t again;
        ASSERT_TRUE(parse_native_config(format_native_config(c), again, why));
        EXPECT_TRUE(again == c) << text;
        all += (all.empty() ? "" : ",") + text;
    }
    // Every key at once comes back in schema order.
    native_config_t c = config(all);
    EXPECT_EQ(format_native_config(c), all);
    EXPECT_TRUE(native_config_t {}.empty());
    EXPECT_EQ(format_native_config(native_config_t {}), "");
    EXPECT_TRUE(parses("") && parses("  "));
}

// The key docs of the schema dump list every key, in order; each built-in
// value is heuristic_params_t's, and each key moves the field its doc reads.
TEST(NativeConfigText, KeyDocsMatchTheKeyTable) {
    size_t count = 0;
    const native_config_key_t *keys = native_config_keys(count);
    ASSERT_EQ(count, std::size(kKeyDocs));
    const heuristic_params_t defaults;
    for (size_t i = 0; i < count; ++i) {
        const native_config_key_t &k = keys[i];
        const key_doc_t &d = kKeyDocs[i];
        ASSERT_EQ(std::string(k.name), d.name) << i;
        const int64_t v = d.value(defaults);
        int64_t text_value = 0;
        const bool integer = k.type == native_key_type_t::int32
                || k.type == native_key_type_t::int64;
        if (integer && !native_text_int64(d.builtin, text_value)) {
            // -1: col_groups by the schedule, cm_tall_pct by the L3 domain.
            const bool minus_one = std::string(k.name) == "col_groups"
                    || std::string(k.name) == "cm_tall_pct";
            EXPECT_EQ(v, minus_one ? -1 : 0) << k.name;
        } else {
            const native_config_t c
                    = config(std::string(k.name) + "=" + d.builtin);
            EXPECT_EQ(d.value(apply_native_config(defaults, c)), v) << k.name;
        }
        const native_config_t other
                = config(std::string(k.name) + "=" + sample_value(k));
        EXPECT_NE(d.value(apply_native_config(defaults, other)), v) << k.name;
        const std::vector<std::string> names
                = native_brgemm_test::native_enum_values(k.type);
        for (const std::string &n : names)
            EXPECT_TRUE(parses(std::string(k.name) + "=" + n)) << k.name;
        if (!names.empty()) {
            EXPECT_FALSE(parses(std::string(k.name) + "=x" + names[0]));
        }
    }
}

TEST(NativeConfigText, RejectsBadTokensAndRanges) {
    for (const char *bad : {"nr=24", "nr=8", "nr=48", "nr=96", "nr=256",
                 "nr=16,nr=32", "foo=1", "nr", "nr=", "=16", "nr=16,", ",nr=16",
                 "nr=16;mb=8", "nr=1 6", "nr=16x", "nr=+16", "mb=99999999999",
                 "mr=0", "mr=13", "kc=1", "kc=1073741825", "col_groups=-1",
                 "col_groups=1025", "schedule=rows", "traversal=k_inner",
                 "threads_cap=0", "min_tiles_per_thread=65", "row_unit=0",
                 "tile_rows=13", "tall_rows=0", "block_rows=8193",
                 "min_tile_cols=8", "min_tile_cols=64", "row_split=2",
                 "row_split=yes", "min_pass_k=255", "min_pass_k=0",
                 "bf16_l2_pct=0", "bf16_l2_pct=1001", "kc_split=half",
                 "l3_pct=0", "domain_threads=0", "domain_threads=1025",
                 "int8_mr=13", "int8_mr_wide=1", "int8_cw=32", "int8_cw=64/16",
                 "int8_cw=8/16", "int8_cw=", "int8_cw=16/", "int8_l2_pct=0",
                 "int8_kc=6", "int8_kc=0", "direct_store=-1",
                 "bias_in_dequant=2", "copy_block=0", "copy_block=4097",
                 "bf16_copy_par_min=-1", "int8_copy_par_min=1099511627777",
                 "split=0", "k_split=1", "makespan=2", "makespan=-1",
                 "min_gain_pct=101", "idle_pct=0", "idle_pct=101",
                 "decode_m=-1", "l1_pass=on", "l1_pct=0", "l1_pct=1001",
                 "l1_pass_max_mb=0", "l1_pass_max_mb=8193", "traffic_groups=2",
                 "bf16_alias=2", "bf16_alias=-1", "bf16_alias_mr=0",
                 "bf16_alias_mr=13", "bf16_narrow_n=2", "bf16_narrow_n=-1",
                 "bf16_narrow_n_min_m=0", "bf16_narrow_n_min_m=1073741825",
                 "unit_tiles=0", "unit_tiles=4097", "blocks=2", "blocks=-1",
                 "bf16_blocks_min_m=0", "int8_blocks_min_m=1073741825"}) {
        EXPECT_FALSE(parses(bad)) << bad;
    }
    // Every integer key rejects one past each end of its range.
    size_t count = 0;
    const native_config_key_t *keys = native_config_keys(count);
    for (size_t i = 0; i < count; ++i) {
        const native_config_key_t &k = keys[i];
        if (k.type != native_key_type_t::int32
                && k.type != native_key_type_t::int64)
            continue;
        const std::string name = std::string(k.name) + "=";
        EXPECT_FALSE(parses(name + std::to_string(k.min - 1))) << k.name;
        EXPECT_FALSE(parses(name + std::to_string(k.max + 1))) << k.name;
        EXPECT_TRUE(parses(name + std::to_string(k.min))) << k.name;
        EXPECT_TRUE(parses(name + std::to_string(k.max))) << k.name;
        if (k.step > 1) {
            EXPECT_FALSE(parses(name + std::to_string(k.min + 1))) << k.name;
        }
    }
    native_config_t c;
    std::string why;
    EXPECT_FALSE(parse_native_config("foo=1", c, why));
    EXPECT_NE(why.find("foo"), std::string::npos) << why;
}

TEST(NativeConfigText, CrossFieldRules) {
    EXPECT_FALSE(parses("schedule=slice,col_groups=4"));
    EXPECT_FALSE(parses("col_groups=0,schedule=row_groups"));
    EXPECT_FALSE(parses("schedule=blocks,col_groups=0"));
    EXPECT_FALSE(parses("col_groups=2,schedule=blocks"));
    EXPECT_TRUE(parses("schedule=blocks,unit_tiles=4"));
    EXPECT_TRUE(parses("schedule=slice,col_groups=0"));
    EXPECT_TRUE(parses("schedule=row_groups,col_groups=4"));
    EXPECT_TRUE(parses("schedule=auto,col_groups=0"));
    EXPECT_TRUE(parses("schedule=auto,col_groups=8"));
    native_config_t c;
    c.schedule = native_schedule_t::slice;
    c.col_groups = 2;
    std::string why;
    EXPECT_FALSE(check_native_config(c, why));
    c.col_groups = 0;
    EXPECT_TRUE(check_native_config(c, why));
}

TEST(NativeConfigText, KeysAndValuesIgnoreCaseAndSpaces) {
    const native_config_t c = config(
            " NR = 16 , Schedule = ROW_GROUPS , Row_Split = Off , "
            "int8_cw = 16 / 32 , direct_store = TRUE , Blocks = 0 ");
    EXPECT_EQ(format_native_config(c),
            "nr=16,schedule=row_groups,row_split=0,int8_cw=32/16,"
            "direct_store=1,blocks=0");
}

TEST(NativeConfigText, RuleSwitchesSetOff) {
    const heuristic_params_t off = apply_native_config(
            heuristic_params_t {}, config(native_brgemm_test::kNativeRulesOff));
    EXPECT_EQ(off.makespan, 0);
    EXPECT_EQ(off.l1_pass, 0);
    EXPECT_EQ(off.traffic_groups, 0);
    EXPECT_EQ(off.bf16_alias, 0);
    EXPECT_EQ(off.bf16_narrow_n, 0);
    EXPECT_EQ(off.blocks, 0);
    // The rule constants take any legal value; the switches stay as they
    // are.
    const heuristic_params_t tuned = apply_native_config(heuristic_params_t {},
            config("min_gain_pct=0,idle_pct=100,decode_m=0,l1_pct=1000,"
                   "l1_pass_max_mb=8192,bf16_alias_mr=12,"
                   "bf16_narrow_n_min_m=1,bf16_blocks_min_m=1,"
                   "int8_blocks_min_m=1073741824"));
    EXPECT_EQ(tuned.makespan, 1);
    EXPECT_EQ(tuned.min_gain_pct, 0);
    EXPECT_EQ(tuned.idle_pct, 100);
    EXPECT_EQ(tuned.decode_m, 0);
    EXPECT_EQ(tuned.l1_pct, 1000);
    EXPECT_EQ(tuned.l1_pass_max_mb, 8192);
    EXPECT_EQ(tuned.bf16_alias_mr, 12);
    EXPECT_EQ(tuned.bf16_narrow_n_min_m, 1);
    EXPECT_EQ(tuned.bf16_blocks_min_m, 1);
    EXPECT_EQ(tuned.int8_blocks_min_m, 1073741824);
}

TEST(NativeConfigText, MergeAndApply) {
    native_config_t base = config("nr=16,mb=24,kc_split=equal");
    const native_config_t overlay = config("mb=48,threads_cap=8");
    merge_native_config(base, overlay);
    EXPECT_EQ(format_native_config(base),
            "nr=16,mb=48,threads_cap=8,kc_split=equal");
    EXPECT_FALSE(base == overlay);
    native_config_t unchanged = overlay;
    merge_native_config(unchanged, native_config_t {});
    EXPECT_TRUE(unchanged == overlay);

    const heuristic_params_t hp
            = apply_native_config(heuristic_params_t {}, base);
    EXPECT_EQ(hp.nr, 16);
    EXPECT_EQ(hp.mb, 48);
    EXPECT_EQ(hp.threads_cap, 8);
    EXPECT_EQ(hp.kc_split, native_kc_split_t::equal);
    EXPECT_EQ(hp.block_rows, 192);
    EXPECT_EQ(hp.col_groups, -1);
    const heuristic_params_t same
            = apply_native_config(heuristic_params_t {}, native_config_t {});
    EXPECT_EQ(same.nr, 0);
    EXPECT_EQ(same.int8_copy_par_min, 65536);
    EXPECT_EQ(same.int8_cw_mask, kNativeInt8CwAll);
}

TEST(NativeConfigValidate, BuiltInPlansAreLegal) {
    if (!has_isa()) GTEST_SKIP() << "the CPU lacks AVX512-BF16 or AVX512-VNNI";
    const auto planner = make_native_planner(nullptr, nullptr);
    ASSERT_FALSE(planner->has_sources());
    for (const bool bf16 : {true, false})
        for (const int M : {1, 2, 5, 7, 12, 13, 16, 24, 64, 100, 192, 193, 512,
                     1024, 4096})
            for (const int N : {32, 64, 96, 1024, 4096, 14336})
                for (const int K : {64, 100, 4096, 14336})
                    for (const int T : {1, 8, 24, 64, 128, 256}) {
                        const native_problem_t p = problem(M, N, K, bf16, T);
                        const native_plan_t q = heuristic_plan(
                                p, native_plan_uarch(), heuristic_params_t {});
                        const char *why = validate_native_plan(q, p);
                        ASSERT_EQ(why, nullptr)
                                << why << " M=" << M << " N=" << N << " K=" << K
                                << " T=" << T << " bf16=" << bf16;
                        ASSERT_TRUE(same_plan(planner->plan(p), q));
                    }
}

TEST(NativeConfigValidate, RejectsEachRule) {
    if (!has_isa()) GTEST_SKIP() << "the CPU lacks AVX512-BF16 or AVX512-VNNI";
    const UarchParams &u = detect_uarch();
    const native_problem_t bp = problem(256, 4096, 4096, true, 24);
    const native_plan_t b = heuristic_plan(bp, u, heuristic_params_t {});
    ASSERT_EQ(validate_native_plan(b, bp), nullptr);
    const auto rejects
            = [](const native_problem_t &p, native_plan_t q, auto &&change) {
        change(q);
        return validate_native_plan(q, p) != nullptr;
    };
    // Whether rule 1 (the tile width) is the rule a plan breaks.
    const auto width_rejects
            = [](const native_plan_t &q, const native_problem_t &p) {
        const char *why = validate_native_plan(q, p);
        return why != nullptr && std::string(why).rfind("nr ", 0) == 0;
    };
    // 1. nr is 16 or 32 and divides pack_nr; a BF16 tile of 64 or 128
    // columns takes 32-column slabs inside its pack blocks: of 64- or
    // 128-column blocks only.
    native_plan_t n48 = b;
    n48.nr = 48;
    EXPECT_TRUE(width_rejects(n48, bp));
    for (const int nr : {64, 128}) {
        heuristic_params_t wide_tiles;
        wide_tiles.nr = nr;
        EXPECT_EQ(validate_native_plan(heuristic_plan(bp, u, wide_tiles), bp),
                nullptr)
                << nr;
        for (const int pack_nr : {16, 48, 64, 128}) {
            native_problem_t other = bp;
            other.pack_nr = pack_nr;
            const native_plan_t q = heuristic_plan(other, u, wide_tiles);
            if (pack_nr == 64 || pack_nr == 128) {
                EXPECT_EQ(validate_native_plan(q, other), nullptr)
                        << nr << " columns of " << pack_nr << "-column blocks";
            } else {
                EXPECT_TRUE(width_rejects(q, other))
                        << nr << " columns of " << pack_nr << "-column blocks";
            }
        }
    }
    // 2. The grid covers the problem.
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) { ++q.n_tiles; }));
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) { ++q.m_blocks; }));
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) { q.mb = 0; }));
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) { q.mb = 257; }));
    // 3. Rows per call, and a BF16 kernel for every row count up to mr.
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) { q.mr = 0; }));
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) { q.mr = 13; }));
    heuristic_params_t narrow;
    narrow.nr = 16;
    narrow.mr = 6;
    EXPECT_EQ(validate_native_plan(heuristic_plan(bp, u, narrow), bp), nullptr);
    narrow.mr = 7;
    EXPECT_NE(validate_native_plan(heuristic_plan(bp, u, narrow), bp), nullptr);
    // 4. A BF16 K pass below K is even.
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) {
        q.kc = 255;
        q.scratch_floats = static_cast<size_t>(q.mb) * q.nr;
    }));
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) {
        q.kc = 1;
        q.scratch_floats = static_cast<size_t>(q.mb) * q.nr;
    }));
    // 5. The team.
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) { q.threads = 0; }));
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) { q.threads = 25; }));
    // 6. Column groups divide the team and the column tiles.
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) { q.col_groups = 5; }));
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) { q.col_groups = -1; }));
    // 7. Units and copy sizes.
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) { q.row_unit = 0; }));
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) { q.copy_block = 0; }));
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) { q.copy_par_min = -1; }));
    // 10. The scratch.
    EXPECT_TRUE(rejects(bp, b, [](native_plan_t &q) { ++q.scratch_floats; }));
    // 11. The blocks schedule: at most n_tiles tiles per unit, no column
    // groups, domains of at least one thread, and with K passes the
    // accumulators of every tile of a unit.
    heuristic_params_t blocks;
    blocks.schedule = native_schedule_t::blocks;
    blocks.unit_tiles = 3;
    blocks.kc = 1024;
    const native_plan_t k = heuristic_plan(bp, u, blocks);
    ASSERT_EQ(validate_native_plan(k, bp), nullptr);
    ASSERT_EQ(k.unit_tiles, 3);
    ASSERT_LT(k.kc, bp.K);
    EXPECT_EQ(k.col_groups, 0);
    EXPECT_EQ(k.scratch_floats, static_cast<size_t>(k.mb) * k.nr * 3);
    EXPECT_TRUE(rejects(bp, k, [](native_plan_t &q) { q.unit_tiles = -1; }));
    EXPECT_TRUE(rejects(bp, k, [](native_plan_t &q) {
        q.unit_tiles = q.n_tiles + 1;
        q.scratch_floats = static_cast<size_t>(q.mb) * q.nr * q.unit_tiles;
    }));
    EXPECT_TRUE(rejects(bp, k, [](native_plan_t &q) { q.col_groups = 2; }));
    EXPECT_TRUE(rejects(bp, k, [](native_plan_t &q) { q.domain_threads = 0; }));
    EXPECT_TRUE(rejects(bp, k, [](native_plan_t &q) {
        q.scratch_floats = static_cast<size_t>(q.mb) * q.nr;
    }));
    // A unit of every tile; under BF16 K passes at most the tiles whose
    // accumulators fit the bf16_l2_pct share of L2.
    UarchParams l2 = u;
    l2.l2_bytes = 1 << 20;
    blocks.kc = bp.K;
    blocks.unit_tiles = 100000;
    const native_plan_t one = heuristic_plan(bp, l2, blocks);
    ASSERT_EQ(validate_native_plan(one, bp), nullptr);
    EXPECT_EQ(one.unit_tiles, one.n_tiles);
    EXPECT_EQ(one.scratch_floats, static_cast<size_t>(one.mr) * one.nr);
    blocks.kc = 1024;
    const native_plan_t bound = heuristic_plan(bp, l2, blocks);
    ASSERT_EQ(validate_native_plan(bound, bp), nullptr);
    const int64_t share = l2.l2_bytes / 2;
    const int64_t tile_bytes
            = static_cast<int64_t>(bound.mb) * bound.nr * sizeof(float);
    EXPECT_EQ(bound.unit_tiles,
            std::clamp<int64_t>(share / tile_bytes, 1, bound.n_tiles));
    EXPECT_LT(bound.unit_tiles, bound.n_tiles);
    EXPECT_LE(
            static_cast<int64_t>(bound.scratch_floats * sizeof(float)), share);

    // 8, 9. INT8 widths, rows per call and K passes.
    const native_problem_t ip = problem(64, 4096, 4096, false, 24);
    const native_plan_t i = heuristic_plan(ip, u, heuristic_params_t {});
    ASSERT_EQ(validate_native_plan(i, ip), nullptr);
    ASSERT_EQ(i.nr, 32);
    // 64 columns are BF16 only: an INT8 plan of 64 columns, its grid and
    // scratch made for them, breaks the width rule.
    native_plan_t i64 = i;
    i64.nr = 64;
    i64.n_tiles = ip.N / 64;
    i64.scratch_floats = native_plan_scratch_floats(i64, ip);
    EXPECT_TRUE(width_rejects(i64, ip));
    EXPECT_TRUE(rejects(
            ip, i, [](native_plan_t &q) { q.int8_cw_mask = kNativeInt8Cw32; }));
    EXPECT_TRUE(rejects(ip, i, [](native_plan_t &q) { q.int8_rows_nv1 = 13; }));
    EXPECT_TRUE(rejects(ip, i, [](native_plan_t &q) { q.int8_rows_nv1 = 0; }));
    EXPECT_TRUE(rejects(ip, i, [](native_plan_t &q) { q.int8_rows_nv2 = 13; }));
    EXPECT_TRUE(
            rejects(ip, i, [](native_plan_t &q) { q.int8_pass_quads = -1; }));
    EXPECT_TRUE(
            rejects(ip, i, [](native_plan_t &q) { q.int8_pass_budget = -1; }));
    // A per-group problem takes the group kernel's rows per call, 12 at 16
    // columns and 6 at 32; more rows at 32 columns are illegal for it, and
    // the per-channel plan with them too.
    native_problem_t gp = ip;
    gp.int8_scheme = native_int8_scheme_t::per_group;
    const native_plan_t g = heuristic_plan(gp, u, heuristic_params_t {});
    EXPECT_EQ(g.int8_rows_nv1, 12);
    EXPECT_EQ(g.int8_rows_nv2, 6);
    ASSERT_EQ(validate_native_plan(g, gp), nullptr);
    EXPECT_TRUE(rejects(gp, g, [](native_plan_t &q) { q.int8_rows_nv2 = 7; }));
    EXPECT_NE(validate_native_plan(i, gp), nullptr);
}

TEST(NativeConfigOverride, FiltersAndFirstMatch) {
    const auto src = make_native_override_source(
            "m=1-16,n=4096,k=4096:nr=16,threads_cap=32;dtype=s8:int8_mr=8;"
            "m=1024-:schedule=row_groups,col_groups=8,row_unit=24");
    ASSERT_NE(src, nullptr);
    const native_config_entry_t *e = src->lookup(problem(8, 4096, 4096));
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(format_native_config(e->config), "nr=16,threads_cap=32");
    EXPECT_NE(e->label.find("ZENDNNL_MATMUL_NATIVE_CONFIG clause 1 "),
            std::string::npos)
            << e->label;
    EXPECT_EQ(config_of(src.get(), problem(17, 4096, 4096)), "<none>");
    // The first clause whose filters hold wins.
    EXPECT_EQ(config_of(src.get(), problem(8, 4096, 4096, false)),
            "nr=16,threads_cap=32");
    EXPECT_EQ(
            config_of(src.get(), problem(64, 4096, 4096, false)), "int8_mr=8");
    EXPECT_EQ(config_of(src.get(), problem(2048, 1024, 1024)),
            "col_groups=8,schedule=row_groups,row_unit=24");
    const auto all = make_native_override_source("nr=16,mb=24");
    EXPECT_EQ(
            config_of(all.get(), problem(1, 32, 16, false, 1)), "nr=16,mb=24");
}

TEST(NativeConfigOverride, EveryFilter) {
    const auto holds = [](const char *filters, const native_problem_t &p) {
        const auto src
                = make_native_override_source(std::string(filters) + ":nr=16");
        EXPECT_NE(src, nullptr) << filters;
        return src != nullptr && src->lookup(p) != nullptr;
    };
    const native_problem_t p = problem(5, 4096, 2048, true, 24);
    EXPECT_TRUE(holds("m=5", p));
    EXPECT_FALSE(holds("m=6", p));
    EXPECT_TRUE(holds("m=3-5", p));
    EXPECT_FALSE(holds("m=6-9", p));
    EXPECT_TRUE(holds("m=5-", p));
    EXPECT_FALSE(holds("m=6-", p));
    EXPECT_TRUE(holds("n=4096,k=2048,threads=24,pack_nr=32", p));
    EXPECT_FALSE(holds("n=4095", p));
    EXPECT_FALSE(holds("k=2049-", p));
    EXPECT_FALSE(holds("threads=25-", p));
    EXPECT_FALSE(holds("pack_nr=64", p));
    EXPECT_TRUE(holds("dtype=bf16/s8", p));
    EXPECT_FALSE(holds("dtype=s8", p));
    EXPECT_TRUE(holds("src=bf16", p));
    EXPECT_FALSE(holds("src=u8/s8/f32", p));
    EXPECT_TRUE(holds("dst=bf16", p));
    EXPECT_FALSE(holds("dst=f32/f16/u8/s8/s32", p));
    EXPECT_TRUE(holds("epilogue=store", p));
    EXPECT_FALSE(holds("epilogue=bias/postops/gated", p));
    EXPECT_TRUE(holds("trans_a=0,dynamic=false,has_bias=off", p));
    EXPECT_FALSE(holds("trans_a=1", p));
    native_problem_t q = problem(5, 4096, 2048, false, 24);
    q.src_dt = data_type_t::f32;
    q.dst_dt = data_type_t::f32;
    q.dynamic_quant = true;
    q.trans_a = true;
    q.has_bias = true;
    q.epilogue = native_epilogue_class_t::gated;
    EXPECT_TRUE(
            holds("dtype=s8,src=f32,dst=f32,dynamic=1,trans_a=on,"
                  "has_bias=true,epilogue=gated",
                    q));
    EXPECT_FALSE(holds("dynamic=0", q));
    EXPECT_FALSE(holds("has_bias=0", q));
    EXPECT_TRUE(holds(" M = 1-16 , DTYPE = S8 , Epilogue = Gated ", q));
    // The group size of an s4 weight; 0 for the others.
    const native_problem_t g = int4_problem(5, 4096, 2048, 128);
    EXPECT_TRUE(holds("dtype=s4,group=128,src=bf16,dynamic=1", g));
    EXPECT_TRUE(holds("group=64-256", g));
    EXPECT_FALSE(holds("group=32", g));
    EXPECT_FALSE(holds("dtype=s8", g));
    EXPECT_TRUE(holds("dtype=s8/s4", g));
    EXPECT_TRUE(holds("group=0", p));
    EXPECT_FALSE(holds("group=1-", q));
}

TEST(NativeConfigOverride, EmptySettingsExemptLaterClauses) {
    const auto src = make_native_override_source("m=1-16:;nr=16");
    ASSERT_NE(src, nullptr);
    const native_config_entry_t *e = src->lookup(problem(8, 4096, 4096));
    ASSERT_NE(e, nullptr);
    EXPECT_TRUE(e->config.empty());
    EXPECT_NE(e->label.find("clause 1 "), std::string::npos) << e->label;
    EXPECT_EQ(config_of(src.get(), problem(64, 4096, 4096)), "nr=16");
}

TEST(NativeConfigOverride, DropsBadClausesWithOneWarningEach) {
    std::unique_ptr<native_config_source_t> src;
    EXPECT_EQ(warnings_of([&] {
        src = make_native_override_source("m=1-16:nr=24;nr=16");
    }),
            shown(1));
    ASSERT_NE(src, nullptr);
    const native_config_entry_t *e = src->lookup(problem(8, 4096, 4096));
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(format_native_config(e->config), "nr=16");
    EXPECT_NE(e->label.find("clause 2 "), std::string::npos) << e->label;
    for (const char *bad : {"q=1:nr=16", "m=a:nr=16", "m=5-2:nr=16",
                 "m=-5:nr=16", "m=:nr=16", "dtype=f64:nr=16",
                 "dtype=bf16/:nr=16", "trans_a=maybe:nr=16", ":nr=16",
                 "m=1,m=2:nr=16", "m=1,,n=2:nr=16", "m=1:nr=16:mb=8",
                 "nr=16,foo=1", "m=1-16", "m=1:schedule=slice,col_groups=4"}) {
        EXPECT_EQ(warnings_of([&] {
            EXPECT_EQ(make_native_override_source(bad), nullptr) << bad;
        }),
                shown(1))
                << bad;
    }
    EXPECT_EQ(make_native_override_source(""), nullptr);
    EXPECT_EQ(warnings_of([] {
        EXPECT_EQ(make_native_override_source(" ; ;"), nullptr);
    }),
            0);
}

TEST(NativeConfigOverride, ReadsAtMost64Clauses) {
    std::string text;
    for (int i = 1; i <= 70; ++i)
        text += "m=" + std::to_string(i) + ":nr=16;";
    std::unique_ptr<native_config_source_t> src;
    EXPECT_EQ(warnings_of([&] { src = make_native_override_source(text); }),
            shown(1));
    ASSERT_NE(src, nullptr);
    const native_config_entry_t *e = src->lookup(problem(64, 4096, 4096));
    ASSERT_NE(e, nullptr);
    EXPECT_NE(e->label.find("clause 64 "), std::string::npos) << e->label;
    EXPECT_EQ(src->lookup(problem(65, 4096, 4096)), nullptr);
}

TEST(NativeConfigTuningFile, FormatAndVersion) {
    const std::string entries
            = R"("entries": [{"id": "all", "config": {"nr": 16}}])";
    native_tuning_sources_t s = load(tuning(entries));
    ASSERT_NE(s.table, nullptr);
    EXPECT_EQ(s.rules, nullptr);
    EXPECT_EQ(config_of(s.table.get(), problem(8, 4096, 4096)), "nr=16");
    for (const std::string &bad :
            {std::string("{\"version\": 2, ") + entries + "}",
                    std::string("{\"format\": \"other\", \"version\": 1, ")
                            + entries + "}",
                    std::string("{\"format\": "
                                "\"zendnnl-native-brgemm-tuning\", "
                                "\"version\": 3, ")
                            + entries + "}",
                    std::string("{\"format\": "
                                "\"zendnnl-native-brgemm-tuning\", "
                                "\"version\": 1, ")
                            + entries + "}",
                    std::string("{\"format\": "
                                "\"zendnnl-native-brgemm-tuning\", "
                                "\"version\": 0, ")
                            + entries + "}",
                    std::string("{\"format\": "
                                "\"zendnnl-native-brgemm-tuning\", "
                                "\"version\": 2.0, ")
                            + entries + "}",
                    std::string("{\"format\": "
                                "\"zendnnl-native-brgemm-tuning\", "
                                "\"version\": \"1\", ")
                            + entries + "}",
                    std::string("{\"format\": "
                                "\"zendnnl-native-brgemm-tuning\", "
                                "\"version\": 1.0, ")
                            + entries + "}",
                    std::string("{\"format\": "
                                "\"zendnnl-native-brgemm-tuning\", ")
                            + entries + "}",
                    std::string("{"), std::string("[]"), std::string(""),
                    std::string("{\"format\": "
                                "\"zendnnl-native-brgemm-tuning\", "
                                "\"version\": 2, // comment\n}")}) {
        EXPECT_EQ(warnings_of([&] {
            s = load(bad);
            EXPECT_EQ(s.table, nullptr) << bad;
            EXPECT_EQ(s.rules, nullptr) << bad;
        }),
                shown(1))
                << bad;
    }
    // An unknown top-level key is ignored with one warning; meta is free-form.
    EXPECT_EQ(warnings_of([&] {
        s = load(tuning(R"("extra": [1], "meta": {"tool": "x", "n": [1, 2]}, )"
                + entries));
    }),
            shown(1));
    EXPECT_NE(s.table, nullptr);
}

namespace {

/// A version 2 file with the given hardware block and an entry for every
/// problem.
std::string tuning_with_hardware(const std::string &hardware) {
    return tuning(R"("hardware": )" + hardware
            + R"(, "entries": [{"config": {"nr": 16}}])");
}

bool hardware_applies(const std::string &hardware, const native_hw_t &hw) {
    return load(tuning_with_hardware(hardware), hw).table != nullptr;
}

} // namespace

TEST(NativeConfigTuningFile, HardwareMatch) {
    const native_hw_t hw = test_hw();
    EXPECT_TRUE(hardware_applies("{}", hw));
    EXPECT_TRUE(
            hardware_applies(R"({"isa": ["avx512bf16", "avx512vnni"]})", hw));
    EXPECT_FALSE(hardware_applies(R"({"isa": ["avx512bf16", "amx"]})", hw));
    native_hw_t no_vnni = hw;
    no_vnni.uarch.avx512vnni = false;
    EXPECT_FALSE(hardware_applies(R"({"isa": ["avx512vnni"]})", no_vnni));
    EXPECT_TRUE(
            hardware_applies(R"({"l1d_kib": 48, "l2_kib": 1024, "l3_kib": 32768,
                           "l3_threads": 16, "smt": 2, "cores": 128})",
                    hw));
    EXPECT_FALSE(hardware_applies(R"({"l2_kib": 2048})", hw));
    EXPECT_TRUE(
            hardware_applies(R"({"l2_kib": {"min": 512, "max": 2048}})", hw));
    EXPECT_FALSE(hardware_applies(R"({"l2_kib": {"min": 2048}})", hw));
    EXPECT_TRUE(hardware_applies(R"({"cores": {"min": 64}})", hw));
    EXPECT_FALSE(hardware_applies(R"({"cores": {"max": 64}})", hw));
    EXPECT_FALSE(hardware_applies(R"({"l3_threads": 8})", hw));
    EXPECT_FALSE(hardware_applies(R"({"cores": {"least": 64}})", hw));
    EXPECT_FALSE(hardware_applies(R"({"cores": "128"})", hw));
    EXPECT_FALSE(hardware_applies(R"({"frequency": 3})", hw));
    EXPECT_FALSE(hardware_applies(R"(["isa"])", hw));
    EXPECT_EQ(warnings_of([&] { hardware_applies(R"({"smt": 1})", hw); }),
            shown(1));
}

namespace {

/// test_hw() with a CPU identity.
native_hw_t identified_hw() {
    native_hw_t hw = test_hw();
    hw.vendor = "TestVendorId";
    hw.family = 26;
    hw.model = 17;
    hw.stepping = 1;
    return hw;
}

/// What f writes to stdout: the warnings, when they are on.
template <typename F>
std::string log_of(F &&f) {
    testing::internal::CaptureStdout();
    f();
    std::cout.flush();
    return testing::internal::GetCapturedStdout();
}

/// Checks that the version 2 file with the given hardware block is ignored
/// on hw with one warning that names what.
void hardware_ignored(const std::string &hardware, const native_hw_t &hw,
        const std::string &what) {
    bool applied = true;
    const std::string out
            = log_of([&] { applied = hardware_applies(hardware, hw); });
    EXPECT_FALSE(applied) << hardware;
    int lines = 0;
    for (size_t at = out.find("native BRGEMM:"); at != std::string::npos;
            at = out.find("native BRGEMM:", at + 1))
        ++lines;
    EXPECT_EQ(lines, shown(1)) << hardware << ": " << out;
    if (shown(1) == 1) {
        EXPECT_NE(out.find(what), std::string::npos) << hardware << ": " << out;
    }
}

} // namespace

TEST(NativeConfigTuningFile, HardwareFields) {
    const native_hw_t hw = identified_hw();
    EXPECT_TRUE(hardware_applies("{}", hw));
    // Each key at this CPU's value, then all of them together.
    const std::pair<const char *, const char *> same[]
            = {{"vendor", R"("TestVendorId")"}, {"family", "26"},
                    {"model", "17"}, {"stepping", "1"},
                    {"isa", R"(["avx512f", "avx512bf16", "avx512vnni"])"},
                    {"l1d_kib", "48"}, {"l2_kib", "1024"}, {"l3_kib", "32768"},
                    {"l3_cores", "8"}, {"smt", "2"}, {"l3_threads", "16"},
                    {"cores", "128"}, {"logical_cpus", "256"}};
    std::string all;
    for (const auto &[key, value] : same) {
        const std::string one = "\"" + std::string(key) + "\": " + value;
        EXPECT_TRUE(hardware_applies("{" + one + "}", hw)) << one;
        all += (all.empty() ? "" : ", ") + one;
    }
    EXPECT_TRUE(hardware_applies("{" + all + "}", hw));
    // Each key at another value: the warning names the key and this CPU's
    // value.
    const std::pair<const char *, const char *> other[] = {
            {"vendor", R"("OtherVendor0")"}, {"family", "25"}, {"model", "16"},
            {"stepping", "2"}, {"l1d_kib", "32"}, {"l2_kib", "2048"},
            {"l3_kib", "131072"}, {"l3_cores", "32"}, {"smt", "1"},
            {"l3_threads", "8"}, {"cores", "256"}, {"logical_cpus", "512"}};
    for (size_t i = 0; i < std::size(other); ++i) {
        const std::string key = other[i].first;
        hardware_ignored("{\"" + key + "\": " + other[i].second + "}", hw,
                "hardware." + key + "=" + other[i].second + " does not hold "
                        + same[i < 4 ? i : i + 1].second);
    }
    // The first key that fails, in the order of the key list.
    hardware_ignored(R"({"smt": 1, "model": 16, "l2_kib": 2048})", hw,
            "hardware.model=16 does not hold 17");
}

TEST(NativeConfigTuningFile, HardwareListsAndRanges) {
    const native_hw_t hw = identified_hw();
    for (const char *yes : {R"({"model": [16, 17, 81]})",
                 R"({"model": {"min": 17}})", R"({"model": {"max": 17}})",
                 R"({"model": {"min": 0, "max": 31}})",
                 R"({"model": {"min": 17, "max": 17}})",
                 R"({"vendor": ["OtherVendor0", "TestVendorId"]})",
                 R"({"l3_kib": {"min": 32768, "max": 131072},
                     "l3_cores": [8, 32]})",
                 R"({"family": [25, 26], "cores": {"min": 64},
                     "logical_cpus": {"max": 512}})"}) {
        EXPECT_TRUE(hardware_applies(yes, hw)) << yes;
    }
    for (const char *no : {R"({"model": [16, 81]})",
                 R"({"model": {"min": 18}})", R"({"model": {"max": 16}})",
                 R"({"model": {"min": 32, "max": 47}})",
                 R"({"vendor": ["OtherVendor0"]})",
                 R"({"vendor": "testvendorid"})",
                 R"({"l3_kib": {"min": 65536}})",
                 R"({"l3_cores": [16, 32]})"}) {
        EXPECT_FALSE(hardware_applies(no, hw)) << no;
    }
    // The warning shows a short list or range as written.
    hardware_ignored(R"({"model": [16, 81]})", hw,
            "hardware.model=[16,81] does not hold 17");
    hardware_ignored(R"({"model": {"min": 18, "max": 31}})", hw,
            R"(hardware.model={"max":31,"min":18} does not hold 17)");
    hardware_ignored(R"({"vendor": ["OtherVendor0", "OtherVendor1"]})", hw,
            R"(hardware.vendor=["OtherVendor0","OtherVendor1"] does not hold "TestVendorId")");
}

TEST(NativeConfigTuningFile, HardwareIsaSubset) {
    const native_hw_t hw = identified_hw();
    for (const char *yes : {R"({"isa": []})", R"({"isa": "avx512vnni"})",
                 R"({"isa": ["avx512bf16", "avx512vnni"]})",
                 R"({"isa": ["avx512vnni", "avx512f", "avx512bf16"]})"}) {
        EXPECT_TRUE(hardware_applies(yes, hw)) << yes;
    }
    native_hw_t no_vnni = hw;
    no_vnni.uarch.avx512vnni = false;
    EXPECT_TRUE(
            hardware_applies(R"({"isa": ["avx512f", "avx512bf16"]})", no_vnni));
    hardware_ignored(R"({"isa": ["avx512bf16", "avx512vnni"]})", no_vnni,
            "hardware.isa lists \"avx512vnni\", which this CPU lacks");
    hardware_ignored(R"({"isa": "avx512vnni"})", no_vnni, "avx512vnni");
}

TEST(NativeConfigTuningFile, HardwareOneFileForAFamily) {
    // Two machines of one family that differ in model, L3 size, L3 domain
    // and core count.
    native_hw_t a = identified_hw();
    a.model = 2;
    a.smt = 1;
    a.logical_cpus = a.physical_cores = 256;
    a.l3_domain_threads = 8;
    native_hw_t b = a;
    b.model = 81;
    b.uarch.l3_domain_bytes = 128 << 20;
    b.logical_cpus = b.physical_cores = 512;
    b.l3_domain_threads = 32;
    const std::string common = R"("vendor": "TestVendorId", "family": 26,
            "isa": ["avx512bf16", "avx512vnni"], "l1d_kib": 48, "l2_kib": 1024)";
    const std::string both = "{" + common
            + R"(, "model": [2, 81], "l3_kib": {"min": 32768, "max": 131072},
                 "l3_cores": {"min": 8, "max": 32}})";
    const std::string for_a
            = "{" + common + R"(, "model": 2, "l3_kib": 32768, "l3_cores": 8})";
    const std::string for_b = "{" + common
            + R"(, "model": 81, "l3_kib": 131072, "l3_cores": 32})";
    EXPECT_TRUE(hardware_applies(both, a));
    EXPECT_TRUE(hardware_applies(both, b));
    EXPECT_TRUE(hardware_applies(for_a, a));
    hardware_ignored(for_a, b, "hardware.model=2 does not hold 81");
    EXPECT_TRUE(hardware_applies(for_b, b));
    hardware_ignored(for_b, a, "hardware.model=81 does not hold 2");
    // The same part with SMT on, or with fewer cores, keeps the file made
    // for a.
    native_hw_t smt_on = a;
    smt_on.smt = 2;
    smt_on.logical_cpus = 512;
    smt_on.l3_domain_threads = 16;
    native_hw_t fewer = a;
    fewer.logical_cpus = fewer.physical_cores = 128;
    EXPECT_TRUE(hardware_applies(for_a, smt_on));
    EXPECT_TRUE(hardware_applies(for_a, fewer));
}

TEST(NativeConfigTuningFile, HardwareFieldsTheCpuDoesNotReport) {
    const native_hw_t hw = identified_hw();
    const struct {
        const char *block;
        void (*forget)(native_hw_t &);
    } cases[] = {
            {R"({"vendor": "TestVendorId"})",
                    [](native_hw_t &h) { h.vendor.clear(); }},
            {R"({"family": {"min": 0}})",
                    [](native_hw_t &h) { h.family = -1; }},
            {R"({"model": {"min": 0}})", [](native_hw_t &h) { h.model = -1; }},
            {R"({"stepping": [0, 1, 2]})",
                    [](native_hw_t &h) { h.stepping = -1; }},
            {R"({"smt": {"max": 2}})", [](native_hw_t &h) { h.smt = 0; }},
            {R"({"l3_threads": {"min": 0}})",
                    [](native_hw_t &h) { h.l3_domain_threads = 0; }},
            {R"({"cores": {"max": 1024}})",
                    [](native_hw_t &h) { h.physical_cores = 0; }},
            {R"({"logical_cpus": {"min": 0}})",
                    [](native_hw_t &h) { h.logical_cpus = 0; }},
            {R"({"l1d_kib": {"min": 0}})",
                    [](native_hw_t &h) { h.uarch.l1d_detected = false; }},
            {R"({"l2_kib": {"min": 0}})",
                    [](native_hw_t &h) { h.uarch.l2_detected = false; }},
            {R"({"l3_kib": [32768]})",
                    [](native_hw_t &h) { h.uarch.l3_detected = false; }},
            {R"({"l3_cores": {"min": 0}})",
                    [](native_hw_t &h) {
        h.l3_domain_threads = 0;
        h.uarch.family_l3_cores = 0;
    }},
    };
    for (const auto &c : cases) {
        native_hw_t forgot = hw;
        c.forget(forgot);
        EXPECT_TRUE(hardware_applies(c.block, hw)) << c.block;
        hardware_ignored(c.block, forgot, "this CPU does not report it");
        // A file that does not ask for the field still applies.
        EXPECT_TRUE(hardware_applies(R"({"isa": "avx512f"})", forgot))
                << c.block;
    }
    // Without the L3 topology the plan's domain, and l3_cores, is
    // family_l3_cores.
    native_hw_t no_topology = hw;
    no_topology.l3_domain_threads = 0;
    no_topology.uarch.family_l3_cores = 4;
    EXPECT_TRUE(hardware_applies(R"({"l3_cores": 4})", no_topology));
    EXPECT_FALSE(hardware_applies(R"({"l3_cores": 8})", no_topology));
}

TEST(NativeConfigTuningFile, HardwareUnusableFiles) {
    const native_hw_t hw = identified_hw();
    const std::pair<const char *, const char *> bad[] = {
            {R"({"signature": "v1;isa=none"})",
                    "unknown hardware key 'signature'"},
            {R"({"frequency": 3})", "unknown hardware key 'frequency'"},
            {R"({"Family": 26})", "unknown hardware key 'Family'"},
            {R"({"family": 25, "sockets": 2})",
                    "unknown hardware key 'sockets'"},
            {R"({"model": {"min": 5, "max": 2}})", "hardware.model has a bad"},
            {R"({"model": {}})", "hardware.model has a bad"},
            {R"({"model": {"min": 1.5}})", "hardware.model has a bad"},
            {R"({"model": {"lo": 1}})", "hardware.model has a bad"},
            {R"({"model": {"min": 1, "max": 20, "step": 1}})",
                    "hardware.model has a bad"},
            {R"({"model": []})", "hardware.model has a bad"},
            {R"({"model": [1, "17"]})", "hardware.model has a bad"},
            {R"({"model": "17"})", "hardware.model has a bad"},
            {R"({"model": 17.0})", "hardware.model has a bad"},
            {R"({"model": true})", "hardware.model has a bad"},
            {R"({"model": null})", "hardware.model has a bad"},
            {R"({"l3_kib": [[32768]]})", "hardware.l3_kib has a bad"},
            {R"({"vendor": 5})", "hardware.vendor has a bad"},
            {R"({"vendor": []})", "hardware.vendor has a bad"},
            {R"({"vendor": ["TestVendorId", 1]})", "hardware.vendor has a bad"},
            {R"({"isa": ["avx512f", 3]})", "hardware.isa has a bad"},
            {R"({"isa": {"avx512f": true}})", "hardware.isa has a bad"},
            {R"({"isa": ["avx512bf16", "amx"]})",
                    "hardware.isa names \"amx\", which this build does not "
                    "know"},
            {R"(["family"])", "hardware is not an object"},
            {R"(26)", "hardware is not an object"},
    };
    for (const auto &[block, what] : bad)
        hardware_ignored(block, hw, what);
}

// This CPU's value of every hardware key, as the schema dump prints it: the
// block applies here, and each key at another value does not.
TEST(NativeConfigTuningFile, HardwareBlockOfThisCpu) {
    const native_hw_t &hw = native_hw();
    const json m = json::parse(native_brgemm_test::hw_match_text(hw));
    EXPECT_TRUE(hardware_applies(m.dump(), hw));
    for (auto it = m.begin(); it != m.end(); ++it) {
        if (it.key() == "isa") continue;
        json other = m;
        if (it.key() == "vendor")
            other["vendor"] = "OtherVendor0";
        else
            other[it.key()] = it->get<int64_t>() + 1;
        EXPECT_FALSE(hardware_applies(other.dump(), hw)) << it.key();
    }
    EXPECT_EQ(m.contains("vendor"), !hw.vendor.empty());
    if (!hw.vendor.empty()) { EXPECT_EQ(hw.vendor.size(), 12u); }
    EXPECT_GE(hw.family, 0);
    EXPECT_GE(hw.model, 0);
    EXPECT_GE(hw.stepping, 0);
    EXPECT_EQ(m.at("l2_kib"), hw.uarch.l2_bytes / 1024);
    const UarchParams &plan = native_plan_uarch();
    EXPECT_EQ(m.at("l3_cores"),
            plan.l3_domain_cores > 0 ? plan.l3_domain_cores
                                     : plan.family_l3_cores);
    // Every key the block gives is a hardware key of the loader, in its
    // order.
    const nlohmann::ordered_json ordered = nlohmann::ordered_json::parse(
            native_brgemm_test::hw_match_text(hw));
    size_t at = 0;
    for (auto it = ordered.begin(); it != ordered.end(); ++it) {
        while (at < std::size(kNativeHardwareKeys)
                && it.key() != kNativeHardwareKeys[at])
            ++at;
        EXPECT_LT(at, std::size(kNativeHardwareKeys)) << it.key();
    }
}

TEST(NativeConfigTuningFile, DropsBadEntriesWithCappedWarnings) {
    const char *bad[] = {R"({"match": {"q": 1}, "config": {"nr": 16}})",
            R"({"config": {"foo": 1}})", R"({"match": {"n": "4096"}})",
            R"({"match": {"trans_a": 1}})", R"({"config": {"nr": 16.0}})",
            R"({"config": {"row_split": 1}})",
            R"({"config": {"int8_cw": "64/16"}})",
            R"({"config": {"schedule": 3}})",
            R"({"config": {"schedule": "slice,nr=16"}})",
            R"({"config": {"nr": 24}})",
            R"({"config": {"schedule": "slice", "col_groups": 4}})", R"(17)",
            R"({"weight": 1})", R"({"id": 5})", R"({"match": [1]})",
            R"({"config": []})", R"({"match": {"n": []}})",
            R"({"match": {"m": {"min": 5, "max": 2}}})",
            R"({"match": {"m": {"lo": 1}}})", R"({"match": {"dtype": "f64"}})"};
    std::string list;
    for (const char *e : bad)
        list += std::string(e) + ", ";
    list += R"({"id": "good", "config": {"nr": 16}})";
    native_tuning_sources_t s;
    EXPECT_EQ(warnings_of(
                      [&] { s = load(tuning("\"entries\": [" + list + "]")); }),
            shown(16 + 1));
    ASSERT_NE(s.table, nullptr);
    const native_config_entry_t *e = s.table->lookup(problem(8, 4096, 4096));
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(format_native_config(e->config), "nr=16");
    const std::string label = e->label;
    EXPECT_NE(label.find("test.json entry 20 (id good)"), std::string::npos)
            << label;
}

TEST(NativeConfigTuningFile, MatchValues) {
    const native_tuning_sources_t s = load(tuning(R"("entries": [
        {"id": "a", "match": {"m": {"min": 1, "max": 16}, "n": [1024, 4096],
                              "dtype": ["bf16"], "trans_a": false},
         "config": {"nr": 16}},
        {"id": "b", "match": {"m": {"min": 17}, "k": 4096, "src": "u8",
                              "dst": ["bf16", "f32"], "epilogue": "store",
                              "has_bias": false, "dynamic": false,
                              "threads": {"max": 64}, "pack_nr": 32},
         "config": {"int8_cw": [16], "direct_store": false}},
        {"id": "c", "match": {"m": {"max": 16}}, "config": {}}])"));
    ASSERT_NE(s.table, nullptr);
    const native_config_source_t *t = s.table.get();
    EXPECT_EQ(config_of(t, problem(16, 4096, 4096)), "nr=16");
    EXPECT_EQ(config_of(t, problem(16, 2048, 4096)), "");
    EXPECT_EQ(config_of(t, problem(17, 4096, 4096)), "<none>");
    EXPECT_EQ(config_of(t, problem(64, 4096, 4096, false, 64)),
            "int8_cw=16,direct_store=0");
    EXPECT_EQ(config_of(t, problem(64, 4096, 4096, false, 65)), "<none>");
    EXPECT_EQ(config_of(t, problem(64, 4096, 2048, false)), "<none>");
}

// An s4 weight's entries name its dtype and group size; those of one dtype
// and an exact n and k are indexed apart from the s8 weight's.
TEST(NativeConfigTuningFile, Int4Entries) {
    const native_tuning_sources_t s = load(tuning(R"("entries": [
        {"id": "a", "match": {"dtype": "s4", "group": 128, "n": 4096,
                              "k": 4096},
         "config": {"int8_mr": 4}},
        {"id": "b", "match": {"dtype": "s8", "n": 4096, "k": 4096},
         "config": {"int8_mr": 2}},
        {"id": "c", "match": {"dtype": ["s4"], "group": {"max": 64}},
         "config": {"int8_cw": [16]}},
        {"id": "d", "match": {"group": 0, "m": {"max": 16}},
         "config": {"nr": 16}}])"));
    ASSERT_NE(s.table, nullptr);
    const native_config_source_t *t = s.table.get();
    EXPECT_EQ(config_of(t, int4_problem(8, 4096, 4096, 128)), "int8_mr=4");
    EXPECT_EQ(config_of(t, int4_problem(8, 4096, 4096, 32)), "int8_cw=16");
    EXPECT_EQ(config_of(t, int4_problem(8, 4096, 4096, 256)), "<none>");
    EXPECT_EQ(config_of(t, int4_problem(8, 1024, 4096, 64)), "int8_cw=16");
    EXPECT_EQ(config_of(t, problem(8, 4096, 4096, false)), "int8_mr=2");
    EXPECT_EQ(config_of(t, problem(8, 1024, 4096, false)), "nr=16");
    EXPECT_EQ(config_of(t, problem(8, 1024, 4096)), "nr=16");
    // An entry of a dtype the planner does not name is dropped.
    native_tuning_sources_t d;
    EXPECT_EQ(warnings_of([&] {
        d = load(tuning(R"("entries": [
            {"match": {"dtype": "u4"}, "config": {"nr": 16}},
            {"id": "good", "config": {"mb": 24}}])"));
    }),
            shown(1));
    ASSERT_NE(d.table, nullptr);
    EXPECT_EQ(config_of(d.table.get(), int4_problem(8, 4096, 4096, 128)),
            "mb=24");
}

TEST(NativeConfigTuningFile, RuleKeys) {
    const native_tuning_sources_t s = load(tuning(R"("entries": [
        {"id": "small", "match": {"m": {"max": 64}},
         "config": {"blocks": 0, "l1_pass": 0, "l1_pct": 75}},
        {"id": "other", "config": {"makespan": 0}}])"));
    ASSERT_NE(s.table, nullptr);
    EXPECT_EQ(config_of(s.table.get(), problem(8, 4096, 4096)),
            "l1_pass=0,l1_pct=75,blocks=0");
    EXPECT_EQ(config_of(s.table.get(), problem(65, 4096, 4096)), "makespan=0");
    // Value names are lower case, and a switch is a number.
    for (const char *bad : {R"({"config": {"schedule": "SLICE"}})",
                 R"({"config": {"schedule": "slice "}})",
                 R"({"config": {"l1_pass": true}})"}) {
        native_tuning_sources_t d;
        EXPECT_EQ(warnings_of([&] {
            d = load(tuning(std::string("\"entries\": [") + bad + "]"));
        }),
                shown(1))
                << bad;
        EXPECT_EQ(d.table, nullptr) << bad;
    }
}

TEST(NativeConfigTuningFile, IndexKeepsFileOrder) {
    const std::string wide
            = R"({"id": "wide", "match": {"m": {"max": 16}}, "config": {"nr": 16}})";
    const std::string exact = R"({"id": "exact", "match": {"dtype": "bf16",
            "n": 4096, "k": 4096}, "config": {"nr": 32}})";
    const std::string other = R"({"id": "other", "match": {"dtype": "bf16",
            "n": 1024, "k": 4096}, "config": {"mb": 24}})";
    const native_tuning_sources_t a = load(tuning(
            "\"entries\": [" + wide + ", " + exact + ", " + other + "]"));
    ASSERT_NE(a.table, nullptr);
    EXPECT_EQ(config_of(a.table.get(), problem(8, 4096, 4096)), "nr=16");
    EXPECT_EQ(config_of(a.table.get(), problem(64, 4096, 4096)), "nr=32");
    EXPECT_EQ(config_of(a.table.get(), problem(64, 1024, 4096)), "mb=24");
    EXPECT_EQ(
            config_of(a.table.get(), problem(64, 4096, 4096, false)), "<none>");
    EXPECT_EQ(config_of(a.table.get(), problem(8, 4096, 4096, false)), "nr=16");
    const native_tuning_sources_t b = load(tuning(
            "\"entries\": [" + exact + ", " + wide + ", " + other + "]"));
    ASSERT_NE(b.table, nullptr);
    EXPECT_EQ(config_of(b.table.get(), problem(8, 4096, 4096)), "nr=32");
    EXPECT_EQ(config_of(b.table.get(), problem(8, 1024, 4096)), "nr=16");
}

TEST(NativeConfigTuningFile, TreeRules) {
    const native_tuning_sources_t s = load(tuning(R"("rules": {"kind": "tree",
        "root": {"feature": "m", "le": 16,
                 "then": {"feature": "n_per_thread", "le": 48,
                          "then": {"id": "narrow", "config": {"nr": 16}},
                          "else": {"config": {}}},
                 "else": {"config": {"kc_split": "equal"}}}})"));
    EXPECT_EQ(s.table, nullptr);
    ASSERT_NE(s.rules, nullptr);
    const native_config_entry_t *e
            = s.rules->lookup(problem(8, 6144, 4096, true, 128));
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(format_native_config(e->config), "nr=16");
    EXPECT_NE(e->label.find("(id narrow)"), std::string::npos) << e->label;
    EXPECT_EQ(config_of(s.rules.get(), problem(8, 8192, 4096, true, 128)),
            "<none>");
    EXPECT_EQ(config_of(s.rules.get(), problem(17, 8192, 4096, true, 128)),
            "kc_split=equal");
}

TEST(NativeConfigTuningFile, OrderedRules) {
    const native_tuning_sources_t s
            = load(tuning(R"("rules": {"kind": "ordered",
        "list": [
          {"id": "a", "when": [["m", "<=", 16], ["n_per_thread", "<=", 48]],
           "config": {"nr": 16}},
          {"id": "b", "when": [["k", ">", 8192]], "config": {"kc_split": "equal"}},
          {"id": "e", "when": [["m", "==", 100]], "config": {}},
          {"id": "c", "when": [], "config": {"mb": 24}}]})"));
    ASSERT_NE(s.rules, nullptr);
    const native_config_source_t *r = s.rules.get();
    EXPECT_EQ(config_of(r, problem(8, 6144, 4096, true, 128)), "nr=16");
    EXPECT_EQ(config_of(r, problem(8, 8192, 4096, true, 128)), "mb=24");
    EXPECT_EQ(
            config_of(r, problem(8, 8192, 14336, true, 128)), "kc_split=equal");
    EXPECT_EQ(config_of(r, problem(100, 8192, 4096, true, 128)), "<none>");
    for (const auto &[op, holds] : std::vector<std::pair<std::string, bool>> {
                 {"<", false}, {"<=", true}, {">", false}, {">=", true},
                 {"==", true}, {"!=", false}}) {
        const native_tuning_sources_t o = load(tuning(
                R"("rules": {"kind": "ordered", "list": [{"when": [["m", ")"
                + op + R"(", 8]], "config": {"nr": 16}}]})"));
        ASSERT_NE(o.rules, nullptr) << op;
        EXPECT_EQ(o.rules->lookup(problem(8, 4096, 4096)) != nullptr, holds)
                << op;
    }
}

TEST(NativeConfigTuningFile, EveryFeature) {
    native_problem_t p = problem(24, 4096, 2048, true, 16);
    p.trans_a = true;
    p.has_bias = true;
    p.epilogue = native_epilogue_class_t::bias;
    const std::vector<std::pair<const char *, double>> expected = {{"m", 24},
            {"n", 4096}, {"k", 2048}, {"threads", 16}, {"pack_nr", 32},
            {"bf16", 1}, {"dynamic", 0}, {"trans_a", 1}, {"has_bias", 1},
            {"epilogue", 1}, {"dst_bytes", 2}, {"esz", 2}, {"a_bytes", 98304},
            {"b_bytes", 16777216}, {"col_tiles", 128}, {"m_per_thread", 1.5},
            {"n_per_thread", 256}, {"flops_per_thread", 25165824},
            {"l2", 1048576}, {"l3", 33554432}, {"l3_threads", 16}, {"smt", 2},
            {"cores", 128}, {"a_l3", 0.0029296875}, {"b_l3", 0.5},
            {"slab_l2", 0.125}, {"group", 0}};
    ASSERT_EQ(expected.size(), std::size(kNativeRuleFeatures));
    // An s4 weight's: its elements are half a byte, its source's one.
    const native_problem_t g = int4_problem(24, 4096, 2048, 128, 16);
    const std::vector<std::pair<const char *, double>> int4
            = {{"bf16", 0}, {"dynamic", 1}, {"esz", 0.5}, {"a_bytes", 49152},
                    {"b_bytes", 4194304}, {"b_l3", 0.125}, {"slab_l2", 0.03125},
                    {"group", 128}};
    const auto holds = [](const char *feature, double value, const char *op,
                               const native_problem_t &q) {
        json rule;
        rule["when"] = json::array({json::array({feature, op, value})});
        rule["config"]["nr"] = 16;
        json doc;
        doc["format"] = "zendnnl-native-brgemm-tuning";
        doc["version"] = 2;
        doc["rules"]["kind"] = "ordered";
        doc["rules"]["list"] = json::array({rule});
        const native_tuning_sources_t r = load(doc.dump());
        EXPECT_NE(r.rules, nullptr) << feature;
        return r.rules != nullptr && r.rules->lookup(q) != nullptr;
    };
    for (const auto &[feature, value] : int4) {
        EXPECT_TRUE(holds(feature, value, "==", g))
                << feature << " != " << value;
        EXPECT_FALSE(holds(feature, value, ">", g)) << feature;
    }
    for (const auto &[feature, value] : expected) {
        const auto rules = [&, name = feature, v = value](const char *op) {
            json rule;
            rule["when"] = json::array({json::array({name, op, v})});
            rule["config"]["nr"] = 16;
            json doc;
            doc["format"] = "zendnnl-native-brgemm-tuning";
            doc["version"] = 2;
            doc["rules"]["kind"] = "ordered";
            doc["rules"]["list"] = json::array({rule});
            return load(doc.dump());
        };
        const native_tuning_sources_t eq = rules("==");
        ASSERT_NE(eq.rules, nullptr) << feature;
        EXPECT_NE(eq.rules->lookup(p), nullptr) << feature << " != " << value;
        const native_tuning_sources_t gt = rules(">");
        ASSERT_NE(gt.rules, nullptr) << feature;
        EXPECT_EQ(gt.rules->lookup(p), nullptr) << feature;
    }
}

namespace {

/// A full tree of the given depth over m; only the leftmost leaf has an
/// opinion.
json full_tree(int depth, bool &first) {
    if (depth == 1) {
        json leaf = {{"config", json::object()}};
        if (first) leaf["config"]["nr"] = 16;
        first = false;
        return leaf;
    }
    json n = {{"feature", "m"}, {"le", 1 << (depth - 2)}};
    n["then"] = full_tree(depth - 1, first);
    n["else"] = full_tree(depth - 1, first);
    return n;
}

native_tuning_sources_t load_tree(const json &root) {
    const json doc = {{"format", "zendnnl-native-brgemm-tuning"},
            {"version", 2}, {"rules", {{"kind", "tree"}, {"root", root}}}};
    return load(doc.dump());
}

native_tuning_sources_t load_ordered(const json &list) {
    const json doc = {{"format", "zendnnl-native-brgemm-tuning"},
            {"version", 2}, {"rules", {{"kind", "ordered"}, {"list", list}}}};
    return load(doc.dump());
}

} // namespace

TEST(NativeConfigTuningFile, RulesBounds) {
    const native_problem_t one_row = problem(1, 4096, 4096);
    // Depth: the root is depth 1; 64 levels load, 65 are dropped.
    for (const int depth : {64, 65}) {
        json root = {{"config", {{"nr", 16}}}};
        for (int d = 1; d < depth; ++d)
            root = {{"feature", "m"}, {"le", 1000 + d}, {"then", root},
                    {"else", {{"config", json::object()}}}};
        native_tuning_sources_t s;
        EXPECT_EQ(warnings_of([&] { s = load_tree(root); }),
                depth == 64 ? 0 : shown(1));
        EXPECT_EQ(s.rules != nullptr, depth == 64) << depth;
    }
    // Nodes: a full tree of 65535 nodes loads; two more are dropped.
    bool first = true;
    json tree = full_tree(16, first);
    native_tuning_sources_t s = load_tree(tree);
    ASSERT_NE(s.rules, nullptr);
    EXPECT_EQ(config_of(s.rules.get(), one_row), "nr=16");
    json *leftmost = &tree;
    while (leftmost->contains("then"))
        leftmost = &(*leftmost)["then"];
    *leftmost = {{"feature", "k"}, {"le", 1}, {"then", *leftmost},
            {"else", *leftmost}};
    EXPECT_EQ(load_tree(tree).rules, nullptr);
    // Ordered: 4096 rules of 32 conditions load; one more of either is dropped.
    const json cond = json::array({"m", ">=", 1});
    json rule = {{"when", json::array()}, {"config", {{"nr", 16}}}};
    for (int i = 0; i < 32; ++i)
        rule["when"].push_back(cond);
    json list = json::array();
    for (int i = 0; i < 4096; ++i)
        list.push_back(rule);
    ASSERT_NE(load_ordered(list).rules, nullptr);
    list.push_back(rule);
    EXPECT_EQ(load_ordered(list).rules, nullptr);
    rule["when"].push_back(cond);
    EXPECT_EQ(load_ordered(json::array({rule})).rules, nullptr);
    // Malformed rules are dropped whole, with one warning.
    for (const char *bad : {
                 R"({"kind": "tree", "root": {"feature": "q", "le": 1, "then": {"config": {}}, "else": {"config": {}}}})",
                 R"({"kind": "tree", "root": {"feature": "m", "le": 1, "then": {"config": {}}}})",
                 R"({"kind": "tree", "root": {"feature": "m", "le": "1", "then": {"config": {}}, "else": {"config": {}}}})",
                 R"({"kind": "tree", "root": {"feature": "m", "le": 1, "then": {"config": {}}, "else": {"config": {}}, "x": 1}})",
                 R"({"kind": "tree", "root": {"config": {"nr": 16}, "n": 3}})",
                 R"({"kind": "tree", "root": {"id": "x"}})",
                 R"({"kind": "tree", "root": [1]})",
                 R"({"kind": "tree", "root": {"config": {}}, "list": []})",
                 R"({"kind": "forest", "root": {"config": {}}})",
                 R"({"root": {"config": {}}})", R"([1])",
                 R"({"kind": "ordered", "list": [{"when": [["m", "=~", 1]], "config": {}}]})",
                 R"({"kind": "ordered", "list": [{"when": [["q", "<", 1]], "config": {}}]})",
                 R"({"kind": "ordered", "list": [{"when": [["m", "<"]], "config": {}}]})",
                 R"({"kind": "ordered", "list": [{"when": [["m", "<", "1"]], "config": {}}]})",
                 R"({"kind": "ordered", "list": [{"when": []}]})",
                 R"({"kind": "ordered", "list": [{"when": [], "config": {}, "then": 1}]})",
                 R"({"kind": "ordered", "list": {"when": []}})"}) {
        native_tuning_sources_t d;
        EXPECT_EQ(warnings_of([&] {
            d = load(tuning(std::string("\"rules\": ") + bad));
        }),
                shown(1))
                << bad;
        EXPECT_EQ(d.rules, nullptr) << bad;
    }
    // A leaf with a bad config is emptied alone, with one warning.
    native_tuning_sources_t leaf;
    EXPECT_EQ(warnings_of([&] {
        leaf = load(tuning(R"("rules": {"kind": "tree", "root": {"feature": "m",
            "le": 16, "then": {"config": {"nr": 24}}, "else": {"config": {"mb": 24}}}})"));
    }),
            shown(1));
    ASSERT_NE(leaf.rules, nullptr);
    EXPECT_EQ(config_of(leaf.rules.get(), one_row), "<none>");
    EXPECT_EQ(config_of(leaf.rules.get(), problem(17, 4096, 4096)), "mb=24");
}

TEST(NativeConfigTuningFile, LoadsFilesOnlyWhenReadable) {
    const native_brgemm_test::temp_file_t file(
            "test.json", tuning(R"("entries": [{"config": {"nr": 16}}])"));
    const std::string &path = file.path();
    native_tuning_sources_t s = load_native_tuning_file(path, test_hw());
    ASSERT_NE(s.table, nullptr);
    EXPECT_EQ(config_of(s.table.get(), problem(8, 4096, 4096)), "nr=16");
    EXPECT_EQ(warnings_of([&] {
        s = load_native_tuning_file(path + ".missing", test_hw());
    }),
            shown(1));
    EXPECT_EQ(s.table, nullptr);
    const native_brgemm_test::temp_file_t big_file("big.json", "");
    const std::string &big = big_file.path();
    {
        std::ofstream f(big, std::ios::binary);
        f.seekp((int64_t {64} << 20) + 16);
        f.put('}');
    }
    EXPECT_EQ(warnings_of([&] { s = load_native_tuning_file(big, test_hw()); }),
            shown(1));
    EXPECT_EQ(s.table, nullptr);
    // A directory, and a FIFO no process writes, are not read: opening the
    // FIFO would block.
    EXPECT_EQ(warnings_of([&] {
        s = load_native_tuning_file(testing::TempDir(), test_hw());
    }),
            shown(1));
    EXPECT_EQ(s.table, nullptr);
#if !defined(_WIN32)
    const native_brgemm_test::temp_file_t fifo_file("fifo", "");
    const std::string &fifo = fifo_file.path();
    std::remove(fifo.c_str());
    ASSERT_EQ(mkfifo(fifo.c_str(), 0600), 0);
    EXPECT_EQ(
            warnings_of([&] { s = load_native_tuning_file(fifo, test_hw()); }),
            shown(1));
    EXPECT_EQ(s.table, nullptr);
#endif
}

TEST(NativeConfigPlanner, PrecedenceAndFieldMerge) {
    if (!has_isa()) GTEST_SKIP() << "the CPU lacks AVX512-BF16 or AVX512-VNNI";
    const UarchParams &u = detect_uarch();
    native_planner_t planner(u, one_entry("nr=16", "o"),
            one_entry("nr=32,mb=24", "t"),
            one_entry("mb=48,threads_cap=8", "r"));
    ASSERT_TRUE(planner.has_sources());
    const native_problem_t p = problem(256, 4096, 4096, true, 64);
    const native_plan_t q = planner.plan(p);
    heuristic_params_t hp;
    hp.nr = 16;
    hp.mb = 24;
    hp.threads_cap = 8;
    EXPECT_TRUE(same_grid(q, heuristic_plan(p, u, hp)));
    EXPECT_EQ(q.nr, 16);
    EXPECT_EQ(q.mb, 24);
    EXPECT_LE(q.threads, 8);
    EXPECT_EQ(top_source(q), source_kind::override);
    EXPECT_NE(q.override_entry, nullptr);
    EXPECT_NE(q.table_entry, nullptr);
    EXPECT_NE(q.rules_entry, nullptr);
    EXPECT_EQ(native_plan_source_names(q), "override+table+rules");
    EXPECT_EQ(native_plan_config_text(q), "nr=16,mb=24,threads_cap=8");
    EXPECT_EQ(validate_native_plan(q, p), nullptr);
}

TEST(NativeConfigPlanner, IncrementalFallbackWarnsOncePerEntry) {
    if (!has_isa()) GTEST_SKIP() << "the CPU lacks AVX512-BF16 or AVX512-VNNI";
    const UarchParams &u = detect_uarch();
    // M above the cache bound: every call resolves again.
    const native_problem_t p = problem(2048, 4096, 14336, true, 64);
    {
        auto bad = one_entry("nr=16,mr=7", "bad o");
        const native_config_entry_t &bad_entry = bad->entry();
        native_planner_t planner(u, std::move(bad),
                one_entry("threads_cap=8", "t"),
                one_entry("kc_split=equal", "r"));
        native_plan_t q;
        EXPECT_EQ(warnings_of([&] { q = planner.plan(p); }), shown(1));
        EXPECT_TRUE(bad_entry.warned.load());
        EXPECT_EQ(warnings_of([&] { q = planner.plan(p); }), 0);
        heuristic_params_t hp;
        hp.threads_cap = 8;
        hp.kc_split = native_kc_split_t::equal;
        EXPECT_TRUE(same_grid(q, heuristic_plan(p, u, hp)));
        EXPECT_EQ(top_source(q), source_kind::table);
        EXPECT_EQ(q.override_entry, nullptr);
        EXPECT_EQ(native_plan_source_names(q), "table+rules");
        EXPECT_EQ(native_plan_config_text(q), "threads_cap=8,kc_split=equal");
    }
    {
        // A lower layer that breaks the plan goes; a legal higher one stays.
        native_planner_t planner(u, one_entry("mb=24", "o"),
                one_entry("nr=16,mr=7", "bad t"), nullptr);
        native_plan_t q;
        EXPECT_EQ(warnings_of([&] { q = planner.plan(p); }), shown(1));
        heuristic_params_t hp;
        hp.mb = 24;
        EXPECT_TRUE(same_grid(q, heuristic_plan(p, u, hp)));
        EXPECT_EQ(native_plan_source_names(q), "override");
        EXPECT_EQ(native_plan_config_text(q), "mb=24");
        EXPECT_EQ(q.table_entry, nullptr);
    }
    {
        native_planner_t planner(
                u, one_entry("nr=16,mr=7", "bad o"), nullptr, nullptr);
        native_plan_t q;
        EXPECT_EQ(warnings_of([&] { q = planner.plan(p); }), shown(1));
        EXPECT_TRUE(same_plan(q, heuristic_plan(p, u, heuristic_params_t {})));
        EXPECT_EQ(q.sources, 0);
        EXPECT_EQ(top_source(q), source_kind::heuristic);
        EXPECT_EQ(native_plan_source_names(q), "heuristic");
        EXPECT_EQ(native_plan_config_text(q), "-");
    }
}

// A lower layer legal alone but illegal under a higher one is dropped as a
// conflict; the drops past the first 16 warnings get one more line.
TEST(NativeConfigPlanner, DropWarningsNameConflictsAndAreCapped) {
    if (!has_isa()) GTEST_SKIP() << "the CPU lacks AVX512-BF16 or AVX512-VNNI";
    const UarchParams &u = detect_uarch();
    const native_problem_t p = problem(2048, 4096, 14336, true, 64);
    {
        native_planner_t planner(
                u, one_entry("mr=12", "o"), one_entry("nr=16", "t"), nullptr);
        native_plan_t q;
        const std::string out = log_of([&] { q = planner.plan(p); });
        EXPECT_EQ(native_plan_source_names(q), "override");
        EXPECT_EQ(q.mr, 12);
        if (shown(1) == 1) {
            EXPECT_NE(out.find("t conflicts with a higher layer"),
                    std::string::npos)
                    << out;
        }
    }
    std::string entries;
    for (int m = 1; m <= 20; ++m)
        entries += std::string(m > 1 ? ", " : "") + R"({"match": {"m": )"
                + std::to_string(m) + R"(}, "config": {"nr": 16, "mr": 7}})";
    native_tuning_sources_t s = load(tuning("\"entries\": [" + entries + "]"));
    ASSERT_NE(s.table, nullptr);
    native_planner_t planner(u, nullptr, std::move(s.table), nullptr);
    EXPECT_EQ(warnings_of([&] {
        for (int m = 1; m <= 20; ++m)
            planner.plan(problem(m, 4096, 4096, true, 64));
    }),
            shown(17));
}

// A per-group INT8 problem does not apply the keys its scheme does not
// apply (native_config_key_t::int8_schemes): the heuristic decides them, the
// config text leaves them out, the entry notes them once at the verbose
// level and no warning is written, and a layer with no other key is not
// applied.  A per-channel problem and a BF16 problem apply them.
TEST(NativeConfigPlanner, PerChannelKeysSkipPerGroupProblems) {
    if (!has_isa()) GTEST_SKIP() << "the CPU lacks AVX512-BF16 or AVX512-VNNI";
    const UarchParams &u = detect_uarch();
    const native_problem_t channel = problem(64, 4096, 4096, false);
    native_problem_t group = channel;
    group.int8_scheme = native_int8_scheme_t::per_group;
    {
        auto o = one_entry("int8_mr=4,int8_kc=64,direct_store=0", "o");
        const native_config_entry_t &entry = o->entry();
        native_planner_t planner(u, std::move(o), nullptr, nullptr);
        native_plan_t q;
        EXPECT_EQ(warnings_of([&] { q = planner.plan(group); }), 0);
        EXPECT_TRUE(entry.scope_noted.load());
        heuristic_params_t hp;
        hp.int8_mr = 4;
        hp.direct_store = false;
        EXPECT_TRUE(same_grid(q, heuristic_plan(group, u, hp)));
        EXPECT_FALSE(q.int8_direct_store);
        EXPECT_EQ(q.int8_scheme, native_int8_scheme_t::per_group);
        EXPECT_EQ(top_source(q), source_kind::override);
        EXPECT_EQ(native_plan_config_text(q), "int8_mr=4,direct_store=0");
        EXPECT_EQ(validate_native_plan(q, group), nullptr);
        const native_plan_t c = planner.plan(channel);
        hp.int8_kc = 64;
        EXPECT_TRUE(same_grid(c, heuristic_plan(channel, u, hp)));
        EXPECT_EQ(c.int8_pass_quads, 16);
        EXPECT_FALSE(c.int8_direct_store);
        EXPECT_EQ(native_plan_config_text(c),
                "int8_mr=4,int8_kc=64,direct_store=0");
    }
    {
        native_planner_t planner(u, one_entry("int8_kc=64", "o"),
                one_entry("l1_pass=0,min_pass_k=512", "t"), nullptr);
        const native_plan_t q = planner.plan(group);
        EXPECT_TRUE(
                same_plan(q, heuristic_plan(group, u, heuristic_params_t {})));
        EXPECT_EQ(top_source(q), source_kind::heuristic);
        EXPECT_EQ(native_plan_config_text(q), "-");
        const native_plan_t c = planner.plan(channel);
        EXPECT_EQ(native_plan_source_names(c), "override+table");
        EXPECT_EQ(native_plan_config_text(c),
                "min_pass_k=512,int8_kc=64,l1_pass=0");
        const native_plan_t b = planner.plan(problem(64, 4096, 4096, true));
        EXPECT_EQ(native_plan_source_names(b), "override+table");
    }
}

TEST(NativeConfigPlanner, SettingsPlanner) {
    EXPECT_FALSE(make_native_planner(nullptr, nullptr)->has_sources());
    EXPECT_FALSE(make_native_planner("", "")->has_sources());
    EXPECT_TRUE(make_native_planner("nr=16", nullptr)->has_sources());
    std::unique_ptr<native_planner_t> planner;
    EXPECT_EQ(warnings_of([&] { planner = make_native_planner("nr=24", ""); }),
            shown(1));
    EXPECT_FALSE(planner->has_sources());
    EXPECT_EQ(warnings_of([&] {
        planner = make_native_planner(nullptr, "/nonexistent/tuning.json");
    }),
            shown(1));
    EXPECT_FALSE(planner->has_sources());
    const native_brgemm_test::temp_file_t file(
            "planner.json", tuning(R"("entries": [{"config": {"nr": 16}}])"));
    EXPECT_TRUE(
            make_native_planner(nullptr, file.path().c_str())->has_sources());
}

// The process planner reads both settings once, at its first use: in a child
// process that has them, the override and the file are its layers.
TEST(NativeConfigPlanner, ProcessPlannerReadsTheSettings) {
    if (!has_isa()) GTEST_SKIP() << "the CPU lacks AVX512-BF16 or AVX512-VNNI";
    const native_brgemm_test::temp_file_t file("process.json",
            tuning(R"("entries": [{"id": "file", "config": {"mb": 24}}])"));
    native_brgemm_test::expect_in_child(
            {{"ZENDNNL_MATMUL_NATIVE_CONFIG", "m=1-64:nr=16"},
                    {"ZENDNNL_MATMUL_NATIVE_TUNING_FILE", file.path().c_str()}},
            [] {
        int bad = 0;
        const native_planner_t &planner = native_planner();
        if (!planner.has_sources()) {
            std::fprintf(stderr, "the process planner has no layer\n");
            return 1;
        }
        const native_plan_t low_m = planner.plan(problem(8, 4096, 4096));
        if (native_plan_source_names(low_m) != "override+table"
                || low_m.nr != 16 || low_m.mb != 8) {
            std::fprintf(stderr, "M=8: %s nr=%d mb=%d\n",
                    native_plan_source_names(low_m).c_str(), low_m.nr,
                    low_m.mb);
            ++bad;
        }
        const native_plan_t large = planner.plan(problem(512, 4096, 4096));
        if (native_plan_source_names(large) != "table" || large.mb != 24) {
            std::fprintf(stderr, "M=512: %s mb=%d\n",
                    native_plan_source_names(large).c_str(), large.mb);
            ++bad;
        }
        return bad;
    }, "ZENDNNL_MATMUL_NATIVE_CONFIG and ZENDNNL_MATMUL_NATIVE_TUNING_FILE");
}

TEST(NativeConfigPlanner, RuleSwitchThroughTheOverride) {
    if (!has_isa()) GTEST_SKIP() << "the CPU lacks AVX512-BF16 or AVX512-VNNI";
    const UarchParams &u = native_plan_uarch();
    heuristic_params_t off;
    off.makespan = 0;
    const auto planner = make_native_planner("makespan=0", nullptr);
    ASSERT_TRUE(planner->has_sources());
    for (const bool bf16 : {true, false})
        for (const int M : {1, 32, 1031})
            for (const int N : {1536, 4096}) {
                const native_problem_t p = problem(M, N, 4096, bf16, 32);
                const native_plan_t q = planner->plan(p);
                EXPECT_TRUE(same_grid(q, heuristic_plan(p, u, off)))
                        << "M=" << M << " N=" << N << " bf16=" << bf16;
                EXPECT_EQ(top_source(q), source_kind::override);
                EXPECT_EQ(native_plan_config_text(q), "makespan=0");
                EXPECT_EQ(validate_native_plan(q, p), nullptr);
            }
}

// A miss, a hit, another problem of one shape, and M on both sides of the
// cache's bound give the plan of the source.
TEST(NativeConfigCache, HitMissAndLargeM) {
    if (!has_isa()) GTEST_SKIP() << "the CPU lacks AVX512-BF16 or AVX512-VNNI";
    const UarchParams &u = detect_uarch();
    const native_problem_t a = problem(8, 4096, 4096);
    native_planner_t none(u, nullptr, nullptr, nullptr);
    EXPECT_TRUE(same_plan(none.plan(a), heuristic_plan(a, u, {})));

    native_planner_t planner(
            u, make_native_override_source("nr=16"), nullptr, nullptr);
    const native_plan_t miss = planner.plan(a);
    const native_plan_t hit = planner.plan(a);
    EXPECT_TRUE(same_plan(miss, hit));
    EXPECT_EQ(miss.nr, 16);
    EXPECT_EQ(top_source(miss), source_kind::override);
    native_problem_t b = a;
    b.has_bias = true;
    heuristic_params_t hp;
    hp.nr = 16;
    for (const native_problem_t &p :
            {b, problem(kNativePlanCacheMaxM + 1, 4096, 4096),
                    problem(kNativePlanCacheMaxM, 4096, 4096)})
        for (int pass = 0; pass < 2; ++pass) {
            const native_plan_t q = planner.plan(p);
            EXPECT_TRUE(same_grid(q, heuristic_plan(p, u, hp))) << p.M;
            EXPECT_EQ(top_source(q), source_kind::override) << p.M;
        }
}

TEST(NativeConfigCache, FullWindowsStayCorrect) {
    if (!has_isa()) GTEST_SKIP() << "the CPU lacks AVX512-BF16 or AVX512-VNNI";
    const UarchParams &u = detect_uarch();
    native_planner_t planner(
            u, make_native_override_source("nr=16"), nullptr, nullptr);
    heuristic_params_t hp;
    hp.nr = 16;
    std::vector<native_problem_t> problems;
    for (int M = 1; M <= 1000; ++M)
        for (int j = 1; j <= 20; ++j)
            problems.push_back(problem(M, 32 * j, 256));
    int wrong = 0;
    for (const native_problem_t &p : problems) {
        const native_plan_t q = planner.plan(p);
        wrong += !same_grid(q, heuristic_plan(p, u, hp))
                || top_source(q) != source_kind::override;
    }
    EXPECT_EQ(wrong, 0);
    for (const native_problem_t &p : problems) {
        const native_plan_t q = planner.plan(p);
        wrong += !same_grid(q, heuristic_plan(p, u, hp))
                || top_source(q) != source_kind::override;
    }
    EXPECT_EQ(wrong, 0);
}

TEST(NativeConfigCache, ConcurrentHits) {
    if (!has_isa()) GTEST_SKIP() << "the CPU lacks AVX512-BF16 or AVX512-VNNI";
    const UarchParams &u = detect_uarch();
    native_planner_t planner(u,
            make_native_override_source("m=1-256:nr=16;mb=24"), nullptr,
            nullptr);
    std::vector<native_problem_t> problems;
    std::vector<native_plan_t> expected;
    for (int M = 1; M <= 512; ++M) {
        problems.push_back(problem(M, 4096, 4096, M % 2 == 0, 24));
        heuristic_params_t hp;
        if (M <= 256)
            hp.nr = 16;
        else
            hp.mb = 24;
        expected.push_back(heuristic_plan(problems.back(), u, hp));
    }
    std::atomic<int> wrong {0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            for (int round = 0; round < 4; ++round)
                for (size_t i = 0; i < problems.size(); ++i) {
                    const size_t at = (i + 64 * t) % problems.size();
                    if (!same_grid(planner.plan(problems[at]), expected[at]))
                        wrong.fetch_add(1);
                }
        });
    }
    for (std::thread &t : threads)
        t.join();
    EXPECT_EQ(wrong.load(), 0);
}

namespace {

/// Bit mr - 1 set for every row count kind has a kernel for at nr (INT8:
/// for both sources).
uint32_t kernel_rows(native_kernel_kind_t kind, int nr) {
    uint32_t mask = 0;
    for (int mr = 1; mr <= 12; ++mr) {
        bool served = false;
        if (kind == native_kernel_kind_t::bf16)
            served = select_bf16_brgemm_tall_kernel(
                             mr, native_bf16_slab_cols(nr))
                    != nullptr;
        else if (kind == native_kernel_kind_t::f32)
            served = select_native_fp32_ukernel(mr, nr / 16) != nullptr;
        else
            served = select_native_int8_ukernel(mr, nr / 16, false) != nullptr
                    && select_native_int8_ukernel(mr, nr / 16, true) != nullptr;
        if (served) mask |= 1u << (mr - 1);
    }
    return mask;
}

/// Largest m such that every row count 1 .. m is in mask.
int full_rows(uint32_t mask) {
    int m = 0;
    while (m < 32 && ((mask >> m) & 1u) != 0)
        ++m;
    return m;
}

} // namespace

TEST(NativeConfigRegistry, MatchesTheSelectors) {
    const native_kernel_registry_t &r = native_kernel_registry();
    const UarchParams &u = detect_uarch();
    // Tiles of 64 and 128 columns run the nr 32 kernels slab by slab.
    for (const int nr : {16, 32, 64, 128}) {
        const native_kernel_caps_t *c = r.caps(native_kernel_kind_t::bf16, nr);
        if (!u.avx512bf16) {
            EXPECT_EQ(c, nullptr);
            EXPECT_EQ(r.bf16_table(nr), nullptr);
            continue;
        }
        ASSERT_NE(c, nullptr) << nr;
        for (int mr = 1; mr <= 12; ++mr)
            EXPECT_EQ(r.bf16_table(nr)[mr],
                    select_bf16_brgemm_tall_kernel(
                            mr, native_bf16_slab_cols(nr)))
                    << nr << " " << mr;
        const int max_mr
                = full_rows(kernel_rows(native_kernel_kind_t::bf16, nr));
        EXPECT_EQ(c->max_mr, max_mr);
        EXPECT_TRUE(r.covers(native_kernel_kind_t::bf16, max_mr, nr));
        EXPECT_FALSE(r.covers(native_kernel_kind_t::bf16, max_mr + 1, nr));
        EXPECT_EQ(c->int8_schemes, 0u);
    }
    for (const int nr : {16, 32}) {
        const native_kernel_caps_t *c = r.caps(native_kernel_kind_t::int8, nr);
        if (!u.avx512vnni) {
            EXPECT_EQ(c, nullptr);
            continue;
        }
        ASSERT_NE(c, nullptr) << nr;
        EXPECT_EQ(c->max_mr,
                full_rows(kernel_rows(native_kernel_kind_t::int8, nr)));
        EXPECT_EQ(c->max_mr, native_int8_ukernel_max_mr(nr / 16));
        // The per-channel kernels serve per-channel calls, the group kernels
        // per-group calls.
        const native_kernel_caps_t *g
                = r.int8_caps(nr, native_int8_scheme_t::per_group);
        ASSERT_NE(g, nullptr) << nr;
        EXPECT_EQ(r.int8_caps(nr, native_int8_scheme_t::per_channel), c);
        EXPECT_EQ(g->max_mr, native_int8_group_ukernel_max_mr(nr / 16));
        for (const native_kernel_caps_t *k : {c, g}) {
            const native_int8_scheme_t s = k == c
                    ? native_int8_scheme_t::per_channel
                    : native_int8_scheme_t::per_group;
            EXPECT_TRUE(k->serves(s)) << nr;
            EXPECT_FALSE(k->serves(k == c ? native_int8_scheme_t::per_group
                                          : native_int8_scheme_t::per_channel))
                    << nr;
            EXPECT_TRUE(r.covers_int8(k->max_mr, nr, s)) << nr;
            EXPECT_FALSE(r.covers_int8(k->max_mr + 1, nr, s)) << nr;
            EXPECT_FALSE(r.covers_int8(0, nr, s)) << nr;
        }
    }
    // The s4 group kernels, both orders, serve per-group calls.
    for (const int nr : {16, 32}) {
        const native_kernel_caps_t *c = r.caps(native_kernel_kind_t::int4, nr);
        if (!u.avx512vnni) {
            EXPECT_EQ(c, nullptr);
            continue;
        }
        ASSERT_NE(c, nullptr) << nr;
        EXPECT_EQ(c->max_mr, native_int4_group_ukernel_max_mr(nr / 16));
        for (int mr = 1; mr <= c->max_mr; ++mr)
            for (const native_int8_order_t order : {native_int8_order_t::fused,
                         native_int8_order_t::reference})
                EXPECT_NE(select_native_int4_group_ukernel(mr, nr / 16, order),
                        nullptr)
                        << nr << " " << mr;
        EXPECT_TRUE(c->serves(native_int8_scheme_t::per_group)) << nr;
        EXPECT_FALSE(c->serves(native_int8_scheme_t::per_channel)) << nr;
        EXPECT_TRUE(r.covers(native_kernel_kind_t::int4, c->max_mr, nr)) << nr;
        EXPECT_FALSE(r.covers(native_kernel_kind_t::int4, c->max_mr + 1, nr))
                << nr;
        EXPECT_NE(r.int8_caps(nr, native_int8_scheme_t::per_group), c) << nr;
    }
    // FP32 tiles are 16 or 32 columns wide, every row count 1 .. 12.
    for (const int nr : {16, 32}) {
        const native_kernel_caps_t *c = r.caps(native_kernel_kind_t::f32, nr);
        if (!u.avx512f) {
            EXPECT_EQ(c, nullptr);
            EXPECT_EQ(r.f32_table(nr), nullptr);
            continue;
        }
        ASSERT_NE(c, nullptr) << nr;
        for (int mr = 1; mr <= 12; ++mr)
            EXPECT_EQ(r.f32_table(nr)[mr],
                    select_native_fp32_ukernel(mr, nr / 16))
                    << nr << " " << mr;
        EXPECT_EQ(c->max_mr,
                full_rows(kernel_rows(native_kernel_kind_t::f32, nr)));
        EXPECT_EQ(c->max_mr, native_fp32_ukernel_max_mr(nr / 16));
        EXPECT_EQ(c->max_mr, 12);
        EXPECT_TRUE(r.covers(native_kernel_kind_t::f32, 12, nr));
        EXPECT_FALSE(r.covers(native_kernel_kind_t::f32, 13, nr));
        EXPECT_FALSE(r.covers(native_kernel_kind_t::f32, 0, nr));
        EXPECT_EQ(c->int8_schemes, 0u);
    }
    for (const int nr : {48, 96, 256}) {
        EXPECT_EQ(r.caps(native_kernel_kind_t::bf16, nr), nullptr) << nr;
        EXPECT_EQ(r.caps(native_kernel_kind_t::int8, nr), nullptr) << nr;
        EXPECT_EQ(r.caps(native_kernel_kind_t::int4, nr), nullptr) << nr;
        EXPECT_EQ(r.caps(native_kernel_kind_t::f32, nr), nullptr) << nr;
        EXPECT_EQ(r.bf16_table(nr), nullptr) << nr;
        EXPECT_EQ(r.f32_table(nr), nullptr) << nr;
    }
    for (const int nr : {64, 128}) {
        EXPECT_EQ(r.caps(native_kernel_kind_t::int8, nr), nullptr) << nr;
        EXPECT_EQ(r.caps(native_kernel_kind_t::int4, nr), nullptr) << nr;
        EXPECT_EQ(r.caps(native_kernel_kind_t::f32, nr), nullptr) << nr;
        EXPECT_EQ(r.f32_table(nr), nullptr) << nr;
    }
    EXPECT_FALSE(r.covers(native_kernel_kind_t::int8, 0, 16));
    if (u.avx512bf16 && u.avx512vnni) {
        EXPECT_EQ(kernel_rows(native_kernel_kind_t::bf16, 16), 0x3fu);
        EXPECT_EQ(kernel_rows(native_kernel_kind_t::bf16, 32), 0xfffu);
        EXPECT_EQ(r.caps(native_kernel_kind_t::int8, 16)->max_mr, 12);
        EXPECT_EQ(r.caps(native_kernel_kind_t::int8, 32)->max_mr, 12);
    }
}

// The CPU's topology and the L3 domain the built-in plan uses: the physical
// cores of one L3 when the topology gives them, else the family's.
TEST(NativeConfigHw, Probe) {
    const UarchParams &u = detect_uarch();
    const native_hw_t &hw = native_hw();
    const UarchParams &plan = native_plan_uarch();
    const int domain = plan.l3_domain_cores > 0 ? plan.l3_domain_cores
                                                : plan.family_l3_cores;
    EXPECT_EQ(hw.uarch.l2_bytes, u.l2_bytes);
    EXPECT_GT(hw.logical_cpus, 0);
#if !defined(_WIN32)
    if (hw.smt > 0) { EXPECT_EQ(hw.physical_cores, hw.logical_cpus / hw.smt); }
#endif
    EXPECT_EQ(u.l3_domain_cores, 0);
    const int smt = hw.smt > 0 ? hw.smt : 1;
    const int cores = hw.l3_domain_threads / smt;
    EXPECT_EQ(plan.l3_domain_cores,
            hw.l3_domain_threads > 0 ? (cores > 0 ? cores : 1) : 0);
    EXPECT_GE(domain, 1);
    EXPECT_EQ(plan.family_l3_cores, u.family_l3_cores);
    EXPECT_EQ(plan.l1d_bytes, u.l1d_bytes);
    EXPECT_EQ(plan.l2_bytes, u.l2_bytes);
    EXPECT_EQ(plan.l3_domain_bytes, u.l3_detected ? u.l3_domain_bytes : 0);
    EXPECT_EQ(native_plan_uarch(test_hw()).l3_domain_cores, 8);
}

namespace {

const char *key_type_name(native_key_type_t t) {
    switch (t) {
        case native_key_type_t::int32: return "int";
        case native_key_type_t::int64: return "int64";
        case native_key_type_t::boolean: return "bool";
        case native_key_type_t::cw_list: return "list";
        default: return "enum";
    }
}

/// The INT8 schemes of a mask of native_int8_scheme_bit.
json scheme_names(uint8_t mask) {
    json names = json::array();
    for (const native_int8_scheme_t s : {native_int8_scheme_t::per_channel,
                 native_int8_scheme_t::per_group}) {
        if ((mask & native_int8_scheme_bit(s)) != 0)
            names.push_back(native_int8_scheme_name(s));
    }
    return names;
}

json cw_widths(int64_t mask) {
    json widths = json::array();
    if ((mask & kNativeInt8Cw32) != 0) widths.push_back(32);
    if ((mask & kNativeInt8Cw16) != 0) widths.push_back(16);
    return widths;
}

/// The schema of key k: its type, legal values and default, the problems
/// and INT8 schemes it applies to, and its effect on the output bits.
nlohmann::ordered_json key_json(const native_config_key_t &k) {
    const key_doc_t *d = key_doc(k);
    nlohmann::ordered_json j;
    j["name"] = k.name;
    j["type"] = key_type_name(k.type);
    if (d == nullptr) return j;
    const int64_t builtin = d->value(heuristic_params_t {});
    switch (k.type) {
        case native_key_type_t::int32:
        case native_key_type_t::int64: {
            int64_t text_value = 0;
            if (native_text_int64(d->builtin, text_value)) {
                j["default"] = builtin;
            } else {
                j["default"] = d->builtin;
                j["auto"] = builtin;
            }
            if (k.pow2) {
                json values = json::array();
                for (int64_t v = k.min; v <= k.max; v *= 2)
                    values.push_back(v);
                j["values"] = values;
            } else {
                j["min"] = k.min;
                j["max"] = k.max;
                if (k.step > 1) j["multiple_of"] = k.step;
            }
            break;
        }
        case native_key_type_t::boolean: j["default"] = builtin != 0; break;
        case native_key_type_t::cw_list:
            j["default"] = cw_widths(builtin);
            j["values"] = cw_widths(kNativeInt8CwAll);
            j["must_contain"] = json::array({16});
            break;
        default:
            j["default"] = d->builtin;
            j["values"] = native_brgemm_test::native_enum_values(k.type);
            break;
    }
    j["applies"] = d->applies;
    j["int8_schemes"] = scheme_names(
            std::string(d->applies) == "bf16" ? 0 : k.int8_schemes);
    j["bits"] = d->same_bits ? "same" : "changes";
    j["kind"] = d->pin ? "pin" : "param";
    return j;
}

/// What the planner reads and offers on this build and CPU: the config keys,
/// the match keys, rule features and comparisons, the kernels, the hardware
/// and what the plan reads of it (uarch), and this CPU's value of every
/// hardware key (hw_match).
nlohmann::ordered_json schema_json() {
    const native_hw_t &hw = native_hw();
    const UarchParams &u = native_plan_uarch();
    nlohmann::ordered_json doc;
    doc["format"] = "zendnnl-native-brgemm-schema";
    doc["version"] = 1;
    json isa = json::array();
    for (const char *f : kNativeIsaNames) {
        const std::string name = f;
        if ((name == "avx512f" && hw.uarch.avx512f)
                || (name == "avx512bf16" && hw.uarch.avx512bf16)
                || (name == "avx512vnni" && hw.uarch.avx512vnni))
            isa.push_back(name);
    }
    doc["hw"] = {{"vendor", hw.vendor}, {"family", hw.family},
            {"model", hw.model}, {"stepping", hw.stepping}, {"isa", isa},
            {"l1d_kib", hw.uarch.l1d_bytes / 1024},
            {"l2_kib", hw.uarch.l2_bytes / 1024},
            {"l3_kib", hw.uarch.l3_domain_bytes / 1024},
            {"logical_cpus", hw.logical_cpus}, {"smt", hw.smt},
            {"physical_cores", hw.physical_cores},
            {"l3_domain_threads", hw.l3_domain_threads}};
    doc["uarch"] = {{"l1d_bytes", u.l1d_bytes}, {"l2_bytes", u.l2_bytes},
            {"l3_bytes", u.l3_domain_bytes}, {"num_cores", u.num_cores},
            {"family_l3_cores", u.family_l3_cores},
            {"l3_domain_cores", u.l3_domain_cores}, {"avx512f", u.avx512f},
            {"avx512bf16", u.avx512bf16}, {"avx512vnni", u.avx512vnni}};
    doc["hw_match"] = nlohmann::ordered_json::parse(
            native_brgemm_test::hw_match_text(hw));

    size_t count = 0;
    const native_config_key_t *keys = native_config_keys(count);
    nlohmann::ordered_json key_list = nlohmann::ordered_json::array();
    for (size_t i = 0; i < count; ++i)
        key_list.push_back(key_json(keys[i]));
    doc["keys"] = key_list;

    nlohmann::ordered_json match = nlohmann::ordered_json::object();
    const std::vector<std::vector<std::string>> named = {
            {std::begin(kNativeDtypeNames), std::end(kNativeDtypeNames)},
            {std::begin(kNativeSrcNames), std::end(kNativeSrcNames)},
            {std::begin(kNativeDstNames), std::end(kNativeDstNames)},
            {std::begin(kNativeEpilogueNames), std::end(kNativeEpilogueNames)}};
    for (int i = 0; i < static_cast<int>(std::size(kNativeMatchKeys)); ++i) {
        if (i < kNativeNumericMatchKeys)
            match[kNativeMatchKeys[i]] = "range";
        else if (i < kNativeNumericMatchKeys + kNativeNamedMatchKeys)
            match[kNativeMatchKeys[i]] = named[i - kNativeNumericMatchKeys];
        else
            match[kNativeMatchKeys[i]] = "bool";
    }
    doc["match_keys"] = match;
    doc["features"] = std::vector<std::string>(
            std::begin(kNativeRuleFeatures), std::end(kNativeRuleFeatures));
    doc["rule_ops"] = std::vector<std::string>(
            std::begin(kNativeRuleOps), std::end(kNativeRuleOps));

    nlohmann::ordered_json kernels = nlohmann::ordered_json::array();
    const native_kernel_registry_t &r = native_kernel_registry();
    for (const native_kernel_kind_t kind :
            {native_kernel_kind_t::bf16, native_kernel_kind_t::int8})
        for (const int nr : {16, 32, 64, 128}) {
            const native_kernel_caps_t *c = r.caps(kind, nr);
            if (c == nullptr) continue;
            const bool bf16 = kind == native_kernel_kind_t::bf16;
            kernels.push_back({{"kind", bf16 ? "bf16" : "int8"}, {"nr", nr},
                    {"mr_mask", kernel_rows(kind, nr)}, {"max_mr", c->max_mr},
                    {"k_unit", bf16 ? 2 : 4},
                    {"int8_schemes", scheme_names(c->int8_schemes)}});
        }
    for (const int nr : {16, 32}) {
        const native_kernel_caps_t *c
                = r.int8_caps(nr, native_int8_scheme_t::per_group);
        if (c == nullptr) continue;
        uint32_t mask = 0;
        for (int mr = 1; mr <= 12; ++mr)
            if (select_native_int8_group_ukernel(mr, nr / 16, false) != nullptr
                    && select_native_int8_group_ukernel(mr, nr / 16, true)
                            != nullptr)
                mask |= 1u << (mr - 1);
        kernels.push_back({{"kind", "int8"}, {"nr", nr}, {"mr_mask", mask},
                {"max_mr", c->max_mr}, {"k_unit", 4},
                {"int8_schemes", scheme_names(c->int8_schemes)}});
    }
    for (const int nr : {16, 32}) {
        const native_kernel_caps_t *c = r.caps(native_kernel_kind_t::int4, nr);
        if (c == nullptr) continue;
        uint32_t mask = 0;
        for (int mr = 1; mr <= 12; ++mr)
            if (select_native_int4_group_ukernel(
                        mr, nr / 16, native_int8_order_t::fused)
                            != nullptr
                    && select_native_int4_group_ukernel(
                               mr, nr / 16, native_int8_order_t::reference)
                            != nullptr)
                mask |= 1u << (mr - 1);
        kernels.push_back({{"kind", "int4"}, {"nr", nr}, {"mr_mask", mask},
                {"max_mr", c->max_mr}, {"k_unit", kNativeS4Octet},
                {"int8_schemes", scheme_names(c->int8_schemes)}});
    }
    for (const int nr : {16, 32}) {
        const native_kernel_caps_t *c = r.caps(native_kernel_kind_t::f32, nr);
        if (c == nullptr) continue;
        kernels.push_back({{"kind", "f32"}, {"nr", nr},
                {"mr_mask", kernel_rows(native_kernel_kind_t::f32, nr)},
                {"max_mr", c->max_mr}, {"k_unit", 1},
                {"int8_schemes", scheme_names(c->int8_schemes)}});
    }
    doc["kernels"] = kernels;
    doc["tuning_file"] = {{"format", kNativeTuningFormat},
            {"version", kNativeTuningVersion},
            {"hardware_keys",
                    std::vector<std::string>(std::begin(kNativeHardwareKeys),
                            std::end(kNativeHardwareKeys))},
            {"isa",
                    std::vector<std::string>(std::begin(kNativeIsaNames),
                            std::end(kNativeIsaNames))}};
    return doc;
}

} // namespace

// Prints the schema between two marker lines, which delimit it for a parser
// of the test output.
TEST(NativeConfigSchema, SchemaDump) {
    const std::string text = schema_json().dump(2);
    std::cout << "BEGIN NATIVE TUNING SCHEMA\n"
              << text << "\nEND NATIVE TUNING SCHEMA" << std::endl;
    const json doc = json::parse(text, nullptr, false);
    ASSERT_FALSE(doc.is_discarded());
    EXPECT_EQ(doc.at("format"), "zendnnl-native-brgemm-schema");
    EXPECT_EQ(doc.at("version"), 1);
    const native_hw_t &hw = native_hw();
    EXPECT_EQ(doc.at("hw").at("vendor"), hw.vendor);
    EXPECT_EQ(doc.at("hw").at("smt"), hw.smt);
    EXPECT_EQ(doc.at("uarch").at("l2_bytes"), detect_uarch().l2_bytes);
    EXPECT_EQ(doc.at("uarch").at("family_l3_cores"),
            detect_uarch().family_l3_cores);
    EXPECT_EQ(doc.at("uarch").at("l3_domain_cores"),
            native_plan_uarch().l3_domain_cores);
    size_t count = 0;
    const native_config_key_t *keys = native_config_keys(count);
    const json &k = doc.at("keys");
    ASSERT_EQ(k.size(), count);
    for (size_t i = 0; i < count; ++i)
        EXPECT_EQ(k[i].at("name"), keys[i].name);
    const auto key = [&k](const std::string &name) -> const json & {
        for (const json &e : k)
            if (e.at("name") == name) return e;
        static const json none = json::object();
        return none;
    };
    EXPECT_EQ(key("nr").value("default", json()), "auto");
    EXPECT_EQ(key("nr").value("auto", json()), 0);
    EXPECT_EQ(
            key("nr").value("values", json()), json::array({16, 32, 64, 128}));
    EXPECT_EQ(key("col_groups").value("auto", json()), -1);
    EXPECT_EQ(key("schedule").value("values", json()),
            json::array({"auto", "slice", "row_groups", "blocks"}));
    EXPECT_EQ(key("unit_tiles").value("auto", json()), 0);
    EXPECT_EQ(key("row_split").value("default", json()), true);
    EXPECT_EQ(key("min_pass_k").value("multiple_of", json()), 2);
    EXPECT_EQ(key("min_pass_k").value("applies", json()), "both");
    EXPECT_EQ(key("int8_cw").value("default", json()), json::array({32, 16}));
    EXPECT_EQ(key("int8_cw").value("must_contain", json()), json::array({16}));
    EXPECT_EQ(key("bias_in_dequant").value("bits", json()), "changes");
    EXPECT_EQ(key("threads_cap").value("default", json()), "none");
    EXPECT_EQ(
            key("domain_threads").value("default", json()), "l3_domain_cores");
    // The rule switches, on by default.
    const std::vector<std::string> switches
            = native_brgemm_test::native_rule_switches();
    EXPECT_EQ(switches,
            (std::vector<std::string> {"makespan", "l1_pass", "traffic_groups",
                    "bf16_alias", "bf16_narrow_n", "blocks", "cost_model"}));
    for (const std::string &name : switches) {
        EXPECT_EQ(key(name).value("default", json()), 1) << name;
        EXPECT_FALSE(key(name).contains("auto")) << name;
        EXPECT_EQ(key(name).value("min", json()), 0) << name;
        EXPECT_EQ(key(name).value("max", json()), 1) << name;
        EXPECT_EQ(key(name).value("bits", json()), "same") << name;
    }
    for (const auto &[name, value] : std::vector<std::pair<const char *, int>> {
                 {"min_gain_pct", 5}, {"idle_pct", 10}, {"decode_m", 128},
                 {"l1_pct", 100}, {"l1_pass_max_mb", 64}, {"bf16_alias_mr", 8},
                 {"bf16_narrow_n_min_m", 1024}, {"bf16_blocks_min_m", 1024},
                 {"int8_blocks_min_m", 3072}, {"cm_gain_pct", 5},
                 {"cm_short_rows", 8}, {"cm_fma_lat", 6},
                 {"cm_call_cycles", 800}, {"cm_l2_bytes", 16},
                 {"cm_l3_cycles", 10}, {"cm_prefetch_kib", 4},
                 {"cm_overlap_pct", 20}, {"cm_reuse_pct", 50},
                 {"cm_stream_cycles", 600}, {"cm_sync_tiles", 5},
                 {"cm_group_slab_pct", 75}, {"cm_group_l3_pct", 75},
                 {"cm_group_skew_pct", 110}, {"cm_group_rows_pct", 115},
                 {"cm_drift_kib", 4096}, {"cm_pass_kib", 128},
                 {"cm_pass_calls", 4}, {"cm_block_kib", 16384},
                 {"cm_int8_l1_pct", 0}, {"cm_int4_min_m", 48},
                 {"int4_l1_pct", 125}}) {
        EXPECT_EQ(key(name).value("default", json()), value) << name;
        EXPECT_EQ(key(name).value("bits", json()), "same") << name;
    }
    // The cost model's inputs and its row-group bounds apply to BF16 and
    // per-channel INT8 plans, its stream rules to BF16 and its INT8 rule to
    // INT8; the FMA pipes default to the detected ports and the memory rate
    // to the L3 domain.
    EXPECT_EQ(key("cm_fma_pipes").value("default", json()), "fma_ports");
    EXPECT_EQ(key("cm_fma_pipes").value("auto", json()), 0);
    EXPECT_EQ(key("cm_mem_cycles").value("default", json()), "domain_threads");
    EXPECT_EQ(key("cm_mem_cycles").value("auto", json()), 0);
    EXPECT_EQ(key("cm_tall_pct").value("default", json()), "domain_threads");
    EXPECT_EQ(key("cm_tall_pct").value("auto", json()), -1);
    for (const char *name : {"cost_model", "cm_gain_pct", "cm_short_rows",
                 "cm_fma_pipes", "cm_fma_lat", "cm_call_cycles", "cm_tall_pct",
                 "cm_l2_bytes", "cm_l3_cycles", "cm_mem_cycles",
                 "cm_prefetch_kib", "cm_overlap_pct", "cm_reuse_pct",
                 "cm_stream_cycles", "cm_sync_tiles", "cm_group_slab_pct",
                 "cm_group_l3_pct", "cm_group_skew_pct", "cm_group_rows_pct"}) {
        EXPECT_EQ(key(name).value("applies", json()), "both") << name;
        EXPECT_EQ(key(name).value("kind", json()), "param") << name;
    }
    for (const char *name :
            {"cm_drift_kib", "cm_pass_kib", "cm_pass_calls", "cm_block_kib"}) {
        EXPECT_EQ(key(name).value("applies", json()), "bf16") << name;
        EXPECT_EQ(key(name).value("kind", json()), "param") << name;
    }
    for (const char *name : {"cm_int8_l1_pct", "cm_int4_min_m"}) {
        EXPECT_EQ(key(name).value("applies", json()), "int8") << name;
        EXPECT_EQ(key(name).value("kind", json()), "param") << name;
    }
    // The INT8 schemes a key applies to: the per-group tile reads neither
    // the K pass nor the bias placement; the cost model's switch, margin and
    // the inputs of its price apply to both (it prices the per-group tile of
    // an s4 weight), its row-group bounds and INT8 rule to per-channel calls
    // and its s4 row blocks to per-group calls.
    const json both = json::array({"per_channel", "per_group"});
    for (const json &e : k) {
        const std::string name = e.at("name");
        const bool per_channel = name == "int8_kc" || name == "int8_l2_pct"
                || name == "bias_in_dequant" || name == "min_pass_k"
                || name == "l1_pass" || name == "l1_pct"
                || name == "l1_pass_max_mb" || name == "cm_reuse_pct"
                || name.rfind("cm_group_", 0) == 0 || name == "cm_int8_l1_pct";
        const bool per_group = name == "cm_int4_min_m" || name == "int4_l1_pct";
        const json want = e.at("applies") == "bf16" ? json::array()
                : per_channel ? json::array({"per_channel"})
                : per_group   ? json::array({"per_group"})
                              : both;
        EXPECT_EQ(e.at("int8_schemes"), want) << name;
    }
    // Each INT8 kernel family serves one scheme; the s4 kernels per group,
    // K in octets, every row count to their widest tile.
    size_t group_kernels = 0;
    size_t int4_kernels = 0;
    for (const json &c : doc.at("kernels")) {
        if (c.at("kind") == "int4") {
            ++int4_kernels;
            EXPECT_EQ(c.at("int8_schemes"), json::array({"per_group"}));
            EXPECT_EQ(c.at("k_unit"), 8);
            const int max_mr = c.at("max_mr");
            EXPECT_EQ(c.at("mr_mask"), (1u << max_mr) - 1) << c.at("nr");
            continue;
        }
        if (c.at("kind") != "int8") {
            EXPECT_EQ(c.at("int8_schemes"), json::array());
            continue;
        }
        const bool group = c.at("int8_schemes") == json::array({"per_group"});
        group_kernels += group ? 1 : 0;
        EXPECT_TRUE(
                group || c.at("int8_schemes") == json::array({"per_channel"}));
    }
    EXPECT_EQ(group_kernels, detect_uarch().avx512vnni ? 2u : 0u);
    EXPECT_EQ(int4_kernels, detect_uarch().avx512vnni ? 2u : 0u);
    // The FP32 kernels: every row count 1 .. 12 at 16 and 32 columns, K in
    // steps of one.
    std::vector<json> f32_kernels;
    for (const json &c : doc.at("kernels"))
        if (c.at("kind") == "f32") f32_kernels.push_back(c);
    if (detect_uarch().avx512f) {
        ASSERT_EQ(f32_kernels.size(), 2u);
        for (size_t i = 0; i < 2; ++i) {
            EXPECT_EQ(f32_kernels[i].at("nr"), 16 << i);
            EXPECT_EQ(f32_kernels[i].at("mr_mask"), 0xfff);
            EXPECT_EQ(f32_kernels[i].at("max_mr"), 12);
            EXPECT_EQ(f32_kernels[i].at("k_unit"), 1);
        }
    } else {
        EXPECT_TRUE(f32_kernels.empty());
    }
    EXPECT_EQ(doc.at("match_keys").size(), 13u);
    EXPECT_EQ(doc.at("match_keys").at("group"), "range");
    EXPECT_EQ(doc.at("match_keys").at("dtype"),
            json::array({"bf16", "s8", "f32", "s4"}));
    EXPECT_EQ(doc.at("features").size(), 27u);
    EXPECT_EQ(doc.at("features").back(), "group");
    EXPECT_EQ(
            doc.at("tuning_file").at("format"), "zendnnl-native-brgemm-tuning");
    EXPECT_EQ(doc.at("tuning_file").at("version"), 2);
}
