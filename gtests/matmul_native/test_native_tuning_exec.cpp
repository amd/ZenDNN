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

// Tuning inputs of the native BRGEMM (ALGO 10) through matmul_direct.  The
// process planner reads ZENDNNL_MATMUL_NATIVE_CONFIG and
// ZENDNNL_MATMUL_NATIVE_TUNING_FILE once, so the calls under a setting run
// in a child process; the outputs under the built-in plan come from the
// parent.  Every key whose schema bits are "same" must give the output bits
// of the built-in plan, on BF16, INT8 and FP32 calls that reach the paths the keys
// steer (narrowing, row split, K passes, row groups, the rule switches, INT8
// call widths, rows and K passes, direct store, bias in the dequantization,
// source copies, gated and softmax epilogues, alpha / beta);
// bias_in_dequant=0 must stay in the comparison band.  Tuning files (a table
// with tree rules, ordered rules, and an override on top of a file) must
// keep the bits too.  A bad input never fails a call: illegal or malformed
// overrides, unreadable or mismatching files and bad entries fall back to
// the next source, with one warning per entry.  Planning from many threads
// over many problems, and calls from many threads, must give the plans and
// outputs of one thread.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "common/zendnnl_global.hpp"
#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_call.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_file.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_tuning_keys.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"
#include "nlohmann/json.hpp"

namespace {

namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
namespace mm = zendnnl::lowoha::matmul;
using nt::buffer_t;
using source_t = nat::native_plan_source_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::post_op_type_t;
using zendnnl::error_handling::status_t;
using zendnnl::memory::data_type_t;
using gated_t = mm::matmul_gated_act_t;

constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kS8 = data_type_t::s8;
constexpr data_type_t kU8 = data_type_t::u8;
constexpr data_type_t kS32 = data_type_t::s32;
constexpr data_type_t kNone = data_type_t::none;

constexpr const char *kConfig = "ZENDNNL_MATMUL_NATIVE_CONFIG";
constexpr const char *kTuningFile = "ZENDNNL_MATMUL_NATIVE_TUNING_FILE";

/// dyn_s4 / dyn_s4_groups: a bf16 source quantized per token / per group of
/// K times an s4 weight in groups of kInt4Group with bf16 scales.
enum class src_t { bf16, dyn_s8, st_u8, st_s8, f32, dyn_s4, dyn_s4_groups };

constexpr int kInt4Group = 128;

/// One dense call; threads is its params.num_threads.
struct case_t {
    const char *name;
    src_t src;
    int M, N, K;
    data_type_t dst;
    int threads;
    bool trans_a = false;
    data_type_t bias = kNone;
    post_op_type_t post_op = post_op_type_t::none;
    gated_t gated = gated_t::none;
    float alpha = 1.0f;
    float beta = 0.0f;

    bool int8() const { return src != src_t::bf16 && src != src_t::f32; }
    bool int4() const {
        return src == src_t::dyn_s4 || src == src_t::dyn_s4_groups;
    }
    int cols() const { return gated == gated_t::none ? N : N / 2; }
};

/// Calls that reach every path a tuning key steers.
std::vector<case_t> all_cases() {
    const post_op_type_t none = post_op_type_t::none;
    return {
            {"bf16_decode", src_t::bf16, 1, 1536, 1000, kBf16, 8},
            {"bf16_decode_narrowed", src_t::bf16, 2, 320, 512, kBf16, 8},
            {"bf16_narrow_odd_k", src_t::bf16, 4, 64, 777, kF32, 8},
            {"bf16_row_split", src_t::bf16, 100, 96, 128, kBf16, 8},
            {"bf16_k_passes", src_t::bf16, 200, 64, 9000, kBf16, 8},
            {"bf16_trans_bias_gelu", src_t::bf16, 300, 256, 320, kBf16, 8, true,
                    kF32, post_op_type_t::gelu_erf},
            {"bf16_gated_bias", src_t::bf16, 24, 1024, 320, kBf16, 8, false,
                    kF32, none, gated_t::silu_and_mul},
            {"bf16_softmax", src_t::bf16, 16, 128, 64, kBf16, 4, false, kNone,
                    post_op_type_t::softmax},
            {"bf16_alpha_beta", src_t::bf16, 16, 96, 100, kBf16, 8, false,
                    kNone, none, gated_t::none, 0.5f, 2.0f},
            {"int8_dyn_decode", src_t::dyn_s8, 1, 1536, 1024, kBf16, 8},
            {"int8_dyn_decode_narrowed", src_t::dyn_s8, 3, 320, 256, kBf16, 8},
            {"int8_dyn_bias_k_passes", src_t::dyn_s8, 40, 256, 7200, kBf16, 8,
                    false, kF32},
            {"int8_u8_relu_padded_k", src_t::st_u8, 65, 256, 1001, kF32, 8,
                    false, kNone, post_op_type_t::relu},
            {"int8_u8_trans", src_t::st_u8, 7, 96, 100, kF32, 4, true},
            {"int8_s8", src_t::st_s8, 20, 64, 96, kBf16, 4},
            {"int8_dyn_trans_prompt", src_t::dyn_s8, 300, 512, 256, kF32, 8,
                    true},
            {"int8_dyn_gated", src_t::dyn_s8, 8, 256, 128, kBf16, 4, false,
                    kNone, none, gated_t::silu_and_mul},
            {"int8_u8_alpha_beta", src_t::st_u8, 16, 96, 100, kF32, 4, false,
                    kNone, none, gated_t::none, 0.5f, 2.0f},
            {"int8_dyn_bf16_bias_softmax", src_t::dyn_s8, 16, 128, 64, kF32, 4,
                    false, kBf16, post_op_type_t::softmax},
            {"f32_decode", src_t::f32, 1, 1536, 1000, kF32, 8},
            {"f32_decode_narrowed", src_t::f32, 2, 320, 512, kF32, 8},
            {"f32_row_split", src_t::f32, 100, 96, 128, kF32, 8},
            {"f32_k_passes", src_t::f32, 200, 64, 9000, kF32, 8},
            {"f32_trans_bias_gelu", src_t::f32, 300, 256, 320, kF32, 8, true,
                    kF32, post_op_type_t::gelu_erf},
            {"f32_gated_bias", src_t::f32, 24, 1024, 320, kF32, 8, false, kF32,
                    none, gated_t::silu_and_mul},
            {"f32_softmax", src_t::f32, 16, 128, 64, kF32, 4, false, kNone,
                    post_op_type_t::softmax},
            {"f32_alpha_beta", src_t::f32, 16, 96, 100, kF32, 8, false, kNone,
                    none, gated_t::none, 0.5f, 2.0f},
    };
}

/// Calls of the fallback tests: some of all_cases, and M above the plan
/// cache's bound (resolved on every call).
std::vector<case_t> fallback_cases() {
    std::vector<case_t> v;
    for (const case_t &c : all_cases()) {
        const std::string n = c.name;
        if (n == "bf16_decode" || n == "bf16_softmax" || n == "bf16_k_passes"
                || n == "int8_dyn_decode" || n == "int8_dyn_gated"
                || n == "int8_u8_trans")
            v.push_back(c);
    }
    v.push_back({"bf16_large_m", src_t::bf16, 1100, 64, 64, kBf16, 8});
    v.push_back({"int8_large_m", src_t::dyn_s8, 1100, 64, 64, kBf16, 8});
    return v;
}

/// Settings whose keys keep the output bits (schema bits "same"), each
/// statically legal; one key per entry, then combinations.
const char *const kSameBitsSettings[] = {"nr=16", "nr=32", "mr=1", "mr=4",
        "mr=6", "nr=32,mr=12", "mb=1", "mb=7", "mb=24", "mb=8192", "kc=2",
        "kc=64", "kc=1000", "kc=1073741824", "col_groups=0", "col_groups=1",
        "col_groups=2", "schedule=slice", "schedule=row_groups",
        "schedule=row_groups,domain_threads=2", "traversal=n_inner",
        "unit_tiles=2,bf16_blocks_min_m=1,int8_blocks_min_m=1,domain_threads=2",
        "schedule=blocks", "schedule=blocks,unit_tiles=1",
        "schedule=blocks,unit_tiles=3", "schedule=blocks,unit_tiles=4096",
        "schedule=blocks,unit_tiles=2,mb=7,kc=64",
        "schedule=blocks,unit_tiles=3,domain_threads=2,threads_cap=7",
        "schedule=blocks,nb=16", "schedule=blocks,nb=64,mb=7,kc=64",
        "schedule=blocks,unit_tiles=3,nb=128",
        "nb=96,bf16_blocks_min_m=1,int8_blocks_min_m=1,domain_threads=2",
        "threads_cap=1", "threads_cap=3", "threads_cap=7",
        "min_tiles_per_thread=2", "min_tiles_per_thread=64", "row_unit=1",
        "row_unit=5", "row_unit=4096", "col_groups=2,row_unit=1", "tile_rows=1",
        "tile_rows=4", "tall_rows=1", "tall_rows=8", "block_rows=1",
        "block_rows=24", "block_rows=100", "block_rows=8192",
        "min_tile_cols=32", "row_split=0", "min_pass_k=2", "min_pass_k=64",
        "bf16_l2_pct=1", "bf16_l2_pct=10", "bf16_l2_pct=1000", "kc_split=equal",
        "bf16_l2_pct=5,kc_split=equal", "l3_pct=1", "l3_pct=1000",
        "domain_threads=1", "domain_threads=16", "l3_pct=1,domain_threads=4",
        "int8_mr=1", "int8_mr=5", "int8_mr=12", "int8_cw=16", "int8_cw=32/16",
        "int8_l2_pct=1", "int8_l2_pct=1000", "int8_kc=4", "int8_kc=64",
        "int8_kc=1024", "direct_store=0", "copy_block=1", "copy_block=7",
        "copy_block=4096", "bf16_copy_par_min=1",
        "bf16_copy_par_min=1099511627776", "int8_copy_par_min=0",
        "int8_copy_par_min=1099511627776",
        "nr=16,mb=24,kc=64,traversal=n_inner,row_unit=6",
        "int8_mr=5,int8_cw=32/16,int8_kc=64,direct_store=0,copy_block=16",
        "l3_pct=1,domain_threads=32", "schedule=row_groups,domain_threads=64",
        nt::kNativeRulesOff, "makespan=0,l1_pass=0,traffic_groups=0",
        "min_gain_pct=0", "min_gain_pct=100", "idle_pct=1,min_gain_pct=0",
        "decode_m=0", "decode_m=1073741824", "l1_pct=1",
        "l1_pct=1000,l1_pass_max_mb=8192", "l1_pass_max_mb=1",
        "min_pass_k=2,l1_pct=1", "makespan=0", "l1_pass=0",
        "makespan=0,traffic_groups=0,l3_pct=1,domain_threads=4",
        "l3_pct=1,domain_threads=2,threads_cap=7", "bf16_alias=0",
        "makespan=0,l1_pass=0,traffic_groups=0,bf16_narrow_n=0,blocks=0,"
        "bf16_alias=1",
        "bf16_alias_mr=1", "bf16_narrow_n=0",
        "bf16_narrow_n_min_m=1,domain_threads=1",
        "makespan=0,l1_pass=0,traffic_groups=0,bf16_alias=0,blocks=0,"
        "bf16_narrow_n=1,bf16_narrow_n_min_m=1,domain_threads=1",
        "cost_model=0", "cm_short_rows=4", "cm_gain_pct=0",
        "cm_tall_pct=100,cm_call_cycles=0", "cm_fma_pipes=1,cm_fma_lat=64",
        "cm_prefetch_kib=0,cm_reuse_pct=1000,domain_threads=64",
        "cm_mem_cycles=1,cm_l2_bytes=1,cm_l3_cycles=1048576,cm_overlap_pct=0",
        "cm_stream_cycles=0,cm_sync_tiles=0",
        "cm_group_slab_pct=100000,cm_group_l3_pct=100000,cm_block_kib=0,"
        "cm_block_k=0",
        "cm_group_skew_pct=100,cm_group_rows_pct=200",
        "cm_drift_kib=0,cm_pass_kib=1048576,cm_pass_calls=1",
        "cm_int8_l1_pct=1"};

uint32_t seed_of(const std::string &s) {
    uint32_t h = 2166136261u;
    for (char ch : s)
        h = (h ^ static_cast<uint8_t>(ch)) * 16777619u;
    return h;
}

/// FNV-1a of the bytes of b.
uint64_t digest_of(const buffer_t &b) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < b.bytes; ++i)
        h = (h ^ b.as<uint8_t>()[i]) * 1099511628211ull;
    return h;
}

buffer_t copy_of(const buffer_t &b) {
    buffer_t c = nt::make_buffer(b.bytes);
    std::copy(b.as<uint8_t>(), b.as<uint8_t>() + b.bytes, c.as<uint8_t>());
    return c;
}

/// The operands of a case, for calls of K up to k_max: a source of k_max
/// columns (rows when transposed) and a weight of k_max rows, of which a
/// call reads the first K.
struct operands_t {
    buffer_t src, wei, bias, wei_scale, src_scale, src_zp;
    /// The destination before the call (read by beta).
    buffer_t init;
    int k_max = 0;
    mm::matmul_params params;
};

void set_quant(mm::matmul_quantization_params_t::matmul_quant_t &q,
        const buffer_t &b, data_type_t dt, std::vector<int64_t> dims) {
    q.buff = b.get();
    q.dt = dt;
    q.dims = std::move(dims);
}

/// k_max 0: the case's K.  wc is the call's weight cache mode.
operands_t make_operands(const case_t &c, int k_max = 0, int32_t wc = 1) {
    operands_t o;
    o.k_max = k_max > 0 ? k_max : c.K;
    const uint32_t seed = seed_of(c.name);
    const size_t a_n = static_cast<size_t>(c.M) * o.k_max;
    const size_t b_n = static_cast<size_t>(o.k_max) * c.N;
    const data_type_t float_dt = c.src == src_t::f32 ? kF32 : kBf16;
    if (c.src == src_t::st_u8)
        o.src = nt::random_buffer(kU8, a_n, seed, 96, 160);
    else if (c.src == src_t::st_s8)
        o.src = nt::random_buffer(kS8, a_n, seed, -64, 64);
    else
        o.src = nt::random_buffer(float_dt, a_n, seed, -1.0f, 1.0f);
    // An s4 weight holds two elements per byte, every byte drawn.
    if (c.int4())
        o.wei = nt::random_buffer(kU8, (b_n + 1) / 2, seed + 1, 0, 255);
    else if (c.int8())
        o.wei = nt::random_buffer(kS8, b_n, seed + 1, -127, 127);
    else
        o.wei = nt::random_buffer(float_dt, b_n, seed + 1, -1.0f, 1.0f);
    if (c.bias != kNone)
        o.bias = nt::random_buffer(c.bias, c.N, seed + 2, -1.0f, 1.0f);
    o.init = nt::random_buffer(
            c.dst, static_cast<size_t>(c.M) * c.cols(), seed + 6, -1.0f, 1.0f);

    mm::matmul_params &p = o.params;
    p.dtypes.src = c.src == src_t::st_u8
            ? kU8
            : (c.src == src_t::st_s8 ? kS8 : float_dt);
    p.dtypes.wei = c.int4() ? data_type_t::s4 : c.int8() ? kS8 : float_dt;
    p.dtypes.dst = c.dst;
    p.dtypes.bias = c.bias;
    p.num_threads = c.threads;
    p.weight_cache_type = wc;
    auto &qp = p.quant_params;
    const int64_t groups = o.k_max / kInt4Group;
    if (c.int4()) {
        o.wei_scale = nt::random_buffer(kBf16,
                static_cast<size_t>(groups) * c.N, seed + 3, 0.002f, 0.01f);
        set_quant(qp.wei_scale, o.wei_scale, kBf16, {groups, c.N});
    } else if (c.int8()) {
        o.wei_scale = nt::random_buffer(kF32, c.N, seed + 3, 0.002f, 0.01f);
        set_quant(qp.wei_scale, o.wei_scale, kF32, {1, c.N});
    }
    if (c.src == src_t::dyn_s8 || c.int4()) {
        p.dynamic_quant = true;
        p.dtypes.compute = kS8;
        const int64_t cols = c.src == src_t::dyn_s4_groups ? groups : 1;
        o.src_scale = nt::make_buffer(static_cast<size_t>(c.M) * cols * 4);
        set_quant(qp.src_scale, o.src_scale, kF32, {c.M, cols});
    } else if (c.int8()) {
        o.src_scale = nt::random_buffer(kF32, 1, seed + 4, 0.005f, 0.02f);
        set_quant(qp.src_scale, o.src_scale, kF32, {1, 1});
        if (c.src == src_t::st_u8) {
            o.src_zp = nt::random_buffer(kS32, 1, seed + 5, 120, 136);
            set_quant(qp.src_zp, o.src_zp, kS32, {1, 1});
        }
    }
    if (c.post_op != post_op_type_t::none) {
        mm::matmul_post_op po;
        po.po_type = c.post_op;
        p.postop_.push_back(po);
    }
    return o;
}

/// The call at K (0: the case's).
nt::call_t make_call(
        const case_t &c, const operands_t &o, void *dst, int K = 0) {
    nt::call_t call;
    call.M = c.M;
    call.N = c.N;
    call.K = K > 0 ? K : c.K;
    call.transA = c.trans_a;
    call.lda = c.trans_a ? 0 : o.k_max;
    call.alpha = c.alpha;
    call.beta = c.beta;
    call.src = o.src.get();
    call.wei = o.wei.get();
    call.bias = c.bias == kNone ? nullptr : o.bias.get();
    call.dst = dst;
    call.params = o.params;
    call.gated = c.gated;
    return call;
}

/// The problem native_brgemm_execute builds for the call (num_threads set).
nat::native_problem_t problem_of(
        const case_t &c, const operands_t &o, int K = 0) {
    const nt::call_t call = make_call(c, o, nullptr, K);
    nat::GemmDescriptor desc;
    desc.M = call.M;
    desc.N = call.N;
    desc.K = call.K;
    desc.lda = call.lda_or_tight();
    desc.ldb = call.ldb_or_tight();
    desc.ldc = call.ldc_or_tight();
    desc.transA = call.transA;
    desc.transB = call.transB;
    desc.alpha = call.alpha;
    desc.beta = call.beta;
    desc.src_dt = call.params.dtypes.src;
    desc.wei_dt = call.params.dtypes.wei;
    desc.dst_dt = call.params.dtypes.dst;
    desc.bias_dt = call.params.dtypes.bias;
    desc.bias = call.bias;
    desc.is_weights_const = call.is_weights_const;
    desc.num_threads = call.params.num_threads;
    desc.gated_act = call.gated;
    return nat::make_native_problem(desc, call.params, call.bias,
            nat::kNativeLayoutPackNr,
            nat::decide_int8_call(desc, call.params, call.bias));
}

nat::native_problem_t make_problem(
        int M, int N, int K, int threads, bool bf16) {
    nat::native_problem_t p;
    p.M = M;
    p.N = N;
    p.K = K;
    p.pack_nr = nat::kNativeLayoutPackNr;
    p.threads = threads;
    p.src_dt = kBf16;
    p.wei_dt = bf16 ? kBf16 : kS8;
    p.dst_dt = kBf16;
    p.dynamic_quant = !bf16;
    return p;
}

struct run_t {
    buffer_t dst;
    status_t status = status_t::failure;
    /// The native BRGEMM keeps a layout of the weight after it (weight cache
    /// mode 1: it served the call).
    bool native = false;
};

/// The call at K (0: the case's) under the process planner, from its initial
/// destination.
run_t run_call(const case_t &c, const operands_t &o, int K = 0) {
    run_t r;
    r.dst = copy_of(o.init);
    r.status = nt::run_matmul(make_call(c, o, r.dst.get(), K));
    r.native = nt::native_layouts(o.wei.get()) != 0;
    return r;
}

/// Whether the native BRGEMM serves the call of c at K on a weight of k_max
/// rows: run at weight cache mode 1 on a weight of its own, it leaves a
/// layout of that weight.  The native caches are cleared after it.
bool native_serves(const case_t &c, int k_max, int K) {
    const operands_t probe = make_operands(c, k_max, 1);
    const bool served = run_call(c, probe, K).native;
    mm::clear_matmul_native_weight_caches();
    return served;
}

/// Why r is not the native BRGEMM's output of the bits of ref, or "" when
/// it is.
std::string bits_differ(const case_t &c, const run_t &r, const buffer_t &ref) {
    if (r.status != status_t::success)
        return "status " + std::to_string(static_cast<int>(r.status));
    if (!r.native) return "not served by the native BRGEMM";
    const ::testing::AssertionResult same = nt::within_steps(r.dst.get(),
            c.cols(), ref.get(), c.cols(), c.dst, c.M, c.cols(), 0);
    return same ? std::string() : same.message();
}

/// Lists the first failures of a child process on stderr.
class child_report_t {
public:
    void fail(const std::string &what) {
        if (failures_++ < kListed) std::fprintf(stderr, "%s\n", what.c_str());
    }
    int failures() const {
        if (failures_ > kListed)
            std::fprintf(stderr, "%d more failures\n", failures_ - kListed);
        return failures_;
    }

private:
    static constexpr int kListed = 20;
    int failures_ = 0;
};

/// The cases the CPU serves, their operands, and their outputs under the
/// built-in plan: computed in the parent, which publishes them in shared,
/// and read from shared in a child.
struct fixture_t {
    std::vector<case_t> cases;
    std::vector<operands_t> operands;
    std::vector<buffer_t> refs;
};

/// The cases the CPU serves.
std::vector<case_t> served(const std::vector<case_t> &cases) {
    const nat::UarchParams &uarch = nat::detect_uarch();
    std::vector<case_t> v;
    for (const case_t &c : cases) {
        const bool isa = c.src == src_t::f32
                ? nat::native_base_isa()
                : (c.int8() ? uarch.avx512vnni : uarch.avx512bf16);
        if (isa) v.push_back(c);
    }
    return v;
}

fixture_t make_fixture(
        const std::vector<case_t> &cases, nt::child_buffers_t &shared) {
    fixture_t f;
    f.cases = served(cases);
    for (const case_t &c : f.cases)
        f.operands.push_back(make_operands(c));
    if (nt::in_child()) {
        f.refs = shared.buffers();
        return f;
    }
    for (size_t i = 0; i < f.cases.size(); ++i) {
        const run_t r = run_call(f.cases[i], f.operands[i]);
        EXPECT_EQ(r.status, status_t::success) << f.cases[i].name;
        EXPECT_TRUE(r.native)
                << f.cases[i].name << " not served by the native BRGEMM";
        f.refs.push_back(r.dst);
    }
    shared.publish(f.refs);
    return f;
}

/// In a child: every case once under the process planner against the
/// parent's output; returns the failures, listed on stderr.
int child_same_bits(const fixture_t &f) {
    child_report_t report;
    if (f.refs.size() != f.cases.size()) {
        report.fail("the parent's outputs are missing");
        return report.failures();
    }
    for (size_t i = 0; i < f.cases.size(); ++i) {
        const std::string why = bits_differ(
                f.cases[i], run_call(f.cases[i], f.operands[i]), f.refs[i]);
        if (!why.empty())
            report.fail(std::string(f.cases[i].name) + ": " + why);
    }
    return report.failures();
}

/// Counts checks and lists the first failures in full.
class report_t {
public:
    template <typename Describe>
    void check(bool ok, Describe &&describe) {
        ++checked_;
        if (ok) return;
        if (failed_++ < kListed) ADD_FAILURE() << describe();
    }
    void finish() const {
        EXPECT_EQ(failed_, 0u) << "of " << checked_ << " checks (at most "
                               << kListed << " listed above)";
    }

private:
    static constexpr size_t kListed = 20;
    size_t checked_ = 0;
    size_t failed_ = 0;
};

/// What the library writes to stdout (its log) while f runs.
template <typename F>
std::string stdout_of(F &&f) {
    ::testing::internal::CaptureStdout();
    f();
    return ::testing::internal::GetCapturedStdout();
}

size_t count_of(const std::string &text, const std::string &what) {
    size_t n = 0;
    for (size_t at = text.find(what); at != std::string::npos;
            at = text.find(what, at + what.size()))
        ++n;
    return n;
}

bool contains(const std::string &text, const std::string &what) {
    return text.find(what) != std::string::npos;
}

/// Tuning warnings follow the API log's warning level.
bool warnings_on() {
    return zendnnl::error_handling::apilog_warning_enabled();
}

std::string label_of(const nat::native_config_entry_t *e) {
    return e == nullptr ? "-" : e->label;
}

std::string tuning_json(const std::string &hardware, const std::string &entries,
        const std::string &rules) {
    std::string s = R"({"format": "zendnnl-native-brgemm-tuning", "version": 2)"
                    R"(, "meta": {"tool": "gtest", "note": "ignored"})";
    if (!hardware.empty()) s += ",\n \"hardware\": " + hardware;
    if (!entries.empty()) s += ",\n \"entries\": [" + entries + "]";
    if (!rules.empty()) s += ",\n \"rules\": " + rules;
    return s + "}\n";
}

/// The hardware block of this CPU (every key the schema gives), or with key
/// set to value.
std::string this_hardware(
        const std::string &key = "", const nlohmann::json &value = {}) {
    nlohmann::json block
            = nlohmann::json::parse(nt::hw_match_text(nat::native_hw()));
    if (!key.empty()) block[key] = value;
    return block.dump();
}

/// A match of exactly the problem p.
nlohmann::json match_of(const nat::native_problem_t &p) {
    const auto dt = [](data_type_t t) -> const char * {
        switch (t) {
            case kF32: return "f32";
            case kBf16: return "bf16";
            case data_type_t::f16: return "f16";
            case kU8: return "u8";
            case kS8: return "s8";
            default: return "s32";
        }
    };
    const char *const epilogues[] = {"store", "bias", "postops", "gated"};
    const char *dtype = p.f32() ? "f32" : (p.bf16() ? "bf16" : "s8");
    return {{"m", p.M}, {"n", p.N}, {"k", p.K}, {"threads", p.threads},
            {"pack_nr", p.pack_nr}, {"dtype", dtype}, {"src", dt(p.src_dt)},
            {"dst", dt(p.dst_dt)},
            {"epilogue", epilogues[static_cast<int>(p.epilogue)]},
            {"trans_a", p.trans_a}, {"dynamic", p.dynamic_quant},
            {"has_bias", p.has_bias}};
}

/// The tuning-file config of a setting "key=value,...".
nlohmann::json config_json(const std::string &setting) {
    size_t count = 0;
    const nat::native_config_key_t *keys = nat::native_config_keys(count);
    nlohmann::json cfg = nlohmann::json::object();
    std::stringstream items(setting);
    std::string item;
    while (std::getline(items, item, ',')) {
        const size_t eq = item.find('=');
        const std::string name = item.substr(0, eq);
        const std::string value = item.substr(eq + 1);
        const nat::native_config_key_t *key = nullptr;
        for (size_t i = 0; i < count && key == nullptr; ++i)
            if (name == keys[i].name) key = &keys[i];
        if (key == nullptr) {
            ADD_FAILURE() << "unknown key in " << setting;
            continue;
        }
        switch (key->type) {
            case nat::native_key_type_t::int32:
            case nat::native_key_type_t::int64:
                cfg[name] = std::stoll(value);
                break;
            case nat::native_key_type_t::boolean:
                cfg[name] = value == "1";
                break;
            case nat::native_key_type_t::cw_list: {
                nlohmann::json widths = nlohmann::json::array();
                std::stringstream ws(value);
                std::string w;
                while (std::getline(ws, w, '/'))
                    widths.push_back(std::stoi(w));
                cfg[name] = widths;
                break;
            }
            default: cfg[name] = value;
        }
    }
    return cfg;
}

class NativeTuningExec : public ::testing::Test {
protected:
    void SetUp() override {
        // The outputs under the built-in plan come from the process planner
        // of the parent; a child has the settings it is given.
        if (!nt::in_child()) {
            for (const char *name : {kConfig, kTuningFile}) {
                const char *value = std::getenv(name);
                if (value != nullptr && *value != '\0')
                    GTEST_SKIP() << name << " is set in this process";
            }
        }
        ::reset_grp_matmul_caches();
        cache_ = std::make_unique<WeightCacheGuard>(2);
        algo_ = std::make_unique<nt::algo_pin_t>(matmul_algo_t::native_brgemm);
    }
    void TearDown() override {
        algo_.reset();
        cache_.reset();
        ::reset_grp_matmul_caches();
    }

private:
    std::unique_ptr<WeightCacheGuard> cache_;
    std::unique_ptr<nt::algo_pin_t> algo_;
};

// ── Settings and tuning files that keep the bits ─────────────────────────

// One child process runs every setting on every case, through a table that
// gives each pair a problem of its own: setting j runs at K + 4 j, which
// keeps the parity of K and K mod 4.  The weight is packed for each call, so
// no layout of it shows that the native BRGEMM served the call: the parent
// checks that on a weight of the same shape at weight cache mode 1.
TEST_F(NativeTuningExec, SameBitsOverrideKeysKeepTheBits) {
    const std::vector<case_t> cases = served(all_cases());
    ASSERT_FALSE(cases.empty());
    constexpr int kSettings = static_cast<int>(std::size(kSameBitsSettings));
    const auto k_of = [](const case_t &c, int j) { return c.K + 4 * j; };
    std::vector<operands_t> ops;
    for (const case_t &c : cases)
        ops.push_back(make_operands(c, k_of(c, kSettings - 1), 0));
    std::string entries;
    for (size_t i = 0; i < cases.size(); ++i)
        for (int j = 0; j < kSettings; ++j) {
            nlohmann::json e;
            e["id"] = std::string(cases[i].name) + "/" + kSameBitsSettings[j];
            e["match"]
                    = match_of(problem_of(cases[i], ops[i], k_of(cases[i], j)));
            e["config"] = config_json(kSameBitsSettings[j]);
            entries += (entries.empty() ? "" : ",\n") + e.dump();
        }
    const nt::temp_file_t file("same_bits.json", tuning_json("", entries, ""));
    nt::child_buffers_t shared;
    if (!nt::in_child()) {
        std::unique_ptr<nat::native_planner_t> planner;
        const std::string out = stdout_of([&] {
            planner = nat::make_native_planner(nullptr, file.path().c_str());
        });
        ASSERT_TRUE(planner->has_sources());
        if (warnings_on()) { EXPECT_FALSE(contains(out, " dropped: ")) << out; }
        std::vector<int> applied(kSettings, 0);
        buffer_t digests
                = nt::make_buffer(cases.size() * kSettings * sizeof(uint64_t));
        for (size_t i = 0; i < cases.size(); ++i)
            for (int j = 0; j < kSettings; ++j) {
                const int K = k_of(cases[i], j);
                applied[j] += nt::top_source(planner->plan(
                                      problem_of(cases[i], ops[i], K)))
                        == source_t::table;
                ASSERT_TRUE(native_serves(
                        cases[i], k_of(cases[i], kSettings - 1), K))
                        << cases[i].name << " at K " << K;
                const run_t r = run_call(cases[i], ops[i], K);
                ASSERT_EQ(r.status, status_t::success) << cases[i].name;
                digests.as<uint64_t>()[i * kSettings + j] = digest_of(r.dst);
            }
        for (int j = 0; j < kSettings; ++j)
            EXPECT_GT(applied[j], 0)
                    << "'" << kSameBitsSettings[j] << "' applied to no call";
        shared.publish({digests});
    }
    nt::expect_in_child(
            {{kConfig, nullptr}, {kTuningFile, file.path().c_str()}}, [&] {
        child_report_t report;
        const std::vector<buffer_t> &got = shared.buffers();
        if (got.size() != 1
                || got[0].bytes
                        != cases.size() * kSettings * sizeof(uint64_t)) {
            report.fail("the parent's outputs are missing");
            return report.failures();
        }
        if (!nat::native_planner().has_sources())
            report.fail("the process planner has no tuning file");
        for (size_t i = 0; i < cases.size(); ++i)
            for (int j = 0; j < kSettings; ++j) {
                const run_t r = run_call(cases[i], ops[i], k_of(cases[i], j));
                const bool same = r.status == status_t::success
                        && digest_of(r.dst)
                                == got[0].as<uint64_t>()[i * kSettings + j];
                if (!same)
                    report.fail(std::string(cases[i].name) + " at K "
                            + std::to_string(k_of(cases[i], j)) + " with '"
                            + kSameBitsSettings[j]
                            + "' differs from the built-in plan");
            }
        return report.failures();
    }, "the same-bits settings");
}

// BF16 tiles of several 32-column slabs (with odd K, K passes, row tails, a
// transposed source and a gated epilogue among the cases) keep the bits of
// the built-in plan.  A setting runs on the BF16 cases it gives a legal plan,
// as in SameBitsOverrideKeysKeepTheBits.
TEST_F(NativeTuningExec, SameBitsBf16TileWidthsKeepTheBits) {
    const char *const settings[] = {"nr=64", "nr=128", "nr=64,mr=4",
            "nr=128,mr=2", "nr=64,mr=1", "nr=64,kc=64", "nr=128,kc=1000",
            "nr=128,mr=3,kc=2", "nr=64,mb=7", "nr=128,schedule=row_groups",
            "nr=64,traversal=n_inner", "nr=64,schedule=blocks,unit_tiles=2",
            "nr=128,schedule=blocks,unit_tiles=3,kc=1000"};
    constexpr int kSettings = static_cast<int>(std::size(settings));
    std::vector<case_t> cases;
    for (const case_t &c : served(all_cases()))
        if (c.src == src_t::bf16) cases.push_back(c);
    if (cases.empty()) GTEST_SKIP() << "needs AVX512-BF16";
    const auto k_of = [](const case_t &c, int j) { return c.K + 4 * j; };
    std::vector<operands_t> ops;
    for (const case_t &c : cases)
        ops.push_back(make_operands(c, k_of(c, kSettings - 1), 0));
    const nat::UarchParams &uarch = nat::native_plan_uarch();
    const auto legal = [&](size_t i, int j) {
        nat::native_config_t cfg;
        std::string why;
        if (!nat::parse_native_config(settings[j], cfg, why)) return false;
        const nat::native_problem_t p
                = problem_of(cases[i], ops[i], k_of(cases[i], j));
        return nat::validate_native_plan(
                       nat::heuristic_plan(
                               p, uarch, nat::apply_native_config({}, cfg)),
                       p)
                == nullptr;
    };
    std::string entries;
    std::vector<int> listed(kSettings, 0);
    for (size_t i = 0; i < cases.size(); ++i)
        for (int j = 0; j < kSettings; ++j) {
            if (!legal(i, j)) continue;
            ++listed[j];
            nlohmann::json e;
            e["id"] = std::string(cases[i].name) + "/" + settings[j];
            e["match"]
                    = match_of(problem_of(cases[i], ops[i], k_of(cases[i], j)));
            e["config"] = config_json(settings[j]);
            entries += (entries.empty() ? "" : ",\n") + e.dump();
        }
    for (int j = 0; j < kSettings; ++j)
        EXPECT_GT(listed[j], 0) << "'" << settings[j] << "' serves no case";
    const nt::temp_file_t file(
            "same_bits_bf16.json", tuning_json("", entries, ""));
    nt::child_buffers_t shared;
    if (!nt::in_child()) {
        std::unique_ptr<nat::native_planner_t> planner;
        const std::string out = stdout_of([&] {
            planner = nat::make_native_planner(nullptr, file.path().c_str());
        });
        ASSERT_TRUE(planner->has_sources());
        if (warnings_on()) { EXPECT_FALSE(contains(out, " dropped: ")) << out; }
        buffer_t digests
                = nt::make_buffer(cases.size() * kSettings * sizeof(uint64_t));
        for (size_t i = 0; i < cases.size(); ++i)
            for (int j = 0; j < kSettings; ++j) {
                const int K = k_of(cases[i], j);
                ASSERT_TRUE(native_serves(
                        cases[i], k_of(cases[i], kSettings - 1), K))
                        << cases[i].name << " at K " << K;
                if (legal(i, j)) {
                    EXPECT_EQ(nt::top_source(planner->plan(
                                      problem_of(cases[i], ops[i], K))),
                            source_t::table)
                            << cases[i].name << " " << settings[j];
                }
                const run_t r = run_call(cases[i], ops[i], K);
                ASSERT_EQ(r.status, status_t::success) << cases[i].name;
                digests.as<uint64_t>()[i * kSettings + j] = digest_of(r.dst);
            }
        shared.publish({digests});
    }
    nt::expect_in_child(
            {{kConfig, nullptr}, {kTuningFile, file.path().c_str()}}, [&] {
        child_report_t report;
        const std::vector<buffer_t> &got = shared.buffers();
        if (got.size() != 1
                || got[0].bytes
                        != cases.size() * kSettings * sizeof(uint64_t)) {
            report.fail("the parent's outputs are missing");
            return report.failures();
        }
        for (size_t i = 0; i < cases.size(); ++i)
            for (int j = 0; j < kSettings; ++j) {
                const run_t r = run_call(cases[i], ops[i], k_of(cases[i], j));
                const bool same = r.status == status_t::success
                        && digest_of(r.dst)
                                == got[0].as<uint64_t>()[i * kSettings + j];
                if (!same)
                    report.fail(std::string(cases[i].name) + " at K "
                            + std::to_string(k_of(cases[i], j)) + " with '"
                            + settings[j] + "' differs from the built-in plan");
            }
        return report.failures();
    }, "the BF16 tile widths");
}

/// Whether a plan with the BF16 aliasing rule (on) takes its rows per call,
/// against the plan without it (off).
bool takes_alias_rows(
        const nat::native_plan_t &on, const nat::native_plan_t &off) {
    return on.mr == nat::heuristic_params_t {}.bf16_alias_mr && off.mr > on.mr;
}

/// Whether a plan with the BF16 narrow-N rule (on) takes tiles of several
/// slabs, against the plan without it (off).
bool takes_wide_tiles(
        const nat::native_plan_t &on, const nat::native_plan_t &off) {
    return on.nr > off.nr;
}

/// Whether the cost model's plan (on) differs from the rules' (off) in the
/// rows per call, the K pass or the row groups.
bool takes_model_plan(
        const nat::native_plan_t &on, const nat::native_plan_t &off) {
    return on.mr != off.mr || on.kc != off.kc
            || on.col_groups != off.col_groups;
}

/// A planner of setting over one cache hierarchy (48 KiB L1, 1 MiB L2,
/// 32 MiB of L3 per 8 cores), so that the rules it runs take the same calls
/// on any host.
std::unique_ptr<nat::native_planner_t> fixed_planner(const char *setting) {
    nat::UarchParams u = nat::detect_uarch();
    u.l1d_bytes = 48 << 10;
    u.l2_bytes = 1 << 20;
    u.l3_domain_bytes = 32 << 20;
    u.family_l3_cores = 8;
    u.l3_domain_cores = 8;
    return std::make_unique<nat::native_planner_t>(
            u, nat::make_native_override_source(setting), nullptr, nullptr);
}

/// How many of cases the planners of on and off over the detected caches,
/// which a child of expect_in_child plans with, find taken: where none is,
/// both children would run the built-in plan.
size_t taken_on_this_host(const fixture_t &f, const char *on, const char *off,
        bool (*taken)(const nat::native_plan_t &, const nat::native_plan_t &)) {
    const std::unique_ptr<nat::native_planner_t> with
            = nat::make_native_planner(on, nullptr);
    const std::unique_ptr<nat::native_planner_t> without
            = nat::make_native_planner(off, nullptr);
    size_t n = 0;
    for (size_t i = 0; i < f.cases.size(); ++i) {
        const nat::native_problem_t p = problem_of(f.cases[i], f.operands[i]);
        n += taken(with->plan(p), without->plan(p)) ? 1 : 0;
    }
    return n;
}

// The BF16 rules keep the bits on the calls they take (one with a transposed
// source, an epilogue and K passes): a planner of each rule's setting takes
// every case, and a child with the rule and one without it both give the
// outputs of the built-in plan.  The narrow-N rule takes them with L3 domains
// of two threads.  A rule's children run only where the detected caches let
// it take a case; the alias rule is on in the built-in plan, so only its
// child without it runs.
TEST_F(NativeTuningExec, Bf16RulesKeepTheBits) {
    const post_op_type_t gelu = post_op_type_t::gelu_erf;
    nt::child_buffers_t shared;
    const fixture_t f = make_fixture(
            {{"bf16_rows_4k_apart", src_t::bf16, 1100, 64, 2048, kBf16, 8},
                    {"bf16_rows_4k_apart_trans", src_t::bf16, 1030, 128, 4096,
                            kF32, 8, true, kF32, gelu}},
            shared);
    if (f.cases.empty()) GTEST_SKIP() << "needs AVX512-BF16";
    const struct {
        const char *on, *off;
        bool (*taken)(const nat::native_plan_t &, const nat::native_plan_t &);
        bool on_is_built_in;
    } rules[] = {{"bf16_alias=1", "bf16_alias=0", takes_alias_rows, true},
            {"domain_threads=2", "domain_threads=2,bf16_narrow_n=0",
                    takes_wide_tiles, false}};
    int compared = 0;
    for (const auto &rule : rules) {
        if (!nt::in_child()) {
            const std::unique_ptr<nat::native_planner_t> on
                    = fixed_planner(rule.on);
            const std::unique_ptr<nat::native_planner_t> off
                    = fixed_planner(rule.off);
            for (size_t i = 0; i < f.cases.size(); ++i) {
                const nat::native_problem_t p
                        = problem_of(f.cases[i], f.operands[i]);
                EXPECT_TRUE(rule.taken(on->plan(p), off->plan(p)))
                        << f.cases[i].name << ": '" << rule.on
                        << "' does not take the call";
            }
        }
        if (taken_on_this_host(f, rule.on, rule.off, rule.taken) == 0) {
            if (!nt::in_child())
                std::printf(
                        "'%s' takes no case over the detected caches: its "
                        "children are skipped\n",
                        rule.on);
            continue;
        }
        ++compared;
        for (const char *setting : {rule.on, rule.off})
            if (setting != rule.on || !rule.on_is_built_in)
                nt::expect_in_child(
                        {{kConfig, setting}, {kTuningFile, nullptr}},
                        [&] { return child_same_bits(f); }, setting);
    }
    if (compared == 0) GTEST_SKIP() << "the detected caches take no case";
}

// The cost model keeps the bits on the calls it takes: 8 rows per call with
// the L1 pass at M = 16 (12 + 4 under the rules), 8-row calls above
// decode_m at the price of taller calls of a 32-thread domain, and two row
// groups of 4 threads at M = 55 (an 8-thread team in one 8-thread domain,
// 4.5 column tiles a thread).
// A planner over 8-core domains takes every case, and a child with the model
// and one without it both give the outputs of the built-in plan, where the
// detected caches let the model take a case.
TEST_F(NativeTuningExec, CostModelKeepsTheBits) {
    nt::child_buffers_t shared;
    const fixture_t f = make_fixture(
            {{"bf16_model_rows", src_t::bf16, 16, 1024, 4096, kBf16, 8},
                    {"bf16_model_prefill", src_t::bf16, 300, 512, 2560, kBf16,
                            8},
                    {"bf16_model_groups", src_t::bf16, 55, 1152, 4096, kBf16, 8,
                            false, kF32, post_op_type_t::gelu_erf}},
            shared);
    if (f.cases.empty()) GTEST_SKIP() << "needs AVX512-BF16";
    const char *const on = "cost_model=1,cm_tall_pct=25";
    const char *const off = "cost_model=0";
    if (!nt::in_child()) {
        const std::unique_ptr<nat::native_planner_t> with = fixed_planner(on);
        const std::unique_ptr<nat::native_planner_t> without
                = fixed_planner(off);
        for (size_t i = 0; i < f.cases.size(); ++i) {
            const nat::native_problem_t p
                    = problem_of(f.cases[i], f.operands[i]);
            EXPECT_TRUE(takes_model_plan(with->plan(p), without->plan(p)))
                    << f.cases[i].name << ": the model does not take the call";
        }
    }
    if (taken_on_this_host(f, on, off, takes_model_plan) == 0)
        GTEST_SKIP() << "the detected caches take no case";
    for (const char *setting : {on, off})
        nt::expect_in_child({{kConfig, setting}, {kTuningFile, nullptr}},
                [&] { return child_same_bits(f); }, setting);
}

/// Whether the cost model's plan (on) differs from the rules' (off) in the
/// M block or the INT8 rows per call or K pass, or as takes_model_plan.
bool takes_model_tile(
        const nat::native_plan_t &on, const nat::native_plan_t &off) {
    return takes_model_plan(on, off) || on.mb != off.mb
            || on.int8_rows_nv2 != off.int8_rows_nv2
            || on.int8_pass_quads != off.int8_pass_quads;
}

// The cost model's rules for domains where one core streams slower than its
// share of the link keep the bits on the calls they take: with that
// limit lifted (cm_stream_cycles above the domain's rate), a BF16 block of
// five calls that computes longer than it streams takes the long K pass, one
// of three calls that streams longer takes M blocks of 6 rows over all of K
// (its K of 2048 with cm_block_k 0), and an INT8 block takes 6 rows per
// call with the shorter L1 pass (cm_int8_l1_pct) or two row groups.  The
// built-in plan of an s4 call of
// one row block whose column tiles do not divide over its threads takes two
// to four M blocks (cm_int4_min_m), with a per-token and with a per-group
// source.  A planner over 8-core domains takes each setting's cases, and a
// child with each setting and one with cost_model=0 give the outputs of the
// built-in plan, where the detected caches let a setting take a case.
TEST_F(NativeTuningExec, CostModelChoicesKeepTheBits) {
    nt::child_buffers_t shared;
    const fixture_t f = make_fixture(
            {{"bf16_long_pass", src_t::bf16, 55, 2048, 4096, kBf16, 8},
                    {"bf16_short_blocks", src_t::bf16, 32, 2048, 2048, kBf16,
                            8},
                    {"int8_three_calls", src_t::dyn_s8, 32, 1024, 4096, kBf16,
                            8},
                    {"int8_groups", src_t::dyn_s8, 55, 512, 2048, kBf16, 8},
                    {"int4_row_blocks", src_t::dyn_s4, 128, 1664, 2048, kBf16,
                            8},
                    {"int4_row_blocks_groups", src_t::dyn_s4_groups, 96, 960,
                            2048, kBf16, 8}},
            shared);
    if (f.cases.size() != 6)
        GTEST_SKIP() << "needs AVX512-BF16 and AVX512-VNNI";
    const char *const off = "cost_model=0";
    const struct {
        const char *on;
        std::vector<std::string> cases;
    } settings[] = {{"cm_stream_cycles=1048576,cm_mem_cycles=50,cm_drift_kib=0,"
                     "cm_int8_l1_pct=40",
                            {"bf16_long_pass", "int8_three_calls"}},
            {"cm_stream_cycles=1048576,cm_block_kib=0,cm_block_k=0",
                    {"bf16_short_blocks"}},
            {"cm_stream_cycles=1048576,cm_mem_cycles=424,cm_group_skew_pct=100",
                    {"int8_groups"}},
            {"cost_model=1", {"int4_row_blocks", "int4_row_blocks_groups"}}};
    int compared = 0;
    for (const auto &s : settings) {
        if (!nt::in_child()) {
            const std::unique_ptr<nat::native_planner_t> on
                    = fixed_planner(s.on);
            const std::unique_ptr<nat::native_planner_t> without
                    = fixed_planner(off);
            for (size_t i = 0; i < f.cases.size(); ++i) {
                if (std::find(s.cases.begin(), s.cases.end(), f.cases[i].name)
                        == s.cases.end())
                    continue;
                const nat::native_problem_t p
                        = problem_of(f.cases[i], f.operands[i]);
                EXPECT_TRUE(takes_model_tile(on->plan(p), without->plan(p)))
                        << f.cases[i].name << ": '" << s.on
                        << "' does not take the call";
            }
        }
        if (taken_on_this_host(f, s.on, off, takes_model_tile) == 0) {
            if (!nt::in_child())
                std::printf(
                        "'%s' takes no case over the detected caches: its "
                        "child is skipped\n",
                        s.on);
            continue;
        }
        ++compared;
        nt::expect_in_child({{kConfig, s.on}, {kTuningFile, nullptr}},
                [&] { return child_same_bits(f); }, s.on);
    }
    if (compared == 0) GTEST_SKIP() << "the detected caches take no case";
    nt::expect_in_child({{kConfig, off}, {kTuningFile, nullptr}},
            [&] { return child_same_bits(f); }, off);
}

bool takes_blocks(const nat::native_plan_t &on, const nat::native_plan_t &off) {
    return on.unit_tiles > 0 && off.unit_tiles == 0;
}

// The blocks rule takes a BF16 call with K passes and an INT8 call the slice
// serves, with domains of two threads and the rule's row bounds lowered;
// with the rule and without it both give the outputs of the built-in plan,
// where the detected caches let the rule take a case.
TEST_F(NativeTuningExec, BlocksRuleKeepsTheBits) {
    nt::child_buffers_t shared;
    const fixture_t f = make_fixture(
            {{"bf16_blocks_k_passes", src_t::bf16, 1100, 256, 3000, kBf16, 8},
                    {"int8_blocks_slice", src_t::dyn_s8, 1100, 256, 512, kBf16,
                            8}},
            shared);
    if (f.cases.empty()) GTEST_SKIP() << "needs AVX512-BF16 or AVX512-VNNI";
    const char *const on
            = "bf16_blocks_min_m=1,int8_blocks_min_m=1,"
              "domain_threads=2,bf16_l2_pct=5";
    const char *const off = "blocks=0,domain_threads=2,bf16_l2_pct=5";
    if (!nt::in_child()) {
        const std::unique_ptr<nat::native_planner_t> with = fixed_planner(on);
        const std::unique_ptr<nat::native_planner_t> without
                = fixed_planner(off);
        for (size_t i = 0; i < f.cases.size(); ++i) {
            const nat::native_problem_t p
                    = problem_of(f.cases[i], f.operands[i]);
            EXPECT_TRUE(takes_blocks(with->plan(p), without->plan(p)))
                    << f.cases[i].name << ": the rule does not take the call";
        }
    }
    // The children plan over the detected caches.
    if (taken_on_this_host(f, on, off, takes_blocks) == 0)
        GTEST_SKIP() << "the detected caches take no case";
    for (const char *setting : {on, off})
        nt::expect_in_child({{kConfig, setting}, {kTuningFile, nullptr}},
                [&] { return child_same_bits(f); }, setting);
}

TEST_F(NativeTuningExec, BiasInDequantOffStaysInTheBand) {
    nt::child_buffers_t shared;
    const fixture_t f = make_fixture(all_cases(), shared);
    nt::expect_in_child(
            {{kConfig, "bias_in_dequant=0"}, {kTuningFile, nullptr}}, [&] {
        child_report_t report;
        if (f.refs.size() != f.cases.size()) {
            report.fail("the parent's outputs are missing");
            return report.failures();
        }
        size_t moved = 0;
        for (size_t i = 0; i < f.cases.size(); ++i) {
            const case_t &c = f.cases[i];
            const run_t r = run_call(c, f.operands[i]);
            if (!c.int8() || c.bias == kNone) {
                // Nothing for the key to move: the bits stay.
                const std::string why = bits_differ(c, r, f.refs[i]);
                if (!why.empty()) report.fail(std::string(c.name) + ": " + why);
                continue;
            }
            ++moved;
            if (r.status != status_t::success || !r.native) {
                report.fail(std::string(c.name) + ": not served");
                continue;
            }
            const float abs_bound = nt::comparison_abs_bound(
                    c.K, epsilon_bf16, c.alpha, c.beta, true, true, false);
            const ::testing::AssertionResult band = nt::within_band(r.dst.get(),
                    c.cols(), f.refs[i].get(), c.cols(), c.dst, c.M, c.cols(),
                    rtol_bf16, abs_bound);
            if (!band) report.fail(std::string(c.name) + ": " + band.message());
        }
        if (moved == 0) report.fail("no INT8 call with a bias");
        return report.failures();
    }, "bias_in_dequant=0");
}

/// Table entries for some of all_cases (file order decides).
const char *const kTableEntries = R"(
  {"id": "bf16-decode",
   "match": {"dtype": "bf16", "n": 1536, "k": 1000, "m": {"max": 16}},
   "config": {"nr": 16, "mb": 1, "threads_cap": 4}},
  {"id": "bf16-k-passes", "match": {"dtype": "bf16", "n": 64, "k": 9000},
   "config": {"kc": 1000, "kc_split": "equal", "traversal": "n_inner"}},
  {"id": "int8-bias",
   "match": {"dtype": "s8", "n": 256, "k": 7200, "has_bias": true},
   "config": {"int8_kc": 256, "int8_mr": 5, "int8_cw": [32, 16]}},
  {"id": "int8-static-u8", "match": {"dtype": "s8", "src": "u8", "n": [256, 96]},
   "config": {"direct_store": false, "copy_block": 7,
              "schedule": "row_groups", "domain_threads": 2}},
  {"id": "few-threads", "match": {"threads": {"min": 2, "max": 4}},
   "config": {"row_unit": 5, "col_groups": 0}})";

/// The table entry each case matches first.
const std::map<std::string, std::string> &table_entry_ids() {
    static const std::map<std::string, std::string> ids = {
            {"bf16_decode", "bf16-decode"}, {"bf16_k_passes", "bf16-k-passes"},
            {"int8_dyn_bias_k_passes", "int8-bias"},
            {"int8_u8_relu_padded_k", "int8-static-u8"},
            {"int8_u8_trans", "int8-static-u8"},
            {"int8_u8_alpha_beta", "int8-static-u8"},
            {"bf16_softmax", "few-threads"}, {"int8_s8", "few-threads"},
            {"int8_dyn_gated", "few-threads"},
            {"int8_dyn_bf16_bias_softmax", "few-threads"},
            {"f32_softmax", "few-threads"}};
    return ids;
}

const char *const kTreeRules = R"({"kind": "tree",
  "root": {"feature": "m", "le": 16,
           "then": {"id": "small-m", "config": {"min_tile_cols": 32}},
           "else": {"feature": "bf16", "le": 0,
                    "then": {"id": "int8-large-m",
                             "config": {"int8_mr": 3, "int8_l2_pct": 10}},
                    "else": {"id": "bf16-large-m",
                             "config": {"block_rows": 24, "bf16_l2_pct": 10}}}}})";

/// An FP32 problem's bf16 feature is 0.
const char *tree_leaf_id(const case_t &c) {
    if (c.M <= 16) return "small-m";
    return c.src == src_t::bf16 ? "bf16-large-m" : "int8-large-m";
}

constexpr unsigned bit(source_t s) {
    return 1u << static_cast<unsigned>(s);
}

TEST_F(NativeTuningExec, TuningFileTableAndTreeRulesKeepTheBits) {
    nt::child_buffers_t shared;
    const fixture_t f = make_fixture(all_cases(), shared);
    const nt::temp_file_t file("table_tree.json",
            tuning_json(this_hardware(), kTableEntries, kTreeRules));
    // The file alone, then with an override on top of it.
    for (const char *config :
            {static_cast<const char *>(nullptr), "traversal=n_inner"}) {
        SCOPED_TRACE(config == nullptr ? "file" : "file and override");
        if (!nt::in_child()) {
            const std::unique_ptr<nat::native_planner_t> planner
                    = nat::make_native_planner(config, file.path().c_str());
            ASSERT_TRUE(planner->has_sources());
            for (size_t i = 0; i < f.cases.size(); ++i) {
                const case_t &c = f.cases[i];
                SCOPED_TRACE(c.name);
                const nat::native_plan_t plan
                        = planner->plan(problem_of(c, f.operands[i]));
                const auto entry = table_entry_ids().find(c.name);
                const bool table = entry != table_entry_ids().end();
                const source_t want = config != nullptr
                        ? source_t::override
                        : (table ? source_t::table : source_t::rules);
                EXPECT_EQ(nt::top_source(plan), want);
                EXPECT_NE(plan.sources & bit(source_t::rules), 0u);
                EXPECT_TRUE(
                        contains(label_of(plan.rules_entry), tree_leaf_id(c)))
                        << label_of(plan.rules_entry);
                if (table) {
                    EXPECT_TRUE(
                            contains(label_of(plan.table_entry), entry->second))
                            << label_of(plan.table_entry);
                } else {
                    EXPECT_EQ(plan.table_entry, nullptr);
                }
            }
        }
        nt::expect_in_child(
                {{kConfig, config}, {kTuningFile, file.path().c_str()}}, [&] {
            return child_same_bits(f);
        }, config == nullptr ? "the file" : "the file and an override");
    }
}

TEST_F(NativeTuningExec, TuningFileOrderedRulesKeepTheBits) {
    nt::child_buffers_t shared;
    const fixture_t f = make_fixture(all_cases(), shared);
    const char *const rules = R"({"kind": "ordered", "list": [
  {"id": "decode", "when": [["m", "<=", 16], ["n_per_thread", "<=", 256]],
   "config": {"nr": 16}},
  {"id": "int8", "when": [["bf16", "==", 0]],
   "config": {"int8_cw": [16], "int8_kc": 128}},
  {"id": "rest", "when": [], "config": {"kc_split": "equal", "row_split": false}}]})";
    const nt::temp_file_t file("ordered.json", tuning_json("", "", rules));
    if (!nt::in_child()) {
        const std::unique_ptr<nat::native_planner_t> planner
                = nat::make_native_planner(nullptr, file.path().c_str());
        ASSERT_TRUE(planner->has_sources());
        for (size_t i = 0; i < f.cases.size(); ++i) {
            const case_t &c = f.cases[i];
            SCOPED_TRACE(c.name);
            const nat::native_plan_t plan
                    = planner->plan(problem_of(c, f.operands[i]));
            // An FP32 problem's bf16 feature is 0.
            const char *want = c.M <= 16 && c.N / c.threads <= 256
                    ? "decode"
                    : (c.src != src_t::bf16 ? "int8" : "rest");
            EXPECT_EQ(nt::top_source(plan), source_t::rules);
            EXPECT_TRUE(contains(label_of(plan.rules_entry), want))
                    << label_of(plan.rules_entry) << ", expected rule " << want;
        }
    }
    nt::expect_in_child(
            {{kConfig, nullptr}, {kTuningFile, file.path().c_str()}},
            [&] { return child_same_bits(f); }, "ordered rules");
}

// ── Bad inputs fall back with one warning ────────────────────────────────

/// Calls on which a statically legal override stays legal.
enum class kept_on_t { none, int8, int8_or_short_k };

struct illegal_t {
    const char *setting;
    kept_on_t kept_on;
};

bool kept(kept_on_t k, const case_t &c) {
    switch (k) {
        case kept_on_t::int8: return c.int8();
        case kept_on_t::int8_or_short_k: return c.int8() || c.K <= 255;
        default: return false;
    }
}

TEST_F(NativeTuningExec, IllegalOverrideFallsBackWithOneWarning) {
    nt::child_buffers_t shared;
    const fixture_t f = make_fixture(fallback_cases(), shared);
    // A BF16 row count without a kernel at nr 16; column groups that divide
    // no team; an odd BF16 kc below K; each with the rule switches off, since
    // with them on the makespan rule would fit its grid to the column-group
    // pin and make it legal.
    const illegal_t illegal[] = {{"nr=16,mr=12", kept_on_t::int8},
            {"col_groups=3", kept_on_t::none},
            {"kc=255", kept_on_t::int8_or_short_k}};
    for (const illegal_t &bad : illegal) {
        SCOPED_TRACE(bad.setting);
        const std::string setting
                = std::string(nt::kNativeRulesOff) + "," + bad.setting;
        if (!nt::in_child()) {
            const std::unique_ptr<nat::native_planner_t> planner
                    = nat::make_native_planner(setting.c_str(), nullptr);
            ASSERT_TRUE(planner->has_sources());
            std::vector<nat::native_plan_t> plans;
            stdout_of([&] {
                for (size_t i = 0; i < f.cases.size(); ++i)
                    plans.push_back(planner->plan(
                            problem_of(f.cases[i], f.operands[i])));
            });
            for (size_t i = 0; i < f.cases.size(); ++i)
                EXPECT_EQ(nt::top_source(plans[i]),
                        kept(bad.kept_on, f.cases[i]) ? source_t::override
                                                      : source_t::heuristic)
                        << f.cases[i].name;
        }
        nt::expect_in_child(
                {{kConfig, setting.c_str()}, {kTuningFile, nullptr}}, [&] {
            child_report_t report;
            if (f.refs.size() != f.cases.size()) {
                report.fail("the parent's outputs are missing");
                return report.failures();
            }
            // Every call twice: the warning comes once per entry, and
            // calls above the cache bound plan again on every call.
            std::vector<run_t> runs;
            const std::string out = stdout_of([&] {
                for (int pass = 0; pass < 2; ++pass)
                    for (size_t i = 0; i < f.cases.size(); ++i)
                        runs.push_back(run_call(f.cases[i], f.operands[i]));
            });
            size_t dropped = 0;
            for (size_t i = 0; i < f.cases.size(); ++i) {
                const case_t &c = f.cases[i];
                dropped += !kept(bad.kept_on, c);
                for (int pass = 0; pass < 2; ++pass) {
                    const std::string why = bits_differ(
                            c, runs[pass * f.cases.size() + i], f.refs[i]);
                    if (!why.empty())
                        report.fail(std::string(c.name) + " pass "
                                + std::to_string(pass) + ": " + why);
                }
            }
            if (warnings_on()
                    && count_of(out, "gives an illegal plan")
                            != (dropped > 0 ? 1u : 0u))
                report.fail("one warning expected: " + out);
            if (warnings_on() && dropped > 0 && !contains(out, "; dropped"))
                report.fail("the warning says dropped: " + out);
            return report.failures();
        }, setting);
    }
}

TEST_F(NativeTuningExec, MalformedOverrideClausesAreDroppedWithAWarning) {
    nt::child_buffers_t shared;
    const fixture_t f = make_fixture(fallback_cases(), shared);
    // Unknown key, malformed value, out of range, conflicting schedule,
    // INT8 widths without 16, odd min_pass_k, int8_kc not a multiple of 4,
    // malformed filter, unknown filter name.
    const char *const malformed[] = {"foo=1", "nr=abc", "nr=48",
            "schedule=slice,col_groups=2", "schedule=row_groups,col_groups=0",
            "int8_cw=32/64", "min_pass_k=255", "int8_kc=6", "m=abc:nr=16",
            "nr==16", "dtype=f64:nr=16"};
    std::string all;
    for (const char *text : malformed)
        all += std::string(text) + ";";
    // The bad clauses before a good one: the good one stays.
    all += "traversal=n_inner";
    if (!nt::in_child()) {
        for (const char *text : malformed) {
            SCOPED_TRACE(text);
            std::unique_ptr<nat::native_planner_t> made;
            const std::string out = stdout_of(
                    [&] { made = nat::make_native_planner(text, nullptr); });
            ASSERT_NE(made, nullptr);
            EXPECT_FALSE(made->has_sources()) << "its only clause is bad";
            if (warnings_on()) {
                EXPECT_TRUE(contains(out, text))
                        << "the warning names the clause: " << out;
            }
        }
        std::unique_ptr<nat::native_planner_t> made;
        const std::string out = stdout_of(
                [&] { made = nat::make_native_planner(all.c_str(), nullptr); });
        ASSERT_TRUE(made->has_sources());
        if (warnings_on()) {
            EXPECT_EQ(count_of(out, "dropped"), std::size(malformed)) << out;
        }
        for (size_t i = 0; i < f.cases.size(); ++i)
            EXPECT_EQ(nt::top_source(made->plan(
                              problem_of(f.cases[i], f.operands[i]))),
                    source_t::override)
                    << f.cases[i].name;

        // At most 64 clauses are read.
        std::string text;
        for (int m = 1; m <= 65; ++m)
            text += (m > 1 ? ";m=" : "m=") + std::to_string(m)
                    + ":traversal=n_inner";
        stdout_of([&] {
            made = nat::make_native_planner(text.c_str(), nullptr);
        });
        ASSERT_NE(made, nullptr);
        EXPECT_EQ(
                nt::top_source(made->plan(make_problem(64, 256, 64, 8, true))),
                source_t::override);
        EXPECT_EQ(
                nt::top_source(made->plan(make_problem(65, 256, 64, 8, true))),
                source_t::heuristic);
    }
    nt::expect_in_child({{kConfig, all.c_str()}, {kTuningFile, nullptr}}, [&] {
        if (!nat::native_planner().has_sources()) {
            std::fprintf(stderr, "the good clause was dropped\n");
            return 1;
        }
        return child_same_bits(f);
    }, "malformed clauses before a good one");
}

TEST_F(NativeTuningExec, BadTuningFilesAreIgnoredWithAWarning) {
    nt::child_buffers_t shared;
    const fixture_t f = make_fixture(fallback_cases(), shared);
    const std::string entry
            = R"({"id": "all", "config": {"traversal": "n_inner"}})";
    const std::string good_head
            = R"({"format": "zendnnl-native-brgemm-tuning", )";
    const std::string l2_kib
            = std::to_string(nat::native_hw().uarch.l2_bytes / 1024 + 1);
    struct bad_file_t {
        const char *name;
        std::string text;
        const char *named; ///< what the warning must name besides the path
    };
    const bad_file_t bad[] = {
            {"not_json", good_head + R"("version": 2, "entries": [)", ""},
            {"wrong_format",
                    R"({"format": "other", "version": 2, "entries": [)" + entry
                            + "]}",
                    ""},
            {"old_version",
                    good_head + R"("version": 1, "entries": [)" + entry + "]}",
                    "version"},
            {"future_version",
                    good_head + R"("version": 3, "entries": [)" + entry + "]}",
                    ""},
            {"no_version", good_head + R"("entries": [)" + entry + "]}", ""},
            {"other_l2",
                    tuning_json(R"({"l2_kib": )" + l2_kib + "}", entry, ""),
                    "l2_kib"},
            {"other_model",
                    tuning_json(this_hardware("model",
                                        nlohmann::json::array(
                                                {nat::native_hw().model + 1})),
                            entry, ""),
                    "hardware.model"},
            {"signature",
                    tuning_json(this_hardware("signature", "v1;isa=none"),
                            entry, ""),
                    "unknown hardware key 'signature'"},
            {"bad_range",
                    tuning_json(
                            this_hardware("l2_kib",
                                    nlohmann::json {{"min", 2}, {"max", 1}}),
                            entry, ""),
                    "hardware.l2_kib has a bad value"},
    };
    std::vector<std::unique_ptr<nt::temp_file_t>> files;
    std::vector<std::pair<std::string, std::string>> cases; // path, named
    for (const bad_file_t &b : bad) {
        files.push_back(std::make_unique<nt::temp_file_t>(
                std::string(b.name) + ".json", b.text));
        cases.emplace_back(files.back()->path(), b.named);
    }
    cases.emplace_back(
            ::testing::TempDir() + "native_tuning_no_such_dir/none.json", "");
    if (!nt::in_child()) {
        for (const auto &bad_file : cases) {
            SCOPED_TRACE(bad_file.first);
            std::unique_ptr<nat::native_planner_t> made;
            const std::string out = stdout_of([&] {
                made = nat::make_native_planner(
                        nullptr, bad_file.first.c_str());
            });
            ASSERT_NE(made, nullptr);
            EXPECT_FALSE(made->has_sources());
            if (warnings_on()) {
                EXPECT_TRUE(contains(out, bad_file.first)) << out;
                EXPECT_TRUE(contains(out, bad_file.second)) << out;
            }
        }
    }
    // A file made for another CPU leaves the calls on the built-in plan.
    nt::expect_in_child(
            {{kConfig, nullptr}, {kTuningFile, cases[6].first.c_str()}}, [&] {
        if (nat::native_planner().has_sources()) {
            std::fprintf(stderr, "the file was not ignored\n");
            return 1;
        }
        return child_same_bits(f);
    }, "a file made for another CPU");
}

TEST_F(NativeTuningExec, BadTuningEntriesFallBackToTheNextSource) {
    nt::child_buffers_t shared;
    const fixture_t f = make_fixture(fallback_cases(), shared);
    // Four entries are dropped when read; too-tall is legal when read but
    // gives BF16 calls a row count without a kernel at its tile width.
    const char *const entries = R"(
  {"id": "bad-key", "config": {"foo": 1}},
  {"id": "bad-type", "config": {"nr": "x"}},
  {"id": "bad-range", "config": {"nr": 48}},
  {"id": "bad-match", "match": {"color": "red"}, "config": {"nr": 16}},
  {"id": "too-tall", "match": {"dtype": "bf16"}, "config": {"nr": 16, "mr": 12}},
  {"id": "int8-order", "match": {"dtype": "s8"},
   "config": {"traversal": "n_inner"}})";
    const char *const rules
            = R"({"kind": "tree", "root": {"id": "everything", "config": {"row_unit": 6}}})";
    const nt::temp_file_t file(
            "bad_entries.json", tuning_json(this_hardware(), entries, rules));
    size_t bf16_calls = 0;
    for (const case_t &c : f.cases)
        bf16_calls += !c.int8();
    if (!nt::in_child()) {
        std::unique_ptr<nat::native_planner_t> made;
        const std::string loaded = stdout_of([&] {
            made = nat::make_native_planner(nullptr, file.path().c_str());
        });
        ASSERT_NE(made, nullptr);
        ASSERT_TRUE(made->has_sources());
        if (warnings_on()) {
            for (const char *id :
                    {"bad-key", "bad-type", "bad-range", "bad-match"})
                EXPECT_TRUE(contains(loaded, id)) << id << ": " << loaded;
        }
        std::vector<nat::native_plan_t> plans;
        stdout_of([&] {
            for (size_t i = 0; i < f.cases.size(); ++i)
                plans.push_back(
                        made->plan(problem_of(f.cases[i], f.operands[i])));
        });
        for (size_t i = 0; i < f.cases.size(); ++i) {
            const case_t &c = f.cases[i];
            const nat::native_plan_t &plan = plans[i];
            SCOPED_TRACE(c.name);
            EXPECT_TRUE(contains(label_of(plan.rules_entry), "everything"))
                    << label_of(plan.rules_entry);
            if (c.int8()) {
                EXPECT_EQ(nt::top_source(plan), source_t::table);
                EXPECT_TRUE(contains(label_of(plan.table_entry), "int8-order"))
                        << label_of(plan.table_entry);
            } else {
                EXPECT_EQ(nt::top_source(plan), source_t::rules);
                EXPECT_EQ(plan.table_entry, nullptr);
            }
        }
    }
    nt::expect_in_child(
            {{kConfig, nullptr}, {kTuningFile, file.path().c_str()}}, [&] {
        child_report_t report;
        if (f.refs.size() != f.cases.size()) {
            report.fail("the parent's outputs are missing");
            return report.failures();
        }
        std::vector<run_t> runs;
        const std::string out = stdout_of([&] {
            for (int pass = 0; pass < 2; ++pass)
                for (size_t i = 0; i < f.cases.size(); ++i)
                    runs.push_back(run_call(f.cases[i], f.operands[i]));
        });
        for (size_t i = 0; i < f.cases.size(); ++i)
            for (int pass = 0; pass < 2; ++pass) {
                const std::string why = bits_differ(
                        f.cases[i], runs[pass * f.cases.size() + i], f.refs[i]);
                if (!why.empty())
                    report.fail(std::string(f.cases[i].name) + " pass "
                            + std::to_string(pass) + ": " + why);
            }
        if (warnings_on() && bf16_calls > 0
                && (count_of(out, "gives an illegal plan") != 1
                        || !contains(out, "too-tall")
                        || !contains(out, "; dropped")))
            report.fail("one warning for too-tall expected: " + out);
        return report.failures();
    }, "bad entries");
}

// ── Concurrency ──────────────────────────────────────────────────────────

/// Plans of two planners built from the same settings: equal values, and
/// entries with equal labels.
bool same_plan(const nat::native_plan_t &a, const nat::native_plan_t &b) {
    return a.nr == b.nr && a.mr == b.mr && a.mb == b.mb
            && a.m_blocks == b.m_blocks && a.n_tiles == b.n_tiles
            && a.kc == b.kc && a.threads == b.threads
            && a.col_groups == b.col_groups && a.row_unit == b.row_unit
            && a.traversal == b.traversal
            && a.scratch_floats == b.scratch_floats
            && a.int8_rows_nv1 == b.int8_rows_nv1
            && a.int8_rows_nv2 == b.int8_rows_nv2
            && a.int8_cw_mask == b.int8_cw_mask
            && a.int8_pass_budget == b.int8_pass_budget
            && a.int8_pass_quads == b.int8_pass_quads
            && a.int8_direct_store == b.int8_direct_store
            && a.int8_bias_in_dequant == b.int8_bias_in_dequant
            && a.copy_block == b.copy_block && a.copy_par_min == b.copy_par_min
            && a.sources == b.sources
            && label_of(a.rules_entry) == label_of(b.rules_entry)
            && label_of(a.table_entry) == label_of(b.table_entry)
            && label_of(a.override_entry) == label_of(b.override_entry);
}

TEST_F(NativeTuningExec, PlanCacheUnderConcurrentPlanning) {
    const char *settings
            = "m=1-64:nr=16,mb=24;m=65-1024:schedule=row_groups,"
              "domain_threads=8;m=1025-:kc_split=equal,traversal=n_inner";
    std::vector<nat::native_problem_t> problems;
    for (bool bf16 : {true, false})
        for (int threads : {1, 4, 8, 32})
            for (int N : {64, 256, 1024, 4096})
                for (int K : {64, 1000, 4096})
                    for (int M : {1, 2, 3, 5, 8, 13, 16, 24, 32, 48, 64, 65,
                                 100, 128, 300, 512, 1024, 1025, 2048, 4096})
                        problems.push_back(
                                make_problem(M, N, K, threads, bf16));
    // The plans of one thread, from a planner of its own.
    const std::unique_ptr<nat::native_planner_t> serial
            = nat::make_native_planner(settings, nullptr);
    ASSERT_TRUE(serial->has_sources());
    std::vector<nat::native_plan_t> want;
    for (const nat::native_problem_t &p : problems)
        want.push_back(serial->plan(p));

    // A cold planner shared by many threads: misses and inserts race.
    const std::unique_ptr<nat::native_planner_t> shared
            = nat::make_native_planner(settings, nullptr);
    std::atomic<size_t> mismatches {0};
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < 16; ++t)
        pool.emplace_back([&, t] {
            std::mt19937 rng(t + 1);
            for (int i = 0; i < 20000; ++i) {
                const size_t k = rng() % problems.size();
                if (!same_plan(shared->plan(problems[k]), want[k]))
                    mismatches.fetch_add(1, std::memory_order_relaxed);
            }
        });
    for (std::thread &t : pool)
        t.join();
    EXPECT_EQ(mismatches.load(), 0u);
    report_t report;
    for (size_t k = 0; k < problems.size(); ++k)
        report.check(same_plan(shared->plan(problems[k]), want[k]), [&] {
            return "M " + std::to_string(problems[k].M) + " N "
                    + std::to_string(problems[k].N) + " K "
                    + std::to_string(problems[k].K) + " threads "
                    + std::to_string(problems[k].threads)
                    + (problems[k].bf16() ? " bf16" : " int8");
        });
    report.finish();
}

TEST_F(NativeTuningExec, ConcurrentCallsKeepPlansAndOutputs) {
    const post_op_type_t none = post_op_type_t::none;
    nt::child_buffers_t shared;
    const fixture_t f = make_fixture(
            {
                    {"c_bf16_m1", src_t::bf16, 1, 512, 256, kBf16, 2},
                    {"c_bf16_m8", src_t::bf16, 8, 256, 320, kF32, 2},
                    {"c_bf16_m24_bias", src_t::bf16, 24, 128, 200, kBf16, 2,
                            false, kF32},
                    {"c_bf16_m70_gelu", src_t::bf16, 70, 96, 128, kBf16, 1,
                            false, kNone, post_op_type_t::gelu_erf},
                    {"c_bf16_trans", src_t::bf16, 9, 64, 72, kF32, 2, true},
                    {"c_int8_m1", src_t::dyn_s8, 1, 512, 256, kBf16, 1},
                    {"c_int8_m8_bias", src_t::dyn_s8, 8, 256, 320, kBf16, 2,
                            false, kF32},
                    {"c_int8_u8_m24", src_t::st_u8, 24, 128, 201, kF32, 2},
                    {"c_int8_m70", src_t::dyn_s8, 70, 96, 128, kF32, 1, false,
                            kNone, none},
                    {"c_int8_s8_m3", src_t::st_s8, 3, 64, 64, kBf16, 2},
            },
            shared);
    ASSERT_FALSE(f.cases.empty());
    const char *const settings
            = "m=1-8:nr=16;dtype=s8:int8_cw=32/16,int8_kc=64;"
              "m=9-:traversal=n_inner,row_unit=6,mb=24,kc_split=equal";
    nt::expect_in_child({{kConfig, settings}, {kTuningFile, nullptr}}, [&] {
        child_report_t report;
        if (f.refs.size() != f.cases.size()) {
            report.fail("the parent's outputs are missing");
            return report.failures();
        }
        // The weights are packed and the plans cached on this thread
        // first.
        if (child_same_bits(f) != 0) {
            report.fail("one thread differs from the built-in plan");
            return report.failures();
        }
        constexpr int kWorkers = 8, kRounds = 4;
        std::atomic<size_t> failed {0}, differ {0};
        std::vector<std::thread> workers;
        for (int t = 0; t < kWorkers; ++t)
            workers.emplace_back([&, t] {
                // A source scale buffer of its own: dynamic
                // quantization writes it.
                std::vector<operands_t> mine = f.operands;
                for (size_t i = 0; i < mine.size(); ++i) {
                    if (f.cases[i].src != src_t::dyn_s8) continue;
                    mine[i].src_scale = nt::make_buffer(
                            static_cast<size_t>(f.cases[i].M) * 4);
                    mine[i].params.quant_params.src_scale.buff
                            = mine[i].src_scale.get();
                }
                const size_t n = f.cases.size();
                for (int round = 0; round < kRounds; ++round)
                    for (size_t j = 0; j < n; ++j) {
                        const size_t i
                                = (j + static_cast<size_t>(t + round)) % n;
                        const case_t &c = f.cases[i];
                        const run_t r = run_call(c, mine[i]);
                        if (r.status != status_t::success)
                            failed.fetch_add(1);
                        else if (!nt::within_steps(r.dst.get(), c.cols(),
                                         f.refs[i].get(), c.cols(), c.dst, c.M,
                                         c.cols(), 0))
                            differ.fetch_add(1);
                    }
            });
        for (std::thread &w : workers)
            w.join();
        if (failed.load() != 0) report.fail("calls failed");
        if (differ.load() != 0)
            report.fail("outputs that differ from one thread's");
        for (size_t i = 0; i < f.cases.size(); ++i)
            if (nt::native_layouts(f.operands[i].wei.get()) == 0)
                report.fail(std::string(f.cases[i].name)
                        + " not served by the native BRGEMM");
        return report.failures();
    }, "concurrent calls");
}

} // namespace
