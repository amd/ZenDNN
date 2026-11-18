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

// The GEMV kernels of M = 1 calls on the native BRGEMM.  Every M = 1 call
// reads the weight's one layout on the GEMV kernels; their results must be
// the tiles' bit for bit: each case runs on the tiles
// (native_brgemm_execute_on_tiles, over a copy of the raw weight packed for
// the call) and through matmul_direct under ZENDNNL_MATMUL_ALGO=10 at the
// case's weight cache mode (a first call that packs, caches or rewrites the
// weight in place, then a second that reads what the first left), over every
// type the native BRGEMM serves, K tails, groups of an odd number of K quads,
// f16 over several blocks of its K steps, N with a padded tail, the INT8 and
// s4 kernels on a team, gated activations over whole blocks of each half,
// over runs of 16 columns and over staged rows, swiglu pairs, every bias
// type, post-ops, alpha and beta and every dst each type takes.  The whole
// dst, the ldc padding included, is compared.  A subset, of products of one
// sign in each column, is checked against an f64 model within the bound of
// each kernel's accumulation; the plan line names the GEMV kernels, in every
// M = 1 call of every bit case, and keeps the tiles' plan, and a weight
// shared by M = 1 calls and calls of larger M keeps one layout.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "matmul_native/native_brgemm_test_utils.hpp"

namespace {

namespace nt = native_brgemm_test;
namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
using nt::buffer_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::post_op_type_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;
using gated_t = mm::matmul_gated_act_t;
using quant_t = mm::matmul_quantization_params_t::matmul_quant_t;

constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kF16 = data_type_t::f16;
constexpr data_type_t kS8 = data_type_t::s8;
constexpr data_type_t kU8 = data_type_t::u8;
constexpr data_type_t kS32 = data_type_t::s32;
constexpr data_type_t kS4 = data_type_t::s4;
constexpr data_type_t kNone = data_type_t::none;

/// The data types and quantization of a call: bf16, f32, f16; a static s8
/// source with a per-tensor scale, or a u8 one with a zero point, over
/// per-channel s8 weights; a bf16 source quantized in the call to s8, or to
/// u8 with a zero point; per-group s8 weights over an s8 source of per-group
/// scales, or over a u8 source; per-channel s8 weights over an s8 source of
/// per-group scales (the weight scales apply after the groups); s4 weights
/// over a static s8 source (fused, or reference order with alpha / beta or an
/// integer dst), over an s8 source of per-group scales, over a bf16 source
/// quantized to s8, and of one group over K.
enum class form_t {
    bf16,
    f32,
    f16,
    st_s8,
    st_u8,
    dyn_s8,
    dyn_u8,
    grp_s8,
    grp_u8,
    after,
    s4,
    s4_grp,
    s4_dyn,
    s4_one
};

const char *form_name(form_t f) {
    static const char *const kNames[]
            = {"bf16", "f32", "f16", "sts8", "stu8", "dyns8", "dynu8", "grps8",
                    "grpu8", "after", "s4", "s4grp", "s4dyn", "s4one"};
    return kNames[static_cast<int>(f)];
}

bool int8_form(form_t f) {
    return f != form_t::bf16 && f != form_t::f32 && f != form_t::f16;
}

bool s4_form(form_t f) {
    return f == form_t::s4 || f == form_t::s4_grp || f == form_t::s4_dyn
            || f == form_t::s4_one;
}

/// The K values of a group of the per-group forms, unless a case gives its
/// own.
constexpr int kGroup = 32;

struct gcase_t {
    std::string name;
    form_t form = form_t::bf16;
    int N = 64, K = 64;
    data_type_t dst = kBf16;
    bool transB = false;
    int threads = 1;
    float alpha = 1.0f, beta = 0.0f;
    data_type_t bias = kNone;
    post_op_type_t po = post_op_type_t::none;
    data_type_t po_dt = kF32;
    gated_t gated = gated_t::none;
    int wc = 1;
    data_type_t wscale = kF32; ///< weight scale type of the INT8 forms
    /// A u8 source's zero point far outside the byte range: the kernels take
    /// the zero-point term in int64.
    bool wide_zp = false;
    int group = kGroup; ///< K values of a group of the per-group forms
    /// The call runs on a team: its layout is above a quarter of the L2.
    bool team = false;
    int pad = 3; ///< ldc columns past the output

    int out_cols() const { return gated == gated_t::none ? N : N / 2; }
    /// beta of a gated call reads the dst's N pre-activation columns.
    int ldc() const {
        return (gated != gated_t::none && beta != 0.0f ? N : out_cols()) + pad;
    }
    int groups() const { return form == form_t::s4_one ? 1 : K / group; }
};

void PrintTo(const gcase_t &c, std::ostream *os) {
    *os << c.name;
}

const char *dt_name(data_type_t dt) {
    switch (dt) {
        case kF32: return "f32";
        case kBf16: return "bf16";
        case kF16: return "f16";
        case kS8: return "s8";
        case kU8: return "u8";
        case kS32: return "s32";
        default: return "none";
    }
}

std::string describe(const gcase_t &c) {
    std::string s = std::string(form_name(c.form)) + "_n" + std::to_string(c.N)
            + "_k" + std::to_string(c.K) + "_" + dt_name(c.dst) + "_t"
            + std::to_string(c.threads) + "_wc" + std::to_string(c.wc);
    if (c.transB) s += "_tb";
    if (int8_form(c.form)) s += std::string("_ws") + dt_name(c.wscale);
    if (c.wide_zp) s += "_widezp";
    if (c.group != kGroup) s += "_g" + std::to_string(c.group);
    if (c.alpha != 1.0f || c.beta != 0.0f)
        s += "_a" + std::to_string(static_cast<int>(c.alpha * 100)) + "b"
                + std::to_string(static_cast<int>(c.beta * 100));
    if (c.bias != kNone) s += std::string("_bias") + dt_name(c.bias);
    switch (c.po) {
        case post_op_type_t::relu: s += "_relu"; break;
        case post_op_type_t::gelu_erf: s += "_gelu"; break;
        case post_op_type_t::softmax: s += "_softmax"; break;
        case post_op_type_t::binary_add:
            s += std::string("_add") + dt_name(c.po_dt);
            break;
        case post_op_type_t::binary_mul:
            s += std::string("_mul") + dt_name(c.po_dt);
            break;
        default: break;
    }
    if (c.gated == gated_t::silu_and_mul) s += "_silu";
    if (c.gated == gated_t::gelu_tanh_and_mul) s += "_gelutanh";
    if (c.gated == gated_t::swiglu_oai_mul) s += "_swiglu";
    return s;
}

std::vector<gcase_t> named(std::vector<gcase_t> v) {
    std::vector<std::string> seen;
    for (gcase_t &c : v) {
        std::string n = describe(c);
        const int dup = static_cast<int>(
                std::count_if(seen.begin(), seen.end(), [&](const auto &s) {
            return s == n || s.rfind(n + "_dup", 0) == 0;
        }));
        if (dup) n += "_dup" + std::to_string(dup);
        seen.push_back(n);
        c.name = n;
    }
    return v;
}

uint32_t seed_of(const std::string &s) {
    uint32_t h = 2166136261u;
    for (char ch : s)
        h = (h ^ static_cast<uint8_t>(ch)) * 16777619u;
    return h;
}

void set_quant(quant_t &q, const buffer_t &b, data_type_t dt,
        std::vector<int64_t> dims) {
    q.buff = b.get();
    q.dt = dt;
    q.dims = std::move(dims);
}

/// The operands of a case and its params pointing at them.
struct operands_t {
    data_type_t src_dt = kBf16, wei_dt = kBf16;
    int ldb = 0;
    buffer_t src, wei, bias, wei_scale, src_scale, src_zp, dst_scale, dst_zp,
            po_buf, init;
    mm::matmul_params params;
};

/// Bytes of the case's raw weight.
size_t weight_bytes(const gcase_t &c, data_type_t wei_dt) {
    const size_t n = static_cast<size_t>(c.K) * c.N;
    return wei_dt == kS4 ? (n + 1) / 2 : n * nt::dt_bytes(wei_dt);
}

operands_t make_operands(const gcase_t &c) {
    operands_t o;
    const uint32_t seed = seed_of(c.name);
    const form_t f = c.form;
    const bool s4 = s4_form(f);
    const bool dyn
            = f == form_t::dyn_s8 || f == form_t::dyn_u8 || f == form_t::s4_dyn;
    const bool st_s8 = f == form_t::st_s8 || f == form_t::grp_s8
            || f == form_t::after || f == form_t::s4 || f == form_t::s4_grp
            || f == form_t::s4_one;
    const bool st_u8 = f == form_t::st_u8 || f == form_t::grp_u8;
    o.src_dt = st_s8           ? kS8
            : st_u8            ? kU8
            : f == form_t::f32 ? kF32
                               : (f == form_t::f16 ? kF16 : kBf16);
    o.wei_dt = s4 ? kS4
            : int8_form(f)
            ? kS8
            : (f == form_t::f32 ? kF32 : (f == form_t::f16 ? kF16 : kBf16));
    o.ldb = c.transB ? c.K : c.N;
    const size_t a_n = static_cast<size_t>(c.K);
    if (o.src_dt == kS8)
        o.src = nt::random_buffer(kS8, a_n, seed, -64, 64);
    else if (o.src_dt == kU8)
        o.src = nt::random_buffer(kU8, a_n, seed, 64, 192);
    else
        o.src = nt::random_buffer(o.src_dt, a_n, seed, -1.0f, 1.0f);
    if (s4) {
        // Two's-complement nibbles at the linear element index, -7 .. 7.
        o.wei = nt::make_buffer(weight_bytes(c, kS4));
        const buffer_t r = nt::random_buffer(
                kS8, static_cast<size_t>(c.K) * c.N, seed + 1, -7, 7);
        auto *w = o.wei.as<uint8_t>();
        for (size_t i = 0; i < static_cast<size_t>(c.K) * c.N; ++i) {
            const uint8_t v = static_cast<uint8_t>(r.as<int8_t>()[i]) & 15u;
            w[i / 2] = static_cast<uint8_t>((i % 2 == 0)
                            ? ((w[i / 2] & 0xF0u) | v)
                            : ((w[i / 2] & 0x0Fu) | (v << 4)));
        }
    } else if (int8_form(f)) {
        o.wei = nt::random_buffer(
                kS8, static_cast<size_t>(c.K) * c.N, seed + 1, -127, 127);
    } else {
        o.wei = nt::random_buffer(o.wei_dt, static_cast<size_t>(c.K) * c.N,
                seed + 1, -1.0f, 1.0f);
    }
    if (c.bias != kNone)
        o.bias = nt::random_buffer(c.bias, c.N, seed + 2, -2.0f, 2.0f);

    mm::matmul_params &p = o.params;
    p.dtypes.src = o.src_dt;
    p.dtypes.wei = o.wei_dt;
    p.dtypes.dst = c.dst;
    p.dtypes.bias = c.bias;
    p.num_threads = c.threads;
    p.weight_cache_type = c.wc;
    // matmul_direct takes an s4 weight with an s8 compute type.
    if (s4) p.dtypes.compute = kS8;
    auto &qp = p.quant_params;
    if (int8_form(f)) {
        const int G = c.groups();
        const bool wgroups = f == form_t::grp_s8 || f == form_t::grp_u8 || s4;
        const int rows = wgroups ? G : 1;
        o.wei_scale = nt::random_buffer(c.wscale,
                static_cast<size_t>(rows) * c.N, seed + 3, 0.002f, 0.01f);
        set_quant(qp.wei_scale, o.wei_scale, c.wscale, {rows, c.N});
        if (dyn) {
            p.dynamic_quant = true;
            p.dtypes.compute = f == form_t::dyn_u8 ? kU8 : kS8;
            o.src_scale = nt::make_buffer(4);
            set_quant(qp.src_scale, o.src_scale, kF32, {1, 1});
            if (f == form_t::dyn_u8) {
                o.src_zp = nt::make_buffer(4);
                set_quant(qp.src_zp, o.src_zp, kS32, {1, 1});
            }
        } else if (f == form_t::grp_s8 || f == form_t::after
                || f == form_t::s4_grp) {
            o.src_scale = nt::random_buffer(
                    kF32, static_cast<size_t>(G), seed + 4, 0.005f, 0.02f);
            set_quant(qp.src_scale, o.src_scale, kF32, {1, G});
        } else {
            o.src_scale = nt::random_buffer(kF32, 1, seed + 4, 0.005f, 0.02f);
            set_quant(qp.src_scale, o.src_scale, kF32, {1, 1});
            if (st_u8) {
                o.src_zp = nt::random_buffer(kS32, 1, seed + 5, 100, 156);
                if (c.wide_zp) o.src_zp.as<int32_t>()[0] = 1 << 22;
                set_quant(qp.src_zp, o.src_zp, kS32, {1, 1});
            }
        }
        if (c.dst == kU8 || c.dst == kS8) {
            o.dst_scale = nt::make_buffer(4);
            nt::store_f32(o.dst_scale.get(), kF32, 0, 4.0f);
            set_quant(qp.dst_scale, o.dst_scale, kF32, {1, 1});
            o.dst_zp = nt::make_buffer(4);
            nt::store_f32(
                    o.dst_zp.get(), kS32, 0, c.dst == kU8 ? 128.0f : 0.0f);
            set_quant(qp.dst_zp, o.dst_zp, kS32, {1, 1});
        }
    }
    if (c.po != post_op_type_t::none) {
        mm::matmul_post_op m;
        m.po_type = c.po;
        if (c.po == post_op_type_t::binary_add
                || c.po == post_op_type_t::binary_mul) {
            // A gated call's binary operand indexes its N pre-activation
            // columns.
            const int cols = c.gated == gated_t::none ? c.N : c.N;
            o.po_buf = nt::random_buffer(
                    c.po_dt, static_cast<size_t>(cols), seed + 6, -1.0f, 1.0f);
            m.buff = o.po_buf.get();
            m.dtype = c.po_dt;
            m.dims = {1, cols};
        }
        p.postop_.push_back(m);
    }
    o.init = nt::random_buffer(c.dst, static_cast<size_t>(c.ldc()), seed + 7,
            c.dst == kU8 ? 0.0f : -2.0f, c.dst == kU8 ? 255.0f : 2.0f);
    return o;
}

/// A copy of b in a buffer of its own.
buffer_t copy_of(const buffer_t &b) {
    buffer_t out = nt::make_buffer(b.bytes);
    if (b.bytes) std::memcpy(out.get(), b.get(), b.bytes);
    return out;
}

/// The descriptor native_layout_matmul builds for the call.
nat::GemmDescriptor descriptor_of(const nt::call_t &c) {
    nat::GemmDescriptor d;
    d.M = c.M;
    d.N = c.N;
    d.K = c.K;
    d.lda = c.lda_or_tight();
    d.ldb = c.ldb_or_tight();
    d.ldc = c.ldc_or_tight();
    d.transA = c.transA;
    d.transB = c.transB;
    d.alpha = c.alpha;
    d.beta = c.beta;
    d.src_dt = c.params.dtypes.src;
    d.wei_dt = c.params.dtypes.wei;
    d.dst_dt = c.params.dtypes.dst;
    d.bias_dt = c.params.dtypes.bias;
    d.bias = c.bias;
    d.is_weights_const = c.is_weights_const;
    d.num_threads = zendnnl::lowoha::resolve_num_threads(
            c.params.num_threads, zendnnl::lowoha::thread_guard::max_threads());
    d.gated_act = c.gated;
    return d;
}

/// The call on the tiles, or (gemv) through native_brgemm_execute, over its
/// raw weight packed for the call.
status_t run_tiles(const nt::call_t &c, bool gemv = false) {
    nat::native_weight_ref_t ref;
    ref.weight = c.wei;
    ref.pack_nr = nat::kNativeLayoutPackNr;
    ref.weight_cache_mode = 0;
    const zendnnl::lowoha::thread_guard tg(
            c.params.num_threads, zendnnl::lowoha::thread_guard::max_threads());
    const nat::GemmDescriptor d = descriptor_of(c);
    return gemv
            ? nat::native_brgemm_execute(d, ref, c.src, c.dst, c.bias, c.params)
            : nat::native_brgemm_execute_on_tiles(
                      d, ref, c.src, c.dst, c.bias, c.params);
}

/// matmul_direct's checks take the case: source groups over per-channel
/// weight scales, an s4 weight into an integer dst, an f32 source into a
/// bf16 dst and a bf16 source the call quantizes into an f16 dst reach the
/// native BRGEMM through native_brgemm_execute only.
bool matmul_direct_takes(const gcase_t &c) {
    if (s4_form(c.form)) return c.dst == kF32 || c.dst == kBf16;
    if (c.form == form_t::f32) return c.dst == kF32;
    if (c.form == form_t::dyn_s8 || c.form == form_t::dyn_u8)
        return c.dst != kF16;
    return c.form != form_t::after;
}

/// The source scales of M rows of a call of per-tensor (static) source
/// scales: matmul_direct takes an s4 weight's static scales per row, as
/// many as rows ({M, 1}), each the first row's.
buffer_t row_scales(const operands_t &o, int M) {
    buffer_t b = nt::make_buffer(static_cast<size_t>(M) * 4);
    for (int m = 0; m < M; ++m)
        b.as<float>()[m] = o.src_scale.as<float>()[0];
    return b;
}

nt::call_t make_call(
        const gcase_t &c, const operands_t &o, const void *wei, void *dst) {
    nt::call_t call;
    call.M = 1;
    call.N = c.N;
    call.K = c.K;
    call.transB = c.transB;
    call.lda = c.K;
    call.ldb = o.ldb;
    call.ldc = c.ldc();
    call.alpha = c.alpha;
    call.beta = c.beta;
    call.src = o.src.get();
    call.wei = wei;
    call.bias = o.bias.get();
    call.dst = dst;
    call.params = o.params;
    call.gated = c.gated;
    return call;
}

/// Whether the CPU runs the case's kernels: AVX-512 F / BW / VL / DQ and FMA,
/// with AVX512-BF16 for bf16 and AVX512-VNNI for the INT8 forms.
bool cpu_serves(form_t f) {
    const nat::UarchParams &u = nat::detect_uarch();
    if (!nat::native_base_isa()) return false;
    if (f == form_t::bf16) return u.avx512bf16;
    if (int8_form(f)) return u.avx512vnni;
    return true;
}

/// The bytes of the dst the case compares: its ldc columns, padding
/// included.
size_t dst_bytes(const gcase_t &c) {
    return static_cast<size_t>(c.ldc()) * nt::dt_bytes(c.dst);
}

class NativeGemvBits : public ::testing::TestWithParam<gcase_t> {};

// The GEMV kernels give the tiles' bits: through matmul_direct under
// ZENDNNL_MATMUL_ALGO=10 at the case's weight cache mode, twice on one
// weight, against the tiles on a raw copy of it.
TEST_P(NativeGemvBits, EqualTheTiles) {
    const gcase_t &c = GetParam();
    if (!cpu_serves(c.form)) GTEST_SKIP() << "the CPU lacks the kernels' ISA";
    const WeightCacheGuard mode(2);
    const nt::algo_pin_t algo(matmul_algo_t::native_brgemm);
    const operands_t o = make_operands(c);
    const buffer_t raw = copy_of(o.wei);
    const buffer_t tiles = copy_of(o.init);
    ASSERT_EQ(run_tiles(make_call(c, o, raw.get(), tiles.get())),
            status_t::success);
    {
        const buffer_t gemv = copy_of(o.init);
        ASSERT_EQ(run_tiles(make_call(c, o, raw.get(), gemv.get()), true),
                status_t::success);
        EXPECT_EQ(std::memcmp(gemv.get(), tiles.get(), dst_bytes(c)), 0)
                << "native_brgemm_execute: the GEMV kernels' dst differs from "
                << "the tiles'";
    }
    if (!matmul_direct_takes(c)) return;
    // The weight the GEMV calls read, which mode 2 may rewrite in place.
    const buffer_t wei = nt::kept_until_exit(copy_of(o.wei));
    for (int call = 0; call < 2; ++call) {
        const buffer_t gemv = copy_of(o.init);
        ASSERT_EQ(nt::run_matmul(make_call(c, o, wei.get(), gemv.get())),
                status_t::success)
                << "call " << call;
        EXPECT_EQ(std::memcmp(gemv.get(), tiles.get(), dst_bytes(c)), 0)
                << "call " << call << ": the GEMV kernels' dst differs from "
                << "the tiles'";
    }
    // The raw weight is the tiles' input still.
    EXPECT_EQ(std::memcmp(raw.get(), o.wei.get(), o.wei.bytes), 0);
}

std::vector<gcase_t> bit_cases() {
    std::vector<gcase_t> v;
    const form_t all[] = {form_t::bf16, form_t::f32, form_t::f16, form_t::st_s8,
            form_t::st_u8, form_t::dyn_s8, form_t::dyn_u8, form_t::grp_s8,
            form_t::grp_u8, form_t::after, form_t::s4, form_t::s4_grp,
            form_t::s4_dyn, form_t::s4_one};
    for (const form_t f : all) {
        const bool i8 = int8_form(f);
        const data_type_t dst = f == form_t::f16 ? kF16
                : f == form_t::f32               ? kF32
                                                 : kBf16;
        // K: whole groups for the per-group forms; tails elsewhere.
        const bool grouped = f == form_t::grp_s8 || f == form_t::grp_u8
                || f == form_t::after || s4_form(f);
        const std::vector<int> ks = grouped ? std::vector<int> {64, 256}
                                            : (i8 ? std::vector<int> {61, 256}
                                                  : std::vector<int> {67, 256});
        // N: one block, a padded tail, whole blocks over several workers.
        for (const int N : {32, 100, 768}) {
            for (const int K : ks) {
                for (const int wc : {0, 1, 2}) {
                    gcase_t c;
                    c.form = f;
                    c.N = N;
                    c.K = K;
                    c.dst = dst;
                    c.wc = wc;
                    c.threads = N == 768 ? 8 : 1;
                    c.transB = (N + K + wc) % 2 == 0;
                    c.wscale = (K + wc) % 3 == 0 ? kBf16
                            : (K + wc) % 3 == 1  ? kF32
                                                 : kF16;
                    v.push_back(c);
                }
            }
        }
        // The epilogue: biases, post-ops, alpha / beta, dsts.
        gcase_t b;
        b.form = f;
        b.N = 160;
        b.K = grouped ? 128 : (i8 ? 92 : 96);
        b.dst = dst;
        b.threads = 4;
        b.wscale = kBf16;
        // Every bias type into the form's dst and into an f32 dst, which
        // keeps every bit of the f32 result.
        for (const data_type_t bias : {kF32, kBf16, kF16, kS8, kU8, kS32}) {
            gcase_t c = b;
            c.bias = bias;
            v.push_back(c);
            if (b.dst == kF32) continue;
            c.dst = kF32;
            v.push_back(c);
        }
        for (const post_op_type_t po : {post_op_type_t::relu,
                     post_op_type_t::gelu_erf, post_op_type_t::softmax}) {
            gcase_t c = b;
            c.po = po;
            c.bias = kF32;
            v.push_back(c);
        }
        for (const data_type_t po_dt : {kF32, kBf16}) {
            gcase_t c = b;
            c.po = post_op_type_t::binary_add;
            c.po_dt = po_dt;
            v.push_back(c);
            c.po = post_op_type_t::binary_mul;
            v.push_back(c);
        }
        {
            gcase_t c = b;
            c.alpha = 0.5f;
            c.beta = 0.75f;
            c.bias = kBf16;
            v.push_back(c);
        }
        for (const data_type_t d : {kF32, kBf16, kF16}) {
            // An f32 call writes f32 or bf16, an f16 call f16 or f32; an f16
            // dst takes no per-group call.
            if (f == form_t::f16 && d == kBf16) continue;
            if (d == kF16 && (f == form_t::f32 || grouped)) continue;
            gcase_t c = b;
            c.dst = d;
            v.push_back(c);
        }
        // A bf16 call and the INT8 forms write the integer dsts too.
        if (i8 || f == form_t::bf16) {
            for (const data_type_t d : {kS8, kU8, kS32}) {
                gcase_t c = b;
                c.dst = d;
                v.push_back(c);
            }
        }
        if (f == form_t::st_u8 || f == form_t::grp_u8) {
            for (const int N : {100, 768}) {
                gcase_t c = b;
                c.N = N;
                c.wide_zp = true;
                c.bias = f == form_t::st_u8 ? kBf16 : kNone;
                c.dst = kF32;
                v.push_back(c);
            }
        }
        // Gated: halves of whole blocks, of runs of 16 and staged; swiglu.
        for (const int N : {256, 96, 100}) {
            for (const gated_t g : {gated_t::silu_and_mul,
                         gated_t::gelu_tanh_and_mul, gated_t::swiglu_oai_mul}) {
                gcase_t c = b;
                c.N = N;
                c.gated = g;
                c.bias = kF32;
                c.threads = 4;
                c.wc = N == 256 ? 2 : 1;
                v.push_back(c);
            }
            gcase_t c = b;
            c.N = N;
            c.gated = gated_t::silu_and_mul;
            c.beta = 0.5f;
            c.po = post_op_type_t::binary_add;
            v.push_back(c);
        }
    }
    // Groups of an odd number of K quads (4 and 12 K values): an s8 group
    // ends on a quad of its own, an s4 group starts or ends half way through
    // an octet; in the fused order, and in the reference order of alpha /
    // beta and of an s8 dst.  And one s4 group over a K of an odd number of
    // quads.
    for (const form_t f : {form_t::grp_s8, form_t::grp_u8, form_t::after,
                 form_t::s4, form_t::s4_grp, form_t::s4_dyn}) {
        for (const int group : {4, 12}) {
            for (const int K : {36, 132}) {
                for (const int N : {32, 100}) {
                    gcase_t c;
                    c.form = f;
                    c.N = N;
                    c.K = K;
                    c.group = group;
                    c.dst = N == 32 ? kF32 : kBf16;
                    c.wc = (K / 4 + N + group) % 3;
                    c.transB = (K / 12 + N / 4 + group / 4) % 2 == 0;
                    c.wscale = (K + N + group) % 3 == 0 ? kBf16
                            : (K + N + group) % 3 == 1  ? kF32
                                                        : kF16;
                    v.push_back(c);
                }
            }
            gcase_t c;
            c.form = f;
            c.N = 100;
            c.K = 132;
            c.group = group;
            c.threads = 4;
            c.alpha = 0.5f;
            c.beta = 0.75f;
            c.bias = kBf16;
            c.dst = kF32;
            v.push_back(c);
            c.alpha = 1.0f;
            c.beta = 0.0f;
            c.bias = kNone;
            c.dst = kS8;
            v.push_back(c);
        }
    }
    for (const int K : {12, 100}) {
        for (const int N : {32, 100}) {
            gcase_t c;
            c.form = form_t::s4_one;
            c.N = N;
            c.K = K;
            c.dst = N == 32 ? kF32 : kBf16;
            c.wc = (K + N) % 3;
            v.push_back(c);
        }
    }
    // The INT8 and s4 kernels on a team: layouts of 2 MiB or more, above a
    // quarter of the L2 of a core of up to 8 MiB, at 8 threads, into an f32
    // dst, which keeps every bit of the sums; padded N, fewer units than
    // threads (N 160), gated halves of whole blocks (N 2048) and of runs of
    // 16 (N 2080).
    struct team_case_t {
        form_t form;
        int N, K;
        gated_t gated;
        int group;
    };
    const team_case_t teams[] = {
            {form_t::st_s8, 2048, 1024, gated_t::none, kGroup},
            {form_t::st_u8, 1000, 2048, gated_t::none, kGroup},
            {form_t::dyn_s8, 2048, 1024, gated_t::none, kGroup},
            {form_t::dyn_u8, 1000, 2048, gated_t::none, kGroup},
            {form_t::st_s8, 2048, 1024, gated_t::silu_and_mul, kGroup},
            {form_t::grp_s8, 1000, 2048, gated_t::none, kGroup},
            {form_t::grp_u8, 2048, 1024, gated_t::none, kGroup},
            {form_t::grp_u8, 1000, 2052, gated_t::none, 12},
            {form_t::grp_s8, 160, 16384, gated_t::none, kGroup},
            {form_t::after, 1000, 2048, gated_t::none, kGroup},
            {form_t::grp_s8, 2048, 1024, gated_t::silu_and_mul, kGroup},
            {form_t::grp_s8, 2080, 1024, gated_t::silu_and_mul, kGroup},
            {form_t::s4, 2048, 2048, gated_t::none, kGroup},
            {form_t::s4_grp, 1000, 4096, gated_t::none, kGroup},
            {form_t::s4_dyn, 2048, 2048, gated_t::none, kGroup},
            {form_t::s4_one, 2048, 2048, gated_t::none, kGroup},
            {form_t::s4, 1000, 4104, gated_t::none, 12},
            {form_t::s4, 2048, 2048, gated_t::silu_and_mul, kGroup},
            {form_t::s4, 2080, 2048, gated_t::silu_and_mul, kGroup},
    };
    int i = 0;
    for (const team_case_t &t : teams) {
        gcase_t c;
        c.form = t.form;
        c.N = t.N;
        c.K = t.K;
        c.gated = t.gated;
        c.group = t.group;
        c.dst = kF32;
        c.threads = 8;
        c.team = true;
        c.wc = i % 3;
        c.transB = i % 2 == 1;
        c.wscale = i % 3 == 0 ? kBf16 : (i % 3 == 1 ? kF32 : kF16);
        v.push_back(c);
        ++i;
    }
    // f16 over more than one block of kNativeF16AccumBlockK steps, whose f16
    // sums the f16_ph kernels add in f32: K 257, 515, 1024 and 1030, at one
    // thread and on a team (layouts of 2 MiB or more at 8 threads), an f16
    // bias, swiglu pairs over staged rows.
    struct f16_case_t {
        int N, K, threads;
        data_type_t dst, bias;
        gated_t gated;
    };
    const f16_case_t f16s[] = {
            {48, 515, 1, kF16, kNone, gated_t::none},
            {48, 1024, 1, kF32, kNone, gated_t::none},
            {100, 257, 1, kF16, kF16, gated_t::none},
            {96, 515, 4, kF16, kF32, gated_t::swiglu_oai_mul},
            {1024, 1024, 8, kF32, kNone, gated_t::none},
            {1000, 1030, 8, kF16, kNone, gated_t::none},
    };
    for (const f16_case_t &t : f16s) {
        gcase_t c;
        c.form = form_t::f16;
        c.N = t.N;
        c.K = t.K;
        c.threads = t.threads;
        c.dst = t.dst;
        c.bias = t.bias;
        c.gated = t.gated;
        c.team = t.threads == 8;
        c.wc = i % 3;
        c.transB = i % 2 == 1;
        v.push_back(c);
        ++i;
    }
    return named(v);
}

INSTANTIATE_TEST_SUITE_P(Native, NativeGemvBits,
        ::testing::ValuesIn(bit_cases()),
        [](const ::testing::TestParamInfo<gcase_t> &i) {
            return i.param.name;
        });

// ── An f64 model ─────────────────────────────────────────────────────────

/// The s4 weight's element i (its linear index), and its store.
int s4_at(const buffer_t &w, size_t i) {
    const uint8_t b = w.as<uint8_t>()[i / 2];
    const int v = (i % 2 == 0) ? (b & 15) : (b >> 4);
    return v >= 8 ? v - 16 : v;
}

void set_s4(const buffer_t &w, size_t i, int v) {
    uint8_t &b = w.as<uint8_t>()[i / 2];
    const uint8_t nibble = static_cast<uint8_t>(v) & 15u;
    b = static_cast<uint8_t>(
            i % 2 == 0 ? (b & 0xF0u) | nibble : (b & 0x0Fu) | (nibble << 4));
}

/// The operands of a model case, made into products of one sign in each
/// column (positive in even columns, negative in odd ones), none small: no
/// column's sum cancels, so the bound of model_of is a small part of its
/// value, and no step of the kernels leaves the normal range of its type.
/// The dst starts with the other sign in each column: an unwritten column
/// is off by more than its value, as a zeroed one is off by its value.
void make_same_sign(const gcase_t &c, operands_t &o) {
    const auto negative = [](int n) { return (n & 1) != 0; };
    const auto wei_index = [&](int k, int n) {
        return c.transB ? static_cast<size_t>(n) * c.K + k
                        : static_cast<size_t>(k) * c.N + n;
    };
    // Floating forms: magnitudes in [0.25, 1].
    const auto lift = [](float v) {
        return std::copysign(0.25f + 0.75f * std::fabs(v), v);
    };
    // Integer forms: the source less its zero point and the weight of
    // magnitude 1 at least.
    const int32_t zp
            = o.src_zp.get() != nullptr ? o.src_zp.as<int32_t>()[0] : 0;
    for (int k = 0; k < c.K; ++k) {
        bool flip = false;
        if (int8_form(c.form)) {
            int a = static_cast<int>(nt::load_f32(o.src.get(), o.src_dt, k));
            if (a == zp) {
                a = zp + 1;
                nt::store_f32(o.src.get(), o.src_dt, k, static_cast<float>(a));
            }
            flip = a < zp;
        } else {
            nt::store_f32(o.src.get(), o.src_dt, k,
                    lift(nt::load_f32(o.src.get(), o.src_dt, k)));
            flip = nt::load_f32(o.src.get(), o.src_dt, k) < 0.0f;
        }
        for (int n = 0; n < c.N; ++n) {
            const size_t i = wei_index(k, n);
            const bool neg = flip != negative(n);
            if (o.wei_dt == kS4) {
                const int m = std::max(1, std::abs(s4_at(o.wei, i)));
                set_s4(o.wei, i, neg ? -m : m);
            } else if (int8_form(c.form)) {
                const float m = std::max(
                        1.0f, std::fabs(nt::load_f32(o.wei.get(), kS8, i)));
                nt::store_f32(o.wei.get(), kS8, i, neg ? -m : m);
            } else {
                const float m = std::fabs(
                        lift(nt::load_f32(o.wei.get(), o.wei_dt, i)));
                nt::store_f32(o.wei.get(), o.wei_dt, i, neg ? -m : m);
            }
        }
    }
    for (int n = 0; n < c.out_cols(); ++n) {
        const float v = std::fabs(nt::load_f32(o.init.get(), c.dst, n));
        nt::store_f32(o.init.get(), c.dst, n, negative(n) ? v : -v);
    }
}

/// Unit roundoffs of rounding to nearest in f32 and in f16.
constexpr double kF32Unit = 1.0 / (1 << 24);
constexpr double kF16Unit = 1.0 / 2048;

/// The bound of n roundings to nearest of unit roundoff u on a sum, as a
/// part of the magnitudes of its terms: n u / (1 - n u).
double gamma_of(int n, double u) {
    return n * u / (1.0 - n * u);
}

/// The f64 value of output column n of a model case (no post-ops, alpha,
/// beta, bias or gated activation) from the operands as the call reads them,
/// and the bound of the call's error.
struct model_t {
    double ref = 0.0;
    double bound = 0.0;
};

/// The bound follows the kernel's accumulation; its products are exact (the
/// kernels multiply inside fused multiply-adds, or bf16 / f16 values whose
/// products f32 holds) and each step rounds to nearest.  bf16, f32 and the
/// widening f16 kernels: one f32 chain over K.  f16_ph: f16 chains of
/// kNativeF16AccumBlockK steps from zero, whose sums an f32 chain adds.  INT8
/// and s4: exact int32 sums, one per group the kernel finishes (one over K
/// with neither weight nor source groups), each group's term rounded at most
/// three times in f32 (the conversion and the scale products) and once per
/// group as the terms add up.  Then the dst's rounding, and one f32 rounding
/// of the magnitudes for the model's own f64 roundings.
model_t model_of(const gcase_t &c, const operands_t &o, int n, bool f16_ph) {
    const auto wei_at = [&](int k) -> double {
        const size_t i = c.transB ? static_cast<size_t>(n) * c.K + k
                                  : static_cast<size_t>(k) * c.N + n;
        if (o.wei_dt == kS4) return s4_at(o.wei, i);
        return nt::load_f32(o.wei.get(), o.wei_dt, i);
    };
    model_t m;
    double mag = 0.0; // the magnitudes of the terms
    double err = 0.0;
    if (!int8_form(c.form)) {
        const bool ph = c.form == form_t::f16 && f16_ph;
        const int block = ph ? nat::kNativeF16AccumBlockK : c.K;
        const double u = ph ? kF16Unit : kF32Unit;
        double sums = 0.0; // the magnitudes of the blocks' rounded sums
        int blocks = 0;
        for (int k0 = 0; k0 < c.K; k0 += block, ++blocks) {
            const int kb = std::min(block, c.K - k0);
            double part = 0.0;
            for (int k = k0; k < k0 + kb; ++k) {
                const double p = static_cast<double>(
                                         nt::load_f32(o.src.get(), o.src_dt, k))
                        * wei_at(k);
                m.ref += p;
                part += std::fabs(p);
            }
            mag += part;
            err += gamma_of(kb, u) * part;
            sums += (1.0 + gamma_of(kb, u)) * part;
        }
        if (ph) err += gamma_of(blocks, kF32Unit) * sums;
    } else {
        const bool wgroups = c.form == form_t::grp_s8
                || c.form == form_t::grp_u8 || s4_form(c.form);
        const bool sgroups = o.params.quant_params.src_scale.dims.back() > 1;
        const int G = wgroups || sgroups ? c.groups() : 1;
        const int gk = c.K / G;
        const int32_t zp
                = o.src_zp.get() != nullptr ? o.src_zp.as<int32_t>()[0] : 0;
        for (int g = 0; g < G; ++g) {
            int64_t sum = 0;
            for (int k = g * gk; k < (g + 1) * gk; ++k)
                sum += (static_cast<int64_t>(
                                nt::load_f32(o.src.get(), o.src_dt, k))
                               - zp)
                        * static_cast<int64_t>(wei_at(k));
            const double ss
                    = nt::load_f32(o.src_scale.get(), kF32, sgroups ? g : 0);
            const double ws = nt::load_f32(o.wei_scale.get(), c.wscale,
                    static_cast<size_t>(wgroups ? g : 0) * c.N + n);
            const double t = static_cast<double>(sum) * ss * ws;
            m.ref += t;
            mag += std::fabs(t);
        }
        err = gamma_of(G + 3, kF32Unit) * mag;
    }
    const double dst_u = c.dst == kBf16 ? 1.0 / 256
            : c.dst == kF16             ? kF16Unit
                                        : 0.0;
    m.bound = err + dst_u * (std::fabs(m.ref) + err) + kF32Unit * mag;
    return m;
}

class NativeGemvModel : public ::testing::TestWithParam<gcase_t> {};

// M = 1 calls on the GEMV kernels against the f64 model, within the bound of
// the kernel's accumulation, over products of one sign in each column; in
// every column the bound refuses a zeroed and an unwritten dst.
TEST_P(NativeGemvModel, MatchesAnF64Model) {
    const gcase_t &c = GetParam();
    if (!cpu_serves(c.form)) GTEST_SKIP() << "the CPU lacks the kernels' ISA";
    const nt::algo_pin_t algo(matmul_algo_t::native_brgemm);
    operands_t o = make_operands(c);
    make_same_sign(c, o);
    const buffer_t wei = nt::kept_until_exit(copy_of(o.wei));
    const buffer_t dst = copy_of(o.init);
    ASSERT_EQ(nt::run_matmul(make_call(c, o, wei.get(), dst.get())),
            status_t::success);
    const bool ph = nat::native_kernel_registry().f16_kind()
            == nat::native_kernel_kind_t::f16_ph;
    int bad = 0, weak = 0;
    for (int n = 0; n < c.N; ++n) {
        const model_t m = model_of(c, o, n, ph);
        const auto takes
                = [&](double v) { return std::fabs(v - m.ref) <= m.bound; };
        if ((takes(0.0) || takes(nt::load_f32(o.init.get(), c.dst, n)))
                && weak++ < 5)
            ADD_FAILURE() << "column " << n << ": the bound " << m.bound
                          << " of " << m.ref
                          << " takes a zeroed or an unwritten dst";
        const double got = nt::load_f32(dst.get(), c.dst, n);
        if (!takes(got) && bad++ < 5)
            ADD_FAILURE() << "column " << n << ": " << got << " against "
                          << m.ref << " (bound " << m.bound << ")";
    }
    EXPECT_EQ(weak, 0);
    EXPECT_EQ(bad, 0);
}

std::vector<gcase_t> model_cases() {
    std::vector<gcase_t> v;
    for (const form_t f : {form_t::bf16, form_t::f32, form_t::f16,
                 form_t::st_s8, form_t::st_u8, form_t::grp_s8, form_t::grp_u8,
                 form_t::s4, form_t::s4_grp, form_t::s4_one}) {
        for (const int N : {48, 512}) {
            for (const bool tb : {false, true}) {
                gcase_t c;
                c.form = f;
                c.N = N;
                c.K = int8_form(f) ? 512 : 515;
                c.transB = tb;
                c.dst = f == form_t::f16 ? kF16 : kF32;
                c.threads = N == 512 ? 8 : 1;
                c.wc = tb ? 2 : 1;
                c.wscale = kF32;
                v.push_back(c);
            }
        }
    }
    // An f16 source of K 4096, whose rows the widening kernels pad to 4112
    // floats, and one of K 4113, whose rows they do not pad.
    for (const int K : {4096, 4113}) {
        gcase_t c;
        c.form = form_t::f16;
        c.N = 48;
        c.K = K;
        c.dst = kF16;
        v.push_back(c);
    }
    return named(v);
}

INSTANTIATE_TEST_SUITE_P(Native, NativeGemvModel,
        ::testing::ValuesIn(model_cases()),
        [](const ::testing::TestParamInfo<gcase_t> &i) {
            return i.param.name;
        });

// ── The plan line ────────────────────────────────────────────────────────

/// The plan line of a call's log, without its gemv field and its team, and
/// the field's value.
std::string plan_line(const std::string &log, std::string &gemv) {
    const size_t at = log.find("[MATMUL.NATIVE_BRGEMM] M=");
    if (at == std::string::npos) return std::string();
    std::string line = log.substr(at, log.find('\n', at) - at);
    const size_t g = line.find(" gemv=");
    if (g == std::string::npos) return std::string();
    const size_t end = line.find(' ', g + 1);
    gemv = line.substr(g + 6, end - g - 6);
    line.erase(g, end - g);
    return line;
}

// The plan line of an M = 1 call names its GEMV kernels in its gemv field,
// and every other field is the one the same call on the tiles logs; a call
// of M 2 logs gemv=none.  Under ZENDNNL_API_LOG_LEVEL=4, read once per
// process: in a child.
TEST(NativeGemv, PlanLineNamesTheGemvKernels) {
    if (!cpu_serves(form_t::bf16) || !cpu_serves(form_t::st_s8))
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512";
    auto body = [] {
        const bool ph = nat::native_kernel_registry().f16_kind()
                == nat::native_kernel_kind_t::f16_ph;
        const std::vector<std::pair<form_t, const char *>> kinds
                = {{form_t::bf16, "bf16"}, {form_t::f32, "f32"},
                        {form_t::f16, ph ? "f16_ph" : "f16"},
                        {form_t::st_u8, "s8"}, {form_t::dyn_s8, "s8"},
                        {form_t::grp_s8, "s8_group"}, {form_t::s4, "s4"}};
        int bad = 0;
        for (const auto &kind : kinds) {
            gcase_t c;
            c.form = kind.first;
            c.N = 256;
            c.K = 128;
            c.threads = 4;
            c.dst = c.form == form_t::f16   ? kF16
                    : c.form == form_t::f32 ? kF32
                                            : kBf16;
            c.name = std::string("plan_") + form_name(c.form);
            operands_t o = make_operands(c);
            o.params.lowoha_algo = matmul_algo_t::native_brgemm;
            const buffer_t wei = nt::kept_until_exit(copy_of(o.wei));
            const buffer_t dst = copy_of(o.init);
            std::string gemv, tiles_gemv;
            ::testing::internal::CaptureStdout();
            const status_t st
                    = nt::run_matmul(make_call(c, o, wei.get(), dst.get()));
            const std::string line
                    = plan_line(::testing::internal::GetCapturedStdout(), gemv);
            ::testing::internal::CaptureStdout();
            const status_t ts
                    = run_tiles(make_call(c, o, wei.get(), dst.get()));
            const std::string tiles = plan_line(
                    ::testing::internal::GetCapturedStdout(), tiles_gemv);
            if (st != status_t::success || ts != status_t::success
                    || gemv != kind.second || tiles_gemv != "none"
                    || line.empty() || line != tiles) {
                std::fprintf(stderr,
                        "%s: status %d / %d, gemv=%s (expected %s), tiles "
                        "gemv=%s\n  gemv  %s\n  tiles %s\n",
                        form_name(c.form), static_cast<int>(st),
                        static_cast<int>(ts), gemv.c_str(), kind.second,
                        tiles_gemv.c_str(), line.c_str(), tiles.c_str());
                ++bad;
            }
            // M 2 on the same weight runs on the tiles (a form of per-group
            // source scales takes a scale per row: not here).
            if (c.form == form_t::grp_s8) continue;
            nt::call_t two = make_call(c, o, wei.get(), nullptr);
            const size_t row
                    = static_cast<size_t>(c.K) * nt::dt_bytes(o.src_dt);
            const buffer_t src2 = nt::make_buffer(2 * row);
            std::memcpy(src2.as<uint8_t>(), o.src.get(), row);
            std::memcpy(src2.as<uint8_t>() + row, o.src.get(), row);
            const buffer_t dst2 = nt::make_buffer(
                    static_cast<size_t>(2) * c.ldc() * nt::dt_bytes(c.dst));
            two.M = 2;
            two.src = src2.get();
            two.dst = dst2.get();
            const buffer_t scales2
                    = s4_form(c.form) ? row_scales(o, 2) : buffer_t {};
            if (s4_form(c.form)) {
                two.params.quant_params.src_scale.buff = scales2.get();
                two.params.quant_params.src_scale.dims = {2, 1};
            }
            ::testing::internal::CaptureStdout();
            const status_t s2 = nt::run_matmul(two);
            plan_line(::testing::internal::GetCapturedStdout(), gemv);
            if (s2 != status_t::success || gemv != "none") {
                std::fprintf(stderr, "%s M 2: status %d, gemv=%s\n",
                        form_name(c.form), static_cast<int>(s2), gemv.c_str());
                ++bad;
            }
        }
        return bad;
    };
    nt::expect_in_child(
            {{"ZENDNNL_API_LOG_LEVEL", "4"}}, body, "ZENDNNL_API_LOG_LEVEL=4");
}

/// The gemv field of the plan line of a form's M = 1 calls.
const char *gemv_kind(form_t f, bool f16_ph) {
    switch (f) {
        case form_t::bf16: return "bf16";
        case form_t::f32: return "f32";
        case form_t::f16: return f16_ph ? "f16_ph" : "f16";
        case form_t::st_s8:
        case form_t::st_u8:
        case form_t::dyn_s8:
        case form_t::dyn_u8: return "s8";
        case form_t::grp_s8:
        case form_t::grp_u8:
        case form_t::after: return "s8_group";
        default: return "s4";
    }
}

/// The value of field name of the plan line of an M = 1 call in log, empty
/// without such a line.
std::string m1_field(const std::string &log, const char *name) {
    const size_t at = log.find("[MATMUL.NATIVE_BRGEMM] M=1 ");
    if (at == std::string::npos) return std::string();
    const std::string line = log.substr(at, log.find('\n', at) - at);
    const std::string key = std::string(" ") + name + "=";
    const size_t f = line.find(key);
    if (f == std::string::npos) return std::string();
    const size_t v = f + key.size();
    return line.substr(v, line.find(' ', v) - v);
}

// Each M = 1 call of every bit case runs on the GEMV kernels of its type,
// which the gemv field of its plan line names, and on a team where the case
// says so (its team field): through native_brgemm_execute and, where its
// checks take the case, through matmul_direct twice on one weight.  A call
// that ran on the tiles instead gives the tiles' bits, which EqualTheTiles
// cannot tell from the GEMV kernels'.  Under ZENDNNL_API_LOG_LEVEL=4, read
// once per process: in a child.
TEST(NativeGemv, EveryBitCaseRunsItsGemvKernels) {
    if (!nat::native_base_isa()) GTEST_SKIP() << "needs AVX-512 and FMA";
    auto body = [] {
        const WeightCacheGuard mode(2);
        const nt::algo_pin_t algo(matmul_algo_t::native_brgemm);
        const bool ph = nat::native_kernel_registry().f16_kind()
                == nat::native_kernel_kind_t::f16_ph;
        int bad = 0, checked = 0;
        for (const gcase_t &c : bit_cases()) {
            if (!cpu_serves(c.form)) continue;
            const operands_t o = make_operands(c);
            const buffer_t raw = copy_of(o.wei);
            const buffer_t wei = nt::kept_until_exit(copy_of(o.wei));
            const int calls = matmul_direct_takes(c) ? 3 : 1;
            for (int call = 0; call < calls; ++call, ++checked) {
                const buffer_t dst = copy_of(o.init);
                ::testing::internal::CaptureStdout();
                const status_t st = call == 0
                        ? run_tiles(make_call(c, o, raw.get(), dst.get()), true)
                        : nt::run_matmul(make_call(c, o, wei.get(), dst.get()));
                const std::string log
                        = ::testing::internal::GetCapturedStdout();
                const std::string gemv = m1_field(log, "gemv");
                const int team = std::atoi(m1_field(log, "team").c_str());
                if (st == status_t::success && gemv == gemv_kind(c.form, ph)
                        && (!c.team || team > 1))
                    continue;
                if (bad++ < 20)
                    std::fprintf(stderr,
                            "%s, %s: status %d, gemv=%s, team=%d (expected "
                            "%s%s)\n",
                            c.name.c_str(),
                            call == 0 ? "native_brgemm_execute"
                                      : "matmul_direct",
                            static_cast<int>(st), gemv.c_str(), team,
                            gemv_kind(c.form, ph), c.team ? " on a team" : "");
            }
        }
        if (bad)
            std::fprintf(stderr,
                    "%d of %d M = 1 calls did not run on the GEMV kernels of "
                    "their type\n",
                    bad, checked);
        return bad;
    };
    nt::expect_in_child(
            {{"ZENDNNL_API_LOG_LEVEL", "4"}}, body, "ZENDNNL_API_LOG_LEVEL=4");
}

// ── One layout per weight ────────────────────────────────────────────────

// A weight that M = 1 calls and calls of M 7 share, in either order, at
// weight cache modes 1 and 2: it keeps one layout (the cached one, or its
// image), and every call gives the tiles' bits.
TEST(NativeGemv, OneLayoutServesM1AndLargerCalls) {
    if (!cpu_serves(form_t::bf16) || !cpu_serves(form_t::st_s8))
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512";
    const WeightCacheGuard mode(2);
    const nt::algo_pin_t algo(matmul_algo_t::native_brgemm);
    for (const form_t f :
            {form_t::bf16, form_t::dyn_s8, form_t::s4, form_t::f16}) {
        for (const int wc : {1, 2}) {
            for (const bool m1_first : {true, false}) {
                gcase_t c;
                c.form = f;
                c.N = 320;
                c.K = 128;
                c.wc = wc;
                c.threads = 4;
                c.dst = f == form_t::f16 ? kF16 : kBf16;
                c.name = std::string("one_") + form_name(f) + "_"
                        + std::to_string(wc) + (m1_first ? "a" : "b");
                const operands_t o = make_operands(c);
                const buffer_t raw = copy_of(o.wei);
                const buffer_t wei = nt::kept_until_exit(copy_of(o.wei));
                // Seven rows: the M = 1 call's source and six others.
                const int M = 7;
                const size_t esz = nt::dt_bytes(o.src_dt);
                buffer_t src7 = nt::random_buffer(o.src_dt,
                        static_cast<size_t>(M) * c.K, seed_of(c.name) + 9,
                        o.src_dt == kU8 ? 64.0f : -1.0f,
                        o.src_dt == kU8 ? 192.0f : 1.0f);
                std::memcpy(src7.get(), o.src.get(), c.K * esz);
                for (int step = 0; step < 2; ++step) {
                    const bool m1 = (step == 0) == m1_first;
                    nt::call_t call = make_call(c, o, wei.get(), nullptr);
                    nt::call_t ref = make_call(c, o, raw.get(), nullptr);
                    const int rows = m1 ? 1 : M;
                    call.M = ref.M = rows;
                    call.src = ref.src = m1 ? o.src.get() : src7.get();
                    if (!m1 && f == form_t::dyn_s8) {
                        call.params.quant_params.src_scale.dims = {M, 1};
                        ref.params.quant_params.src_scale.dims = {M, 1};
                    }
                    buffer_t s7a = nt::make_buffer(static_cast<size_t>(M) * 4);
                    buffer_t s7b = nt::make_buffer(static_cast<size_t>(M) * 4);
                    if (f == form_t::dyn_s8) {
                        call.params.quant_params.src_scale.buff = s7a.get();
                        ref.params.quant_params.src_scale.buff = s7b.get();
                    }
                    const buffer_t s4_rows = f == form_t::s4
                            ? row_scales(o, rows)
                            : buffer_t {};
                    if (f == form_t::s4) {
                        for (nt::call_t *x : {&call, &ref}) {
                            x->params.quant_params.src_scale.buff
                                    = s4_rows.get();
                            x->params.quant_params.src_scale.dims = {rows, 1};
                        }
                    }
                    const size_t bytes = static_cast<size_t>(rows) * c.ldc()
                            * nt::dt_bytes(c.dst);
                    const buffer_t got = nt::make_buffer(bytes);
                    const buffer_t want = nt::make_buffer(bytes);
                    call.dst = got.get();
                    ref.dst = want.get();
                    ASSERT_EQ(nt::run_matmul(call), status_t::success)
                            << c.name << " step " << step;
                    ASSERT_EQ(run_tiles(ref), status_t::success)
                            << c.name << " step " << step;
                    EXPECT_EQ(std::memcmp(got.get(), want.get(), bytes), 0)
                            << c.name << " step " << step << " (M " << rows
                            << ")";
                    EXPECT_EQ(nt::native_layouts(wei.get()), 1u)
                            << c.name << " step " << step;
                }
            }
        }
    }
}

// ── The widened f16 source ───────────────────────────────────────────────

// An M = 1 f16 call on the widening kernels leaves a kept buffer of exactly
// its widened row rounded up to 64 bytes: 4112 floats at K 4096, 4113 at K
// 4113.  The GEMV kernels give the tiles' bits.  In a child with
// ZENDNNL_MATMUL_NATIVE_F16_ACCUM=f32, which keeps f16 calls on the widening
// kernels on every CPU.
TEST(NativeGemv, WidenedF16SourceKeepsItsRow) {
    if (!cpu_serves(form_t::f16)) GTEST_SKIP() << "needs AVX-512";
    auto body = [] {
        const nt::algo_pin_t algo(matmul_algo_t::native_brgemm);
        if (nat::native_kernel_registry().f16_kind()
                != nat::native_kernel_kind_t::f16) {
            std::fprintf(stderr, "f16 calls are not on the widening kernels\n");
            return 1;
        }
        int bad = 0;
        const std::pair<int, size_t> sizes[]
                = {{4096, 4112 * 4}, {4113, (4113 * 4 + 63) & ~63}};
        for (const auto &[K, kept] : sizes) {
            gcase_t c;
            c.form = form_t::f16;
            c.N = 64;
            c.K = K;
            c.dst = kF16;
            c.name = "widened_k" + std::to_string(K);
            const operands_t o = make_operands(c);
            const buffer_t wei = nt::kept_until_exit(copy_of(o.wei));
            const buffer_t tiles = copy_of(o.init);
            const status_t st0
                    = run_tiles(make_call(c, o, wei.get(), tiles.get()));
            nat::release_native_f16_sources();
            const buffer_t gemv = copy_of(o.init);
            const status_t st1
                    = nt::run_matmul(make_call(c, o, wei.get(), gemv.get()));
            const size_t after = nat::native_f16_kept_source_bytes();
            const bool bits
                    = std::memcmp(gemv.get(), tiles.get(), dst_bytes(c)) == 0;
            if (st0 != status_t::success || st1 != status_t::success || !bits
                    || after != kept) {
                std::fprintf(stderr,
                        "%s: status %d / %d, kept %zu (expected %zu), %s "
                        "bits\n",
                        c.name.c_str(), static_cast<int>(st0),
                        static_cast<int>(st1), after, kept,
                        bits ? "the tiles'" : "not the tiles'");
                ++bad;
            }
        }
        nat::release_native_f16_sources();
        return bad;
    };
    nt::expect_in_child({{"ZENDNNL_MATMUL_NATIVE_F16_ACCUM", "f32"}}, body,
            "ZENDNNL_MATMUL_NATIVE_F16_ACCUM=f32");
}

} // namespace
