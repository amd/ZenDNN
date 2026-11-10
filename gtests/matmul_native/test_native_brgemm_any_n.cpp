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

// The native BRGEMM over weights of any N: a weight whose N is not a
// multiple of the 32-column blocks is held padded with zero columns, its
// tiles run on the padded width, and its last column tile stores only the
// columns below N.  Every call runs on operands that end where unreadable
// memory starts, into a destination whose columns past the output (the ldc
// padding) hold sentinels: a read or a write past the end of an operand
// faults, and a write past the output columns shows.  The results
// are compared with the reference algorithm (or, for per-group weight
// scales over a u8 source, which it does not apply per group, with an f64
// model) within each data type's band.  The weight is never rewritten.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#endif

#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/backends/reference/reference_kernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
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
constexpr data_type_t kS8 = data_type_t::s8;
constexpr data_type_t kU8 = data_type_t::u8;
constexpr data_type_t kS32 = data_type_t::s32;
constexpr data_type_t kNone = data_type_t::none;

/// Storage that ends where an unreadable page starts: an access past its end
/// faults.  An unreadable page also precedes its span of whole pages, which
/// the storage starts at only when its size is a whole number of pages.
class guarded_t {
public:
    explicit guarded_t(size_t bytes) : bytes_(bytes) {
#ifndef _WIN32
        page_ = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        span_ = (std::max<size_t>(bytes, 1) + page_ - 1) / page_ * page_;
        void *m = mmap(nullptr, span_ + 2 * page_, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) return;
        base_ = static_cast<uint8_t *>(m);
        if (mprotect(base_, page_, PROT_NONE) != 0
                || mprotect(base_ + page_ + span_, page_, PROT_NONE) != 0)
            return;
        data_ = base_ + page_ + span_ - bytes;
#else
        heap_ = nt::make_buffer(bytes);
        data_ = heap_.as<uint8_t>();
#endif
    }
    ~guarded_t() {
#ifndef _WIN32
        if (base_ != nullptr) munmap(base_, span_ + 2 * page_);
#endif
    }
    guarded_t(const guarded_t &) = delete;
    guarded_t &operator=(const guarded_t &) = delete;
    uint8_t *get() const { return data_; }
    size_t bytes() const { return bytes_; }

private:
    size_t bytes_ = 0;
    uint8_t *data_ = nullptr;
#ifndef _WIN32
    size_t page_ = 0;
    size_t span_ = 0;
    uint8_t *base_ = nullptr;
#else
    buffer_t heap_;
#endif
};

/// A guarded copy of b (nullptr when b holds nothing).
std::unique_ptr<guarded_t> guarded_copy(const buffer_t &b) {
    if (b.get() == nullptr || b.bytes == 0) return nullptr;
    auto g = std::make_unique<guarded_t>(b.bytes);
    if (g->get() != nullptr) std::memcpy(g->get(), b.get(), b.bytes);
    return g;
}

/// The data types of a call: bf16; f32; an s8 source with a per-tensor scale
/// or a u8 source with a per-tensor scale and zero point over per-channel s8
/// weights; a bf16 source quantized per token to s8, or to u8 with a zero
/// point; per-group s8 weights over an s8 source with {M, G} scales, or over
/// a u8 source with a per-tensor scale and zero point; and per-channel s8
/// weights over an s8 source with {M, G} scales, whose weight scales apply
/// after the groups.
enum class q_t {
    bf16,
    f32,
    st_s8,
    st_u8,
    dyn_s8,
    dyn_u8,
    grp_s8,
    grp_u8,
    after
};

/// How the call reaches the native BRGEMM: native_brgemm_execute on the
/// weight reference matmul_direct builds; matmul_direct under
/// ZENDNNL_MATMUL_ALGO=10; or matmul_direct with lowoha_algo = native_brgemm.
enum class route_t { execute, algo10, per_call };

/// The group size of per-group weight or source scales.
constexpr int kGroup = 32;

struct ncase_t {
    std::string name;
    q_t q = q_t::bf16;
    route_t route = route_t::execute;
    int M = 1, N = 1, K = 64;
    data_type_t dst = kBf16;
    bool transB = false;
    int threads = 1; ///< 0: every thread
    float alpha = 1.0f, beta = 0.0f;
    data_type_t bias = kNone;
    post_op_type_t po = post_op_type_t::none;
    data_type_t po_dt = kF32; ///< a binary operand's type
    bool po_row = false; ///< a binary operand of [1, N], else [M, N]
    gated_t gated = gated_t::none;
    int wc = 1; ///< weight cache mode of the call
    int pad = 0; ///< ldc columns past the output (past N for gated beta)
    int calls = 1; ///< calls on the same weight

    bool int8() const { return q != q_t::bf16 && q != q_t::f32; }
    bool groups() const {
        return q == q_t::grp_s8 || q == q_t::grp_u8 || q == q_t::after;
    }
    int out_cols() const { return gated == gated_t::none ? N : N / 2; }
    /// beta of a gated call reads the dst's N pre-activation columns.
    int ldc() const {
        return (gated != gated_t::none && beta != 0.0f ? N : out_cols()) + pad;
    }
};

void PrintTo(const ncase_t &c, std::ostream *os) {
    *os << c.name;
}

const char *dt_name(data_type_t dt) {
    switch (dt) {
        case kF32: return "f32";
        case kBf16: return "bf16";
        case kS8: return "s8";
        case kU8: return "u8";
        case kS32: return "s32";
        default: return "none";
    }
}

std::string describe(const ncase_t &c) {
    static const char *const kQ[] = {"bf16", "f32", "sts8", "stu8", "dyns8",
            "dynu8", "grps8", "grpu8", "after"};
    static const char *const kRoute[] = {"exec", "a10", "percall"};
    std::string s = std::string(kRoute[static_cast<int>(c.route)]) + "_"
            + kQ[static_cast<int>(c.q)] + "_m" + std::to_string(c.M) + "_n"
            + std::to_string(c.N) + "_k" + std::to_string(c.K) + "_"
            + dt_name(c.dst) + "_t" + std::to_string(c.threads) + "_wc"
            + std::to_string(c.wc);
    if (c.transB) s += "_tb";
    if (c.pad) s += "_pad" + std::to_string(c.pad);
    if (c.alpha != 1.0f || c.beta != 0.0f)
        s += "_a" + std::to_string(static_cast<int>(c.alpha * 100)) + "b"
                + std::to_string(static_cast<int>(c.beta * 100));
    if (c.bias != kNone) s += std::string("_bias") + dt_name(c.bias);
    switch (c.po) {
        case post_op_type_t::relu: s += "_relu"; break;
        case post_op_type_t::gelu_erf: s += "_gelu"; break;
        case post_op_type_t::tanh: s += "_tanh"; break;
        case post_op_type_t::softmax: s += "_softmax"; break;
        case post_op_type_t::binary_add:
        case post_op_type_t::binary_mul:
            s += std::string(
                         c.po == post_op_type_t::binary_add ? "_add" : "_mul")
                    + dt_name(c.po_dt) + (c.po_row ? "row" : "full");
            break;
        default: break;
    }
    if (c.gated == gated_t::silu_and_mul) s += "_silu";
    if (c.gated == gated_t::gelu_and_mul) s += "_geluand";
    if (c.gated == gated_t::swiglu_oai_mul) s += "_swiglu";
    if (c.calls != 1) s += "_x" + std::to_string(c.calls);
    for (char &ch : s)
        if (ch == '-') ch = 'm';
    return s;
}

std::vector<ncase_t> named(std::vector<ncase_t> v) {
    std::vector<std::string> seen;
    for (ncase_t &c : v) {
        std::string n = describe(c);
        int dup = 0;
        for (const std::string &s : seen)
            dup += s == n || s.rfind(n + "_dup", 0) == 0;
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

/// The operands of a case in ordinary buffers, and its params pointing at
/// them.
struct operands_t {
    data_type_t src_dt = kBf16, wei_dt = kBf16;
    int lda = 0, ldb = 0;
    buffer_t src, wei, bias, wei_scale, src_scale, src_zp, dst_scale, dst_zp,
            po_buf, init;
    mm::matmul_params params;
};

operands_t make_operands(const ncase_t &c) {
    operands_t o;
    const uint32_t seed = seed_of(c.name);
    const bool st_s8
            = c.q == q_t::st_s8 || c.q == q_t::grp_s8 || c.q == q_t::after;
    const bool st_u8 = c.q == q_t::st_u8 || c.q == q_t::grp_u8;
    const bool dyn = c.q == q_t::dyn_s8 || c.q == q_t::dyn_u8;
    o.src_dt = st_s8 ? kS8 : (st_u8 ? kU8 : (c.q == q_t::f32 ? kF32 : kBf16));
    o.wei_dt = c.int8() ? kS8 : (c.q == q_t::f32 ? kF32 : kBf16);
    o.lda = c.K;
    o.ldb = c.transB ? c.K : c.N;
    const size_t a_n = static_cast<size_t>(c.M) * c.K;
    const size_t b_n = static_cast<size_t>(c.K) * c.N;
    if (o.src_dt == kS8)
        o.src = nt::random_buffer(kS8, a_n, seed, -64, 64);
    else if (o.src_dt == kU8)
        o.src = nt::random_buffer(kU8, a_n, seed, 64, 192);
    else
        o.src = nt::random_buffer(o.src_dt, a_n, seed, -1.0f, 1.0f);
    o.wei = c.int8() ? nt::random_buffer(kS8, b_n, seed + 1, -127, 127)
                     : nt::random_buffer(o.wei_dt, b_n, seed + 1, -1.0f, 1.0f);
    if (c.bias != kNone)
        o.bias = nt::random_buffer(c.bias, c.N, seed + 2, -1.0f, 1.0f);

    mm::matmul_params &p = o.params;
    p.dtypes.src = o.src_dt;
    p.dtypes.wei = o.wei_dt;
    p.dtypes.dst = c.dst;
    p.dtypes.bias = c.bias;
    p.num_threads = c.threads;
    p.weight_cache_type = c.wc;
    auto &qp = p.quant_params;
    if (c.int8()) {
        const int G = c.K / kGroup;
        if (c.q == q_t::grp_s8 || c.q == q_t::grp_u8) {
            o.wei_scale = nt::random_buffer(kF32, static_cast<size_t>(G) * c.N,
                    seed + 3, 0.002f, 0.01f);
            set_quant(qp.wei_scale, o.wei_scale, kF32, {G, c.N});
        } else {
            o.wei_scale = nt::random_buffer(kF32, c.N, seed + 3, 0.002f, 0.01f);
            set_quant(qp.wei_scale, o.wei_scale, kF32, {1, c.N});
        }
        if (dyn) {
            p.dynamic_quant = true;
            p.dtypes.compute = c.q == q_t::dyn_s8 ? kS8 : kU8;
            o.src_scale = nt::make_buffer(static_cast<size_t>(c.M) * 4);
            set_quant(qp.src_scale, o.src_scale, kF32, {c.M, 1});
            if (c.q == q_t::dyn_u8) {
                o.src_zp = nt::make_buffer(static_cast<size_t>(c.M) * 4);
                set_quant(qp.src_zp, o.src_zp, kS32, {c.M, 1});
            }
        } else if (c.q == q_t::grp_s8 || c.q == q_t::after) {
            o.src_scale = nt::random_buffer(kF32, static_cast<size_t>(c.M) * G,
                    seed + 4, 0.005f, 0.02f);
            set_quant(qp.src_scale, o.src_scale, kF32, {c.M, G});
        } else {
            o.src_scale = nt::random_buffer(kF32, 1, seed + 4, 0.005f, 0.02f);
            set_quant(qp.src_scale, o.src_scale, kF32, {1, 1});
            if (st_u8) {
                o.src_zp = nt::random_buffer(kS32, 1, seed + 5, 100, 156);
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
            const int rows = c.po_row ? 1 : c.M;
            o.po_buf = nt::random_buffer(c.po_dt,
                    static_cast<size_t>(rows) * c.out_cols(), seed + 6, -1.0f,
                    1.0f);
            m.buff = o.po_buf.get();
            m.dtype = c.po_dt;
            m.dims = {rows, c.out_cols()};
        }
        p.postop_.push_back(m);
    }
    o.init = nt::random_buffer(c.dst, static_cast<size_t>(c.M) * c.ldc(),
            seed + 7, c.dst == kU8 ? 0.0f : -2.0f,
            c.dst == kU8 ? 255.0f : 2.0f);
    return o;
}

nt::call_t make_call(const ncase_t &c, const operands_t &o) {
    nt::call_t call;
    call.M = c.M;
    call.N = c.N;
    call.K = c.K;
    call.transB = c.transB;
    call.lda = o.lda;
    call.ldb = o.ldb;
    call.ldc = c.ldc();
    call.alpha = c.alpha;
    call.beta = c.beta;
    call.src = o.src.get();
    call.wei = o.wei.get();
    call.bias = o.bias.get();
    call.params = o.params;
    call.gated = c.gated;
    return call;
}

/// The call through native_brgemm_execute with the descriptor and weight
/// reference native_layout_matmul builds for it (no GEMV choice: the tiles).
status_t run_execute(const nt::call_t &c) {
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
    nat::native_weight_ref_t ref;
    ref.weight = c.wei;
    ref.pack_nr = nat::kNativeLayoutPackNr;
    ref.weight_cache_mode = c.params.weight_cache_type;
    return nat::native_brgemm_execute(d, ref, c.src, c.dst, c.bias, c.params);
}

/// The f64 value of element (m, n) of a call of per-group weight scales over
/// a u8 source with a per-tensor scale and zero point (no alpha, beta or
/// post-op other than relu), with the bound of its f32 roundings.
void grp_u8_at(const ncase_t &c, const operands_t &o, int m, int n,
        double &value, double &bound) {
    const auto *a = o.src.as<uint8_t>();
    const auto *w = o.wei.as<int8_t>();
    const float *ws = o.wei_scale.as<float>();
    const double sa = o.src_scale.as<float>()[0];
    const int32_t zp = o.src_zp.as<int32_t>()[0];
    double sum = 0.0, mag = 0.0;
    for (int g = 0; g < c.K / kGroup; ++g) {
        int64_t s = 0;
        for (int k = g * kGroup; k < (g + 1) * kGroup; ++k) {
            const size_t wi = c.transB ? static_cast<size_t>(n) * c.K + k
                                       : static_cast<size_t>(k) * c.N + n;
            s += (static_cast<int64_t>(a[static_cast<size_t>(m) * c.K + k])
                         - zp)
                    * w[wi];
        }
        const double t = static_cast<double>(s) * sa
                * ws[static_cast<size_t>(g) * c.N + n];
        sum += t;
        mag += std::fabs(t);
    }
    if (c.bias != kNone) {
        const double b = nt::load_f32(o.bias.get(), c.bias, n);
        sum += b;
        mag += std::fabs(b);
    }
    if (c.po == post_op_type_t::relu) sum = std::max(sum, 0.0);
    value = sum;
    bound = 4.0 * std::ldexp(mag, -24) + 1e-30;
}

/// The reference algorithm on rc into dst.  A call of {M, G} source scales
/// runs whole (run_reference would run it row by row, one scale per row):
/// through matmul_direct over per-group weight scales, through
/// reference_matmul_execute itself over per-channel ones, which
/// matmul_direct's checks refuse on every route.
status_t reference_into(const ncase_t &c, const nt::call_t &rc, void *dst) {
    if (c.q != q_t::after && c.q != q_t::grp_s8)
        return nt::run_reference(rc, rc.wei, dst);
    nt::call_t r = rc;
    r.dst = dst;
    r.params.lowoha_algo = matmul_algo_t::reference;
    r.params.weight_cache_type = 0;
    r.params.num_threads = 0;
    if (c.q == q_t::grp_s8) return nt::run_matmul(r);
    mm::matmul_batch_params_t batch;
    return mm::reference_matmul_execute('r', r.transA, r.transB, r.M, r.N, r.K,
            r.alpha, r.src, r.lda_or_tight(), r.wei, r.ldb_or_tight(), r.bias,
            r.beta, dst, r.ldc_or_tight(), false, batch, r.params);
}

/// The expected output of the case, [M, ldc] of the dst type, where only the
/// output columns of each row count.
buffer_t expected_of(const ncase_t &c, const operands_t &o) {
    buffer_t want = nt::make_buffer(o.init.bytes);
    std::memcpy(want.get(), o.init.get(), o.init.bytes);
    nt::call_t rc = make_call(c, o);
    if (c.q == q_t::grp_u8) {
        for (int m = 0; m < c.M; ++m)
            for (int n = 0; n < c.N; ++n) {
                double v = 0.0, b = 0.0;
                grp_u8_at(c, o, m, n, v, b);
                nt::store_f32(want.get(), c.dst,
                        static_cast<size_t>(m) * c.ldc() + n,
                        static_cast<float>(v));
            }
        return want;
    }
    if (c.gated == gated_t::none) {
        EXPECT_EQ(reference_into(c, rc, want.get()), status_t::success);
        return want;
    }
    // A gated call: the reference pre-activation into f32 [M, N] (beta
    // reading the dst's N pre-activation columns), then the activation here.
    buffer_t pre = nt::make_buffer(static_cast<size_t>(c.M) * c.N * 4);
    for (int m = 0; m < c.M; ++m)
        for (int j = 0; j < c.N && c.beta != 0.0f; ++j)
            nt::store_f32(pre.get(), kF32, static_cast<size_t>(m) * c.N + j,
                    nt::load_f32(o.init.get(), c.dst,
                            static_cast<size_t>(m) * c.ldc() + j));
    nt::call_t pc = rc;
    pc.gated = gated_t::none;
    pc.ldc = c.N;
    pc.params.dtypes.dst = kF32;
    pc.params.postop_.clear();
    EXPECT_EQ(reference_into(c, pc, pre.get()), status_t::success);
    std::vector<float> act(static_cast<size_t>(c.M) * c.out_cols());
    nt::gated_act_ref(pre.as<float>(), c.M, c.N, c.gated, act.data());
    for (int m = 0; m < c.M; ++m)
        for (int i = 0; i < c.out_cols(); ++i)
            nt::store_f32(want.get(), c.dst,
                    static_cast<size_t>(m) * c.ldc() + i,
                    act[static_cast<size_t>(m) * c.out_cols() + i]);
    return want;
}

/// got against want over the output columns: the band of the data type, one
/// step for an integer dst, the f64 bound of the model for grp_u8.
::testing::AssertionResult matches(const ncase_t &c, const operands_t &o,
        const void *got, const buffer_t &want) {
    const int cols = c.out_cols();
    if (c.q == q_t::grp_u8) {
        size_t bad = 0;
        std::string first;
        for (int m = 0; m < c.M; ++m)
            for (int n = 0; n < c.N; ++n) {
                double v = 0.0, b = 0.0;
                grp_u8_at(c, o, m, n, v, b);
                if (c.dst == kBf16) b += std::ldexp(std::fabs(v), -8);
                const float g = nt::load_f32(
                        got, c.dst, static_cast<size_t>(m) * c.ldc() + n);
                if (!(std::fabs(g - v) <= b) && bad++ == 0)
                    first = "(" + std::to_string(m) + ", " + std::to_string(n)
                            + "): got " + std::to_string(g) + ", f64 "
                            + std::to_string(v);
            }
        if (bad == 0) return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure()
                << bad << " elements outside the f64 bound, first " << first;
    }
    if (c.dst == kU8 || c.dst == kS8 || c.dst == kS32)
        return nt::within_steps(
                got, c.ldc(), want.get(), c.ldc(), c.dst, c.M, cols, 1);
    const bool low = c.q != q_t::f32;
    const float rtol = low ? rtol_bf16 : rtol_f32;
    const float abs_bound
            = nt::comparison_abs_bound(c.K, low ? epsilon_bf16 : epsilon_f32,
                    c.alpha, c.beta, low, c.int8(), false);
    return nt::within_band(got, c.ldc(), want.get(), c.ldc(), c.dst, c.M, cols,
            rtol, abs_bound);
}

class NativeBrgemmAnyN : public ::testing::TestWithParam<ncase_t> {
protected:
    void SetUp() override {
        const ncase_t &c = GetParam();
        const nat::UarchParams &u = nat::detect_uarch();
        if (!nat::native_base_isa()) GTEST_SKIP() << "no AVX-512 base ISA";
        if (c.q == q_t::bf16 && !u.avx512bf16) GTEST_SKIP() << "no BF16";
        if (c.int8() && !u.avx512vnni) GTEST_SKIP() << "no VNNI";
        ::reset_grp_matmul_caches();
    }
    void TearDown() override { ::reset_grp_matmul_caches(); }
    WeightCacheGuard global_cache_ {2};
};

TEST_P(NativeBrgemmAnyN, MatchesTheReferenceAndStaysInItsBuffers) {
    const ncase_t &c = GetParam();
    const operands_t o = make_operands(c);
    const buffer_t want = expected_of(c, o);
    if (HasFailure()) return;

    // The native call's operands, each in guarded storage.
    const auto src = guarded_copy(o.src);
    const auto wei = guarded_copy(o.wei);
    const auto bias = guarded_copy(o.bias);
    const auto wei_scale = guarded_copy(o.wei_scale);
    const auto src_scale = guarded_copy(o.src_scale);
    const auto src_zp = guarded_copy(o.src_zp);
    const auto dst_scale = guarded_copy(o.dst_scale);
    const auto dst_zp = guarded_copy(o.dst_zp);
    const auto po_buf = guarded_copy(o.po_buf);
    nt::call_t call = make_call(c, o);
    call.src = src->get();
    call.wei = wei->get();
    call.bias = bias ? bias->get() : nullptr;
    auto &qp = call.params.quant_params;
    if (wei_scale) qp.wei_scale.buff = wei_scale->get();
    if (src_scale) qp.src_scale.buff = src_scale->get();
    if (src_zp) qp.src_zp.buff = src_zp->get();
    if (dst_scale) qp.dst_scale.buff = dst_scale->get();
    if (dst_zp) qp.dst_zp.buff = dst_zp->get();
    if (po_buf) call.params.postop_[0].buff = po_buf->get();
    if (c.route == route_t::per_call)
        call.params.lowoha_algo = matmul_algo_t::native_brgemm;
    std::unique_ptr<nt::algo_pin_t> pin;
    if (c.route == route_t::algo10)
        pin.reset(new nt::algo_pin_t(matmul_algo_t::native_brgemm));
    const bool on_gemv = c.route != route_t::execute
            && nt::gemv_kernel(call) != nat::native_gemv_t::none;

    // The destination: the initial values, which beta reads (over a gated
    // call's N pre-activation columns), and sentinels in the columns past
    // them.
    guarded_t dst(o.init.bytes);
    ASSERT_NE(dst.get(), nullptr);
    const size_t esz = nt::dt_bytes(c.dst);
    const int written = c.out_cols();
    const int kept = c.ldc() - c.pad;
    std::vector<uint8_t> before(o.init.bytes);
    for (int i = 0; i < c.calls; ++i) {
        std::memcpy(dst.get(), o.init.get(), o.init.bytes);
        for (int m = 0; m < c.M; ++m)
            std::memset(
                    dst.get() + (static_cast<size_t>(m) * c.ldc() + kept) * esz,
                    0x5A, static_cast<size_t>(c.pad) * esz);
        std::memcpy(before.data(), dst.get(), before.size());
        call.dst = dst.get();
        const status_t st = c.route == route_t::execute ? run_execute(call)
                                                        : nt::run_matmul(call);
        ASSERT_EQ(st, status_t::success) << "call " << i;
        EXPECT_TRUE(matches(c, o, dst.get(), want)) << "call " << i;
        size_t touched = 0;
        for (int m = 0; m < c.M; ++m) {
            const size_t from
                    = (static_cast<size_t>(m) * c.ldc() + written) * esz;
            const size_t to = static_cast<size_t>(m + 1) * c.ldc() * esz;
            for (size_t b = from; b < to; ++b)
                touched += dst.get()[b] != before[b];
        }
        EXPECT_EQ(touched, 0u)
                << "call " << i << " wrote bytes past the output columns";
    }
    // A bf16 weight of whole blocks at weight cache mode 2 under the
    // process-wide algorithm, or through native_brgemm_execute, is rewritten
    // in place; a weight of padded columns never is.
    const bool in_place = c.N % nat::kNativeLayoutPackNr == 0 && c.wc == 2
            && c.q == q_t::bf16 && c.route != route_t::per_call && !on_gemv;
    if (!in_place) {
        EXPECT_EQ(std::memcmp(wei->get(), o.wei.get(), o.wei.bytes), 0)
                << "the weight was rewritten";
    }
    if (c.wc != 0 && !on_gemv) {
        EXPECT_EQ(nt::native_layouts(wei->get()), 1u)
                << "one layout (an image in place, else packed out of place)";
    }
    EXPECT_EQ(std::memcmp(src->get(), o.src.get(), o.src.bytes), 0)
            << "the source was written";
    // Cached layouts go with the weight's storage.
    ::reset_grp_matmul_caches();
}

// The problem of a call over a weight of N not a multiple of 32 has the
// padded width, which the planner's legal plans tile in whole column tiles;
// a call of N a multiple of 32 keeps its N.
TEST(NativeBrgemmAnyNPlan, ProblemTakesThePaddedWidth) {
    for (int N : {1, 8, 31, 32, 33, 72, 4096, 4097, 151937})
        for (int threads : {1, 8, 32})
            for (data_type_t wei : {kBf16, kS8, kF32}) {
                SCOPED_TRACE("N=" + std::to_string(N) + " threads="
                        + std::to_string(threads) + " " + dt_name(wei));
                nat::GemmDescriptor d;
                d.M = 33;
                d.N = N;
                d.K = 256;
                d.lda = d.K;
                d.ldb = d.K;
                d.ldc = N;
                d.transB = true;
                d.src_dt = wei == kS8 ? kU8 : wei;
                d.wei_dt = wei;
                d.dst_dt = wei == kF32 ? kF32 : kBf16;
                d.num_threads = threads;
                mm::matmul_params params;
                params.dtypes.src = d.src_dt;
                params.dtypes.wei = wei;
                params.dtypes.dst = d.dst_dt;
                const nat::native_problem_t p = nat::make_native_problem(d,
                        params, nullptr, nat::kNativeLayoutPackNr,
                        nat::native_int8_decision_t {});
                EXPECT_EQ(
                        p.N, nat::native_padded_n(N, nat::kNativeLayoutPackNr));
                const nat::native_plan_t plan = nat::native_planner().plan(p);
                EXPECT_EQ(nat::validate_native_plan(plan, p), nullptr);
                EXPECT_EQ(plan.n_tiles * plan.nr, p.N);
            }
}

// The [MATMUL.NATIVE_BRGEMM] line of a call of padded columns names the
// width that its plan, the tuning file and ZENDNNL_MATMUL_NATIVE_CONFIG
// match (padded_n=): a clause on that width applies to a call of N 4113,
// and a call of N a multiple of 32 names no padded width.
TEST(NativeBrgemmAnyNPlan, PlanLineNamesThePaddedWidth) {
    const nat::UarchParams &u = nat::detect_uarch();
    if (!nat::native_base_isa() || !u.avx512bf16) GTEST_SKIP() << "no BF16";
    nt::expect_in_child(
            {{"ZENDNNL_API_LOG_LEVEL", "4"},
                    {"ZENDNNL_MATMUL_NATIVE_CONFIG", "n=4128:nr=16"},
                    {"ZENDNNL_MATMUL_NATIVE_TUNING_FILE", nullptr}},
            [] {
        int bad = 0;
        for (const int N : {4113, 4096}) {
            const int M = 8, K = 64;
            const buffer_t src = nt::random_buffer(
                    kBf16, static_cast<size_t>(M) * K, 9100, -1.0f, 1.0f);
            const buffer_t wei = nt::kept_until_exit(nt::random_buffer(
                    kBf16, static_cast<size_t>(K) * N, 9101, -1.0f, 1.0f));
            const buffer_t dst
                    = nt::make_buffer(static_cast<size_t>(M) * N * 4);
            nt::call_t c;
            c.M = M;
            c.N = N;
            c.K = K;
            c.src = src.get();
            c.wei = wei.get();
            c.dst = dst.get();
            c.params.dtypes.src = kBf16;
            c.params.dtypes.wei = kBf16;
            c.params.dtypes.dst = kF32;
            c.params.num_threads = 4;
            c.params.lowoha_algo = matmul_algo_t::native_brgemm;
            ::testing::internal::CaptureStdout();
            const status_t st = nt::run_matmul(c);
            const std::string log = ::testing::internal::GetCapturedStdout();
            const std::string head
                    = "[MATMUL.NATIVE_BRGEMM] M=8 N=" + std::to_string(N) + " ";
            const size_t at = log.find(head);
            const std::string line = at == std::string::npos
                    ? std::string()
                    : log.substr(at, log.find('\n', at) - at);
            const bool padded = N % nat::kNativeLayoutPackNr != 0;
            const auto has = [&](const char *what) {
                return line.find(what) != std::string::npos;
            };
            const bool ok = st == status_t::success && !line.empty()
                    && has(" source=override") == padded
                    && (padded ? has(" padded_n=4128") && has(" nr=16 ")
                               : !has("padded_n="));
            if (!ok) {
                std::fprintf(stderr, "N=%d status %d, line: %s\n", N,
                        static_cast<int>(st), line.c_str());
                ++bad;
            }
        }
        return bad;
    }, "the plan line of a call of padded columns");
}

constexpr int kMs[] = {1, 2, 7, 8, 33, 256, 1031};
constexpr int kKs[] = {64, 3, 100, 16, 257};

data_type_t pick_dst(q_t q, int i) {
    switch (q) {
        case q_t::bf16: return i % 2 ? kF32 : kBf16;
        case q_t::f32: return kF32;
        case q_t::st_s8: {
            static const data_type_t d[] = {kF32, kBf16, kS8, kU8};
            return d[i % 4];
        }
        case q_t::st_u8: {
            static const data_type_t d[] = {kBf16, kF32, kU8, kS32};
            return d[i % 4];
        }
        default: return i % 2 ? kBf16 : kF32;
    }
}

/// alpha / beta and one post-op of a case, by i; an integer dst takes no
/// transcendental post-op, a call of per-group weight scales over a u8
/// source takes at most relu.
void pick_epilogue(ncase_t &c, int i) {
    const bool integer = c.dst == kU8 || c.dst == kS8 || c.dst == kS32;
    switch (i % 8) {
        case 1: c.po = post_op_type_t::relu; break;
        case 2:
            c.alpha = 0.5f;
            c.beta = 0.25f;
            break;
        case 3: c.po = post_op_type_t::binary_add; break;
        case 4:
            c.po = post_op_type_t::binary_mul;
            c.po_dt = kBf16;
            c.po_row = true;
            break;
        case 5:
            if (!integer) c.po = post_op_type_t::gelu_erf;
            break;
        case 6:
            if (!integer) c.po = post_op_type_t::softmax;
            break;
        case 7:
            if (!integer) c.po = post_op_type_t::tanh;
            break;
        default: break;
    }
    if (c.q == q_t::grp_u8) {
        c.alpha = 1.0f;
        c.beta = 0.0f;
        if (c.po != post_op_type_t::relu) c.po = post_op_type_t::none;
    }
}

/// Every N from 1 to 100 for every data type, with M, K, the weight's
/// storage, threads, weight cache mode, ldc padding, dst, bias, alpha /
/// beta and post-op turning with N.
std::vector<ncase_t> every_n_cases(route_t route, int offset) {
    static const q_t kQs[] = {q_t::bf16, q_t::f32, q_t::st_s8, q_t::st_u8,
            q_t::dyn_s8, q_t::dyn_u8, q_t::grp_s8, q_t::grp_u8};
    std::vector<ncase_t> v;
    for (int qi = 0; qi < 8; ++qi)
        for (int N = 1; N <= 100; ++N) {
            const int i = N + 3 * qi + offset;
            ncase_t c;
            c.q = kQs[qi];
            c.route = route;
            c.N = N;
            // matmul_direct takes per-token scales of an asymmetric source on
            // one row only.
            c.M = c.q == q_t::dyn_u8 ? 1 : kMs[i % 7];
            c.K = c.groups() ? kGroup * (1 + i % 4) : kKs[i % 5];
            c.transB = i % 2 == 1;
            c.threads = (i / 2) % 3 == 0 ? 1 : ((i / 2) % 3 == 1 ? 3 : 0);
            c.wc = (i / 3) % 3;
            c.pad = i % 4 == 0 ? 0 : 1 + i % 7;
            c.dst = pick_dst(c.q, i);
            static const data_type_t kBias[] = {kNone, kF32, kBf16};
            c.bias = kBias[(i / 4) % 3];
            pick_epilogue(c, i);
            v.push_back(c);
        }
    return named(v);
}

/// The tail of a block above a large N: 4096 + 1, + 17, + 31, and an odd
/// vocabulary width.
std::vector<ncase_t> large_n_cases(route_t route) {
    std::vector<ncase_t> v;
    int i = 0;
    for (int N : {4097, 4113, 4127, 151937})
        for (q_t q :
                {q_t::bf16, q_t::f32, q_t::st_u8, q_t::dyn_s8, q_t::grp_s8}) {
            ncase_t c;
            c.q = q;
            c.route = route;
            c.N = N;
            c.M = N > 5000 ? (i % 2 ? 2 : 7) : kMs[i % 5];
            c.K = 64;
            c.transB = i % 2 == 0;
            c.threads = 0;
            c.wc = i % 3;
            c.pad = i % 2 ? 3 : 0;
            c.dst = q == q_t::f32 ? kF32 : (i % 2 ? kBf16 : kF32);
            c.bias = i % 3 == 0 ? kF32 : kNone;
            ++i;
            v.push_back(c);
        }
    return named(v);
}

/// Gated activations over even N: silu / gelu over the gate / up halves of
/// the weight's plain layout (paired in the tiles, or staged rows), swiglu
/// over weight columns that interleave the pairs; with a bias, beta over the
/// dst's pre-activation columns, BF16, FP32, INT8 per-channel (static, and
/// per token quantized in the call) and per-group, and per-channel weight
/// scales after {M, G} source groups.
std::vector<ncase_t> gated_cases(route_t route) {
    static const gated_t kActs[]
            = {gated_t::silu_and_mul, gated_t::gelu_and_mul,
                    gated_t::swiglu_oai_mul, gated_t::gelu_tanh_and_mul};
    std::vector<q_t> qs
            = {q_t::bf16, q_t::f32, q_t::st_u8, q_t::dyn_s8, q_t::grp_s8};
    // A call of {M, G} source scales and per-channel weight scales reaches
    // the native BRGEMM through matmul_direct only without its checks.
    if (route == route_t::execute) qs.push_back(q_t::after);
    std::vector<ncase_t> v;
    int i = 0;
    for (q_t q : qs)
        for (int N = 2; N <= 100;
                N += (q == q_t::bf16 || q == q_t::dyn_s8 ? 2 : 6)) {
            ncase_t c;
            c.q = q;
            c.route = route;
            c.N = N;
            c.M = kMs[i % 6];
            c.K = q == q_t::grp_s8 || q == q_t::after ? kGroup * (1 + i % 3)
                                                      : kKs[i % 5];
            c.transB = i % 2 == 0;
            c.threads = i % 3 == 0 ? 1 : 0;
            c.wc = i % 3;
            c.pad = i % 4;
            c.dst = q == q_t::f32 ? kF32 : (i % 2 ? kBf16 : kF32);
            c.gated = kActs[i % 4];
            c.bias = i % 4 == 1 ? kF32 : kNone;
            // beta with swiglu reads a copy of the pre-activation columns.
            if (i % 5 == 2) c.beta = 0.5f;
            ++i;
            v.push_back(c);
        }
    return named(v);
}

/// Repeated calls on one weight at every weight cache mode, the second and
/// later served from the layout the first one packed out of place.
std::vector<ncase_t> repeat_cases(route_t route) {
    std::vector<ncase_t> v;
    for (int wc : {0, 1, 2})
        for (q_t q : {q_t::bf16, q_t::st_u8, q_t::f32})
            for (int N : {1, 40, 72}) {
                ncase_t c;
                c.q = q;
                c.route = route;
                c.N = N;
                c.M = N == 1 ? 8 : 33;
                c.K = 2048 / (N == 1 ? 1 : 16);
                c.transB = N != 72;
                c.threads = 0;
                c.wc = wc;
                c.calls = 3;
                c.dst = q == q_t::bf16 ? kBf16 : kF32;
                v.push_back(c);
            }
    return named(v);
}

INSTANTIATE_TEST_SUITE_P(EveryN, NativeBrgemmAnyN,
        ::testing::ValuesIn(every_n_cases(route_t::execute, 0)),
        [](const ::testing::TestParamInfo<ncase_t> &info) {
            return info.param.name;
        });

INSTANTIATE_TEST_SUITE_P(LargeN, NativeBrgemmAnyN,
        ::testing::ValuesIn(large_n_cases(route_t::execute)),
        [](const ::testing::TestParamInfo<ncase_t> &info) {
            return info.param.name;
        });

INSTANTIATE_TEST_SUITE_P(Gated, NativeBrgemmAnyN,
        ::testing::ValuesIn(gated_cases(route_t::execute)),
        [](const ::testing::TestParamInfo<ncase_t> &info) {
            return info.param.name;
        });

INSTANTIATE_TEST_SUITE_P(Repeat, NativeBrgemmAnyN,
        ::testing::ValuesIn(repeat_cases(route_t::execute)),
        [](const ::testing::TestParamInfo<ncase_t> &info) {
            return info.param.name;
        });

/// The cases of route of every fifth N.
std::vector<ncase_t> every_fifth_n(std::vector<ncase_t> v) {
    v.erase(std::remove_if(v.begin(), v.end(),
                    [](const ncase_t &c) { return c.N % 5 != 0; }),
            v.end());
    return v;
}

INSTANTIATE_TEST_SUITE_P(EveryNAlgo10, NativeBrgemmAnyN,
        ::testing::ValuesIn(every_n_cases(route_t::algo10, 5)),
        [](const ::testing::TestParamInfo<ncase_t> &info) {
            return info.param.name;
        });

INSTANTIATE_TEST_SUITE_P(EveryFifthNPerCall, NativeBrgemmAnyN,
        ::testing::ValuesIn(
                every_fifth_n(every_n_cases(route_t::per_call, 11))),
        [](const ::testing::TestParamInfo<ncase_t> &info) {
            return info.param.name;
        });

INSTANTIATE_TEST_SUITE_P(LargeNAlgo10, NativeBrgemmAnyN,
        ::testing::ValuesIn(large_n_cases(route_t::algo10)),
        [](const ::testing::TestParamInfo<ncase_t> &info) {
            return info.param.name;
        });

INSTANTIATE_TEST_SUITE_P(GatedAlgo10, NativeBrgemmAnyN,
        ::testing::ValuesIn(gated_cases(route_t::algo10)),
        [](const ::testing::TestParamInfo<ncase_t> &info) {
            return info.param.name;
        });

INSTANTIATE_TEST_SUITE_P(RepeatAlgo10, NativeBrgemmAnyN,
        ::testing::ValuesIn(repeat_cases(route_t::algo10)),
        [](const ::testing::TestParamInfo<ncase_t> &info) {
            return info.param.name;
        });

INSTANTIATE_TEST_SUITE_P(RepeatPerCall, NativeBrgemmAnyN,
        ::testing::ValuesIn(repeat_cases(route_t::per_call)),
        [](const ::testing::TestParamInfo<ncase_t> &info) {
            return info.param.name;
        });

} // namespace
