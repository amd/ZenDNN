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

// The A/B corpus of the native BRGEMM: the same calls, layouts and layout
// rule verdicts on any build, for ab.py to compare two builds by.
//
// calls   Dense calls through matmul_direct with lowoha_algo native_brgemm,
//         under ZENDNNL_MATMUL_ALGO=10 (which lets weight cache mode 2
//         rewrite a weight in place), on every route: bf16, f32, f16, s8
//         with per-channel and with per-group weight scales, s4 in groups
//         and s4 of one group over K.  Around a base call per route it
//         varies M (1 .. 13, 64, 1024), N (tails), K (long K, K tails,
//         groups), bias, post-ops and a chain of two, gated activations,
//         alpha and beta, the dst type and its requantization, transA,
//         transB, a padded ldb and ldc, the source (u8 with or without a
//         zero point, s8, or bf16 quantized to s8 or u8) and its scales, a
//         per-tensor weight scale, the scale types, the weight cache mode,
//         a weight buffer larger than the weight (an in-place image of its
//         capacity) and the thread count.  A few calls leave lowoha_algo
//         none, which ZENDNNL_MATMUL_ALGO=10 alone brings to the native
//         BRGEMM: a call it refuses goes back to the default route or keeps
//         its error.  Each case runs twice on the same buffers (the second
//         call reads the layout the first cached or wrote in place) and
//         prints
//           BEGIN <case>
//           <the [MATMUL.NATIVE_BRGEMM] line of each call that ran>
//           CASE <case> st=<status> out=<hash> again=<hash> wei=<hash>
//                st2=<status> [sq=<hash> sq2=<hash>]
//         with the status of each call and the 64-bit FNV-1a hashes of the
//         dst allocation after each call (rows, their padding up to ldc and
//         kGuard bytes after the last row), of the weight buffer after both
//         (an image written in place, up to its capacity, and kGuard bytes
//         of kCanary after it) and, for a quantized route, of the source's
//         scales and zero point after each call, each with kGuard bytes of
//         kCanary after it (a dynamic quantization writes them, into
//         buffers of kCanary at each call).  The weight caches are cleared
//         after each case and its buffers stay allocated, so no case
//         depends on another.
// layout  native_pack_weight of every format over K and N tails, transB,
//         padded ldb and thread counts, and for bf16 and f16 weights of
//         whole blocks the unpack of the layout; the hashes take kGuard
//         bytes after the layout and after the unpacked weight:
//           LAYOUT <case> bytes=<n> hash=<hash>
//           UNPACK <case> hash=<hash>
// rules   The native BRGEMM's verdict on a call (native_layout_check: every
//         check up to the weight's layout, matmul_direct's own checks left
//         out) over source, weight and dst types, static and dynamic
//         quantization, weight scale groups, K bounds and ldb; each call
//         with lowoha_algo native_brgemm, then with lowoha_algo none
//         (<case>/algo10), whose refusal may hand it back to the default
//         route (taken=0); --filter selects verdicts by these names:
//           RULE <case> taken=<0|1> st=<status> why=<reason>
// pack    The prepack time of native_pack_weight per format on weights of
//         model size and, for bf16 and f16, the time of the unpack of their
//         layout (median and minimum of --reps runs after one warm-up); a
//         pack that fails ends the run with exit status 1:
//           PACK <case> median_ms=<ms> min_ms=<ms>
//           UNPACK <case> median_ms=<ms> min_ms=<ms>
// build.sh compiles this corpus against the tree of each build, also one
// from before native_unpack_weight, whose unpacks are native_unpack_bf16
// and native_unpack_f16.
// layout and pack call native_pack_weight, which needs the native BRGEMM's
// base instruction sets (native_base_isa): without them they exit with
// status 2.  Every input comes from a generator seeded by its case name: a
// case gets the same data in any run, with any filter and on any build.
//
// Usage: ab_corpus calls|layout|rules|pack [--filter <substring>]
//                  [--reps <n>]
// Logs the plan lines with ZENDNNL_API_LOG_LEVEL=4.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <type_traits>

#include "zendnnl.hpp"

#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"
#include "lowoha_operators/matmul/matmul_native/native_matmul.hpp"

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
using zendnnl::common::bfloat16_t;
using zendnnl::common::data_type_t;
using zendnnl::common::float16_t;
using zendnnl::common::matmul_algo_t;
using zendnnl::common::post_op_type_t;
using zendnnl::error_handling::status_t;
using gated_t = mm::matmul_gated_act_t;

/// Stand-ins for the unpack functions of a library that lacks them.  A
/// library function of the same name outranks its stand-in (any argument
/// converts to any_t, the others pass through the ellipsis), so a call
/// resolves to a stand-in only where the library has no function of its
/// name; unpack_layout reads that from the call's type and runs none.
namespace ab_unpack_api {
struct missing_t {};
struct any_t {
    template <typename T>
    any_t(const T &) {}
};
missing_t native_unpack_weight(any_t, ...);
missing_t native_unpack_bf16(any_t, ...);
missing_t native_unpack_f16(any_t, ...);
} // namespace ab_unpack_api

namespace {

constexpr data_type_t kBf16 = data_type_t::bf16;
constexpr data_type_t kF32 = data_type_t::f32;
constexpr data_type_t kF16 = data_type_t::f16;
constexpr data_type_t kS8 = data_type_t::s8;
constexpr data_type_t kU8 = data_type_t::u8;
constexpr data_type_t kS4 = data_type_t::s4;
constexpr data_type_t kS32 = data_type_t::s32;
constexpr data_type_t kNone = data_type_t::none;

/// Bytes after dst, a weight buffer, the source's scales and zero point, a
/// layout or an unpacked weight that no call may write, hashed with them.
constexpr size_t kGuard = 64;
/// The byte that guards and buffers a call writes start as: a store of
/// zeros changes them.
constexpr uint8_t kCanary = 0xA5;

/// The 64-bit FNV-1a hash of n bytes at p, continuing from h.
uint64_t fnv1a(const void *p, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto *b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i)
        h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

/// splitmix64, seeded by a case name.
struct rng_t {
    uint64_t s;
    explicit rng_t(const std::string &name)
        : s(fnv1a(name.data(), name.size())) {}
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    /// Uniform in [lo, hi), 24 random bits: sums of such values change
    /// with the order of their terms.
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * static_cast<float>(next() >> 40) * 0x1p-24f;
    }
};

struct free_t {
    void operator()(void *p) const { std::free(p); }
};
using buffer_t = std::unique_ptr<uint8_t, free_t>;

/// 64-byte aligned, zero-filled storage of at least one byte.
buffer_t make_buffer(size_t bytes) {
    bytes = std::max<size_t>(64, (bytes + 63) & ~size_t {63});
    void *p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) {
        std::fprintf(stderr, "out of memory (%zu bytes)\n", bytes);
        std::exit(1);
    }
    std::memset(p, 0, bytes);
    return buffer_t(static_cast<uint8_t *>(p));
}

/// Weight buffers live until the process exits: no later buffer takes the
/// address of a weight that a weight cache record may still name.
std::vector<buffer_t> &kept() {
    static std::vector<buffer_t> k;
    return k;
}

uint8_t *keep(buffer_t b) {
    kept().push_back(std::move(b));
    return kept().back().get();
}

size_t dt_bytes(data_type_t dt) {
    switch (dt) {
        case kF32:
        case kS32: return 4;
        case kBf16:
        case kF16: return 2;
        default: return 1;
    }
}

const char *dt_name(data_type_t dt) {
    switch (dt) {
        case kBf16: return "bf16";
        case kF32: return "f32";
        case kF16: return "f16";
        case kS8: return "s8";
        case kU8: return "u8";
        case kS4: return "s4";
        case data_type_t::u4: return "u4";
        case kS32: return "s32";
        default: return "none";
    }
}

/// v as an element of a float or integer buffer of dt.
void store(void *base, data_type_t dt, size_t i, float v) {
    switch (dt) {
        case kF32: static_cast<float *>(base)[i] = v; break;
        case kBf16: static_cast<bfloat16_t *>(base)[i] = bfloat16_t(v); break;
        case kF16: static_cast<float16_t *>(base)[i] = float16_t(v); break;
        case kS32:
            static_cast<int32_t *>(base)[i] = static_cast<int32_t>(v);
            break;
        case kS8:
            static_cast<int8_t *>(base)[i] = static_cast<int8_t>(v);
            break;
        default:
            static_cast<uint8_t *>(base)[i] = static_cast<uint8_t>(v);
            break;
    }
}

/// n elements of dt drawn from [lo, hi] (whole numbers for integer types),
/// then extra zero bytes.
buffer_t random_buffer(rng_t &r, data_type_t dt, size_t n, float lo, float hi,
        size_t extra = 0) {
    buffer_t b = make_buffer(n * dt_bytes(dt) + extra);
    const bool whole = dt == kS8 || dt == kU8 || dt == kS32;
    for (size_t i = 0; i < n; ++i) {
        float v = r.uniform(lo, whole ? hi + 1.0f : hi);
        if (whole) v = static_cast<float>(static_cast<int64_t>(v - lo) + lo);
        store(b.get(), dt, i, v);
    }
    return b;
}

buffer_t random_bytes(rng_t &r, size_t n, size_t extra = 0) {
    buffer_t b = make_buffer(n + extra);
    for (size_t i = 0; i < n; ++i)
        b.get()[i] = static_cast<uint8_t>(r.next() >> 56);
    return b;
}

// ---------------------------------------------------------------- calls

enum class route_t { bf16, f32, f16, s8_channel, s8_group, s4_group, s4_one };
enum class gran_t { tensor, token, group };
enum class po_t {
    none,
    relu,
    gelu,
    tanh,
    swish,
    clip,
    add,
    add_row,
    mul,
    abs,
    elu,
    exp,
    gelu_tanh,
    leaky_relu,
    log,
    mish,
    sigmoid,
    softmax,
    sqrt,
    square,
    sum
};

const char *route_name(route_t r) {
    switch (r) {
        case route_t::bf16: return "bf16";
        case route_t::f32: return "f32";
        case route_t::f16: return "f16";
        case route_t::s8_channel: return "s8c";
        case route_t::s8_group: return "s8g";
        case route_t::s4_group: return "s4g";
        default: return "s4one";
    }
}

const char *gran_name(gran_t g) {
    return g == gran_t::tensor   ? "tensor"
            : g == gran_t::token ? "token"
                                 : "group";
}

const char *po_name(po_t p) {
    switch (p) {
        case po_t::relu: return "relu";
        case po_t::gelu: return "gelu";
        case po_t::tanh: return "tanh";
        case po_t::swish: return "swish";
        case po_t::clip: return "clip";
        case po_t::add: return "add";
        case po_t::add_row: return "add_row";
        case po_t::mul: return "mul";
        case po_t::abs: return "abs";
        case po_t::elu: return "elu";
        case po_t::exp: return "exp";
        case po_t::gelu_tanh: return "gelu_tanh";
        case po_t::leaky_relu: return "leaky_relu";
        case po_t::log: return "log";
        case po_t::mish: return "mish";
        case po_t::sigmoid: return "sigmoid";
        case po_t::softmax: return "softmax";
        case po_t::sqrt: return "sqrt";
        case po_t::square: return "square";
        case po_t::sum: return "sum";
        default: return "none";
    }
}

const char *gated_name(gated_t g) {
    switch (g) {
        case gated_t::silu_and_mul: return "silu";
        case gated_t::gelu_and_mul: return "gelu";
        case gated_t::gelu_tanh_and_mul: return "gelu_tanh";
        case gated_t::swiglu_oai_mul: return "swiglu_oai";
        default: return "none";
    }
}

bool quantized(route_t r) {
    return r == route_t::s8_channel || r == route_t::s8_group
            || r == route_t::s4_group || r == route_t::s4_one;
}

bool int4(route_t r) {
    return r == route_t::s4_group || r == route_t::s4_one;
}

/// Weight scales in groups along K (s8 per group, s4 in groups).
bool grouped(route_t r) {
    return r == route_t::s8_group || r == route_t::s4_group;
}

data_type_t wei_of(route_t r) {
    switch (r) {
        case route_t::bf16: return kBf16;
        case route_t::f32: return kF32;
        case route_t::f16: return kF16;
        case route_t::s8_channel:
        case route_t::s8_group: return kS8;
        default: return kS4;
    }
}

struct call_case_t {
    route_t route = route_t::bf16;
    int M = 8, N = 96, K = 256;
    bool transA = false, transB = true;
    int ldb_pad = 0;
    /// The call's source and compute type; dynamic: a bf16 source the call
    /// quantizes to the compute type.
    data_type_t src = kBf16, compute = kNone;
    bool dynamic = false;
    data_type_t dst = kBf16, bias = kNone;
    /// K values per weight scale group (s8 per group, s4 in groups).
    int group = 0;
    gran_t gran = gran_t::token;
    data_type_t wei_scale_dt = kF32, src_scale_dt = kF32;
    /// Post-ops po, then po2; the type of their binary operands.
    po_t po = po_t::none, po2 = po_t::none;
    data_type_t po_dt = kF32;
    gated_t gated = gated_t::none;
    float alpha = 1.0f, beta = 0.0f;
    int wc = 2;
    int threads = 0;
    /// ldc past the output columns.
    int ldc_pad = 0;
    /// Requantization of dst: a per-tensor dst scale, and a dst zero point.
    bool dst_scale = false, dst_zp = false;
    /// A per-tensor weight scale ({1, 1}); the zero point of a static u8
    /// source.
    bool wei_scale_1x1 = false, src_zp = false;
    /// A weight buffer of the layout's size (wei_buffer_capacity_bytes),
    /// when larger than the weight.
    bool capacity = false;
    /// lowoha_algo none: the native BRGEMM through ZENDNNL_MATMUL_ALGO=10.
    bool algo10 = false;

    std::string name() const {
        char s[512];
        std::snprintf(s, sizeof(s),
                "%s/M%d/N%d/K%d/tA%d/tB%d/pad%d/src=%s%s%s/dst=%s/bias=%s/g%d/"
                "%s/ws=%s/ss=%s/po=%s/gate=%s/a%g/b%g/wc%d/t%d",
                route_name(route), M, N, K, transA ? 1 : 0, transB ? 1 : 0,
                ldb_pad, dt_name(src), dynamic ? ":" : "",
                dynamic ? dt_name(compute) : "", dt_name(dst), dt_name(bias),
                group, gran_name(gran), dt_name(wei_scale_dt),
                dt_name(src_scale_dt), po_name(po), gated_name(gated),
                static_cast<double>(alpha), static_cast<double>(beta), wc,
                threads);
        std::string out = s;
        if (po2 != po_t::none) out += std::string("/po2=") + po_name(po2);
        if (po_dt != kF32) out += std::string("/po_dt=") + dt_name(po_dt);
        if (ldc_pad != 0) out += "/ldc+" + std::to_string(ldc_pad);
        if (dst_scale) out += dst_zp ? "/dst_scale+zp" : "/dst_scale";
        if (wei_scale_1x1) out += "/ws1x1";
        if (src_zp) out += "/src_zp";
        if (capacity) out += "/capacity";
        if (algo10) out += "/algo10";
        return out;
    }
};

void set_quant(mm::matmul_quantization_params_t::matmul_quant_t &q,
        const void *buff, data_type_t dt, std::vector<int64_t> dims) {
    q.buff = buff;
    q.dt = dt;
    q.dims = std::move(dims);
}

/// The post-op of po; the operand of a binary one (rows of out_cols values
/// of po_dt) is drawn from r into operand.
mm::matmul_post_op make_post_op(po_t po_kind, rng_t &r, int M, int out_cols,
        data_type_t po_dt, buffer_t &operand) {
    mm::matmul_post_op po;
    switch (po_kind) {
        case po_t::relu: po.po_type = post_op_type_t::relu; break;
        case po_t::gelu: po.po_type = post_op_type_t::gelu_erf; break;
        case po_t::tanh: po.po_type = post_op_type_t::tanh; break;
        case po_t::swish: po.po_type = post_op_type_t::swish; break;
        case po_t::clip:
            po.po_type = post_op_type_t::clip;
            po.alpha = -0.5f;
            po.beta = 0.75f;
            break;
        case po_t::abs: po.po_type = post_op_type_t::abs; break;
        case po_t::elu:
            po.po_type = post_op_type_t::elu;
            po.alpha = 0.5f;
            break;
        case po_t::exp: po.po_type = post_op_type_t::exp; break;
        case po_t::gelu_tanh: po.po_type = post_op_type_t::gelu_tanh; break;
        case po_t::leaky_relu:
            po.po_type = post_op_type_t::leaky_relu;
            po.alpha = 0.1f;
            break;
        case po_t::log: po.po_type = post_op_type_t::log; break;
        case po_t::mish: po.po_type = post_op_type_t::mish; break;
        case po_t::sigmoid: po.po_type = post_op_type_t::sigmoid; break;
        case po_t::softmax: po.po_type = post_op_type_t::softmax; break;
        case po_t::sqrt: po.po_type = post_op_type_t::sqrt; break;
        case po_t::square: po.po_type = post_op_type_t::square; break;
        case po_t::sum:
            po.po_type = post_op_type_t::sum;
            po.alpha = 0.5f;
            break;
        default: {
            const int rows = po_kind == po_t::add_row ? 1 : M;
            operand = random_buffer(r, po_dt,
                    static_cast<size_t>(rows) * out_cols, -1.0f, 1.0f);
            po.po_type = po_kind == po_t::mul ? post_op_type_t::binary_mul
                                              : post_op_type_t::binary_add;
            po.buff = operand.get();
            po.dtype = po_dt;
            po.dims = {rows, out_cols};
            po.leading_dim = out_cols;
            break;
        }
    }
    return po;
}

int run_call_case(const call_case_t &c) {
    const std::string name = c.name();
    rng_t r(name);
    const data_type_t wei_dt = wei_of(c.route);
    const bool gated = c.gated != gated_t::none;
    const int M = c.M, N = c.N, K = c.K;
    const int lda = c.transA ? M : K;
    const int ldb = (c.transB ? K : N) + c.ldb_pad;
    const int out_cols = gated ? N / 2 : N;
    const int ldc = out_cols + c.ldc_pad;
    const size_t w_elems = static_cast<size_t>((c.transB ? N : K) - 1) * ldb
            + (c.transB ? K : N);
    const size_t w_bytes
            = wei_dt == kS4 ? (w_elems + 1) / 2 : w_elems * dt_bytes(wei_dt);
    // The weight buffer: the weight, room up to its stated capacity, then
    // the guard.
    const size_t owned = c.capacity
            ? std::max(w_bytes,
                      nat::native_packed_bytes(
                              wei_dt, K, N, nat::kNativeLayoutPackNr))
            : w_bytes;
    const size_t w_tail = owned - w_bytes + kGuard;

    uint8_t *wei = nullptr;
    if (wei_dt == kS4)
        wei = keep(random_bytes(r, w_bytes, w_tail));
    else if (wei_dt == kS8)
        wei = keep(random_buffer(r, kS8, w_elems, -128.0f, 127.0f, w_tail));
    else
        wei = keep(random_buffer(r, wei_dt, w_elems, -1.0f, 1.0f, w_tail));
    std::memset(wei + owned, kCanary, kGuard);

    const size_t a_elems = static_cast<size_t>(M) * K;
    buffer_t src = c.src == kU8 ? random_buffer(r, kU8, a_elems, 0.0f, 255.0f)
            : c.src == kS8 ? random_buffer(r, kS8, a_elems, -128.0f, 127.0f)
                           : random_buffer(r, c.src, a_elems, -2.0f, 2.0f);

    mm::matmul_params p;
    p.dtypes.src = c.src;
    p.dtypes.wei = wei_dt;
    p.dtypes.dst = c.dst;
    p.dtypes.bias = c.bias;
    p.dtypes.compute = c.compute;
    p.dynamic_quant = c.dynamic;
    p.lowoha_algo
            = c.algo10 ? matmul_algo_t::none : matmul_algo_t::native_brgemm;
    p.num_threads = c.threads;
    p.weight_cache_type = c.wc;
    if (c.capacity) p.wei_buffer_capacity_bytes = owned;

    buffer_t wei_scale, src_scale, src_zp;
    // The bytes of the source's scales and of its zero point, each with
    // kGuard bytes after it.
    size_t scale_bytes = 0, zp_bytes = 0;
    if (quantized(c.route)) {
        const int G = grouped(c.route) ? K / c.group : 1;
        wei_scale = random_buffer(
                r, c.wei_scale_dt, static_cast<size_t>(G) * N, 0.002f, 0.01f);
        set_quant(p.quant_params.wei_scale, wei_scale.get(), c.wei_scale_dt,
                c.wei_scale_1x1 ? std::vector<int64_t> {1, 1}
                                : std::vector<int64_t> {G, N});
        const int groups = c.gran == gran_t::group ? G : 1;
        const int rows = c.gran == gran_t::tensor ? 1 : M;
        const size_t n_scale = static_cast<size_t>(rows) * groups;
        scale_bytes = n_scale * dt_bytes(c.src_scale_dt) + kGuard;
        if (p.dynamic_quant) {
            src_scale = make_buffer(scale_bytes);
            if (c.compute == kU8) {
                zp_bytes = n_scale * 4 + kGuard;
                src_zp = make_buffer(zp_bytes);
                set_quant(p.quant_params.src_zp, src_zp.get(), kS32,
                        {rows, groups});
            }
        } else {
            src_scale = random_buffer(
                    r, c.src_scale_dt, n_scale, 0.002f, 0.01f, kGuard);
            std::memset(
                    src_scale.get() + scale_bytes - kGuard, kCanary, kGuard);
            if (c.src_zp) {
                zp_bytes = 4 + kGuard;
                src_zp = random_buffer(r, kS32, 1, 1.0f, 9.0f, kGuard);
                std::memset(src_zp.get() + 4, kCanary, kGuard);
                set_quant(p.quant_params.src_zp, src_zp.get(), kS32, {1, 1});
            }
        }
        set_quant(p.quant_params.src_scale, src_scale.get(), c.src_scale_dt,
                {rows, groups});
    }
    buffer_t dst_scale, dst_zp;
    if (c.dst_scale) {
        dst_scale = random_buffer(r, kF32, 1, 2.0f, 8.0f);
        set_quant(p.quant_params.dst_scale, dst_scale.get(), kF32, {1, 1});
        if (c.dst_zp) {
            dst_zp = c.dst == kU8 ? random_buffer(r, kS32, 1, 100.0f, 120.0f)
                                  : random_buffer(r, kS32, 1, -5.0f, 5.0f);
            set_quant(p.quant_params.dst_zp, dst_zp.get(), kS32, {1, 1});
        }
    }

    buffer_t bias;
    if (c.bias != kNone) bias = random_buffer(r, c.bias, N, -1.0f, 1.0f);
    buffer_t operand, operand2;
    if (c.po != po_t::none)
        p.postop_.push_back(
                make_post_op(c.po, r, M, out_cols, c.po_dt, operand));
    if (c.po2 != po_t::none)
        p.postop_.push_back(
                make_post_op(c.po2, r, M, out_cols, c.po_dt, operand2));

    // dst: M rows of ldc, then the guard; all of it starts as init.
    const size_t c_elems
            = static_cast<size_t>(M) * ldc + kGuard / dt_bytes(c.dst);
    const size_t c_bytes = c_elems * dt_bytes(c.dst);
    const buffer_t init = c.dst == kU8
            ? random_buffer(r, kU8, c_elems, 0.0f, 255.0f)
            : random_buffer(r, c.dst, c_elems, -2.0f, 2.0f);
    buffer_t dst = make_buffer(c_bytes);
    mm::matmul_batch_params_t bp;
    mm::matmul_gated_act_params g;
    g.act = c.gated;
    const auto call = [&]() {
        std::memcpy(dst.get(), init.get(), c_bytes);
        if (p.dynamic_quant && src_scale) {
            std::memset(src_scale.get(), kCanary, scale_bytes);
            if (src_zp) std::memset(src_zp.get(), kCanary, zp_bytes);
        }
        const status_t st = mm::matmul_direct('r', c.transA, c.transB, M, N, K,
                c.alpha, src.get(), lda, wei, ldb, bias.get(), c.beta,
                dst.get(), ldc, true, bp, p, gated ? &g : nullptr);
        std::fflush(stdout);
        return st;
    };
    // The source's scales and zero point.
    const auto quant_hash = [&]() {
        const uint64_t h = fnv1a(src_scale.get(), scale_bytes);
        return src_zp ? fnv1a(src_zp.get(), zp_bytes, h) : h;
    };

    std::printf("BEGIN %s\n", name.c_str());
    std::fflush(stdout);
    const status_t st = call();
    const uint64_t out = fnv1a(dst.get(), c_bytes);
    const uint64_t sq = src_scale ? quant_hash() : 0;
    const status_t st2 = call();
    const uint64_t again = fnv1a(dst.get(), c_bytes);
    const uint64_t sq2 = src_scale ? quant_hash() : 0;
    const uint64_t w_hash = fnv1a(wei, owned + kGuard);
    nat::clear_native_weight_caches();
    char quant[64] = "";
    if (src_scale)
        std::snprintf(quant, sizeof(quant), " sq=%016llx sq2=%016llx",
                static_cast<unsigned long long>(sq),
                static_cast<unsigned long long>(sq2));
    std::printf(
            "CASE %s st=%d out=%016llx again=%016llx wei=%016llx st2=%d%s\n",
            name.c_str(), static_cast<int>(st),
            static_cast<unsigned long long>(out),
            static_cast<unsigned long long>(again),
            static_cast<unsigned long long>(w_hash), static_cast<int>(st2),
            quant);
    std::fflush(stdout);
    return st == status_t::success ? 1 : 0;
}

/// The base call of a route: a source and scales it serves, a K it takes.
/// A quantized route's source is a bf16 one the call quantizes to s8 per
/// token.
call_case_t base_of(route_t route) {
    call_case_t c;
    c.route = route;
    c.dynamic = quantized(route);
    switch (route) {
        case route_t::bf16: c.K = 263; break;
        case route_t::f32:
            c.src = c.dst = kF32;
            c.K = 263;
            break;
        case route_t::f16:
            c.src = c.dst = kF16;
            c.K = 263;
            break;
        case route_t::s8_channel:
            c.compute = kS8;
            c.K = 262;
            break;
        case route_t::s8_group:
            c.compute = kS8;
            c.K = 256;
            c.group = 64;
            break;
        case route_t::s4_group:
            c.compute = kS8;
            c.K = 256;
            c.group = 64;
            c.wei_scale_dt = kBf16;
            break;
        case route_t::s4_one:
            c.compute = kS8;
            c.K = 260;
            c.wei_scale_dt = kBf16;
            break;
    }
    return c;
}

/// The calls, each once: a variation that gives the call of an earlier case
/// is left out.
std::vector<call_case_t> call_cases() {
    std::vector<call_case_t> out;
    std::set<std::string> names;
    const route_t routes[]
            = {route_t::bf16, route_t::f32, route_t::f16, route_t::s8_channel,
                    route_t::s8_group, route_t::s4_group, route_t::s4_one};
    for (const route_t route : routes) {
        const call_case_t base = base_of(route);
        const bool q = quantized(route);
        const auto add = [&](const call_case_t &c) {
            if (names.insert(c.name()).second) out.push_back(c);
        };
        // M, over whole blocks and a tail, transB and not; M = 1 (the GEMV
        // kernels) also with ldc past N.
        for (const int M :
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 64, 1024})
            for (const int N : {96, 100})
                for (const bool tB : {true, false}) {
                    call_case_t c = base;
                    c.M = M;
                    c.N = N;
                    c.transB = tB;
                    add(c);
                    if (M != 1) continue;
                    c.ldc_pad = 5;
                    add(c);
                }
        // N tails, with ldc N and N + 5: a store past N changes the padding
        // of a row or the guard after dst.
        for (const int N : {1, 7, 16, 31, 33, 64, 130, 250})
            for (const int M : {1, 5, 32})
                for (const int pad : {0, 5}) {
                    call_case_t c = base;
                    c.M = M;
                    c.N = N;
                    c.ldc_pad = pad;
                    add(c);
                }
        // K: tails of the K groups, long K (several K passes) and groups.
        std::vector<std::pair<int, int>> ks; // K, group
        switch (route) {
            case route_t::s8_channel:
                for (const int K : {1, 3, 4, 63, 64, 1000, 2051, 4096, 8192})
                    ks.push_back({K, 0});
                break;
            case route_t::s8_group:
                ks = {{4, 4}, {12, 4}, {64, 32}, {1000, 8}, {1024, 64},
                        {2048, 128}, {4096, 128}, {4096, 1024}, {8192, 256}};
                break;
            case route_t::s4_group:
                ks = {{8, 4}, {64, 32}, {1000, 8}, {1024, 128}, {2052, 4},
                        {4096, 128}, {4096, 2048}, {8192, 256}};
                break;
            case route_t::s4_one:
                for (const int K : {4, 8, 12, 260, 1024, 4100, 8192})
                    ks.push_back({K, 0});
                break;
            default:
                for (const int K : {1, 2, 15, 64, 255, 1000, 2049, 4096, 8192})
                    ks.push_back({K, 0});
                break;
        }
        for (const auto &kg : ks)
            for (const int M : {1, 4, 13, 64}) {
                call_case_t c = base;
                c.M = M;
                c.N = 160;
                c.K = kg.first;
                c.group = kg.second;
                add(c);
            }
        // Bias and post-ops: each post-op the epilogue applies, a chain of
        // two (in an order whose result differs from the other) and a bf16
        // binary operand.
        for (const int M : {1, 6, 33}) {
            for (const data_type_t b : {kF32, kBf16}) {
                call_case_t c = base;
                c.M = M;
                c.bias = b;
                add(c);
            }
            for (const po_t po : {po_t::relu, po_t::gelu, po_t::tanh,
                         po_t::swish, po_t::clip, po_t::add, po_t::add_row,
                         po_t::mul, po_t::abs, po_t::elu, po_t::exp,
                         po_t::gelu_tanh, po_t::leaky_relu, po_t::log,
                         po_t::mish, po_t::sigmoid, po_t::softmax, po_t::sqrt,
                         po_t::square}) {
                call_case_t c = base;
                c.M = M;
                c.N = 100;
                c.bias = kF32;
                c.po = po;
                add(c);
            }
            call_case_t c = base;
            c.M = M;
            c.N = 100;
            c.bias = kF32;
            c.po = po_t::relu;
            c.po2 = po_t::add;
            add(c);
            c.po = po_t::add;
            c.po2 = po_t::none;
            c.po_dt = kBf16;
            add(c);
        }
        // Gated activations: halves paired (N a multiple of 32: by runs of
        // 16 columns, or by whole blocks over enough tiles) or staged; at
        // M 1 and 7 also with ldc past the output columns.
        for (const gated_t gt : {gated_t::silu_and_mul, gated_t::gelu_and_mul,
                     gated_t::gelu_tanh_and_mul, gated_t::swiglu_oai_mul})
            for (const int M : {1, 7, 64})
                for (const int N : {128, 1024, 100}) {
                    call_case_t c = base;
                    c.M = M;
                    c.N = N;
                    c.gated = gt;
                    c.bias = M == 7 ? kF32 : kNone;
                    add(c);
                    if (M == 64) continue;
                    c.ldc_pad = 5;
                    add(c);
                }
        // alpha and beta.
        for (const int M : {3, 40})
            for (const auto &ab : {std::pair<float, float> {1.5f, 0.0f},
                         std::pair<float, float> {1.0f, 0.5f},
                         std::pair<float, float> {0.75f, -1.0f}}) {
                call_case_t c = base;
                c.M = M;
                c.alpha = ab.first;
                c.beta = ab.second;
                add(c);
            }
        // dst types matmul_direct takes for the route.
        std::vector<data_type_t> dsts = {kBf16, kF32};
        if (route == route_t::f32) dsts = {kF32};
        if (route == route_t::f16) dsts = {kF16, kF32};
        if (q && !int4(route)) dsts = {kBf16, kF32, kS8, kU8, kS32};
        for (const data_type_t d : dsts)
            for (const int M : {1, 9, 64}) {
                call_case_t c = base;
                c.M = M;
                c.dst = d;
                add(c);
            }
        // transA.
        for (const int M : {1, 5, 13, 64}) {
            call_case_t c = base;
            c.M = M;
            c.transA = true;
            add(c);
        }
        // Weight cache modes 0 and 1 (2 is the base).
        for (const int wc : {0, 1})
            for (const bool tB : {true, false})
                for (const int M : {1, 8}) {
                    call_case_t c = base;
                    c.M = M;
                    c.wc = wc;
                    c.transB = tB;
                    add(c);
                }
        // Thread counts.
        for (const int t : {1, 3, 5})
            for (const int M : {1, 8, 64}) {
                call_case_t c = base;
                c.M = M;
                c.threads = t;
                add(c);
            }
        // Padded ldb: even (rows of whole bytes for s4) and odd.
        for (const int pad : {8, 1})
            for (const bool tB : {true, false})
                for (const int M : {1, 8}) {
                    call_case_t c = base;
                    c.M = M;
                    c.ldb_pad = pad;
                    c.transB = tB;
                    add(c);
                }
        // A weight buffer that states the layout's size as its capacity: a
        // layout larger than the weight (bf16 of odd K, s4 of K in quads)
        // is then written in place.
        for (const bool tB : {true, false})
            for (const int M : {1, 8}) {
                call_case_t c = base;
                c.M = M;
                c.transB = tB;
                c.capacity = true;
                add(c);
            }
        // lowoha_algo none: the native BRGEMM serves the base call; a sum
        // post-op, which its epilogue does not apply, goes back to the
        // default route (and is refused with lowoha_algo native_brgemm).
        for (const int M : {1, 8}) {
            call_case_t c = base;
            c.M = M;
            c.algo10 = true;
            add(c);
            c.po = po_t::sum;
            add(c);
            c.algo10 = false;
            add(c);
        }
        if (!q) continue;
        // Sources and their scales as matmul_direct takes them: static u8
        // (per tensor) and s8 sources, a bf16 source quantized to s8, or to
        // u8 (asymmetric, per tensor); an s4 weight takes s8 sources scaled
        // per token or per group.
        struct src_t {
            data_type_t src, compute;
            bool dynamic;
            gran_t gran;
        };
        std::vector<src_t> srcs = int4(route)
                ? std::vector<src_t> {{kS8, kS8, false, gran_t::token},
                          {kBf16, kS8, true, gran_t::token}}
                : std::vector<src_t> {{kU8, kNone, false, gran_t::tensor},
                          {kS8, kNone, false, gran_t::tensor},
                          {kS8, kNone, false, gran_t::token},
                          {kBf16, kS8, true, gran_t::tensor},
                          {kBf16, kS8, true, gran_t::token},
                          {kBf16, kU8, true, gran_t::tensor}};
        if (grouped(route)) {
            srcs.push_back(
                    {kS8, int4(route) ? kS8 : kNone, false, gran_t::group});
            srcs.push_back({kBf16, kS8, true, gran_t::group});
        }
        for (const src_t &s : srcs)
            for (const int M : {1, 8, 64}) {
                call_case_t c = base;
                c.M = M;
                c.src = s.src;
                c.compute = s.compute;
                c.dynamic = s.dynamic;
                c.gran = s.gran;
                add(c);
            }
        // Scale types.
        for (const data_type_t ws : {kF32, kBf16})
            for (const data_type_t ss : {kF32, kBf16})
                for (const int M : {1, 16}) {
                    call_case_t c = base;
                    c.M = M;
                    c.wei_scale_dt = ws;
                    c.src_scale_dt = ss;
                    add(c);
                }
        if (int4(route)) continue;
        // A weight scale of one value ({1, 1}) on per-channel s8 (an s4
        // weight takes {G, N} only), with a source scale of one value
        // (matmul_direct takes per-token source scales with per-channel
        // weight scales only): a source quantized per tensor, and a static
        // u8 source.
        if (!grouped(route))
            for (const int M : {1, 8, 64}) {
                call_case_t c = base;
                c.M = M;
                c.gran = gran_t::tensor;
                c.wei_scale_1x1 = true;
                add(c);
                c.src = kU8;
                c.compute = kNone;
                c.dynamic = false;
                add(c);
            }
        // A static u8 source with a zero point, and an s8 or u8 dst
        // requantized by a per-tensor dst scale, with and without a dst
        // zero point.
        for (const int M : {1, 8, 64}) {
            call_case_t c = base;
            c.M = M;
            c.src = kU8;
            c.compute = kNone;
            c.dynamic = false;
            c.gran = gran_t::tensor;
            c.src_zp = true;
            add(c);
            for (const data_type_t d : {kS8, kU8})
                for (const bool zp : {false, true}) {
                    call_case_t rq = base;
                    rq.M = M;
                    rq.dst = d;
                    rq.dst_scale = true;
                    rq.dst_zp = zp;
                    add(rq);
                }
        }
    }
    return out;
}

int run_calls(const char *filter) {
    int cases = 0, served = 0;
    for (const call_case_t &c : call_cases()) {
        if (filter != nullptr && c.name().find(filter) == std::string::npos)
            continue;
        ++cases;
        served += run_call_case(c);
    }
    std::printf("SUMMARY calls cases=%d served=%d\n", cases, served);
    return 0;
}

// --------------------------------------------------------------- layout

struct layout_case_t {
    data_type_t wei = kBf16;
    int K = 1, N = 1;
    bool transB = false;
    int ldb_pad = 0;
    int threads = 1;

    std::string name() const {
        char s[128];
        std::snprintf(s, sizeof(s), "%s/K%d/N%d/tB%d/pad%d/t%d", dt_name(wei),
                K, N, transB ? 1 : 0, ldb_pad, threads);
        return s;
    }
    int ldb() const { return (transB ? K : N) + ldb_pad; }
    size_t raw_bytes() const {
        const size_t elems = static_cast<size_t>((transB ? N : K) - 1) * ldb()
                + (transB ? K : N);
        return wei == kS4 ? (elems + 1) / 2 : elems * dt_bytes(wei);
    }
    /// Whether the layout of the case unpacks: bf16 and f16 of whole blocks.
    bool unpacks() const {
        return (wei == kBf16 || wei == kF16)
                && N % nat::kNativeLayoutPackNr == 0;
    }
};

/// The unpack of the bf16 or f16 layout of c into weight: the library's
/// native_unpack_weight, or the unpack of the weight's type in a library
/// from before it.
void unpack_layout(
        const layout_case_t &c, const void *packed, void *weight, int threads) {
    using namespace ab_unpack_api;
    using namespace nat;
    const int nr = kNativeLayoutPackNr;
    using generic_t = decltype(native_unpack_weight(
            c.wei, packed, c.K, c.N, c.ldb(), nr, c.transB, weight, threads));
    if constexpr (!std::is_same<generic_t, missing_t>::value)
        native_unpack_weight(c.wei, packed, c.K, c.N, c.ldb(), nr, c.transB,
                weight, threads);
    else if (c.wei == kF16)
        native_unpack_f16(
                packed, c.K, c.N, c.ldb(), nr, c.transB, weight, threads);
    else
        native_unpack_bf16(
                packed, c.K, c.N, c.ldb(), nr, c.transB, weight, threads);
}

std::vector<layout_case_t> layout_cases() {
    std::vector<layout_case_t> out;
    for (const data_type_t w : {kBf16, kS8, kS4, kF32, kF16}) {
        for (const int K : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 64, 100,
                     1023, 1024, 1029, 2050})
            for (const int N : {1, 5, 31, 32, 33, 64, 100, 257})
                for (const bool tB : {false, true})
                    for (const int pad : {0, 3, 8}) {
                        if (K > 1024 && pad == 3) continue;
                        layout_case_t c;
                        c.wei = w;
                        c.K = K;
                        c.N = N;
                        c.transB = tB;
                        c.ldb_pad = pad;
                        out.push_back(c);
                    }
        // Several threads, over the tiles of a long K and over blocks.
        for (const int K : {64, 1024, 2050})
            for (const int N : {512, 1000})
                for (const bool tB : {false, true})
                    for (const int t : {3, 8}) {
                        layout_case_t c;
                        c.wei = w;
                        c.K = K;
                        c.N = N;
                        c.transB = tB;
                        c.threads = t;
                        out.push_back(c);
                    }
    }
    return out;
}

int run_layout(const char *filter) {
    int cases = 0;
    for (const layout_case_t &c : layout_cases()) {
        const std::string name = c.name();
        if (filter != nullptr && name.find(filter) == std::string::npos)
            continue;
        ++cases;
        rng_t r(name);
        const buffer_t w = random_bytes(r, c.raw_bytes());
        const size_t bytes = nat::native_packed_bytes(
                c.wei, c.K, c.N, nat::kNativeLayoutPackNr);
        buffer_t packed = make_buffer(bytes + kGuard);
        std::memset(packed.get(), kCanary, bytes + kGuard);
        const status_t st = nat::native_pack_weight(c.wei, w.get(), c.K, c.N,
                c.ldb(), nat::kNativeLayoutPackNr, c.transB, packed.get(),
                c.threads);
        std::printf("LAYOUT %s st=%d bytes=%zu hash=%016llx\n", name.c_str(),
                static_cast<int>(st), bytes,
                static_cast<unsigned long long>(
                        fnv1a(packed.get(), bytes + kGuard)));
        if (c.unpacks()) {
            const size_t back_bytes = c.raw_bytes() + kGuard;
            buffer_t back = make_buffer(back_bytes);
            std::memset(back.get(), 0x5A, back_bytes);
            unpack_layout(c, packed.get(), back.get(), c.threads);
            std::printf("UNPACK %s hash=%016llx\n", name.c_str(),
                    static_cast<unsigned long long>(
                            fnv1a(back.get(), back_bytes)));
        }
    }
    std::printf("SUMMARY layout cases=%d\n", cases);
    return 0;
}

// ---------------------------------------------------------------- rules

/// A source the call passes as is (static) or quantizes (dynamic) to the
/// compute type.
struct quant_t {
    const char *name;
    data_type_t compute;
    bool dynamic;
};

int run_rules(const char *filter) {
    int verdicts = 0;
    const auto rule
            = [&](const std::string &name, data_type_t src, data_type_t wei,
                      data_type_t dst, const quant_t &q, int scale_groups,
                      int K, int N, int ldb_cut) {
        // The verdicts with lowoha_algo native_brgemm and none, by the names
        // the filter selects them by; the inputs are seeded by name.
        const matmul_algo_t algos[2]
                = {matmul_algo_t::native_brgemm, matmul_algo_t::none};
        const std::string names[2] = {name, name + "/algo10"};
        bool kept[2];
        for (int i = 0; i < 2; ++i)
            kept[i] = filter == nullptr
                    || names[i].find(filter) != std::string::npos;
        if (!kept[0] && !kept[1]) return;
        const int M = 3;
        rng_t r(name);
        const int ldb = K - ldb_cut;
        const size_t w_bytes = static_cast<size_t>(N) * K
                * (wei == kS4 || wei == data_type_t::u4 ? 1 : dt_bytes(wei));
        const buffer_t w = random_bytes(r, w_bytes);
        const buffer_t a = random_bytes(r, static_cast<size_t>(M) * K * 4);
        buffer_t d = make_buffer(static_cast<size_t>(M) * N * 4);
        const buffer_t ws = make_buffer(static_cast<size_t>(K + 3) * N * 4);
        buffer_t ss = make_buffer(static_cast<size_t>(M) * 4);
        buffer_t zp = make_buffer(4);
        mm::matmul_params p;
        p.dtypes.src = src;
        p.dtypes.wei = wei;
        p.dtypes.dst = dst;
        p.dtypes.compute = q.compute;
        p.dynamic_quant = q.dynamic;
        // scale_groups: 0 none, 1 per channel, -1 groups that do not split
        // K evenly (the fewest above one), else that many groups.
        if (scale_groups != 0) {
            int64_t G = scale_groups;
            if (scale_groups == -1) {
                G = 2;
                while (K % G == 0)
                    ++G;
            }
            for (int64_t i = 0; i < G * N; ++i)
                store(ws.get(), kF32, static_cast<size_t>(i), 0.01f);
            set_quant(p.quant_params.wei_scale, ws.get(), kF32, {G, N});
        }
        // A static source one scale; a quantized one a scale per token, or
        // to u8 one scale and one zero point.
        if (q.dynamic && q.compute == kU8) {
            set_quant(p.quant_params.src_scale, ss.get(), kF32, {1, 1});
            set_quant(p.quant_params.src_zp, zp.get(), kS32, {1, 1});
        } else if (q.dynamic) {
            set_quant(p.quant_params.src_scale, ss.get(), kF32, {M, 1});
        } else if (src == kU8 || src == kS8) {
            store(ss.get(), kF32, 0, 0.02f);
            set_quant(p.quant_params.src_scale, ss.get(), kF32, {1, 1});
        }
        // The verdict on a call that asks for the native BRGEMM (lowoha_algo
        // native_brgemm), then on the same call that ZENDNNL_MATMUL_ALGO=10
        // alone selects (lowoha_algo none), which a refusal may hand back
        // to the default route (taken=0).
        for (int i = 0; i < 2; ++i) {
            if (!kept[i]) continue;
            ++verdicts;
            p.lowoha_algo = algos[i];
            status_t st = status_t::success;
            const char *why = nullptr;
            const bool taken = nat::native_layout_check('r', false, true, M, N,
                    K, 1.0f, a.get(), K, w.get(), ldb, nullptr, 0.0f, d.get(),
                    N, true, 1, 0, gated_t::none, p, st, why);
            std::printf("RULE %s taken=%d st=%d why=%s\n", names[i].c_str(),
                    taken ? 1 : 0, static_cast<int>(st),
                    why != nullptr ? why : "");
        }
    };
    const data_type_t weis[]
            = {kBf16, kS8, kS4, kF32, kF16, kU8, data_type_t::u4};
    const data_type_t srcs[] = {kBf16, kF32, kF16, kU8, kS8};
    const data_type_t dsts[] = {kBf16, kF32, kF16, kS8, kS32};
    const quant_t quants[] = {{"none", kNone, false}, {"dyn_s8", kS8, true},
            {"dyn_u8", kU8, true}, {"s8", kS8, false}};
    // Weight scale groups by K: none, per channel, uneven, groups of 32
    // values (whole quads), groups of 6 values (not whole quads).
    for (const data_type_t wei : weis)
        for (const data_type_t src : srcs)
            for (const data_type_t dst : dsts)
                for (const quant_t &q : quants)
                    for (const int K : {1, 6, 64, 96})
                        for (const int sg : {0, 1, -1, 32, 6}) {
                            const int groups = sg == 32 || sg == 6
                                    ? (K % sg == 0 ? K / sg : -1)
                                    : sg;
                            char s[160];
                            std::snprintf(s, sizeof(s),
                                    "wei=%s/src=%s/dst=%s/q=%s/K%d/sg%d",
                                    dt_name(wei), dt_name(src), dt_name(dst),
                                    q.name, K, sg);
                            rule(s, src, wei, dst, q, groups, K, 40, 0);
                        }
    // K bounds of the int32 accumulation, and ldb below the row length.
    const quant_t none = quants[0], dyn_s8 = quants[1];
    rule("bound/s8/K65793", kU8, kS8, kF32, none, 1, 65793, 1, 0);
    rule("bound/s8/K65794", kU8, kS8, kF32, none, 1, 65794, 1, 0);
    rule("bound/s4/K2097148", kBf16, kS4, kF32, dyn_s8, 1, 2097148, 1, 0);
    rule("bound/s4/K2097152", kBf16, kS4, kF32, dyn_s8, 1, 2097152, 1, 0);
    for (const data_type_t wei : {kBf16, kS8, kS4, kF32, kF16}) {
        const data_type_t src = wei == kS8 ? kU8 : wei == kS4 ? kBf16 : wei;
        rule(std::string("ldb/") + dt_name(wei), src, wei, kF32,
                wei == kS4 ? dyn_s8 : none, wei == kS8 || wei == kS4 ? 1 : 0,
                64, 40, 1);
    }
    std::printf("SUMMARY rules verdicts=%d\n", verdicts);
    return 0;
}

// ----------------------------------------------------------------- pack

/// Prints "<what> <name> median_ms=<ms> min_ms=<ms>" over reps runs of run
/// after one warm-up run; false, printing nothing, when a run fails.
template <typename Run>
bool time_runs(const char *what, const std::string &name, int reps, Run run) {
    std::vector<double> ms;
    for (int i = 0; i <= reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!run()) return false;
        const auto t1 = std::chrono::steady_clock::now();
        if (i > 0)
            ms.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    std::printf("%s %s median_ms=%.4f min_ms=%.4f\n", what, name.c_str(),
            ms[ms.size() / 2], ms.front());
    std::fflush(stdout);
    return true;
}

int run_pack(const char *filter, int reps) {
    struct shape_t {
        int N, K;
    };
    const shape_t shapes[] = {{4096, 4096}, {14336, 4096}, {4096, 14336}};
    const int threads = omp_get_max_threads();
    for (const data_type_t w : {kBf16, kS8, kS4, kF32, kF16})
        for (const shape_t &sh : shapes)
            for (const bool tB : {true, false}) {
                layout_case_t c;
                c.wei = w;
                c.K = sh.K;
                c.N = sh.N;
                c.transB = tB;
                const std::string name = c.name();
                if (filter != nullptr && name.find(filter) == std::string::npos)
                    continue;
                rng_t r(name);
                const buffer_t src = random_bytes(r, c.raw_bytes());
                const size_t bytes = nat::native_packed_bytes(
                        w, c.K, c.N, nat::kNativeLayoutPackNr);
                buffer_t dst = make_buffer(bytes);
                status_t st = status_t::success;
                if (!time_runs("PACK", name, reps, [&] {
                    st = nat::native_pack_weight(w, src.get(), c.K, c.N,
                            c.ldb(), nat::kNativeLayoutPackNr, c.transB,
                            dst.get(), threads);
                    return st == status_t::success;
                })) {
                    std::fprintf(stderr,
                            "pack %s: native_pack_weight returned status %d\n",
                            name.c_str(), static_cast<int>(st));
                    return 1;
                }
                if (!c.unpacks()) continue;
                buffer_t back = make_buffer(c.raw_bytes());
                time_runs("UNPACK", name, reps, [&] {
                    unpack_layout(c, dst.get(), back.get(), threads);
                    return true;
                });
            }
    return 0;
}

const char *arg(int argc, char **argv, const char *name, const char *def) {
    for (int i = 2; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    return def;
}

} // namespace

int main(int argc, char **argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    const char *filter = arg(argc, argv, "--filter", nullptr);
    if ((mode == "layout" || mode == "pack") && !nat::native_base_isa()) {
        std::fprintf(stderr,
                "%s: native_pack_weight needs the native BRGEMM's base "
                "instruction sets (native_base_isa), which this CPU lacks\n",
                mode.c_str());
        return 2;
    }
    if (mode == "calls") return run_calls(filter);
    if (mode == "layout") return run_layout(filter);
    if (mode == "rules") return run_rules(filter);
    if (mode == "pack")
        return run_pack(filter,
                std::max(1, std::atoi(arg(argc, argv, "--reps", "15"))));
    std::fprintf(stderr,
            "usage: %s calls|layout|rules|pack [--filter <substring>] "
            "[--reps <n>]\n",
            argv[0]);
    return 2;
}
