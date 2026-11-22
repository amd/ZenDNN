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

// Op-level replay: times matmul_direct exactly as the framework plugin's dense
// linear layers call it.
//
// DA8W8 (the plugin's dynamic quantized linear): layout 'r', transA=false,
// transB=true, lda=ldb=K, ldc=N, alpha=1, beta=0, weights const, src bf16,
// wei s8 [N, K], dst bf16, compute s8, dynamic_quant, src_scale {nullptr,
// scale_dt, {M,1}}, wei_scale {buf, scale_dt, {1,N}}, weight_cache_type left
// at its default (2), num_threads 0.
// BF16 (--wei bf16, the plugin's linear through its direct kernel): the same
// call with src / wei / dst bf16, compute none, no quant params; the raw
// [N, K] weight is packed in place by its first call (weight cache 2).
// F16 (--wei f16): the BF16 call with an f16 source and weight and an f16
// dst (--dst f32 or bf16 for another dst).
// F32 (--wei f32): the BF16 call with an f32 source, weight and dst.
// DA8W4 (--wei s4, the plugin's dynamic quantized linear over int4 weights):
// the DA8W8 call with wei s4 packed [N, K / 2] (two values a byte, the even
// k in the low nibble, two's complement), ldb = K in values, and wei_scale
// {buf, scale_dt, {K / gs, N}} (--group gs; -1: one group over K, {1, N}).
// --gated silu_and_mul (the plugin's gated linear, bf16, s8 or s4 weights):
// the same call over a [gate | up] weight of N = 2I rows with the gated
// activation, writing act(gate) * up into an M x I output (ldc = N / 2).
// --check 1: after the timed passes of an M, one more call per shape (layer 0)
// and a 64-bit FNV-1a hash of its M x N output buffer (CHECK line; a gated
// call's M x I output and the zeros after it), to compare builds or plans bit
// for bit.
// Weights and activations are drawn in parallel 1 Mi blocks, each from its own
// seeded generator (float weights k / 1280 and activations k / 64 for
// small integers k, s8 weights in [-127, 127], s4 weights in [-8, 7], u8
// sources in [0, 255]). Every seed comes from the shape's N and K, so a shape
// gets the same data in any run.
//
// Weights: --layers distinct weight sets per shape, visited in layer order
// (layer l: every shape in --shapes order), so each weight is cold when it
// comes back, as in a model; the activation of a shape is one buffer (warm).
// Every weight is packed during the warm-up passes; the timed passes only
// replay.
//
// Usage: replay --shapes name:N:K[,..] --ms 1,8,32,4096 [--layers 4]
//               [--iters 30] [--warmup 2] [--scale-dt bf16|f32|f16]
//               [--bias none|bf16|f32] [--budget-ms 1500] [--src bf16|u8|s8]
//               [--wei s8|bf16|s4|f16|f32] [--group gs] [--dst bf16|f16|f32]
//               [--gated none|silu_and_mul] [--check 0|1]
// Prints one RESULT line per shape and M (per-call median / mean / min / p90
// in ms) and a TOTAL per M.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "zendnnl.hpp"

namespace mm = zendnnl::lowoha::matmul;
using zendnnl::common::bfloat16_t;
using zendnnl::common::data_type_t;
using zendnnl::error_handling::status_t;

namespace {

// plugin_op is only logged; labels longer than the small-string buffer keep
// the plugin's per-call string copy.
constexpr const char *kLinearOp = "framework_plugin::linear";
constexpr const char *kDynamicQLinearOp = "framework_plugin::dynamic_qlinear";

struct shape_t {
    std::string name;
    int N = 0, K = 0;
};

std::vector<std::string> split(const std::string &s, char c) {
    std::vector<std::string> out;
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(c, p);
        if (q == std::string::npos) q = s.size();
        if (q > p) out.push_back(s.substr(p, q - p));
        p = q + 1;
    }
    return out;
}

const char *arg(int argc, char **argv, const char *name, const char *def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    return def;
}

data_type_t dt_of(const std::string &s) {
    if (s == "bf16") return data_type_t::bf16;
    if (s == "f32") return data_type_t::f32;
    if (s == "f16") return data_type_t::f16;
    return data_type_t::none;
}

size_t dt_bytes(data_type_t dt) {
    return dt == data_type_t::f32 ? 4 : 2;
}

/// v stored as an element of a bf16, f16 or f32 buffer.
void store(void *base, data_type_t dt, size_t i, float v) {
    if (dt == data_type_t::f32)
        static_cast<float *>(base)[i] = v;
    else if (dt == data_type_t::f16)
        static_cast<zendnnl::common::float16_t *>(base)[i]
                = zendnnl::common::float16_t(v);
    else
        static_cast<bfloat16_t *>(base)[i] = bfloat16_t(v);
}

void *alloc64(size_t bytes) {
    void *p = nullptr;
    if (posix_memalign(&p, 64, (bytes + 63) & ~size_t {63}) != 0)
        return nullptr;
    return p;
}

double pct(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const size_t i = std::min(
            v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5));
    return v[i];
}

// A shape's data must not depend on the other shapes of a run: output hashes
// are compared across runs with different shape lists.
uint32_t shape_seed(int N, int K, int stream) {
    uint64_t h = 1469598103934665603ull;
    for (const uint64_t v : {static_cast<uint64_t>(N), static_cast<uint64_t>(K),
                 static_cast<uint64_t>(stream)}) {
        h ^= v;
        h *= 1099511628211ull;
    }
    return static_cast<uint32_t>(h ^ (h >> 32));
}

// f(i, gen) for i < n in parallel blocks of 1 Mi, block b drawing from its own
// generator seeded with (seed, b): the same values for the same seed at any
// thread count.
template <typename F>
void par_fill(size_t n, uint32_t seed, F &&f) {
    constexpr size_t kBlock = size_t {1} << 20;
    const int64_t blocks = static_cast<int64_t>((n + kBlock - 1) / kBlock);
#pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < blocks; ++b) {
        std::mt19937 g(seed * 1000003u + static_cast<uint32_t>(b));
        const size_t hi = std::min(n, static_cast<size_t>(b + 1) * kBlock);
        for (size_t i = static_cast<size_t>(b) * kBlock; i < hi; ++i)
            f(i, g);
    }
}

} // namespace

int main(int argc, char **argv) {
    std::vector<shape_t> shapes;
    for (const auto &t :
            split(arg(argc, argv, "--shapes", "qkv:6144:5120"), ',')) {
        const auto f = split(t, ':');
        if (f.size() != 3) {
            std::fprintf(stderr, "bad shape %s (name:N:K)\n", t.c_str());
            return 2;
        }
        shapes.push_back(
                {f[0], std::atoi(f[1].c_str()), std::atoi(f[2].c_str())});
    }
    std::vector<int> ms;
    for (const auto &t : split(arg(argc, argv, "--ms", "1,8,32"), ','))
        ms.push_back(std::atoi(t.c_str()));
    const int layers = std::atoi(arg(argc, argv, "--layers", "4"));
    const int iters_max = std::atoi(arg(argc, argv, "--iters", "30"));
    const int warmup = std::atoi(arg(argc, argv, "--warmup", "2"));
    const double budget_ms = std::atof(arg(argc, argv, "--budget-ms", "1500"));
    const data_type_t scale_dt = dt_of(arg(argc, argv, "--scale-dt", "bf16"));
    const data_type_t bias_dt = dt_of(arg(argc, argv, "--bias", "none"));
    // --src bf16 (default): dynamic per-token quantization as the plugin does;
    // u8 / s8: a pre-quantized source with one f32 scale (static), to compare
    // the kernels a source dtype selects.
    const std::string src_mode = arg(argc, argv, "--src", "bf16");
    const data_type_t src_dt = src_mode == "u8" ? data_type_t::u8
            : src_mode == "s8"                  ? data_type_t::s8
                                                : data_type_t::bf16;
    const std::string wei_mode = arg(argc, argv, "--wei", "s8");
    const bool wei_f16 = wei_mode == "f16";
    const bool wei_f32 = wei_mode == "f32";
    const bool wei_s4 = wei_mode == "s4";
    // An f16 or f32 call is a float call as a bf16 one, over operands of its
    // type.
    const bool wei_float = wei_mode == "bf16" || wei_f16 || wei_f32;
    if (!wei_float && !wei_s4 && wei_mode != "s8") {
        std::fprintf(stderr, "bad --wei %s (s8, bf16, s4, f16, f32)\n",
                wei_mode.c_str());
        return 2;
    }
    const data_type_t float_dt = wei_f16 ? data_type_t::f16
            : wei_f32                    ? data_type_t::f32
                                         : data_type_t::bf16;
    const char *dst_default = wei_f16 ? "f16" : wei_f32 ? "f32" : "bf16";
    const data_type_t out_dt = dt_of(arg(argc, argv, "--dst", dst_default));
    const int group = std::atoi(arg(argc, argv, "--group", "128"));
    const bool check = std::atoi(arg(argc, argv, "--check", "0")) != 0;
    const std::string gated_mode = arg(argc, argv, "--gated", "none");
    mm::matmul_gated_act_params gated;
    if (gated_mode == "silu_and_mul") {
        gated.act = mm::matmul_gated_act_t::silu_and_mul;
    } else if (gated_mode != "none") {
        std::fprintf(stderr, "--gated takes none or silu_and_mul\n");
        return 2;
    }
    const bool is_gated = gated.act != mm::matmul_gated_act_t::none;
    const int m_max = *std::max_element(ms.begin(), ms.end());
    const char *algo = std::getenv("ZENDNNL_MATMUL_ALGO");
    if (out_dt == data_type_t::none || scale_dt == data_type_t::none) {
        std::fprintf(stderr, "bad --dst or --scale-dt\n");
        return 2;
    }
    std::printf(
            "CONFIG algo=%s threads=%d layers=%d iters_max=%d warmup=%d "
            "scale_dt=%s bias=%s src=%s wei=%s group=%d dst=%s gated=%s\n",
            algo ? algo : "unset", omp_get_max_threads(), layers, iters_max,
            warmup, arg(argc, argv, "--scale-dt", "bf16"),
            arg(argc, argv, "--bias", "none"), src_mode.c_str(),
            wei_mode.c_str(), wei_s4 ? group : 0,
            arg(argc, argv, "--dst", dst_default), gated_mode.c_str());
    // --group -1: one group over K ({1, N} weight scales).
    for (const auto &sh : shapes) {
        if (wei_s4
                && (sh.K % 2 != 0
                        || (group != -1
                                && (group <= 0 || sh.K % group != 0)))) {
            std::fprintf(
                    stderr, "group %d does not divide K=%d\n", group, sh.K);
            return 2;
        }
    }
    if (is_gated && (wei_f16 || wei_f32)) {
        std::fprintf(stderr, "--gated takes bf16, s8 or s4 weights\n");
        return 2;
    }
    for (const auto &sh : shapes) {
        if (is_gated && sh.N % 2 != 0) {
            std::fprintf(stderr, "--gated needs an even N, not N=%d\n", sh.N);
            return 2;
        }
    }
    const auto groups_of
            = [&](int K) { return wei_s4 && group > 0 ? K / group : 1; };

    const size_t s_es = dt_bytes(scale_dt);
    const size_t b_es = dt_bytes(bias_dt);
    const size_t o_es = dt_bytes(out_dt);

    struct buffers_t {
        std::vector<void *> w, scale, bias;
        void *a = nullptr, *c = nullptr;
    };
    std::vector<buffers_t> buf(shapes.size());
    for (size_t s = 0; s < shapes.size(); ++s) {
        const int N = shapes[s].N, K = shapes[s].K;
        const int G = groups_of(K);
        for (int l = 0; l < layers; ++l) {
            const size_t w_bytes = wei_s4 ? static_cast<size_t>(N) * K / 2
                                          : static_cast<size_t>(N) * K
                            * (wei_float ? dt_bytes(float_dt) : 1);
            auto *w = static_cast<int8_t *>(alloc64(w_bytes));
            auto *sc = static_cast<uint8_t *>(
                    alloc64(static_cast<size_t>(G) * N * s_es));
            if (!w || !sc) {
                std::fprintf(stderr, "out of memory\n");
                return 1;
            }
            const uint32_t wseed = shape_seed(N, K, l + 1);
            if (wei_float) {
                par_fill(static_cast<size_t>(N) * K, wseed,
                        [w, float_dt](size_t i, std::mt19937 &g) {
                    store(w, float_dt, i,
                            static_cast<float>(static_cast<int>(g() % 129) - 64)
                                    / 1280.0f);
                });
            } else if (wei_s4) {
                par_fill(w_bytes, wseed, [w](size_t i, std::mt19937 &g) {
                    const uint32_t r = g();
                    const int lo = static_cast<int>(r % 16) - 8;
                    const int hi = static_cast<int>((r >> 8) % 16) - 8;
                    w[i] = static_cast<int8_t>(
                            (lo & 0x0F) | ((hi & 0x0F) << 4));
                });
            } else {
                par_fill(static_cast<size_t>(N) * K, wseed,
                        [w](size_t i, std::mt19937 &g) {
                    w[i] = static_cast<int8_t>(
                            static_cast<int>(g() % 255) - 127);
                });
            }
            std::mt19937 srng(shape_seed(N, K, 1000 + l));
            std::uniform_real_distribution<float> sd(0.0005f, 0.004f);
            for (size_t n = 0; n < static_cast<size_t>(G) * N; ++n)
                store(sc, scale_dt, n, sd(srng));
            buf[s].w.push_back(w);
            buf[s].scale.push_back(sc);
            void *b = nullptr;
            if (bias_dt != data_type_t::none) {
                b = alloc64(N * b_es);
                std::mt19937 brng(shape_seed(N, K, 2000 + l));
                std::normal_distribution<float> nd(0.0f, 1.0f);
                for (int n = 0; n < N; ++n)
                    store(b, bias_dt, n, 0.1f * nd(brng));
            }
            buf[s].bias.push_back(b);
        }
        void *a = alloc64(static_cast<size_t>(m_max) * K * (wei_f32 ? 4 : 2));
        buf[s].c = alloc64(static_cast<size_t>(m_max) * N * o_es);
        const uint32_t aseed = shape_seed(N, K, 63);
        if (src_dt == data_type_t::bf16 || wei_float) {
            const data_type_t a_dt = wei_float ? float_dt : data_type_t::bf16;
            par_fill(static_cast<size_t>(m_max) * K, aseed,
                    [a, a_dt](size_t i, std::mt19937 &g) {
                store(a, a_dt, i,
                        static_cast<float>(static_cast<int>(g() % 257) - 128)
                                / 64.0f);
            });
        } else {
            auto *au = static_cast<uint8_t *>(a);
            par_fill(static_cast<size_t>(m_max) * K, aseed,
                    [au](size_t i, std::mt19937 &g) {
                au[i] = static_cast<uint8_t>(g() % 256);
            });
        }
        buf[s].a = a;
        std::memset(buf[s].c, 0, static_cast<size_t>(m_max) * N * o_es);
    }

    auto call = [&](size_t s, int l, int M) {
        const int N = shapes[s].N, K = shapes[s].K;
        const int ldc = is_gated ? N / 2 : N;
        const mm::matmul_gated_act_params *gated_act
                = is_gated ? &gated : nullptr;
        static const float kStaticSrcScale = 0.02f;
        mm::matmul_params p;
        p.dtypes.src = src_dt;
        p.dtypes.wei = wei_s4 ? data_type_t::s4 : data_type_t::s8;
        p.dtypes.dst = out_dt;
        p.dtypes.bias = bias_dt;
        mm::matmul_batch_params_t bp;
        if (wei_float) {
            p.dtypes.src = float_dt;
            p.dtypes.wei = float_dt;
            p.dtypes.compute = data_type_t::none;
            p.plugin_op = kLinearOp;
            return mm::matmul_direct('r', false, true, M, N, K, 1.0f, buf[s].a,
                    K, buf[s].w[l], K, buf[s].bias[l], 0.0f, buf[s].c, ldc,
                    true, bp, p, gated_act);
        }
        if (src_dt == data_type_t::bf16) {
            p.dtypes.compute = data_type_t::s8;
            p.dynamic_quant = true;
            p.plugin_op = kDynamicQLinearOp;
            p.quant_params.src_scale.buff = nullptr;
            p.quant_params.src_scale.dt = scale_dt;
            p.quant_params.src_scale.dims = {M, 1};
        } else {
            p.quant_params.src_scale.buff = &kStaticSrcScale;
            p.quant_params.src_scale.dt = data_type_t::f32;
            p.quant_params.src_scale.dims = {1};
        }
        p.quant_params.wei_scale.buff = buf[s].scale[l];
        p.quant_params.wei_scale.dt = scale_dt;
        p.quant_params.wei_scale.dims = {groups_of(K), N};
        return mm::matmul_direct('r', false, true, M, N, K, 1.0f, buf[s].a, K,
                buf[s].w[l], K, buf[s].bias[l], 0.0f, buf[s].c, ldc, true, bp,
                p, gated_act);
    };

    for (int M : ms) {
        std::vector<std::vector<double>> t(shapes.size());
        for (int wi = 0; wi < warmup; ++wi)
            for (int l = 0; l < layers; ++l)
                for (size_t s = 0; s < shapes.size(); ++s)
                    if (call(s, l, M) != status_t::success) {
                        std::fprintf(stderr, "matmul_direct failed: %s M=%d\n",
                                shapes[s].name.c_str(), M);
                        return 1;
                    }
        // Size the timed passes from one probe pass.
        const auto p0 = std::chrono::steady_clock::now();
        for (int l = 0; l < layers; ++l)
            for (size_t s = 0; s < shapes.size(); ++s)
                call(s, l, M);
        const double pass_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - p0)
                                       .count();
        const int iters = std::max(3,
                std::min(iters_max,
                        static_cast<int>(budget_ms / std::max(pass_ms, 1e-3))));
        for (int it = 0; it < iters; ++it)
            for (int l = 0; l < layers; ++l)
                for (size_t s = 0; s < shapes.size(); ++s) {
                    const auto t0 = std::chrono::steady_clock::now();
                    const status_t st = call(s, l, M);
                    const auto t1 = std::chrono::steady_clock::now();
                    if (st != status_t::success) {
                        std::fprintf(stderr, "matmul_direct failed: %s M=%d\n",
                                shapes[s].name.c_str(), M);
                        return 1;
                    }
                    t[s].push_back(
                            std::chrono::duration<double, std::milli>(t1 - t0)
                                    .count());
                }
        double total = 0;
        for (size_t s = 0; s < shapes.size(); ++s) {
            double sum = 0;
            for (double x : t[s])
                sum += x;
            const double med = pct(t[s], 0.5);
            total += med;
            std::printf(
                    "RESULT shape=%s N=%d K=%d M=%d calls=%zu median_ms=%.4f "
                    "mean_ms=%.4f min_ms=%.4f "
                    "p90_ms=%.4f\n",
                    shapes[s].name.c_str(), shapes[s].N, shapes[s].K, M,
                    t[s].size(), med, sum / t[s].size(), pct(t[s], 0.0),
                    pct(t[s], 0.9));
        }
        std::printf("TOTAL M=%d sum_median_ms=%.4f\n", M, total);
        if (check) {
            for (size_t s = 0; s < shapes.size(); ++s) {
                const size_t bytes
                        = static_cast<size_t>(M) * shapes[s].N * o_es;
                std::memset(buf[s].c, 0, bytes);
                if (call(s, 0, M) != status_t::success) return 1;
                const auto *c = static_cast<const uint8_t *>(buf[s].c);
                uint64_t h = 1469598103934665603ull;
                for (size_t i = 0; i < bytes; ++i)
                    h = (h ^ c[i]) * 1099511628211ull;
                std::printf("CHECK shape=%s N=%d K=%d M=%d hash=%016llx\n",
                        shapes[s].name.c_str(), shapes[s].N, shapes[s].K, M,
                        static_cast<unsigned long long>(h));
            }
        }
        std::fflush(stdout);
    }
    return 0;
}
