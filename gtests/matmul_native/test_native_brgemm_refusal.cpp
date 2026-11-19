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

// The calls that ask for the native BRGEMM and that it does not serve: with
// lowoha_algo native_brgemm each returns its error status from matmul_direct,
// with dst untouched, the weight left raw, and one error line naming the
// reason; under ZENDNNL_MATMUL_ALGO=10 a call without lowoha_algo takes the
// default route instead, unless its arguments are wrong, its buffer holds
// an image the native BRGEMM wrote, or the default route does not compute
// its form correctly.  Calls of N not a multiple of 32 are served.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_call.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/native_matmul.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "native_brgemm_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
namespace rdr = zendnnl::lowoha::reorder;
using zendnnl::common::matmul_algo_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;

const data_type_t bf16 = data_type_t::bf16, f32 = data_type_t::f32,
                  s8 = data_type_t::s8, u8 = data_type_t::u8,
                  s32 = data_type_t::s32, s4 = data_type_t::s4,
                  u4 = data_type_t::u4;

bool native_isa() {
    const nat::UarchParams &u = nat::detect_uarch();
    return u.avx512f && u.avx512bf16 && u.avx512vnni;
}

/// One refused call: operands, params, and what it is refused with.
struct refused_t {
    const char *name;
    status_t status = status_t::unimplemented;
    const char *reason = "";
    nt::call_t c;
    nt::buffer_t src, wei, dst, prior;
    float src_scale = 0.5f;
    std::vector<float> wei_scales;
    int32_t zp = 3;
    /// Under ZENDNNL_MATMUL_ALGO=10: why the call keeps its error instead of
    /// taking the default route (null: it takes the default route).
    const char *kept = nullptr;
    char layout = 'r';
    std::vector<uint16_t> half_scales;
    std::vector<int8_t> wei_zps;
    /// A binary_add operand of [M, N] whose rows are operand_ld apart.
    std::vector<float> operand;
    int operand_ld = 0;
};

/// A call of M x N x K from src_dt x wei_dt into dst_dt on its own constant
/// weight, with per-tensor f32 scales for an s8 weight; dst holds random
/// values, which a refused call must keep.
std::unique_ptr<refused_t> make_case(const char *name, const char *reason,
        data_type_t src_dt, data_type_t wei_dt, data_type_t dst_dt, int M,
        int N, int K, uint32_t seed) {
    auto r = std::make_unique<refused_t>();
    r->name = name;
    r->reason = reason;
    const bool int8 = wei_dt == s8;
    r->src = nt::random_buffer(src_dt, static_cast<size_t>(M) * K, seed,
            int8 ? 0.0f : -1.0f, int8 ? 15.0f : 1.0f);
    const size_t wei_elems = static_cast<size_t>(K) * N;
    r->wei = nt::kept_until_exit(wei_dt == s4 || wei_dt == u4
                    ? nt::random_buffer(
                              u8, (wei_elems + 1) / 2, seed + 1, 0, 255)
                    : nt::random_buffer(wei_dt, wei_elems, seed + 1,
                              int8 ? -8.0f : -1.0f, int8 ? 8.0f : 1.0f));
    const size_t dst_bytes = static_cast<size_t>(M) * N * nt::dt_bytes(dst_dt);
    r->dst = nt::make_buffer(dst_bytes);
    r->prior = nt::random_buffer(dst_dt == s32 ? s32 : dst_dt,
            static_cast<size_t>(M) * N, seed + 2,
            dst_dt == bf16 || dst_dt == f32 ? -1.0f : 0.0f,
            dst_dt == bf16 || dst_dt == f32 ? 1.0f : 9.0f);
    std::memcpy(r->dst.get(), r->prior.get(), dst_bytes);
    nt::call_t &c = r->c;
    c.M = M;
    c.N = N;
    c.K = K;
    c.src = r->src.get();
    c.wei = r->wei.get();
    c.dst = r->dst.get();
    c.is_weights_const = true;
    c.params.dtypes.src = src_dt;
    c.params.dtypes.wei = wei_dt;
    c.params.dtypes.dst = dst_dt;
    c.params.weight_cache_type = 1;
    if (int8 || wei_dt == s4 || wei_dt == u4) {
        auto &q = c.params.quant_params;
        r->wei_scales.assign(N, 0.25f);
        q.wei_scale.buff = r->wei_scales.data();
        q.wei_scale.dt = f32;
        q.wei_scale.dims = {1, N};
        if (int8) {
            q.src_scale.buff = &r->src_scale;
            q.src_scale.dt = f32;
            q.src_scale.dims = {1, 1};
        }
    }
    return r;
}

/// Calls of N not a multiple of 32, which the native BRGEMM serves over the
/// padded layout of their weight.
std::vector<std::unique_ptr<refused_t>> any_n_cases() {
    std::vector<std::unique_ptr<refused_t>> v;
    uint32_t seed = 6000;
    v.push_back(
            make_case("bf16 N 48", "", bf16, bf16, bf16, 4, 48, 64, seed += 4));
    v.push_back(make_case("bf16 M 1 N 3 on an [N, K] weight", "", bf16, bf16,
            f32, 1, 3, 200, seed += 4));
    v.back()->c.transB = true;
    v.push_back(make_case(
            "bf16 M 1 N 100", "", bf16, bf16, bf16, 1, 100, 128, seed += 4));
    v.push_back(
            make_case("u8 x s8 N 40", "", u8, s8, f32, 4, 40, 64, seed += 4));
    v.push_back(make_case("f32 N 40", "", f32, f32, f32, 4, 40, 64, seed += 4));
    return v;
}

/// The refused calls this suite builds: one per class matmul_direct passes
/// on to the native BRGEMM.
std::vector<std::unique_ptr<refused_t>> refused_cases() {
    std::vector<std::unique_ptr<refused_t>> v;
    uint32_t seed = 7000;
    v.push_back(make_case("u8 x s8 with a weight zero point",
            "weight zero points", u8, s8, f32, 4, 64, 64, seed += 4));
    {
        refused_t &r = *v.back();
        r.c.params.quant_params.wei_zp.buff = &r.zp;
        r.c.params.quant_params.wei_zp.dt = s32;
        r.c.params.quant_params.wei_zp.dims = {1};
    }
    v.push_back(make_case("bf16 x s4 without dynamic quantization",
            "s4 weight with a source other than s8", bf16, s4, bf16, 4, 64, 64,
            seed += 4));
    v.push_back(make_case("bf16 x u4",
            "weight type other than bf16 / s8 / s4 / f32 / f16", bf16, u4, bf16,
            4, 64, 64, seed += 4));
    {
        refused_t &r = *v.back();
        r.c.params.quant_params.wei_zp.buff = &r.zp;
        r.c.params.quant_params.wei_zp.dt = s8;
        r.c.params.quant_params.wei_zp.dims = {1, 1};
    }
    v.push_back(make_case("bf16 x s8 without dynamic quantization",
            "s8 weight with a source other than u8 / s8", bf16, s8, bf16, 4, 64,
            64, seed += 4));
    v.back()->c.params.quant_params.src_scale = {};
    v.back()->kept = "W8A16";
    v.push_back(make_case("bf16 quantized to s8 with a source zero point",
            "a source the call quantizes to s8 with a source zero point", bf16,
            s8, f32, 4, 64, 64, seed += 4));
    {
        refused_t &r = *v.back();
        r.c.params.dynamic_quant = true;
        r.c.params.dtypes.compute = s8;
        r.c.params.quant_params.src_zp.buff = &r.zp;
        r.c.params.quant_params.src_zp.dt = s32;
        r.c.params.quant_params.src_zp.dims = {1, 1};
        r.kept = "a source quantized to s8 with a source zero point";
    }
    v.push_back(make_case("u8 x s8 with K past the int32 bound",
            "K exceeds the int32 accumulation bound", u8, s8, f32, 1, 32,
            nat::kMaxSafeInt8AccumK + 4, seed += 4));
    v.push_back(make_case("u8 x s8 in a GGML pack", "a GGML pack format", u8,
            s8, f32, 4, 64, 64, seed += 4));
    v.back()->c.params.packing.pack_format_b = 1;
    if (!ZENDNNL_DEPENDS_AOCLDLP)
        v.back()->kept = "a call on a reordered or GGML packed weight";
    v.push_back(make_case("bf16 with a reordered source", "a reordered source",
            bf16, bf16, bf16, 4, 64, 64, seed += 4));
    v.back()->c.params.mem_format_a = 'r';
    v.back()->kept = "a call with a reordered source";
    return v;
}

/// matmul_direct of c with layout layout and its batch counts, its stdout
/// in log.
status_t run_logged(const nt::call_t &c, std::string &log, char layout = 'r',
        int batch_a = 1, int batch_b = 1) {
    mm::matmul_batch_params_t batch;
    batch.Batch_A = batch_a;
    batch.Batch_B = batch_b;
    mm::matmul_gated_act_params gated;
    gated.act = c.gated;
    ::testing::internal::CaptureStdout();
    const status_t st = mm::matmul_direct(layout, c.transA, c.transB, c.M, c.N,
            c.K, c.alpha, c.src, c.lda_or_tight(), c.wei, c.ldb_or_tight(),
            c.bias, c.beta, c.dst, c.ldc_or_tight(), c.is_weights_const, batch,
            c.params,
            c.gated == mm::matmul_gated_act_t::none ? nullptr : &gated);
    log = ::testing::internal::GetCapturedStdout();
    return st;
}

/// The status native_brgemm_refusal gives c with layout layout, with reason
/// its reason.
status_t refusal_in(const nt::call_t &c, char layout, std::string &reason) {
    mm::matmul_batch_params_t batch;
    mm::matmul_gated_act_params gated;
    gated.act = c.gated;
    const char *why = nullptr;
    const status_t st = mm::native_brgemm_refusal(layout, c.transA, c.transB,
            c.M, c.N, c.K, c.alpha, c.src, c.lda_or_tight(), c.wei,
            c.ldb_or_tight(), c.bias, c.beta, c.dst, c.ldc_or_tight(),
            c.is_weights_const, batch, c.params,
            c.gated == mm::matmul_gated_act_t::none ? nullptr : &gated, &why);
    reason = why != nullptr ? why : "";
    return st;
}

/// The lines of log that hold text.
int lines_with(const std::string &log, const std::string &text) {
    int n = 0;
    for (size_t at = log.find(text); at != std::string::npos;
            at = log.find(text, at + text.size()))
        ++n;
    return n;
}

/// The line matmul_direct logs once per reason for the calls of
/// ZENDNNL_MATMUL_ALGO=10 the native BRGEMM hands to the default route.
std::string default_route_line(const std::string &reason) {
    return "the native BRGEMM does not serve (" + reason
            + ") takes the default route";
}

// Each class, per call (lowoha_algo native_brgemm): native_brgemm_refusal
// gives its status and names its reason, and matmul_direct returns that
// status, with dst holding its values, the weight its bytes, no native
// layout of the weight, and an error line naming the reason
// (run_matmul_direct) and that no other route takes the call.
TEST(NativeBrgemmRefusal, EveryClassReturnsItsStatus) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    for (const auto &rp : refused_cases()) {
        refused_t &r = *rp;
        SCOPED_TRACE(std::string(r.name) + ", lowoha_algo native_brgemm");
        r.c.params.lowoha_algo = matmul_algo_t::native_brgemm;
        const std::vector<uint8_t> raw(static_cast<uint8_t *>(r.wei.get()),
                static_cast<uint8_t *>(r.wei.get()) + r.wei.bytes);
        std::string reason;
        EXPECT_EQ(nt::refusal_of(r.c, &reason), r.status);
        EXPECT_NE(reason.find(r.reason), std::string::npos) << reason;
        EXPECT_TRUE(nt::refused_with(r.c, r.status));
        std::string log;
        EXPECT_EQ(run_logged(r.c, log), r.status);
        if (apilog_error_enabled()) {
            EXPECT_EQ(lines_with(log,
                              "lowoha_algo native_brgemm takes no other route"),
                    1)
                    << log;
        }
        EXPECT_EQ(lines_with(log, "takes the default route"), 0) << log;
        EXPECT_EQ(std::memcmp(r.dst.get(), r.prior.get(), r.dst.bytes), 0);
        EXPECT_EQ(std::memcmp(r.wei.get(), raw.data(), raw.size()), 0);
        EXPECT_EQ(nt::native_layouts(r.wei.get()), 0u);
    }
}

/// The values of r's dst that differ from the f64 model of its call (an
/// s8 weight with a u8 source and an optional weight zero point, an s8
/// weight with a bf16 source it does not quantize, an s4 weight, or a u4
/// weight with a zero point, each with r's scales and r's binary_add
/// operand) by more than the rounding of the dst, described; "no model" for
/// any other class.
std::string model_mismatches(const refused_t &r) {
    const nt::call_t &c = r.c;
    const auto &q = c.params.quant_params;
    const data_type_t src_dt = c.params.dtypes.src,
                      wei_dt = c.params.dtypes.wei,
                      dst_dt = c.params.dtypes.dst;
    const bool int_src = src_dt == u8;
    if (c.transA || c.transB || r.layout != 'r' || c.params.dynamic_quant
            || c.params.mem_format_a != 'n' || c.params.packing.pack_format_b
            || (int_src ? wei_dt != s8
                        : src_dt != bf16
                                    || (wei_dt != s8 && wei_dt != s4
                                            && wei_dt != u4)))
        return "no model";
    const auto *nibbles = static_cast<const uint8_t *>(c.wei);
    const auto weight = [&](int k, int n) -> double {
        const size_t i = static_cast<size_t>(k) * c.N + n;
        if (wei_dt == s8) return static_cast<const int8_t *>(c.wei)[i];
        const int nib = (nibbles[i / 2] >> (i % 2 * 4)) & 0xF;
        return wei_dt == s4 ? (nib ^ 8) - 8 : nib;
    };
    double zp = 0.0;
    if (q.wei_zp.buff != nullptr)
        zp = q.wei_zp.dt == s8 ? *static_cast<const int8_t *>(q.wei_zp.buff)
                               : *static_cast<const int32_t *>(q.wei_zp.buff);
    const double src_scale = int_src ? r.src_scale : 1.0;
    std::string out;
    for (int m = 0; m < c.M; ++m)
        for (int n = 0; n < c.N; ++n) {
            double want = 0.0, mag = 0.0;
            for (int k = 0; k < c.K; ++k) {
                const double t = nt::load_f32(c.src, src_dt,
                                         static_cast<size_t>(m) * c.K + k)
                        * (weight(k, n) - zp);
                want += t;
                mag += std::fabs(t);
            }
            const double scale = src_scale * r.wei_scales[n];
            want *= scale;
            mag *= scale;
            if (!r.operand.empty()) {
                const double v
                        = r.operand[static_cast<size_t>(m) * r.operand_ld + n];
                want += v;
                mag += std::fabs(v);
            }
            const size_t i = static_cast<size_t>(m) * c.N + n;
            const double got = nt::load_f32(c.dst, dst_dt, i);
            const double tol = (dst_dt == bf16 ? std::fabs(want) / 128 : 0.0)
                    + 1e-5 * mag + 1e-6;
            if (std::fabs(got - want) > tol && out.size() < 400)
                out += " (" + std::to_string(m) + ", " + std::to_string(n)
                        + "): " + std::to_string(got) + " for "
                        + std::to_string(want);
        }
    return out;
}

// The same classes under ZENDNNL_MATMUL_ALGO=10, without lowoha_algo: a
// class that returns success there has the dst of the f64 model of its call
// (model_mismatches), and a class the default route has no such dst for
// keeps its error (WhatTheDefaultRouteComputesWronglyKeepsItsError).  The
// native BRGEMM hands each other class to the default route
// (native_brgemm_refusal gives success), and matmul_direct returns the
// status and the dst of the same call with the algorithm unset, with no
// native layout of the weight and the weight's bytes as they were.
TEST(NativeBrgemmRefusal, UnderTheProcessAlgoEveryClassTakesTheDefaultRoute) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    for (const auto &rp : refused_cases()) {
        refused_t &r = *rp;
        SCOPED_TRACE(r.name);
        const std::vector<uint8_t> raw(static_cast<uint8_t *>(r.wei.get()),
                static_cast<uint8_t *>(r.wei.get()) + r.wei.bytes);
        nt::buffer_t unset = nt::make_buffer(r.dst.bytes);
        std::memcpy(unset.get(), r.prior.get(), r.dst.bytes);
        status_t unset_status = status_t::failure;
        {
            const nt::algo_pin_t pin(matmul_algo_t::none);
            nt::call_t c = r.c;
            c.dst = unset.get();
            unset_status = nt::run_matmul(c);
        }
        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
        std::string reason;
        const status_t refusal = nt::refusal_of(r.c, &reason);
        const status_t st = nt::run_matmul(r.c);
        if (st == status_t::success) {
            EXPECT_EQ(model_mismatches(r), "")
                    << "dst differs from the model of the call";
        }
        if (r.kept == nullptr) {
            EXPECT_EQ(refusal, status_t::success) << reason;
            EXPECT_EQ(st, unset_status);
            EXPECT_EQ(std::memcmp(r.dst.get(), unset.get(), r.dst.bytes), 0)
                    << "dst differs from the default route's";
        }
        EXPECT_EQ(std::memcmp(r.wei.get(), raw.data(), raw.size()), 0);
        EXPECT_EQ(nt::native_layouts(r.wei.get()), 0u);
        mm::clear_matmul_weight_caches();
    }
}

// One warning line per reason and process: in a process of its own, two
// classes run in turn from four threads, then every class twice and a
// batched call twice under ZENDNNL_MATMUL_ALGO=10, log one line each that
// the call takes the default route and why; the classes that keep their
// error log none.  A column-major call of K below N with the
// tight ldb, which the layout rule refuses for a row-major ldb, logs "a
// column-major call" (in a build without AOCL-DLP it keeps its error, whose
// line names that reason), and no line names the ldb.
TEST(NativeBrgemmRefusal, TheDefaultRouteIsLoggedOncePerReason) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    if (!apilog_warning_enabled()) GTEST_SKIP() << "API warnings are off";
    nt::expect_in_child({}, [] {
        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
        auto cases = refused_cases();
        cases.push_back(make_case("bf16 column-major of K below N",
                "a column-major call", bf16, bf16, f32, 4, 128, 64, 7304));
        {
            refused_t &r = *cases.back();
            r.layout = 'c';
            r.c.lda = r.c.M;
            r.c.ldb = r.c.K;
            r.c.ldc = r.c.M;
            r.c.is_weights_const = false;
        }
        auto batched = make_case("bf16 batched", "a batched call", bf16, bf16,
                bf16, 4, 64, 64, 7300);
        nt::buffer_t two = nt::make_buffer(2 * batched->src.bytes);
        std::memcpy(two.as<uint8_t>(), batched->src.get(), batched->src.bytes);
        std::memcpy(two.as<uint8_t>() + batched->src.bytes, batched->src.get(),
                batched->src.bytes);
        nt::buffer_t two_dst = nt::make_buffer(2 * batched->dst.bytes);
        batched->c.src = two.get();
        batched->c.dst = two_dst.get();
        // First, four threads that run the s4 and the u4 class in turn, each
        // into a dst of its own.
        ::testing::internal::CaptureStdout();
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; ++t)
            threads.emplace_back([&cases] {
                nt::buffer_t own = nt::make_buffer(cases[1]->dst.bytes);
                for (int i = 0; i < 50; ++i) {
                    nt::call_t c = cases[1 + i % 2]->c;
                    c.dst = own.get();
                    (void)nt::run_matmul(c);
                }
            });
        for (std::thread &t : threads)
            t.join();
        std::string log = ::testing::internal::GetCapturedStdout();
        for (int round = 0; round < 2; ++round) {
            for (const auto &rp : cases) {
                std::string one;
                (void)run_logged(rp->c, one, rp->layout);
                log += one;
            }
            std::string one;
            if (run_logged(batched->c, one, 'r', 2, 1) != status_t::success) {
                std::fprintf(stderr, "the batched call failed\n");
                return 1;
            }
            log += one;
        }
        int bad = 0;
        std::vector<const char *> reasons {"weight zero points",
                "s4 weight with a source other than s8",
                "weight type other than bf16 / s8 / s4 / f32 / f16",
                "K exceeds the int32 accumulation bound", "a batched call"};
        if (ZENDNNL_DEPENDS_AOCLDLP) {
            reasons.push_back("a GGML pack format");
            reasons.push_back("a column-major call");
        } else if (apilog_error_enabled()
                && lines_with(log,
                           "did not serve the call (a column-major call); "
                           "not handed back")
                        != 2) {
            std::fprintf(
                    stderr, "no kept column-major line:\n%s\n", log.c_str());
            ++bad;
        }
        for (const char *reason : reasons) {
            const int n = lines_with(log, default_route_line(reason));
            if (n != 1) {
                std::fprintf(stderr, "%d lines for \"%s\"\n", n, reason);
                ++bad;
            }
        }
        if (lines_with(log, "ldb below the weight row length") != 0) {
            std::fprintf(stderr, "a line names the ldb:\n%s\n", log.c_str());
            ++bad;
        }
        if (lines_with(log, "takes the default route")
                != static_cast<int>(reasons.size())) {
            std::fprintf(stderr, "default-route lines:\n%s\n", log.c_str());
            ++bad;
        }
        return bad;
    }, "one default-route line per reason");
}

// Calls whose arguments are wrong keep their status under
// ZENDNNL_MATMUL_ALGO=10: an ldb below the weight row (unimplemented) and an
// lda below K (failure, the native BRGEMM's check of a bad shape or leading
// dimension), with dst untouched and no default-route line.  So do an ldc
// below N and an lda below K of a call the layout rule refuses before
// those checks (u8 x s8 of K past the int32 bound).
TEST(NativeBrgemmRefusal, WrongArgumentsKeepTheirStatus) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const int long_k = nat::kMaxSafeInt8AccumK + 4;
    for (int form = 0; form < 4; ++form) {
        const bool short_ldb = form == 0, early = form >= 2;
        auto rp = early
                ? make_case(form == 2
                                  ? "u8 x s8 of K past the int32 bound, ldc "
                                    "below N"
                                  : "u8 x s8 of K past the int32 bound, lda "
                                    "below K",
                          "a bad shape or leading dimension", u8, s8, f32, 2,
                          32, long_k, 7208 + form)
                : make_case(short_ldb ? "bf16 ldb below N" : "bf16 lda below K",
                          short_ldb ? "ldb below the weight row length"
                                    : "a bad shape or leading dimension",
                          bf16, bf16, bf16, 4, 64, 64,
                          7200 + (short_ldb ? 0 : 4));
        refused_t &r = *rp;
        SCOPED_TRACE(r.name);
        if (short_ldb)
            r.c.ldb = 48;
        else if (form == 2)
            r.c.ldc = 16;
        else
            r.c.lda = r.c.K - (early ? 1 : 32);
        const status_t want
                = short_ldb ? status_t::unimplemented : status_t::failure;
        std::string reason;
        EXPECT_EQ(nt::refusal_of(r.c, &reason), want);
        EXPECT_EQ(reason, r.reason);
        EXPECT_TRUE(nt::refused_with(r.c, want));
        std::string log;
        EXPECT_EQ(run_logged(r.c, log), want);
        EXPECT_EQ(lines_with(log, "takes the default route"), 0) << log;
        EXPECT_EQ(std::memcmp(r.dst.get(), r.prior.get(), r.dst.bytes), 0);
    }
}

// A call matmul_direct's checks take only for the native BRGEMM (f16 weight
// scales of W4A8) that the native BRGEMM hands back (a gated call with beta
// over a dst of N / 2 columns)
// fails the default route's checks under ZENDNNL_MATMUL_ALGO=10: no route
// reads its scales, and dst keeps its values.  With lowoha_algo
// native_brgemm it is the native BRGEMM's refusal.
TEST(NativeBrgemmRefusal, TheDefaultRouteChecksWhatOnlyTheNativeRulesTake) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    const int M = 4, N = 64, K = 128, G = 2;
    const nt::buffer_t src = nt::random_buffer(
            bf16, static_cast<size_t>(M) * K, 7500, -1.0f, 1.0f);
    const nt::buffer_t wei = nt::kept_until_exit(nt::random_buffer(
            u8, static_cast<size_t>(N) * K / 2, 7501, 0, 255));
    const nt::buffer_t prior = nt::random_buffer(
            bf16, static_cast<size_t>(M) * N, 7502, -1.0f, 1.0f);
    nt::buffer_t dst = nt::make_buffer(prior.bytes);
    std::vector<uint16_t> wei_scales(static_cast<size_t>(G) * N, 0x2000);
    std::vector<uint16_t> src_scales(M, 0);
    for (const bool per_call : {false, true}) {
        SCOPED_TRACE(per_call ? "lowoha_algo native_brgemm"
                              : "ZENDNNL_MATMUL_ALGO=10");
        const nt::algo_pin_t pin(
                per_call ? matmul_algo_t::none : matmul_algo_t::native_brgemm);
        std::memcpy(dst.get(), prior.get(), prior.bytes);
        nt::call_t c;
        c.M = M;
        c.N = N;
        c.K = K;
        c.transB = true;
        c.src = src.get();
        c.wei = wei.get();
        c.dst = dst.get();
        c.params.dtypes.src = bf16;
        c.params.dtypes.wei = s4;
        c.params.dtypes.dst = bf16;
        c.params.dtypes.compute = s8;
        c.params.dynamic_quant = true;
        c.params.weight_cache_type = 1;
        c.gated = mm::matmul_gated_act_t::silu_and_mul;
        c.beta = 0.5f;
        if (per_call) c.params.lowoha_algo = matmul_algo_t::native_brgemm;
        auto &q = c.params.quant_params;
        q.wei_scale.buff = wei_scales.data();
        q.wei_scale.dt = data_type_t::f16;
        q.wei_scale.dims = {G, N};
        q.src_scale.buff = src_scales.data();
        q.src_scale.dt = bf16;
        q.src_scale.dims = {M, 1};
        std::string log;
        EXPECT_EQ(run_logged(c, log),
                per_call ? status_t::unimplemented : status_t::failure);
        if (commonlog_error_enabled()) {
            EXPECT_EQ(lines_with(log,
                              "W4A8 weight scale supports only f32 or "
                              "bf16 data type"),
                    per_call ? 0 : 1)
                    << log;
        }
        EXPECT_EQ(std::memcmp(dst.get(), prior.get(), prior.bytes), 0);
        EXPECT_EQ(nt::native_layouts(wei.get()), 0u);
    }
}

// Weight cache mode 2 under ZENDNNL_MATMUL_ALGO=10: once a call the native
// BRGEMM serves has rewritten a bf16 weight in place, a call on that buffer
// it does not serve (a reordered source, a column-major call, a batched
// call) fails, dst untouched, with an error line saying no other route reads
// the buffer: the default route never reads the image as a raw weight.
TEST(NativeBrgemmRefusal, ABufferRewrittenInPlaceHasNoOtherReader) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    WeightCacheGuard global_cache(2);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    auto rp = make_case("bf16 in place", "", bf16, bf16, bf16, 4, 64, 64, 7600);
    refused_t &r = *rp;
    r.c.params.weight_cache_type = 2;
    ASSERT_EQ(nt::run_matmul(r.c), status_t::success);
    ASSERT_TRUE(nat::native_weight_packed_in_place(r.wei.get(), r.wei.bytes))
            << "the first call did not rewrite its weight in place";
    nt::buffer_t two = nt::make_buffer(2 * r.src.bytes);
    nt::buffer_t two_dst = nt::make_buffer(2 * r.dst.bytes);
    struct kind_t {
        const char *name;
        char layout;
        int batch_a;
        char mem_format_a;
    };
    for (const kind_t &k : {kind_t {"a reordered source", 'r', 1, 'r'},
                 kind_t {"a column-major call", 'c', 1, 'n'},
                 kind_t {"a batched call", 'r', 2, 'n'}}) {
        SCOPED_TRACE(k.name);
        nt::call_t c = r.c;
        c.params.mem_format_a = k.mem_format_a;
        nt::buffer_t out = k.batch_a == 2 ? two_dst : r.dst;
        if (k.batch_a == 2) c.src = two.get();
        c.dst = out.get();
        std::vector<uint8_t> before(
                out.as<uint8_t>(), out.as<uint8_t>() + out.bytes);
        std::string log;
        EXPECT_EQ(
                run_logged(c, log, k.layout, k.batch_a, 1), status_t::failure);
        if (apilog_error_enabled()) {
            EXPECT_EQ(lines_with(log, k.name), 1) << log;
            EXPECT_EQ(
                    lines_with(log, "no other route reads a buffer it rewrote"),
                    1)
                    << log;
        }
        EXPECT_EQ(lines_with(log, "takes the default route"), 0) << log;
        EXPECT_EQ(std::memcmp(out.get(), before.data(), out.bytes), 0);
    }
    mm::clear_matmul_weight_caches();
}

// The f32 column-major call on a constant weight at weight cache mode 2,
// which the default route rewrites in place with the algorithm unset and
// leaves its dst unwritten, keeps its error under ZENDNNL_MATMUL_ALGO=10:
// the weight stays raw, and a later call on the buffer that the native
// BRGEMM serves reads the raw weight, with the bits of the same call on a
// fresh copy of the weight.
TEST(NativeBrgemmRefusal, AHandedBackCallLeavesTheWeightRaw) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    WeightCacheGuard global_cache(2);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const int M = 128, N = 256, K = 256;
    auto rp = make_case("f32 column-major", "a column-major call", f32, f32,
            f32, M, N, K, 7700);
    refused_t &r = *rp;
    const std::vector<uint8_t> raw(
            r.wei.as<uint8_t>(), r.wei.as<uint8_t>() + r.wei.bytes);
    nt::call_t c = r.c;
    c.params.weight_cache_type = 2;
    c.lda = M;
    c.ldb = K;
    c.ldc = M;
    std::string log;
    EXPECT_EQ(run_logged(c, log, 'c'), status_t::unimplemented);
    EXPECT_EQ(std::memcmp(r.wei.get(), raw.data(), raw.size()), 0)
            << "the default route rewrote the weight in place";
    EXPECT_EQ(nt::native_layouts(r.wei.get()), 0u);

    // A row-major call on the buffer and on a fresh copy of the weight: the
    // native BRGEMM serves both.
    nt::call_t served = r.c;
    served.params.weight_cache_type = 2;
    EXPECT_EQ(nt::run_matmul(served), status_t::success);
    EXPECT_GE(nt::native_layouts(r.wei.get()), 1u);
    nt::buffer_t fresh = nt::kept_until_exit(nt::make_buffer(r.wei.bytes));
    std::memcpy(fresh.get(), raw.data(), raw.size());
    nt::buffer_t want = nt::make_buffer(r.dst.bytes);
    nt::call_t on_fresh = r.c;
    on_fresh.wei = fresh.get();
    on_fresh.dst = want.get();
    EXPECT_EQ(nt::run_matmul(on_fresh), status_t::success);
    EXPECT_EQ(std::memcmp(r.dst.get(), want.get(), r.dst.bytes), 0)
            << "the native BRGEMM read a rewritten weight";
    mm::clear_matmul_weight_caches();
}

// A call handed back under ZENDNNL_MATMUL_ALGO=10 at weight cache mode 2
// reads its raw weight while no in-place rewrite of it runs.  A column-major
// bf16 call over a weight it marks as changing, long enough to read the
// weight for a while, and a first native call on the same buffer started up
// to 300 us after it, which rewrites the buffer in place: the column-major
// call either reads the raw weight, with the result of the same call on a
// private copy, or finds the image and fails; it never returns success with
// a result read in part from the image.  The native call has the result of
// the same call on a private copy every time.
TEST(NativeBrgemmRefusal, AHandedBackCallReadsNoBufferARewriteWrites) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    if (!ZENDNNL_DEPENDS_AOCLDLP)
        GTEST_SKIP() << "the default route of a build without AOCL-DLP "
                        "computes no column-major call";
    WeightCacheGuard global_cache(2);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const int K = 4096, N = 4096, MA = 32, MB = 256, threads = 8, trials = 8;
    const size_t w_elems = static_cast<size_t>(K) * N;
    const nt::buffer_t w0 = nt::random_buffer(bf16, w_elems, 8100, -1.0f, 1.0f);
    const nt::buffer_t src_a = nt::random_buffer(
            bf16, static_cast<size_t>(MA) * K, 8101, -1.0f, 1.0f);
    const nt::buffer_t src_b = nt::random_buffer(
            bf16, static_cast<size_t>(MB) * K, 8102, -1.0f, 1.0f);
    const auto params = [&](int wc) {
        mm::matmul_params p;
        p.dtypes.src = bf16;
        p.dtypes.wei = bf16;
        p.dtypes.dst = f32;
        p.weight_cache_type = wc;
        p.num_threads = threads;
        return p;
    };
    // The native call: row-major [K, N], constant, so its first call at mode
    // 2 rewrites the buffer in place.
    const auto native_call = [&](const void *w, void *dst) {
        mm::matmul_batch_params_t batch;
        mm::matmul_params p = params(2);
        return mm::matmul_direct('r', false, false, MA, N, K, 1.0f, src_a.get(),
                K, w, N, nullptr, 0.0f, dst, N, true, batch, p, nullptr);
    };
    // The handed-back call: column-major, the same bytes as an [N, K]
    // weight of ld N under transB.
    const auto handed_back = [&](const void *w, void *dst) {
        mm::matmul_batch_params_t batch;
        mm::matmul_params p = params(2);
        return mm::matmul_direct('c', false, true, MB, N, K, 1.0f, src_b.get(),
                MB, w, N, nullptr, 0.0f, dst, MB, false, batch, p, nullptr);
    };
    const size_t bytes_a = static_cast<size_t>(MA) * N * sizeof(float);
    const size_t bytes_b = static_cast<size_t>(MB) * N * sizeof(float);
    nt::buffer_t want_a = nt::make_buffer(bytes_a);
    nt::buffer_t want_b = nt::make_buffer(bytes_b);
    {
        nt::buffer_t own_a = nt::make_buffer(w_elems * 2);
        nt::buffer_t own_b = nt::make_buffer(w_elems * 2);
        std::memcpy(own_a.get(), w0.get(), w_elems * 2);
        std::memcpy(own_b.get(), w0.get(), w_elems * 2);
        ASSERT_EQ(handed_back(own_b.get(), want_b.get()), status_t::success);
        ASSERT_EQ(native_call(own_a.get(), want_a.get()), status_t::success);
        mm::clear_matmul_weight_caches();
    }
    const auto far_off = [&](const nt::buffer_t &got) {
        float err = 0.0f, mag = 0.0f;
        for (int i = 0; i < MB * N; ++i) {
            err = std::max(
                    err, std::fabs(got.as<float>()[i] - want_b.as<float>()[i]));
            mag = std::max(mag, std::fabs(want_b.as<float>()[i]));
        }
        return err > 1e-2f * mag;
    };
    std::mt19937 rng(8103);
    std::uniform_int_distribution<int> delay_us(0, 300);
    int read_raw = 0, found_image = 0, wrong = 0;
    for (int t = 0; t < trials; ++t) {
        const int delay = delay_us(rng);
        nt::buffer_t shared = nt::make_buffer(w_elems * 2);
        std::memcpy(shared.get(), w0.get(), w_elems * 2);
        nt::buffer_t dst_a = nt::make_buffer(bytes_a);
        nt::buffer_t dst_b = nt::make_buffer(bytes_b);
        status_t st_a = status_t::failure, st_b = status_t::failure;
        std::atomic<int> ready {0};
        const auto start = [&ready] {
            ready.fetch_add(1);
            while (ready.load() < 2) {}
        };
        std::thread tb([&] {
            start();
            st_b = handed_back(shared.get(), dst_b.get());
        });
        std::thread ta([&] {
            start();
            std::this_thread::sleep_for(std::chrono::microseconds(delay));
            st_a = native_call(shared.get(), dst_a.get());
        });
        tb.join();
        ta.join();
        EXPECT_EQ(st_a, status_t::success) << "trial " << t;
        EXPECT_EQ(std::memcmp(dst_a.get(), want_a.get(), bytes_a), 0)
                << "trial " << t;
        if (st_b != status_t::success)
            ++found_image;
        else if (far_off(dst_b))
            ++wrong;
        else
            ++read_raw;
        mm::clear_matmul_weight_caches();
    }
    EXPECT_EQ(wrong, 0) << "of " << trials << " handed-back calls, " << wrong
                        << " returned success with a result read in part "
                           "from the image ("
                        << read_raw << " read the raw weight, " << found_image
                        << " found the image)";
}

// The hold of a handed-back call spans its weight as the caller lays it
// out: a column-major bf16 call of K 32 and N 256 with ldb 32, not
// transposed, reads N columns of K elements, 16 KiB from its weight, and so
// does an f16 call on a row-major [N, K] weight of ldb 32 (transB).  While
// its hold is kept, a first native call at weight cache mode 2 on a
// row-major 32 x 32 weight 4 KiB into the buffer waits to rewrite it in
// place, and every byte of the span reads as it was; once the hold ends,
// the native call rewrites the 32 x 32 weight in place, with the result of
// the same call on a private copy.
TEST(NativeBrgemmRefusal, AHandedBackCallHoldsItsWeightAsTheCallerLaysItOut) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    WeightCacheGuard global_cache(2);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const int K = 32, N = 256, ldb = 32, VK = 32, VN = 32, VM = 4;
    const size_t span = static_cast<size_t>(N) * ldb * 2, offset = 4096;
    const nt::buffer_t raw
            = nt::random_buffer(bf16, span / 2, 8300, -1.0f, 1.0f);
    const nt::buffer_t src = nt::random_buffer(
            bf16, static_cast<size_t>(VM) * VK, 8301, -1.0f, 1.0f);
    const size_t dst_bytes = static_cast<size_t>(VM) * VN * sizeof(float);
    const auto native_call = [&](const void *w, void *dst) {
        mm::matmul_batch_params_t batch;
        mm::matmul_params p;
        p.dtypes.src = bf16;
        p.dtypes.wei = bf16;
        p.dtypes.dst = f32;
        p.weight_cache_type = 2;
        p.num_threads = 1;
        return mm::matmul_direct('r', false, false, VM, VN, VK, 1.0f, src.get(),
                VK, w, VN, nullptr, 0.0f, dst, VN, true, batch, p, nullptr);
    };
    nt::buffer_t want = nt::make_buffer(dst_bytes);
    {
        nt::buffer_t own = nt::kept_until_exit(nt::make_buffer(span));
        std::memcpy(own.get(), raw.get(), span);
        ASSERT_EQ(native_call(own.as<uint8_t>() + offset, want.get()),
                status_t::success);
    }
    for (const data_type_t wei_dt : {bf16, data_type_t::f16}) {
        SCOPED_TRACE(dtype_info(wei_dt));
        const bool transB = wei_dt != bf16;
        nt::buffer_t shared = nt::kept_until_exit(nt::make_buffer(span));
        std::memcpy(shared.get(), raw.get(), span);
        mm::matmul_params handed_back;
        handed_back.dtypes.src = wei_dt;
        handed_back.dtypes.wei = wei_dt;
        handed_back.dtypes.dst = f32;
        handed_back.weight_cache_type = 2;
        auto hold = std::make_unique<nat::native_weight_raw_hold_t>();
        ASSERT_EQ(nat::native_hold_handed_back_weight(*hold, transB ? 'r' : 'c',
                          shared.get(), K, N, ldb, transB, handed_back),
                status_t::success);
        nt::buffer_t dst = nt::make_buffer(dst_bytes);
        std::atomic<bool> done {false};
        status_t st = status_t::failure;
        std::thread native([&] {
            st = native_call(shared.as<uint8_t>() + offset, dst.get());
            done.store(true);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::vector<uint8_t> read(span);
        std::memcpy(read.data(), shared.get(), span);
        const bool waited = !done.load();
        hold.reset();
        native.join();
        EXPECT_TRUE(waited) << "the native call did not wait for the hold";
        EXPECT_EQ(std::memcmp(read.data(), raw.get(), span), 0)
                << "an in-place rewrite wrote bytes the hold holds";
        EXPECT_EQ(st, status_t::success);
        EXPECT_TRUE(nat::native_weight_packed_in_place(
                shared.as<uint8_t>() + offset,
                static_cast<size_t>(VK) * VN * 2))
                << "the native call did not rewrite its weight in place";
        EXPECT_EQ(std::memcmp(dst.get(), want.get(), dst_bytes), 0);
        mm::clear_matmul_weight_caches();
    }
}

// The hold of a handed-back call spans a raw weight of any type as the
// caller lays it out, two u4 elements a byte and two bytes an f16 element.
// A first native call at weight cache mode 2 rewrites a 64 x 64 bf16
// weight 64 KiB into a buffer in place; then the hold of a call on a
// [K, N] weight of 128 KiB at the start of the buffer, u4 or f16, finds
// the image and fails, and so does that call through matmul_direct under
// ZENDNNL_MATMUL_ALGO=10 (W4A16 with a u4 weight, and f16 x f16 on a CPU
// with AVX512-FP16), with dst untouched.  With the native call started up
// to 300 us after a W4A16 call on the same buffer, the W4A16 call reads its
// raw weight, with the result of the same call on a private copy, or
// fails; it never returns success with a result read in part from the
// image.
TEST(NativeBrgemmRefusal, AHandedBackCallOfAnyWeightTypeHoldsAllItsBytes) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    WeightCacheGuard global_cache(2);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const int N = 1024, VK = 64, VN = 64, VM = 4, MB = 256, trials = 8;
    const size_t span = static_cast<size_t>(128) << 10,
                 offset = static_cast<size_t>(64) << 10;
    const nt::buffer_t raw = nt::random_buffer(u8, span, 8600, 0.0f, 255.0f);
    const nt::buffer_t src_a = nt::random_buffer(
            bf16, static_cast<size_t>(VM) * VK, 8601, -1.0f, 1.0f);
    const std::vector<float> wei_scales(N, 0.25f);
    const int32_t zp = 8;
    std::vector<float> dst_a(static_cast<size_t>(VM) * VN);
    const auto native_call = [&](uint8_t *base) {
        mm::matmul_batch_params_t batch;
        mm::matmul_params p;
        p.dtypes.src = bf16;
        p.dtypes.wei = bf16;
        p.dtypes.dst = f32;
        p.weight_cache_type = 2;
        return mm::matmul_direct('r', false, false, VM, VN, VK, 1.0f,
                src_a.get(), VK, base + offset, VN, nullptr, 0.0f, dst_a.data(),
                VN, true, batch, p, nullptr);
    };
    // The handed-back call: bf16 x u4 with per-column scales and a zero
    // point, or f16 x f16, into f32, on a [K, N] weight of span bytes.
    const auto params = [&](data_type_t wei_dt) {
        mm::matmul_params p;
        p.dtypes.src = wei_dt == u4 ? bf16 : data_type_t::f16;
        p.dtypes.wei = wei_dt;
        p.dtypes.dst = f32;
        p.weight_cache_type = 2;
        if (wei_dt == u4) {
            auto &q = p.quant_params;
            q.wei_scale.buff = wei_scales.data();
            q.wei_scale.dt = f32;
            q.wei_scale.dims = {1, N};
            q.wei_zp.buff = &zp;
            q.wei_zp.dt = s8;
            q.wei_zp.dims = {1, 1};
        }
        return p;
    };
    const auto handed_back = [&](data_type_t wei_dt, int M, const void *src,
                                     const uint8_t *w, float *dst) {
        const int K = wei_dt == u4 ? 256 : 64;
        mm::matmul_batch_params_t batch;
        return mm::matmul_direct('r', false, false, M, N, K, 1.0f, src, K, w, N,
                nullptr, 0.0f, dst, N, true, batch, params(wei_dt), nullptr);
    };
    const nt::buffer_t src_b = nt::random_buffer(
            bf16, static_cast<size_t>(MB) * 256, 8602, -1.0f, 1.0f);
    const nt::buffer_t src_f16 = nt::random_buffer(
            data_type_t::f16, static_cast<size_t>(MB) * 64, 8603, -1.0f, 1.0f);
    const size_t bytes_b = static_cast<size_t>(MB) * N * sizeof(float);
    for (const data_type_t wei_dt : {u4, data_type_t::f16}) {
        SCOPED_TRACE(dtype_info(wei_dt));
        nt::buffer_t shared = nt::kept_until_exit(nt::make_buffer(span));
        std::memcpy(shared.get(), raw.get(), span);
        ASSERT_EQ(native_call(shared.as<uint8_t>()), status_t::success);
        ASSERT_TRUE(nat::native_weight_packed_in_place(
                shared.as<uint8_t>() + offset,
                static_cast<size_t>(VK) * VN * 2));
        nat::native_weight_raw_hold_t hold;
        EXPECT_EQ(nat::native_hold_handed_back_weight(hold, 'r', shared.get(),
                          wei_dt == u4 ? 256 : 64, N, N, false, params(wei_dt)),
                status_t::failure);
        if (wei_dt != u4
                && !zendnnl::common::zendnnl_platform_info()
                            .get_avx512_f16_status())
            continue;
        std::vector<float> dst(static_cast<size_t>(VM) * N, 123.0f);
        EXPECT_EQ(handed_back(wei_dt, VM,
                          wei_dt == u4 ? src_b.get() : src_f16.get(),
                          shared.as<uint8_t>(), dst.data()),
                status_t::failure);
        EXPECT_TRUE(std::all_of(dst.begin(), dst.end(), [](float v) {
            return v == 123.0f;
        })) << "a call that failed wrote dst";
        mm::clear_matmul_weight_caches();
    }
    nt::buffer_t want = nt::make_buffer(bytes_b);
    {
        nt::buffer_t own = nt::kept_until_exit(nt::make_buffer(span));
        std::memcpy(own.get(), raw.get(), span);
        ASSERT_EQ(handed_back(u4, MB, src_b.get(), own.as<uint8_t>(),
                          want.as<float>()),
                status_t::success);
        mm::clear_matmul_weight_caches();
    }
    std::mt19937 rng(8604);
    std::uniform_int_distribution<int> delay_us(0, 300);
    int read_raw = 0, found_image = 0, wrong = 0;
    for (int t = 0; t < trials; ++t) {
        const int delay = delay_us(rng);
        nt::buffer_t shared = nt::kept_until_exit(nt::make_buffer(span));
        std::memcpy(shared.get(), raw.get(), span);
        nt::buffer_t got = nt::make_buffer(bytes_b);
        status_t st_a = status_t::failure, st_b = status_t::failure;
        std::atomic<int> ready {0};
        const auto start = [&ready] {
            ready.fetch_add(1);
            while (ready.load() < 2) {}
        };
        std::thread tb([&] {
            start();
            st_b = handed_back(
                    u4, MB, src_b.get(), shared.as<uint8_t>(), got.as<float>());
        });
        std::thread ta([&] {
            start();
            std::this_thread::sleep_for(std::chrono::microseconds(delay));
            st_a = native_call(shared.as<uint8_t>());
        });
        tb.join();
        ta.join();
        EXPECT_EQ(st_a, status_t::success) << "trial " << t;
        if (st_b != status_t::success)
            ++found_image;
        else if (std::memcmp(got.get(), want.get(), bytes_b) != 0)
            ++wrong;
        else
            ++read_raw;
        mm::clear_matmul_weight_caches();
    }
    EXPECT_EQ(wrong, 0) << "of " << trials << " handed-back calls, " << wrong
                        << " returned success with a result read in part "
                           "from the image ("
                        << read_raw << " read the raw weight, " << found_image
                        << " found the image)";
}

/// A row-major 32 x 32 bf16 weight at w, constant, on one thread at weight
/// cache mode 2: its first call under ZENDNNL_MATMUL_ALGO=10 rewrites the
/// 2 KiB at w in place.
status_t native_view_call(const void *w, const void *src, float *dst) {
    mm::matmul_batch_params_t batch;
    mm::matmul_params p;
    p.dtypes.src = bf16;
    p.dtypes.wei = bf16;
    p.dtypes.dst = f32;
    p.weight_cache_type = 2;
    p.num_threads = 1;
    return mm::matmul_direct('r', false, false, 4, 32, 32, 1.0f, src, 32, w, 32,
            nullptr, 0.0f, dst, 32, true, batch, p, nullptr);
}

/// While the hold of a handed-back call (of batch, when not null) over the
/// bytes of weight is kept, a first native call on the 32 x 32 bf16 weight
/// offset bytes into them waits to rewrite it in place and every byte of
/// them reads as it was; once the hold ends, it rewrites that weight in
/// place.
void expect_view_waits_for_hold(const mm::matmul_params &params,
        uint8_t *weight, size_t bytes, size_t offset, int K, int N, int ldb,
        bool transB, const mm::matmul_batch_params_t *batch = nullptr) {
    const nt::buffer_t src = nt::random_buffer(
            bf16, static_cast<size_t>(4) * 32, 8701, -1.0f, 1.0f);
    const std::vector<uint8_t> before(weight, weight + bytes);
    auto hold = std::make_unique<nat::native_weight_raw_hold_t>();
    ASSERT_EQ(nat::native_hold_handed_back_weight(
                      *hold, 'r', weight, K, N, ldb, transB, params, batch),
            status_t::success);
    std::vector<float> dst(static_cast<size_t>(4) * 32);
    std::atomic<bool> done {false};
    status_t st = status_t::failure;
    std::thread native([&] {
        st = native_view_call(weight + offset, src.get(), dst.data());
        done.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const std::vector<uint8_t> read(weight, weight + bytes);
    const bool waited = !done.load();
    hold.reset();
    native.join();
    EXPECT_TRUE(waited) << "the native call did not wait for the hold";
    EXPECT_EQ(read, before) << "an in-place rewrite wrote bytes the hold holds";
    EXPECT_EQ(st, status_t::success);
    EXPECT_TRUE(nat::native_weight_packed_in_place(weight + offset, 2048))
            << "the native call did not rewrite its weight in place";
}

// The hold of a handed-back call on an AOCL-DLP prepack spans the prepack
// as the prepack lays it out: a static u8 x s8 call of K 64 and N 512 with
// a source zero point, prepacked under a DLP M hint (no unreorder of it),
// reads N int32 column sums in the last 2 KiB of its prepack, past the
// K x N bytes of its raw weight, with per-tensor weight scales or scales in
// groups of 32.  While its hold is kept, a first native call at weight
// cache mode 2 on a 32 x 32 bf16 weight there waits to rewrite it in
// place.  Once that weight is an image, the call through matmul_direct
// under ZENDNNL_MATMUL_ALGO=10 fails with dst untouched; on a copy of the
// prepack it returns the result of the call's model.
TEST(NativeBrgemmRefusal, AHandedBackCallHoldsAllOfItsPrepack) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    if (!ZENDNNL_DEPENDS_AOCLDLP)
        GTEST_SKIP() << "an AOCL-DLP prepack needs AOCL-DLP";
    WeightCacheGuard global_cache(2);
    const DlpMHintGuard m_hint(4);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const int M = 4, K = 64, N = 512;
    const nt::buffer_t raw = nt::random_buffer(
            s8, static_cast<size_t>(K) * N, 8700, -8.0f, 8.0f);
    rdr::reorder_params_t rp;
    rp.is_prepack = true;
    rp.prepack.algo = matmul_algo_t::aocl_dlp_blocked;
    rp.prepack.wei_dtype = s8;
    rp.prepack.src_dtype = u8;
    rp.prepack.K = K;
    rp.prepack.N = N;
    rp.prepack.ldb = N;
    const size_t bytes = rdr::weight_prepack_size(rp),
                 sums = static_cast<size_t>(N) * sizeof(int32_t);
    ASSERT_GE(bytes, static_cast<size_t>(K) * N + sums);
    nt::buffer_t packed = nt::kept_until_exit(nt::make_buffer(bytes));
    ASSERT_EQ(rdr::reorder_direct(raw.get(), packed.get(), rp),
            status_t::success);
    const std::vector<uint8_t> prepack(
            packed.as<uint8_t>(), packed.as<uint8_t>() + bytes);
    const nt::buffer_t src = nt::random_buffer(
            u8, static_cast<size_t>(M) * K, 8702, 0.0f, 255.0f);
    const float src_scale = 0.03125f, wei_scale = 0.0625f;
    const int32_t src_zp = 7;
    mm::matmul_params p;
    p.dtypes.src = u8;
    p.dtypes.wei = s8;
    p.dtypes.dst = f32;
    p.mem_format_b = 'r';
    p.weight_cache_type = 2;
    p.quant_params.src_scale.buff = &src_scale;
    p.quant_params.src_scale.dt = f32;
    p.quant_params.src_scale.dims = {1};
    p.quant_params.src_zp.buff = &src_zp;
    p.quant_params.src_zp.dt = s32;
    p.quant_params.src_zp.dims = {1};
    p.quant_params.wei_scale.buff = &wei_scale;
    p.quant_params.wei_scale.dt = f32;
    p.quant_params.wei_scale.dims = {1};
    const auto call = [&](const void *w, float *dst) {
        mm::matmul_batch_params_t batch;
        mm::matmul_params cp = p;
        return mm::matmul_direct('r', false, false, M, N, K, 1.0f, src.get(), K,
                w, N, nullptr, 0.0f, dst, N, true, batch, cp, nullptr);
    };
    expect_view_waits_for_hold(
            p, packed.as<uint8_t>(), bytes, bytes - sums, K, N, N, false);
    {
        SCOPED_TRACE("weight scales in groups of 32");
        mm::matmul_params grouped = p;
        const std::vector<float> scales(static_cast<size_t>(2) * N, wei_scale);
        grouped.quant_params.wei_scale.buff = scales.data();
        grouped.quant_params.wei_scale.dims = {2, N};
        nt::buffer_t copy = nt::kept_until_exit(nt::make_buffer(bytes));
        std::memcpy(copy.get(), prepack.data(), bytes);
        expect_view_waits_for_hold(grouped, copy.as<uint8_t>(), bytes,
                bytes - sums, K, N, N, false);
    }
    std::vector<float> dst(static_cast<size_t>(M) * N, 123.0f);
    EXPECT_EQ(call(packed.get(), dst.data()), status_t::failure);
    EXPECT_TRUE(std::all_of(dst.begin(), dst.end(), [](float v) {
        return v == 123.0f;
    })) << "a call that failed wrote dst";
    mm::clear_matmul_weight_caches();
    nt::buffer_t own = nt::kept_until_exit(nt::make_buffer(bytes));
    std::memcpy(own.get(), prepack.data(), bytes);
    ASSERT_EQ(call(own.get(), dst.data()), status_t::success);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k)
                sum += (static_cast<double>(src.as<uint8_t>()[m * K + k])
                               - src_zp)
                        * raw.as<int8_t>()[static_cast<size_t>(k) * N + n];
            EXPECT_NEAR(dst[static_cast<size_t>(m) * N + n],
                    sum * src_scale * wei_scale, 1e-3)
                    << "row " << m << " column " << n;
        }
    mm::clear_matmul_weight_caches();
}

// The hold of a handed-back call on a GGML pack spans the pack's blocks:
// with N = K = 256, transB and ldb 256, a Q8_0 pack holds 34 bytes for
// each 32 weights, 69,632 bytes, past the 65,536 of the raw s8 weight of
// the shape, and a Q4_0 pack 18, 36,864 bytes, past the 32,768 of the raw
// s4 weight.  While its hold is kept, a first native call at weight cache
// mode 2 on a 32 x 32 bf16 weight at the end of that raw weight waits to
// rewrite it in place.  Once that weight is an image, the Q8_0 call
// through matmul_direct under ZENDNNL_MATMUL_ALGO=10 fails with dst
// untouched; on a copy of the pack it returns the result of the call's
// model.
TEST(NativeBrgemmRefusal, AHandedBackCallHoldsAllOfItsGgmlPack) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    if (!ZENDNNL_DEPENDS_AOCLDLP)
        GTEST_SKIP() << "the default route of a build without AOCL-DLP "
                        "computes no GGML pack";
    WeightCacheGuard global_cache(2);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const int M = 4, K = 256, N = 256, groups = K / 32;
    const nt::buffer_t src = nt::random_buffer(
            s8, static_cast<size_t>(M) * K, 8710, -8.0f, 8.0f);
    const std::vector<float> src_scale(static_cast<size_t>(M) * groups, 0.5f);
    const auto params = [&](data_type_t wei_dt) {
        mm::matmul_params p;
        p.dtypes.src = s8;
        p.dtypes.wei = wei_dt;
        p.dtypes.dst = f32;
        p.packing.pack_format_b = 1;
        p.weight_cache_type = 2;
        p.quant_params.src_scale.buff = src_scale.data();
        p.quant_params.src_scale.dt = f32;
        p.quant_params.src_scale.dims = {M, groups};
        return p;
    };
    // Blocks of an f16 scale of 1 and random values.
    const auto pack_of = [&](data_type_t wei_dt) {
        const size_t block = wei_dt == s8 ? 34 : 18;
        nt::buffer_t b = nt::kept_until_exit(nt::random_buffer(u8,
                static_cast<size_t>(N) * groups * block, 8711, 0.0f, 255.0f));
        for (size_t i = 0; i < static_cast<size_t>(N) * groups; ++i) {
            const uint16_t one = 0x3c00;
            std::memcpy(b.as<uint8_t>() + i * block, &one, sizeof(one));
        }
        return b;
    };
    for (const data_type_t wei_dt : {s8, s4}) {
        SCOPED_TRACE(dtype_info(wei_dt));
        nt::buffer_t pack = pack_of(wei_dt);
        expect_view_waits_for_hold(params(wei_dt), pack.as<uint8_t>(),
                pack.bytes, wei_dt == s8 ? 65536 : 32768, K, N, K, true);
        mm::clear_matmul_weight_caches();
    }
    nt::buffer_t pack = pack_of(s8);
    const std::vector<uint8_t> blocks(
            pack.as<uint8_t>(), pack.as<uint8_t>() + pack.bytes);
    const auto call = [&](const void *w, float *dst) {
        mm::matmul_batch_params_t batch;
        mm::matmul_params p = params(s8);
        return mm::matmul_direct('r', false, true, M, N, K, 1.0f, src.get(), K,
                w, K, nullptr, 0.0f, dst, N, true, batch, p, nullptr);
    };
    const nt::buffer_t view_src = nt::random_buffer(
            bf16, static_cast<size_t>(4) * 32, 8712, -1.0f, 1.0f);
    std::vector<float> view_dst(static_cast<size_t>(4) * 32);
    ASSERT_EQ(native_view_call(pack.as<uint8_t>() + 65536, view_src.get(),
                      view_dst.data()),
            status_t::success);
    std::vector<float> dst(static_cast<size_t>(M) * N, 123.0f);
    EXPECT_EQ(call(pack.get(), dst.data()), status_t::failure);
    EXPECT_TRUE(std::all_of(dst.begin(), dst.end(), [](float v) {
        return v == 123.0f;
    })) << "a call that failed wrote dst";
    mm::clear_matmul_weight_caches();
    nt::buffer_t own = nt::kept_until_exit(nt::make_buffer(blocks.size()));
    std::memcpy(own.get(), blocks.data(), blocks.size());
    ASSERT_EQ(call(own.get(), dst.data()), status_t::success);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double sum = 0.0;
            for (int g = 0; g < groups; ++g) {
                const int8_t *w = reinterpret_cast<const int8_t *>(blocks.data()
                        + (static_cast<size_t>(n) * groups + g) * 34 + 2);
                double part = 0.0;
                for (int k = 0; k < 32; ++k)
                    part += static_cast<double>(
                                    src.as<int8_t>()[m * K + g * 32 + k])
                            * w[k];
                sum += part * src_scale[static_cast<size_t>(m) * groups + g];
            }
            EXPECT_NEAR(dst[static_cast<size_t>(m) * N + n], sum,
                    1e-3 * std::max(1.0, std::fabs(sum)))
                    << "row " << m << " column " << n;
        }
    mm::clear_matmul_weight_caches();
}

// The hold of a handed-back call outlives a drop of the process's weight
// cache mode from 2 to 1 (as kernel selection and group matmul make): an
// image a first native call wrote in place 4 KiB into a buffer before the
// drop makes the hold of a bf16 column-major call of K 32, N 256 and ldb
// 32 at the start of the buffer fail, and so does a W4A16 call on a u4
// weight there through matmul_direct, with dst untouched.  A native call
// that took mode 2 before the drop and waits for an earlier hold to
// rewrite its weight in place: a hold taken after the drop finds that
// rewrite and fails, or the rewrite waits for it too, and every held byte
// reads as it was until that hold ends.
TEST(NativeBrgemmRefusal, AHandedBackCallKeepsItsHoldAfterTheCacheModeDrops) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    WeightCacheGuard global_cache(2);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    auto &config = zendnnl::common::matmul_config_t::instance();
    const int K = 32, N = 256, ldb = 32, VK = 32, VN = 32, VM = 4;
    const size_t span = static_cast<size_t>(N) * ldb * 2, offset = 4096;
    const nt::buffer_t raw
            = nt::random_buffer(bf16, span / 2, 8700, -1.0f, 1.0f);
    const nt::buffer_t src = nt::random_buffer(
            bf16, static_cast<size_t>(VM) * VK, 8701, -1.0f, 1.0f);
    std::vector<float> dst_a(static_cast<size_t>(VM) * VN);
    const auto native_call = [&](uint8_t *base) {
        mm::matmul_batch_params_t batch;
        mm::matmul_params p;
        p.dtypes.src = bf16;
        p.dtypes.wei = bf16;
        p.dtypes.dst = f32;
        p.weight_cache_type = 2;
        p.num_threads = 1;
        return mm::matmul_direct('r', false, false, VM, VN, VK, 1.0f, src.get(),
                VK, base + offset, VN, nullptr, 0.0f, dst_a.data(), VN, true,
                batch, p, nullptr);
    };
    mm::matmul_params column_major;
    column_major.dtypes.src = bf16;
    column_major.dtypes.wei = bf16;
    column_major.dtypes.dst = f32;
    column_major.weight_cache_type = 2;
    const auto hold
            = [&](const void *w,
                      std::unique_ptr<nat::native_weight_raw_hold_t> &h) {
        h = std::make_unique<nat::native_weight_raw_hold_t>();
        return nat::native_hold_handed_back_weight(
                *h, 'c', w, K, N, ldb, false, column_major);
    };
    {
        SCOPED_TRACE("an image written before the drop");
        nt::buffer_t shared = nt::kept_until_exit(nt::make_buffer(span));
        std::memcpy(shared.get(), raw.get(), span);
        ASSERT_EQ(native_call(shared.as<uint8_t>()), status_t::success);
        ASSERT_TRUE(nat::native_weight_packed_in_place(
                shared.as<uint8_t>() + offset,
                static_cast<size_t>(VK) * VN * 2));
        config.downgrade_weight_cache(1);
        std::unique_ptr<nat::native_weight_raw_hold_t> h;
        EXPECT_EQ(hold(shared.get(), h), status_t::failure);
        // W4A16: bf16 x u4 of K 128 and N 256, 16 KiB of nibbles.
        const std::vector<float> wei_scales(N, 0.25f);
        const int32_t zp = 8;
        mm::matmul_params p;
        p.dtypes.src = bf16;
        p.dtypes.wei = u4;
        p.dtypes.dst = f32;
        p.weight_cache_type = 2;
        auto &q = p.quant_params;
        q.wei_scale.buff = wei_scales.data();
        q.wei_scale.dt = f32;
        q.wei_scale.dims = {1, N};
        q.wei_zp.buff = &zp;
        q.wei_zp.dt = s8;
        q.wei_zp.dims = {1, 1};
        const nt::buffer_t src_b = nt::random_buffer(
                bf16, static_cast<size_t>(VM) * 128, 8702, -1.0f, 1.0f);
        std::vector<float> dst(static_cast<size_t>(VM) * N, 123.0f);
        mm::matmul_batch_params_t batch;
        EXPECT_EQ(mm::matmul_direct('r', false, false, VM, N, 128, 1.0f,
                          src_b.get(), 128, shared.get(), N, nullptr, 0.0f,
                          dst.data(), N, true, batch, p, nullptr),
                status_t::failure);
        EXPECT_TRUE(std::all_of(dst.begin(), dst.end(), [](float v) {
            return v == 123.0f;
        })) << "a call that failed wrote dst";
        config.set_weight_cache(2);
        mm::clear_matmul_weight_caches();
    }
    {
        SCOPED_TRACE("a rewrite that took mode 2 before the drop");
        nt::buffer_t shared = nt::kept_until_exit(nt::make_buffer(span));
        std::memcpy(shared.get(), raw.get(), span);
        std::unique_ptr<nat::native_weight_raw_hold_t> first, second;
        ASSERT_EQ(hold(shared.get(), first), status_t::success);
        std::atomic<bool> done {false};
        status_t st = status_t::failure;
        std::thread native([&] {
            st = native_call(shared.as<uint8_t>());
            done.store(true);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        config.downgrade_weight_cache(1);
        const status_t held = hold(shared.get(), second);
        first.reset();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::vector<uint8_t> read(span);
        std::memcpy(read.data(), shared.get(), span);
        const bool waited = !done.load();
        second.reset();
        native.join();
        if (held == status_t::success) {
            EXPECT_TRUE(waited) << "the native call did not wait for the hold";
            EXPECT_EQ(std::memcmp(read.data(), raw.get(), span), 0)
                    << "an in-place rewrite wrote bytes the hold holds";
        } else {
            EXPECT_EQ(held, status_t::failure);
        }
        EXPECT_EQ(st, status_t::success);
        EXPECT_TRUE(nat::native_weight_packed_in_place(
                shared.as<uint8_t>() + offset,
                static_cast<size_t>(VK) * VN * 2))
                << "the native call did not rewrite its weight in place";
        mm::clear_matmul_weight_caches();
    }
}

/// A bf16 call of two batches over Batch_B row-major 32 x 32 weights at w
/// (ldb 32), stride elements apart (-1: the batched kernels' stride, one
/// weight), M 4, into f32, on a weight that changes: under
/// ZENDNNL_MATMUL_ALGO=10 the native BRGEMM hands it to the default route.
status_t batched_call(const void *w, const void *src, float *dst, int batch_b,
        size_t stride = static_cast<size_t>(-1)) {
    mm::matmul_batch_params_t batch;
    batch.Batch_A = 2;
    batch.Batch_B = batch_b;
    batch.batch_stride_wei = stride;
    mm::matmul_params p;
    p.dtypes.src = bf16;
    p.dtypes.wei = bf16;
    p.dtypes.dst = f32;
    p.weight_cache_type = 2;
    return mm::matmul_direct('r', false, false, 4, 32, 32, 1.0f, src, 32, w, 32,
            nullptr, 0.0f, dst, 32, false, batch, p, nullptr);
}

// A batched call that ZENDNNL_MATMUL_ALGO=10 hands to the default route,
// whose second weight holds an in-place image, fails with one error line
// that names the image's start and bytes and what helps: for a live image,
// writing the raw weight back into the image's buffer, a clear and a reload
// of every weight a mode 2 call rewrote; for its tombstone after a clear,
// the raw weight of the image's whole buffer, with no further clear.
TEST(NativeBrgemmRefusal, AHandedBackCallOverAnImageNamesWhatHelps) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    nt::expect_in_child({{"ZENDNNL_API_LOG_LEVEL", nullptr},
                                {"ZENDNNL_COMMON_LOG_LEVEL", nullptr}},
            [] {
        WeightCacheGuard global_cache(2);
        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
        const nt::buffer_t w = nt::random_buffer(bf16, 4096, 8920, -1.0f, 1.0f);
        const nt::buffer_t src
                = nt::random_buffer(bf16, 2 * 4 * 32, 8921, -1.0f, 1.0f);
        std::vector<float> dst(2 * 4 * 32);
        mm::matmul_batch_params_t one;
        mm::matmul_params p;
        p.dtypes.src = bf16;
        p.dtypes.wei = bf16;
        p.dtypes.dst = f32;
        p.weight_cache_type = 2;
        const uint8_t *second = w.as<uint8_t>() + 2048;
        if (mm::matmul_direct('r', false, false, 4, 32, 32, 1.0f, src.get(), 32,
                    second, 32, nullptr, 0.0f, dst.data(), 32, true, one, p,
                    nullptr)
                        != status_t::success
                || !nat::native_weight_packed_in_place(second, 2048))
            return 1;
        int bad = 0;
        const auto line = [&](const char *what) {
            std::fill(dst.begin(), dst.end(), 123.0f);
            ::testing::internal::CaptureStdout();
            const status_t st = batched_call(w.get(), src.get(), dst.data(), 2);
            const std::string log = ::testing::internal::GetCapturedStdout();
            size_t lines = 0;
            for (size_t at = log.find("][error"); at != std::string::npos;
                    at = log.find("][error", at + 1))
                ++lines;
            if (st != status_t::failure || lines != 1
                    || !std::all_of(dst.begin(), dst.end(),
                            [](float v) { return v == 123.0f; })
                    || log.find("the image recorded at ") == std::string::npos
                    || log.find(", 2048 bytes") == std::string::npos) {
                std::fprintf(stderr, "%s:\n%s\n", what, log.c_str());
                ++bad;
            }
            return log;
        };
        const std::string live = line("a live image");
        mm::clear_matmul_native_weight_caches();
        const std::string tomb = line("a tombstone");
        if (live.find("write the raw weight back into the image's buffer, "
                      "clear the weight caches and reload every weight a mode "
                      "2 call rewrote")
                == std::string::npos)
            ++bad;
        if (tomb.find("reload the raw weight of the image's whole buffer")
                        == std::string::npos
                || tomb.find("clear the weight caches") != std::string::npos)
            ++bad;
        return bad;
    }, "the error line of a handed-back call over an image");
}

// A batched call that ZENDNNL_MATMUL_ALGO=10 hands to the default route
// reads every one of its Batch_B weights: once a first native call at
// weight cache mode 2 has rewritten the second weight in place, at the
// batched kernels' stride (2 KiB) or at a batch_stride_wei of 1536
// elements, the batched call fails with dst untouched; a call of one weight
// (Batch_B 1) does not read the image right after it and gives the result
// of the same call on a pristine copy.
TEST(NativeBrgemmRefusal, ABatchedCallOverAnImageInALaterMatrixFails) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    WeightCacheGuard global_cache(2);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const size_t bytes = 8192, out = 2 * 4 * 32;
    const nt::buffer_t raw
            = nt::random_buffer(bf16, bytes / 2, 8900, -1.0f, 1.0f);
    const nt::buffer_t src
            = nt::random_buffer(bf16, 2 * 4 * 32, 8901, -1.0f, 1.0f);
    const nt::buffer_t view_src
            = nt::random_buffer(bf16, 4 * 32, 8902, -1.0f, 1.0f);
    struct case_t {
        int batch_b;
        size_t stride;
        size_t image_at;
        bool reads_image;
    };
    for (const case_t &c : {case_t {2, static_cast<size_t>(-1), 2048, true},
                 case_t {2, 1536, 3072, true},
                 case_t {1, static_cast<size_t>(-1), 2048, false}}) {
        SCOPED_TRACE(::testing::Message()
                << "Batch_B " << c.batch_b << " image at " << c.image_at);
        std::vector<float> want(out);
        {
            nt::buffer_t own = nt::kept_until_exit(nt::make_buffer(bytes));
            std::memcpy(own.get(), raw.get(), bytes);
            ASSERT_EQ(batched_call(own.get(), src.get(), want.data(), c.batch_b,
                              c.stride),
                    status_t::success);
        }
        nt::buffer_t shared = nt::kept_until_exit(nt::make_buffer(bytes));
        std::memcpy(shared.get(), raw.get(), bytes);
        std::vector<float> view_dst(4 * 32);
        ASSERT_EQ(native_view_call(shared.as<uint8_t>() + c.image_at,
                          view_src.get(), view_dst.data()),
                status_t::success);
        ASSERT_TRUE(nat::native_weight_packed_in_place(
                shared.as<uint8_t>() + c.image_at, 2048));
        std::vector<float> dst(out, 123.0f);
        const status_t st = batched_call(
                shared.get(), src.get(), dst.data(), c.batch_b, c.stride);
        if (c.reads_image) {
            EXPECT_EQ(st, status_t::failure);
            EXPECT_TRUE(std::all_of(dst.begin(), dst.end(), [](float v) {
                return v == 123.0f;
            })) << "a call that failed wrote dst";
        } else {
            EXPECT_EQ(st, status_t::success);
            EXPECT_EQ(std::memcmp(dst.data(), want.data(), out * 4), 0);
        }
        mm::clear_matmul_weight_caches();
    }
}

// The hold of a batched call spans every weight it reads: while it is kept,
// a first native call at weight cache mode 2 on the second of two 32 x 32
// weights waits to rewrite it in place, and every byte of both reads as it
// was.  With either call started up to 300 us after the other on the same
// buffer, the batched call reads the raw weights, with the result of the
// same call on a pristine copy, or fails; it never returns success with a
// result read in part from the image.
TEST(NativeBrgemmRefusal, ABatchedCallHoldsEveryMatrixItReads) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    WeightCacheGuard global_cache(2);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const size_t bytes = 4096, out = 2 * 4 * 32;
    const int trials = 16;
    const nt::buffer_t raw
            = nt::random_buffer(bf16, bytes / 2, 8910, -1.0f, 1.0f);
    const nt::buffer_t src
            = nt::random_buffer(bf16, 2 * 4 * 32, 8911, -1.0f, 1.0f);
    const nt::buffer_t view_src
            = nt::random_buffer(bf16, 4 * 32, 8912, -1.0f, 1.0f);
    {
        nt::buffer_t shared = nt::kept_until_exit(nt::make_buffer(bytes));
        std::memcpy(shared.get(), raw.get(), bytes);
        mm::matmul_params p;
        p.dtypes.src = bf16;
        p.dtypes.wei = bf16;
        p.dtypes.dst = f32;
        p.weight_cache_type = 2;
        mm::matmul_batch_params_t batch;
        batch.Batch_A = 2;
        batch.Batch_B = 2;
        expect_view_waits_for_hold(p, shared.as<uint8_t>(), bytes, 2048, 32, 32,
                32, false, &batch);
        mm::clear_matmul_weight_caches();
    }
    std::vector<float> want(out);
    {
        nt::buffer_t own = nt::kept_until_exit(nt::make_buffer(bytes));
        std::memcpy(own.get(), raw.get(), bytes);
        ASSERT_EQ(batched_call(own.get(), src.get(), want.data(), 2),
                status_t::success);
    }
    std::mt19937 rng(8913);
    std::uniform_int_distribution<int> delay_us(0, 300);
    int read_raw = 0, found_image = 0, wrong = 0;
    for (int t = 0; t < trials; ++t) {
        const int delay = delay_us(rng);
        nt::buffer_t shared = nt::kept_until_exit(nt::make_buffer(bytes));
        std::memcpy(shared.get(), raw.get(), bytes);
        std::vector<float> view_dst(4 * 32), got(out, 123.0f);
        status_t st_a = status_t::failure, st_b = status_t::failure;
        std::atomic<int> ready {0};
        const auto start = [&ready] {
            ready.fetch_add(1);
            while (ready.load() < 2) {}
        };
        const bool batched_first = t % 2 == 0;
        std::thread tb([&] {
            start();
            if (!batched_first)
                std::this_thread::sleep_for(std::chrono::microseconds(delay));
            st_b = batched_call(shared.get(), src.get(), got.data(), 2);
        });
        std::thread ta([&] {
            start();
            if (batched_first)
                std::this_thread::sleep_for(std::chrono::microseconds(delay));
            st_a = native_view_call(shared.as<uint8_t>() + 2048, view_src.get(),
                    view_dst.data());
        });
        tb.join();
        ta.join();
        EXPECT_EQ(st_a, status_t::success) << "trial " << t;
        if (st_b != status_t::success)
            ++found_image;
        else if (std::memcmp(got.data(), want.data(), out * 4) != 0)
            ++wrong;
        else
            ++read_raw;
        mm::clear_matmul_weight_caches();
    }
    EXPECT_EQ(wrong, 0) << "of " << trials << " batched calls, " << wrong
                        << " returned success with a result read in part "
                           "from the image ("
                        << read_raw << " read the raw weights, " << found_image
                        << " found the image)";
}

// The same column-major form through matmul_direct under
// ZENDNNL_MATMUL_ALGO=10 at weight cache mode 2: a handed-back bf16 call of
// K 256, N 8192 and ldb 256 over a weight it marks as changing, and a first
// native call on a row-major 256 x 256 weight 1 MiB into the same buffer,
// started up to 300 us after it, which rewrites those bytes in place.  The
// handed-back call reads its raw weight, with the result of the same call
// on a private copy, or finds the image and fails; it never returns success
// with a result read in part from the image.  The native call has the
// result of the same call on a private copy every time.
TEST(NativeBrgemmRefusal,
        AHandedBackColumnMajorCallReadsNoBytesARewriteWrites) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    if (!ZENDNNL_DEPENDS_AOCLDLP)
        GTEST_SKIP() << "the default route of a build without AOCL-DLP "
                        "computes no column-major call";
    WeightCacheGuard global_cache(2);
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    const int K = 256, N = 8192, ldb = 256, VK = 256, VN = 256, MA = 16, MB = 8,
              threads = 8, trials = 16;
    const size_t span = static_cast<size_t>(N) * ldb * 2,
                 offset = static_cast<size_t>(1) << 20;
    const nt::buffer_t w0
            = nt::random_buffer(bf16, span / 2, 8400, -1.0f, 1.0f);
    const nt::buffer_t src_a = nt::random_buffer(
            bf16, static_cast<size_t>(MA) * VK, 8401, -1.0f, 1.0f);
    const nt::buffer_t src_b = nt::random_buffer(
            bf16, static_cast<size_t>(K) * MB, 8402, -1.0f, 1.0f);
    const auto params = [&](int wc) {
        mm::matmul_params p;
        p.dtypes.src = bf16;
        p.dtypes.wei = bf16;
        p.dtypes.dst = f32;
        p.weight_cache_type = wc;
        p.num_threads = threads;
        return p;
    };
    // The native call: row-major [VK, VN] at offset, constant, so its first
    // call at mode 2 rewrites those bytes in place.
    const auto native_call = [&](const uint8_t *base, void *dst) {
        mm::matmul_batch_params_t batch;
        mm::matmul_params p = params(2);
        return mm::matmul_direct('r', false, false, MA, VN, VK, 1.0f,
                src_a.get(), VK, base + offset, VN, nullptr, 0.0f, dst, VN,
                true, batch, p, nullptr);
    };
    // The handed-back call: column-major [K, N], N columns of K elements
    // ldb apart.
    const auto handed_back = [&](const uint8_t *base, void *dst) {
        mm::matmul_batch_params_t batch;
        mm::matmul_params p = params(2);
        return mm::matmul_direct('c', false, false, MB, N, K, 1.0f, src_b.get(),
                MB, base, ldb, nullptr, 0.0f, dst, MB, false, batch, p,
                nullptr);
    };
    const size_t bytes_a = static_cast<size_t>(MA) * VN * sizeof(float);
    const size_t bytes_b = static_cast<size_t>(MB) * N * sizeof(float);
    nt::buffer_t want_a = nt::make_buffer(bytes_a);
    nt::buffer_t want_b = nt::make_buffer(bytes_b);
    {
        nt::buffer_t own_a = nt::make_buffer(span);
        nt::buffer_t own_b = nt::make_buffer(span);
        std::memcpy(own_a.get(), w0.get(), span);
        std::memcpy(own_b.get(), w0.get(), span);
        ASSERT_EQ(handed_back(own_b.as<uint8_t>(), want_b.get()),
                status_t::success);
        ASSERT_EQ(native_call(own_a.as<uint8_t>(), want_a.get()),
                status_t::success);
        mm::clear_matmul_weight_caches();
    }
    const auto far_off = [&](const nt::buffer_t &got) {
        float err = 0.0f, mag = 0.0f;
        for (int i = 0; i < MB * N; ++i) {
            err = std::max(
                    err, std::fabs(got.as<float>()[i] - want_b.as<float>()[i]));
            mag = std::max(mag, std::fabs(want_b.as<float>()[i]));
        }
        return err > 1e-2f * mag;
    };
    std::mt19937 rng(8403);
    std::uniform_int_distribution<int> delay_us(0, 300);
    int read_raw = 0, found_image = 0, wrong = 0;
    for (int t = 0; t < trials; ++t) {
        const int delay = delay_us(rng);
        nt::buffer_t shared = nt::make_buffer(span);
        std::memcpy(shared.get(), w0.get(), span);
        nt::buffer_t dst_a = nt::make_buffer(bytes_a);
        nt::buffer_t dst_b = nt::make_buffer(bytes_b);
        status_t st_a = status_t::failure, st_b = status_t::failure;
        std::atomic<int> ready {0};
        const auto start = [&ready] {
            ready.fetch_add(1);
            while (ready.load() < 2) {}
        };
        std::thread tb([&] {
            start();
            st_b = handed_back(shared.as<uint8_t>(), dst_b.get());
        });
        std::thread ta([&] {
            start();
            std::this_thread::sleep_for(std::chrono::microseconds(delay));
            st_a = native_call(shared.as<uint8_t>(), dst_a.get());
        });
        tb.join();
        ta.join();
        EXPECT_EQ(st_a, status_t::success) << "trial " << t;
        EXPECT_EQ(std::memcmp(dst_a.get(), want_a.get(), bytes_a), 0)
                << "trial " << t;
        if (st_b != status_t::success)
            ++found_image;
        else if (far_off(dst_b))
            ++wrong;
        else
            ++read_raw;
        mm::clear_matmul_weight_caches();
    }
    EXPECT_EQ(wrong, 0) << "of " << trials << " handed-back calls, " << wrong
                        << " returned success with a result read in part "
                           "from the image ("
                        << read_raw << " read the raw weight, " << found_image
                        << " found the image)";
}

// Calls of N not a multiple of 32, per call and under
// ZENDNNL_MATMUL_ALGO=10: native_brgemm_refusal gives success, and
// matmul_direct serves them over a native layout of the weight, which stays
// raw, with the reference algorithm's result within the band of the type.
TEST(NativeBrgemmRefusal, CallsOfAnyNAreServed) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    const WeightCacheGuard mode(2);
    for (const bool per_call : {true, false})
        for (const auto &rp : any_n_cases()) {
            refused_t &r = *rp;
            SCOPED_TRACE(std::string(r.name)
                    + (per_call ? ", lowoha_algo native_brgemm"
                                : ", ZENDNNL_MATMUL_ALGO=10"));
            nt::buffer_t ref = nt::make_buffer(r.dst.bytes);
            std::memcpy(ref.get(), r.prior.get(), r.dst.bytes);
            ASSERT_EQ(nt::run_reference(r.c, r.wei.get(), ref.get()),
                    status_t::success);
            std::unique_ptr<nt::algo_pin_t> pin;
            if (per_call)
                r.c.params.lowoha_algo = matmul_algo_t::native_brgemm;
            else
                pin.reset(new nt::algo_pin_t(matmul_algo_t::native_brgemm));
            const std::vector<uint8_t> raw(static_cast<uint8_t *>(r.wei.get()),
                    static_cast<uint8_t *>(r.wei.get()) + r.wei.bytes);
            std::string reason;
            EXPECT_EQ(nt::refusal_of(r.c, &reason), status_t::success)
                    << reason;
            ASSERT_EQ(nt::run_matmul(r.c), status_t::success);
            const data_type_t dst_dt = r.c.params.dtypes.dst;
            const bool low = r.c.params.dtypes.wei != f32;
            EXPECT_TRUE(nt::within_band(r.dst.get(), r.c.N, ref.get(), r.c.N,
                    dst_dt, r.c.M, r.c.N, low ? rtol_bf16 : rtol_f32,
                    nt::comparison_abs_bound(r.c.K,
                            low ? epsilon_bf16 : epsilon_f32, 1.0f, 0.0f, low,
                            r.c.params.dtypes.wei == s8, false)));
            EXPECT_EQ(std::memcmp(r.wei.get(), raw.data(), raw.size()), 0);
            EXPECT_EQ(nt::native_layouts(r.wei.get()), 1u);
            ::reset_grp_matmul_caches();
        }
}

// The check of a call that quantizes its bf16 / f32 source writes none of
// the caller's buffers: native_brgemm_refusal leaves the source scale and
// zero point, dst, the source and the weight as they were (an s8
// quantization with a scale per tensor and per row, a u8 one with a zero
// point), per call and under ZENDNNL_MATMUL_ALGO=10.  matmul_direct then
// serves the call and writes the scales it computed into those buffers.
TEST(NativeBrgemmRefusal, TheCheckWritesNoCallerBuffer) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    const int M = 4, N = 64, K = 128;
    const float kScale = -7.0f;
    const int32_t kZp = 77;
    const struct {
        const char *name;
        data_type_t compute;
        int rows;
    } kinds[] = {{"s8, a scale per tensor", s8, 1},
            {"s8, a scale per row", s8, M},
            {"u8, a scale and zero point per tensor", u8, 1}};
    uint32_t seed = 6500;
    for (const bool per_call : {true, false})
        for (const data_type_t src_dt : {bf16, f32})
            for (const auto &k : kinds) {
                SCOPED_TRACE(std::string(dtype_info(src_dt)) + " source, "
                        + k.name
                        + (per_call ? ", lowoha_algo native_brgemm"
                                    : ", ZENDNNL_MATMUL_ALGO=10"));
                auto rp = make_case(
                        k.name, "", src_dt, s8, f32, M, N, K, seed += 4);
                refused_t &r = *rp;
                std::vector<float> scales(k.rows, kScale);
                std::vector<int32_t> zps(k.rows, kZp);
                auto &q = r.c.params.quant_params;
                r.c.params.dynamic_quant = true;
                r.c.params.dtypes.compute = k.compute;
                q.src_scale.buff = scales.data();
                q.src_scale.dt = f32;
                q.src_scale.dims = {k.rows, 1};
                if (k.compute == u8) {
                    q.src_zp.buff = zps.data();
                    q.src_zp.dt = s32;
                    q.src_zp.dims = {k.rows, 1};
                }
                std::unique_ptr<nt::algo_pin_t> pin;
                if (per_call)
                    r.c.params.lowoha_algo = matmul_algo_t::native_brgemm;
                else
                    pin.reset(new nt::algo_pin_t(matmul_algo_t::native_brgemm));
                const std::vector<uint8_t> src(
                        static_cast<uint8_t *>(r.src.get()),
                        static_cast<uint8_t *>(r.src.get()) + r.src.bytes);
                const std::vector<uint8_t> wei(
                        static_cast<uint8_t *>(r.wei.get()),
                        static_cast<uint8_t *>(r.wei.get()) + r.wei.bytes);

                std::string reason;
                EXPECT_EQ(nt::refusal_of(r.c, &reason), status_t::success)
                        << reason;
                EXPECT_EQ(scales, std::vector<float>(k.rows, kScale))
                        << "the check wrote the source scales";
                EXPECT_EQ(zps, std::vector<int32_t>(k.rows, kZp))
                        << "the check wrote the source zero points";
                EXPECT_EQ(std::memcmp(r.dst.get(), r.prior.get(), r.dst.bytes),
                        0);
                EXPECT_EQ(std::memcmp(r.src.get(), src.data(), src.size()), 0);
                EXPECT_EQ(std::memcmp(r.wei.get(), wei.data(), wei.size()), 0);

                ASSERT_EQ(nt::run_matmul(r.c), status_t::success);
                EXPECT_NE(scales, std::vector<float>(k.rows, kScale))
                        << "matmul_direct wrote no source scale";
                ::reset_grp_matmul_caches();
            }
}

// The check of a refusal in the suites that draw calls the native BRGEMM
// serves (native_refusal_verified) stands on the call itself: each class of
// this suite that those suites draw on purpose passes it, and the refusal
// of a class they do not draw (K past the int32 bound) fails it, as a
// refusal of a call they expect to be served would.
TEST(NativeBrgemmRefusal, BroadSuitesTakeOnlyRefusalsTheirCallsExplain) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    for (const auto &rp : refused_cases()) {
        refused_t &r = *rp;
        SCOPED_TRACE(r.name);
        r.c.params.lowoha_algo = matmul_algo_t::native_brgemm;
        const nt::call_t &c = r.c;
        mm::matmul_batch_params_t batch;
        run_matmul_direct('r', c.transA, c.transB, c.M, c.N, c.K, c.alpha,
                c.src, c.lda_or_tight(), c.wei, c.ldb_or_tight(), c.bias,
                c.beta, c.dst, c.ldc_or_tight(), c.is_weights_const, batch,
                c.params);
        if (std::string(r.reason) == "K exceeds the int32 accumulation bound")
            EXPECT_NONFATAL_FAILURE(EXPECT_TRUE(native_refusal_verified()),
                    "refused a call these tests expect it to serve");
        else
            EXPECT_TRUE(native_refusal_verified());
    }
}

// A call that names native_brgemm on a reordered weight (mem_format_b 'r'),
// bf16 and u8 x s8: unimplemented, with dst holding its values and the
// weight its bytes, and a reason naming what the call lacks: with the
// algorithm unset, ZENDNNL_MATMUL_ALGO=10; under it, a weight the native
// BRGEMM takes as AOCL-DLP blocked.
TEST(NativeBrgemmRefusal, AReorderedWeightIsRefusedForWhatTheCallLacks) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    uint32_t seed = 7400;
    for (const bool algo10 : {false, true})
        for (const bool int8 : {false, true}) {
            const nt::algo_pin_t pin(algo10 ? matmul_algo_t::native_brgemm
                                            : matmul_algo_t::none);
            auto rp = make_case(int8 ? "u8 x s8 on a reordered weight"
                                     : "bf16 on a reordered weight",
                    algo10 ? "a reordered weight other than an AOCL-DLP "
                             "blocked one under ZENDNNL_MATMUL_ALGO=10"
                           : "a reordered weight without "
                             "ZENDNNL_MATMUL_ALGO=10",
                    int8 ? u8 : bf16, int8 ? s8 : bf16, int8 ? f32 : bf16, 4,
                    64, 64, seed += 4);
            refused_t &r = *rp;
            SCOPED_TRACE(std::string(r.name)
                    + (algo10 ? ", ZENDNNL_MATMUL_ALGO=10"
                              : ", ZENDNNL_MATMUL_ALGO unset"));
            r.c.params.mem_format_b = 'r';
            r.c.params.lowoha_algo = matmul_algo_t::native_brgemm;
            const std::vector<uint8_t> raw(static_cast<uint8_t *>(r.wei.get()),
                    static_cast<uint8_t *>(r.wei.get()) + r.wei.bytes);
            std::string reason;
            EXPECT_EQ(nt::refusal_of(r.c, &reason), status_t::unimplemented);
            EXPECT_EQ(reason, r.reason);
            EXPECT_TRUE(nt::refused_with(r.c, status_t::unimplemented));
            EXPECT_EQ(std::memcmp(r.dst.get(), r.prior.get(), r.dst.bytes), 0);
            EXPECT_EQ(std::memcmp(r.wei.get(), raw.data(), raw.size()), 0);
        }
}

// The same calls with the algorithm unset are not the native BRGEMM's:
// native_brgemm_refusal gives success.
TEST(NativeBrgemmRefusal, CallsThatDoNotAskForItAreNotRefused) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    const nt::algo_pin_t pin(matmul_algo_t::none);
    for (const auto &rp : refused_cases()) {
        SCOPED_TRACE(rp->name);
        EXPECT_EQ(nt::refusal_of(rp->c), status_t::success);
    }
}

/// The calls of each form the default route does not compute correctly
/// (it returns success with a dst that is wrong or not written, or does not
/// implement the activation) that the native BRGEMM refuses, kept naming
/// what their error line says under ZENDNNL_MATMUL_ALGO=10; a form the
/// native BRGEMM serves is refused as a gated call with beta over a dst of
/// N / 2 columns, or for a column-major layout.
std::vector<std::unique_ptr<refused_t>> kept_cases() {
    std::vector<std::unique_ptr<refused_t>> v;
    uint32_t seed = 7800;
    const int M = 4, N = 64, K = 64, G = 2;
    const auto per_group = [&](refused_t &r) {
        r.wei_scales.assign(static_cast<size_t>(G) * N, 0.25f);
        auto &q = r.c.params.quant_params;
        q.wei_scale.buff = r.wei_scales.data();
        q.wei_scale.dims = {G, N};
    };
    // Scales of 0.25 in f16, and in bf16.
    const auto half_scales = [&](refused_t &r, size_t n, uint16_t bits) {
        r.half_scales.assign(n, bits);
        return static_cast<void *>(r.half_scales.data());
    };
    const char *w4a16 = "W4A16 with per-group or f16 weight scales";
    const char *column
            = "a column-major call on a constant weight or with "
              "integer operands";
    const char *scales = "these weight and source scale types";
    v.push_back(make_case("bf16 x s4 per group",
            "s4 weight with a source other than s8", bf16, s4, bf16, M, N, K,
            seed += 4));
    per_group(*v.back());
    v.back()->kept = w4a16;
    v.push_back(make_case("bf16 x u4 per group with zero points",
            "weight type other than bf16 / s8 / s4 / f32 / f16", bf16, u4, bf16,
            M, N, K, seed += 4));
    {
        refused_t &r = *v.back();
        per_group(r);
        r.wei_zps.assign(static_cast<size_t>(G) * N, 8);
        auto &q = r.c.params.quant_params;
        q.wei_zp.buff = r.wei_zps.data();
        q.wei_zp.dt = s8;
        q.wei_zp.dims = {G, N};
        r.kept = w4a16;
    }
    v.push_back(make_case("bf16 x s4 with f16 weight scales",
            "s4 weight with a source other than s8", bf16, s4, bf16, M, N, K,
            seed += 4));
    {
        refused_t &r = *v.back();
        auto &q = r.c.params.quant_params;
        q.wei_scale.buff = half_scales(r, N, 0x3400);
        q.wei_scale.dt = data_type_t::f16;
        r.kept = w4a16;
    }
    for (const data_type_t src_dt : {bf16, f32}) {
        v.push_back(make_case(
                src_dt == bf16 ? "W8A16, bf16 source" : "W8A16, f32 source",
                "s8 weight with a source other than u8 / s8", src_dt, s8,
                src_dt, M, N, K, seed += 4));
        v.back()->c.params.quant_params.src_scale = {};
        v.back()->kept = "W8A16";
    }
    v.push_back(make_case(
            "f32 into an s32 dst", "", f32, f32, s32, M, N, K, seed += 4));
    v.back()->kept = "a call of float operands into an integer dst";
    v.push_back(make_case(
            "bf16 into an s32 dst, column-major on a weight "
            "that changes",
            "a column-major call", bf16, bf16, s32, M, N, K, seed += 4));
    {
        refused_t &r = *v.back();
        r.layout = 'c';
        r.c.lda = r.c.M;
        r.c.ldb = r.c.K;
        r.c.ldc = r.c.M;
        r.c.is_weights_const = false;
        r.kept = "a call of float operands into an integer dst";
    }
    // Column-major: lda, ldb and ldc of the column-major operands.
    const auto column_major = [&](refused_t &r) {
        r.layout = 'c';
        r.c.lda = r.c.M;
        r.c.ldb = r.c.K;
        r.c.ldc = r.c.M;
        r.kept = column;
    };
    v.push_back(make_case("bf16 column-major on a constant weight",
            "a column-major call", bf16, bf16, bf16, M, N, K, seed += 4));
    column_major(*v.back());
    v.push_back(make_case("u8 x s8 column-major on a weight that changes",
            "a column-major call", u8, s8, f32, M, N, K, seed += 4));
    column_major(*v.back());
    v.back()->c.is_weights_const = false;
    v.push_back(make_case("gelu_tanh_and_mul column-major",
            "a column-major call", bf16, bf16, bf16, M, N, K, seed += 4));
    column_major(*v.back());
    v.back()->c.gated = mm::matmul_gated_act_t::gelu_tanh_and_mul;
    v.back()->kept = "the default route does not implement gelu_tanh_and_mul";
    // A gated call with beta over its dst of N / 2 columns, which the native
    // BRGEMM refuses before its kernel.
    const char *gated_beta
            = "beta with a gated activation over a dst of "
              "fewer than N columns";
    const auto gated_with_beta = [&](refused_t &r) {
        r.c.gated = mm::matmul_gated_act_t::silu_and_mul;
        r.c.beta = 0.5f;
    };
    // A bf16 source quantized per token into an f32 dst.
    const auto dynamic
            = [&](const char *name, data_type_t wei_dt, data_type_t compute) {
        v.push_back(make_case(
                name, gated_beta, bf16, wei_dt, f32, M, N, K, seed += 4));
        refused_t &r = *v.back();
        gated_with_beta(r);
        r.c.params.dynamic_quant = true;
        r.c.params.dtypes.compute = compute;
        auto &q = r.c.params.quant_params;
        q.src_scale.buff = nullptr;
        q.src_scale.dt = f32;
        q.src_scale.dims = {M, 1};
        return &r;
    };
    {
        refused_t &r = *dynamic("DA8W8 with f16 scales", s8, s8);
        auto &q = r.c.params.quant_params;
        q.wei_scale.buff = half_scales(r, N, 0x3400);
        q.wei_scale.dt = data_type_t::f16;
        q.src_scale.dt = data_type_t::f16;
        r.kept = scales;
    }
    {
        refused_t &r = *dynamic(
                "DA8W8 with f32 weight and bf16 source scales", s8, s8);
        r.c.params.quant_params.src_scale.dt = bf16;
        r.kept = scales;
    }
    {
        refused_t &r = *dynamic(
                "W4A8 with bf16 weight and f32 source scales", s4, s8);
        per_group(r);
        auto &q = r.c.params.quant_params;
        q.wei_scale.buff = half_scales(r, static_cast<size_t>(G) * N, 0x3e80);
        q.wei_scale.dt = bf16;
        r.kept = scales;
    }
    {
        refused_t &r
                = *dynamic("a source quantized to u8 with zero points", s8, u8);
        auto &q = r.c.params.quant_params;
        q.src_zp.buff = nullptr;
        q.src_zp.dt = s32;
        q.src_zp.dims = {M, 1};
        r.kept = "a source quantized to u8 with a zero point per row";
    }
    v.push_back(make_case("s8 x s8 with f32 weight and bf16 source scales",
            gated_beta, s8, s8, f32, M, N, K, seed += 4));
    {
        refused_t &r = *v.back();
        gated_with_beta(r);
        auto &q = r.c.params.quant_params;
        q.src_scale.buff = half_scales(r, 1, 0x3f00);
        q.src_scale.dt = bf16;
        r.kept = scales;
    }
    // W4A16 of {4, N} weight scales, and an f32 column-major call on a
    // constant weight.
    v.push_back(make_case("bf16 x s4 with {4, N} weight scales",
            "s4 weight with a source other than s8", bf16, s4, bf16, M, N, K,
            7880));
    {
        refused_t &r = *v.back();
        r.wei_scales.assign(static_cast<size_t>(4) * N, 0.25f);
        auto &q = r.c.params.quant_params;
        q.wei_scale.buff = r.wei_scales.data();
        q.wei_scale.dims = {4, N};
        r.kept = w4a16;
    }
    v.push_back(make_case("f32 column-major on a constant weight",
            "a column-major call", f32, f32, f32, M, N, K, 7884));
    column_major(*v.back());
    // Operand types the default route has no kernel for: float operands of
    // two types, an integer source with a float weight, W4A16 into an
    // integer dst, and an s4 weight with an f32 or u8 source (without
    // scales, which matmul_direct's checks take with a bf16 source only).
    const char *types = "a call of these source, weight and dst types";
    v.push_back(make_case("bf16 x f32 into f32",
            "f32 weight with a non-f32 source", bf16, f32, f32, M, N, K, 7888));
    v.back()->kept = types;
    v.push_back(make_case("f32 x bf16 into f32",
            "bf16 weight with a non-bf16 source", f32, bf16, f32, M, N, K,
            7892));
    v.back()->kept = types;
    v.push_back(make_case("u8 x bf16 into f32",
            "bf16 weight with a non-bf16 source", u8, bf16, f32, M, N, K,
            7896));
    v.back()->kept = types;
    v.push_back(make_case("s8 x f32 into f32",
            "f32 weight with a non-f32 source", s8, f32, f32, M, N, K, 7900));
    v.back()->kept = types;
    v.push_back(make_case("bf16 x s4 into an s32 dst",
            "s4 weight with a source other than s8", bf16, s4, s32, M, N, K,
            7904));
    v.back()->kept = "a call of float operands into an integer dst";
    for (const data_type_t src_dt : {f32, u8}) {
        v.push_back(make_case(src_dt == f32 ? "f32 x s4 without scales"
                                            : "u8 x s4 without scales",
                src_dt == f32 ? "s4 weight with a source other than s8"
                              : "a u8 source with an s4 weight",
                src_dt, s4, f32, M, N, K, src_dt == f32 ? 7908 : 7912));
        v.back()->c.params.quant_params.wei_scale = {};
        v.back()->kept = types;
    }
    // Weight zero points: of an s4 weight, of one value per column, over a
    // source the call quantizes, and with alpha.
    const char *zero_point
            = "a weight zero point of more than one value, or over a source "
              "it quantizes";
    v.push_back(make_case("bf16 x s4 with a weight zero point",
            "s4 weight with a source other than s8", bf16, s4, f32, M, N, K,
            7916));
    {
        refused_t &r = *v.back();
        auto &q = r.c.params.quant_params;
        q.wei_zp.buff = &r.zp;
        q.wei_zp.dt = s32;
        q.wei_zp.dims = {1};
        r.kept = "W4A16 with a zero point of its s4 weight";
    }
    v.push_back(make_case("u8 x s8 with a weight zero point per column",
            "weight zero points", u8, s8, f32, M, N, K, 7920));
    {
        refused_t &r = *v.back();
        r.wei_zps.resize(N);
        for (int n = 0; n < N; ++n)
            r.wei_zps[n] = static_cast<int8_t>(n % 7 - 3);
        auto &q = r.c.params.quant_params;
        q.wei_zp.buff = r.wei_zps.data();
        q.wei_zp.dt = s8;
        q.wei_zp.dims = {1, N};
        r.kept = zero_point;
    }
    {
        v.push_back(make_case("DA8W8 with a weight zero point",
                "weight zero points", bf16, s8, f32, M, N, K, 7924));
        refused_t &r = *v.back();
        r.c.params.dynamic_quant = true;
        r.c.params.dtypes.compute = s8;
        auto &q = r.c.params.quant_params;
        q.src_scale.buff = nullptr;
        q.src_scale.dims = {M, 1};
        q.wei_zp.buff = &r.zp;
        q.wei_zp.dt = s32;
        q.wei_zp.dims = {1};
        r.kept = zero_point;
    }
    v.push_back(make_case("u8 x s8 with a weight zero point and alpha",
            "weight zero points", u8, s8, f32, M, N, K, 7928));
    {
        refused_t &r = *v.back();
        r.c.alpha = 0.5f;
        auto &q = r.c.params.quant_params;
        q.wei_zp.buff = &r.zp;
        q.wei_zp.dt = s32;
        q.wei_zp.dims = {1};
        r.kept = "zero points with alpha";
    }
    // An f16 source with dynamic quantization, which the default route
    // does not quantize (matmul_direct refuses an f16 source on a CPU
    // without AVX512-FP16 first).
    if (zendnnl::common::zendnnl_platform_info().get_avx512_f16_status()) {
        v.push_back(make_case("f16 x s8 with dynamic quantization", "",
                data_type_t::f16, s8, f32, M, N, K, 7932));
        refused_t &r = *v.back();
        r.c.is_weights_const = false;
        r.c.params.dynamic_quant = true;
        r.c.params.dtypes.compute = s8;
        auto &q = r.c.params.quant_params;
        q.src_scale.buff = nullptr;
        q.src_scale.dims = {M, 1};
        r.kept = "W8A16";
    }
    for (auto &rp : refused_cases())
        if (rp->kept != nullptr) v.push_back(std::move(rp));
    return v;
}

// Under ZENDNNL_MATMUL_ALGO=10 a call of a form the default route does not
// compute correctly keeps the native BRGEMM's error (native_brgemm_refusal
// and matmul_direct give unimplemented), with dst holding its values, the
// weight its bytes, no native layout, and one error line that names the
// native reason and why the call is not handed back, and no default-route
// line.  With lowoha_algo native_brgemm the error and its line are as for
// any other refusal.
TEST(NativeBrgemmRefusal, WhatTheDefaultRouteComputesWronglyKeepsItsError) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    for (const bool per_call : {false, true})
        for (const auto &rp : kept_cases()) {
            refused_t &r = *rp;
            SCOPED_TRACE(std::string(r.name)
                    + (per_call ? ", lowoha_algo native_brgemm"
                                : ", ZENDNNL_MATMUL_ALGO=10"));
            const nt::algo_pin_t pin(per_call ? matmul_algo_t::none
                                              : matmul_algo_t::native_brgemm);
            nt::call_t c = r.c;
            if (per_call) c.params.lowoha_algo = matmul_algo_t::native_brgemm;
            const std::vector<uint8_t> raw(static_cast<uint8_t *>(r.wei.get()),
                    static_cast<uint8_t *>(r.wei.get()) + r.wei.bytes);
            std::string reason;
            EXPECT_EQ(refusal_in(c, r.layout, reason), status_t::unimplemented);
            EXPECT_NE(reason.find(r.reason), std::string::npos) << reason;
            std::string log;
            EXPECT_EQ(run_logged(c, log, r.layout), status_t::unimplemented);
            if (apilog_error_enabled()) {
                EXPECT_EQ(
                        lines_with(log, "native BRGEMM did not serve the call"),
                        1)
                        << log;
                EXPECT_EQ(
                        lines_with(log, "; not handed back: the default route"),
                        per_call ? 0 : 1)
                        << log;
                EXPECT_EQ(lines_with(log, r.kept), per_call ? 0 : 1) << log;
                EXPECT_EQ(lines_with(log,
                                  "lowoha_algo native_brgemm takes no other "
                                  "route"),
                        per_call ? 1 : 0)
                        << log;
            }
            EXPECT_EQ(lines_with(log, "takes the default route"), 0) << log;
            EXPECT_EQ(lines_with(log, "runs only on the native BRGEMM"), 0)
                    << log;
            EXPECT_EQ(std::memcmp(r.dst.get(), r.prior.get(), r.dst.bytes), 0);
            EXPECT_EQ(std::memcmp(r.wei.get(), raw.data(), raw.size()), 0);
            EXPECT_EQ(nt::native_layouts(r.wei.get()), 0u);
        }
}

/// The values of the dst of a row-major call c of float operands that
/// differ from the f64 model of the call by more than f32 accumulation
/// error and the rounding of the dst, described.
std::string float_mismatches(const nt::call_t &c) {
    const data_type_t src_dt = c.params.dtypes.src,
                      wei_dt = c.params.dtypes.wei,
                      dst_dt = c.params.dtypes.dst;
    std::string out;
    for (int m = 0; m < c.M; ++m)
        for (int n = 0; n < c.N; ++n) {
            double want = 0.0, mag = 0.0;
            for (int k = 0; k < c.K; ++k) {
                const double t = static_cast<double>(nt::load_f32(c.src, src_dt,
                                         static_cast<size_t>(m) * c.K + k))
                        * nt::load_f32(c.wei, wei_dt,
                                static_cast<size_t>(k) * c.N + n);
                want += t;
                mag += std::fabs(t);
            }
            const double got = nt::load_f32(
                    c.dst, dst_dt, static_cast<size_t>(m) * c.N + n);
            const double tol = (dst_dt == bf16 ? std::fabs(want) / 128 : 0.0)
                    + 1e-5 * mag + 1e-6;
            if (std::fabs(got - want) > tol && out.size() < 400)
                out += " (" + std::to_string(m) + ", " + std::to_string(n)
                        + "): " + std::to_string(got) + " for "
                        + std::to_string(want);
        }
    return out;
}

// Under ZENDNNL_MATMUL_ALGO=10, a row-major call of each pair of f32 / bf16
// source and weight types into each f32 / bf16 dst matmul_direct's checks
// take either returns success with the dst of the f64 model of the call,
// whose values are not zero, or keeps the native BRGEMM's error with dst
// holding its values: a pair of two types, which the native BRGEMM does
// not serve and the default route has no kernel for, keeps its error.
TEST(NativeBrgemmRefusal, FloatOperandsOfAnyTypesMeetTheModelOrKeepTheError) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    uint32_t seed = 8200;
    for (const data_type_t src_dt : {f32, bf16})
        for (const data_type_t wei_dt : {f32, bf16})
            for (const data_type_t dst_dt : {f32, bf16}) {
                if (src_dt == f32 && dst_dt == bf16) continue;
                auto rp = make_case(
                        "", "", src_dt, wei_dt, dst_dt, 4, 64, 64, seed += 4);
                refused_t &r = *rp;
                SCOPED_TRACE(std::string(dtype_info(src_dt)) + " x "
                        + dtype_info(wei_dt) + " into " + dtype_info(dst_dt));
                const status_t st = nt::run_matmul(r.c);
                if (st == status_t::success) {
                    EXPECT_EQ(float_mismatches(r.c), "");
                } else {
                    EXPECT_NE(src_dt, wei_dt);
                    EXPECT_EQ(st, status_t::unimplemented);
                    EXPECT_EQ(std::memcmp(
                                      r.dst.get(), r.prior.get(), r.dst.bytes),
                            0);
                }
                mm::clear_matmul_weight_caches();
            }
}

// Next to those forms, the calls the default route computes still take it
// under ZENDNNL_MATMUL_ALGO=10 (native_brgemm_refusal gives success, and
// matmul_direct returns the status and dst of the same call with the
// algorithm unset): a column-major bf16 call on a weight that changes (in a
// build with AOCL-DLP: AColumnMajorCallIsHandedBackOnlyWhereItIsComputed),
// and a source quantized per token with matched f32 scales and K past the
// int32 bound.
TEST(NativeBrgemmRefusal, TheFormsNextToThemStillTakeTheDefaultRoute) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    std::vector<std::unique_ptr<refused_t>> v;
    v.push_back(make_case("bf16 column-major on a weight that changes",
            "a column-major call", bf16, bf16, bf16, 4, 64, 64, 7900));
    {
        refused_t &r = *v.back();
        r.layout = 'c';
        r.c.lda = r.c.M;
        r.c.ldb = r.c.K;
        r.c.ldc = r.c.M;
        r.c.is_weights_const = false;
    }
    v.push_back(make_case("DA8W8 with f32 scales and K past the int32 bound",
            "K exceeds the int32 accumulation bound", bf16, s8, f32, 2, 32,
            nat::kMaxSafeInt8AccumK + 4, 7904));
    {
        refused_t &r = *v.back();
        r.c.params.dynamic_quant = true;
        r.c.params.dtypes.compute = s8;
        auto &q = r.c.params.quant_params;
        q.src_scale.buff = nullptr;
        q.src_scale.dims = {2, 1};
    }
    for (const auto &rp : v) {
        refused_t &r = *rp;
        if (r.layout != 'r' && !ZENDNNL_DEPENDS_AOCLDLP) continue;
        SCOPED_TRACE(r.name);
        nt::buffer_t unset = nt::make_buffer(r.dst.bytes);
        std::memcpy(unset.get(), r.prior.get(), r.dst.bytes);
        status_t unset_status = status_t::failure;
        {
            const nt::algo_pin_t pin(matmul_algo_t::none);
            nt::call_t c = r.c;
            c.dst = unset.get();
            std::string log;
            unset_status = run_logged(c, log, r.layout);
        }
        const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
        std::string reason;
        EXPECT_EQ(refusal_in(r.c, r.layout, reason), status_t::success)
                << reason;
        std::string log;
        EXPECT_EQ(run_logged(r.c, log, r.layout), unset_status);
        EXPECT_EQ(lines_with(log, "not handed back"), 0) << log;
        EXPECT_EQ(std::memcmp(r.dst.get(), unset.get(), r.dst.bytes), 0)
                << "dst differs from the default route's";
        EXPECT_EQ(nt::native_layouts(r.wei.get()), 0u);
        mm::clear_matmul_weight_caches();
    }
}

/// A column-major bf16 call into an f32 dst on a weight that changes: src
/// [M, K] M apart, the weight [K, N] ldb apart, or [N, K] when transB, dst
/// [M, N] M apart.
std::unique_ptr<refused_t> column_major_case(
        int N, int K, bool transB, int ldb, uint32_t seed) {
    const int M = 4;
    auto r = make_case("bf16 column-major on a weight that changes",
            "a column-major call", bf16, bf16, f32, M, N, K, seed);
    r->layout = 'c';
    r->c.transB = transB;
    r->c.lda = M;
    r->c.ldb = ldb;
    r->c.ldc = M;
    r->c.is_weights_const = false;
    const size_t elems = static_cast<size_t>(ldb) * (transB ? K : N);
    if (elems * 2 > r->wei.bytes) {
        r->wei = nt::kept_until_exit(
                nt::random_buffer(bf16, elems, seed + 1, -1.0f, 1.0f));
        r->c.wei = r->wei.get();
    }
    return r;
}

/// The values of a column_major_case's dst that differ from the f64 model
/// of the call by more than f32 accumulation error, described.
std::string column_major_mismatches(const nt::call_t &c) {
    std::string out;
    for (int n = 0; n < c.N; ++n)
        for (int m = 0; m < c.M; ++m) {
            double want = 0.0, mag = 0.0;
            for (int k = 0; k < c.K; ++k) {
                const size_t w = c.transB ? static_cast<size_t>(k) * c.ldb + n
                                          : static_cast<size_t>(n) * c.ldb + k;
                const double t = static_cast<double>(nt::load_f32(c.src, bf16,
                                         static_cast<size_t>(k) * c.lda + m))
                        * nt::load_f32(c.wei, bf16, w);
                want += t;
                mag += std::fabs(t);
            }
            const double got = nt::load_f32(
                    c.dst, f32, static_cast<size_t>(n) * c.ldc + m);
            if (std::fabs(got - want) > 1e-5 * mag + 1e-6 && out.size() < 400)
                out += " (" + std::to_string(m) + ", " + std::to_string(n)
                        + "): " + std::to_string(got) + " for "
                        + std::to_string(want);
        }
    return out;
}

// A column-major call on a weight that changes, under ZENDNNL_MATMUL_ALGO=10,
// of N 64 and K 64, and of the tight ldb with K below N, or N below K under
// transB: its ldb spans a row of the weight as the caller lays it out, K
// elements of a [K, N] weight and N of an [N, K] one.  In a build with
// AOCL-DLP the default route computes it, and the call is handed back
// (native_brgemm_refusal gives success) with the f64 model's result;
// without AOCL-DLP the default route is the reference kernel, which
// computes row-major calls only, and the call keeps the native error, with
// dst untouched and one error line that says why.
TEST(NativeBrgemmRefusal, AColumnMajorCallIsHandedBackOnlyWhereItIsComputed) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    struct shape_t {
        int N, K;
        bool transB;
        uint32_t seed;
    };
    for (const shape_t s : {shape_t {64, 64, false, 7950},
                 shape_t {64, 32, false, 8024}, shape_t {32, 64, true, 7992}}) {
        SCOPED_TRACE("N " + std::to_string(s.N) + " K " + std::to_string(s.K)
                + (s.transB ? " transB" : ""));
        auto rp = column_major_case(
                s.N, s.K, s.transB, s.transB ? s.N : s.K, s.seed);
        refused_t &r = *rp;
        std::string reason, log;
        const status_t refusal = refusal_in(r.c, 'c', reason);
        const status_t st = run_logged(r.c, log, 'c');
        if (ZENDNNL_DEPENDS_AOCLDLP) {
            EXPECT_EQ(refusal, status_t::success) << reason;
            EXPECT_EQ(st, status_t::success) << log;
            EXPECT_EQ(column_major_mismatches(r.c), "");
        } else {
            EXPECT_EQ(refusal, status_t::unimplemented);
            EXPECT_EQ(st, status_t::unimplemented);
            if (apilog_error_enabled()) {
                EXPECT_EQ(lines_with(log,
                                  "not handed back: the default route of a "
                                  "build without AOCL-DLP does not compute a "
                                  "column-major call"),
                        1)
                        << log;
            }
            EXPECT_EQ(std::memcmp(r.dst.get(), r.prior.get(), r.dst.bytes), 0);
        }
    }
}

/// Refused calls of the forms the reference kernel does not compute, kept
/// naming what their error line says in a build without AOCL-DLP: a
/// binary_add operand whose rows are N + 3 values apart (W4A16, u4 with a
/// zero point, u8 x s8 with a weight zero point), weight scales in groups
/// of 6 with one source scale value, and a pooling post-op.
std::vector<std::unique_ptr<refused_t>> reference_kernel_cases() {
    std::vector<std::unique_ptr<refused_t>> v;
    const int M = 8, N = 64, K = 128;
    const char *operand = "a binary operand whose rows are not N values apart";
    const auto padded_operand = [&](refused_t &r) {
        r.operand_ld = N + 3;
        r.operand.resize(static_cast<size_t>(M) * r.operand_ld);
        for (size_t i = 0; i < r.operand.size(); ++i)
            r.operand[i] = static_cast<float>(i % 13) * 0.125f - 0.75f;
        mm::matmul_post_op po;
        po.po_type = zendnnl::common::post_op_type_t::binary_add;
        po.buff = r.operand.data();
        po.dtype = f32;
        po.dims = {M, N};
        po.leading_dim = r.operand_ld;
        r.c.params.postop_.push_back(po);
        r.kept = operand;
    };
    v.push_back(make_case("W4A16 with a padded binary operand", "", bf16, s4,
            f32, M, N, K, 7970));
    padded_operand(*v.back());
    v.push_back(
            make_case("bf16 x u4 with a zero point and a padded binary "
                      "operand",
                    "", bf16, u4, bf16, M, N, K, 7974));
    {
        refused_t &r = *v.back();
        padded_operand(r);
        auto &q = r.c.params.quant_params;
        q.wei_zp.buff = &r.zp;
        q.wei_zp.dt = s8;
        q.wei_zp.dims = {1, 1};
    }
    v.push_back(
            make_case("u8 x s8 with a weight zero point and a padded "
                      "binary operand",
                    "", u8, s8, f32, M, N, K, 7978));
    {
        refused_t &r = *v.back();
        padded_operand(r);
        auto &q = r.c.params.quant_params;
        q.wei_zp.buff = &r.zp;
        q.wei_zp.dt = s32;
        q.wei_zp.dims = {1};
    }
    for (const data_type_t src_dt : {s8, u8}) {
        v.push_back(make_case(src_dt == s8 ? "s8 x s8 in groups of 6"
                                           : "u8 x s8 in groups of 6",
                "", src_dt, s8, f32, M, N, 96, src_dt == s8 ? 7982 : 7986));
        refused_t &r = *v.back();
        r.wei_scales.assign(static_cast<size_t>(16) * N, 0.25f);
        auto &q = r.c.params.quant_params;
        q.wei_scale.buff = r.wei_scales.data();
        q.wei_scale.dims = {16, N};
        r.kept = "per-group weight scales without source scales of the same "
                 "groups";
    }
    v.push_back(make_case(
            "bf16 with a pooling post-op", "", bf16, bf16, f32, M, N, K, 7990));
    {
        mm::matmul_post_op po;
        po.po_type = zendnnl::common::post_op_type_t::pooling;
        v.back()->c.params.postop_.push_back(po);
        v.back()->kept = "a pooling post-op";
    }
    return v;
}

// Under ZENDNNL_MATMUL_ALGO=10 the forms the reference kernel does not
// compute: in a build without AOCL-DLP, whose default route it is, each
// keeps the native error, with dst holding its values and one error line
// that says why; with AOCL-DLP each is handed back and has the dst of the
// f64 model of the call, or the default route's error with dst holding its
// values.
TEST(NativeBrgemmRefusal, WhatTheReferenceKernelDoesNotComputeKeepsItsError) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    const nt::algo_pin_t pin(matmul_algo_t::native_brgemm);
    for (const auto &rp : reference_kernel_cases()) {
        refused_t &r = *rp;
        SCOPED_TRACE(r.name);
        std::string reason, log;
        const status_t refusal = refusal_in(r.c, 'r', reason);
        const status_t st = run_logged(r.c, log);
        const bool untouched
                = std::memcmp(r.dst.get(), r.prior.get(), r.dst.bytes) == 0;
        if (!ZENDNNL_DEPENDS_AOCLDLP) {
            EXPECT_EQ(refusal, status_t::unimplemented) << reason;
            EXPECT_EQ(st, status_t::unimplemented);
            if (apilog_error_enabled()) {
                EXPECT_EQ(lines_with(log, r.kept), 1) << log;
                EXPECT_EQ(lines_with(log,
                                  "not handed back: the default route "
                                  "of a build without AOCL-DLP"),
                        1)
                        << log;
            }
            EXPECT_TRUE(untouched);
        } else if (st == status_t::success) {
            EXPECT_EQ(refusal, status_t::success) << reason;
            EXPECT_EQ(model_mismatches(r), "")
                    << "dst differs from the model of the call";
        } else {
            EXPECT_EQ(lines_with(log, "not handed back"), 0) << log;
            EXPECT_TRUE(untouched);
        }
        mm::clear_matmul_weight_caches();
    }
}

// A per-group INT8 call into an f16 dst, which matmul_direct passes on to
// the native BRGEMM (on a CPU without AVX512-FP16 too, for a call that asks
// for it): the INT8 tile's check turns it away and names the class.
TEST(NativeBrgemmRefusal, PerGroupF16OutputIsUnimplemented) {
    const int M = 4, N = 64, K = 128, G = 4;
    std::vector<float> wei_scales(static_cast<size_t>(G) * N, 0.5f);
    const float src_scale = 0.25f;
    mm::matmul_params p;
    p.dtypes.src = s8;
    p.dtypes.wei = s8;
    p.dtypes.dst = data_type_t::f16;
    p.quant_params.src_scale.buff = &src_scale;
    p.quant_params.src_scale.dt = f32;
    p.quant_params.src_scale.dims = {1, 1};
    p.quant_params.wei_scale.buff = wei_scales.data();
    p.quant_params.wei_scale.dt = f32;
    p.quant_params.wei_scale.dims = {G, N};
    nat::GemmDescriptor d;
    d.M = M;
    d.N = N;
    d.K = K;
    d.lda = K;
    d.ldb = N;
    d.ldc = N;
    d.src_dt = s8;
    d.wei_dt = s8;
    d.dst_dt = data_type_t::f16;
    const char *why = nullptr;
    EXPECT_EQ(nat::check_int8_call(
                      d, p, nat::decide_int8_call(d, p, nullptr), &why),
            status_t::unimplemented);
    ASSERT_NE(why, nullptr);
    EXPECT_NE(std::string(why).find("per-group call into a dst"),
            std::string::npos)
            << why;
}

} // namespace
