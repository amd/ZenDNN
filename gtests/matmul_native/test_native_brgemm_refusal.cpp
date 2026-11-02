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

// The calls that ask for the native BRGEMM and that it does not serve: each
// returns its error status from matmul_direct, with dst untouched, the
// weight left raw, and one error line naming the reason.  Calls of N not a
// multiple of 32 are served.

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_call.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "native_brgemm_test_utils.hpp"

namespace {

namespace mm = zendnnl::lowoha::matmul;
namespace nat = zendnnl::lowoha::matmul::native;
namespace nt = native_brgemm_test;
using zendnnl::common::matmul_algo_t;
using zendnnl::memory::data_type_t;
using zendnnl::memory::status_t;

const data_type_t bf16 = data_type_t::bf16, f32 = data_type_t::f32,
                  s8 = data_type_t::s8, u8 = data_type_t::u8,
                  s32 = data_type_t::s32, s4 = data_type_t::s4;

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
    r->wei = nt::kept_until_exit(wei_dt == s4
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
    if (int8 || wei_dt == s4) {
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
    v.push_back(make_case("bf16 x s4", "weight type other than bf16 / s8 / f32",
            bf16, s4, bf16, 4, 64, 64, seed += 4));
    v.push_back(make_case("bf16 x s8 without dynamic quantization",
            "s8 weight with a source other than u8 / s8", bf16, s8, bf16, 4, 64,
            64, seed += 4));
    v.back()->c.params.quant_params.src_scale = {};
    v.push_back(make_case("u8 x s8 with K past the int32 bound",
            "K exceeds the int32 accumulation bound", u8, s8, f32, 1, 32,
            nat::kMaxSafeInt8AccumK + 4, seed += 4));
    v.push_back(make_case("u8 x s8 in a GGML pack", "a GGML pack format", u8,
            s8, f32, 4, 64, 64, seed += 4));
    v.back()->c.params.packing.pack_format_b = 1;
    v.push_back(make_case("bf16 with a reordered source", "a reordered source",
            bf16, bf16, bf16, 4, 64, 64, seed += 4));
    v.back()->c.params.mem_format_a = 'r';
    return v;
}

// Each class, per call (lowoha_algo native_brgemm) and under
// ZENDNNL_MATMUL_ALGO=11: native_brgemm_refusal gives its status and names
// its reason, and matmul_direct returns that status, with dst holding its
// values, the weight its bytes, no native layout of the weight, and an error
// line naming the reason (run_matmul_direct).
TEST(NativeBrgemmRefusal, EveryClassReturnsItsStatus) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    for (const bool per_call : {true, false})
        for (const auto &rp : refused_cases()) {
            refused_t &r = *rp;
            SCOPED_TRACE(std::string(r.name)
                    + (per_call ? ", lowoha_algo native_brgemm"
                                : ", ZENDNNL_MATMUL_ALGO=11"));
            std::unique_ptr<nt::algo_pin_t> pin;
            if (per_call)
                r.c.params.lowoha_algo = matmul_algo_t::native_brgemm;
            else
                pin.reset(new nt::algo_pin_t(matmul_algo_t::native_brgemm));
            const std::vector<uint8_t> raw(static_cast<uint8_t *>(r.wei.get()),
                    static_cast<uint8_t *>(r.wei.get()) + r.wei.bytes);
            std::string reason;
            EXPECT_EQ(nt::refusal_of(r.c, &reason), r.status);
            EXPECT_NE(reason.find(r.reason), std::string::npos) << reason;
            EXPECT_TRUE(nt::refused_with(r.c, r.status));
            EXPECT_EQ(std::memcmp(r.dst.get(), r.prior.get(), r.dst.bytes), 0);
            EXPECT_EQ(std::memcmp(r.wei.get(), raw.data(), raw.size()), 0);
            EXPECT_EQ(nt::native_layouts(r.wei.get()), 0u);
        }
}

// Calls of N not a multiple of 32, per call and under
// ZENDNNL_MATMUL_ALGO=11: native_brgemm_refusal gives success, and
// matmul_direct serves them over a native layout of the weight, which stays
// raw, with the reference algorithm's result within the band of the type.
TEST(NativeBrgemmRefusal, CallsOfAnyNAreServed) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    for (const bool per_call : {true, false})
        for (const auto &rp : any_n_cases()) {
            refused_t &r = *rp;
            SCOPED_TRACE(std::string(r.name)
                    + (per_call ? ", lowoha_algo native_brgemm"
                                : ", ZENDNNL_MATMUL_ALGO=11"));
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
// point), per call and under ZENDNNL_MATMUL_ALGO=11.  matmul_direct then
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
                                    : ", ZENDNNL_MATMUL_ALGO=11"));
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
// algorithm unset, ZENDNNL_MATMUL_ALGO=11; under it, a weight the native
// BRGEMM takes as AOCL-DLP blocked.
TEST(NativeBrgemmRefusal, AReorderedWeightIsRefusedForWhatTheCallLacks) {
    if (!native_isa())
        GTEST_SKIP() << "needs AVX512-BF16, AVX512-VNNI and AVX-512 F";
    uint32_t seed = 7400;
    for (const bool algo11 : {false, true})
        for (const bool int8 : {false, true}) {
            const nt::algo_pin_t pin(algo11 ? matmul_algo_t::native_brgemm
                                            : matmul_algo_t::none);
            auto rp = make_case(int8 ? "u8 x s8 on a reordered weight"
                                     : "bf16 on a reordered weight",
                    algo11 ? "a reordered weight other than an AOCL-DLP "
                             "blocked one under ZENDNNL_MATMUL_ALGO=11"
                           : "a reordered weight without "
                             "ZENDNNL_MATMUL_ALGO=11",
                    int8 ? u8 : bf16, int8 ? s8 : bf16, int8 ? f32 : bf16, 4,
                    64, 64, seed += 4);
            refused_t &r = *rp;
            SCOPED_TRACE(std::string(r.name)
                    + (algo11 ? ", ZENDNNL_MATMUL_ALGO=11"
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

// A per-group INT8 call into an f16 dst, which matmul_direct passes on only
// on a CPU with AVX512-FP16: the INT8 tile's check turns it away and names
// the class.
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
