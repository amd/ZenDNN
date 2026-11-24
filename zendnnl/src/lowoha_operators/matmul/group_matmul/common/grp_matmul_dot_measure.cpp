/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_machine.hpp"

#if defined(_MSC_VER) && !defined(__clang__)

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace grp_machine {

bool measure_dot_rates(
        const native::UarchParams &, std::array<dot_rate, kDotPaths> &out) {
    out = {};
    return false;
}

} // namespace grp_machine
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#else

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>

#if defined(ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS) \
        || (defined(__GNUC__) && (__GNUC__ >= 12))
#define ZENDNNL_GRP_MACHINE_FP16 1
#else
#define ZENDNNL_GRP_MACHINE_FP16 0
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace grp_machine {

namespace {

// The loops issue their instructions from volatile asm: with intrinsics a
// compiler may copy every accumulator between iterations, and the copies,
// not the instruction, then set the rate; volatile, every chain runs
// whatever a loop returns.

using loop_fn = double (*)(uint64_t);

volatile int g_one = 1;

constexpr int kAddsPerIter = 32;

double add_chain(uint64_t iters) {
    uint64_t x = iters;
    for (uint64_t i = 0; i < iters; ++i) {
#define GRP_ADD4 "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
        asm volatile(GRP_ADD4 GRP_ADD4 GRP_ADD4 GRP_ADD4 GRP_ADD4 GRP_ADD4
                             GRP_ADD4 GRP_ADD4
                     : "+r"(x));
#undef GRP_ADD4
    }
    return static_cast<double>(x);
}

#define GRP_OP(insn, n) insn " %[x], %[x], %[a" #n "]\n\t"
#define GRP_OPS8(insn, b0, b1, b2, b3, b4, b5, b6, b7) \
    asm volatile(GRP_OP(insn, 0) GRP_OP(insn, 1) GRP_OP(insn, 2) GRP_OP(insn, \
            3) GRP_OP(insn, 4) GRP_OP(insn, 5) GRP_OP(insn, 6) GRP_OP(insn, 7) \
                 : [a0] "+v"(b0), [a1] "+v"(b1), [a2] "+v"(b2), [a3] "+v"(b3), \
                 [a4] "+v"(b4), [a5] "+v"(b5), [a6] "+v"(b6), [a7] "+v"(b7) \
                 : [x] "v"(x))
/// 24 independent chains of insn on a0 .. a23, more than any core needs to
/// hide its latency.
#define GRP_TPUT24(T, insn) \
    T a0 = x, a1 = x, a2 = x, a3 = x, a4 = x, a5 = x, a6 = x, a7 = x, a8 = x, \
      a9 = x, a10 = x, a11 = x, a12 = x, a13 = x, a14 = x, a15 = x, a16 = x, \
      a17 = x, a18 = x, a19 = x, a20 = x, a21 = x, a22 = x, a23 = x; \
    for (uint64_t it = 0; it < iters; ++it) { \
        GRP_OPS8(insn, a0, a1, a2, a3, a4, a5, a6, a7); \
        GRP_OPS8(insn, a8, a9, a10, a11, a12, a13, a14, a15); \
        GRP_OPS8(insn, a16, a17, a18, a19, a20, a21, a22, a23); \
    }
/// One chain of 8 dependent insn an iteration, on a0.
#define GRP_LAT8(T, insn) \
    T a0 = x; \
    for (uint64_t it = 0; it < iters; ++it) \
    asm volatile(GRP_OP(insn, 0) GRP_OP(insn, 0) GRP_OP(insn, 0) GRP_OP(insn, \
            0) GRP_OP(insn, 0) GRP_OP(insn, 0) GRP_OP(insn, 0) GRP_OP(insn, 0) \
                 : [a0] "+v"(a0) \
                 : [x] "v"(x))

__attribute__((target("avx512f"))) double tput_fp32(uint64_t iters) {
    const __m512 x = _mm512_set1_ps(static_cast<float>(g_one) * 1e-7f);
    GRP_TPUT24(__m512, "vfmadd231ps");
    return _mm512_reduce_add_ps(a0);
}

__attribute__((target("avx512f"))) double lat_fp32(uint64_t iters) {
    const __m512 x = _mm512_set1_ps(static_cast<float>(g_one) * 1e-7f);
    GRP_LAT8(__m512, "vfmadd231ps");
    return _mm512_reduce_add_ps(a0);
}

// bf16 0.001 (0x3a83) pairs: an accumulator grows by 2e-6 an instruction.
__attribute__((target("avx512f,avx512bf16"))) double tput_bf16(uint64_t iters) {
    const __m512 x = _mm512_castsi512_ps(
            _mm512_set1_epi16(static_cast<short>(0x3a83 * g_one)));
    GRP_TPUT24(__m512, "vdpbf16ps");
    return _mm512_reduce_add_ps(a0);
}

__attribute__((target("avx512f,avx512bf16"))) double lat_bf16(uint64_t iters) {
    const __m512 x = _mm512_castsi512_ps(
            _mm512_set1_epi16(static_cast<short>(0x3a83 * g_one)));
    GRP_LAT8(__m512, "vdpbf16ps");
    return _mm512_reduce_add_ps(a0);
}

// u8 1 times s8 1, four a lane: an int32 accumulator wraps only after 5e8
// instructions.
__attribute__((target("avx512f,avx512vnni"))) double tput_int8(uint64_t iters) {
    const __m512i x = _mm512_set1_epi8(static_cast<char>(g_one));
    GRP_TPUT24(__m512i, "vpdpbusd");
    return static_cast<double>(_mm512_reduce_add_epi32(a0));
}

__attribute__((target("avx512f,avx512vnni"))) double lat_int8(uint64_t iters) {
    const __m512i x = _mm512_set1_epi8(static_cast<char>(g_one));
    GRP_LAT8(__m512i, "vpdpbusd");
    return static_cast<double>(_mm512_reduce_add_epi32(a0));
}

#if ZENDNNL_GRP_MACHINE_FP16
// The halves of 0 * 0 + acc: an accumulator stays at its start.
__attribute__((target("avx512f"))) double tput_fp16(uint64_t iters) {
    const __m512i x = _mm512_set1_epi16(static_cast<short>(0 * g_one));
    GRP_TPUT24(__m512i, "vfmadd231ph");
    return static_cast<double>(_mm512_reduce_or_epi32(a0));
}

__attribute__((target("avx512f"))) double lat_fp16(uint64_t iters) {
    const __m512i x = _mm512_set1_epi16(static_cast<short>(0 * g_one));
    GRP_LAT8(__m512i, "vfmadd231ph");
    return static_cast<double>(_mm512_reduce_or_epi32(a0));
}
#endif

#undef GRP_LAT8
#undef GRP_TPUT24
#undef GRP_OPS8
#undef GRP_OP

volatile double g_sink = 0;

/// Nanoseconds of one run of f(iters) after a warm-up.
__attribute__((noinline)) double nanoseconds(loop_fn f, uint64_t iters) {
    g_sink = g_sink + f(iters / 8 + 1);
    const auto t0 = std::chrono::steady_clock::now();
    g_sink = g_sink + f(iters);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

/// One-cycle adds per timing of them, and iterations per timing of a path's
/// throughput and latency loops.
constexpr uint64_t kAddIters = 16384;
constexpr uint64_t kTputIters = 32768;
constexpr uint64_t kLatIters = 8192;

struct measured_path_t {
    dot_path path;
    bool have;
    loop_fn tput;
    loop_fn lat;
};

} // namespace

bool measure_dot_rates(
        const native::UarchParams &u, std::array<dot_rate, kDotPaths> &out) {
    out = {};
    const measured_path_t paths[] = {
            {dot_path::fp32, u.avx512f, tput_fp32, lat_fp32},
            {dot_path::bf16, u.avx512bf16, tput_bf16, lat_bf16},
            {dot_path::int8, u.avx512vnni, tput_int8, lat_int8},
#if ZENDNNL_GRP_MACHINE_FP16
            {dot_path::fp16, u.avx512fp16, tput_fp16, lat_fp16},
#endif
    };
    // Each path's adds, throughput loop and latency loop keep their fastest
    // run over rounds that take every path in turn: a run that interference
    // slows, whether the loop's or the adds' it is scaled by, is not kept.
    constexpr int kRounds = 3;
    struct fastest_t {
        double add = 1e30, tput = 1e30, lat = 1e30;
    } best[kDotPaths];
    // An untimed AVX-512 loop first: on a CPU that lowers its clock for
    // AVX-512 code, every timing of the adds then runs at the clock the
    // loops run at.
    for (const measured_path_t &p : paths) {
        if (!p.have) continue;
        g_sink = g_sink + p.tput(kTputIters);
        break;
    }
    bool any = false;
    for (int r = 0; r < kRounds; ++r) {
        for (const measured_path_t &p : paths) {
            if (!p.have) continue;
            any = true;
            fastest_t &f = best[static_cast<int>(p.path)];
            f.add = std::min(f.add, nanoseconds(add_chain, kAddIters));
            f.tput = std::min(f.tput, nanoseconds(p.tput, kTputIters));
            f.lat = std::min(f.lat, nanoseconds(p.lat, kLatIters));
        }
    }
    for (const measured_path_t &p : paths) {
        if (!p.have) continue;
        const fastest_t &f = best[static_cast<int>(p.path)];
        const double add_ns
                = f.add / (static_cast<double>(kAddIters) * kAddsPerIter);
        const double tput_cycles = f.tput / kTputIters / add_ns;
        const double lat_cycles = f.lat / kLatIters / add_ns;
        dot_rate &d = out[static_cast<int>(p.path)];
        d.ipc = std::max(0.25, std::round(24 / tput_cycles * 4) / 4);
        d.latency = std::max(1, static_cast<int>(std::lround(lat_cycles / 8)));
    }
    return any;
}

} // namespace grp_machine
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif
