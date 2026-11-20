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

/// The op performance utility's probe: the compute and memory ceilings of this
/// machine at one thread count and pinning, written as the ceilings file
/// tools/op_perf_utility/common/ceilings.py reads (format op-perf-ceilings,
/// version 1).
///
///   probe --out ceilings.json [--threads N] [--domain-threads N]
///         [--repeats R] [--seconds S] [--quick] [--hugepages]
///
/// Run it the way the measured work runs: the same cpuset and
/// OMP_NUM_THREADS, with OMP_PROC_BIND=close OMP_PLACES=cores.  It measures
/// - the sustained clock, from a chain of dependent 1-cycle adds, on one core
///   and on every thread at once;
/// - per type the throughput (instructions per cycle, GFLOP/s of one core and
///   of the team) of the dot product the kernels use: FP32 FMA, the BF16 dot
///   product, the INT8 VNNI dot product and FP16 FMA (where the CPU has
///   AVX-512-FP16), with enough independent accumulators to hide latency, and
///   their latency in cycles from a dependent chain;
/// - the read bandwidth of 1 thread, one L3 domain's threads and the team,
///   each streaming a private buffer it touched first, sized for L1, L2, its
///   share of its domain's L3 and DRAM.  The buffers get the pages a plain
///   allocation gets, as the measured calls' buffers do; --hugepages advises
///   huge pages (MADV_HUGEPAGE) for them.
/// The CPU identity and caches come from the library's own detection
/// (native_hw()), named as the native BRGEMM tuner's schema names them.

#include <immintrin.h>
#include <omp.h>
#include <unistd.h>
#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"

#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 12)
#define PROBE_HAVE_FP16 1
#else
#define PROBE_HAVE_FP16 0
#endif

namespace {

namespace nat = zendnnl::lowoha::matmul::native;

constexpr const char *kProbeVersion = "1.1";
constexpr int kAcc = 16; // accumulators of a throughput kernel (PROBE_TPUT16)

// Operands the compiler cannot fold.
volatile float g_one = 1.0f;
volatile float g_tiny = 1.0e-7f;
volatile int g_byte = 1;

struct options_t {
    int threads = 0;
    int domain_threads = 0;
    int repeats = 5;
    double seconds = 0.2;
    bool quick = false;
    bool hugepages = false;
    std::string out;
};

struct stat_t {
    double median = 0.0;
    double rel_mad = 0.0;
};

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

stat_t stat_of(const std::vector<double> &v) {
    stat_t s;
    s.median = median_of(v);
    if (v.size() >= 2 && s.median > 0) {
        std::vector<double> d;
        for (double x : v)
            d.push_back(std::fabs(x - s.median));
        s.rel_mad = 1.4826 * median_of(d) / s.median;
    }
    return s;
}

// ---------------------------------------------------------------- clock

#define PROBE_ADD1 "add $1, %0\n\t"
#define PROBE_ADD8 \
    PROBE_ADD1 PROBE_ADD1 PROBE_ADD1 PROBE_ADD1 PROBE_ADD1 PROBE_ADD1 \
            PROBE_ADD1 PROBE_ADD1
constexpr double kAddsPerIter = 64.0;

uint64_t add_chain(uint64_t iters) {
    uint64_t x = iters;
    for (uint64_t i = 0; i < iters; ++i) {
        asm volatile(PROBE_ADD8 PROBE_ADD8 PROBE_ADD8 PROBE_ADD8 PROBE_ADD8
                             PROBE_ADD8 PROBE_ADD8 PROBE_ADD8
                     : "+r"(x));
    }
    return x;
}

/// The clock (GHz) of `threads` threads running the add chain at once: the
/// median of each thread's rate.
double clock_ghz(int threads, double seconds) {
    const auto iters = static_cast<uint64_t>(
            std::max(1000.0, seconds * 3.0e9 / kAddsPerIter));
    std::vector<double> rates(threads);
    volatile uint64_t sink = 0;
#pragma omp parallel num_threads(threads)
    {
        const int t = omp_get_thread_num();
        uint64_t r = add_chain(iters / 8);
#pragma omp barrier
        const double t0 = omp_get_wtime();
        r += add_chain(iters);
        const double t1 = omp_get_wtime();
        rates[t] = static_cast<double>(iters) * kAddsPerIter / (t1 - t0) / 1e9;
        if (r == 1) sink = r;
    }
    (void)sink;
    return median_of(rates);
}

// ---------------------------------------------------------------- compute

using kernel_fn = double (*)(uint64_t);

// The kernels issue their instructions from asm: with the intrinsics a compiler may
// copy every accumulator between iterations (GCC does so for VPDPBUSD), and the
// copies, not the instruction, then set the rate.
#define PROBE_DECL16(T, init) \
    T a0 = (init), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, \
      a7 = a0, a8 = a0, a9 = a0, a10 = a0, a11 = a0, a12 = a0, a13 = a0, \
      a14 = a0, a15 = a0
#define PROBE_OP(insn, n) insn " %[x], %[y], %[a" #n "]\n\t"
/// 16 instructions acc_i = insn(acc_i, y, x), one dependent chain per accumulator, in two
/// statements (an asm takes at most 30 operands, an in-out one counting twice).
#define PROBE_TPUT8(insn, b0, b1, b2, b3, b4, b5, b6, b7) \
    asm volatile(PROBE_OP(insn, 0) PROBE_OP(insn, 1) PROBE_OP(insn, 2) \
                         PROBE_OP(insn, 3) PROBE_OP(insn, 4) PROBE_OP(insn, 5) \
                                 PROBE_OP(insn, 6) PROBE_OP(insn, 7) \
                 : [a0] "+v"(b0), [a1] "+v"(b1), [a2] "+v"(b2), [a3] "+v"(b3), \
                 [a4] "+v"(b4), [a5] "+v"(b5), [a6] "+v"(b6), [a7] "+v"(b7) \
                 : [x] "v"(x), [y] "v"(y))
#define PROBE_TPUT16(insn) \
    PROBE_TPUT8(insn, a0, a1, a2, a3, a4, a5, a6, a7); \
    PROBE_TPUT8(insn, a8, a9, a10, a11, a12, a13, a14, a15)
/// 8 instructions of one dependent chain on a0.
#define PROBE_LAT8(insn) \
    asm volatile(PROBE_OP(insn, 0) PROBE_OP(insn, 0) PROBE_OP(insn, 0) \
                         PROBE_OP(insn, 0) PROBE_OP(insn, 0) PROBE_OP(insn, 0) \
                                 PROBE_OP(insn, 0) PROBE_OP(insn, 0) \
                 : [a0] "+v"(a0) \
                 : [x] "v"(x), [y] "v"(y))
#define PROBE_SUM16(add) \
    add(add(add(add(a0, a1), add(a2, a3)), add(add(a4, a5), add(a6, a7))), \
            add(add(add(a8, a9), add(a10, a11)), \
                    add(add(a12, a13), add(a14, a15))))
constexpr int kLatChain = 8; // instructions of one PROBE_LAT8

// acc += y * x: y 1e-7 and x 1, so acc grows slowly, never denormal.
__attribute__((target("avx512f"))) double tput_fp32(uint64_t iters) {
    PROBE_DECL16(__m512, _mm512_set1_ps(1.0f));
    const __m512 x = _mm512_set1_ps(g_one), y = _mm512_set1_ps(g_tiny);
    for (uint64_t it = 0; it < iters; ++it) {
        PROBE_TPUT16("vfmadd231ps");
    }
    return _mm512_reduce_add_ps(PROBE_SUM16(_mm512_add_ps));
}

__attribute__((target("avx512f"))) double lat_fp32(uint64_t iters) {
    __m512 a0 = _mm512_set1_ps(1.0f);
    const __m512 x = _mm512_set1_ps(g_one), y = _mm512_set1_ps(g_tiny);
    for (uint64_t it = 0; it < iters; ++it)
        PROBE_LAT8("vfmadd231ps");
    return _mm512_reduce_add_ps(a0);
}

// bf16 0.001 (0x3a83) pairs: acc grows by 2e-6 per instruction and lane.
__attribute__((target("avx512f,avx512bf16"))) double tput_bf16(uint64_t iters) {
    PROBE_DECL16(__m512, _mm512_setzero_ps());
    const __m512i x = _mm512_set1_epi16(static_cast<short>(0x3a83 * g_byte)),
                  y = x;
    for (uint64_t it = 0; it < iters; ++it) {
        PROBE_TPUT16("vdpbf16ps");
    }
    return _mm512_reduce_add_ps(PROBE_SUM16(_mm512_add_ps));
}

__attribute__((target("avx512f,avx512bf16"))) double lat_bf16(uint64_t iters) {
    __m512 a0 = _mm512_setzero_ps();
    const __m512i x = _mm512_set1_epi16(static_cast<short>(0x3a83 * g_byte)),
                  y = x;
    for (uint64_t it = 0; it < iters; ++it)
        PROBE_LAT8("vdpbf16ps");
    return _mm512_reduce_add_ps(a0);
}

// u8 1 times s8 1, four a lane: int32 acc wraps only after 5e8 instructions.
__attribute__((target("avx512f,avx512vnni"))) double tput_int8(uint64_t iters) {
    PROBE_DECL16(__m512i, _mm512_setzero_si512());
    const __m512i x = _mm512_set1_epi8(static_cast<char>(g_byte)), y = x;
    for (uint64_t it = 0; it < iters; ++it) {
        PROBE_TPUT16("vpdpbusd");
    }
    return static_cast<double>(
            _mm512_reduce_add_epi32(PROBE_SUM16(_mm512_add_epi32)));
}

__attribute__((target("avx512f,avx512vnni"))) double lat_int8(uint64_t iters) {
    __m512i a0 = _mm512_setzero_si512();
    const __m512i x = _mm512_set1_epi8(static_cast<char>(g_byte)), y = x;
    for (uint64_t it = 0; it < iters; ++it)
        PROBE_LAT8("vpdpbusd");
    return static_cast<double>(_mm512_reduce_add_epi32(a0));
}

#if PROBE_HAVE_FP16
// acc += 0 * 1: acc stays 1.
__attribute__((target("avx512f,avx512fp16"))) double tput_fp16(uint64_t iters) {
    PROBE_DECL16(__m512h, _mm512_set1_ph(static_cast<_Float16>(1.0f)));
    const __m512h x = _mm512_set1_ph(static_cast<_Float16>(g_one));
    const __m512h y = _mm512_set1_ph(static_cast<_Float16>(0.0f * g_tiny));
    for (uint64_t it = 0; it < iters; ++it) {
        PROBE_TPUT16("vfmadd231ph");
    }
    return static_cast<double>(
            _mm512_reduce_add_ph(PROBE_SUM16(_mm512_add_ph)));
}

__attribute__((target("avx512f,avx512fp16"))) double lat_fp16(uint64_t iters) {
    __m512h a0 = _mm512_set1_ph(static_cast<_Float16>(1.0f));
    const __m512h x = _mm512_set1_ph(static_cast<_Float16>(g_one));
    const __m512h y = _mm512_set1_ph(static_cast<_Float16>(0.0f * g_tiny));
    for (uint64_t it = 0; it < iters; ++it)
        PROBE_LAT8("vfmadd231ph");
    return static_cast<double>(_mm512_reduce_add_ph(a0));
}
#endif

// The INT4 tile's inner loop at its widest shape, 6 rows of 2 vectors (as
// native_int4_ukernel.cpp): per K octet each vector's 64 bytes of nibbles
// unpacked into two planes (and, shift, and), and per row two source dwords
// broadcast and two VPDPBUSD a vector, the high plane's on the low one's
// accumulator.
alignas(64) uint8_t g_nibbles[128];
alignas(64) uint8_t g_src8[64];
constexpr int kInt4PerIter = 24; // VPDPBUSD of one iteration

#define PROBE_INT4_ROW(m, a, b) \
    "vpbroadcastd " #m \
    "*8(%[src]), %[q0]\n\t" \
    "vpbroadcastd " #m \
    "*8+4(%[src]), %[q1]\n\t" \
    "vpdpbusd %[q0], %[lo0], %[" #a \
    "]\n\t" \
    "vpdpbusd %[q1], %[hi0], %[" #a \
    "]\n\t" \
    "vpdpbusd %[q0], %[lo1], %[" #b \
    "]\n\t" \
    "vpdpbusd %[q1], %[hi1], %[" #b "]\n\t"
#define PROBE_INT4_ROWS(r0, r1, r2, a0, b0, a1, b1, a2, b2) \
    asm volatile(PROBE_INT4_ROW(r0, a0, b0) PROBE_INT4_ROW(r1, a1, b1) \
                         PROBE_INT4_ROW(r2, a2, b2) \
                 : [a0] "+v"(a0), [b0] "+v"(b0), [a1] "+v"(a1), [b1] "+v"(b1), \
                 [a2] "+v"(a2), [b2] "+v"(b2), [q0] "=&v"(q0), [q1] "=&v"(q1) \
                 : [lo0] "v"(lo0), [hi0] "v"(hi0), [lo1] "v"(lo1), \
                 [hi1] "v"(hi1), [src] "r"(g_src8) \
                 : "memory")

// Nibbles 1 over source bytes 1 (init_tile_operands()): an accumulator grows by
// 16 an iteration.
__attribute__((target("avx512f,avx512bw,avx512vnni"))) double tput_int4(
        uint64_t iters) {
    const __m512i mask = _mm512_set1_epi8(0x0f);
    __m512i c00 = _mm512_setzero_si512(), c01 = c00, c10 = c00, c11 = c00,
            c20 = c00, c21 = c00, c30 = c00, c31 = c00, c40 = c00, c41 = c00,
            c50 = c00, c51 = c00;
    __m512i lo0, hi0, lo1, hi1, q0, q1;
    for (uint64_t it = 0; it < iters; ++it) {
        asm volatile(
                "vmovdqu64 (%[b]), %[lo0]\n\t"
                "vpsrlw $4, %[lo0], %[hi0]\n\t"
                "vpandd %[mask], %[lo0], %[lo0]\n\t"
                "vpandd %[mask], %[hi0], %[hi0]\n\t"
                "vmovdqu64 64(%[b]), %[lo1]\n\t"
                "vpsrlw $4, %[lo1], %[hi1]\n\t"
                "vpandd %[mask], %[lo1], %[lo1]\n\t"
                "vpandd %[mask], %[hi1], %[hi1]\n\t"
                : [lo0] "=&v"(lo0), [hi0] "=&v"(hi0), [lo1] "=&v"(lo1),
                [hi1] "=&v"(hi1)
                : [b] "r"(g_nibbles), [mask] "v"(mask)
                : "memory");
        PROBE_INT4_ROWS(0, 1, 2, c00, c01, c10, c11, c20, c21);
        PROBE_INT4_ROWS(3, 4, 5, c30, c31, c40, c41, c50, c51);
    }
    const __m512i s = _mm512_add_epi32(
            _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(c00, c01),
                                     _mm512_add_epi32(c10, c11)),
                    _mm512_add_epi32(_mm512_add_epi32(c20, c21),
                            _mm512_add_epi32(c30, c31))),
            _mm512_add_epi32(
                    _mm512_add_epi32(c40, c41), _mm512_add_epi32(c50, c51)));
    return static_cast<double>(_mm512_reduce_add_epi32(s));
}

// The F16 kernels' inner loop at their widest shape, 12 rows of 2 vectors (as
// native_f16_ukernel.cpp, where the CPU has no AVX512-FP16): per K step each
// vector's 16 halves widened to f32, and per row a broadcast and an FMA a
// vector.
alignas(64) uint16_t g_halves[32];
alignas(64) float g_rows[16];
constexpr int kF16PerIter = 24; // VFMADD231PS of one iteration

// off: the row's byte offset in g_rows.
#define PROBE_F16_ROW(off, x, y) \
    "vbroadcastss " #off \
    "(%[src]), %[t]\n\t" \
    "vfmadd231ps %[t], %[w0], %[" #x \
    "]\n\t" \
    "vfmadd231ps %[t], %[w1], %[" #y "]\n\t"
#define PROBE_F16_ROWS(o0, o1, o2, o3, o4, o5, x0, y0, x1, y1, x2, y2, x3, y3, \
        x4, y4, x5, y5) \
    asm volatile(PROBE_F16_ROW(o0, x0, y0) PROBE_F16_ROW( \
            o1, x1, y1) PROBE_F16_ROW(o2, x2, y2) PROBE_F16_ROW(o3, x3, y3) \
                         PROBE_F16_ROW(o4, x4, y4) PROBE_F16_ROW(o5, x5, y5) \
                 : [x0] "+v"(x0), [y0] "+v"(y0), [x1] "+v"(x1), [y1] "+v"(y1), \
                 [x2] "+v"(x2), [y2] "+v"(y2), [x3] "+v"(x3), [y3] "+v"(y3), \
                 [x4] "+v"(x4), [y4] "+v"(y4), [x5] "+v"(x5), [y5] "+v"(y5), \
                 [t] "=&v"(t) \
                 : [w0] "v"(w0), [w1] "v"(w1), [src] "r"(g_rows) \
                 : "memory")

// Halves 2^-14 times rows of 1 (init_tile_operands()): an accumulator grows by
// 2^-14 an iteration.
__attribute__((target("avx512f"))) double tput_f16_widen(uint64_t iters) {
    __m512 c00 = _mm512_setzero_ps(), c01 = c00, c10 = c00, c11 = c00,
           c20 = c00, c21 = c00, c30 = c00, c31 = c00, c40 = c00, c41 = c00,
           c50 = c00, c51 = c00, c60 = c00, c61 = c00, c70 = c00, c71 = c00,
           c80 = c00, c81 = c00, c90 = c00, c91 = c00, ca0 = c00, ca1 = c00,
           cb0 = c00, cb1 = c00;
    __m512 w0, w1, t;
    for (uint64_t it = 0; it < iters; ++it) {
        asm volatile(
                "vcvtph2ps (%[h]), %[w0]\n\t"
                "vcvtph2ps 32(%[h]), %[w1]\n\t"
                : [w0] "=&v"(w0), [w1] "=&v"(w1)
                : [h] "r"(g_halves)
                : "memory");
        PROBE_F16_ROWS(0, 4, 8, 12, 16, 20, c00, c01, c10, c11, c20, c21, c30,
                c31, c40, c41, c50, c51);
        PROBE_F16_ROWS(24, 28, 32, 36, 40, 44, c60, c61, c70, c71, c80, c81,
                c90, c91, ca0, ca1, cb0, cb1);
    }
    const __m512 s = _mm512_add_ps(
            _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(_mm512_add_ps(c00, c01),
                                                _mm512_add_ps(c10, c11)),
                                  _mm512_add_ps(_mm512_add_ps(c20, c21),
                                          _mm512_add_ps(c30, c31))),
                    _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(c40, c41),
                                          _mm512_add_ps(c50, c51)),
                            _mm512_add_ps(_mm512_add_ps(c60, c61),
                                    _mm512_add_ps(c70, c71)))),
            _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(c80, c81),
                                  _mm512_add_ps(c90, c91)),
                    _mm512_add_ps(
                            _mm512_add_ps(ca0, ca1), _mm512_add_ps(cb0, cb1))));
    return static_cast<double>(_mm512_reduce_add_ps(s));
}

/// The tile kernels' operands, read by every thread of a measurement: set once,
/// before any runs.
void init_tile_operands() {
    std::memset(g_nibbles, 0x11 * g_byte, sizeof(g_nibbles));
    std::memset(g_src8, g_byte, sizeof(g_src8));
    for (uint16_t &h : g_halves)
        h = static_cast<uint16_t>(0x0400 * g_byte);
    for (float &r : g_rows)
        r = g_one;
}

struct path_t {
    const char *name;
    kernel_fn tput;
    kernel_fn lat;
    double flops_per_instr; // a 512-bit instruction's FLOPs (2 per multiply-add)
    int per_iter = kAcc; // instructions of one iteration of tput
};

/// Iterations of k that take about `seconds` on one thread.
uint64_t calibrate(kernel_fn k, double seconds) {
    uint64_t n = 4096;
    volatile double sink = 0;
    for (;;) {
        const double t0 = omp_get_wtime();
        sink = sink + k(n);
        const double dt = omp_get_wtime() - t0;
        if (dt > 0.02 || n > (1ull << 40))
            return std::max<uint64_t>(1,
                    static_cast<uint64_t>(
                            static_cast<double>(n) * seconds / dt));
        n *= 4;
    }
}

/// Instructions per second of each of `threads` threads running k at once
/// (iters iterations of `per_iter` instructions each).
std::vector<double> instr_rates(
        kernel_fn k, uint64_t iters, double per_iter, int threads) {
    std::vector<double> rates(threads);
    volatile double sink = 0;
#pragma omp parallel num_threads(threads)
    {
        const int t = omp_get_thread_num();
        double r = k(iters / 8);
#pragma omp barrier
        const double t0 = omp_get_wtime();
        r += k(iters);
        const double t1 = omp_get_wtime();
        rates[t] = static_cast<double>(iters) * per_iter / (t1 - t0);
        if (r == 0.5) sink = r;
    }
    (void)sink;
    return rates;
}

// ---------------------------------------------------------------- bandwidth

// noinline, and every call behind a compiler memory barrier: a pass is a pure
// function of memory no one writes, which the compiler would otherwise run once
// for all the passes of a timed loop.
__attribute__((target("avx512f"), noinline)) uint64_t read_pass(
        const uint8_t *p, size_t bytes, size_t start) {
    __m512i s0 = _mm512_setzero_si512(), s1 = s0, s2 = s0, s3 = s0, s4 = s0,
            s5 = s0, s6 = s0, s7 = s0;
    const size_t end = bytes / 512 * 512;
    for (int part = 0; part < 2; ++part) {
        const size_t lo = part == 0 ? start : 0, hi = part == 0 ? end : start;
        for (size_t i = lo; i + 512 <= hi; i += 512) {
            s0 = _mm512_xor_si512(s0, _mm512_load_si512(p + i));
            s1 = _mm512_xor_si512(s1, _mm512_load_si512(p + i + 64));
            s2 = _mm512_xor_si512(s2, _mm512_load_si512(p + i + 128));
            s3 = _mm512_xor_si512(s3, _mm512_load_si512(p + i + 192));
            s4 = _mm512_xor_si512(s4, _mm512_load_si512(p + i + 256));
            s5 = _mm512_xor_si512(s5, _mm512_load_si512(p + i + 320));
            s6 = _mm512_xor_si512(s6, _mm512_load_si512(p + i + 384));
            s7 = _mm512_xor_si512(s7, _mm512_load_si512(p + i + 448));
        }
    }
    s0 = _mm512_xor_si512(_mm512_xor_si512(_mm512_xor_si512(s0, s1),
                                  _mm512_xor_si512(s2, s3)),
            _mm512_xor_si512(
                    _mm512_xor_si512(s4, s5), _mm512_xor_si512(s6, s7)));
    return static_cast<uint64_t>(_mm512_reduce_add_epi64(s0));
}

/// The stores of a pass over p, as an output tile's: whole lines, each read
/// first by the store unless the line is held.  The barrier keeps the stores
/// from becoming a memset.
__attribute__((target("avx512f"), noinline)) void write_pass(
        uint8_t *p, size_t bytes, size_t start, int value) {
    const __m512i v = _mm512_set1_epi32(value);
    const size_t end = bytes / 512 * 512;
    for (int part = 0; part < 2; ++part) {
        const size_t lo = part == 0 ? start : 0, hi = part == 0 ? end : start;
        for (size_t i = lo; i + 512 <= hi; i += 512) {
            for (size_t j = 0; j < 512; j += 64)
                _mm512_store_si512(p + i + j, v);
            asm volatile("" : : "r"(p + i) : "memory");
        }
    }
}

/// Whether a fault in an MADV_HUGEPAGE buffer gets a huge page here.  Where it does
/// not (THP off, or memory too fragmented), the advice only makes every fault try to
/// compact memory, which can take seconds a buffer.  One huge page says little
/// about the thousands the DRAM buffers take: on fragmented memory a team's
/// faults can still spend minutes compacting.
bool hugepages_work() {
    constexpr size_t kHuge = size_t(2) << 20;
    void *mem = nullptr;
    if (posix_memalign(&mem, kHuge, kHuge) != 0) return false;
    madvise(mem, kHuge, MADV_HUGEPAGE);
    static_cast<volatile char *>(mem)[0] = 1;
    bool huge = false;
    if (FILE *f = std::fopen("/proc/self/smaps", "r")) {
        const auto at = reinterpret_cast<unsigned long>(mem);
        char line[512];
        bool in = false;
        while (std::fgets(line, sizeof(line), f)) {
            unsigned long lo = 0, hi = 0, kb = 0;
            if (std::sscanf(line, "%lx-%lx ", &lo, &hi) == 2) {
                in = at >= lo && at < hi;
            } else if (in
                    && std::sscanf(line, "AnonHugePages: %lu kB", &kb) == 1) {
                huge = kb >= 2048;
                break;
            }
        }
        std::fclose(f);
    }
    std::free(mem);
    return huge;
}

/// Where thread t starts each pass over its buffer (a multiple of 512 bytes): the
/// buffers are aligned alike, and threads that start at one offset stream the
/// same DRAM channels and banks in step.
size_t stagger(int t, size_t bytes) {
    const uint64_t units = std::max<uint64_t>(1, bytes / 512);
    return static_cast<size_t>(
                   (static_cast<uint64_t>(t) * 0x9E3779B97F4A7C15ull >> 17)
                   % units)
            * 512;
}

struct bw_t {
    std::string level;
    int threads = 0;
    size_t bytes_per_thread = 0;
    stat_t gbps;
};

/// The total read (or write) bandwidth (GB/s) of `threads` threads, each
/// streaming its own `bytes` (first touched by itself), over `repeats` timed
/// rounds: per round the bytes of every thread over the slowest thread's time.
stat_t stream_bandwidth(int threads, size_t bytes, double seconds, int repeats,
        bool huge, bool write) {
    constexpr size_t kAlign = size_t(2) << 20;
    const size_t alloc = (bytes + kAlign - 1) / kAlign * kAlign;
    std::vector<std::vector<double>> times(
            threads, std::vector<double>(repeats));
    uint64_t passes = 1;
    bool failed = false;
    volatile uint64_t sink = 0;
#pragma omp parallel num_threads(threads)
    {
        const int t = omp_get_thread_num();
        void *mem = nullptr;
        if (posix_memalign(&mem, kAlign, alloc) != 0) mem = nullptr;
        if (mem == nullptr) {
#pragma omp atomic write
            failed = true;
        } else {
            if (huge) madvise(mem, alloc, MADV_HUGEPAGE);
            std::memset(mem, t + 1, bytes);
        }
#pragma omp barrier
        if (!failed) {
            auto *buf = static_cast<uint8_t *>(mem);
            const size_t start = stagger(t, bytes);
            uint64_t r = 0;
            auto pass = [&](int i) {
                if (write)
                    write_pass(buf, bytes, start, t + i);
                else
                    r += read_pass(buf, bytes, start);
            };
            pass(0);
            pass(1);
#pragma omp barrier
            const double t0 = omp_get_wtime();
            pass(2);
            const double one = omp_get_wtime() - t0;
#pragma omp barrier
#pragma omp single
            passes = std::max<uint64_t>(2,
                    static_cast<uint64_t>(
                            std::ceil(seconds / std::max(one, 1e-9))));
            for (int rep = 0; rep < repeats; ++rep) {
#pragma omp barrier
                const double s0 = omp_get_wtime();
                for (uint64_t i = 0; i < passes; ++i) {
                    asm volatile("" : : "r"(buf) : "memory");
                    pass(static_cast<int>(i));
                }
                times[t][rep] = omp_get_wtime() - s0;
            }
            if (r == 1) sink = r;
        }
        std::free(mem);
    }
    (void)sink;
    if (failed) return stat_t {};
    std::vector<double> gbps;
    for (int rep = 0; rep < repeats; ++rep) {
        double slowest = 0.0;
        for (int t = 0; t < threads; ++t)
            slowest = std::max(slowest, times[t][rep]);
        gbps.push_back(static_cast<double>(bytes) * static_cast<double>(passes)
                * threads / slowest / 1e9);
    }
    return stat_of(gbps);
}

// ---------------------------------------------------------------- output

std::string json_str(const std::string &s) {
    std::string o = "\"";
    for (char ch : s) {
        if (ch == '"' || ch == '\\') {
            o += '\\';
            o += ch;
        } else if (static_cast<unsigned char>(ch) >= 0x20) {
            o += ch;
        }
    }
    return o + "\"";
}

void add_place(int place, std::vector<int> &ids) {
    std::vector<int> p(static_cast<size_t>(omp_get_place_num_procs(place)));
    omp_get_place_proc_ids(place, p.data());
    ids.insert(ids.end(), p.begin(), p.end());
}

/// "a-b,c" for a set of CPU ids.
std::string cpu_list(std::vector<int> ids) {
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::string out;
    for (size_t i = 0; i < ids.size();) {
        size_t j = i;
        while (j + 1 < ids.size() && ids[j + 1] == ids[j] + 1)
            ++j;
        out += (out.empty() ? "" : ",") + std::to_string(ids[i])
                + (j > i ? "-" + std::to_string(ids[j]) : "");
        i = j + 1;
    }
    return out;
}

/// The CPUs the process started with.  The OpenMP places cover them; when
/// there are places the runtime has already pinned this thread to the first,
/// so its own affinity says nothing.
std::string cpuset() {
    std::vector<int> ids;
    for (int place = 0; place < omp_get_num_places(); ++place)
        add_place(place, ids);
    if (!ids.empty()) return cpu_list(ids);
    std::string s;
    FILE *f = std::fopen("/proc/self/status", "r");
    if (f == nullptr) return s;
    char line[4096];
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "Cpus_allowed_list:", 18) == 0) {
            s = line + 18;
            s.erase(0, s.find_first_not_of(" \t"));
            s.erase(s.find_last_not_of(" \t\r\n") + 1);
            break;
        }
    }
    std::fclose(f);
    return s;
}

/// The CPUs of the places a team of `threads` runs on, or "" when its threads
/// are not bound to places.
std::string team_cpus(int threads) {
    std::vector<int> ids;
#pragma omp parallel num_threads(threads)
    {
        const int place = omp_get_place_num();
        if (place >= 0) {
#pragma omp critical
            add_place(place, ids);
        }
    }
    return cpu_list(ids);
}

/// A CPU's (package, core) from sysfs, or -1.
long core_of(int cpu) {
    auto read_id = [](const std::string &path) {
        long v = -1;
        if (FILE *f = std::fopen(path.c_str(), "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    const std::string dir = "/sys/devices/system/cpu/cpu" + std::to_string(cpu)
            + "/topology/";
    const long core = read_id(dir + "core_id");
    const long pkg = read_id(dir + "physical_package_id");
    return core < 0 || pkg < 0 ? -1 : (pkg << 20) + core;
}

/// The physical cores a team of `threads` runs on, two threads on the siblings
/// of one core counting once; 0 when it is not known (unbound threads, a place
/// of several cores, no topology).
int team_cores(int threads) {
    std::vector<std::vector<int>> places(static_cast<size_t>(threads));
#pragma omp parallel num_threads(threads)
    {
        const int place = omp_get_place_num();
        if (place >= 0)
            add_place(place, places[static_cast<size_t>(omp_get_thread_num())]);
    }
    std::vector<long> cores;
    for (const std::vector<int> &cpus : places) {
        long core = -1;
        for (int cpu : cpus) {
            const long c = core_of(cpu);
            if (c < 0 || (core >= 0 && c != core)) return 0;
            core = c;
        }
        if (core < 0) return 0;
        cores.push_back(core);
    }
    std::sort(cores.begin(), cores.end());
    return static_cast<int>(
            std::unique(cores.begin(), cores.end()) - cores.begin());
}

std::string env_or(const char *name, const char *fallback) {
    const char *v = std::getenv(name);
    return v ? v : fallback;
}

std::string utc_now() {
    char buf[32];
    const std::time_t now = std::time(nullptr);
    std::tm tm {};
    gmtime_r(&now, &tm);
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

[[noreturn]] void usage(const char *why) {
    std::fprintf(stderr,
            "%s\nusage: probe --out ceilings.json [--threads N] "
            "[--domain-threads N] [--repeats R] [--seconds S] [--quick] "
            "[--hugepages]\n",
            why);
    std::exit(2);
}

options_t parse(int argc, char **argv) {
    options_t o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char *name) -> std::string {
            if (i + 1 >= argc)
                usage((std::string(name) + " needs a value").c_str());
            return argv[++i];
        };
        if (a == "--out") {
            o.out = value("--out");
        } else if (a == "--threads") {
            o.threads = std::atoi(value("--threads").c_str());
        } else if (a == "--domain-threads") {
            o.domain_threads = std::atoi(value("--domain-threads").c_str());
        } else if (a == "--repeats") {
            o.repeats = std::atoi(value("--repeats").c_str());
        } else if (a == "--seconds") {
            o.seconds = std::atof(value("--seconds").c_str());
        } else if (a == "--quick") {
            o.quick = true;
        } else if (a == "--hugepages") {
            o.hugepages = true;
        } else {
            usage(("unknown argument " + a).c_str());
        }
    }
    if (o.out.empty()) usage("--out is required");
    if (o.quick) {
        o.seconds = 0.01;
        o.repeats = 2;
    }
    if (o.repeats < 1 || !(o.seconds > 0)) usage("bad --repeats or --seconds");
    return o;
}

} // namespace

int main(int argc, char **argv) {
    options_t o = parse(argc, argv);
    const std::string cpus = cpuset();
    const bool huge = o.hugepages && hugepages_work();
    if (o.hugepages && !huge)
        std::fprintf(stderr,
                "probe: a fault gets no huge page here; the buffers take the "
                "default pages\n");
    const double start = omp_get_wtime();
    const nat::native_hw_t &hw = nat::native_hw();
    const nat::UarchParams &u = nat::native_plan_uarch();
    if (!u.avx512f) {
        std::fprintf(stderr, "probe: the CPU has no AVX-512F\n");
        return 1;
    }
    const int threads = o.threads > 0 ? o.threads : omp_get_max_threads();
    int domain = o.domain_threads > 0
            ? o.domain_threads
            : (u.l3_domain_cores > 0 ? u.l3_domain_cores : u.family_l3_cores);
    domain = std::max(1, std::min(domain, threads));
    const double secs = o.seconds;

    // Clock.
    std::vector<double> c1, ct;
    for (int r = 0; r < o.repeats; ++r) {
        c1.push_back(clock_ghz(1, secs));
        ct.push_back(clock_ghz(threads, secs));
    }
    const stat_t ghz1 = stat_of(c1), ghzt = stat_of(ct);

    // Compute.
    init_tile_operands();
    std::vector<path_t> paths = {{"fp32", tput_fp32, lat_fp32, 32.0}};
    if (u.avx512bf16) paths.push_back({"bf16", tput_bf16, lat_bf16, 64.0});
    if (u.avx512vnni) paths.push_back({"int8", tput_int8, lat_int8, 128.0});
    if (u.avx512vnni)
        paths.push_back({"int4", tput_int4, lat_int8, 128.0, kInt4PerIter});
#if PROBE_HAVE_FP16
    if (u.avx512fp16) paths.push_back({"fp16", tput_fp16, lat_fp16, 64.0});
#endif
    // Where the CPU has AVX512-FP16, f16 calls run on the F16_PH kernels instead.
    if (!u.avx512fp16)
        paths.push_back(
                {"f16_widen", tput_f16_widen, lat_fp32, 32.0, kF16PerIter});
    std::string compute_json;
    for (const path_t &p : paths) {
        const uint64_t it = calibrate(p.tput, secs);
        const uint64_t lit = calibrate(p.lat, secs);
        std::vector<double> core, team, lat;
        for (int r = 0; r < o.repeats; ++r) {
            core.push_back(instr_rates(p.tput, it, p.per_iter, 1)[0]);
            double sum = 0.0;
            for (double v : instr_rates(p.tput, it, p.per_iter, threads))
                sum += v;
            team.push_back(sum);
            lat.push_back(ghz1.median * 1e9
                    / instr_rates(p.lat, lit, kLatChain, 1)[0]);
        }
        const stat_t sc = stat_of(core), st = stat_of(team), sl = stat_of(lat);
        char buf[640];
        std::snprintf(buf, sizeof(buf),
                "%s  \"%s\": {\"flops_per_instr\": %.0f, \"instr_per_cycle\": "
                "%.4f, "
                "\"instr_per_cycle_team\": %.4f, \"latency_cycles\": %.3f, "
                "\"gflops_core\": %.3f, \"gflops_team\": %.3f, \"spread\": "
                "%.4f}",
                compute_json.empty() ? "" : ",\n", p.name, p.flops_per_instr,
                sc.median / (ghz1.median * 1e9),
                st.median / threads / (ghzt.median * 1e9), sl.median,
                sc.median * p.flops_per_instr / 1e9,
                st.median * p.flops_per_instr / 1e9, st.rel_mad);
        compute_json += buf;
    }

    // Bandwidth.
    std::vector<int> counts = {1, domain, threads};
    std::sort(counts.begin(), counts.end());
    counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
    const size_t l3 = static_cast<size_t>(u.l3_domain_bytes);
    const size_t l3_share = l3 / static_cast<size_t>(domain);
    std::vector<bw_t> bws, wbws;
    for (const char *level : {"l1", "l2", "l3", "dram"}) {
        for (int t : counts) {
            size_t bytes = 0;
            const std::string lv = level;
            if (lv == "l1") {
                bytes = static_cast<size_t>(u.l1d_bytes) / 2;
            } else if (lv == "l2") {
                bytes = static_cast<size_t>(u.l2_bytes) / 2;
            } else if (lv == "l3") {
                if (l3 == 0) continue;
                bytes = std::min(
                        std::max(static_cast<size_t>(u.l2_bytes) * 3 / 2,
                                l3_share / 2),
                        l3_share * 4 / 5);
            } else {
                const size_t domains
                        = static_cast<size_t>((t + domain - 1) / domain);
                const size_t floor_mb = o.quick ? 16 : 64;
                bytes = std::min(
                        std::max(floor_mb << 20,
                                4 * domains * std::max(l3, size_t(1) << 20)
                                        / static_cast<size_t>(t)),
                        size_t(1) << 30);
            }
            bytes = std::max<size_t>(bytes / 512 * 512, 512);
            bw_t b;
            b.level = lv;
            b.threads = t;
            b.bytes_per_thread = bytes;
            b.gbps = stream_bandwidth(t, bytes, secs, o.repeats, huge, false);
            if (b.gbps.median > 0) bws.push_back(b);
            bw_t w = b;
            w.gbps = stream_bandwidth(t, bytes, secs, o.repeats, huge, true);
            if (w.gbps.median > 0) wbws.push_back(w);
        }
    }

    // Output.
    std::string isa;
    for (auto [have, name] : {
                 std::pair<bool, const char *> {u.avx512f, "avx512f"},
                 {u.avx512bf16, "avx512bf16"}, {u.avx512vnni, "avx512vnni"}}) {
        if (have)
            isa += std::string(isa.empty() ? "" : ", ") + "\"" + name + "\"";
    }
    FILE *f = std::fopen(o.out.c_str(), "w");
    if (f == nullptr) {
        std::perror(o.out.c_str());
        return 1;
    }
    std::fprintf(
            f, "{\n \"format\": \"op-perf-ceilings\",\n \"version\": 1,\n");
    std::fprintf(f, " \"probe_version\": \"%s\",\n \"created\": \"%s\",\n",
            kProbeVersion, utc_now().c_str());
    std::fprintf(f, " \"duration_s\": %.1f,\n", omp_get_wtime() - start);
    std::fprintf(f,
            " \"cpu\": {\"vendor\": %s, \"family\": %d, \"model\": %d, "
            "\"stepping\": %d, "
            "\"isa\": [%s], \"avx512fp16\": %s, \"l1d_kib\": %d, \"l2_kib\": "
            "%d, "
            "\"l3_kib\": %d, \"logical_cpus\": %d, \"smt\": %d, "
            "\"physical_cores\": %d, "
            "\"l3_domain_threads\": %d},\n",
            json_str(hw.vendor).c_str(), hw.family, hw.model, hw.stepping,
            isa.c_str(), u.avx512fp16 ? "true" : "false",
            u.l1d_detected ? u.l1d_bytes / 1024 : 0,
            u.l2_detected ? u.l2_bytes / 1024 : 0,
            u.l3_detected ? u.l3_domain_bytes / 1024 : 0, hw.logical_cpus,
            hw.smt, hw.physical_cores, hw.l3_domain_threads);
    std::fprintf(f,
            " \"uarch\": {\"l1d_bytes\": %d, \"l2_bytes\": %d, "
            "\"l3_domain_bytes\": %d, "
            "\"l3_domain_cores\": %d, \"family_l3_cores\": %d, \"fma_ports\": "
            "%d, "
            "\"vec_width_bits\": %d},\n",
            u.l1d_bytes, u.l2_bytes, u.l3_domain_bytes, u.l3_domain_cores,
            u.family_l3_cores, u.fma_ports, u.vec_width_bits);
    std::fprintf(f,
            " \"run\": {\"threads\": %d, \"domain_threads\": %d, \"cpuset\": "
            "%s, "
            "\"team_cpus\": %s, \"omp_proc_bind\": %s, \"omp_places\": %s, "
            "\"team_cores\": %d, \"repeats\": %d, \"seconds\": %.3f, "
            "\"quick\": %s, \"hugepages\": %s},\n",
            threads, domain, json_str(cpus).c_str(),
            json_str(team_cpus(threads)).c_str(),
            json_str(env_or("OMP_PROC_BIND", "unset")).c_str(),
            json_str(env_or("OMP_PLACES", "unset")).c_str(),
            team_cores(threads), o.repeats, secs, o.quick ? "true" : "false",
            huge ? "true" : "false");
    std::fprintf(f,
            " \"clock\": {\"ghz_1core\": %.4f, \"ghz_team\": %.4f, "
            "\"spread_1core\": %.4f, "
            "\"spread_team\": %.4f},\n",
            ghz1.median, ghzt.median, ghz1.rel_mad, ghzt.rel_mad);
    std::fprintf(f, " \"compute\": {\n%s\n },\n", compute_json.c_str());
    for (const auto &[name, list] :
            {std::pair<const char *, const std::vector<bw_t> *> {
                     "bandwidth", &bws},
                    {"write_bandwidth", &wbws}}) {
        std::fprintf(f, " \"%s\": [\n", name);
        for (size_t i = 0; i < list->size(); ++i) {
            const bw_t &b = (*list)[i];
            std::fprintf(f,
                    "  {\"level\": \"%s\", \"threads\": %d, "
                    "\"bytes_per_thread\": %zu, \"gbps\": %.3f, \"spread\": "
                    "%.4f}%s\n",
                    b.level.c_str(), b.threads, b.bytes_per_thread,
                    b.gbps.median, b.gbps.rel_mad,
                    i + 1 < list->size() ? "," : "");
        }
        std::fprintf(f, " ]%s\n", list == &wbws ? "" : ",");
    }
    std::fprintf(f, "}\n");
    std::fclose(f);
    std::printf("probe: wrote %s (%d threads, L3 domain %d threads, %.1f s)\n",
            o.out.c_str(), threads, domain, omp_get_wtime() - start);
    return 0;
}
