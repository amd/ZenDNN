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

/// Group matmul's view of the machine: the caches, the L3 domain and each
/// dot-product path's rate, which ALGO 3's custom kernel and N-tile flat
/// parallel decide from.  The caches and the topology come from
/// `native::native_hw()` (`machine()`).  The rates are measured once, on the
/// first call that needs them, on the calling thread (`dot_rates()`,
/// grp_matmul_dot_measure.cpp), so a call that reads only the caches never
/// waits for the measurement.

#ifndef ZENDNNL_GRP_MATMUL_MACHINE_HPP
#define ZENDNNL_GRP_MATMUL_MACHINE_HPP

#include <array>
#include <cstdint>
#include <string>

#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace grp_machine {

/// A dot-product path of the kernels: the FP32 FMA, BF16's VDPBF16PS, INT8's
/// VPDPBUSD (which the s4 kernels run too) and the AVX-512 FP16 FMA.
enum class dot_path : uint8_t { fp32, bf16, int8, fp16 };
inline constexpr int kDotPaths = 4;

/// Indexed by dot_path.
inline constexpr const char *kDotPathNames[kDotPaths]
        = {"fp32", "bf16", "int8", "fp16"};

/// One core's rate on a path: its instructions a cycle with enough
/// independent accumulators (ipc, in quarters), and the cycles from one into
/// an accumulator to the next one that reads it (latency).  Zero for a path
/// the CPU lacks or that was not measured.
struct dot_rate {
    double ipc = 0.0;
    int latency = 0;
};

/// Measures `u`'s CPU on each of its paths, on the calling thread: each path's
/// loops timed against one-cycle adds, each timing the fastest of three runs
/// in rounds that take every path in turn.  A few milliseconds in all.
/// Whether it measured a path: false, with `out` all zero, in a build that
/// cannot measure (MSVC, which has no inline assembly for the loops) and on a
/// CPU without AVX-512.
bool measure_dot_rates(
        const native::UarchParams &u, std::array<dot_rate, kDotPaths> &out);

/// The caches and the topology.
struct machine_t {
    int64_t l1d_bytes = 0;
    int64_t l2_bytes = 0;
    int64_t l3_domain_bytes = 0;
    /// Physical cores sharing one L3, from the OS; 0 when it doesn't say.
    int l3_domain_cores = 0;
    /// Logical CPUs per core, at least 1.
    int smt = 1;
    /// The L2 plans size for: the L2 reported, or 256 KiB when that is under
    /// 64 KiB.
    int64_t l2_plan_bytes = 0;
    /// The share of it a kernel's weights may fill: 80%.
    int64_t l2_budget_bytes = 0;
};

/// One core's rates, and what the kernels decide from them.
struct dot_rates_t {
    std::array<dot_rate, kDotPaths> dot {};
    /// Whether `dot` was measured.
    bool measured = false;
    /// Milliseconds the measurement took.
    double measure_ms = 0.0;
    /// Whether DA8W8 kernels that read sources already XOR-ed with 0x80 pay
    /// for flipping each row once: they save one XOR per broadcast, which
    /// costs throughput only where INT8 dot products issue well beyond the
    /// other paths, more than 1.5 times the faster of BF16's and FP32's
    /// measured rates.  Without measured rates, on the CPUs they were
    /// measured to pay on: AMD family 1Ah models 50h-5Fh (Zen 6).
    bool u8_src_pays = false;

    const dot_rate &rate(const dot_path p) const {
        return dot[static_cast<int>(p)];
    }
    /// Independent accumulator chains a kernel on `p` needs to stay
    /// compute-bound: ipc x latency, rounded up; 1 for an unmeasured path.
    int chains(dot_path p) const;
};

/// The caches and the topology `hw` describes.
machine_t make_machine(const native::native_hw_t &hw);

/// The rates `dot` of `hw`'s CPU (measured when `measured`).
dot_rates_t make_dot_rates(const native::native_hw_t &hw,
        const std::array<dot_rate, kDotPaths> &dot, bool measured);

/// make_machine(native::native_hw()), made once.
const machine_t &machine();

/// make_dot_rates(native::native_hw(), its measured rates), made on first
/// use.
const dot_rates_t &dot_rates();

/// One line for the API log.
std::string describe(const machine_t &m, const dot_rates_t &d);

} // namespace grp_machine
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GRP_MATMUL_MACHINE_HPP
