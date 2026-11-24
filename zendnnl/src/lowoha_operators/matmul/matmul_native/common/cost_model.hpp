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

#ifndef MATMUL_NATIVE_COST_MODEL_HPP
#define MATMUL_NATIVE_COST_MODEL_HPP

#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// The compute paths of the native kernels, one per kernel family: the dot
/// product of its inner loop.  int4 is the s4 tile's loop (the INT8 dot
/// product over nibbles it unpacks), f16_widen the F16 kernels' FP32 FMA
/// over halves they widen, fp16 the FP16 FMA of the F16_PH kernels.
enum class native_dot_path_t : uint8_t {
    fp32,
    bf16,
    int8,
    int4,
    fp16,
    f16_widen
};
inline constexpr int kNativeDotPaths = 6;

/// Indexed by native_dot_path_t.
inline constexpr const char *kNativeDotPathNames[kNativeDotPaths]
        = {"fp32", "bf16", "int8", "int4", "fp16", "f16_widen"};

/// One compute path of a core: its dot-product instructions a cycle with
/// enough independent accumulators (ipc), and the cycles from one into an
/// accumulator to the next one that reads it (latency).
struct native_dot_t {
    double ipc;
    int latency;
};

/// Independent accumulator chains that keep path d at its ipc:
/// ceil(ipc * latency).
int native_dot_chains(const native_dot_t &d);

/// Where the per-path model comes from: the class default, or the values
/// ZENDNNL_MATMUL_NATIVE_UARCH sets (a machine's own values come only from
/// running the op perf probe on it).
enum class native_dot_source_t : uint8_t { class_default, set };

/// Indexed by native_dot_source_t.
inline constexpr const char *kNativeDotSourceNames[] = {"default", "env"};

/// Micro-architecture parameters detected at runtime via CPUID.
struct UarchParams {
    int l1d_bytes;
    int l2_bytes;
    /// L3 bytes per L3 domain; detect_uarch() gives the L2 size when CPUID
    /// reports no L3 (l3_detected false), the last cache level there.
    int l3_domain_bytes;
    int num_cores; ///< OMP thread count (may exceed physical cores with SMT)
    /// dot[bf16].ipc rounded; the native BRGEMM reads dot.
    int fma_ports;
    int vec_width_bits;
    bool avx512f;
    bool avx512bf16;
    bool avx512vnni;
    bool avx512fp16;
    /// Cores per L3 domain that the group matmul's prompt tiles assume where
    /// the OS does not describe the domains, and the native BRGEMM plan's L3
    /// domain when l3_domain_cores is 0: a fixed count of physical cores (not
    /// SMT siblings), 8 on the CPU families detect_uarch() recognizes, else 1.
    int family_l3_cores;
    /// Physical cores sharing one L3; 0 when unknown.  detect_uarch() leaves
    /// it 0: only the native BRGEMM plan reads it (native_plan_uarch).
    int l3_domain_cores;
    /// Whether CPUID reported the L1d, L2 and L3 size; set_cache_sizes()
    /// gives the size of one it does not report.  Only the native BRGEMM
    /// reads them: its tuning-file matching, and its plan, which has no L3
    /// without l3_detected.
    bool l1d_detected;
    bool l2_detected;
    bool l3_detected;
    /// The native kernels' compute model, indexed by native_dot_path_t;
    /// dot_source says where it comes from.  The constructor gives the class
    /// default, the cost model's prices without a model: 2 dot products a
    /// cycle and 6 cycles (cm_fma_pipes' and cm_fma_lat's defaults) on every
    /// path.
    native_dot_t dot[kNativeDotPaths];
    native_dot_source_t dot_source;
    /// The cost model's memory terms in cycles of this machine, 0 where not
    /// set (the planner's constants then, 2400 and 600): mem_cycles,
    /// the KiB cost of one thread's weight stream while its L3 domain
    /// streams times the square root of the domain's threads
    /// (cm_mem_cycles); stream_cycles, the KiB cost of one core's stream
    /// alone (cm_stream_cycles).
    int mem_cycles;
    int stream_cycles;

    UarchParams()
        : l1d_bytes(32768)
        , l2_bytes(1048576)
        , l3_domain_bytes(33554432)
        , num_cores(1)
        , fma_ports(2)
        , vec_width_bits(512)
        , avx512f(false)
        , avx512bf16(false)
        , avx512vnni(false)
        , avx512fp16(false)
        , family_l3_cores(1)
        , l3_domain_cores(0)
        , l1d_detected(true)
        , l2_detected(true)
        , l3_detected(true)
        , dot {{2.0, 6}, {2.0, 6}, {2.0, 6}, {2.0, 6}, {2.0, 6}, {2.0, 6}}
        , dot_source(native_dot_source_t::class_default)
        , mem_cycles(0)
        , stream_cycles(0) {}
};

/// Whether the cost model prices kernels at u.dot's dot products a cycle: a
/// model ZENDNNL_MATMUL_NATIVE_UARCH sets.  Otherwise at 2 a cycle on every
/// path.
bool native_dot_prices(const UarchParams &u);

/// CPUID leaf 0's EBX, and leaf 1's family, model and stepping with the
/// extended fields folded in as the processor manuals define; -1 for each
/// without leaf 1.
struct native_cpuid_identity_t {
    uint32_t vendor_ebx = 0;
    int family = -1;
    int model = -1;
    int stepping = -1;
};
native_cpuid_identity_t native_cpuid_identity();

/// The cache fields of p from the sizes CPUID reports, 0 for a level it
/// does not report: a reported size as is; for an unreported L1d or L2 the
/// constructor's size, for an unreported L3 the L2 size.
void set_cache_sizes(UarchParams &p, int l1d, int l2, int l3);

/// Detect micro-architecture parameters via CPUID (cached after first call).
const UarchParams &detect_uarch();

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COST_MODEL_HPP
