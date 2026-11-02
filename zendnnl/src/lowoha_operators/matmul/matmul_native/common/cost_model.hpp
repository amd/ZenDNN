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

/// Micro-architecture parameters detected at runtime via CPUID.
struct UarchParams {
    int l1d_bytes;
    int l2_bytes;
    /// L3 bytes per L3 domain; detect_uarch() gives the L2 size when CPUID
    /// reports no L3 (l3_detected false), the last cache level there.
    int l3_domain_bytes;
    int num_cores; ///< OMP thread count (may exceed physical cores with SMT)
    int fma_ports;
    int vec_width_bits;
    bool avx512f;
    bool avx512bf16;
    bool avx512vnni;
    /// Cores per L3 domain that the M=1 GEMV slice permutation assumes, and
    /// the native BRGEMM plan's L3 domain when l3_domain_cores is 0: a fixed
    /// count of physical cores (not SMT siblings), 8 on the CPU families
    /// detect_uarch() recognizes, else 1 (column slices in linear tid order).
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
        , family_l3_cores(1)
        , l3_domain_cores(0)
        , l1d_detected(true)
        , l2_detected(true)
        , l3_detected(true) {}
};

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
