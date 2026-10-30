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
    int l3_bytes_per_ccd;
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
    int ccx_cores;
    /// Physical cores sharing one L3; 0 when unknown.  detect_uarch() leaves
    /// it 0: only the native BRGEMM plan reads it (native_plan_uarch).
    int l3_domain_cores;
    /// Whether CPUID reported the L1d, L2 and L3 size; a size it does not
    /// report keeps the value of the constructor.  Only the native BRGEMM's
    /// tuning-file matching reads them.
    bool l1d_detected;
    bool l2_detected;
    bool l3_detected;

    UarchParams()
        : l1d_bytes(32768)
        , l2_bytes(1048576)
        , l3_bytes_per_ccd(33554432)
        , num_cores(1)
        , fma_ports(2)
        , vec_width_bits(512)
        , avx512f(false)
        , avx512bf16(false)
        , avx512vnni(false)
        , ccx_cores(1)
        , l3_domain_cores(0)
        , l1d_detected(true)
        , l2_detected(true)
        , l3_detected(true) {}
};

/// Detect micro-architecture parameters via CPUID (cached after first call).
const UarchParams &detect_uarch();

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COST_MODEL_HPP
