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

#ifndef MATMUL_NATIVE_COMMON_NATIVE_HW_HPP
#define MATMUL_NATIVE_COMMON_NATIVE_HW_HPP

#include <string>

#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Hardware facts of the process.  The built-in heuristic reads
/// native_plan_uarch(*this): uarch and the physical cores of one L3 domain
/// (l3_domain_threads / smt).  The other topology fields feed only
/// tuning-file matching and rule features, the CPU identity only tuning-file
/// matching.  A count that could not be detected is 0; an identity field,
/// empty or -1.  A cache size CPUID does not report holds detect_uarch()'s
/// default, which tuning-file matching treats as not reported.
struct native_hw_t {
    UarchParams uarch; ///< detect_uarch()
    std::string vendor; ///< CPUID vendor string
    int family = -1; ///< CPUID family, extended family included
    int model = -1; ///< CPUID model, extended model included
    int stepping = -1; ///< CPUID stepping
    int logical_cpus = 0; ///< online logical CPUs, from the OS
    int smt = 0; ///< logical CPUs per core
    int physical_cores = 0; ///< logical_cpus / smt
    int l3_domain_threads = 0; ///< logical CPUs sharing the L3 of CPU 0
};

/// Detected once per process, on first use: by the first native BRGEMM plan,
/// or earlier by a tuning file.
const native_hw_t &native_hw();

/// What the native BRGEMM plan reads on hw: hw.uarch with l3_domain_cores =
/// l3_domain_threads / smt (smt 1 when unknown; at least 1), or 0 when
/// l3_domain_threads is unknown, where the plan uses ccx_cores instead.
UarchParams native_plan_uarch(const native_hw_t &hw);

/// native_plan_uarch(native_hw()), made once.
const UarchParams &native_plan_uarch();

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_HW_HPP
