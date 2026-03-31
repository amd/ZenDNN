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
  int num_cores;  ///< OMP thread count (may exceed physical cores with SMT)
  int fma_ports;
  int vec_width_bits;
  bool avx512f;
  bool avx512bf16;
  bool avx512vnni;
  /// Logical cores per CCX for M=1 GEMV OpenMP slice scheduling (AMD Zen ≈ 8).
  /// When 1, column slices follow linear tid order.
  int ccx_cores;

  UarchParams()
    : l1d_bytes(32768), l2_bytes(1048576), l3_bytes_per_ccd(33554432),
      num_cores(1), fma_ports(2), vec_width_bits(512),
      avx512f(false), avx512bf16(false), avx512vnni(false), ccx_cores(1) {}
};

/// Detect micro-architecture parameters via CPUID (cached after first call).
const UarchParams &detect_uarch();

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COST_MODEL_HPP
