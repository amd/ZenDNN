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
 ******************************************************************************/

#ifndef MATMUL_NATIVE_BRGEMM_PLANNER_HPP
#define MATMUL_NATIVE_BRGEMM_PLANNER_HPP

#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// BRGEMM blocking plan.
///
/// Unlike GEMM (where the thread loop iterates KC blocks and the microkernel
/// processes one KC block per call), BRGEMM gives the microkernel the FULL K
/// dimension. The microkernel internally iterates K in BK-sized blocks,
/// keeping MR×NR accumulators live in registers across all K-blocks.
///
/// Thread loop: parallel over MC × NC tiles only (no PC loop).
/// Each thread calls the BRGEMM microkernel once per (MC, NC) tile.
struct BrgemmPlan {
  int MB;           ///< M-block size (MC tile)
  int NB;           ///< N-block size (NC tile)
  int BK;           ///< K-block size inside microkernel
  int MR;           ///< Register tile rows
  int NR;           ///< Register tile columns
  int num_threads;  ///< Thread count
};

/// Select FP32 BRGEMM blocking parameters (MR=6, NR=16).
BrgemmPlan plan_fp32_brgemm(const GemmDescriptor &desc,
                            const UarchParams &uarch);

/// Select BF16 BRGEMM blocking parameters (NR=64, VNNI layout).
/// Computes MR, BK, NB, MB from scratch for BF16 dpbf16ps microkernels.
BrgemmPlan plan_bf16_brgemm(const GemmDescriptor &desc,
                            const UarchParams &uarch);

/// Select INT8 BRGEMM blocking parameters (NR=64, INT8 VNNI layout).
/// INT8 uses 1-byte elements and 4-byte VNNI groups (vpdpbusd).
BrgemmPlan plan_int8_brgemm(const GemmDescriptor &desc,
                            const UarchParams &uarch);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_PLANNER_HPP
