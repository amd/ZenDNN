/********************************************************************************
# * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

/// @file group_matmul_test_helpers.hpp
/// @brief Group-matmul-specific declarations lifted out of `gtest_utils.hpp`
///        so the operator-agnostic header stays focused on shared
///        infrastructure (tensor_factory_t, RNG, comparators, multi-suite
///        param structs, etc.).
///
/// Surface:
///   * `GroupQuantMatmulType`         — fixture parameter for
///                                       `TestGroupMatmulQuant` (constrained
///                                       random shape/algo/dtype, no
///                                       transpose, alpha=1, beta=0, no
///                                       random post-op chain).
///   * `quant_matmul_test`            — global vector populated by
///                                       `gtest_main.cpp::main()`.
///   * `PrintTo(GroupQuantMatmulType, …)`  — googletest pretty-printer.
///   * `group_matmul_kernel_test(…)`  — dispatch shim that wraps vectors
///                                       of (input, weight, bias, output)
///                                       tensors into `group_matmul_direct`.
///
/// Consumers:
///   * `gtest_main.cpp` — the only file outside `group_matmul/` that
///     touches these symbols (it owns the `quant_matmul_test` fill loop).
///   * `group_matmul/test_*.cpp` — the test bodies.

#ifndef ZENDNNL_GTESTS_GROUP_MATMUL_TEST_HELPERS_HPP
#define ZENDNNL_GTESTS_GROUP_MATMUL_TEST_HELPERS_HPP

#include <ostream>
#include <vector>

#include "gtest_utils.hpp"

/** @brief Constrained Group-Matmul parameters for the quantized test suites.
 *
 *  The shared `MatmulType` random grid (alpha, beta ∈ U[0,10]; random
 *  transA/transB; arbitrary K) drives the quantized group-matmul kernels
 *  outside the regime where a meaningful tolerance can be derived for an
 *  activated comparison: `silu(g) · u`-style outputs inherit an O(α²·k²·ε)
 *  noise term that swamps any rel/abs bound for large α·β, and INT8 quant
 *  rounding on transposed inputs can deviate from the f32 reference by
 *  amounts that the standard `compare_tensor_2D_matrix` envelope (tuned for
 *  the un-quantized BF16/F32 path) cannot bound.
 *
 *  This struct pins the destabilizing knobs to safe defaults via the
 *  fixture's member constants — no transpose, alpha = 1, beta = 0, no
 *  random post-op chain — and only randomizes the shape / algo / dtype
 *  axes that produce useful coverage.  K is rounded down to a multiple of
 *  4 so INT8 K-grouping in the symmetric / dynamic-quant tests doesn't
 *  need per-test re-rounding.
 */
struct GroupQuantMatmulType {
  uint64_t matmul_m;
  uint64_t matmul_k;
  uint64_t matmul_n;
  matmul_algo_t algo = matmul_algo_t::none;
  data_type_t source_dtype;
  data_type_t output_dtype;
  quant_granularity_t weight_granularity;
  int32_t num_threads;
  GroupQuantMatmulType(uint32_t test_index = 0, uint32_t total_tests = 1);
};

/// Global vector populated once by `gtest_main.cpp::main()`; consumed by
/// `TestGroupMatmulQuant` (`group_matmul/test_quant.cpp`) via `ValuesIn`.
extern std::vector<GroupQuantMatmulType> quant_matmul_test;

/** @brief Print GroupQuantMatmulType for GTest parameterized test failure messages. */
void PrintTo(const GroupQuantMatmulType &value, ::std::ostream *os);

/** @fn group_matmul_kernel_test
 *  @brief Run group matmul through group_matmul_direct for multiple experts.
 *
 *  Wraps vectors of (input, weight, bias, output) tensors into the
 *  group_matmul_direct API. For each expert, derives transA/transB from tensor
 *  order tags and lda/ldb from tensor strides — matching how matmul_kernel_test
 *  works for the single-op path. Automatically extracts quantization parameters
 *  (WOQ, INT8, dynamic quant) from tensor metadata.
 *
 *  @param inputs   Vector of input tensors (one per expert)
 *  @param weights  Vector of weight tensors (one per expert)
 *  @param biases   Vector of bias tensors (one per expert, empty tensor for no bias)
 *  @param outputs  Vector of output tensors (one per expert)
 *  @param algo     Matmul algorithm to use
 *  @param alpha    Scaling factor for A*B (default 1.0)
 *  @param beta     Scaling factor for existing C values (default 0.0)
 *  @param moe_postop Optional MoE post-op params (default nullptr)
 *  @param gated_act  Optional gated-activation params (default nullptr).
 *                    When provided with act != none, the kernel applies the
 *                    activation in-place to the first N/2 columns of each
 *                    expert's output (requires even N and f32/bf16 dst).
 *  @return group_matmul_direct status
 */
status_t group_matmul_kernel_test(
  std::vector<tensor_t> &inputs,
  std::vector<tensor_t> &weights,
  std::vector<tensor_t> &biases,
  std::vector<tensor_t> &outputs,
  matmul_algo_t algo,
  float alpha = 1.0f,
  float beta = 0.0f,
  const group_matmul_moe_postop_params *moe_postop = nullptr,
  const grp_matmul_gated_act_params *gated_act = nullptr);

#endif // ZENDNNL_GTESTS_GROUP_MATMUL_TEST_HELPERS_HPP
