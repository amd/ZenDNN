/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// @file reorder_test_helpers.hpp
/// @brief Reorder-specific declarations lifted out of `gtest_utils.hpp` so the
///        operator-agnostic header stays focused on shared infrastructure
///        (tensor_factory_t, RNG, comparators, multi-suite param structs, etc.).
///
/// Mirrors the `group_matmul/group_matmul_test_helpers.{hpp,cpp}` split.
///
/// Surface:
///   * `ReorderType`               — fixture parameter for `TestReorder`
///                                    (embeds MatmulType + LOWOHA fields).
///   * `reorder_test`              — global vector populated by
///                                    `gtest_main.cpp::main()`.
///   * `PrintTo(ReorderType, …)`   — googletest pretty-printer.
///   * reorder kernel shims        — `reorder_kernel_test` (regular path) and
///                                    `lowoha_reorder_kernel_test` (LOWOHA path).
///   * LOWOHA compare / shape / log helpers used by the reorder TEST_P bodies.
///
/// Consumers:
///   * `gtest_main.cpp` — the only file outside `reorder/` that touches these
///     symbols (it owns the `reorder_test` fill loop and definition).
///   * `reorder/reorder_test_common.hpp` (the shared TestReorder fixture)
///     includes this header, so every `reorder/test_*.cpp` sees it.
///
/// Note: `ReorderInput` (the CLI/file input struct) intentionally stays in
/// `gtest_utils.hpp` because it is a member of the shared `CLIParams`, and
/// `read_reorder_inputs(...)` stays in `gtest_utils.cpp` alongside the other
/// `read_*_inputs` file-parsing helpers (it shares `parse_bool_field`).

#ifndef ZENDNNL_GTESTS_REORDER_REORDER_TEST_HELPERS_HPP
#define ZENDNNL_GTESTS_REORDER_REORDER_TEST_HELPERS_HPP

#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "gtest_utils.hpp"

/** @brief Reorder Op Parameters Structure (fixture parameter for TestReorder).
 *
 *  Embeds a `MatmulType` for the legacy regular reorder + matmul path and a
 *  set of LOWOHA-specific fields used when `is_lowoha_test == true`.
 */
struct ReorderType {
  bool inplace_reorder = false;
  MatmulType mat{};

  // LOWOHA-specific parameters (used when is_lowoha_test = true)
  // Flag to distinguish LOWOHA reorder tests
  bool is_lowoha_test = false;
  // Rows dimension (LOWOHA)
  uint64_t M = 0;
  //  Columns dimension (LOWOHA)
  uint64_t N = 0;
  //  Batch dimension (0=1D, 1=2D, >1=3D) (LOWOHA)
  uint64_t batch = 0;
  //  Source data type (LOWOHA, set by TEST_P)
  data_type_t src_dtype = data_type_t::f32;
  //  Destination data type (LOWOHA, set by TEST_P)
  data_type_t dst_dtype = data_type_t::s8;
  //  Quantization granularity (LOWOHA)
  quant_granularity_t granularity = quant_granularity_t::tensor;
  //  Number of groups for per-group quant (LOWOHA)
  uint64_t num_groups = 0;
  //  Use strided source memory (LOWOHA, set by TEST_P)
  bool use_strided_src = false;
  //  LOWOHA algorithm selection (LOWOHA, set by TEST_P)
  reorder_algo_t lowoha_algo = reorder_algo_t::native;
  //  Number of threads (LOWOHA)
  int32_t num_threads = 1;

  // Per-instance sub-mode selections, fixed at construction so coverage stays
  // diverse yet reproducible and order-independent for a fixed --seed.
  bool is_symmetric = true;        // quant: S8 (symmetric) vs U8 (asymmetric)
  bool cvt_direction_swap = false; // type conversion: which dtype is the source
  bool use_col_variant = false;    // quant scale/zp shape: per-col vs per-row
  uint64_t row_padding = 0;        // strided: extra row padding (0 = contiguous)

  /// @param test_index Index of current test (for partitioning)
  /// @param total_tests Total number of tests
  ReorderType(const ReorderInput &reorder_input = ReorderInput(),
              uint32_t test_index = 0, uint32_t total_tests = 1);
};

/// Global vector populated once by `gtest_main.cpp::main()`; consumed by
/// `TestReorder` (`reorder/test_reorder_regular.cpp`) via `ValuesIn`.
extern std::vector<ReorderType> reorder_test;

/** @brief Print ReorderType for GTest parameterized test failure messages. */
void PrintTo(const ReorderType &value, ::std::ostream *os);

/** @fn reorder_kernel_test
 *  @brief Function to Reorder tensor
 *
 *  This function reorders/unreorder the tensor either by Inplace or OutofPlace.
 *  @return Updated tensor and status
 *
 * */
std::pair<tensor_t, status_t> reorder_kernel_test(tensor_t &input_tensor,
    bool inplace_reorder, void **reorder_weights,
    data_type_t source_dtype = data_type_t::f32);

/** @fn lowoha_reorder_kernel_test
 *  @brief Test function for LOWOHA reorder kernel (quantization/dequantization)
 *
 *  Executes the LOWOHA reorder_direct API with the specified parameters.
 *
 *  @param src_tensor Source tensor
 *  @param dst_tensor Destination tensor
 *  @param scale_tensor Scale tensor for quantization
 *  @param zp_tensor Zero-point tensor for quantization
 *  @param params LOWOHA reorder test parameters
 *  @param dynamic_quant If true, enables dynamic quantization (default: false)
 *  @return status_t Success or failure status
 */
status_t lowoha_reorder_kernel_test(tensor_t &src_tensor,
                                    tensor_t &dst_tensor,
                                    tensor_t &scale_tensor,
                                    tensor_t &zp_tensor,
                                    const ReorderType &params,
                                    bool dynamic_quant = false);

/** @fn compare_lowoha_reorder_output
 *  @brief Compare LOWOHA reorder output with reference
 *
 *  Compares actual output tensor with expected reference tensor element by element.
 *
 *  @param output_tensor Actual output tensor
 *  @param ref_tensor Reference tensor
 *  @param params LOWOHA reorder test parameters
 *  @param is_comparison_successful Output flag indicating comparison result
 */
void compare_lowoha_reorder_output(tensor_t &output_tensor,
                                   tensor_t &ref_tensor,
                                   const ReorderType &params,
                                   bool &is_comparison_successful);

/** @fn lowoha_granularity_to_str
 *  @brief Convert quantization granularity enum to string
 *
 *  @param granularity The quant_granularity_t enum value
 *  @return std::string String representation
 */
std::string lowoha_granularity_to_str(quant_granularity_t granularity);

/** @fn lowoha_reorder_algo_to_str
 *  @brief Convert LOWOHA reorder algorithm enum to string
 *
 *  @param algo The reorder_algo_t enum value
 *  @return std::string String representation
 */
std::string lowoha_reorder_algo_to_str(reorder_algo_t algo);

/** @fn log_lowoha_test_info
 *  @brief Log LOWOHA test information
 *
 *  @param params LOWOHA reorder test parameters
 *  @param src_dt Source data type
 *  @param dst_dt Destination data type
 *  @param strided Whether strided memory is used
 *  @param use_scale_zp Whether scale/zero-point is used
 */
void log_lowoha_test_info(const ReorderType &params, data_type_t src_dt,
                          data_type_t dst_dt, bool strided, bool use_scale_zp);

/** @fn get_lowoha_shape
 *  @brief Get LOWOHA tensor shape based on batch dimension
 *
 *  @param params LOWOHA reorder test parameters
 *  @return std::vector<size_t> Shape vector (1D, 2D, or 3D)
 */
std::vector<size_t> get_lowoha_shape(const ReorderType &params);

/** @fn get_lowoha_strided_shape
 *  @brief Get strided shape with row padding for LOWOHA tests
 *
 *  Row padding is taken from `params.row_padding` (chosen once per instance in
 *  the ReorderType constructor), so the strided layout is reproducible and
 *  order-independent for a fixed --seed.
 *
 *  @param params LOWOHA reorder test parameters
 *  @return std::vector<size_t> Strided shape vector
 */
std::vector<size_t> get_lowoha_strided_shape(const ReorderType &params);

/** @fn get_lowoha_quant_shape
 *  @brief Get quantization parameter shape based on granularity
 *
 *  Supports tensor/channel/group granularity and returns the corresponding
 *  shape used for scale/zero-point tensors.
 *
 *  @param params LOWOHA reorder test parameters
 *  @return std::vector<size_t> Scale/zero-point shape vector
 */
std::vector<size_t> get_lowoha_quant_shape(const ReorderType &params);

/** @fn compare_lowoha_quant_output
 *  @brief Compare original input with dequantized output for quantization
 *         round-trip validation
 *
 *  Compares the original source tensor with the dequantized tensor after a
 *  quantization → dequantization round trip. Tolerance is computed
 *  based on the quantization scale (max error ≈ scale/2 + numerical epsilon).
 *
 *  @param original_tensor Original source tensor (f32 or bf16)
 *  @param dequant_tensor Dequantized tensor (same dtype as original)
 *  @param scale_tensor Computed scale tensor from quantization
 *  @param params LOWOHA reorder test parameters
 *  @param is_comparison_successful Output flag indicating comparison result
 */
void compare_lowoha_quant_output(tensor_t &original_tensor,
                                 tensor_t &dequant_tensor,
                                 tensor_t &scale_tensor,
                                 const ReorderType &params,
                                 bool &is_comparison_successful);

#endif  // ZENDNNL_GTESTS_REORDER_REORDER_TEST_HELPERS_HPP
