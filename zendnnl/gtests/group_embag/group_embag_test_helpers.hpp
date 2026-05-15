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

/// @file group_embag_test_helpers.hpp
/// @brief Group-embag gtest declarations.
///
/// Surface:
///   * `GroupEmbagType`              - random fixture parameter.
///   * `group_embag_test`            - global parameter vector.
///   * `PrintTo(GroupEmbagType, ...)`- googletest pretty-printer.
///   * `GroupTensors`                - per-table tensor container.
///   * `build_group_embag_tensors`            - float-table builder (bag mode).
///   * `build_group_embag_quant_tensors`      - quantized-table builder (bag mode).
///   * `build_group_embedding_tensors`        - float-table builder (lookup mode).
///   * `build_group_embedding_quant_tensors`  - quantized-table builder (lookup mode).
///   * `free_quant_tables`           - free the raw buffers behind
///                                     quantized tables.
///   * `compare_group_outputs`       - per-table tensor compare.
///   * `group_embag_kernel_test`     - dispatch shim into
///                                     `group_embedding_bag_direct`.
///   * `group_embag_forced_ref_kernel_test` - reference shim that loops
///                                     the per-table forced-reference
///                                     kernel.

#ifndef ZENDNNL_GTESTS_GROUP_EMBAG_TEST_HELPERS_HPP
#define ZENDNNL_GTESTS_GROUP_EMBAG_TEST_HELPERS_HPP

#include <ostream>
#include <vector>

#include "gtest_utils.hpp"
#include "operators/embag/embag_config.hpp"

/** @brief Random parameter set for the TestGroupEmbag* fixtures.
 *
 *  `thread_algo` is picked uniformly across {batch, table, ccd, hybrid}
 *  so every scheduler gets coverage across parameterized runs.  F16
 *  cases self-skip when the host lacks AVX-512 FP16.
 */
struct GroupEmbagType {
  uint64_t num_embeddings;
  uint64_t embedding_dim;
  uint64_t num_indices;
  uint64_t num_bags;
  embag_algo_t algo;            // sum / mean / max
  int64_t  padding_index;
  bool     include_last_offset;
  bool     is_weights;
  data_type_t indices_dtype;    // s32 or s64 (offsets_dtype tracks this)
  data_type_t offsets_dtype;
  bool     fp16_scale_bias;
  size_t   group_size;          // 2..5 tables
  eb_thread_algo_t thread_algo; // batch / table / ccd / hybrid
  int32_t  num_threads;
  GroupEmbagType(uint32_t test_index = 0, uint32_t total_tests = 1);
};

/// Global vector populated once by `gtest_main.cpp::main()`.
extern std::vector<GroupEmbagType> group_embag_test;

/** @brief Pretty-print GroupEmbagType for GTest failure messages. */
void PrintTo(const GroupEmbagType &value, ::std::ostream *os);

/** @brief Per-table tensor container used by both test files. */
struct GroupTensors {
  std::vector<tensor_t> tables;
  std::vector<tensor_t> indices;
  std::vector<tensor_t> offsets;     // empty in lookup mode
  std::vector<tensor_t> weights;     // empty when is_weights == false
  std::vector<tensor_t> outputs;
  std::vector<tensor_t> outputs_ref;
  std::vector<embag_algo_t> algos;
  std::vector<int64_t> padding_idxs;
  std::vector<bool> include_last_offsets;
  std::vector<bool> fp16_scale_bias;
};

/** @brief Build a group of float-tabled bag-mode tensors with the
 *         same params shared across every table. */
GroupTensors build_group_embag_tensors(
  tensor_factory_t &factory,
  size_t group_size,
  uint64_t num_embeddings, uint64_t embedding_dim,
  uint64_t num_indices, uint64_t num_bags,
  embag_algo_t algo, int64_t padding_index,
  bool include_last_offset, bool is_weights, bool fp16_scale_bias_v,
  data_type_t table_dtype, data_type_t output_dtype,
  data_type_t indices_dtype, data_type_t offsets_dtype);

/** @brief Build a group of quantized-table (INT8/S4/U4) bag-mode
 *         tensors with the same params shared across every table.
 *         Table buffers are heap-allocated; call free_quant_tables()
 *         before the GroupTensors goes out of scope. */
GroupTensors build_group_embag_quant_tensors(
  tensor_factory_t &factory,
  size_t group_size,
  uint64_t num_embeddings, uint64_t embedding_dim,
  uint64_t num_indices, uint64_t num_bags,
  embag_algo_t algo, int64_t padding_index,
  bool include_last_offset, bool is_weights, bool fp16_scale_bias_v,
  data_type_t table_dtype, data_type_t output_dtype,
  data_type_t indices_dtype, data_type_t offsets_dtype);

/** @brief Build a group of float-tabled lookup-mode tensors
 *         (algo = none, offsets left empty so the shim forwards
 *         nullptr).  Output shape per table is
 *         [num_indices, embedding_dim]. */
GroupTensors build_group_embedding_tensors(
  tensor_factory_t &factory,
  size_t group_size,
  uint64_t num_embeddings, uint64_t embedding_dim,
  uint64_t num_indices,
  int64_t padding_index, bool is_weights, bool fp16_scale_bias_v,
  data_type_t table_dtype, data_type_t output_dtype,
  data_type_t indices_dtype);

/** @brief Build a group of quantized-table lookup-mode tensors.
 *         Table buffers are heap-allocated; call free_quant_tables()
 *         before the GroupTensors goes out of scope. */
GroupTensors build_group_embedding_quant_tensors(
  tensor_factory_t &factory,
  size_t group_size,
  uint64_t num_embeddings, uint64_t embedding_dim,
  uint64_t num_indices,
  int64_t padding_index, bool is_weights, bool fp16_scale_bias_v,
  data_type_t table_dtype, data_type_t output_dtype,
  data_type_t indices_dtype);

/** @brief free() the raw buffers behind quantized tables and reset the
 *         tensors.  Must be called before the GroupTensors built by
 *         the *_quant_* builders goes out of scope, since
 *         `quantized_embedding_tensor_random` returns a heap-allocated
 *         buffer that the tensor_t does not own. */
void free_quant_tables(GroupTensors &g);

/** @brief Per-table tensor compare; logs the first mismatched table.
 *  @return true if every table is within tolerance. */
bool compare_group_outputs(GroupTensors &g, uint64_t output_rows,
                           uint64_t embedding_dim, float tol);

/** @fn group_embag_kernel_test
 *  @brief Dispatch shim into group_embedding_bag_direct.
 *
 *  Lookup mode (group embedding): pass `algos[i] == embag_algo_t::none`
 *  and an empty `offsets[i]` tensor; the shim then forwards nullptr
 *  for that table's offsets pointer.
 *
 *  Pins `thread_algo` for the call by setting `ZENDNNL_EMBAG_THREAD_ALGO`
 *  via an RAII guard.  The env var is the source of truth because
 *  `group_embedding_bag_direct` re-reads it on every call through
 *  `embag_config_t::set_env_config()`; the guard saves and restores
 *  the previous env value so consecutive tests do not leak scheduler
 *  state.
 */
status_t group_embag_kernel_test(
  std::vector<tensor_t> &tables,
  std::vector<tensor_t> &indices,
  std::vector<tensor_t> &offsets,
  std::vector<tensor_t> &weights,
  std::vector<tensor_t> &outputs,
  const std::vector<embag_algo_t> &algos,
  const std::vector<int64_t> &padding_idxs,
  const std::vector<bool> &include_last_offsets,
  const std::vector<bool> &fp16_scale_bias,
  eb_thread_algo_t thread_algo);

/** @brief Reference: loops the single-op forced-reference kernel
 *         (`embag_forced_ref_kernel_test` or
 *         `embedding_forced_ref_kernel_test` depending on
 *         `algos[i]`) per table; returns failure on the first
 *         per-table reference failure. */
status_t group_embag_forced_ref_kernel_test(
  std::vector<tensor_t> &tables,
  std::vector<tensor_t> &indices,
  std::vector<tensor_t> &offsets,
  std::vector<tensor_t> &weights,
  std::vector<tensor_t> &outputs,
  const std::vector<embag_algo_t> &algos,
  const std::vector<int64_t> &padding_idxs,
  const std::vector<bool> &include_last_offsets,
  const std::vector<bool> &fp16_scale_bias);

#endif // ZENDNNL_GTESTS_GROUP_EMBAG_TEST_HELPERS_HPP
