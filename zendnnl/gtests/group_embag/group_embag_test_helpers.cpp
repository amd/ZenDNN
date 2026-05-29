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

/// @file group_embag_test_helpers.cpp
/// @brief Definitions for the group-embag gtest helpers.

#include "group_embag_test_helpers.hpp"

#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

namespace {

// RAII pin of the embag thread algorithm via ZENDNNL_EMBAG_THREAD_ALGO.
//
// group_embedding_bag_direct calls embag_config_t::set_env_config()
// before each dispatch, which re-reads this env var and overwrites
// the in-memory thread_algo field.  So pinning the singleton directly
// has no effect — the env var is the actual source of truth and must
// be set here to force a specific scheduler.
//
// Not thread-safe: setenv/getenv are racy across threads.  Fine for
// the gtests binary since tests run serially within the process.
class scoped_thread_algo {
 public:
  explicit scoped_thread_algo(eb_thread_algo_t algo) {
    if (const char *p = std::getenv(kEnvName)) {
      prev_value_ = p;
      had_prev_   = true;
    }
    setenv(kEnvName,
           std::to_string(static_cast<int32_t>(algo)).c_str(),
           /*overwrite=*/1);
  }
  ~scoped_thread_algo() {
    if (had_prev_) {
      setenv(kEnvName, prev_value_.c_str(), /*overwrite=*/1);
    }
    else {
      unsetenv(kEnvName);
    }
  }
  scoped_thread_algo(const scoped_thread_algo &) = delete;
  scoped_thread_algo &operator=(const scoped_thread_algo &) = delete;

 private:
  static constexpr const char *kEnvName = "ZENDNNL_EMBAG_THREAD_ALGO";
  std::string prev_value_;
  bool        had_prev_ = false;
};

const char *thread_algo_str(eb_thread_algo_t algo) {
  switch (algo) {
  case eb_thread_algo_t::batch_threaded:
    return "batch_threaded";
  case eb_thread_algo_t::table_threaded:
    return "table_threaded";
  case eb_thread_algo_t::ccd_threaded:
    return "ccd_threaded";
  case eb_thread_algo_t::hybrid_threaded:
    return "hybrid_threaded";
  case eb_thread_algo_t::auto_tuner:
    return "auto_tuner";
  case eb_thread_algo_t::dynamic_dispatch:
    return "dynamic_dispatch";
  default:
    return "unknown";
  }
}

const char *algo_str(embag_algo_t algo) {
  switch (algo) {
  case embag_algo_t::none:
    return "none";
  case embag_algo_t::sum:
    return "sum";
  case embag_algo_t::mean:
    return "mean";
  case embag_algo_t::max:
    return "max";
  default:
    return "unknown";
  }
}

} // namespace

// ── GroupEmbagType constructor ──────────────────────────────────────────────
// Each axis is either CLI-overridable (`cli_params`) or randomly
// drawn within the EMBEDDING_* / NUM_* / NUM_BAGS_* envelopes defined
// in `gtest_utils.hpp`.
GroupEmbagType::GroupEmbagType(uint32_t test_index, uint32_t total_tests) {
  num_embeddings = (cli_params.embag_input.embedding_input.num_embeddings &&
                    *cli_params.embag_input.embedding_input.num_embeddings > 0)
                   ? *cli_params.embag_input.embedding_input.num_embeddings
                   : (EMBEDDING_SIZE_START + std::rand() % EMBEDDING_SIZE_END);
  embedding_dim  = (cli_params.embag_input.embedding_input.embedding_dim &&
                    *cli_params.embag_input.embedding_input.embedding_dim > 0)
                   ? *cli_params.embag_input.embedding_input.embedding_dim
                   : (EMBEDDING_DIM_START + std::rand() % EMBEDDING_DIM_END);
  num_indices    = (cli_params.embag_input.embedding_input.num_indices &&
                    *cli_params.embag_input.embedding_input.num_indices > 0)
                   ? *cli_params.embag_input.embedding_input.num_indices
                   : (NUM_INDICES_START + std::rand() % NUM_INDICES_END);
  num_bags       = (cli_params.embag_input.num_bags &&
                    *cli_params.embag_input.num_bags > 0)
                   ? *cli_params.embag_input.num_bags
                   : (NUM_BAGS_START + std::rand() % NUM_BAGS_END);

  algo = cli_params.embag_input.embag_algo
         ? *cli_params.embag_input.embag_algo
         : static_cast<embag_algo_t>(1 + std::rand() % 3); // sum / mean / max

  padding_index       = cli_params.embag_input.embedding_input.padding_index ?
                        *cli_params.embag_input.embedding_input.padding_index : -1;
  include_last_offset = cli_params.embag_input.include_last_offset
                        ? *cli_params.embag_input.include_last_offset
                        : (std::rand() % 2);
  is_weights = cli_params.embag_input.embedding_input.is_weights ?
               *cli_params.embag_input.embedding_input.is_weights : (std::rand() % 2);

  if (cli_params.embag_input.embedding_input.indices_dtype &&
      (*cli_params.embag_input.embedding_input.indices_dtype == data_type_t::s32 ||
       *cli_params.embag_input.embedding_input.indices_dtype == data_type_t::s64)) {
    indices_dtype = *cli_params.embag_input.embedding_input.indices_dtype;
  }
  else {
    indices_dtype = (std::rand() % 2 == 0) ? data_type_t::s32 : data_type_t::s64;
  }
  offsets_dtype = indices_dtype;

  fp16_scale_bias = cli_params.embag_input.embedding_input.fp16_scale_bias ?
                    *cli_params.embag_input.embedding_input.fp16_scale_bias :
                    (std::rand() % 2);

  // 2..5 tables — realistic deployment range, also where the four
  // thread strategies meaningfully differ.
  group_size = 2 + (std::rand() % 4);

  static constexpr eb_thread_algo_t thread_algo_choices[4] = {
    eb_thread_algo_t::batch_threaded,
    eb_thread_algo_t::table_threaded,
    eb_thread_algo_t::ccd_threaded,
    eb_thread_algo_t::hybrid_threaded,
  };
  thread_algo = thread_algo_choices[std::rand() % 4];

  if (cmd_num_threads) {
    num_threads = static_cast<int32_t>(cmd_num_threads);
  }
  else {
    int max_threads = omp_get_max_threads();
    static std::mt19937 gen(std::rand());
    std::uniform_int_distribution<int> thread_dist(1, max_threads);
    num_threads = thread_dist(gen);
  }
  (void)test_index;
  (void)total_tests;
}

void PrintTo(const GroupEmbagType &v, ::std::ostream *os) {
  *os << "num_embeddings=" << v.num_embeddings
      << ", embedding_dim=" << v.embedding_dim
      << ", num_indices=" << v.num_indices
      << ", num_bags=" << v.num_bags
      << ", algo=" << algo_str(v.algo)
      << ", padding_index=" << v.padding_index
      << ", include_last_offset=" << v.include_last_offset
      << ", is_weights=" << v.is_weights
      << ", indices_dtype=" << dtype_info(v.indices_dtype)
      << ", offsets_dtype=" << dtype_info(v.offsets_dtype)
      << ", fp16_scale_bias=" << v.fp16_scale_bias
      << ", group_size=" << v.group_size
      << ", thread_algo=" << thread_algo_str(v.thread_algo)
      << ", num_threads=" << v.num_threads
      << ", seed=" << seed;
}

// ── GroupTensors builders ───────────────────────────────────────────────────

GroupTensors build_group_embag_tensors(
  tensor_factory_t &factory,
  size_t group_size,
  uint64_t num_embeddings, uint64_t embedding_dim,
  uint64_t num_indices, uint64_t num_bags,
  embag_algo_t algo, int64_t padding_index,
  bool include_last_offset, bool is_weights, bool fp16_scale_bias_v,
  data_type_t table_dtype, data_type_t output_dtype,
  data_type_t indices_dtype, data_type_t offsets_dtype) {
  GroupTensors g;
  g.tables.reserve(group_size);
  g.indices.reserve(group_size);
  g.offsets.reserve(group_size);
  g.weights.reserve(group_size);
  g.outputs.reserve(group_size);
  g.outputs_ref.reserve(group_size);
  g.algos.assign(group_size, algo);
  g.padding_idxs.assign(group_size, padding_index);
  g.include_last_offsets.assign(group_size, include_last_offset);
  g.fp16_scale_bias.assign(group_size, fp16_scale_bias_v);

  const uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;

  for (size_t i = 0; i < group_size; ++i) {
    g.tables.push_back(factory.uniform_dist_tensor(
    {num_embeddings, embedding_dim}, table_dtype, 2.0f));
    g.indices.push_back(factory.random_indices_tensor(
    {num_indices}, num_embeddings, indices_dtype));
    g.offsets.push_back(factory.random_offsets_tensor(
    {offsets_size}, num_indices, offsets_dtype,
    include_last_offset));
    g.weights.push_back(is_weights
                        ? factory.uniform_dist_tensor({num_indices},
                            data_type_t::f32, 2.0f)
                        : tensor_t{});
    g.outputs.push_back(factory.zero_tensor(
    {num_bags, embedding_dim}, output_dtype));
    g.outputs_ref.push_back(factory.zero_tensor(
    {num_bags, embedding_dim}, output_dtype));
  }
  return g;
}

GroupTensors build_group_embag_quant_tensors(
  tensor_factory_t &factory,
  size_t group_size,
  uint64_t num_embeddings, uint64_t embedding_dim,
  uint64_t num_indices, uint64_t num_bags,
  embag_algo_t algo, int64_t padding_index,
  bool include_last_offset, bool is_weights, bool fp16_scale_bias_v,
  data_type_t table_dtype, data_type_t output_dtype,
  data_type_t indices_dtype, data_type_t offsets_dtype) {
  GroupTensors g;
  g.tables.reserve(group_size);
  g.indices.reserve(group_size);
  g.offsets.reserve(group_size);
  g.weights.reserve(group_size);
  g.outputs.reserve(group_size);
  g.outputs_ref.reserve(group_size);
  g.algos.assign(group_size, algo);
  g.padding_idxs.assign(group_size, padding_index);
  g.include_last_offsets.assign(group_size, include_last_offset);
  g.fp16_scale_bias.assign(group_size, fp16_scale_bias_v);

  const uint64_t offsets_size = include_last_offset ? num_bags + 1 : num_bags;

  for (size_t i = 0; i < group_size; ++i) {
    g.tables.push_back(factory.quantized_embedding_tensor_random(
    {num_embeddings, embedding_dim}, table_dtype,
    "group_quant_table", fp16_scale_bias_v));
    g.indices.push_back(factory.random_indices_tensor(
    {num_indices}, num_embeddings, indices_dtype));
    g.offsets.push_back(factory.random_offsets_tensor(
    {offsets_size}, num_indices, offsets_dtype,
    include_last_offset));
    g.weights.push_back(is_weights
                        ? factory.uniform_dist_tensor({num_indices},
                            data_type_t::f32, 2.0f)
                        : tensor_t{});
    g.outputs.push_back(factory.zero_tensor(
    {num_bags, embedding_dim}, output_dtype));
    g.outputs_ref.push_back(factory.zero_tensor(
    {num_bags, embedding_dim}, output_dtype));
  }
  return g;
}

GroupTensors build_group_embedding_tensors(
  tensor_factory_t &factory,
  size_t group_size,
  uint64_t num_embeddings, uint64_t embedding_dim,
  uint64_t num_indices,
  int64_t padding_index, bool is_weights, bool fp16_scale_bias_v,
  data_type_t table_dtype, data_type_t output_dtype,
  data_type_t indices_dtype) {
  GroupTensors g;
  g.tables.reserve(group_size);
  g.indices.reserve(group_size);
  g.offsets.reserve(group_size);
  g.weights.reserve(group_size);
  g.outputs.reserve(group_size);
  g.outputs_ref.reserve(group_size);
  g.algos.assign(group_size, embag_algo_t::none);
  g.padding_idxs.assign(group_size, padding_index);
  g.include_last_offsets.assign(group_size, false);
  g.fp16_scale_bias.assign(group_size, fp16_scale_bias_v);

  for (size_t i = 0; i < group_size; ++i) {
    g.tables.push_back(factory.uniform_dist_tensor(
    {num_embeddings, embedding_dim}, table_dtype, 2.0f));
    g.indices.push_back(factory.random_indices_tensor(
    {num_indices}, num_embeddings, indices_dtype));
    g.offsets.push_back(tensor_t{});
    g.weights.push_back(is_weights
                        ? factory.uniform_dist_tensor({num_indices},
                            data_type_t::f32, 2.0f)
                        : tensor_t{});
    g.outputs.push_back(factory.zero_tensor(
    {num_indices, embedding_dim}, output_dtype));
    g.outputs_ref.push_back(factory.zero_tensor(
    {num_indices, embedding_dim}, output_dtype));
  }
  return g;
}

GroupTensors build_group_embedding_quant_tensors(
  tensor_factory_t &factory,
  size_t group_size,
  uint64_t num_embeddings, uint64_t embedding_dim,
  uint64_t num_indices,
  int64_t padding_index, bool is_weights, bool fp16_scale_bias_v,
  data_type_t table_dtype, data_type_t output_dtype,
  data_type_t indices_dtype) {
  GroupTensors g;
  g.tables.reserve(group_size);
  g.indices.reserve(group_size);
  g.offsets.reserve(group_size);
  g.weights.reserve(group_size);
  g.outputs.reserve(group_size);
  g.outputs_ref.reserve(group_size);
  g.algos.assign(group_size, embag_algo_t::none);
  g.padding_idxs.assign(group_size, padding_index);
  g.include_last_offsets.assign(group_size, false);
  g.fp16_scale_bias.assign(group_size, fp16_scale_bias_v);

  for (size_t i = 0; i < group_size; ++i) {
    g.tables.push_back(factory.quantized_embedding_tensor_random(
    {num_embeddings, embedding_dim}, table_dtype,
    "group_quant_lookup_table", fp16_scale_bias_v));
    g.indices.push_back(factory.random_indices_tensor(
    {num_indices}, num_embeddings, indices_dtype));
    g.offsets.push_back(tensor_t{});
    g.weights.push_back(is_weights
                        ? factory.uniform_dist_tensor({num_indices},
                            data_type_t::f32, 2.0f)
                        : tensor_t{});
    g.outputs.push_back(factory.zero_tensor(
    {num_indices, embedding_dim}, output_dtype));
    g.outputs_ref.push_back(factory.zero_tensor(
    {num_indices, embedding_dim}, output_dtype));
  }
  return g;
}

void free_quant_tables(GroupTensors &g) {
  for (auto &t : g.tables) {
    if (t.check()) {
      free(t.get_raw_handle_unsafe());
      t.reset();
    }
  }
}

bool compare_group_outputs(GroupTensors &g, uint64_t output_rows,
                           uint64_t embedding_dim, float tol) {
  bool ok = true;
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    bool table_ok = true;
    compare_tensor_2D(g.outputs[i], g.outputs_ref[i],
                      output_rows, embedding_dim, tol, table_ok);
    if (!table_ok) {
      log_error("group embag mismatch at table ", i);
      ok = false;
    }
  }
  return ok;
}

// ── Kernel dispatch shims ───────────────────────────────────────────────────

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
  eb_thread_algo_t thread_algo) {
  try {
    const size_t num_tables = tables.size();
    if (num_tables == 0 ||
        indices.size() != num_tables ||
        offsets.size() != num_tables ||
        weights.size() != num_tables ||
        outputs.size() != num_tables ||
        algos.size() != num_tables ||
        padding_idxs.size() != num_tables ||
        include_last_offsets.size() != num_tables ||
        fp16_scale_bias.size() != num_tables) {
      log_error("group_embag_kernel_test: vector size mismatch");
      return status_t::failure;
    }

    std::vector<const void *> table_ptrs(num_tables);
    std::vector<const void *> indices_ptrs(num_tables);
    std::vector<const void *> offsets_ptrs(num_tables);
    std::vector<const float *> weights_ptrs(num_tables);
    std::vector<void *>       output_ptrs(num_tables);
    std::vector<embag_params_t> params_v(num_tables);

    for (size_t i = 0; i < num_tables; ++i) {
      if (!tables[i].check() || !indices[i].check() || !outputs[i].check()) {
        log_error("group_embag_kernel_test: invalid tensor at table ", i);
        return status_t::failure;
      }

      const bool lookup_mode = (algos[i] == embag_algo_t::none);
      if (!lookup_mode && !offsets[i].check()) {
        log_error("group_embag_kernel_test: offsets required at table ", i,
                  " for reduction algo");
        return status_t::failure;
      }

      table_ptrs[i]   = tables[i].get_raw_handle_unsafe();
      indices_ptrs[i] = indices[i].get_raw_handle_unsafe();
      offsets_ptrs[i] = lookup_mode ? nullptr
                        : offsets[i].get_raw_handle_unsafe();
      // Track presence of the weights tensor (the fixture has already
      // decided whether to allocate it based on EmbagType::is_weights).
      weights_ptrs[i] = weights[i].check()
                        ? static_cast<const float *>(weights[i].get_raw_handle_unsafe())
                        : nullptr;
      output_ptrs[i]  = outputs[i].get_raw_handle_unsafe();

      embag_params_t &p = params_v[i];
      p.dtypes.table   = tables[i].get_data_type();
      p.dtypes.output  = outputs[i].get_data_type();
      p.dtypes.indices = indices[i].get_data_type();
      p.dtypes.offsets = lookup_mode ? data_type_t::none
                         : offsets[i].get_data_type();
      p.algo            = algos[i];
      p.num_embeddings  = tables[i].get_size(0);
      p.embedding_dim   = tables[i].get_size(1);
      p.num_indices     = indices[i].get_size(0);
      p.num_bags        = lookup_mode ? 0 :
                          (include_last_offsets[i]
                           ? offsets[i].get_size(0) - 1
                           : offsets[i].get_size(0));
      p.is_weights          = (weights_ptrs[i] != nullptr);
      p.include_last_offset = include_last_offsets[i];
      p.padding_idx         = padding_idxs[i];
      p.fp16_scale_bias     = fp16_scale_bias[i];
      p.dst_stride          = outputs[i].get_stride()[0];
      // group_embedding_bag_direct reads num_threads from params[0]
      // only.  Leave it 0 (auto) so the fixture's
      // omp_set_num_threads() value drives execution.
      p.num_threads = 0;
    }

    scoped_thread_algo guard(thread_algo);

    return group_embedding_bag_direct(
             table_ptrs, indices_ptrs, offsets_ptrs, weights_ptrs,
             output_ptrs, params_v);
  }
  catch (const std::exception &e) {
    log_error("group_embag_kernel_test: ", e.what());
    return status_t::failure;
  }
}

status_t group_embag_forced_ref_kernel_test(
  std::vector<tensor_t> &tables,
  std::vector<tensor_t> &indices,
  std::vector<tensor_t> &offsets,
  std::vector<tensor_t> &weights,
  std::vector<tensor_t> &outputs,
  const std::vector<embag_algo_t> &algos,
  const std::vector<int64_t> &padding_idxs,
  const std::vector<bool> &include_last_offsets,
  const std::vector<bool> &fp16_scale_bias) {
  const size_t num_tables = tables.size();
  if (num_tables == 0 ||
      indices.size() != num_tables ||
      offsets.size() != num_tables ||
      weights.size() != num_tables ||
      outputs.size() != num_tables ||
      algos.size() != num_tables ||
      padding_idxs.size() != num_tables ||
      include_last_offsets.size() != num_tables ||
      fp16_scale_bias.size() != num_tables) {
    log_error("group_embag_forced_ref_kernel_test: vector size mismatch");
    return status_t::failure;
  }

  for (size_t i = 0; i < num_tables; ++i) {
    const bool lookup_mode = (algos[i] == embag_algo_t::none);
    const bool is_weights  = weights[i].check();
    status_t st = lookup_mode
                  ? embedding_forced_ref_kernel_test(
                    tables[i], indices[i], weights[i], outputs[i],
                    padding_idxs[i], is_weights, fp16_scale_bias[i])
                  : embag_forced_ref_kernel_test(
                    tables[i], indices[i], offsets[i], weights[i],
                    outputs[i], algos[i], padding_idxs[i],
                    include_last_offsets[i], is_weights,
                    fp16_scale_bias[i]);
    if (st != status_t::success) {
      return st;
    }
  }
  return status_t::success;
}
