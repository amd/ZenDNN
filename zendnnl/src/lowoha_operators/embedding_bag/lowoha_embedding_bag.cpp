/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_embedding_bag.hpp"
#include "dispatch_kernel.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

#include <sstream>
#include <vector>

namespace zendnnl {
namespace lowoha {
namespace embag {

using namespace ::zendnnl::common;
using namespace ::zendnnl::profile;

status_t embedding_bag_direct(
  const void *table,
  const void *indices,
  const void *offsets,
  const float *weights,
  void *dst,
  embag_params_t params) {
  // Create profiler instance for timing
  profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile) {
    profiler.tbp_start();
  }

  if (validate_embag_inputs(table, indices, dst, params) != status_t::success) {
    return status_t::failure;
  }

  if (params.algo != embag_algo_t::none && offsets == nullptr) {
    log_error("embedding_bag_direct: offsets required for reduction operations");
    return status_t::failure;
  }

  const int num_threads = (params.num_threads > 0)
                          ? static_cast<int>(params.num_threads)
                          : omp_get_max_threads();

  embag_threadlimit thread_guard(num_threads);

  // Dispatch to the appropriate kernel
  dispatch_avx512_kernel(table, indices, offsets, weights, dst, params);

  if (is_profile) {
    profiler.tbp_stop();
  }

  if (apilog_info_enabled() || is_profile) {
    [[maybe_unused]] std::ostringstream ss;
    ss << "LOWOHA embedding_bag_direct: "
       << "num_embeddings=" << params.num_embeddings
       << ", embedding_dim=" << params.embedding_dim
       << ", num_indices=" << params.num_indices
       << ", num_bags=" << params.num_bags
       << ", algo=" << algo_to_string(params.algo)
       << ", table_dtype=" << dtype_to_string(params.dtypes.table)
       << ", output_dtype=" << dtype_to_string(params.dtypes.output)
       << ", num_threads=" << num_threads;
    apilog_info(ss.str());
    if (is_profile) {
      profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(),
                         profiler.get_res_str());
    }
  }
  return status_t::success;
}

// Embedding bag direct implementation
status_t embedding_direct(
  const void *table,
  const void *indices,
  const float *weights,
  void *dst,
  embag_params_t params) {

  params.algo = embag_algo_t::none;
  return embedding_bag_direct(table, indices, nullptr, weights, dst, params);
}

// Group embedding bag direct implementation
status_t group_embedding_bag_direct(
  const std::vector<const void *> &tables,
  const std::vector<const void *> &indices,
  const std::vector<const void *> &offsets,
  const std::vector<const float *> &weights,
  const std::vector<void *> &dsts,
  const std::vector<embag_params_t> &params) {

  // Create profiler instance for timing
  profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile) {
    profiler.tbp_start();
  }

  const int num_tables = static_cast<int>(tables.size());

  // Validate that all vectors have the same size
  if (indices.size() != static_cast<size_t>(num_tables) ||
      offsets.size() != static_cast<size_t>(num_tables) ||
      weights.size() != static_cast<size_t>(num_tables) ||
      dsts.size() != static_cast<size_t>(num_tables) ||
      params.size() != static_cast<size_t>(num_tables)) {
    log_error("group_embedding_bag_direct: all input vectors must have the same size");
    return status_t::failure;
  }

  // Read environment configuration
  using namespace zendnnl::ops;
  embag_config_t &embag_config = embag_config_t::instance();
  embag_config.set_env_config();

  unsigned int eb_thread_qty = params[0].num_threads > 0 ? params[0].num_threads :
                               omp_get_max_threads();
  eb_thread_algo_t thread_algo = thread_algo_select();
  const char *thread_type = thread_algo_to_string(thread_algo);

  // Make a mutable copy of params for dispatch
  std::vector<embag_params_t> mutable_params = params;

  // Thread algorithm dispatch
  if (thread_algo == eb_thread_algo_t::ccd_threaded) {
    // CCD-aware threading with nested parallelism
    omp_set_max_active_levels(2);
    int ccd_num_threads = CCD_NUM_THREADS;
    unsigned int outer_threads = (eb_thread_qty % ccd_num_threads) == 0 ?
                                 eb_thread_qty / ccd_num_threads :
                                 ((eb_thread_qty / ccd_num_threads) + 1);
    unsigned int rem = (eb_thread_qty % ccd_num_threads) == 0 ?
                       ccd_num_threads :
                       eb_thread_qty % ccd_num_threads;
    unsigned int loopCount = (num_tables % outer_threads) == 0 ?
                             num_tables / outer_threads :
                             ((num_tables / outer_threads) + 1);

    #pragma omp parallel num_threads(outer_threads)
    {
      unsigned int inner_threads = ccd_num_threads;
      unsigned int thid = omp_get_thread_num();
      if (thid == outer_threads - 1) {
        inner_threads = rem;
      }

      for (unsigned int i = 0; i < loopCount; i++) {
        int threadOffset = thid + (i * outer_threads);
        if (threadOffset >= num_tables) {
          break;
        }

        embag_threadlimit thread_guard(inner_threads);
        dispatch_avx512_kernel(
          tables[threadOffset], indices[threadOffset], offsets[threadOffset],
          weights[threadOffset], dsts[threadOffset], mutable_params[threadOffset]);
      }
    }
  }
  else if (num_tables < static_cast<int>(eb_thread_qty) &&
           thread_algo == eb_thread_algo_t::hybrid_threaded) {
    // Hybrid threading when tables < threads
    unsigned int outer_threads = num_tables;
    unsigned int rem = eb_thread_qty % num_tables;

    #pragma omp parallel num_threads(outer_threads)
    {
      unsigned int inner_threads = eb_thread_qty / num_tables;
      unsigned int threadOffset = omp_get_thread_num();
      if (threadOffset < rem) {
        inner_threads++;
      }

      embag_threadlimit thread_guard(inner_threads);
      dispatch_avx512_kernel(
        tables[threadOffset], indices[threadOffset], offsets[threadOffset],
        weights[threadOffset], dsts[threadOffset], mutable_params[threadOffset]);
    }
  }
  else if (thread_algo == eb_thread_algo_t::table_threaded) {
    // Thread-per-table parallelism
    unsigned int loopCount = (num_tables % eb_thread_qty) == 0 ?
                             num_tables / eb_thread_qty :
                             ((num_tables / eb_thread_qty) + 1);

    #pragma omp parallel num_threads(eb_thread_qty)
    {
      for (unsigned int i = 0; i < loopCount; i++) {
        int threadOffset = omp_get_thread_num() + (i * eb_thread_qty);
        if (threadOffset >= num_tables) {
          break;
        }

        dispatch_avx512_kernel(
          tables[threadOffset], indices[threadOffset], offsets[threadOffset],
          weights[threadOffset], dsts[threadOffset], mutable_params[threadOffset]);
      }
    }
  }

  else {
    // Default: batch_threaded - Sequential tables with batch-level threading
    embag_threadlimit thread_guard(eb_thread_qty);
    for (int i = 0; i < num_tables; i++) {
      dispatch_avx512_kernel(
        tables[i], indices[i], offsets[i],
        weights[i], dsts[i], mutable_params[i]);
    }
  }

  if (is_profile) {
    profiler.tbp_stop();
  }

  if (apilog_info_enabled() || is_profile) {
    [[maybe_unused]] std::ostringstream ss;
    ss << "LOWOHA group_embedding_bag_direct: "
       << "num_tables=" << num_tables
       << ", eb_thread_qty=" << eb_thread_qty
       << ", thread_algo=" << thread_type;
    apilog_info(ss.str());
    if (is_profile) {
      profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(),
                         profiler.get_res_str());
    }
  }

  return status_t::success;
}

} // namespace embag
} // namespace lowoha
} // namespace zendnnl
