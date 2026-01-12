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

#include <sstream>

namespace zendnnl {
namespace lowoha {
namespace embag {

using namespace zendnnl::common;
using namespace zendnnl::profile;

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

  [[maybe_unused]] std::ostringstream ss;
  if (apilog_info_enabled()) {
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
  }
  dispatch_avx512_kernel(table, indices, offsets, weights, dst, params);

  if (is_profile) {
    profiler.tbp_stop();
    profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(),
                       profiler.get_res_str());
  }
  return status_t::success;
}

status_t embedding_direct(
  const void *table,
  const void *indices,
  const float *weights,
  void *dst,
  embag_params_t params) {

  params.algo = embag_algo_t::none;
  return embedding_bag_direct(table, indices, nullptr, weights, dst, params);
}

} // namespace embag
} // namespace lowoha
} // namespace zendnnl
