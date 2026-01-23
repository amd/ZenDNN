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

#ifndef _LOWOHA_EMBEDDING_BAG_HPP
#define _LOWOHA_EMBEDDING_BAG_HPP

#include <omp.h>
#include <vector>
#include "lowoha_embag_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace embag {

/**
 * @brief RAII helper to temporarily set OpenMP thread count.
 *        Automatically restores original thread count on scope exit.
 *
 * @example
 *   {
 *       embag_threadlimit guard(4);   // Set to 4 threads
 *       // ... parallel work ...
 *   }  // Restored to original
 */
struct embag_threadlimit {
  int old_num_threads;
  bool is_modified;

  embag_threadlimit(int num_threads) : old_num_threads(0), is_modified(false) {
    if (num_threads != omp_get_max_threads()) {
      old_num_threads = omp_get_max_threads();
      omp_set_num_threads(num_threads);
      is_modified = true;
    }
  }

  ~embag_threadlimit() {
    if (is_modified) {
      omp_set_num_threads(old_num_threads);
    }
  }
};

/**
 * @brief Direct API for embedding bag operation
 *
 * Performs embedding bag lookup and reduction operation on the given embedding table.
 * Given indices and offsets, looks up embeddings from the table and applies
 * the specified reduction algorithm (sum, mean, max) for each bag.
 *
 * @param table           Pointer to embedding table data [num_embeddings x embedding_dim]
 * @param indices         Pointer to indices array (int32 or int64)
 * @param offsets         Pointer to offsets array (int32 or int64), can be nullptr for
 *                        embedding lookup (no reduction)
 * @param weights         Pointer to (optional)weights array (float)
 * @param dst             Pointer to output buffer [num_bags x embedding_dim]
 * @param params          Embedding bag parameters (dtypes, algo, dimensions, etc.)
 *
 * @return status_t::success on successful execution, status_t::failure otherwise
 *
 * @par Example Usage:
 * @code
 *   // Setup parameters
 *   embag_params_t params;
 *   params.dtypes.table = data_type_t::f32;
 *   params.dtypes.output = data_type_t::f32;
 *   params.algo = embag_algo_t::sum;
 *   params.num_embeddings = 1000;
 *   params.embedding_dim = 128;
 *   params.num_indices = 50;
 *   params.num_bags = 10;
 *   params.dtypes.indices = data_type_t::s64;
 *   params.dtypes.offsets = data_type_t::s64;
 *
 *   // Execute
 *   status_t status = embedding_bag_direct(
 *       table_ptr, indices_ptr, offsets_ptr, output_ptr, params);
 * @endcode
 */
zendnnl::common::status_t embedding_bag_direct(
  const void *table,
  const void *indices,
  const void *offsets,
  const float *weights,
  void *dst,
  embag_params_t params);

/**
 * @brief Simplified direct API for embedding lookup (no reduction)
 *
 * Performs simple embedding lookup without any reduction operation.
 * Each index maps directly to one output row.
 *
 * @param table           Pointer to embedding table data [num_embeddings x embedding_dim]
 * @param indices         Pointer to indices array (int32 or int64)
 * @param weights         Pointer to (optional)weights array (float)
 * @param dst             Pointer to output buffer [num_indices x embedding_dim]
 * @param params          Embedding parameters (dtypes, dimensions, etc.)
 *
 * @return status_t::success on successful execution, status_t::failure otherwise
 */
zendnnl::common::status_t embedding_direct(
  const void *table,
  const void *indices,
  const float *weights,
  void *dst,
  embag_params_t params);

/**
 * @brief Direct API for group embedding bag operation
 *
 * Performs multiple embedding bag operations in a single call. Each operation
 * performs embedding bag lookup and reduction on its corresponding embedding table.
 * This is useful for batching multiple embedding bag operations together.
 *
 * @param tables          Vector of pointers to embedding table data
 * @param indices         Vector of pointers to indices arrays (int32 or int64)
 * @param offsets         Vector of pointers to offsets arrays (int32 or int64)
 * @param weights         Vector of pointers to (optional) weights arrays (float)
 * @param dsts            Vector of pointers to output buffers
 * @param params          Vector of embedding bag parameters for each operation
 *
 * @return status_t::success if all operations succeed, status_t::failure otherwise
 *
 * @note All vectors must have the same size. Each index i in the vectors
 *       corresponds to one embedding bag operation.
 *
 * @par Example Usage:
 * @code
 *   std::vector<const void*> tables = {table1, table2};
 *   std::vector<const void*> indices = {idx1, idx2};
 *   std::vector<const void*> offsets = {off1, off2};
 *   std::vector<const float*> weights = {nullptr, nullptr};
 *   std::vector<void*> outputs = {out1, out2};
 *   std::vector<embag_params_t> params = {params1, params2};
 *
 *   status_t status = group_embedding_bag_direct(
 *       tables, indices, offsets, weights, outputs, params);
 * @endcode
 */
zendnnl::common::status_t group_embedding_bag_direct(
  const std::vector<const void*> &tables,
  const std::vector<const void*> &indices,
  const std::vector<const void*> &offsets,
  const std::vector<const float*> &weights,
  const std::vector<void*> &dsts,
  const std::vector<embag_params_t> &params);

} // namespace embag
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_EMBEDDING_BAG_HPP
