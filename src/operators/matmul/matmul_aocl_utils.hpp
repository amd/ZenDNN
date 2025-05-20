/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _MATMUL_AOCL_CONTEXT_HPP_
#define _MATMUL_AOCL_CONTEXT_HPP_

#include <vector>
#include <map>
#include <memory>
#include <optional>

#include "common/zendnnl_global.hpp"
#include "memory/tensor.hpp"
#include "operators/common/post_op.hpp"

#include "blis.h"

namespace zendnnl {
namespace ops {

using namespace zendnnl::memory;

/** @class aocl_utils_t
 *  @brief reordering to blocked format for @c tensor_t and setting AOCL post-ops.
 *
 * Reorders the tensor to blocked format according to data type.
 * Sets AOCL post-ops for MatMul
 */
class aocl_utils_t {
public:
  aocl_utils_t();
  ~aocl_utils_t();

  using tensor_map_type = std::map<std::string, tensor_t>;

  /** @brief function pointer type for getting the reorder buffer size */
  using get_reorder_buff_size_func_ptr = long unsigned int (*)(const char, const char,
                                                               const char, const dim_t,
                                                               const dim_t);

  /** @brief template function pointer type for reordering */
  template <typename T>
  using reorder_func_ptr = void (*)(const char, const char, const char, const T *, T *,
                               const dim_t, const dim_t, const dim_t);

  /** @brief entry function for tensor reordering for the AOCL */
  status_t      reorder_weights(std::optional<tensor_t> weights);

  /** @brief weight reordering for the AOCL */
  template <typename T>
  size_t        reorder_weights_execute(
                 const void *weights,
                 const int k,
                 const int n,
                 const int ldb,
                 const char order,
                 const char trans,
                 get_reorder_buff_size_func_ptr get_reorder_buf_size,
                 reorder_func_ptr<T> reorder_func);

  /** @brief allocate memory for the AOCL post-ops */
  status_t      aocl_post_op_memory_alloc(const std::vector<post_op_t> post_op_vec_,
                                          bool is_bias);

  /** @brief initialize the post-ops */
  status_t      aocl_post_op_initialize(const std::vector<post_op_t> post_op_vec_,
                                        int &post_op_count);

  /** @brief allocate aocl post op */
  status_t      alloc_post_op(const std::vector<post_op_t> post_op_vec_,
                              std::optional<tensor_t> optional_bias_tensor_);

  /** @brief free aocl post op */
  void          free_post_op();

  /** @brief sets runtime post-op buffers in aocl_po_ptr */
  status_t      set_runtime_post_op_buffer(tensor_map_type &inputs);

  /** @brief get the post op pointer */
  aocl_post_op* get_aocl_post_op_ptr_unsafe() const;

  /** @brief get the reordered weights pointer*/
  void*         get_aocl_reordered_weights_ptr_unsafe() const;

protected:
  std::map<std::string, uint32_t> post_op_size;
  aocl_post_op* aocl_po_ptr;
  void* reordered_weights_ptr;
};

} // namespace ops
} // namespace zendnnl
#endif
