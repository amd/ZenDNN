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
#ifndef _MATMUL_DLP_UTILS_HPP_
#define _MATMUL_DLP_UTILS_HPP_

#include <vector>
#include <map>
#include <memory>
#include <optional>

#include "common/zendnnl_global.hpp"
#include "memory/tensor.hpp"
#include "memory/memory_utils.hpp"
#include "operators/common/post_op.hpp"
#include "memory/memory_utils.hpp"
#include "aocl_dlp.h" // aocl-dlp header
namespace zendnnl {
namespace ops {
using namespace zendnnl::memory;
/** @class aocl_dlp_utils_t
 *  @brief reordering to blocked format for @c tensor_t and setting AOCL DLP post-ops.
 *
 * Reorders the tensor to blocked format according to data type.
 * Sets AOCL DLP post-ops for MatMul using the new dlp_metadata_t structure
 */
class aocl_dlp_utils_t {
 public:
  aocl_dlp_utils_t();
  ~aocl_dlp_utils_t();
  using tensor_map_type = std::map<std::string, tensor_t>;
  /** @brief function pointer type for getting the reorder buffer size */
  using get_reorder_buff_size_func_ptr = long unsigned int (*)(const char,
                                         const char,
                                         const char, const md_t,
                                         const md_t, dlp_metadata_t *);
  /** @brief template function pointer type for reordering */
  template <typename T>
  using reorder_func_ptr = void (*)(const char, const char, const char, const T *,
                                    T *,
                                    const md_t, const md_t, const md_t, dlp_metadata_t *);
  /** @brief entry function for tensor reordering for the AOCL */
  status_t      reorder_weights(std::optional<tensor_t> weights, data_type_t src_dt);
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

  /** @brief compute zero-point compensation for INT8 */
  void          zero_point_compensation(int M, int N, int K, tensor_t &src,
                                        tensor_t &wei, int32_t src_zero_point,
                                        int32_t wei_zero_point);
  /** @brief allocate memory for the AOCL DLP post-ops */
  status_t      aocl_post_op_memory_alloc(const std::vector<post_op_t>
                                          &post_op_vec_,
                                          bool is_bias, std::map<std::string, zendnnl::memory::tensor_t> &inputs_);
  /** @brief initialize the post-ops */
  status_t      aocl_post_op_initialize(const std::vector<post_op_t> &post_op_vec_,
                                        int &post_op_count, bool is_bias,
                                        std::map<std::string, zendnnl::memory::tensor_t> &inputs_,
                                        zendnnl::memory::tensor_t &output_tensor,
                                        size_t eltwise_index, size_t add_index_2d, size_t mul_index_1d,
                                        size_t mul_index_2d);
  /** @brief allocate aocl post op */
  status_t      alloc_post_op(const std::vector<post_op_t> &post_op_vec_,
                              std::optional<tensor_t> optional_bias_tensor_,
                              tensor_t &weight_tensor,
                              std::map<std::string, zendnnl::memory::tensor_t> &inputs_,
                              zendnnl::memory::tensor_t &output_tensor);
  /** @brief free aocl post op */
  void          free_post_op();
  /** @brief sets runtime post-op buffers in dlp_metadata_t */
  status_t      set_runtime_post_op_buffer(tensor_map_type &inputs, bool is_bias,
      tensor_t &output_tensor);
  /** @brief get the post op pointer */
  dlp_metadata_t *get_aocl_dlp_post_op_ptr_unsafe() const;
  /** @brief get the reordered weights pointer*/
  void         *get_aocl_dlp_reordered_weights_ptr_unsafe() const;
 protected:
  std::map<std::string, uint32_t> post_op_size;
  unsigned int zp_comp_ndim;
  int32_t dummy_zp;
  float dummy_scale;
  int32_t *zp_comp_acc;
  dlp_metadata_t *aocl_dlp_po_ptr;  // Changed from aocl_post_op to dlp_metadata_t
  void *reordered_weights_ptr;
};
} // namespace ops
} // namespace zendnnl
#endif