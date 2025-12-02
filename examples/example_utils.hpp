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
#ifndef _EXAMPLE_UTILS_HPP_
#define _EXAMPLE_UTILS_HPP_

#include <vector>
#include <cstring>
#include <random>
#include <algorithm>
#include <variant>

#include "zendnnl.hpp"

namespace zendnnl {
/** @namespace zendnnl::examples
 *  @brief A namespace that contains examples of how to use ZenDNNL.
 */
namespace examples {
using namespace zendnnl::interface;
using StorageParam = std::variant<std::pair<size_t, void *>, tensor_t>;

/** @class tensor_factory_t
 * @brief Quick generation of predefined tensors.
 */
class tensor_factory_t {
 public:
  /** @brief Index type */
  using index_type = tensor_t::index_type;
  using data_type  = common::data_type_t;

  /** @brief zero tensor */
  tensor_t zero_tensor(const std::vector<index_type> size_, data_type dtype_,
                       std::string tensor_name_="zero", tensor_t scale = tensor_t(),
                       tensor_t zp = tensor_t());

  /** @brief uniform tensor */
  tensor_t uniform_tensor(const std::vector<index_type> size_, data_type dtype_,
                          float val_, std::string tensor_name_="uniform", tensor_t scale = tensor_t(),
                          tensor_t zp = tensor_t());

  /** @brief broadcasted uniform tensor */
  tensor_t broadcast_uniform_tensor(const std::vector<index_type> size_,
                                    const std::vector<index_type> stride_, data_type dtype_, float val_,
                                    std::string tensor_name_="broadcasted uniform", tensor_t scale = tensor_t(),
                                    tensor_t zp = tensor_t());

  /** @brief non-uniform tensor */
  tensor_t non_uniform_tensor(const std::vector<index_type> size_,
                              data_type dtype_,
                              std::vector<int64_t> val_, std::string tensor_name_="non_uniform",
                              tensor_t scale = tensor_t(), tensor_t zp = tensor_t());

  /** @brief uniform distributed tensor */
  tensor_t uniform_dist_tensor(const std::vector<index_type> size_,
                               data_type dtype_, float range_,
                               std::string tensor_name_="uniform dist", bool trans = false,
                               tensor_t scale = tensor_t(),
                               tensor_t zp = tensor_t());

  /** @brief uniform distributed strided tensor */
  tensor_t uniform_dist_strided_tensor(const std::vector<index_type> size_,
                                       const std::vector<index_type> stride_,
                                       data_type dtype_, float range_,
                                       std::string tensor_name_="strided uniform dist", tensor_t scale = tensor_t(),
                                       tensor_t zp = tensor_t());

  /** @brief blocked tensor */
  tensor_t blocked_tensor(const std::vector<index_type> size_, data_type dtype_,
                          float range_, std::string tensor_name_="blocked",
                          tensor_t scale = tensor_t(), tensor_t zp = tensor_t());

  /** @brief copy tensor */
  tensor_t copy_tensor(const std::vector<index_type> size_, data_type dtype_,
                       StorageParam param, bool trans, bool is_blocked,
                       std::string tensor_name_="copy", tensor_t scale = tensor_t(),
                       tensor_t zp = tensor_t());

  /** @brief Generate random indices tensor with optional padding index */
  tensor_t random_indices_tensor(const std::vector<index_type> size_,
                                 uint64_t num_embeddings);

  /** @brief Generate random offsets tensor for bag boundaries */
  tensor_t random_offsets_tensor(const std::vector<index_type> size_,
                                 uint64_t num_indices, bool include_last_offset = true);

  /** @brief quantized embedding tensor random */
  tensor_t quantized_embedding_tensor_random(const std::vector<index_type> size_,
      data_type dtype_, std::string tensor_name_="quant random",
      bool fp16_scale_bias = true, float scale_min = 0.10,
      float scale_max = 0.19, int8_t zp_min = 0, int8_t zp_max = 7);
};

/** @class tensor_functions
 * @brief Quick generation of predefined tensors.
 */
class tensor_functions_t {
 public:
  void tensor_pretty_print(const tensor_t &tensor_);
};

/** @fn get_aligned_size
 *  @brief Function to align the given size_ according to the alignment
 */
size_t get_aligned_size(size_t alignment, size_t size_);

} //examples
} //zendnnl

#endif
