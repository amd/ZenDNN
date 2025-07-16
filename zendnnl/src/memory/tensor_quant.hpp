/********************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _TENSOR_QUANT_HPP_
#define _TENSOR_QUANT_HPP_

#include <cstdint>
#include <vector>
#include <string>
#include <optional>

#include "common/zendnnl_global.hpp"
#include "common/bfloat16.hpp"
#include "common/data_types.hpp"
#include "common/hash_object.hpp"
#include "memory/tensor_storage.hpp"
#include "memory/tensor_options.hpp"

namespace zendnnl {
namespace memory {

using namespace zendnnl::common;
using namespace zendnnl::error_handling;

using quant_storage_t = tensor_storage_t;

/** @enum quant_type_t
 *
 */
enum class quant_type_t : uint8_t {
  none,
  uniform,     /*!< uniform quantization */
  nonuniform   /*!< nonuniform quantization */
};

/** @enum quant_subtype_t
 *
 */
enum class quant_subtype_t : uint8_t {
  none,
  symmetric,     /*!< symmetric quantization */
  asymmetric     /*!< asymmetric quantization */
};

/** @enum quant_granularity_t
 *
 */
enum class quant_granularity_t : uint8_t {
  none,          /*!< unspecified */
  tensor,        /*!< per tensor */
  channel,       /*!< per channel */
  group          /*!< per group */
};

/** @class tensor_quant_t
 *  @brief A class to hold tensor quantization data.
 *
 * @sa tensor_t
 */
class tensor_quant_t final : public hash_object_t {
public:
  friend class tensor_t;
  friend class tensor_option_t;

public:
  /** @brief Parent type */
  using parent_type       = hash_object_t;
  using index_type        = tensor_option_t::index_type;
  using index_vec_type    = std::vector<index_type>;
  using storage_sptr_type = std::shared_ptr<quant_storage_t>;

public:
  /** @name Constructors, Destructors and Assignment
   */
  /**@{*/
  /** @brief Default constructor */
  tensor_quant_t();
  /**@}*/

  /** @name Reset and Hash
   */
  /**@{*/
  /** @brief Reset the object.
   *
   * Resets all quant data of a tensor. Used by
   * @c tensor_t::reset() to reset the tensor.
   */
  void        reset();

  /** @brief Generate hash
   *
   * Hash generated is used by @c tensor_t::hash() to generate
   * tensor hash.
   * @return Generated hash.
   */
  std::size_t hash() override;
  /**@}*/

private:
  quant_type_t                   type;
  quant_subtype_t                subtype;

  index_vec_type                 scale_size;
  index_vec_type                 scale_stride;
  index_vec_type                 scale_block_size;
  data_type_t                    scale_data_type;
  storage_sptr_type              scales;

  index_vec_type                 zero_size;
  index_vec_type                 zero_stride;
  index_vec_type                 zero_block_size;
  data_type_t                    zero_data_type;
  storage_sptr_type              zeros;
};

}//memory

namespace interface {
using quant_type_t        = zendnnl::memory::quant_type_t;
using quant_subtype_t     = zendnnl::memory::quant_subtype_t;
using quant_granularity_t = zendnnl::memory::quant_granularity_t;
} //export

}//zendnnl

#endif
