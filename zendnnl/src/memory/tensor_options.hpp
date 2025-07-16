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
#ifndef _TENSOR_OPTIONS_HPP_
#define _TENSOR_OPTIONS_HPP_

#include <cstdint>
#include <vector>
#include <string>

#include "common/zendnnl_global.hpp"
#include "common/bfloat16.hpp"
#include "common/data_types.hpp"
#include "common/hash_object.hpp"

namespace zendnnl {
namespace memory {

using namespace zendnnl::common;
using namespace zendnnl::error_handling;

/** @enum tensor_layout_t
 *  @brief Enumeration of tensor data layout in the tensor storage.
 *
 * Tensor layout refers to how tensor data is layed out in
 * the tensor memory (contiguous, blocked or strided etc.).
 */
enum class tensor_layout_t : uint8_t {
  contiguous, /*!< Contiguous layout */
  aligned,    /*!< Memory aligned layout */
  blocked,    /*!< Blocked layout */
  oblique     /*!< Oblique layout */
};

/** @class tensor_option_t
 *  @brief A class to hold tensor meta data.
 *
 * This class consists of all tensor meta data like size, data type, tensor format etc.
 * It is used as a member of tensor_t class.
 *
 * Tensor meta data is set by tensor_t functions prefixed by "set_".
 *
 * @sa tensor_t, tensor_format_t.
 */
class tensor_option_t final : public hash_object_t {
  friend class tensor_t;

  /** @brief Parent type */
  using parent_type = hash_object_t;

  /** @brief Index type */
  using index_type = uint64_t;

  /** @brief Index vector type */
  using index_vec_type = std::vector<index_type>;

public:
  /** @name Constructors, Destructors and Assignment
   */
  /**@{*/
  /** @brief Default constructor */
  tensor_option_t();
  /**@}*/

  /** @name Reset and Hash
   */
  /**@{*/
  /** @brief Reset the object.
   *
   * Resets all meta data. Used by @c tensor_t::reset().
   */
  void        reset();

  /** @brief Genarate hash value.
   *
   * Hash generated is used by @c tensor_t::hash() to generate tensor hash.
   * @return Generated hash.
   */
  std::size_t hash() override;
  /**@}*/

private:
  index_vec_type     size;           /**< Tensor size. Tensor dimensions are
                                        decided by length of size vector. */
  index_vec_type     aligned_size;   /**< Tensor aligned size. */
  index_vec_type     stride;         /**< Tensor stride that defines access
                                        pattern */
  index_vec_type     base;           /**< Index of the element to be consider
                                        first element. */
  uint64_t           nelem;          /**< Number of elements, computed from
                                        size */
  uint64_t           aligned_nelem;  /**< Memory buffer size in terms of
                                        elements computed from aligned_size */
  uint64_t           base_offset;    /**< Base offset, computed from @c base
                                        and @c stride */
  data_type_t        data_type;      /**< Tensor data type */
  tensor_layout_t    layout;         /**< Tensor layout */
  bool               is_const;       /**< Tensor constness */
  std::string        order;          /**< Tensor channel order(for example
                                        NCHW or NHCW) */
};

/** @class tensor_quant_t
 *  @brief A class to hold tensor quantization data.
 *
 * @sa tensor_t
 */
class tensor_quant_t final : public hash_object_t {
  friend class tensor_t;

  /** @brief Parent type */
  using parent_type = hash_object_t;
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
  float    zero_point; /**< Zero point of quantization. */
  float    scale; /**< Quantization scale. */
};

} //memory

namespace interface {
using tensor_layout_t = zendnnl::memory::tensor_layout_t;
// using tensor_option_t = zendnnl::memory::tensor_option_t;
// using tensor_quant_t  = zendnnl::memory::tensor_quant_t;
} //export

} //zendnnl
#endif
