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
#ifndef _TENSOR_STORAGE_HPP_
#define _TENSOR_STORAGE_HPP_

#include <cstdint>
#include <cstdlib>
#include <string>

#include "common/zendnnl_global.hpp"
#include "common/hash_object.hpp"

namespace zendnnl {
namespace memory {

using namespace zendnnl::common;
using namespace zendnnl::error_handling;

/** @class tensor_storage_t
 *  @brief A class to hold memory buffer for tensor data.
 *
 * This class holds memory buffer for tensor data and other information
 * (like buffer size) related to the memory buffer.
 *
 * Memory buffer can either be allocated, including aligned allocation, or
 * it can be borrowed from a deep learning framework. In case of allocated
 * buffer this class take care of releasing it if it is no longer referenced
 * by any tensor.
 *
 * @sa tensor_t, hash_object_t
 */
class tensor_storage_t final : public hash_object_t {
  friend class tensor_t;

public:
  /** @brief Parent type */
  using parent_type = hash_object_t;

  /** @name Constructors, Destructors and Assignment
   */
  /**@{*/
  /** @brief Default constuctor */
  tensor_storage_t();

  /** @brief Releases allocated memory */
  ~tensor_storage_t();
  /**@}*/

  /** @name Create, Reset and Hash
   */
  /**@{*/
  /** @brief Generate hash
   *
   * Hash generated is used by @c tensor_t::hash() to generate
   * tensor hash.
   * @return Generated hash.
   */
  std::size_t   hash() override;

  /** @brief Reset the object.
   *
   * Resets storage. Used by @c tensor_t::reset().
   */
  void          reset();
  /**@}*/

private:

  /** @name Memory Management
   */
  /**{*/
  /** @brief Allocate memory buffer for tensor data.
   *
   * Throws exception if memory could not be allocated.
   * @param size_ : buffer size in bytes.
   */
  void          allocate(std::size_t size_);

  /** @brief Borrow a memory buffer from a deep learning framework.
   *
   * @param ptr_ : pointer to the memory buffer.
   * @param size_ : buffer size.
   */
  void          set_raw_handle(void* ptr_, std::size_t size_);

  /** @brief Get a raw pointer to the memory buffer.
   *
   * This is potentially unsafe.
   * @return a void pointer to the memory buffer.
   */
  void*         get_raw_handle();
  /**}*/

private:
  /* @name Private Variables
   */
  /*{*/
  bool         allocated; /**< Whether memory buffer is allocated or borrowed
                             This is used to decide whether to release memory or not.*/
  uint32_t     aligned_to; /**< Memory boundary the buffer is aligned to. Zero
                            represents unaligned allocation. */
  std::size_t  size; /**< Buffer size in bytes. In case of borrowed buffer this is
                        given by the framework, and ZenDNNL trusts the size given. */
  void*        raw_ptr; /**< Raw pointer to the memory buffer */
  /*}*/
};

} //memory
} //zendnnl
#endif
