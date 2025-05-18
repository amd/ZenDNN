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
#ifndef _HASH_OBJECT_HPP_
#define _HASH_OBJECT_HPP_

#include <string>
#include "zendnnl_global.hpp"
#include "hash_utils.hpp"

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;

/** @class hash_object_t
 *  @brief An abstract class for hashable objects.
 *
 * This class is an abstract parent class of all hashable objects. A hashable
 * object is an object that can create a unique hash based on its parameters
 * or member objects. This unique hash can be used to identify an object, cache
 * an object or compare objects. If a hashable object is a member of a class,
 * then its hash can also be used to combine with other member hashes to create
 * hash of an object of the class.
 *
 * @todo Revisit hash_utils to see that all hash_combine functions are robust.
 */
class hash_object_t {
public:
  /** @brief default destructor */
  virtual ~hash_object_t() = default;

  /** @brief default copy constructor
   *  @param other_ : other hash object
   */
  hash_object_t(const hash_object_t& other_) = default;

  /** @brief default copy assignment
   *  @param other_ : other hash object
   */
  hash_object_t& operator=(const hash_object_t& other_) = default;

  /** @brief move constructor
   *  @param other_ : other hash object
   */
  hash_object_t(hash_object_t&& other_);

  /** @brief move assignment */
  hash_object_t& operator=(hash_object_t&& other_);

  /** @brief Compare hash objects for equality
   *
   * Two hash objects are considered equal if their hash is nonzero and equal.
   */
  bool operator==(const hash_object_t& other_) const;

  /** @brief Compare hash objects for equality
   *
   * Two hash objects are considered equal if their hash is nonzero and equal.
   */
  bool operator!=(const hash_object_t& other_) const;

  /** @brief Set the last status of a hashable object.
   *
   * Many hashable objects like tensor_t and operator_t are created by chaining
   * creation apis. In such cases, last status being status_t::success denotes
   * successful object creation.
   *
   * In general last status other than status_t::success denotes some problem
   * with the object, like object creation failure.
   *
   * Default status of a hashable object is status_t::bad_hash_object.
   *
   * @param status_ : object status to set.
   */
  void                  set_last_status(status_t status_);

  /** @brief Reset the object
   *
   * Resets the object to make hash_key = 0, and status = status_t::bad_hash_object.
   */
  void                  reset();

  /** @brief Get the hash.
   * @return hash of the object.
   */
  std::size_t           get_hash() const;

  /** @brief Get the last status.
   * @sa set_last_status().
   * @return last status of the object.
   */
  status_t              get_last_status() const;

  /** @brief Check the last status.
   * @sa set_last_status()
   * @return True if the last status is status_t::success, else false.
   */
  bool                  check() const;

  /** @brief Compute hash of the object.
   *
   * Hash computation uses parameters and members of an object that make it unique.
   *
   * This is a pure virtual function and makes the class a pure virtual class.
   * @return computed hash of the object.
   */
  virtual std::size_t   hash() = 0;

protected:
  /** @brief Default constructor.
   *
   * Though this class is made pure virtual class by hash(), ZenDNNL follows the convension
   * of making constructors protected (or private), where a class need to serve as a
   * virtual base class, and no object of the class should be created.
   */
  hash_object_t();

  std::size_t   hash_key; /*!< Hash of the object. If hash is not generated its default
                            value is zero. */
  status_t      status; /*!< Last status of the object. */
};

} //common
} //zendnnl

#endif
