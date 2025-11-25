/********************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _OBJECT_TYPE_HPP_
#define _OBJECT_TYPE_HPP_

#include <type_traits>
#include <string>

#include "nlohmann/json.hpp"
#include "common/hash_utils.hpp"
#include "common/error_status.hpp"
#include "common/zendnnl_exceptions.hpp"
#include "common/zendnnl_global.hpp"

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;
using json         = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

/** @class object_t
 *  @brief An abstract class for named objects.
 *
 * This class is an abstract parent class of all named objects. A named object
 * has a name and is capable of having object information for profiling and diagnostic
 * purposes. This class some basic functionalities for diagnostics.
 * These functionalities are
 *
 * 1. This class provides a name string, that can be used to give any object a name.
 *    This name can be used to identify the object in profiling and dignostics.
 * 2. This class provides a json object which contains all object information for
 *    diagnostic and profiling purposes.
 * 3. This class also provides object summary and object runtime summary for short
 *    log messages.
 * 4. Since many objects are created using creation api chaining, status of their
 *    creation can be checked using get_last_status.
 * @todo Revisit hash_utils to see that all hash_combine functions are robust.
 */

class object_t {
public:
  /** @brief A string for unknown object name */
  static const std::string unknown_object_str;

public:
  /** @brief default destructor */
  virtual ~object_t() = default;

  /** @brief default copy constructor
   *  @param other_ : other hash object
   */
  object_t(const object_t& other_) = default;

  /** @brief default copy assignment
   *  @param other_ : other hash object
   */
  object_t& operator=(const object_t& other_) = default;

  /** @brief move constructor
   *  @param other_ : other hash object
   */
  object_t(object_t&& other_) = default;

  /** @brief move assignment */
  object_t& operator=(object_t&& other_) = default;

  /** @brief Get the object name. */
  void  set_name(std::string name_);

  /** @brief Get the object name.
   * @sa set_name().
   * @return object name string.
   */
  std::string  get_name() const;

  /** @brief Get the object information.
   * @return Object information json.
   */
  nlohmann::ordered_json  get_object_info() const;

  /** @brief Get the object runtime information.
   * @return Object runtime information json.
   */
  nlohmann::ordered_json  get_object_runtime_info() const;

  /** @brief Get the object summary information.
   * @return Object information string.
   */
  virtual std::string  get_object_summary() const;

  /** @brief Get the object runtime summary information.
   * @return Object runtime information string.
   */
  virtual std::string  get_object_runtime_summary() const;

  /** @brief Get the last status.
   * @sa set_last_status().
   * @return last status of the object.
   */
  status_t  get_last_status() const;

  /** @brief Check if it is a bad object
   * @sa set_last_status()
   * @return True if the last status is not status_t::success, else false.
   */
  bool is_bad_object() const;

  /** @brief Check if it is a bad object
   * @sa set_last_status()
   * @return True if the last status is not status_t::success, else false.
   */
  bool is_unnamed_object() const;

protected:
  /** @brief Default constructor.
   *
   * Though this class is made pure virtual class by hash(), ZenDNNL follows the convension
   * of making constructors protected (or private), where a class need to serve as a
   * virtual base class, and no object of the class should be created.
   */
  object_t();

  /** @brief Set the last status of a zendnnl object.
   *
   * Many objects like tensor_t and operator_t are created by chaining
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
  void set_last_status(status_t status_);

  /** @brief Set object info
   *
   * Sets object info.
   */
  virtual void set_object_info();

  /** @brief Set object info
   *
   * Sets object info.
   */
  virtual void set_object_runtime_info();

  /** @brief Reset the object
   *
   * Resets the object to make hash_key = 0, and status = status_t::bad_hash_object.
   */
  virtual void reset();

  status_t                status;         /*!< Last status of the object. */
  std::string             obj_name;       /*!< Object name */
  nlohmann::ordered_json  obj_info_json;  /*!< Object json */
  nlohmann::ordered_json  obj_runtime_info_json;  /*!< Object json */
};

} //common
} //zendnnl

#endif
