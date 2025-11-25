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
#ifndef _HASHABLE_OBJECT_TYPE_HPP_
#define _HASHABLE_OBJECT_TYPE_HPP_

#include "nlohmann/json.hpp"
#include "common/zendnnl_object.hpp"
#include "common/hash_utils.hpp"
#include "common/error_status.hpp"

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;
using json         = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

class hashable_object_t : public object_t {
public:
  using parent_type = object_t;

public:
  /** @brief Default constructor */
  hashable_object_t();

  /** @brief default destructor */
  virtual ~hashable_object_t() = default;

  /** @brief comparison equal */
  bool operator==(const hashable_object_t& other_) const;
  bool operator!=(const hashable_object_t& other_) const;

  /** @brief Set object info */
  virtual void set_object_info() override;

  /** @brief Set object runtime info */
  virtual void set_object_runtime_info() override;

  /** @brief get the hash */
  std::size_t get_hash() const;

  /** @brief create the object */
  virtual void create();

protected:
  /** @brief reset the object */
  void reset() override;

  /** @brief Compute hash of the object */
  virtual std::size_t hash();

protected:
  std::size_t  hash_key;       /*!< Object hash with default as zero. */
};

} //common
} //zendnnl

#endif
