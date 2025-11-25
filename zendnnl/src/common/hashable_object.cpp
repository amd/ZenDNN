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

#include "common/hashable_object.hpp"

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;
using json         = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

hashable_object_t::hashable_object_t():
  object_t{}, hash_key{0} {
}

bool hashable_object_t::operator==(const hashable_object_t& other_) const {
  if(hash_key && other_.hash_key) {
    if(hash_key == other_.hash_key)
      return true;
  }
  else {
    std::string message = "attempt to compare bad objects";
    EXCEPTION_WITH_LOC(message);
  }

  return false;
}

bool hashable_object_t::operator!=(const hashable_object_t& other_) const {
  if(hash_key && other_.hash_key) {
    if(hash_key != other_.hash_key)
      return true;
  }
  else {
    std::string message = "attempt to compare bad objects";
    EXCEPTION_WITH_LOC(message);
  }

  return false;
}

void hashable_object_t::reset() {
  parent_type::reset();
  hash_key = 0;
}

void hashable_object_t::set_object_info() {
  parent_type::set_object_info();
  obj_info_json["hash_key"] = hash_key;
}

void hashable_object_t::set_object_runtime_info() {
  parent_type::set_object_runtime_info();
  obj_runtime_info_json["hash_key"] = hash_key;
}

std::size_t hashable_object_t::get_hash() const {
  return hash_key;
}

void hashable_object_t::create() {
  set_object_info();
  status = status_t::success;
}

std::size_t hashable_object_t::hash() {
  return hash_key;
}

} //common
} //zendnnl

