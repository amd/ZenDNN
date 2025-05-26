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

#include "hash_object.hpp"

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;

hash_object_t::hash_object_t():
  hash_key{0},
  status{status_t::bad_hash_object} {
}

hash_object_t::hash_object_t(hash_object_t&& other_) {
  hash_key = other_.hash_key;
  status   = other_.status;

  other_.hash_key = 0;
  other_.status   = status_t::bad_hash_object;
}

hash_object_t& hash_object_t::operator=(hash_object_t&& other_) {
  if(hash_key != other_.hash_key) {
    hash_key = other_.hash_key;
    status   = other_.status;

    other_.hash_key = 0;
    other_.status   = status_t::bad_hash_object;
  }

  return *this;
}

void hash_object_t::reset() {
  LOG_DEBUG_INFO("Reset hash object");
  hash_key = 0;
  status   = status_t::bad_hash_object;
}

void hash_object_t::set_last_status(status_t status_) {
  status = status_;
}

status_t hash_object_t::get_last_status() const {
  return status;
}

std::size_t hash_object_t::get_hash() const {
  LOG_DEBUG_INFO("Get the hash key of object");
  return hash_key;
}

bool hash_object_t::check() const {
  LOG_DEBUG_INFO("Validate status of hash object");
  return (status == status_t::success);
}

bool hash_object_t::operator==(const hash_object_t& other_) const {
  LOG_DEBUG_INFO("Check if hash objects are equal");
  if (hash_key && other_.hash_key && (hash_key == other_.hash_key))
    return true;

  return false;
}

bool hash_object_t::operator!=(const hash_object_t& other_) const {
  LOG_DEBUG_INFO("Check if hash objects are not equal");
  if (hash_key != other_.hash_key)
    return true;

  return false;
}

}//commom
}//zendnnl
