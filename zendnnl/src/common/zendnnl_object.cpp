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

#include "common/zendnnl_object.hpp"

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;
using json         = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

const std::string object_t::unknown_object_str = "unknown_object";

object_t::object_t():
  status{status_t::bad_object},
  obj_name{unknown_object_str},
  obj_info_json{},
  obj_runtime_info_json{} {
}

void object_t::reset() {
  status                = status_t::bad_object;
  obj_name              = unknown_object_str;
  obj_info_json         = ordered_json{};
  obj_runtime_info_json = ordered_json{};
}

void object_t::set_name(std::string obj_name_) {

  //return if object has been created.
  if (status == status_t::success)
    return;

  obj_name = obj_name_;
}

std::string object_t::get_name() const {
  return obj_name;
}

void object_t::set_object_info() {
  obj_info_json["name"] = obj_name;
}

void object_t::set_object_runtime_info() {
  obj_runtime_info_json["name"] = obj_name;
}

nlohmann::ordered_json object_t::get_object_info() const {
  return obj_info_json;
}

nlohmann::ordered_json object_t::get_object_runtime_info() const {
  return obj_runtime_info_json;
}

std::string object_t::get_object_summary() const {
  return obj_info_json["name"].dump();
}

std::string object_t::get_object_runtime_summary() const {
  return obj_runtime_info_json["name"].dump();
}

void object_t::set_last_status(status_t status_) {

  //return if object has been created.
  if (status == status_t::success)
    return;

  status = status_;
}

status_t object_t::get_last_status() const {
  return status;
}

bool object_t::is_bad_object() const {
  return (status_t::success != status);
}

bool object_t::is_unnamed_object() const {
  return (unknown_object_str == obj_name);
}

} //common
} //zendnnl
