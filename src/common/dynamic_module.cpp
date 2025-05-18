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

#include "dynamic_module.hpp"

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;

//implementation
dynamic_module_t::dynamic_module_t():
  module_handle{nullptr}, module_name{} {
}

dynamic_module_t::~dynamic_module_t() {
  unload();
}

dynamic_module_t& dynamic_module_t::set_name(std::string module_name_) {
  LOG_DEBUG_INFO("Set name for dynamic module");
  module_name = module_name_;
  return *this;
}

std::string dynamic_module_t::get_name() const {
  LOG_DEBUG_INFO("Get dynamic module name");
  return module_name;
}

#ifdef __linux__
status_t dynamic_module_t::load() {
  LOG_DEBUG_INFO("Load dynamic module");
  if (module_name.empty())
    return status_t::utils_bad_module_name;

  try {
    module_name   = "lib" + module_name + ".so";
    module_handle = dlopen(module_name.c_str(), RTLD_NOW);
    if (!module_handle) {
      EXCEPTION_WITH_LOC(dlerror());
    }
  } catch(const exception_t& ex) {
    EXCEPTION_WITH_LOC(ex.what());
  }

  return status_t::success;
}

void dynamic_module_t::unload() {
  LOG_DEBUG_INFO("Unload dynamic module");
  if (module_handle) {
    dlclose(module_handle);
    module_handle = nullptr;
  }
}

void* dynamic_module_t::get_symbol(std::string symbol_name_) {
  LOG_DEBUG_INFO("Get dynamic module symbol");
  try {
    if (module_handle) {
      void* symbol_handle = dlsym(module_handle, symbol_name_.c_str());
      if (dlerror() != NULL)
        EXCEPTION_WITH_LOC(dlerror());

      if (symbol_handle)
        return symbol_handle;
    }
  } catch(const exception_t& ex) {
    EXCEPTION_WITH_LOC(ex.what());
  }
  return nullptr;
}

#else
status_t dynamic_module_t::load() {
  return status_t::unimplemented;
}

void dynamic_module_t::unload() {
}

void* dynamic_module_t::get_symbol(std::string symbol_name_) {
  return nullptr;
}

#endif

} //common
} //zendnnl

