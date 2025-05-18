/*******************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#include "zendnnl_global_block.hpp"

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;

//zendnnl_global_block_t instance initialization
std::mutex              zendnnl_global_block_t::instance_mutex;
zendnnl_global_block_t* zendnnl_global_block_t::instance = nullptr;

zendnnl_global_block_t::zendnnl_global_block_t()
  :platform_info{}, logger{} {

  platform_info.populate();
}

zendnnl_global_block_t* zendnnl_global_block_t::get() {
  //try to allocate if not allocated
  if (instance == nullptr) {
    std::lock_guard<std::mutex> lock(instance_mutex);

    //recheck again if another thread has acquired mutex before and created
    //instance.
    if (instance == nullptr) {
      instance = new zendnnl_global_block_t();
    }
  }

  return instance;
}

platform_info_t& zendnnl_global_block_t::get_platform_info() {
  return platform_info;
}

logger_t& zendnnl_global_block_t::get_logger() {
  return logger;
}

}//common
}//zendnnl
