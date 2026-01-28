/*******************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
std::once_flag              zendnnl_global_block_t::init_flag;
zendnnl_global_block_t* zendnnl_global_block_t::instance = nullptr;

zendnnl_global_block_t::zendnnl_global_block_t()
  :config_manager{}, platform_info{},
   logger{}, lru_cache{} {

  platform_info.populate();
  config_manager.config();
  logger.set_config(config_manager.get_logger_config());
  lru_cache.set_config(config_manager.get_lru_cache_config());
}

zendnnl_global_block_t* zendnnl_global_block_t::get() {
  // Thread-safe initialization
  std::call_once(init_flag, []() {
    instance = new zendnnl_global_block_t();
  });

  return instance;
}

config_manager_t& zendnnl_global_block_t::get_config_manager() {
  return config_manager;
}

platform_info_t& zendnnl_global_block_t::get_platform_info() {
  return platform_info;
}

logger_t& zendnnl_global_block_t::get_logger() {
  return logger;
}

sptr_lru_cache_t& zendnnl_global_block_t::get_lru_cache() {
  return lru_cache;
}

}//common
}//zendnnl
