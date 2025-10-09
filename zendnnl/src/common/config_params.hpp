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
#ifndef _CONFIG_PARAMS_HPP_
#define _CONFIG_PARAMS_HPP_

#include <map>
#include "common/logging_support.hpp"

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;

struct config_logger_t {
  std::map<log_module_t, log_level_t> log_level_map;
};
struct config_profiler_t {
  bool enable_profiler;
};

}//common
}//zendnnl
#endif
