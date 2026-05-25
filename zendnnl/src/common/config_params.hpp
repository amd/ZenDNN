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
#ifndef _CONFIG_PARAMS_HPP_
#define _CONFIG_PARAMS_HPP_

#include <map>
#include <cstdint>

#include "common/logging_support.hpp"

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;

struct config_logger_t {
  std::map<log_module_t, log_level_t> log_level_map;
};

struct config_profiler_t {
  bool enable_profiler;

  config_profiler_t() : enable_profiler(false) {}
};

struct config_lru_cache_t {
uint32_t capacity = 0;
};

// Toggle for the per-layer AOCL DLP post-op metadata cache built by
// create_dlp_post_op (see aocl_postop.cpp). When `enable` is false the
// cache is cleared at the start of every create_dlp_post_op call, which
// forces a cold-path rebuild on every invocation and makes behavior
// bit-equivalent to pre-cache zendnnl. Intended as a runtime kill
// switch for triage and as a safety valve for integrators (zentorch,
// vLLM, etc.) who hit unexpected behavior in the field. Driven by
// the ZENDNNL_ENABLE_POSTOP_CACHE environment variable; sampled once
// per process via is_postop_cache_enabled() in zendnnl_global.hpp.
struct config_postop_cache_t {
  // Cache is enabled by default. ZENDNNL_ENABLE_POSTOP_CACHE=0
  // (or false/off/no) acts as a runtime kill switch that forces every
  // create_dlp_post_op() call through the cold path — used for triage
  // and as a safety valve for integrators (zentorch, vLLM, etc.) who
  // hit unexpected behavior in the field. The default was previously
  // false during the initial cache soak; it was flipped to true once
  // the cache had been validated across integrator releases.
  bool enable = true;
};

}//common
}//zendnnl
#endif
