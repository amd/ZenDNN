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

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>

#include "common/logging_support.hpp"

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;

inline bool parse_cache_bool(const char *raw, bool &out) {
  if (raw == nullptr) {
    return false;
  }

  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(),
  [](unsigned char c) {
    return std::tolower(c);
  });

  if (value == "0" || value == "false" || value == "off" || value == "no") {
    out = false;
    return true;
  }
  if (value == "1" || value == "true" || value == "on" || value == "yes") {
    out = true;
    return true;
  }

  return false;
}

inline bool is_global_cache_off_env() {
  static const bool cache_off = []() {
    bool value = false;
    parse_cache_bool(std::getenv("ZENDNNL_CACHE_OFF"), value);
    return value;
  }
  ();
  return cache_off;
}

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
// vLLM, etc.) who hit unexpected behavior in the field.
// ZENDNNL_CACHE_OFF / JSON `global_cache_off` act as a process-wide kill
// switch: when set truthy this cache is forced off. Otherwise the local
// ZENDNNL_ENABLE_POSTOP_CACHE knob retains its usual default/override
// behavior; sampled once per process via
// is_postop_cache_enabled() in zendnnl_global.hpp.
struct config_postop_cache_t {
  // Cache is enabled by default. ZENDNNL_CACHE_OFF=1 (or JSON
  // `global_cache_off=true`) is a global kill switch that forces all
  // cache infrastructure off; otherwise
  // ZENDNNL_ENABLE_POSTOP_CACHE controls this cache specifically.
  // The disabled path forces every create_dlp_post_op() call through
  // the cold path for triage and as a safety valve for integrators
  // (zentorch, vLLM, etc.).
  bool enable = true;
};

}//common
}//zendnnl
#endif
