/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "config_manager.hpp"

#include <algorithm>
#include <cctype>
#include <string>

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;
using json = nlohmann::json;
using namespace zendnnl::ops;

namespace {

// Throw-free parse of a `*_LOG_LEVEL` environment value.  Accepts EITHER a
// numeric level ("0".."4") OR a case-insensitive name ("disabled" / "error"
// / "warning" / "info" / "verbose") — the same names the JSON config path
// already accepts via `logger_support_t::str_to_log_level`.  Returns true
// and writes `out` on a recognised value; returns false (leaving the
// caller's current setting untouched) on null / empty / out-of-range /
// unrecognised input.
//
// Rationale: the previous `std::stoi(std::getenv(...))` ABORTED the whole
// process with an uncaught `std::invalid_argument` when an operator set a
// non-numeric level such as `ZENDNNL_API_LOG_LEVEL=info` (a natural value,
// and exactly the one the JSON path accepts).  A logging-config knob must
// never crash the library or its tools (benchdnn / gtests).
bool parse_env_log_level(const char *raw, log_level_t &out) {
  if (raw == nullptr) {
    return false;
  }
  std::string s(raw);
  const auto not_space = [](unsigned char c) {
    return std::isspace(c) == 0;
  };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  if (s.empty()) {
    return false;
  }

  // All-digits -> numeric level (manual parse: cannot throw).
  if (std::all_of(s.begin(), s.end(),
  [](unsigned char c) {
  return std::isdigit(c) != 0;
  })) {
    unsigned long v = 0;
    for (char c : s) {
      v = v * 10UL + static_cast<unsigned long>(c - '0');
      if (v >= static_cast<unsigned long>(log_level_t::log_level_count)) {
        return false;  // out of range (also bounds the accumulation)
      }
    }
    out = log_level_t(static_cast<uint32_t>(v));
    return true;
  }

  // Otherwise a name.  Accept ONLY the known tokens; an unrecognised name is
  // ignored (keep the current setting) rather than silently disabling logs.
  std::string lower = s;
  std::transform(lower.begin(), lower.end(), lower.begin(),
  [](unsigned char c) {
    return std::tolower(c);
  });
  if (lower == "disabled" || lower == "error" || lower == "warning" ||
      lower == "info" || lower == "verbose") {
    out = logger_support_t::str_to_log_level(lower);
    return true;
  }
  return false;
}

void apply_global_cache_off_to_matmul(
  zendnnl::ops::matmul_config_t &matmul_config, bool global_cache_off) {
  if (!global_cache_off) {
    return;
  }

  matmul_config.set_weight_cache(0);
  matmul_config.set_zp_comp_cache(false);
}

}  // namespace

void config_manager_t::config() {
  //default config
  set_default_config();

  //if config file is given set config as per config file
  char *config_file_str = std::getenv("ZENDNNL_CONFIG_FILE");

  if (config_file_str) {
    if (parse(config_file_str) == status_t::success) {
      set_user_config();
      return;
    }
  }

  //if config file is unavailable setup config as per env variables
  set_env_config();
}

const config_logger_t &config_manager_t::get_logger_config() const {
  return config_logger;
}

const config_profiler_t &config_manager_t::get_profiler_config() const {
  return config_profiler;
}

const config_lru_cache_t &config_manager_t::get_lru_cache_config() const {
  return config_lru_cache;
}

const config_postop_cache_t &
config_manager_t::get_postop_cache_config() const {
  return config_postop_cache;
}

bool config_manager_t::is_global_cache_off() const {
  return global_cache_off;
}

status_t config_manager_t::parse(std::string file_name_) {

  std::ifstream json_file(file_name_);

  if (! json_file.is_open()) {
    return status_t::config_bad_json_file;
  }

  config_json = json::parse(json_file);

  json_file.close();

  return status_t::success;
}

void config_manager_t::set_default_config() {
  set_default_logger_config();
  set_default_profiler_config();
  set_default_lru_cache_config();
  set_default_global_cache_config();
  set_default_postop_cache_config();

  // Retrieve the singleton instance of matmul_config_t and set default configuration.
  matmul_config_t &matmul_config = matmul_config_t::instance();
  matmul_config.set_default_config();
}

void config_manager_t::set_user_config() {
  set_user_logger_config();
  set_user_profiler_config();
  set_user_lru_cache_config();
  set_user_global_cache_config();
  set_user_postop_cache_config();
  if (global_cache_off) {
    config_postop_cache.enable = false;
  }

  // Retrieve the singleton instance of matmul_config_t and set user configuration.
  matmul_config_t &matmul_config = matmul_config_t::instance();
  matmul_config.set_user_config(config_json);
  apply_global_cache_off_to_matmul(matmul_config, global_cache_off);
}

void config_manager_t::set_env_config() {
  set_env_logger_config();
  set_env_profiler_config();
  set_env_lru_cache_config();
  set_env_global_cache_config();
  set_env_postop_cache_config();

  // Retrieve the singleton instance of matmul_config_t and set env configuration.
  matmul_config_t &matmul_config = matmul_config_t::instance();
  matmul_config.set_env_config();
  apply_global_cache_off_to_matmul(matmul_config, global_cache_off);
}

status_t config_manager_t::set_default_logger_config() {
  config_logger.log_level_map.clear();
  uint32_t module_count = uint32_t(log_module_t::log_module_count);
  for (uint32_t i = 0; i < module_count; ++i) {
    config_logger.log_level_map[log_module_t(i)] = log_level_t::warning;
  }
  config_logger.log_level_map[log_module_t::profile] = log_level_t::verbose;

  return status_t::success;
}

status_t config_manager_t::set_user_logger_config() {
  //check for log_levels key
  auto logger_json = config_json["log_levels"];
  if (logger_json.empty()) {
    return status_t::failure;
  }

  //get log levels of each log
  auto common_level_json = logger_json["common"];
  if (! common_level_json.empty()) {
    auto log_level_str = common_level_json.template get<std::string>();
    if (! log_level_str.empty()) {
      config_logger.log_level_map[log_module_t::common]
        = logger_support_t::str_to_log_level(log_level_str);
    }
  }

  auto api_level_json = logger_json["api"];
  if (! api_level_json.empty()) {
    auto log_level_str = api_level_json.template get<std::string>();
    if (! log_level_str.empty()) {
      config_logger.log_level_map[log_module_t::api]
        = logger_support_t::str_to_log_level(log_level_str);
    }
  }

  auto test_level_json = logger_json["test"];
  if (! test_level_json.empty()) {
    auto log_level_str = test_level_json.template get<std::string>();
    if (! log_level_str.empty()) {
      config_logger.log_level_map[log_module_t::test]
        = logger_support_t::str_to_log_level(log_level_str);
    }
  }

  auto profile_level_json = logger_json["profile"];
  if (! profile_level_json.empty()) {
    auto log_level_str = profile_level_json.template get<std::string>();
    if (! log_level_str.empty()) {
      config_logger.log_level_map[log_module_t::profile]
        = logger_support_t::str_to_log_level(log_level_str);
    }
  }

  auto debug_level_json = logger_json["debug"];
  if (! debug_level_json.empty()) {
    auto log_level_str = debug_level_json.template get<std::string>();
    if (! log_level_str.empty()) {
      config_logger.log_level_map[log_module_t::debug]
        = logger_support_t::str_to_log_level(log_level_str);
    }
  }

  return status_t::success;
}

status_t config_manager_t::set_env_logger_config() {
  {
    log_level_t level;
    if (parse_env_log_level(std::getenv("ZENDNNL_COMMON_LOG_LEVEL"), level)) {
      config_logger.log_level_map[log_module_t::common] = level;
    }
  }

  {
    log_level_t level;
    if (parse_env_log_level(std::getenv("ZENDNNL_API_LOG_LEVEL"), level)) {
      config_logger.log_level_map[log_module_t::api] = level;
    }
  }

  {
    log_level_t level;
    if (parse_env_log_level(std::getenv("ZENDNNL_TEST_LOG_LEVEL"), level)) {
      config_logger.log_level_map[log_module_t::test] = level;
    }
  }

  {
    log_level_t level;
    if (parse_env_log_level(std::getenv("ZENDNNL_PROFILE_LOG_LEVEL"), level)) {
      config_logger.log_level_map[log_module_t::profile] = level;
    }
  }

  {
    log_level_t level;
    if (parse_env_log_level(std::getenv("ZENDNNL_DEBUG_LOG_LEVEL"), level)) {
      config_logger.log_level_map[log_module_t::debug] = level;
    }
  }

  return status_t::success;

}

status_t config_manager_t::set_default_profiler_config() {
  config_profiler.enable_profiler = false;

  return status_t::success;
}

status_t config_manager_t::set_user_profiler_config() {
  //check for log_levels key
  auto profiler_json = config_json["profiler"];
  if (profiler_json.empty()) {
    return status_t::failure;
  }

  //get log levels of each log
  auto enable_profiler_json = profiler_json["enable_profiler"];
  if (!enable_profiler_json.empty()) {
    config_profiler.enable_profiler = enable_profiler_json.get<bool>();
  }

  return status_t::success;
}

status_t config_manager_t::set_env_profiler_config() {
  {
    char *enable_profiler_str = std::getenv("ZENDNNL_ENABLE_PROFILER");
    if (enable_profiler_str) {
      config_profiler.enable_profiler = (std::string(enable_profiler_str) == "1");
    }
  }

  return status_t::success;
}

status_t config_manager_t::set_default_lru_cache_config() {
  config_lru_cache.capacity = UINT_MAX;

  return status_t::success;
}

status_t config_manager_t::set_user_lru_cache_config() {
  //check for lru_cache json object
  auto lru_cache_json = config_json["lru_cache"];
  if (lru_cache_json.empty()) {
    return status_t::failure;
  }

  //get log levels of each log
  auto capacity_json = lru_cache_json["capacity"];
  if (!capacity_json.empty()) {
    config_lru_cache.capacity = capacity_json.get<uint32_t>();
  }

  return status_t::success;
}

status_t config_manager_t::set_env_lru_cache_config() {
  {
    char *lru_cache_capacity_str = std::getenv("ZENDNNL_LRU_CACHE_CAPACITY");
    char *endptr;
    if (lru_cache_capacity_str) {
      config_lru_cache.capacity = std::strtoul(lru_cache_capacity_str, &endptr, 10);
    }
  }

  return status_t::success;
}

status_t config_manager_t::set_default_global_cache_config() {
  global_cache_off = false;
  return status_t::success;
}

status_t config_manager_t::set_user_global_cache_config() {
  auto global_cache_json = config_json["global_cache_off"];
  if (global_cache_json.empty()) {
    return status_t::failure;
  }

  global_cache_off = global_cache_json.get<bool>();
  return status_t::success;
}

status_t config_manager_t::set_env_global_cache_config() {
  global_cache_off = is_global_cache_off_env();
  return status_t::success;
}

status_t config_manager_t::set_default_postop_cache_config() {
  // Cache is enabled by default. ZENDNNL_CACHE_OFF=1 acts only as a global
  // kill switch; when it is enabled or unset, the local postop cache
  // policy remains in control.
  config_postop_cache.enable = true;
  return status_t::success;
}

status_t config_manager_t::set_user_postop_cache_config() {
  auto postop_cache_json = config_json["postop_cache"];
  if (postop_cache_json.empty()) {
    return status_t::failure;
  }

  auto enable_json = postop_cache_json["enable"];
  if (!enable_json.empty()) {
    config_postop_cache.enable = enable_json.get<bool>();
  }

  return status_t::success;
}

status_t config_manager_t::set_env_postop_cache_config() {
  // ZENDNNL_CACHE_OFF only acts as a process-wide kill switch. When it is
  // enabled, all local cache knobs are forced off.
  if (global_cache_off) {
    config_postop_cache.enable = false;
    return status_t::success;
  }

  // Accept the same truthy/falsy spellings as the other ZENDNNL_ENABLE_*
  // toggles use today ("1"/"0"), plus the more readable "true"/"false"
  // and "on"/"off" so integrators don't have to remember which we picked.
  // Any unrecognized non-empty value leaves the local default
  // untouched rather than silently disabling, so a typo can't accidentally
  // turn the cache off.
  char *enable_str = std::getenv("ZENDNNL_ENABLE_POSTOP_CACHE");
  if (!enable_str) {
    return status_t::success;
  }

  bool postop_cache_enabled = config_postop_cache.enable;
  if (parse_cache_bool(enable_str, postop_cache_enabled)) {
    config_postop_cache.enable = postop_cache_enabled;
  }

  return status_t::success;
}

} //common
} //zendnnl


