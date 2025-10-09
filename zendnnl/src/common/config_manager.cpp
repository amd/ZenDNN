/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

namespace zendnnl {
namespace common {

using namespace zendnnl::error_handling;
using json = nlohmann::json;
using namespace zendnnl::ops;

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

  // Retrieve the singleton instance of matmul_config_t and set default configuration.
  matmul_config_t &matmul_config = matmul_config_t::instance();
  matmul_config.set_default_config();
}

void config_manager_t::set_user_config() {
  set_user_logger_config();
  set_user_profiler_config();

  // Retrieve the singleton instance of matmul_config_t and set user configuration.
  matmul_config_t &matmul_config = matmul_config_t::instance();
  matmul_config.set_user_config(config_json);
}

void config_manager_t::set_env_config() {
  set_env_logger_config();
  set_env_profiler_config();

  // Retrieve the singleton instance of matmul_config_t and set env configuration.
  matmul_config_t &matmul_config = matmul_config_t::instance();
  matmul_config.set_env_config();
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
    char *log_level_str = std::getenv("ZENDNNL_COMMON_LOG_LEVEL");
    if (log_level_str) {
      uint32_t log_level = std::stoi(log_level_str);
      if (log_level < uint32_t(log_level_t::log_level_count))
        config_logger.log_level_map[log_module_t::common]
          = log_level_t(log_level);
    }
  }

  {
    char *log_level_str = std::getenv("ZENDNNL_API_LOG_LEVEL");
    if (log_level_str) {
      uint32_t log_level = std::stoi(log_level_str);
      if (log_level < uint32_t(log_level_t::log_level_count))
        config_logger.log_level_map[log_module_t::api]
          = log_level_t(log_level);
    }
  }

  {
    char *log_level_str = std::getenv("ZENDNNL_TEST_LOG_LEVEL");
    if (log_level_str) {
      uint32_t log_level = std::stoi(log_level_str);
      if (log_level < uint32_t(log_level_t::log_level_count))
        config_logger.log_level_map[log_module_t::test]
          = log_level_t(log_level);
    }
  }

  {
    char *log_level_str = std::getenv("ZENDNNL_PROFILE_LOG_LEVEL");
    if (log_level_str) {
      uint32_t log_level = std::stoi(log_level_str);
      if (log_level < uint32_t(log_level_t::log_level_count))
        config_logger.log_level_map[log_module_t::profile]
          = log_level_t(log_level);
    }
  }

  {
    char *log_level_str = std::getenv("ZENDNNL_DEBUG_LOG_LEVEL");
    if (log_level_str) {
      uint32_t log_level = std::stoi(log_level_str);
      if (log_level < uint32_t(log_level_t::log_level_count))
        config_logger.log_level_map[log_module_t::debug]
          = log_level_t(log_level);
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

} //common
} //zendnnl


