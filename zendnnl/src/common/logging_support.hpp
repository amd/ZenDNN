/*******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _LOGGING_SUPPORT_HPP_
#define _LOGGING_SUPPORT_HPP_

#include <algorithm>
#include <string>

namespace zendnnl {
namespace error_handling {

/** @enum log_level_t
 *  @brief defines message levels from error to verbose.
 *
 *  A log with certain level enabled will print all the message at that level
 *  and below.
 */
enum class log_level_t : uint32_t {
  disabled = 0,    /*!< Log disabled, no messages go to the log */
  error    = 1,    /*!< Log only error messages */
  warning  = 2,    /*!< Log error and warning messages */
  info     = 3,    /*!< Log error, warning and info messages */
  verbose  = 4,    /*!< Log all messages */
  log_level_count  /*!< Log level count */
};

/** @enum log_modules_t
 *  @brief defines different log modules.
 *
 * Messages not categorized into a log module will go to common log.
 */
enum class log_module_t : uint32_t {
  common = 0,       /*!< Common log */
  api,              /*!< API log */
  test,             /*!< Test log */
  profile,          /*!< Profile log */
  debug,            /*!< Debug log */
  trace,            /*!< Trace log */ 
  log_module_count  /*!< Log module count */
};

/** @class logger_support_t
 *  @brief A class to hold support functions for logger.
 *
 *  This class holds support functions that both logger and config
 *  manager use.
 */
class logger_support_t final {
 public:
  /** @brief Convert from log module to string
   *
   *  @param module_ : log module
   *  @return A log module string.
   */
  static std::string log_module_to_str(log_module_t log_module_) {
    switch (log_module_) {
    case log_module_t::common:
      return "COMMON ";
    case log_module_t::api:
      return "API    ";
    case log_module_t::test:
      return "TEST   ";
    case log_module_t::profile:
      return "PROF   ";
    case log_module_t::debug:
      return "DEBUG  ";
    case log_module_t::trace:
      return "TRACE  ";
    default:
      return "unknown";
    }

    return "unknown";
  }

  /** @brief Convert from string to log module.
   *
   *  @param str_ : string containing log module name.
   *  @return log module for appropriate string.
   *          log_module_t::log_module_count if string is not
   *          appropriate.
   */
  static log_module_t str_to_log_module(std::string str_) {
    //transform str_ to lowercase
    std::transform(str_.begin(), str_.end(), str_.begin(),
    [](unsigned char c) {
      return std::tolower(c);
    });
    //convert to integer value
    if ("common" == str_) {
      return log_module_t::common;
    }
    if ("api" == str_) {
      return log_module_t::api;
    }
    if ("test" == str_) {
      return log_module_t::test;
    }
    if ("profile" == str_) {
      return log_module_t::profile;
    }
    if ("debug" == str_) {
      return log_module_t::debug;
    }
    if ("trace" == str_) {
      return log_module_t::trace;
    }

    return log_module_t::log_module_count;
  }

  /** @brief Convert from log level to string
   *
   *  @param log_level_ : log level
   *  @return A log level string.
   */
  static std::string log_level_to_str(log_level_t log_level_) {
    switch (log_level_) {
    case log_level_t::disabled:
      return "disabled";
    case log_level_t::error:
      return "error  ";
    case log_level_t::warning:
      return "warning";
    case log_level_t::info:
      return "info   ";
    case log_level_t::verbose:
      return "verbose";
    default:
      return "unknown";
    }

    return "unknown";
  }

  /** @brief Convert from string to log level.
   *
   *  @param str_ : string containng log level name.
   *  @return log level for appropriate string,
   *  else log_level_t::disabled.
   */
  static log_level_t str_to_log_level(std::string str_) {
    //transform str_ to lowercase
    std::transform(str_.begin(), str_.end(), str_.begin(),
    [](unsigned char c) {
      return std::tolower(c);
    });
    //convert to integer value
    if ("disabled" == str_) {
      return log_level_t::disabled;
    }
    if ("error" == str_) {
      return log_level_t::error;
    }
    if ("warning" == str_) {
      return log_level_t::warning;
    }
    if ("info" == str_) {
      return log_level_t::info;
    }
    if ("verbose" == str_) {
      return log_level_t::verbose;
    }

    return log_level_t::disabled;
  }

};

}//error_handling
}//zendnnl
#endif
