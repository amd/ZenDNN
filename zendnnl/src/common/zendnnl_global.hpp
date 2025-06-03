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
#ifndef  _ZENDNNL_GLOBAL_HPP_
#define  _ZENDNNL_GLOBAL_HPP_

#include "common/bfloat16.hpp"
#include "common/data_types.hpp"
#include "common/zendnnl_global_block.hpp"

/** @def LOG_DEBUG_INFO(...)
 *
 * Print the debug information about the file and function call. Mostly
 * used to trace which functions are called in debug mode.
 */
#ifndef NDEBUG
#define LOG_DEBUG_INFO(...)                                                  \
  do                                                                         \
    debuglog_verbose("[",get_relative_path(__FILE__),"], [",                 \
                     __PRETTY_FUNCTION__,"]: ", __VA_ARGS__);                \
  while(0);
#else
#define LOG_DEBUG_INFO(...)
#endif

/** @def LOGGER_MACRO(LOG_MODULE, LOG_LEVEL)
 *
 * Used to expand various logger functions for different log modules and
 * log levels.
 */
#define LOGGER_MACRO(LOG_MODULE, LOG_LEVEL)                                  \
  template<typename... Ts>                                                   \
  static inline void LOG_MODULE##log##_##LOG_LEVEL(Ts...vargs) {             \
  zendnnl::common::zendnnl_global_block()                                    \
  .get_logger()                                                              \
  .log_msg(zendnnl::error_handling::log_module_t::LOG_MODULE,                \
           zendnnl::error_handling::log_level_t::LOG_LEVEL, vargs...);       \
  }

/** @def LOGGER_MACRO(LOG_MODULE, LOG_LEVEL)
 *
 * Used to expand various logger functions for common log for various log
 * levels. These functions do not have log module name prefixed and write
 * to common log.
 */
#define COMMON_LOGGER_MACRO(LOG_LEVEL)                                      \
  template<typename... Ts>                                                  \
  static inline void log##_##LOG_LEVEL(Ts...vargs) {                        \
  zendnnl::common::zendnnl_global_block()                                   \
    .get_logger()                                                           \
    .log_msg(zendnnl::error_handling::log_module_t::common,                 \
             zendnnl::error_handling::log_level_t::LOG_LEVEL, vargs...);    \
  }

namespace zendnnl {
namespace common {

/** @fn zendnnl_init()
 * @brief zendnn initialization function
 *
 * Initializes ZenDNNL by creating its persistent global block singleton
 */
static inline void zendnnl_init() {
  //zendnnl_global_block_t* ins = zendnnl_global_block_t::get();
  //platform_info_t& platform_info = ins->get_platform_info();
}

/** @fn zendnnl_global_block()
 * @brief Get a reference to zendnnl global block singleton
 * @return A reference to zendnnl global block singleton.
 */
static inline zendnnl_global_block_t& zendnnl_global_block() {
  return (*(zendnnl_global_block_t::get()));
}

/** @fn zendnnl_platform_info()
 * @brief Get a reference to zendnnl platform information block
 * @return A reference to zendnnl platform information block.
 */
static inline platform_info_t &zendnnl_platform_info() {
  return (zendnnl_global_block_t::get())->get_platform_info();
}

/** @fn get_relative_path
 * @brief Convert an absolute path to a ralative path.
 * @param abs_path : absolute path
 * @return A Relative path string
 */
static inline const char *get_relative_path(const char *abs_path_) {
  const char *rel = std::strstr(abs_path_, "ZenDNN/");
  return (rel? rel : abs_path_);
}

}//common

namespace error_handling {

using namespace zendnnl::common;

//logger functions
COMMON_LOGGER_MACRO(error)
COMMON_LOGGER_MACRO(warning)
COMMON_LOGGER_MACRO(info)
COMMON_LOGGER_MACRO(verbose)

LOGGER_MACRO(common, error)
LOGGER_MACRO(common, warning)
LOGGER_MACRO(common, info)
LOGGER_MACRO(common, verbose)

LOGGER_MACRO(api, error)
LOGGER_MACRO(api, warning)
LOGGER_MACRO(api, info)
LOGGER_MACRO(api, verbose)

LOGGER_MACRO(test, error)
LOGGER_MACRO(test, warning)
LOGGER_MACRO(test, info)
LOGGER_MACRO(test, verbose)

LOGGER_MACRO(profile, error)
LOGGER_MACRO(profile, warning)
LOGGER_MACRO(profile, info)
LOGGER_MACRO(profile, verbose)

LOGGER_MACRO(debug, error)
LOGGER_MACRO(debug, warning)
LOGGER_MACRO(debug, info)
LOGGER_MACRO(debug, verbose)

}//error_handling

namespace interface {
COMMON_LOGGER_MACRO(error)
COMMON_LOGGER_MACRO(warning)
COMMON_LOGGER_MACRO(info)
COMMON_LOGGER_MACRO(verbose)

LOGGER_MACRO(common, error)
LOGGER_MACRO(common, warning)
LOGGER_MACRO(common, info)
LOGGER_MACRO(common, verbose)

LOGGER_MACRO(api, error)
LOGGER_MACRO(api, warning)
LOGGER_MACRO(api, info)
LOGGER_MACRO(api, verbose)

LOGGER_MACRO(test, error)
LOGGER_MACRO(test, warning)
LOGGER_MACRO(test, info)
LOGGER_MACRO(test, verbose)

// LOGGER_MACRO(profile, error)
// LOGGER_MACRO(profile, warning)
// LOGGER_MACRO(profile, info)
// LOGGER_MACRO(profile, verbose)

// LOGGER_MACRO(debug, error)
// LOGGER_MACRO(debug, warning)
// LOGGER_MACRO(debug, info)
// LOGGER_MACRO(debug, verbose)

}//interface

}//zendnnl
#endif
