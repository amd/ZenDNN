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
#ifndef _LOGGING_HPP_
#define _LOGGING_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdarg>
#include <chrono>
#include <string>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <map>
#include <vector>

#include "common/zendnnl_exceptions.hpp"

namespace zendnnl {
namespace error_handling {
namespace cn = std::chrono;

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
  log_module_count  /*!< Log module count */
};

/** @class logger_t
 *  @brief A thread-safe logger class.
 *
 *  Logger is owned by @c zendnnl_global_block. Though this class is not a
 *  singleton, only one logger owned by @c zendnnl_global_block singleton
 *  will be used for logging purposes.
 *
 *  The logger is thread-safe and serializes logging messages. However no
 *  synchronization is implemented for time stamps, and different threads can
 *  log messages with out of order time stamps.
 *
 *  It is capable of writing to a log file. If no log file is provided, log is
 *  written at standard output.
 *
 *  The message are of the form
 *  @verbatim
 *  [<log module>][<log level>][<elapsed time>]: <message>
 *  @endverbatim
 */
class logger_t {
public:
  using log_level_map_type = std::map<log_module_t, log_level_t>;

  /** @brief default constructor */
  logger_t();

  /** @brief Set log module level
   * @param module_ : log module
   * @param level_  : log level to setup.
   * @return A reference to self.
   */
  logger_t& set_log_level(log_module_t module_, log_level_t level_);

  /** @brief Get log module level
   * @param module_ : log module
   * @param level_  : log level to setup.
   * @return A reference to self.
   */
  log_level_t get_log_level(log_module_t module_);

  /** @brief Set log file
   * @param log_file_ : log file name.
   * @return A reference to self.
   */
  logger_t& set_log_file(std::string log_file_);

  /** @brief Get log file
   * @return log file name
   */
  std::string get_log_file();

  /** @brief log a message
   * @param log_module_ : log module the message should go
   * @param log_level_  : level of the message.
   * @param msg_args_   : variable message arguments forming message.
   */
  template<typename MSG_T, typename... MSG_TS>
  void log_msg(log_module_t log_module_, log_level_t log_level_,
               MSG_T msg_arg0_, MSG_TS... msg_args_);

private:
  /** @brief recursive function to log a message */
  template<typename MSG_T, typename... MSG_TS>
  void log_msg_r(log_module_t log_module_, log_level_t log_level_,
                 std::string& message_, MSG_T msg_arg0_, MSG_TS... msg_args_);

  /** @brief recursive terminating function to log a message */
  void log_msg_r(log_module_t log_module_, log_level_t log_level_, std::string& message_);

  /** @brief convert from module enum to string */
  std::string log_module_to_str(log_module_t module_);

  /** @brief convert from level enum to string */
  std::string log_level_to_str(log_level_t level_);

private:
  std::string                   log_file;         /*!< Log file name */
  std::ofstream                 log_ofstream;     /*!< Log file stream */
  cn::steady_clock::time_point  log_start_time;   /*!< Logger creation time stamp */
  bool                          log_cout_flag;    /*!< Write to a log file or cout */
  std::mutex                    log_mutex;        /*!< Mutex for thread safety */
  log_level_map_type            log_level_map;    /*!< Log level map */
};

template<typename MSG_T, typename... MSG_TS>
void logger_t::log_msg(log_module_t log_module_, log_level_t log_level_,
                       MSG_T msg_arg0_, MSG_TS... msg_args_) {

  if (log_level_map.at(log_module_) >= log_level_) {
    std::stringstream stream;
    stream << msg_arg0_;
    std::string message = stream.str();

    log_msg_r(log_module_, log_level_, message, msg_args_...);
  }
}

template<typename MSG_T, typename... MSG_TS>
void logger_t::log_msg_r(log_module_t log_module_, log_level_t log_level_,
                         std::string& message_, MSG_T msg_arg0_, MSG_TS... msg_args_) {
  std::stringstream stream;
  stream << msg_arg0_;
  message_ += stream.str();

  log_msg_r(log_module_, log_level_, message_, msg_args_...);
}

}//error_handling
}//zendnnl

#endif
