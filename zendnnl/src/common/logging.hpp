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
#include <algorithm>

#include "common/zendnnl_exceptions.hpp"
#include "common/config_params.hpp"

namespace zendnnl {
namespace error_handling {
using namespace zendnnl::common;

namespace cn = std::chrono;

/** @class logger_t
 *  @brief A thread-safe logger class.
 *
 *  Logger is owned by @c zendnnl_global_block_t. Though this class is not a
 *  singleton, only one logger owned by @c zendnnl_global_block_t singleton
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

  /** @brief Set logger configuration
   *
   *  Sets logger configuration as received by config manager.
   * @param config_logger_ : config received by config manager.
   * @return A reference to self.
   */
  logger_t& set_config(const config_logger_t& config_logger_);

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
                 std::string &message_, MSG_T msg_arg0_, MSG_TS... msg_args_);

  /** @brief recursive terminating function to log a message */
  void log_msg_r(log_module_t log_module_, log_level_t log_level_,
                 std::string &message_);

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
                         std::string &message_, MSG_T msg_arg0_, MSG_TS... msg_args_) {
  std::stringstream stream;
  stream << msg_arg0_;
  message_ += stream.str();

  log_msg_r(log_module_, log_level_, message_, msg_args_...);
}

}//error_handling
}//zendnnl

#endif
