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
#include "logging.hpp"

namespace zendnnl {
namespace error_handling {
using namespace zendnnl::common;

namespace cn = std::chrono;

logger_t::logger_t()
  :log_file{}, log_ofstream{}, log_start_time{cn::steady_clock::now()},
   log_cout_flag{true} {
  log_level_map[log_module_t::common]  = log_level_t::warning;
  log_level_map[log_module_t::api]     = log_level_t::warning;
  log_level_map[log_module_t::test]    = log_level_t::warning;
  log_level_map[log_module_t::profile] = log_level_t::warning;
  log_level_map[log_module_t::debug]   = log_level_t::warning;
  log_level_map[log_module_t::trace]   = log_level_t::disabled;
}

logger_t &logger_t::set_log_level(log_module_t module_, log_level_t level_) {
  log_level_map[module_] = level_;
  return *this;
}

log_level_t logger_t::get_log_level(log_module_t module_) {
  return log_level_map[module_];
}

logger_t &logger_t::set_log_file(std::string log_file_) {
  //if log_stream already open throw exception
  if (log_ofstream.is_open()) {
    log_ofstream.close();
    std::string message = "trying to reopen < " + log_file + " >";
    message += " with new name < ";
    message += log_file_;
    message += ">.";
    EXCEPTION_WITH_LOC(message);
  }

  //open log_stream
  log_ofstream.open(log_file_);
  if (!log_ofstream.is_open()) {
    std::string message = "unable to open log file < " + log_file_ + " >.";
    EXCEPTION_WITH_LOC(message);
  }

  log_file      = log_file_;
  log_cout_flag = false;

  return *this;
}

std::string logger_t::get_log_file() {
  return log_file;
}

logger_t &logger_t::set_config(const config_logger_t &config_logger_) {

  for (auto& [key, value] : config_logger_.log_level_map) {
    log_level_map[key] = value;
  }

  return *this;
}

void logger_t::log_msg_r(log_module_t log_module_, log_level_t log_level_,
                         std::string &message_) {

  //get time lapse from start time
  auto lapsed_time  = cn::steady_clock::now() - log_start_time;

  auto us  = cn::duration_cast<cn::microseconds>(lapsed_time).count();
  auto sec = float(us)/1000000.0;

  std::stringstream stream;
  stream << std::fixed << std::setprecision(6) << sec;

  //get log module string
  auto log_module_str = logger_support_t::log_module_to_str(log_module_);

  //get level string
  auto log_level_str  = logger_support_t::log_level_to_str(log_level_);

  //prepare message header
  std::string module_hdr = std::string("[") + log_module_str + std::string("]");
  std::string level_hdr  = std::string("[") + log_level_str + std::string("]");
  std::string time_hdr   = std::string("[") + stream.str() + std::string("]:");
  std::string hdr        = module_hdr + level_hdr + time_hdr;
  std::string empty_hdr  = std::string(hdr.length(), ' ');

  if (log_cout_flag) {
    std::lock_guard<std::mutex> lk{log_mutex};

    std::size_t nl_pos = message_.find('\n');
    if (nl_pos != std::string::npos) {
      std::cout << hdr <<  message_.substr(0, nl_pos) << "\n";
      std::string sub_message = message_.substr(nl_pos + 1);

      nl_pos = sub_message.find('\n');
      while(nl_pos != std::string::npos) {
        std::cout << empty_hdr << sub_message.substr(0, nl_pos) << "\n";
        sub_message = sub_message.substr(nl_pos + 1);
        nl_pos = sub_message.find('\n');
      }
      std::cout << empty_hdr << sub_message << "\n";
    }
    else {
      std::cout << hdr << message_ << "\n";
    }
  }
  else {
    std::lock_guard<std::mutex> lk{log_mutex};
    log_ofstream << hdr << "\n" << message_ << "\n";
  }
}

}//error_handling
}//zendnnl
