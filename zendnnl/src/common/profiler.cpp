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

#include "profiler.hpp"

namespace zendnnl {
namespace profile {

profiler_t::profiler_t()
  : timer_started(false),
    res_str("ms"),
    resolution(time_res_t::milliseconds),
    elapsed_time(0.0) {}

void profiler_t::tbp_set_default_res(time_res_t res) {
  resolution = res;
}

status_t profiler_t::tbp_start() {
  if(timer_started) {
    return status_t::failure;
  }
  start_time = std::chrono::high_resolution_clock::now();
  timer_started = true;

  return status_t::success;
}

status_t profiler_t::tbp_stop() {
  if(!timer_started) {
    return status_t::failure;
  }
  stop_time = std::chrono::high_resolution_clock::now();
  calculate_elapsed();
  timer_started = false;

  return status_t::success;
}

double profiler_t::tbp_elapsedtime() const {
  return elapsed_time;
}

std::string profiler_t::get_res_str() {
  return res_str;
}

void profiler_t::calculate_elapsed() {
  using namespace std::chrono;
  auto time_duration = stop_time - start_time;
  switch (resolution) {
  case time_res_t::milliseconds:
    elapsed_time = duration<double, std::milli>(time_duration).count();
    res_str = "ms";
    break;
  case time_res_t::microseconds:
    elapsed_time = duration<double, std::micro>(time_duration).count();
    res_str = "us";
    break;
  case time_res_t::seconds:
    elapsed_time = duration<double>(time_duration).count();
    res_str = "sec";
    break;
  }
}

} // profile
} // zendnnl