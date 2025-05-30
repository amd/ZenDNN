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

#ifndef _PROFILER_HPP_
#define _PROFILER_HPP_

#include <chrono>
#include <string>

#include "error_status.hpp"

namespace zendnnl {
namespace profile {
using namespace zendnnl::error_handling;

/**
 * @enum time_res_t
 * @brief Supported time resolution.
 */
enum class time_res_t : uint32_t {
  milliseconds,
  microseconds,
  seconds
};

/** @class profiler_t
 *  @brief A utility class for profiling and measuring time.
 *
 * This class provides functions to start, stop, and measure elapsed time
 * using std::chrono. The time resolution can be configured by the user.
 */
class profiler_t {
 public:

  /** @brief default constructor
  * Initializes a profiler_t object with the default
  * time resolution set to MILLISECONDS.
  */
  profiler_t();

  /** @brief Set the time resolution
   * Configures the time unit in which elapsed time will be measured.
   *
   * @param res The time resolution to use.
  */
  void tbp_set_default_res(time_res_t res);

  /** @brief Start the timer
   * Records the start time using std::chrono::steady_clock.
  */
  status_t tbp_start();

  /** @brief Stop the timer
   * Records the stop time using std::chrono::steady_clock and
   * calculates the elapsed time.
  */
  status_t tbp_stop();

  /** @brief Get the elapsed time
   * Returns the elapsed time measured between the last calls to
   * tbp_start() and tbp_stop().
   *
   * @return The elapsed time in the currently configured time unit.
  */
  double tbp_elapsedtime() const;

  /** @brief Get resolution string
   *
   * @return Resolution string
  */
  std::string get_res_str();

 private:
  /** @brief Calculate elapsed time based on the selected unit
   * Computes the elapsed time from the recorded start and stop times and
   * updates the elapsed_time variable.
  */
  void calculate_elapsed();

  bool timer_started;                                        /*!< Set to true after tbp_start()*/
  std::string res_str;                                       /*!< Set resolution string*/
  time_res_t resolution;                                     /*!< Time resolution */
  std::chrono::high_resolution_clock::time_point start_time; /*!< Time point when tbp_start() was called */
  std::chrono::high_resolution_clock::time_point stop_time;  /*!< Time point when tbp_stop() was called */
  double elapsed_time;                                       /*!< Computed elapsed time */
};

} // profile
} // zendnnl

#endif