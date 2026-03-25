/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef LOWOHA_OPERATOR_INSTRUMENTATION_HPP
#define LOWOHA_OPERATOR_INSTRUMENTATION_HPP

#include <cstdlib>

#include "common/error_status.hpp"

namespace zendnnl {
namespace common {

// ═══════════════════════════════════════════════════════════════════════════
//  Operator instrumentation — runtime-gated diagnostic layer.
//
//  Provides a lightweight gate for expensive diagnostic operations
//  (e.g., input validation) that should run during development but
//  can be skipped in production for zero overhead.
//
//  Controlled by a process-lifetime boolean initialized once from
//  the ZENDNNL_DIAGNOSTICS_ENABLE environment variable. When disabled
//  (default), each method reduces to a single predicted-not-taken
//  branch.
//
//  Profiling and logging are handled directly by their respective
//  optimized subsystems (is_profile_enabled(), apilog_info_enabled())
//  and do not require this gate.
//
//  Usage:
//    export ZENDNNL_DIAGNOSTICS_ENABLE=1   # enable before launching application
//    (unset or =0)                  # disabled — near-zero overhead
// ═══════════════════════════════════════════════════════════════════════════
struct op_instrumentation {

  static bool is_enabled() {
    static const bool val = []() {
      const char *env = std::getenv("ZENDNNL_DIAGNOSTICS_ENABLE");
      return env && env[0] == '1';
    }();
    return val;
  }

  template <typename Fn>
  static inline status_t validate(Fn &&fn) {
    if (__builtin_expect(is_enabled(), 0)) {
      return fn();
    }
    return status_t::success;
  }

};

} // namespace common
} // namespace zendnnl

#endif // LOWOHA_OPERATOR_INSTRUMENTATION_HPP
