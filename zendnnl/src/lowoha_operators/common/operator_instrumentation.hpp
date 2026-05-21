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
//  (e.g., input validation) that run by default to surface contract
//  violations early, and can be turned off explicitly in production
//  hot paths where the per-call validation cost is unacceptable.
//
//  Controlled by a process-lifetime boolean initialized once from
//  the ZENDNNL_DIAGNOSTICS_ENABLE environment variable. The flag
//  defaults to ENABLED; setting the variable to "0" disables it,
//  in which case each method reduces to a single predicted-taken
//  branch with no validator body executed.
//
//  Profiling and logging are handled directly by their respective
//  optimized subsystems (is_profile_enabled(), apilog_info_enabled())
//  and do not require this gate.
//
//  Usage:
//    (unset, or set to anything other than "0")  # enabled (default)
//    export ZENDNNL_DIAGNOSTICS_ENABLE=0         # disable for near-zero overhead
// ═══════════════════════════════════════════════════════════════════════════
struct op_instrumentation {

  static bool is_enabled() {
    static const bool val = []() {
      const char *env = std::getenv("ZENDNNL_DIAGNOSTICS_ENABLE");
      // Default ON: only an explicit leading '0' disables diagnostics.
      return !env || env[0] != '0';
    }();
    return val;
  }

  template <typename Fn>
  static inline status_t validate(Fn &&fn) {
    if (__builtin_expect(is_enabled(), 1)) {
      return fn();
    }
    return status_t::success;
  }

};

} // namespace common
} // namespace zendnnl

#endif // LOWOHA_OPERATOR_INSTRUMENTATION_HPP
