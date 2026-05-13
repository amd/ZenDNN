/*******************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// @file detect_internal_alloc.hpp
/// @brief Shared Op1/Op2 internal-alloc buffer-source detection helper.
///
/// "Internal-alloc" is the fused-MoE pattern where the library, not the
/// caller, owns one or both intermediate buffers (Op1's gate+up output
/// `dst[]`, or Op2's down_proj output `fused.dst_down[]`).  Each side is
/// signalled by the caller passing either an empty vector or a vector
/// of all-null pointers in the active range; any non-null entry in the
/// active range means "caller owns that side's destination".
///
/// Three call sites historically open-coded this detection with
/// subtly different shapes:
///
///   1. `group_matmul_direct::validate_group_matmul_direct_inputs`
///      (Phase A, full sweep, log_error on mixed null/non-null,
///      gated on `fused_moe != nullptr` because the validator runs
///      for non-fused dispatch too)
///   2. `group_matmul_direct::group_matmul_direct(...)` always-on
///      inline guard (single-slot probe at `v[0]`, no log_error,
///      runs in production builds with diagnostics off)
///   3. `group_matmul_fused_moe::group_matmul_fused_moe_execute`
///      (full sweep, log_error on mixed, runs only when fused-MoE
///      is engaged so the `fused_moe_present` gate is implicit)
///
/// This header centralises the predicate.  Logging stays at the call
/// site so each location keeps its distinct error-message prefix
/// ("group_matmul_direct: fused_moe ..." vs
/// "group_matmul_fused_moe: ...") and the diagnostic-only call path
/// remains free of cross-TU logging dependencies.
///
/// Modes:
///
///   * `internal_alloc_mode::quick_o1` — single-slot probe at `v[0]`.
///     Used by the production inline guard.  Never reports mixed
///     state: the diagnostic full sweep catches that.  Cost: O(1).
///
///   * `internal_alloc_mode::sweep_active` — iterates
///     `[0, min(num_ops, v.size()))` and reports `status_t::failure`
///     when the active range is mixed null/non-null (a hard caller-
///     contract break).  Used by both diagnostic call sites.
///     Cost: O(num_ops).
///
/// Empty-vector semantics:
///   * `fused_moe_present == false` → `*out_internal = false`.
///     ("No fused-MoE → the question is moot; tell the caller's
///     internal-alloc gate not to engage.")
///   * `fused_moe_present == true && v.empty()` → `*out_internal = true`.
///     ("Caller is signalling library-managed via empty vector.")

#ifndef ZENDNNL_GROUP_MATMUL_DETECT_INTERNAL_ALLOC_HPP
#define ZENDNNL_GROUP_MATMUL_DETECT_INTERNAL_ALLOC_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#include "common/error_status.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace group_matmul_internal {

using zendnnl::error_handling::status_t;

enum class internal_alloc_mode {
  /// O(1) probe at `v[0]` only.  Treats empty vector and `v[0] ==
  /// nullptr` (when fused_moe_present) as internal.  Never reports
  /// mixed state; assumes a sibling diagnostic-mode call catches
  /// caller-contract breaks.  Used by the production inline guard.
  quick_o1,

  /// O(num_ops) sweep of `[0, min(num_ops, v.size()))`.  Detects mixed
  /// null/non-null active ranges and returns `status_t::failure` so
  /// the caller can `log_error` with its own prefix.
  sweep_active,
};

/// Detect whether `v` signals internal-alloc for one fused-MoE side.
///
/// @param v                 dst[] (Op1) or fused.dst_down[] (Op2) vector
/// @param num_ops           active matmul count (firing experts)
/// @param fused_moe_present caller knows fused-MoE is engaged
/// @param mode              `quick_o1` (production) or `sweep_active`
///                          (diagnostic).  See enum doc.
/// @param out_internal      [out] true iff library-managed for this side
///
/// @returns
///   `status_t::success` — `*out_internal` populated.
///   `status_t::failure` — sweep mode only; mixed null/non-null
///                          detected in the active range.  Caller
///                          should `log_error` with its prefix.
inline status_t detect_internal_alloc(const std::vector<void *> &v,
                                      size_t                      num_ops,
                                      bool                        fused_moe_present,
                                      internal_alloc_mode         mode,
                                      bool                       *out_internal) {
  if (!fused_moe_present) {
    *out_internal = false;
    return status_t::success;
  }
  if (v.empty()) {
    *out_internal = true;
    return status_t::success;
  }
  if (mode == internal_alloc_mode::quick_o1) {
    *out_internal = (v[0] == nullptr);
    return status_t::success;
  }

  // sweep_active
  bool any_null = false;
  bool any_nonnull = false;
  const size_t sweep = std::min<size_t>(num_ops, v.size());
  for (size_t i = 0; i < sweep; ++i) {
    if (v[i] == nullptr) any_null = true;
    else any_nonnull = true;
  }
  if (any_null && any_nonnull) {
    return status_t::failure;
  }
  *out_internal = any_null;
  return status_t::success;
}

} // namespace group_matmul_internal
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_DETECT_INTERNAL_ALLOC_HPP
