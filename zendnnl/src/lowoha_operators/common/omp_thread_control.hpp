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

#ifndef LOWOHA_OMP_THREAD_CONTROL_HPP
#define LOWOHA_OMP_THREAD_CONTROL_HPP

#include <cstdint>
#include <omp.h>

namespace zendnnl {
namespace lowoha {

/**
 * @brief Resolves a thread count from an operator params struct.
 *
 * Resolves the int32_t num_threads field used by _params structs for
 * OpenMP APIs. A value of 0 is the "auto" sentinel used by external
 * callers (e.g. zentorch) to indicate that the operator should use the
 * current OMP thread limit.
 *
 * @param requested  Thread count from _params struct (0 = auto)
 * @param current_max  Value previously captured via thread_guard::max_threads()
 * @return current_max when requested == 0, otherwise requested
 */
inline int32_t resolve_num_threads(int32_t requested,
                                   int32_t current_max) noexcept {
    return (requested == 0) ? current_max : requested;
}

/**
 * @brief RAII guard for the OpenMP nthreads-var ICV.
 *
 * Sets omp_set_num_threads(desired) on construction and restores the
 * previous value on destruction. Both calls are elided when the desired
 * count already matches the current setting.
 *
 * The two-argument constructor accepts a pre-captured current_max to avoid
 * a redundant omp_get_max_threads() call when the caller has already
 * queried the ICV (e.g. for resolve_num_threads).
 *
 * @note omp_set_num_threads() modifies a task-scoped ICV. When constructed
 *       in the initial task region (outside any parallel region), it controls
 *       the thread count for subsequent parallel regions — the common case.
 *       When constructed inside an active parallel region, it sets the ICV
 *       for that task only, controlling nested parallelism per outer thread.
 *       Both uses are valid:
 *       - Top-level: use the two-arg constructor with max_threads().
 *       - Nested (no loop): the single-arg constructor is convenient;
 *         it captures the per-task ICV via omp_get_max_threads() internally.
 *       - Nested (inside a loop): prefer the two-arg constructor with a
 *         pre-captured per-task ICV (via omp_get_max_threads() before the
 *         loop) to avoid repeated OMP API calls per iteration.
 *
 * @example
 *   {
 *       const int32_t omp_mt = thread_guard::max_threads();
 *       const int32_t nt = resolve_num_threads(params.num_threads, omp_mt);
 *       thread_guard guard(nt, omp_mt);
 *       // ... parallel work using nt threads ...
 *   }  // nthreads-var restored to omp_mt
 */
class thread_guard final {
public:
    /**
     * @brief Returns the cached OpenMP nthreads-var ICV baseline.
     *
     * Captures omp_get_max_threads() on first invocation and returns
     * the cached value on all subsequent calls. This eliminates the
     * OMP runtime call from operator hot paths.
     *
     * @pre OMP_NUM_THREADS must be set before the first ZenDNNL API call.
     *      The nthreads-var ICV must not be modified externally between
     *      operator invocations (thread_guard restores it on scope exit).
     *
     * @return Cached baseline value of the nthreads-var ICV
     */
    static int32_t max_threads() noexcept {
        static const int32_t cached = omp_get_max_threads();
        return cached;
    }

    /**
     * @brief Construct with a pre-captured current max thread count.
     *
     * @param desired      Desired thread count for the scoped region
     * @param current_max  The nthreads-var ICV to compare against and
     *                     restore to on destruction. Use max_threads()
     *                     at top-level, or omp_get_max_threads() for
     *                     the per-task ICV inside a parallel region.
     */
    thread_guard(int32_t desired, int32_t current_max) noexcept
        : old_(current_max)
        , modified_(desired != current_max) {
        if (modified_)
            omp_set_num_threads(desired);
    }

    /**
     * @brief Construct by querying the current nthreads-var ICV internally.
     *
     * @param desired  Desired thread count for the scoped region
     */
    explicit thread_guard(int32_t desired) noexcept
        : thread_guard(desired, omp_get_max_threads()) {}

    ~thread_guard() noexcept {
        if (modified_)
            omp_set_num_threads(old_);
    }

    thread_guard(const thread_guard&) = delete;
    thread_guard& operator=(const thread_guard&) = delete;

private:
    const int32_t old_;
    const bool    modified_;
};

/**
 * @brief RAII guard for the OpenMP max-active-levels-var ICV.
 *
 * Sets omp_set_max_active_levels(desired) on construction and restores
 * the previous value on destruction. Both calls are elided when the
 * desired level already matches the current setting.
 *
 * @note Must be constructed in the initial task region (outside any active
 *       parallel region), same as thread_guard.
 *
 * @example
 *   {
 *       scoped_active_levels guard(2);  // Enable 2 levels of nesting
 *       #pragma omp parallel num_threads(outer)
 *       {
 *           #pragma omp parallel num_threads(inner)
 *           { ... }  // nested work
 *       }
 *   }  // max-active-levels restored here
 */
class scoped_active_levels final {
public:
    /**
     * @brief Construct by querying the current max-active-levels-var ICV.
     *
     * @param desired  Desired maximum number of active nested parallel regions
     */
    explicit scoped_active_levels(int32_t desired) noexcept
        : old_(omp_get_max_active_levels())
        , modified_(desired != old_) {
        if (modified_)
            omp_set_max_active_levels(desired);
    }

    ~scoped_active_levels() noexcept {
        if (modified_)
            omp_set_max_active_levels(old_);
    }

    scoped_active_levels(const scoped_active_levels&) = delete;
    scoped_active_levels& operator=(const scoped_active_levels&) = delete;

private:
    const int32_t old_;
    const bool    modified_;
};

} // namespace lowoha
} // namespace zendnnl

#endif
