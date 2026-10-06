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
inline int32_t resolve_num_threads(
        int32_t requested, int32_t current_max) noexcept {
    return (requested == 0) ? current_max : requested;
}

#if defined(_MSC_VER) && !defined(__clang__)
/**
 * @brief Per-thread team size recorded by thread_guard, MSVC only.
 *
 * vcomp is OpenMP 2.0 and has no task-scoped nthreads-var: omp_set_num_threads
 * is ignored inside an active parallel region, so a guard constructed there
 * cannot publish its count through the ICV. This thread_local is ZenDNNL-owned
 * storage that does carry it, with the same per-thread meaning.
 *
 * Not compiled on any other toolchain — those provide OpenMP 3.0 task-scoped
 * ICV semantics, so the ICV already is the per-thread home for this value.
 *
 * @return Reference to the calling thread's count; 0 means "unset".
 */
inline int32_t &nested_thread_budget() noexcept {
    static thread_local int32_t budget = 0;
    return budget;
}
#endif

/**
 * @brief The thread count currently in effect for the calling thread.
 *
 * Reads back what a thread_guard installed. Callers opening a nested parallel
 * region should pass this to a num_threads() clause rather than relying on the
 * region inheriting the ICV, which does not carry per-thread counts on vcomp.
 *
 * Outside MSVC this is exactly omp_get_max_threads().
 *
 * @return The active budget when one is set, else the nthreads-var ICV
 */
inline int32_t effective_num_threads() noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    const int32_t budget = nested_thread_budget();
    if (budget > 0) { return budget; }
#endif
    return static_cast<int32_t>(omp_get_max_threads());
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
 * @note The task-scoped ICV above is OpenMP 3.0. On MSVC (vcomp, OpenMP 2.0)
 *       omp_set_num_threads() is ignored inside an active parallel region, so
 *       the nested uses would silently keep the full outer team. There the
 *       guard also records the count in nested_thread_budget(). Read it back
 *       with effective_num_threads() — including in a num_threads() clause on
 *       the nested region, since that region will not inherit it.
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
#if defined(_MSC_VER) && !defined(__clang__)
        , old_budget_(nested_thread_budget())
#endif
        , modified_(desired != current_max) {
        if (modified_) {
            omp_set_num_threads(desired);
#if defined(_MSC_VER) && !defined(__clang__)
            nested_thread_budget() = desired;
#endif
        }
    }

    /**
     * @brief Construct by querying the thread count currently in effect.
     *
     * @param desired  Desired thread count for the scoped region
     */
    explicit thread_guard(int32_t desired) noexcept
        : thread_guard(desired, effective_num_threads()) {}

    ~thread_guard() noexcept {
        if (modified_) {
            omp_set_num_threads(old_);
#if defined(_MSC_VER) && !defined(__clang__)
            // Restore the previous budget verbatim, including the 0 "unset"
            // state: leaving a stale non-zero value here would make
            // effective_num_threads() shadow the live ICV for every later
            // caller on this thread.
            nested_thread_budget() = old_budget_;
#endif
        }
    }

    thread_guard(const thread_guard &) = delete;
    thread_guard &operator=(const thread_guard &) = delete;

private:
    const int32_t old_;
#if defined(_MSC_VER) && !defined(__clang__)
    const int32_t old_budget_;
#endif
    const bool modified_;
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
 * @note On MSVC (vcomp, OpenMP 2.0) the level count degrades to the boolean
 *       nest-var: a desired level >= 2 enables nesting but does not bound it
 *       to that depth.
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
        : old_(query_levels()), modified_(desired != old_) {
        if (modified_) apply_levels(desired);
    }

    ~scoped_active_levels() noexcept {
        if (modified_) apply_levels(old_);
    }

    scoped_active_levels(const scoped_active_levels &) = delete;
    scoped_active_levels &operator=(const scoped_active_levels &) = delete;

    // max-active-levels-var is OpenMP 3.0. MSVC's vcomp is OpenMP 2.0 and
    // exposes only the boolean nest-var, so levels collapse to on/off there:
    // any desired >= 2 enables nesting without capping its depth, and the
    // saved state round-trips as 1 or 2.
    //
    // Public so callers that must observe or restore the ICV outside a scope
    // (fixtures, tests) can do so through the same mapping the guard applies,
    // instead of calling the OpenMP 3.0 entry points directly.
    static int32_t query_levels() noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
        return omp_get_nested() ? 2 : 1;
#else
        return static_cast<int32_t>(omp_get_max_active_levels());
#endif
    }

    static void apply_levels(int32_t levels) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
        omp_set_nested(levels > 1 ? 1 : 0);
#else
        omp_set_max_active_levels(levels);
#endif
    }

private:
    const int32_t old_;
    const bool modified_;
};

} // namespace lowoha
} // namespace zendnnl

#endif
