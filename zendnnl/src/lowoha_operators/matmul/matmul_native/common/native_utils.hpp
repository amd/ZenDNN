/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef MATMUL_NATIVE_COMMON_NATIVE_UTILS_HPP
#define MATMUL_NATIVE_COMMON_NATIVE_UTILS_HPP

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "common/error_status.hpp"
#include "common/zendnnl_compat.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Frees memory from zendnnl_aligned_alloc.
struct native_aligned_free_t {
    void operator()(void *p) const { zendnnl_aligned_free(p); }
};

/// st for a call turned away, with *why (when why is not null) naming the
/// reason.
inline zendnnl::error_handling::status_t turned_away(
        zendnnl::error_handling::status_t st, const char *reason,
        const char **why) {
    if (why != nullptr) *why = reason;
    return st;
}

/// a / b rounded up, for a >= 0 and b > 0; no intermediate overflows.
inline int ceil_div(int a, int b) {
    return a / b + (a % b != 0);
}

/// range(fn, i0, i1) on each thread of a region of threads threads, over the
/// iterations i0 .. i1 of 0 .. n that schedule(static) gives the thread.
void native_for_parallel(
        int n, int threads, void (*range)(void *, int, int), void *fn);

/// (*fn)(i0) .. (*fn)(i1 - 1) for the Fn at fn on the calling thread: the
/// one copy of the body both loops of native_for run.
template <typename Fn>
ZENDNNL_NOINLINE void native_for_range(void *fn, int i0, int i1) {
    Fn &f = *static_cast<Fn *>(fn);
    for (int i = i0; i < i1; ++i)
        f(i);
}

/// fn(0) .. fn(n - 1): with parallel and threads > 1 in a region of threads
/// threads (native_for_parallel); else on the calling thread with no region
/// of its own (a region whose if clause is false still makes the runtime
/// start a team of one), so inside a caller's region on that region's
/// thread.  fn must not depend on the OpenMP thread number, team or level.
template <typename Fn>
inline void native_for(int n, int threads, bool parallel, Fn &&fn) {
    using F = std::remove_reference_t<Fn>;
    void *const f = const_cast<void *>(static_cast<const void *>(&fn));
    if (!parallel || threads <= 1)
        native_for_range<F>(f, 0, n);
    else
        native_for_parallel(n, threads, &native_for_range<F>, f);
}

/// The K passes of at most kc values over K (K >= 0, kc >= 1) in order:
/// next() moves to the next pass, k0 its first value and kl its length,
/// and is false past the last.  A pass starts where the one before ends,
/// so no start passes K, at any K an int holds.
struct native_k_passes_t {
    int K = 0, kc = 1;
    int k0 = 0, kl = 0;
    bool next() {
        k0 += kl;
        kl = std::min(kc, K - k0);
        return k0 < K;
    }
};

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_UTILS_HPP
