/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// Per-thread aligned scratch buffer shared by the M-tile and N-tile
/// executors.

#ifndef ZENDNNL_GROUP_MATMUL_PER_THREAD_SCRATCH_HPP
#define ZENDNNL_GROUP_MATMUL_PER_THREAD_SCRATCH_HPP

#include <cstddef>

#include "common/zendnnl_compat.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// =====================================================================
// Section A.0 — AlignedTileScratch
// =====================================================================
//
// Per-thread aligned heap buffer used by the tight-fused-epilogue
// path of `execute_tile()`.  When `fused_epilogue=swiglu` AND the caller's
// dst is a tight [M, I]-layout buffer (ldc < N), the classic
// matmul-then-in-place-compact pattern no longer fits (matmul writes
// 2I cols but dst only has I cols per row).  In that case `execute_tile()`
// switches to a per-thread scratch + out-of-place activation flow:
//
//   1. matmul the thread's N-tile slice into a thread-local scratch
//      buffer (wide, `n_tile` cols, ldc=n_tile),
//   2. run `apply_swiglu_oai_tile_rows_out_of_place()` from that scratch into
//      the caller's tight dst at halved col offset `col_start / 2`.
//
// This is barrier-free: each thread writes to its own scratch then to
// a disjoint column range of the caller's tight dst — no cross-thread
// reads needed inside the fused-epilogue step.
//
// Lifetime: `static thread_local` inside the OMP region; monotonically
// grows to the high-water mark this thread has seen.  Freed by the
// destructor on thread exit.  Per-thread, so no contention.
//
// The buffer is 64-byte aligned so the custom kernel's `vmovdqu`
// reads start on a cache-line boundary.
//
// Move/copy operations are deleted because the struct owns `buf` via
// `std::free` in the destructor — accidental copying would leave two
// instances pointing at the same allocation and double-free on
// destruction.  Production use is always `static thread_local` inside
// an OMP region (one instance per worker, fixed lifetime); the delete
// catches misuse if a future caller or test instantiates it as a
// stack variable and accidentally returns / passes it by value.
struct AlignedTileScratch {
    void *buf = nullptr;
    size_t cap = 0;
    AlignedTileScratch() = default;
    ~AlignedTileScratch() { zendnnl_aligned_free(buf); }
    AlignedTileScratch(const AlignedTileScratch &) = delete;
    AlignedTileScratch &operator=(const AlignedTileScratch &) = delete;
    AlignedTileScratch(AlignedTileScratch &&) = delete;
    AlignedTileScratch &operator=(AlignedTileScratch &&) = delete;
};

// Grow a per-thread scratch to at least `need` bytes, 64-byte aligned.
// Returns false on alloc failure (caller signals via the abort_code atomic
// + post-OMP-region check).
inline bool grow_scratch(AlignedTileScratch &s, size_t need) {
    if (need <= s.cap) return true;
    zendnnl_aligned_free(s.buf);
    s.buf = nullptr;
    s.cap = 0;
    void *tmp = nullptr;
    if (zendnnl_posix_memalign(&tmp, 64, need) != 0 || tmp == nullptr)
        return false;
    s.buf = tmp;
    s.cap = need;
    return true;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_PER_THREAD_SCRATCH_HPP
