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
#ifndef _ZENDNNL_COMPAT_HPP_
#define _ZENDNNL_COMPAT_HPP_

/*
 * Cross-compiler portability shims for GCC/Clang function attributes that MSVC
 * (cl) does not support.
 *
 * The most important is per-function ISA targeting via
 * __attribute__((target(...))), which lets one translation unit hold functions
 * compiled for different instruction sets. MSVC has no per-function equivalent:
 * it compiles a whole translation unit for one ISA via /arch (ZenDNN builds the
 * kernels with /arch:AVX512, its Zen 4+ baseline). So on MSVC the target
 * attribute expands to nothing and the ISA comes from the command-line flag.
 *
 * These are genuine COMPILER-language features (not platform/ABI), so the gate
 * is (_MSC_VER && !__clang__) - the real Microsoft compiler only. clang-cl also
 * defines _MSC_VER but supports __attribute__, so it must take the GCC/Clang
 * branch (this mirrors the guard style used elsewhere for MSVC vs clang-cl).
 *
 * | Macro                       | GCC / Clang                              | MSVC (cl)             |
 * |-----------------------------|------------------------------------------|-----------------------|
 * | ZENDNNL_TARGET(isa)         | __attribute__((target(isa)))             | (empty)               |
 * | ZENDNNL_INLINE_TARGET(isa)  | __attribute__((always_inline,target(isa)))| __forceinline        |
 * | ZENDNNL_TARGET_NOINLINE(isa)| __attribute__((target(isa),noinline))    | __declspec(noinline)  |
 * | ZENDNNL_ALWAYS_INLINE       | __attribute__((always_inline))           | __forceinline         |
 * | ZENDNNL_LAMBDA_ALWAYS_INLINE| __attribute__((always_inline))           | (empty)               |
 * | ZENDNNL_NOINLINE            | __attribute__((noinline))                | __declspec(noinline)  |
 * | ZENDNNL_HOT                 | __attribute__((hot))                     | (empty)               |
 * | ZENDNNL_FLATTEN             | __attribute__((flatten))                 | (empty)               |
 * | ZENDNNL_EXPECT(e, v)        | __builtin_expect((e), (v))               | (e)                   |
 *
 * ZENDNNL_LAMBDA_ALWAYS_INLINE is the lambda-safe form of ZENDNNL_ALWAYS_INLINE.
 * A lambda's inline attribute sits in the TRAILING position (after the parameter
 * list), where GCC/Clang accept __attribute__((always_inline)) but MSVC rejects
 * __forceinline (MSVC's lambda spelling [[msvc::forceinline]] is a LEADING
 * attribute - an incompatible position). The MSVC expansion is therefore empty;
 * force-inline is only a hint, so behaviour is unchanged.
 */

#if defined(_MSC_VER) && !defined(__clang__)

#define ZENDNNL_TARGET(isa)
#define ZENDNNL_INLINE_TARGET(isa) __forceinline
#define ZENDNNL_TARGET_NOINLINE(isa) __declspec(noinline)
#define ZENDNNL_ALWAYS_INLINE __forceinline
#define ZENDNNL_LAMBDA_ALWAYS_INLINE
#define ZENDNNL_NOINLINE __declspec(noinline)
#define ZENDNNL_HOT
#define ZENDNNL_FLATTEN
#define ZENDNNL_EXPECT(expr, val) (expr)

#else

#define ZENDNNL_TARGET(isa) __attribute__((target(isa)))
#define ZENDNNL_INLINE_TARGET(isa) __attribute__((always_inline, target(isa)))
#define ZENDNNL_TARGET_NOINLINE(isa) __attribute__((target(isa), noinline))
#define ZENDNNL_ALWAYS_INLINE __attribute__((always_inline))
#define ZENDNNL_LAMBDA_ALWAYS_INLINE __attribute__((always_inline))
#define ZENDNNL_NOINLINE __attribute__((noinline))
#define ZENDNNL_HOT __attribute__((hot))
#define ZENDNNL_FLATTEN __attribute__((flatten))
#define ZENDNNL_EXPECT(expr, val) __builtin_expect((expr), (val))

#endif

/* ---------------------------------------------------------------------------
 * OpenMP `collapse` clause neutralization on MSVC.
 *
 * MSVC's OpenMP (including /openmp:llvm with the bundled libomp) mishandles the
 * runtime collapsed loop-nest path: a `#pragma omp parallel for collapse(n)`
 * over 64-bit loop counters divides by zero inside libomp
 * (__kmpc_process_loop_nest_rectang -> kmp_calculate_trip_count) and crashes the
 * process once more than one thread participates. Following oneDNN
 * (src/common/dnnl_thread.hpp), strip the `collapse` clause on the genuine MSVC
 * compiler so those loops degrade to a plain parallel-for over the outer loop
 * instead of crashing. `collapse` is only ever an OpenMP clause here (never a
 * C++ identifier), so redefining it as a function-like macro is safe; MSVC
 * expands macros in `#pragma omp` clauses, so `collapse(2)` disappears.
 *
 * Compiler-language concern -> gated on (_MSC_VER && !__clang__) (clang-cl and
 * MinGW keep the real clause). This header is force-included on MSVC (see the
 * /FI flag in zendnnl/CMakeLists.txt) so the macro is active in every
 * translation unit before any OpenMP pragma. Loops whose outer extent is too
 * small to parallelize well should instead be hand-flattened into a single
 * parallel loop (see oneDNN's parallel_nd/for_nd) — a perf follow-up, not a
 * correctness concern.
 * ------------------------------------------------------------------------- */
#if defined(_MSC_VER) && !defined(__clang__)
#define collapse(x)
#endif

/* ---------------------------------------------------------------------------
 * Portable aligned allocation.
 *
 * Platform/CRT feature -> gated on _WIN32 (works for MSVC, clang-cl, MinGW).
 * Windows uses _aligned_malloc/_aligned_free (note the SWAPPED argument order,
 * and that _aligned_malloc'd memory MUST be released with _aligned_free, never
 * plain free()); other platforms use the C11 aligned_alloc / posix_memalign,
 * released with free().
 * ------------------------------------------------------------------------- */
#include <cstddef>
#include <cstdlib>
#if defined(_WIN32)
#include <cerrno>
#include <malloc.h>
#endif

static inline void *zendnnl_aligned_alloc(size_t alignment, size_t size) {
    // Round the request up to a multiple of `alignment`. The C11 std::aligned_alloc
    // used on the non-Windows path requires `size` to be an integral multiple of
    // `alignment` (it may otherwise return nullptr - e.g. glibc rejects it),
    // whereas Windows _aligned_malloc imposes no such rule. Rounding here lets
    // callers pass any size and get identical behavior on both platforms; it only
    // ever over-allocates by < alignment bytes. (`alignment` is always a power of
    // two - a requirement of both underlying allocators.)
    if (alignment != 0) { size = (size + alignment - 1) & ~(alignment - 1); }
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

static inline void zendnnl_aligned_free(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

static inline int zendnnl_posix_memalign(
        void **memptr, size_t alignment, size_t size) {
#if defined(_WIN32)
    void *p = _aligned_malloc(size, alignment);
    if (p == nullptr) { return errno ? errno : ENOMEM; }
    *memptr = p;
    return 0;
#else
    return ::posix_memalign(memptr, alignment, size);
#endif
}

/* ---------------------------------------------------------------------------
 * Checked arithmetic. Compiler-language feature -> gated on _MSC_VER (real cl
 * has no __builtin_*_overflow; clang/clang-cl do). The MSVC fallbacks assume
 * unsigned operands (the sizes/counts these guard), which is where ZenDNN uses
 * them.
 * ------------------------------------------------------------------------- */
template <typename T>
static inline bool zendnnl_mul_overflow(T a, T b, T *result) {
#if defined(_MSC_VER) && !defined(__clang__)
    *result = static_cast<T>(a * b);
    return (a != 0) && ((*result / a) != b);
#else
    return __builtin_mul_overflow(a, b, result);
#endif
}

template <typename T>
static inline bool zendnnl_add_overflow(T a, T b, T *result) {
#if defined(_MSC_VER) && !defined(__clang__)
    *result = static_cast<T>(a + b);
    return (*result < a);
#else
    return __builtin_add_overflow(a, b, result);
#endif
}

#endif // _ZENDNNL_COMPAT_HPP_
