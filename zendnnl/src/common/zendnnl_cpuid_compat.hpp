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
#ifndef _ZENDNNL_CPUID_COMPAT_HPP_
#define _ZENDNNL_CPUID_COMPAT_HPP_

/*
 * Portable CPUID access.
 *
 * GCC/Clang expose the raw CPUID instruction through <cpuid.h> as the
 * lvalue-filling macros __cpuid(level, a, b, c, d) and
 * __cpuid_count(level, count, a, b, c, d). MSVC has no <cpuid.h>; it exposes
 * __cpuid / __cpuidex in <intrin.h> with a different (array-filling) signature.
 *
 * Gating: use <intrin.h>/__cpuidex whenever _MSC_VER is defined, which covers
 * BOTH real MSVC (cl) and clang-cl. clang-cl targets the MSVC headers/CRT
 * (no <cpuid.h>) yet provides __cpuidex via <intrin.h>, so it must take the
 * MSVC path here. This intentionally differs from the language-feature gates
 * elsewhere (`_MSC_VER && !__clang__`, which route clang-cl down the GCC/Clang
 * branch): clang-cl supports __attribute__/__builtin_*, but not the GNU
 * <cpuid.h> layout. Only non-MSVC GCC/Clang (e.g. Linux) uses <cpuid.h>.
 *
 * Rather than macro-redefine the reserved __cpuid* spellings, expose thin
 * wrapper functions that behave identically on both toolchains, so the shared
 * detection code reads the same everywhere.
 */
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

static inline void zendnnl_cpuid_count(unsigned level, unsigned count,
        unsigned &a, unsigned &b, unsigned &c, unsigned &d) {
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, static_cast<int>(level), static_cast<int>(count));
    a = static_cast<unsigned>(regs[0]);
    b = static_cast<unsigned>(regs[1]);
    c = static_cast<unsigned>(regs[2]);
    d = static_cast<unsigned>(regs[3]);
#else
    __cpuid_count(level, count, a, b, c, d);
#endif
}

static inline void zendnnl_cpuid(
        unsigned level, unsigned &a, unsigned &b, unsigned &c, unsigned &d) {
    // Subleaf 0 is the correct/safe input for the leaves that ignore ECX.
    zendnnl_cpuid_count(level, 0u, a, b, c, d);
}

#endif // _ZENDNNL_CPUID_COMPAT_HPP_
