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
#ifndef _ZENDNNL_API_HPP_
#define _ZENDNNL_API_HPP_

/*
 * ZENDNNL_API - cross-compiler symbol export/visibility decoration.
 *
 * A pair of platform helper macros, gated by a two-level switch.
 *
 *   Windows (MSVC, clang-cl, MinGW, Cygwin): shared-library symbols must be
 *   explicitly exported with __declspec(dllexport) while building the DLL and
 *   imported with __declspec(dllimport) while consuming it.
 *
 *   GCC/Clang (ELF): map to the default-visibility attribute, which is
 *   harmless when the build does not hide symbols by default and correct if
 *   it ever compiles with -fvisibility=hidden.
 *
 * The Windows branch is gated on _WIN32 (the platform/ABI) rather than
 * _MSC_VER (one specific compiler). DLL export/import is a PE/COFF ABI
 * property, and __declspec(dllexport/dllimport) is understood by every
 * Windows-targeting compiler - MSVC, clang-cl, and MinGW GCC - not only by
 * cl. _MSC_VER would exclude MinGW GCC (which defines _WIN32 and __GNUC__ but
 * not _MSC_VER); such a build would then wrongly fall through to the ELF
 * visibility branch and emit no export decoration for a Windows DLL. Gating on
 * _WIN32 keeps the macro correct across all Windows toolchains with no further
 * change (clang-cl also defines _WIN32, so it is covered as well).
 *
 * Two-level gate:
 *   - ZENDNNL_DLL         defined => "DLL mode" (producing or consuming a
 *                         shared library). When absent (e.g. the static
 *                         archive, or any consumer that links it), ZENDNNL_API
 *                         is empty, so a static consumer never inherits an
 *                         accidental dllimport.
 *   - ZENDNNL_DLL_EXPORTS defined (in addition) => we are BUILDING the DLL, so
 *                         export; otherwise we are consuming it, so import.
 * Both are set by the shared-library CMake target; consumers that link the DLL
 * define only ZENDNNL_DLL.
 */

#if defined(_WIN32) || defined(__CYGWIN__)
#define ZENDNNL_HELPER_DLL_EXPORT __declspec(dllexport)
#define ZENDNNL_HELPER_DLL_IMPORT __declspec(dllimport)
#elif defined(__GNUC__) && (__GNUC__ >= 4)
#define ZENDNNL_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#define ZENDNNL_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#else
#define ZENDNNL_HELPER_DLL_EXPORT
#define ZENDNNL_HELPER_DLL_IMPORT
#endif

#if defined(ZENDNNL_DLL)
#if defined(ZENDNNL_DLL_EXPORTS)
#define ZENDNNL_API ZENDNNL_HELPER_DLL_EXPORT
#else
#define ZENDNNL_API ZENDNNL_HELPER_DLL_IMPORT
#endif
#else
#define ZENDNNL_API
#endif

#endif // _ZENDNNL_API_HPP_
