---
name: op-development-optimization-windows
description: >-
  Windows platform file for developing or optimizing a ZenDNN operator, and
  the guardrails for keeping the native Windows (MSVC / cl >= 19.43) build
  working when editing C++ or CMake. Use together with
  op-development-optimization on Windows, and alongside any change to the
  zendnnl library, kernels, dependency CMake, or gtests. Covers portable
  aligned alloc/free pairing, GNU keywords and attributes, LLP64 integer
  widths, the OpenMP 2.0 limits of MSVC's vcomp runtime, DLL exports
  (ZENDNNL_API), CRT (/MD) linking, the Windows SDK, and the /FI compat
  header.
version: 1.1.0
---

# Windows

Platform file for operator development and optimization on Windows. The shared
procedure is in the parent [SUBSKILL.md](../SUBSKILL.md).

Windows-only optimization tricks are not written here yet. Do not invent any.

The rest of this file is the MSVC portability checklist. Apply it on every
C++ or CMake edit, including kernels, gtests, and dependency CMake. Operator
review on Windows uses this same file.

## Windows / MSVC portability guardrails

ZenDNN builds natively with MSVC (`cl` ≥ 19.43, native Windows ABI) in addition to
Linux GCC/Clang. A Linux/Clang-only change easily breaks the Windows build:
`cl` rejects several GNU-isms and differs on integer widths, CRT, OpenMP, and
DLL exports. This skill lists the recurring breakages and their fixes so a change
stays green on **both** paths. Apply it whenever you touch memory allocation,
intrinsics/attributes, integer/size types, OpenMP loops, dependency CMake, or the
public API surface.

## Golden rules
1. **Keep Linux behavior identical.** Gate MSVC-only *source* changes on
   `#if defined(_MSC_VER) && !defined(__clang__)`; gate platform/CRT/OS changes on
   `_WIN32`. Never alter the GCC/Clang code path.
2. **Fix real bugs cross-platform, unguarded.** If the port exposes a genuine
   defect (out-of-bounds read, mismatched free), fix it for all platforms — do
   NOT `_WIN32`-guard a correctness fix. Windows tends to *crash* where Linux
   silently over-reads into mapped heap.
3. **Prefer the portable wrapper/macro over a raw GNU construct** — most already
   exist in `common/zendnnl_compat.hpp`, which is force-included into every
   translation unit via `/FI` on the MSVC build.

## Review checklist (scan any C++/CMake diff for these)
- [ ] No `aligned_alloc(` and no raw `free`/`std::free` on an aligned buffer — use the wrappers, and pair free with alloc.
- [ ] No GNU `__restrict__`, `__attribute__((...))`, `__attribute__((target(...)))`, or GNU inline `asm`.
- [ ] No `long` / `unsigned long` for a size/count/dim, or in a function-pointer typedef that must match a 64-bit ABI.
- [ ] Any new `#pragma omp ... collapse(n)` has a flattened MSVC path.
- [ ] OpenMP `for` loop counters are signed (`int64_t`), never unsigned or `size_t`.
- [ ] No `reduction(min:…)` / `reduction(max:…)` without the MSVC per-thread fallback.
- [ ] Nesting is set through `scoped_active_levels`, not `omp_get/set_max_active_levels`.
- [ ] A nested parallel region under a `thread_guard` passes `effective_num_threads()` to `num_threads()`.
- [ ] Any new public `*_direct` / API entry point is stamped `ZENDNNL_API`.
- [ ] No captureless lambda reading a non-`static` `constexpr` local; no VLA.
- [ ] New dependency CMake links `/MD` and handles Windows `.lib` byproducts.

## Rules by category

### Memory: aligned alloc/free must be paired
- `aligned_alloc` (C11) is absent from the MSVC CRT (`C3861: identifier not
  found`). Use `zendnnl_aligned_alloc(alignment, size)` /
  `zendnnl_aligned_free(ptr)` from `common/zendnnl_compat.hpp`
  (`_aligned_malloc`/`_aligned_free` on Windows, `std::aligned_alloc`/`std::free`
  elsewhere).
- **Pair the free with the alloc.** `_aligned_free` on `malloc`/`calloc`/`new`
  memory — or `free`/`delete` on `_aligned_malloc` memory — corrupts the Windows
  heap → an `0xC0000005` access violation or a cascading, misleading
  `resource deadlock would occur` from later `std::mutex` use.
- The `lru_cache_t` evictor releases cached pointer values with
  `zendnnl_aligned_free`, so **every value stored in an `lru_cache_t` must be
  `zendnnl_aligned_alloc`'d**. A `std::calloc`'d holder in that cache was the
  post-op metadata heap-corruption bug — align the allocation instead.
- `aligned_alloc`/`std::aligned_alloc` require `size % alignment == 0`.
  `sizeof(T)` is always a multiple of `alignof(T)`, so
  `zendnnl_aligned_alloc(alignof(T), sizeof(T))` is safe. Aligned storage is not
  zeroed — `memset` if you relied on `calloc`'s zero-init.

### GNU keywords, attributes, intrinsics
| GNU construct | MSVC symptom | Fix |
|---|---|---|
| `__restrict__` | `C3646` unknown override specifier / `C2146` missing `)` | use `__restrict` (accepted by cl, gcc, clang) |
| `__attribute__((...))`, `__attribute__((target("...")))` | `C3646` / `__target__` not found | use the compat attribute macros; drop target-clones on MSVC |
| GNU inline `asm(...)` | not supported by `cl` | provide a portable intrinsics path (e.g. AVX-512 intrinsics for the fp32 GEMV microkernel) |
| `-mrdrnd` and other `-m<isa>` flags | `D9002 ignoring unknown option` | use `/arch:AVX512` + `_MSC_VER`-guarded intrinsics |

### Integer widths — LLP64
Win64 is **LLP64**: `long`/`unsigned long` are **32-bit** (they are 64-bit on
Linux LP64). Never use `long` for a size/count/dim, nor in a function-pointer
type that must match a 64-bit ABI:
- Size/return types: use `size_t`, `int64_t`/`uint64_t`, or the dependency's own
  type. The reorder buffer-size function-pointer typedef had to change
  `long unsigned int` → `msz_t` (AOCL-DLP's 64-bit size type) in `aocl_kernel.hpp`.
- oneDNN dims: `std::vector<int64_t>`, not `std::vector<long>` — `dnnl::memory::dims`
  is `int64_t`; `long` truncates under MSVC. `int64_t` is correct on Linux too.

### OpenMP: MSVC builds use vcomp (OpenMP 2.0)
On Windows, ZenDNN builds with MSVC's own OpenMP runtime, vcomp (`/openmp`,
which `OpenMP::OpenMP_CXX` carries). vcomp implements only OpenMP 2.0, so any
OpenMP 3.0+ construct needs an MSVC path gated on
`#if defined(_MSC_VER) && !defined(__clang__)` (clang-cl keeps full OpenMP):
- **One runtime per process.** Do not pin ZenDNN or a dependency it builds to
  LLVM's libomp (`OpenMP_RUNTIME_MSVC=llvm`, `/openmp:llvm`). Two OpenMP
  runtimes in one process corrupt oneDNN's threading: wrong brgemm results at 2
  threads and out-of-bounds reads in the AVX-512 kernels at higher counts.
- **Signed loop counters.** vcomp rejects an unsigned `omp for` index
  (`C3016`). Use `int64_t` and cast the bound: `i < static_cast<int64_t>(n)`.
- **No `collapse`** (OpenMP 3.0). `zendnnl_compat.hpp` defines `collapse(x)` as
  an empty macro on MSVC (applied everywhere via `/FI`), so a `collapse(n)` nest
  runs parallel only over the *outer* loop — for a GEMV (M=1) or batch=1 that is
  a single core, which caused nightly gtest timeouts. **When you add a
  `collapse` nest, also flatten it** on MSVC: one parallel-for over the whole
  index space, decode the sub-indices inside. Leave the Linux `collapse` path
  unchanged.
- **No `min`/`max` reductions** (OpenMP 3.1, `C7660`). Keep per-thread extremes
  inside the parallel region and combine them once per thread in an
  `omp critical`, as `compute_dynamic_quant_params` in `lowoha_reorder_utils.cpp`
  does.
- **No `max_active_levels`** (OpenMP 3.0). Set nesting through
  `scoped_active_levels`, or its `query_levels()` / `apply_levels()` in fixtures
  and tests; on MSVC they map to vcomp's on/off nesting. Don't call
  `omp_get/set_max_active_levels` directly.
- **`omp_set_num_threads` is ignored inside a parallel region.** `thread_guard`
  also records its count in a `thread_local` budget on MSVC. Read it back with
  `effective_num_threads()` and pass that to a nested region's `num_threads()`
  clause, since the region does not inherit it on vcomp.

### Lambdas, VLAs
- A captureless lambda reading a `constexpr` *local* → `C3493`. Make the local
  `static constexpr` (static storage needs no capture; same value, no runtime cost).
- `always_inline` on a lambda → `C3260`; use `ZENDNNL_LAMBDA_ALWAYS_INLINE`.
- Variable-length arrays → `C2131`; use `std::vector`.

### CRT / environment
- `setenv`/`unsetenv` are missing from the MSVC CRT → `_putenv_s` shim. Gate it on
  **`_MSC_VER`, not `_WIN32`**: MinGW/Clang-on-Windows already provide the POSIX
  functions and the shim would clash with them.
- Link everything against the **dynamic CRT (`/MD`)**. Static-CRT (`/MT`) objects
  folded into the `/MD` archive/DLL cause `LNK2038`/`LNK4098`. Force dependencies
  to `/MD` (`gtest_force_shared_crt=ON`, `AU_STATIC_FORCE_CRT_MD=ON`, …).

### DLL exports
`ZENDNNL_API` (`common/zendnnl_api.hpp`) expands to
`__declspec(dllexport/dllimport)` only when `ZENDNNL_DLL` is defined (the shared
target sets it). Stamp every public `.cpp`-defined free-function entry point
(`*_direct`, `size_of`, `dtype_info`, …) with `ZENDNNL_API` and include
`common/zendnnl_api.hpp`. On Windows the shared target also sets
`WINDOWS_EXPORT_ALL_SYMBOLS`, so in-tree gtests, BenchDNN and examples can call
internal functions; that exports functions only, so exported **data** still
needs `ZENDNNL_API`. Do not export classes/types (avoids `C4251` STL-ABI
exposure). Static consumers are unaffected because `ZENDNNL_API` expands to
nothing without `ZENDNNL_DLL`.

### Windows SDK specifics
- The `_SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` union is anonymous: access
  `.Processor` / `.Cache` / `.Group` directly, **not** `.u.Processor`. Do NOT pass
  `/DNONAMELESSUNION` (it names the union and breaks direct access → `C2039`).

## Backends on Windows (dependency-level)
- **libxsmm**: bf16 matmul crashes on Windows (JIT stack overflow, or the
  reference-GEMM fallback over-reads). The `can_use_libxsmm` guard rejects all
  `src=bf16` on Windows so it falls back to AOCL-DLP. `_WIN32`-only; Linux JITs
  these fine.
- **oneDNN**: enabled on Windows with two Windows-only fixes — `int64_t` dims,
  and stripping the compiled `version.rc.res` from static `dnnl.lib` (else
  `LNK1241` under `/WHOLEARCHIVE`). It builds with vcomp like the rest of the
  process.
- **aocl-utils**: build **static-only** on Windows (`-DBUILD_SHARED_LIBS=OFF`) —
  its Cpuid CMake gives shared+static targets the same `OUTPUT_NAME aoclutils_c`,
  colliding on `aoclutils_c.lib` under Ninja; ZenDNN only whole-archives the
  static lib. Windows also needs a **CRT-MD-capable aocl-utils source** — stock
  public 5.3 does NOT build under `cl` (GNU `__attribute__((target))` in
  `HardwareRng.cc`, old `pInfo->u.Processor` anonymous-union access, clang-tidy
  enabled, no `AU_STATIC_FORCE_CRT_MD`). Point local mode at the CRT-MD drop
  (`-DZENDNNL_LOCAL_AOCLUTILS=ON -DAOCLUTILS_ROOT_DIR=<drop>`); this applies to CI
  too. The resolver in `ZenDnnlMacros.cmake` prefers `AoclUtils::aoclutils_static`.

## Cross-platform correctness bugs the port exposed
The Windows bring-up surfaced latent defects that were "safe" on Linux (the
over-read landed in mapped heap). Fix these at the root, **unguarded**:
- FBGEMM sharded embedding_bag passed the *global* `index_size` to every chunk →
  out-of-bounds prefetch; pass the per-chunk count instead.
- WOQ `binary_mul` routed a `{1,N}` row-broadcast operand as a dense `MATRIX_MUL`
  read past the buffer for `M>1` → materialize a dense `M×N` replica.

## Build & validate on Windows before landing
- Load the toolchain in Git Bash: `source scripts/load_msvc.sh` (sets `cl`,
  `link`, `rc`, `mt`, and `INCLUDE`/`LIB`/`PATH`; auto-discovers VS via vswhere).
- Configure with Ninja + `cl` ≥ 19.43, CMake ≥ 3.26, and always
  `-DZENDNNL_BUILD_TYPE=Release`; build with `cmake --build build-msvc`.
- After a Linux/Clang change, do a Windows build **and** run the relevant
  `gtests.exe` suites. Watch for the tell-tale symptoms: `C3861 aligned_alloc`,
  `C3646`/`C2146` (GNU attr / `__restrict__`), `LNK2038` (CRT mismatch),
  `0xC0000005` or `resource deadlock would occur` (alloc/free mismatch).

## Where the machinery lives
- `zendnnl/src/common/zendnnl_compat.hpp` — force-included via `/FI`; holds the
  `collapse` neutralization macro, `zendnnl_aligned_alloc`/`zendnnl_aligned_free`,
  and the lambda/attribute macros.
- `zendnnl/src/common/zendnnl_api.hpp` — the `ZENDNNL_API` export macro.
- `zendnnl/src/lowoha_operators/common/omp_thread_control.hpp` — `thread_guard`,
  `effective_num_threads()` and `scoped_active_levels`, with their vcomp
  fallbacks.
- `cmake/` — per-compiler flag split and the `ExternProj*.cmake` dependency
  wiring (aocl-utils / oneDNN / gtest / …); `ZenDnnlMacros.cmake` resolves the
  aocl-utils target.
- `scripts/load_msvc.sh` — MSVC environment loader for Git Bash.
