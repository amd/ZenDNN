---
name: build-zendnn
description: Build the standalone ZenDNN native library (zendnnl) from source with the alternate compute backends OFF (no oneDNN, libxsmm, parlooper, fbgemm), keeping AOCL DLP, which is the GEMM backend zendnnl links against. Use when asked to build ZenDNN alone, produce the zendnnl install tree for zentorch/vLLM, or set up the "zendnn-only" backend. This is the single source of truth for ZenDNN backend build steps — the zentorch repo invokes this skill via its third_party/ZenDNN submodule.
version: 1.2.0
---

# Build ZenDNN (zendnn-only backend)

Produce a clean `zendnnl` build + install tree from a chosen ZenDNN git ref.
This install tree is the ZenDNN backend consumed by downstream builds — e.g.
zentorch (`third_party/ZenDNN` submodule) and the vLLM CPU integration.

## FAILURE POLICY (applies to EVERY step)
If ANY command fails — non-zero exit, cmake configure error, compile/link
error, missing file, bad git ref, OOM kill, or unexpected output — **STOP
immediately and ask the user how to proceed. Do NOT auto-retry, auto-fix,
work around, or continue to the next step on your own.** Report to the user:
the step that failed, the exact command, and the first real error line(s) from
the log. Then wait for the user's input before doing anything else. The
per-step "On failure" notes below describe likely causes to include in your
report — they are NOT license to fix silently.

## Sections
- Flow
- Inputs to confirm
- Steps
- Smoke test
- Gotchas
- Output

## Flow

Four steps plus a smoke test. This is the reusable sub-flow that
`build-vllm-onednn-zendnn` invokes as its Stage 1.

The full flowchart lives in **[`build-zendnn-flow.mmd`](build-zendnn-flow.mmd)**
next to this file. Every gate in it has an explicit STOP terminal — there are no
retry loops, matching the FAILURE POLICY above.

In brief: confirm inputs → Step 1 check out the ref → Step 2 configure with the
four alternate-backend `ZENDNNL_DEPENDS_*` OFF → Step 3 build the `all` target
(which also installs) → Step 4 record `ZENDBUILDPATH` → smoke test the install
tree and record the source hash for provenance.

## Inputs to confirm
- **ZenDNN git ref** — ask the user. Default `main`. Accept:
  - a branch or tag (`git checkout <ref>`)
  - a PR number `N` → `git fetch origin pull/N/head:pr<N> && git checkout pr<N>`
- **ZenDNN dir** — default the current repo root (this repo, where the skill lives).

## Steps

1. **Check out the ref** in the ZenDNN repo:
   ```bash
   git -C <ZenDNN> fetch origin
   # branch/tag:
   git -C <ZenDNN> checkout <ref>
   # or PR:
   git -C <ZenDNN> fetch origin pull/<N>/head:pr<N> && git -C <ZenDNN> checkout pr<N>
   ```
   **On failure** (bad/unknown ref → `fatal: ... did not match any` or
   `couldn't find remote ref`): stop. Do not continue with a stale checkout.
   Re-confirm the exact branch/tag/PR number with the user and re-run this step.

2. **Configure (zendnn-only)** — turn off the four *alternate* compute backends
   and keep AOCL DLP. The options are defined in
   `cmake/ZenDnnlDependenciesDefaults.cmake`:
   ```bash
   cd <ZenDNN>
   rm -rf build && mkdir build && cd build
   cmake -DZENDNNL_DEPENDS_ONEDNN=OFF \
         -DZENDNNL_DEPENDS_LIBXSMM=OFF \
         -DZENDNNL_DEPENDS_PARLOOPER=OFF \
         -DZENDNNL_DEPENDS_FBGEMM=OFF ..
   ```
   **Do NOT pass `-DZENDNNL_DEPENDS_AOCLDLP=OFF`.** AOCL DLP is not an
   interchangeable backend — it is the GEMM library `zendnnl` itself links
   against. The built `libzendnnl_archive.a` carries ~144 undefined `aocl_*`
   references (`aocl_batch_gemm_bf`, `aocl_dlp_experts`, …), so turning it off
   does not yield a leaner zendnn-only library, it yields an unusable one.
   `ZENDNNL_DEPENDS_AOCLUTILS` and `ZENDNNL_DEPENDS_JSON` are likewise not
   yours to turn off — the cmake sets them `ON` with `FORCE`.

   The default install prefix is set by the build system itself:
   `cmake/ZenDnnlProjectOptions.cmake` sets
   `CMAKE_INSTALL_PREFIX = <ZenDNN>/build/install`, so no
   `-DCMAKE_INSTALL_PREFIX` flag is needed.
   **On failure** (cmake configure error): fix the reported cause (missing
   compiler/OpenMP, unsupported CMake < 3.26, or a leftover `build/` cache),
   then stop and re-run this step from the `rm -rf build` line — do not proceed
   to Build with a half-configured tree.

3. **Build** (this also installs — see below):
   ```bash
   cmake --build . --target all
   ```
   The `all` target **already installs** into `build/install/zendnnl`: the
   top-level `CMakeLists.txt` adds `zendnnl` via `ExternalProject_ADD`
   (`cmake/ExternProjZENDNNL.cmake`) whose `INSTALL_COMMAND cmake --build .
   --target install` runs as part of building (`BUILD_ALWAYS TRUE`, not
   excluded from `all`), installing under `CMAKE_INSTALL_PREFIX` (=
   `build/install`) into the `zendnnl/` subdir. Therefore **do not** add a
   separate `--target install`/`cmake --install .` step — Steps 4-5 consume the
   tree this step produces.
   **On failure** (compile/link error from `cmake --build`): read the first
   error in the log, fix it (or report the exact error to the user if it is a
   source-level break in the chosen ref), then stop and re-run this step. Do not
   record an install path in Step 4 if the build did not complete.

4. **Record the install path** for downstream skills. `$(pwd)` here must be
   `<ZenDNN>/build` (the dir created in Step 2); otherwise use the absolute
   path directly:
   ```bash
   # from <ZenDNN>/build:
   export ZENDBUILDPATH="$(pwd)/install/zendnnl"   # <ZenDNN>/build/install/zendnnl
   # or, cwd-independent:
   export ZENDBUILDPATH="<ZenDNN>/build/install/zendnnl"
   ```

## Smoke test (build sanity, not benchmarking)
- Install tree exists: `test -d <ZenDNN>/build/install/zendnnl && echo OK`.
  **If this returns non-zero** (dir missing): the build did not install — the
  `all` target in Step 3 did not complete. Re-check the Step 3 build log for
  errors, fix the cause, and re-run Steps 2-3. Do not hand `ZENDBUILDPATH` to a
  downstream skill until this prints `OK`.
- Note the exact source hash compiled, so downstream can prove provenance:
  ```bash
  git -C <ZenDNN> rev-parse HEAD
  ```
  Downstream (zentorch) will match this against
  `zentorch._C.show_config()`'s `ZENDNNL ... Git Hash` line.

## Gotchas
- Always `rm -rf build` first — a stale cmake cache silently keeps old
  dependency flags, defeating the zendnn-only intent.
- The `ZENDNNL_DEPENDS_*` **defaults vary** in
  `cmake/ZenDnnlDependenciesDefaults.cmake`: `AOCLDLP`, `ONEDNN`, `LIBXSMM` and
  `FBGEMM` default **ON**, `PARLOOPER` defaults **OFF**. Pass the four
  alternate backends OFF explicitly rather than relying on the defaults, so the
  result does not drift if a default changes.
- "zendnn-only" means *no oneDNN / libxsmm / parlooper / fbgemm*. It does **not**
  mean zero dependencies: a correct zendnn-only build still pulls AOCL DLP,
  AOCL UTILS and JSON, and you can confirm that afterwards —
  `ls build/install/deps` shows exactly `aocldlp aoclutils json`.
- There is no `ZENDNNL_DEPENDS_AMDBLIS` option. Passing it is silently ignored
  (it lands in the cache as `UNINITIALIZED` and changes nothing), so a build
  that appears to have disabled BLIS that way has in fact disabled nothing.
- **Do not point `CC`/`CXX` at a conda toolchain.** Configuring with conda's
  `x86_64-conda-linux-gnu-cc` while `CXX` is the system `/usr/bin/c++` breaks
  the bundled AOCL-UTILS build with `/usr/bin/ld: cannot find /lib64/libm.so.6`
  — conda's gcc resolves system libs against its own sysroot. Build ZenDNN with
  a consistent system toolchain (`/usr/bin/cc` + `/usr/bin/c++`); `deactivate`
  or unset `CC`/`CXX` if a conda env has exported them.

## Output
Artifacts produced by a successful run:
- **Install tree**: `<ZenDNN>/build/install/zendnnl` — the installed `zendnnl`
  library (`lib/`, `include/`, package config). Produced by the `all` target in
  Step 3. This is the directory downstream skills point at.
- **`ZENDBUILDPATH`** (exported shell variable) = `<ZenDNN>/build/install/zendnnl`
  — consumed by downstream skills (`zentorch`/vLLM builds) to locate the install
  tree above.
