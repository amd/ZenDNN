---
name: build-vllm-onednn-zendnn
description: Build vLLM (CPU) end-to-end with the oneDNN + ZenDNN (zen64) backend via direct integration (NO zentorch). Use when asked to build vLLM CPU against oneDNN/ZenDNN or produce a ZenDNN-enabled vLLM wheel. Orchestrates the ZenDNN native lib (via the build-zendnn skill), upstream oneDNN source prep, a conda build env, the vLLM build, and two-tier verification.
version: 1.4.0
---

# Build vLLM (CPU) with oneDNN + ZenDNN (zen64)

Take a user from clean repos to a working, **verified** vLLM CPU build using the
direct oneDNN → vLLM CPU integration path (no zentorch). This skill encodes the
flow as a linear, stop-on-failure orchestration.

The opt-in **zen64 ZenDNN backend is now in upstream oneDNN `main`**
(`uxlfoundation/oneDNN#5511`, merged 2026-07-16), so this skill uses upstream
oneDNN directly. **Caveat:** as of this writing the fix is only on `main`; the
latest tagged release (`v3.13`) does **not** contain it, so `main` is the
default (see Stage 2).

**Upstream vLLM + a bundled patch.** The vLLM-side changes (the
`cpu_extension.cmake` wiring) are bundled here as a single patch,
**`vllm-onednn-zendnn.patch`**, applied on top of **upstream
`vllm-project/vllm`** — no downstream vLLM fork is required. Because the patch
can break if upstream edits `cpu_extension.cmake`, Stage 0 runs a **version
check** (`git apply --check`) before applying and, on mismatch, stops and asks
the user how to proceed (see Stage 0).

ZenDNN is compiled into vLLM's CPU extension: vLLM's `cpu_extension.cmake`
fetches and compiles the oneDNN source itself (via
`FETCHCONTENT_SOURCE_DIR_ONEDNN`) and links the standalone ZenDNN native lib
(via `ZENDNN_DIR`). ZenDNN is enabled **only when BOTH env vars are set**.

## FAILURE POLICY (applies to EVERY stage)
If ANY command fails — non-zero exit, clone/checkout error, bad git ref, conda
solve error, cmake can't find ZenDNN, compile/link error, OOM kill, or a
verification tier not matching — **STOP immediately and ask the user how to
proceed. Do NOT auto-retry, auto-fix, work around, or continue to the next
stage on your own.** Report to the user: the stage that failed, the exact
command, and the first real error line(s) from the log (e.g. `build.log`). Then
wait for the user's input before doing anything else. The per-stage "On
failure" notes below describe likely causes to include in your report — they
are NOT license to fix silently.

## Sections
- Flow
- Inputs to confirm
- Steps
- Smoke test (verification, not benchmarking)
- Stage 6 — Benchmark (optional, only after Stage 5 passes)
- Output

## Flow

Linear spine, Stage 0 → 6. Every gate is an explicit decision with a STOP
terminal — no retry loops.

The full flowchart lives in
**[`build-vllm-onednn-zendnn-flow.mmd`](build-vllm-onednn-zendnn-flow.mmd)** next
to this file. Stage 1 delegates to the `build-zendnn` skill, whose own sub-flow
is in `build-zendnn/build-zendnn-flow.mmd`.

In brief: Stage 0 confirm inputs → Stage 0b apply the bundled patch (with a
`git apply --check` version guard) → Stage 1 build ZenDNN via `build-zendnn` →
Stage 2 prep upstream oneDNN source (no build) → Stage 3 create the conda env →
Stage 4 build and install the wheel → Stage 5 verify both tiers → Stage 6
optional clean benchmark.

## Inputs to confirm
- **`REPO_ROOT`** — site-specific value the user sets: the directory that
  *contains* `vllm/` (and the sibling `oneDNN/`, `ZenDNN/` checkouts). All
  paths below are expressed relative to it.
- **vLLM repo location** — default: clone **upstream `vllm-project/vllm`** into
  `${REPO_ROOT}/vllm` (sibling of this repo). Accept an existing checkout
  override. No downstream vLLM fork is used.
- **vLLM ref** — default the **pinned base commit** the bundled patch was
  generated against: `c638f9216a08bfb5644d8a266ddd35421e04118d`
  (upstream `vllm-project/vllm`, `#47265`). This guarantees a clean patch apply.
  You may instead pick a newer ref (`main`, a tag) — the patch is tolerant of
  line-number shifts and will often still apply — but then the **Stage 0 version
  check must pass** first.
  > **The vLLM changes come from a patch, not a fork.** The direct
  > oneDNN+ZenDNN → vLLM CPU integration (wiring `cpu_extension.cmake` to
  > `FETCHCONTENT_SOURCE_DIR_ONEDNN` + `ZENDNN_DIR`, no zentorch) is **not yet in
  > upstream `vllm-project/vllm`** — upstream only carries the *zentorch*-based
  > ZenCPU path, which this skill avoids. Those changes are captured here as
  > **`vllm-onednn-zendnn.patch`** (touches only `cmake/cpu_extension.cmake`) and
  > applied to upstream vLLM in Stage 0.
  >
  > **Plan / interim guidance.** These changes are intended to be upstreamed.
  > Until then the bundled patch is the source of truth. If upstream diverges so
  > the patch no longer applies, regenerate it (see "Regenerating the patch"
  > under Stage 0). Once the changes merge upstream, drop the patch and use that
  > ref directly.
- **ZenDNN ref** — passed to the `build-zendnn` skill. Default `main` / current
  repo.
- **oneDNN ref** — **ask the user each run.** Source is upstream
  `uxlfoundation/oneDNN` (default sibling `../oneDNN`). Offer two choices:
  - **`main`** (default) — currently the **only** ref with the zen64 backend.
  - **a release tag** the user names that is known to contain the fix
    (`uxlfoundation/oneDNN#5511`). As of 2026-07 no tag qualifies yet — `v3.13`
    does **not** include it — so confirm containment before using a tag (see
    Stage 2).
- **Build method** — **wheel (default)** or editable.

> **Layout assumption (important).** The bundled `create_env.sh` and
> `build_wheel.sh` assume they live in the **vLLM repo root's parent** — i.e.
> `REPO_ROOT` is the dir that *contains* `vllm/` (the actual vLLM checkout is at
> `${REPO_ROOT}/vllm`). Copy both scripts next to your `vllm/` checkout (i.e.
> into `${REPO_ROOT}/`) before running them, or invoke them from there. Rust
> state lands under `${REPO_ROOT}/.rust/`.

## Steps

1. **Stage 0 — Confirm inputs.** Resolve the inputs above with the user
   (including the per-run oneDNN ref: `main` vs a qualifying tag).

   **First check for an existing checkout — do NOT clone over a staged tree.**
   The repos are often pre-staged as siblings (`vllm/`, `oneDNN/`,
   `ZenDNN/`). If `${REPO_ROOT}/vllm` already exists, skip the clone and just
   verify the refs are what you expect:
   ```bash
   # existing checkout: verify, don't clone
   git -C vllm remote get-url origin    # expect vllm-project/vllm (upstream)
   git -C vllm log --oneline -1
   git -C ../oneDNN branch --show-current  # expect main (or the agreed tag)
   git -C ZenDNN branch --show-current
   ```
   Only if `vllm/` is absent, clone **upstream** and check out the agreed ref
   (default = the pinned base commit the patch was generated against):
   ```bash
   PATCH_BASE=c638f9216a08bfb5644d8a266ddd35421e04118d   # upstream vllm #47265
   git clone https://github.com/vllm-project/vllm.git "${REPO_ROOT}/vllm"
   cd "${REPO_ROOT}/vllm"
   git checkout "${PATCH_BASE}"     # default; or: git checkout main / <tag>
   ```
   **On failure** (clone/network error, or bad ref → `couldn't find remote ref` /
   `did not match any`): stop. Do not continue with a missing or stale checkout.
   Re-confirm the repo URL and exact ref with the user, then re-run. If an
   existing checkout is on an unexpected ref, confirm with the user before moving
   it — it may be their in-progress work.

   **Apply the bundled vLLM patch (with a version check).** The vLLM changes live
   in `vllm-onednn-zendnn.patch` (bundled next to this skill). **Always dry-run
   the version check first, then apply:**
   ```bash
   PATCH=/path/to/build-vllm-onednn-zendnn/vllm-onednn-zendnn.patch
   PATCH_BASE=c638f9216a08bfb5644d8a266ddd35421e04118d   # upstream vllm #47265
   cd "${REPO_ROOT}/vllm"

   # Skip if already applied (idempotency guard):
   if grep -q "DNNL_X64_USE_ZEN" cmake/cpu_extension.cmake; then
       echo "patch already applied — skipping"
   # VERSION CHECK: does the patch still apply to this vLLM checkout?
   elif git apply --check "$PATCH" 2>/dev/null; then
       # even after --check, apply can fail — do NOT swallow it:
       git apply "$PATCH" && echo "patch applied" || {
           echo "ERROR: git apply FAILED on $(git rev-parse --short HEAD)" >&2
           exit 1
       }
   else
       echo "VERSION MISMATCH: patch does not apply to $(git rev-parse --short HEAD)" >&2
       exit 1
   fi
   # Confirm the marker landed:
   grep -n "Using ZenDNN from\|DNNL_X64_USE_ZEN" cmake/cpu_extension.cmake
   ```
   **On failure** (`git apply --check` fails, or `git apply` itself fails →
   upstream changed `cpu_extension.cmake`): **STOP. Do NOT force it, do NOT
   auto-`--3way`, and do NOT auto-checkout a different ref.** Per the FAILURE
   POLICY, **report to the user and ask how they want to proceed.** Include in
   your report: the vLLM ref/SHA (`git rev-parse HEAD`), the exact failing
   command, and the first error line(s) from `git apply`. Then present the
   options and **wait for the user's input**:
   - (a) check out the pinned base commit `${PATCH_BASE}` where the patch is known
     to apply, then re-run the apply; or
   - (b) regenerate the patch against their chosen ref (see below); or
   - (c) something else the user directs.

   A forced or half-landed hunk silently drops the ZenDNN wiring, so never work
   around this on your own.

   > **Regenerating the patch** (only if upstream diverged): from a vLLM checkout
   > that carries the integration changes, diff the base vs. the change tip for
   > the touched file:
   > ```bash
   > git diff <base-ref> <change-ref> -- cmake/cpu_extension.cmake \
   >   > vllm-onednn-zendnn.patch
   > ```

2. **Stage 1 — ZenDNN native lib.** **Invoke the existing `build-zendnn`
   skill** — it is the single source of truth for the ZenDNN build. Do **not**
   duplicate its cmake commands here. It produces:
   ```
   ZENDBUILDPATH = <ZenDNN>/build/install/zendnnl
   ```
   **On failure**: stop. Fix per the `build-zendnn` skill's own on-failure
   guidance; do not proceed to Stage 4 without a completed ZenDNN install tree.

3. **Stage 2 — oneDNN source prep (NO build).** Use **upstream
   `uxlfoundation/oneDNN`** — the zen64 ZenDNN backend is now in its `main`
   (`#5511`). A standalone oneDNN build is **NOT REQUIRED FOR VLLM** — vLLM's
   `cpu_extension.cmake` fetches and compiles this source itself via
   `FETCHCONTENT_SOURCE_DIR_ONEDNN`.

   **Ask the user each run which oneDNN ref to use** (default `main`), then check
   it out and record its path. Clone upstream if the sibling `../oneDNN` is
   absent:
   ```bash
   # clone upstream only if missing:
   [ -d ../oneDNN ] || git clone https://github.com/uxlfoundation/oneDNN.git ../oneDNN
   git -C ../oneDNN fetch origin --tags

   # ---- choose ONE ref (ask the user) ----
   # (a) DEFAULT: main — currently the only ref with the zen64 backend
   git -C ../oneDNN checkout main && git -C ../oneDNN pull --ff-only
   # (b) a release tag the user names, e.g.:
   # git -C ../oneDNN checkout <tag>

   export ONEDNN_DIR="$(cd ../oneDNN && pwd)"
   ```
   **If the user picks a tag, verify it actually contains the fix first** — the
   zen64 merge commit is `e4feb31` (`uxlfoundation/oneDNN#5511`). As of 2026-07
   **no released tag qualifies** (`v3.13`, cut 2026-07-17, does NOT contain it):
   ```bash
   git -C ../oneDNN merge-base --is-ancestor e4feb31 <tag> \
     && echo "OK: <tag> has zen64" || echo "REJECT: <tag> lacks zen64 — use main"
   ```
   **On failure** (ref missing → `did not match any`, or a tag that lacks the
   fix): stop and re-confirm the oneDNN ref with the user. Do not point vLLM at a
   source tree without the zen64 backend — the build will silently drop ZenDNN.

4. **Stage 3 — Conda build env.** Drive this with the bundled
   **`create_env.sh`** (do not hand-roll the env). It creates/reuses conda env
   `vllm-build-env-312` (python 3.12), installs conda gcc/g++ `>=12.3,<13` +
   libnuma + numactl, installs rust `1.95` off the home quota (under
   `${REPO_ROOT}/.rust/`), and installs vLLM build deps. `--runtime` also
   installs runtime deps and pins torch/torchvision/torchaudio to the CPU index.
   - **gcc is pinned to 12.x (`>=12.3,<13`)** so libstdc++ symbols stay
     compatible with Ubuntu 22.04 targets; a newer gcc pulls in newer libstdc++
     symbols.
   - **Rust is placed off the home quota** — `RUSTUP_HOME`/`CARGO_HOME` live
     under `${REPO_ROOT}/.rust/` because the home directory quota is too small
     for a toolchain.
   - **CPU torch only** — `--runtime` installs torch/torchvision/torchaudio from
     the CPU index (`https://download.pytorch.org/whl/cpu`); otherwise CUDA
     builds slip in.
   ```bash
   # from the parent of vllm/ (see layout assumption above):
   bash create_env.sh              # build-only deps
   # or: bash create_env.sh --runtime   # + runtime deps + CPU torch
   conda activate vllm-build-env-312
   ```
   **On failure** (missing `requirements/*` file, conda solve error, rustup
   download blocked): stop. The script errors out under `set -euo pipefail`; fix
   the reported cause (usually a missing vLLM checkout or network) and re-run. Do
   not proceed to the build with a half-provisioned env.

5. **Stage 4 — vLLM build.** Drive this with the bundled **`build_wheel.sh`**
   (wheel method) or the editable command below. The build must run with:
   - `VLLM_TARGET_DEVICE=cpu`
   - `FETCHCONTENT_SOURCE_DIR_ONEDNN=<ONEDNN_DIR>` (from Stage 2)
   - `ZENDNN_DIR` — **either form works**: the ZenDNN repo dir (whose install
     tree is at `build/install/zendnnl`) or the `ZENDBUILDPATH` install tree
     itself from Stage 1. The patch probes for `build/install/zendnnl/include`
     first, then falls back to treating `ZENDNN_DIR` as the install tree. If
     neither resolves, cmake stops with a `FATAL_ERROR` — a wrong path fails
     loudly instead of silently producing a ZenDNN-less wheel. Either way,
     **Stage 1 must have completed first**; pointing at an unbuilt repo fails.
   - `MAX_JOBS=${MAX_JOBS:-32}` — the box is a shared 192-core machine, so an
     unbounded `-j` triggers the OOM killer mid-build. Keep the cap (lower if
     still OOMing).
   - gcc/g++ `>= 12.3`; if `CC`/`CXX` are unset, fall back to the conda
     compilers (`$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-{gcc,g++}`).

   **Wheel (default):** edit the two placeholders at the top of `build_wheel.sh`
   (or export `ZENDNN_DIR` / `FETCHCONTENT_SOURCE_DIR_ONEDNN` first), then:
   ```bash
   bash build_wheel.sh 2>&1 | tee build.log
   pip install dist/*.whl
   ```
   The script itself runs `pip wheel . --no-deps --no-build-isolation -w dist/`.

   **Editable (alternative):**
   ```bash
   cd vllm
   VLLM_TARGET_DEVICE=cpu MAX_JOBS=${MAX_JOBS:-32} \
   FETCHCONTENT_SOURCE_DIR_ONEDNN="$ONEDNN_DIR" ZENDNN_DIR="<ZenDNN dir>" \
     pip install -e . --no-build-isolation -v 2>&1 | tee build.log
   ```
   **On failure** (cmake can't find ZenDNN, compile/link error, OOM kill): stop.
   Read the first error in `build.log`. If it's an OOM (killed processes, signal
   9), lower `MAX_JOBS`. For a clean rebuild against a changed oneDNN or
   ZenDNN, `rm -rf build/ .deps/` in `vllm/` and re-run — stale cmake/fetch
   caches silently keep old sources and defeat the integration. Do not
   `pip install` a wheel from a failed build.

6. **Stage 5 — Verification (BOTH tiers).**
   > **These env vars and this command are for VERIFICATION ONLY — NEVER for
   > benchmarking.** They enable verbose logging/profiling that distorts
   > performance; do not report any numbers from this run.

   - **Build-time** — confirm ZenDNN was actually linked in:
     ```bash
     grep "Using ZenDNN from" build.log
     ```
     Must match. **On failure** (no match): ZenDNN was NOT enabled — verify BOTH
     `FETCHCONTENT_SOURCE_DIR_ONEDNN` and `ZENDNN_DIR` were set for the build,
     then `rm -rf build/ .deps/` and rebuild (Stage 4).
   - **Runtime smoke:**
     ```bash
     export LD_PRELOAD=$CONDA_PREFIX/lib/libiomp5.so
     export VLLM_CPU_KVCACHE_SPACE=40   # REQUIRED — see note below
     ZENDNNL_ENABLE_PROFILER=1 ONEDNN_VERBOSE=1 vllm bench throughput \
       --model Qwen/Qwen3-0.6B --gpu-memory-utilization 0.2 \
       --random-input-len 32 --random-output-len 1 --num-prompts 1 \
       --max-num-seqs 1 --dtype bfloat16 --trust_remote_code \
       | tee out.log | tail -30
     ```
     Expect both `onednn_verbose,...,zen:matmul...` lines and
     `[PROF ...]:LOWOHA matmul_direct...` lines in `out.log`. **On failure** (no
     `zen:matmul` / no `[PROF ...]` lines): the ZenDNN path is not active at
     runtime — re-check the build-time grep and the two env vars, then rebuild.

   > **CPU KV-cache sizing (`VLLM_CPU_KVCACHE_SPACE`).** On CPU,
   > `--gpu-memory-utilization` barely bounds memory — vLLM auto-sizes the KV
   > cache to a fraction of **total box RAM** (seen: ~225 GiB up to ~1 TB on a
   > big box), which OOMs the process or trips a scheduler mem limit. Always set
   > `VLLM_CPU_KVCACHE_SPACE` (GB) to cap it. `40` is plenty for a smoke run.
   > The log line `Explicitly set (40.0/...) GiB for KV cache` confirms the cap
   > took; `Auto set (1041.28/...) GiB` means it did NOT and you'll blow up.

## Smoke test (verification, not benchmarking)
The Stage 5 checks above **are** the smoke test. A run is only "good" when:
- `grep "Using ZenDNN from" build.log` matches (build-time), AND
- `out.log` shows `zen:matmul` (oneDNN verbose) and `LOWOHA matmul_direct`
  (ZenDNN profiler) lines (runtime).

Do not treat the build as done until both pass. Again: **verification only, not
a benchmark** — the verbose/profiler env vars skew timing.

### Fallback: verify an already-built extension (no build.log)
If you inherit a tree whose `build.log`/`out.log` are gone, you can still prove
statically that both backends were compiled and linked in, by inspecting the
built extension for the two symbol families:
```bash
SO=<vllm>/vllm/_C.abi3.so
nm -C "$SO" | grep -c 'zendnnl::'                    # ZenDNN linked in
nm -C "$SO" | grep -c 'dnnl::impl::cpu::x64::zen'    # oneDNN zen64 backend built
nm -C "$SO" | grep ' T .*zendnnl::lowoha::matmul::matmul_direct'   # the bridge
```
All three must be non-empty. Use `nm -C` (full symbol table), **not `nm -D`**:
the zen64 symbols are internal to the extension and do not appear in the
dynamic table, so a `-D` check reports zero and looks like a failure when the
build is in fact correct. This is weaker than the Stage 5 checks — it proves the
integration was *linked*, not that it is *selected at runtime* — so still do the
`out.log` check when you can actually run the model.

## Stage 6 — Benchmark (optional, only after Stage 5 passes)
Verification ≠ benchmark. Stage 5 proves ZenDNN is *active*; it says nothing
about performance. To measure throughput, run a **clean** pass — the opposite of
Stage 5's env:

- **NEVER set** `ZENDNNL_ENABLE_PROFILER` or `ONEDNN_VERBOSE` for a perf run —
  both log per-op and destroy timing. (If you want to *inspect* oneDNN dispatch,
  `ONEDNN_VERBOSE=1` is fine, but treat that run's numbers as invalid and say so.)
- **DO set** `VLLM_CPU_KVCACHE_SPACE` (same reason as Stage 5b — else OOM).
- Keep `LD_PRELOAD=$CONDA_PREFIX/lib/libiomp5.so`.
- Use realistic seqlens/prompt counts, not the 1-token smoke shape.

```bash
export LD_PRELOAD=$CONDA_PREFIX/lib/libiomp5.so
export VLLM_CPU_KVCACHE_SPACE=40
vllm bench throughput \
  --model Qwen/Qwen3-0.6B --dtype bfloat16 \
  --input-len 128 --output-len 128 --num-prompts 64 \
  --trust_remote_code | tee bench.out
```
Report the `Throughput: ... requests/s, ... total tokens/s` line.
Qwen3-0.6B is a small, currently-supported model — numbers are small
by design; swap `--model` for the real target to get meaningful figures. (Use
any current vLLM-supported model; avoid stale ones like `facebook/opt-125m`.)

## Output
Artifacts produced by a successful run:
- **Patched upstream vLLM** — `vllm/cmake/cpu_extension.cmake` carries the
  ZenDNN wiring from `vllm-onednn-zendnn.patch` (confirmed by the
  `DNNL_X64_USE_ZEN` / `Using ZenDNN from` markers), applied on top of upstream
  `vllm-project/vllm` — no internal fork.
- **Installed wheel** in `vllm/dist/*.whl` — the ZenDNN-enabled vLLM CPU wheel
  (produced by `build_wheel.sh`, then `pip install`ed).
- **`build.log`** — the tee'd build output; contains the `Using ZenDNN from`
  build-time proof line.
- **A verified ZenDNN-enabled vLLM** — confirmed by both Stage 5 tiers
  (`build.log` grep + `out.log` `zen:matmul` / `LOWOHA matmul_direct` lines).
