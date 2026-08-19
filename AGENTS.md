# AGENTS.md

Guidance for AI coding agents working in the **ZenDNN** repository. This file
captures always-useful context about the repo and its architecture, and points
to specialized agent skills for detailed, step-by-step workflows. Read this
first; follow a skill when your task matches one below.

## What this repo is

**ZenDNN** (Zen Deep Neural Network Library) accelerates deep learning
*inference* on AMD CPUs. This repository builds the standalone native library
**`zendnnl`** — a redesigned, re-architected successor to `ZenDNN_legacy`.

- ZenDNN's performance-first interface is the **Low Overhead API (LowOHA)**:
  direct, function-based entry points (e.g. `matmul_direct`,
  `group_matmul_direct`, `reorder_direct`) operating on raw pointers with
  minimal per-call overhead. It is complemented by a modular, object-oriented
  **Tensor Operator API**.
- ZenDNN is a *library*, not an application. It is consumed by inference
  frameworks through plugins (see [Consumers / plugins](#consumers--plugins))
  and can also be linked directly into serving stacks.
- ZenDNN can optionally plug in alternate backends (oneDNN, LibXSMM, parlooper,
  FBGEMM) but also builds "zendnn-only" with those off. Note that zendnn-only
  is not dependency-free: AOCL DLP is the GEMM library `zendnnl` links against
  and stays enabled.

For a deeper architecture overview see `docs/zendnnl_architecture.md`; for the
public overview see `README.md`.

## Repository layout

```
ZenDNN
|- CMakeLists.txt   : top-level CMake entry point.
|- cmake/           : CMake modules (dependency defaults, project options,
|                     ExternalProject wiring).
|- zendnnl/         : the library source.
|   |- src/
|   |   |- common          : high-level library utilities.
|   |   |- memory          : tensor_t class, storage, quantization, options.
|   |   |- lowoha_operators: direct low-overhead operators (LowOHA path):
|   |   |                    matmul, reorder, normalization, sdpa, softmax, pooling.
|   |   |- operators       : object-oriented Tensor Operator API.
|   |   |- gtests          : GoogleTest files.
|- benchdnn/        : benchmarking utility for ZenDNN operators.
|- examples/        : tutorial examples using the ZenDNN API.
|- dependencies/    : third-party dependency download/build.
|- docs/            : architecture, build, and logging documentation.
|- scripts/         : supporting shell scripts.
|- tools/, fwk/     : additional tooling.
|- .claude/skills/  : agent skills (see below).
```

## Build system

- **CMake** (>= 3.26) drives the whole build. Optional backend dependencies are
  controlled by `ZENDNNL_DEPENDS_*` flags defined in
  `cmake/ZenDnnlDependenciesDefaults.cmake`. The five toggles are `AOCLDLP`,
  `ONEDNN`, `LIBXSMM`, `PARLOOPER` and `FBGEMM`; **defaults vary** (`PARLOOPER`
  is OFF, the rest ON), so pass each one explicitly rather than assuming.
  `AOCLDLP` is a toggle but not a *choice*: `zendnnl` links AOCL DLP directly,
  so a build with it OFF is broken rather than lean. `AOCLUTILS` and `JSON` are
  forced ON by the cmake and are not user-controllable. There is no `AMDBLIS`
  toggle — passing `-DZENDNNL_DEPENDS_AMDBLIS=...` is silently ignored.
- `cmake/ZenDnnlProjectOptions.cmake` sets the install prefix to
  `<ZenDNN>/build/install`, so no `-DCMAKE_INSTALL_PREFIX` is needed.
- The top-level `CMakeLists.txt` adds `zendnnl` via `ExternalProject_ADD`
  (`cmake/ExternProjZENDNNL.cmake`); the `all` target both builds **and
  installs** into `build/install/zendnnl` — there is no separate install step.
- ZenDNN downloads and builds its dependencies during its own build, and
  forwards them to downstream consumers, so a consumer does not have to resolve
  ZenDNN's dependencies itself.

For the detailed and authoritative build workflow, **use the `build-zendnn`
skill** rather than reconstructing commands by hand.

## Consumers / plugins

ZenDNN is designed to be used through framework plugins and integrations rather
than standalone:

- **zentf — TensorFlow Plugin** (`ZenDNN_TensorFlow_Plugin`): integrates ZenDNN
  into TensorFlow so TF inference can dispatch supported ops to ZenDNN kernels
  on AMD CPUs. It consumes the `zendnnl` library produced by this repo. For
  supported TensorFlow/Python versions and install instructions, see the public
  zentf README (linked from this repo's `README.md`). Architectural specifics of
  the plugin itself live in the zentf repository, not here — treat that repo as
  the source of truth for zentf internals.
- **zentorch — PyTorch Plugin** (`ZenDNN_PyTorch_Plugin`): integrates ZenDNN
  into PyTorch. zentorch vendors this ZenDNN repo as a `third_party/ZenDNN`
  submodule and builds the ZenDNN backend via the `build-zendnn` skill (the
  single source of truth for the ZenDNN backend build steps).
- **Direct vLLM (CPU) integration**: vLLM's CPU extension can compile **upstream
  oneDNN** (`uxlfoundation/oneDNN`, whose `main` carries the opt-in zen64 ZenDNN
  backend) and link the standalone ZenDNN native library directly (no zentorch).
  This path is enabled only when both the oneDNN source dir
  (`FETCHCONTENT_SOURCE_DIR_ONEDNN`) and the ZenDNN dir (`ZENDNN_DIR`) are
  provided at build time. The vLLM-side wiring is not yet upstream, so it is
  carried as a patch bundled with the `build-vllm-onednn-zendnn` skill. See that
  skill.

## Agent skills

Specialized, step-by-step workflows live under `.claude/skills/`. When a task
matches one of these, **follow the skill** — it is the authoritative,
stop-on-failure procedure. Do not duplicate or paraphrase its commands here.

- **`build-zendnn`** — `.claude/skills/build-zendnn/SKILL.md`
  Build the standalone ZenDNN native library (`zendnnl`) from a chosen git ref
  with the alternate backends OFF (no oneDNN, LibXSMM, parlooper, FBGEMM),
  keeping AOCL DLP. Produces the `build/install/zendnnl` install tree (exported as
  `ZENDBUILDPATH`) that downstream consumers (zentorch, vLLM) point at. This is
  the single source of truth for the ZenDNN backend build. Inputs: ZenDNN git
  ref (branch/tag/PR; default `main`) and ZenDNN dir (default this repo).

- **`build-vllm-onednn-zendnn`** — `.claude/skills/build-vllm-onednn-zendnn/SKILL.md`
  Build vLLM (CPU) end-to-end with the upstream oneDNN + ZenDNN (zen64) backend
  via direct integration (no zentorch). Orchestrates the `build-zendnn` skill for
  the native lib, upstream oneDNN source prep, a conda build env, the vLLM wheel
  build, and two-tier (build-time + runtime) verification. Inputs: repo root,
  vLLM repo/ref, ZenDNN ref, oneDNN dir, and build method.

Each skill ships its flowchart as a standalone `.mmd` file next to its
`SKILL.md` — `build-zendnn/build-zendnn-flow.mmd` and
`build-vllm-onednn-zendnn/build-vllm-onednn-zendnn-flow.mmd`. Those diagrams are
the end-to-end decision flow, with `build-zendnn` as the reusable Stage 1
sub-flow of the vLLM build.

### Working principle for skills

Both build skills enforce a strict **stop-on-failure policy**: if any command
fails, stop and report the failing step, the exact command, and the first real
error line — do not auto-retry, auto-fix, or continue on your own. Honor that
policy when executing these workflows.

## Conventions

- Keep documentation factual: do not invent APIs, flags, or version numbers.
- Prefer editing existing files over creating new ones.
- Commit messages describe only the change — no AI/tool attribution.
