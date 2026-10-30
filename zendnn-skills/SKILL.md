---
name: zendnn-skills
description: >-
  Single entry point for working with the ZenDNN library (zendnnl). Routes each
  request to the matching sub-skill, collects shared inputs once, runs
  multi-step requests in order and reports one result. Covers building
  zendnnl, building vLLM for CPU against upstream oneDNN and ZenDNN without
  zentorch, developing and optimizing an operator, reviewing an operator
  change or a ZenDNN pull request, running gtests, benchmarking with
  BenchDNN, and keeping C++ and CMake changes working on the native Windows
  (MSVC) build. Use for any task that involves changing ZenDNN's C++ or CMake,
  or building, testing, benchmarking, configuring, integrating or reviewing
  ZenDNN. When no sub-skill covers the task, it says so and works only from
  the repository's own documentation.
---

# ZenDNN skills

Every ZenDNN request starts here. This file picks the sub-skill, gathers what
it needs and hands over. The procedures live in the sub-skills, and a
sub-skill is only ever opened through this file.

## Handling a request

1. Match the request to a sub-skill below. If it needs more than one, see
   Chaining. If none fits, see Gaps.
2. Resolve the shared inputs and the sub-skill's own inputs. Ask only for what
   the user hasn't given, and offer the defaults.
3. Tell the user the sub-skill's side effects. Get a yes before it switches a
   checkout that has uncommitted changes, or replaces a build directory that
   already exists.
4. Open only that sub-skill's SUBSKILL.md and follow it exactly. When it
   points at a platform folder or an operator file, open those too. Its rules,
   including its failure policy, override this file.
5. Report as described in Reporting.

## Sub-skills

### [build-zendnn](building/build-zendnn/SUBSKILL.md)
- **For:** building zendnnl with oneDNN, LibXSMM, parlooper and FBGEMM off
  (AOCL DLP stays on), to hand to zentorch or vLLM.
- **Inputs:** the shared inputs.
- **Side effects:** checks out the ZenDNN ref in the ZenDNN checkout (a PR
  becomes a local `pr<N>` branch); deletes and recreates `<ZenDNN>/build`.
- **Produces:** the install tree `<ZenDNN>/build/install/zendnnl`, exported as
  `ZENDBUILDPATH`, and the commit it was built from.

### [build-vllm-onednn-zendnn](building/build-vllm-onednn-zendnn/SUBSKILL.md)
- **For:** building vLLM for CPU against upstream oneDNN and ZenDNN, without
  zentorch.
- **Inputs:** the shared inputs, plus the workspace directory containing
  `vllm/`; the vLLM ref (default: the commit its patch was made against); the
  oneDNN ref (ask every run, default `main`); wheel or editable install
  (default wheel).
- **Side effects:** everything build-zendnn does, because it runs build-zendnn
  itself. It also clones vLLM and oneDNN if they're missing, moves oneDNN to
  the chosen ref, and patches `vllm/cmake/cpu_extension.cmake`. It creates or
  reuses the conda env `vllm-build-env-312`, installs Rust under
  `<workspace>/.rust/`, and installs the built vLLM into that env.
- **Produces:** a vLLM build with ZenDNN enabled, verified at build time and at
  run time, plus `build.log`.

### [op-development-optimization](op-development-optimization/SUBSKILL.md)
- **For:** developing or optimizing an operator. Also open
  [windows](op-development-optimization/windows/SUBSKILL.md) or
  [linux](op-development-optimization/linux/SUBSKILL.md) for the platform, and
  the operator file in that folder when one exists.
- **Inputs:** the shared inputs, the operator, and the platform (default
  Linux).
- **Side effects:** edits operator source. A build or benchmark happens only
  when that sub-skill is chained in.
- **Produces:** the operator change. The shared procedure and the Linux notes
  are not written yet. The Windows file holds the MSVC portability checklist.

### [review](review/SUBSKILL.md)
- **For:** reviewing an operator change in the working tree, or a ZenDNN pull
  request. It hands every review to its
  [orchestrator](review/orchestrator/SUBSKILL.md), which opens the common
  checks, the Linux or Windows file, and the operator file when one exists.
  The pull-request workflow runs only when the user gives a specific PR URL
  and asks for a review.
- **Inputs:** the change to review or the PR URL, the operator, and the
  platform (default Linux). For a PR, any validation the user wants skipped,
  and `gh` signed in to GitHub.
- **Side effects:** a local review only reads the change. A PR review posts
  comments and a review decision on GitHub, and builds and tests in a
  separate worktree, leaving the user's checkout and build untouched.
- **Produces:** the findings in chat for a local review, or the posted review
  and its URL for a PR. The Linux-only and Windows-only review checks are not
  written yet; on Windows, the MSVC checklist applies to C++ and CMake edits.

### [running-tests](running-tests/SUBSKILL.md)
- **For:** running the gtest suite.
- **Inputs:** the shared inputs, plus any filter the user names.
- **Side effects:** may configure a gtest build. Confirm before replacing an
  existing build directory.
- **Produces:** the gtest result.

### [benchmark](benchmark/SUBSKILL.md)
- **For:** measuring an operator with BenchDNN.
- **Inputs:** the shared inputs, the operator, and the shape or input file.
- **Side effects:** may configure a BenchDNN build. Confirm before replacing
  an existing build directory.
- **Produces:** the BenchDNN result.

### [Windows MSVC checklist](op-development-optimization/windows/SUBSKILL.md)
- **For:** changing ZenDNN's C++ or CMake without breaking the native Windows
  (MSVC) build. Apply it alongside any task that edits C++ or CMake, including
  kernels, gtests, and dependency CMake. The same file is the Windows platform
  file for op-development-optimization, and the review Windows file applies
  it too.
- **Inputs:** none.
- **Side effects:** none; it guides the edit.
- **Produces:** a change that passes its review checklist, plus a Windows build
  and gtest run when a Windows machine with Visual Studio is available.

## Shared inputs

Collect these once per conversation and reuse them until the user changes them:

- **ZenDNN checkout:** default is the repository that contains this folder.
- **ZenDNN ref:** a branch, tag or PR number; default `main`.

## Chaining

- Run a multi-step request in order, handing each step's output to the next,
  such as the install tree from build-zendnn.
- build-vllm-onednn-zendnn runs build-zendnn as its first stage, so don't run
  build-zendnn separately before it.
- If a step fails, stop the chain and don't start the next step.

## Gaps

When no sub-skill covers the request:

1. Tell the user that no sub-skill covers it.
2. Work from the repository's documentation for that area, using only commands
   it documents. Paths are relative to the ZenDNN checkout:
   - builds with gtests, BenchDNN or examples, and Debug builds:
     `docs/zendnnl_build.md`
   - runtime environment variables: `docs/runtime_env.md`
   - logging and profiling: `docs/logging.md`
   - architecture and operators: `docs/zendnnl_architecture.md`, `docs/operator/`
3. Where documents disagree, the build files under `cmake/` and the source are
   authoritative. Say which one you followed.
4. Don't invent steps the documentation doesn't contain.

## Reporting

Finish with one summary: which sub-skills ran and in what order, the ZenDNN
commit they ran against, where the outputs are, and the result. If a step
failed, give the step, the exact command and the first real error line, then
wait for the user.
