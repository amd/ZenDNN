# ZenDNN pull-request validation reference

Read this before validating a pull request. The code-review checks are in
[common](../common/SUBSKILL.md). This file covers how much to build, test, and
benchmark, and how to compare a pull request against its merge base.

## User-requested validation exclusions

The validation guidance in this file is enabled by default. If the user
explicitly says not to run a named activity, do not run it. Skip only the named
activity, continue all other applicable validation, and report it as
`skipped by user request`.

An execution exclusion does not remove the corresponding code-review duty.
Continue inspecting implementation correctness, existing test quality, and
whether the PR needs additional tests. Never describe excluded validation as
passed.

## Risk-based validation depth

Classify risk from behavior, not line count. A one-line default change can be
higher risk than a large test refactor.

The tiers below define the default validation depth. Apply every listed
activity except those explicitly excluded by the user.

### Tier 0 — non-behavioral

Examples:

- comments or spelling only
- formatting-only changes
- test naming with no test logic change
- documentation aligned to already-shipped behavior

Validation:

- diff integrity
- documentation/link/format checks that apply
- no build or benchmark unless the diff unexpectedly touches generated code,
  build configuration, public declarations, or executable examples

### Tier 1 — localized low-risk fix

Examples:

- narrow bug fix with a clear existing regression test
- defensive validation before execution
- no default, kernel, allocator, layout, concurrency, or public API change

Validation:

- build the affected target
- run the new regression test and nearby suite
- prove the test fails on the old behavior when practical
- exercise one adjacent negative/fallback case
- BenchDNN is normally unnecessary unless the touched code is hot or the PR
  makes a performance claim

### Tier 2 — moderate feature or execution-flow change

Examples:

- new supported dtype/layout/post-op
- routing or planner rule that is not the global default
- changed quantization metadata
- new allocation/lifetime path
- changed thread partitioning in a bounded regime

Validation:

- Release build and affected regression suites
- default, forced-on, forced-off, fallback, and invalid-input paths
- cross-product of relevant dtype/layout/stride/shape boundaries
- sanitizer or reduced-team/topology testing when applicable
- baseline-vs-PR BenchDNN when execution cost or a hot path can change

### Tier 3 — high-risk/default/kernel/scheduler change

Examples:

- production default or AUTO policy change
- numerical kernel, packing format, or reduction-order change
- memory ownership or pointer lifetime change
- OpenMP mapping, atomics, barriers, or CCD/topology scheduling
- broad public API behavior or fused pipeline change
- cache/prepack policy
- a supported path becoming serial

Validation:

- isolated, equivalent builds of merge-base and PR head
- all affected functional suites and meaningful neighboring suites
- execution-level regression tests with independent oracles
- sanitizer/UB checks relevant to the changed risk
- default and all escape-hatch/fallback configurations
- BenchDNN matrix using the protocol below
- model/profile benchmarks when the optimization premise is model-level
- longer review time is expected; never reduce validation merely to finish
  quickly

Escalate one tier when:

- tests are missing or only selector/tag based
- the code is architecture-gated and cannot be exercised locally
- the PR force-pushes away a previously measured behavior
- performance evidence conflicts across microbenchmark and model levels

## Building a functionality-preservation matrix

Derive the matrix from changed behavior:

If the user excluded the corresponding accuracy or regression-test execution,
use the matrix to inspect coverage and report gaps, but do not execute the
excluded cases.

1. List each new capability, changed default, gate, and fallback.
2. Identify the old supported behavior that shares its code path.
3. Identify boundary values immediately below/at/above each gate.
4. Add sparse, empty, single-item, many-item, heterogeneous, and capacity-limit
   cases where the API allows them.
5. Add every relevant dtype, destination epilogue, layout, transpose, bias,
   activation, quantization granularity, and padded stride.
6. Run the PR's default path.
7. Run forced optimized and forced fallback paths.
8. Compare against an independent reference or merge-base behavior as
   appropriate.
9. Verify all output, including canary padding and inactive destinations.
10. Verify path reachability separately so a silent fallback cannot pass.

For a bug fix, the minimum useful proof is:

```text
old code + regression input -> fails
new code + regression input -> passes
new code + neighboring valid inputs -> still pass
new code + fallback/negative inputs -> still take the correct fallback
```

Do not require an old-code failure when the test is inherently non-deterministic
or destructive; explain the limitation and use a stronger independent oracle.

## BenchDNN decision

Unless the user explicitly excludes BenchDNN or performance testing, run
BenchDNN when the PR can affect runtime cost:

- kernel, packing, reorder, quantization, fusion, cache, or prepack changes
- AUTO/default algorithm changes
- M/N/K tiling, planner, OpenMP, CCD, affinity, or work-pool changes
- scratch allocation, conversion, copy, barrier, or atomic changes
- a performance claim in the PR
- a bug fix on a hot path where the fix may alter throughput

BenchDNN is usually unnecessary for:

- documentation-only or comment-only changes
- isolated test-only changes
- build metadata with no generated/runtime impact
- trivial validation fixes before any hot computation

When unsure, inspect the call frequency and hot-path placement. Benchmark the
small change if it executes per element, per tile, per expert, or per inference.

## BenchDNN baseline protocol

Benchmark merge-base and PR head, not merely two forced modes in the PR binary.
Use identical:

- compiler and build type
- dependencies and build flags
- OpenMP runtime
- thread count, affinity, CPU/NUMA placement, and environment
- input files, warmups, timed iterations, cache mode, and machine state

Protocol:

1. Build baseline and PR in isolated worktrees.
2. Confirm both binaries execute the intended paths.
3. Keep the machine otherwise idle; avoid benchmarking while unrelated tests
   or builds consume the same CPUs.
4. Pin CPU/NUMA placement and record topology, OpenMP runtime, and affinity.
5. Warm both arms before collecting data.
6. Run enough repeats to establish a noise floor. Prefer at least five
   interleaved baseline/PR pairs for stable kernels; increase repeats for noisy
   or short cases.
7. Alternate order (`baseline, PR`, then `PR, baseline`) to reduce thermal,
   frequency, and drift bias.
8. Cover:
   - PR-advertised model/profile shapes
   - boundary shapes around new gates
   - common existing workloads sharing the path
   - relevant dtypes and thread counts
   - default behavior and documented escape hatch
   - hot and cold cache modes when the change concerns caching
9. Report median per case, aggregate geomean where useful, noise floor, and
   worst individual regression. Never hide a supported-case regression behind
   an aggregate gain.
10. Re-run suspicious outliers with order reversal and increased repetitions.
11. Preserve commands, input files, environment, raw CSV/output, and summary.
12. Clean generated benchmark artifacts after recording results.

Use repository-defined acceptance thresholds when available. Otherwise do not
invent a universal percentage. Compare against measured noise and explain why
each observed change is significant or inconclusive.

Performance is not validated when:

- baseline and PR use different dependencies or affinity
- only best-of-N is reported without the full distribution
- the benchmark does not reach the changed path
- only a forced mode is compared while the shipped default differs
- microbenchmark gains conflict with model-level results and the conflict is
  unresolved

## Scaling review time intelligently

Apply this scaling only to non-excluded validation activities.

- Tier 0: finish after focused inspection and lightweight validation.
- Tier 1: spend enough time to prove the regression and nearby compatibility.
- Tier 2: allow builds, targeted matrices, sanitizer work, and BenchDNN.
- Tier 3: expect an extended review with parallel analysis, baseline/head
  builds, repeated benchmarks, and possibly model/profile runs.

Do not impose a fixed time budget. Stop expanding only when added tests no
longer address a plausible risk introduced by the diff. Do not run broad,
expensive suites for a minor change solely to appear thorough.

