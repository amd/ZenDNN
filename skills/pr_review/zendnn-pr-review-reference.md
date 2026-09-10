# ZenDNN Detailed Review Checklist

Read this checklist before reviewing code. Apply every relevant section and
record anything that could not be validated.

## User-requested validation exclusions

The validation guidance in this file is enabled by default. If the user
explicitly says not to run a named activity, do not run it. Skip only the named
activity, continue all other applicable validation, and report it as
`skipped by user request`.

An execution exclusion does not remove the corresponding code-review duty.
Continue inspecting implementation correctness, existing test quality, and
whether the PR needs additional tests. Never describe excluded validation as
passed.

## Functional correctness

- Public API behavior matches code and documentation.
- Every new branch is reachable under the claimed configuration.
- Every fallback is correct and preserves semantics.
- Empty groups, zero-M experts, sparse experts, leading inactive experts,
  single expert, many experts, and capacity limits are handled.
- Mixed per-expert M/N/K values do not get collapsed into fictitious shapes.
- Active expert count is not confused with padded slot count.
- Row-major/column-major, transposition, alpha/beta, bias, and post-ops are
  preserved.
- Tight and padded `lda`, `ldb`, and `ldc` use element units correctly.
- BF16, F16, F32, S8/U8, S4/U4, symmetric/asymmetric, per-token,
  per-channel, and per-group paths use the correct element sizes and metadata.
- Fused activation changes output width and pointer arithmetic consistently.
- Environment overrides honor documented precedence, including invalid input.
- Release behavior does not rely only on debug assertions.

## Numerical correctness

- Reduction order changes have an independent reference or justified tolerance.
- Tests cannot pass with all-zero, stale, partially written, or repeated-block
  output.
- Test data avoids cancellation that makes the oracle ineffective.
- Tolerances reflect dtype and accumulation error; do not use a broad tolerance
  that exceeds expected output magnitude.
- Every distinct destination epilogue is covered.
- Nonzero tile offsets and boundary/tail tiles are validated.
- Negative-path tests prove the optimized path did not run.
- A capture hook verifies intended path reachability when silent fallback could
  make a test pass.

## Memory and lifetime

- Every allocation has one clear owner and matched deallocator.
- `malloc/free`, aligned alloc/free, and `new/delete` families are paired.
- Aliased pointers never outlive their backing vectors, arenas, or stack scope.
- Warmup and timed benchmark calls do not retain pointers to setup temporaries.
- Early returns and failed preparation release all owned buffers.
- Context reset clears every newly added state field.
- Thread-local vectors are not ownership leaks when destroyed at thread exit,
  but retained high-water memory is still audited and bounded where needed.
- Scratch sizes use checked arithmetic and cannot overflow.
- Stack arrays use named maximum constants, not fragile magic sizes.
- Cache ownership and mutable-weight behavior are correct.
- ASan/Valgrind inputs actually reach the changed lifetime branch; report ISA
  or instrumentation limitations.

## Concurrency and OpenMP

- Requested OpenMP team size is not assumed to be the actual team size.
- Reduced teams do not skip experts, rows, columns, or activation work.
- Every thread reaches barriers under uniform conditions.
- Atomic memory order is sufficient.
- Test capture hooks do not create production cache-line contention.
- Fixed thread roles cannot leave one pool as the unstealable long pole.
- Planner mappings remain valid under nested OpenMP and thread limits.
- Thread ID grouping is not assumed to equal physical CCD/L3 grouping unless
  affinity/topology is detected or enforced.
- Partial CCDs receive capacity-weighted work.
- Sorting and tie-breaking are deterministic and preserve intended balance.
- No data races occur in shared scratch, output tiles, caches, or telemetry.

## Planner and integer arithmetic

- Producer and consumer use the same floor/ceil semantics.
- Alignment is applied to the effective backend/kernel NR.
- Both sides of an imbalance bound are enforced.
- Per-expert constraints stay per-expert where inputs can differ.
- Inactive slots do not influence active-work planning.
- Products such as `K*N*bytes`, thread thresholds, and ceil division use
  checked 64-bit arithmetic.
- No signed overflow occurs before a later clamp.
- Capacity overflow has a correct and acceptably parallel fallback.
- A performance heuristic must not silently become a correctness condition.

## Performance and overhead

Audit hot paths for:

- per-call and per-tile heap allocations
- vectors that could be reused or stack-bound safely
- repeated environment parsing
- atomics/logging/capture loads in every tile
- scalar dtype conversion where vector conversion exists
- duplicate full-output scratch and copies
- unnecessary barriers or nested parallel regions
- serial fallback replacing a previously parallel supported path
- cache/CCD assumptions unsupported by deployment affinity
- work estimates that ignore N, K, dtype, or post-op cost
- benchmark claims measured only in a constrained runtime configuration

Require before/after evidence when:

- a production default changes
- AUTO routing changes
- a supported path becomes serial
- a new numerical kernel becomes automatic
- topology/affinity is part of the performance premise

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

## Tests and validation quality

- Selection-only tests do not substitute for execution tests.
- Branch-tag-only tests do not substitute for output parity.
- Every bug fix has a regression test that fails on the old code.
- Cached environment getters have deterministic overrides or process-isolated
  tests.
- RAII test guards restore prior state, including nested use where relevant.
- ISA-skipped tests do not hide a wrong expected branch on capable hardware.
- Test inputs reach the intended branch and assert that they did.
- Sanitizer tests exercise the changed dtype/configuration.
- Canary padding detects out-of-bounds or omitted writes.
- Heterogeneous and inactive-expert cases are included where the API permits
  them.
- Default/auto behavior is tested separately from forced override behavior.

## Code quality and documentation

- No unused variables, stale constants, unreachable tags, dead branches, or
  superseded comments remain.
- Names describe what predicates actually check.
- Inline comments explain durable invariants, not historical narration.
- Public docs, source docs, tests, PR body, and telemetry agree.
- Environment domains/defaults/invalid-value behavior are documented.
- PR checkboxes do not claim coverage that the code or tests contradict.
- Formatting, licenses, and repository conventions pass.
