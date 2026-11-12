---
name: review-common
description: >-
  Checks every ZenDNN review runs, on Linux and Windows, for a local operator
  change and for a pull request: functional and numerical correctness, memory
  and lifetime, OpenMP concurrency, planner arithmetic, hot-path overhead,
  test quality, code quality, and documentation. Also the severity scale and
  the rule that a finding is raised only when a change is required. Opened
  through the review orchestrator.
---

# Common review checks

Every review applies this file, then the platform file
([linux](../linux/SUBSKILL.md) or [windows](../windows/SUBSKILL.md)), then the
operator file in `review/` when one exists. The
[orchestrator](../orchestrator/SUBSKILL.md) decides which ones open.

Apply every relevant section. Record any item that could not be validated
rather than silently treating it as passed.

## Checks

### Functional correctness

- Public API behavior matches code and documentation.
- Every new branch is reachable under the claimed configuration.
- Every fallback is correct and preserves semantics.
- Row-major/column-major, transposition, alpha/beta, bias, and post-ops are
  preserved.
- Tight and padded `lda`, `ldb`, and `ldc` use element units correctly.
- BF16, F16, F32, S8/U8, S4/U4, symmetric/asymmetric, per-token,
  per-channel, and per-group paths use the correct element sizes and metadata.
- Environment overrides honor documented precedence, including invalid input.
- Release behavior does not rely only on debug assertions.

### Numerical correctness

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

### Memory and lifetime

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

### Caches and kept state

Apply these to any state that outlives a call: caches, pools, image records,
plans, per-thread buffers and values read once. An operator file with a
section of the same name adds that operator's state and test matrix, which
runs on every PR of that operator.

- List the state the PR adds:
  `rg -n '\bstatic\b|thread_local' <changed files>`, minus `static_cast`,
  `static_assert` and `constexpr`. For each one, record what it holds, its
  key, its entry and byte bounds, who frees it, which public clear reaches it,
  its locking, and what happens at exit and in a forked child.
- The key holds every input the cached value reads, or the value provably
  does not depend on it. Inputs to check include dtypes, leading dimensions,
  transposes, quantization parameters, the thread count, the ISA, env vars
  and tuning files.
- A pointer-keyed cache is tested with a weight refilled at the same address
  and shape, once with the bytes its change check reads changed and once with
  only other bytes changed, against the result at a fresh address. Its docs
  state what the check detects and nothing more.
- Every public clear frees what its header, the operator doc and the env docs
  say it frees, and nothing a running call may still read: the quiescence
  contract or a generation counter covers each such read. A test fills the
  new state, clears it, checks that the kept bytes are zero, and checks the
  next call against an independent reference.
- Every kept buffer has an entry and a byte bound, says what happens when the
  bound is reached, and says how large a per-thread buffer grows and when it
  shrinks.
- Process-wide state that a call or a clear can reach at exit is never
  destroyed (allocated with `new` and kept). A test runs the clear from an
  exit hook, and `exit()` while a call runs.
- A new mutex is a fork hazard: a child forked after the caches filled is
  tested, and the docs say when a fork is safe.
- A value read once is documented as read once. Tests change it in a child
  process or through a pin that restores it, never with `setenv` after the
  first call.
- The suites that touch cached state pass with `--gtest_shuffle` for three
  seeds and with `--gtest_repeat=2`. A test that needs cold caches clears
  them at its start, or keeps its buffers until exit.

### Concurrency and OpenMP

- Requested OpenMP team size is not assumed to be the actual team size.
- Reduced teams do not skip rows or columns.
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

### Planner and integer arithmetic

- Producer and consumer use the same floor/ceil semantics.
- Alignment is applied to the effective backend/kernel NR.
- Both sides of an imbalance bound are enforced.
- Products such as `K*N*bytes`, thread thresholds, and ceil division use
  checked 64-bit arithmetic.
- No signed overflow occurs before a later clamp.
- Capacity overflow has a correct and acceptably parallel fallback.
- A performance heuristic must not silently become a correctness condition.

### Performance and overhead

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

### Tests and validation quality

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
- Default/auto behavior is tested separately from forced override behavior.

### Code quality and documentation

- No unused variables, stale constants, unreachable tags, dead branches, or
  superseded comments remain.
- Names describe what predicates actually check.
- Inline comments explain durable invariants, not historical narration.
- Public docs, source docs, tests, PR body, and telemetry agree.
- Environment domains/defaults/invalid-value behavior are documented.
- PR checkboxes do not claim coverage that the code or tests contradict.
- Formatting, licenses, and repository conventions pass.

## Findings

Raise a finding only when a code, test, documentation, or design change is
required.

### Severity

- **P0**: security issue, data corruption, crash, use-after-free, deadlock, or
  broad catastrophic behavior.
- **P1**: merge blocker: wrong output, skipped work, material default-path
  regression, untested automatic numerical path, or unsupported public
  behavior.
- **P2**: required correction with narrower impact: misleading contract,
  significant avoidable overhead, ineffective test, telemetry error, or
  configuration bug.
- **P3**: small but necessary correctness/documentation cleanup. Use sparingly.

### What a finding contains

Every finding must contain:

1. Severity
2. Concrete triggering input/configuration
3. Actual behavior
4. Expected behavior
5. User-visible correctness/performance/maintenance impact
6. Requested code or test change
7. Evidence: code path, calculation, test failure, sanitizer output, or
   benchmark

Template:

```text
[P1] <concise defect statement>

With <specific shape/configuration>, <producer> computes/chooses <X>, but
<consumer> uses <Y>. This causes <concrete impact>. Please <required change>
and add <specific regression test>.
```

Do not raise a finding when:

- the change is correct and needs no action
- only wording/style preference is involved
- the scenario is impossible under validated API contracts
- performance concern has no hot-path or reachable evidence
- the requested change is optional
