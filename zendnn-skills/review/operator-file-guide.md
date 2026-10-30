# Writing an operator review file

Guide for operator owners adding or extending `review/<operator>.md`. An
operator file holds the review checks that only make sense for one operator.
It sharpens the review; it does not replace any part of it.

## How the file is used

Every review, local or pull request, opens these in order:

1. [common](common/SUBSKILL.md): checks for every operator.
2. The platform file: [linux](linux/SUBSKILL.md) or
   [windows](windows/SUBSKILL.md).
3. The operator file for each operator the change touches.

An agent reads the file and applies each check to the diff. Write for that
reader: it already knows C++, OpenMP, and the common checks. It does not know
your operator's invariants, failure history, or which inputs reach which path.

An operator file adds checks. It never relaxes, overrides, or repeats a check
in common or the platform files.

## Where a check belongs

Decide placement per check, not per file.

| The check is true for | Put it in |
|---|---|
| Any operator, on any platform | `common/SUBSKILL.md` |
| Any operator, on one platform only | `linux/` or `windows/SUBSKILL.md` |
| One operator family (for example SDPA and HSTU attention) | The family's shared file, or the parent operator file |
| One operator | That operator's file |
| One operator, on one platform only | That operator's file, under `Linux only` or `Windows only` |

To test a check, ask whether another operator's code could break it. Rewrite
it without your operator's own terms (experts, bags, heads, KV cache, block
formats). If the rewritten rule names something other operators have, it is
generic. If only your operator has it, the check stays local, however
general it sounds.

When you can't tell, keep the check local. Common loads on every review of
every operator, so a check placed there by mistake costs every run. Move it
up when a second operator needs it.

Split a mixed check. Keep the generic part in common and the operator part in
your file. For example, "Reduced teams do not skip experts, rows, columns, or
activation work" became "Reduced teams do not skip rows or columns" in common
and "Reduced teams do not skip experts or activation work" in
[group_matmul](group_matmul.md).

A local check may sharpen a common one with operator values. Common says
tolerances reflect dtype and accumulation error. Your file can say which
accumulation your operator uses and what tolerance that justifies.

When two operator files carry the same check, move it up: to common if any
operator could hit it, or to the family's file if only one family can. Delete
it from the operator files in the same change.

## Aspects to work through

Go through each aspect for your operator. Write a check only where your
operator has a rule that common does not already state.

- **API contract:** public `*_direct` entry points, argument validation,
  return status on rejected input, behavior that docs promise, and what
  zentorch, vLLM, and TensorFlow rely on.
- **Shapes and boundaries:** zero sizes, tails, the smallest and largest
  supported dimensions, and values on each side of a dispatch threshold.
- **Data types and quantization:** supported combinations, scale and
  zero-point granularity, and the metadata each one needs.
- **Layouts and strides:** supported formats, leading dimensions, transposes,
  and padded or non-contiguous inputs.
- **Post-ops and fusion:** which post-ops apply, their order, and how fusion
  changes output shape or pointer arithmetic.
- **Dispatch and fallback:** backends, algorithm selection, AUTO rules, the
  environment variables that force a path, and what each fallback must keep
  the same.
- **ISA gating:** instruction sets each kernel needs, and the runtime CPU check
  that guards it.
- **Caching and prepacking:** weight caches, prepacked layouts, and the
  constant-weights condition they rely on.
- **Threading:** how work is partitioned across threads, nested regions, and
  what must hold when the team is smaller than requested.
- **Memory:** scratch sizes, ownership, and lifetimes particular to the
  operator.
- **Numerics:** accumulation order and type, the reference to compare
  against, and the tolerance per data type.
- **Performance:** the hot loop, the shapes that matter in production, the
  baseline the operator must not fall behind, and the regressions that block
  a merge. Use measured numbers only.
- **Tests and benchmarks:** the gtest fixtures and BenchDNN driver that cover
  the operator, and the cases a change must add.
- **Known failure modes:** past bugs, written as the invariant they broke.

## How specific to write

Each check is one bullet that names an invariant a reviewer can confirm or
refute from the diff, the source, or a test.

- State what must hold, not what to be careful about.
- Name the trigger when it isn't obvious: the shape, data type, flag, or
  configuration that reaches the case.
- Use stable names: entry points, file paths, gtest fixtures, environment
  variables. Avoid line numbers and current default values unless the check
  is about that default.
- Write a past bug as the invariant it broke. A pull-request number in
  parentheses is fine as evidence.
- When the diff alone can't confirm a check, say what does: a gtest fixture,
  an assertion, or a BenchDNN case.
- When confirming a check needs a build, test, or benchmark, say which changes
  need it, so other reviews skip it.
- In each section, list first the checks whose failure is P0 or P1.
- Leave out general C++ advice, style preferences, and anything the agent
  already applies from common.

| Too vague | Checkable |
|---|---|
| Be careful with experts. | Active expert count is not confused with padded slot count. |
| Make sure the tests are good. | Tests cannot pass with all-zero, stale, partially written, or repeated-block output. |
| Check the AVX-512 path. | The AVX-512 embedding bag kernels run only after the runtime CPU check passes (#811). |

## What stays out

- Checks already in common or a platform file.
- The validation procedure, risk tiers, and BenchDNN protocol. They are in
  [review-reference](orchestrator/review-reference.md).
- MSVC portability rules. They are in the
  [Windows checklist](../op-development-optimization/windows/SUBSKILL.md).
- Design explanations and support matrices. Link the operator document under
  `docs/operator/low_overhead_operator/` instead of copying it.
- Commands, flags, or checks the repository does not document or support.
- Placeholders. Leave a section out until it has a check.

## File format

- **Name:** the operator's directory under `zendnnl/src/lowoha_operators/`,
  for example `reorder.md` or `embedding_bag.md`. A nested component with its
  own public entry point and tests gets its own file named after its
  directory, such as `group_matmul.md` for `matmul/group_matmul/`. Internals
  without their own entry point, such as backends, partitioning, or
  quantization helpers, are covered by the parent's file.
- **Shared family file:** operators that share most of their checks share one
  file named after the family, such as `attention.md` for `sdpa/` and
  `hstu_attention/`. It has one section for the shared checks and one per
  operator.
- **Family checks across files:** an operator file that needs another file's
  checks says so in its first lines with a link, for example "Also apply
  matmul" linking to `matmul.md`.
- **Headings:** use the section names from common, so findings map onto the
  same areas. Add `Validation`, `Linux only`, and `Windows only` when needed.
  Include only sections that have checks.
- **Length:** every review of the operator loads the whole file, so each line
  costs time on every run. A file over about 150 lines usually repeats common
  or copies documentation.

## Template

```markdown
# <Operator> review checks

Checks for a change to <operator>, under
`zendnnl/src/lowoha_operators/<dir>/`. Apply them after
[common](common/SUBSKILL.md) and the platform file.
<Optional: Also apply [<parent>](<parent>.md).>

Reference: `docs/operator/low_overhead_operator/lowoha_<operator>_operator.md`.

## Functional correctness

- <invariant>

## Numerical correctness

- <invariant>

## Memory and lifetime

## Concurrency and OpenMP

## Planner and integer arithmetic

## Performance and overhead

## Tests and validation quality

## Code quality and documentation

## Validation

- gtests: <fixtures or files under `gtests/`>
- BenchDNN: `benchdnn/doc/<driver>.md`
- Shapes: <input files in the repository for the production shapes>
- Baseline: <what a change is compared against, and the regression that
  blocks a merge>
- Paths: <environment variables that force the optimized path and the
  fallback>
- ISA: <instruction sets a kernel needs, so a skip is recorded as not
  validated>

## Linux only

## Windows only
```

## Before you submit

- Every check names an invariant someone can verify.
- No check repeats common, a platform file, or another operator file.
- Generic parts that another operator's code could break are split out and
  proposed for common. Anything uncertain stays local.
- The file is named after the operator's directory or its family, and the
  sections that have no checks are removed.
- Every path, fixture, environment variable, and document named in the file
  exists in the repository.

## Ownership

The operator owner reviews changes to the operator file. A change to common
or a platform file needs agreement from the owners of the operators it
affects. Update the operator file in the same pull request that changes the
invariant it describes.

When a defect in the operator is found after review, add the invariant it
broke in the pull request that fixes it. Delete checks for code that no
longer exists.
