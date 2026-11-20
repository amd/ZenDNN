# Matmul review checks

Checks for a change to scalar matmul, under
`zendnnl/src/lowoha_operators/matmul/`, outside `group_matmul/`. Apply them
after [common](common/SUBSKILL.md) and the platform file. A change that also
touches `group_matmul/` still opens [group_matmul](group_matmul.md).

`matmul_direct` in `lowoha_matmul.cpp` is the public entry. On current code
it tries, in order: native BRGEMM when that algorithm is selected, GGML
unpack, one-op `group_matmul_direct` when the call is grouped-routable or
requests a gated activation, then the scalar path (`kernel_select`, then
reference, BMM, or `matmul_execute`).

## Functional correctness

- Layout, transposition, alpha, beta, bias, post-ops, leading dimensions,
  batch counts, and dtype combinations stay correct on every backend the call
  can reach.
- Dynamic quantization, GGML unpack, and reorder-quant rewrite only the
  per-call copies. The caller's `matmul_params` and weight bytes stay
  unchanged except on the documented in-place weight-cache path.
- A declined grouped route falls through to the scalar path and produces the
  result that path would have produced on its own.
- An active gated activation has no scalar equivalent. If `group_matmul_direct`
  cannot run it, the call fails with that status and does not fall through.
- AOCL-prepacked weights (`mem_format_b == 'r'` with AOCL provenance),
  custom-kernel packed weights, and foreign prepacked weights dispatch only
  to a backend that can read that layout, or fail closed.
- Reference, AOCL-DLP, oneDNN, libxsmm, batched SGEMM, the auto-tuner, and
  native selection keep their current dtype, batch, bias, and post-op limits,
  including the fallbacks those limits already have.

## Post-ops and data types

For every post-op and every source, weight, destination, bias, and scale
dtype the PR adds, removes, or can newly reach, name the path that executes
it. A combination is supported only on a path whose code accepts it.
Highlight a change that moves a combination onto a path that did not accept
it, or drops one a path used to accept. An unsupported post-op is not
skipped. AOCL (`aocl_postop.cpp`) and oneDNN (`onednn_kernel.cpp`) `default`
cases drop an unknown post-op. That is a finding unless the PR adds the case
or refuses the path before execute.

Current pairing:

- **Group route.** BF16, INT8, and S4 variants, plus F16 only when the
  single-expert custom kernel covers that layer. Any post-op declines
  `single_matmul_group_routable` except static u8 INT8 when
  `int8_ck_postops_supported` says yes: one relu or sigmoid with no buffer,
  or f32 `binary_mul` then f32 `binary_add` with shape `[M, N]` and leading
  dimension at least N. Alpha is 1 and beta is 0.
- **AOCL-DLP**, including the default `aocl_dlp_blocked` path. Its chain
  takes relu, leaky relu, gelu tanh, gelu erf, sigmoid, swish, tanh, clip,
  mish, binary add, binary mul. A call with elu, square, abs, sqrt, exp,
  log, or softmax that would run on AOCL-DLP runs on the reference kernel;
  what that kernel does not serve (`reference_declines`: column-major,
  AOCL-prepacked or GGML packed weights, per-group weight scales without a
  source scale of the same groups, a bf16 or f32 source with s8 weights and
  no source scale, a binary operand that is neither `{1, N}` nor rows of N)
  and pooling return `status_t::unimplemented`. So do weight scales and
  INT8 alpha / beta it cannot apply (`aocl_dlp_applies_wei_scale`,
  `aocl_dlp_applies_wei_scale_type`, `aocl_dlp_applies_alpha_beta`), for every
  kernel that passes calls to AOCL-DLP. Under the auto tuner they apply only
  when a candidate runs the call on AOCL-DLP. `group_matmul_direct` refuses
  an operation with those post-ops unless it runs on oneDNN. Dtypes include f32, bf16 to bf16 or f32, u8 or s8 times
  s8, WOQ bf16 times s4 or u4, W4A8, and f16. WOQ, W4A8, symmetric quant, and
  a non-f32 int8 source scale are forced here by `kernel_select`.
- **oneDNN.** Post-ops: elu, relu, leaky relu, gelu tanh, gelu erf, tanh,
  square, abs, sqrt, exp, log, sigmoid, swish, mish, clip, binary add,
  binary mul. Not softmax or pooling. F16 and bf16 map to oneDNN types.
  M=1 with an f32 source leaves oneDNN. Unequal batch counts leave oneDNN.
- **libxsmm.** Post-ops: binary add, binary mul, gelu erf, relu, tanh,
  sigmoid, and swish only when alpha is 1. Dtypes: f32 to f32, bf16 to f32,
  bf16 to bf16. The Windows build rejects bf16. Anything else returns false
  and must not stay on libxsmm.
- **BMM.** Batch count greater than 1. `batched_sgemm` is the f16 batch path
  only with no bias and no post-op. Other batches use the selected BMM
  algorithm and must apply that algorithm's post-op and dtype limits.
- **Reference.** Elu, relu, leaky relu, gelu tanh, gelu erf, swish, sigmoid,
  tanh, softmax, square, abs, sqrt, exp, log, clip, mish, binary add, and
  binary mul. It is the fallback when a requested backend is not built, and
  it still rejects an AOCL-reordered weight.
- **Native BRGEMM.** The calls it serves, with their data types, scales,
  zero points, weight layouts, and post-ops, are the list in the Native
  BRGEMM section of
  `docs/operator/low_overhead_operator/lowoha_matmul_operator.md`: bf16
  times bf16, u8 or s8 times s8 (a bf16 or f32 source it quantizes
  included), s8 times symmetric s4, f32 times f32, and f16 times f16;
  per-tensor, per-row, or `{M, G}` source scales and per-tensor,
  per-column, or per-group weight scales; and every
  post-op and gated activation of `matmul_direct` except a softmax with a
  gated activation. Its epilogue (`native_epilogue.cpp`) applies them, mish
  included. A call outside that list returns an error status, or under
  `ZENDNNL_MATMUL_ALGO=10` without `lowoha_algo` goes back to the default
  route (see Native matmul below). Every call reads the weight's one
  layout: M = 1 on the GEMV kernels of `brgemm/kernel/gemv/`, whose
  results must keep the tiles' bits, every other M on the native tiles.
  Check the kernel that runs, not only `native_epilogue.cpp`.

Binary post-op leading dimension `-1` still becomes N. An f16 binary post-op
requires AVX512-FP16 on every route but the native BRGEMM: a call that asks
for the native BRGEMM takes f16 operands without it (it widens them to f32
with AVX-512 F), while a call ZENDNNL_MATMUL_ALGO=10 hands back to the
default route still needs it and gets `isa_unsupported` without it. Bias
dtype, scale dtype, and zero-point dtype travel with the same path as the
activation and weight. A chain is applied
in order. Dropping or reordering one entry changes the result.

## Hot-path overhead

- `matmul_direct`, `kernel_select`, and `matmul_execute` do not gain a
  per-call heap allocation, environment read, lock, or extra copy on a path
  that previously did none.
- A new branch sits behind a predicate the call already needed. A check that
  runs for every matmul, including shapes that never take the new path, is
  overhead. Flag it.
- Logging on the execute path stays behind the existing info or verbose
  switch.
- A cache lookup, fingerprint, or unreorder added for one backend is not
  done for backends that do not use it.
- Ask for before/after evidence when the diff touches the default
  `aocl_dlp_blocked` path, AUTO routing, or a per-tile loop.

## Routing into group matmul

The hand-off is `matmul_direct` calling `group_matmul_direct` for one op when
`single_matmul_group_routable` accepts the call, and for every active gated
activation. Treat that predicate and the success / decline handling as the
contract.

- Leave who enters and who declines that route unchanged unless the PR is
  about that route. If the predicate, the one-op argument packing, or the
  success / decline handling changes, highlight it: previous gate, new gate,
  and which shapes move.
- A successful grouped return returns immediately. It does not also run the
  scalar kernel.
- A declined grouped call, other than a gated activation, continues on the
  scalar path. It still runs reorder-quant and `kernel_select`.
- A mutated-weight conflict on the grouped attempt fails the public call. It
  does not fall through into a scalar reorder of the same buffer.
- Entry still requires effective algorithm `aocl_dlp_blocked`, agreeing
  weight-cache modes, constant weights, batch count 1, no `transA`, alpha 1,
  and beta 0, plus the existing dtype and post-op gates. An explicit oneDNN,
  libxsmm, reference, or native algorithm stays off this route.
- GGML weights are unpacked before the route. A still-packed GGML buffer does
  not enter `group_matmul_direct`.

## Framework prepack and weight cache

- Framework prepack (`mem_format_b == 'r'`, `reorder_direct`, GGML unpack)
  reaches only the backend that packed the weight. AOCL provenance stays on
  `aocl_dlp_blocked`. A custom-kernel image has no scalar consumer and fails
  closed when the grouped route does not take it.
- Foreign prepacked provenance is rejected.
- Weight-cache modes stay distinct. Mode 0 does not cache. Mode 1 caches out
  of place and leaves the caller's buffer intact. Mode 2 reorders in place
  only when a single backend owns that buffer.
- `auto_tuner` and `dynamic_dispatch` still downgrade mode 2 to mode 1 for
  the process. Those paths do not reorder in place.
- A scalar fallback whose weight can be reused by a later grouped
  custom-kernel call still caps the cache at mode 1, so the raw bytes
  survive. See `custom_kernel_cross_phase_weight_candidate`.
- A cache hit is reused only by the backend, layout, and dtype that produced
  it. Reset and test-only clear hooks drop every entry the change adds.
- Under `ZENDNNL_MATMUL_ALGO=10`, mode 2 may rewrite a buffer in place for
  the native BRGEMM. `lowoha_algo == native_brgemm` alone still caches out
  of place, and so does a weight whose N is not a multiple of 32 (its layout
  holds zero columns up to whole blocks). Do not extend in-place rewrite to
  any other algorithm.

## oneDNN, libxsmm, BMM, and the other scalar paths

With no per-op algorithm and no `ZENDNNL_MATMUL_ALGO`, `kernel_select`
resolves to `aocl_dlp_blocked`. The paths below are separate from a grouped
or native change. Highlight any edit that moves them.

- oneDNN and oneDNN-blocked (`matmul/backends/onednn/`) keep their entry
  conditions, the constant-weight requirement for the blocked variant, the
  M=1 F32 fallback, and the unequal-batch rejection.
- libxsmm and libxsmm-blocked (`matmul/backends/libxsmm/`), including the
  partitioner, run only when that algorithm is selected.
- Batch count greater than 1 selects the BMM algorithm (`get_bmm_algo`,
  default libxsmm) and `bmm::bmm_execute` under `matmul/partitioning/bmm/`.
  It does not use the single-matmul kernel or `group_matmul_direct`.
- `batched_sgemm` stays the F16 batch path only when bias and post-ops are
  absent.
- The reference kernel remains the fallback when a requested backend is not
  built, and it still rejects AOCL-reordered weights.
- Auto-tuner v1/v2 and the dynamic-dispatch heuristics are unchanged unless
  the PR is about them. If a heuristic result changes, name the shapes that
  move.

## Native matmul

`matmul/matmul_native/` stays off the default run until an algorithm selects
it. `native_brgemm_selected` is `lowoha_algo == native_brgemm` or
`ZENDNNL_MATMUL_ALGO=10`.

- A change in this folder does not change the algorithm or the result of a
  default call (`lowoha_algo == none`, algo unset), which stays on
  `aocl_dlp_blocked`. Flag it if it does.
- Dynamic dispatch and the auto-tuner never return `native_brgemm`: the
  native BRGEMM runs only the calls that ask for it. Flag a PR that makes
  them pick it.
- The native BRGEMM serves any N: its layout pads a weight to whole
  32-column blocks with zero columns, and the last column tile reads and
  writes only the columns below N. Flag a change that reads or writes an
  operand or dst past N.
- A call with `lowoha_algo == native_brgemm` that uses a dtype, layout,
  post-op, or shape the native BRGEMM does not implement returns an error
  status (unimplemented, isa_unsupported, or failure for a weight it cannot
  read), with dst untouched and one error line naming the reason; it never
  takes another route. Tests decide the expected status with
  `native_brgemm_refusal`, not with a copy of the rules.
- Under `ZENDNNL_MATMUL_ALGO=10`, a single call without `lowoha_algo` that
  the native BRGEMM refuses with unimplemented or isa_unsupported is handed
  back to the default route (`native_layout_matmul` returns false), after
  the default route's own input checks, with its weight cache capped at
  mode 1 and its raw weight held against an in-place rewrite while the
  default route reads it. It keeps the native error instead when its
  arguments are wrong (ldb checked against a weight row of the caller's
  layout), when its buffer holds an image a native call wrote in place,
  and for the forms `kept_from_default_route` lists, which the default route
  computes wrongly or does not implement (in a build without AOCL-DLP also
  what `kept_in_this_build` adds for the reference kernel). Flag a hand-back of a form without a test that
  compares its dst with an independent model, and a kept form whose error
  line does not say why it is not handed back.
- Do not call native code from the `aocl_dlp_blocked`, oneDNN, libxsmm, BMM,
  or reference paths.
- The offline tuner (`tools/native_brgemm_tuning`) follows the native
  planner: its schema fixture holds the planner's keys, match keys, rule
  features, kernels and plan-line fields, and `plan.py` / `cost_model.py`
  reproduce every rule of the built-in plan and the cost model. A change to
  `matmul_native/brgemm/planner/`, to a `ZENDNNL_MATMUL_NATIVE_CONFIG` key, a
  match key or a rule feature, to the kernel registry, or to the
  `[MATMUL.NATIVE_BRGEMM]` line updates the tuner in the same PR: the mirror
  follows, `refresh_fixtures.sh` rewrites both fixtures, and
  `NativeConfigSchema.TheTunerFollowsThePlanner`,
  `NativeConfigSchema.PlanLineFields`,
  `NativeConfigText.KeyDocsMatchTheKeyTable`, the plan comparison (the
  build's plan dump of the `tests/plan_cases.py` cases against `plan.py`:
  `plan_cases.py compare --schema <the SchemaDump log>`, 0 mismatches; the
  schema checks do not see a rule, cost-model term or default) and the
  tuner's tests
  (`python3 -m unittest discover -s tools/native_brgemm_tuning/tests -t tools`)
  pass. Flag a planner change that leaves the tuner behind, and a fixture
  refreshed without the mirror following it.
- The op performance utility (`tools/op_perf_utility`) follows the native
  kernels, the cost model and BenchDNN: its probe times the kernels'
  dot-product instructions, its bounds take each weight type's compute path,
  its calibration writes the cost model's `cm_*` inputs, and its reader
  parses BenchDNN's matmul output. A change to a kernel's dot-product
  instruction or kind, a matmul data type, scale granularity or post-op, a
  `cm_*` input, its units or the calls the cost model prices, the fields of
  `native_hw()` or `native_plan_uarch()`, or BenchDNN's matmul CSV or result
  line updates the utility in the same PR, and
  `python3 -m unittest discover -s tools/op_perf_utility/tests -t tools`
  and the tuner's tests pass. Flag a change that leaves the utility behind.
  Also flag committed machine data: a ceilings file, numbers measured on a
  lab machine, or the identity or figures of an unreleased part.

## Caches and kept state

Run this section on every matmul PR, whatever its diff touches. A change to
routing, a dtype, a layout, a kernel, a post-op or a test can make a cached
value stale, read another route's image as raw bytes, or keep memory that no
clear frees, and each of these returns success with wrong values. Apply the
generic checks in [common](common/SUBSKILL.md) too.

What a matmul call keeps across calls (extend this list when a PR adds state):

- Native layouts (`matmul_native/common/native_weight_cache.cpp`), keyed on
  the weight pointer, K, N, ldb, pack_nr, transB, the weight type and whether
  the buffer is an AOCL-DLP prepack. At weight cache mode 2 under
  `ZENDNNL_MATMUL_ALGO=10`, the layout replaces a bf16, f16 or s4 weight (or
  an AOCL-DLP prepack) in the caller's buffer, and an image record remembers
  it; a clear keeps the record as a tombstone, with the bytes it keeps (up
  to 64 KiB a record, outside the layout capacity), which no clear frees.
  Change detection reads the two 64-byte ends of the bytes a layout was
  packed from. A call an image may serve (at its start, of its layout)
  checks two 64-byte lines only: the first and the last where the image
  differs from the weight it replaced (its own ends may hold the weight's
  bytes). Any other call on an image record or tombstone compares the bytes
  it keeps (all of an image up to 64 KiB, the two ends of a larger one)
  wherever the call's buffer overlaps them, and those two lines and up to
  1024 sample lines of a larger one. A cached layout of bytes
  an image holds serves only while that image is there: it names the record
  of the image it was packed from or carried across.
  A dropped layout is freed once no call can read it, by the next call that
  drops a layout or caches one it packed (or finds the cache full), or by a
  clear; calls served a cached layout or an image free none.
- The widened f16 sources (`matmul_native/brgemm/looper/`), at most 32
  buffers and 512 MiB, and the per-thread staged rows of gated calls
  (`matmul_native/common/native_epilogue.cpp`), kept up to 4 MiB a thread;
  the clears release the widened sources, and the staged rows of the
  clearing thread at once and of another thread at its next staged call or
  its exit (rows above 4 MiB stay on an idle thread until then).
- Native plans (`matmul_native/brgemm/planner/`) and the values read once at
  the first native call: CPU features, `ZENDNNL_MATMUL_WEIGHT_CACHE`,
  `ZENDNNL_MATMUL_NATIVE_CONFIG`, `ZENDNNL_MATMUL_NATIVE_TUNING_FILE` and
  `ZENDNNL_MATMUL_NATIVE_F16_ACCUM`.
- The default route's AOCL-DLP and oneDNN weight caches, keyed on pointer and
  shape with no check of the bytes. Calls the native BRGEMM hands back fill
  them too.

Checks:

- A native layout depends on the weight alone. Flag a layout or image that
  starts to depend on M, the post-op, the source type, the thread count or
  the weight cache mode.
- A native plan may depend on all of those, and its cache key holds every
  input it reads: flag a new field a plan reads that `native_problem_t`, its
  `operator==` and its hash do not hold.
- Under `ZENDNNL_MATMUL_ALGO=10`, every reader of a weight buffer checks for
  an in-place image or holds the bytes before it reads, and caps its own cache
  at mode 1 when native calls may read the buffer later. This covers a new
  route, a hand-back, group matmul, the reorder API and an explicit
  `lowoha_algo` call. Flag a path that reads a buffer another route rewrote in
  place as raw bytes, in either direction.
- A change that rewrites more weights in place, or widens mode 2, says in the
  API header, the operator doc and `docs/runtime_env.md` what a framework may
  no longer do with that buffer: read it, save it, update it, or share it with
  another tensor.
- A clear at mode 2 is process-wide: it turns the image records of every
  model in the process into tombstones, and a weight that stays loaded fails
  its calls until its raw weight is reloaded. Flag a doc or test that treats
  a clear as safe while in-place weights stay loaded.
- `clear_matmul_native_weight_caches()` reaches only the native state.
  Handed-back calls keep the default route's caches, which only
  `clear_matmul_weight_caches()` clears. A doc that names a clear for reloaded
  weights under `ZENDNNL_MATMUL_ALGO=10` names the one that reaches both.
- New native state is freed by `clear_all_weight_caches()`, and
  `benchdnn/matmul/matmul_lowoha.cpp`'s clear between lines reaches it.

For a change to a cache, an image, a clear or a route, test each scenario
below against an independent reference. Each must give the right result or a
documented, loud failure, never success with wrong values:

- a clear while in-place weights stay loaded, another model's weights
  included;
- a weight allocated inside a freed image, as in a model swap without a
  clear;
- a view of a weight (rows or columns, at the buffer's start or inside it)
  cached before its buffer is rewritten in place, then the buffer rewritten
  again for another weight;
- a view's own raw elements written back over an image, after a clear or
  with none, then the whole weight called;
- a raw weight (or AOCL-DLP prepack) written back over its image with no
  clear, as a model loaded again at the same address, for every weight type
  rewritten in place;
- a framework that reads or updates a weight after its first mode-2 call: a
  tied embedding, a checkpoint save, a LoRA merge;
- a handed-back constant weight refilled in place, then each clear;
- another route's in-place image under each `ZENDNNL_MATMUL_ALGO` setting;
- a clear from an exit hook, and `exit()` while a call runs;
- a forked child after the caches filled, at modes 1 and 2.

Test matrix for every matmul PR, on its head. Compare each failure with the
merge base, and name a failure the base also shows as pre-existing:

1. `--gtest_filter='Native*:*/Native*:Int8InPlaceWarm.*:*/Int8InPlaceWarm.*:AIGemvTests*:*/AIGemvTests*'`,
   under `ZENDNNL_MATMUL_ALGO=10` and unset, each in its own process under
   `ZENDNNL_MATMUL_WEIGHT_CACHE` 0, 1 and 2. A test that asserts a cached
   layout, an image or a pack count pins the mode it needs
   (`WeightCacheGuard`), so the mode-0 run fails only on outputs. Flag a new
   test that relies on the process default instead.
2. The same filter with `--gtest_shuffle` for three seeds. The cache suites
   (`NativeWeightLifetime*`, `NativeCacheModes*`, `NativeOneLayout*`,
   `NativeLayoutRuleRoute*`, `NativeBrgemmRefusal*`) also run with
   `--gtest_repeat=2`.
3. When the PR touches a retire, free or clear path: an ASan build of
   `NativeWeightLifetime*` and `NativeBrgemmRefusal*`, and RSS measured over
   many refills of one weight and over many distinct weights.

## Tests and validation quality

- A routing change has one test that shows the call entered
  `group_matmul_direct` and one that shows a declined call stayed on the
  scalar path. Both check the destination against an independent oracle.
- Prepack and each weight-cache mode are tested with the buffer that mode
  claims to preserve or reuse. Mode 2 is tested only for the single backend
  that owns the buffer.
- oneDNN, libxsmm, and BMM tests still pass when the diff can reach their
  sources or their selection. A native-only diff still runs one default-path
  case that shows those backends were not selected.
- Native tests force `native_brgemm` (algo 10). They do not stand in for the
  default path.
- A fallback test shows the declined kernel did not run, and that the
  destination matches the backend that did.
- Each post-op and dtype pair the PR claims is tested on the path that
  accepts it. The output differs from the matmul with that post-op removed,
  so a skipped epilogue cannot pass. A pair the path does not accept has a
  test that the call fell back or failed, and that the unsupported path did
  not run.
