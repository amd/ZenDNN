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
- **AOCL-DLP**, including the default `aocl_dlp_blocked` path. Post-ops:
  relu, leaky relu, gelu tanh, gelu erf, sigmoid, swish, tanh, clip, mish,
  binary add, binary mul. Dtypes include f32, bf16 to bf16 or f32, u8 or s8
  times s8, WOQ bf16 times s4 or u4, W4A8, and f16. WOQ, W4A8, symmetric
  quant, and a non-f32 int8 source scale are forced here by `kernel_select`.
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
- **Native.** f32 to f32; bf16 to bf16 or f32; u8 or s8 times s8 to f32,
  bf16, s8, or u8. `transA`, mish, and a swish alpha other than 1 fall back
  to AOCL-DLP. Native BRGEMM also falls back when alpha is 0. INT8 scales
  stay per-tensor for the source, and per-tensor or per-channel for the
  weight. Per-group and per-token INT8 fall back. The GEMM and INT8 BRGEMM
  loopers fuse only relu, gelu tanh, gelu erf, sigmoid, tanh, and swish, and
  send every other accepted post-op through the remaining epilogue. Check
  the kernel that runs, not only `native_epilogue.cpp`.

Binary post-op leading dimension `-1` still becomes N. An f16 binary post-op
still requires AVX512-FP16. Bias dtype, scale dtype, and zero-point dtype
travel with the same path as the activation and weight. A chain is applied
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
- Under `ZENDNNL_MATMUL_ALGO=11`, mode 2 may rewrite a buffer in place for
  the native BRGEMM. `lowoha_algo == native_brgemm` alone still caches out
  of place. Do not extend in-place rewrite to any other algorithm.

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
`ZENDNNL_MATMUL_ALGO=11`. `matmul_execute` also enters
`native_matmul_execute` when `kernel_select` returns `native_gemm` or
`native_brgemm`.

- A change in this folder does not change the algorithm or the result of a
  default call (`lowoha_algo == none`, algo unset), which stays on
  `aocl_dlp_blocked`. Flag it if it does.
- Dynamic dispatch and the auto-tuner may already return `native_brgemm` for
  documented small-M BF16 and single-thread INT8 shapes. A native-folder
  change may affect those shapes only while that algorithm is selected.
  Highlight a PR that adds new default or AUTO shapes to that set.
- A native dtype, `transA`, post-op, or shape the kernel does not implement
  falls back to AOCL-DLP. The fallback writes a correct result and does not
  leave a partial destination. `native_layout_matmul` returns without stores
  when the call does not qualify.
- Do not call native code from the `aocl_dlp_blocked`, oneDNN, libxsmm, BMM,
  or reference paths.

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
- Native tests force `native_gemm` or `native_brgemm` (algo 10 or 11). They
  do not stand in for the default path.
- A fallback test shows the declined kernel did not run, and that the
  destination matches the backend that did.
- Each post-op and dtype pair the PR claims is tested on the path that
  accepts it. The output differs from the matmul with that post-op removed,
  so a skipped epilogue cannot pass. A pair the path does not accept has a
  test that the call fell back or failed, and that the unsupported path did
  not run.
