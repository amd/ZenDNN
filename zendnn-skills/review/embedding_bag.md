# Embedding bag review checks

Checks for a change to embedding bag, embedding lookup, and group embedding
bag, under `zendnnl/src/lowoha_operators/embedding_bag/`. Apply them after
[common](common/SUBSKILL.md) and the platform file.

Reference: `docs/operator/low_overhead_operator/lowoha_embedding_bag_operator.md`.

## Functional correctness

- The AVX-512 kernels run only when `can_use_avx512_embag_kernel()` holds and
  `ZENDNNL_EMBAG_ISA` is not `avx2`. Otherwise a request that neither FBGEMM
  nor the reference takes runs `embag_avx2_kernel` for FP32 and BF16 on CPUs
  with AVX2, and returns `status_t::isa_unsupported` for other types (#811).
  An instruction added to `native_kernels/` that AVX-512F and AVX512-BF16 do
  not imply extends that predicate.
- An F16 table or output returns `status_t::isa_unsupported` without
  AVX512-FP16 from every entry point, on every backend except the reference.
- An all-empty group passes `group_embedding_bag_direct`'s size check, then
  reads `params[0].num_threads` from the empty `params`, and with
  `ZENDNNL_EMBAG_THREAD_ALGO=4` (`hybrid_threaded`) it also divides by
  `num_tables` (`eb_thread_qty % num_tables`). It must return before either
  happens.
- `group_embedding_bag_direct` picks a kernel and backend for every table on
  the calling thread, and returns before any table writes output when one has
  none (`EmbagIsa.NoKernelReturnsIsaUnsupportedBeforeRunning`).
- `is_weights = true` with a null `weights` fails on every backend. Only
  `embedding_bag_ref_direct` checks it; the native kernels dereference it.
- `dst_stride = 0` means `embedding_dim` on every backend, as the operator doc
  states. A nonzero stride below `embedding_dim` overlaps output rows, so it
  is rejected. Neither rule is implemented: no path resolves 0,
  `validate_embag_inputs` never reads `dst_stride`, and the kernels index rows
  as `oi * dst_stride` with no check. Every gtest and BenchDNN call sets the
  stride, so a change to stride handling needs both a `dst_stride = 0` case
  and a rejection of `0 < dst_stride < embedding_dim`.
- `can_use_fbgemm` stays false for `padding_idx >= 0`, any `algo` other than
  `sum`, s8 and s4 tables, and FP32 scale/bias. FBGEMM has no padding index
  and summed the padding row before #813
  (`EmbagFbgemm.PaddedSumSkipsPaddingRow`).
- A chunk that FBGEMM rejects (an index outside `[0, num_embeddings)`, or
  offsets that do not use the chunk's indices) makes the call return
  `status_t::failure`, since FBGEMM stops mid-chunk (#813,
  `EmbagFbgemm.RejectedIndexReturnsFailure`). Each chunk passes its own index
  count; passing `num_indices` made the prefetch read past the indices buffer.
- `padding_idx` is left out of the sum, the max, and the mean denominator. An
  empty or all-padding bag, and a lookup of `padding_idx`, write zeros.
- With `include_last_offset = false`, no kernel reads `offsets[num_bags]`, and
  the last bag ends at `num_indices`.
- Weights are documented as sum-only but are not rejected for `mean`, `max`,
  or lookup. For `max` and lookup, FP tables ignore them while INT8 and INT4
  tables scale by them, in the kernels and the reference alike. A change to
  weight handling rejects those cases or makes every table type agree;
  comparing each type with its own reference cannot catch the difference.

## Numerical correctness

- Every kernel and the reference dequantize a row as `scale * q + bias`, with
  `scale` and then `bias` stored right after the packed bytes. The operator
  doc's `Scale × (raw − ZeroPoint)` does not describe this; a change to
  either brings them in line.
- A quantized row is `D` bytes (s8) or `ceil(D/2)` bytes (s4, u4), then scale
  and bias at 2 bytes each (FP16) or 4 (FP32), with no padding. INT4 element
  `2k` is the low nibble of byte `k`, and s4 sign-extends bit 3. The kernels,
  the reference, FBGEMM's default input stride, and the gtest and BenchDNN
  table factories agree, odd `D` included.
- `embag_int4_w128_sum_specialized` runs only for `embedding_dim == 128`,
  `sum`, no weights, u4, and FP16 scale/bias, and matches the generic INT4
  path. Random `TestEmbag` widths rarely hit 128, so a change to it or to its
  gate adds `D = 128` and a neighbouring width.
- Any F16 table or output, FP32-to-F16 and F16-to-FP32 included, accumulates
  in FP16 when the F16 FMA kernels run (`__GNUC__ >= 12` and
  `can_use_f16_fma_kernel()`), and `ref_uses_f16_fma` gives the reference the
  same mode. Those pairs need FP16-level tolerance even with one FP32 side,
  and a change to either guard changes both.

## Memory and lifetime

- The INT8 and INT4 kernels keep their accumulators in a stack array of
  `MAX_ACC_BLOCKS` vectors, which an `embedding_dim` above
  `MAX_ACC_BLOCKS` × 16 (× 32 for F16 FMA) overflows. Only an `assert` guards
  it, release builds drop it, and validation sets no width limit. A change
  there enforces the bound in release builds or replaces the fixed array.
- Per-worker accumulator scratch is allocated only by workers that get a row,
  is null-checked, and reports failure as `status_t::failure`, as
  `embag_avx2_kernel` does (#811). The AVX-512 and F16 FMA kernels do not
  check `_mm_malloc`, and the `__GNUC__ < 12` BF16 branches of
  `embag_native_kernel` drop the AVX2 kernel's status.

## Concurrency and OpenMP

- A team smaller than requested does not skip tables. Today
  `table_threaded` strides tables by the requested count, `eb_thread_qty`;
  `ccd_threaded` strides them by `outer_threads`, which is `eb_thread_qty`
  over `CCD_NUM_THREADS` rounded up; and `hybrid_threaded` uses the thread id
  as the table index. With a smaller team, the tables meant for the missing
  threads never run and the call still succeeds. `hybrid_threaded` runs only
  when `num_tables` is below `eb_thread_qty`; otherwise the call falls
  through to the sequential `batch_threaded` loop.
- Nothing inside `group_embedding_bag_direct`'s parallel regions writes
  `embag_config_t`. Each table's kernel and backend are resolved before the
  region, and the reference receives its accumulation mode as an argument.
- `ccd_threaded` and `hybrid_threaded` both nest the kernels' parallel
  regions, so both need `scoped_active_levels(2)`. `hybrid_threaded` lacks
  it, so by default its inner teams run on one thread.

## Planner and integer arithmetic

- Bag and index loops use 64-bit counters, as `embag_avx2_kernel` has since
  #811. The other kernels and the reference still count bags, and some count
  indices, in `int`, which truncates past 2^31 − 1.

## Performance and overhead

- FBGEMM is compiled in only with `-DZENDNNL_DEPENDS_FBGEMM=ON`, off by
  default (#734), so default builds run `fbgemm` requests on the native
  kernels. A performance claim records `ZENDNNL_DEPENDS_FBGEMM` and the
  backend `select_embag_backend` returned, which `dispatch_avx512_kernel`
  logs as a `Using ... kernel` line with `ZENDNNL_COMMON_LOG_LEVEL` at 3 or
  higher. `kernel=` in the `embedding_bag_direct` apilog line is not that
  backend: it reads `native` for both the AVX-512 and the AVX2 kernels, and
  `group_embedding_bag_direct` prints no `kernel=`.
- Builds where `__GNUC__` is below 12 or undefined (GCC before 12, Clang,
  MSVC) run BF16 tables on `embag_avx2_kernel` even on AVX-512 hosts, so BF16
  results name the compiler.

## Tests and validation quality

- `TestEmbag`, `TestEmbedding`, and the group suites leave `padding_idx` at -1
  unless the gtest command line or input file sets one, and no `TestEmbagAI`
  case sets `use_padding_idx`. Without FBGEMM no test pins padding, so a
  change to padding adds explicit cases.
- A change to group thread mapping runs `TestGroupEmbag` and
  `TestGroupEmbedding` under all four `ZENDNNL_EMBAG_THREAD_ALGO` strategies,
  with fewer tables than threads in the `hybrid_threaded` cases (which
  otherwise run `batch_threaded`), plus a call from inside an active parallel
  region, where the team is smaller than requested.

## Validation

- gtests: `TestEmbag`, `EmbagIsa`, and, in FBGEMM builds only, `EmbagFbgemm`
  (`gtests/test_embag.cpp`); `TestEmbedding`; `TestGroupEmbag` and
  `TestGroupEmbedding` (`group_embag/`); `TestEmbagAI`. `TestEmbag`,
  `TestEmbedding`, and the group suites compare with
  `embag_kernel_t::reference` under the `EMBAG_*_TOL` values in
  `gtest_main.cpp` (INT8 uses `EMBAG_INT4_TOL`); `TestEmbagAI` compares with
  the Tensor Operator API's forced reference kernel.
- BenchDNN: `benchdnn/doc/embag.md`. It times `embedding_bag_direct` only, so
  group and lookup changes need another measurement.
- Shapes: the repository has no embedding bag input files; record the shapes
  used.
- Baseline: merge-base with the same FBGEMM setting, `ZENDNNL_EMBAG_ALGO`, and
  `ZENDNNL_EMBAG_THREAD_ALGO`. The repository sets no regression threshold.
- Paths: `ZENDNNL_EMBAG_ALGO=1` (native) or `2` (FBGEMM), which applies only
  while `params.kernel` is `none`; `params.kernel = embag_kernel_t::reference`
  for the reference; `ZENDNNL_EMBAG_ISA=avx2` with native selected
  (`ZENDNNL_EMBAG_ALGO=1` or `params.kernel = embag_kernel_t::native`) for the
  AVX2 kernel, since FBGEMM builds otherwise still send eligible sums to
  FBGEMM; `ZENDNNL_EMBAG_THREAD_ALGO` `1` to `4`;
  `-DZENDNNL_NATIVE_F32_ACCUM=ON` for FP32 accumulation instead of F16 FMA.
- ISA: AVX-512F and AVX512-BF16 for the native kernels, AVX512-FP16 for F16
  tables and outputs, and AVX2 for the fallback, which only a GCC build
  validates: Clang and MSVC builds compile every file for AVX-512.
- Builds: a change to `fbgemm_kernel.hpp` or the FBGEMM branches needs
  `-DZENDNNL_DEPENDS_FBGEMM=ON`; a change to the F16 FMA kernels needs GCC 12
  or newer and an AVX512-FP16 host.

## Windows only

- BF16 tables with FP32 output never reach FBGEMM on `_WIN32`
  (`can_use_fbgemm`), because FBGEMM's `EmbeddingSpMDM` throws for them there.
- `ccd_threaded` runs the kernels' parallel regions nested under a
  `thread_guard`, so on MSVC they need `num_threads(effective_num_threads())`.
  Without it they keep the full `eb_thread_qty` team instead of the inner
  guard's count, and oversubscribe.
