# Normalization review checks

Checks for a change to normalization (LayerNorm, RMSNorm, FusedAddRMSNorm,
FusedLayerNormAdd, FusedGatedLayerNorm, and BatchNorm), under
`zendnnl/src/lowoha_operators/normalization/`. Apply them after
[common](common/SUBSKILL.md) and the platform file.

Reference: `docs/operator/low_overhead_operator/lowoha_normalization_operator.md`.

## Functional correctness

- The FP32-accumulating AVX-512 kernels run only when the runtime gate covers
  what their `#pragma GCC target` enables: AVX-512F, BW, DQ, VL, BF16, and
  FMA. `normalization_kernel_wrapper` checks only `get_avx512f_status()`, so
  writing BF16 faults on a CPU with AVX-512F but no AVX512-BF16; embedding
  bag's gate checks both (#811). An instruction added to `kernel/` extends
  the gate.
- Without `-DZENDNNL_NATIVE_F32_ACCUM=ON`, an F16 buffer the call uses (src,
  dst, gamma with `use_scale`, beta with `use_shift` on LayerNorm and
  FusedLayerNormAdd) returns `status_t::isa_unsupported` before dispatch on a
  host without AVX512-FP16. Only `norm_algo_t::reference`, BatchNorm, and
  FusedGatedLayerNorm, which converts F16 with F16C, skip this gate.
- Checks whose absence gives silent wrong output or undefined behavior stay
  outside `op_instrumentation::validate`, which `ZENDNNL_DIAGNOSTICS_ENABLE=0`
  skips: the F16 ISA gate, f16/bf16 mixing between `src_dt` and `dst_dt`, a
  null residual on FusedAddRMSNorm and FusedLayerNormAdd (the kernels drop
  the add), `residual == output` on FusedLayerNormAdd, and
  `validate_gated_norm_params`.
- In-place calls (`output == input`, `src_dt == dst_dt`) must keep working:
  BenchDNN's normalization driver runs them by default (`isInplace`), and the
  doc allows them for FusedGatedLayerNorm. No kernel reads an input element
  after storing the output element over it. That order is not enough for the
  non-gated row kernels: they declare `in_row` and `out_row` `__restrict`, so
  aliasing them is undefined behavior however the loads and stores are
  ordered. In-place support there needs an alias-safe pointer contract or a
  dedicated in-place path.
- FusedAddRMSNorm stores `residual + input` into the residual in `src_dt`,
  and pass 2 normalizes the rounded value it reads back, as the reference
  does. FusedLayerNormAdd reads its residual in `dst_dt` and never writes it
  (`residual_unmodified` in `TestNormalization`).
- An F16 kernel returns `status_t::unimplemented` for a combination or norm
  type it does not run (its stub, built when `__GNUC__` is below 12 or
  undefined, returns `isa_unsupported`), and only those two statuses fall
  through to the FP32 kernel. A new F16
  combination goes into the eligibility flags in
  `normalization_kernel_wrapper` and the kernel's selector together.
- FusedGatedLayerNorm strides count elements, 0 means `norm_size`, and a
  nonzero stride below `norm_size` fails. Strides and `norm_params::gated`
  fields fail on the other norm types rather than being ignored
  (`GatedLayerNormGuards`).

## Numerical correctness

- The AVX512-FP16 RMSNorm, FusedAddRMSNorm, LayerNorm, and FusedLayerNormAdd
  kernels accumulate each sum in four 32-lane `__m512h` registers over
  128-element blocks, fold the 32-lane cleanup and the tail into the first,
  and widen each to FP32 with `reduce_add_ph_to_fp32` before summing.
  `reference_kernel.cpp` mirrors this with `F16_ACCUM_LANES` and
  `simd_accum_lane`; both sides change together, or the F16 gtests stop
  matching.
- Each dispatch branch that runs an AVX512-FP16 kernel sets
  `params.accum_type` to f16 first and restores f32 when it falls through.
  The reference reads it, `norm_algo_t::reference` does not reset it, and the
  gtests rely on that by passing one `norm_params` to the native and
  reference runs.
- FusedAddRMSNorm accumulates in FP32 unless built with
  `-DZENDNNL_FUSED_ADD_RMS_F16=ON`, and then only when `src_dt`, `dst_dt`, and
  `gamma_dt` are all f16. FP16 accumulation of the in-place add and the sum of
  squares lost too much precision (#556); `-DZENDNNL_NATIVE_F32_ACCUM=ON`
  overrides the flag.
- F16 gtests compare the kernels with a reference that copies their FP16
  rounding, which shows parity, not accuracy. A change that moves more work
  into FP16 also measures its error against FP32 accumulation on long rows.
- The SIMD LayerNorm kernels compute variance as `E[x²] − mean²` clamped at
  zero, and the FP32 reference uses two passes. `TestNormalization` inputs are
  zero-mean, which hides the cancellation, so a change to the statistics pass
  is also tested on rows whose mean is large against their spread.
- FusedGatedLayerNorm's silu is `group_matmul_act_avx512::silu_avx512`
  (fast exp, relative error about 5e-5), and the reference uses `std::exp`.
  A change to that silu in `group_matmul_act_avx512.hpp` re-runs
  `GatedLayerNorm`.

## Concurrency and OpenMP

- Every kernel but BatchNorm's splits a call by rows only, so `batch <= 1`
  runs on one thread; BatchNorm splits by `batch` × `num_channels`. A change
  that splits a row across threads keeps the output independent of the thread
  count, which `GatedLayerNorm.ThreadInvariant` checks only for the gated
  type.

## Performance and overhead

- With the row split, small-`batch` decode shapes use few threads, so a
  performance claim states `batch`. BenchDNN runs in place by default, and the
  merge-base run uses the same `isInplace`.

## Tests and validation quality

- `TestNormalization` compares only the output on FusedAddRMSNorm. A change
  to the fused add also compares the updated residual with
  `residual_tensor_ref`.
- A change to a row loop's blocks or tail adds sizes on both sides of each
  block width (16 and 64 elements in the FP32 kernels, 32 and 128 in the FP16
  ones) to `NormalizationTail` (#513). FusedAddRMSNorm is not in that suite;
  its F16 tail runs only in `NormalizationFusedTailF16`, built with
  `-DZENDNNL_FUSED_ADD_RMS_F16=ON` (#556).
- No `GatedLayerNorm` case runs with `output == input`; a change to the gated
  chunk loop adds one.

## Validation

- gtests: `TestNormalization`, instantiated as `Normalization` and
  `NormalizationTail`, and `NormalizationFusedTailF16`
  (`gtests/test_normalization.cpp`); `GatedLayerNorm` and
  `GatedLayerNormGuards` (`test_gated_layer_norm.cpp`). The oracle is
  `norm_algo_t::reference` with `NORM_F32_TOL`, `NORM_BF16_TOL`, and
  `NORM_F16_TOL` from `gtest_main.cpp`; the gated test has its own FP64
  oracle.
- BenchDNN: `benchdnn/doc/normalization.md`. It has no FusedGatedLayerNorm
  type, and it logs `isa_unsupported` rows as failures. A row the host's ISA
  cannot run counts as not validated; a row that succeeded at merge-base on
  the same hardware and build flags and now returns `isa_unsupported` is a
  regression.
- Shapes: the repository has no normalization input files; record the shapes
  used.
- Baseline: merge-base with the same input lines (`algorithm`, `num_threads`,
  and `isInplace` included) and build flags. The repository sets no
  regression threshold.
- Paths: `reference` in the BenchDNN `algorithm` field, or
  `norm_algo_t::reference`, for the scalar kernel;
  `-DZENDNNL_NATIVE_F32_ACCUM=ON` to drop the FP16 FMA path;
  `-DZENDNNL_FUSED_ADD_RMS_F16=ON` to add FusedAddRMSNorm's;
  `ZENDNNL_DIAGNOSTICS_ENABLE=0` to confirm the always-on checks still run.
- ISA: AVX-512F with BW, DQ, VL, BF16, and FMA for the FP32 kernels, and
  AVX512-FP16 for the FP16 kernels and, by default, any F16 buffer.
  `GatedLayerNorm` skips without AVX-512F, BF16, and BW/VL.
- Builds: code under `ZENDNNL_FUSED_ADD_RMS_F16` needs that flag, and the FP16
  kernels need GCC 12 or newer; other compilers build stubs that fall through.

## Windows only

- MSVC drops `collapse`, so the `collapse(2)` over `batch` × `num_channels` in
  `batch_norm_impl` parallelizes over `batch` alone and a single image runs on
  one thread. A BatchNorm change there adds the flattened MSVC loop.
