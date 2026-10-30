# Reorder review checks

Checks for a change to LowOHA reorder, under
`zendnnl/src/lowoha_operators/reorder/`. Apply them after
[common](common/SUBSKILL.md) and the platform file.

Reference:
`docs/operator/low_overhead_operator/lowoha_reorder_operator.md`.

`reorder_direct` in `lowoha_reorder.cpp` is the main entry. Mode precedence is
mutually-exclusive prepack/unreorder rejection, weight prepack, weight
unreorder, dynamic quantization, then standard conversion/static
quantization. The other entries are the unchecked
`dynamic_per_token_quant_bf16_s8_native`, native-only
`group_dynamic_quant`, sequential `group_reorder`, and the prepack/unreorder
size and data-movement helpers.

```text
reorder_direct
  ├─ both prepack flags       -> failure
  ├─ is_prepack              -> weight_prepack_into
  ├─ is_unreorder            -> weight_unreorder_into
  ├─ dynamic_quant
  │    ├─ eligible fast shape -> fused/unfused dynamic dispatcher
  │    └─ otherwise           -> compute params -> standard reorder
  └─ standard                -> validate -> select algo -> reorder_wrapper
```

## Functional correctness

- `is_prepack` and `is_unreorder` together fail before either mode touches a
  buffer. Each selected mode ignores only the fields its documented contract
  says it ignores; adding a mode does not change the established precedence.
- Standard reorder accepts exactly the dtype pairs in `is_reorder_supported`;
  identity and unsupported pairs fail before dispatch.
- Execution supports positive 1D, 2D, or 3D source shapes and an identical
  destination shape. Rank greater than three must fail without stores; do not
  infer rank support only because a shape helper accepted it. Destination
  storage is contiguous; `dst_strides` remains unsupported.
- Source strides have the shape rank and positive element strides. A fast
  path that assumes contiguous rows accepts only an empty stride vector or a
  unit innermost stride and uses the correct row pitch.
- `reorder_direct` allows `dst == nullptr` only for dynamic-quant
  compute-only mode. Standard conversion, prepack, and unreorder require
  caller-owned source and destination buffers.
- The void `dynamic_per_token_quant_bf16_s8_native` remains unchecked:
  callers provide positive M/N, contiguous non-overlapping capacity, and ISA.
- `group_dynamic_quant` keeps all per-op vectors aligned with `M.size()`.
  Every op has nonnegative M, positive K, and valid `{row_stride, 1}`
  source/destination strides with row pitch at least K. Active ops additionally
  have non-null buffers and K divisible by G for per-group; M=0 skips only
  those active-only pointer and divisibility checks.
- `group_reorder` rejects an empty group or vector-size mismatch, executes
  each op through `reorder_direct` in order, stops on the first error, and
  does not promise rollback of already completed outputs.
- `group_dynamic_quant`, `group_reorder`, and the prepack size helpers are
  declared without `ZENDNNL_API`. A change that claims shared-library
  consumer support adds the export decoration and a shared-build link test.

## Current support matrix

Use the detailed matrix in
`docs/operator/low_overhead_operator/lowoha_reorder_operator.md` for design
background. The bullets below are the current executable review boundary; a
change that expands one row also updates validation, dispatch, tests, and that
document.

- Static quant/dequant supports f32, bf16, or f16 activation storage against
  s8/u8. Scale storage is f32/bf16/f16 and zero point is s32. Float-only
  conversions may omit both; any change to whether integer conversions may
  omit metadata agrees across validation, kernels, tests, and documentation.
- Quantization metadata has the tensor rank and one documented shape:
  per-tensor, row/column channel, or a proper row/column group. Group counts
  are positive proper divisors of the grouped dimension. Scale and zero-point
  dims are each valid for the shape; mixed granularities keep their existing
  independent indexing.
- A `{N}` 1D per-channel descriptor is not claimed as covered until a test
  proves every metadata index is consumed; the row/column helpers used by
  scalar granular reorder must not collapse every element onto index zero.
- Dynamic quant supports f32/bf16/f16 to s8 when zero point is absent and to
  u8 when an s32 zero-point output is present. Scale output is mandatory.
  Compute-only mode writes parameters but not destination data.
- Dynamic fast paths cover 2D per-token and contiguous 2D per-group-column
  shapes. A decline falls through to `compute_dynamic_quant_params` followed
  by standard reorder without changing scale dtype, granularity, or results.
- `group_dynamic_quant` is narrower than `reorder_direct`: bf16/f32 source,
  s8 destination, f32/bf16 scales; per-channel currently requires bf16.
  It has no scalar fallback and returns `isa_unsupported` without
  AVX-512F/BW/VL.
- AOCL weight prepack covers its current float, S8, S4/U4 WOQ, W4A8, static
  U8xS8, and grouped-symmetric S8 formats. Custom-kernel prepack covers
  BF16/F16/S8 only. Unreorder covers AOCL F32/BF16/F16/S8 only; CK, S4/U4,
  and grouped-symmetric packs are not invertible.
## Dispatch, ISA, and numerical paths

- `DT` selects native only at the current size threshold and only with
  AVX-512F/BW/VL. An explicitly requested native algorithm falls back to
  reference when that ISA is unavailable.
- Per-channel/per-token/per-group static metadata still routes through the
  granular scalar implementation unless a specific optimized path proves it
  supports the same indexing. Do not infer support from the per-tensor
  native kernel.
- For eligible 2D per-token input, `ZENDNNL_DYNAMIC_QUANT_ALGO=1` selects
  vector fused; values 2/3 select vector-unfused/scalar-fused only with a
  contiguous source. For contiguous 2D per-group-column input, only value 1
  forces fused execution; values 2/3/4 fall through to parameter computation
  plus normal algorithm selection. Unset/invalid values keep default behavior.
- FP16 native dispatch uses `can_use_f16_fma_kernel`: the build-time
  `ZENDNNL_NATIVE_F32_ACCUM` kill switch, compiler support, and runtime
  AVX512-FP16 check stay in agreement. The F32-FMA path remains the fallback.
- Scalar, F32-FMA, and FP16-FMA paths use the same scale floor, rounding,
  saturation, and zero-point contract.

## Prepack, unreorder, memory, and lifetime

- `weight_prepack_size` and `weight_prepack_into` validate the same K/N/ldb,
  transpose, dtype, algorithm, group-size, pack-NR, and interleave contract.
  Callers reuse identical params between size query and write.
- Every prepack writes into caller storage. The API does not know capacity:
  the caller allocates at least the queried, alignment-rounded byte count.
  Any size-formula change updates writer, consumer, documentation, and a
  canary/sanitizer test together.
- AOCL prepack supports only the layouts its matmul consumer can identify.
  Static U8xS8 appends the aligned int32 column-sum tail exactly when the
  consumer will read it. Prepack `src_dtype` and matmul source dtype must
  agree; a missing tail is an out-of-bounds read, not only wrong math.
- In a build without AOCL-DLP, `aocl_dlp_blocked` prepack currently produces
  its explicit dense reference fallback, while unreorder is unavailable.
  A consumer never labels those dense fallback bytes as AOCL-blocked.
- Custom-kernel prepack supports its current bf16/f16/s8 families, pack width
  32/64 dividing N, s8 K-quad alignment, and even-N interleave. Its output is
  CK-only and a non-CK consumer fails closed.
- Reorder prepack itself does not publish a persistent weight-cache entry or
  retain the caller pointer. A change must not create both a caller-owned
  packed buffer and an unbounded duplicate cache image.
- `prepack.cached_size` resets before every size query, records only a
  successful prepack size, and is not overwritten by unreorder.
- Unreorder accepts only AOCL-DLP f32/bf16/f16/s8 layouts with no symmetric
  group pack. It preserves the packed source, writes the caller's plain
  buffer using destination ldb/transpose, and uses the same m-hint contract
  as the matching pack.
- Temporary f32 scale vectors used to narrow bf16/f16 dynamic scales are
  scoped to the call and sized to the exact scale count. New staging does not
  add a second full tensor copy or survive the call.
- Matmul integrations keep quantized source/scale/ZP storage in their RAII
  owners or bounded thread-local pools. AOCL-to-CK conversion may hold one
  temporary raw image while producing the final CK arena, but publishes only
  the final image and frees the temporary.
## Framework and operator integration

- `reorder_quantization_wrapper` in matmul owns or borrows every generated
  source/scale/ZP buffer through its RAII holders. On reorder decline it keeps
  the original source metadata and intentionally returns success so the
  caller can continue on its non-quantized path; direct API failure semantics
  are not copied into that soft-fallback wrapper.
- Thread-local quantization pools are bounded/released through their existing
  reset path. A pointer into a pool is not retained beyond the lifetime
  promised by the wrapper or across a later call that may resize the arena.
- `ZENDNNL_ENABLE_GROUP_DQ` controls the grouped-matmul prepass only; it does
  not govern single-op `matmul_direct`. Flash and varlen SDPA invoke
  `group_dynamic_quant` independently, including per-channel V quantization.
  Every integration updates only owned/per-call metadata and preserves caller
  descriptors and source buffers.
- Model-load prepack through `group_reorder` stays outside the inference hot
  loop. Framework code queries size once, owns the packed storage, records
  the exact backend provenance, and keeps it alive through the last consumer.
- A framework-facing API addition has an exported symbol, installed header,
  stable ownership contract, and shared-library link test. In-tree/static
  visibility alone is not public support.

## Error handling and return contracts

- With diagnostics enabled, invalid standard and dynamic inputs return
  `status_t::failure`. Group API checks and unreorder validation are
  unconditional; unavailable grouped ISA returns `isa_unsupported`, and
  unsupported prepack/unreorder combinations keep their documented status.
- Size-query APIs return zero on rejection and never expose a partial size.
  Data-movement APIs propagate backend errors and do not advertise a failed
  destination as valid.
- AOCL prepack entry points return through void backend reorder calls, unlike
  unreorder's metadata error check. A change must not claim backend failure
  propagation unless the called API exposes and tests it.
- `validate_reorder_inputs`, dynamic parameter/shape validation, and prepack
  parameter/input validation run inside `op_instrumentation::validate` and
  may be skipped with `ZENDNNL_DIAGNOSTICS_ENABLE=0`. New or changed code does
  not rely on them for memory safety: required preconditions are checked
  unconditionally, as current mode-conflict, dynamic-stride, grouped, and
  unreorder pointer checks are.
- The void native per-token API cannot report rejection; no new validation
  or fallback assumption is assigned to it without changing its API contract.

## Configuration, logging, performance, and threading

- `ZENDNNL_DIAGNOSTICS_ENABLE` may remove diagnostic validation. Checks whose
  absence permits invalid pointer arithmetic, buffer access, or shape/stride
  execution are added outside that gate.
- `ZENDNNL_DYNAMIC_QUANT_ALGO` changes only the documented dynamic strategy;
  `ZENDNNL_ENABLE_GROUP_DQ` changes only the grouped matmul prepass; neither
  silently changes the public dtype or granularity contract.
- `ZENDNNL_DLP_M_HINT`, transpose, thread hint, K/N/ldb, and dtype form one
  AOCL pack identity. Size, pack, cache lookup, and unreorder use compatible
  values or decline.
- `ZENDNNL_NATIVE_F32_ACCUM` is a build-time FP16-FMA kill switch. There is
  no reorder runtime environment override for the FP16 backend; runtime ISA
  selection remains authoritative.
- API visibility uses `ZENDNNL_API_LOG_LEVEL`; profiling uses
  `ZENDNNL_ENABLE_PROFILER` and `ZENDNNL_PROFILE_LOG_LEVEL`. A new log does
  not execute formatting or inspect optional metadata when its channel is
  disabled.
- `group_reorder` stays sequential because each reorder may create its own
  OpenMP team. Grouped dynamic quantization owns one team and does not skip
  active rows when the actual team is smaller than requested.

## Tests and validation quality

- Standard/static/conversion/stride changes run the corresponding
  `TestReorder` cases under `gtests/reorder/`.
- Dynamic changes cover symmetric/asymmetric, compute-only, every affected
  granularity, source stride, scale dtype, fused path, and fall-through path.
- Shape/index validation changes cover rank greater than three, 1D
  per-channel metadata, checked element-count overflow, zero scale, and a
  non-finite scale outside element zero where those contracts are touched.
- Prepack/unreorder changes run
  `gtests/test_static_quant_prepack.cpp`,
  `gtests/reorder/test_lowoha_unreorder_aocl.cpp`,
  `gtests/reorder/test_aocl_raw_weight_cache.cpp`, and group-matmul
  prepack/unreorder route tests when those consumers can change.
- Group API changes run `gtests/reorder/test_group_reorder.cpp` and
  `gtests/group_matmul/e2e/test_group_per_group.cpp`, including M=0,
  heterogeneous shapes, bad vectors/strides, and first-failure behavior.
- Grouped dynamic-quant changes also run dense and varlen INT8 coverage in
  `gtests/test_sdpa.cpp`, including QK, PV, windowed/causal, ragged,
  and per-channel V-quantization paths.
- BenchDNN changes under `benchdnn/reorder/` include the dtype, granularity,
  stride, and algorithm whose behavior changed.
