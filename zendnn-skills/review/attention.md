# Attention operator review checks

Checks for a change to SDPA, under
`zendnnl/src/lowoha_operators/sdpa/`, or HSTU attention, under
`zendnnl/src/lowoha_operators/hstu_attention/`. Apply them after
[common](common/SUBSKILL.md) and the platform file. Apply "Both
operators" first, then each operator the change touches.

Reference: `docs/operator/low_overhead_operator/lowoha_sdpa_operator.md`
and
`docs/operator/low_overhead_operator/lowoha_hstu_attention_operator.md`.

## Both operators

### Functional correctness

- A query row with no visible key writes zeros.
- A sequence with no queries writes nothing.
- Q, K, V, and the output use their token and head strides, including a
  fused-projection view and a gap between heads.

### Numerical correctness

- The attention accumulator is FP32 for f32, bf16, and f16.

### Concurrency and OpenMP

- `sdpa_direct` and `hstu_attention_direct` return a failure after the
  OpenMP region. They do not throw or return inside it.
- Scratch for that region is allocated before the region starts.
- A flash or tiled work item writes only its own output rows.
- A flash or tiled work item lies inside one sequence.
- A flash or tiled work item is one non-empty query tile of one
  (sequence, head) in a single flat `parallel for`.
- The SDPA reference uses `collapse(2)` over batch or sequence and head,
  and each iteration is one whole sequence.
- The HSTU reference runs one query row of one head on a dynamic
  schedule.

### Code quality and documentation

- Every `ZENDNNL_SDPA_*` and `ZENDNNL_HSTU_ATTENTION_*` variable in the
  source is in `docs/runtime_env.md`, with its default and invalid
  values.
- A change to BF16 `pack_rows`, `pack_cols`, `packed_size`, or `pack_b`
  in `sdpa_gemm_ck/` or `hstu_gemm_ck/` is applied to the other tree.

## SDPA

### Functional correctness

- Query head `h` reads K/V head `h / (num_heads / kv_num_heads)`.
- `kv_num_heads == 0` selects `num_heads` K/V heads.
- Output does not depend on the query tile size: every size
  `balanced_q_split_size` can pick, including a partial last tile, gives
  the reference output with causal, window, and mask bands.
- `balanced_q_split_size` returns either `preferred` itself or a smaller
  multiple of 8 that is at least 32, and returns `preferred` when that is 32
  or less or one thread runs the call. Both packed kernels pass
  `varlen_min_gain_percent`; the dense kernels pass no margin.
- Its `tasks_for(split)` counts the tiles the loop runs at that split:
  `batch * num_heads * ceil(q_len / split)` dense, and
  `num_heads * varlen_tile_count(...)` packed, matching
  `build_varlen_tiles`.
- A dense causal or window band places query `i` at key `i`.
- A packed band places row `r` of a sequence at key
  `r + kv_len - q_len`, with `q_len` and `kv_len` that sequence's
  lengths, including when they differ.
- The window band is `|i - j| <= sliding_window_size - 1`, and
  `is_causal` drops the right half, in flash FP, flash INT8, packed,
  and reference.
- A skipped KV tile holds no key inside that band.
- A packed sequence attends only to its own keys.
- Each packed query length, `cu_seqlens_q[b + 1] - cu_seqlens_q[b]`, is
  at most `seq_len`.
- Each packed key length, `cu_seqlens_k[b + 1] - cu_seqlens_k[b]`, is
  at most `kv_seq_len`, where 0 means `seq_len`.
- A null `cu_seqlens_k` uses `cu_seqlens_q` for the key lengths.
- A packed-mode change leaves the output of a dense call
  (`cu_seqlens_q == nullptr`) unchanged.
- A packed sequence with `q_len == kv_len` gives the dense output for
  that length.
- Dynamic INT8 runs only for BF16 Q, K, and V. Another `qkv_dt` stays
  on the FP path, and `sdpa_direct` logs a warning.
- `is_qk_quant` and `is_pv_quant` select independent modes.
- A set `ZENDNNL_SDPA_INT8_QK` overrides `is_qk_quant`.
- A set `ZENDNNL_SDPA_INT8_PV` overrides `is_pv_quant`.
- A feature the selected kernel does not implement returns
  `status_t::unimplemented`, for example `attn_mask` in packed mode.
- A missing ISA returns `status_t::isa_unsupported`: f16 without
  AVX512-FP16, and dynamic INT8 without AVX512-F, AVX512-BW,
  AVX512-VL, and AVX512-VNNI.
- With `ZENDNNL_DIAGNOSTICS_ENABLE=0`, a guard of a formed pointer, an
  ISA requirement, or packed `attn_mask` still runs in `sdpa_direct`.
- The dense validator enforces a limit dense mode can reach. Only the
  packed validator enforces a packed-only limit.

### Numerical correctness

- With `is_qk_quant`, Q and K are stored as int8. Without it they stay
  BF16.
- With `is_pv_quant`, P is stored as u8 and V as int8. Without it they
  stay BF16.
- The score buffer stays FP32 in every dynamic INT8 mode.
- A row masked in every KV tile writes zeros, not NaN from the `-inf`
  row maximum.
- A row masked only in its first KV tile, and live later, writes its
  unmasked output.
- The softmax scale is applied once: in the FP score step, in the
  per-row Q scales when `is_qk_quant` is set, and in QK `alpha` when
  only `is_pv_quant` is set.
- The additive mask is added after that scale and is not scaled.
- The online-softmax accumulator starts at the first live KV tile with
  `beta = 0` and no rescale.
- Packed INT8 V scales come from that sequence's own V rows.

### Memory and lifetime

- Matmul receives K, V, and their INT8 copies with `is_weights_const`
  false, and the INT8 helper sets `weight_cache_type = 0`.
- The calling thread releases flash scratch through
  `sdpa_flash_cpu_free_scratch()`.

### Concurrency and OpenMP

- The thread count passed to `balanced_q_split_size` is the
  `num_threads` of the tile loop it sizes, and `loop_team_size` caps it to
  the team that loop gets: the thread limit, and one thread inside an
  active parallel region, since the loops run under
  `scoped_active_levels(1)`.
- A GEMM in the tile loop is single-threaded: each flash loop holds
  `scoped_active_levels(1)`, and the INT8 matmul params set
  `num_threads = 1`.

### Performance and overhead

- The SDPA operator doc's tile-size table matches `varlen_q_split_size`
  and the dense FP and INT8 `q_seq_len` selectors, including the
  sliding-window exemption and the `batch > 4` override, and the doc
  describes the thread-count fit in `balanced_q_split_size` (shrink-only,
  cost `ceil(tasks / threads) * (rows + 12)`).
- A GEMM whose `alpha`, `beta`, or row count misses the custom-kernel
  rules in `docs/runtime_env.md` runs on AOCL-DLP.
- A change to `varlen_q_split_size`, those selectors, or
  `balanced_q_split_size` is benchmarked on both sides of each threshold,
  at `seq_len == 1`, and at 8, 16, and 32 threads on a few-head shape
  (3 heads, as in EmbeddingGemma), for FP and dynamic INT8.
- A tile-size rule for a `schedule(dynamic, 1)` loop is benchmarked on a
  ragged packed batch (for example lengths 1024, 300, 50, 700 at 32
  threads), where dynamic scheduling already absorbs uneven tiles.

### Tests and validation quality

- `SdpaGqaTest.F32_GQA_MASK_MATCHES_EXPANDED_KV` compares GQA with
  expanded K/V.
- `SdpaVarlenTest.BF16_EQUAL_LENGTHS_MATCH_DENSE` and
  `SdpaVarlenTest.BF16_CAUSAL_EQUAL_LENGTHS_MATCH_DENSE` compare packed
  sequences with `q_len == kv_len` to dense output.
- `SdpaVarlenTest.F32_CHUNKED_PREFILL_MATCHES_FULL_PREFILL` compares
  chunked prefill with a full prefill.
- A flash test covers a row masked only in its first KV tile, a row
  masked in every tile, and decode (`seq_len == 1`).
- `SdpaInt8DynamicQuantTest` and `SdpaVarlenTest` cover QK-only,
  PV-only, and both, with a mask, GQA, a window, and packed mode.
- `SdpaCkGemmApi` and `SdpaCkInt8GemmApi` count `ck::test_api` calls for
  a forced custom-kernel hit and for the AOCL-DLP fallback.
- Resized tiles run in `SdpaVarlenTest.BF16_FUSED_QKV_THREAD_COUNTS`
  (packed), `SdpaTileSplitTest.F32_DENSE_THREAD_COUNTS` (dense FP), and
  `SdpaInt8DynamicQuantTest.BF16_DENSE_TILE_SIZE_INDEPENDENT` (dense
  INT8), and `SdpaTileSplitTest.BALANCED_Q_SPLIT_PICKS` checks the sizes
  they rely on. `TestSdpa` shapes (`seq_len` up to 128) never leave the
  preferred size, so a tile-sizing change keeps these tests on sizes the
  rule actually picks.

### Code quality and documentation

- The dense and packed flash loops, FP and INT8, and the reference
  kernel use the same mask, scale, and fully-masked-row rules.

## HSTU attention

### Functional correctness

- In delta mode, query and output row `r` of sequence `b` is row
  `b * delta_size + r` at position `len - delta_size + r`.
- `attn_scale_per_token` with `delta_size > 0` returns
  `status_t::failure`.
- `is_varlen = false` with `delta_size > 0` stays variable-length, and
  `hstu_attention_direct` warns once.
- `delta_size > 0` and `causal = false` uses absolute id distance.
- For a causal row, `seq_mask_t::band` plus the diagonal equals
  `seq_mask_t::valid` (`HstuAttentionMask.CausalBandMatchesPredicate`).
- `seq_mask_t::valid` implements `_get_valid_attn_mask`, including
  targets, the window, `contextual_seq_len`, and
  `min_full_attn_seq_len`.
- `hstu_attention_direct` returns `status_t::failure` for a null tensor
  or `seq_offsets`, `seq_offsets[0] < 0`, a decreasing offset, a length
  outside `[0, max_seq_len]` or below `delta_size`, `num_targets`
  outside `[0, length]`, a non-positive stride, or overlapping output
  rows.
- A new pointer or index field is covered by
  `HstuAttentionApi.RejectsInvalidArguments` or
  `HstuAttentionApi.RejectsInvalidParameters`.
- `hstu_attention_tiled_supported` is false unless the mask is causal,
  Q, K, and V are BF16, and the CPU has AVX512-BF16, AVX512-BW, and
  AVX512-VL.
- `ZENDNNL_HSTU_ATTENTION_KERNEL=tiled` returns
  `status_t::unimplemented` when that support check is false.
- An automatic kernel the tiled path cannot run uses the reference
  kernel, and `hstu_attention_direct` warns once.

### Numerical correctness

- The scale is `attn_scale_per_token` when set, else `attn_scale` when
  `has_attn_scale` is set (0 and negative included), else
  `1 / max_seq_len`.
- A masked key contributes 0. The row is not divided by a sum.
- The scale is multiplied once, after silu.
- `is_varlen = false` and `delta_size == 0` matches variable-length
  output (`HstuAttentionVarlen.PaddedMatchesVarlen`).
- P and the weight of an `extra_diag` row are rounded to BF16. That row
  has its diagonal outside `[lo, hi)`, for example a target row.
- `diag()` computes silu with `std::exp` for both fast-silu settings.
- Unset `ZENDNNL_HSTU_ATTENTION_FAST_SILU` matches a build without it
  (`HstuAttentionFastSilu.DefaultPathUnchanged`).
- The reference kernel ignores `ZENDNNL_HSTU_ATTENTION_FAST_SILU`.
- Output matches across thread counts
  (`HstuAttentionTiled.DeterministicAcrossThreads`).
- A second call that reuses scratch matches the first
  (`HstuAttentionTiled.RepeatedCallsReuseScratch`).

### Memory and lifetime

- `std::bad_alloc` in `hstu_attention_direct` returns
  `status_t::failure`.
- `arena_t::get` leaves the arena empty when `new` throws.
- Tiled scratch starts uninitialized, and a slot is read only after
  that call writes it.
- A tile reads only key blocks in `kv_blocks_t::jobs()`.
- The tiled kernel writes P for every key of the row window before
  reading that P.
- An arena larger than `kRetainBytes` is trimmed when
  `hstu_attention_direct` returns.
- Padded-run buffers (`is_varlen = false`) stay allocated for reuse.

### Concurrency and OpenMP

- `hstu_attention_direct` uses this call's `omp_get_max_threads()` and
  restores the caller
  (`HstuAttentionApi.KeepsTheCallersThreadCount`).
- The tile pool has one slot per thread, indexed by
  `omp_get_thread_num()`.

### Planner and integer arithmetic

- `key_ranges`, `kv_blocks_t::jobs()`, and `block_window` cover the same
  keys (`HstuAttentionPlan.KeyBlocksCoverKeyRanges`,
  `HstuAttentionDriver.BlockWindowsCoverBands`).
- `choose_block_m` and `work_plan_t` split the same query rows
  (`HstuAttentionPlan.WorkItemsInSdpaOrder`).

### Performance and overhead

- The tiled item loop uses `schedule(dynamic, 1)`.
- K^T and V are packed once per call, only for blocks some row reads.
- A change that packs more blocks, or packs inside the tile loop, is
  benchmarked on a delta-mode call and a windowed call.
- A change to the tiled item loop is benchmarked at batch size 1, in
  delta mode, and on a full-attention call.

### Tests and validation quality

- `HstuAttentionKernels.MatchOracle` compares reference, automatic, and
  tiled output with the FP64 transcription for full and delta mode,
  targets, window, contextual prefix, a non-causal mask, f32 and bf16,
  per-token scale, a fused view, and a gap between heads.
- `HstuAttentionKernels.MatchOracle` skips when
  `ZENDNNL_HSTU_ATTENTION_KERNEL` is set.
- `HstuAttentionFastSilu.MatchesOracle` checks fast-silu error against
  the FP64 transcription.

### Code quality and documentation

- Data-type-independent code stays in `hstu_attention/common/`. A data
  type is a compile-time kernel policy in its own file.

## Validation

- gtests: `gtests/test_sdpa.cpp`, `gtests/test_sdpa_ck_gemm.cpp`,
  `gtests/test_sdpa_ck_int8.cpp`, and `gtests/test_hstu_attention.cpp`.
- BenchDNN: `benchdnn/doc/sdpa.md` covers dense FP SDPA only. A GQA,
  sliding-window, dynamic INT8, or packed change brings its own
  benchmark. An HSTU performance change brings its own benchmark.
- Shapes: `benchdnn/input/sdpa/sdpa_models.txt`.
- Baseline: a static schedule of the HSTU tiled item loop measured
  0.73-0.77x `schedule(dynamic, 1)` at model level on DLRM v3.
- Paths: `ZENDNNL_SDPA_BF16_CK`, `ZENDNNL_SDPA_INT8_CK`,
  `ZENDNNL_SDPA_INT8_QK`, `ZENDNNL_SDPA_INT8_PV`,
  `ZENDNNL_HSTU_ATTENTION_KERNEL`, `ZENDNNL_HSTU_ATTENTION_FAST_SILU`,
  `ZENDNNL_HSTU_ATTENTION_VARLEN`.
- ISA: HSTU tiled needs AVX512-BF16, AVX512-BW, and AVX512-VL. SDPA f16
  needs AVX512-FP16. SDPA dynamic INT8 needs AVX512-F, AVX512-BW,
  AVX512-VL, and AVX512-VNNI. A skip without that ISA is not a pass.
