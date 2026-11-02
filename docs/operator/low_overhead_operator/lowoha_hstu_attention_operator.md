
(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA HSTU Attention Operator

## Overview

The **HSTU attention API** (`hstu_attention_direct`) implements the attention of HSTU (Hierarchical Sequential Transduction Units), the sequential encoder of generative recommenders such as DLRM v3. It takes raw pointers and strides over jagged sequences and has no framework dependency.

For sequence *b* of length *n* and head *h*:

$$
O_i = \sum_{j} M_{ij} \cdot \text{silu}\!\left(\alpha \, Q_i \cdot K_j\right) \cdot s_i \cdot V_j
$$

Where:
- *α*: score scale, usually 1/√d<sub>qk</sub>
- *s<sub>i</sub>*: output scale, 1/`max_seq_len` by default, or a scalar or per-token value
- *M*: the HSTU valid-attention mask (0 or 1)

Unlike SDPA there is **no softmax**: masked scores contribute zero and no row normalization takes place. The semantics match `pytorch_hstu_mha` (full mode) and `pytorch_cached_hstu_mha` (delta mode) of [generative-recommenders](https://github.com/meta-recsys/generative-recommenders).

### Mask

Positions get the id `id(x) = min(c > 0 ? max(0, x - c + 1) : x, max_ids)` with `max_ids = n - t - (c > 0 ? c - 1 : 0)`, where *t* = `num_targets[b]` and *c* = `contextual_seq_len`. All targets (candidates) share the id of the first target. With `d = id(i) - id(j)` (`|d|` when not causal), key *j* is valid for query *i* when:

- `i == j`, or `d > 0`;
- and, with a window *w* = `max_attn_len` > 0, `d <= w`, unless `id(i) >= max_ids - min_full_attn_seq_len`;
- or, with contextual tokens, `id(i) == 0` and `id(j) < max_ids`.

In a causal mask the valid keys of a row are one interval plus the row itself. History rows see `[i - w, i]`, and target rows see the whole history (or its last *w* items) and themselves, but not the other targets.

## Layout

Rows are jagged: `seq_offsets[b]` is the first row of sequence *b*, and the last dimension of every tensor is contiguous.

| Tensor | Full mode (`delta_size == 0`) | Delta mode (`delta_size > 0`) |
|--------|-------------------------------|-------------------------------|
| Q | `[L, H, qk_dim]` | `[num_seqs * delta_size, H, qk_dim]`: queries of the last `delta_size` positions of every sequence |
| K | `[L, H, qk_dim]` | `[L, H, qk_dim]` |
| V | `[L, H, v_dim]` | `[L, H, v_dim]` |
| Output | `[L, H, v_dim]` | `[num_seqs * delta_size, H, v_dim]` |

`L = seq_offsets[num_seqs]`. Token and head strides are free, so Q, K and V can be views into one fused projection (for example `u, v, q, k = split(uvqk)` in the HSTU layer) without a copy.

## API

Include `lowoha_operators/hstu_attention/lowoha_hstu_attention.hpp` (namespace `zendnnl::lowoha::hstu_attention`).

```cpp
status_t hstu_attention_direct(const void *query, const void *key,
                               const void *value, void *output,
                               const hstu_attention_params &params);
```

### Parameters (`hstu_attention_params`)

| Field | Meaning |
|-------|---------|
| `num_seqs`, `num_heads`, `qk_dim`, `v_dim` | Shapes |
| `max_seq_len` | Upper bound of the sequence lengths; the default output scale is `1 / max_seq_len` |
| `seq_offsets` | `[num_seqs + 1]` int64 row offsets of K and V (and Q in full mode) |
| `delta_size` | 0 for full mode; > 0 for delta mode |
| `causal` | Causal mask (DLRM v3 always uses `true`) |
| `num_targets` | `[num_seqs]` int64 targets per sequence, or `nullptr` |
| `max_attn_len` | Attention window, 0 for none |
| `contextual_seq_len` | Leading contextual tokens, 0 for none |
| `min_full_attn_seq_len` | Trailing history rows exempt from the window |
| `alpha` | Score scale |
| `has_attn_scale`, `attn_scale` | Scalar output scale, used as given (including 0 or negative values) when `has_attn_scale`; otherwise `1 / max_seq_len` |
| `attn_scale_per_token` | Optional `[L]` f32 per-token output scale (full mode only) |
| `q_stride_t`, `q_stride_h`, ... `o_stride_h` | Token and head strides of Q, K, V and the output, in elements |
| `qkv_dt`, `out_dt` | `f32` or `bf16` |
| `is_varlen` | `true` (default): variable-length attention. `false`: every sequence padded to `max_seq_len`, for comparisons (see [Variable length](#variable-length)); full mode only |
| `is_qk_quant`, `is_pv_quant` | Dynamic INT8 Q·K<sup>T</sup> and P·V in the tiled kernel, independently (see [Dynamic INT8](#dynamic-int8)); BF16 Q/K/V only, otherwise ignored with a warning. `ZENDNNL_HSTU_ATTENTION_INT8_QK` / `ZENDNNL_HSTU_ATTENTION_INT8_PV` override them |
| `kernel` | `none` (automatic), `tiled` or `reference` |
| `num_threads` | 0 uses the current `omp_get_max_threads()`; the count in effect before the call is restored afterwards |

The call returns `status_t::failure` for invalid arguments; these O(num_seqs) checks always run, independent of `ZENDNNL_DIAGNOSTICS_ENABLE`. Offsets must be non-decreasing from `seq_offsets[0] >= 0`, sequences no longer than `max_seq_len` and at least `delta_size` long, every `num_targets[b]` in `[0, n]`, and output rows must not overlap; the output must not alias Q, K or V. Requesting a kernel that does not support the call returns `status_t::unimplemented`, and running out of memory for scratch returns `status_t::failure`. When the automatic choice falls back to the scalar reference kernel, a warning is logged once. The kernels assume the default floating-point rounding mode (round to nearest): a caller's MXCSR mode reaches only the calling thread, not the OpenMP workers, so under another mode the FP32 results depend on which thread computes which tile. The INT8 quantizers round to nearest whatever the mode.

## Kernels

| Kernel | Supports | Design |
|--------|----------|--------|
| `tiled` | Causal mask, BF16 Q/K/V, f32 or BF16 output, AVX512-BF16/BW/VL; dynamic INT8 also needs AVX512-VNNI | See [Tiled kernel](#tiled-kernel) and [Dynamic INT8](#dynamic-int8) |
| `reference` | Every valid call | Scalar, one (sequence, head, row) per OpenMP iteration, FP32 accumulation, mask predicate evaluated per pair |

The automatic choice is `tiled` when it supports the call, `reference` otherwise. `ZENDNNL_HSTU_ATTENTION_KERNEL` overrides both (see [runtime_env.md](../../runtime_env.md)).

### Tiled kernel

1. **Pack once.** The 256-key blocks of K<sup>T</sup> and V that some query of the call reads (the union of each sequence's query bands, so delta mode and windows pack only what they use) are packed once per call, in a static parallel loop, as 32-column blocks of [ceil(K/2)][32][2] BF16, the pair layout `vdpbf16ps` reads (`ck::bf16::pack_b`). Each block takes its exact packed size; the buffer is not zero-filled, is reused by the calling thread, and is released after a call that needed more than 256 MiB.
2. **Work items.** A work item is one query tile of one (sequence, head). The tile height is the largest power of two in [32, 128] that still gives every thread four items. Calls with fewer query tiles than threads (for example delta mode with a small `delta_size` at batch size 1) leave threads idle, since keys are not split across threads. As in SDPA, the items are a flat index in (sequence, head, query tile) order in one `parallel for`, but with `schedule(dynamic, 1)`: under a causal mask the last tiles of a sequence read the most keys, and SDPA's `schedule(static)` left cores idle (0.73–0.77× at model level on DLRM v3; sorting the tiles by cost first gains at most 4.5% at batch size 1 and nothing at batch size 4). Each tile copies its Q rows to a contiguous buffer, because rows of a strided view such as the fused `uvqk` projection alias in L1.
3. **Q·K<sup>T</sup> with a fused epilogue.** An AVX512-BF16 tile kernel (6 rows × 64 keys, or 8 × 32 at a block edge) accumulates S in registers. It then applies silu (`group_matmul_act_avx512::silu_avx512`, shared with the grouped MatMul and routed MoE epilogues), the row scale and the row's key interval, and stores P in BF16 (as the PyTorch reference rounds P). S is never written to memory. A tile whose rows keep no key skips the GEMM. See [Fast silu](#fast-silu) for an opt-in cheaper epilogue.
4. **P·V.** O += P·V runs on a `vdpbf16ps` tile kernel (8 rows × 32 columns, `ck::bf16::gemm_block`), one packed 32-column V block at a time over only the keys of the block that the tile's rows read.
5. **Diagonal and output.** A history row's diagonal is part of its band. The diagonal of a target row (outside its band) is added the same way by every policy (`diag_bf16`): an AVX512-BF16 dot product, the exact silu and a weight rounded to BF16 like P. The FP32 O tile is stored in the output dtype.

Only the key blocks of each tile's band are visited. History rows see `[i - w, i]` and target rows see the history and themselves, so a causal call does about half the work of a dense one, and a windowed call proportionally less.

**Code layout.** Files named `lowoha_hstu_attention*` are the public API; every internal file is `lowoha_hstu_<component>`. The planning, the scratch, the two parallel loops of steps 1 and 2, the per-tile windows and the store do not depend on the data type and live in `hstu_attention/tiled/`. What does (the pack format, the Q tile, the Q·K<sup>T</sup> and P·V tiles and the diagonal) is a compile-time policy of `run_tiled` with a folder of its own (`tiled/bf16/`, `tiled/int8/`, as in `hstu_gemm_ck/`); the policy folders do not include each other. Another data type is one more policy folder and one more entry point in `tiled/lowoha_hstu_tiled.hpp`:
- `common/`: what the dispatcher and every kernel share. `lowoha_hstu_mask.hpp` holds the mask predicate, the row bands and the rows and scales of a sequence; `lowoha_hstu_select.hpp` the kernel and INT8-flag selection and the parsing of the environment variables (pure functions); `lowoha_hstu_pad` the padded (`is_varlen = false`) run of either kernel; `lowoha_hstu_utils.hpp` small helpers (lane masks) of the kernels, GEMMs and quantizers.
- `reference/lowoha_hstu_ref_kernel`: the scalar kernel.
- `tiled/lowoha_hstu_tiled.hpp`: the entry points of the tiled kernel, one per policy (`hstu_attention_tiled_bf16`, `hstu_attention_tiled_int8`), and its support checks.
- `tiled/lowoha_hstu_tiled_driver.hpp`: `run_tiled<policy_t, out_t>`, which owns the two parallel loops, the per-tile row and key windows, the diagonal loop and the store.
- `tiled/lowoha_hstu_tiled_plan`: key ranges, the 256-key K/V blocks and pack jobs, the query tile height, and the work items in SDPA order.
- `tiled/lowoha_hstu_tiled_scratch`: the thread-owned arenas, the packed K/V storage (one slot per block and head, laid out by the policy), and the per-thread tile scratch.
- `tiled/lowoha_hstu_tiled_store`: the FP32 O row to the output dtype.
- `tiled/lowoha_hstu_tiled_common.hpp`: what the policies share: the Q·K<sup>T</sup> tile heights, the opt-in fast silu of their epilogues, the diagonal term every policy adds the same way (`diag_bf16`), and the BF16 P·V of every policy with a BF16 half.
- `tiled/bf16/`: the BF16 policy `bf16_policy_t` and its micro-kernels (`lowoha_hstu_tiled_bf16.cpp`). It carries its layout (key pairs, BF16 P, packed K<sup>T</sup> then V per slot) and five members: `pack`, `load_q`, `qk`, `pv` and `diag`. They are inlined into the driver, so the hot loop has no indirection.
- `tiled/int8/`: the dynamic-INT8 policy `int8_policy_t<QkInt8, PvInt8, FastSilu>` and its Q·K<sup>T</sup> tile kernels (`lowoha_hstu_tiled_int8.cpp`); it decides where the `hstu_quant/` quantizers run (Q in `load_q`, K and V in `pack`, P after each block's Q·K<sup>T</sup>).
- `hstu_gemm_ck/`: HSTU attention's own custom-kernel GEMMs, laid out like `sdpa/sdpa_gemm_ck/` but separate from it (no kernel code is shared with SDPA): `lowoha_hstu_ck.hpp` (the API, namespace `ck::bf16` and `ck::int8`), `lowoha_hstu_ck_common.hpp` (the block width, ISA targets and the pair step and quad step, which the fused Q·K<sup>T</sup> tile kernels also use), `bf16/` (the BF16 pack of K<sup>T</sup> and V and the P·V tile kernel) and `int8/` (the INT8 pack of K<sup>T</sup> and V with their column sums, and the P·V tile kernel).
- `hstu_quant/`: HSTU attention's dynamic quantizers (`quant::quantize_row` for Q and K per token, `quantize_cols` for V per channel of a block, `quantize_rows_amax` for P per row with the max the Q·K<sup>T</sup> epilogue tracked), separate from the kernels that call them and from the GEMMs. The INT8 policy decides where each runs: in the Q tile copy, the pack loop and after each block's Q·K<sup>T</sup>.

### Dynamic INT8

With `is_qk_quant` and / or `is_pv_quant` the tiled kernel runs Q·K<sup>T</sup> and / or P·V on INT8, like the dynamic-INT8 flash SDPA: Q/K/V stay BF16 in the API, and the kernel quantizes them itself. Every scale is local to what one query tile reads (one token, or one 256-key block, which is also the unit of P·V accumulation in FP32), so quantization needs no extra pass over the tensors. Every group is symmetric S8 with scale `max|x| / 127` and values `round_to_nearest_even(x · 127 / max|x|)` (`max|x|` at least 1e-10, so a zero group quantizes to zeros). The quantizers round the exact product `x · (127 / max|x|)` once, with one FMA against the constant 1.5 · 2<sup>23</sup> + offset whose low byte is the code; the FMA rounds to nearest even whatever the caller's MXCSR rounding mode. A group holding a NaN gets a NaN scale and one holding an infinity an infinite scale, so non-finite inputs reach the output in every INT8 mode, as on the BF16 path (they may spread to more outputs of the same group).

| Tensor | Group | Where |
|--------|-------|-------|
| Q | one token of one head; α is folded into the scale | the Q tile copy, as U8 = S8 + 128 (what `vpdpbusd` reads) |
| K | one token of one head | the pack loop, with the column sums of the packed S8 K<sup>T</sup> |
| V | one channel of one 256-key block of one head | the pack loop, with the column sums of the packed S8 V |
| P = mask · silu(α q·k) | one row of one 256-key block | after the tile's Q·K<sup>T</sup> of the block, as U8 = S8 + 128 |

- **Q·K<sup>T</sup>.** The INT8 tile kernel (6 rows × 64 keys, or 8 × 32) accumulates `Σ (q + 128) · k` in int32 with `vpdpbusd`. Its epilogue removes the offset with the K column sums, dequantizes by the Q and K scales, and applies silu and the band mask as the BF16 kernel does. With BF16 P·V it also applies the row scale and stores BF16 P.
- **P·V.** Unlike SDPA's softmax P in [0, 1], HSTU's P is signed (silu ≥ −0.28) and unbounded, so a fixed U8 scale does not fit it: the epilogue stores FP32 P and keeps each row's max |P|, and a short pass then quantizes the block's P per row. P·V runs on a `vpdpbusd` tile kernel (8 rows × 32 columns, `ck::int8::gemm_block`), one packed 32-column V block at a time over the keys the tile reads. It removes the +128 offset of P with the V column sums over those keys, dequantizes by the P scale times the row scale *s<sub>i</sub>* and by the V scales, and adds to the FP32 O. Keeping *s<sub>i</sub>* out of P costs nothing, since P·V has a row scale anyway, and a zero or negative *s<sub>i</sub>* never reaches the quantizer.
- **Diagonal.** The extra diagonal term of a target row comes from the BF16 rows as in the BF16 kernel (`diag_bf16`), so with no history (every row a target) the INT8 output is the BF16 one bit for bit.
- **Selection.** The flags (or `ZENDNNL_HSTU_ATTENTION_INT8_QK` / `_PV`) need BF16 Q/K/V: with f32 inputs they are dropped with a warning. INT8 Q·K<sup>T</sup> needs `qk_dim` ≤ 133,144 (`kInt8MaxQkDim`: its int32 dot products reach 127² · `qk_dim`); past it `is_qk_quant` is dropped with a warning and INT8 P·V still applies. The INT8 kernel needs AVX512-VNNI on top of the tiled kernel's requirements; without it the tiled kernel runs on BF16 and a warning is logged once. The reference kernel ignores the flags (logged once at info level). The API log shows the flags in effect.
- **Accuracy.** On DLRM v3 shapes with N(0, 1) inputs the relative L2 error against an FP64 reference is about 1.0e-2 with INT8 Q·K<sup>T</sup>, 1.2e-2 with INT8 P·V and 1.6e-2 with both (BF16: 2.3e-3). The output is bitwise deterministic across thread counts, as in BF16.

### Fast silu

`ZENDNNL_HSTU_ATTENTION_FAST_SILU=1` (off by default; see [runtime_env.md](../../runtime_env.md)) swaps the silu of the Q·K<sup>T</sup> epilogue for a cheaper one, 11 instead of 21 vector operations per 16 scores:
- exp(−x) = 2<sup>t</sup>, t = −x·log<sub>2</sub>e, from a degree-3 minimax polynomial of 2<sup>frac(t)</sup> and `vscalefps` for 2<sup>floor(t)</sup> (no clamps: it saturates to 0 or ∞, so the sigmoid saturates to 1 or 0);
- the sigmoid's reciprocal is `rcp14` without a Newton step;
- α is folded into the exponent and into the row scale.

The relative error of silu is at most 1.2e-4, against 5e-5 for `group_matmul_act_avx512::silu_avx512`. Both are far below the 2<sup>−9</sup> rounding of BF16 P, so the output error against FP64 does not change. Silu is about 11% of the BF16 kernel's time on DLRM v3 full-attention calls; the fast silu saves 6% there and 3–5% on delta-mode and short-sequence calls (Turin, 32 cores). When most scores are below −88, as in the recorded DLRM v3 benchmark calls, the default silu computes on denormals (its exponent clamp leaves a denormal reciprocal for the Newton step), and the fast silu, which saturates to exact zeros, saves 15–23%. The setting is a compile-time parameter of the BF16 policy (`bf16_policy_t<FastSilu>`): the default path has no runtime branch and is bit-identical to running without the variable. The target-row diagonal (step 5) keeps the exact silu, and the reference kernel ignores the setting.

The [dynamic INT8](#dynamic-int8) kernel reads the same setting (`int8_policy_t<QkInt8, PvInt8, FastSilu>`), with the same function on its dequantized scores, so its default path is unchanged too. The silu error is far below the INT8 rounding of P (at most half a step, 1/254 of a row's max per block), and the output error against FP64 does not change. With INT8 P·V, a P within the silu error of a rounding boundary can round to the neighbouring code, so the outputs move by up to about 2e-3 against the oracle with the kernel's quantization.

### Variable length

Both kernels are variable-length: every sequence is computed over its own length from `seq_offsets`, with no padding to the longest one. To measure what that saves, `is_varlen = false` (or `ZENDNNL_HSTU_ATTENTION_VARLEN=0`, which overrides the field; see [runtime_env.md](../../runtime_env.md)) runs the call as a framework without variable-length support would (`common/lowoha_hstu_pad`):
1. Q, K, V (and the per-token scale) are copied in one parallel pass to contiguous buffers of `num_seqs * max_seq_len` rows, each sequence's rows first and zero rows after them (one copy per token when its heads are adjacent, as in the fused `uvqk` projection);
2. the kernel runs on the uniform layout, with the zero rows counted as extra targets;
3. the real rows are copied to the output.

The output is the variable-length one bit for bit: the mask depends on the position ids, which padding does not change for real rows, and a zero key adds silu(0)·v = 0. The cost is the attention of the padded rows and the copies. The buffers are kept per thread for reuse. Delta mode, whose queries are the last rows of every sequence, stays variable-length (with a warning, once).

## Example

```cpp
using namespace zendnnl::lowoha::hstu_attention;

hstu_attention_params p;
p.num_seqs = B;
p.num_heads = 4;
p.qk_dim = 128;
p.v_dim = 128;
p.max_seq_len = max_len;
p.seq_offsets = offsets;          // int64 [B + 1]
p.num_targets = num_targets;      // int64 [B]
p.alpha = 1.f / std::sqrt(128.f);
// q, k, v: views into uvqk [L, W] (W = row width of the fused projection)
p.q_stride_t = p.k_stride_t = p.v_stride_t = W;
p.q_stride_h = p.k_stride_h = 128;
p.v_stride_h = 128;
p.o_stride_t = 4 * 128;
p.o_stride_h = 128;
p.qkv_dt = p.out_dt = data_type_t::bf16;

status_t st = hstu_attention_direct(q, k, v, out, p);
```

## Tests

`gtests/test_hstu_attention.cpp` (BF16 and the reference kernel, and what every kernel shares) and `gtests/test_hstu_attention_int8.cpp` (dynamic INT8 and the quantizers) share their cases and FP64 oracles through `gtests/hstu_attention_test_utils.hpp`, so the tests of another data type are one more file. They contain these groups of tests:
- the causal row band against the mask predicate, exhaustively over lengths, targets, windows, contextual prefixes and `min_full_attn_seq_len`;
- both kernels against an FP64 transcription of the PyTorch reference: full and delta modes, targets, window, contextual prefix, non-causal masks, f32/BF16, per-token scale, DLRM-shaped fused-projection views, and heads with gaps between them in Q, K, V and the output;
- bitwise determinism of the tiled kernel across thread counts and repeated calls;
- the fast silu: the FP64 oracle at the default path's error (including scores large enough to saturate exp), the default path bit-identical to the explicit default, and the parsing of `ZENDNNL_HSTU_ATTENTION_FAST_SILU`;
- padding (`is_varlen = false`): bit-identical to the variable-length output on every case, both kernels and several thread counts, the canaries untouched, and the parsing of `ZENDNNL_HSTU_ATTENTION_VARLEN`;
- the tiled planning and driver helpers: work-item order and coverage, key ranges and pack jobs, the tile height, and the per-tile block windows;
- the dynamic quantizers code for code against the scalar definition (ties under every rounding mode, zero groups, outliers including in the tail chunk, tails and strides);
- dynamic INT8 on every tiled case, for INT8 Q·K<sup>T</sup>, P·V and both: against an FP64 oracle that applies the same quantization (tight bound) and against the exact oracle (loose bound), zero and outlier rows, NaN and infinity in Q, K and V, the `qk_dim` limit, padding, bitwise determinism, the diagonal of target rows bit-identical to the BF16 kernel, scratch reuse with BF16 calls (BF16 output checked), the flags where INT8 does not run, kernel selection (including a CPU without VNNI), the flag resolution, and the environment overrides in fresh processes;
- argument validation.

`--gtest_filter='*HstuAttentionInt8*'` runs every dynamic INT8 test (the cases run as `Cases/HstuAttentionInt8`, the others as `HstuAttentionInt8Api`, `HstuAttentionInt8Quant` and `HstuAttentionInt8DeathTest`).
