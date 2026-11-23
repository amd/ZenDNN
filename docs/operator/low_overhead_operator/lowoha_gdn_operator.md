
(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA Gated DeltaNet (GDN) Operator

## Overview

The **GDN API** (`gdn_direct`) implements the linear-attention core of Gated DeltaNet layers, as used by Qwen3-Next and Qwen3.5 / Qwen3.6 / Qwen3.8: the causal depthwise conv1d, the gating and the delta-rule recurrence of one batch, in a single call. One call takes a whole non-speculative vLLM batch, decode sequences first and prefill sequences after them, with the packed variable-length offsets of SDPA's `cu_seqlens` (vLLM's `query_start_loc`); a speculative-verify call takes the draft tokens of vLLM's speculative decodes. It takes raw pointers and strides over vLLM-style state pools and has no framework dependency.

For value head *hv* (key head *h* = *hv* / (`num_v_heads` / `num_k_heads`)) and token *t*:

$$
\begin{aligned}
x_t &= \text{silu}\left(\text{conv1d}(\text{mixed\_qkv})_t\right), \quad [q_t, k_t, v_t] = x_t \\
g_t &= -e^{A_{log}} \cdot \text{softplus}(a_t + dt_{bias}), \qquad \beta_t = \sigma(b_t) \\
S &\leftarrow e^{g_t} S, \qquad S \leftarrow S + k_t \left(\beta_t (v_t - S^T k_t)\right)^T \\
o_t &= \text{scale} \cdot S^T q_t
\end{aligned}
$$

Where:
- *q*, *k* are L2-normalised when `use_qk_l2norm` is set, and `scale` defaults to 1/√`head_k_dim`;
- β<sub>t</sub> = 2σ(b<sub>t</sub>) with `allow_neg_eigval` (FLA's option, as OLMo's hybrid layers set it), so the transition *I* − β*kk*<sup>T</sup> has eigenvalues down to −1 (with L2-normalised *k*);
- the conv output is rounded to the source type before the recurrence (`round_conv_out`), as an unfused conv → recurrence pipeline does;
- *S* is the `head_k_dim` × `head_v_dim` recurrent state of one (pool slot, value head), accumulated in FP32.

An optional gated RMSNorm, `out = rmsnorm(o) * norm_weight * act(z)` with `act` = silu or sigmoid, can be fused on the output. It matches vLLM's `RMSNormGated` (norm before gate, one group): by default the core output is rounded to the source type before the norm (`round_core_out`), as vLLM's unfused pipeline reads it.

## Batches

A call runs `batch` sequences. Sequence *i* owns token rows `[cu_seqlens[i], cu_seqlens[i + 1])` of every per-token tensor (`mixed_qkv`, `a`, `b`, `z`, `output`) and pool slot `state_indices[i]`.

- **Decodes first.** The first `num_decodes` sequences are single-token continuations: each advances its slot by one token on the fused decode kernel. `cu_seqlens[i] == i` for `i <= num_decodes`, which is how vLLM orders the decode tokens.
- **Then prefills.** Sequences `num_decodes .. batch - 1` are prefills of any length (0 included) on the prefill kernel: token-recurrent, or the chunked delta rule once the prefills of the call average 16 tokens (bf16 sources), 32 (f16) or 24 (f32).
- **`cu_seqlens`** holds `batch + 1` non-decreasing offsets from 0 and `num_tokens == cu_seqlens[batch]`. A null `cu_seqlens` means every sequence has one token (sequence *i* owns row *i*) and `num_tokens == batch`; the sequences after the decodes are then single-token prefills.
- **`has_initial_state`** has one flag per sequence, indexed like `state_indices`, exactly as vLLM's `GDNAttentionMetadata::has_initial_state`, so a caller passes vLLM's tensors without slicing or copying. For a prefill, `true` continues from the conv and SSM state in its slot and `false` starts from zero states. The flags of the decodes are not read: a decode always continues from its slot. A null `has_initial_state` starts every prefill from zero states.
- **Entries past the batch** of `cu_seqlens` and `has_initial_state` are not read, so vLLM's padded metadata (trailing empty requests) can be passed as is.
- **Pad slots.** A sequence whose slot equals `pad_slot_id` touches neither pool: a padded decode writes a zero output row, a padded prefill leaves its output rows untouched, as does a prefill with no tokens. Two sequences of one call must not share a slot, unless `update_state` is false.
- Every sequence that runs leaves its final conv state (its last `conv_width - 1` raw inputs) and SSM state in its slot, so decode can continue from a prefill and vice versa, and from states written by vLLM's own CPU kernels.
- **The inputs of `chunk_gated_delta_rule`.** A prefill-only call may also take what FLA's and transformers' `chunk_gated_delta_rule` / `recurrent_gated_delta_rule` take: q, k and v as separate tensors (`k` and `v` set, `mixed_qkv` pointing at q; each `[token][head][dim]` with its own token and head strides; `conv_width = 0`, as these are already convolved) and the gates as given (`gate_input = precomputed`: `a` holds the log decay g, of either sign, and `b` holds beta, of types `a_dt` and `b_dt`; `A_log` and `dt_bias` are not read). The kernels are the same; only the reads of q / k / v and of the gates change.
- **`update_state = false`** (default `true`) keeps both pools as they are, for prefill-only calls: the prefills read their initial states as `has_initial_state` says and write nothing back. Several sequences may then continue from one slot (prefixes shared read-only, such as questions continuing from one encoded context), and when no prefill reads a slot the pools are not accessed at all and may be null (a stateless prefill, such as an encoder pass).
- **Split passes.** A prefill continued in a later call from the state an earlier call left in its slot (`has_initial_state`), as prefix caching and prompts answered from a shared context run it, gives the outputs of one call over all its tokens, bit for bit, when the SSM state is FP32 (`state_dt = f32`) and both calls run the same kernel, the same `prefill_algo` and the same rounding options; when both calls write their states (`update_state`), the final states match too. The token-recurrent form matches at any cut; the chunked form at a cut on one of the sequence's 32-token chunk boundaries, since its chunks start at each call's first token. Off a chunk boundary the chunked form matches to the precision of its GEMM operands. How the calls group sequences, `update_state` (which only decides whether a call writes its final state) and the thread count (value blocks included) change no bit. `automatic` picks the algorithm per call from the prefills' average length, so a short continuation can run the recurrent form where one call ran the chunked one: pin `prefill_algo` (or `ZENDNNL_GDN_PREFILL`) so that both calls run the same form, and with `chunked`, cut on one of the sequence's 32-token chunk boundaries. A decode continuing a prefill matches it to FP32 rounding, not bit for bit: the native decode kernel sums in another order than the recurrent prefill.

- **Speculative verify** (`spec_state_indices` set), for vLLM's speculative decoding (MTP or DFlash drafts): every sequence of the call checks its draft tokens, up to `spec_state_stride` of them, from the state its previous step accepted, with vLLM's metadata as is. Sequence *i* has `spec_state_stride` SSM slots `spec_state_indices[i * spec_state_stride + t]`: it continues from slot `num_accepted_tokens[i] - 1` and writes the state after its token *t* to slot *t*, so the next step can continue from whichever token it accepts. Its conv state is slot `spec_state_indices[i * spec_state_stride]`, of `conv_width - 1 + spec_state_stride - 1` taps (vLLM's wide conv buffer): the conv reads the `conv_width - 1` taps from tap `num_accepted_tokens[i] - 1` on, then the `conv_width - 2` taps from `num_accepted_tokens[i]` on move to the front and the sequence's *q* tokens follow, the taps after them keeping their values. That is how vLLM's speculative conv update keeps the buffer, so the next step finds its history at tap `num_accepted_tokens - 1` whatever *q* was, including a row that drafted nothing. `state_indices` and `has_initial_state` are not read; a sequence whose first slot is the pad slot writes zero output rows and touches neither pool. A verify call holds no decodes or prefills, needs `update_state`, packed rows, raw gates and `conv_width` 0 or at least 2, and `1 <= num_accepted_tokens[i] <= spec_state_stride`, which may exceed the sequence's tokens. At the same `num_threads`, it gives, bit for bit, what as many decode calls over the same sequences give, one token at a time, on a conv state of the same strides.

## Layout

All strides are in elements.

| Tensor | Shape | Notes |
|--------|-------|-------|
| `mixed_qkv` | `[num_tokens][qkv_stride]` | q (`num_k_heads * head_k_dim`), k (same), v (`num_v_heads * head_v_dim`); pre-conv when `conv_width > 0` |
| `a`, `b` | `[num_tokens][a_stride]`, `[num_tokens][b_stride]` | `num_v_heads` values per token |
| `A_log`, `dt_bias` | `[num_v_heads]` | f32 or the source type |
| `conv_weight`, `conv_bias` | `[conv_dim][conv_width]`, `[conv_dim]` | depthwise, unpacked; the bias may be null |
| `conv_state` | pool `[slot][tap][channel]` | `conv_width - 1` taps (plus `spec_state_stride - 1` in speculative verify); tap and channel strides cover vLLM's SD (channel-contiguous) and DS (tap-contiguous) layouts |
| `ssm_state` | pool `[slot][v_head][...]` | inner matrix contiguous: `kv` = `[head_k_dim][head_v_dim]` (vLLM CPU), `vk` = `[head_v_dim][head_k_dim]` (FLA) |
| `state_indices` | `[batch]` int32 | pool slot per sequence |
| `output`, `z` | `[num_tokens][num_v_heads][head_v_dim]` | token and head strides |

`conv_dim` = 2 · `num_k_heads` · `head_k_dim` + `num_v_heads` · `head_v_dim`.

## API

```cpp
status_t gdn_direct(const void *mixed_qkv, const void *a, const void *b,
        const void *A_log, const void *dt_bias, const void *conv_weight,
        const void *conv_bias, void *conv_state, void *ssm_state,
        const int32_t *state_indices, void *output, gdn_params &params);

bool gdn_native_supported(const gdn_params &params);
```

`gdn_native_supported` reports whether `gdn_direct` runs the native kernels for a shape on this CPU (see [Kernels](#kernels)). It reads only `head_k_dim`, `head_v_dim` and `conv_width`, and a field left at 0 does not restrict the answer, so `gdn_native_supported(gdn_params())` asks about the CPU alone. Frameworks use it to send only natively supported layers to `gdn_direct`.

### Parameters (`gdn_params`)

| Field | Meaning |
|-------|---------|
| `batch`, `num_decodes`, `num_tokens` | sequences (decodes first), leading decode sequences, rows (`cu_seqlens[batch]`, or `batch` without `cu_seqlens`) |
| `cu_seqlens` | `batch + 1` offsets from 0 (vLLM's `query_start_loc`), or null for one token per sequence |
| `has_initial_state` | `batch` flags indexed like `state_indices`; read for prefills only; null: prefills start from zero |
| `num_k_heads`, `num_v_heads`, `head_k_dim`, `head_v_dim` | head layout; `num_v_heads` is a multiple of `num_k_heads` |
| `conv_width`, `silu`, `round_conv_out` | conv1d width (0: `mixed_qkv` is already convolved), SiLU, rounding of the conv output |
| `qkv_stride`, `a_stride`, `b_stride`, `out_stride_t`, `out_stride_h` | row strides |
| `conv_state_{slot,tap,dim}_stride`, `ssm_state_{slot,head}_stride`, `state_layout` | pool layout |
| `num_state_slots` | pool size; when > 0 every non-pad slot is checked against it |
| `update_state` | `false`: prefill-only call that leaves both pools unchanged (slots may be shared; pools nothing reads may be null); default `true` |
| `spec_state_indices`, `spec_state_stride`, `num_accepted_tokens` | speculative verify: `[batch][spec_state_stride]` slots and `[batch]` accepted counts (vLLM's `spec_state_indices_tensor` and `num_accepted_tokens`); null (default): no verify |
| `k`, `v`, `q_stride_h`, `k_stride_{t,h}`, `v_stride_{t,h}` | separate q (at `mixed_qkv`, token stride `qkv_stride`), k and v with their strides; prefill-only, `conv_width = 0`; null (default): packed rows |
| `gate_input`, `a_dt`, `b_dt` | `raw` (default): `a` / `b` are the gate projections in `src_dt`; `precomputed`: `a` = g, `b` = beta, of types `a_dt` / `b_dt` (`none`: `src_dt`); prefill-only |
| `allow_neg_eigval` | raw gates: beta = 2 · sigmoid(b) (FLA's `allow_neg_eigval`, OLMo's hybrid layers); default `false` |
| `src_dt`, `state_dt`, `a_log_dt`, `dt_bias_dt` | f32, bf16 or f16; `src_dt` covers `mixed_qkv`, `a`, `b`, the conv weight / bias / state, `z` and the output |
| `scale`, `use_qk_l2norm`, `l2norm_eps`, `softplus_threshold` | recurrence options |
| `norm_weight`, `norm_weight_dt`, `z`, `z_stride_{t,h}`, `norm_eps`, `norm_gate`, `round_core_out` | optional fused gated RMSNorm: gate `silu` or `sigmoid`; round the core output to `src_dt` first (default on) |
| `prefill_algo` | prefill algorithm of the native kernel: `automatic`, `recurrent` or `chunked` |
| `pad_slot_id`, `num_threads`, `kernel` | padding slot, OpenMP threads (0: all), kernel request |

### Validation

With diagnostics on (the default; `ZENDNNL_DIAGNOSTICS_ENABLE=0` skips it), `gdn_direct` returns `status_t::failure` without touching any buffer when: a head count or head dim is not positive or `num_v_heads` is not a multiple of `num_k_heads`; `num_decodes` is outside `[0, batch]`; `cu_seqlens[0] != 0`, the offsets decrease, a decode does not have exactly one token (so `cu_seqlens[num_decodes] != num_decodes`) or `num_tokens != cu_seqlens[batch]`; `num_tokens != batch` without `cu_seqlens`; a data type is not f32 / bf16 / f16; `update_state` is false, q / k / v are separate or the gates precomputed in a call with decodes; separate q / k / v lack `k` or `v` or come with `conv_width > 0`; a tensor pointer is null (the pools may be null when `update_state` is false and no prefill reads its slot, `A_log` and `dt_bias` with precomputed gates, `state_indices` in a speculative-verify call); `gate_input` is invalid; `allow_neg_eigval` comes with precomputed gates; `num_threads` is negative; a row stride is narrower than its packed width; the conv or SSM state strides overlap taps, channels, heads or slots; the fused norm lacks `z`, valid `z` strides, a float `norm_weight_dt` or a silu / sigmoid gate; or a non-pad slot is outside `[0, num_state_slots)`. A speculative-verify call is also refused without `cu_seqlens`, `num_accepted_tokens` or a positive `spec_state_stride`; with decodes, `update_state` false, separate q / k / v, precomputed gates or `conv_width` 1; when a sequence has more than `spec_state_stride` tokens or an accepted count outside `[1, spec_state_stride]`; or when a slot it reads or writes is negative, the pad slot or outside `[0, num_state_slots)`. A batch of 0 sequences is a no-op.

## Kernels

| Kernel | Supports | Design |
|--------|----------|--------|
| `native` | AVX512-F/BW/VL/BF16; `head_k_dim`, `head_v_dim` multiples of 16 and at most 256; `conv_width` at most 8 | See below |
| `reference` | Every valid call | Scalar FP32, one sequence after another |

One kernel runs both parts of a call. The automatic choice is `native` when it supports the call, `reference` otherwise; a forced `native` request on an unsupported call falls back to `reference` with a warning. `ZENDNNL_GDN_KERNEL` overrides both, and `ZENDNNL_GDN_PREFILL` overrides `prefill_algo` (see [runtime_env.md](../../runtime_env.md)); both are read once. Every call logs its batch, shape, kernel and prefill algorithm at `ZENDNNL_API_LOG_LEVEL=4` and its time with profiling on.

### Data types

The native kernels take f32, bf16 and f16 activations and SSM states in any combination and compute in FP32. Every load and store converts 16 values at a time: f16 with `vcvtph2ps` / `vcvtps2ph` (AVX512-F, round to nearest even; AVX512-FP16 is not needed), bf16 with a shift / `vcvtneps2bf16`. This covers the activations, the conv weights, bias and history, the gates, the state tiles (both layouts), the outputs and the fused norm, on the decode, recurrent and chunked paths. Only the chunked prefill multiplies in a lower precision, at that of the source type (see below).

### Native decode

1. **Conv.** One parallel loop over (64-channel block, sequence) applies the conv1d (+SiLU) to the new token, advances the conv state and writes the FP32 conv output to a per-call scratch buffer. Both vLLM conv-state layouts run vectorised: in the DS layout the taps of a block are contiguous, so they are converted as one block and shifted in place.
2. **Recurrence.** One parallel loop over (sequence, value head, value tile) L2-normalises q and k, computes the gates and updates the FP32 state tile in registers: decay, S<sup>T</sup>k, the rank-1 update and S<sup>T</sup>q in one pass over the tile. Value heads are split into tiles only when there are fewer (sequence, head) pairs than threads, so each thread streams one contiguous state block.
3. **Norm.** With a fused norm, a third loop applies the gated RMSNorm per (sequence, head).

Decode reads and writes every state once per token, so its cost is the state traffic: at 8 sequences with Qwen3.6 shapes (32 value heads, 128 x 128, FP32 state), 34 MB per layer.

### Native recurrent prefill

Work items are (sequence, value head, 32-column value tile) in one `schedule(dynamic, 1)` loop. Each item loads its FP32 state tile once (16 KB at `head_k_dim` = 128, kept in L1), then for every token of the sequence computes the conv for its channels from the raw input and the conv history, normalises q and k, and runs the recurrence on the tile; the tile is stored once at the end. The final conv state, the last `conv_width - 1` raw inputs, is written after a barrier, since every item reads the initial one. With a fused norm on the rounded core output, the items store the rounded rows as usual and a second parallel pass normalises each (token, head) row in place, so value tiles stay 32 wide; on the FP32 core output (`round_core_out` off) each item takes the whole head row and normalises it before storing. Per-thread scratch is allocated before the parallel region.

### Native chunked prefill

Once the prefills of a call average 16 tokens with bf16 sources, 32 with f16 or 24 with f32 (or with `prefill_algo` / `ZENDNNL_GDN_PREFILL` set to `chunked`), the prefill runs the chunked delta rule (the WY form of FLA's `chunk_gated_delta_rule`). It needs head dims that are multiples of 32 and, with a fused norm, `round_core_out`; other calls run the recurrent form, with a warning if `chunked` was forced. A work item is one (sequence, value head), or a block of its value columns (see below); it keeps its FP32 state `S` and, per chunk of 32 tokens with γ the cumulative log decay:

1. computes the conv, normalised q and k, and the gates of the chunk's tokens (prefetching the next rows of `mixed_qkv`);
2. forms `A_ij = β_i e^(γ_i − γ_j) k_i·k_j` (j < i) and the intra-chunk attention `tril(Q Kᵀ) · e^(γ_i − γ_j)`;
3. inverts `I + A` on 32-wide rows and forms `U = (I + A)⁻¹ diag(β) V` and `W = (I + A)⁻¹ diag(β e^γ) K`;
4. computes `Δ = U − W S` and the output `diag(e^γ) Q S + (attention) Δ`;
5. advances `S ← e^(γ_last) S + Kᵀ diag(e^(γ_last − γ)) Δ`.

Every product runs on GDN's own custom-kernel GEMMs (`gdn_gemm_ck/`) with FP32 accumulation, on bf16 operands for bf16 sources and FP32 operands otherwise:

| Source | GEMM operands | Micro-kernel |
|--------|---------------|--------------|
| bf16 | bf16 (round to nearest even) | `vdpbf16ps`; B in the pair layout the instruction reads |
| f16, f32 | the FP32 rows themselves (no copy) | FP32 FMA |

One tiling serves both types (`lowoha_gdn_ck_kernel.hpp`: tiles of up to 4 x 64 FP32 accumulators, a last block of 32 columns when N is an odd multiple of 32, row tails of 2 and 1, packing loops), and each type's translation unit (`bf16/`, `f32/`) supplies its micro-kernel and conversions. A is read through row and step strides, so a packed matrix also serves as its own transpose. f16 sources take FP32 operands: without AVX512-FP16 their products would be FP32 FMA anyway, and f16 operands would overflow on values the FP32 recurrence keeps, such as the decays `e^γ` of growing precomputed gates or an FP32 state past 65504. The products skip their zero triangles: the inverse `(I + A)⁻¹` and the intra-chunk attention are lower triangular, and only the lower triangles of `K Kᵀ` and `Q Kᵀ` are read. With 32-token chunks those two fit one column block, so the saving comes from the three products with a triangular A, about 8% of a chunk's multiply-adds at head dims of 128, without changing any result. The decays, the triangular inverse and the state stay FP32, as in vLLM's CPU chunked kernel.

**Value blocks.** When the (sequence, value head) pairs would leave threads idle, as one long prompt does with 32 value heads on 64 threads, or 48 value heads on 32 threads in the second round, a pair runs as 2 or 4 work items, each with a block of 64 or 32 value columns of the state, output and `v`. Steps 1 to 3 do not depend on the value columns, so every block recomputes them; steps 4 and 5 and `U` are computed column by column in the same order whatever the block width, so blocks give exactly the bits of whole heads. That recomputation costs about 44% of a head's chunk time with bf16 sources and 30% with f16 or f32 (one core, Qwen3.6 shapes), so `gdn_direct` picks the split from a model of the `schedule(dynamic, 1)` loop: whole heads, the pairs of the last round in blocks (so that round fills the threads), or every pair in blocks, whichever is estimated at least 5% faster. The fused norm runs in its pass after all work items, so it needs no change.

The result matches the FP32 recurrent kernel rather than bit for bit: to bf16-level tolerances for bf16 sources, and to FP32 summation noise for f16 and f32 sources (plus the rounding of an f16 output). FP32 operands cost FP32 FMA, half the multiply-adds per instruction of `vdpbf16ps`, so the crossover with the recurrent form is at more tokens. On Turin (32 threads, Qwen3.6 shapes, 1 to 32 prefills) the chunked form at 64 and 128 tokens per sequence is 1.8-2.6x faster than the recurrent one with bf16 sources, 1.3-2.0x with f16 and 1.4-2.0x with f32. Each automatic threshold was set as the shortest average length at which it was at least as fast for 1, 8 and 32 prefills; with chunks of 32 tokens it is from about 12 tokens with bf16 and f32 sources and 20 with f16, so the thresholds are conservative.

### Native speculative verify

1. **Conv.** One parallel loop over 64-channel blocks convolves each sequence's tokens: the history taps from tap `num_accepted_tokens - 1` and the tokens, in FP32, through the conv decode takes on a conv state of the same strides (the vector conv for SD and DS layouts, DS also with draft taps or padding; a scalar one for other strides), then moves the `conv_width - 2` taps from tap `num_accepted_tokens` to the front and writes the tokens after them.
2. **Recurrence.** One parallel loop over (sequence, value head, value tile), with value tiles chosen exactly as a decode call over the same sequences chooses them. Each item runs its tokens in order: token *t* reads the state token *t* − 1 wrote (the accepted one for the first), updates it with decode's tile step and writes it to its own slot, so no state is copied.
3. **Norm.** With a fused norm, a third loop applies the gated RMSNorm per (token, head).

Each token runs decode's arithmetic on decode's tiles, so a verify call gives the bits of as many decode calls at the same `num_threads` (the thread count picks the tiles). It reads the accepted state once per (sequence, head) and writes one state per token: with 16 tokens at Qwen3.5-9B shapes (24 GDN layers of 32 value heads, FP32 state), 768 MiB per sequence per step.

### Mixed batches

A call with both parts runs the decode kernel on the decode sequences, then the prefill kernel on the prefills, each over all threads; the prefill algorithm is chosen from the prefills alone, so adding decodes to a batch does not change how its prefills run. The result is exactly that of a decode-only call followed by a prefill-only call.

## Limits

- One token per decode; draft tokens run in a separate speculative-verify call, which holds only verify sequences.
- Duplicate non-pad slots in one call are not supported.
- Native kernels: head dims multiples of 16 up to 256, conv width up to 8; the chunked prefill also needs head dims that are multiples of 32.
- `cu_seqlens` is int32, so a call holds fewer than 2<sup>31</sup> tokens.

## Example

One vLLM step with `nd` decodes and `np` prefills (Qwen3.6-35B-A3B layer, bf16 activations, FP32 state), passing vLLM's metadata as is:

```cpp
using namespace zendnnl::lowoha::gdn;

gdn_params p;
p.batch = nd + np;
p.num_decodes = nd;
p.cu_seqlens = query_start_loc;         // [nd + np + 1], decodes first
p.num_tokens = query_start_loc[nd + np];
p.has_initial_state = has_initial_state; // [nd + np] or nullptr
p.num_k_heads = 16;
p.num_v_heads = 32;
p.head_k_dim = p.head_v_dim = 128;
p.conv_width = 4;
const int64_t D = p.conv_dim();
p.qkv_stride = qkvz_width;             // view into the fused qkvz projection
p.a_stride = p.b_stride = 2 * 32;      // views into the fused ba projection
p.out_stride_t = 32 * 128;
p.out_stride_h = 128;
p.conv_state_slot_stride = 3 * D;      // SD conv pool [slot][tap][channel]
p.conv_state_tap_stride = D;
p.conv_state_dim_stride = 1;
p.ssm_state_slot_stride = 32 * 128 * 128;
p.ssm_state_head_stride = 128 * 128;
p.num_state_slots = slots;
p.src_dt = data_type_t::bf16;
p.state_dt = data_type_t::f32;
p.dt_bias_dt = data_type_t::bf16;

status_t st = gdn_direct(qkv, a, b, A_log, dt_bias, conv_w, conv_b,
        conv_pool, ssm_pool, slot_ids, out, p);
```

A decode-only step sets `p.num_decodes = p.batch` and may leave `cu_seqlens` and `has_initial_state` null with `num_tokens == batch`. An fp16 model (vLLM `--dtype float16`) sets `p.src_dt = data_type_t::f16` (and `p.dt_bias_dt` to the type of `dt_bias`); nothing else changes.

## Tests

`gtests/test_gdn.cpp` contains these groups of tests:
- the native decode and prefill kernels against the reference: Qwen3.6 and small shapes, bf16, f16 and f32 sources on f32 / bf16 / f16 state, kv and vk layouts, the fused norm, shuffled slots and pad slots (with NaN in a pad slot's `z` row), several decode steps so state errors compound; tolerances follow the source type (one or two output ulps of it);
- mixed batches (decodes and prefills in one call) against the reference: pad slots in both parts, decode flags that must not be read, mixed initial state, recurrent and chunked prefills chosen automatically or forced, for bf16, f16 and f32 sources;
- the mixed-batch contract, bit for bit: a mixed call equals a decode-only call followed by a prefill-only call on rebased rows; the decode flags, a null `has_initial_state` and padded metadata past the batch; a null `cu_seqlens` equals one token per sequence;
- the reference prefill against repeated reference decode steps;
- the chunked prefill against the reference at the precision of its operands: lengths around the chunk size, mixed initial state, kv and vk layouts, the fused norm and a small shape, for bf16 sources at bf16 tolerance and for f16 and f32 sources at tolerances bf16 operands would fail;
- the chunked prefill in value blocks: the split chosen for one or several prompts, pad slots, thread counts and head dims; against the reference for every source and state type, kv and vk, the fused norm and mixed batches; and bit for bit against whole heads (2 and 4 blocks for every pair, and the last round's pairs in 2 blocks);
- `allow_neg_eigval` on decode and both prefill forms against the reference, which doubles beta as well, for bf16 and f32 sources with and without the norm, and a check that it changes the result; and speculative verify with it on, bit for bit against decode calls;
- every source x state type combination, the DS conv-state layout (also with f16 and f32 sources), padded DS and padded-channel SD conv states, the `use_qk_l2norm`, `silu`, `round_conv_out` and `scale` options, the fused norm with both gates, an FP32 core output and an f16 norm weight, for bf16 and f16 sources, conv widths 1 to 8, thread counts and padded output rows;
- the fallback of a forced native request on unsupported shapes, bit-identical to the reference, and `gdn_native_supported` for the CPU and for supported and unsupported shapes;
- the prefill contract: a null `has_initial_state` equals all-false, and pad slots and empty sequences leave their outputs and slots untouched (outputs start as NaN so a skipped write is caught); `update_state = false` gives the outputs of `true` bit for bit and leaves both pools unchanged, lets several prefills continue from one slot (each as it would alone) and runs without pools when no prefill reads one (an empty prefill reads none, whatever its flag), on both kernels and both prefill algorithms;
- split passes, bit for bit against one call: the chunked form cut on chunk boundaries (after 3 chunks and after 1, at a prompt's end, at 0, before a rest shorter than the conv history, next to a pad slot) for Qwen3.5, Qwen3.6 and Qwen3.8 heads and a small shape, every source type, kv and vk, the fused norm and the DS conv state, at 1, 8 and 64 threads; the recurrent form cut anywhere (inside the conv history included) on both kernels, with the norm on the rounded and on the FP32 core output; prompts continuing read-only from one slot after one call prefilled their shared context, on packed rows and on the `chunk_gated_delta_rule` inputs, both kernels and both algorithms; and the chunked form cut off chunk boundaries within its operand tolerance;
- the `chunk_gated_delta_rule` inputs: separate q / k / v with padded head and token strides and g / beta computed outside the kernel (f32, and the source type) against the packed rows with the gate projections, on both kernels and both prefill algorithms, with mixed initial states, GQA, the fused norm and bf16, f16 and f32 sources, and log decays of either sign (growing states) on both native prefill algorithms against the reference kernel, for f16 sources also past f16's range (decays growing to e^16 in a chunk, an FP32 state of 65536);
- speculative verify, bit for bit against decode calls over the same sequences: sequences of 16, 9 and 2 draft tokens with accepted counts of 1 and of every token, every output, the state stored after every token and the conv state afterwards, for every source and state type, kv and vk, the fused norm and the SD and DS conv states, on both kernels, and on conv states of other strides (vLLM's DS buffer with draft taps decoded in place, a padded DS layout, an SD-like layout with padded channels) for f32 and bf16 sources; a second step continuing from the token the first accepted last, after a full or a short first step and with rows that drafted nothing or have fewer tokens than the step before accepted, at conv widths 4, 2 and 0; and pad and empty sequences;
- argument validation, including every rule on `num_decodes`, `cu_seqlens` and `num_tokens`, `update_state = false` with decodes or with a prefill that reads null pools, the rules of the `chunk_gated_delta_rule` inputs and of speculative verify (against a call that passes), and the batch-0 no-op.

`gtests/test_gdn_ck_gemm.cpp` checks GDN's CK GEMMs for each operand type (bf16, f32) against a naive reference on the operand values: `gemm` on row-major A and on A read as the transpose of a B operand (row tails of 1 to 7, the chunk's 64-row shapes, accumulation and padded C rows), the chunk's K Kᵀ through `pack_b_transposed`, `to_a` rounding (ties, binade carries, signed zeros and infinities), `pack_b` with strided rows, `pack_b_transposed`, and the triangular products (`lower_a`, `lower_c`) bit for bit against the full product.

## Benchmarking

`benchdnn --op=gdn` times `gdn_direct` on Qwen3.5 / Qwen3.6 / Qwen3.8 shapes, decode, prefill and mixed batches, with a cold cache by default; see [benchdnn/doc/gdn.md](../../../benchdnn/doc/gdn.md) and `benchdnn/input/gdn/gdn_models.txt`.
