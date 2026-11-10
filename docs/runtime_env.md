
(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# ZenDNNL Runtime Environment Variables

This document lists all environment variables available for configuring ZenDNNL at runtime.

---

## Configuration File

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_CONFIG_FILE` | Path to a JSON configuration file. When set, configuration is loaded from this file instead of individual environment variables. | None | Valid file path to a JSON config file |

---

## MatMul Algorithm Configuration
| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_MATMUL_ALGO` | Selects the MatMul algorithm/kernel to use for 2D matrix multiplication | `none` (auto-select) | `auto` (auto_tuner)<br>`0` (dynamic_dispatch)<br>`1` (aocl_dlp_blocked)<br>`2` (onednn_blocked)<br>`3` (libxsmm_blocked)<br>`4` (aocl_dlp)<br>`5` (onednn)<br>`6` (libxsmm)<br>`7` (batched_sgemm, AOCL-DLP's batch GEMM for batched calls; a single call runs on AOCL-DLP)<br>`8` (auto_tuner)<br>`9` (reference)<br>`10` (native_brgemm; `matmul_direct` runs every single (not batched) call without a per-call `lowoha_algo` on the native BRGEMM over the native VNNI layout, which serves the row-major BF16, INT8 (s8 and s4 weights) and FP32 calls listed on the matmul operator page. A call it does not implement (status `unimplemented`), or that needs an instruction set the CPU lacks (`isa_unsupported`), runs on the default route instead, as with the algorithm unset: through that route's input checks and with its weight cached out of place (weight cache mode 1 at most), with one warning line per process for each reason, `matmul_direct: under ZENDNNL_MATMUL_ALGO=10 a call the native BRGEMM does not serve (<reason>) takes the default route (logged once per reason)`. A call with wrong arguments (N or K below 1, an ldb below a weight row as the call lays the weight out, a bad shape, leading dimension or buffer), a call on a buffer the native BRGEMM rewrote in place or on a custom-kernel image, and a call the native BRGEMM fails after it took it keep their error status, with dst untouched when the failure comes before the call's tiles run (its checks, its weight's layout, its buffers); a thread that cannot allocate its scratch once the tiles run fails the call with dst partly written. So does a call of a form the default route returns success for with a wrong or unwritten dst, with one error line ending `not handed back: the default route does not compute <form> correctly`: a column-major call on a constant weight or with integer operands, a reordered source, W8A16, W4A16 with per-group or f16 weight scales, a float source the call does not quantize into an s32 / s8 / u8 dst, source, weight and dst types the default route has no kernel for (a float source and weight of two types, a u8 / s8 source with a float or u4 weight, an s4 weight with an f32 or u8 source), a source quantized to s8 with a source zero point or to u8 with a zero point per row, a weight zero point of an s4 weight, or of an s8 weight with more than one value or over a source the call quantizes, zero points with an alpha other than 1 over an s8 weight, f16 scales over s8 weights, f32 weight scales with bf16 source scales, and bf16 weight scales with f32 source scales over s4 weights; a `gelu_tanh_and_mul` call, which only the native BRGEMM implements; and in a build without AOCL-DLP what that build's default route (the reference kernel) does not compute: a column-major call, a reordered or GGML packed weight, per-group weight scales of an s8 weight without source scales of the same groups, a binary operand that is neither `{1, N}` nor rows of N values, and a pooling post-op. A call with `lowoha_algo = native_brgemm` gets the error for every call it does not serve. A batched call takes its kernel from `ZENDNNL_BMM_ALGO` (and logs the line once), and group matmul runs as with the algorithm unset)<br>`11` (moe_custom_kernel, the algorithm of group matmul's MoE custom kernel: a single call on a weight packed for that kernel returns failure, and a single call on any other weight runs on AOCL-DLP)<br>Any other value, `-1` included, names no algorithm and does not select the default: a single call without a per-call `lowoha_algo` then runs on `aocl_dlp` (`reference` in a build without AOCL-DLP) |
| `ZENDNNL_MATMUL_NATIVE_CONFIG` | Developer override of the native BRGEMM's plan (ALGO 10): clauses `[filters:]key=value,...` separated by `;`, the first clause whose filters hold applies to a call. A call whose N is not a multiple of 32 matches `n` on its width padded to whole 32-column blocks, which the `[MATMUL.NATIVE_BRGEMM]` line shows as `padded_n=`. A call on s4 weights matches `dtype=s4` and `group` on its group size (`group` is 0 for other weights). A bad clause is dropped with one warning; a setting that is illegal for a call is dropped for that call, with one warning the first time; a call never fails because of either. Read once, at the first native call. Takes precedence over `ZENDNNL_MATMUL_NATIVE_TUNING_FILE`. | unset (built-in plan) | e.g. `nr=16,mb=24` or `m=1-16,n=4096:nr=16;dtype=s8:int8_mr=8` |
| `ZENDNNL_MATMUL_NATIVE_TUNING_FILE` | Path of a tuning file for the native BRGEMM (ALGO 10): versioned JSON (`zendnnl-native-brgemm-tuning`, version 2) with a hardware match by CPU fields and ranges, entries (a problem match and a plan setting) and rules; a call whose N is not a multiple of 32 matches `n`, and the rule features computed from N, on its padded width (`padded_n=` on the `[MATMUL.NATIVE_BRGEMM]` line). A file that is unreadable, malformed or made for other hardware is ignored with one warning; a bad entry is dropped with one warning (for the first 16; one line counts the rest). Read once, at the first native call. | unset (built-in plan) | path to a `.json` file |
| `ZENDNNL_BMM_ALGO` | Selects the Batch MatMul algorithm/kernel to use for 3D batch matrix multiplication | unset (`none`); see note below | Integer ID or name. `0` (dynamic_dispatch)<br>`1` (aocl_dlp_blocked)<br>`2` (onednn_blocked)<br>`4` (aocl_dlp)<br>`5` (onednn)<br>`6` (libxsmm)<br>`7` (batched_sgemm)<br>Any other value, `-1` included, names no algorithm and does not select the default: a batched call without a per-call `lowoha_algo` then runs on `aocl_dlp` (`reference` in a build without AOCL-DLP) |
| `ZENDNNL_CACHE_OFF` | Process-wide cache kill switch for covered cache infrastructure. Equivalent JSON knob: top-level `global_cache_off`. | `0` (disabled) | `1`/`true`/`on`/`yes` (force covered caches off)<br>`0`/`false`/`off`/`no` (leave local cache knobs in control) |
| `ZENDNNL_MATMUL_WEIGHT_CACHE` | Weight-reorder caching mode for blocked / auto-tuner algorithms and grouped-matmul custom paths. W8A8 grouped-MoE ALGO 4 requires an enabled persistent pack and declines when this is `0`; BF16 and caller-prequantized S8 may then use generic fallback. | `2` (in-place where supported) | `0` (disabled)<br>`1` (out-of-place, enabled)<br>`2` (in-place: reuse the user weight buffer as the reorder destination when the layout is safe, otherwise fall back to out-of-place; the buffer must be writable, so pass a per-call `weight_cache_type` of `1` for read-only weights such as a read-only mmap; auto-downgraded to `1` under auto_tuner / dynamic_dispatch; grouped-matmul AUTO keeps `2` only with prepack, cross-warm, the custom-kernel decode pack and unlimited LRU capacity. A BF16 or FP16 layer of several experts then uses one ALGO 3 custom-kernel layout in both phases and keeps AOCL layouts out-of-place; eligible BF16/FP16 packs may use the caller's buffer. An INT8 / W4A8 ALGO-3 MoE layer uses one out-of-place custom-kernel pack in both phases. Other grouped calls may write only their AOCL prompt reorder in place). INT8 in-place needs `matmul_params::wei_buffer_capacity_bytes` when the blocked size exceeds `k*n` (`0` means exactly `k*n`). A partitioned BMM that shares one weight and has a source zero-point stays out-of-place even when that image already equals `k*n`. A call's `matmul_params::weight_cache_type` caps that call to the lower of this mode and the per-call value; the GGML Q8_0/Q4_0 unpack cache and the W4A8 plain s8 cache that ALGO 3 reads ignore a per-call `0`. The ALGO 10 native BRGEMM, its GEMV kernels included, takes the lower of the two values and rewrites a buffer in place only under `ZENDNNL_MATMUL_ALGO=10`, and then only a bf16 or raw s4 weight or an AOCL-DLP blocked prepack whose layout fits the buffer. At mode 2 under `ZENDNNL_MATMUL_ALGO=10` a constant weight has that one layout: its M = 1 calls run on it (in place, or out of place where the layout does not fit the buffer) instead of on a GEMV kernel's own pack; the GEMV kernels serve M = 1 calls at mode 0 or 1. A bf16 or s4 weight is rewritten only with a contiguous `ldb` (its row length) and N a multiple of 32, and its layout fits when K is even (bf16) or a multiple of 8 (s4), otherwise only within a stated `matmul_params::wei_buffer_capacity_bytes`. Since mode 2 is the default, a bf16 or s4 weight in a read-only buffer, such as a read-only mmap of a checkpoint, is written to under `ZENDNNL_MATMUL_ALGO=10` unless its calls pass a `weight_cache_type` of `1` (or this variable is `1`). An f32 or raw s8 weight and a weight whose N is not a multiple of 32 are packed out of place, keeping the raw buffer next to the layout (for an f32 weight, a second K x N x 4-byte copy). A call it hands to the default route under `ZENDNNL_MATMUL_ALGO=10` caches its weight out of place (mode 1 at most), as a later call on the buffer may read it raw on the native BRGEMM, and holds the raw weight until it returns (so does a batched call without a per-call `lowoha_algo`, over every weight it reads): an in-place rewrite of the buffer by a concurrent native call waits for it. |
| `ZENDNNL_MATMUL_CUSTOM_KERNEL` | Routes eligible wide-N `matmul_direct` calls through one-op grouped AUTO: decode-class (`M <= 32`) executes ALGO 3 CK, and a prompt executes the CK prompt tile when `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_TILE` lists its dtype, else full-weight AOCL while warming the future decode layout. A single-expert call that `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_TILE` / `..._DECODE_TILE` covers (by default both phases of every dtype) also routes below decode's N-tile count when `..._FORCE_CK` (default `all`) lists its dtype, at any N the custom-kernel pack divides (see the group-matmul notes below): a covered phase executes ALGO 3 CK, and a raw weight routes in both phases so grouped AUTO owns all of its layouts (an AOCL-prepacked weight routes only its covered phases). Supported scalar families are BF16 (BF16 destination), DQ-INT8 (per-token source, per-channel S8 weight scales), W4A8 (per-group S4 weight scales), FP16 (single-expert policy only) and static per-tensor U8×S8 with a positive F32/BF16 source scale, S32 source zero-point, symmetric S8 weights with scalar or per-channel `{N}` / `{1,N}` F32/BF16 scales, and BF16/F32/U8 destination. U8 destination additionally requires a positive F32/BF16 destination multiplier and S32 destination zero-point. Static U8×S8 may use either no post-op, one buffer-free ReLU/sigmoid, or the exact `binary_mul(F32[M,N]) -> binary_add(F32[M,N])` chain. Scalar routing requires the effective backend to resolve to AOCL-DLP blocked; on an AOCL build this includes the unset/default MatMul algorithm as well as explicit `ZENDNNL_MATMUL_ALGO=1`, while explicit non-AOCL backends remain configured. Plain and supported caller-prepacked weights retain their representation through AUTO selection. For scalar prepack compatibility, `mem_format_b='r'` with unset or AOCL-DLP provenance is treated as AOCL-DLP blocked; `lowoha_algo=moe_custom_kernel` explicitly identifies CK-VNNI. Only a validated ALGO-3 custom-kernel path converts supported AOCL-DLP BF16 or pure static U8×S8 bytes: AOCL unreorders into temporary raw storage and CK packs directly into the final bounded-cache arena, so no persistent raw copy or full-size CK memcpy remains. AOCL symmetric-quant S8 packs remain on AOCL because that format has no unreorder API. Prompt/AOCL and every CK fallback retain the original blocked pointer. Also requires constant weights and compatible cache/threading settings. Equivalent JSON knob: `runtime_variables.matmul.custom_kernel`. | `1` (enabled) | `0` (standard path)<br>`1` (eligible prompt/decode routed through grouped AUTO when the effective backend is AOCL-DLP blocked) |
| `ZENDNNL_DLP_M_HINT` | Expected M dimension for supported AOCL-DLP GEMMs. Equivalent JSON knob: `runtime_variables.matmul.dlp_m_hint`. A valid JSON config replaces environment-variable configuration. | `0` (hints disabled) | Non-negative base-10 integer; `0` disables hints |
| `ZENDNNL_ENABLE_ROUTED_MOE` | Enable the routed-MoE fast executor for eligible calls. Set to `0` to skip the fast executor and use the internal vector `group_matmul_direct` fallback. The grouped fallback still follows `ZENDNNL_GRP_MATMUL_ALGO`; set that to `4` to request ALGO 4. Unsupported calls use the generic fallback regardless of this setting. | `1` (enabled) | `1` (routed fast-then-fallback policy)<br>`0` (disable routed fast and use generic fallback) |
| `ZENDNNL_ZP_COMP_CACHE` | Enable/disable zero-point compensation caching for quantized operations. Forced off when `ZENDNNL_CACHE_OFF=1` or JSON `global_cache_off=true`. | `1` (enabled) | `1`/`true`/`on`/`yes` (enabled)<br>`0`/`false`/`off`/`no` (disabled) |
| `ZENDNNL_ENABLE_POSTOP_CACHE` | Enable/disable AOCL DLP post-op metadata caching for the matmul fast path. When ON, the per-call build of post-op metadata (bias, binary_add, binary_mul, sum scales, src/wei/dst quant scales) is memoized and reused on subsequent calls with the same shape, dtypes, and post-op chain; mutable fields (data pointers, dynamic scale buffers) are refreshed on every hit. When OFF, the cache is cleared on every call and the cold path runs each time. Cached on first read after config resolution. Forced off when `ZENDNNL_CACHE_OFF=1` or JSON `global_cache_off=true`. Recognized spellings are case-insensitive; unrecognized or empty values leave the default untouched. | `1` (enabled) | `1`/`true`/`on`/`yes` (enabled)<br>`0`/`false`/`off`/`no` (disabled) |

> **`ZENDNNL_BMM_ALGO` default is path- and build-dependent.** The stored config value when the variable is unset is `none` (`-1`). The operator (`matmul`) path resolves `none` to `aocl_dlp` (4). The LowOHA `matmul_direct` batched path initially selects `libxsmm` (6), but uses it only when ZenDNN was built with `ZENDNNL_DEPENDS_LIBXSMM=ON`. Otherwise, backend availability remaps it to `aocl_dlp` when AOCL-DLP is available, or to `reference` when AOCL-DLP is also unavailable. Therefore, the default LIBXSMM-disabled build runs LowOHA BMM with AOCL-DLP. Set the variable explicitly to request one backend across both paths; requests for a backend that was not built are still remapped to an available fallback.

### Algorithm Details

| Algorithm ID | Name | Description |
|--------------|------|-------------|
| unset | `none` | No algorithm selected (uses default selection logic); setting `-1` does not select it (see `ZENDNNL_MATMUL_ALGO` above) |
| 0 | `dynamic_dispatch` | Dynamic kernel dispatch based on heuristics (supported by LOA) |
| 1 | `aocl_dlp_blocked` | Blocked AOCL DLP algorithm |
| 2 | `onednn_blocked` | Blocked OneDNN algorithm |
| 3 | `libxsmm_blocked` | Blocked LIBXSMM algorithm (supported by LOA) |
| 4 | `aocl_dlp` | AOCL DLP algorithm |
| 5 | `onednn` | OneDNN algorithm |
| 6 | `libxsmm` | LIBXSMM algorithm (supported by LOA) |
| 10 | `native_brgemm` | Native BRGEMM over the native VNNI weight layout (supported by LOA); under `ZENDNNL_MATMUL_ALGO=10` a call it does not implement takes the default route, and with `lowoha_algo = native_brgemm` it returns an error status |
| auto | `auto_tuner` | Auto-tuner (automatically selects best algorithm, supported by LOA) |

---

## Group MatMul Configuration

User-facing knob for `group_matmul_direct` (the grouped GEMM dispatcher used by MoE expert layers and other parallel-GEMM workloads).  See `docs/operator/low_overhead_operator/lowoha_group_matmul_operator.md` for full operator semantics.

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_GRP_MATMUL_ALGO` | Selects the grouped-matmul whole-call mode. Unset, AUTO picks per call and runs every dense single-expert layer, and every MoE layer of BF16, FP16, symmetric DQ-INT8 or W4A8 experts the custom kernel serves, on ALGO 3 with the custom kernel in both phases, AOCL-DLP only where the custom kernel cannot serve the layer. An explicit `0` is AUTO with the earlier routing: the single-expert prompt on AOCL-DLP (ALGO 1) and its decode on ALGO 3 with the custom kernel; BF16 / FP16 MoE layers still on ALGO 3 in both phases; INT8 / W4A8 MoE layers on the shape rules in both phases (prompt Rule 0.7, ALGO 1; decode Rules 0.5 / 0.6). `{1,2,3,5,6}` pin a generic scheduler globally when the active phase knob is unset. `4` attempts the private symmetric per-channel W8A8 fused-MoE fast path in both phases, in either BF16-dynamic or caller-prequantized S8 mode. ALGO 4's direct-S8 fast path requires positive finite per-token BF16/F32 source scales, caller-owned BF16 `dst_down`, and `row_ptrs` that permute destination rows. `dst_down[i]` must be the tight BF16 view over the exact same BF16-sized backing as `src[i]` (`ldc_down[i] == hidden`), whose leading bytes hold the S8 source prefix; separate, offset, partial, padded, and cross-expert destinations decline ALGO 4. Any eligibility decline writes nothing and falls back to generic AUTO. ALGO 4 allocation/execution errors remain terminal. Read once and cached on first reference. | unset (AUTO) | `0` (AUTO, single-expert prompt on AOCL-DLP), `1` (sequential), `2` (flat M-tile), `3` (flat N-tile), `4` (two-mode W8A8 fused-MoE attempt), `5` (per-expert), `6` (multilevel CCD) |
| `ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO` | Selects the decode-phase policy (`max active M <= 32`) and, when explicitly set, outranks `ZENDNNL_GRP_MATMUL_ALGO`. Unset, an N-tile-safe call of an ALGO-3 MoE layer (see `ZENDNNL_GRP_MATMUL_MULTI_EXPERT_ALGO3`) runs ALGO 3. `{1,2,3,5,6}` pin that scheduler; `0` selects the legacy cascade; `4` attempts W8A8, then inherits the decode default (ALGO 3) on decline. Tile-safety failures clamp to ALGO 1. | `3` | `0`, `1`, `2`, `3`, `4`, `5`, `6` |
| `ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO` | Selects the prompt-phase policy (`max active M > 32`) and, when explicitly set, outranks `ZENDNNL_GRP_MATMUL_ALGO`. Unset, an N-tile-safe call of an ALGO-3 MoE layer (see `ZENDNNL_GRP_MATMUL_MULTI_EXPERT_ALGO3`) runs ALGO 3; every other multi-expert prompt selects ALGO 1. `{1,2,3,5,6}` pin that scheduler with tile-safety clamps; `0` selects the legacy cascade; `4` attempts W8A8, then inherits the unpinned prompt policy on decline. | unset (ALGO 1 for shape-rule prompts) | `0`, `1`, `2`, `3`, `4`, `5`, `6` |
| `ZENDNNL_GRP_MATMUL_MULTI_EXPERT_ALGO3` | Dtype set of the MoE layers (at least two experts) unpinned AUTO keeps on ALGO 3 with the custom kernel in both phases; tokens as for the single-expert knobs (`bf16`, `f16`, `int8`, `w4a8`, `all`, `none`, joined by `,` `+` or spaces; `u8s8` has no effect). An INT8 / W4A8 layer also needs every condition in the group-matmul notes below. A layer whose dtype is not listed keeps the shape rules. Does not change the WEIGHT_CACHE=2 ownership guards. A value with no known token keeps the default. Read once. | `all` (`bf16,f16` under an explicit `ZENDNNL_GRP_MATMUL_ALGO=0`) | dtype set |

**Selector precedence (highest first):**

1. `ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO` / `ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO`
   — the knob matching *this call's phase*, when explicitly set. It outranks
   `ZENDNNL_GRP_MATMUL_ALGO` whatever the global is set to, so tuning one phase
   does not require clearing the global first. A phase `4` requests W8A8 for
   that phase even under a global generic pin; an explicit non-`4` phase knob
   suppresses a global `4` for that phase.
2. `ZENDNNL_GRP_MATMUL_ALGO` — governs the phase whose knob is **unset**. A
   global `4` attempts W8A8; a global generic pin `{1,2,3,5,6}` pins that
   scheduler.
3. AUTO heuristics, when neither the matching phase knob nor the global is set.

The knob for the inactive phase has no effect on the current call. Safety
clamps always apply last: ALGO 2 requires `m_tile_safe` and ALGO 3 requires
`n_tile_safe`, and either failure falls back to **ALGO 1**. A phase `=5` pin is
still subject to the ALGO 5 occupancy/dtype qualifier. All three variables are
cached; set them before the first grouped-matmul call.

> With `ZENDNNL_MATMUL_WEIGHT_CACHE=2`, a multi-expert BF16/FP16 layer, an
> INT8 MoE layer whose weights declare a writable buffer capacity, and a
> raw BF16/FP16 single expert whose both phases the single-expert knobs below
> cover, do not support splitting a non-AUTO global ALGO pin with a phase
> override (INT8 and W4A8 custom-kernel packs are always out of place),
> whatever `ZENDNNL_GRP_MATMUL_MULTI_EXPERT_ALGO3` lists. The
> call fails before prepack or destination writes. Use weight-cache mode `1`,
> remove the phase override, or use global ALGO `0`.
>
> Internal tuning knobs (planner thresholds, custom-kernel sub-knobs,
> prepack-cache sizing, etc.) live inside the implementation and are
> not part of the user-facing contract.  Shared ones are documented in
> `zendnnl/src/lowoha_operators/matmul/group_matmul/
> common/group_matmul_parallel_common.hpp`; per-ALGO ones live with their ALGO
> (`executors/m_tile/`, `executors/n_tile/`, `multi_expert/expert_parallel/`, `single_expert/`).  Production
> deployments should not need them.  If a workload requires one, file an issue
> describing the problem and we will assess whether to promote the knob
> to user-visible status.
>
> One exception is worth knowing, because it changes what executes
> rather than only how fast it runs: under a resolved ALGO 5, a fused
> MoE call runs W13, the gated activation and W2 in a single per-expert
> pass instead of two passes.  It is on by default for the dtype
> regimes that support it and declines to the two-pass elsewhere.  Set
> `ZENDNNL_GRP_MATMUL_ALGO5_VERTICAL_FUSION=0` to force the two-pass
> everywhere — the A/B baseline, and the escape hatch if a deployment
> sees a regression.
>
> Similarly, under a resolved ALGO 3, prompt calls (`max active M > 32`) on
> the custom microkernel run through the prompt tile executor, which
> schedules (expert, M-block, N-slice) tiles across L3-domain thread teams
> without per-round barriers.  The output is bitwise identical to the round
> executor.  AUTO runs both phases of a layer of at least two experts on
> ALGO 3 (an "ALGO-3 MoE layer") when every active expert has a dtype
> `ZENDNNL_GRP_MATMUL_MULTI_EXPERT_ALGO3` lists and is either BF16 or FP16 on
> a weight the caller did not AOCL-prepack, or symmetric DQ-INT8
> (per-channel S8 weights) or W4A8 (per-group S4 weights, groups a multiple
> of 8) with one source scale per row, no zero point, no post-op, a weight
> the custom kernel packs (`ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL`,
> `..._CUSTOM_KERNEL_INT8`, ISA, K a multiple of 4 for INT8 and at most the
> INT8 accumulation depth, an N the pack divides), not AOCL-prepacked, and a
> family the prompt tile executor serves (its `ZENDNNL_GRP_MATMUL_PROMPT_TILE`
> bit set and `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY` not `1`).  Any other layer
> keeps the shape rules: prompt on Rule 0.7 (AOCL-DLP, ALGO 1), decode on
> Rules 0.5 / 0.6.  Asymmetric (`compute=u8`) INT8 keeps the shape rules.
> `ZENDNNL_GRP_MATMUL_MULTI_EXPERT_ALGO3` takes the dtype tokens of the
> single-expert knobs below (`bf16`, `f16`, `int8`, `w4a8`, `all`, `none`;
> `u8s8` has no effect) and defaults to `all`, or `bf16,f16` under an
> explicit `ZENDNNL_GRP_MATMUL_ALGO=0`; it is read once per process.  The
> custom-kernel pack of a BF16 / FP16 layer may use the caller's buffer at
> `ZENDNNL_MATMUL_WEIGHT_CACHE=2`; INT8 and W4A8 packs stay out of place and
> those layers keep the mixed WEIGHT_CACHE=2 policy.  The executor needs at least two active experts, an N the
> custom-kernel pack divides and `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY` at any
> value but `1`.
>
> Group-matmul calls of exactly one expert (one expert in the pool, not one
> active expert of an MoE pool) follow a single-expert custom-kernel policy
> of three dtype-set knobs.  A value is a list of tokens joined by `,`, `+`
> or whitespace (case-insensitive): `bf16`, `f16`, `int8` (DQ-INT8), `u8s8`
> (static U8×S8), `w4a8`, `all` / `1` / `on` / `true` / `yes` (every dtype)
> and `none` / `0` / `off` / `false` / `no` (nothing); tokens are OR-ed.
> Unknown tokens are ignored, and a value with no known token keeps the
> knob's default; both log one warning.  The knobs are read once per
> process.  A phase of a call runs ALGO 3 on the custom kernel when a knob
> covering that phase lists the call's dtype, the custom kernel serves it
> (`ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL`, `..._CUSTOM_KERNEL_INT8`, `..._F16`,
> ISA), the custom-kernel pack divides N and the custom kernel takes the
> epilogue:
>
> * `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_TILE` (default `all`) — the
>   prompt runs ALGO 3 on the prompt tile executor, which splits M across the
>   thread teams.  A BF16 prompt uses one run of equal M-blocks per team
>   (each block's A within half of L2) and one equal N-slice per thread
>   unless `ZENDNNL_GRP_MATMUL_PROMPT_TILE_MB` / `_NS` force a size, and at
>   pack NR 32 runs 12-row microkernel calls.  The other dtypes use the
>   multi-expert tile sizing, INT8 / W4A8 on the source quantized once per
>   call, and an FP16 prompt at pack NR 32 runs 12-row calls over two
>   o-blocks.  Not static U8×S8 (the executor does not tile it), and not under
>   `ZENDNNL_GRP_MATMUL_PROMPT_TILE=0`, K-blocking or
>   `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=1`; a prompt the executor does not
>   take runs the multi-expert kernels, without the prompt MR bound, K
>   chunks or FP16 prompt kernel.  A dtype it leaves out keeps
>   Rule 0.7 (AOCL-DLP, ALGO 1) for the prompt; a layer whose weights the
>   custom kernel owns should list its dtype, since its prompt then needs
>   no AOCL layout.
> * `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_KC` (default `4096`) — a prompt
>   tile that can chunk K (act=none BF16 at pack NR 32 with an even K and no
>   bias, or act=none FP16) and is taller than one microkernel call
>   accumulates in K chunks of this depth when the B of one call (one
>   pack-width column block for BF16, two for FP16) over the whole K
>   overflows L2 (deep-K layers such as FFN down projections); the tile
>   planner sizes its M-blocks and N-slices for one chunk.  BF16 caps the
>   chunk so the tile's B chunk stays in L2, and no chunk is shallower than
>   64.  Gated tiles keep the full K.  The result matches the full-K tile
>   bit for bit (an F16-destination FP16 tile sums its chunks in FP32 and
>   narrows once).  `0` keeps the full K.  The prompt tile knob
>   `ZENDNNL_GRP_MATMUL_PROMPT_TILE_KC` (`kc` in
>   `ZENDNNL_GRP_MATMUL_PROMPT_TILE_PROFILE`) sets the chunk per shape.
> * `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_DECODE_TILE` (default `all`) — the
>   decode runs ALGO 3 on the custom-kernel N-tile path, also at one thread.
>   Off, decode keeps Rule 0.45 (which also picks ALGO 3 unless
>   `ZENDNNL_GRP_MATMUL_DENSE_DECODE_NTILE=0`) and, through `matmul_direct`,
>   main's N-gated routing.
> * `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_FORCE_CK` (default `all`) — the two
>   phase knobs cover a dtype it lists at any N; any other dtype only at
>   decode's N-tile count (at least two custom-kernel column blocks per
>   thread, the gate `matmul_direct` applies).  It never adds a phase: the
>   phase knobs take precedence.
>
> The defaults therefore run both phases of a single expert on the custom
> kernel, at any N; an explicit `ZENDNNL_GRP_MATMUL_ALGO=0` keeps its prompt
> on AOCL-DLP (the prompt knob reads as empty).  All three at `0` reproduce
> main.
> No knob applies under `ZENDNNL_GRP_MATMUL_N_TILE_STRATEGY=0`.  A one-thread
> call stays on ALGO 3 when the knob of its own phase covers it.  The policy
> covers both passes of a dense fused FFN (ZenTorch's fused FFN) and, through
> the `matmul_direct` route, linears at any N for a dtype `FORCE_CK`
> lists.  A raw weight routes in both phases once
> either is covered, so grouped AUTO owns all of its layouts (even where it
> then selects ALGO 1); an AOCL-prepacked weight routes only its covered
> phases.  With both phases covered the layer has one custom-kernel pack:
> out of place at `ZENDNNL_MATMUL_WEIGHT_CACHE=1`; at
> `ZENDNNL_MATMUL_WEIGHT_CACHE=2` a BF16 (even K) or FP16 pack may go in the
> caller's buffer (no AOCL layout is written there), while INT8 and W4A8
> packs stay out of place.  A later call on that buffer that does not reach
> the custom kernel (for example a `matmul_direct` with an F32 destination,
> `beta != 0` or a post-op the route refuses) fails closed rather than read
> the packed bytes.  With one phase covered, the other keeps the AUTO rules
> and grouped AUTO's mixed WEIGHT_CACHE=2 policy.  An AOCL-prepacked weight is
> covered only for BF16 and static U8×S8, which the custom kernel converts.
> An explicit `ZENDNNL_GRP_MATMUL_AUTO_{PROMPT,DECODE}_ALGO` pin still wins.
> Examples: decode on the custom kernel only for layers with decode's N-tile
> count, `..._FORCE_CK=0`; decode on the custom kernel for W4A8 only,
> `..._DECODE_TILE=w4a8`; the prompt on AOCL-DLP,
> `..._PROMPT_TILE=0`; the prompt tile for BF16 only at any N and for
> every other dtype only above the gate, `..._PROMPT_TILE=all
> ..._FORCE_CK=bf16`.  Set all three to `0` for the AUTO shape rules (prompt
> on ALGO 1, Rule 0.45 decode) and main's N-gated
> `matmul_direct` routing.  Output matches the AOCL path to the dtype's
> tolerance, not bitwise.  A single-expert call ignores the family bits of
> `ZENDNNL_GRP_MATMUL_PROMPT_TILE` below; only `0` keeps it off the tile.
> `ZENDNNL_GRP_MATMUL_PROMPT_TILE` is a family bitmask (1 BF16, 2 DQ-INT8,
> 4 W4A8, 8 FP16; default 15, every family); `0` keeps every prompt on the
> round executor.
>
> A lone expert's custom-kernel decode (DQ-INT8 symmetric or BF16, no bias,
> no activation or post-op) can run a blocking plan instead of the N-only
> split:
> * `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_BLOCK` (default unset: the N-only
>   split) — `key=value` pairs: `split=legacy|remainder|grid`;
>   `mparts=<n>` and `kparts=<n>|auto` (the thread grid is M parts x N parts
>   x K parts, N parts = threads / (M parts x K parts); `remainder` gives
>   every thread its even share of whole column blocks and splits only the
>   remainder blocks along K); `kc=<n>` (K chunk per accumulation);
>   `mr=<n>` (rows per microkernel call, 1-12); `order=n|k` (loop order inside
>   a work unit).  A malformed value is warned once and ignored; a plan the
>   shape cannot take falls back to the N-only split.
> * `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_KSPLIT` (default `0`) — with no
>   `..._BLOCK` value, `1` gives the `remainder` plan to every decode whose
>   column blocks do not divide evenly over the threads (K ranges of at
>   least 256).
>
> Split K ranges are summed exactly for DQ-INT8 and in f32 for BF16.
>
> Two further ALGO 5 knobs are documented here only because a declined pin
> names them in a default-level warning, so an operator can be pointed at
> them by the library itself:
> `ZENDNNL_GRP_MATMUL_DECODE_ALGO5_GATE` and
> `ZENDNNL_GRP_MATMUL_PROMPT_ALGO5_GATE`, both `{0,1}` defaulting to `1`.
> At `1` an `AUTO_{DECODE,PROMPT}_ALGO=5` pin is honoured only where ALGO 5
> was measured to win (decode also requires s8 weights on every active
> expert; both require the active experts to saturate the thread team).
> Set either to `0` to honour the corresponding pin verbatim.  Both are read
> once and cached, so setting them after the first grouped call has no
> effect.  Any non-numeric or out-of-domain value resolves to the default.
>
> `ZENDNNL_ENABLE_GROUP_DQ` (`{0,1}`, default `1`) selects the grouped
> source-quantization pre-pass; `0` falls back to per-expert quantization.
> It is likewise cached from this release onward — previously it was re-read
> on every call, so an in-process `setenv` no longer takes effect.

### Examples

```bash
# Default: auto-select (recommended for production MoE inference).
./your_app

# Pin ALGO 3 (N-tile) for the whole process (e.g. for A/B
# benchmarking against the auto choice).
export ZENDNNL_GRP_MATMUL_ALGO=3
```

---

## Auto-Tuner Configuration

The auto-tuner automatically selects the best-performing algorithm for matrix multiplication by benchmarking different kernels.

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_AUTO_TUNER_TYPE` | Selects the auto-tuner version/strategy | `1` | `1` (v1 strategy), `2` (v2 strategy) |
| `ZENDNNL_MATMUL_SKIP_ITER` | Number of warmup iterations before algorithm evaluation begins | `2` | Positive integer |
| `ZENDNNL_MATMUL_EVAL_ITER` | Number of evaluation iterations to benchmark each algorithm | `3` | Positive integer |
| `ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES` | Comma-separated list of algorithm IDs to evaluate | `1,2` | Comma-separated list of algorithm IDs; `0` (dynamic_dispatch), `8` (auto_tuner) and `10` (native_brgemm, which runs only the calls that ask for it) are skipped |

### Auto-Tuner Phases

1. **Skip Phase** (`ZENDNNL_MATMUL_SKIP_ITER`): Initial warmup iterations to stabilize caches before measurements.
2. **Evaluation Phase** (`ZENDNNL_MATMUL_EVAL_ITER`): Tests multiple algorithms and measures execution times.
3. **Execution Phase**: Uses the cached best-performing algorithm for subsequent calls.

---

## Embedding Bag Kernel Configuration

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_EMBAG_ALGO` | Selects the Embedding Bag kernel to use | `none` (auto-select) | `1` (native kernel), `2`  (FBGEMM; requires a build with `-DZENDNNL_DEPENDS_FBGEMM=ON`, otherwise runs native)
---

## Embedding Bag Threading Algorithm Configuration

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_EMBAG_THREAD_ALGO` | Selects the Embedding Bag threading algorithm to use | `1` (table_threaded) | `0` (dynamic_dispatch), `1` (table_threaded), `2` (batch_threaded), `3` (ccd_threaded), `4` (hybrid_threaded), `5` or `auto` (auto_tuner). The integer maps directly to the `eb_thread_algo_t` enum; `dynamic_dispatch` and `auto_tuner` both fall back to `table_threaded` at execution time.
---

## Embedding Bag ISA Selection

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_EMBAG_ISA` | Makes the LowOHA Embedding Bag run the AVX2 kernel instead of the native AVX-512 kernels on CPUs that support both, to test the path that CPUs without AVX-512 (e.g. AMD Zen 2 and Zen 3) take. Operations the AVX2 kernel does not cover (FP16, INT8 and INT4 tables) then return `status_t::isa_unsupported` unless they run on the reference kernel, which this variable does not affect. An unsigned INT4 table (`u4`) also runs on FBGEMM when the build has it (`-DZENDNNL_DEPENDS_FBGEMM=ON`) and the operation is a sum with FP16 scale/bias; FBGEMM does not take signed INT8 or INT4 tables. FP16 tables and outputs never do: they are rejected with `status_t::isa_unsupported` on a CPU without AVX512-FP16 before a backend is selected, and FBGEMM has no FP16 kernel. Read on every call. | unset (selected by CPU) | `avx2`
---

## Group MatMul Deep-K Blocking

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_GRP_MATMUL_KBLOCK` | Shared deep-K policy for supported BF16 and symmetric dynamic-INT8 custom-kernel tiles. Automatic mode is limited to single-expert decode; dtype, epilogue, M, and L2-capacity gates still apply. A single-expert prompt tile chunks K under `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_KC` instead, whatever this is set to; `1` keeps BF16 prompts off the prompt tile executor. Cached on first read. | unset (automatic) | `0` (off), `1` (on where shape gates allow) |
| `ZENDNNL_GRP_MATMUL_INT8_KBLOCK` | Narrow override for the symmetric dynamic-INT8 path only. Useful as an INT8 escape hatch or A/B switch without changing the established BF16 policy. Cached on first read. | unset (defer to shared policy) | `0` (INT8 off), `1` (INT8 on where shape gates allow) |

The INT8 path is native custom-kernel execution; these controls do not add or
select an AOCL fallback. Unsupported shapes continue through the pre-existing
dispatcher.

---

## SDPA Flash Kernel Configuration

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_SDPA_BF16_CK` | Runs the BF16 GEMMs of flash SDPA on the BF16 custom-kernel (CK) microkernel instead of AOCL-DLP, packing B on the fly; the rest of the flash loop is unchanged. Applies to every BF16 flash GEMM with alpha 1, beta 0 or 1 and at least 3 rows: Q*K^T and P*V of the BF16 path, and the BF16 P*V of the INT8 path with INT8 Q*K^T only (its scaled BF16 Q*K^T, and one- or two-row GEMMs such as decode, stay on AOCL-DLP). On by default; needs AVX512-BF16, AVX512-BW and AVX512-VL, otherwise AOCL-DLP runs (with a warning only when the variable is set). Cached on first read. | `1` | `0`/`false`/`off`/`no` (AOCL-DLP), `1`/`true`/`on`/`yes` (custom kernel); other values keep the default |
| `ZENDNNL_SDPA_INT8_CK` | Runs the INT8 matmuls of dynamic-INT8 flash SDPA (`is_qk_quant` / `is_pv_quant`, or ZenTorch's `ZENTORCH_SDPA_QK_QUANT` / `ZENTORCH_SDPA_PV_QUANT`) on the INT8 custom-kernel (CK) `vpdpbusd` microkernel instead of AOCL-DLP, packing B on the fly: the S8 Q*K^T (any rows) and the U8 x S8 P*V (two rows or more; one-row P*V stays on AOCL-DLP). Quantization and the rest of the flash loop are unchanged. On by default; needs AVX512-VNNI, AVX512-BW and AVX512-VL, otherwise AOCL-DLP runs (with a warning only when the variable is set). Independent of `ZENDNNL_SDPA_BF16_CK`, which covers the BF16 P*V of the QK-only mode; the PV-only mode's BF16 Q*K^T is scaled through alpha and stays on AOCL-DLP. Cached on first read. | `1` | `0`/`false`/`off`/`no` (AOCL-DLP), `1`/`true`/`on`/`yes` (custom kernel); other values keep the default |

---

## HSTU Attention Configuration

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_HSTU_ATTENTION_KERNEL` | Kernel of `hstu_attention_direct`, overriding `hstu_attention_params::kernel`. Without it the tiled kernel runs when it supports the call (causal mask, BF16 Q/K/V, AVX512-BF16/BW/VL) and the reference kernel otherwise; forcing `tiled` on an unsupported call returns `status_t::unimplemented`. Cached on first read. | unset (automatic) | `tiled`, `reference`; other values are ignored with a warning (then `hstu_attention_params::kernel` applies) |
| `ZENDNNL_HSTU_ATTENTION_FAST_SILU` | Opt-in cheaper silu in the Q*K^T epilogue of the tiled kernel, BF16 and dynamic INT8: a degree-3 polynomial exp with `vscalefps` and `rcp14` without a Newton step (relative error of silu at most 1.2e-4, against 5e-5 by default; P is rounded to BF16, or to S8 per row and 256-key block with INT8 P*V, either way). About 6% faster on DLRM v3 full-attention calls (3–5% on delta-mode and short-sequence calls), with the output error against FP64 unchanged; in the INT8 kernel the error is far below the INT8 rounding of P. The default path is bit-identical to running without the variable; the reference kernel ignores it. Cached on first read. | unset (off) | `0`/`false`/`off`/`no`, `1`/`true`/`on`/`yes`; other values are ignored with a warning (off) |
| `ZENDNNL_HSTU_ATTENTION_VARLEN` | Variable-length attention, overriding `hstu_attention_params::is_varlen`. `0` pads every sequence to `max_seq_len` and runs the kernel on the uniform layout, as without variable-length support: the output is unchanged, the cost is the padded rows and the copies (for comparisons). Full mode only; delta mode stays variable-length with a warning. Cached on first read. | unset (`is_varlen`, default `true`) | `0`/`false`/`off`/`no`, `1`/`true`/`on`/`yes`; other values are ignored with a warning (then `is_varlen` applies) |
| `ZENDNNL_HSTU_ATTENTION_INT8_QK` | Overrides `hstu_attention_params::is_qk_quant`: dynamic INT8 Q*K^T in the tiled kernel (Q and K quantized to S8 per token and head inside the kernel, `vpdpbusd` GEMM). Needs BF16 Q/K/V (otherwise dropped with a warning), `qk_dim` <= 133,144 (otherwise dropped with a warning) and AVX512-VNNI (otherwise the tiled kernel runs on BF16, with a warning); the reference kernel ignores it. Independent of `ZENDNNL_HSTU_ATTENTION_INT8_PV`. Cached on first read. | unset (`is_qk_quant` applies) | `0`/`false`/`off`/`no`, `1`/`true`/`on`/`yes`; other values leave `is_qk_quant` in effect |
| `ZENDNNL_HSTU_ATTENTION_INT8_PV` | Overrides `hstu_attention_params::is_pv_quant`: dynamic INT8 P*V in the tiled kernel (V quantized to S8 per 256-key block and channel, P per row and 256-key block, `vpdpbusd` GEMM). Same requirements and fallbacks as `ZENDNNL_HSTU_ATTENTION_INT8_QK`, except the `qk_dim` limit. Cached on first read. | unset (`is_pv_quant` applies) | `0`/`false`/`off`/`no`, `1`/`true`/`on`/`yes`; other values leave `is_pv_quant` in effect |

---

## Logging Configuration

Log levels control the verbosity of messages for different modules.

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_COMMON_LOG_LEVEL` | Log level for common module | `2` (warning) | `0` (disabled), `1` (error), `2` (warning), `3` (info), `4` (verbose) |
| `ZENDNNL_API_LOG_LEVEL` | Log level for API module | `2` (warning) | `0` (disabled), `1` (error), `2` (warning), `3` (info), `4` (verbose) |
| `ZENDNNL_TEST_LOG_LEVEL` | Log level for test module | `2` (warning) | `0` (disabled), `1` (error), `2` (warning), `3` (info), `4` (verbose) |
| `ZENDNNL_PROFILE_LOG_LEVEL` | Log level for profile module | `4` (verbose) | `0` (disabled), `1` (error), `2` (warning), `3` (info), `4` (verbose) |
| `ZENDNNL_DEBUG_LOG_LEVEL` | Log level for debug module | `2` (warning) | `0` (disabled), `1` (error), `2` (warning), `3` (info), `4` (verbose) |

### Log Level Descriptions

| Level | Name | Description |
|-------|------|-------------|
| 0 | `disabled` | Print no messages |
| 1 | `error` | Print only error messages |
| 2 | `warning` | Print error and warning messages |
| 3 | `info` | Print error, warning, and info messages |
| 4 | `verbose` | Print all messages |

---

## Profiler Configuration

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_ENABLE_PROFILER` | Enable/disable the profiler functionality | `false` (disabled) | `1` (enabled), any other value (disabled) |

---

## Diagnostics Configuration

The diagnostics layer gates expensive input-validation paths (null-pointer checks, dimension checks, quantization-parameter checks, and fused-MoE / group-MatMul contract checks) inside the low-overhead operators (`matmul`, `group_matmul`, `flash_sdpa`, `normalization`).  See `src/lowoha_operators/common/operator_instrumentation.hpp` for the gate implementation.

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_DIAGNOSTICS_ENABLE` | Master switch for runtime input validation in low-overhead operators.  When enabled, validators run and emit rich `log_error` diagnostics on contract violations.  When disabled, the gate collapses to a single predicted-taken branch with no validator body executed.  Cached on first read. | `1` (enabled) | `0` (disabled); unset or any other value (enabled) |

### Notes

- The gate covers diagnostic-only checks.  Memory-safety checks that are always required (e.g. empty-vector reject in `group_matmul_direct`, post-op `leading_dim` defaulting in `matmul`) execute unconditionally regardless of this flag.
- Profiling and logging are separate subsystems (`ZENDNNL_ENABLE_PROFILER`, `ZENDNNL_*_LOG_LEVEL`) and are NOT controlled by this flag.
- The value is captured once per process via `static const bool`, so in-process `setenv` / `unsetenv` calls after the first validator call have no effect.  Tests that need to deterministically toggle the gate must do so via `fork()` / `execve()` subprocesses (see `[15] TestDispatcherActiveTotalNegative` in `gtests/group_matmul/fused_moe/test_fused_moe.cpp`).

---

## LRU Cache Configuration

The LRU (Least Recently Used) cache stores precomputed data for reuse.

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_LRU_CACHE_CAPACITY` | Maximum number of entries in the LRU cache | `UINT_MAX` (unlimited) | Positive integer (uint32_t) |

### Eviction semantics by cache layer

Four distinct caches are involved in MatMul / Group MatMul execution
with different eviction semantics — readers of `ZENDNNL_LRU_CACHE_CAPACITY`
should know which one is actually governed:

- **AOCL DLP reorder cache** — backed by the generic `lru_cache_t`
  layer.  Default capacity `UINT32_MAX`, so populated entries are
  held for process lifetime and any prior prepack / warm-up
  guarantee holds.  Deployments that lower the capacity via
  `ZENDNNL_LRU_CACHE_CAPACITY` (env or matmul-config JSON) shrink
  the upper bound; prepacked entries can then be evicted under
  steady-state pressure and a subsequent runtime reorder spike
  can return.  Set the LRU capacity high enough to fit your full
  warm working set if you need eager prepack to hold.

- **Custom-kernel pack arena** — process-wide singleton that
  intentionally ignores the LRU eviction path
  (`pack.cpp::clear_custom_kernel_pack_cache_bf16` is the only entry
  that frees entries; regular operation never evicts).
  `ZENDNNL_LRU_CACHE_CAPACITY` does NOT govern this cache.
  Populated entries are held for process lifetime regardless of
  the LRU knob.

- **Native BRGEMM (ALGO 10) layout cache** — process-wide; holds at most
  `ZENDNNL_LRU_CACHE_CAPACITY` layouts by refusing new entries, never by
  evicting: once full, a call on another weight packs it for that call
  only.  When the bytes of a cached weight change (a weight refilled or
  reloaded at the same address), its next call packs it again and its old
  layout stays allocated until `clear_matmul_native_weight_caches()`: one
  whole layout per change (32 MiB for a 4096 x 4096 bf16 weight), counted
  toward `ZENDNNL_LRU_CACHE_CAPACITY` with the cached layouts.  With the
  default capacity the memory then grows with every such change until the
  caches are cleared.  With a finite capacity, once the layouts held reach
  it, every weight that is not cached yet, the changed one included, is
  packed on every call until the next clear; only the verbose pack line
  shows it.  `clear_matmul_native_weight_caches()` frees the cached and the
  dropped layouts.  Each calling thread also keeps a memo of the layouts it
  was served (about 50 KB, freed at thread exit); the memo holds no layout
  memory, and its entries stop serving after a cache clear.

- **W8A8 ALGO-4 complete-tensor cache** — a separate `lru_cache_t` holding
  complete W13/W2 packs. It honors `ZENDNNL_LRU_CACHE_CAPACITY` exactly;
  in-flight calls retain evicted entries through `shared_ptr`. Set
  `params[i].weight_cache_type=0` or the process-wide weight-cache mode to
  `0` to make ALGO 4 decline without publishing an entry.

For memory-bounded deployments (multi-tenant, container quotas):

- `ZENDNNL_MATMUL_WEIGHT_CACHE=0` disables AOCL lazy/eager population.
  The grouped custom-kernel path uses per-call caller-owned packs instead of
  publishing persistent entries, and W8A8 grouped-MoE ALGO 4 declines before
  writes; BF16 and caller-prequantized S8 may continue through generic
  execution.
  Entries created before a runtime mode change are not implicitly reclaimed.
  From an outside-OMP quiescent window,
  `clear_grp_matmul_weight_caches()` releases the weight caches and
  `clear_fused_moe_scratch()` independently releases fused-MoE scratch.
  Under weight-cache mode 2, the weight-cache clear removes the BF16/FP16
  in-place image records but cannot restore the caller's raw weight bytes,
  so weights a call packed in place must be reinitialized before reuse.
  Do not clear and then reuse an already-packed buffer; release/replace it, or
  reload the original raw weights before the next call. For a `1 → 0 → 1`
  transition in which weights may change, quiesce, clear, replace or reload
  the weights, and only then re-enable caching.
  W8A8 weight scales are copied from the current call and may change without
  flushing the packed-weight generation.
- Under the native BRGEMM (ALGO 10), clear the native caches after
  replacing or reloading weights: from a quiescent window,
  `clear_matmul_native_weight_caches()` (or `clear_matmul_weight_caches()`)
  frees the layouts of the weights whose bytes changed, which the native
  layout cache otherwise holds until then.
- `ZENDNNL_LRU_CACHE_CAPACITY=<N>` caps the AOCL reorder cache at
  `N` entries.  Smaller `N` yields lower steady-state memory but
  reintroduces reorder spikes when prepacked entries are evicted
  under capacity pressure.  It also caps the layouts the native BRGEMM
  layout cache holds, dropped ones included.  No effect on the
  custom-kernel pack arena.

---

## Testing/Debug Configuration

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `AI_GTEST_DEBUG` | Enable debug print output in AI gtests | `false` (disabled) | `1` or `true` (enabled) |

---

## Usage Examples

### Basic Configuration

```bash
# Use AOCL DLP blocked algorithm with weight caching
export ZENDNNL_MATMUL_ALGO=1
export ZENDNNL_MATMUL_WEIGHT_CACHE=1

# Enable verbose logging for debugging
export ZENDNNL_COMMON_LOG_LEVEL=4
export ZENDNNL_API_LOG_LEVEL=4
```

### Auto-Tuner Configuration

```bash
# Enable auto-tuner with custom iteration counts
export ZENDNNL_MATMUL_ALGO=auto
export ZENDNNL_MATMUL_SKIP_ITER=3
export ZENDNNL_MATMUL_EVAL_ITER=5
```

### Using Configuration File

```bash
# Use a JSON configuration file instead of individual env vars
export ZENDNNL_CONFIG_FILE=/path/to/zendnnl_config.json
```

### Performance Logs

```bash
# Enable profiler for performance analysis
export ZENDNNL_ENABLE_PROFILER=1
export ZENDNNL_PROFILE_LOG_LEVEL=4
```

---

## Configuration Priority

1. **JSON Configuration File**: If `ZENDNNL_CONFIG_FILE` is set and the file is valid, configuration is loaded from the file.
2. **Environment Variables**: If no config file is specified, individual environment variables are used.
3. **Default Values**: Built-in defaults are used for any unspecified settings.

---

## See Also

- `config/zendnnl_user_config.json` - Example JSON configuration file
- `zendnnl/src/common/config_manager.cpp` - Configuration manager implementation
- `zendnnl/src/common/op_config.cpp` - MatMul configuration implementation

