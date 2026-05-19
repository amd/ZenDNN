# `group_matmul/` gtests

Self-contained gtest surface for the `group_matmul` operator family
(basic group_matmul, fused-MoE, scheduling-ALGO matrix, quantization,
prepack module).  Every group-matmul-specific gtest artifact lives in
this folder; the only outside-of-folder coupling is the
`#include "group_matmul/group_matmul_test_helpers.hpp"` in
`gtests/gtest_main.cpp` which keeps the existing `quant_matmul_test`
fill loop compiling.

This README is the maintainer's first stop: it answers "what's tested,
what isn't, where do I add a new test, what env knobs do these tests
consume, and which bug class does each test lock down".

---

## 1. Overview

| Question | Answer |
|---|---|
| What gets tested? | The public `group_matmul_direct(...)` dispatcher and everything reachable from it: scheduling ALGOs 1..5, custom BF16 microkernel, fused-MoE (Op1 + activation + Op2), gated activations, MoE post-op (weighted reduce), per-expert active/total contract, internal-alloc patterns, prepack module, quantization (WOQ + INT8 + dynamic). |
| What isn't tested here? | Operator-agnostic infrastructure tests (`test_matmul.cpp`, `test_batchmatmul.cpp`, etc.) live at the parent `zendnnl/gtests/` level. The AI-gtests framework (`ai_gtests/`) is its own subsystem. |
| Single binary? | Yes. All test files in this folder compile into the same `gtests` executable produced by the parent CMakeLists. Filter via `--gtest_filter=*Prepack*`, `--gtest_filter=*FusedMoE*`, etc. |
| Helpers reuse policy? | One sibling header (`moe_test_utils.hpp`) for cross-file helpers; one helper TU (`group_matmul_test_helpers.{hpp,cpp}`) for the dispatch shim + quant fixture. File-local helpers stay in anonymous namespaces inside their owning `.cpp`. |

---

## 2. File layout

```
zendnnl/gtests/group_matmul/
  CMakeLists.txt                    target_sources(...) into the parent gtests target
  README.md                         this file
  moe_test_utils.hpp                shared test helpers (data fills, TypedBuffers,
                                    GemmVecs, ref activations, tolerances, env-var
                                    RAII guards, fused-MoE 2-call reference,
                                    verify_per_expert_2d)
  group_matmul_test_helpers.hpp     declarations: GroupQuantMatmulType,
                                    quant_matmul_test extern, group_matmul_kernel_test
                                    proto, PrintTo(GroupQuantMatmulType, ...)
  group_matmul_test_helpers.cpp     implementations of the above
  test_basic.cpp                    [2] TestGroupMatmul, [3] TestGatedAct, [4] TestMoEPostop
  test_fused_moe.cpp                [5]/[5b]/[5c] TestFusedMoE family,
                                    [6] TestGroupMatmulCombined,
                                    [10] TestFusedMoEDstSplit,
                                    [11] TestFusedMoEActiveMatmul,
                                    [12] TestFusedMoEWarmPackPipeline,
                                    [13] TestFusedMoEArchGrid,
                                    [14] TestFusedMoEActiveTotalEdge,
                                    [15] TestDispatcherActiveTotalNegative,
                                    [16] TestFusedMoEQuant (fused-MoE quant suite:
                                         TestFusedMoEQuantWOQ.BothPasses (48 tuples)
                                         + TestFusedMoEQuantDynINT8.BothPasses
                                         (24 tuples, M >= 16 only — AOCL BF16->S8
                                         reorder floor).  Shared base
                                         TestFusedMoEQuantBase holds setup/teardown;
                                         each scheme gets its own subclass + grid
                                         + INSTANTIATE_TEST_SUITE_P so no test is
                                         GTEST_SKIP()'d at run time)
  test_algos.cpp                    [7] TestFusedMoEAlgos,
                                    [7b] TestFusedMoEAlgoCustom,
                                    [8] TestGroupMatmulAlgoCustom,
                                    [8b] TestGroupMatmulAutoSelectAlgo
  test_quant.cpp                    [9] TestGroupMatmulQuant
  test_prepack.cpp                  [16] TestFusedMoECacheStress,
                                    [17] TestFusedMoEPointerChurn,
                                    [18] TestPrepackPerAlgoFunctions,
                                    [19] TestPrepackFusedMoEEndToEnd,
                                    [20] TestPrepackKDownSynthesisAllActs,
                                    [21] TestPrepackResultInvariance,
                                    [22] TestPrepackClearCacheDirect,
                                    [23] TestPrepackAoclDlpFullWeight,
                                    [24] TestPrepackVariableNExperts,
                                    [25] TestPrepackStressManyExperts,
                                    [26] TestPrepackEnvBucketA,
                                    [27] TestPrepackEnvBucketB,
                                    [28] TestPrepackEnvInteractionMatrix,
                                    [29] TestPrepackCrossWarmRegimes,
                                    [30] TestPrepackFingerprintInvariance,
                                    [32] TestPrepackCkGateSymmetry
```

LOC summary (current tree, `wc -l`):

| File | Lines |
|---|---|
| `moe_test_utils.hpp`            |  ~647 |
| `group_matmul_test_helpers.hpp` |  ~120 |
| `group_matmul_test_helpers.cpp` |  ~244 |
| `test_basic.cpp`                |  ~688 |
| `test_fused_moe.cpp`            | ~1794 |
| `test_algos.cpp`                |  ~935 |
| `test_quant.cpp`                | ~1081 |
| `test_prepack.cpp`              | ~2593 |
| **Total**                       | **~8102 LOC** |

---

## 3. Tested API surface

### 3.1 Public dispatcher

```cpp
zendnnl::lowoha::matmul::group_matmul_direct(
    layout, transA, transB, M, N, K, alpha,
    src, lda, weight, ldb, bias, beta, dst, ldc,
    is_weights_const, params,
    moe_postop          /* optional */,
    gated_act           /* optional */,
    fused_moe           /* optional */)
```

This is the single public entry point that every test eventually calls.
Lives in `zendnnl/src/lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp`.

### 3.2 Param structs

| Type | Drives |
|---|---|
| `matmul_params`                       | per-expert dtype + (post-PR) `active_matmul / total_matmul` prepack-extras contract |
| `grp_matmul_gated_act_params`         | gated activation kind (`silu_and_mul`, `gelu_and_mul`, `swiglu_oai_mul`, `none`) |
| `grp_matmul_fused_moe_params`         | Op2 down-projection weights + bias + N_down/ldb_down/dst_down/ldc_down + **Op2 weight quant** (`down_scale`, `down_zp`) |
| `group_matmul_moe_postop_params`      | weighted-reduce post-op (e.g. `topk`-routed sum after Op2) |
| `quant_params` (in `matmul_params`)   | per-tensor scale/zp for WOQ / INT8 paths on Op1; **the same scheme** (dynamic_quant flag, dtypes.compute, src_scale.dims) is inherited by Op2's internal params_down |
| `down_scale` / `down_zp` (in `grp_matmul_fused_moe_params`) | per-expert **weight** scale + zero-point for `down_weight[i]` (the only Op2-specific quant artefact — every other knob is inherited from `params[i]`).  Empty ⇒ Op2 weight un-quantized (default). |

### 3.3 Prepack module

```cpp
zendnnl::lowoha::matmul::group_matmul_prepack::
    prepack_for_algo_1(p) ... prepack_for_algo_5(p);
zendnnl::lowoha::matmul::group_matmul_prepack::aocl_dlp::
    warm_pack_all_aocl_dlp_experts(...);
zendnnl::lowoha::matmul::group_matmul_prepack::aocl_dlp::
    warm_pack_all_aocl_dlp_experts_n_tile(...);
zendnnl::lowoha::matmul::group_matmul_prepack::custom_kernel::
    warm_pack_all_custom_kernel_experts(...);
zendnnl::lowoha::matmul::group_matmul_prepack::
    clear_fingerprint_cache_for_test();
```

Lives in `zendnnl/src/lowoha_operators/matmul/group_matmul/prepack/`.
Tests in `test_prepack.cpp` cover the public API + observable cache state
via the custom-kernel `PackProbeStats` (the AOCL DLP LRU is private and
isn't directly inspectable; per-tile correctness is asserted via
`total_attempted` math instead).

### 3.4 Test-local dispatch shim

```cpp
group_matmul_kernel_test(
    inputs, weights, biases, outputs,
    algo, alpha, beta, moe_postop, gated_act);
```

Defined in `group_matmul_test_helpers.{hpp,cpp}`.  Wraps vectors of
`tensor_t` into `group_matmul_direct` for the basic + algo + quant suites
(extracts dtype + transpose + ldX from tensor metadata; auto-handles
WOQ / INT8 / dynamic-quant params).  Mirrors the design of
`matmul_kernel_test` in `gtest_utils.{hpp,cpp}`.

---

## 4. Tested feature axes (which file owns each)

### 4.1 Datatypes
- F32 / F32                         — `test_basic.cpp` ([2])
- BF16 / F32                        — `test_basic.cpp`
- BF16 / BF16                       — `test_basic.cpp`
- BF16-src x F32-dst (mixed)        — `test_algos.cpp` ([7], `FusedAlgoTestParam::mixed_prec`)
- WOQ S4 / U4                       — `test_quant.cpp`
- INT8 (sym/asym, per-tensor/channel/group/token) — `test_quant.cpp`
- INT8 dynamic-quant                — `test_quant.cpp`

### 4.2 Activations
- `none`                            — every section
- `silu_and_mul`                    — `test_basic.cpp`, `test_fused_moe.cpp`, `test_prepack.cpp` (sweep in [20])
- `gelu_and_mul`                    — same
- `swiglu_oai_mul` (interleaved)    — same
- Unfused `silu` / `gelu`           — out-of-scope (those run as a separate post-pass; not part of `grp_matmul_gated_act_t`)

### 4.3 Internal-alloc patterns (Op1 internal x Op2 internal)
4-pattern truth table:
| dst[] | dst_down | Op1 internal? | Op2 internal? |
|---|---|---|---|
| caller   | caller   | no  | no  |
| nullptr  | caller   | yes | no  |
| caller   | nullptr  | no  | yes (in-place src reuse) |
| nullptr  | nullptr  | yes | yes |
Covered by `[10] TestFusedMoEDstSplit` in `test_fused_moe.cpp`.

### 4.4 Active / total contract (prepack-extras tail)
The framework signals "I'm sending all `total_matmul` weights but only
the first `active_matmul` are firing" via `params[0].active_matmul` /
`params[0].total_matmul`.  The prepack module warms `[0, total)` while
the dispatcher computes only `[0, active)`.

Covered by `[11] TestFusedMoEActiveMatmul`, `[14] TestFusedMoEActiveTotalEdge`,
`[19] TestPrepackFusedMoEEndToEnd` (Pass 2 K_down sizing regression),
and the entire prepack section.

### 4.5 Scheduling ALGOs (1..5)
- 1 `sequential_experts`            — covered by `[7]` and `[8]` in `test_algos.cpp`
- 2 `flat_m_tile`                   — covered by `[7]` and `[8]`
- 3 `flat_n_tile`                   — covered by `[7]`, `[8]`, `[18]`-`[28]` (the prepack module's hot path is ALGO 3)
- 4 `parallel_multilevel`           — covered by `[7]` and `[8]`
- 5 `parallel_per_expert`           — covered by `[7]` and `[8]`

The env-knob matrix in `test_prepack.cpp` `[26]`-`[28]` runs each ALGO
in its own subprocess for full coverage of the `static const`-cached
env-getter paths.

### 4.6 Custom BF16 microkernel
- Pack/unpack semantics             — `test_prepack.cpp` `[18]`, `[22]` (clear/re-fire)
- NR=32 vs NR=64                    — `test_prepack.cpp` `[26]` (env matrix)
- Fused swiglu_oai epilogue         — `test_fused_moe.cpp` `[7b]`, `test_prepack.cpp`
- Wide-swiglu correctness guard     — `test_prepack.cpp` `[28]` `algo3_wide_layout`

### 4.7 Quantization

Single-call (non-fused MoE) coverage in `test_quant.cpp` (`[9] TestGroupMatmulQuant`):

| Variant | Test |
|---|---|
| `WOQ_BF16_S4`                            | `test_quant.cpp` |
| `WOQ_BF16_U4`                            | `test_quant.cpp` |
| `INT8` (per-tensor / per-channel)        | `test_quant.cpp` |
| `INT8_SYM_QUANT_PER_GROUP_BF16`/`F32`    | `test_quant.cpp` |
| `INT8_SYM_QUANT_PER_TOKEN_BF16`/`F32`    | `test_quant.cpp` |
| `INT8_DYNAMIC_GEMM_BF16`/`F32`           | `test_quant.cpp` |

Fused MoE (Op1 + activation + Op2) coverage in `test_fused_moe.cpp`
(`[16] TestFusedMoEQuant`).  Layered after `TestGroupMatmulQuant`
but with one deliberate twist: each scheme owns its own fixture
subclass and parameter grid, so every enumerated tuple is one the
AOCL kernel layer is contractually obligated to handle.  No
runtime `GTEST_SKIP()` is needed — the previous `M < 16` skips
have been pushed up to the parameter source.

| Fixture | Param grid | Tuples | What it covers |
|---|---|---|---|
| `TestFusedMoEQuantBase` | — (abstract) | — | shared setup/teardown (shape config, ALGO 1 pin via `AlgoEnvGuard`) |
| `TestFusedMoEQuantWOQ` | 3 acts × 2 dims × **M ∈ {1, 4, 16, 32}** × 2 num_ops | **48** | AOCL DLP's WOQ fast path is M-agnostic, so all M values run. |
| `TestFusedMoEQuantDynINT8` | 3 acts × 2 dims × **M ∈ {16, 32}** × 2 num_ops | **24** | M trimmed at the source — AOCL BF16→S8 reorder rejects per-token `{M, 1}` for M < 16 (same constraint as `test_quant.cpp::INT8_DYNAMIC_GEMM_BF16`). |

| Variant | Op1 quant | Op2 quant carrier | Test (in `[16]`) |
|---|---|---|---|
| WOQ S4 on both passes | `params[i].quant_params.wei_scale` | `fused.down_scale[i]` | `TestFusedMoEQuantWOQ.BothPasses` |
| Dynamic INT8 on both passes (runtime BF16→S8 reorder) | `params[i].dynamic_quant=true` + `params[i].quant_params.src_scale` (drives both passes) | `fused.down_scale[i]` (down_weight only — every other knob inherited) | `TestFusedMoEQuantDynINT8.BothPasses` |

Both fused-MoE quant tests carry a non-zero sanity guard (sum of
absolute values of the reference + test outputs MUST exceed
`1e-3`) at the bottom of the body — this catches the failure mode
where AOCL DLP rejects an unsupported quant combo and leaves the
dst buffer at the post-alloc all-zero state.  Without the guard a
silent kernel bail-out on **both** routes would trivially pass the
per-element comparison on `0 == 0`.

### 4.8 Prepack module
| Surface | Test |
|---|---|
| Per-ALGO functions ([1..5])      | `test_prepack.cpp` `[18]` |
| Backend warmer (custom-kernel)   | `[16]`, `[17]` (pointer churn) |
| Backend warmer (AOCL DLP, full)  | `[23]` |
| Backend warmer (AOCL DLP, per-tile) | `[18]` AoclDlpNTile* |
| K_down synthesis (all gated_acts)| `[20]` parameterised over 4 acts |
| Fingerprint cache + `clear_fingerprint_cache_for_test()` | `[22]` |
| `clear_custom_kernel_pack_cache()` | `[22]` (regression test for the size_t-underflow BLOCKER) |
| Result invariance (cold->warm, ALGO 1<->3) | `[21]` |
| Variable-N per-expert            | `[24]` |
| Stress (E=64 multi-iter, E=256 boundary) | `[25]` |
| Env-knob matrix (subprocess-isolated) | `[26]` Bucket A perf-critical, `[27]` Bucket B tuning, `[28]` interaction matrix |

---

## 5. Env vars consumed

All knobs the prepack / dispatch / kernel layers read are listed below.
`gtests/group_matmul/test_prepack.cpp` `[26]`-`[28]` exercises every
*cached* knob in a subprocess (because `static const` IIFEs cache on
first read for the process lifetime, in-process `setenv` is a no-op).

| Env var | Default | Cached? | What it gates |
|---|---|---|---|
| `ZENDNNL_GRP_MATMUL_ALGO` | auto | NO (re-read each call) | Force scheduling ALGO 1..5; tests use `AlgoEnvGuard` for in-process flips |
| `ZENDNNL_GRP_MATMUL_PREPACK` | ON | yes | Master prepack switch (PR-443) |
| `ZENDNNL_GRP_MATMUL_CROSS_WARM` | ON | yes | Opportunistic CK-aware cross-regime warm in `prepack/prepack.cpp::cross_warm` (eliminates decode-first-call spike when prompt-only warmup runs) |
| `ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE` | ON | yes | Pin n_thr to a num_threads-only formula -> AOCL cache key stability |
| `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL` | OFF | yes | BF16 microkernel master switch |
| `ZENDNNL_GRP_MATMUL_FUSED_MOE_TIGHT` | 1 | yes | Force tight-dst layout for fused-MoE Op1 |
| `ZENDNNL_GRP_MATMUL_N_TILE_FUSED_ACT` | ON | yes | Fuse swiglu_oai into ALGO 3 epilogue |
| `ZENDNNL_GRP_MATMUL_N_ROUNDS` | 0 (auto) | yes | ALGO 3 round-mode (single/multi/balanced) |
| `ZENDNNL_GRP_MATMUL_N_ORDER` | 0 (auto) | yes | Expert ordering within rounds |
| `ZENDNNL_GRP_MATMUL_AOCL_TARGET_SLOTS` | 16 | yes | Divisor for stable n_thr |
| `ZENDNNL_GRP_MATMUL_AOCL_BLIS_NC` | 128 | yes | BLIS-bf16 inner-N block (narrow-N density estimator) |
| `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_NR` | 0 (auto -> 32) | yes | Pack/microkernel NR (32 or 64) |
| `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_SUBTILE_PER_EXPERT` | OFF | yes | Per-expert L2-friendly subtile_cols |
| `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_N_TILE` | 0 (off) | yes | Override outer N-tile minimum |
| `ZENDNNL_MATMUL_WEIGHT_CACHE` | 1 | runtime-mutable via `set_weight_cache(...)` | AOCL DLP cache; 0 short-circuits the prepack AOCL warmer |
| `ZENDNNL_LRU_CACHE_CAPACITY` | UINT32_MAX | yes (matmul_config) | AOCL DLP reorder LRU capacity.  Caps the prepack guarantee — populated entries can be evicted under pressure.  No effect on the custom-kernel pack arena (intentionally eviction-immune). |
| `ZENDNNL_DIAGNOSTICS_ENABLE` | OFF | yes | Wraps `validate_group_matmul_direct_inputs` (Phase B-G + log_error).  Used by `[15]` subprocess test to exercise the diagnostic-mode reject paths for `am > M.size()` / `tm < am`. |

Cached = read once via `static const v = []() { getenv(...); }()` at first
call.  Production deployments set these once at process start; tests
override them either with `AlgoEnvGuard` / `EnvVarGuard` (uncached only)
or via the subprocess pattern in `test_prepack.cpp`.

---

## 6. How to run

```bash
# Build (from the build directory).
make gtests -j$(nproc)

# Run only this folder's tests:
./gtests --gtest_filter='*GroupMatmul*:*FusedMoE*:*Prepack*'

# Common targeted filters:
./gtests --gtest_filter='*Prepack*'                # all prepack module tests
./gtests --gtest_filter='*FusedMoEAlgos*'          # ALGO sweep
./gtests --gtest_filter='*GroupMatmulQuant*'       # quantization
./gtests --gtest_filter='*EnvBucketA*'             # perf-critical env knobs
./gtests --gtest_filter='*EnvInteraction*'         # production env combos

# Single test by parameterised name:
./gtests --gtest_filter='*KDownSynthesis*/act_silu_and_mul_E8_N1_64'
```

Default seed is from `time(nullptr)`; override with the binary's CLI:
```bash
./gtests --seed=12345
```

---

## 7. How to add a new test

### Decision tree

```
Is it a basic group_matmul correctness test (no MoE / no fused / no quant)?
  yes -> test_basic.cpp

Is it a fused-MoE end-to-end correctness test (any (act, fused, moe_postop) combo)?
  yes -> test_fused_moe.cpp

Is it sweeping the scheduling-ALGO matrix or custom-kernel env knobs in-process?
  yes -> test_algos.cpp

Is it WOQ / INT8 / dynamic-quant?
  yes -> test_quant.cpp

Is it about the prepack module (per-ALGO, K_down, fingerprint cache,
per-tile vs full-weight, env-knob subprocess matrix, stress)?
  yes -> test_prepack.cpp
```

### Helpers

- **Need a new shared helper used by 2+ test files?** -> add to
  `moe_test_utils.hpp` (header-only, all definitions inline).
- **Need a quant-fixture-specific helper?** -> add to
  `group_matmul_test_helpers.hpp`/`cpp`.
- **Single-file helper?** -> file-local anonymous namespace inside the
  owning `.cpp`.

### Class / parameter naming

Existing test class names and gtest filters are stable; do not rename
`TestGroupMatmul`, `TestFusedMoE*`, `TestPrepack*`, etc.  CI filters
and external tooling depend on them.

### Subprocess pattern (env knobs)

When you need to test a `static const`-cached env knob, use the
`EnvCase` helper in `test_prepack.cpp`:

```cpp
return env_case("my_case_label",
                /*envs=*/{{"ZENDNNL_GRP_MATMUL_FOO", "1"}});
```

Then add to one of `make_bucket_{a,b}_cases()` /
`make_interaction_matrix_cases()`.  The `run_env_matrix_subprocess_test`
driver does the `EXPECT_EXIT` + threadsafe death-test plumbing for
you.

---

## 8. Bug-class regression coverage

Every regression caught during the PR #443 review now has a locked-in
test.  Re-reads of the test logic should treat these as load-bearing
invariants:

| ID | Bug class (severity) | Test |
|---|---|---|
| B1 | `K_down_synth` derived half-K from `swiglu_oai_mul` only (BLOCKER) | `[20]` TestPrepackKDownSynthesisAllActs (24 cases) |
| B2 | `scratch.K_down` sized to `num_ops` not `N.size()` (BLOCKER) | `[19]` TestPrepackFusedMoEEndToEnd |
| B3 | `scratch.params_down` not propagating active/total (BLOCKER) | `[19]` |
| B4 | `clear_custom_kernel_pack_cache()` size_t underflow (BLOCKER) | `[22]` CustomKernelCacheClearEvictsEntries |
| B5 | AOCL warmer ignored `is_weights_const` (MEDIUM) | `[18]` AoclDlpNTileSkipsNonConstExperts (per-tile), `[23]` SkipsNonConstExperts (full-weight) |
| B6 | AOCL warmer ignored `WEIGHT_CACHE=0` (HIGH) | `[28]` algo3_weight_cache_off_custom_on (subprocess) |
| B7 | Per-call warm overhead (HIGH) | `[12]` TestFusedMoEWarmPackPipeline + `[25]` E64MultiIter |
| B8 | Per-tile cache key mismatch with runtime (HIGH) | `[18]` count assertions + `[21]` Algo1VsAlgo3 |
| B9 | Test pollution via fingerprint cache | `[22]` FingerprintClearEnablesPrepackReFire |
| B10 | `internal_alloc` Op1/Op2 conflation | `[10]` TestFusedMoEDstSplit |
| B11 | Per-expert K_down clamping for narrow N | `[24]` MixedNAcrossExperts |
| B12 | Deferred decode-time prepack spike (vLLM prompt-only warmup leaves regimes 2/3 cold) | `[29]` TestPrepackCrossWarmRegimes (9 cases pinning the 5-quadrant decision tree) |
| B13 | Order-dependent fingerprint hash flips on active-set rotation; XOR-collision false-hits (Copilot review #1) | `[30]` TestPrepackFingerprintInvariance (3 cases: permutation, membership change, pool-size change) |
| B14 | Dispatcher contract validator missed `active_matmul > M.size()` and `total_matmul < active_matmul` reject paths (PR-443 review G2) | `[15]` TestDispatcherActiveTotalNegative (3 cases incl. `RejectsWeightShorterThanTotalMatmul` for B15) |
| B15 | Silent prepack-extras truncation when weight-side metadata < total_matmul (Copilot review #2) | `[15]` TestDispatcherActiveTotalNegative.RejectsWeightShorterThanTotalMatmul |
| B16 | Phase-F validator under-restricted `ldb_down` for `act == none` (used `N/2` unconditionally) | `[15]` TestDispatcherActiveTotalNegative.RejectsFusedMoEActNoneLdbBelowN |
| B17 | Prepack `ck_eligible` only checked dtype trio + CK env; runtime `prepare_for_call` refuses on 8 more grounds (act, act_dtype, bias_dtype, pack_nr divides N, etc.).  Asymmetric verdict caused ~1.5 GB wasted CK arena + ~12 k lazy AOCL DLP reorders at first decode call when the runtime falls back. | `[32]` TestPrepackCkGateSymmetry (9 cases pinning each new gate's accept/refuse + a fingerprint sensitivity case) |

---

## 9. Subprocess pattern explanation

`test_prepack.cpp` `[26]`-`[28]` use gtest's `EXPECT_EXIT` with
`testing::FLAGS_gtest_death_test_style = "threadsafe"` to run each
test case in a *fresh process*.  This is the only way to exercise the
~13 `static const`-cached env-getter functions in
`group_matmul_parallel_common.hpp` at values other than what the
running binary first read.

How it works:

1. Parent `setenv("ZENDNNL_GRP_MATMUL_FOO", "1", 1)`.
2. `EXPECT_EXIT({ child_body }, ExitedWithCode(0), "")`.
3. gtest `fork()` + `execve()`s the binary -> child gets a fresh
   process image, every `static const` is uninitialised.
4. Child inherits the parent's environment via execve.  Its first read
   of `get_grp_matmul_foo()` hits the IIFE, observes "1", caches.
5. Child runs the body (fused-MoE call, cache probe, etc.), exits with
   `::testing::Test::HasFailure() ? 1 : 0`.
6. Parent's `EXPECT_EXIT` matches the exit code.

Per-case overhead is ~150-450 ms — fork+exec is the dominant cost
(varies with binary size and child startup); the matmul work in the
child is <50 ms at the tiny shapes used.  The ~50-case env matrix
[26]-[28] completes in ~7-12 s; the 3-case `[15]` dispatcher
negative suite adds ~1 s; the 9-case `[29]` cross-warm suite uses
in-process state inspection (`test_api::LastInvocationStats`) so it
runs without the subprocess overhead.

---

## 10. Maintenance notes

- **Don't grow `gtest_utils.{hpp,cpp}`** with new group_matmul-specific
  symbols.  If you find yourself adding "yet another group-matmul
  helper" to those files, lift it into
  `group_matmul_test_helpers.{hpp,cpp}` instead.
- **Keep test files under 2200 lines.**  If a section grows past that,
  consider extracting it into its own file in this directory (e.g.
  if the prepack env-matrix grows large, split `[26]`-`[28]` into
  `test_prepack_env.cpp`).
- **Don't introduce a hard dependency from `gtest_main.cpp` on more
  than one symbol from this folder.**  The single `quant_matmul_test`
  fill loop is the only acceptable cross-folder reach.
