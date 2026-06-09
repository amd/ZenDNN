# `reorder/` gtests

Self-contained gtest surface for the `reorder` operator family (legacy regular reorder + matmul, and the LOWOHA quantize / dequantize / type-convert kernels).  The reorder tests were split out of the monolithic `gtests/test_reorder.cpp` into this folder, mirroring the `group_matmul/` layout.  Every reorder-specific gtest artifact lives here — including the `ReorderType` param struct and the reorder kernel/compare/shape helpers, which were lifted out of `gtest_utils.{hpp,cpp}` into `reorder_test_helpers.{hpp,cpp}` (see §4).  The only outside-of-folder coupling is `gtest_main.cpp`, which `#include`s `reorder/reorder_test_helpers.hpp` to *define* and fill the `reorder_test` global.  `ReorderInput` and `read_reorder_inputs` stay in `gtest_utils.{hpp,cpp}` because `ReorderInput` is a member of the shared `CLIParams`.

This README is the maintainer's first stop: it answers "what's tested, how the files are split, where do I add a new test, and why is it laid out this way".

---

## 1. Overview

| Question | Answer |
|---|---|
| What gets tested? | The reorder surface reachable from the reorder helper shims in `reorder_test_helpers.{hpp,cpp}`: the legacy regular reorder + matmul path (`reorder_kernel_test` + the shared `matmul_kernel_test`) and the LOWOHA reorder path (`lowoha_reorder_kernel_test`) covering static quant/dequant, dynamic quant, strided sources, and floating-point type conversion (FP32/BF16/F16, incl. FP16-typed scale buffers). |
| What isn't tested here? | Operator-agnostic infrastructure and other operators (`test_matmul.cpp`, `test_batchmatmul.cpp`, etc.) live at the parent `zendnnl/gtests/` level. Shared RNG / tensor factory stay in `gtest_utils.{hpp,cpp}`; the reorder param struct, kernel shims, and comparators were lifted into `reorder_test_helpers.{hpp,cpp}` (see §4). |
| Single binary? | Yes. All files in this folder compile into the same `gtests` executable produced by the parent CMakeLists (via `add_subdirectory(reorder)` + `target_sources`). Filter via `--gtest_filter=Reorder/TestReorder.*`. |
| One fixture or many? | **One.** Every reorder `TEST_P` shares the single `TestReorder` fixture (defined in `reorder_test_common.hpp`) and the `Reorder` instantiation prefix. Splitting `TEST_P` for one fixture across translation units is fully supported by GoogleTest as long as exactly one `INSTANTIATE_TEST_SUITE_P` exists (it lives in `test_reorder_regular.cpp`). This keeps every existing `--gtest_filter=Reorder/TestReorder.*` invocation working unchanged. |

---

## 2. File layout

```
zendnnl/gtests/reorder/
  CMakeLists.txt                  target_sources(...) into the parent gtests target
  README.md                       this file
  reorder_test_common.hpp         shared TestReorder fixture (SetUp/TearDown,
                                  LOWOHA-only masking) included by every test file
  reorder_test_helpers.{hpp,cpp}  reorder-only helpers lifted out of
                                  gtest_utils.{hpp,cpp}: ReorderType + ctor,
                                  reorder_test extern, PrintTo, the reorder
                                  kernel shims, and the LOWOHA compare/shape/log
                                  helpers (mirrors group_matmul_test_helpers)
  test_reorder_regular.cpp        legacy regular reorder + matmul tests
                                  (always SKIPPED on the LOWOHA-only path)
                                  + the single INSTANTIATE_TEST_SUITE_P
  test_static_quant_dequant.cpp   LOWOHA static quant/dequant round-trips
  test_dynamic_quant.cpp          LOWOHA dynamic-quant round-trips
  test_strided_cases.cpp          LOWOHA strided quant/conversion round-trips
  test_conversion.cpp             LOWOHA float<->float type-conversion round-trips
```

---

## 3. Test inventory (which file owns each `TEST_P`)

All tests share the `TestReorder` fixture and the `Reorder` prefix, so each filters as `Reorder/TestReorder.<NAME>/*`.

### 3.1 `test_reorder_regular.cpp` — legacy regular reorder + matmul

These begin with `if (use_LOWOHA) GTEST_SKIP();` and are therefore **always SKIPPED** on the LOWOHA-only suite (`use_LOWOHA` is always true unless the user passes a non-`true`/`1` `--lowoha`, in which case `SetUp()` masks the whole fixture).  Kept for reference / reactivation.  This TU also hosts the lone `INSTANTIATE_TEST_SUITE_P(Reorder, TestReorder, ::testing::ValuesIn(reorder_test))`.

| Test | What it does |
|---|---|
| `F32_F32`, `BF16_F32`, `BF16_BF16` | Reorder weights, run matmul, compare vs un-reordered matmul reference |
| `F32`, `BF16`, `S8`, `F16` | Reorder contiguous→blocked and back (round-trip), compare to original. `F16` skips with `isa_unsupported` when AVX512-FP16 is absent |
| `F32_F32_Stride`, `BF16_F32_Stride`, `BF16_BF16_Stride` | Same reorder+matmul flow with strided tensors vs a forced reference kernel |

### 3.2 `test_static_quant_dequant.cpp` — LOWOHA static quant/dequant

Round-trip: Source → INT8 (S8 symmetric / U8 asymmetric) → Source using a **user-provided** scale/zp, compared via `compare_lowoha_quant_output`.

| Test | Source dtype | Notes |
|---|---|---|
| `BF16_QUANT_DEQUANT` | BF16 | |
| `FP32_QUANT_DEQUANT` | FP32 | |
| `F16_QUANT_DEQUANT` | F16 | skips on non-AVX512-FP16; FMA backend auto-selected |
| `F16_QUANT_DEQUANT_F16_SCALE` | F16 | FP16-typed scale buffer (read path of `get_scale_value`) |

### 3.3 `test_dynamic_quant.cpp` — LOWOHA dynamic quant

Round-trip with the scale/zp **computed by the kernel** (`dynamic_quant=true`) on the forward pass and consumed by the static dequant dispatcher on reverse.

| Test | Source dtype | Notes |
|---|---|---|
| `FP32_DYN_QUANT` | FP32 | |
| `BF16_DYN_QUANT` | BF16 | |
| `F16_DYN_QUANT` | F16 | dequant target f32 (compare widens f16→f32); skips on non-AVX512-FP16 |
| `F16_DYN_QUANT_F16_SCALE` | F16 | FP16-typed scale buffer (write-side narrow + read-side widen), dequant target f16 |

### 3.4 `test_strided_cases.cpp` — LOWOHA strided sources

Forward pass reads a strided (optionally row-padded) source; reverse writes contiguous; result compared to original.

| Test | What it covers |
|---|---|
| `BF16_QUANT_DEQUANT_STRIDED` | strided BF16 quant/dequant |
| `FP32_QUANT_DEQUANT_STRIDED` | strided FP32 quant/dequant |
| `FP32_BF16_CVT_STRIDED` | strided scaled FP32↔BF16 conversion |

### 3.5 `test_conversion.cpp` — LOWOHA float↔float type conversion

Round-trip between floating-point dtypes (no INT8), with and without scale/zp. Tolerance pinned to the lossy intermediate dtype.

| Test | Pair | Scaled? |
|---|---|---|
| `FP32_BF16_CVT` / `FP32_BF16_CVT_SCALED` | FP32 ↔ BF16 | no / yes |
| `FP32_F16_CVT` / `FP32_F16_CVT_SCALED` | FP32 ↔ F16 | no / yes |
| `BF16_F16_CVT` / `BF16_F16_CVT_SCALED` | BF16 ↔ F16 | no / yes |

---

## 4. Helper layout — what lives where

Reorder-specific helpers were lifted out of `gtest_utils.{hpp,cpp}` into the
folder-local `reorder_test_helpers.{hpp,cpp}`, keeping the shared header
operator-agnostic (the same policy `group_matmul/` follows).

**In `reorder/reorder_test_helpers.{hpp,cpp}` (reorder-only):**
- **Param struct**: `ReorderType` (embeds `MatmulType` + LOWOHA fields) and its constructor.
- **Global vector**: `reorder_test` — declared `extern` here; *defined* and filled in `gtest_main.cpp` (random / CLI-override / `--input_file` modes), which `#include`s this header. Consumed by the single `INSTANTIATE_TEST_SUITE_P` in `test_reorder_regular.cpp`.
- **Pretty-printer**: `PrintTo(const ReorderType &, …)`.
- **Kernel shims**: `reorder_kernel_test` (regular path), `lowoha_reorder_kernel_test` (LOWOHA path).
- **Comparators**: `compare_lowoha_reorder_output`, `compare_lowoha_quant_output`.
- **Shape / log / str helpers**: `get_lowoha_shape`, `get_lowoha_strided_shape`, `get_lowoha_quant_shape`, `log_lowoha_test_info`, `lowoha_reorder_algo_to_str`, `lowoha_granularity_to_str`.

**Stays in `gtest_utils.{hpp,cpp}` (shared / CLI infra):**
- `ReorderInput` — it's a member of the shared `CLIParams`.
- `read_reorder_inputs` — file-input parser alongside the other `read_*_inputs` helpers (shares `parse_bool_field`).
- Genuinely shared helpers the reorder tests reuse: `tensor_factory_t`, `matmul_kernel_test`, `matmul_forced_ref_kernel_test`, `compare_tensor_2D`, `get_aligned_size`.

Full CLI / input-file / granularity / tolerance documentation lives in the parent `gtests/Readme.md` (see the **LOWOHA Reorder Tests** and **Reorder Input File Format** sections).

---

## 5. How to run

```bash
# Build (from the build directory).
make gtests -j$(nproc)

# Run every reorder test (LOWOHA on by default):
./install/gtests/gtests --gtest_filter='Reorder/TestReorder.*'

# Common targeted filters:
./install/gtests/gtests --gtest_filter='Reorder/TestReorder.*QUANT_DEQUANT*'   # static quant family
./install/gtests/gtests --gtest_filter='Reorder/TestReorder.*DYN_QUANT*'       # dynamic quant family
./install/gtests/gtests --gtest_filter='Reorder/TestReorder.*STRIDED*'         # strided cases
./install/gtests/gtests --gtest_filter='Reorder/TestReorder.*CVT*'             # type conversions
./install/gtests/gtests --gtest_filter='Reorder/TestReorder.F16*'              # all FP16 paths

# Single parameterised instance with a fixed seed + thread count:
./install/gtests/gtests --gtest_filter='Reorder/TestReorder.BF16_QUANT_DEQUANT/0' \
  --seed 42 --num_threads 8
```

The legacy regular tests are SKIPPED unless the LOWOHA-only suite is disabled; passing `--lowoha false` masks the whole fixture rather than enabling them (see the parent `gtests/Readme.md` for the LOWOHA flag semantics).

---

## 6. How to add a new test

### Decision tree

```
Is it a legacy regular (non-LOWOHA) reorder + matmul test?
  yes -> test_reorder_regular.cpp

Is it a LOWOHA static quant/dequant (user-provided scale/zp) round-trip?
  yes -> test_static_quant_dequant.cpp

Is it a LOWOHA dynamic-quant (kernel-computed scale/zp) round-trip?
  yes -> test_dynamic_quant.cpp

Is the source strided / row-padded?
  yes -> test_strided_cases.cpp

Is it a float<->float type conversion (no INT8)?
  yes -> test_conversion.cpp
```

### Rules

- **Reuse the fixture.** Add a `TEST_P(TestReorder, <NAME>) { ... }` to the appropriate file and `#include "reorder_test_common.hpp"`.  Do **not** add a second `INSTANTIATE_TEST_SUITE_P` — the one in `test_reorder_regular.cpp`
  covers the whole suite.
- **Keep names stable.** `Reorder/TestReorder.<NAME>` filters are referenced by CI and external tooling; don't rename existing cases.
- **Mask correctly.** LOWOHA tests start with `if (!use_LOWOHA) GTEST_SKIP();`; legacy regular tests start with `if (use_LOWOHA) GTEST_SKIP();`.
- **FP16 cases** must handle `status_t::isa_unsupported` with `GTEST_SKIP() << "F16 not supported: requires AVX512-FP16 ISA";`.
- **Don't grow `gtest_utils.{hpp,cpp}`** with reorder-only symbols. Put reorder-only helpers shared across these test files in `reorder_test_helpers.{hpp,cpp}`; keep single-file helpers in an anonymous namespace inside the owning `.cpp`. Only add to `gtest_utils.{hpp,cpp}` when a helper is genuinely shared across operators.
