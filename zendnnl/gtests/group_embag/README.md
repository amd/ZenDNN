# `group_embag/` gtests

Self-contained gtest surface for the `group_embedding_bag_direct` LOWOHA API and the lookup-mode (group embedding) variant exercised through the same entry point.

The single-table fixtures `TestEmbedding` (`gtests/test_embedding.cpp`) and `TestEmbag` (`gtests/test_embag.cpp`) do not exercise any of the group APIs; this folder is the only place where `group_embedding_bag_direct` is covered with numerical correctness checks across the dtype matrix.  Each test pins `ZENDNNL_EMBAG_THREAD_ALGO` via an RAII guard in `group_embag_test_helpers.cpp` so the four thread strategies (`batch_threaded`, `table_threaded`, `ccd_threaded`, `hybrid_threaded`) are exercised across runs — `embag_config_t::set_env_config()` is called inside the dispatcher and reads back the pinned value.

---

## 1. Overview

| Question | Answer |
|---|---|
| What gets tested? | `zendnnl::lowoha::embag::group_embedding_bag_direct(...)` and the lookup variant (`algo == none` + `offsets == nullptr` per table). |
| What isn't tested here? | Single-table embedding / embedding bag (covered by `gtests/test_embedding.cpp` and `gtests/test_embag.cpp`); AI suite (covered by `gtests/ai_gtests/test_embag_ai.cpp`). |
| Single binary? | Yes. All test files compile into the same `gtests` executable. Filter via `--gtest_filter='*GroupEmbag*'`. |
| Helpers? | One helper TU `group_embag_test_helpers.{hpp,cpp}` for the parameter struct, `PrintTo`, the per-table tensor builders, and the dispatch shims. File-local helpers stay anonymous. |

---

## 2. File layout

```text
zendnnl/gtests/group_embag/
  CMakeLists.txt                  target_sources(...) into the parent gtests target
  README.md                       this file
  group_embag_test_helpers.hpp    declarations: GroupEmbagType, group_embag_test extern,
                                  GroupTensors, build_group_embag[_quant]_tensors,
                                  build_group_embedding[_quant]_tensors,
                                  free_quant_tables, compare_group_outputs,
                                  group_embag_kernel_test / forced_ref shim, PrintTo
  group_embag_test_helpers.cpp    implementations of the above
  test_group_embag.cpp            TestGroupEmbag               - F32/BF16/F16 + INT8/S4/U4
                                                                 bag-mode correctness
                                                                 (per-test ZENDNNL_EMBAG_
                                                                 THREAD_ALGO override)
  test_group_embedding.cpp        TestGroupEmbedding           - F32/BF16/F16 + INT8/S4/U4
                                                                 lookup-mode correctness
```

Naming convention: `TestGroupEmbag` for bag-mode tests, `TestGroupEmbedding` for lookup-mode tests.

---

## 3. Tested API surface

### 3.1 Public LOWOHA dispatcher

```cpp
zendnnl::lowoha::embag::group_embedding_bag_direct(
    tables, indices, offsets, weights, dsts, params);
```

`offsets[i] == nullptr` + `params[i].algo == none` selects per-table embedding-lookup mode (one output row per index, no reduction). The fixtures in this folder exercise each mode in a homogeneous group (every table in the call shares the same mode); mixed-mode groups within a single call are not covered.

### 3.2 Param structs

| Type | Drives |
|---|---|
| `embag_params_t` (per-table) | dtypes, algo, dimensions, padding_idx, include_last_offset, fp16_scale_bias, dst_stride, num_threads |
| `eb_thread_algo_t` (group-wide) | thread strategy: `batch_threaded`, `table_threaded`, `ccd_threaded`, `hybrid_threaded` |

### 3.3 Test-local dispatch shims

```cpp
status_t group_embag_kernel_test(tables, indices, offsets, weights,
                                 outputs, algos, padding_idxs,
                                 include_last_offsets, fp16_scale_bias,
                                 thread_algo);

status_t group_embag_forced_ref_kernel_test(...same vectors...);
```

DUT side calls `group_embedding_bag_direct`. Reference side loops the existing single-op `embag_forced_ref_kernel_test` / `embedding_forced_ref_kernel_test` from `gtests/gtest_utils.{hpp,cpp}` per table.

---

## 4. Coverage matrix

| Axis | File / class |
|---|---|
| F32/BF16/F16 bag-mode dtype combos (sum/mean/max) | `test_group_embag.cpp::TestGroupEmbag` |
| INT8 / S4 / U4 bag-mode + `fp16_scale_bias` toggle | `test_group_embag.cpp::TestGroupEmbag` |
| F32/BF16/F16 lookup-mode dtype combos | `test_group_embedding.cpp::TestGroupEmbedding` |
| INT8 / S4 / U4 lookup-mode + `fp16_scale_bias` toggle | `test_group_embedding.cpp::TestGroupEmbedding` |
| `eb_thread_algo_t` ∈ {batch, table, ccd, hybrid} | random axis in `GroupEmbagType`, applied by the `scoped_thread_algo` env-var guard in `group_embag_test_helpers.cpp` |

---

## 5. How to run

```bash
make gtests -j$(nproc)
./install/gtests/gtests --gtest_filter='*GroupEmbag*:*GroupEmbedding*'
./install/gtests/gtests --gtest_filter='GroupEmbag/TestGroupEmbag.F32_F32/*'
./install/gtests/gtests --gtest_filter='GroupEmbedding/TestGroupEmbedding.*'
```

The randomized parameter vector `group_embag_test` is filled in `gtests/gtest_main.cpp` from `--test <N>` (default 400).
Seed override:

```bash
./install/gtests/gtests --seed 12345 --test 10
```

---

## 6. How to add a new test

```text
Is it bag mode (offsets present, algo = sum/mean/max)?
  Float / BF16 / F16 / INT8 / S4 / U4 correctness?
    yes -> test_group_embag.cpp::TestGroupEmbag

Is it lookup mode (no reduction, offsets = nullptr)?
  yes -> test_group_embedding.cpp::TestGroupEmbedding
```

If a helper would be reused across 2+ files, add it to `group_embag_test_helpers.{hpp,cpp}`. Single-file helpers stay in anonymous namespaces in their owning `.cpp`.

---

## 7. Maintenance notes

- **Don't grow `gtests/gtest_utils.{hpp,cpp}`** with group-embag-specific symbols; lift them into `group_embag_test_helpers.{hpp,cpp}` instead.
- **Keep test files under ~1500 lines.** Split if they exceed that.
- The `embag_accum_type` field on `embag_config_t` is process-wide and unsynchronized (see TODO in `embag_config.hpp`); mixed-dtype group invocations may flake if reference validation is ever wired in. The current group tests use a single dtype combo per call, side-stepping the race.
