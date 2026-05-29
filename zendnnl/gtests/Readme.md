
(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# ZenDNN Google Test Infrastructure

## **Overview of GTest**
Google Test (GTest) is a unit testing framework for C++ that provides a robust and flexible way to write and execute tests. In this project, GTest is used to validate the functionality and performance of ZenDNN operations. It supports parameterized tests, filtering, and detailed reporting, making it ideal for testing complex numerical computations.

## **Purpose and Audience**
The ZenDNN GTest infrastructure is designed for:
- **Developers**: To verify the correctness of ZenDNN operations during development.
- **Test Engineers**: To run automated tests and validate functionality across different configurations.
- **Researchers**: To experiment with numerical precision and post-operations in deep learning workloads.

This infrastructure ensures that ZenDNN operations meet expected accuracy standards under various conditions.

## **Flexible Configuration**
ZenDNN GTest provides flexibility in configuring tests through command-line arguments and code-level parameters. Parameters are built in one of three ways — **fully randomized**, **command-line overrides** (random mode with pinned fields), or **input file** — described in [Running Tests](#running-tests). Users can:
- **Filter Tests**: Run specific test suites or cases using `--gtest_filter`.
- **Set Random Seed**: Provide a seed value for reproducible random test data generation.
- **Specify Post-Operations**: Apply post-operations like `relu`, `gelu_tanh`, etc., during matrix multiplication tests.
- **Backend Selection**: Choose specific computational backends using `--backend` parameter to control algorithm selection (e.g., `aocl_dlp`, `onednn`, `libxsmm`).
- **LOWOHA**: Enable or disable Low Overhead API using `--lowoha` parameter.
- **Thread Control**: Specify the number of threads for parallel execution using `--num_threads` parameter.
- **Input File Support**: Use `--input_file` with `--op` (and `--ndims 3` for batch matmul, or `--lowoha` for reorder) to read per-line test configurations from a file instead of random generation. Supported operators: `matmul`, `reorder`, `embeddingbag`, `embedding`, and `normalization`.
- **Per-suite overrides**: Optional flags (`--m`, `--k`, `--batch_size`, `--dim_choice`, `--norm_type`, etc.) fix specific fields when building **randomized** `MatmulType`, `BatchMatmulType`, `ReorderType`, embedding, or normalization parameters (no `--input_file`). They are **not** applied in input-file mode; see [CLI flags vs input-file mode](#cli-flags-vs-input-file-mode). See also [Command-line parameters (reference)](#command-line-parameters-reference).
- **LOWOHA dimension mode**: With `--lowoha true`, `--dim_choice` selects how LOWOHA reorder shapes `M`, `N`, and `batch` are constrained (see [LOWOHA Reorder Tests](#lowoha-reorder-tests)).

## **Configurable Parameters**
You can modify the following parameters in the source code (`gtest_main.cpp`):
- `MATMUL_F32_TOL`: Tolerance for floating-point precision in tests (default: `0.001`).
- `MATMUL_BF16_TOL`: Tolerance for BF16 precision in tests (default: `0.01`).
- F16 tests use the same tolerance as BF16 (`MATMUL_BF16_TOL`).
- `EMBAG_F16_TOL`: Tolerance for EmbeddingBag FP16 precision in tests (default: `0.01`).
- `NORM_F32_TOL`: Tolerance for normalization F32 tests (default: `0.001`).
- `NORM_BF16_TOL`: Tolerance for normalization BF16 tests (default: `0.01`).
- `NORM_F16_TOL`: Tolerance for normalization F16 tests (default: `0.01`).
- `SOFTMAX_F32_TOL`: Tolerance for softmax F32 tests (default: `0.001`).
- `SOFTMAX_BF16_TOL`: Tolerance for softmax BF16 tests (default: `0.01`).
- `TEST_NUM`: Number of test cases to generate (default: `400`).
- `POST_OPS_LIMIT`: Maximum number of post-ops allowed in a single chain (default: `3`). Defined in `gtest_utils.hpp`. Applies to `--postop`, matmul/batch-matmul/reorder input file `postOp` fields, and random post-op generation (which picks a chain length uniformly from `1` … `POST_OPS_LIMIT`). If the parsed chain exceeds this limit, gtest reports an error (`Post-op chain length exceeds POST_OPS_LIMIT.`).

## **Matrix Dimension Ranges**

The following dimension ranges are used for randomly generated test cases (defined in `gtest_utils.hpp`):

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| **M** | 1 | 3000 | Number of rows in the output matrix |
| **K** | 1 | 3000 | Inner dimension (columns of A / rows of B) |
| **N** | 1 | 3000 | Number of columns in the output matrix |
| **Batch Size** | 1 | 256 | Batch size for batch matrix multiplication |

**Dimension Formula:**
```
dimension = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END
```
Where `MATMUL_SIZE_START = 1` and `MATMUL_SIZE_END = 3000`.

## **Buffer Value Distribution**

Test tensors are initialized with uniformly distributed random values. The distribution ranges vary by data type and tensor role:

### **Input and Weight Tensors**

| Data Type | Distribution Range | Description |
|-----------|-------------------|-------------|
| **F32** | `[-2.0, 2.0]` | Standard floating-point tests |
| **BF16** | `[-2.0, 2.0]` | BFloat16 tests |
| **F16** | `[-2.0, 2.0]` | Half-precision (IEEE 754) tests |
| **S8** (INT8 weights) | `[-25.0, 25.0]` | Signed 8-bit integer quantized weights |
| **U8** (INT8 source) | `[0, 25.0]` | Unsigned 8-bit integer quantized input |
| **S4** (WOQ weights) | `[-8, 7]` | 4-bit signed integer for weight-only quantization |

### **Other Tensors**

| Tensor Type | Distribution Range | Description |
|-------------|-------------------|-------------|
| **Bias** | `[-2.0, 2.0]` | Bias tensor (F32 or BF16) |
| **Binary Post-op** | `[-2.0, 2.0]` | Binary add/mul tensors |
| **Output** | `[-2.0, 2.0]` | Initial output buffer values |
| **Quantization Scales** | `[-0.2, 0.2]` to `[-2.0, 2.0]` | Scale factors for quantized operations |
| **Zero Points** | Integer values | Zero points for asymmetric quantization |

### **Uniform Constant Value Tensors**

Some tensors are initialized with a single constant value (all elements set to the same value) using `uniform_tensor()` instead of random distribution:

| Tensor Type | Constant Value | Data Type | Description |
|-------------|---------------|-----------|-------------|
| **Weight Zero Point (WOQ)** | `0` | S8 | Zero point for S4 weight-only quantization |
| **Source Zero Point (INT8)** | `16` | S32 | Zero point for U8 asymmetric quantization |
| **Weight Zero Point (INT8)** | `16` | S32 | Zero point for per-tensor weight quantization |
| **Destination Zero Point** | `53` | S32 | Zero point for U8 output quantization |

### **Alpha and Beta Parameters**
- **Alpha**: Uniformly distributed in `[0.0, 10.0]`
- **Beta**: Uniformly distributed in `[0.0, 10.0]`
- **Note**: For LIBXSMM backends, alpha is fixed to `1.0` and beta is `0` or `1`.

## **Accuracy Validation**

ZenDNN GTest validates numerical accuracy by comparing optimized implementations against reference results using dynamic tolerance algorithms that adapt to matrix dimensions and data types.

### **Tolerance Calculation**

**BF16/F16 Operations:**
```
abs_bound = k * epsilon_bf16
allowed_error = abs_bound + rtol_bf16 * |reference_value|
```
> **Note:** F16 uses the same tolerance bounds as BF16 since both are low-precision 16-bit formats.

**F32 Operations:**
```
abs_bound = ((20 + log2(k)/4) * k + 15) * epsilon_f32
allowed_error = abs_bound + rtol_f32 * |reference_value|
```

Where `k` is the inner matrix dimension that determines accumulation error scaling.

**Relaxation Features:**
- **Zero-reference tolerance**: `8e-4` for near-zero reference values (`< 1e-6`)
- **Additional slack**: `2e-4` buffer for F32 operations

**Note**: Relaxation can be enabled via `ENABLE_F32_RELAXATION` macro in `gtest_utils.hpp` and is enabled by default for LIBXSMM backends.

### **Configuration**
Key tolerance parameters in `gtest_main.cpp`:
```cpp
const float epsilon_f32     = 1.19e-7;  // Base F32 tolerance
const float epsilon_bf16    = 9.76e-4;    // Base BF16 tolerance
const float rtol_f32        = 1e-5;    // F32 relative tolerance
const float rtol_bf16       = 1e-2;    // BF16 relative tolerance
```

### **Adaptive Scaling**
- **Small matrices**: Tolerance based on base epsilon values
- **Large matrices**: Logarithmic scaling prevents excessive tolerance growth
- **Post-operations**: Additional margin (P=15) for accumulation errors

This ensures accuracy validation scales with computational complexity while maintaining precision standards across different backends and data types.


## **Supported Post-Operations**
The following post-operations are supported for matmul and its variations:
- `relu`
- `gelu_tanh`
- `gelu_erf`
- `sigmoid`
- `swish`
- `tanh`
- `mish`
- `binary_add`
- `binary_mul`

**Note:** If specified post-op is not from above mention list, then no post-op will be applied to operator.

**Post-op chain length:** At most `POST_OPS_LIMIT` post-ops may appear in one chain (see **Configurable Parameters**). Longer chains from `--postop` or an input file are rejected.

## **Bias Support**
- **Default Behavior**: Bias is enabled by default for all matrix multiplication tests
- **Data Types**: Bias tensors support both F32 and BF16 data types, randomly selected during test execution

**Note**: GTest currently excludes bias operations for LIBXSMM BF16 backends

## **Directory Structure**

The `zendnnl/gtests/` directory contains all test-related files and subdirectories. Below is the structure:

```plaintext
gtests/
├── gtest_main.cpp           # Entry point for all tests.
├── test_matmul.cpp          # Single-op matmul testsuite (F32, BF16, INT8, WOQ, stride).
├── test_group_matmul.cpp    # Group matmul testsuite (MoE postop, gated activation, fused MoE).
├── test_batchmatmul.cpp     # Batch matmul testsuite with different test cases.
├── test_reorder.cpp         # Reorder testsuite (regular + LOWOHA quantization/dequantization).
├── test_embag.cpp           # Embedding bag testsuite with different test cases.
├── test_embedding.cpp       # Embedding testsuite with different test cases.
├── test_normalization.cpp   # Normalization testsuite with different test cases.
├── test_omp_api.cpp         # OpenMP thread-control utility testsuite (omp_thread_control.hpp).
├── test_softmax.cpp         # Softmax testsuite (OneDNN vs Reference).
└── gtest_utils.cpp/hpp      # Utility functions for tests.
```

## **Configure and build GTest with ZenDNN**
- Configure:
```bash
cmake -DZENDNNL_BUILD_GTEST=ON ..
```
- Build:
```bash
cmake --build .
```
## **Running Tests**

Test parameters come from one of three sources. Use `--gtest_filter` to pick suites and cases in any mode.

| Mode | How |
|------|-----|
| **Default** | No `--input_file`, no override flags — RNG in `gtest_utils.cpp` |
| **CLI** | Override flags (`--m`, `--postop`, …) — unset fields stay random |
| **Input file** | `--input_file` + `--op` (+ `--ndims` / `--lowoha` when needed) — empty CSV cells stay random |

**Post-ops:** join with **`:`** in `--postop` or the file `postOp` column (max **`POST_OPS_LIMIT`**, default `3`).

---

### **1. Default (fully randomized)**

No `--input_file`, no override flags — RNG in `gtest_utils.cpp` fills each test through the active suite's `*Type` constructor (shapes, dtypes, backends, post-ops where supported, LOWOHA when applicable, thread count, and tensor values). Per-suite defaults: [Command-line parameters (reference)](#command-line-parameters-reference). Shared tensor initialization: [Buffer Value Distribution](#buffer-value-distribution).

```bash
./install/gtests/gtests --gtest_filter=<TestSuite>/<TestCase>[/<Index>] \
  [--seed <Seed>] [--test <N>] [--num_threads <N>]
```

Omitted: `--seed` → wall-clock; `--test` → **400**; `--num_threads` → random.

```bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.BF16_BF16/*
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F32_F32/* --seed 42 --test 10
```

---

### **2. CLI overrides**

Per-suite override flags pin fields; anything you omit stays random. Only when **`--input_file` is not set**. Flag list: [Command-line parameters (reference)](#command-line-parameters-reference). Standard `--gtest_*` flags also apply.

```bash
./install/gtests/gtests --gtest_filter=<TestSuite>/<TestCase>[/<Index>] \
  [--seed <Seed>] [--test <N>] [--num_threads <N>] [--<flag> <value>] ...
```

```bash
./install/gtests/gtests --gtest_filter=<TestSuite>/<TestCase>[/<Index>] --seed <Seed> --<flag> <value>
./install/gtests/gtests --gtest_filter=<TestSuite>/* --<flag1> <value1> --<flag2> <value2>
```

---

### **3. Input file**

`--input_file` + `--op` — one config per non-empty line. Operators: `matmul`, `reorder`, `embeddingbag`, `embedding`, `normalization`. Empty CSV cells stay random; per-suite CLI overrides are ignored (see [CLI flags vs input-file mode](#cli-flags-vs-input-file-mode)).

**Scope note:** At startup, only the parameter vector for the selected `--op` is populated from the file; other operator suites are left without parameters. Always use `--gtest_filter` to run the matching suite (e.g. `Matmul/*` for `--op matmul --ndims 2`, `BatchMatmul/*` for `--ndims 3`, `Reorder/*` for `--op reorder`).

```bash
./install/gtests/gtests --gtest_filter=<TestSuite>/<TestCase>[/<Index>] \
  --input_file <InputFile> --op <Operator> \
  [--ndims 2|3] [--lowoha true|false] [--test <N>] [--num_threads <N>] [--seed <Seed>]
```

**Operator-specific requirements:**

| `--op` | Extra flags | Line format (summary) |
|--------|-------------|------------------------|
| `matmul` | `--ndims 2` (default) or `3` for batch matmul | 12 fields (2D) or 13 fields (3D); see [Matmul (2D)](#matmul-2d-input-file-format) / [Batch Matmul (3D)](#batch-matmul-3d-input-file-format) |
| `reorder` | **`--lowoha true` or `false`** required | 8 fields; layout depends on `--lowoha` — [Reorder](#reorder-input-file-format) |
| `embeddingbag` | none | 11 fields — [Embedding Bag](#embedding-bag-input-file-format) |
| `embedding` | none | 8 fields — [Embedding](#embedding-input-file-format) |
| `normalization` | none | 6 fields — [Normalization](#normalization-input-file-format) |

**What applies in input-file mode:**

Each line is parsed into a `*Input` struct; **`cli_params` is not consulted** for operator fields. Only the knobs below (plus file columns) shape a run.

| Category | Flags / knobs |
|----------|----------------|
| **Global CLI** | `--input_file`, `--op`, `--test`, `--seed`, `--num_threads`, `--gtest_filter` |
| **Operator routing** | `--ndims` (`2` default / `3` for batch matmul); `--lowoha` `true` or `false` (**required** for `--op reorder`) |
| **Environment** | `ZENDNNL_MATMUL_ALGO`, `ZENDNNL_BMM_ALGO` — process-wide; override file `kernel` on matmul lines ([precedence](#backend-selection-precedence)) |
| **Not applied** | Per-suite overrides (`--m`, `--postop`, `--backend`, `--src_dtype`, `--norm_type`, …) — set them in the file; empty cells randomize per instance ([details](#cli-flags-vs-input-file-mode)) |

**Per-line behavior:**

- **Non-empty fields** set that parameter for each expanded test instance.
- **Empty fields** use the suite’s random default **on each** `*Type` construction (not once per line). With `--test` > 1, empty columns can yield **different** random fills per repetition — use `--test 1` or fill every column you need fixed.
- Invalid lines are logged and skipped.

**Examples:**
```bash
# 2D matmul from file
./install/gtests/gtests --gtest_filter=Matmul/* --input_file matmul.txt --op matmul

# Batch matmul (3D)
./install/gtests/gtests --gtest_filter=BatchMatmul/* --input_file batch_tests.txt --op matmul --ndims 3

# Regular vs LOWOHA reorder (separate line layouts)
./install/gtests/gtests --gtest_filter=Reorder/* --input_file reorder.txt --op reorder --lowoha false
./install/gtests/gtests --gtest_filter=Reorder/* --input_file lowoha_reorder.txt --op reorder --lowoha true

# One resolved case per line, 8 threads
./install/gtests/gtests --gtest_filter=Matmul/* --input_file matmul_input.txt --op matmul --num_threads 8 --test 1
```

Full column definitions: [Input File Format](#input-file-format).

## **Command-line parameters (reference)**

ZenDNN flags: **`--<name> <value>`** (two tokens). Args whose name contains `gtest` are for GoogleTest only (`InitGoogleTest`).

**Booleans:** `true`/`1` or `false`/`0` (case-insensitive). Invalid values → logged, then random default. **Exception:** `--input_file` + `--op reorder` requires valid `--lowoha` (else exit). See [3. Input file](#3-input-file).

**Global flags** (all modes — default, CLI override, and input file):

| Flag | Values / description |
|------|----------------------|
| `--seed` | RNG seed; wall-clock timestamp if omitted. |
| `--test` | Repetitions per suite or file line (default **400**). With `--input_file`, empty CSV columns re-randomize on **each** repetition — use `--test 1` for one resolved case per line. |
| `--num_threads` | OpenMP thread count; **0** = randomize per test. |
| `--input_file` | CSV path; requires `--op` (add `--ndims 3` for batch matmul or `--lowoha` for reorder). Per-operator overrides are ignored — see [CLI flags vs input-file mode](#cli-flags-vs-input-file-mode). |
| `--op` | With `--input_file`: `matmul`, `reorder`, `embeddingbag`, `embedding`, `normalization`. |
| `--ai_test_mode` | AI gtests only (`ZENDNNL_BUILD_AI_GTESTS=ON`): `presub`, `nightly`, `minimal`, `accuracy`, … |

**Random-mode overrides** (CLI override mode only; **not** applied with `--input_file` — set values in the CSV instead):

### Matmul, batch matmul, and reorder

Shared flags for `--op matmul` (2D or 3D) and `--op reorder`:

| Flag | Values / description |
|------|----------------------|
| `--m`, `--k`, `--n`, `--batch_size` | Positive integers (`--batch_size` for 3D / batch matmul). |
| `--transA`, `--transB`, `--inplace_reorder` | Boolean |
| `--alpha`, `--beta` | Float scalars |
| `--src_dtype`, `--dst_dtype` | Lowercase dtype strings (see **Dtypes** below) |
| `--postop` | Post-op or **`:`**-separated chain (max **`POST_OPS_LIMIT`**, default `3`). |
| `--backend` | Kernel backend (`aocl_dlp`, `onednn`, `libxsmm`, …). Overridden by `ZENDNNL_MATMUL_ALGO` / `ZENDNNL_BMM_ALGO` when set — see [Backend selection precedence](#backend-selection-precedence). |
| `--lowoha` | `true`/`false`/`1`/`0`; routes matmul/reorder/embedding suites in random mode (see **Defaults** below). **Required** with `--input_file --op reorder` to select the line layout. |

**Matmul / batch matmul** — `--op matmul`; `--ndims` selects 2D vs 3D:

| Flag | Values / description |
|------|----------------------|
| `--ndims` | **`2`** (default, 2D matmul) or **`3`** (batch matmul). |
| `--weight_granularity` | `tensor` / `per_tensor` / `channel` / `per_channel` |

**Reorder** — `--op reorder` (plus shared flags above):

| Flag | Values / description |
|------|----------------------|
| `--weight_granularity` | Regular: `tensor` / `per_tensor` / `channel` / `per_channel`. LOWOHA (`--lowoha true`): also `group` / `per_group`. |
| `--num_groups` | LOWOHA only; required for `group` / `per_group`. |
| `--dim_choice` | LOWOHA only; **`1`**–**`3`**: `1` → `M=1, batch=0`; `2` → `batch=1`; `3` → keep 3D batch. Omitted/invalid → random ~20% / 50% / 30% for 1D / 2D / 3D. |

### Embedding bag and embedding

Shared flags (`EmbeddingInput` ⊂ `EmbagInput`):

| Flag | Values / description |
|------|----------------------|
| `--num_embeddings`, `--embedding_dim`, `--num_indices` | Positive integers |
| `--padding_index` | Signed int (e.g. `-1`) |
| `--is_weights`, `--fp16_scale_bias`, `--strided` | Boolean |
| `--indices_dtype` | `s32` or `s64` |

**Embedding bag only** — `--op embeddingbag`:

| Flag | Values / description |
|------|----------------------|
| `--num_bags`, `--embag_algo`, `--include_last_offset` | `--embag_algo`: `sum` / `mean` / `max` |

**Embedding** — `--op embedding`; uses the shared flags above only.

### Normalization

Requires `--op normalization`:

| Flag | Values / description |
|------|----------------------|
| `--norm_type` | `layer`, `batch`, `rms`, `fusedaddrms` |
| `--norm_shape` | Comma-separated dims, e.g. `2,4096` (`batch` needs ≥2). Input file uses **`:`** — see [Normalization](#normalization-input-file-format) |
| `--use_scale`, `--use_shift` | Boolean |
| `--gamma_dt`, `--beta_dt` | **`f32`**, **`bf16`**, or **`f16`** (`f16` only on AVX512-FP16-capable hosts; otherwise rejected with a log message and a random fallback) |

**Defaults (random mode):** Omitted flags randomize (`--seed` → timestamp, `--test` → 400). **`--lowoha` omitted:** reorder 50/50 LOWOHA/regular; matmul covers both; embedding 50/50; normalization/softmax always LOWOHA. **`--input_file`:** per-suite overrides ignored; globals (`--test`, `--seed`, `--num_threads`) and env `ZENDNNL_MATMUL_ALGO` / `ZENDNNL_BMM_ALGO` still apply — see [CLI flags vs input-file mode](#cli-flags-vs-input-file-mode), [Backend selection precedence](#backend-selection-precedence).

**Dtypes:** CLI accepts `f32`, `f16`, `bf16`, `s32`, `s16`, `s8`, `s4`, `u32`, `u16`, `u8`, `u4`. Matmul honors **`src_dtype`**: `s8`/`u8` only; **`dst_dtype`**: `s8`/`u8`/`f32`/`bf16`/`f16` (others randomized). `s64` is **`--indices_dtype`** only. Reorder random mode shares matmul flags; use `--gtest_filter` to select suites. Embedding uses the shared embag flags only (`EmbeddingInput` ⊂ `EmbagInput`). Unset flags use `*Type` random defaults (random mode only; not with `--input_file`).

## **Input File Format**

One non-empty line = one test case. Fields are **comma-separated**; an **empty field** uses the suite’s random default when that test instance is built (not from CLI override flags).

**Rules (all operators):**
- Invalid lines are logged and skipped.
- **`--test N`:** each line is expanded `N` times; empty fields are re-randomized on every instance (use `--test 1` or fill all columns for a fixed case).
- **`--num_threads`:** CLI only (not in the file).
- String tokens (`f32`, `tensor`, `channel`, …) match the [CLI dtype tables](#command-line-parameters-reference) unless noted below.

### **CLI flags vs input-file mode**

Parameters come from the file, not from per-operator CLI flags (`cli_params` is unused in input-file mode), except for the specific global / operator exceptions listed below.


| Honored | Ignored (use file columns) |
|---------|----------------------------|
| `--input_file`, `--op`, `--ndims` (default `2`), `--lowoha` (reorder layout; also matmul, embedding, and embag via `cmd_lowoha` / `parse_cmd_lowoha()`), `--test`, `--seed`, `--num_threads`, `--gtest_filter` | `--m`, `--k`, `--n`, `--batch_size`, `--transA`, `--transB`, `--alpha`, `--beta`, `--postop`, `--backend`, dtypes, `--inplace_reorder`, `--num_groups`, `--dim_choice`, embedding/normalization flags, … |
| `ZENDNNL_MATMUL_ALGO` / `ZENDNNL_BMM_ALGO` on matmul (see below) | |

Example: `--input_file matmul.txt --op matmul --m 64 --postop relu` does not set `M` or post-op; only the CSV line counts. However, `--lowoha` remains an exception for the operators noted above.

### **Backend selection precedence**

Matmul / batch matmul only:

1. `ZENDNNL_MATMUL_ALGO` (2D) or `ZENDNNL_BMM_ALGO` (3D) if set (enum `0`–`11` or `auto`)
2. File **`kernel`** column (or `--backend` in random mode)
3. Random among backends enabled in the build

`ZENDNNL_MATMUL_ALGO=4` forces AOCL DLP for every line, even when the file names another `kernel`.

### **Matmul (2D) Input File Format**

`--op matmul` (`--ndims` defaults to `2`). **12** fields per line:

```
M,K,N,postOp,kernel,transA,transB,alpha,beta,src_dtype,dst_dtype,weight_granularity
```

Matmul dtypes: **`src_dtype`** — `s8`, `u8` only (others → random + log); **`dst_dtype`** — `s8`, `u8`, `f32`, `bf16`, `f16`.

```
128,256,512,relu,aocl_dlp,false,false,1.0,0.0,,,
64,128,256,binary_add:relu,onednn,false,true,2.0,1.0,s8,bf16,tensor
,,512,relu:gelu_tanh,onednn,false,false,2.0,1.0,u8,bf16,tensor
```

### **Batch Matmul (3D) Input File Format**

`--op matmul --ndims 3`. **13** fields per line:

```
BS,M,K,N,postOp,kernel,transA,transB,alpha,beta,src_dtype,dst_dtype,weight_granularity
```

Same dtype rules as 2D matmul.

```
32,128,256,512,relu,aocl_dlp,false,false,1.0,0.0,,,
16,64,128,256,,libxsmm,false,true,1.0,0.0,s8,bf16,channel
```

### **Reorder Input File Format**

`--op reorder` plus **`--lowoha true`** or **`--lowoha false`** (layout selector; use separate files for each).

**Regular (`--lowoha false`)** — **8** fields:

```
M,K,N,postOp,kernel,transA,transB,inplace_reorder
```

```
4,6,28,relu:gelu_tanh,aocl_dlp,true,false,false
4,,,binary_add,aocl_dlp,true,false,false
```

**LOWOHA (`--lowoha true`)** — **8** fields:

```
batch_size,M,N,src_dtype,dst_dtype,weight_granularity,num_groups,dim_choice
```

`dim_choice`: `1` / `2` / `3` (1D / 2D / 3D). See [LOWOHA Reorder Tests](#lowoha-reorder-tests).

```
2,32,256,f32,bf16,channel,4,3
2,,256,f32,bf16,group,4,3
```

### **Embedding Bag Input File Format**

`--op embeddingbag`. **11** fields:

```
num_embeddings,embedding_dim,num_bags,num_indices,embag_algo,padding_index,include_last_offset,is_weights,indices_dtype,fp16_scale_bias,strided
```

`embag_algo`: `sum`, `mean`, `max`. `padding_index`: signed int (e.g. `-1`). `indices_dtype`: `s32`, `s64`.

```
4,2,5,6,mean,,false,true,s32,true,true
4,,5,6,,,false,true,,,
```

### **Embedding Input File Format**

`--op embedding`. **8** fields:

```
num_embeddings,embedding_dim,num_indices,padding_index,is_weights,indices_dtype,fp16_scale_bias,strided
```

`indices_dtype`: `s32`, `s64`.

```
2,3,6,0,true,s32,true,false
,3,6,,true,,false,
```

### **Normalization Input File Format**

`--op normalization`. **6** fields:

```
norm_type,norm_shape,use_scale,use_shift,gamma_dt,beta_dt
```

`norm_type`: `layer`, `batch`, `rms`, `fusedaddrms`. `norm_shape`: colon-separated dims (e.g. `2:4096`, `32:64:56:56`) — not commas; CLI `--norm_shape` uses commas instead. `gamma_dt` / `beta_dt`: `f32`, `bf16`.

```
layer,2:3,,,,
batch,8:2,,,,
rms,4096,true,false,bf16,f32
```

### **Shared field reference**

| Field | Notes |
|-------|--------|
| **M, K, N** | Positive integers |
| **BS** | Batch matmul or LOWOHA reorder batch size |
| **postOp** | Chain with `:` (`relu:gelu_tanh`). Tokens: `relu`, `gelu_tanh`, `gelu_erf`, `sigmoid`, `swish`, `tanh`, `binary_add`, `binary_mul`, empty/`none`. Max length `POST_OPS_LIMIT` (default 3) |
| **kernel** | `aocl_dlp`, `aocl_dlp_blocked`, `onednn`, `onednn_blocked`, `libxsmm`, `libxsmm_blocked`, `native_gemm`, `native_brgemm`. Overridden by matmul env vars above |
| **Booleans** | `true`/`1`, `false`/`0`, or empty (random). Other values skip the line |
| **alpha, beta** | Float GEMM scales |
| **src_dtype** | Matmul: `s8`, `u8`. LOWOHA reorder: also `f32`, `bf16` |
| **dst_dtype** | Matmul: `s8`, `u8`, `f32`, `bf16`, `f16` |
| **weight_granularity** | `tensor`/`per_tensor`, `channel`/`per_channel`; `group`/`per_group` (LOWOHA only) |
| **num_groups** | Required when granularity is `group` / `per_group` |

## **LOWOHA Reorder Tests**

LOWOHA reorder tests validate quantization, dequantization, and type conversion kernels using a **round-trip methodology**. Each test performs a forward operation followed by its inverse, and then compares the result against the original input to verify correctness.

### **Test Categories**

| Category | Test Cases | Description |
|----------|-----------|-------------|
| **Static Quantization** | `BF16_QUANT_DEQUANT`, `FP32_QUANT_DEQUANT` | Round-trip: Source → INT8 (S8/U8) → Source using user-provided scale/zp |
| **Strided Static Quantization** | `BF16_QUANT_DEQUANT_STRIDED`, `FP32_QUANT_DEQUANT_STRIDED` | Same as above but with strided (non-contiguous) source memory |
| **Type Conversion** | `FP32_BF16_CVT` | Round-trip: FP32 ↔ BF16 without scale/zp |
| **Scaled Type Conversion** | `FP32_BF16_CVT_SCALED` | Round-trip: FP32 ↔ BF16 with scale/zp |
| **Strided Scaled Conversion** | `FP32_BF16_CVT_STRIDED` | Same as scaled conversion but with strided source memory |
| **Dynamic Quantization** | `FP32_DYN_QUANT`, `BF16_DYN_QUANT` | Round-trip: Source → INT8 (S8/U8) → Source where scale/zp are computed by the kernel |

### **Round-Trip Methodology**

All LOWOHA reorder tests follow a three-step pattern:

1. **Forward pass**: Apply the operation (quantize, dequantize, or convert) on the source tensor to produce an intermediate result.
2. **Backward pass**: Apply the inverse operation on the intermediate result to reconstruct the original data type.
3. **Compare**: Compare the reconstructed output against the original input using a tolerance-based comparison.

### **Comparison Methods**

- **Quantization tests** (`*QUANT_DEQUANT*`, `*DYN_QUANT*`, `*CVT_SCALED*`, `*CVT_STRIDED*`): Use `compare_lowoha_quant_output`, which computes a scale-aware tolerance:
  ```
  tolerance = max_scale / 2  +  epsilon
  ```
  Where `epsilon = 0.03` for BF16 sources and `0.001` for FP32 sources. The `max_scale / 2` term accounts for the maximum rounding error introduced by quantization.

- **Type conversion tests** (`FP32_BF16_CVT`): Use `compare_lowoha_reorder_output` with fixed tolerances based on the output data type (BF16 tolerance for BF16, FP32 tolerance for FP32).

### **Randomized Test Parameters**

Each LOWOHA reorder test instance is parameterized with randomly generated values:

| Parameter | Range / Distribution | Description |
|-----------|---------------------|-------------|
| **Dimensionality** | 1D (20%), 2D (50%), 3D (30%) unless overridden | Tensor dimensionality; use `--dim_choice` `1`/`2`/`3` to fix rank (`--dim_choice` must be in range or it is ignored; see [Command-line parameters (reference)](#command-line-parameters-reference)) |
| **M** | [1, 3000] | Row dimension (fixed to 1 for 1D) |
| **N** | [1, 3000] | Column dimension |
| **Batch** | [2, 256] for 3D | Batch dimension |
| **Symmetric/Asymmetric** | 50/50 random | S8 (symmetric, no zp) or U8 (asymmetric, with zp) |
| **Granularity** | per-tensor (60-70%), per-channel (30%), per-group (10%) | Quantization granularity |
| **Strided padding** | [0, 16] elements | Row padding for strided tests (50% chance of zero padding) |
| **Threads** | [1, max_threads] | Number of OMP threads |

### **Quantization Granularities**

LOWOHA reorder tests support five quantization granularities:

| Granularity | Scale/ZP Shape (2D) | Scale/ZP Shape (3D) | Description |
|------------|--------------------|--------------------|-------------|
| **Per-tensor** | `{1, 1}` | `{1, 1, 1}` | Single scale/zp for all elements |
| **Per-channel-row** | `{M, 1}` | `{1, M, 1}` | One scale/zp per row (per-token) |
| **Per-channel-col** | `{1, N}` | `{1, 1, N}` | One scale/zp per column |
| **Per-group-row** | `{G, N}` | `{1, G, N}` | Groups along M dimension (M/G rows per group) |
| **Per-group-col** | `{M, G}` | `{1, M, G}` | Groups along N dimension (N/G cols per group) |

For 1D tensors, only per-tensor and per-channel granularities are supported.

## **Examples**

### Reorder Tests
 - The Reorder TestSuite contains both LOWOHA reorder tests and Regular reorder tests (Reorder + Matmul).
 - By default (no `--lowoha` flag), each test instance is randomly assigned as LOWOHA or regular (50/50).
 - Use `--lowoha true` to run only LOWOHA reorder tests, or `--lowoha false` to run only regular reorder tests.
 - **LOWOHA test cases**: `BF16_QUANT_DEQUANT`, `FP32_QUANT_DEQUANT`, `FP32_BF16_CVT`, `FP32_BF16_CVT_SCALED`, `BF16_QUANT_DEQUANT_STRIDED`, `FP32_QUANT_DEQUANT_STRIDED`, `FP32_BF16_CVT_STRIDED`, `FP32_DYN_QUANT`, `BF16_DYN_QUANT`
 - **Regular test cases**: `F32_F32`, `BF16_F32`, `BF16_BF16`, `F32`, `BF16`, `S8`, `F32_F32_Stride`, `BF16_F32_Stride`, `BF16_BF16_Stride`

#### LOWOHA Reorder Tests

1. Run all LOWOHA reorder tests:
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.* --lowoha true
```
2. Run LOWOHA static quantization round-trip tests (BF16 and FP32):
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_QUANT_DEQUANT/* --lowoha true
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.FP32_QUANT_DEQUANT/* --lowoha true
```
3. Run LOWOHA strided quantization round-trip tests:
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_QUANT_DEQUANT_STRIDED/* --lowoha true
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.FP32_QUANT_DEQUANT_STRIDED/* --lowoha true
```
4. Run LOWOHA type conversion tests (FP32 ↔ BF16):
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.FP32_BF16_CVT/* --lowoha true
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.FP32_BF16_CVT_SCALED/* --lowoha true
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.FP32_BF16_CVT_STRIDED/* --lowoha true
```
5. Run LOWOHA dynamic quantization tests:
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.FP32_DYN_QUANT/* --lowoha true
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_DYN_QUANT/* --lowoha true
```
6. Run a specific LOWOHA test with fixed seed and thread count:
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_QUANT_DEQUANT/0 --seed 42 --num_threads 8 --lowoha true
```
7. Force LOWOHA tensor rank with `--dim_choice` (`1` = 1D, `2` = 2D, `3` = 3D; invalid values are ignored):
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_QUANT_DEQUANT/* --lowoha true --dim_choice 2
```

#### Regular Reorder Tests (Reorder + Matmul)

8. Run all regular reorder tests:
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.* --lowoha false
```
9. Run FP32 reorder + matmul tests:
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.F32_F32/* --lowoha false
```
10. Run BF16 reorder + matmul tests (F32 output):
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_F32/* --lowoha false
```
11. Run BF16 reorder + matmul tests (BF16 output):
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_BF16/* --lowoha false
```

### Matmul Tests
 - Matmul TestSuite has nine testcases (F32_F32, BF16_F32, BF16_BF16, F16_F16, F16_F32, F32_F32_Stride, BF16_F32_Stride, BF16_BF16_Stride, F16_F16_Stride)

> **Note:** F16 tests (F16_F16, F16_F32, F16_F16_Stride) require **AVX512-FP16** support. On unsupported platforms, these tests are automatically skipped via `GTEST_SKIP()` with an informative message.

1. Run all BF16 Input, F32 Output matmul tests:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.BF16_F32/*
```
2. Run all BF16 Input, BF16 Output matmul tests:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.BF16_BF16/*
```
3. Run all F32 Input, F32 Output matmul tests:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F32_F32/*
```
4. Run F32_F32_Stride matmul tests:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F32_F32_Stride/*
```
5. Run all F16 Input, F16 Output matmul tests:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F16_F16/*
```
6. Run all F16 Input, F32 Output matmul tests:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F16_F32/*
```
7. Run F16_F16_Stride matmul tests:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F16_F16_Stride/*
```

### Embedding Bag Tests
 - Embedding Bag TestSuite has the following testcases: F32_F32, F32_BF16, F32_F16, BF16_F32, BF16_BF16, F16_F32, F16_F16, INT8_F32, INT8_BF16, INT8_F16, S4_F32, S4_BF16, S4_F16, U4_F32, U4_BF16, U4_F16

> **Note:** F16 tests (F32_F16, F16_F32, F16_F16, INT8_F16, S4_F16, U4_F16) require **AVX512-FP16** support. On platforms without AVX512-FP16, these tests are automatically skipped (`GTEST_SKIP`).

1. Run all F32 Input, F32 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.F32_F32/*
```
2. Run all F32 Input, BF16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.F32_BF16/*
```
3. Run all F32 Input, F16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.F32_F16/*
```
4. Run all BF16 Input, F32 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.BF16_F32/*
```
5. Run all BF16 Input, BF16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.BF16_BF16/*
```
6. Run all F16 Input, F32 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.F16_F32/*
```
7. Run all F16 Input, F16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.F16_F16/*
```
8. Run all INT8 Input, F32 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.INT8_F32/*
```
9. Run all INT8 Input, BF16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.INT8_BF16/*
```
10. Run all INT8 Input, F16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.INT8_F16/*
```
11. Run all S4 Input, F32 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.S4_F32/*
```
12. Run all S4 Input, BF16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.S4_BF16/*
```
13. Run all S4 Input, F16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.S4_F16/*
```
14. Run all U4 Input, F32 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.U4_F32/*
```
15. Run all U4 Input, BF16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.U4_BF16/*
```
16. Run all U4 Input, F16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.U4_F16/*
```

### Embedding Tests
 - Embedding TestSuite has the following testcases: F32_F32, F32_BF16, F32_F16, BF16_F32, BF16_BF16, F16_F32, F16_F16, INT8_F32, INT8_BF16, INT8_F16, S4_F32, S4_BF16, S4_F16, U4_F32, U4_BF16, U4_F16

> **Note:** F16 tests (F32_F16, F16_F32, F16_F16, INT8_F16, S4_F16, U4_F16) require **AVX512-FP16** support. On platforms without AVX512-FP16, these tests are automatically skipped (`GTEST_SKIP`).

1. Run all F32 Input, F32 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.F32_F32/*
```
2. Run all F32 Input, BF16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.F32_BF16/*
```
3. Run all F32 Input, F16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.F32_F16/*
```
4. Run all BF16 Input, F32 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.BF16_F32/*
```
5. Run all BF16 Input, BF16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.BF16_BF16/*
```
6. Run all F16 Input, F32 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.F16_F32/*
```
7. Run all F16 Input, F16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.F16_F16/*
```
8. Run all INT8 Input, F32 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.INT8_F32/*
```
9. Run all INT8 Input, BF16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.INT8_BF16/*
```
10. Run all INT8 Input, F16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.INT8_F16/*
```
11. Run all S4 Input, F32 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.S4_F32/*
```
12. Run all S4 Input, BF16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.S4_BF16/*
```
13. Run all S4 Input, F16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.S4_F16/*
```
14. Run all U4 Input, F32 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.U4_F32/*
```
15. Run all U4 Input, BF16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.U4_BF16/*
```
16. Run all U4 Input, F16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.U4_F16/*
```

### Group Embedding Bag Tests
 - Group Embedding Bag TestSuite has the following testcases: F32_F32, F32_BF16, F32_F16, BF16_F32, BF16_BF16, F16_F32, F16_F16, INT8_F32, INT8_BF16, INT8_F16, S4_F32, S4_BF16, S4_F16, U4_F32, U4_BF16, U4_F16
 - Each parameterized case picks a thread strategy (`batch_threaded`, `table_threaded`, `ccd_threaded`, `hybrid_threaded`) uniformly at random per test parameter and pins it for the call via `ZENDNNL_EMBAG_THREAD_ALGO` (an RAII guard in `group_embag/group_embag_test_helpers.cpp` saves and restores the env var). The dispatcher reads the pinned value back through `embag_config_t::set_env_config()`, so all four schedulers are exercised across runs.

> **Note:** F16 tests (F32_F16, F16_F32, F16_F16, INT8_F16, S4_F16, U4_F16) require **AVX512-FP16** support. On platforms without AVX512-FP16, these tests are automatically skipped (`GTEST_SKIP`).

1. Run all group embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=GroupEmbag/TestGroupEmbag.*/*
```
2. Run all F32 Input, F32 Output group embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=GroupEmbag/TestGroupEmbag.F32_F32/*
```
3. Run all BF16 Input, BF16 Output group embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=GroupEmbag/TestGroupEmbag.BF16_BF16/*
```
4. Run all U4 Input, BF16 Output group embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=GroupEmbag/TestGroupEmbag.U4_BF16/*
```

### Group Embedding Tests
 - Group Embedding TestSuite has the following testcases: F32_F32, F32_BF16, F32_F16, BF16_F32, BF16_BF16, F16_F32, F16_F16, INT8_F32, INT8_BF16, INT8_F16, S4_F32, S4_BF16, S4_F16, U4_F32, U4_BF16, U4_F16
 - Exercises lookup mode (algo = none, offsets = nullptr per table) through `group_embedding_bag_direct`.

> **Note:** F16 tests (F32_F16, F16_F32, F16_F16, INT8_F16, S4_F16, U4_F16) require **AVX512-FP16** support. On platforms without AVX512-FP16, these tests are automatically skipped (`GTEST_SKIP`).

1. Run all group embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=GroupEmbedding/TestGroupEmbedding.*/*
```
2. Run all F32 Input, F32 Output group embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=GroupEmbedding/TestGroupEmbedding.F32_F32/*
```
3. Run all BF16 Input, BF16 Output group embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=GroupEmbedding/TestGroupEmbedding.BF16_BF16/*
```
4. Run all U4 Input, BF16 Output group embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=GroupEmbedding/TestGroupEmbedding.U4_BF16/*
```

### Normalization Tests
 - Normalization TestSuite has seven testcases (F32_F32, BF16_BF16, BF16_F32,
   F32_BF16, F16_F16, F16_F32, F32_F16)
 - Supports four normalization types: LayerNorm, RMSNorm, FusedAddRMSNorm, BatchNorm
 - Test parameters (shape, norm_ndims, use_scale, use_shift, gamma/beta data types) are randomly generated
 - Validates native kernel output against the reference (scalar) kernel output
 - F16 tests skip at runtime if the platform lacks AVX512-FP16 ISA support
   (returns `status_t::isa_unsupported`)

1. Run all F32 Input, F32 Output normalization tests:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/TestNormalization.F32_F32/*
```
2. Run all BF16 Input, BF16 Output normalization tests:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/TestNormalization.BF16_BF16/*
```
3. Run all BF16 Input, F32 Output normalization tests:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/TestNormalization.BF16_F32/*
```
4. Run all F32 Input, BF16 Output normalization tests:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/TestNormalization.F32_BF16/*
```
5. Run all F16 Input, F16 Output normalization tests:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/TestNormalization.F16_F16/*
```
6. Run all F16 Input, F32 Output normalization tests:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/TestNormalization.F16_F32/*
```
7. Run all F32 Input, F16 Output normalization tests:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/TestNormalization.F32_F16/*
```
8. Run all normalization tests:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/*
```
9. Run normalization tests with 8 threads:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/* --num_threads 8
```

### OMP API Tests
 - OMP API TestSuite validates the OpenMP thread-control utilities in `omp_thread_control.hpp`
 - Uses fixed `TEST_F` tests (not parameterized) since each test targets a specific code path or contract
 - Self-contained: does not depend on `gtest_utils.hpp` or randomized data from `gtest_main.cpp`
 - The `OmpApiTest` fixture manages OpenMP state (disables `omp_set_dynamic`, restores ICVs between tests)

#### Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| **resolve_num_threads** | 6 tests | Pure logic tests for the thread-count resolution function (auto, explicit, single-thread, boundary, single-core, negative) |
| **thread_guard::max_threads** | 3 tests | Singleton caching: stability across repeated calls, cache immunity to external ICV mutation, thread-safe static init |
| **thread_guard RAII** | 8 tests | Set/restore correctness: no-op elision, two-arg top-level, single-arg capture, per-task ICV inside parallel regions, loop pattern, nested guards, sequential guards, over-request restore |
| **Combined / Production** | 2 tests | Full operator entry-point pattern (`resolve_num_threads` + `thread_guard`), combined `thread_guard` + `scoped_active_levels` (group_gemm pattern) |
| **scoped_active_levels** | 2 tests | RAII correctness: no-op elision when desired equals current, set/restore of max-active-levels ICV |

#### Environment Setup

Set these environment variables before running OMP API tests to ensure deterministic behavior:
```bash
export OMP_NUM_THREADS=8
export OMP_MAX_ACTIVE_LEVELS=2
```

#### Running OMP API Tests

1. Run all OMP API tests:
```bash
./install/gtests/gtests --gtest_filter='OmpApiTest.*'
```
2. Run only resolve_num_threads tests:
```bash
./install/gtests/gtests --gtest_filter='OmpApiTest.ResolveNumThreads*'
```
3. Run only thread_guard RAII tests:
```bash
./install/gtests/gtests --gtest_filter='OmpApiTest.*ThreadGuard*'
```
4. Run only scoped_active_levels tests:
```bash
./install/gtests/gtests --gtest_filter='OmpApiTest.ScopedActiveLevels*'
```
5. Run only the production pattern and combined tests:
```bash
./install/gtests/gtests --gtest_filter='OmpApiTest.ProductionPattern*:OmpApiTest.Combined*'
```
6. Run only the parallel-region tests (per-task ICV, loop pattern, concurrent max_threads):
```bash
./install/gtests/gtests --gtest_filter='OmpApiTest.*ParallelRegion*:OmpApiTest.*PerTaskIcv*:OmpApiTest.*ParallelThreads*'
```

### Softmax Tests
 - Softmax TestSuite has two testcases (F32_F32, BF16_BF16)
 - Validates **OneDNN kernel** output against the **reference kernel** output
 - Randomized per-parameter: `ndims` (1D–5D), `shape`, `log_softmax`, `softmin`, `num_threads`
 - Fixed: `axis = -1` (last-axis only); dtype is selected by the test case (F32_F32 or BF16_BF16), not randomized
 - Supports both standard softmax and log-softmax variants, plus the `softmin` mode
   (computes `softmax(-x)`; composes with `log_softmax` when both are true)

1. Run all softmax tests:
``` bash
./install/gtests/gtests --gtest_filter=Softmax/*
```
2. Run all F32 Input, F32 Output softmax tests:
``` bash
./install/gtests/gtests --gtest_filter=Softmax/TestSoftmax.F32_F32/*
```
3. Run all BF16 Input, BF16 Output softmax tests:
``` bash
./install/gtests/gtests --gtest_filter=Softmax/TestSoftmax.BF16_BF16/*
```
4. Run softmax tests with fixed seed and thread count:
``` bash
./install/gtests/gtests --gtest_filter=Softmax/* --seed 42 --num_threads 4
```
5. Run 100 softmax tests:
``` bash
./install/gtests/gtests --gtest_filter=Softmax/* --test 100
```

### Example with more Arguments Support
1. Run the 23rd BF16 Input, BF16 Output matmul test with seed 1245:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.BF16_BF16/23 --seed 1245
```
2. Run the 23rd BF16 Input, BF16 Output matmul test with seed 1245 and relu post-operation:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.BF16_BF16/23 --seed 1245 --postop relu
```
3. Run the BF16 Input, BF16 Output matmul test for 10 randomly generated valid tests with random seed and random postop selection:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.BF16_BF16/* --test 10
```
4. Run F32 matmul tests with AOCL DLP backend:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F32_F32/* --backend aocl_dlp
```
5. Run matmul tests with oneDNN backend and LOWOHA enabled:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F32_F32/* --backend onednn --lowoha true
```
6. Run matmul tests using input file configurations for 2D matmul:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/* --input_file test.txt --op matmul
```
7. Run batch matmul tests using input file configurations for 3D matmul:
``` bash
./install/gtests/gtests --gtest_filter=BatchMatmul/* --input_file batch_tests.txt --op matmul --ndims 3
```
8. Run regular reorder tests from an input file:
``` bash
./install/gtests/gtests --gtest_filter=Reorder/* --input_file input.txt --op reorder --lowoha false
```
9. Run LOWOHA reorder tests from an input file:
``` bash
./install/gtests/gtests --gtest_filter=Reorder/* --input_file input.txt --op reorder --lowoha true
```
10. Run embedding bag tests from an input file:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/* --input_file input.txt --op embeddingbag
```
11. Run embedding tests from an input file:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/* --input_file input.txt --op embedding
```
12. Run normalization tests from an input file:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/* --input_file input.txt --op normalization
```
13. Run matmul tests with 16 threads:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F32_F32/* --num_threads 16
```
14. Run input-file matmul with fixed thread count and test count:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/* --input_file matmul_input.txt --op matmul --num_threads 8 --test 1
```

### Run All testcases of a TestSuite
```bash
./install/gtests/gtests --gtest_filter=<TestSuite>/*
```

### Run All available unit tests
```bash
./install/gtests/gtests
```

## **Future Enhancements**
- Add dynamic tolerance values based on tensor dimensions, data type, and value range.
- Implement a command-line parser to avoid hardcoding parameters.
