# `group_matmul/`

This folder implements `group_matmul_direct` and the fused MoE path. Source
files are listed in `CMakeLists.txt`, with paths relative to
`lowoha_operators/matmul/`.

## Layout

- `group_matmul_direct.hpp` is the installed public header and stays at this
  path. `group_matmul_direct.cpp` is the entry point that implements it.
- `common/` holds leaf helpers: the constants (`grp_matmul_constants.hpp`),
  the environment knobs (`grp_matmul_env.hpp`), the N-split helpers
  (`grp_matmul_n_split.hpp`), the per-expert primitives
  (`grp_matmul_expert_slice.hpp`), the per-thread scratch buffer, the AVX-512
  activation math and the internal-allocation check.
  `group_matmul_parallel_common.hpp` is an umbrella that includes these
  pieces together with `selection/algo_select.hpp`,
  `custom_kernel/ck_enable.hpp` and `multi_expert/multi_expert_policy.hpp`, so
  existing includers compile unchanged. New code includes the piece it uses.
- `selection/` selects the algorithm and runs the parallel dispatch.
  `algo_select.hpp` holds the AUTO per-phase settings and declares the
  selector, which `algo_select.cpp` implements.
- `executors/sequential/` is ALGO 1, the sequential executor that is also the
  fallback of every safety clamp.
- `executors/m_tile/` is ALGO 2, the M-tile executor and its planner, and
  `m_tile_internal.hpp` holds the slice helper it shares with the
  vertical-fusion pipeline.
- `executors/n_tile/` is ALGO 3, the N-tile executor, its planner, its
  environment knobs (`n_tile_knobs.hpp`) and its admission check
  (`n_tile_admission.hpp`), and `executors/n_tile/prompt_tile/` is its prompt
  tile executor.
- `single_expert/` holds the single-expert policy, hooks, plan, executor and
  the `matmul_direct` route. The hooks include the single-expert adaptive
  N-tile sizer that ALGO 3 calls.
- `multi_expert/` holds the MoE-only policy (`multi_expert_policy.hpp`: the
  AUTO ALGO 3 layer test, the CcdExpertWaves knobs and gate, and the CCD-fill
  knob) and the MoE-only algorithms:
  `ntile_flat_parallel/` (ALGO 4), `expert_parallel/` (ALGO 5) and
  `multilevel/` (ALGO 6).
- `fused_moe/` is the fused MoE entry and its vertical-fusion pipeline
  (`vertical_fusion_pipeline.hpp`), which runs W13, the gated activation and
  W2 per M-tile slice.
- `epilogue/` holds the gated activations and the MoE weighted-reduce post-op.
- `custom_kernel/` holds the custom microkernels, their weight packing and
  their dispatch, and `ck_enable.hpp` holds the custom-kernel enable knobs.
  `w8a8/` is the ALGO 4 W8A8 pack and microkernel, and `qwen_gemma_llama/`
  holds the Qwen, Gemma and Llama BF16 decode kernels.
- `prepack/` holds the weight prepacking for each algorithm.

## Dependency rules

The layers below are listed lowest first. A file may include anything allowed
for its own layer and for the layers before it, and nothing from a later
layer. New code must not add includes that break these rules. Three known
exceptions remain: the `common/group_matmul_parallel_common.hpp` umbrella,
which includes pieces from later layers; `custom_kernel/dispatch.cpp`, which
includes the single-expert hooks and policy for its K-blocking, prompt MR and
o-block-outer decisions; and `prepack/prepack.cpp`, which includes
`single_expert/single_expert_policy.hpp` for its prepack-layer test.

1. `common/` includes nothing from this folder except
   `group_matmul_direct.hpp`. Outside this folder it may use
   `lowoha_common.hpp`, `lowoha_matmul_utils.hpp`, `quantization/` and
   `common/`.
2. `custom_kernel/` and `epilogue/` may also use `backends/`,
   `matmul_native/common/`, `lru_cache/` and `aocl_unreorder_cache.hpp`.
3. `prepack/` may also include `custom_kernel/`.
4. `executors/` may also include `epilogue/` and `prepack/`. From the
   workload folders it may include only the seam headers
   `single_expert/single_expert_hooks.hpp`,
   `single_expert/single_expert_executor.hpp` and
   `multi_expert/multi_expert_policy.hpp`. It never includes
   `selection/`, `fused_moe/` or any other workload header.
5. `single_expert/` and `multi_expert/` may include anything from layers 1
   to 4, including the executor headers. They never include `selection/`,
   `fused_moe/`, the entry point or each other.
6. `selection/` may include everything except `fused_moe/` and the entry
   point.
7. `fused_moe/` may also include `selection/`.
8. The entry point, `group_matmul_direct.cpp`, may include everything.

Tests in `gtests/group_matmul/` mirror this tree and may include
anything.
