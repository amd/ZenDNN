# *******************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

# Group matmul: parallel expert dispatch, MoE post-op, shared header.
# Paths are relative to lowoha_operators/matmul/ (parent listfile).

set(lowoha_group_matmul_sources
  group_matmul/selection/group_matmul_dispatch.cpp
  group_matmul/selection/algo_select.cpp
  group_matmul/selection/mxfp4_route.cpp
  group_matmul/executors/sequential/sequential_experts.cpp
  group_matmul/executors/m_tile/group_matmul_m_tile.cpp
  group_matmul/executors/n_tile/group_matmul_n_tile.cpp
  group_matmul/executors/n_tile/n_tile_admission.cpp
  group_matmul/executors/n_tile/src_group_sums.cpp
  group_matmul/multi_expert/expert_parallel/group_matmul_expert_parallel.cpp
  group_matmul/multi_expert/multilevel/parallel_multilevel.cpp
  group_matmul/multi_expert/multi_expert_policy.cpp
  group_matmul/single_expert/single_expert_policy.cpp
  group_matmul/single_expert/single_expert_plan.cpp
  group_matmul/single_expert/single_expert_executor.cpp
  group_matmul/epilogue/group_matmul_moe_postop.cpp
  group_matmul/epilogue/group_matmul_moe_act.cpp
  group_matmul/fused_moe/group_matmul_fused_moe.cpp
  group_matmul/fused_moe/vertical_fusion_pipeline.cpp
  group_matmul/prepack/prepack.cpp
  group_matmul/prepack/prepack_aocl_dlp.cpp
  group_matmul/prepack/prepack_custom_kernel.cpp
  group_matmul/custom_kernel/pack.cpp
  group_matmul/custom_kernel/qwen_gemma_llama/bf16_decode_policy.cpp
  group_matmul/custom_kernel/ukernel/bf16_microkernel.cpp
  group_matmul/custom_kernel/qwen_gemma_llama/bf16_decode_microkernel.cpp
  group_matmul/custom_kernel/ukernel/int8_microkernel.cpp
  group_matmul/custom_kernel/ukernel/s4_microkernel.cpp
  group_matmul/custom_kernel/ukernel/mxfp4_microkernel.cpp
  group_matmul/custom_kernel/ukernel/s8_group_sums.cpp
  group_matmul/custom_kernel/ukernel/f16_microkernel.cpp
  group_matmul/custom_kernel/dispatch.cpp
  group_matmul/custom_kernel/ck_enable.cpp
  group_matmul/common/grp_matmul_dot_measure.cpp
  group_matmul/common/grp_matmul_machine.cpp
  group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel_config.cpp
  group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel.cpp
  group_matmul/multi_expert/ntile_flat_parallel/nr32_pack.cpp
  group_matmul/multi_expert/ntile_flat_parallel/nr32_tile.cpp
  group_matmul/multi_expert/ntile_flat_parallel/nr32_ukernel.cpp
  group_matmul/multi_expert/ntile_flat_parallel/nr32_ukernel_bf16.cpp
  group_matmul/multi_expert/ntile_flat_parallel/nr32_ukernel_f16.cpp
  group_matmul/multi_expert/ntile_flat_parallel/nr32_ukernel_s4.cpp
  group_matmul/multi_expert/ntile_flat_parallel/nr32_ukernel_s8.cpp
  group_matmul/executors/n_tile/prompt_tile/prompt_tile_topology.cpp
  group_matmul/executors/n_tile/prompt_tile/prompt_tile_tunables.cpp
  group_matmul/executors/n_tile/prompt_tile/prompt_tile_executor.cpp)

# AOCL-DLP backed MoE gated activations: only built when AOCL-DLP is enabled.
if(ZENDNNL_DEPENDS_AOCLDLP)
  list(APPEND lowoha_group_matmul_sources
    group_matmul/epilogue/group_matmul_moe_act_dlp.cpp)
endif()

set(lowoha_group_matmul_sources ${lowoha_group_matmul_sources} PARENT_SCOPE)

set(lowoha_group_matmul_public_headers
  group_matmul/group_matmul_direct.hpp
  PARENT_SCOPE)
