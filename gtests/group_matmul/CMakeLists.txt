# ********************************************************************************
# * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Lightweight CMakeLists invoked from the parent gtests/CMakeLists.txt
# via add_subdirectory(group_matmul).  Appends sources + headers to the
# existing `gtests` target so the binary stays single-output.
#
# Layout: every group_matmul-specific gtest artifact lives under this
# folder.  See group_matmul/README.md for the maintainer overview.
target_sources(gtests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/group_matmul_test_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e/test_basic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e/test_gated_act_forms.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fused_moe/test_fused_moe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_algos.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e/test_quant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e/test_group_per_group.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e/test_ggml_per_group.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fused_moe/test_fused_moe_ggml.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/prepack/test_prepack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e/test_w4a8_per_group.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e/test_e2m1_group_matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e/test_group_reorder_e2e.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/e2e/test_wei_buffer_capacity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/prepack/test_int8_inplace_warm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/executors/n_tile/prompt_tile/test_prompt_tile.cpp
    # Custom-kernel direct-surface tests — see gtests/group_matmul/
    # custom_kernel/ck_test_helpers.hpp for scope (unit-level on
    # `bf16_isa_supported`, `resolve_variant`, `prepare_for_call`,
    # `plan_pack_nr`, end-to-end correctness for both BF16 and FP32
    # dst, feature-integration on resolved CallContext fields, and
    # CK -> DLP fallback chain).
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_dispatch_supported.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_resolve_variant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_prepare_for_call.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_pack_bf16.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_ukernel_bf16.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_matmul_unreorder_route.cpp
    # FP16 sibling suites -- mirror test_ukernel_bf16.cpp /
    # test_pack_bf16.cpp for the native AVX-512-FP16 microkernel family
    # (f16:f16:f16 / f16:f16:f32) and its disjoint f16 pack cache, gated
    # on CK_SKIP_IF_NO_F16_ISA() so they skip cleanly on hosts /
    # toolchains without AVX-512-FP16.
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_ukernel_f16.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_pack_f16.cpp
    # DQ-INT8 sibling suites — mirror the bf16 set 1:1 (microkernel
    # direct-surface correctness + pack module coverage with K/4
    # interleave + compensation row).
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_pack_int8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_ukernel_int8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_pack_w4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_ukernel_s4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_ukernel_mxfp4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_features.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/test_dispatch_routing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel/qwen_gemma_llama/test_bf16_decode.cpp
    # Single-expert policy (selector rules, phase knobs, dtype masks) and its
    # `matmul_direct` route; executor-level single-expert tests stay beside
    # their executors.
    ${CMAKE_CURRENT_SOURCE_DIR}/single_expert/test_single_expert_policy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/single_expert/test_single_expert_route.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/single_expert/test_single_expert_executor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_expert/test_multi_expert_policy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_expert/ntile_flat_parallel/test_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_expert/ntile_flat_parallel/test_format.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_expert/ntile_flat_parallel/test_grouped.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_expert/ntile_flat_parallel/test_nr32.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/routed_moe/test_routed_moe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/routed_moe/test_routed_moe_s4.cpp
)
