# ********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Native BRGEMM over native-layout weights: layout rule and route, layout
# and weight view, INT8 (per-channel and per-group), s4 weights, FP32, F16,
# epilogue, route parity, refused calls, what matmul_direct's checks take for
# it, weights of any N, gated calls over the plain layout, planner config
# and tuning file, plan, cost model, partition and execution, hardening,
# weight lifetime and group-matmul scope suites. Appended to the parent
# `gtests` target.
target_sources(gtests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/native_brgemm_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_layout_rule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_brgemm_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_brgemm_fp32.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_brgemm_f16.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_brgemm_int8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_brgemm_int8_group.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_brgemm_int8_scales.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_brgemm_int4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_brgemm_epilogue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_brgemm_parity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_brgemm_refusal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_brgemm_checks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_brgemm_any_n.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_gated_pack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_tuning_plan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_cost_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_tuning_rules.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_tuning_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_tuning_exec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_brgemm_hardening.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_weight_lifetime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_brgemm_group.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_matmul_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_gemv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_gemv_operands.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_one_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_postop_chain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_native_cache_fallback.cpp)
