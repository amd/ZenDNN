/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/// Library-internal helpers shared by the generic scheduling-ALGO paths.
///
/// Each ALGO implementation (`sequential_experts`, `flat_m_tile`,
/// `flat_n_tile`, `parallel_multilevel`, `parallel_per_expert`) is
/// split into its own translation unit so the files stay small and
/// ownership is clear.  This header hosts the bits they all need:
///   - env-driven feature flags
///   - the `resolve_kernel()` / `execute_expert_slice()` primitives
///   - tile-size constants referenced by ALGO 0 auto-select
///   - forward declarations of the per-strategy entry points
///
/// None of these symbols are part of the public ZenDNN API.
///
/// The code lives in the headers included below; this umbrella includes
/// them all so existing includers compile unchanged.

#ifndef ZENDNNL_GROUP_MATMUL_PARALLEL_COMMON_HPP
#define ZENDNNL_GROUP_MATMUL_PARALLEL_COMMON_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "../group_matmul_direct.hpp"
#include "common/op_config.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_env.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_expert_slice.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_n_split.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ck_enable.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/multi_expert_policy.hpp"
#include "lowoha_operators/matmul/group_matmul/selection/algo_select.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"

#endif // ZENDNNL_GROUP_MATMUL_PARALLEL_COMMON_HPP
