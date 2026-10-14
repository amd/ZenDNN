/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// @file prompt_tile_executor.hpp
/// @brief ALGO 3 prompt tile executor (custom kernel, fat-M calls).
///
/// Replaces the round executor for prompt calls that the custom kernel
/// (CK) serves, for each dtype family whose `enable` bit is set (BF16 and
/// FP16 by default; INT8 and W4A8 opt in).  The call is cut into (expert,
/// M-block, N-slice) tiles (`prompt_tile_planner.hpp`); each L3 team walks
/// its own contiguous, equal-cost range through a shared atomic cursor,
/// then drains other teams' ranges.  One OMP region, no per-round barriers.
///
/// Output is bitwise equal to the round executor: every tile runs the
/// same `custom_kernel::dispatch_tile` on a row / column sub-range (INT8
/// and W4A8 on the hoisted s8 / u8 A and its per-row scales), and the
/// planner keeps every M-block in the microkernel regime a full-M call
/// uses (see `FamilyFacts::min_mr`).  Weights stay in the CK's shared
/// pack; no new copies.  Knobs: `prompt_tile_tunables.hpp`.

#ifndef ZENDNNL_GROUP_MATMUL_PROMPT_TILE_EXECUTOR_HPP
#define ZENDNNL_GROUP_MATMUL_PROMPT_TILE_EXECUTOR_HPP

#include <vector>

#include "../custom_kernel/dispatch.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace prompt_tile {

using zendnnl::common::data_type_t;

/// An INT8 or W4A8 expert's hoisted A and CK scale views, as the round
/// executor hands them to `dispatch_tile`.
struct QuantRow {
    const void *src = nullptr; ///< s8 / u8 A.
    int lda = 0;
    const void *src_scale = nullptr; ///< Per row, read per `kctx.scale_kind`.
    const int32_t *src_zp = nullptr; ///< Per row when asymmetric, else null.
    const void *wei_scale = nullptr; ///< Whole; the CK slices it by column.
};

/// Borrowed views of `flat_n_tile`'s per-expert arguments.
struct CallView {
    const std::vector<int> &M;
    const std::vector<int> &N;
    const std::vector<int> &K;
    const std::vector<const void *> &src;
    const std::vector<int> &lda;
    const std::vector<const void *> &bias;
    const std::vector<void *> &dst;
    const std::vector<int> &ldc;
    /// One per expert for INT8 and W4A8, else null.
    const QuantRow *quant = nullptr;
};

/// True when the call should run on the prompt tile executor: CK engaged
/// with a variant whose family bit is set in the `enable` knob, a
/// prompt-phase call that the planner sent to the round executor, deep-K
/// blocking off for a family the CK K-blocks (its use depends on the
/// per-call M), and every active expert's N a multiple of pack_nr.  A call
/// with one active expert stays on the round executor, except a BF16 call of
/// exactly one expert (a dense layer) while
/// `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_PROMPT_TILE` is on.
bool should_run(const custom_kernel::CallContext &kctx, bool use_custom,
        bool rounds_strategy, bool prompt_phase, const std::vector<int> &M,
        const std::vector<int> &N);

/// Run every tile of the call on an OMP region of `num_threads`.
/// `log_plan` emits one `[GRP_MATMUL.PLAN.PTILE]` apilog line.
void execute(const custom_kernel::CallContext &kctx, const CallView &call,
        int num_threads, bool log_plan);

/// `gemm_mode` literal for a prompt-tile call (keeps the `flat_n_tile`
/// prefix, so it maps to exec_algo=3).
const char *gemm_mode_label(bool fused_epilogue, grp_matmul_gated_act_t act);

struct Plan;

namespace test_api {
/// Plans `call` as `execute` does, without running it (the planner
/// micro-benchmark times this).  The plan lives in the calling thread's
/// scratch until its next call.
const Plan &plan_only(const custom_kernel::CallContext &kctx,
        const CallView &call, int num_threads);
} // namespace test_api

} // namespace prompt_tile
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_PROMPT_TILE_EXECUTOR_HPP
