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

/// MXFP4 routing.  f4_e2m1 weights have one backend, the custom kernel, and
/// this module owns the decisions that keep them on it:
///
///   * a grouped call runs ALGO 3, the one scheduler that engages the custom
///     kernel (`kGrpMatmulMxfp4Algo`, applied by `group_matmul_algo_select`);
///   * `grp_matmul_mxfp4_admit` refuses, before any work and with the
///     reason, a call the kernel cannot serve;
///   * `n_tile_execute` fails a call the kernel declines at run time
///     (`kGrpMatmulErrMxfp4NoCk`), so no refusal becomes a fallback;
///   * `matmul_direct` sends every MXFP4 call through the one-op grouped
///     route.
///
/// Serving MXFP4 on another backend starts here.

#ifndef ZENDNNL_GROUP_MATMUL_SELECTION_MXFP4_ROUTE_HPP
#define ZENDNNL_GROUP_MATMUL_SELECTION_MXFP4_ROUTE_HPP

#include <cstddef>
#include <vector>

#include "common/error_status.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

/// The grouped scheduler every MXFP4 call runs on.
inline constexpr int kGrpMatmulMxfp4Algo = 3;

/// The one pass over the active experts in `[0, num_ops)` that runs before
/// admission, with diagnostics on or off: fails an expert that uses an MX
/// type outside its role (`mx_types_in_role`), and sets `has_mxfp4` when any
/// expert carries f4_e2m1 weights, so admission sees a call that mixes
/// weight types.
status_t grp_matmul_scan_mx_types(const std::vector<int> &M,
        const std::vector<matmul_params> &params, size_t num_ops,
        bool &has_mxfp4);

/// True when the first active expert carries f4_e2m1 weights.  After
/// `grp_matmul_mxfp4_admit` a call is MXFP4 for every expert or for none, so
/// this is the O(1) test for everything downstream of admission.
bool grp_matmul_is_mxfp4(const std::vector<int> &M,
        const std::vector<matmul_params> &params, size_t num_ops);

/// Refuses, before any work, an MXFP4 call the custom kernel cannot serve:
///   * an active expert that is not f4_e2m1, or a sequential chain;
///   * an expert whose weight scale is not f8_e8m0 `{K/32, N}`, which the
///     pack reads in full;
///   * an expert outside the rest of the contract (`validate_mxfp4_inputs`);
///   * a grouped scheduler pinned to a scheduler ALGO other than 3, globally
///     or for the call's phase;
///   * the custom kernel switched off, or a CPU without AVX-512 VNNI
///     (`status_t::isa_unsupported`);
///   * more active experts than the kernel takes in one call;
///   * a transposed source, alpha != 1, beta != 0, or an N the kernel's
///     pack width does not divide;
///   * anything the N-tile executor cannot slice (`m_tile_supported`,
///     `n_tile_supported`).
/// Refusals the kernel can only make at preparation time surface from
/// `n_tile_execute` as a failed call.
status_t grp_matmul_mxfp4_admit(const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K,
        const std::vector<float> &alpha, const std::vector<float> &beta,
        const std::vector<bool> &is_weights_const,
        const std::vector<matmul_params> &params, size_t num_ops,
        grp_matmul_gated_act_t fused_act, bool sequential_chain);

/// Whether the custom kernel can fuse `act` into an MXFP4 call's store: its
/// gated stores write bf16 (`act_dtype`), and a split-halves activation
/// (silu, gelu) takes no bias there.  Where it cannot, MXFP4, which has no
/// other backend, runs the plain kernel into a wide destination and the
/// activation as a separate pass.
bool grp_matmul_mxfp4_fuses_gated_store(const std::vector<int> &M,
        const std::vector<matmul_params> &params, grp_matmul_gated_act_t act,
        data_type_t act_dtype);

/// Refuses, with the reason, a gated activation into a tight destination
/// (ldc < N) whose store the kernel cannot fuse: such a destination has no
/// room to run the activation after the kernel.
status_t grp_matmul_mxfp4_admit_tight_gated_store(const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &ldc,
        const std::vector<matmul_params> &params, size_t num_ops,
        grp_matmul_gated_act_t act);

/// The operand half of `grp_matmul_mxfp4_admit`, also run on a projection the
/// caller builds internally (the fused-MoE down projection): every active
/// expert is f4_e2m1 with an f8_e8m0 `{K/32, N}` scale (`mxfp4_scale_view`),
/// the pack width divides every N, and the expert count fits one call.
status_t grp_matmul_mxfp4_admit_operands(const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K,
        const std::vector<matmul_params> &params, size_t num_ops);
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_SELECTION_MXFP4_ROUTE_HPP
