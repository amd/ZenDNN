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

/// Single-expert BLOCKING PLAN.
///
/// How a lone expert's decode on the custom kernel is partitioned: the thread
/// grid (M parts x N parts x K parts), the K chunk and rows per microkernel
/// call inside a work unit, and the loop order.  `make_single_expert_plan` picks it;
/// `single_expert_executor` runs it.  A legacy plan leaves the call on the
/// N-tile executor's own N-only split.

#ifndef ZENDNNL_GROUP_MATMUL_SINGLE_EXPERT_PLAN_HPP
#define ZENDNNL_GROUP_MATMUL_SINGLE_EXPERT_PLAN_HPP

#include <atomic>
#include <cstdint>
#include <string>

namespace zendnnl {
namespace lowoha {
namespace matmul {

enum class SingleExpertSplit : uint8_t {
    /// The N-tile executor's N-only split (no single-expert executor).
    n_tile_default,
    /// Whole NR blocks evenly per thread; the remainder blocks split along K.
    n_remainder_k_split,
    /// M parts x N parts x K parts over the team.
    mnk_grid,
};

enum class SingleExpertLoopOrder : uint8_t {
    /// Inside a work unit: every K chunk of a block, then the next block.
    n_outer,
    /// Inside a work unit: one K chunk across its blocks, then the next.
    k_outer,
};

/// `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_BLOCK`: `key=value` pairs joined by `,`
/// or whitespace (case-insensitive), read once per process.
///   * `split=legacy|remainder|grid` (default `legacy`).
///   * `mparts=<n>` (grid, default 1), `kparts=<n>|auto` (grid default 1;
///     remainder default `auto` = threads / remainder blocks).
///   * `kc=<n>` K chunk per accumulation inside a unit (default 0: the whole
///     range, through the regular kernel when the unit is not K-split).
///   * `mr=<n>` rows per microkernel call, 1..12 (default 0: kernel max).
///   * `order=n|k` loop order inside a unit (default `n`).
/// An unknown key or a bad value is warned once and keeps the default plan.
/// A valid non-empty value, even `split=legacy`, overrides `..._KSPLIT`; an
/// empty value counts as unset.
struct SingleExpertBlockConfig {
    bool given = false; ///< parsed from a valid non-empty value
    SingleExpertSplit split = SingleExpertSplit::n_tile_default;
    int m_parts = 1;
    int k_parts = 0; ///< 0: the split's default
    int kc = 0;
    int mr = 0;
    SingleExpertLoopOrder order = SingleExpertLoopOrder::n_outer;
};

/// Parse a knob value; false (and `*error` set) on a malformed value.
bool parse_single_expert_block(
        const char *value, SingleExpertBlockConfig *knob, std::string *error);

namespace test_api {
/// A knob for the lifetime of a test; nullptr uses the cached env value.
inline std::atomic<const SingleExpertBlockConfig *>
        s_single_expert_block_override {nullptr};
/// `0` / `1` force `ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_KSPLIT`; `-1` env.
inline std::atomic<int> s_grp_matmul_single_expert_ksplit_override {-1};
/// Calls `single_expert_plan_execute` ran, for tests to confirm a plan ran.
inline std::atomic<int> s_single_expert_executor_runs {0};
} // namespace test_api

const SingleExpertBlockConfig &get_single_expert_block_knob();

/// ZENDNNL_GRP_MATMUL_SINGLE_EXPERT_KSPLIT = { 0, 1 } — cached, default 0.
///   With no `..._BLOCK` value, `1` gives a lone expert's decode the
///   `remainder` plan whenever its NR blocks do not divide evenly over the
///   team.  Non-numeric input → default.
bool get_grp_matmul_single_expert_ksplit();

/// Minimum K depth of one K range in an automatic `remainder` plan.
inline constexpr int kSeKsplitMinK = 256;

struct SingleExpertBlockPlan {
    SingleExpertSplit split = SingleExpertSplit::n_tile_default;
    int m_parts = 1;
    int n_parts = 1;
    int k_parts = 1;
    int whole_blocks = 0; ///< remainder: whole blocks per thread
    int split_blocks = 0; ///< remainder: blocks split along K
    int kc = 0;
    int mr = 0;
    SingleExpertLoopOrder order = SingleExpertLoopOrder::n_outer;
    bool active() const { return split != SingleExpertSplit::n_tile_default; }
};

/// Whether any plan but `legacy` can come out of `make_single_expert_plan` in this
/// process (a cached knob test, so the default path pays nothing).
bool single_expert_plan_enabled();

/// The plan for one call.  `k_step` is the custom kernel's K granularity
/// (`custom_kernel::block_k_step`).  Legacy outside a lone expert's decode,
/// for an N the pack width does not divide, or for a plan the shape cannot
/// take.
SingleExpertBlockPlan make_single_expert_plan(int num_ops, int M, int N, int K,
        int pack_nr, int k_step, int num_threads);

/// Short label for logs, e.g. `grid m2 n8 k2 kc1024 mr0 order=n`.
std::string se_block_plan_label(const SingleExpertBlockPlan &plan);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_SINGLE_EXPERT_PLAN_HPP
