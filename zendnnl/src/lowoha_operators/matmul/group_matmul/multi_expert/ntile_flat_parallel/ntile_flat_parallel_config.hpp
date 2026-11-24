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

/// N-tile flat parallel's settings.  Their defaults follow group matmul's
/// machine view (grp_matmul_machine.hpp), and
/// `ZENDNNL_GRP_MATMUL_NTILE_FLAT_PARALLEL_CONFIG`, read once at the first
/// N-tile flat parallel call, overrides any of them: clauses
/// `[filters:]key=value,...` separated by `;`, as the native BRGEMM's config
/// takes them.  For each setting, the first clause whose filters hold and that
/// sets it applies.  Filters: `phase=decode|prompt`, `dtype=` a format's name
/// (`nr32_format.hpp`: bf16, f16, s8, s4) and `api=routed|grouped`.  A weight's
/// plans read the plan settings, so those take no phase filter, and the process
/// limits take no filter at all.  A bad clause is dropped with one warning.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_CONFIG_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {

/// The API a call came through.
enum class api_kind : uint8_t { routed, grouped };

/// Which units a decode call's threads take first: `automatic` runs each
/// thread's own units first when the team spans several L3 domains, `queue`
/// takes queue order, `own` always runs its own units first.
enum class unit_order : uint8_t { automatic, queue, own };

/// A setting with a machine-derived default.
enum class switch3 : uint8_t { automatic, off, on };

/// What a call's unit queue reads.
struct call_knobs {
    /// Units per thread a decode call's queue aims for.
    int units = 2;
    /// Units per thread a prompt call cuts each projection into.
    int prompt_units = 6;
    /// Rows above which a call's units sweep L2-sized column chunks.
    int sweep_rows = kDecodeMaxM;
    unit_order order = unit_order::automatic;
    /// Whether a sweeping call's queue alternates its heaviest and lightest
    /// experts' units.
    bool heavy_light = true;
};

/// What a weight's plans read: fixed for a (weight type, API) once read.
struct plan_knobs {
    /// The share of L2 a kernel's weights may fill; 0 keeps the machine
    /// view's budget (80%).
    double l2_frac = 0.0;
    /// Source rows the L2 budget reserves beside the weights.
    int subtile_rows = 16;
    /// DA8W8's kernels that read sources already XOR-ed with 0x80.
    switch3 u8_src = switch3::automatic;
};

/// Process-wide limits.
struct limits_t {
    /// Rows a W13 unit copies at a time when it shares a run's gather.
    int gather_rows = 16;
    /// Pauses a unit spends waiting for another thread's rows before it
    /// starts yielding the core.
    int wait_spins = 4096;
    /// A calling thread's scratch above this many bytes is freed after a
    /// call.
    size_t scratch_keep_bytes = size_t {1} << 30;
    /// Grouped call sites kept before they are all dropped.
    size_t call_sites = 1024;
    /// Prepack-extras pool fingerprints kept before they are all dropped.
    size_t warmed_pools = 4096;
};

/// The settings a call of `phase` on `wei` weights through `api` reads.
const call_knobs &call_knobs_for(
        grp_matmul_phase phase, data_type_t wei, api_kind api);

/// The settings plans of `wei` weights for `api` read.
const plan_knobs &plan_knobs_for(data_type_t wei, api_kind api);

const limits_t &limits();

/// The L2 bytes plans of `wei` weights for `api` give their weights.
int64_t l2_budget_bytes(data_type_t wei, api_kind api);

/// Whether plans of `wei` weights for `api` run the u8-source kernels: never
/// for a format without them, which DA8W8's s8 alone has.
bool u8_src(data_type_t wei, api_kind api);

} // namespace ntile_flat_parallel

namespace test_api {

/// Test-only: reads `text` as N-tile flat parallel's config (nullptr: the
/// environment's), as the first N-tile flat parallel call would.  Plans made
/// before keep their settings, so flush N-tile flat parallel first.  Not safe
/// while N-tile flat parallel calls run.
void set_ntile_flat_parallel_config(const char *text);

/// The warnings N-tile flat parallel's config last read gave, one per dropped
/// clause.
std::vector<std::string> ntile_flat_parallel_config_warnings();

} // namespace test_api

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_CONFIG_HPP
