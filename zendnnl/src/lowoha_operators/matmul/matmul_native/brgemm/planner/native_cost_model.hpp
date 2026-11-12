/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_COST_MODEL_HPP
#define MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_COST_MODEL_HPP

#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_config.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Threads of one L3 domain the plan of p takes: domain_threads, else
/// UarchParams::l3_domain_cores, else family_l3_cores; at least 1.
int native_plan_domain_threads(
        const UarchParams &uarch, const heuristic_params_t &hp);

/// Cycles per KiB one thread streams from memory while its L3 domain
/// streams: cm_mem_cycles, or, at 0, 2400 over the square root of
/// native_plan_domain_threads.
double native_plan_mem_cycles(
        const UarchParams &uarch, const heuristic_params_t &hp);

/// Percent more cycles per row a call of more than cm_short_rows rows
/// takes: cm_tall_pct, or, at -1, 12.5 per doubling of
/// native_plan_domain_threads past 8, at most 50.
double native_plan_tall_pct(
        const UarchParams &uarch, const heuristic_params_t &hp);

/// The modeled time of a plan, in cycles of the busiest worker.
struct native_plan_cost_t {
    /// Kernel calls, slab re-reads of the later calls of each K pass and
    /// source rows a tile reads from L3.
    double core = 0;
    /// The weight tiles the worker streams, from memory or, for the second
    /// read of a tile in one L3 domain, from L3, and the time its weight
    /// stream idles behind the calls after the first of each K pass: of
    /// every tile, or of its first cm_sync_tiles tiles where one core
    /// streams faster than cm_mem_cycles (the domain's other streams fill
    /// the link).
    double memory = 0;
    /// max(core, memory) plus cm_overlap_pct percent of the smaller.
    double cycles = 0;
    /// Of memory, the cycles of the worker's share of its domain's weight
    /// stream (and of its second reads from L3), without the idle.
    double stream = 0;
    /// KiB of that share.
    double stream_kib = 0;
};

/// True where one core streams weights slower than its share of its L3
/// domain's link: native_plan_mem_cycles below cm_stream_cycles.  At a tie
/// (16 threads to a domain at the defaults) neither this nor the link's
/// filling of the idle (native_plan_cost_t::memory) holds.
bool native_plan_core_streams(
        const UarchParams &uarch, const heuristic_params_t &hp);

/// True for the INT8 tiles the model prices: per-channel weight scales (the
/// INT8 tile's calls, K passes and weight bytes; the per-group tile batches
/// its groups instead).
bool native_plan_cost_int8(const native_problem_t &p);

/// Bytes the INT4 tile streams per weight element of p: half a byte plus
/// the group's scale (wei_scale_bytes over the group size).
double native_int4_weight_bytes(const native_problem_t &p);

/// True while the weights of p (N x K elements of the weight type) take at
/// most cm_group_l3_pct percent of the L3 of the domains plan q's team
/// spans (native_plan_domain_threads threads to a domain).
bool native_plan_groups_fit_l3(const native_problem_t &p,
        const native_plan_t &q, const UarchParams &uarch,
        const heuristic_params_t &hp);

/// True while the busiest worker of the slice of q (ceil(n_tiles / threads)
/// column tiles) takes at least cm_group_skew_pct percent of the mean
/// column tiles per thread.
bool native_plan_slice_skewed(
        const native_plan_t &q, const heuristic_params_t &hp);

/// The modeled time of plan q of a BF16 or F16 problem p, of an INT8 one
/// that native_plan_cost_int8 accepts, or of an s4 one, on uarch with the
/// cost inputs of hp; zero for another.  An F16 call takes one FMA step a K
/// row of two-byte weights, over source rows widened to f32 (four bytes an
/// element); on the F16_PH kernels (q.kind) one FMA a K row of 32 weights,
/// over its two-byte source rows.  An INT8 call takes K in quads at the
/// plan's call width and rows per call, a weight element is one byte, and
/// the K pass follows the tile's rows (native_int8_pass_quads).  An s4 call
/// is priced on its per-group tile: the INT8 call over all of K, weight
/// elements of native_int4_weight_bytes, and per group one FMA per row and
/// output vector.  Weight tiles a worker visits are streamed once; under row
/// groups each tile is read once per row group, from memory once per L3
/// domain the threads of its column group span (the threads of the team
/// taken in order, domain_threads to a domain), and from L3 for the other
/// reads of the domain while the domain's weight bytes times cm_reuse_pct
/// percent fit its l3_pct share.  A plan of the blocks schedule is weighed
/// as the slice of its grid.  No allocation.
native_plan_cost_t native_plan_cost(const native_plan_t &q,
        const native_problem_t &p, const UarchParams &uarch,
        const heuristic_params_t &hp);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_COST_MODEL_HPP
