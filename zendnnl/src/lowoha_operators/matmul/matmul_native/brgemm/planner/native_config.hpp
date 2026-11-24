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

#ifndef MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_CONFIG_HPP
#define MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_CONFIG_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "lowoha_operators/matmul/matmul_native/common/native_int8_scheme.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// blocks: units of a row block by unit_tiles column tiles, dealt to the
/// team from one counter per L3 domain.
enum class native_schedule_t : uint8_t { automatic, slice, row_groups, blocks };
enum class native_traversal_t : uint8_t { m_inner, n_inner };
enum class native_kc_split_t : uint8_t { budget, equal };

/// INT8 call widths as a mask.
inline constexpr uint8_t kNativeInt8Cw16 = 1;
inline constexpr uint8_t kNativeInt8Cw32 = 2;
inline constexpr uint8_t kNativeInt8CwAll = 3;

/// Every knob of the native BRGEMM plan at its built-in value.  A pin left at
/// its auto value (0, or -1 for col_groups) leaves the decision to the rule
/// that reads the other fields.  heuristic_plan with these defaults is the
/// built-in plan.
struct heuristic_params_t {
    // Tile grid and schedule.
    /// pin; 0: pack_nr, at most 32 (64 for F16_PH by f16_wide_min_m),
    /// halved while tiles < team and one row block, or 64 / 128 by
    /// bf16_narrow_n.  64 and 128 are tiles of 32-column slabs, the widest
    /// kernel call of every type but F16_PH, whose widest call is its tile
    /// of 64 (two pack blocks) and whose 128 is two of those; the rules
    /// take them for BF16 and F16_PH 64 only.
    int nr = 0;
    /// pin; 0: tall_rows for a BF16 tile of 32 columns or more, else
    /// tile_rows
    int mr = 0;
    int mb = 0; ///< pin, no row split; 0: block_rows, then the row split
    /// pin, blocks schedule: columns per unit (the N block); the unit takes
    /// nb / nr column tiles, at least one, under the bounds of unit_tiles,
    /// so its width holds whatever nr the plan takes; set, it replaces
    /// unit_tiles; 0: unit_tiles
    int nb = 0;
    int kc = 0; ///< pin, BF16; 0: the L2 rule
    int col_groups = -1; ///< pin; -1: schedule; 0: slice; > 0: row groups
    native_schedule_t schedule = native_schedule_t::automatic;
    native_traversal_t traversal = native_traversal_t::m_inner;
    /// pin, blocks schedule: column tiles per unit, at most n_tiles and,
    /// under BF16 K passes, the tiles whose accumulators fit the bf16_l2_pct
    /// share of L2; 0: one
    int unit_tiles = 0;
    int threads_cap = 0; ///< 0: the requested team
    int min_tiles_per_thread = 1;
    int row_unit = 12; ///< rows per unit of the row-group ranges
    // Rule constants.
    int tile_rows = 6; ///< BF16 rows per call at nr 16; INT8 split unit
    int tall_rows = 12; ///< BF16 rows per call of a tile of 32 columns or more
    int block_rows = 192; ///< M block before the row split
    int min_tile_cols = 16; ///< narrowest tile of the narrowing
    bool row_split = true;
    /// Shortest K pass of the BF16 L2 rule and of the BF16 and INT8 L1
    /// passes; even.
    int min_pass_k = 256;
    int bf16_l2_pct = 50; ///< L2 share of one BF16 B slab
    native_kc_split_t kc_split = native_kc_split_t::budget;
    int l3_pct = 100; ///< L3 share of the row-group test
    /// Threads of one L3 domain; 0: UarchParams::l3_domain_cores, or
    /// family_l3_cores when that is 0.  Thread w of a team is taken to run
    /// in L3 domain w / domain_threads, as a compact binding of the team to
    /// cores gives (OMP_PROC_BIND=close with OMP_PLACES=cores, or a binding
    /// of thread w to core w): the blocks schedule and the cost model's row
    /// groups count on it.
    int domain_threads = 0;
    // INT8 tile.
    int int8_mr = 12; ///< rows per call, at most
    uint8_t int8_cw_mask = kNativeInt8CwAll;
    int int8_l2_pct = 50; ///< L2 share of one pass's B slab and A rows
    int int8_kc = 0; ///< pin; 0: the L2 rule
    /// Percent of L1 one batch of an s4 tile's groups (its B slab and one
    /// call's source rows) takes in a plan of at most decode_m rows, where
    /// the tile's later row calls read each batch again; 0: batches of the
    /// int8_l2_pct share of L2; -1: 50 where the weight fits the L2 of the
    /// plan's threads (one L2 a physical core) and the source rows of its
    /// row block fit one L2, else 125.  One group over K takes K passes of
    /// that share without the cost model; with it, of the int8_l2_pct share
    /// of L2 (one pass where the group's slab fits it).
    int int4_l1_pct = -1;
    bool direct_store = true;
    bool bias_in_dequant = true;
    // Source preparation.
    int copy_block = 64;
    /// BF16 transpose goes parallel at M * K >= this.
    int64_t bf16_copy_par_min = 0;
    /// INT8 copy goes parallel at M * K >= this.
    int64_t int8_copy_par_min = 65536;
    // Rule switches: 1 applies the rule, 0 leaves it out (traffic_groups=0
    // runs the L3 row-group test in its place).
    /// nr, mb and the row split by the smallest makespan, and row groups
    /// when the slice leaves more than idle_pct percent of the team idle;
    /// either replaces the grid of the narrowing and the row split, or the
    /// slice, only for a makespan more than min_gain_pct percent smaller.
    int makespan = 1;
    /// The makespan's margin (above), and the extra work, in percent of the
    /// busiest worker's, that bf16_narrow_n lets a wider tile cost.
    int min_gain_pct = 5;
    int idle_pct = 10;
    /// The makespan counts a tile's bytes up to this M, its flops above;
    /// also the M above which bf16_alias applies.
    int decode_m = 128;
    /// K passes of at most l1_pct percent of L1 for one pass's B slab and
    /// one call's A rows, in tiles of 2 * mr .. l1_pass_max_mb rows.
    int l1_pass = 1;
    int l1_pct = 100;
    int l1_pass_max_mb = 64;
    /// col_groups by the source and weight bytes an L3 domain reads from
    /// memory, whether or not the column tiles divide over the threads.  0:
    /// row groups by the L3 test instead, when the source and one domain's
    /// B slabs exceed the l3_pct share of its L3 and the column tiles do not
    /// divide over the threads.
    int traffic_groups = 1;
    /// BF16 rows per call at most bf16_alias_mr above decode_m when K is a
    /// multiple of 2048, for a block of more than bf16_alias_mr rows: source
    /// rows K elements apart (a transposed source is copied to such rows)
    /// are then a multiple of 4 KiB apart, and the rows of one call share an
    /// L1 set.  The problem carries no source row stride: the rule takes a
    /// row-major source as rows K apart.
    int bf16_alias = 1;
    int bf16_alias_mr = 8;
    /// BF16 tiles of 64 or 128 columns from bf16_narrow_n_min_m rows when
    /// the plan of 32-column tiles is the slice with fewer than two column
    /// tiles per L3 domain of the team, and the wider tiles are fewer than
    /// the domains: fewer domains then read each source row from memory.
    /// The busiest worker of the wider tiles has at most min_gain_pct
    /// percent more work; the wider plan, planned by every rule at its
    /// width, may take the blocks schedule and is weighed as the slice of
    /// its grid.
    int bf16_narrow_n = 1;
    int bf16_narrow_n_min_m = 1024;
    /// F16_PH tiles of 64 columns, two pack blocks of 32 under one source
    /// broadcast, from f16_wide_min_m rows where N is whole tiles of 64;
    /// tiles of 32 below.  Narrowing for the threads applies to both.
    int f16_wide_min_m = 192;
    /// The blocks schedule, on a team of more than one L3 domain, from
    /// bf16_blocks_min_m rows (BF16) or int8_blocks_min_m rows (INT8) when
    /// the rows of a full L3 domain's share of the source (M * domain_threads
    /// / threads) fit the l3_pct share of its L3: in place of the
    /// schedule of a BF16 plan whose B slab over all of K exceeds the
    /// bf16_l2_pct share of L2, of a BF16 slice of 32-column tiles with at
    /// most two column tiles per L3 domain that bf16_narrow_n leaves, or of
    /// the slice of an INT8 plan.
    int blocks = 1;
    int bf16_blocks_min_m = 1024;
    int int8_blocks_min_m = 3072;
    /// 1: a BF16 plan of tiles of 32 columns or more and at least as many
    /// column tiles as threads gives way to the variant of the rules
    /// (cm_short_rows rows per call, two row groups for a single row block)
    /// or pair of variants that the cost model rates more than cm_gain_pct
    /// percent faster, and to the model's rules for long weight streams
    /// (cm_pass_kib, cm_block_kib); a per-channel INT8 plan likewise to two
    /// row groups, and with cm_int8_l1_pct to its rule for blocks of three
    /// calls; an s4 plan to M blocks (cm_int4_min_m) and, over one group, to
    /// its K passes (int4_l1_pct).  The model places the team's threads in
    /// L3 domains in order, domain_threads to a domain.
    int cost_model = 1;
    int cm_gain_pct = 5;
    // Cost model (native_cost_model.hpp): the inputs of the modeled time of
    // a BF16, per-channel INT8 or s4 plan, in cycles of one core.
    /// Rows per call of the shorter BF16 call the model weighs against
    /// tall_rows; a BF16 or INT8 call of more rows takes cm_tall_pct percent
    /// more cycles per row.
    int cm_short_rows = 8;
    /// Dot products a cycle of every kernel family; 0: its compute path's
    /// (UarchParams::dot) where the machine's model prices kernels
    /// (native_dot_prices), else 2.
    int cm_fma_pipes = 0;
    /// Cycles one step of a kernel call takes at least, whatever its
    /// accumulator sets; 0: its compute path's latency over the sets of the
    /// call's kernel (UarchParams::dot, native_int8_kernel_sets): a call of
    /// few accumulators waits for its chains.
    int cm_fma_lat = 6;
    /// Cycles of one kernel call besides its multiply-adds.
    int cm_call_cycles = 800;
    /// Percent more cycles per row a call of more than cm_short_rows rows
    /// takes; -1: 12.5 per doubling of the threads of an L3 domain
    /// (domain_threads) past 8, at most 50.
    int cm_tall_pct = -1;
    /// Bytes a cycle a core reads from L2: a later call of a K pass that
    /// does not fit the l1_pct share of L1 re-reads the pass's slab at most
    /// this fast.
    int cm_l2_bytes = 16;
    /// Cycles per KiB read from the L3 domain: the source rows of a tile
    /// whose rows and slab over all of K exceed the bf16_l2_pct share of
    /// L2 (int8_l2_pct for an INT8 tile), and the second read of a weight
    /// tile in one domain.
    int cm_l3_cycles = 10;
    /// Cycles per KiB one thread streams weights from memory while every
    /// thread of its L3 domain streams; 0: the machine's
    /// (UarchParams::mem_cycles, 2400 where not measured) over the square
    /// root of the threads of an L3 domain (domain_threads).
    int cm_mem_cycles = 0;
    /// KiB a weight stream is fetched ahead: the calls after the first of a
    /// K pass read no new weight, and the stream idles for the part of them
    /// beyond the cm_mem_cycles this many KiB take.
    int cm_prefetch_kib = 4;
    /// Cycles per KiB one core streams weights at most: while cm_mem_cycles
    /// is slower, the link of the L3 domain limits the streams, and the
    /// other streams of the domain fill the link while one idles, past the
    /// first cm_sync_tiles tiles of each worker (the workers start in step);
    /// while it is faster, one core's stream limits, and the long K pass
    /// (cm_drift_kib), the INT8 rule of cm_int8_l1_pct and the K bound of
    /// the M blocks of cm_block_kib (cm_block_k) apply.
    /// At a tie neither holds: the derived cm_mem_cycles of 16 threads to a
    /// domain is 600 where neither is measured.  -1: the machine's
    /// (UarchParams::stream_cycles, 600 where not measured).
    int cm_stream_cycles = -1;
    int cm_sync_tiles = 5;
    /// Percent of the smaller of the core and the memory time that adds to
    /// the larger.
    int cm_overlap_pct = 20;
    /// Percent of the L3 domain's share that the weight tiles read between
    /// the first and the second read of a tile by two row groups of one
    /// domain fill, of the domain's weight bytes: the second read hits L3
    /// while that fits.  The domain a thread reads in follows
    /// domain_threads.
    int cm_reuse_pct = 50;
    /// Percent of L2 that the slab of a column tile over all of K may take
    /// for two row groups to be weighed.
    int cm_group_slab_pct = 75;
    /// Percent of the L3 of the domains the team spans that the weights of
    /// a call may take for two row groups to be weighed: each row group
    /// reads every weight tile of its column group.
    int cm_group_l3_pct = 75;
    /// Percent of the mean column tiles per thread that the busiest worker
    /// of the slice must take for two row groups to be weighed: they gain
    /// by balancing a slice of whole tiles.
    int cm_group_skew_pct = 110;
    /// Percent of half the block's rows the larger of two row groups may
    /// take: its workers take the row groups' common column tiles.
    int cm_group_rows_pct = 115;
    /// KiB of weights each worker streams in step with the others of its L3
    /// domain.  Where one core streams slower than its share of the
    /// domain's link, the busiest worker of a single row block of at least
    /// cm_pass_calls calls of the rules' rows that computes longer than it
    /// streams, over more than these KiB, has drifted out of step (its
    /// stream runs ahead of its calls): the block takes a K pass of
    /// cm_pass_kib KiB of B slab, whose later calls read the slab from L2,
    /// at the rules' rows per call and row unit (tall_rows, row_unit); the
    /// model's row groups take it only where that row unit splits the rows
    /// within cm_group_rows_pct.  cm_pass_kib 0: no such pass.
    int cm_drift_kib = 4096;
    int cm_pass_kib = 128;
    int cm_pass_calls = 4;
    /// KiB of weights a worker of the slice streams past which, where the
    /// worker streams longer than it computes, a single row block of more
    /// than two calls takes M blocks of tile_rows rows over all of K.
    int cm_block_kib = 16384;
    /// K past which, where one core streams slower than its share of the
    /// domain's link, the M blocks of cm_block_kib of a BF16 plan whose team
    /// fills each L3 domain it spans (threads a multiple of domain_threads)
    /// apply; a K of at most this keeps the block's K passes.  0: no such
    /// bound.
    int cm_block_k = 2048;
    /// Percent of L1 the INT8 L1 pass of tile_rows rows per call takes
    /// where the cost model gives a single INT8 row block of three calls
    /// those rows; 0: no such rule.
    int cm_int8_l1_pct = 0;
    /// Rows from which a single s4 row block whose slice does not divide
    /// its column tiles over the threads takes M blocks of ceil(M / b) rows,
    /// b 2 to 4 (at least tile_rows each), where the cost model rates them
    /// more than cm_gain_pct percent faster: the row blocks divide the
    /// slice more evenly.  0: no such rule.
    int cm_int4_min_m = 48;
};

/// A partial heuristic_params_t: what one source sets.  Fields mirror
/// heuristic_params_t one to one.
struct native_config_t {
    std::optional<int> nr, mr, mb, nb, kc, col_groups;
    std::optional<native_schedule_t> schedule;
    std::optional<native_traversal_t> traversal;
    std::optional<int> unit_tiles;
    std::optional<int> threads_cap, min_tiles_per_thread, row_unit;
    std::optional<int> tile_rows, tall_rows, block_rows, min_tile_cols;
    std::optional<bool> row_split;
    std::optional<int> min_pass_k, bf16_l2_pct;
    std::optional<native_kc_split_t> kc_split;
    std::optional<int> l3_pct, domain_threads;
    std::optional<int> int8_mr;
    std::optional<uint8_t> int8_cw_mask;
    std::optional<int> int8_l2_pct, int8_kc, int4_l1_pct;
    std::optional<bool> direct_store, bias_in_dequant;
    std::optional<int> copy_block;
    std::optional<int64_t> bf16_copy_par_min, int8_copy_par_min;
    std::optional<int> makespan, min_gain_pct, idle_pct, decode_m;
    std::optional<int> l1_pass, l1_pct, l1_pass_max_mb;
    std::optional<int> traffic_groups;
    std::optional<int> bf16_alias, bf16_alias_mr;
    std::optional<int> bf16_narrow_n, bf16_narrow_n_min_m;
    std::optional<int> f16_wide_min_m;
    std::optional<int> blocks, bf16_blocks_min_m, int8_blocks_min_m;
    std::optional<int> cost_model, cm_gain_pct, cm_short_rows, cm_fma_pipes;
    std::optional<int> cm_fma_lat, cm_call_cycles, cm_tall_pct;
    std::optional<int> cm_l2_bytes, cm_l3_cycles, cm_mem_cycles;
    std::optional<int> cm_prefetch_kib, cm_overlap_pct, cm_reuse_pct;
    std::optional<int> cm_stream_cycles, cm_sync_tiles;
    std::optional<int> cm_group_slab_pct, cm_group_l3_pct, cm_group_skew_pct;
    std::optional<int> cm_group_rows_pct, cm_drift_kib, cm_pass_kib;
    std::optional<int> cm_pass_calls, cm_block_kib, cm_block_k;
    std::optional<int> cm_int8_l1_pct, cm_int4_min_m;

    bool empty() const;
    bool operator==(const native_config_t &o) const;
};

/// Parses "key=value{,key=value}".  On an unknown key, a malformed value or
/// a value outside its range (check_native_config) returns false, sets why
/// and leaves cfg unspecified.
bool parse_native_config(
        std::string_view text, native_config_t &cfg, std::string &why);

/// The set fields as "key=value,...", in schema order; "" when empty.
/// parse_native_config(format_native_config(c)) == c.
std::string format_native_config(const native_config_t &cfg);

/// Fields set in overlay replace those of base; schedule and col_groups
/// choose one schedule together, and nb and unit_tiles one unit width, so an
/// overlay setting either key of a pair replaces both.
void merge_native_config(native_config_t &base, const native_config_t &overlay);

/// The key ranges and the cross-field rules; nothing that depends on the
/// problem.  why names the first violation.
bool check_native_config(const native_config_t &cfg, std::string &why);

/// params with every field set in cfg replaced.
heuristic_params_t apply_native_config(
        heuristic_params_t params, const native_config_t &cfg);

/// cfg without the keys an INT8 call of scheme does not apply; the names of
/// those cfg sets are appended to dropped, comma separated, unless nullptr.
native_config_t native_config_for_scheme(const native_config_t &cfg,
        native_int8_scheme_t scheme, std::string *dropped);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_CONFIG_HPP
