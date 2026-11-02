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
    /// pin; 0: pack_nr, at most 32, halved while tiles < team and one row
    /// block, or 64 / 128 by bf16_narrow_n.  64 and 128 are BF16 only:
    /// tiles of 32-column slabs.
    int nr = 0;
    /// pin; 0: tall_rows for a BF16 tile of 32 columns or more, else
    /// tile_rows
    int mr = 0;
    int mb = 0; ///< pin, no row split; 0: block_rows, then the row split
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
    /// family_l3_cores when that is 0.
    int domain_threads = 0;
    // INT8 tile.
    int int8_mr = 12; ///< rows per call, at most
    uint8_t int8_cw_mask = kNativeInt8CwAll;
    int int8_l2_pct = 50; ///< L2 share of one pass's B slab and A rows
    int int8_kc = 0; ///< pin; 0: the L2 rule
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
};

/// A partial heuristic_params_t: what one source sets.  Fields mirror
/// heuristic_params_t one to one.
struct native_config_t {
    std::optional<int> nr, mr, mb, kc, col_groups;
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
    std::optional<int> int8_l2_pct, int8_kc;
    std::optional<bool> direct_store, bias_in_dequant;
    std::optional<int> copy_block;
    std::optional<int64_t> bf16_copy_par_min, int8_copy_par_min;
    std::optional<int> makespan, min_gain_pct, idle_pct, decode_m;
    std::optional<int> l1_pass, l1_pct, l1_pass_max_mb;
    std::optional<int> traffic_groups;
    std::optional<int> bf16_alias, bf16_alias_mr;
    std::optional<int> bf16_narrow_n, bf16_narrow_n_min_m;
    std::optional<int> blocks, bf16_blocks_min_m, int8_blocks_min_m;

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
/// choose one schedule together, so an overlay setting either replaces both.
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
