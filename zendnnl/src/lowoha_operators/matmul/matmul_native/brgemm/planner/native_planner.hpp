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

#ifndef MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_PLANNER_HPP
#define MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_PLANNER_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_config.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// What the epilogue of a call does after the accumulation (first match):
/// gated: a gated activation; postops: any post-op, INT8 requantization, or
/// alpha / beta other than 1 / 0; bias: a bias the epilogue must add (BF16, or
/// an INT8 bias the dequantization cannot take); store: the store only (no
/// bias, or an INT8 bias the dequantization takes).
enum class native_epilogue_class_t : uint8_t { store, bias, postops, gated };

/// Everything a plan may depend on.  Equal problems get equal plans on one
/// process; a field a source can match on must be here.
struct native_problem_t {
    int M = 0;
    int N = 0;
    int K = 0;
    int pack_nr = 0; ///< native_weight_ref_t::pack_nr
    /// 1 inside a parallel region, else max(1, desc.num_threads).
    int threads = 1;
    /// bf16, u8, s8, or bf16 / f32 quantized in the call.
    data_type_t src_dt = data_type_t::none;
    data_type_t wei_dt = data_type_t::none; ///< bf16 or s8
    data_type_t dst_dt = data_type_t::none;
    bool trans_a = false;
    bool dynamic_quant = false; ///< the call quantizes a bf16 / f32 source
    bool has_bias = false;
    native_epilogue_class_t epilogue = native_epilogue_class_t::store;
    /// INT8: decide_int8_call's scheme.  The plan check consults it, and a
    /// key a scheme does not apply (native_config_key_t::int8_schemes) is not
    /// applied to it.
    native_int8_scheme_t int8_scheme = native_int8_scheme_t::per_channel;

    bool bf16() const { return wei_dt == data_type_t::bf16; }
    bool operator==(const native_problem_t &o) const;
};

size_t hash_native_problem(const native_problem_t &p) noexcept;

/// The problem of a call native_brgemm_execute serves.  Cheap; no allocation.
native_problem_t make_native_problem(const GemmDescriptor &desc,
        const matmul_params &params, const void *bias, int pack_nr);

enum class native_plan_source_t : uint8_t { heuristic, rules, table, override };

struct native_config_entry_t;

/// The resolved plan of one call: everything the looper, the partition and
/// the INT8 tile read.  Legal for its problem (validate_native_plan).
struct native_plan_t {
    // Tile grid and schedule.
    int nr = 0; ///< tile width: 16 or 32, dividing pack_nr
    int mr = 0; ///< BF16 rows per call; rows of the row-split rounding
    int mb = 0; ///< rows per M block
    int m_blocks = 0;
    int n_tiles = 0;
    int kc = 0; ///< BF16 K per pass; even unless it covers K
    int threads = 1; ///< compute team
    int col_groups = 0; ///< 0: slice schedule; > 0: row-group schedule
    int row_unit = 12; ///< rows per unit of the row-group ranges
    native_traversal_t traversal = native_traversal_t::m_inner;
    size_t scratch_floats = 0; ///< accumulators per thread
    // INT8 tile.
    int int8_rows_nv1 = 12; ///< rows per INT8 call at NR 16
    int int8_rows_nv2 = 12; ///< at NR 32
    uint8_t int8_cw_mask = kNativeInt8CwAll;
    /// Bytes for one pass's B slab and one call's A rows.
    int64_t int8_pass_budget = 0;
    int int8_pass_quads = 0; ///< > 0: quads per pass, replacing the rule
    bool int8_direct_store = true; ///< permission; the call needs the gate
    bool int8_bias_in_dequant = true; ///< permission; the call needs the gate
    /// The problem's: which INT8 keys the plan applies.
    native_int8_scheme_t int8_scheme = native_int8_scheme_t::per_channel;
    // Source preparation.
    int copy_block = 64;
    int64_t copy_par_min = 0; ///< of the problem's dtype
    // Provenance.
    uint8_t sources = 0; ///< bit 1 << native_plan_source_t per layer applied
    const native_config_entry_t *rules_entry = nullptr;
    const native_config_entry_t *table_entry = nullptr;
    const native_config_entry_t *override_entry = nullptr;

    int tiles() const { return m_blocks * n_tiles; }
};

/// The built-in planner.  With heuristic_params_t{} it returns the built-in
/// tile plan plus the INT8 tile's choices; a set pin replaces its decision,
/// and later rules read the pinned value.  Pure: no allocation, no global
/// state other than the kernel bounds.
native_plan_t heuristic_plan(const native_problem_t &p,
        const UarchParams &uarch, const heuristic_params_t &params);

/// nullptr when plan is legal for p, else the first rule it breaks.
const char *validate_native_plan(
        const native_plan_t &plan, const native_problem_t &p);

/// INT8 tile helpers; the INT8 tile calls these instead of its constants.
/// Widest allowed width <= remaining (remaining is a multiple of 16).
inline int native_int8_col_width(const native_plan_t &p, int remaining) {
    if ((p.int8_cw_mask & kNativeInt8Cw32) != 0 && remaining >= 32) return 32;
    return 16;
}
/// Rows per INT8 call at NR = 16 * nv, nv 1 or 2.
inline int native_int8_call_rows(const native_plan_t &p, int nv) {
    return nv == 2 ? p.int8_rows_nv2 : p.int8_rows_nv1;
}
/// Quads per K pass of a tile cw columns wide and rows tall, max_mr rows per
/// call, with the pass budget of the plan.
int native_int8_pass_quads(
        const native_plan_t &p, int k_quads, int cw, int rows, int max_mr);

/// A partial config and where it came from.  Owned by its source and
/// immutable, except for the one-warning flag.
struct native_config_entry_t {
    native_config_t config;
    /// For warnings and the log, e.g. "<file> entry 17 (id ...)".
    std::string label;
    mutable std::atomic<bool> warned {false};
    /// Set once the entry's keys an INT8 scheme does not apply are logged.
    mutable std::atomic<bool> scope_noted {false};
};

/// One source of partial configs: the developer override, a tuning table or
/// rules.  Built once; lookup runs concurrently and does not allocate, lock
/// or touch files.
class native_config_source_t {
public:
    virtual ~native_config_source_t() = default;
    /// The entry this source gives for p (its first match), or nullptr.  The
    /// entry lives as long as the source.
    virtual const native_config_entry_t *lookup(
            const native_problem_t &p) const noexcept
            = 0;
};

inline constexpr size_t kNativePlanCacheSlots = 8192;
inline constexpr int kNativePlanCacheMaxM = 1024;

/// Plans calls from the built-in heuristic and up to three sources.
class native_planner_t {
public:
    /// Any source may be null.  uarch is what heuristic_plan reads.
    native_planner_t(const UarchParams &uarch,
            std::unique_ptr<const native_config_source_t> override_source,
            std::unique_ptr<const native_config_source_t> table_source,
            std::unique_ptr<const native_config_source_t> rules_source);
    ~native_planner_t();
    native_planner_t(const native_planner_t &) = delete;
    native_planner_t &operator=(const native_planner_t &) = delete;

    /// The plan of p; never fails.  Without sources this is heuristic_plan
    /// with the built-in parameters: no cache, no allocation, no lock.
    native_plan_t plan(const native_problem_t &p) const noexcept;

    bool has_sources() const noexcept;

private:
    struct cache_entry_t;

    /// Merges the entries of the sources for p; a layer that makes the plan
    /// illegal is dropped with one warning.
    native_plan_t resolve(const native_problem_t &p) const noexcept;
    native_plan_t resolve_layers(const native_problem_t &p) const;
    /// Caches plan for p unless the probe window of p is full.
    void remember(const native_problem_t &p,
            const native_plan_t &plan) const noexcept;

    UarchParams uarch_;
    std::unique_ptr<const native_config_source_t> override_;
    std::unique_ptr<const native_config_source_t> table_;
    std::unique_ptr<const native_config_source_t> rules_;
    /// kNativePlanCacheSlots slots with sources, else null.
    std::unique_ptr<std::atomic<const cache_entry_t *>[]> cache_;
    /// Entries dropped with a warning.
    mutable std::atomic<uint32_t> drop_warnings_ {0};
};

/// The planner native_brgemm_execute uses.  Built on the first call from
/// ZENDNNL_MATMUL_NATIVE_CONFIG and ZENDNNL_MATMUL_NATIVE_TUNING_FILE, and
/// never destroyed.
const native_planner_t &native_planner();

/// The planner native_planner() builds, over native_plan_uarch(), from the
/// two settings' values (nullptr or "" = unset); warnings as at process start.
std::unique_ptr<native_planner_t> make_native_planner(
        const char *config, const char *tuning_file);

/// The layers kept in plan, highest first, joined with '+'; "heuristic"
/// when none.
std::string native_plan_source_names(const native_plan_t &plan);

/// format_native_config of the merged config of the layers kept in plan;
/// "-" when none.
std::string native_plan_config_text(const native_plan_t &plan);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_PLANNER_HPP
