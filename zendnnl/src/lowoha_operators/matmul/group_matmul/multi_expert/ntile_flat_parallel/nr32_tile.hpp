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

/// Tiles of one projection over an `nr32_pack.hpp` pack.
///
/// A call's epilogue decides how the pack is read, through a plan made once
/// per projection.  Callers cut a projection's work into ranges of plan
/// columns, `step` at a time, as if it were one weight of `cols` columns:
///   * plain: column c is weight and output column c;
///   * swiglu_pairs: column c is weight column c, of output column c / 2;
///   * silu_split, gelu_split: column c is output column c / 2, whose gate is
///     weight column c / 2 and whose up is weight column I + c / 2.
/// A tile runs its rows in balanced calls of as many rows as its kernels take
/// against L2-sized subtiles of its columns, every call over the subtile's
/// steps.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_NR32_TILE_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_NR32_TILE_HPP

#include <cstdint>

#include "nr32_pack.hpp"
#include "nr32_ukernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {

/// How a projection's tiles read its pack.
struct tile_plan {
    epilogue ep = epilogue::plain;
    bool src_u8 = false;
    /// Quantized weights of several scale groups: each source row has one
    /// scale per group, `layout.groups` scales apart from row to row.
    bool src_groups = false;
    /// Columns of a row's output: N, or N / 2 under a gated epilogue.
    int out_cols = 0;
    /// The columns callers cut, and how many one kernel call covers.
    int cols = 0;
    int step = 0;
    /// Columns a tile walks at a time.
    int subtile = 0;
    /// Split epilogues: the first up column, and whether it starts inside a
    /// kernel vector.
    int up = 0;
    bool unaligned = false;
    int max_mr = 0;
    ukernel_fn kernels[kMaxMr + 1] = {};
    /// Split epilogues of bf16, s8 and s4 whose up columns start on a block:
    /// a step is 64 columns, one wide kernel call for a tile of at most
    /// `wide_rows` rows and two of `kernels` for a taller one.
    bool wide = false;
    int wide_rows = 0;
    int wide_max_mr = 0;
    /// [two_chains][mr]
    ukernel_fn wide_kernels[2][kMaxMr + 1] = {};
};

/// The L2 bytes a plan's tiles may give their rows and weights, and the
/// source rows that budget reserves beside the weights.
struct plan_budget {
    int64_t l2_bytes = 0;
    int subtile_rows = 16;
};

/// The machine view's budget: its L2 budget and 16 rows.
plan_budget default_budget();

/// The plan for tiles of packs of `layout` with epilogue `ep`, reading
/// source bytes XOR-ed with 0x80 when `src_u8`, and one source scale per row
/// and weight scale group when `src_groups` (one per row otherwise, and for a
/// weight of one group), its subtiles sized by `budget`.  False when this CPU
/// has no kernel for it or a gated epilogue meets an odd N.
bool make_plan(const pack_layout &layout, epilogue ep, bool src_u8,
        tile_plan &plan, bool src_groups = false,
        const plan_budget &budget = default_budget());

/// Bytes past a tile's last s8 source row its kernels may read: they read
/// each row four bytes at a time, and a K that is not a multiple of four
/// ends inside a group.  The bytes read past K meet zero weights.
inline constexpr size_t kRowReadSlack = 4;

/// One tile: `M` rows, `lda` elements apart (s8 for quantized weights, each
/// with its scales at `src_scale` as `plan.src_groups` says, else the
/// weight's 16-bit type), against plan columns
/// [col, col + width), both multiples of `plan.step`, of expert pack `px` of
/// `layout`.  The expert's bias row (or null) is in `bias_t`.  Output column
/// j of row r goes to `dst + r * ldc + j`, in bf16 (f16 for f16 weights).
/// `kRowReadSlack` bytes past the last s8 row must be readable.
void run_tile(const tile_plan &plan, const pack_layout &layout,
        const packed_expert &px, int M, const void *src, int lda,
        const float *src_scale, const void *bias, bias_kind bias_t, void *dst,
        int ldc, int col, int width);

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_NR32_TILE_HPP
