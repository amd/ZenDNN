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

/// The engine.  Both APIs run their calls here: a routed call once it has
/// routed its tokens to expert-major rows, a grouped call with its
/// operations' rows.  It names no weight type: the plans and packs it runs
/// carry their format.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_ENGINE_ENGINE_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_ENGINE_ENGINE_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include <omp.h>
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/schedule.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/scratch.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/source.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel_config.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/nr32_pack.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/tile/nr32_tile.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

/// A tile's operands besides its columns: its source rows, its expert's bias
/// row, and where its output rows go.
struct tile_io {
    nr32::tile_source src;
    const void *bias = nullptr;
    nr32::bias_kind bias_t = nr32::bias_kind::none;
    void *dst = nullptr;
    int ldc = 0;
};

/// What the engine runs: `n` experts, expert e's `rows[e]` rows from
/// `S.offset[e]` on, through W13 with its plan's epilogue and, when `fused`,
/// W2; then the weighted `reduce` of `reduce_cols` columns, if any.  `fmt`
/// and `api` pick the call's settings.
struct engine_t {
    int n = 0;
    const int *rows = nullptr;
    const nr32::format_desc *fmt = nullptr;
    api_kind api = api_kind::routed;
    bool fused = false;
    float wei_bytes = 2.0f;
    int K13 = 0;
    int K2 = 0;
    /// What each projection's kernels read.
    source_spec src13, src2;
    const nr32::pack_layout *l13 = nullptr;
    const nr32::pack_layout *l2 = nullptr;
    const nr32::tile_plan *p13 = nullptr;
    const nr32::tile_plan *p2 = nullptr;
    const nr32::packed_expert *const *packs13 = nullptr;
    const nr32::packed_expert *const *packs2 = nullptr;
    const group_matmul_moe_postop_params *reduce = nullptr;
    int reduce_cols = 0;
    bool reduce_f16 = false;
    int num_threads = 0;
};

/// Runs `E` on the calling thread's scratch, whose `offset` the caller has
/// set.  The front end gives the rows, on every thread of the region:
/// `prologue()`, which may run a worksharing loop, before any unit;
/// `before_w13(run)` before a W13 unit reads row run `run`; `w13(block)` and
/// `w2(block)`, a block's tile operands; and `op1_row(e, row)`, Op1's output
/// row, which a fused call whose W2 reads s8 rows quantizes into the
/// scratch's once its run's W13 units are done.
template <typename Front>
status_t run_engine(const engine_t &E, const Front &front) {
    scratch_t &S = scratch();
    const size_t es = E.src13.elem_bytes();
    const int max_rows = *std::max_element(E.rows, E.rows + E.n);
    const grp_matmul_phase phase = classify_grp_matmul_phase(max_rows);
    const call_knobs &knobs = call_knobs_for(phase, *E.fmt, E.api);
    const queue_t q = plan_queue(S, E.rows, E.n, *E.p13, E.K13,
            E.fused ? E.p2 : nullptr, E.K2, es, E.wei_bytes, E.num_threads,
            phase, knobs, l2_budget_bytes(*E.fmt, E.api));
    std::atomic<int> next_unit {0};

    // The region reaches the calling thread's buffers through `S`, a
    // reference, so the workers share them.
    const unit_t *const plan = S.queue.data();
    const row_block_t *const blocks = S.blocks.data();
    const row_run_t *const runs = S.runs.data();
    source_rows &src2 = S.src2;

#pragma omp parallel num_threads(E.num_threads)
    {
        front.prologue();

        // Columns [c.col, c.col + c.width) of every block of `c`'s run, one
        // chunk of columns at a time.
        const auto run_w13 = [&](const unit_t &c) {
            front.before_w13(c.run);
            const row_run_t run = runs[c.run];
            for (int col = c.col; col < c.col + c.width; col += q.chunk13) {
                const int width = std::min(q.chunk13, c.col + c.width - col);
                for (int i = run.block; i < run.block + run.blocks; ++i) {
                    const row_block_t b = blocks[i];
                    const tile_io io = front.w13(b);
                    nr32::run_tile(*E.p13, *E.l13, *E.packs13[b.e], b.rows,
                            io.src, io.bias, io.bias_t, io.dst, io.ldc, col,
                            width);
                }
            }
        };
        const auto run_w2 = [&](const unit_t &c) {
            const int b0 = runs[c.run].block;
            const row_run_t &last = runs[c.run + c.runs - 1];
            for (int col = c.col; col < c.col + c.width; col += q.chunk2) {
                const int width = std::min(q.chunk2, c.col + c.width - col);
                for (int i = b0; i < last.block + last.blocks; ++i) {
                    const row_block_t b = blocks[i];
                    const tile_io io = front.w2(b);
                    nr32::run_tile(*E.p2, *E.l2, *E.packs2[b.e], b.rows, io.src,
                            io.bias, io.bias_t, io.dst, io.ldc, col, width);
                }
            }
        };

        run_units(S, q, next_unit, [&](const int u) {
            const unit_t c = plan[u];
            if (u < q.n13) {
                run_w13(c);
                if (E.fused
                        && S.left[c.run].fetch_sub(1, std::memory_order_acq_rel)
                                == 1) {
                    const row_run_t run = runs[c.run];
                    const row_block_t &head = blocks[run.block];
                    const row_block_t &tail
                            = blocks[run.block + run.blocks - 1];
                    for (int r = head.row;
                            E.src2.s8 && r < tail.row + tail.rows; ++r) {
                        quantize_row(E.src2, front.op1_row(head.e, r), E.K2,
                                src2, r);
                    }
                    S.ready[c.run].store(1, std::memory_order_release);
                }
                return;
            }
            for (int r = c.run; r < c.run + c.runs; ++r) {
                wait_for(S.ready[r], 1);
            }
            run_w2(c);
        });

        if (E.reduce != nullptr) {
#pragma omp barrier
            const int D = E.reduce_cols;
            const int cuts = col_chunks_per_row(
                    E.reduce->num_tokens, D, E.num_threads);
            const int span = col_chunk_span(D, cuts);
#pragma omp for schedule(static)
            for (int it = 0; it < E.reduce->num_tokens * cuts; ++it) {
                const int t = it / cuts;
                const int d0 = (it % cuts) * span;
                if (d0 < D) {
                    const int len = std::min(D, d0 + span) - d0;
                    if (E.reduce_f16) {
                        group_matmul_moe_postop_f16_cols(E.reduce, t, d0, len);
                    } else {
                        group_matmul_moe_postop_bf16_cols(E.reduce, t, d0, len);
                    }
                }
            }
        }
    }
    if (S.bytes() > limits().scratch_keep_bytes) { S = scratch_t {}; }
    return status_t::success;
}

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_ENGINE_ENGINE_HPP
