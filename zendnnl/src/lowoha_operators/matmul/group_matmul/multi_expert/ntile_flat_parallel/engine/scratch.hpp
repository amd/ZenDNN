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

/// The engine's scratch, one per calling thread: a call's row blocks, runs and
/// units, and its source, intermediate and staging rows.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_ENGINE_SCRATCH_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_ENGINE_SCRATCH_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <type_traits>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/source.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/nr32_pack.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

/// Rows [row, row + rows) of expert `e`, at most kDecodeMaxM.
struct row_block_t {
    int e;
    int row;
    int rows;
};

/// Blocks [block, block + blocks) of one expert, `rows` rows in all, cut
/// into row runs [run, run + runs): a decode call's group is one block.
struct block_group_t {
    int block;
    int blocks;
    int rows;
    int run = 0;
    int runs = 0;
};

/// Blocks [block, block + blocks) of one group: the rows a unit covers and
/// that a W2 unit waits for.
struct row_run_t {
    int block;
    int blocks;
};

/// Columns [col, col + width) of row runs [run, run + runs) of one group.
struct unit_t {
    int run; // index into the call's row runs
    int runs;
    int col;
    int width;
};

/// The calling thread's buffers, grown to its largest call.
struct scratch_t {
    std::vector<int> offset, fill, row_token, row_of;
    std::vector<row_block_t> blocks;
    std::vector<block_group_t> groups;
    std::vector<row_run_t> runs;
    std::vector<const void *> row_ptrs;
    /// Source rows: a routed call's tokens' s8 rows, each projection's rows
    /// by row of the call (gathered, quantized or copied), and W2's,
    /// quantized from Op1's.
    source_rows tok, src13, src2;
    std::vector<uint16_t> inter, stage;
    std::vector<unit_t> queue, planned;
    std::vector<std::atomic<int>> left, ready, gather_next, gather_done;
    std::vector<std::atomic<uint8_t>> claimed;
    /// A grouped call's packs, by operation.
    std::vector<const nr32::packed_expert *> packs13, packs2;

    /// Bytes every vector above keeps.
    size_t bytes() const {
        const auto kept = [](const auto &v) {
            return v.capacity()
                    * sizeof(typename std::decay_t<decltype(v)>::value_type);
        };
        return kept(offset) + kept(fill) + kept(row_token) + kept(row_of)
                + kept(blocks) + kept(groups) + kept(runs) + kept(row_ptrs)
                + tok.bytes() + src13.bytes() + src2.bytes() + kept(inter)
                + kept(stage) + kept(queue) + kept(planned) + kept(left)
                + kept(ready) + kept(gather_next) + kept(gather_done)
                + kept(claimed) + kept(packs13) + kept(packs2);
    }
};

scratch_t &scratch();

/// Grow-only sizing: every live element is written before it is read.
template <typename T>
void grow(std::vector<T> &v, const size_t n) {
    if (v.size() < n) { v.resize(n); }
}

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_ENGINE_SCRATCH_HPP
