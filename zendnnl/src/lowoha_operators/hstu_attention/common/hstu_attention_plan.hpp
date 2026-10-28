/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

/// Work decomposition of the tiled HSTU attention kernels: which key blocks
/// are packed, the query tile height and the work items.

#ifndef LOWOHA_HSTU_ATTENTION_PLAN_HPP
#define LOWOHA_HSTU_ATTENTION_PLAN_HPP

#include <algorithm>
#include <cstdint>
#include <vector>
#include "lowoha_operators/hstu_attention/hstu_attention_mask.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

/// Keys per packed K / V block.
constexpr int64_t kBlockN = 256;
/// Query rows per tile: the largest power of two in [kMinBlockM, kMaxBlockM]
/// that still gives every thread kTilesPerThread tiles.
constexpr int64_t kMaxBlockM = 128;
constexpr int64_t kMinBlockM = 32;
constexpr int64_t kTilesPerThread = 4;

/// Keys [lo, hi) of a sequence that some query of the call reads through the
/// packed blocks (diagonal terms outside a band read K/V directly).
struct key_range_t {
    int64_t lo;
    int64_t hi;
};

/// The key range each sequence's queries need.
std::vector<key_range_t> key_ranges(const hstu_attention_params &p);

int64_t choose_block_m(const hstu_attention_params &p, int32_t threads);

/// Block g (global index) holds keys [k0, k0 + nb) of sequence b.
struct kv_block_t {
    int64_t g;
    int64_t b;
    int64_t k0;
    int64_t nb;
};

/// The kBlockN-key blocks of every sequence (the last one of a sequence is
/// short) and those that intersect the sequence's key range.
class kv_blocks_t {
public:
    kv_blocks_t(const hstu_attention_params &p,
            const std::vector<key_range_t> &need);

    int64_t count() const { return first_.back(); }
    /// Blocks to pack (global block indices).
    const std::vector<int64_t> &jobs() const { return jobs_; }
    /// Sequence b owns blocks [first_block(b), first_block(b + 1)).
    int64_t first_block(int64_t b) const { return first_[b]; }
    kv_block_t block(int64_t g) const {
        const int64_t b = std::upper_bound(first_.begin(), first_.end(), g)
                - first_.begin() - 1;
        const int64_t k0 = (g - first_[b]) * kBlockN;
        return {g, b, k0,
                std::min(kBlockN, offsets_[b + 1] - offsets_[b] - k0)};
    }

private:
    const int64_t *offsets_;
    std::vector<int64_t> first_;
    std::vector<int64_t> jobs_;
};

/// One query tile of one (sequence, head): query rows [row0, row0 + rows)
/// of sequence b.
struct work_item_t {
    int64_t b;
    int64_t h;
    int64_t row0;
    int64_t rows;
    seq_rows_t s;
};

/// Work items in SDPA order (sequence, head, query tile).
class work_plan_t {
public:
    work_plan_t(const hstu_attention_params &p, int64_t block_m);

    int64_t size() const { return first_.back(); }
    work_item_t item(int64_t z) const {
        work_item_t w;
        w.b = std::upper_bound(first_.begin(), first_.end(), z) - first_.begin()
                - 1;
        const int64_t tiles = (first_[w.b + 1] - first_[w.b]) / p_.num_heads;
        w.h = (z - first_[w.b]) / tiles;
        w.row0 = (z - first_[w.b]) % tiles * bm_;
        w.s = seq_rows(p_, w.b);
        w.rows = std::min(bm_, w.s.num_q - w.row0);
        return w;
    }

private:
    const hstu_attention_params &p_;
    int64_t bm_;
    /// Sequence b owns items [first_[b], first_[b + 1]).
    std::vector<int64_t> first_;
};

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_ATTENTION_PLAN_HPP
