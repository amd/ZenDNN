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

/// Scratch memory of the tiled HSTU attention kernels.

#ifndef LOWOHA_HSTU_TILED_SCRATCH_HPP
#define LOWOHA_HSTU_TILED_SCRATCH_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "lowoha_operators/hstu_attention/common/lowoha_hstu_mask.hpp"
#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled_plan.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

/// Uninitialized scratch owned by the calling thread and reused across calls.
/// Allocation happens outside OpenMP regions, so std::bad_alloc reaches the
/// caller; buffers above kRetainBytes are released when the call ends.
class arena_t {
public:
    char *get(size_t bytes) {
        if (cap_ < bytes) {
            data_.reset();
            cap_ = 0;
            data_.reset(new char[bytes]);
            cap_ = bytes;
        }
        return data_.get();
    }
    void trim() {
        if (cap_ > kRetainBytes) {
            data_.reset();
            cap_ = 0;
        }
    }

private:
    static constexpr size_t kRetainBytes = size_t(256) << 20;
    std::unique_ptr<char[]> data_;
    size_t cap_ = 0;
};

inline size_t round_up(size_t bytes) {
    return (bytes + 63) / 64 * 64;
}

/// The calling thread's arenas for packed K / V and for the per-thread
/// tiles, shared by every policy and output type.
arena_t &packed_arena();
arena_t &tile_arena();

/// Packed K and V of every (block, head): a block of nb keys has one slot of
/// slot_bytes(nb) bytes per head, laid out by the policy. Only the blocks in
/// `kv_blocks_t::jobs()` are packed (and touched).
class packed_kv_t {
public:
    template <typename slot_bytes_fn>
    packed_kv_t(const kv_blocks_t &blocks, int64_t heads,
            slot_bytes_fn slot_bytes, arena_t &arena) {
        offset_.resize(blocks.count() + 1, 0);
        slot_.resize(blocks.count());
        for (int64_t g = 0; g < blocks.count(); ++g) {
            slot_[g] = slot_bytes(blocks.block(g).nb);
            offset_[g + 1] = offset_[g] + slot_[g] * heads;
        }
        base_ = arena.get(offset_.back());
    }

    char *slot(int64_t g, int64_t h) const {
        return base_ + offset_[g] + h * slot_[g];
    }

private:
    std::vector<size_t> offset_;
    std::vector<size_t> slot_;
    char *base_ = nullptr;
};

/// Per-thread scratch of one query tile; the policy lays out `q` (the tile's
/// queries) and `p` (P for one key block).
struct tile_t {
    void *q;
    void *p;
    float *o;
    float *scales;
    row_band_t *bands;
    int32_t *lo;
    int32_t *hi;
};

class tile_pool_t {
public:
    tile_pool_t(int64_t threads, int64_t bm, size_t q_bytes, size_t p_bytes,
            int64_t dv, arena_t &arena)
        : q_(round_up(q_bytes))
        , p_(round_up(p_bytes))
        , o_(round_up(bm * dv * sizeof(float)))
        , scales_(round_up(bm * sizeof(float)))
        , bands_(round_up(bm * sizeof(row_band_t)))
        , bounds_(round_up(bm * sizeof(int32_t))) {
        per_thread_ = q_ + p_ + o_ + scales_ + bands_ + 2 * bounds_;
        base_ = arena.get(per_thread_ * threads);
    }

    tile_t get(int thread) const {
        char *c = base_ + per_thread_ * thread;
        tile_t t;
        t.q = c;
        c += q_;
        t.p = c;
        c += p_;
        t.o = reinterpret_cast<float *>(c);
        c += o_;
        t.scales = reinterpret_cast<float *>(c);
        c += scales_;
        t.bands = reinterpret_cast<row_band_t *>(c);
        c += bands_;
        t.lo = reinterpret_cast<int32_t *>(c);
        c += bounds_;
        t.hi = reinterpret_cast<int32_t *>(c);
        return t;
    }

private:
    size_t q_, p_, o_, scales_, bands_, bounds_;
    size_t per_thread_ = 0;
    char *base_ = nullptr;
};

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_TILED_SCRATCH_HPP
