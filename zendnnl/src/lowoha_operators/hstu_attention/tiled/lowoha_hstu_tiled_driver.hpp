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

/// The data-type independent part of the tiled HSTU attention kernels: the
/// parallel loops, the per-tile row and key windows and the output store.
/// The data type lives in a policy (see run_tiled), so each policy
/// is its own translation unit and the hot loop has no indirection.

#ifndef LOWOHA_HSTU_TILED_DRIVER_HPP
#define LOWOHA_HSTU_TILED_DRIVER_HPP

#include <algorithm>
#include <cstdint>
#include <omp.h>
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/hstu_attention/common/lowoha_hstu_mask.hpp"
#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled_plan.hpp"
#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled_scratch.hpp"
#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled_store.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

/// Rows [ra, rb) of a query tile and keys [plo, phi) of block g (keys
/// [k0, k0 + nb) of the sequence) that some row of the tile reads.
struct block_window_t {
    int64_t g;
    int64_t k0;
    int64_t nb;
    int64_t ra, rb;
    int64_t plo, phi;
};

/// Bands and scales of the tile's rows and O = 0; returns the keys the bands
/// span (empty when no row has a band).
inline key_range_t prepare_rows(
        const hstu_attention_params &p, const work_item_t &w, const tile_t &t) {
    const seq_mask_t mask(p, w.s.len, w.s.targets);
    const int64_t pos0 = w.s.q_pos + w.row0;
    key_range_t keys {w.s.len, 0};
    for (int64_t r = 0; r < w.rows; ++r) {
        t.bands[r] = mask.band(pos0 + r);
        t.scales[r] = row_scale(p, w.s, pos0 + r);
        if (t.bands[r].lo < t.bands[r].hi) {
            keys.lo = std::min(keys.lo, t.bands[r].lo);
            keys.hi = std::max(keys.hi, t.bands[r].hi);
        }
    }
    std::fill_n(t.o, w.rows * p.v_dim, 0.f);
    return keys;
}

/// The window of block g, keys [k0, k0 + nb), for rows [0, rows) with `bands`;
/// false when no row reads the block. plo and phi are rounded out to
/// multiples of key_align (phi to at most nb), so packed V groups stay
/// whole; the policy writes P for every key of the window.
template <int64_t key_align>
inline bool block_window(const row_band_t *bands, int64_t rows, int64_t g,
        int64_t k0, int64_t nb, block_window_t &bw) {
    int64_t ra = rows, rb = 0, plo = nb, phi = 0;
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t lo = std::max(bands[r].lo, k0) - k0;
        const int64_t hi = std::min(bands[r].hi, k0 + nb) - k0;
        if (lo < hi) {
            ra = std::min(ra, r);
            rb = r + 1;
            plo = std::min(plo, lo);
            phi = std::max(phi, hi);
        }
    }
    if (ra >= rb) return false;
    bw.g = g;
    bw.k0 = k0;
    bw.nb = nb;
    bw.ra = ra;
    bw.rb = rb;
    bw.plo = plo / key_align * key_align;
    bw.phi = std::min(nb, (phi + key_align - 1) / key_align * key_align);
    return true;
}

/// Keys [lo[r], hi[r]) that rows [ra, rb) keep among keys [c0, c0 + width)
/// of the sequence, relative to c0.
inline void chunk_bounds(const row_band_t *bands, int64_t ra, int64_t rb,
        int64_t c0, int64_t width, int32_t *lo, int32_t *hi) {
    for (int64_t r = ra; r < rb; ++r) {
        lo[r] = int32_t(std::clamp<int64_t>(bands[r].lo - c0, 0, width));
        hi[r] = int32_t(std::clamp<int64_t>(bands[r].hi - c0, 0, width));
    }
}

/**
 * @brief One call of a tiled kernel: thread count, planning, packing, tiles.
 *
 * K/V blocks are packed once (static loop over (block, head) jobs), then the
 * work items run in SDPA order with schedule(dynamic, 1). A work item visits
 * the key blocks of its rows' bands; the diagonal of a row outside its band
 * is added separately, and O is stored in the output type.
 *
 * `policy_t` is the data-type policy; its members are inlined:
 *   static constexpr int64_t kKeyAlign;   key group of packed V
 *   static size_t q_bytes(int64_t bm, int64_t d);   Q tile of bm rows
 *   static size_t p_bytes(int64_t bm);              P tile, bm x kBlockN
 *   size_t slot_bytes(int64_t nb) const;            packed K/V, nb keys, 1 head
 *   // `scratch`: the thread's tile, free while packing
 *   void pack(const kv_block_t &, int64_t h, char *slot,
 *           const tile_t &scratch) const;
 *   void load_q(const work_item_t &, const tile_t &) const;
 *   // P of rows [ra, rb) x keys [plo, phi), then O[ra, rb) += P V
 *   void qk(const block_window_t &, const char *slot, const tile_t &) const;
 *   void pv(const block_window_t &, const char *slot, const tile_t &) const;
 *   // o += the diagonal term of row r
 *   void diag(const work_item_t &, int64_t r, const tile_t &, float *o) const;
 */
template <typename policy_t, typename out_t>
void run_tiled(
        const hstu_attention_params &p, const policy_t &policy, out_t *out) {
    const int64_t H = p.num_heads, Dv = p.v_dim;

    // The live ICV, not thread_guard::max_threads(): restoring the count cached
    // by the first ZenDNN call would undo a later omp_set_num_threads().
    const int32_t omp_max = omp_get_max_threads();
    const int32_t nt = resolve_num_threads(p.num_threads, omp_max);
    thread_guard guard(nt, omp_max);

    const int64_t BM = choose_block_m(p, nt);
    const kv_blocks_t blocks(p, key_ranges(p));
    const packed_kv_t packed(blocks, H,
            [&](int64_t nb) { return policy.slot_bytes(nb); }, packed_arena());
    const tile_pool_t pool(nt, BM, policy_t::q_bytes(BM, p.qk_dim),
            policy_t::p_bytes(BM), Dv, tile_arena());
    const work_plan_t plan(p, BM);
    const int64_t num_jobs = static_cast<int64_t>(blocks.jobs().size());

#pragma omp parallel for schedule(static) num_threads(nt)
    for (int64_t j = 0; j < num_jobs * H; ++j) {
        const kv_block_t blk = blocks.block(blocks.jobs()[j / H]);
        const int64_t h = j % H;
        policy.pack(
                blk, h, packed.slot(blk.g, h), pool.get(omp_get_thread_num()));
    }

    // Dynamic rather than SDPA's static schedule: under a causal mask the
    // last tiles of a sequence read the most keys, and a static split leaves
    // cores idle (0.73-0.77x at model level on DLRM v3).
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
    for (int64_t z = 0; z < plan.size(); ++z) {
        const work_item_t w = plan.item(z);
        const tile_t t = pool.get(omp_get_thread_num());
        policy.load_q(w, t);
        const key_range_t keys = prepare_rows(p, w, t);
        for (int64_t k0 = keys.lo / kBlockN * kBlockN; k0 < keys.hi;
                k0 += kBlockN) {
            block_window_t bw;
            if (!block_window<policy_t::kKeyAlign>(t.bands, w.rows,
                        blocks.first_block(w.b) + k0 / kBlockN, k0,
                        std::min(kBlockN, w.s.len - k0), bw))
                continue;
            const char *slot = packed.slot(bw.g, w.h);
            policy.qk(bw, slot, t);
            policy.pv(bw, slot, t);
        }
        for (int64_t r = 0; r < w.rows; ++r) {
            float *o = t.o + r * Dv;
            if (t.bands[r].extra_diag) policy.diag(w, r, t, o);
            store_row(o, Dv,
                    out + (w.s.q_row + w.row0 + r) * p.o_stride_t
                            + w.h * p.o_stride_h);
        }
    }
    packed_arena().trim();
    tile_arena().trim();
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_TILED_DRIVER_HPP
