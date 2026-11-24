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

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/nr32_pack.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "common/zendnnl_compat.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {

namespace {

float scale_at(const void *scales, data_type_t dt, size_t i) {
    if (dt == data_type_t::bf16)
        return static_cast<float>(static_cast<const bfloat16_t *>(scales)[i]);
    return static_cast<const float *>(scales)[i];
}

/// `out`'s scales: `{G, N}` weight scales `scales` as a pack of `layout`
/// keeps them, `[G][n_pad]` f32 with zeros in the padded columns.
void fill_scales(const pack_layout &layout, const void *scales,
        const data_type_t dt, packed_expert &out) {
    const int N = layout.desc.N;
    out.scale.assign(static_cast<size_t>(layout.groups) * layout.n_pad, 0.0f);
    float max_scale = 0.0f;
    bool nan = false;
    for (int g = 0; g < layout.groups; ++g) {
        for (int col = 0; col < N; ++col) {
            const float v
                    = scale_at(scales, dt, static_cast<size_t>(g) * N + col);
            out.scale[static_cast<size_t>(g) * layout.n_pad + col] = v;
            nan = nan || std::isnan(v);
            max_scale = std::max(max_scale, std::fabs(v));
        }
    }
    out.max_scale = nan ? std::numeric_limits<float>::quiet_NaN() : max_scale;
}

} // namespace

int padded_cols(const int N) {
    return (N + kNr - 1) / kNr * kNr;
}

size_t block_bytes(const weight_desc &d) {
    const format_desc *f = d.fmt;
    if (f == nullptr) return 0;
    const size_t rows
            = static_cast<size_t>((d.K + f->k_per_word - 1) / f->k_per_word);
    return rows * kNr * f->word_bytes;
}

bool packable(const weight_desc &d) {
    const format_desc *f = d.fmt;
    if (f == nullptr || d.K <= 0 || d.N <= 0) return false;
    if (d.ldb < (d.trans ? d.K : d.N)) return false;
    if (f->always_grouped || (f->scaled() && d.group_size != 0)) {
        return d.group_size > 0 && d.group_size % f->k_per_word == 0
                && d.K % d.group_size == 0;
    }
    return d.group_size == 0;
}

pack_layout layout_of(const weight_desc &desc) {
    pack_layout layout;
    layout.desc = desc;
    layout.n_pad = padded_cols(desc.N);
    layout.groups = desc.group_size > 0 ? desc.K / desc.group_size : 1;
    layout.block_bytes = block_bytes(desc);
    return layout;
}

size_t packed_weights::bytes() const {
    size_t total = 0;
    for (const packed_expert &e : experts) {
        if (e.blocks != nullptr) total += block_bytes * (n_pad / kNr);
        total += (e.comp != nullptr ? e.comp->size() * sizeof(int32_t) : 0)
                + e.scale.size() * sizeof(float);
    }
    return total;
}

status_t prepack(const weight_desc &desc,
        const std::vector<const void *> &weights,
        const std::vector<const void *> &scales, const data_type_t scale_dt,
        const int num_threads, packed_weights &out) {
    out = packed_weights {};
    const int E = static_cast<int>(weights.size());
    const format_desc *f = desc.fmt;
    const bool scaled = f != nullptr && f->scaled() && !scales.empty();
    if (!packable(desc) || E == 0
            || (scaled
                    && (static_cast<int>(scales.size()) != E
                            || (scale_dt != data_type_t::f32
                                    && scale_dt != data_type_t::bf16)))) {
        return status_t::unimplemented;
    }
    for (int e = 0; e < E; ++e) {
        if (weights[e] == nullptr || (scaled && scales[e] == nullptr))
            return status_t::unimplemented;
    }

    packed_weights pw;
    static_cast<pack_layout &>(pw) = layout_of(desc);
    pw.experts.resize(static_cast<size_t>(E));
    const size_t expert_bytes = pw.block_bytes * (pw.n_pad / kNr);
    const size_t cols = static_cast<size_t>(pw.groups) * pw.n_pad;

    const pack_fn pack = f->pack;
    std::atomic<bool> failed {false};
    // No exception may leave the region: an allocation that fails fails the
    // pack.
#pragma omp parallel for schedule(dynamic) num_threads(std::max(num_threads, 1))
    for (int e = 0; e < E; ++e) {
        if (failed.load(std::memory_order_relaxed)) continue;
        try {
            if (test_api::s_pack_allocation_fails.load(
                        std::memory_order_relaxed)) {
                throw std::bad_alloc();
            }
            packed_expert &px = pw.experts[e];
            void *raw = zendnnl_aligned_alloc(64, expert_bytes);
            if (raw == nullptr) {
                failed.store(true, std::memory_order_relaxed);
                continue;
            }
            px.blocks = std::shared_ptr<uint8_t>(static_cast<uint8_t *>(raw),
                    [](uint8_t *p) { zendnnl_aligned_free(p); });
            std::shared_ptr<std::vector<int32_t>> comp;
            if (f->sums) comp = std::make_shared<std::vector<int32_t>>(cols, 0);
            if (scaled) fill_scales(pw, scales[e], scale_dt, px);
            pack_out to;
            to.blocks = px.blocks.get();
            to.block_bytes = pw.block_bytes;
            to.n_pad = pw.n_pad;
            to.sums = comp != nullptr ? comp->data() : nullptr;
            pack(weights[e], desc, to);
            px.comp = std::move(comp);
        } catch (const std::bad_alloc &) {
            failed.store(true, std::memory_order_relaxed);
        }
    }
    if (failed.load()) return status_t::memory_bad_size;
    out = std::move(pw);
    return status_t::success;
}

status_t with_scales(const pack_layout &layout, const packed_expert &bare,
        const void *scales, const data_type_t scale_dt, packed_expert &out) {
    if (scales == nullptr
            || (scale_dt != data_type_t::f32 && scale_dt != data_type_t::bf16)
            || layout.desc.fmt == nullptr || !layout.format().scaled()) {
        return status_t::unimplemented;
    }
    out.blocks = bare.blocks;
    out.comp = bare.comp;
    fill_scales(layout, scales, scale_dt, out);
    return status_t::success;
}

void unpack(const packed_weights &pw, const int e, void *dst, const int ld) {
    const weight_desc &d = pw.desc;
    const format_desc &f = pw.format();
    const uint8_t *blocks = pw.blocks(e);
    const size_t ldz = static_cast<size_t>(ld);
    const size_t es = static_cast<size_t>(f.elem_bits / 8);
    for (int col = 0; col < d.N; ++col) {
        const uint8_t *blk = blocks + (col / kNr) * pw.block_bytes;
        const int n = col % kNr;
        for (int k = 0; k < d.K; ++k) {
            const uint8_t *word = blk
                    + (static_cast<size_t>(k / f.k_per_word) * kNr + n)
                            * f.word_bytes;
            const size_t idx = col * ldz + k;
            if (f.enc == encode::copy) {
                std::memcpy(static_cast<uint8_t *>(dst) + idx * es,
                        word + (k % f.k_per_word) * es, es);
                continue;
            }
            const int q = k % 8;
            const uint8_t byte = word[q & 3];
            const uint8_t raw = static_cast<uint8_t>(
                    ((q < 4 ? byte : byte >> 4) & 0x0Fu) ^ 0x08u);
            auto *out = static_cast<uint8_t *>(dst) + (idx >> 1);
            *out = (idx & 1u) == 0u
                    ? static_cast<uint8_t>((*out & 0xF0u) | raw)
                    : static_cast<uint8_t>((*out & 0x0Fu) | (raw << 4));
        }
    }
}

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
