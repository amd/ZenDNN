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

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/api/routed.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>
#include <unordered_map>

#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/api/api_common.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/engine.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/scratch.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/engine/source.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/formats/formats.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/scales.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/store.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

namespace {

struct layer_key_hash_t {
    size_t operator()(const layer_key &k) const noexcept {
        size_t h = 0;
        const auto mix
                = [&h](size_t v) { h = (h ^ v) * 0x9E3779B97F4A7C15ULL; };
        for (const projection_key *p : {&k.w13, &k.w2}) {
            mix(std::hash<const void *> {}(p->weight));
            mix(static_cast<size_t>(p->ldb));
            mix(p->expert_stride);
            mix(static_cast<size_t>(p->trans_weight));
        }
        for (const int v : {k.num_experts, k.H, k.I, static_cast<int>(k.act),
                     nr32::format_index(*k.fmt), static_cast<int>(k.w13_bias),
                     static_cast<int>(k.w2_bias),
                     static_cast<int>(k.w13_src_groups),
                     static_cast<int>(k.w2_src_groups)}) {
            mix(static_cast<size_t>(v));
        }
        return h;
    }
};

/// One projection of a routed layer: its experts' packs, from the pack
/// store, and how its tiles read them.
struct projection_state_t {
    nr32::pack_layout layout;
    std::vector<expert_pack> packs;
    /// `packs`' pointers, as the engine reads them.
    std::vector<const nr32::packed_expert *> raw;
    nr32::tile_plan plan;
};

struct layer_state_t {
    std::once_flag packed;
    /// Null when the layer is served, else why not.
    const char *unserved = nullptr;
    /// Both projections run the kernels that read source bytes XOR-ed with
    /// 0x80, so calls flip their s8 rows.
    bool u8_src = false;
    layer_scales scales;
    projection_state_t w13, w2;
};

std::mutex &layer_mutex() {
    static auto *mutex = new std::mutex;
    return *mutex;
}

std::unordered_map<layer_key, std::shared_ptr<layer_state_t>,
        layer_key_hash_t> &
layer_states() {
    static auto *layers = new std::unordered_map<layer_key,
            std::shared_ptr<layer_state_t>, layer_key_hash_t>;
    return *layers;
}

/// What packing a routed layer's projection takes, decided before anything
/// is packed: the weight as its packs read it, and its experts' scales.
struct projection_plan_t {
    nr32::weight_desc desc;
    scale_form_t form;
    std::vector<const void *> scales;
};

/// `p` for all `num_experts` experts of format `f`, with `groups` weight-scale
/// groups along K for a scaled format: in `pp`, the weight and its scales; in
/// `out`, the layout
/// and the plan of its tiles for epilogue `ep` and, with `src_groups`, one
/// source scale per row and group, sized by the routed settings.  Null when
/// this CPU runs it, else why not.
const char *plan_projection(projection_state_t &out, projection_plan_t &pp,
        const projection &p, const int num_experts, const nr32::format_desc &f,
        const int64_t groups, const nr32::epilogue ep, const bool src_u8,
        const bool src_groups) {
    const int E = num_experts;
    if (static_cast<int>(p.weight.size()) != E
            || static_cast<int>(p.params.size()) != E) {
        return "weights_unavailable";
    }
    nr32::weight_desc &desc = pp.desc;
    desc.fmt = &f;
    desc.K = p.K;
    desc.N = p.N;
    desc.trans = p.trans_weight;
    desc.ldb = p.ldb;
    if (f.always_grouped || (f.scaled() && groups > 1)) {
        if (groups <= 0 || p.K % groups != 0) return "unsupported_weight";
        desc.group_size = static_cast<int>(p.K / groups);
    }
    if (!nr32::packable(desc)) return "unsupported_weight";
    pp.scales.assign(static_cast<size_t>(E), nullptr);
    if (f.scaled()) {
        pp.form.dt = p.params[0].quant_params.wei_scale.dt;
        pp.form.groups = std::max<int64_t>(groups, 1);
        for (int e = 0; e < E; ++e)
            pp.scales[e] = p.params[e].quant_params.wei_scale.buff;
    }
    out.layout = nr32::layout_of(desc);
    if (!nr32::make_plan(out.layout, ep, src_u8, out.plan, src_groups,
                budget_for(f, api_kind::routed))) {
        return "no_kernel";
    }
    return nullptr;
}

/// `out`'s packs: `p`'s experts' packs from the pack store, packed now on
/// `num_threads` threads where it lacks them, for a call that began at flush
/// count `since`.  Null on success, else why not.
const char *pack_projection(projection_state_t &out,
        const projection_plan_t &pp, const projection &p, const uint64_t since,
        const int num_threads) {
    const char *unserved = find_packs(pp.desc, pp.form, p.weight, pp.scales,
            since, num_threads, out.packs);
    if (unserved != nullptr) return unserved;
    out.raw.clear();
    for (const expert_pack &pack : out.packs) {
        if (pack == nullptr) return "weights_unavailable";
        out.raw.push_back(pack.get());
    }
    return nullptr;
}

/// The layer's packs and plans, made on its first call and again when its
/// scales change.
std::shared_ptr<const layer_state_t> layer_state(const layer_key &key,
        const layer_scales &scales, const describe_layer &describe,
        const int num_threads) {
    std::shared_ptr<layer_state_t> layer;
    uint64_t since = 0;
    {
        std::lock_guard<std::mutex> guard(layer_mutex());
        since = flushes().load(std::memory_order_relaxed);
        if (since % 2 != 0) {
            // A flush runs: the layer's state is the call's own.
            layer = std::make_shared<layer_state_t>();
            layer->scales = scales;
        } else {
            std::shared_ptr<layer_state_t> &entry = layer_states()[key];
            if (entry == nullptr || !(entry->scales == scales)) {
                entry = std::make_shared<layer_state_t>();
                entry->scales = scales;
            }
            layer = entry;
        }
    }
    std::call_once(layer->packed, [&] {
        layer->u8_src = u8_src_pays(*key.fmt, api_kind::routed);
        projection w13, w2;
        projection_plan_t pp13, pp2;
        layer->unserved = describe(w13, w2);
        if (layer->unserved == nullptr) {
            layer->unserved = plan_projection(layer->w13, pp13, w13,
                    key.num_experts, *key.fmt, scales.w13_groups,
                    epilogue_of(key.act), layer->u8_src, key.w13_src_groups);
        }
        if (layer->unserved == nullptr) {
            layer->unserved = plan_projection(layer->w2, pp2, w2,
                    key.num_experts, *key.fmt, scales.w2_groups,
                    nr32::epilogue::plain, layer->u8_src, key.w2_src_groups);
        }
        if (layer->unserved == nullptr) {
            layer->unserved = pack_projection(
                    layer->w13, pp13, w13, since, num_threads);
        }
        if (layer->unserved == nullptr) {
            layer->unserved
                    = pack_projection(layer->w2, pp2, w2, since, num_threads);
        }
        if (layer->unserved != nullptr) {
            layer->w13 = projection_state_t {};
            layer->w2 = projection_state_t {};
        }
        if (test_api::s_capture_ntile_flat_parallel.load(
                    std::memory_order_relaxed)) {
            test_api::s_ntile_flat_parallel_layer_entries.fetch_add(
                    1, std::memory_order_relaxed);
        }
        static const bool s_log = apilog_info_enabled();
        if (s_log) {
            apilog_info("[GRP_MATMUL.NTILE_FLAT_PARALLEL] layer E=",
                    key.num_experts, " H=", key.H, " I=", key.I,
                    " served=", layer->unserved == nullptr ? 1 : 0);
        }
    });
    if (layer->unserved == kOutOfMemory) {
        // A shortage passes: the next call packs the layer again.
        std::lock_guard<std::mutex> guard(layer_mutex());
        const auto it = layer_states().find(key);
        if (it != layer_states().end() && it->second == layer)
            layer_states().erase(it);
    }
    return layer;
}

/// A routed call's rows: each (token, slot) pair is a row of its expert, in
/// token order, `S.row_token` naming its token.  A layer whose W13 reads s8
/// rows quantizes each token once and copies its row and scales to its
/// slots; another copies the token's row.  The W13 units that reach a run
/// first share the copies of its rows, a few at a time, so the copies overlap
/// other units' work and no unit waits long.  Op1 writes the scratch's
/// intermediate rows and Op2 its staging rows, which the reduce sums into the
/// output.
struct routed_front {
    scratch_t &S;
    const moe_call &call;
    int H, I;
    source_spec src13, src2;
    /// Expert e's biases are row e of the caller's `[E, N]` tensors.
    const char *bias13;
    size_t bias13_row;
    nr32::bias_kind bias13_t;
    const char *bias2;
    size_t bias2_row;
    nr32::bias_kind bias2_t;

    void prologue() const {
        if (!src13.s8) return;
#pragma omp for schedule(static)
        for (int t = 0; t < call.num_tokens; ++t) {
            quantize_row(src13, call.src + static_cast<size_t>(t) * call.src_ld,
                    H, S.tok, t);
        }
    }

    void before_w13(const int r) const {
        const row_run_t run = S.runs[r];
        const row_block_t &tail = S.blocks[run.block + run.blocks - 1];
        const int first = S.blocks[run.block].row;
        const int count = tail.row + tail.rows - first;
        const int gather = limits().gather_rows;
        for (;;) {
            const int i = S.gather_next[r].fetch_add(
                    gather, std::memory_order_relaxed);
            if (i >= count) break;
            const int n = std::min(gather, count - i);
            for (int row = first + i; row < first + i + n; ++row) {
                const int t = S.row_token[row];
                if (src13.s8) {
                    copy_row(src13, S.tok, t, S.src13, row, H);
                } else {
                    copy_in(src13,
                            call.src + static_cast<size_t>(t) * call.src_ld, H,
                            S.src13, row);
                }
            }
            S.gather_done[r].fetch_add(n, std::memory_order_release);
        }
        wait_for(S.gather_done[r], count);
    }

    tile_io w13(const row_block_t &b) const {
        tile_io io;
        io.src = S.src13.view(src13, b.row, H);
        io.bias = bias13 != nullptr ? bias13 + b.e * bias13_row : nullptr;
        io.bias_t = bias13_t;
        io.dst = S.inter.data() + static_cast<size_t>(b.row) * I;
        io.ldc = I;
        return io;
    }

    tile_io w2(const row_block_t &b) const {
        tile_io io;
        if (src2.s8) {
            io.src = S.src2.view(src2, b.row, I);
        } else {
            io.src.rows = op1_row(b.e, b.row);
            io.src.lda = I;
        }
        io.bias = bias2 != nullptr ? bias2 + b.e * bias2_row : nullptr;
        io.bias_t = bias2_t;
        io.dst = S.stage.data() + static_cast<size_t>(b.row) * H;
        io.ldc = H;
        return io;
    }

    const uint16_t *op1_row(int, const int row) const {
        return S.inter.data() + static_cast<size_t>(row) * I;
    }
};

status_t run_call(const layer_key &key, const layer_scales &scales,
        const describe_layer &describe, const moe_call &call) {
    const int E = key.num_experts, H = key.H, I = key.I;
    const int num_tokens = call.num_tokens, topk = call.topk;
    const int num_threads = call.num_threads;
    const int num_pairs = num_tokens * topk;
    const std::shared_ptr<const layer_state_t> layer
            = layer_state(key, scales, describe, num_threads);
    if (layer->unserved != nullptr) { return refuse(layer->unserved); }
    const layer_state_t &L = *layer;
    const int *const counts = call.counts;

    // Expert-major rows: expert e owns rows [offset[e], offset[e+1]), in
    // token order; `row_of` maps each (token, slot) pair to its row for the
    // reduce.
    scratch_t &S = scratch();
    S.offset.assign(static_cast<size_t>(E) + 1, 0);
    for (int e = 0; e < E; ++e)
        S.offset[e + 1] = S.offset[e] + counts[e];
    S.fill.assign(S.offset.begin(), S.offset.end() - 1);
    grow(S.row_token, static_cast<size_t>(num_pairs));
    grow(S.row_of, static_cast<size_t>(num_pairs));
    for (int p = 0; p < num_pairs; ++p) {
        const int r = S.fill[call.local_experts[p]]++;
        S.row_token[r] = p / topk;
        S.row_of[p] = r;
    }

    // What the projections' kernels read: s8 rows of the source and of the
    // intermediate with their scales, or both as they are.
    const source_spec src13 = source_of(L.w13.layout, L.w13.plan);
    const source_spec src2 = source_of(L.w2.layout, L.w2.plan);
    S.src13.grow(src13, static_cast<size_t>(num_pairs), H);
    if (src13.s8) S.tok.grow(src13, static_cast<size_t>(num_tokens), H);
    if (src2.s8) S.src2.grow(src2, static_cast<size_t>(num_pairs), I);
    grow(S.inter, static_cast<size_t>(num_pairs) * I);
    grow(S.stage, static_cast<size_t>(num_pairs) * H);
    grow(S.row_ptrs, static_cast<size_t>(num_pairs));
    for (int p = 0; p < num_pairs; ++p) {
        S.row_ptrs[p] = S.stage.data() + static_cast<size_t>(S.row_of[p]) * H;
    }
    group_matmul_moe_postop_params reduce;
    reduce.num_tokens = num_tokens;
    reduce.topk = topk;
    reduce.output = call.dst;
    reduce.ldc_output = call.dst_ld;
    reduce.topk_weights = call.topk_weights;
    reduce.skip_weighted = call.skip_weighted;
    reduce.row_ptrs = S.row_ptrs.data();

    const char *const bias13 = static_cast<const char *>(call.w13_bias);
    const char *const bias2 = static_cast<const char *>(call.w2_bias);
    const routed_front front {S, call, H, I, src13, src2, bias13,
            bias13 != nullptr
                    ? static_cast<size_t>(size_of(key.w13_bias)) * 2 * I
                    : 0,
            bias_kind_of(key.w13_bias), bias2,
            bias2 != nullptr ? static_cast<size_t>(size_of(key.w2_bias)) * H
                             : 0,
            bias_kind_of(key.w2_bias)};
    engine_t en;
    en.n = E;
    en.rows = counts;
    en.fmt = key.fmt;
    en.api = api_kind::routed;
    en.fused = true;
    en.wei_bytes = key.fmt->elem_bits / 8.0f;
    en.K13 = H;
    en.K2 = I;
    en.src13 = src13;
    en.src2 = src2;
    en.l13 = &L.w13.layout;
    en.l2 = &L.w2.layout;
    en.p13 = &L.w13.plan;
    en.p2 = &L.w2.plan;
    en.packs13 = L.w13.raw.data();
    en.packs2 = L.w2.raw.data();
    en.reduce = &reduce;
    en.reduce_cols = H;
    en.reduce_f16 = key.fmt->dst == data_type_t::f16;
    en.num_threads = num_threads;
    const status_t status = run_engine(en, front);
    if (test_api::s_capture_ntile_flat_parallel.load(
                std::memory_order_relaxed)) {
        test_api::s_ntile_flat_parallel_completed.fetch_add(
                1, std::memory_order_relaxed);
        if (L.u8_src) {
            test_api::s_ntile_flat_parallel_u8_src_calls.fetch_add(
                    1, std::memory_order_relaxed);
        }
    }
    return status;
}

} // namespace

void clear_layers() {
    assert(flushes().load(std::memory_order_relaxed) % 2 == 1);
    std::lock_guard<std::mutex> guard(layer_mutex());
    layer_states().clear();
}

size_t layer_count() {
    std::lock_guard<std::mutex> guard(layer_mutex());
    return layer_states().size();
}

} // namespace detail

status_t execute(const layer_key &key, const layer_scales &scales,
        const describe_layer &describe, const moe_call &call) {
    if (test_api::s_pause_ntile_flat_parallel.load(std::memory_order_acquire)) {
        test_api::s_ntile_flat_parallel_paused.store(
                true, std::memory_order_release);
        while (test_api::s_pause_ntile_flat_parallel.load(
                std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }
    if (key.fmt == nullptr) return detail::decline("unsupported_types");
    if (key.act == grp_matmul_gated_act_t::none || key.num_experts < 1
            || call.num_tokens <= 0 || call.topk <= 0
            || call.num_threads <= 0) {
        return detail::decline("shape");
    }
    const int num_pairs = call.num_tokens * call.topk;
    for (int p = 0; p < num_pairs; ++p) {
        if (call.local_experts[p] < 0)
            return detail::decline("non_resident_expert");
        if (call.local_experts[p] >= key.num_experts)
            return detail::decline("shape");
    }
    try {
        return detail::run_call(key, scales, describe, call);
    } catch (const std::length_error &) {
        return status_t::memory_bad_size;
    } catch (const std::bad_alloc &) { return status_t::memory_bad_storage; }
}

} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
