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

/// ALGO 4 — N-tile flat-parallel MoE.
///
/// Library-internal.  Runs one call of an MoE layer (W13 with its gated
/// activation, W2, and the top-k weighted reduce) in a single parallel region
/// over expert-major rows, on one engine for both APIs: a routed call
/// (`execute`) first routes its tokens to their experts' rows, and a grouped
/// call (`execute_grouped`) brings its operations' rows already grouped.  For
/// DA8W8 and W4A8 experts a routed call quantizes each token once and copies
/// its s8 row to its slots; BF16 and FP16 rows are copied as they are.  Each
/// expert's rows are cut into blocks of at most `kDecodeMaxM`
/// rows, each expert's blocks into row runs, and both projections into
/// (row runs, column range) units of whole kernel blocks of columns, queued
/// every W13 unit before any W2 unit.  The first W13 units to reach a run
/// share the copies of its rows.  The thread that finishes a run's last W13
/// unit quantizes that run's intermediate rows when the weights are
/// quantized, and the W2 units over the run wait only for that.
///   * A decode call (one block per expert) has one block per run, and its
///     queue takes the heaviest expert first.  When the team has more
///     threads than one L3 has cores, each thread runs its own units first,
///     every T-th of the queue, its W13 units before its W2 units, so a call
///     repeated with the same experts and rows keeps each unit on one thread
///     and its weights in that thread's L3; then it takes any unit still
///     unclaimed.  A smaller team takes units in queue order.
///   * A prompt call cuts a heavily routed expert into several runs, as many
///     as keep the source rows and the weights its units read about equal; a
///     W2 unit may cover several of an expert's runs.  A unit sweeps its blocks
///     against one L2-sized chunk of its columns at a time, so it streams each
///     of its weights once.  Each projection's units take turns from the
///     heaviest and the lightest experts, so units that keep the cores busy
///     run beside units that stream weights, and threads claim them in queue
///     order, so every W13 unit is claimed before any W2 unit.
/// Units run the `nr32_tile.hpp` tiles over the layer's `nr32_pack.hpp` packs,
/// whose kernels read the gate and up columns of the gated activation where
/// the pack keeps them.  Where the kernels that read source bytes XOR-ed with
/// 0x80 pay, DA8W8 layers run them and calls flip their s8 rows.  Each
/// token's top-k rows are reduced with `group_matmul_moe_postop_execute`'s
/// numerics.
///
/// Kept state: one pack per expert weight (about the size of the weight, with
/// its sums), made by the first call of either API that reads the weight and
/// shared by every later call that reads the same weight and shape, whatever
/// its activation, bias or source scales; for DA8W8 and W4A8 it also keeps
/// the last weight scales a call gave it, which a call with other scale
/// pointers replaces.  Per routed layer, its experts' packs and its tile
/// plans: one routed entry per layer (weights, shapes, activation, dtypes
/// and source scale groups), replaced when the layer is seen with other
/// weight-scale pointers or groups.  Per grouped call site (the weights its
/// calls name, in any order), pointers to its experts' packs, which it keeps
/// alive, and the calls' tile plans: a call that gives an expert other scales
/// drops every site, so a
/// site keeps only packs the store holds; at most 1,024 sites, all dropped
/// when one more comes.  Packs stay until `ntile_flat_parallel::flush()`,
/// which drops them, every routed entry and every site, even those of weights
/// no call reads anymore.
///
/// Scratch, per calling thread: about (3H + 3I + 24) bytes per (token, slot)
/// pair and (H + 4) per token of the largest quantized call the thread has
/// run, and (4H + 2I + 16) per pair of the largest 16-bit one, kept until the
/// thread exits or calls `reset_thread_local_scratch()`; a call that needs
/// more than 1 GiB frees its scratch when it returns.
///
/// Reached from `routed_fused_moe_direct` when ALGO 4 is requested for the
/// call's phase (`resolve_grp_matmul_ntile_flat_parallel_request`): a decode
/// call while no expert has more than `kDecodeMaxM` rows, else a prompt call.
/// `group_matmul_direct` sends a call of two or more operations in such a
/// phase to `execute_grouped`, which runs it on the same engine and packs
/// with the operations as the experts.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "../../common/group_matmul_parallel_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {

/// One projection of the layer for every expert, as the prepack reads it.
struct projection {
    int K = 0;
    int N = 0;
    int ldb = 0;
    bool trans_weight = true;
    /// One per expert.
    std::vector<const void *> weight;
    /// One per expert, with that expert's weight scales.
    std::vector<matmul_params> params;
};

/// One projection's weights.
struct projection_key {
    const void *weight = nullptr;
    int ldb = 0;
    size_t expert_stride = 0;
    bool trans_weight = true;

    bool operator==(const projection_key &o) const {
        return weight == o.weight && ldb == o.ldb
                && expert_stride == o.expert_stride
                && trans_weight == o.trans_weight;
    }
};

/// Identifies a layer: its weights and shapes.
struct layer_key {
    projection_key w13;
    projection_key w2;
    int num_experts = 0;
    /// Hidden size: W13's K and W2's N.
    int H = 0;
    /// Intermediate size: W2's K; W13's N is `2 * I`.
    int I = 0;
    grp_matmul_gated_act_t act = grp_matmul_gated_act_t::none;
    data_type_t wei = data_type_t::none;
    /// Each projection's bias type: `none` without a bias.
    data_type_t w13_bias = data_type_t::none;
    data_type_t w2_bias = data_type_t::none;
    /// Each quantized projection's tokens take one source scale per weight
    /// scale group rather than one.
    bool w13_src_groups = false;
    bool w2_src_groups = false;

    bool operator==(const layer_key &o) const {
        return w13 == o.w13 && w2 == o.w2 && num_experts == o.num_experts
                && H == o.H && I == o.I && act == o.act && wei == o.wei
                && w13_bias == o.w13_bias && w2_bias == o.w2_bias
                && w13_src_groups == o.w13_src_groups
                && w2_src_groups == o.w2_src_groups;
    }
};

/// The layer's weight scales.  A layer seen with other scales than the ones
/// its state was packed from is packed again.
struct layer_scales {
    const void *w13 = nullptr;
    data_type_t w13_dt = data_type_t::none;
    const void *w2 = nullptr;
    data_type_t w2_dt = data_type_t::none;
    /// Scale groups along K per expert: 1 for per-channel scales.
    int64_t w13_groups = 1;
    int64_t w2_groups = 1;

    bool operator==(const layer_scales &o) const {
        return w13 == o.w13 && w13_dt == o.w13_dt && w2 == o.w2
                && w2_dt == o.w2_dt && w13_groups == o.w13_groups
                && w2_groups == o.w2_groups;
    }
};

/// One call of a layer.
struct moe_call {
    /// Token rows, `src_ld` elements apart: FP16 for an FP16 layer, else BF16.
    const uint16_t *src = nullptr;
    int src_ld = 0;
    int num_tokens = 0;
    int topk = 0;
    /// `[num_tokens * topk]`: the local expert of each (token, slot), below
    /// the layer's expert count, or negative when it is not resident.
    const int *local_experts = nullptr;
    /// `[num_experts]`: how many slots route to each expert.
    const int *counts = nullptr;
    /// `[num_tokens * topk]` router weights; unread when `skip_weighted`.
    const float *topk_weights = nullptr;
    bool skip_weighted = false;
    /// `[num_experts, 2 * I]` and `[num_experts, H]` biases in the layer
    /// key's bias types; read on every call, so they stay the caller's.
    const void *w13_bias = nullptr;
    const void *w2_bias = nullptr;
    /// `[num_tokens, H]` output in the type of `src`, `dst_ld` elements apart.
    void *dst = nullptr;
    int dst_ld = 0;
    /// The team size of the call's parallel region.
    int num_threads = 0;
};

/// Fills both projections for every expert of the layer; null on success,
/// else why it cannot, which the layer's calls then report.  Called only to
/// pack a layer, which reads the weights it names after it returns: a lock
/// that keeps them unchanged must be held until `execute` returns.
using describe_layer
        = std::function<const char *(projection &w13, projection &w2)>;

/// Runs `call` on the layer `key` names: routes its tokens to their experts'
/// rows and runs them on the engine `execute_grouped` uses.  The layer's
/// first call takes its experts' packs from the pack store, packing through
/// `describe` the weights no call has packed.  Returns `unimplemented`,
/// before writing anything, for a
/// non-gated activation, an empty call, a slot whose expert is not resident or
/// out of range, and a layer it cannot pack or has no kernels for on this CPU
/// (a layer that fails to pack fails its calls until `flush()`); and
/// `memory_bad_size` / `memory_bad_storage` when an allocation fails, after
/// which the next call packs the layer again.
status_t execute(const layer_key &key, const layer_scales &scales,
        const describe_layer &describe, const moe_call &call);

/// Runs a `group_matmul_direct` call of `num_ops` (two or more) operations,
/// its arguments as that function takes them, `num_threads` resolved: every
/// operation's rows against its expert's W13 weight with the gated
/// activation `act` fused, then `fused`'s down projection and `reduce`'s
/// weighted reduce, each when not null, in one parallel region as `execute`
/// runs a layer.  Operations with rows must share their shapes, types and
/// scale layouts.  Packs each expert weight the first time a call of either
/// API reads it, and the prepack extras' (`matmul_params::total_matmul`) with
/// it, keyed by the weight and its shape; the pack stands for the weight until
/// `flush()`.  DA8W8 and W4A8 extras may come without `params` and down
/// scales: their weights are packed alone, and their scales join the pack the
/// first time a call gives them.  A later call that names the same weights,
/// with the same activation and source scale form, finds their packs and its
/// tile plans through its call site.  With a gated activation, Op1's output is
/// the activation's `N / 2` columns of each row; the rest of a wider `dst`
/// row is not written.  Quantized experts take weight scales per tensor,
/// channel or group, and source scales per row or per (row, weight group):
/// bf16 rows it quantizes so, or the caller's s8 rows (DA8W8) with such
/// scales or one for all of them.  Returns `unimplemented`, with the reason
/// logged and before writing anything, for a call it cannot serve: other than
/// BF16, FP16, DA8W8 or W4A8 experts; other scale layouts; post-ops, zero
/// points, transposed or scaled sources, non-constant or pre-packed weights;
/// weight cache mode 0; and a weight it cannot pack or has no kernels for on
/// this CPU.  A malformed `reduce` returns its validation status.  `since`
/// is the flush count (`flush_count()`) as the call read it on entering the
/// API: after a flush that ran since, the call runs on packs of its own and
/// publishes nothing.
status_t execute_grouped(size_t num_ops, const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<bool> &transB,
        const std::vector<int> &M, const std::vector<int> &N,
        const std::vector<int> &K, const std::vector<float> &alpha,
        const std::vector<const void *> &src, const std::vector<int> &lda,
        const std::vector<const void *> &weight, const std::vector<int> &ldb,
        const std::vector<const void *> &bias, const std::vector<float> &beta,
        const std::vector<void *> &dst, const std::vector<int> &ldc,
        const std::vector<bool> &is_weights_const,
        const std::vector<matmul_params> &params, grp_matmul_gated_act_t act,
        const grp_matmul_fused_moe_params *fused,
        const group_matmul_moe_postop_params *reduce, int num_threads,
        uint64_t since);

/// The flush count, which a grouped call reads on entering the API: odd
/// while a flush runs, and greater after one.
uint64_t flush_count();

/// Drops every pack, every routed layer's state and every grouped call site.
/// A grouped call that read the flush count before the flush finished
/// dropping, and a routed call that looked its layer up before then, finish
/// on packs made for them alone, and keep no pack, layer or site after they
/// return.  A grouped call that reads it later packs the bytes it finds and
/// keeps those packs, though its caller may have made it before the flush,
/// so grouped calls on weights the host reloads or frees in place return
/// before the flush is called.  Called by both
/// `group_matmul_routed_moe_flush_weight_cache()` and
/// `clear_grp_matmul_weight_caches()`, with their contracts.
void flush();

/// Frees the calling thread's scratch.
void reset_thread_local_scratch();

} // namespace ntile_flat_parallel

namespace test_api {

/// Dormant test-only counters: while armed, the calls `execute` and
/// `execute_grouped` completed and the routed layer entries `execute` made.
inline std::atomic<bool> s_capture_ntile_flat_parallel {false};
inline std::atomic<uint64_t> s_ntile_flat_parallel_completed {0};
inline std::atomic<uint64_t> s_ntile_flat_parallel_grouped_completed {0};
inline std::atomic<uint64_t> s_ntile_flat_parallel_layer_entries {0};

/// Test-only pin of whether DA8W8 layers prepacked from now on run the
/// kernels that read source bytes XOR-ed with 0x80: -1 leaves it to the CPU,
/// 0 and 1 force it off and on.
inline std::atomic<int> s_ntile_flat_parallel_u8_src_override {-1};
/// Counts, while `s_capture_ntile_flat_parallel` is set, the calls of either
/// API that ran the kernels that read source bytes XOR-ed with 0x80.
inline std::atomic<uint64_t> s_ntile_flat_parallel_u8_src_calls {0};

/// Test-only pin of whether a decode call's threads run their own units
/// first: -1 leaves it to the team size, 0 forces queue order, 1 own units.
inline std::atomic<int> s_ntile_flat_parallel_own_units_override {-1};

/// Test-only: while `s_pause_ntile_flat_parallel` is set, a routed call waits
/// at the start of `execute`, after setting `s_ntile_flat_parallel_paused`.
inline std::atomic<bool> s_pause_ntile_flat_parallel {false};
inline std::atomic<bool> s_ntile_flat_parallel_paused {false};
/// Test-only: while set, the next pack lookup flushes ALGO 4's state between
/// finding its weights and publishing their packs, and clears the flag.
inline std::atomic<bool> s_ntile_flat_parallel_flush_before_publish {false};
/// Test-only: while set, the next `group_matmul_direct` call flushes ALGO
/// 4's state right after it reads the flush count, and clears the flag.
inline std::atomic<bool> s_ntile_flat_parallel_flush_after_entry {false};
/// Test-only: the next flush calls this after its first step, dropping the
/// store, and clears it.
inline std::atomic<void (*)()> s_ntile_flat_parallel_during_flush {nullptr};
/// The packs alive, whoever keeps them: the store, a call site, a routed
/// layer or a call.
inline std::atomic<int64_t> s_ntile_flat_parallel_live_packs {0};

/// How many layers have their state kept.
size_t ntile_flat_parallel_layers();

/// How many expert weights have packs.
size_t ntile_flat_parallel_packs();
/// The grouped call sites ALGO 4 keeps the packs of (see `execute_grouped`).
size_t ntile_flat_parallel_call_sites();
/// The projection pools whose prepack extras grouped calls have packed.
size_t ntile_flat_parallel_warmed_pools();

} // namespace test_api

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_HPP
