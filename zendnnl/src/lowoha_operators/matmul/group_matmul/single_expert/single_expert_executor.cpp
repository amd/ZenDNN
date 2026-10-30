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

#include "single_expert_executor.hpp"

#include <algorithm>
#include <atomic>
#include <vector>

#include <omp.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace {

namespace ck = custom_kernel;

struct Range {
    int begin;
    int count;
};

// Part `i` of `parts` near-equal contiguous parts of [0, total).
inline Range part_range(int total, int parts, int i) {
    const int base = total / parts;
    const int extra = total % parts;
    return {i * base + std::min(i, extra), base + (i < extra ? 1 : 0)};
}

// Row-offset views of one call: `m_begin` rows into src, src scale and dst.
struct RowView {
    const void *src;
    const void *src_scale;
    void *dst;
};

inline RowView rows_at(const SingleExpertCallView &c, int m_begin) {
    const ck::CallContext &kc = *c.kctx;
    const size_t src_elem = ck::is_int8_variant(kc.variant) ? 1 : 2;
    RowView r;
    r.src = static_cast<const char *>(c.src)
            + static_cast<size_t>(m_begin) * c.lda * src_elem;
    r.src_scale = c.src_scale == nullptr
            ? nullptr
            : static_cast<const char *>(c.src_scale)
                    + static_cast<size_t>(m_begin)
                            * ck::block_scale_elem_bytes(kc);
    r.dst = static_cast<char *>(c.dst)
            + static_cast<size_t>(m_begin) * c.ldc
                    * ck::block_dst_elem_bytes(kc);
    return r;
}

// Accumulate K range [k0, k1) of blocks [b0, b1) for `m` rows into
// `acc_of(b)`, in `plan.kc` chunks and `plan.order`.
template <typename AccOf>
void accum_blocks(const SingleExpertCallView &c,
        const SingleExpertBlockPlan &plan, int m, const void *src, int b0,
        int b1, int k0, int k1, AccOf acc_of) {
    const ck::CallContext &kc = *c.kctx;
    const int chunk = plan.kc > 0 ? plan.kc : k1 - k0;
    if (plan.order == SingleExpertLoopOrder::n_outer) {
        for (int b = b0; b < b1; ++b)
            for (int k = k0; k < k1; k += chunk)
                ck::accumulate_block(kc, 0, m, c.K, b, k,
                        std::min(k + chunk, k1), src, c.lda, acc_of(b), k != k0,
                        plan.mr);
        return;
    }
    for (int k = k0; k < k1; k += chunk)
        for (int b = b0; b < b1; ++b)
            ck::accumulate_block(kc, 0, m, c.K, b, k, std::min(k + chunk, k1),
                    src, c.lda, acc_of(b), k != k0, plan.mr);
}

// Full-K blocks [b0, b1) for rows [m_begin, m_begin + m), straight to dst.
// Without a K chunk or MR cap this is the regular tile kernel.
void run_whole_blocks(const SingleExpertCallView &c,
        const SingleExpertBlockPlan &plan, int m_begin, int m, int b0, int b1) {
    if (b1 <= b0 || m <= 0) return;
    const ck::CallContext &kc = *c.kctx;
    const int nr = kc.pack_nr;
    const RowView r = rows_at(c, m_begin);
    if (plan.kc == 0 && plan.mr == 0) {
        static const ck::Int8EpilogueConfig kNoEpilogue {};
        ck::dispatch_tile(kc, 0, m, c.K, (b1 - b0) * nr, b0 * nr, r.src, c.lda,
                /*bias=*/nullptr, r.dst, c.ldc, r.src_scale, c.src_zp,
                c.wei_scale,
                c.int8_epilogue != nullptr ? *c.int8_epilogue : kNoEpilogue);
        return;
    }
    thread_local std::vector<int32_t> local;
    const size_t per_block = static_cast<size_t>(m) * nr;
    local.resize(static_cast<size_t>(b1 - b0) * per_block);
    accum_blocks(c, plan, m, r.src, b0, b1, 0, c.K,
            [&](int b) { return local.data() + (b - b0) * per_block; });
    for (int b = b0; b < b1; ++b)
        ck::finish_block(kc, 0, m, c.K, b, local.data() + (b - b0) * per_block,
                r.src_scale, c.wei_scale, r.dst, c.ldc);
}

// Add `parts` partial sums of `elems` elements, stored `stride` elements
// apart, into the first.
inline void sum_parts(const ck::CallContext &kc, void *acc, int parts,
        size_t elems, size_t stride) {
    if (ck::is_int8_variant(kc.variant)) {
        auto *a = static_cast<int32_t *>(acc);
        for (int p = 1; p < parts; ++p)
            for (size_t i = 0; i < elems; ++i)
                a[i] += a[p * stride + i];
        return;
    }
    auto *a = static_cast<float *>(acc);
    for (int p = 1; p < parts; ++p)
        for (size_t i = 0; i < elems; ++i)
            a[i] += a[p * stride + i];
}

// K range `part` of `parts`, in multiples of the kernel's K step.
inline int k_at(const ck::CallContext &kc, int K, int parts, int part) {
    const int step = ck::block_k_step(kc);
    return static_cast<int>(static_cast<int64_t>(K / step) * part / parts)
            * step;
}

} // namespace

bool single_expert_call_supported(const SingleExpertCallView &c) {
    if (c.kctx == nullptr || c.M <= 0 || c.N <= 0 || c.K <= 0
            || c.num_threads < 1 || c.src == nullptr || c.dst == nullptr)
        return false;
    const ck::CallContext &kc = *c.kctx;
    if (kc.pack_nr <= 0 || c.N % kc.pack_nr != 0) return false;
    static const ck::Int8EpilogueConfig kNoEpilogue {};
    if (ck::is_int8_variant(kc.variant)
            && (c.src_scale == nullptr || c.wei_scale == nullptr
                    || c.int8_epilogue == nullptr))
        return false;
    return ck::block_api_supported(kc, c.bias, c.src_zp,
            c.int8_epilogue != nullptr ? *c.int8_epilogue : kNoEpilogue);
}

bool single_expert_plan_execute(
        const SingleExpertCallView &c, const SingleExpertBlockPlan &plan) {
    const ck::CallContext &kc = *c.kctx;
    const int nr = kc.pack_nr;
    const int blocks = c.N / nr;
    const int T = c.num_threads;
    std::atomic<bool> short_team {false};
    test_api::s_single_expert_executor_runs.fetch_add(
            1, std::memory_order_relaxed);

    if (plan.split == SingleExpertSplit::n_remainder_k_split) {
        const int first_split = plan.whole_blocks * T;
        const size_t acc_elems = static_cast<size_t>(c.M) * nr;
        thread_local std::vector<int32_t> scratch;
        scratch.resize(static_cast<size_t>(plan.split_blocks) * plan.k_parts
                * acc_elems);
        int32_t *acc = scratch.data();
#pragma omp parallel num_threads(T)
        {
            const bool short_now = omp_get_num_threads() < T;
            if (short_now) short_team.store(true, std::memory_order_relaxed);
            const int tid = omp_get_thread_num();
            if (!short_now) {
                run_whole_blocks(c, plan, 0, c.M, tid * plan.whole_blocks,
                        (tid + 1) * plan.whole_blocks);
                if (tid < plan.split_blocks * plan.k_parts) {
                    const int part = tid % plan.k_parts;
                    const int b = first_split + tid / plan.k_parts;
                    int32_t *a = acc + tid * acc_elems;
                    accum_blocks(c, plan, c.M, c.src, b, b + 1,
                            k_at(kc, c.K, plan.k_parts, part),
                            k_at(kc, c.K, plan.k_parts, part + 1),
                            [&](int) { return a; });
                }
            }
#pragma omp barrier
            if (!short_now && tid < plan.split_blocks) {
                int32_t *a = acc
                        + static_cast<size_t>(tid) * plan.k_parts * acc_elems;
                sum_parts(kc, a, plan.k_parts, acc_elems, acc_elems);
                ck::finish_block(kc, 0, c.M, c.K, first_split + tid, a,
                        c.src_scale, c.wei_scale, c.dst, c.ldc);
            }
        }
        return !short_team.load();
    }

    // Grid: thread -> (m part, n part, k part), K parts innermost.
    const int mp_n = plan.m_parts;
    const int np_n = plan.n_parts;
    const int kp_n = plan.k_parts;
    const int rows_max = (c.M + mp_n - 1) / mp_n;
    const size_t acc_elems = static_cast<size_t>(rows_max) * nr;
    thread_local std::vector<int32_t> scratch;
    if (kp_n > 1)
        scratch.resize(static_cast<size_t>(mp_n) * blocks * kp_n * acc_elems);
    int32_t *acc = scratch.data();
    const auto acc_at = [&](int mp, int b, int kp) {
        return acc
                + ((static_cast<size_t>(mp) * blocks + b) * kp_n + kp)
                * acc_elems;
    };
#pragma omp parallel num_threads(T)
    {
        const bool short_now = omp_get_num_threads() < T;
        if (short_now) short_team.store(true, std::memory_order_relaxed);
        const int tid = omp_get_thread_num();
        if (!short_now && tid < mp_n * np_n * kp_n) {
            const int kp = tid % kp_n;
            const int np = (tid / kp_n) % np_n;
            const int mp = tid / (kp_n * np_n);
            const Range rows = part_range(c.M, mp_n, mp);
            const Range blk = part_range(blocks, np_n, np);
            if (kp_n == 1) {
                run_whole_blocks(c, plan, rows.begin, rows.count, blk.begin,
                        blk.begin + blk.count);
            } else if (rows.count > 0) {
                accum_blocks(c, plan, rows.count, rows_at(c, rows.begin).src,
                        blk.begin, blk.begin + blk.count,
                        k_at(kc, c.K, kp_n, kp), k_at(kc, c.K, kp_n, kp + 1),
                        [&](int b) { return acc_at(mp, b, kp); });
            }
        }
        if (kp_n > 1) {
#pragma omp barrier
            if (!short_now) {
                for (int idx = tid; idx < mp_n * blocks; idx += T) {
                    const int mp = idx / blocks;
                    const int b = idx % blocks;
                    const Range rows = part_range(c.M, mp_n, mp);
                    if (rows.count <= 0) continue;
                    int32_t *a = acc_at(mp, b, 0);
                    sum_parts(kc, a, kp_n, static_cast<size_t>(rows.count) * nr,
                            acc_elems);
                    const RowView r = rows_at(c, rows.begin);
                    ck::finish_block(kc, 0, rows.count, c.K, b, a, r.src_scale,
                            c.wei_scale, r.dst, c.ldc);
                }
            }
        }
    }
    return !short_team.load();
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
