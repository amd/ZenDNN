/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *******************************************************************************/

#include "bf16_decode_policy.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include "bf16_decode_microkernel.hpp"
#include "common/zendnnl_compat.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

QwenBf16DecodeRoute apply_qwen_bf16_decode_policy(CallContext &ctx,
        grp_matmul_gated_act_t act, const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K) {
    if (!ctx.enabled || ctx.variant != KernelVariant::kBF16_BF16_BF16
            || ctx.NV != 2 || ctx.bias_kind != BiasKind::none || M.size() != 1
            || N.size() != 1 || K.size() != 1 || M[0] <= 0
            || M[0] > kDecodeMaxM) {
        return QwenBf16DecodeRoute::none;
    }

    const int n = N[0];
    const int k = K[0];
    const bool qwen35_gated = M[0] > 8
            && act == grp_matmul_gated_act_t::silu_and_mul && k == 4096
            && n == 24576;
    const bool qwen35_w2 = M[0] > 8 && act == grp_matmul_gated_act_t::none
            && k == 12288 && n == 4096;
    const bool qwen35_plain = M[0] > 8 && act == grp_matmul_gated_act_t::none
            && k == 4096
            && (n == 4096 || n == 10240 || n == 12288 || n == 248320);
    const bool qwen36_plain = act == grp_matmul_gated_act_t::none
            && ((k == 2048 && (n == 256 || n == 12288 || n == 248320))
                    || (k == 4096 && n == 2048));

    if (qwen35_gated) {
        bool complete = true;
        for (int mr = 9; mr <= 12; ++mr) {
            ctx.kfn_table[mr] = decode_specialized::select_gated_bf16_one_panel(
                    mr, ctx.act_kind);
            complete = complete && ctx.kfn_table[mr] != nullptr;
        }
        if (complete) { ctx.max_mr = 12; }
        return QwenBf16DecodeRoute::none;
    }
    if (qwen35_w2) {
        const int max_mr = ctx.max_mr;
        const int n_calls = (M[0] + max_mr - 1) / max_mr;
        const int mr_small = M[0] / n_calls;
        const int mr_big = mr_small + ((M[0] % n_calls) != 0 ? 1 : 0);
        const auto initial_small
                = select_ukernel(mr_small, ctx.NV, ActKind::none, DstDt::kF32);
        const auto initial_big
                = select_ukernel(mr_big, ctx.NV, ActKind::none, DstDt::kF32);
        const auto continue_small
                = decode_specialized::select_plain_bf16_f32_continue(
                        mr_small, ctx.NV);
        const auto continue_big
                = decode_specialized::select_plain_bf16_f32_continue(
                        mr_big, ctx.NV);
        if (initial_small != nullptr && initial_big != nullptr
                && continue_small != nullptr && continue_big != nullptr) {
            return QwenBf16DecodeRoute::qwen35_w2_resume;
        }
        return QwenBf16DecodeRoute::none;
    }
    if (qwen35_plain) {
        bool complete = true;
        for (int mr = 9; mr <= 12; ++mr) {
            ctx.kfn_table[mr]
                    = decode_specialized::select_plain_bf16_one_panel(mr);
            complete = complete && ctx.kfn_table[mr] != nullptr;
        }
        if (complete) ctx.max_mr = 12;
        return QwenBf16DecodeRoute::none;
    }
    if (qwen36_plain) {
        bool complete = true;
        for (int mr = 8; mr <= 12; ++mr) {
            ctx.kfn_table[mr]
                    = decode_specialized::select_plain_bf16_fixed_k(mr, k);
            complete = complete && ctx.kfn_table[mr] != nullptr;
        }
        if (complete) ctx.max_mr = 12;
    }
    return QwenBf16DecodeRoute::none;
}

bool dispatch_qwen_bf16_decode_tile(QwenBf16DecodeRoute route,
        const CallContext &ctx, int expert_idx, int M, int K, int n_tile,
        int col_start, const void *src, int lda, void *dst, int ldc) {
    if (route != QwenBf16DecodeRoute::qwen35_w2_resume) return false;

    assert(ctx.enabled && ctx.variant == KernelVariant::kBF16_BF16_BF16);
    assert(ctx.pack_nr == 32 && ctx.NV == 2 && ctx.bias_kind == BiasKind::none);
    assert(expert_idx >= 0 && expert_idx < CallContext::kMaxExperts);
    assert(ctx.packed_ptrs[expert_idx] != nullptr);
    assert(M > 8 && M <= kDecodeMaxM && K == 12288);
    assert(n_tile > 0 && n_tile % ctx.pack_nr == 0);
    assert(col_start >= 0 && col_start % ctx.pack_nr == 0);

    constexpr int resume_kc = 320;
    const int pack_nr = ctx.pack_nr;
    const int max_mr = ctx.max_mr;
    const int n_blocks = n_tile / pack_nr;
    const int n_calls = (M + max_mr - 1) / max_mr;
    const int mr_small = M / n_calls;
    const int n_big = M - mr_small * n_calls;
    const int mr_big = mr_small + (n_big > 0 ? 1 : 0);

    const ukernel_fn_t initial_small
            = select_ukernel(mr_small, ctx.NV, ActKind::none, DstDt::kF32);
    const ukernel_fn_t initial_big = (n_big > 0)
            ? select_ukernel(mr_big, ctx.NV, ActKind::none, DstDt::kF32)
            : initial_small;
    const ukernel_fn_t continue_small
            = decode_specialized::select_plain_bf16_f32_continue(
                    mr_small, ctx.NV);
    const ukernel_fn_t continue_big = (n_big > 0)
            ? decode_specialized::select_plain_bf16_f32_continue(mr_big, ctx.NV)
            : continue_small;
    assert(initial_small != nullptr && initial_big != nullptr);
    assert(continue_small != nullptr && continue_big != nullptr);

    const auto *A = static_cast<const bfloat16_t *>(src);
    const auto *Bpacked_full = ctx.packed_ptrs[expert_idx];
    const int k_pairs = (K + 1) / 2;
    const size_t o_blk_stride
            = static_cast<size_t>(k_pairs) * pack_nr * kVNNIPair;
    const int kp_stride_bf16 = pack_nr * kVNNIPair;

    thread_local std::vector<float> accum;
    accum.resize(static_cast<size_t>(M) * n_tile);

    for (int sub = 0; sub < K; sub += resume_kc) {
        const int sub_now = std::min(resume_kc, K - sub);
        const int kp0 = sub / 2;
        const bool resume = sub != 0;
        for (int b = 0; b < n_blocks; ++b) {
            const bfloat16_t *Bpacked = Bpacked_full
                    + static_cast<size_t>(col_start / pack_nr + b)
                            * o_blk_stride
                    + static_cast<size_t>(kp0) * kp_stride_bf16;
            int m_offset = 0;
            for (int call = 0; call < n_calls; ++call) {
                const bool big = call < n_big;
                const int mr = big ? mr_big : mr_small;
                const ukernel_fn_t kernel = resume
                        ? (big ? continue_big : continue_small)
                        : (big ? initial_big : initial_small);
                const bfloat16_t *A_chunk
                        = A + static_cast<size_t>(m_offset) * lda + sub;
                float *C_chunk = accum.data()
                        + static_cast<size_t>(m_offset) * n_tile
                        + static_cast<size_t>(b) * pack_nr;
                kernel(A_chunk, lda, Bpacked, /*bias=*/nullptr, BiasKind::none,
                        C_chunk, n_tile, /*Cout_tight=*/nullptr,
                        /*ldc_tight=*/0, sub_now);
                m_offset += mr;
            }
        }
    }

    auto *output = static_cast<bfloat16_t *>(dst);
    for (int m = 0; m < M; ++m) {
        const float *accum_row = accum.data() + static_cast<size_t>(m) * n_tile;
        bfloat16_t *output_row
                = output + static_cast<size_t>(m) * ldc + col_start;
        for (int n = 0; n < n_tile; ++n)
            output_row[n] = bfloat16_t(accum_row[n]);
    }
    return true;
}

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
