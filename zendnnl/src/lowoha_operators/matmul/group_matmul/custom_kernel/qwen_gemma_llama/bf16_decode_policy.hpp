/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *******************************************************************************/

#ifndef ZENDNNL_GROUP_MATMUL_QWEN_BF16_DECODE_POLICY_HPP
#define ZENDNNL_GROUP_MATMUL_QWEN_BF16_DECODE_POLICY_HPP

#include <cstdint>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

enum class QwenBf16DecodeRoute : uint8_t {
    none = 0,
    qwen35_w2_resume,
};

/// Apply Qwen BF16 decode microkernels after the normal ALGO 3 context is
/// fully prepared. Unknown shapes return without changing one byte of context.
QwenBf16DecodeRoute apply_qwen_bf16_decode_policy(CallContext &ctx,
        grp_matmul_gated_act_t act, const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K);

/// Execute the isolated tile scheduler selected above. Returns false without
/// touching dst when route is none.
bool dispatch_qwen_bf16_decode_tile(QwenBf16DecodeRoute route,
        const CallContext &ctx, int expert_idx, int M, int K, int n_tile,
        int col_start, const void *src, int lda, void *dst, int ldc);

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif
