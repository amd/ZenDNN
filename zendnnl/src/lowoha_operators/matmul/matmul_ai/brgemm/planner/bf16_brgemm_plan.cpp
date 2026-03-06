/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "lowoha_operators/matmul/matmul_ai/brgemm/planner/bf16_brgemm_plan.hpp"
#include "operators/matmul/matmul_config.hpp"
#include <algorithm>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ai {

using zendnnl::ops::post_op_type_t;

BF16BrgemmFullPlan plan_bf16_brgemm(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;
    const int K_padded = (K + 1) & ~1;

    BrgemmPlan bplan = plan_brgemm(desc, uarch);

    const bool is_decode = (M <= 4);

    if (is_decode) {
        bplan.MR = M;
        bplan.NR = 64;
        if (N < 64) bplan.NR = (N >= 32) ? 32 : 16;
        bplan.MB = M;
        bplan.NB = std::max(bplan.NB / bplan.NR * bplan.NR, bplan.NR);
        bplan.NB = std::min(bplan.NB, N);
    } else {
        bplan.MR = 6;
        bplan.NR = 64;
        if (N < 64) bplan.NR = (N >= 32) ? 32 : 16;
        bplan.NB = std::max(bplan.NB / bplan.NR * bplan.NR, bplan.NR);
        bplan.NB = std::min(bplan.NB, N);
    }

    bool decode_wide_n = (is_decode && K < N);
    int b_panel_bytes = K_padded * NR_PACK * static_cast<int>(sizeof(uint16_t));
    bool b_exceeds_l2 = (b_panel_bytes > uarch.l2_bytes);

    return BF16BrgemmFullPlan{bplan, is_decode, decode_wide_n || b_exceeds_l2};
}

} // namespace ai
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
