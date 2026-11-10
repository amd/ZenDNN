/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_gemv.hpp"

#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/native_int8_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/bf16_gemv_direct.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/int8_gemv_direct.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

/// The largest K of bf16_bkc's calls of more than one thread.
constexpr int kBf16GemvTeamMaxK = 1024;

/// The largest weight (N * K elements) of int8_bkc's calls of more than one
/// thread, and the weight above which its calls of one thread take K <=
/// kInt8GemvMaxK.
constexpr int64_t kInt8GemvTeamMaxWeight = 64 * 1024;
constexpr int kInt8GemvMaxK = 1024;

bool bf16_call(const GemmDescriptor &desc) {
    return desc.src_dt == data_type_t::bf16 && desc.wei_dt == data_type_t::bf16;
}

/// What the GEMV kernels do not apply: a gated activation, alpha 0, or a
/// Swish of alpha other than 1 (they fuse SiLU).
bool beyond_the_gemv_kernels(
        const GemmDescriptor &desc, const matmul_params &params) {
    if (desc.gated_act != matmul_gated_act_t::none || desc.alpha == 0.0f)
        return true;
    for (const auto &po : params.postop_)
        if (po.po_type == post_op_type_t::swish && po.alpha != 1.0f)
            return true;
    return false;
}

/// Whether plan, a BKC kernel whose gates take a call, takes it from the
/// native tiles: N a multiple of the layout's blocks (a weight of other N
/// stays on the tiles of its padded layout), a constant weight, and the
/// bounds above.  At weight cache mode 1 or 2 a BKC kernel caches its own
/// pack of the weight (bf16_bkc's of any size on one thread), next to the
/// native layout that the weight's calls of M > 1 cache when it is out of
/// place.
bool bkc_gemv_takes_the_call(
        const native_gemv_plan_t &plan, const GemmDescriptor &desc) {
    if (desc.N % kNativeLayoutPackNr != 0 || !desc.is_weights_const)
        return false;
    const bool one_thread = desc.num_threads <= 1;
    if (plan.kernel == native_gemv_t::bf16_bkc)
        return one_thread || desc.K <= kBf16GemvTeamMaxK;
    if (plan.kernel == native_gemv_t::int8_bkc)
        return static_cast<int64_t>(desc.N) * desc.K <= kInt8GemvTeamMaxWeight
                || (one_thread && desc.K <= kInt8GemvMaxK);
    return false;
}

} // namespace

const char *native_gemv_name(native_gemv_t kernel) {
    switch (kernel) {
        case native_gemv_t::bf16_narrow: return "bf16_narrow";
        case native_gemv_t::bf16_bkc: return "bf16_bkc";
        case native_gemv_t::int8_bkc: return "int8_bkc";
        default: return "none";
    }
}

native_gemv_plan_t decide_native_gemv(
        const GemmDescriptor &desc, const matmul_params &params) {
    if (desc.M != 1) return native_gemv_plan_t {};
    // The GEMV kernels run on one thread: a bf16 call on the team of
    // m1_gemv_cap_threads (at least 64 columns per thread) when that is one,
    // an INT8 call at any team.
    const bool bf16 = bf16_call(desc);
    // A K the layout rule turns away goes to the tiles, which turn it away
    // with the rule's reason: the GEMV choices count K padded to whole pairs
    // or quads in an int.
    if (desc.K > (bf16 ? kMaxNativeBf16K : kMaxSafeInt8AccumK))
        return native_gemv_plan_t {};
    const int threads
            = bf16 ? m1_gemv_cap_threads(desc.N, desc.num_threads) : 1;
    if (threads > 1 || beyond_the_gemv_kernels(desc, params))
        return native_gemv_plan_t {};
    const UarchParams &uarch = detect_uarch();
    GemmDescriptor d = desc;
    d.num_threads = threads;
    const native_gemv_plan_t plan = bf16 ? bf16_gemv_choice(d, uarch, params)
                                         : int8_gemv_choice(d, uarch, params);
    // The narrow kernel is the only native kernel for its widths.
    if (plan.kernel == native_gemv_t::none
            || plan.kernel == native_gemv_t::bf16_narrow
            || bkc_gemv_takes_the_call(plan, desc))
        return plan;
    return native_gemv_plan_t {};
}

status_t run_native_gemv(const native_gemv_plan_t &plan,
        const GemmDescriptor &desc, const void *src, const void *weight,
        void *dst, const void *bias, const matmul_params &params,
        int32_t weight_cache_mode) {
    if (plan.kernel == native_gemv_t::int8_bkc)
        return int8_gemv_run(
                plan, desc, src, weight, dst, bias, params, weight_cache_mode);
    return bf16_gemv_run(plan, desc, src, weight, dst, bias, weight_cache_mode);
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
