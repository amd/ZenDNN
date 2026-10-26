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

#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_partition.hpp"

#include <memory>

#include "common/zendnnl_compat.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

struct aligned_free_t {
    void operator()(void *p) const { zendnnl_aligned_free(p); }
};

} // namespace

native_partition_t::native_partition_t(
        const native_plan_t &plan, int M, int c0, int c1)
    : M_(M)
    , mb_(plan.mb)
    , nr_(plan.nr)
    , m_blocks_(plan.m_blocks)
    , c0_(c0)
    , c1_(c1)
    , col_groups_(plan.col_groups)
    , threads_(plan.threads)
    , row_unit_(plan.row_unit)
    , traversal_(plan.traversal) {}

native_partition_kind_t native_partition_t::kind(int workers) const noexcept {
    return col_groups_ > 0 && workers == threads_ && workers % col_groups_ == 0
                    && (c1_ - c0_) % col_groups_ == 0
            ? native_partition_kind_t::row_groups
            : native_partition_kind_t::slice;
}

int64_t native_partition_t::units() const noexcept {
    return static_cast<int64_t>(m_blocks_) * (c1_ - c0_);
}

void native_partition_t::row_range(
        int rg, int row_groups, int &r0, int &r1) const noexcept {
    const int64_t row_units
            = (static_cast<int64_t>(M_) + row_unit_ - 1) / row_unit_;
    r0 = static_cast<int>(
            std::min<int64_t>(M_, row_units * rg / row_groups * row_unit_));
    r1 = static_cast<int>(std::min<int64_t>(
            M_, row_units * (rg + 1) / row_groups * row_unit_));
}

float *native_thread_scratch(size_t floats) {
    static thread_local std::unique_ptr<float, aligned_free_t> buf;
    static thread_local size_t cap = 0;
    if (cap < floats) {
        buf.reset(static_cast<float *>(zendnnl_aligned_alloc(
                64, (floats * sizeof(float) + 63) & ~size_t {63})));
        cap = buf ? floats : 0;
    }
    return buf.get();
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
