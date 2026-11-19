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

#include "lowoha_operators/matmul/group_matmul/executors/n_tile/src_group_sums.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>

#include "lowoha_operators/matmul/group_matmul/custom_kernel/ukernel/s8_group_sums.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/n_tile_knobs.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

bool hoist_src_group_sums(const std::vector<SrcGroupSumsJob> &jobs,
        int group_size, int num_threads) {
    size_t src_bytes = 0;
    for (const SrcGroupSumsJob &j : jobs)
        src_bytes += static_cast<size_t>(j.rows) * static_cast<size_t>(j.K);
    if (src_bytes < kSrcGroupSumsHoistMinBytes) return false;
    test_api::s_src_group_sum_hoists.fetch_add(1, std::memory_order_relaxed);

    struct SumTask {
        size_t job;
        int row0;
        int rows;
    };
    constexpr int kSumRowsPerTask = 16;
    std::vector<SumTask> tasks;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const SrcGroupSumsJob &j = jobs[i];
        j.sums->resize(static_cast<size_t>(j.rows)
                * static_cast<size_t>(j.K / group_size));
        for (int r0 = 0; r0 < j.rows; r0 += kSumRowsPerTask)
            tasks.push_back({i, r0, std::min(kSumRowsPerTask, j.rows - r0)});
    }
    const int n_tasks = static_cast<int>(tasks.size());
#pragma omp parallel for num_threads(num_threads)
    for (int t = 0; t < n_tasks; ++t) {
        const SumTask &task = tasks[static_cast<size_t>(t)];
        const SrcGroupSumsJob &j = jobs[task.job];
        const int G = j.K / group_size;
        custom_kernel::s8_group_sums(
                j.src + static_cast<size_t>(task.row0) * j.lda, j.lda,
                task.rows, j.K, group_size,
                j.sums->data() + static_cast<size_t>(task.row0) * G);
    }
    return true;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
