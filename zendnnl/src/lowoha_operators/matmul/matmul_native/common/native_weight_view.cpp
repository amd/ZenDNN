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

#include "lowoha_operators/matmul/matmul_native/common/native_weight_view.hpp"

#include <cassert>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::log_error;

native_weight_lease_t::~native_weight_lease_t() {
    native_free_layout(packed_);
}

void native_weight_lease_t::own(void *packed) {
    native_free_layout(packed_);
    packed_ = packed;
}

status_t acquire_native_weight(const native_weight_ref_t &ref,
        data_type_t wei_dt, int K, int N, int ldb, bool transB,
        native_weight_view_t &view, native_weight_lease_t &lease,
        int num_threads) {
    const int pack_nr = ref.pack_nr;
    assert(native_layout_fits(wei_dt, K, N, pack_nr));
    if (ref.weight == nullptr) {
        log_error("native weight: no weight buffer for wei=",
                zendnnl::common::dtype_info(wei_dt), " K=", K, " N=", N);
        return status_t::failure;
    }

    view = native_weight_view_t {};
    view.wei_dt = wei_dt;
    view.K = K;
    view.N = N;
    view.pack_nr = pack_nr;
    view.interleaved = ref.interleave_split_halves;
    view.k_groups = native_k_groups(wei_dt, K);
    view.block_bytes = native_block_bytes(wei_dt, K, pack_nr);

    if (ref.weight_cache_mode < 0 || ref.weight_cache_mode > 2) {
        log_error("native weight: weight cache mode ", ref.weight_cache_mode,
                " is not 0, 1 or 2");
        return status_t::failure;
    }
    native_raw_weight_t raw;
    raw.wei_dt = wei_dt;
    raw.weight = ref.weight;
    raw.K = K;
    raw.N = N;
    raw.ldb = ref.aocl_blocked ? (transB ? K : N) : ldb;
    raw.transB = transB;
    raw.pack_nr = pack_nr;
    raw.interleave_split_halves = ref.interleave_split_halves;
    raw.aocl_blocked = ref.aocl_blocked;
    native_layout_t layout;
    const status_t st = get_native_layout(raw, ref.weight_cache_mode,
            ref.capacity_bytes, num_threads, layout, &ref.image);
    if (st != status_t::success) return st;
    lease.own(layout.owned);
    view.base = static_cast<const uint8_t *>(layout.data);
    return status_t::success;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
