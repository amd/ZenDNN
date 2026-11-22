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

#ifndef MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_BRGEMM_LOOPER_HPP
#define MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_BRGEMM_LOOPER_HPP

#include <cstddef>
#include <cstdint>

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_view.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Native BRGEMM over a native-layout weight: bf16 x bf16, u8 / s8 (or a
/// bf16 / f32 source quantized dynamically) x s8, s8 (or a bf16 / f32
/// source quantized dynamically) x s4, f32 x f32 and f16 x f16, into the
/// dst types its checks take.  A call it turns away before any work returns
/// unimplemented or isa_unsupported (failure for bad arguments), with *why
/// (when why is not null) naming the reason, and leaves dst and the weight
/// as they were; once work started, every error is failure.  desc.K,
/// desc.N and weight.pack_nr must satisfy native_layout_fits (the layout
/// rule's shape); a release build does not check them.  Every call reads the
/// weight's one layout: M = 1 on the GEMV kernels (native_gemv_ukernel.hpp),
/// whose results have the tiles' bits, every other M on the native tiles.
status_t native_brgemm_execute(const GemmDescriptor &desc,
        const native_weight_ref_t &weight, const void *src, void *dst,
        const void *bias, const matmul_params &params,
        const char **why = nullptr);

/// native_brgemm_execute with an M = 1 call on the native tiles instead of
/// the GEMV kernels: the reference the tests check the GEMV kernels' bits
/// against.
status_t native_brgemm_execute_on_tiles(const GemmDescriptor &desc,
        const native_weight_ref_t &weight, const void *src, void *dst,
        const void *bias, const matmul_params &params,
        const char **why = nullptr);

/// The status native_brgemm_execute would turn the call away with (success
/// when it would run it), with *why naming the reason, without running it:
/// every check up to the acquisition of the weight's layout.  dst and the
/// weight are not touched; a dynamic quantization of a bf16 / f32 source
/// goes into the call's own buffers, and the scales and zero points it
/// makes into the buffers of params.
status_t native_brgemm_check(const GemmDescriptor &desc,
        const native_weight_ref_t &weight, const void *src, void *dst,
        const void *bias, const matmul_params &params, const char **why);

/// The plan the paired tiles of a gated call over [gate | up] halves run
/// (native_weight_view_t) at width (native_paired_width): planned's, with
/// tiles width wide when that is wider (paired_wider), and for BF16 the
/// accumulators of a tile's rows.  Where the wider tiles leave the static
/// deal fewer units than threads, row blocks of fewer rows give back the
/// units the plan was balanced for.
native_plan_t native_paired_plan(const native_plan_t &planned,
        const native_problem_t &problem, int width);

/// Block b of the widening of an f16 source of M rows into f32 rows ld
/// apart in dst: element (m, k), src[m * lda + k] or with trans
/// src[k * lda + m], for every k below K (side at a time with trans) and
/// the block's rows m, from b * rows to that plus what is left, at most
/// rows (b below ceil_div(M, rows)).  No row index passes M, at any M an
/// int holds.  Needs AVX-512 F, BW and VL.
void native_widen_f16_block(const uint16_t *src, int lda, bool trans, int M,
        int K, float *dst, int ld, int rows, int side, int b);

/// Buffers of widened F16 sources the process keeps for later calls at
/// most.
constexpr int kNativeF16KeptSourceSlots = 32;

/// The buffer of a call's widened F16 source, 64-byte aligned, of at least
/// the bytes take asks for: the smallest kept buffer large enough, else a
/// new one (nullptr when none can be allocated).  At scope exit it goes
/// back to the kept buffers, the smallest dropped first beyond 512 MiB, or
/// is freed: when it is larger than that, after a release, or when every
/// slot holds a buffer at least as large.  Giving it back allocates
/// nothing.
class native_f16_source_lease_t {
public:
    native_f16_source_lease_t() = default;
    native_f16_source_lease_t(const native_f16_source_lease_t &) = delete;
    native_f16_source_lease_t &operator=(const native_f16_source_lease_t &)
            = delete;
    ~native_f16_source_lease_t();

    /// The buffer, taken once per lease.
    float *take(size_t bytes);

private:
    float *buf_ = nullptr;
    size_t bytes_ = 0;
    uint64_t generation_ = 0;
};

/// Frees the buffers of widened F16 sources kept for later calls, and those
/// that calls running now hold once they return.
void release_native_f16_sources();

/// Bytes of the buffers of widened F16 sources kept for later calls.
size_t native_f16_kept_source_bytes();

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_LOOPER_NATIVE_BRGEMM_LOOPER_HPP
