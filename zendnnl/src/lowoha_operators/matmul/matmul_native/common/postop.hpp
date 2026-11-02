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

#ifndef MATMUL_NATIVE_POSTOP_HPP
#define MATMUL_NATIVE_POSTOP_HPP

#include <vector>
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

enum class fused_postop_t : int;

/// Bias, then the post-ops in chain order, over an m_count x n_count tile
/// at row m_offset and column n_offset of the output.  AVX-512 F / BW and
/// FMA.
void apply_postops_tile(float *C, int ldc, int m_count, int n_count,
        int n_offset, int m_offset, const float *bias,
        const std::vector<matmul_post_op> &postops);

/// apply_postops_tile without AVX-512: each element by the formulas of the
/// remainder loops of apply_postops_tile, not bit for bit, since the
/// compiler may fuse a multiply and an add of those loops.
void apply_postops_tile_scalar(float *C, int ldc, int m_count, int n_count,
        int n_offset, int m_offset, const float *bias,
        const std::vector<matmul_post_op> &postops);

/// The activation a kernel epilogue fuses from the post-op chain of params,
/// and its index in *index: the first op that is not none, when it is an
/// activation the epilogues implement; else fused_postop_t::none and -1.
/// An epilogue runs before every other op of the chain, so only its head
/// may fuse.
fused_postop_t head_fused_postop(const matmul_params &params, int *index);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_POSTOP_HPP
