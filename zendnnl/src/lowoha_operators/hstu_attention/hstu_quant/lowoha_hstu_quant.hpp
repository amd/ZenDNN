/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

/// HSTU attention's dynamic quantizers, separate from the attention kernels
/// that call them (`tiled/int8/`) and from the GEMMs (`hstu_gemm_ck/`).
///
/// Every group is symmetric S8: with m = max(max |x|, 1e-10), the codes are
/// RNE(x * 127 / m) in [-127, 127] plus `offset` (0 for S8; 128 for the U8
/// that `vpdpbusd` reads) and the scale is m / 127, so an all-zero group
/// quantizes to `offset`. Each code is RNE of the exact product x * (127 / m)
/// (rounded once, in an FMA, independent of the MXCSR rounding mode). A group
/// holding a NaN gets a NaN scale and one holding an infinity an infinite
/// scale, so every INT8 GEMM that reads it gives non-finite results, as on
/// the BF16 path.

#ifndef LOWOHA_HSTU_QUANT_HPP
#define LOWOHA_HSTU_QUANT_HPP

#include <algorithm>
#include <cstdint>
#include "common/bfloat16.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {
namespace quant {

using zendnnl::common::bfloat16_t;

/// 127 / m and m / 127 for a group with max |x| = `amax`.
inline float quant_inv(float amax) {
    return 127.f / std::max(amax, 1e-10f);
}
inline float quant_scale(float amax) {
    return std::max(amax, 1e-10f) / 127.f;
}

/// x[0, n) as one group, codes to y[0, n); returns the scale.
float quantize_row(const bfloat16_t *x, int64_t n, int offset, uint8_t *y);

/// Each of the n <= 32 columns of `rows` x n BF16 x (rows `ldx` elements
/// apart) as one group, codes (offset 0) to y (rows `ldy` bytes apart, 32
/// bytes per row written, 0 past n); scale[0, 32) gets the scales (0 past
/// n).
void quantize_cols(const bfloat16_t *x, int64_t ldx, int64_t rows, int64_t n,
        int8_t *y, int64_t ldy, float *scale);

/// Rows [ra, rb) of FP32 x (rows `ld` apart) over columns [lo, hi), each row
/// one group whose max |x| is the max of amax[16 r, 16 r + 16); codes to the
/// same positions of y (rows `ld` bytes apart). scale[r] = the row's scale *
/// row_mul[r].
void quantize_rows_amax(const float *x, int64_t ld, const float *amax,
        const float *row_mul, int64_t ra, int64_t rb, int64_t lo, int64_t hi,
        int offset, uint8_t *y, float *scale);

} // namespace quant
} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_QUANT_HPP
