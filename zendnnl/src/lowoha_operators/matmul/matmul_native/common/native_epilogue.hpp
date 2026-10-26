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

#ifndef MATMUL_NATIVE_COMMON_NATIVE_EPILOGUE_HPP
#define MATMUL_NATIVE_COMMON_NATIVE_EPILOGUE_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "common/error_status.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::status_t;

/// A tensor read at coordinates (m, n) of the [M, n_out] output the way the
/// reference algorithm reads it: element (m * n_out + n) % count.  One
/// element is a scalar, n_out elements one value per column, M * n_out a
/// matrix with row stride ld; any other count keeps the flat modulo.
struct native_operand_t {
    enum class kind_t : int { scalar, column, matrix, modulo };
    kind_t kind = kind_t::scalar;
    const void *buf = nullptr;
    data_type_t dt = data_type_t::none;
    int esz = 0;
    /// Read as int32, floats truncated (a zero point).
    bool as_int = false;
    int64_t ld = 0;
    int64_t count = 1;
    float value = 0.0f; ///< the scalar
};

/// One post-op of the call, resolved once.
struct native_postop_t {
    post_op_type_t type = post_op_type_t::none;
    float alpha = 0.0f;
    float beta = 0.0f;
    native_operand_t operand; ///< binary_add / binary_mul
};

/// Everything after the accumulation of one call over a native-layout
/// weight: alpha, beta, bias, the gated activation, post-ops and the store to
/// dst (f32, bf16, f16, or u8 / s8 / s32 requantized).  Prepared once per call,
/// then applied by every thread to its own tiles.
struct native_epilogue_t {
    int N = 0; ///< weight columns; a gated activation writes N / 2
    int n_out = 0; ///< dst columns
    float alpha = 1.0f;
    float beta = 0.0f;
    const float *bias = nullptr; ///< an f32 bias of a plain pack
    /// Any other bias, read in its own type at the tile (an interleaved pack
    /// reads gate column i at pack column 2i and up column i at 2i + 1).
    const void *bias_raw = nullptr;
    data_type_t bias_dt = data_type_t::none;
    int bias_esz = 0;
    matmul_gated_act_t gated = matmul_gated_act_t::none;
    bool interleaved = false; ///< pack column 2i is gate i, 2i + 1 is up i
    void *dst = nullptr;
    data_type_t dst_dt = data_type_t::none;
    int dst_esz = 0;
    int ldc = 0;
    std::vector<native_postop_t> ops;

    /// INT8 calls with a dst scale store v * dst_scale + dst_zp.
    bool requant = false;
    bool has_dst_zp = false;
    native_operand_t dst_scale;
    native_operand_t dst_zp;

    /// A softmax needs whole rows: ops up to softmax_at stage each tile in
    /// rows, and the tile that completes a row finishes it.
    size_t softmax_at = 0;
    std::unique_ptr<float[]> rows;
    std::unique_ptr<std::atomic<int>[]> row_cols;
};

/// f32, bf16 or f16: the float types the native BRGEMM reads in their own
/// type and widens exactly (scales, a bias, operands, a destination).
inline bool native_float_dt(data_type_t dt) {
    return dt == data_type_t::f32 || dt == data_type_t::bf16
            || dt == data_type_t::f16;
}

/// Fills ep for the call; unimplemented when the call needs something the
/// epilogue cannot finish.  The alpha and beta of an INT8 call with
/// per-tensor or per-channel weight scales act on the integer sum, so its
/// tile applies them and desc carries alpha 1 and beta 0; those of a call
/// with per-group weight scales apply here, to the sum of the scaled groups.
/// The CPU has AVX512-BF16 or AVX512-VNNI.
status_t prepare_epilogue(const GemmDescriptor &desc,
        const matmul_params &params, const void *bias, void *dst,
        bool interleaved, native_epilogue_t &ep);

/// Finishes C[mr][nr] (f32 accumulators of rows m0 .. m0 + mr and pack
/// columns n0 .. n0 + nr of one block) and stores them to dst.  A gated
/// activation needs n0 and nr even.  C may be overwritten.  Tiles of a call
/// are disjoint and cover the output once.
void apply_epilogue_tile(const native_epilogue_t &ep, float *C, int ldc_acc,
        int m0, int mr, int n0, int nr);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_EPILOGUE_HPP
