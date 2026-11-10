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

/// Whether a gated activation takes its gate and up columns as the [gate |
/// up] halves of the weight (gate column i pairs with up column N / 2 + i);
/// swiglu_oai_mul interleaves its pairs instead.
inline bool native_gated_halves(matmul_gated_act_t gated) {
    return gated == matmul_gated_act_t::silu_and_mul
            || gated == matmul_gated_act_t::gelu_and_mul
            || gated == matmul_gated_act_t::gelu_tanh_and_mul;
}

/// Everything after the accumulation of one call over a native-layout
/// weight: alpha, beta, bias, post-ops, the gated activation and the store
/// to dst (f32, bf16, f16, or u8 / s8 / s32 requantized).  A gated call
/// applies alpha, beta, the bias and the post-ops to its N pre-activation
/// columns, as matmul_direct defines it (the gated activation follows the
/// matmul), then the activation and the store of N / 2 columns.  Prepared
/// once per call, then applied by every thread to its own tiles.
struct native_epilogue_t {
    int N = 0; ///< weight columns; a gated activation writes N / 2
    int n_out = 0; ///< dst columns
    float alpha = 1.0f;
    float beta = 0.0f;
    const float *bias = nullptr; ///< an f32 bias
    /// Any other bias, read in its own type at the tile.
    const void *bias_raw = nullptr;
    data_type_t bias_dt = data_type_t::none;
    int bias_esz = 0;
    matmul_gated_act_t gated = matmul_gated_act_t::none;
    /// The tiles of a gated call over [gate | up] halves hold their gate
    /// columns, then the up columns paired with them (native_weight_view_t).
    bool paired = false;
    /// Halves of a gated call without paired tiles: whole rows of the
    /// pre-activation stage in staged, which pair_staged_halves pairs once
    /// every tile is done.
    bool staged_halves = false;
    void *dst = nullptr;
    data_type_t dst_dt = data_type_t::none;
    int dst_esz = 0;
    int ldc = 0;
    /// Only the store: an f32 or bf16 dst, alpha 1, beta 0, and no bias,
    /// post-op, requantization or gated activation.
    bool store_only = false;
    std::vector<native_postop_t> ops;

    /// INT8 calls with a dst scale store v * dst_scale + dst_zp.
    bool requant = false;
    bool has_dst_zp = false;
    native_operand_t dst_scale;
    native_operand_t dst_zp;

    /// A softmax needs whole rows: ops up to softmax_at stage each tile in
    /// rows, and the tile that completes a row finishes it (rows of n_out
    /// columns).
    size_t softmax_at = 0;
    std::unique_ptr<float[]> rows;
    std::unique_ptr<std::atomic<int>[]> row_cols;
    /// With staged_halves, M rows of N columns: the calling thread's buffer,
    /// kept across its calls (kNativeThreadStagedFloats).
    float *staged = nullptr;

    /// What beta reads in a gated call, rows beta_ld elements apart of the
    /// dst type: dst, or a copy of its N pre-activation columns when the
    /// call's tiles store over columns other tiles read (see
    /// native_brgemm_execute).
    const void *beta_src = nullptr;
    int beta_ld = 0;
};

/// f32, bf16 or f16: the float types the native BRGEMM reads in their own
/// type and widens exactly (scales, a bias, operands, a destination).
inline bool native_float_dt(data_type_t dt) {
    return dt == data_type_t::f32 || dt == data_type_t::bf16
            || dt == data_type_t::f16;
}

/// Fills ep for the call; unimplemented when the call needs something the
/// epilogue cannot finish (a gated call with a softmax, or with beta over a
/// dst of fewer than N columns), with *why (when why is not null) naming
/// it.  paired: the call's tiles are paired (native_weight_view_t), for a
/// gated activation over [gate | up] halves only; such a call without them
/// stages its rows.  Either way the weight's layout is a plain one.  The alpha and beta of an INT8 call with per-tensor or per-channel
/// weight scales act on the integer sum, so its tile applies them and desc
/// carries alpha 1 and beta 0; those of a call with per-group weight scales
/// (any call over an s4 weight) apply here, to the sum of the scaled groups.
/// An INT8 call over an s8 or s4 weight with a dst scale stores the
/// requantized value.  A gated call's post-op
/// operands index its [M, N] pre-activation.  Without buffers (a check that
/// runs no tile) the rows of a softmax and the staged halves are not
/// allocated.  The CPU has AVX-512 F, BW, VL and DQ and FMA
/// (native_base_isa).
status_t prepare_epilogue(const GemmDescriptor &desc,
        const matmul_params &params, const void *bias, void *dst, bool paired,
        native_epilogue_t &ep, const char **why = nullptr, bool buffers = true);

/// The most floats of staged rows a calling thread keeps across calls that
/// fit them (4 MiB).  Rows a call needs above that serve the thread's calls
/// that need them, and its next staged call that fits the bound frees them.
inline constexpr size_t kNativeThreadStagedFloats
        = (size_t {4} << 20) / sizeof(float);

/// The floats of staged rows the calling thread keeps.  A diagnostic; no
/// call reads it.
size_t native_thread_staged_floats();

/// Finishes C[mr][nr] (f32 accumulators of rows m0 .. m0 + mr and pack
/// columns n0 .. n0 + nr, or with paired tiles the columns the tile at n0
/// holds) and stores them to dst.  A gated activation needs n0 and nr even.
/// C may be overwritten.  Tiles of a call are disjoint and cover the output
/// once; with beta they read beta_src while other tiles store.  No column
/// past n0 + nr of any operand or of dst is read or written: the last tile
/// of a weight of padded columns passes only its columns below N.
void apply_epilogue_tile(const native_epilogue_t &ep, float *C, int ldc_acc,
        int m0, int mr, int n0, int nr);

/// With staged_halves, once every tile of the call is done: share i of n of
/// the activation of the staged halves of its M rows, stored to dst.  The n
/// shares cover every output once.
void pair_staged_halves(const native_epilogue_t &ep, int M, int i, int n);

/// The units pair_staged_halves deals over its shares for M rows.
int64_t staged_pair_units(const native_epilogue_t &ep, int M);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_EPILOGUE_HPP
