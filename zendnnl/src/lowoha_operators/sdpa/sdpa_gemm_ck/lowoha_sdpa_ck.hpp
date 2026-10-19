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

/// Custom-kernel (CK) paths for SDPA, one namespace per data type, each with
/// its `<dtype>_mode()` and `<dtype>/lowoha_sdpa_ck_<dtype>.cpp`. flash_sdpa
/// reaches them through its GEMM entry points: every BF16 `zendnn_gemm` call
/// of the flash kernels (including the BF16 GEMMs of the INT8 flash path),
/// and the Q*K^T and P*V matmuls of the dynamic-INT8 flash path.

#ifndef LOWOHA_SDPA_CK_HPP
#define LOWOHA_SDPA_CK_HPP

#include "common/bfloat16.hpp"

#include <cstddef>
#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace sdpa {
namespace ck {

/// How a data type uses the custom kernel.
enum class mode_t {
    off, ///< AOCL-DLP flash path.
    gemm, ///< Both flash GEMMs on CK; softmax in the default flash loop.
};

/// BF16 mode on an AVX512-BF16/BW/VL CPU (off otherwise): the flash GEMMs run
/// on CK unless `ZENDNNL_SDPA_BF16_CK` is false (0/false/off/no). The
/// variable is read once.
mode_t bf16_mode();

/// INT8 mode on an AVX512-VNNI/BW/VL CPU (off otherwise): the INT8 matmuls of
/// the dynamic-INT8 flash path run on CK unless `ZENDNNL_SDPA_INT8_CK` is
/// false. The variable is read once.
mode_t int8_mode();

/// BF16 kernels. They need AVX512-BF16/BW/VL: callers check `bf16_mode()`
/// first.
namespace bf16 {

using zendnnl::common::bfloat16_t;

/// Elements of a packed op(B) of K x N.
size_t packed_size(int64_t k, int64_t n);

/// Packs op(B) (K x N) into 32-column blocks of [ceil(K/2)][32][2] BF16,
/// zero padded past K and N. `trans_b`: B is stored N x K with row stride
/// `ldb`; otherwise K x N.
void pack_b(int64_t k, int64_t n, const bfloat16_t *b, int64_t ldb,
        bool trans_b, bfloat16_t *packed);

/// C[M, N] (+)= A[M, K] * B, B from `pack_b`; adds to C when `accumulate`.
void gemm_packed(int64_t m, int64_t n, int64_t k, const bfloat16_t *a,
        int64_t lda, const bfloat16_t *packed, bool accumulate, float *c,
        int64_t ldc);

/// `zendnn_gemm` contract: C = alpha * A * op(B) + beta * C, FP32 C, packing
/// B on the fly. Covers alpha 1, beta 0 or 1, an untransposed A and M of 0 or
/// at least 3; returns false without touching C otherwise.
bool gemm(int64_t m, int64_t n, int64_t k, float alpha, const bfloat16_t *a,
        int64_t lda, const bfloat16_t *b, int64_t ldb, float beta, float *c,
        int64_t ldc, bool trans_a, bool trans_b);

} // namespace bf16

/// INT8 kernels for the dynamic-INT8 flash path, packing B on the fly. They
/// need AVX512-VNNI/BW/VL: callers check `int8_mode()` first.
namespace int8 {

/// scores[M, N] = (Q * K^T)[m, n] * q_scales[m] * k_scales[n], with S8 Q
/// (M x D) and K (N x D), both row-major and dense. Covers any M.
bool qk(const int8_t *q, const int8_t *k, float *scores, int64_t m, int64_t n,
        int64_t d, const float *q_scales, const float *k_scales);

/// out[M, N] = (P * V)[m, n] * p_scale * v_scales[n], with U8 P (M x KV) and
/// S8 V (KV x N), both row-major and dense. Covers M other than 1 and
/// returns false without touching `out` for M of 1.
bool pv(const uint8_t *p, const int8_t *v, float *out, int64_t m, int64_t n,
        int64_t kv, float p_scale, const float *v_scales);

} // namespace int8

namespace test_api {
/// Override `bf16_mode()` / `int8_mode()`: -1 follows the environment, 0
/// off, 1 GEMM.
void set_bf16_mode(int value);
void set_int8_mode(int value);
/// Resets the counts of GEMM calls taken by CK and turns counting on or off.
void count_calls(bool on);
/// BF16 / INT8 GEMM calls taken by CK since the last `count_calls(true)`.
uint64_t bf16_gemm_calls();
uint64_t int8_gemm_calls();
} // namespace test_api

} // namespace ck
} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#endif
