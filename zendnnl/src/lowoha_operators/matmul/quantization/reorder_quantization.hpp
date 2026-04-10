/*******************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef REORDER_QUANTIZATION_HPP
#define REORDER_QUANTIZATION_HPP

#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include <cstdlib>

namespace zendnnl {
namespace lowoha {
namespace matmul {

/**
 * @brief RAII holder for buffers allocated during reorder quantization.
 *        Automatically frees on scope exit; non-copyable, move-only.
 */
struct reorder_quant_buffers_t {
  uint8_t *src_buf   = nullptr;
  uint8_t *scale_buf = nullptr;
  uint8_t *zp_buf    = nullptr;

  reorder_quant_buffers_t() = default;
  ~reorder_quant_buffers_t() { free(src_buf); free(scale_buf); free(zp_buf); }

  reorder_quant_buffers_t(const reorder_quant_buffers_t &) = delete;
  reorder_quant_buffers_t &operator=(const reorder_quant_buffers_t &) = delete;
  reorder_quant_buffers_t(reorder_quant_buffers_t &&o) noexcept
      : src_buf(o.src_buf), scale_buf(o.scale_buf), zp_buf(o.zp_buf) {
    o.src_buf = o.scale_buf = o.zp_buf = nullptr;
  }
  reorder_quant_buffers_t &operator=(reorder_quant_buffers_t &&o) noexcept {
    if (this != &o) {
      free(src_buf); free(scale_buf); free(zp_buf);
      src_buf = o.src_buf; scale_buf = o.scale_buf; zp_buf = o.zp_buf;
      o.src_buf = o.scale_buf = o.zp_buf = nullptr;
    }
    return *this;
  }
};

/**
 * @brief Attempt reorder quantization of the source tensor if eligible.
 *
 * Checks whether the dtype combination qualifies for reorder quantization
 * (src is BF16/F32, weight is S8, compute is S8/U8). If eligible, quantizes
 * the source via reorder and updates src/params in-place. If not eligible,
 * returns success with no changes.
 *
 * Supports two modes based on params.dynamic_quant:
 * - Dynamic (true):  Computes scale (and zp for u8) on-the-fly.
 * - Static  (false): Uses user-provided scale (and zp for u8) values.
 *
 * @param[in,out] src              Source data pointer; updated on success
 * @param[in]     lda              Leading dimension of original source
 * @param[in,out] reordered_lda    Updated to contiguous lda on success
 * @param[in,out] src_type_size    Updated to quantized element size on success
 * @param[in,out] params           dtypes.src and quant_params updated on success
 * @param[in,out] batch_params     batch_stride_src updated on success
 * @param[in]     transA           Whether source matrix is transposed
 * @param[in]     M                Number of rows
 * @param[in]     K                Shared (inner) dimension
 * @param[in]     num_threads      Thread count for the reorder operation
 * @param[out]    buffers          RAII holder; freed automatically on scope exit
 *
 * @return status_t::success  Quantization performed, skipped (not eligible),
 *                            or reorder failed gracefully
 * @return status_t::failure  Validation failed — caller should propagate
 */
status_t reorder_quantization_wrapper(
    const void *&src, const int lda, int &reordered_lda, size_t &src_type_size,
    matmul_params &params, matmul_batch_params_t &batch_params,
    const bool transA, const int M, const int K, const int num_threads,
    reorder_quant_buffers_t &buffers);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // REORDER_QUANTIZATION_HPP
