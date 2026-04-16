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
 ******************************************************************************/

#include "lowoha_operators/matmul/partitioning/bmm/kernel/bmm_kernel.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/matmul/backends/libxsmm/libxsmm_kernel.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_operators/matmul/backends/onednn/onednn_kernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace bmm {

void bmm_tile_execute(
  int batch_idx, int m_start, int m_len,
  const uint8_t *src_ptr, const uint8_t *weight_ptr, uint8_t *dst_ptr,
  const BmmKernelContext &ctx,
  matmul_params &params,
  matmul_batch_params_t &batch_params) {

  const void *A = get_matrix_block(src_ptr, m_start, 0,
                                   ctx.lda, ctx.transA, ctx.src_type_size);
  void *C = get_output_block(dst_ptr, m_start, 0,
                             ctx.ldc, ctx.out_type_size);

  matmul_params tile_params = params;
  apply_bmm_postop_offsets(tile_params, batch_idx, m_start, ctx.N);

  matmul_algo_t tile_kernel = ctx.kernel;

#if ZENDNNL_DEPENDS_LIBXSMM
  if (tile_kernel == matmul_algo_t::libxsmm) {
    log_info("Using libxsmm kernel");
    if (run_libxsmm(ctx.trans_input, ctx.trans_weight, m_len, ctx.N, ctx.K,
                    ctx.beta, ctx.lda, ctx.ldb, ctx.ldc,
                    A, weight_ptr, C, tile_params.dtypes,
                    tile_params, ctx.bias)) {
      return;
    }
  }
#endif
#if ZENDNNL_DEPENDS_ONEDNN
  if (tile_kernel == matmul_algo_t::onednn ||
      tile_kernel == matmul_algo_t::onednn_blocked) {
    log_info("Using onednn kernel");
    matmul_onednn_wrapper(ctx.trans_input, ctx.trans_weight, m_len, ctx.N, ctx.K,
                          ctx.alpha, A, ctx.lda, weight_ptr, ctx.ldb, ctx.beta, C,
                          ctx.ldc, tile_params, batch_params, ctx.bias,
                          tile_kernel, ctx.is_weights_const);
    return;
  }
#endif
  log_info("Using AOCL DLP kernel");
  run_dlp(ctx.layout, ctx.trans_input, ctx.trans_weight, m_len, ctx.N, ctx.K,
          ctx.alpha, ctx.beta,
          ctx.lda, ctx.ldb, ctx.ldc,
          tile_params.mem_format_a, tile_params.mem_format_b,
          A, weight_ptr, C, tile_params.dtypes, tile_params, ctx.bias,
          tile_kernel, ctx.is_weights_const);
}

} // namespace bmm
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
