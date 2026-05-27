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

// AOCL DLP-backed MoE gated activations (silu_and_mul, gelu_and_mul).
// swiglu_oai_mul stays on the AVX-512 path in group_matmul_moe_act.cpp.
//
// Return value contract for every entry point in this file:
//   true  - the activation has been applied (or was a legitimate no-op,
//           e.g. empty row range / null dst).  Caller does NOT need to
//           run a fallback.
//   false - DLP did not apply the activation (unsupported act, invalid
//           dtype/N, or DLP setup failure).  Caller MUST fall back to
//           apply_gated_act_inplace().

#include <cstdint>

#include "common/bfloat16.hpp"
#include "common/zendnnl_global.hpp"
#include "group_matmul_direct.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_postop.hpp"
#include "aocl_dlp.h"


namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::bfloat16_t;
using zendnnl::memory::data_type_t;

namespace {

// Build a 2-op DLP postop chain: ELTWISE(act_kind) + BINARY_MUL(up_base, ldc).
dlp_metadata_t *build_act_and_mul_metadata(
  post_op_type_t act_kind,
  void *up_base,
  int ldc,
  int m_rows,
  int dim,
  data_type_t dst_dtype) {

  matmul_params lparams;
  lparams.dtypes.src  = dst_dtype;
  lparams.dtypes.wei  = dst_dtype;
  lparams.dtypes.dst  = dst_dtype;
  lparams.dtypes.bias = data_type_t::none;
  lparams.lowoha_algo = matmul_algo_t::aocl_dlp;

  matmul_post_op act_op;
  act_op.po_type = act_kind;
  lparams.postop_.push_back(act_op);

  matmul_post_op mul_op;
  mul_op.po_type     = post_op_type_t::binary_mul;
  mul_op.buff        = up_base;
  mul_op.dtype       = dst_dtype;
  mul_op.leading_dim = ldc;
  mul_op.dims        = { static_cast<int64_t>(m_rows),
                         static_cast<int64_t>(dim)
                       };
  lparams.postop_.push_back(mul_op);

  matmul_data_types dtypes = lparams.dtypes;
  return create_dlp_post_op(
           lparams, /*bias=*/nullptr, dtypes,
           /*N=*/dim, /*K=*/0, /*M=*/m_rows,
           /*zp_comp_acc=*/nullptr, /*zp_comp_ndim=*/0,
           matmul_algo_t::aocl_dlp,
           /*weight_ptr=*/nullptr); // weight_ptr is not used for MoE gated activations
}

// dst[m, 0:dim) = act(dst[m, 0:dim)) * dst[m, dim:2*dim), in-place.
// Returns true on success (or legitimate no-op), false on any failure
// where the caller must fall back to the AVX-512/scalar path.
bool apply_act_and_mul_dlp_impl(
  post_op_type_t act_kind,
  const char *op_name,
  void *dst,
  int row_start, int row_end,
  int N, int ldc,
  data_type_t dst_dtype) {

  // Legitimate no-op: nothing to compute.  Match apply_gated_act_inplace()
  // semantics so the caller can drop the fallback in this case.
  if (row_start >= row_end || dst == nullptr) {
    return true;
  }
  if (dst_dtype != data_type_t::f32 && dst_dtype != data_type_t::bf16) {
    log_error(op_name, ": unsupported dst_dtype (must be f32 or bf16)");
    return false;
  }
  if (N <= 0 || (N & 1) != 0) {
    log_error(op_name, ": N=", N, " must be positive and even");
    return false;
  }

  const int    m_rows = row_end - row_start;
  const int    dim    = N / 2;
  const size_t elem   = (dst_dtype == data_type_t::f32)
                        ? sizeof(float)
                        : sizeof(bfloat16_t);

  uint8_t *base      = static_cast<uint8_t *>(dst)
                       + static_cast<size_t>(row_start) * ldc * elem;
  uint8_t *gate_base = base;
  uint8_t *up_base   = base + static_cast<size_t>(dim) * elem;

  dlp_metadata_t *md = build_act_and_mul_metadata(
                         act_kind,
                         static_cast<void *>(up_base),
                         ldc, m_rows, dim, dst_dtype);
  if (md == nullptr) {
    log_error(op_name, ": create_dlp_post_op returned null");
    return false;
  }

  if (dst_dtype == data_type_t::f32) {
    aocl_gemm_eltwise_ops_f32of32(
      'r', 'n', 'n',
      m_rows, dim,
      reinterpret_cast<const float *>(gate_base), ldc,
      reinterpret_cast<float *>(gate_base),       ldc,
      md);
  }
  else {
    aocl_gemm_eltwise_ops_bf16obf16(
      'r', 'n', 'n',
      m_rows, dim,
      reinterpret_cast<const int16_t *>(gate_base), ldc,
      reinterpret_cast<int16_t *>(gate_base),       ldc,
      md);
  }

  cleanup_dlp_post_op(md);
  return true;
}

}  // namespace

bool silu_and_mul_dlp(
  void *dst,
  int row_start, int row_end,
  int N, int ldc,
  data_type_t dst_dtype) {
  return apply_act_and_mul_dlp_impl(
           post_op_type_t::swish, "silu_and_mul_dlp",
           dst, row_start, row_end, N, ldc, dst_dtype);
}

bool gelu_and_mul_dlp(
  void *dst,
  int row_start, int row_end,
  int N, int ldc,
  data_type_t dst_dtype) {
  return apply_act_and_mul_dlp_impl(
           post_op_type_t::gelu_erf, "gelu_and_mul_dlp",
           dst, row_start, row_end, N, ldc, dst_dtype);
}

// Returns false when DLP cannot or did not apply the activation (caller
// must then fall back to apply_gated_act_inplace()).
bool apply_gated_act_inplace_dlp(
  grp_matmul_gated_act_t act,
  void *dst, int row_start, int row_end,
  int N, int ldc, data_type_t dst_dtype) {
  switch (act) {
  case grp_matmul_gated_act_t::none:
    return true;
  case grp_matmul_gated_act_t::silu_and_mul:
    return silu_and_mul_dlp(dst, row_start, row_end, N, ldc, dst_dtype);
  case grp_matmul_gated_act_t::gelu_and_mul:
    return gelu_and_mul_dlp(dst, row_start, row_end, N, ldc, dst_dtype);
  case grp_matmul_gated_act_t::swiglu_oai_mul:
    return false;
  }
  return false;
}


}  // namespace matmul
}  // namespace lowoha
}  // namespace zendnnl
