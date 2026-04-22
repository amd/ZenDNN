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

#include "lowoha_flash_sdpa.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

namespace {

sdpa_flash_cpu_tensor_view build_tensor_view(
  const void *data, const sdpa_params &p,
  int64_t sb, int64_t sh, int64_t ss, int64_t sd,
  int64_t seq_len) {
  sdpa_flash_cpu_tensor_view v{};
  v.data      = data;
  v.size_b    = p.batch;
  v.size_h    = p.num_heads;
  v.size_s    = seq_len;
  v.size_d    = p.head_dim;
  v.stride_b  = sb;
  v.stride_h  = sh;
  v.stride_s  = ss;
  v.stride_d  = sd;
  return v;
}

} // namespace

status_t sdpa_flash_cpu_standalone(
  const void *query,
  const void *key,
  const void *value,
  const void *attn_mask,
  void *output,
  const sdpa_params &params) {
  if (validate_flash_sdpa_inputs(query, key, value, output, attn_mask, params)
      != status_t::success) {
    return status_t::failure;
  }

  const int64_t eff_kv_seq_len = (params.kv_seq_len > 0)
                                 ? params.kv_seq_len
                                 : params.seq_len;

  sdpa_flash_cpu_tensor_view qv = build_tensor_view(
                                    query, params,
                                    params.q_stride_b, params.q_stride_h,
                                    params.q_stride_s, params.q_stride_d,
                                    params.seq_len);
  sdpa_flash_cpu_tensor_view kv = build_tensor_view(
                                    key, params,
                                    params.k_stride_b, params.k_stride_h,
                                    params.k_stride_s, params.k_stride_d,
                                    eff_kv_seq_len);
  sdpa_flash_cpu_tensor_view vv = build_tensor_view(
                                    value, params,
                                    params.v_stride_b, params.v_stride_h,
                                    params.v_stride_s, params.v_stride_d,
                                    eff_kv_seq_len);
  sdpa_flash_cpu_tensor_view ov = build_tensor_view(
                                    output, params,
                                    params.o_stride_b, params.o_stride_h,
                                    params.o_stride_s, params.o_stride_d,
                                    params.seq_len);

  const sdpa_flash_cpu_mask_view *mask_ptr = nullptr;
  sdpa_flash_cpu_mask_view mask_view{};
  if (attn_mask != nullptr && params.mask_ndims > 0) {
    mask_view.data = attn_mask;
    mask_view.ndim = params.mask_ndims;
    for (int i = 0; i < 4; ++i) {
      mask_view.sizes[i]   = params.mask_sizes[i];
      mask_view.strides[i] = params.mask_strides[i];
    }
    mask_ptr = &mask_view;
  }

  double scale_val = params.scale;
  const double *scale_opt =
    (scale_val != 0.0) ? &scale_val : nullptr;

  return sdpa_flash_cpu_run_internal(
           ov, qv, kv, vv,
           params.dropout_p, params.is_causal,
           mask_ptr, scale_opt,
           params.qkv_dt, params.mask_dt,
           params.num_threads);
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl
