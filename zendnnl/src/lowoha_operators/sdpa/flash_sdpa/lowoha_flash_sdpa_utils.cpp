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

#include "lowoha_flash_sdpa_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

status_t validate_flash_sdpa_inputs(
  const void *query,
  const void *key,
  const void *value,
  void *output,
  const void *attn_mask,
  const sdpa_params &params) {

  // --- pointer checks ---
  if (query == nullptr) {
    log_error("sdpa_flash_cpu: query pointer is null");
    return status_t::failure;
  }
  if (key == nullptr) {
    log_error("sdpa_flash_cpu: key pointer is null");
    return status_t::failure;
  }
  if (value == nullptr) {
    log_error("sdpa_flash_cpu: value pointer is null");
    return status_t::failure;
  }
  if (output == nullptr) {
    log_error("sdpa_flash_cpu: output pointer is null");
    return status_t::failure;
  }

  // --- dimension checks ---
  if (params.batch <= 0) {
    log_error("sdpa_flash_cpu: batch must be > 0");
    return status_t::failure;
  }
  if (params.num_heads <= 0) {
    log_error("sdpa_flash_cpu: num_heads must be > 0");
    return status_t::failure;
  }
  if (params.seq_len <= 0) {
    log_error("sdpa_flash_cpu: seq_len must be > 0");
    return status_t::failure;
  }
  if (params.kv_seq_len < 0) {
    log_error("sdpa_flash_cpu: kv_seq_len must be >= 0 "
              "(0 means same as seq_len)");
    return status_t::failure;
  }
  if (params.head_dim <= 0) {
    log_error("sdpa_flash_cpu: head_dim must be > 0");
    return status_t::failure;
  }

  // --- stride_d checks: Q/K/V must be contiguous along head_dim (GEMM requirement) ---
  if (params.q_stride_d != 1) {
    log_error("sdpa_flash_cpu: q_stride_d must be 1 (contiguous head_dim)");
    return status_t::failure;
  }
  if (params.k_stride_d != 1) {
    log_error("sdpa_flash_cpu: k_stride_d must be 1 (contiguous head_dim)");
    return status_t::failure;
  }
  if (params.v_stride_d != 1) {
    log_error("sdpa_flash_cpu: v_stride_d must be 1 (contiguous head_dim)");
    return status_t::failure;
  }

  // --- stride_s checks: seq stride is the GEMM leading dimension, must be positive ---
  if (params.q_stride_s <= 0) {
    log_error("sdpa_flash_cpu: q_stride_s must be > 0");
    return status_t::failure;
  }
  if (params.k_stride_s <= 0) {
    log_error("sdpa_flash_cpu: k_stride_s must be > 0");
    return status_t::failure;
  }
  if (params.v_stride_s <= 0) {
    log_error("sdpa_flash_cpu: v_stride_s must be > 0");
    return status_t::failure;
  }
  if (params.o_stride_s <= 0) {
    log_error("sdpa_flash_cpu: o_stride_s must be > 0");
    return status_t::failure;
  }
  if (params.o_stride_d <= 0) {
    log_error("sdpa_flash_cpu: o_stride_d must be > 0");
    return status_t::failure;
  }

  // --- batch/head stride checks for output (written in parallel, must not alias) ---
  if (params.batch > 1 && params.o_stride_b <= 0) {
    log_error("sdpa_flash_cpu: o_stride_b must be > 0 when batch > 1");
    return status_t::failure;
  }
  if (params.num_heads > 1 && params.o_stride_h <= 0) {
    log_error("sdpa_flash_cpu: o_stride_h must be > 0 when num_heads > 1");
    return status_t::failure;
  }

  // --- data type checks ---
  if (params.qkv_dt != data_type_t::f32 && params.qkv_dt != data_type_t::bf16) {
    log_error("sdpa_flash_cpu: qkv_dt must be f32 or bf16");
    return status_t::failure;
  }

  // --- dropout (only 0 supported) ---
  if (params.dropout_p != 0.0) {
    log_error("sdpa_flash_cpu: dropout_p must be 0");
    return status_t::failure;
  }

  // --- mask checks ---
  if (params.mask_ndims != 0 && params.mask_ndims != 2
      && params.mask_ndims != 4) {
    log_error("sdpa_flash_cpu: mask_ndims must be 0, 2, or 4");
    return status_t::failure;
  }
  if (params.mask_ndims > 0 || attn_mask != nullptr) {
    if (params.mask_ndims == 0 && attn_mask != nullptr) {
      log_error("sdpa_flash_cpu: attn_mask is non-null but mask_ndims is 0; "
                "set mask_ndims to 2 or 4");
      return status_t::failure;
    }
    if (params.mask_ndims > 0 && attn_mask == nullptr) {
      log_error("sdpa_flash_cpu: attn_mask is null but mask_ndims is %d",
                params.mask_ndims);
      return status_t::failure;
    }
    if (params.qkv_dt == data_type_t::bf16 && params.mask_dt != data_type_t::f32
        && params.mask_dt != data_type_t::bf16) {
      log_error("sdpa_flash_cpu: mask_dt must be f32 or bf16 when qkv_dt is bf16 and mask is "
                "provided");
      return status_t::failure;
    }
    else if (params.qkv_dt == data_type_t::f32 &&
             params.mask_dt != data_type_t::f32) {
      log_error("sdpa_flash_cpu: mask_dt must be f32 when qkv_dt is f32 and mask is "
                "provided");
      return status_t::failure;
    }
  }
  if (params.out_dt != data_type_t::none && params.out_dt != params.qkv_dt) {
    log_error("sdpa_flash_cpu: out_dt must be the same as qkv_dt");
    return status_t::failure;
  }

  // --- num_threads ---
  if (params.num_threads < 0) {
    log_error("sdpa_flash_cpu: num_threads must be >= 0");
    return status_t::failure;
  }

  return status_t::success;
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl
