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

#include "reorder_quantization.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::size_of;

status_t reorder_quantization_wrapper(
    const void *&src, const int lda, int &reordered_lda, size_t &src_type_size,
    matmul_params &params, matmul_batch_params_t &batch_params,
    const bool transA, const int M, const int K, const int num_threads,
    reorder_quant_buffers_t &buffers) {

  const bool eligible =
      params.dynamic_quant && //TODO: Masking static quantization for now, Remove later.
      params.dtypes.wei == data_type_t::s8 &&
      (params.dtypes.src == data_type_t::bf16 ||
       params.dtypes.src == data_type_t::f32) &&
      (params.dtypes.compute == data_type_t::s8 ||
       params.dtypes.compute == data_type_t::u8);

  if (!eligible) return status_t::success;

  const bool is_dynamic = params.dynamic_quant;
  const data_type_t quant_dtype = params.dtypes.compute;
  const data_type_t orig_src_dtype = params.dtypes.src;
  const bool needs_zp = (quant_dtype == data_type_t::u8);

  if (params.quant_params.src_scale.dims.empty() ||
      params.quant_params.src_scale.dt == data_type_t::none) {
    log_error("Reorder quantization requires quant_params.src_scale "
              "dims and dt to be set");
    return status_t::failure;
  }
  if (needs_zp &&
      (params.quant_params.src_zp.dims.empty() ||
       params.quant_params.src_zp.dt == data_type_t::none)) {
    log_error("Asymmetric (u8) quantization requires quant_params.src_zp "
              "dims and dt to be set");
    return status_t::failure;
  }
  if (!is_dynamic) {
    if (!params.quant_params.src_scale.buff) {
      log_error("Static quantization requires quant_params.src_scale.buff "
                "to be provided");
      return status_t::failure;
    }
    if (needs_zp && !params.quant_params.src_zp.buff) {
      log_error("Static quantization requires quant_params.src_zp.buff "
                "to be provided for asymmetric (u8) quantization");
      return status_t::failure;
    }
  }

  apilog_info("Reorder quantization: using ",
              is_dynamic ? "dynamic" : "static",
              needs_zp ? " asymmetric" : " symmetric", " quantization");

  const int64_t phys_rows = transA ? K : M;
  const int64_t phys_cols = transA ? M : K;
  const int64_t batch_A = batch_params.Batch_A;
  const bool is_batched = (batch_A > 1);

  buffers.src_buf = static_cast<uint8_t *>(
      malloc(static_cast<size_t>(batch_A * phys_rows * phys_cols)));
  if (!buffers.src_buf) {
    log_error("Reorder quantization: failed to allocate quantized source buffer");
    return status_t::failure;
  }

  void *scale_buff = const_cast<void *>(params.quant_params.src_scale.buff);
  if (!scale_buff) {
    int64_t n = 1;
    for (auto d : params.quant_params.src_scale.dims) n *= d;
    buffers.scale_buf = static_cast<uint8_t *>(
        malloc(static_cast<size_t>(n) *
               size_of(params.quant_params.src_scale.dt)));
    if (!buffers.scale_buf) {
      log_error("Reorder quantization: failed to allocate scale buffer");
      return status_t::failure;
    }
    scale_buff = buffers.scale_buf;
  }

  void *zp_buff = nullptr;
  if (needs_zp) {
    zp_buff = const_cast<void *>(params.quant_params.src_zp.buff);
    if (!zp_buff) {
      int64_t n = 1;
      for (auto d : params.quant_params.src_zp.dims) n *= d;
      buffers.zp_buf = static_cast<uint8_t *>(
          malloc(static_cast<size_t>(n) *
                 size_of(params.quant_params.src_zp.dt)));
      if (!buffers.zp_buf) {
        log_error("Reorder quantization: failed to allocate zero-point buffer");
        return status_t::failure;
      }
      zp_buff = buffers.zp_buf;
    }
  }

  zendnnl::lowoha::reorder::reorder_params_t rp;
  rp.src_dtype = orig_src_dtype;
  rp.dst_dtype = quant_dtype;
  rp.dynamic_quant = is_dynamic;
  rp.num_threads = num_threads;

  if (is_batched) {
    rp.src_shape = {batch_A, phys_rows, phys_cols};
    rp.dst_shape = rp.src_shape;
    const int64_t batch_stride =
        (batch_params.batch_stride_src != static_cast<size_t>(-1))
            ? static_cast<int64_t>(batch_params.batch_stride_src)
            : phys_rows * lda;
    if (lda != static_cast<int>(phys_cols) ||
        batch_stride != phys_rows * phys_cols) {
      rp.src_strides = {batch_stride, static_cast<int64_t>(lda), 1};
    }
  } else {
    rp.src_shape = {phys_rows, phys_cols};
    rp.dst_shape = rp.src_shape;
    if (lda != static_cast<int>(phys_cols)) {
      rp.src_strides = {static_cast<int64_t>(lda), 1};
    }
  }

  rp.quant_params.scale.buff = scale_buff;
  rp.quant_params.scale.dt = params.quant_params.src_scale.dt;
  rp.quant_params.scale.dims = params.quant_params.src_scale.dims;

  if (needs_zp) {
    rp.quant_params.zero_point.buff = zp_buff;
    rp.quant_params.zero_point.dt = params.quant_params.src_zp.dt;
    rp.quant_params.zero_point.dims = params.quant_params.src_zp.dims;
  }

  const status_t reorder_status = zendnnl::lowoha::reorder::reorder_direct(
      src, buffers.src_buf, std::move(rp));

  if (reorder_status == status_t::success) {
    src = buffers.src_buf;
    reordered_lda = static_cast<int>(phys_cols);
    params.dtypes.src = quant_dtype;
    src_type_size = size_of(quant_dtype);

    if (is_batched) {
      batch_params.batch_stride_src =
          static_cast<size_t>(phys_rows * phys_cols);
    }

    params.quant_params.src_scale.buff = scale_buff;
    if (needs_zp) {
      params.quant_params.src_zp.buff = zp_buff;
    }

    if (apilog_info_enabled()) {
      int64_t ns = 1;
      for (auto d : params.quant_params.src_scale.dims) ns *= d;
      int64_t nz = 0;
      if (needs_zp) {
        nz = 1;
        for (auto d : params.quant_params.src_zp.dims) nz *= d;
      }
      apilog_info(is_dynamic ? "Dynamic" : "Static",
                  needs_zp ? " asymmetric" : " symmetric",
                  " quantization: source converted from ",
                  dtype_info(orig_src_dtype), " to ",
                  dtype_info(quant_dtype),
                  " (scale_elems=", ns, ", zp_elems=", nz, ")");
    }
  } else {
    log_error("Reorder quantization of source failed, "
              "falling back to original path");
  }

  return status_t::success;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
