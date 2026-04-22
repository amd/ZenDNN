/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# *******************************************************************************/

#ifndef LOWOHA_SDPA_FLASH_CPU_HPP
#define LOWOHA_SDPA_FLASH_CPU_HPP

#include <cstdint>

#include "lowoha_operators/sdpa/lowoha_sdpa_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

/// BF16 lane for flash CPU kernel (matches uint16 storage used by lowoha matmul).
struct bf16_elem {
  uint16_t x;
};

/// Layout [Batch, Head, Seq, Dim] (PyTorch-style BHSD).
struct sdpa_flash_cpu_tensor_view {
  const void *data;
  int64_t stride_b;
  int64_t stride_h;
  int64_t stride_s;
  int64_t stride_d;
  int64_t size_b;
  int64_t size_h;
  int64_t size_s;
  int64_t size_d;
};

/// Optional mask: 2D [S_q, S_kv] or 4D [B, H, S_q, S_kv] with broadcast strides.
struct sdpa_flash_cpu_mask_view {
  const void *data;
  int ndim;
  int64_t sizes[4];
  int64_t strides[4];
};

/// Eagerly free the thread-local flash scratch buffer.
/// The buffer is also freed automatically when the thread exits (RAII).
void sdpa_flash_cpu_free_scratch();

status_t sdpa_flash_cpu_run_internal(
  const sdpa_flash_cpu_tensor_view &output,
  const sdpa_flash_cpu_tensor_view &query,
  const sdpa_flash_cpu_tensor_view &key,
  const sdpa_flash_cpu_tensor_view &value, double dropout_p, bool is_causal,
  const sdpa_flash_cpu_mask_view *mask, const double *scale_opt,
  data_type_t qkv_dt, data_type_t mask_dtype,
  int num_threads = 0);

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_SDPA_FLASH_CPU_HPP
