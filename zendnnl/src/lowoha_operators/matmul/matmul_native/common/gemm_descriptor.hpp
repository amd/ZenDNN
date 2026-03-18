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

#ifndef MATMUL_NATIVE_GEMM_DESCRIPTOR_HPP
#define MATMUL_NATIVE_GEMM_DESCRIPTOR_HPP

#include <vector>
#include <cstddef>

#include "common/data_types.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::data_type_t;

/// Canonical description of a GEMM problem for Native kernel dispatch.
struct GemmDescriptor {
  int M, N, K;
  int lda, ldb, ldc;
  bool transA, transB;
  float alpha, beta;
  data_type_t src_dt, wei_dt, dst_dt, bias_dt;
  size_t src_elem_size, wei_elem_size, dst_elem_size;
  const void *bias;
  bool is_weights_const;
  int num_threads;

  GemmDescriptor()
    : M(0), N(0), K(0), lda(0), ldb(0), ldc(0),
      transA(false), transB(false), alpha(1.0f), beta(0.0f),
      src_dt(data_type_t::f32), wei_dt(data_type_t::f32),
      dst_dt(data_type_t::f32), bias_dt(data_type_t::none),
      src_elem_size(4), wei_elem_size(4), dst_elem_size(4),
      bias(nullptr), is_weights_const(false), num_threads(1) {}
};

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_GEMM_DESCRIPTOR_HPP
