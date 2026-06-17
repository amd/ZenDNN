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

#include "reference_kernel.hpp"
#include "common/logging.hpp"
#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "memory/memory_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

namespace zendnnl {
namespace lowoha {
namespace softmax {

using zendnnl::common::float16_t;
using zendnnl::memory::read_and_cast;

// =============================================================================
// store_scalar — narrow an FP32 result back to the storage dtype at the write
// boundary, so the kernel can compute in plain `float`. Reads use the shared
// read_and_cast<float>() helper (memory/memory_utils.hpp); there is no shared
// write counterpart, so this small inline switch mirrors the normalization
// reference kernel's store_scalar.
// =============================================================================

static inline void store_scalar(void *base, data_type_t dt,
                                uint64_t i, float v) {
  switch (dt) {
  case data_type_t::f32:
    static_cast<float *>(base)[i] = v;
    break;
  case data_type_t::bf16:
    // Write through the element's own type; the float constructor narrows.
    static_cast<bfloat16_t *>(base)[i] = bfloat16_t(v);
    break;
  case data_type_t::f16:
    static_cast<float16_t *>(base)[i] = float16_t(v);
    break;
  default:
    break;
  }
}

// Generic softmax reference implementation for all supported dtypes
// (f32, bf16, f16).
//
// All computation is performed in FP32 regardless of the storage dtype:
// - f16 has a 10-bit mantissa and max value ~65504; exp() overflows at ~11,
//   so the exp/sum/log chain must run in FP32 to stay numerically stable.
// - bf16 has a 7-bit mantissa with limited precision for accumulation.
// - f32 reads/writes pass through unchanged.
//
// read_and_cast<float>() widens each stored element to FP32 at the read
// boundary, and store_scalar narrows the FP32 result back to the storage
// dtype at the write boundary. softmin is handled by flipping the input sign
// (softmin(x) == softmax(-x)).
static void softmax_reference_impl(
  const void *input,
  void *output,
  const softmax_params &params,
  int num_threads
) {
  const uint64_t axis_size = params.axis_dim;
  const uint64_t outer_size = params.batch;
  const data_type_t dt = params.src_dt;
  const float sign = params.softmin ? -1.0f : 1.0f;

  #pragma omp parallel for num_threads(num_threads)
  for (uint64_t outer = 0; outer < outer_size; ++outer) {
    const uint64_t offset = outer * axis_size;

    // Find max for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    for (uint64_t i = 0; i < axis_size; ++i) {
      max_val = std::max(max_val,
                         sign * read_and_cast<float>(input, dt, offset + i));
    }

    // Compute exp and sum
    std::vector<float> exp_vals(axis_size);
    float sum_exp = 0.0f;
    for (uint64_t i = 0; i < axis_size; ++i) {
      exp_vals[i] = std::exp(sign * read_and_cast<float>(input, dt, offset + i)
                             - max_val);
      sum_exp += exp_vals[i];
    }

    // Normalize
    if (params.log_softmax) {
      float log_sum_exp = std::log(sum_exp);
      for (uint64_t i = 0; i < axis_size; ++i) {
        float val = sign * read_and_cast<float>(input, dt, offset + i)
                    - max_val - log_sum_exp;
        store_scalar(output, dt, offset + i, val);
      }
    }
    else {
      for (uint64_t i = 0; i < axis_size; ++i) {
        store_scalar(output, dt, offset + i, exp_vals[i] / sum_exp);
      }
    }
  }
}

status_t softmax_reference_wrapper(
  const void *input,
  void *output,
  softmax_params &params
) {
  // Calculate flattened parameters from shape
  if (params.ndims <= 0 || params.ndims > SOFTMAX_MAX_NDIMS) {
    log_error("Softmax Reference: Invalid ndims: ", params.ndims,
              " (must be 1-", SOFTMAX_MAX_NDIMS,
              "). Use setup_softmax_shape() to populate params.");
    return status_t::failure;
  }

  int normalized_axis = params.axis >= 0 ? params.axis : params.ndims +
                        params.axis;

  // Reject an out-of-range axis before it indexes params.shape below; an
  // invalid value would otherwise be an out-of-bounds read that corrupts
  // batch/axis_dim and the subsequent buffer traversal.
  if (normalized_axis < 0 || normalized_axis >= params.ndims) {
    log_error("Softmax Reference: Invalid axis: ", params.axis, " for ",
              params.ndims, "D tensor (must be in [-", params.ndims, ", ",
              params.ndims, ")).");
    return status_t::failure;
  }

  // Calculate batch as product of all dimensions except axis_dim
  params.batch = 1;
  for (int i = 0; i < params.ndims; ++i) {
    if (i != normalized_axis) {
      params.batch *= params.shape[i];
    }
  }
  params.axis_dim = params.shape[normalized_axis];

  log_info("Softmax Reference: ", params.ndims, "D tensor, flattened to batch=",
           params.batch, ", axis_dim=", params.axis_dim);

  const int32_t num_threads = resolve_num_threads(params.num_threads,
                              thread_guard::max_threads());

  // f32, bf16, and f16 all run through the same FP32-compute path; the
  // load/store helpers handle the per-dtype widen/narrow at the boundaries.
  if (params.src_dt == data_type_t::f32 ||
      params.src_dt == data_type_t::bf16 ||
      params.src_dt == data_type_t::f16) {
    softmax_reference_impl(input, output, params, num_threads);
    return status_t::success;
  }

  log_error("Softmax Reference: Unsupported data type");
  return status_t::failure;
}

} // namespace softmax
} // namespace lowoha
} // namespace zendnnl
