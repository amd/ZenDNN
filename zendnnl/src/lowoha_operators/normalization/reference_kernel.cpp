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
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <omp.h>

namespace zendnnl {
namespace lowoha {
namespace normalization {

// ============================================================================
// LayerNorm  –  FP32 reference
//
// For each sample (row) in the batch:
//   mean  = (1/N) * sum(x_i)
//   var   = (1/N) * sum((x_i - mean)^2)
//   y_i   = gamma[i] * (x_i - mean) / sqrt(var + eps) + beta[i]
//
// Where N = norm_size (product of the last norm_ndims dimensions).
// ============================================================================
static void layer_norm_fp32_impl(
  const float *input,
  float *output,
  const float *gamma,
  const float *beta,
  const norm_params &params,
  int num_threads
) {
  const uint64_t batch     = params.batch;
  const uint64_t norm_size = params.norm_size;
  const float    eps       = params.epsilon;

  #pragma omp parallel for num_threads(num_threads)
  for (uint64_t b = 0; b < batch; ++b) {
    const float *x = input  + b * norm_size;
    float       *y = output + b * norm_size;

    // Compute mean
    float mean = 0.0f;
    for (uint64_t i = 0; i < norm_size; ++i) {
      mean += x[i];
    }
    mean /= static_cast<float>(norm_size);

    // Compute variance
    float var = 0.0f;
    for (uint64_t i = 0; i < norm_size; ++i) {
      float diff = x[i] - mean;
      var += diff * diff;
    }
    var /= static_cast<float>(norm_size);

    // Normalize
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (uint64_t i = 0; i < norm_size; ++i) {
      float norm_val = (x[i] - mean) * inv_std;
      if (params.use_scale) {
        norm_val *= gamma[i];
      }
      if (params.use_shift) {
        norm_val += beta[i];
      }
      y[i] = norm_val;
    }
  }
}

// ============================================================================
// RMSNorm  –  FP32 reference
//
// For each sample (row) in the batch:
//   rms   = sqrt( (1/N) * sum(x_i^2) + eps )
//   y_i   = gamma[i] * x_i / rms
//
// Where N = norm_size.
// ============================================================================
static void rms_norm_fp32_impl(
  const float *input,
  float *output,
  const float *gamma,
  const norm_params &params,
  int num_threads
) {
  const uint64_t batch     = params.batch;
  const uint64_t norm_size = params.norm_size;
  const float    eps       = params.epsilon;

  #pragma omp parallel for num_threads(num_threads)
  for (uint64_t b = 0; b < batch; ++b) {
    const float *x = input  + b * norm_size;
    float       *y = output + b * norm_size;

    // Compute mean of squares
    float sum_sq = 0.0f;
    for (uint64_t i = 0; i < norm_size; ++i) {
      sum_sq += x[i] * x[i];
    }
    float rms = std::sqrt(sum_sq / static_cast<float>(norm_size) + eps);
    float inv_rms = 1.0f / rms;

    // Normalize
    for (uint64_t i = 0; i < norm_size; ++i) {
      float norm_val = x[i] * inv_rms;
      if (params.use_scale) {
        norm_val *= gamma[i];
      }
      y[i] = norm_val;
    }
  }
}

// ============================================================================
// BatchNorm  –  FP32 reference  (inference-only)
//
// Uses pre-computed running statistics from training.
// For each channel c across the batch and spatial dimensions:
//   y[n,c,s] = gamma[c] * (x[n,c,s] - running_mean[c])
//              / sqrt(running_var[c] + eps) + beta[c]
//
// Input layout:  [N, C, spatial...]  (contiguous, row-major)
// ============================================================================
static void batch_norm_fp32_impl(
  const float *input,
  float *output,
  const float *gamma,
  const float *beta,
  const float *running_mean,
  const float *running_var,
  const norm_params &params,
  int num_threads
) {
  const uint64_t N            = params.batch;         // batch size
  const uint64_t C            = params.num_channels;  // channels
  const uint64_t spatial_size = params.norm_size;      // product of spatial dims
  const float    eps          = params.epsilon;

  #pragma omp parallel for collapse(2) num_threads(num_threads)
  for (uint64_t n = 0; n < N; ++n) {
    for (uint64_t c = 0; c < C; ++c) {
      const float *x     = input  + (n * C + c) * spatial_size;
      float       *y_out = output + (n * C + c) * spatial_size;

      float inv_std = 1.0f / std::sqrt(running_var[c] + eps);
      float m       = running_mean[c];

      for (uint64_t s = 0; s < spatial_size; ++s) {
        float norm_val = (x[s] - m) * inv_std;
        if (params.use_scale) {
          norm_val *= gamma[c];
        }
        if (params.use_shift) {
          norm_val += beta[c];
        }
        y_out[s] = norm_val;
      }
    }
  }
}

// ============================================================================
// Reference entry point – dispatches on data type and norm type (inference-only)
// ============================================================================
status_t normalization_reference_wrapper(
  const void *input,
  void *output,
  const void *gamma,
  const void *beta,
  const void *running_mean,
  const void *running_var,
  norm_params &params
) {
  // Reference kernel only supports f32 for both src and dst.
  if (params.src_dt != data_type_t::f32 || params.dst_dt != data_type_t::f32) {
    log_error("Normalization Reference: Unsupported data type combination "
              "(src_dt=", static_cast<int>(params.src_dt),
              ", dst_dt=", static_cast<int>(params.dst_dt),
              "). Reference kernel supports f32 only.");
    return status_t::failure;
  }

  const int num_threads = params.num_threads > 0
                          ? static_cast<int>(params.num_threads)
                          : omp_get_max_threads();

  // ---- FP32 path ----
  {
    const float *gamma_f32 = static_cast<const float *>(gamma);
    const float *beta_f32  = static_cast<const float *>(beta);

    switch (params.norm_type) {
    case norm_type_t::LAYER_NORM:
      layer_norm_fp32_impl(
        static_cast<const float *>(input),
        static_cast<float *>(output),
        gamma_f32, beta_f32,
        params, num_threads);
      return status_t::success;

    case norm_type_t::RMS_NORM:
      rms_norm_fp32_impl(
        static_cast<const float *>(input),
        static_cast<float *>(output),
        gamma_f32,
        params, num_threads);
      return status_t::success;

    case norm_type_t::BATCH_NORM:
      batch_norm_fp32_impl(
        static_cast<const float *>(input),
        static_cast<float *>(output),
        gamma_f32, beta_f32,
        static_cast<const float *>(running_mean),
        static_cast<const float *>(running_var),
        params, num_threads);
      return status_t::success;

    default:
      log_error("Normalization Reference: Unknown norm type");
      return status_t::failure;
    }
  }
}

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

