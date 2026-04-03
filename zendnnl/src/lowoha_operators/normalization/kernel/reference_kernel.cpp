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
#include "lowoha_operators/common/omp_thread_control.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

namespace zendnnl {
namespace lowoha {
namespace normalization {

// ============================================================================
// LayerNorm
//
// For each sample (row) in the batch:
//   mean  = (1/N) * sum(x_i)
//   var   = (1/N) * sum((x_i - mean)^2)
//   y_i   = gamma[i] * (x_i - mean) / sqrt(var + eps) + beta[i]
//
// Where N = norm_size (product of the last norm_ndims dimensions).
// ============================================================================
static void layer_norm_impl(
  const void  *input,
  void        *output,
  const void  *gamma,
  const void  *beta,
  const norm_params &params,
  int num_threads
) {
  const uint64_t batch      = params.batch;
  const uint64_t norm_size  = params.norm_size;
  const float    eps        = params.epsilon;
  const bool     src_bf16   = (params.src_dt == data_type_t::bf16);
  const bool     dst_bf16   = (params.dst_dt == data_type_t::bf16);
  const bool     gamma_bf16 = (params.gamma_dt == data_type_t::bf16);
  const bool     beta_bf16  = (params.beta_dt  == data_type_t::bf16);

  const float   *in_f32   = static_cast<const float *>(input);
  const int16_t *in_bf16  = static_cast<const int16_t *>(input);
  float         *out_f32  = static_cast<float *>(output);
  int16_t       *out_bf16 = static_cast<int16_t *>(output);

  #pragma omp parallel for num_threads(num_threads)
  for (uint64_t b = 0; b < batch; ++b) {
    const uint64_t off = b * norm_size;

    float mean = 0.0f;
    for (uint64_t i = 0; i < norm_size; ++i) {
      float x = src_bf16 ? bfloat16_t::bf16_to_f32_val(in_bf16[off + i])
                : in_f32[off + i];
      mean += x;
    }
    mean /= static_cast<float>(norm_size);

    float var = 0.0f;
    for (uint64_t i = 0; i < norm_size; ++i) {
      float x = src_bf16 ? bfloat16_t::bf16_to_f32_val(in_bf16[off + i])
                : in_f32[off + i];
      float diff = x - mean;
      var += diff * diff;
    }
    var /= static_cast<float>(norm_size);

    float inv_std = 1.0f / std::sqrt(var + eps);
    for (uint64_t i = 0; i < norm_size; ++i) {
      float x = src_bf16 ? bfloat16_t::bf16_to_f32_val(in_bf16[off + i])
                : in_f32[off + i];
      float norm_val = (x - mean) * inv_std;
      if (params.use_scale) {
        norm_val *= gamma_bf16
                    ? bfloat16_t::bf16_to_f32_val(static_cast<const int16_t *>(gamma)[i])
                    : static_cast<const float *>(gamma)[i];
      }
      if (params.use_shift) {
        norm_val += beta_bf16
                    ? bfloat16_t::bf16_to_f32_val(static_cast<const int16_t *>(beta)[i])
                    : static_cast<const float *>(beta)[i];
      }

      if (dst_bf16) {
        out_bf16[off + i] = bfloat16_t::f32_to_bf16_val(norm_val);
      }
      else {
        out_f32[off + i] = norm_val;
      }
    }
  }
}

// ============================================================================
// BatchNorm
//
// Uses pre-computed running statistics from training.
// For each channel c across the batch and spatial dimensions:
//   y[n,c,s] = gamma[c] * (x[n,c,s] - running_mean[c])
//              / sqrt(running_var[c] + eps) + beta[c]
//
// Input layout:  [N, C, spatial...]  (contiguous, row-major)
// ============================================================================
static void batch_norm_impl(
  const void  *input,
  void        *output,
  const void  *gamma,
  const void  *beta,
  const float *running_mean,
  const float *running_var,
  const norm_params &params,
  int num_threads
) {
  const uint64_t N            = params.batch;
  const uint64_t C            = params.num_channels;
  const uint64_t spatial_size = params.norm_size;
  const float    eps          = params.epsilon;
  const bool     src_bf16     = (params.src_dt == data_type_t::bf16);
  const bool     dst_bf16     = (params.dst_dt == data_type_t::bf16);
  const bool     gamma_bf16   = (params.gamma_dt == data_type_t::bf16);
  const bool     beta_bf16    = (params.beta_dt  == data_type_t::bf16);

  const float   *in_f32   = static_cast<const float *>(input);
  const int16_t *in_bf16  = static_cast<const int16_t *>(input);
  float         *out_f32  = static_cast<float *>(output);
  int16_t       *out_bf16 = static_cast<int16_t *>(output);

  #pragma omp parallel for collapse(2) num_threads(num_threads)
  for (uint64_t n = 0; n < N; ++n) {
    for (uint64_t c = 0; c < C; ++c) {
      const uint64_t off = (n * C + c) * spatial_size;
      float inv_std = 1.0f / std::sqrt(running_var[c] + eps);
      float m       = running_mean[c];

      for (uint64_t s = 0; s < spatial_size; ++s) {
        float x = src_bf16 ? bfloat16_t::bf16_to_f32_val(in_bf16[off + s])
                  : in_f32[off + s];
        float norm_val = (x - m) * inv_std;
        if (params.use_scale) {
          norm_val *= gamma_bf16
                      ? bfloat16_t::bf16_to_f32_val(static_cast<const int16_t *>(gamma)[c])
                      : static_cast<const float *>(gamma)[c];
        }
        if (params.use_shift) {
          norm_val += beta_bf16
                      ? bfloat16_t::bf16_to_f32_val(static_cast<const int16_t *>(beta)[c])
                      : static_cast<const float *>(beta)[c];
        }

        if (dst_bf16) {
          out_bf16[off + s] = bfloat16_t::f32_to_bf16_val(norm_val);
        }
        else {
          out_f32[off + s] = norm_val;
        }
      }
    }
  }
}

// ============================================================================
// RMSNorm
//
// For each sample (row) in the batch:
//   rms   = sqrt( (1/N) * sum(x_i^2) + eps )
//   y_i   = gamma[i] * x_i / rms
//
// Where N = norm_size.
// ============================================================================
static void rms_norm_impl(
  const void  *input,
  void        *output,
  const void  *gamma,
  const norm_params &params,
  int num_threads
) {
  const uint64_t batch      = params.batch;
  const uint64_t norm_size  = params.norm_size;
  const float    eps        = params.epsilon;
  const bool     src_bf16   = (params.src_dt == data_type_t::bf16);
  const bool     dst_bf16   = (params.dst_dt == data_type_t::bf16);
  const bool     gamma_bf16 = (params.gamma_dt == data_type_t::bf16);

  const float   *in_f32   = static_cast<const float *>(input);
  const int16_t *in_bf16  = static_cast<const int16_t *>(input);
  float         *out_f32  = static_cast<float *>(output);
  int16_t       *out_bf16 = static_cast<int16_t *>(output);

  #pragma omp parallel for num_threads(num_threads)
  for (uint64_t b = 0; b < batch; ++b) {
    const uint64_t off = b * norm_size;

    float sum_sq = 0.0f;
    for (uint64_t i = 0; i < norm_size; ++i) {
      float x = src_bf16 ? bfloat16_t::bf16_to_f32_val(in_bf16[off + i])
                : in_f32[off + i];
      sum_sq += x * x;
    }
    float inv_rms = 1.0f / std::sqrt(
                      sum_sq / static_cast<float>(norm_size) + eps);

    for (uint64_t i = 0; i < norm_size; ++i) {
      float x = src_bf16 ? bfloat16_t::bf16_to_f32_val(in_bf16[off + i])
                : in_f32[off + i];
      float norm_val = x * inv_rms;
      if (params.use_scale) {
        norm_val *= gamma_bf16
                    ? bfloat16_t::bf16_to_f32_val(static_cast<const int16_t *>(gamma)[i])
                    : static_cast<const float *>(gamma)[i];
      }

      if (dst_bf16) {
        out_bf16[off + i] = bfloat16_t::f32_to_bf16_val(norm_val);
      }
      else {
        out_f32[off + i] = norm_val;
      }
    }
  }
}

// ============================================================================
// FusedAddRMSNorm
//
// For each sample (row) in the batch:
//   residual[i] += input[i]
//   rms   = sqrt( (1/N) * sum(residual[i]^2) + eps )
//   y_i   = gamma[i] * residual[i] / rms
// ============================================================================
static void fused_add_rms_norm_impl(
  const void  *input,
  void        *output,
  void        *residual,
  const void  *gamma,
  const norm_params &params,
  int num_threads
) {
  const uint64_t batch      = params.batch;
  const uint64_t norm_size  = params.norm_size;
  const float    eps        = params.epsilon;
  const bool     src_bf16   = (params.src_dt == data_type_t::bf16);
  const bool     dst_bf16   = (params.dst_dt == data_type_t::bf16);
  const bool     gamma_bf16 = (params.gamma_dt == data_type_t::bf16);

  const float   *in_f32    = static_cast<const float *>(input);
  const int16_t *in_bf16   = static_cast<const int16_t *>(input);
  float         *res_f32   = static_cast<float *>(residual);
  int16_t       *res_bf16  = static_cast<int16_t *>(residual);
  float         *out_f32   = static_cast<float *>(output);
  int16_t       *out_bf16  = static_cast<int16_t *>(output);

  #pragma omp parallel for num_threads(num_threads)
  for (uint64_t b = 0; b < batch; ++b) {
    const uint64_t off = b * norm_size;

    // Pass 1: residual[i] += input[i] and accumulate variance
    float sum_sq = 0.0f;
    for (uint64_t i = 0; i < norm_size; ++i) {
      float inp = src_bf16 ? bfloat16_t::bf16_to_f32_val(in_bf16[off + i])
                  : in_f32[off + i];
      float res = src_bf16 ? bfloat16_t::bf16_to_f32_val(res_bf16[off + i])
                  : res_f32[off + i];
      float sum = res + inp;
      sum_sq += sum * sum;
      if (src_bf16) {
        res_bf16[off + i] = bfloat16_t::f32_to_bf16_val(sum);
      }
      else {
        res_f32[off + i] = sum;
      }
    }

    float inv_rms = 1.0f / std::sqrt(
                      sum_sq / static_cast<float>(norm_size) + eps);

    // Pass 2: normalize updated residual and write output
    for (uint64_t i = 0; i < norm_size; ++i) {
      float r = src_bf16 ? bfloat16_t::bf16_to_f32_val(res_bf16[off + i])
                : res_f32[off + i];
      float norm_val = r * inv_rms;
      if (params.use_scale) {
        norm_val *= gamma_bf16
                    ? bfloat16_t::bf16_to_f32_val(static_cast<const int16_t *>(gamma)[i])
                    : static_cast<const float *>(gamma)[i];
      }

      if (dst_bf16) {
        out_bf16[off + i] = bfloat16_t::f32_to_bf16_val(norm_val);
      }
      else {
        out_f32[off + i] = norm_val;
      }
    }
  }
}

// ===================================================
// Reference entry point – dispatches on norm type
// ===================================================
status_t normalization_reference_wrapper(
  const void *input,
  void       *output,
  const void *gamma,
  const void *beta,
  const void *running_mean,
  const void *running_var,
  void       *residual,
  norm_params &params
) {

  const int32_t num_threads = resolve_num_threads(params.num_threads,
                                                  thread_guard::max_threads());

  switch (params.norm_type) {
  case norm_type_t::LAYER_NORM:
    layer_norm_impl(input, output, gamma, beta,
                    params, num_threads);
    return status_t::success;

  case norm_type_t::BATCH_NORM:
    batch_norm_impl(input, output, gamma, beta,
                    static_cast<const float *>(running_mean),
                    static_cast<const float *>(running_var),
                    params, num_threads);
    return status_t::success;

  case norm_type_t::RMS_NORM:
    rms_norm_impl(input, output, gamma,
                  params, num_threads);
    return status_t::success;

  case norm_type_t::FUSED_ADD_RMS_NORM:
    fused_add_rms_norm_impl(input, output, residual, gamma,
                            params, num_threads);
    return status_t::success;

  default:
    log_error("Normalization Reference: Unknown norm type");
    return status_t::failure;
  }
}

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

