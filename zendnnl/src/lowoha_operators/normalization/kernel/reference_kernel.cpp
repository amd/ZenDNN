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
#include "lowoha_operators/common/omp_thread_control.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

namespace zendnnl {
namespace lowoha {
namespace normalization {

using zendnnl::common::float16_t;

// =============================================================================
// FP16-rounded scalar helper.
//
// Each value is round-tripped float -> float16 -> float so the running
// accumulator carries only the precision of an FP16 value. Mirrors what
// `_mm512_fmadd_ph` does in the SIMD kernel: every FMA result is rounded
// to FP16 before the next step.
// =============================================================================
static inline float to_f16_rounded(float v) {
  return float16_t::f16_to_f32_val(float16_t::f32_to_f16_val(v));
}

// =============================================================================
// SIMD-equivalent lane-mapping helper for the F16-accum reference path.
//
// The production AVX512-FP16 kernels for LayerNorm and (plain + fused)
// RMSNorm use 4 parallel __m512h accumulators of 32 lanes each (128
// parallel f16 partial sums) in their main 4x32-lane unrolled loop, then
// fall back to a single 32-lane cleanup loop + a masked-tail load that
// both write into acc0's lanes 0..31. After the row finishes, the SIMD
// kernel widens each accumulator to FP32 once via reduce_add_ph_to_fp32
// and sums the FP32 partials.
//
// To bit-match this, the reference kernel must:
//   1) accumulate per-lane in FP16 (with per-step f16 rounding) using the
//      same lane-mapping rule as the SIMD kernel, and
//   2) widen-then-sum in FP32 at the end of the row (no per-step f16
//      rounding for the cross-lane combine).
//
// A naive single-accumulator FP16 reduction silently saturates for long
// rows (e.g., norm_size >= 2K with sum_sq above ~1024 the FP16 spacing
// exceeds 0.125, so individual increments round to zero). The 128-lane
// layout keeps each lane's running sum small and avoids that loss.
//
// Lane-mapping rule (matches the SIMD kernel exactly):
//   - i < vec128:  lane = i % 128                  (4-acc x 32-lane main)
//   - i >= vec128: lane = (i - vec128) % 32        (32-lane cleanup + tail
//                                                   both write into acc0)
// =============================================================================

// Number of parallel f16 lanes in the SIMD main loop (4 acc x 32 lanes).
static constexpr int F16_ACCUM_LANES = 128;

static inline int simd_accum_lane(uint64_t i, uint64_t vec128) {
  if (i < vec128) {
    return static_cast<int>(i % F16_ACCUM_LANES);
  }
  return static_cast<int>((i - vec128) % 32);
}

static inline uint64_t simd_vec128(uint64_t norm_size) {
  return norm_size & ~127ULL;
}

// Sum the per-lane f16 partial accumulators in FP32 (mirrors
// reduce_add_ph_to_fp32 + the cross-accumulator add in the SIMD kernel).
static inline float reduce_lanes_f32(const float (&lanes)[F16_ACCUM_LANES]) {
  float total = 0.0f;
  for (int j = 0; j < F16_ACCUM_LANES; ++j) {
    total += lanes[j];
  }
  return total;
}

// =============================================================================
// Scalar load/store helpers — convert any supported dtype to/from FP32.
//
// These helpers narrow/widen at the load/store boundaries so the rest of the
// kernel can work with plain `float`. They are tiny inline switches that the
// compiler hoists outside the inner loops since the dtype is loop-invariant
// per call site. The arithmetic precision between load and store is chosen
// by each kernel impl (FP32 by default; FP16-accum branches emulate the
// native FP16-FMA path — see the parallel-lane accumulators below).
// =============================================================================

static inline float load_scalar(const void *base, data_type_t dt, uint64_t i) {
  switch (dt) {
  case data_type_t::f32:
    return static_cast<const float *>(base)[i];
  case data_type_t::bf16:
    return bfloat16_t::bf16_to_f32_val(
             static_cast<const int16_t *>(base)[i]);
  case data_type_t::f16:
    return float16_t::f16_to_f32_val(
             static_cast<const uint16_t *>(base)[i]);
  default:
    return 0.0f;
  }
}

static inline void store_scalar(void *base, data_type_t dt,
                                uint64_t i, float v) {
  switch (dt) {
  case data_type_t::f32:
    static_cast<float *>(base)[i] = v;
    break;
  case data_type_t::bf16:
    static_cast<int16_t *>(base)[i] = bfloat16_t::f32_to_bf16_val(v);
    break;
  case data_type_t::f16:
    static_cast<uint16_t *>(base)[i] = float16_t::f32_to_f16_val(v);
    break;
  default:
    break;
  }
}

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
  const data_type_t src_dt   = params.src_dt;
  const data_type_t dst_dt   = params.dst_dt;
  const data_type_t gamma_dt = params.gamma_dt;
  const data_type_t beta_dt  = params.beta_dt;

  const bool use_f16_accum = (params.accum_type == data_type_t::f16);
  const uint64_t vec128    = simd_vec128(norm_size);

  #pragma omp parallel for num_threads(num_threads)
  for (uint64_t b = 0; b < batch; ++b) {
    const uint64_t off = b * norm_size;

    // Pass 1: compute mean and variance.
    //
    // Two formulas are in play here and we deliberately use different ones
    // in the two branches:
    //
    //   - F16-accum branch (gtest only — set by the dispatcher only just
    //     before invoking the SIMD kernel, never on the production
    //     fallback path) mirrors the SIMD kernel's single-pass layout
    //     (Σx, Σx² in one sweep, 128-lane f16 accumulators, then
    //     var = E[x²] − μ²) so the bit-match against the SIMD kernel
    //     holds.
    //   - FP32 branch is the production scalar fallback when AVX-512 is
    //     unavailable (and the path benchdnn drives when forcing the
    //     reference algorithm). Use the numerically stable two-pass
    //     formula here: pass 1 computes μ, pass 2 accumulates Σ(x − μ)².
    //     The shifted-second-moment identity that the SIMD kernel uses
    //     would suffer catastrophic cancellation on large-offset inputs
    //     here, where there is no f32 accumulator above us to absorb it.
    float mean = 0.0f;
    float var  = 0.0f;
    if (use_f16_accum) {
      float sum_lane[F16_ACCUM_LANES] = {0};
      float sq_lane [F16_ACCUM_LANES] = {0};
      for (uint64_t i = 0; i < norm_size; ++i) {
        float x = to_f16_rounded(load_scalar(input, src_dt, off + i));
        const int lane = simd_accum_lane(i, vec128);
        // Use std::fmaf for x*x + acc to mirror _mm512_fmadd_ph semantics
        // (single rounding for the multiply-add) — bit-equivalent to embag's
        // reference, which also uses std::fmaf to mirror its SIMD FMA.
        sum_lane[lane] = to_f16_rounded(sum_lane[lane] + x);
        sq_lane [lane] = to_f16_rounded(std::fmaf(x, x, sq_lane[lane]));
      }
      const float total_sum = reduce_lanes_f32(sum_lane);
      const float total_sq  = reduce_lanes_f32(sq_lane);
      mean = total_sum / static_cast<float>(norm_size);
      var  = std::max(0.0f,
                      total_sq / static_cast<float>(norm_size) - mean * mean);
    }
    else {
      float sum = 0.0f;
      for (uint64_t i = 0; i < norm_size; ++i) {
        sum += load_scalar(input, src_dt, off + i);
      }
      mean = sum / static_cast<float>(norm_size);

      float sum_sq_dev = 0.0f;
      for (uint64_t i = 0; i < norm_size; ++i) {
        const float d = load_scalar(input, src_dt, off + i) - mean;
        sum_sq_dev += d * d;
      }
      var = sum_sq_dev / static_cast<float>(norm_size);
    }

    float inv_std = 1.0f / std::sqrt(var + eps);
    if (use_f16_accum) {
      // The SIMD kernel broadcasts mean / inv_std via _mm512_set1_ph,
      // i.e. both are FP16-rounded once before pass 2.
      mean    = to_f16_rounded(mean);
      inv_std = to_f16_rounded(inv_std);
    }

    // Pass 2: normalize, apply optional gamma/beta. F16-accum mirrors the
    // SIMD kernel's operation order exactly so bit-equal comparisons hold:
    //   * with use_scale=true, combine gamma * inv_std first to FP16
    //     (matches _mm512_mul_ph(g, inv_std_h)), then multiply (x - mean).
    //     With use_shift=true the multiply+add becomes a single-rounded
    //     fmaf, matching _mm512_fmadd_ph(d, g_eff, beta).
    //   * with use_scale=false, multiply (x - mean) by inv_std directly.
    for (uint64_t i = 0; i < norm_size; ++i) {
      float x = load_scalar(input, src_dt, off + i);
      float norm_val;
      if (use_f16_accum) {
        // Mirror the SIMD load boundary: f16x32_load_typed<float>(...) does
        // _mm512_cvtps_ph, so any f32 gamma/beta is rounded to FP16 before
        // entering the multiply / FMA. Pre-round here so the reference sees
        // the same operand precision as the SIMD kernel.
        float xr = to_f16_rounded(x);
        float d  = to_f16_rounded(xr - mean);
        if (params.use_scale) {
          float g     = to_f16_rounded(load_scalar(gamma, gamma_dt, i));
          float g_eff = to_f16_rounded(g * inv_std);
          if (params.use_shift) {
            float bt = to_f16_rounded(load_scalar(beta, beta_dt, i));
            norm_val = to_f16_rounded(std::fmaf(d, g_eff, bt));
          }
          else {
            norm_val = to_f16_rounded(d * g_eff);
          }
        }
        else {
          norm_val = to_f16_rounded(d * inv_std);
          if (params.use_shift) {
            float bt = to_f16_rounded(load_scalar(beta, beta_dt, i));
            norm_val = to_f16_rounded(norm_val + bt);
          }
        }
      }
      else {
        norm_val = (x - mean) * inv_std;
        if (params.use_scale) {
          norm_val *= load_scalar(gamma, gamma_dt, i);
        }
        if (params.use_shift) {
          norm_val += load_scalar(beta, beta_dt, i);
        }
      }
      store_scalar(output, dst_dt, off + i, norm_val);
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
  const data_type_t src_dt   = params.src_dt;
  const data_type_t dst_dt   = params.dst_dt;
  const data_type_t gamma_dt = params.gamma_dt;
  const data_type_t beta_dt  = params.beta_dt;

  #pragma omp parallel for collapse(2) num_threads(num_threads)
  for (uint64_t n = 0; n < N; ++n) {
    for (uint64_t c = 0; c < C; ++c) {
      const uint64_t off = (n * C + c) * spatial_size;
      float inv_std = 1.0f / std::sqrt(running_var[c] + eps);
      float m       = running_mean[c];
      float g       = params.use_scale ? load_scalar(gamma, gamma_dt, c) : 0.0f;
      float bt      = params.use_shift ? load_scalar(beta,  beta_dt,  c) : 0.0f;

      for (uint64_t s = 0; s < spatial_size; ++s) {
        float x = load_scalar(input, src_dt, off + s);
        float norm_val = (x - m) * inv_std;
        if (params.use_scale) {
          norm_val *= g;
        }
        if (params.use_shift) {
          norm_val += bt;
        }
        store_scalar(output, dst_dt, off + s, norm_val);
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
  const data_type_t src_dt   = params.src_dt;
  const data_type_t dst_dt   = params.dst_dt;
  const data_type_t gamma_dt = params.gamma_dt;

  // F16 accumulation is engaged only when the production dispatch ran the
  // AVX512-FP16 kernel for this call. The gtest reference path uses this
  // to bit-match (within NORM_F16_TOL) the per-step FP16 rounding of
  // _mm512_fmadd_ph; the production path itself never runs through here.
  const bool use_f16_accum = (params.accum_type == data_type_t::f16);
  const uint64_t vec128    = simd_vec128(norm_size);

  #pragma omp parallel for num_threads(num_threads)
  for (uint64_t b = 0; b < batch; ++b) {
    const uint64_t off = b * norm_size;

    float sum_sq = 0.0f;
    if (use_f16_accum) {
      // Mirror the SIMD kernel's 128-lane f16 accumulator layout, then
      // widen-and-sum in FP32 (matches reduce_add_ph_to_fp32). A single
      // f16 accumulator would saturate for long rows.
      float sq_lane[F16_ACCUM_LANES] = {0};
      for (uint64_t i = 0; i < norm_size; ++i) {
        float x = to_f16_rounded(load_scalar(input, src_dt, off + i));
        const int lane = simd_accum_lane(i, vec128);
        // std::fmaf: single rounding for x*x + acc, matches _mm512_fmadd_ph.
        sq_lane[lane] = to_f16_rounded(std::fmaf(x, x, sq_lane[lane]));
      }
      sum_sq = reduce_lanes_f32(sq_lane);
    }
    else {
      for (uint64_t i = 0; i < norm_size; ++i) {
        float x = load_scalar(input, src_dt, off + i);
        sum_sq += x * x;
      }
    }
    float inv_rms = 1.0f / std::sqrt(
                      sum_sq / static_cast<float>(norm_size) + eps);
    // The SIMD kernel broadcasts inv_rms via _mm512_set1_ph((_Float16)v),
    // i.e. it is FP16-rounded once before entering pass 2.
    if (use_f16_accum) {
      inv_rms = to_f16_rounded(inv_rms);
    }

    // Pass 2 mirrors the SIMD operation order so bit-equal comparisons hold:
    // with use_scale, combine gamma * inv_rms first to FP16 (matches
    // _mm512_mul_ph(g, inv_rms_h)), then multiply x. f32 gamma is pre-rounded
    // to FP16 to mirror the load boundary (_mm512_cvtps_ph) in
    // f16x32_load_typed<float>.
    for (uint64_t i = 0; i < norm_size; ++i) {
      float x = load_scalar(input, src_dt, off + i);
      float norm_val;
      if (use_f16_accum) {
        float xr = to_f16_rounded(x);
        if (params.use_scale) {
          float g     = to_f16_rounded(load_scalar(gamma, gamma_dt, i));
          float g_eff = to_f16_rounded(g * inv_rms);
          norm_val    = to_f16_rounded(xr * g_eff);
        }
        else {
          norm_val = to_f16_rounded(xr * inv_rms);
        }
      }
      else {
        norm_val = x * inv_rms;
        if (params.use_scale) {
          norm_val *= load_scalar(gamma, gamma_dt, i);
        }
      }
      store_scalar(output, dst_dt, off + i, norm_val);
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
  const data_type_t src_dt   = params.src_dt;   // residual dtype matches src_dt
  const data_type_t dst_dt   = params.dst_dt;
  const data_type_t gamma_dt = params.gamma_dt;

  const bool use_f16_accum = (params.accum_type == data_type_t::f16);
  const uint64_t vec128    = simd_vec128(norm_size);

  #pragma omp parallel for num_threads(num_threads)
  for (uint64_t b = 0; b < batch; ++b) {
    const uint64_t off = b * norm_size;

    // Pass 1: residual[i] += input[i] and accumulate sum-of-squares.
    // In F16-accum mode the in-place add is rounded to FP16 (residual is
    // already f16 storage by precondition), and sum-of-squares accumulates
    // per-lane in a 128-lane f16 array (mirrors the SIMD kernel's 4 acc x
    // 32 lanes) before widening to FP32 for the final reduction.
    float sum_sq = 0.0f;
    if (use_f16_accum) {
      float sq_lane[F16_ACCUM_LANES] = {0};
      for (uint64_t i = 0; i < norm_size; ++i) {
        float inp = load_scalar(input,    src_dt, off + i);
        float res = load_scalar(residual, src_dt, off + i);
        float sum = to_f16_rounded(res + inp);
        const int lane = simd_accum_lane(i, vec128);
        // std::fmaf: single rounding for sum*sum + acc, matches _mm512_fmadd_ph.
        sq_lane[lane] = to_f16_rounded(std::fmaf(sum, sum, sq_lane[lane]));
        store_scalar(residual, src_dt, off + i, sum);
      }
      sum_sq = reduce_lanes_f32(sq_lane);
    }
    else {
      for (uint64_t i = 0; i < norm_size; ++i) {
        float inp = load_scalar(input,    src_dt, off + i);
        float res = load_scalar(residual, src_dt, off + i);
        float sum = res + inp;
        sum_sq += sum * sum;
        store_scalar(residual, src_dt, off + i, sum);
      }
    }

    float inv_rms = 1.0f / std::sqrt(
                      sum_sq / static_cast<float>(norm_size) + eps);
    if (use_f16_accum) {
      inv_rms = to_f16_rounded(inv_rms);
    }

    // Pass 2 mirrors the SIMD operation order so bit-equal comparisons hold:
    // with use_scale, combine gamma * inv_rms first to FP16, then multiply
    // the residual sample. Both r and gamma are pre-rounded for symmetry
    // with the LayerNorm/RMSNorm references; the dispatch is currently
    // strict all-f16 so these rounds are no-ops, but they keep the kernel
    // correct if the upstream gate ever loosens.
    for (uint64_t i = 0; i < norm_size; ++i) {
      float r = load_scalar(residual, src_dt, off + i);
      float norm_val;
      if (use_f16_accum) {
        float rr = to_f16_rounded(r);
        if (params.use_scale) {
          float g     = to_f16_rounded(load_scalar(gamma, gamma_dt, i));
          float g_eff = to_f16_rounded(g * inv_rms);
          norm_val    = to_f16_rounded(rr * g_eff);
        }
        else {
          norm_val = to_f16_rounded(rr * inv_rms);
        }
      }
      else {
        norm_val = r * inv_rms;
        if (params.use_scale) {
          norm_val *= load_scalar(gamma, gamma_dt, i);
        }
      }
      store_scalar(output, dst_dt, off + i, norm_val);
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

