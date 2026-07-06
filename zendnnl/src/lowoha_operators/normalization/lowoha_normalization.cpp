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

#include "lowoha_normalization.hpp"
#include "lowoha_operators/normalization/kernel/reference_kernel.hpp"
#include "lowoha_operators/normalization/kernel/rmsnorm_avx512_kernel.hpp"
#include "lowoha_operators/normalization/kernel/layernorm_avx512_kernel.hpp"
#include "lowoha_operators/normalization/kernel/rmsnorm_avx512_fp16_kernel.hpp"
#include "lowoha_operators/normalization/kernel/layernorm_avx512_fp16_kernel.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"

namespace zendnnl {
namespace lowoha {
namespace normalization {

using namespace zendnnl::common;

status_t normalization_kernel_wrapper(
  const void *input,
  void *output,
  const void *gamma,
  const void *beta,
  const void *running_mean,
  const void *running_var,
  void       *residual,
  norm_params &params
) {

  // Honor params.num_threads for all AVX-512 fast-path kernels. They use
  // bare `#pragma omp parallel for schedule(static)` with no num_threads
  // clause, so they would otherwise pick up omp_get_max_threads() at the
  // moment of the call. The reference kernel resolves num_threads inside
  // its own dispatch (reference_kernel.cpp:539), but applying the guard
  // here is harmless for that path. 0 means "use the current ICV".
  const int32_t omp_mt = thread_guard::max_threads();
  const int32_t num_threads     = resolve_num_threads(params.num_threads, omp_mt);
  thread_guard tg(num_threads, omp_mt);

  const bool has_avx512f = zendnnl_platform_info().get_avx512f_status();

  // AVX512-FP16 kernel eligibility.
  //
  //   src_or_dst_f16 : at least one of src/dst is f16 — only then is a
  //                    pure FP16-FMA inner loop worth running. f32/f32
  //                    rows stay on the FP32-accumulating AVX-512 kernel.
  //   gamma in {f16, f32} : the kernel multiplies by gamma in __m512h.
  //                    f16 gamma is loaded directly; f32 gamma is narrowed
  //                    once at the load boundary (one vcvtps2ph per 32-lane
  //                    block), which is amortized into the multi-K-lane
  //                    main loop. bf16 gamma is rejected (would need a
  //                    different boundary helper, no real workload demand).
  //   beta            : same rule as gamma when use_shift = true.
  //
  // Plain RMS_NORM and LAYER_NORM accept three (src_dt, dst_dt) combos:
  //   (f16, f16), (f16, f32), (f32, f16).
  const bool src_or_dst_f16   = (params.src_dt == data_type_t::f16 ||
                                 params.dst_dt == data_type_t::f16);
  const bool src_dst_f16_path = src_or_dst_f16 &&
                                (params.src_dt == data_type_t::f16 ||
                                 params.src_dt == data_type_t::f32) &&
                                (params.dst_dt == data_type_t::f16 ||
                                 params.dst_dt == data_type_t::f32);
  const bool gamma_f16_fma_ok = (params.gamma_dt == data_type_t::f16 ||
                                 params.gamma_dt == data_type_t::f32);
  const bool beta_f16_fma_ok  = (params.beta_dt == data_type_t::f16 ||
                                 params.beta_dt == data_type_t::f32);
  const bool rms_f16_fma_eligible = src_dst_f16_path && gamma_f16_fma_ok;
  const bool ln_f16_fma_eligible  = rms_f16_fma_eligible &&
                                    (!params.use_shift || beta_f16_fma_ok);

  // Record the FMA precision the chosen kernel will use so the reference
  // kernel can bit-match it during gtest comparisons. Default to f32
  // before the dispatch decisions below; the F16-FMA branches overwrite
  // it just before invoking their compute path.
  params.accum_type = data_type_t::f32;

  // Plain RMS_NORM with mixed-dtype (f16/f32, f32/f16) eligibility.
  if (can_use_f16_fma_kernel() && rms_f16_fma_eligible &&
      params.norm_type == norm_type_t::RMS_NORM) {
    log_info("Using AVX512-FP16 kernel for ",
             norm_type_to_str(params.norm_type));

    params.accum_type = data_type_t::f16;
    status_t status = rms_norm_avx512_fp16(input, output, residual, gamma,
                                           params);
    if (status == status_t::success) {
      return status;
    }
    if (status != status_t::isa_unsupported &&
        status != status_t::unimplemented) {
      log_error(norm_type_to_str(params.norm_type), " kernel failed");
      return status;
    }
    // Fell through; reset accum_type so the next kernel sees the actual
    // precision the reference will need to bit-match.
    params.accum_type = data_type_t::f32;
  }

#if defined(ZENDNNL_FUSED_ADD_RMS_F16)
  // Opt-in native AVX512-FP16 fast path for FUSED_ADD_RMS_NORM (built with
  // -DZENDNNL_FUSED_ADD_RMS_F16=ON). Kept behind the flag because the in-place
  // residual add plus F16 sum-of-squares accumulation loses too much precision
  // vs. the FP32-accumulating AVX-512 kernel. Strict all-f16 gate: the residual
  // buffer aliases src and is read-modify-written in place, so it must share the
  // f16 storage layout. Mixed-dtype fused-add falls through to the FP32 path.
  const bool fused_add_all_f16 = (params.src_dt   == data_type_t::f16 &&
                                  params.dst_dt   == data_type_t::f16 &&
                                  params.gamma_dt == data_type_t::f16);
  if (can_use_f16_fma_kernel() && fused_add_all_f16 &&
      params.norm_type == norm_type_t::FUSED_ADD_RMS_NORM) {
    log_info("Using AVX512-FP16 kernel for ",
             norm_type_to_str(params.norm_type));

    params.accum_type = data_type_t::f16;
    status_t status = rms_norm_avx512_fp16(input, output, residual, gamma,
                                           params);
    if (status == status_t::success) {
      return status;
    }
    if (status != status_t::isa_unsupported &&
        status != status_t::unimplemented) {
      log_error(norm_type_to_str(params.norm_type), " kernel failed");
      return status;
    }
    // Fell through; reset accum_type so the AVX-512 FP32 path sees f32.
    params.accum_type = data_type_t::f32;
  }
#endif

  // Plain LAYER_NORM with mixed-dtype (f16/f32, f32/f16) eligibility.
  if (can_use_f16_fma_kernel() && ln_f16_fma_eligible &&
      params.norm_type == norm_type_t::LAYER_NORM) {
    log_info("Using AVX512-FP16 kernel for ",
             norm_type_to_str(params.norm_type));

    params.accum_type = data_type_t::f16;
    status_t status = layer_norm_avx512_fp16(input, output, gamma, beta,
                      params);
    if (status == status_t::success) {
      return status;
    }
    if (status != status_t::isa_unsupported &&
        status != status_t::unimplemented) {
      log_error(norm_type_to_str(params.norm_type), " kernel failed");
      return status;
    }
    // Fell through; reset accum_type so the AVX-512 FP32 path sees f32.
    params.accum_type = data_type_t::f32;
  }

  if (has_avx512f && (params.norm_type == norm_type_t::RMS_NORM ||
                      params.norm_type == norm_type_t::FUSED_ADD_RMS_NORM)) {
    log_info("Using AVX512 kernel for ", norm_type_to_str(params.norm_type));
    params.accum_type = data_type_t::f32;
    status_t status = rms_norm_avx512(input, output, residual, gamma, params);
    if (status != status_t::success) {
      log_error(norm_type_to_str(params.norm_type), " kernel failed");
    }
    return status;
  }

  if (has_avx512f && params.norm_type == norm_type_t::LAYER_NORM) {
    log_info("Using AVX512 kernel for ", norm_type_to_str(params.norm_type));
    params.accum_type = data_type_t::f32;
    status_t status = layer_norm_avx512(input, output, gamma, beta, params);
    if (status != status_t::success) {
      log_error(norm_type_to_str(params.norm_type), " kernel failed");
    }
    return status;
  }

  log_info("Using reference kernel for ", norm_type_to_str(params.norm_type));
  status_t status = normalization_reference_wrapper(
                      input, output, gamma, beta,
                      running_mean, running_var, residual, params);

  if (status != status_t::success) {
    log_error("Normalization kernel failed for ",
              norm_type_to_str(params.norm_type));
  }

  return status;
}

status_t normalization_direct(
  const void *input,
  void *output,
  const void *gamma,
  const void *beta,
  const void *running_mean,
  const void *running_var,
  void       *residual,
  norm_params &params
) {
  // Create profiler instance for timing
  zendnnl::profile::profiler_t profiler;
  bool is_profile = is_profile_enabled();
  if (is_profile) {
    profiler.tbp_start();
  }

  // F16 ISA check — requires AVX512-FP16 (CPUID leaf 7, sub 0, EDX bit 23).
  // Performed unconditionally (independent of ZENDNNL_DIAGNOSTICS_ENABLE)
  // since dispatching a kernel without the required ISA causes SIGILL.
  //
  // Only checks dtype fields whose buffer the kernel will actually read or
  // write: src and dst are always touched; gamma is read iff use_scale is
  // true; beta is read iff use_shift is true AND norm_type uses beta
  // (RMS_NORM and FUSED_ADD_RMS_NORM ignore beta regardless of use_shift).
  // This avoids spurious isa_unsupported failures on non-AVX512-FP16
  // hardware when an unused dtype field happens to be f16.
  const bool uses_gamma = params.use_scale;
  const bool uses_beta  = params.use_shift &&
                          params.norm_type != norm_type_t::RMS_NORM &&
                          params.norm_type != norm_type_t::FUSED_ADD_RMS_NORM;
  [[maybe_unused]] const bool is_f16 = (params.src_dt == data_type_t::f16 ||
                       params.dst_dt == data_type_t::f16 ||
                       (uses_gamma && params.gamma_dt == data_type_t::f16) ||
                       (uses_beta  && params.beta_dt  == data_type_t::f16));
  // When ZENDNNL_NATIVE_F32_ACCUM forces the FP32-accumulating kernel, f16
  // storage is handled via F16C convert (_mm512_cvtph_ps), which any AVX-512
  // host has. Only require AVX-512-FP16 when the F16-FMA fast path could
  // actually be selected.
#if !defined(ZENDNNL_NATIVE_F32_ACCUM)
  if (is_f16 && !zendnnl_platform_info().get_avx512_f16_status()) {
    log_error("F16 data type is not supported on this platform "
              "(requires AVX512-FP16 ISA).");
    return status_t::isa_unsupported;
  }
#endif

  // Reject f16<->bf16 cross-mixing between src and dst unconditionally.
  // The AVX-512 load/store helpers support both dtypes individually, so an
  // unguarded mixed call would silently run with semantically wrong
  // conversions (one end interpreted as f16 storage, the other as bf16).
  // The validate_normalization_inputs path also enforces this, but that
  // path only runs when ZENDNNL_DIAGNOSTICS_ENABLE=1, so the gate must
  // also live here on the production hot path.
  const bool src_is_f16  = (params.src_dt == data_type_t::f16);
  const bool src_is_bf16 = (params.src_dt == data_type_t::bf16);
  const bool dst_is_f16  = (params.dst_dt == data_type_t::f16);
  const bool dst_is_bf16 = (params.dst_dt == data_type_t::bf16);
  if ((src_is_f16 && dst_is_bf16) || (src_is_bf16 && dst_is_f16)) {
    log_error("Normalization: f16/bf16 cross-mixing between src and dst "
              "is not supported");
    return status_t::failure;
  }

  // Reject FUSED_ADD_RMS_NORM with a null residual: the kernels treat it as
  // plain RMS_NORM and silently drop the residual-add step. Diagnostics-gated
  // validation catches this, but not in production builds.
  if (params.norm_type == norm_type_t::FUSED_ADD_RMS_NORM && !residual) {
    log_error("Normalization: FUSED_ADD_RMS_NORM requires a non-null "
              "residual buffer");
    return status_t::failure;
  }

  // Validate inputs only when ZENDNNL_DIAGNOSTICS_ENABLE=1. In production this
  // resolves to a single predicted-not-taken branch, skipping the full
  // validation path (null-pointer checks, dimension checks, and
  // quantization-parameter validation).
  status_t status = zendnnl::common::op_instrumentation::validate([&]() {
    return validate_normalization_inputs(input, output, gamma, beta,
                                         running_mean, running_var, residual, params);
  });
  if (status != status_t::success) {
    return status;
  }

  // Execute normalization
  status_t kernel_status = normalization_kernel_wrapper(
                             input, output, gamma, beta,
                             running_mean, running_var, residual, params);

  if (is_profile) {
    profiler.tbp_stop();
  }

  if (kernel_status != status_t::success) {
    return kernel_status;
  }

  if (apilog_info_enabled() || is_profile) {
    [[maybe_unused]] std::ostringstream ss;
    ss << "LOWOHA " << norm_type_to_str(params.norm_type)
       << ": batch=" << params.batch
       << ", norm_size=" << params.norm_size
       << ", num_channels=" << params.num_channels
       << ", epsilon=" << params.epsilon
       << ", use_scale=" << (params.use_scale ? "true" : "false")
       << ", use_shift=" << (params.use_shift ? "true" : "false")
       << ", src_dt=" << dtype_info(params.src_dt)
       << ", dst_dt=" << dtype_info(params.dst_dt)
       << ", gamma_dt=" << dtype_info(params.gamma_dt)
       << ", beta_dt=" << dtype_info(params.beta_dt);

    apilog_info(ss.str());
    if (is_profile) {
      profilelog_verbose(ss.str(), ", time=",
                         profiler.tbp_elapsedtime(),
                         profiler.get_res_str());
    }
  }

  return status_t::success;
}

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

