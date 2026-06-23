/*******************************************************************************
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

#include "lowoha_operators/reorder/prepack/lowoha_prepack.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_common.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"
#include "common/zendnnl_global.hpp"

#include <sstream>
#include <string>

namespace zendnnl {
namespace lowoha {
namespace reorder {

using namespace zendnnl::error_handling;
using namespace zendnnl::profile;
using zendnnl::common::op_instrumentation;
using zendnnl::common::is_profile_enabled;
using zendnnl::profile::profiler_t;
using zendnnl::lowoha::matmul::kernel_to_string;

namespace {

constexpr size_t kPrepackAlign = 64;

inline size_t round_up_align(size_t bytes, size_t align) {
  return (bytes + align - 1) & ~(align - 1);
}

// dtype -> short string: reuse the canonical common helper
using zendnnl::common::dtype_info;

// Params-only validation (no weights pointer needed). Used by both the
// size-query path and the data-movement paths. `caller` is the name of
// the public function being validated, so log lines name the actual
// entry point a user invoked (easier to triage from logs).
status_t validate_prepack_params(const char *caller,
                                 const prepack_params_t &params) {
  if (params.K <= 0 || params.N <= 0) {
    apilog_error(caller, ": invalid K or N (K=", params.K,
                 ", N=", params.N, ")");
    return status_t::failure;
  }
  if (params.ldb <= 0) {
    apilog_error(caller, ": invalid ldb (", params.ldb, ")");
    return status_t::failure;
  }

  const int64_t required_ldb = params.transposed ? params.K : params.N;
  if (params.ldb < required_ldb) {
    apilog_error(caller, ": invalid ldb (", params.ldb,
                 "), expected at least ", required_ldb,
                 " for ",
                 (params.transposed ? "transposed" : "non-transposed"),
                 " weights");
    return status_t::failure;
  }
  // Prepack only supports the AOCL DLP blocked layout. (libxsmm_blocked
  // and onednn_blocked were intentionally dropped -- see lowoha_prepack.hpp
  // for the rationale.)
  if (params.algo != matmul_algo_t::aocl_dlp_blocked) {
    apilog_error(caller,
                 ": prepack.algo must be aocl_dlp_blocked (got ",
                 kernel_to_string(params.algo), ")");
    return status_t::failure;
  }
  return status_t::success;
}

// Full input validation for path the weight buffer
// (weight_prepack_into).
status_t validate_prepack_inputs(const char *caller,
                                 const void *weights,
                                 const prepack_params_t &params) {
  if (!weights) {
    apilog_error(caller, ": weights pointer is null");
    return status_t::failure;
  }
  return validate_prepack_params(caller, params);
}

// ===========================================================================
// Split into two single-purpose functions sharing the same per-dtype
// dispatch table:
//   aocl_compute_size : returns prepacked size in bytes (0 on failure).
//   aocl_prepack      : writes the prepacked layout into the caller's
//                       buffer.
// ===========================================================================
size_t aocl_compute_size(const prepack_params_t &params) {
  using namespace zendnnl::lowoha::matmul;

  const char order = 'r';
  const char trans = params.transposed ? 't' : 'n';
  const md_t k     = static_cast<md_t>(params.K);
  const md_t n     = static_cast<md_t>(params.N);

  if (params.wei_dtype == data_type_t::f32) {
    const size_t req = aocl_get_reorder_buf_size_f32f32f32of32(
                         order, trans, 'B', k, n, nullptr);
    return round_up_align(req, kPrepackAlign);
  }

  if (params.wei_dtype == data_type_t::bf16) {
    const size_t req = aocl_get_reorder_buf_size_bf16bf16f32of32(
                         order, trans, 'B', k, n, nullptr);
    return round_up_align(req, kPrepackAlign);
  }

  if (params.wei_dtype == data_type_t::f16) {
    const size_t req = aocl_get_reorder_buf_size_f16f16f16of16(
                         order, trans, 'B', k, n, nullptr);
    return round_up_align(req, kPrepackAlign);
  }

  if (params.wei_dtype == data_type_t::s4 ||
      params.wei_dtype == data_type_t::u4) {
    const size_t req = aocl_get_reorder_buf_size_bf16s4f32of32(
                         order, trans, 'B', k, n, nullptr);
    return round_up_align(req, kPrepackAlign);
  }

  if (params.wei_dtype == data_type_t::s8) {
    if (params.sym_group_size > 0 &&
        (params.src_dtype == data_type_t::bf16 ||
         params.src_dtype == data_type_t::s8)) {
      DLP_SYMM_STAT_QUANT symq_meta;
      symq_meta.group_size = params.sym_group_size;
      const size_t req = aocl_get_reorder_buf_size_s8s8s32os32_sym_quant(
                           order, trans, 'B', k, n, &symq_meta, nullptr);
      return round_up_align(req, kPrepackAlign);
    }

    if (params.src_dtype == data_type_t::u8) {
      const size_t req = aocl_get_reorder_buf_size_u8s8s32os32(
                           order, trans, 'B', k, n, nullptr);
      return round_up_align(req, kPrepackAlign);
    }

    // src = s8 / bf16 / f32 / unspecified -> s8s8s32os32
    const size_t req = aocl_get_reorder_buf_size_s8s8s32os32(
                         order, trans, 'B', k, n, nullptr);
    return round_up_align(req, kPrepackAlign);
  }

  apilog_error("weight_prepack(aocl_dlp): unsupported wei_dtype=",
               dtype_info(params.wei_dtype));
  return 0;
}

status_t aocl_prepack(const void *weights, const prepack_params_t &params,
                      void *dst) {
  using namespace zendnnl::lowoha::matmul;

  const char order = 'r';
  const char trans = params.transposed ? 't' : 'n';
  const md_t k     = static_cast<md_t>(params.K);
  const md_t n     = static_cast<md_t>(params.N);
  const md_t ldb   = static_cast<md_t>(params.ldb);

  if (params.wei_dtype == data_type_t::f32) {
    aocl_reorder_f32f32f32of32(order, trans, 'B',
                               static_cast<const float *>(weights),
                               static_cast<float *>(dst), k, n, ldb, nullptr);
    return status_t::success;
  }

  if (params.wei_dtype == data_type_t::bf16) {
    aocl_reorder_bf16bf16f32of32(order, trans, 'B',
                                 static_cast<const int16_t *>(weights),
                                 static_cast<int16_t *>(dst), k, n, ldb,
                                 nullptr);
    return status_t::success;
  }

  if (params.wei_dtype == data_type_t::f16) {
    aocl_reorder_f16f16f16of16(order, trans, 'B',
                               static_cast<const uint16_t *>(weights),
                               static_cast<uint16_t *>(dst), k, n, ldb,
                               nullptr);
    return status_t::success;
  }

  if (params.wei_dtype == data_type_t::s4 ||
      params.wei_dtype == data_type_t::u4) {
    aocl_reorder_bf16s4f32of32(order, trans, 'B',
                               static_cast<const int8_t *>(weights),
                               static_cast<int8_t *>(dst), k, n, ldb, nullptr);
    return status_t::success;
  }

  if (params.wei_dtype == data_type_t::s8) {
    if (params.sym_group_size > 0 &&
        (params.src_dtype == data_type_t::bf16 ||
         params.src_dtype == data_type_t::s8)) {
      DLP_SYMM_STAT_QUANT symq_meta;
      symq_meta.group_size = params.sym_group_size;
      aocl_reorder_s8s8s32os32_sym_quant(order, trans, 'B',
                                         static_cast<const int8_t *>(weights),
                                         static_cast<int8_t *>(dst),
                                         k, n, ldb, &symq_meta, nullptr);
      return status_t::success;
    }

    if (params.src_dtype == data_type_t::u8) {
      aocl_reorder_u8s8s32os32(order, trans, 'B',
                               static_cast<const int8_t *>(weights),
                               static_cast<int8_t *>(dst), k, n, ldb, nullptr);
      return status_t::success;
    }

    aocl_reorder_s8s8s32os32(order, trans, 'B',
                             static_cast<const int8_t *>(weights),
                             static_cast<int8_t *>(dst), k, n, ldb, nullptr);
    return status_t::success;
  }

  apilog_error("weight_prepack(aocl_dlp): unsupported wei_dtype=",
               dtype_info(params.wei_dtype));
  return status_t::unimplemented;
}

// =====================================================================
// Algo dispatchers: AOCL DLP is the only supported backend.
// =====================================================================
size_t backend_size_by_algo(const prepack_params_t &params) {
  if (params.algo == matmul_algo_t::aocl_dlp_blocked) {
    return aocl_compute_size(params);
  }
  apilog_error("weight_prepack_size: algo not supported by prepack API (",
               kernel_to_string(params.algo),
               "); only aocl_dlp_blocked is supported");
  return 0;
}

status_t backend_prepack_by_algo(const void *weights,
                                 const prepack_params_t &params,
                                 void *dst) {
  if (params.algo == matmul_algo_t::aocl_dlp_blocked) {
    return aocl_prepack(weights, params, dst);
  }
  apilog_error("weight_prepack_into: algo not supported by prepack API (",
               kernel_to_string(params.algo),
               "); only aocl_dlp_blocked is supported");
  return status_t::unimplemented;
}

} // anonymous namespace

// ===========================================================================
// Public API: weight_prepack_size
// ===========================================================================

size_t weight_prepack_size(const reorder_params_t &params) {
  const prepack_params_t &pp = params.prepack;

  // Always reset the cache and recompute.
  pp.cached_size = 0;

  // Diagnostic-gated params validation (no weights pointer needed here).
  status_t val_status = op_instrumentation::validate([&]() {
    return validate_prepack_params("weight_prepack_size", pp);
  });
  if (val_status != status_t::success) {
    return 0;
  }

  if (apilog_info_enabled()) {
    std::ostringstream ss;
    ss << "LOWOHA weight_prepack_size: algo=" << kernel_to_string(pp.algo)
       << ", K=" << pp.K
       << ", N=" << pp.N
       << ", wei_dtype=" << dtype_info(pp.wei_dtype);
    apilog_info(ss.str());
  }

  const size_t size = backend_size_by_algo(pp);
  if (size > 0) {
    pp.cached_size = size;
  }
  return size;
}

status_t weight_prepack_into(const void *weights,
                             const reorder_params_t &params,
                             void *dst) {
  const prepack_params_t &pp = params.prepack;

  // Thread control: inherited from reorder_direct's thread_guard
  // (which is set up from reorder_params_t::num_threads before
  //  this function is reached). The prepack pipeline does not expose
  //  its own num_threads knob.

  // Diagnostic-gated input validation.
  status_t val_status = op_instrumentation::validate([&]() {
    return validate_prepack_inputs("weight_prepack_into", weights, pp);
  });
  if (val_status != status_t::success) {
    return val_status;
  }
  if (!dst) {
    apilog_error("weight_prepack_into: dst pointer is null");
    return status_t::failure;
  }

  // The caller is contractually required to have allocated at least
  // weight_prepack_size(params) bytes at `dst`. No alignment check
  // here: none of the supported backends require a specific alignment
  // for correctness.

  profiler_t profiler;
  const bool is_profile = is_profile_enabled();

  [[maybe_unused]] std::string log_str;
  if (apilog_info_enabled() || is_profile) {
    std::ostringstream ss;
    ss << "LOWOHA weight_prepack_into: algo=" << kernel_to_string(pp.algo)
       << ", K=" << pp.K
       << ", N=" << pp.N
       << ", ldb=" << pp.ldb
       << ", trans=" << (pp.transposed ? 't' : 'n')
       << ", wei_dtype=" << dtype_info(pp.wei_dtype);
    log_str = ss.str();
    if (apilog_info_enabled()) {
      apilog_info(log_str);
    }
  }

  if (is_profile) {
    profiler.tbp_start();
  }

  const status_t st = backend_prepack_by_algo(weights, pp, dst);

  if (is_profile) {
    profiler.tbp_stop();
    profilelog_verbose(log_str, ", time=", profiler.tbp_elapsedtime(),
                       profiler.get_res_str());
  }

  return st;
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
