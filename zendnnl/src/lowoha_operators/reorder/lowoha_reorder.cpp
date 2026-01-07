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

#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_utils.hpp"
#include "lowoha_operators/reorder/reorder_kernels.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "common/zendnnl_global.hpp"

#include <omp.h>
#include <sstream>

namespace zendnnl {
namespace lowoha {

using namespace zendnnl::error_handling;
using namespace zendnnl::profile;

/**
 * @brief Extract scale value from quant_t (currently only f32 supported)
 */
static inline float get_scale_value(const reorder_quant_params_t::quant_t
                                    &scale_param) {
  if (scale_param.buff == nullptr) {
    return 1.0f;  // Default scale
  }
  // Currently only f32 scale is supported
  return *static_cast<const float *>(scale_param.buff);
}

/**
 * @brief Extract zero_point value from quant_t (currently only s32 supported)
 */
static inline int get_zero_point_value(const reorder_quant_params_t::quant_t
                                       &zp_param) {
  if (zp_param.buff == nullptr) {
    return 0;  // Default zero_point
  }
  // Currently only s32 zero_point is supported
  return *static_cast<const int32_t *>(zp_param.buff);
}

/**
 * @brief Execute reorder kernel based on selected algorithm
 */
static void execute_reorder_kernel(const void *src, void *dst, size_t nelems,
                                   const lowoha_reorder_params_t &params,
                                   reorder_algo_t algo) {
  const float scale = get_scale_value(params.quant_params.scale);
  const int zero_point = get_zero_point_value(params.quant_params.zero_point);

  // BF16 -> INT8 (Quantization)
  if (params.dtypes.src == data_type_t::bf16 &&
      params.dtypes.dst == data_type_t::s8) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    int8_t *dst_int8 = static_cast<int8_t *>(dst);

    if (algo == reorder_algo_t::native) {
      quantize_bf16_to_int8_avx512(src_bf16, dst_int8, nelems, scale, zero_point);
    }
    else {
      quantize_bf16_to_int8_ref(src_bf16, dst_int8, nelems, scale, zero_point);
    }
    return;
  }

  // INT8 -> BF16 (Dequantization)
  if (params.dtypes.src == data_type_t::s8 &&
      params.dtypes.dst == data_type_t::bf16) {
    const int8_t *src_int8 = static_cast<const int8_t *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);

    if (algo == reorder_algo_t::native) {
      dequantize_int8_to_bf16_avx512(src_int8, dst_bf16, nelems, scale, zero_point);
    }
    else {
      dequantize_int8_to_bf16_ref(src_int8, dst_bf16, nelems, scale, zero_point);
    }
    return;
  }

  // BF16 -> UINT8 (Quantization)
  if (params.dtypes.src == data_type_t::bf16 &&
      params.dtypes.dst == data_type_t::u8) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    uint8_t *dst_uint8 = static_cast<uint8_t *>(dst);

    if (algo == reorder_algo_t::native) {
      quantize_bf16_to_uint8_avx512(src_bf16, dst_uint8, nelems, scale, zero_point);
    }
    else {
      quantize_bf16_to_uint8_ref(src_bf16, dst_uint8, nelems, scale, zero_point);
    }
    return;
  }

  // UINT8 -> BF16 (Dequantization)
  if (params.dtypes.src == data_type_t::u8 &&
      params.dtypes.dst == data_type_t::bf16) {
    const uint8_t *src_uint8 = static_cast<const uint8_t *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);

    if (algo == reorder_algo_t::native) {
      dequantize_uint8_to_bf16_avx512(src_uint8, dst_bf16, nelems, scale, zero_point);
    }
    else {
      dequantize_uint8_to_bf16_ref(src_uint8, dst_bf16, nelems, scale, zero_point);
    }
    return;
  }
}

/**
 * @brief Parallel reorder execution for large buffers
 */
static void execute_reorder_parallel(const void *src, void *dst, size_t nelems,
                                     const lowoha_reorder_params_t &params,
                                     reorder_algo_t algo) {
  constexpr int64_t grain_size = 1024;  // Minimum elements per thread

  const size_t src_elem_size = (params.dtypes.src == data_type_t::bf16) ? sizeof(
                                 uint16_t) : sizeof(uint8_t);
  const size_t dst_elem_size = (params.dtypes.dst == data_type_t::bf16) ? sizeof(
                                 uint16_t) : sizeof(uint8_t);

  zendnnl_parallel_for(0, static_cast<int64_t>(nelems), grain_size,
  [&](int64_t begin, int64_t end) {
    const uint8_t *src_ptr = static_cast<const uint8_t *>(src) + begin *
                             src_elem_size;
    uint8_t *dst_ptr = static_cast<uint8_t *>(dst) + begin * dst_elem_size;
    size_t thread_nelems = static_cast<size_t>(end - begin);

    execute_reorder_kernel(src_ptr, dst_ptr, thread_nelems, params, algo);
  });
}

status_t reorder_direct(const void *src, void *dst, size_t nelems,
                        lowoha_reorder_params_t params) {
  // Validate inputs
  if (validate_reorder_inputs(src, dst, nelems, params) != status_t::success) {
    return status_t::failure;
  }

  // Select algorithm
  reorder_algo_t algo = select_reorder_algo(params, nelems);

  // Determine number of threads
  //const int num_threads = params.num_threads > 0 ? static_cast<int>(params.num_threads)
  //                                                : omp_get_max_threads();
  // Create profiler instance for timing
  profiler_t profiler;
  bool is_profile = is_profile_enabled();

  // Build log string for API and profile logging
  [[maybe_unused]] std::ostringstream ss;
  if (apilog_info_enabled() || is_profile) {
    float scale_val = get_scale_value(params.quant_params.scale);
    int zp_val = get_zero_point_value(params.quant_params.zero_point);
    ss << "LOWOHA reorder_direct: nelems=" << nelems
       << ", src_dtype=" << reorder_data_type_to_string(params.dtypes.src)
       << ", dst_dtype=" << reorder_data_type_to_string(params.dtypes.dst)
       << ", scale=" << scale_val
       << ", zero_point=" << zp_val
       << ", algo=" << reorder_algo_to_string(algo);

    if (apilog_info_enabled()) {
      apilog_info(ss.str());
    }
  }

  // Start profiling timer
  if (is_profile) {
    profiler.tbp_start();
  }

  const int num_threads = params.num_threads > 0 ? params.num_threads :
                          omp_get_max_threads();

  reorder_threadlimit thread_guard(num_threads);

  execute_reorder_parallel(src, dst, nelems, params, algo);

  // Stop profiling timer and log
  if (is_profile) {
    profiler.tbp_stop();
    profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(),
                       profiler.get_res_str());
  }

  return status_t::success;
}

} // namespace lowoha
} // namespace zendnnl

