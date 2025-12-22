/*******************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_operators/reorder/lowoha_reorder_utils.hpp"
#include "common/zendnnl_global.hpp"

#include <cmath>

namespace zendnnl {
namespace lowoha {

using namespace zendnnl::error_handling;

status_t validate_reorder_inputs(const void *src, void *dst, size_t nelems,
                                  const lowoha_reorder_params_t &params) {
  // Check for null pointers
  if (!src || !dst) {
    log_error("Null pointer input to reorder_direct: src=",
              static_cast<const void *>(src),
              ", dst=", static_cast<const void *>(dst));
    return status_t::failure;
  }

  // Validate nelems
  if (nelems <= 0) {
    log_error("Invalid nelems: nelems=", nelems);
    return status_t::failure;
  }

  // Validate data type combination
  if (!is_reorder_supported(params.dtypes.src, params.dtypes.dst)) {
    log_error("Unsupported data type combination: src=",
              reorder_data_type_to_string(params.dtypes.src),
              ", dst=", reorder_data_type_to_string(params.dtypes.dst));
    return status_t::failure;
  }

  // Validate scale parameter
  if (params.quant_params.scale.buff != nullptr) {
    // Currently only f32 scale is supported
    if (params.quant_params.scale.dt != data_type_t::f32) {
      log_error("Invalid scale data type: only f32 is currently supported. Got: ",
                reorder_data_type_to_string(params.quant_params.scale.dt));
      return status_t::failure;
    }
    float scale_val = *static_cast<const float *>(params.quant_params.scale.buff);
    if (scale_val <= 0.0f || !std::isfinite(scale_val)) {
      log_error("Invalid scale parameter: scale must be positive and finite. Got: ",
                scale_val);
      return status_t::failure;
    }
  }

  // Validate zero_point parameter
  if (params.quant_params.zero_point.buff != nullptr) {
    // Currently only s32 zero_point is supported
    if (params.quant_params.zero_point.dt != data_type_t::s32) {
      log_error("Invalid zero_point data type: only s32 is currently supported. Got: ",
                reorder_data_type_to_string(params.quant_params.zero_point.dt));
      return status_t::failure;
    }

    // For bf16->int8 quantization, check zero_point is within valid range
    if (params.dtypes.src == data_type_t::bf16 && params.dtypes.dst == data_type_t::s8) {
      int32_t zp_val = *static_cast<const int32_t *>(params.quant_params.zero_point.buff);
      if (zp_val < -128 || zp_val > 127) {
        log_error("Invalid zero_point for int8 quantization. Must be in [-128, 127]. Got: ",
                  zp_val);
        return status_t::failure;
      }
    }

    // For bf16->uint8 quantization, check zero_point is within valid range
    if (params.dtypes.src == data_type_t::bf16 && params.dtypes.dst == data_type_t::u8) {
      int32_t zp_val = *static_cast<const int32_t *>(params.quant_params.zero_point.buff);
      if (zp_val < 0 || zp_val > 255) {
        log_error("Invalid zero_point for uint8 quantization. Must be in [0, 255]. Got: ",
                  zp_val);
        return status_t::failure;
      }
    }
  }

  return status_t::success;
}

const char *reorder_data_type_to_string(data_type_t dtype) {
  switch (dtype) {
  case data_type_t::none:
    return "none";
  case data_type_t::f32:
    return "f32";
  case data_type_t::bf16:
    return "bf16";
  case data_type_t::s4:
    return "s4";
  case data_type_t::s8:
    return "s8";
  case data_type_t::u8:
    return "u8";
  case data_type_t::s32:
    return "s32";
  default:
    return "unknown";
  }
}

const char *reorder_algo_to_string(reorder_algo_t algo) {
  switch (algo) {
  case reorder_algo_t::none:
    return "none";
  case reorder_algo_t::native:
    return "native";
  case reorder_algo_t::reference:
    return "reference";
  case reorder_algo_t::DT:
    return "DT";
  default:
    return "unknown";
  }
}

reorder_algo_t select_reorder_algo(const lowoha_reorder_params_t &params, size_t nelems) {
  // If user explicitly specified an algorithm (not DT), use it
  if (params.algo != reorder_algo_t::DT && params.algo != reorder_algo_t::none) {
    return params.algo;
  }

  // Decision tree based selection logic:
  // Use AVX512 for larger buffers (>= 64 elements) for better performance
  // Use reference for smaller buffers to avoid AVX512 overhead
  if (nelems >= 64) {
    return reorder_algo_t::native;
  }

  return reorder_algo_t::reference;
}

bool is_reorder_supported(data_type_t src_dtype, data_type_t dst_dtype) {
  // Currently supported conversions:
  // 1. bf16 -> s8 (quantization)
  // 2. s8 -> bf16 (dequantization)
  // 3. bf16 -> u8 (quantization)
  // 4. u8 -> bf16 (dequantization)

  return (src_dtype == data_type_t::bf16 && dst_dtype == data_type_t::s8) ||
         (src_dtype == data_type_t::s8 && dst_dtype == data_type_t::bf16) ||
         (src_dtype == data_type_t::bf16 && dst_dtype == data_type_t::u8) ||
         (src_dtype == data_type_t::u8 && dst_dtype == data_type_t::bf16);
}

} // namespace lowoha
} // namespace zendnnl

