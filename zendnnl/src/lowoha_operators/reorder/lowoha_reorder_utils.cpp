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

#include "lowoha_operators/reorder/lowoha_reorder_utils.hpp"
#include "common/zendnnl_global.hpp"

#include <cmath>

namespace zendnnl {
namespace lowoha {
namespace reorder {

using namespace zendnnl::error_handling;

status_t validate_reorder_inputs(const void *src, void *dst, size_t nelems,
                                  const reorder_params_t &params) {
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
  if (!is_reorder_supported(params.src_dtype, params.dst_dtype)) {
    log_error("Unsupported data type combination: src=",
              reorder_data_type_to_string(params.src_dtype),
              ", dst=", reorder_data_type_to_string(params.dst_dtype));
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
    if (!std::isfinite(scale_val)) {
      log_error("Invalid scale parameter: scale must be finite. Got: ",
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

    // For bf16/f32->int8 quantization, check zero_point is within valid range
    if ((params.src_dtype == data_type_t::bf16 || params.src_dtype == data_type_t::f32) && 
        params.dst_dtype == data_type_t::s8) {
      int32_t zp_val = *static_cast<const int32_t *>(params.quant_params.zero_point.buff);
      if (zp_val < -128 || zp_val > 127) {
        log_error("Invalid zero_point for int8 quantization. Must be in [-128, 127]. Got: ",
                  zp_val);
        return status_t::failure;
      }
    }

    // For bf16/f32->uint8 quantization, check zero_point is within valid range
    if ((params.src_dtype == data_type_t::bf16 || params.src_dtype == data_type_t::f32) && 
        params.dst_dtype == data_type_t::u8) {
      int32_t zp_val = *static_cast<const int32_t *>(params.quant_params.zero_point.buff);
      if (zp_val < 0 || zp_val > 255) {
        log_error("Invalid zero_point for uint8 quantization. Must be in [0, 255]. Got: ",
                  zp_val);
        return status_t::failure;
      }
    }
  }

  // Validate shape parameters
  if (validate_reorder_shape(params) != status_t::success) {
    return status_t::failure;
  }

  // Validate stride parameters
  if (validate_reorder_strides(params) != status_t::success) {
    return status_t::failure;
  }

  // Validate quantization parameters
  if (validate_reorder_quant_params(params) != status_t::success) {
    return status_t::failure;
  }

  return status_t::success;
}

status_t validate_reorder_quant_params(const reorder_params_t &params) {
  const auto &scale = params.quant_params.scale;
  const auto &zp = params.quant_params.zero_point;
  const auto &shape = params.src_shape;
  
  // For f32 <-> bf16 conversions, scale/zp are optional
  // If both buffers are null, skip quant params validation
  bool is_f32_bf16_conversion = 
      (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::bf16) ||
      (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::f32);
  
  if (is_f32_bf16_conversion && scale.buff == nullptr && zp.buff == nullptr) {
    // Pure type conversion without scaling - no validation needed
    return status_t::success;
  }
  
  // Get shape dimensions
  const int64_t M = params.M();
  const int64_t N = params.N();
  
  // Helper lambda to validate quant dims
  auto validate_quant_dims = [&](const std::vector<int64_t> &dims, 
                                  const char *param_name) -> status_t {
    // Dims are mandatory
    if (dims.empty()) {
      log_error(param_name, " dims is empty (dims are mandatory)");
      return status_t::failure;
    }
    
    // Dims size must match shape size
    if (dims.size() != shape.size()) {
      log_error(param_name, " dims size (", dims.size(), 
                ") doesn't match shape size (", shape.size(), ")");
      return status_t::failure;
    }
    
    // All dims must be positive
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] <= 0) {
        log_error("Invalid ", param_name, " dims[", i, "]=", dims[i], " (must be > 0)");
        return status_t::failure;
      }
    }
    
    // Validate based on dimensionality
    if (dims.size() == 1) {
      // 1D: dims = {1} (per-tensor) or {N} (per-channel), no per-group
      int64_t N_dim = shape[0];
      if (dims[0] != 1 && dims[0] != N_dim) {
        log_error(param_name, " dims[0]=", dims[0], 
                  " must be 1 (per-tensor) or N=", N_dim, " (per-channel)");
        return status_t::failure;
      }
    } else if (dims.size() == 2) {
      // 2D: dims = {1,1} (per-tensor), {1,N} (per-channel), or {G,N} (per-group)
      if (dims[1] != 1 && dims[1] != N) {
        log_error(param_name, " dims[1]=", dims[1], 
                  " must be 1 (per-tensor) or N=", N, " (per-channel/per-group)");
        return status_t::failure;
      }
      if (dims[0] != 1 && dims[1] == N) {
        // Per-group: check M divisibility
        if (M % dims[0] != 0) {
          log_error(param_name, " per-group: M=", M, 
                    " is not divisible by G=", dims[0]);
          return status_t::failure;
        }
      }
    } else if (dims.size() == 3) {
      // 3D: dims = {1,1,1} (per-tensor), {1,1,N} (per-channel), or {1,G,N} (per-group)
      if (dims[0] != 1) {
        log_error(param_name, " dims[0]=", dims[0], 
                  " must be 1 for 3D tensors");
        return status_t::failure;
      }
      if (dims[2] != 1 && dims[2] != N) {
        log_error(param_name, " dims[2]=", dims[2], 
                  " must be 1 (per-tensor) or N=", N, " (per-channel/per-group)");
        return status_t::failure;
      }
      if (dims[1] != 1 && dims[2] == N) {
        // Per-group: check M divisibility
        if (M % dims[1] != 0) {
          log_error(param_name, " per-group: M=", M, 
                    " is not divisible by G=", dims[1]);
          return status_t::failure;
        }
      }
    }
    
    return status_t::success;
  };
  
  // Validate scale dims
  status_t status = validate_quant_dims(scale.dims, "scale");
  if (status != status_t::success) {
    return status;
  }
  
  // Validate zero_point dims
  status = validate_quant_dims(zp.dims, "zero_point");
  if (status != status_t::success) {
    return status;
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

reorder_algo_t select_reorder_algo(const reorder_params_t &params, size_t nelems) {
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
  // 5. f32 -> s8 (quantization)
  // 6. s8 -> f32 (dequantization)
  // 7. f32 -> u8 (quantization)
  // 8. u8 -> f32 (dequantization)
  // 9. f32 -> bf16 (type conversion with optional quantization)
  // 10. bf16 -> f32 (type conversion with optional dequantization)

  return (src_dtype == data_type_t::bf16 && dst_dtype == data_type_t::s8) ||
         (src_dtype == data_type_t::s8 && dst_dtype == data_type_t::bf16) ||
         (src_dtype == data_type_t::bf16 && dst_dtype == data_type_t::u8) ||
         (src_dtype == data_type_t::u8 && dst_dtype == data_type_t::bf16) ||
         (src_dtype == data_type_t::f32 && dst_dtype == data_type_t::s8) ||
         (src_dtype == data_type_t::s8 && dst_dtype == data_type_t::f32) ||
         (src_dtype == data_type_t::f32 && dst_dtype == data_type_t::u8) ||
         (src_dtype == data_type_t::u8 && dst_dtype == data_type_t::f32) ||
         (src_dtype == data_type_t::f32 && dst_dtype == data_type_t::bf16) ||
         (src_dtype == data_type_t::bf16 && dst_dtype == data_type_t::f32);
}

status_t validate_reorder_shape(const reorder_params_t &params) {
  if (!params.is_shaped()) {
    log_error("Invalid reorder shape: src_shape is empty or contains invalid dimensions");
    return status_t::failure;
  }

  // Check if dst_shape is provided
  if (params.dst_shape.empty()) {
    log_error("Invalid reorder shape: dst_shape is empty");
    return status_t::failure;
  }
  // Check that src_shape and dst_shape match
  if (!params.shapes_match()) {
    log_error("Invalid reorder shape: src_shape and dst_shape must be identical. ",
              "src_shape size=", params.src_shape.size(), 
              ", dst_shape size=", params.dst_shape.size());
    return status_t::failure;
  }

  // Check for reasonable dimensions to avoid overflow
  const int64_t max_dim = 1LL << 30;  // 1 billion elements per dimension
  for (size_t i = 0; i < params.src_shape.size(); ++i) {
    if (params.src_shape[i] > max_dim) {
      log_error("Shape dimension too large at index ", i, ": ", params.src_shape[i]);
      return status_t::failure;
    }
  }

  // Check for overflow in total element count
  const int64_t max_nelems = 1LL << 40;  // ~1 trillion elements max
  if (params.nelems() > max_nelems) {
    log_error("Total element count too large: ", params.nelems());
    return status_t::failure;
  }

  return status_t::success;
}

bool is_stride_contiguous(const reorder_params_t &params) {
  return params.is_src_contiguous();
}

status_t validate_reorder_strides(const reorder_params_t &params) {
  if (!params.has_src_strides()) {
    return status_t::success;  // No strides to validate
  }

  const auto &strides = params.src_strides;

  // Validate stride values are positive
  for (size_t i = 0; i < strides.size(); ++i) {
    if (strides[i] <= 0) {
      log_error("Invalid stride at index ", i, ": ", strides[i], " (must be > 0)");
      return status_t::failure;
    }
  }

  // Validate stride dimensions match shape (shape is mandatory, validated earlier)
  if (strides.size() != params.src_shape.size()) {
    log_error("Stride dimensions (", strides.size(), 
              ") don't match shape dimensions (", params.src_shape.size(), ")");
    return status_t::failure;
  }

  // Validate stride values are reasonable relative to shape
  if (strides.size() == 2) {
    // For 2D, stride_M should be at least N for non-overlapping access
    if (strides[0] < params.N() && strides[1] == 1) {
      log_error("Warning: stride_M (", strides[0], 
                ") is less than N (", params.N(), "), may cause overlapping access");
    }
  } else if (strides.size() == 3) {
    // For 3D, stride_batch should be at least M*N
    const int64_t matrix_size = params.M() * params.N();
    if (strides[0] < matrix_size && strides[1] == params.N() && strides[2] == 1) {
      log_error("Warning: stride_batch (", strides[0], 
                ") is less than M*N (", matrix_size, "), may cause overlapping access");
    }
  }

  return status_t::success;
}

const char *granularity_to_string(granularity_type_t granularity) {
  switch (granularity) {
    case granularity_type_t::invalid: return "invalid";
    case granularity_type_t::per_tensor: return "per_tensor";
    case granularity_type_t::per_channel: return "per_channel";
    case granularity_type_t::per_group: return "per_group";
    case granularity_type_t::mixed: return "mixed";
    default: return "unknown";
  }
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

