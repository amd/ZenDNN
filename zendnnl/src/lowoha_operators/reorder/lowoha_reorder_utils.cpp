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
#include "lowoha_operators/reorder/reorder_data_type/dynamic_quant_impl/dynamic_kernels.hpp"
#include "common/zendnnl_global.hpp"
#include "common/float16.hpp"

#include <cmath>
#include <limits>
#include <vector>

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
    // Supported scale data types: f32, bf16, and f16. bf16 and f16 are
    // widened to f32 transparently on read via get_scale_value(). The
    // dispatchers stage internally in f32 and narrow back to the user's
    // dtype on write for the dynamic-quant path.
    if (params.quant_params.scale.dt != data_type_t::f32  &&
        params.quant_params.scale.dt != data_type_t::bf16 &&
        params.quant_params.scale.dt != data_type_t::f16) {
      log_error("Invalid scale data type: f32, bf16, or f16 expected. Got: ",
                reorder_data_type_to_string(params.quant_params.scale.dt));
      return status_t::failure;
    }
    float scale_val = get_scale_value(params.quant_params.scale);
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

  // For float-only conversions (f32 <-> bf16, f32 <-> f16, bf16 <-> f16),
  // scale/zp are optional. If both buffers are null, skip quant params validation
  bool is_float_only_conversion =
    (params.src_dtype == data_type_t::f32  &&
     params.dst_dtype == data_type_t::bf16) ||
    (params.src_dtype == data_type_t::bf16 &&
     params.dst_dtype == data_type_t::f32)  ||
    (params.src_dtype == data_type_t::f32  &&
     params.dst_dtype == data_type_t::f16)  ||
    (params.src_dtype == data_type_t::f16  &&
     params.dst_dtype == data_type_t::f32)  ||
    (params.src_dtype == data_type_t::bf16 &&
     params.dst_dtype == data_type_t::f16)  ||
    (params.src_dtype == data_type_t::f16  &&
     params.dst_dtype == data_type_t::bf16);

  if (is_float_only_conversion && scale.buff == nullptr && zp.buff == nullptr) {
    // Pure type conversion without scaling - no validation needed
    return status_t::success;
  }

  // Get shape dimensions
  const int64_t M = params.M();
  const int64_t N = params.N();

  // Helper lambda to validate quant dims
  // Supported configurations:
  //   1D: {1} (per-tensor), {N} (per-channel)
  //   2D: {1,1} (per-tensor), {1,N} (per-channel-col), {M,1} (per-channel-row),
  //       {G,N} (per-group-row), {M,G} (per-group-col)
  //   3D: {1,1,1} (per-tensor), {1,1,N} (per-channel-col), {1,M,1} (per-channel-row),
  //       {1,G,N} (per-group-row), {1,M,G} (per-group-col)
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
    }
    else if (dims.size() == 2) {
      // 2D tensor (shape = [M, N]):
      //   - per-tensor:      {1, 1}
      //   - per-channel-col: {1, N}
      //   - per-channel-row: {M, 1}
      //   - per-group-row:   {G, N} where M % G == 0 and G > 1
      //   - per-group-col:   {M, G} where N % G == 0 and G > 1

      bool is_per_tensor = (dims[0] == 1 && dims[1] == 1);
      bool is_per_channel_col = (dims[0] == 1 && dims[1] == N);
      bool is_per_channel_row = (dims[0] == M && dims[1] == 1);
      bool is_per_group_row = (dims[0] > 1 && dims[0] != M && dims[1] == N &&
                               M % dims[0] == 0);
      bool is_per_group_col = (dims[0] == M && dims[1] > 1 && dims[1] != N &&
                               N % dims[1] == 0);

      if (!is_per_tensor && !is_per_channel_col && !is_per_channel_row &&
          !is_per_group_row && !is_per_group_col) {
        log_error(param_name, " invalid dims {", dims[0], ", ", dims[1],
                  "} for shape {", M, ", ", N, "}. "
                  "Expected: {1,1} (per-tensor), {1,", N, "} (per-channel-col), "
                  "{", M, ",1} (per-channel-row), {G,", N, "} (per-group-row), "
                  "or {", M, ",G} (per-group-col)");
        return status_t::failure;
      }

      // Additional validation for per-group divisibility
      if (is_per_group_row && M % dims[0] != 0) {
        log_error(param_name, " per-group-row: M=", M,
                  " is not divisible by G=", dims[0]);
        return status_t::failure;
      }
      if (is_per_group_col && N % dims[1] != 0) {
        log_error(param_name, " per-group-col: N=", N,
                  " is not divisible by G=", dims[1]);
        return status_t::failure;
      }
    }
    else if (dims.size() == 3) {
      // 3D tensor (shape = [batch, M, N]):
      //   - per-tensor:      {1, 1, 1}
      //   - per-channel-col: {1, 1, N}
      //   - per-channel-row: {1, M, 1}
      //   - per-group-row:   {1, G, N} where M % G == 0 and G > 1
      //   - per-group-col:   {1, M, G} where N % G == 0 and G > 1

      if (dims[0] != 1) {
        log_error(param_name, " dims[0]=", dims[0],
                  " must be 1 for 3D tensors");
        return status_t::failure;
      }

      bool is_per_tensor = (dims[1] == 1 && dims[2] == 1);
      bool is_per_channel_col = (dims[1] == 1 && dims[2] == N);
      bool is_per_channel_row = (dims[1] == M && dims[2] == 1);
      bool is_per_group_row = (dims[1] > 1 && dims[1] != M && dims[2] == N &&
                               M % dims[1] == 0);
      bool is_per_group_col = (dims[1] == M && dims[2] > 1 && dims[2] != N &&
                               N % dims[2] == 0);

      if (!is_per_tensor && !is_per_channel_col && !is_per_channel_row &&
          !is_per_group_row && !is_per_group_col) {
        log_error(param_name, " invalid dims {", dims[0], ", ", dims[1], ", ", dims[2],
                  "} for shape {batch, ", M, ", ", N, "}. "
                  "Expected: {1,1,1} (per-tensor), {1,1,", N, "} (per-channel-col), "
                  "{1,", M, ",1} (per-channel-row), {1,G,", N, "} (per-group-row), "
                  "or {1,", M, ",G} (per-group-col)");
        return status_t::failure;
      }

      // Additional validation for per-group divisibility
      if (is_per_group_row && M % dims[1] != 0) {
        log_error(param_name, " per-group-row: M=", M,
                  " is not divisible by G=", dims[1]);
        return status_t::failure;
      }
      if (is_per_group_col && N % dims[2] != 0) {
        log_error(param_name, " per-group-col: N=", N,
                  " is not divisible by G=", dims[2]);
        return status_t::failure;
      }
    }

    return status_t::success;
  };

  // Validate scale dims (only if scale buffer is provided)
  if (scale.buff != nullptr) {
    status_t status = validate_quant_dims(scale.dims, "scale");
    if (status != status_t::success) {
      return status;
    }
  }

  // Validate zero_point dims (only if zero_point buffer is provided)
  // For symmetric quantization, zp.buff can be nullptr and dims can be empty
  if (zp.buff != nullptr) {
    status_t status = validate_quant_dims(zp.dims, "zero_point");
    if (status != status_t::success) {
      return status;
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
  case data_type_t::f16:
    return "f16";
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

reorder_algo_t select_reorder_algo(const reorder_params_t &params,
                                    size_t nelems) {
  // Native reorder / dynamic-quant kernels are compiled for
  // avx512f + avx512bw + avx512vl (they emit VPMOVDB narrowing plus
  // byte/word and 128/256-bit masked AVX-512 instructions). Selecting
  // them on a CPU that has AVX-512F but lacks BW/VL (e.g. some Intel
  // Xeon Phi SKUs) would raise an illegal-instruction fault, so the
  // full feature set is required before choosing the native path.
  static const bool has_native_isa =
    zendnnl::common::zendnnl_platform_info().get_avx512f_status() &&
    zendnnl::common::zendnnl_platform_info().get_avx512_bw_vl_status();
  // If user explicitly specified an algorithm (not DT), use it
  if (params.algo != reorder_algo_t::DT && params.algo != reorder_algo_t::none) {
    if (params.algo == reorder_algo_t::native && !has_native_isa) {
      log_info("Reorder native kernel requires AVX-512 "
                "(avx512f + avx512bw + avx512vl); falling back to reference "
                "kernel on this platform.");
      return reorder_algo_t::reference;
    }
    return params.algo;
  }

  // Decision tree based selection logic:
  // Use AVX512 for larger buffers (>= 64 elements) for better performance
  // Use reference for smaller buffers to avoid AVX512 overhead
  if (nelems >= 64 && has_native_isa) {
    return reorder_algo_t::native;
  }

  return reorder_algo_t::reference;
}

bool is_reorder_supported(data_type_t src_dtype, data_type_t dst_dtype) {
  // Currently supported conversions:
  // 1.  bf16 -> s8   (quantization)
  // 2.  s8   -> bf16 (dequantization)
  // 3.  bf16 -> u8   (quantization)
  // 4.  u8   -> bf16 (dequantization)
  // 5.  f32  -> s8   (quantization)
  // 6.  s8   -> f32  (dequantization)
  // 7.  f32  -> u8   (quantization)
  // 8.  u8   -> f32  (dequantization)
  // 9.  f32  -> bf16 (type conversion with optional quantization)
  // 10. bf16 -> f32  (type conversion with optional dequantization)
  // 11. f32  -> f16  (type conversion with optional quantization)
  // 12. f16  -> f32  (type conversion with optional dequantization)
  // 13. bf16 -> f16  (type conversion with optional quantization)
  // 14. f16  -> bf16 (type conversion with optional dequantization)
  // 15. f16  -> s8   (static + dynamic quantization)
  // 16. s8   -> f16  (static dequantization)
  // 17. f16  -> u8   (static + dynamic quantization)
  // 18. u8   -> f16  (static dequantization)
  

  return (src_dtype == data_type_t::bf16 && dst_dtype == data_type_t::s8)   ||
          (src_dtype == data_type_t::s8   && dst_dtype == data_type_t::bf16) ||
          (src_dtype == data_type_t::bf16 && dst_dtype == data_type_t::u8)   ||
          (src_dtype == data_type_t::u8   && dst_dtype == data_type_t::bf16) ||
          (src_dtype == data_type_t::f32  && dst_dtype == data_type_t::s8)   ||
          (src_dtype == data_type_t::s8   && dst_dtype == data_type_t::f32)  ||
          (src_dtype == data_type_t::f32  && dst_dtype == data_type_t::u8)   ||
          (src_dtype == data_type_t::u8   && dst_dtype == data_type_t::f32)  ||
          (src_dtype == data_type_t::f32  && dst_dtype == data_type_t::bf16) ||
          (src_dtype == data_type_t::bf16 && dst_dtype == data_type_t::f32)  ||
          (src_dtype == data_type_t::f32  && dst_dtype == data_type_t::f16)  ||
          (src_dtype == data_type_t::f16  && dst_dtype == data_type_t::f32)  ||
          (src_dtype == data_type_t::bf16 && dst_dtype == data_type_t::f16)  ||
          (src_dtype == data_type_t::f16  && dst_dtype == data_type_t::bf16) ||
          (src_dtype == data_type_t::f16  && dst_dtype == data_type_t::s8)   ||
          (src_dtype == data_type_t::s8   && dst_dtype == data_type_t::f16)  ||
          (src_dtype == data_type_t::f16  && dst_dtype == data_type_t::u8)   ||
          (src_dtype == data_type_t::u8   && dst_dtype == data_type_t::f16);
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
    }
    else if (strides.size() == 3) {
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
    case granularity_type_t::invalid:
      return "invalid";
    case granularity_type_t::per_tensor:
      return "per_tensor";
    case granularity_type_t::per_channel:
      return "per_channel";
    case granularity_type_t::per_token:
      return "per_token";
    case granularity_type_t::per_group:
      return "per_group";
    case granularity_type_t::mixed:
      return "mixed";
    default:
      return "unknown";
    }
  }

//==============================================================================
// Dynamic Quantization Implementation
//==============================================================================

  status_t validate_dynamic_quant_params(const void *src,
                                         const reorder_params_t &params) {
    // Check source pointer
    if (src == nullptr) {
      log_error("Dynamic quantization requires non-null source buffer");
      return status_t::failure;
    }

  // Check source data type (must be floating point for quantization)
  if (params.src_dtype != data_type_t::bf16 &&
      params.src_dtype != data_type_t::f32  &&
      params.src_dtype != data_type_t::f16) {
    log_error("Dynamic quantization only supports bf16, f32 or f16 source. Got: ",
              reorder_data_type_to_string(params.src_dtype));
    return status_t::failure;
  }

  // Check destination data type (must be int8 for quantization)
  if (params.dst_dtype != data_type_t::s8 && params.dst_dtype != data_type_t::u8) {
    log_error("Dynamic quantization only supports s8 or u8 destination. Got: ",
              reorder_data_type_to_string(params.dst_dtype));
    return status_t::failure;
  }

  // Check that scale buffer is provided (required for both symmetric and asymmetric)
  if (params.quant_params.scale.buff == nullptr) {
    log_error("Dynamic quantization requires scale buffer");
    return status_t::failure;
  }

  // Check that scale dims are provided
  if (params.quant_params.scale.dims.empty()) {
    log_error("Dynamic quantization requires scale dims to be specified");
    return status_t::failure;
  }

  // Validate scale data type — dynamic quantization writes computed scale
  // values into the user-provided buffer. f32, bf16, and f16 are accepted;
  // the output is written in the same data type as the buffer (bf16 / f16
  // are narrowed from the internal f32 scratch via float_to_bf16 /
  // float16_t::f32_to_f16_val).
  if (params.quant_params.scale.dt != data_type_t::f32  &&
      params.quant_params.scale.dt != data_type_t::bf16 &&
      params.quant_params.scale.dt != data_type_t::f16) {
    log_error("Dynamic quantization scale output buffer must be f32, bf16, or f16. Got: ",
              reorder_data_type_to_string(params.quant_params.scale.dt));
    return status_t::failure;
  }

  // Determine quantization mode based on zero_point buffer presence
  // - If zp buffer is nullptr -> symmetric quantization (scale only, zp=0)
  // - If zp buffer is provided -> asymmetric quantization (scale and zp)
  const bool is_symmetric = (params.quant_params.zero_point.buff == nullptr);
  
  if (is_symmetric) {
    // Symmetric quantization: destination must be s8
    if (params.dst_dtype != data_type_t::s8) {
      log_error("Symmetric dynamic quantization (zp=nullptr) requires s8 destination. Got: ",
                reorder_data_type_to_string(params.dst_dtype));
      return status_t::failure;
    }
  } else {
    // Asymmetric quantization: validate zp dims and data type
    if (params.quant_params.zero_point.dims.empty()) {
      log_error("Asymmetric dynamic quantization requires zero_point dims to be specified");
      return status_t::failure;
    }

    if (params.quant_params.zero_point.dt != data_type_t::s32) {
      log_error("Dynamic quantization zero_point must be s32. Got: ",
                reorder_data_type_to_string(params.quant_params.zero_point.dt));
      return status_t::failure;
    }

    // Asymmetric quantization: destination must be u8
    if (params.dst_dtype != data_type_t::u8) {
      log_error("Asymmetric dynamic quantization (zp provided) requires u8 destination. Got: ",
                reorder_data_type_to_string(params.dst_dtype));
      return status_t::failure;
    }
  }

  // Validate shape is provided
  if (!params.is_shaped()) {
    log_error("Dynamic quantization requires shape to be specified");
    return status_t::failure;
  }

  return status_t::success;
}

status_t compute_dynamic_quant_params(const void *src, const reorder_params_t &params) {
  const auto &scale_dims = params.quant_params.scale.dims;
  const auto &shape = params.src_shape;
  
  const int64_t M = params.M();
  const int64_t N = params.N();
  const int64_t batch = params.batch();

  int32_t *zp_out = static_cast<int32_t *>(params.quant_params.zero_point.buff);

  // Write helper that stores the computed f32 scale in the user-provided
  // buffer, narrowing to the user's storage dtype (bf16 or f16) when
  // requested. f32 is written directly; bf16 and f16 share uint16_t storage.
  enum class scale_kind_t { f32, bf16, f16 };
  const scale_kind_t s_kind =
      (params.quant_params.scale.dt == data_type_t::bf16) ? scale_kind_t::bf16 :
      (params.quant_params.scale.dt == data_type_t::f16)  ? scale_kind_t::f16  :
                                                            scale_kind_t::f32;
  float    *scale_out_f32 = (s_kind == scale_kind_t::f32)
                                ? static_cast<float *>(params.quant_params.scale.buff)
                                : nullptr;
  uint16_t *scale_out_u16 = (s_kind != scale_kind_t::f32)
                                ? static_cast<uint16_t *>(params.quant_params.scale.buff)
                                : nullptr;

  auto write_scale = [s_kind, scale_out_f32, scale_out_u16](int64_t idx, float val) {
    switch (s_kind) {
      case scale_kind_t::f32:  scale_out_f32[idx] = val; break;
      case scale_kind_t::bf16: scale_out_u16[idx] = float_to_bf16(val); break;
      case scale_kind_t::f16:
        // Floor-then-narrow: compute_scale_zp()'s 1e-10f f32 floor is
        // below FP16's min positive subnormal and would round to zero
        // on the narrow, breaking the subsequent static-quant
        // get_scale_value() read-back. See narrow_f32_scale_to_f16().
        scale_out_u16[idx] = common::narrow_f32_scale_to_f16(val);
        break;
    }
  };

  // Determine quantization mode based on zero_point buffer presence
  // - Symmetric: zp_out == nullptr -> scale = max(abs) / 127, zp = 0, dst = s8
  // - Asymmetric: zp_out != nullptr -> scale = (max-min) / 255, zp = round(-min/scale), dst = u8
  const bool is_symmetric = (zp_out == nullptr);
  const int nthreads = static_cast<int>(params.num_threads);

  // Determine granularity type from scale dims
  granularity_type_t granularity = get_single_granularity(scale_dims, shape);
  
  if (granularity == granularity_type_t::invalid) {
    log_error("Invalid granularity dims for dynamic quantization");
    return status_t::failure;
  }

  // For asymmetric quantization, validate zero_point dims are valid and match scale dims
  if (!is_symmetric) {
    const auto &zp_dims = params.quant_params.zero_point.dims;
    const int64_t scale_nelems = get_quant_param_nelems(scale_dims);
    const int64_t zp_nelems = get_quant_param_nelems(zp_dims);
    // get_quant_param_nelems() returns -1 on overflow/invalid dims; reject early
    if (scale_nelems < 0 || zp_nelems < 0) {
      log_error("Asymmetric dynamic quantization received invalid scale/zero_point dims");
      return status_t::failure;
    }
    // Require same number of elements and identical dims mapping for scale and zero_point
    if (scale_nelems != zp_nelems || scale_dims != zp_dims) {
      log_error("Asymmetric dynamic quantization requires scale and zero_point to have the same granularity and dims mapping");
      return status_t::failure;
    }
  }

  // Precompute typed source pointer and dtype kind once so the inner loops
  // never re-check params.src_dtype or re-cast the pointer.
  enum class src_kind_t { f32, bf16, f16 };
  const src_kind_t skind =
      params.src_dtype == data_type_t::f32  ? src_kind_t::f32  :
      params.src_dtype == data_type_t::bf16 ? src_kind_t::bf16 :
                                              src_kind_t::f16;
  const uint16_t *src_u16   = static_cast<const uint16_t *>(src);
  const float    *src_f32_p = static_cast<const float *>(src);

  auto read_src_value = [skind, src_u16, src_f32_p](int64_t idx) -> float {
    switch (skind) {
      case src_kind_t::f32:  return src_f32_p[idx];
      case src_kind_t::bf16: return bf16_to_float(src_u16[idx]);
      case src_kind_t::f16:  return common::float16_t::f16_to_f32_val(src_u16[idx]);
    }
    return 0.0f;
  };

  // Precompute stride values once to avoid per-element branching inside hot
  // loops.  When source strides are provided we capture the scalar stride
  // values; otherwise we fall back to the contiguous row-major strides
  // (batch_stride = M*N, row_stride = N, col_stride = 1).
  const bool has_strides = params.has_src_strides();
  const int64_t stride_b = has_strides && params.src_strides.size() == 3
                               ? params.src_strides[0]
                               : M * N;
  const int64_t stride_m = has_strides
                               ? (params.src_strides.size() == 3
                                      ? params.src_strides[1]
                                      : (params.src_strides.size() == 2
                                             ? params.src_strides[0]
                                             : 0))
                               : N;
  const int64_t stride_n = has_strides
                               ? params.src_strides.back()
                               : 1;

  // Branchless source-index computation using the precomputed strides.
  auto compute_src_idx = [=](int64_t b, int64_t m, int64_t n) -> int64_t {
    return b * stride_b + m * stride_m + n * stride_n;
  };

  // Helper lambda to compute scale and zero_point from min/max
  // Based on image formulas:
  // Symmetric:  scale = max(abs(A)) / 127, zp = 0
  // Asymmetric: scale = (max - min) / 255, zp = round(-min / scale)
  auto compute_scale_zp = [is_symmetric](float min_val, float max_val) 
      -> std::pair<float, int32_t> {
    float scale;
    int32_t zp;
    
    // If no finite values were observed in the scanned range, fall back to a
    // deterministic min/max (both zero). This avoids propagating sentinel
    // extremes into scale/zp computation, which would otherwise produce
    // nonsensical quantization parameters. The downstream logic already
    // treats the "all zeros" case robustly.
    if (min_val == std::numeric_limits<float>::max()
        && max_val == std::numeric_limits<float>::lowest()) {
      min_val = 0.0f;
      max_val = 0.0f;
    }
    
    if (is_symmetric) {
      // Symmetric quantization: scale = max(abs(A)) / 127, zp = 0
      float abs_max = std::max(std::abs(min_val), std::abs(max_val));
      // Handle edge case where all values are zero
      if (abs_max < 1e-10f) {
        abs_max = 1e-10f;
      }
      scale = abs_max / 127.0f;
      zp = 0;
    }
    else {
      // Asymmetric quantization: scale = (max - min) / 255, zp = round(-min / scale)
      // Handle edge case where min == max
      if (max_val <= min_val) {
        max_val = min_val + 1.0f;
      }
      scale = (max_val - min_val) / 255.0f;
      // zp = round(-min / scale), with clamping to int32_t range to avoid UB
      {
        double zp_double = std::round(static_cast<double>(-min_val) /
                                      static_cast<double>(scale));
        double int32_min = static_cast<double>(std::numeric_limits<int32_t>::min());
        double int32_max = static_cast<double>(std::numeric_limits<int32_t>::max());
        if (zp_double < int32_min) {
          zp = std::numeric_limits<int32_t>::min();
        }
        else if (zp_double > int32_max) {
          zp = std::numeric_limits<int32_t>::max();
        }
        else {
          zp = static_cast<int32_t>(zp_double);
        }
      }
    }

    // Ensure scale is not zero or denormal
    if (scale < 1e-10f) {
      scale = 1e-10f;
    }

    return {scale, zp};
  };

  //============================================================================
  // Per-Tensor Quantization
  //============================================================================
  if (granularity == granularity_type_t::per_tensor) {
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    const bool is_contiguous = (stride_n == 1 && stride_m == N
                                && stride_b == M * N);

    if (is_contiguous) {
      // Fast path: contiguous memory — flat scan avoids per-element index
      // arithmetic and enables the best possible OpenMP work distribution.
      const int64_t total = batch * M * N;
      #pragma omp parallel for num_threads(nthreads) reduction(min:min_val) reduction(max:max_val)
      for (int64_t i = 0; i < total; ++i) {
        float val = read_src_value(i);
        if (std::isfinite(val)) {
          min_val = std::min(min_val, val);
          max_val = std::max(max_val, val);
        }
      }
    }
    else {
      // Strided path: collapse(2) on (batch, M) gives good parallelism
      // while keeping the inner N loop sequential for cache locality.
      #pragma omp parallel for num_threads(nthreads) collapse(2) reduction(min:min_val) reduction(max:max_val)
      for (int64_t b = 0; b < batch; ++b) {
        for (int64_t m = 0; m < M; ++m) {
          for (int64_t n = 0; n < N; ++n) {
            float val = read_src_value(compute_src_idx(b, m, n));
            if (std::isfinite(val)) {
              min_val = std::min(min_val, val);
              max_val = std::max(max_val, val);
            }
          }
        }
      }
    }

    auto [scale, zp] = compute_scale_zp(min_val, max_val);

    write_scale(0, scale);
    if (!is_symmetric) {
      zp_out[0] = zp;
    }

    return status_t::success;
  }

  //============================================================================
  // Per-Channel Quantization (Row or Column)
  //============================================================================
  if (granularity == granularity_type_t::per_channel ||
      granularity == granularity_type_t::per_token) {
    bool is_per_row = is_per_channel_row_dims(scale_dims, shape);

    if (is_per_row) {
      const bool contiguous_batch1 =
          (batch == 1 && stride_n == 1 && stride_m == N &&
          (!has_strides || stride_b == M * N));

      // Per-token fast path: AVX-accelerated kernels exist only for
      // bf16/f32 source. For f16 source we fall through to the generic
      // scalar/strided code below (which already handles f16 via
      // read_src_value()).
      //
      // The fast-path kernels always emit f32 scales; if the caller
      // requested a narrowed scale dtype (bf16 or f16), we stage into a
      // local f32 buffer and narrow afterwards via write_scale().
      if (contiguous_batch1 && M > 0 && N > 0 && skind != src_kind_t::f16) {
        const bool is_bf16_src = (skind == src_kind_t::bf16);
        // Pick a destination buffer the AVX kernel can write f32 scales
        // into: either the caller's f32 buffer directly (zero-copy), or
        // a small staging buffer that we narrow back via write_scale.
        const bool narrow_scale = (s_kind != scale_kind_t::f32);
        std::vector<float> tmp;
        float *dst_scale_f32 = scale_out_f32;
        if (narrow_scale) {
          tmp.resize(static_cast<size_t>(M));
          dst_scale_f32 = tmp.data();
        }
        auto narrow_back = [&]() {
          if (!narrow_scale) {
            return;
          }
          for (int64_t m = 0; m < M; ++m) {
            write_scale(m, tmp[static_cast<size_t>(m)]);
          }
        };

        if (is_symmetric) {
          if (is_bf16_src) {
            dynamic_per_token_compute_scales_bf16_s8_symmetric(
                src_u16, dst_scale_f32, M, N, nthreads);
          }
          else {
            dynamic_per_token_compute_scales_f32_s8_symmetric(
                src_f32_p, dst_scale_f32, M, N, nthreads);
          }
          narrow_back();
          return status_t::success;
        }
        else if (zp_out != nullptr) {
          if (is_bf16_src) {
            dynamic_per_token_compute_scales_bf16_u8_asymmetric(
                src_u16, dst_scale_f32, zp_out, M, N, nthreads);
          }
          else {
            dynamic_per_token_compute_scales_f32_u8_asymmetric(
                src_f32_p, dst_scale_f32, zp_out, M, N, nthreads);
          }
          narrow_back();
          return status_t::success;
        }
      }

      // Per-channel-row (per-token): M values, one per row
      // This matches the "per-token" quantization from the spec
      #pragma omp parallel for num_threads(nthreads)
      for (int64_t m = 0; m < M; ++m) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();

        for (int64_t b = 0; b < batch; ++b) {
          for (int64_t n = 0; n < N; ++n) {
            int64_t idx = compute_src_idx(b, m, n);
            float val = read_src_value(idx);
            if (std::isfinite(val)) {
              min_val = std::min(min_val, val);
              max_val = std::max(max_val, val);
            }
          }
        }

        auto [scale, zp] = compute_scale_zp(min_val, max_val);
        write_scale(m, scale);
        if (!is_symmetric) {
          zp_out[m] = zp;
        }
      }
    }
    else {
      // Per-channel-col: N values, one per column
      #pragma omp parallel for num_threads(nthreads)
      for (int64_t n = 0; n < N; ++n) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();

        for (int64_t b = 0; b < batch; ++b) {
          for (int64_t m = 0; m < M; ++m) {
            int64_t idx = compute_src_idx(b, m, n);
            float val = read_src_value(idx);
            if (std::isfinite(val)) {
              min_val = std::min(min_val, val);
              max_val = std::max(max_val, val);
            }
          }
        }

        auto [scale, zp] = compute_scale_zp(min_val, max_val);
        write_scale(n, scale);
        if (!is_symmetric) {
          zp_out[n] = zp;
        }
      }
    }

    return status_t::success;
  }

  //============================================================================
  // Per-Group Quantization (Row or Column groups)
  //============================================================================
  if (granularity == granularity_type_t::per_group) {
    bool is_per_group_row = is_per_group_row_dims(scale_dims, shape);

    if (is_per_group_row) {
      // Per-group-row: G groups across rows, each group has N values
      int64_t G = get_num_groups_row(scale_dims);
      int64_t group_size = M / G;

      #pragma omp parallel for num_threads(nthreads) collapse(2)
      for (int64_t g = 0; g < G; ++g) {
        for (int64_t n = 0; n < N; ++n) {
          float min_val = std::numeric_limits<float>::max();
          float max_val = std::numeric_limits<float>::lowest();

          for (int64_t b = 0; b < batch; ++b) {
            for (int64_t m_local = 0; m_local < group_size; ++m_local) {
              int64_t m = g * group_size + m_local;
              int64_t idx = compute_src_idx(b, m, n);
              float val = read_src_value(idx);
              if (std::isfinite(val)) {
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
              }
            }
          }

          auto [scale, zp] = compute_scale_zp(min_val, max_val);
          int64_t param_idx = g * N + n;
          write_scale(param_idx, scale);
          if (!is_symmetric) {
            zp_out[param_idx] = zp;
          }
        }
      }
    }
    else {
      // Per-group-col: G groups across columns, each row has G values
      int64_t G = get_num_groups_col(scale_dims);
      int64_t group_size = N / G;

      #pragma omp parallel for num_threads(nthreads) collapse(2)
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t g = 0; g < G; ++g) {
          float min_val = std::numeric_limits<float>::max();
          float max_val = std::numeric_limits<float>::lowest();

          for (int64_t b = 0; b < batch; ++b) {
            for (int64_t n_local = 0; n_local < group_size; ++n_local) {
              int64_t n = g * group_size + n_local;
              int64_t idx = compute_src_idx(b, m, n);
              float val = read_src_value(idx);
              if (std::isfinite(val)) {
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
              }
            }
          }

          auto [scale, zp] = compute_scale_zp(min_val, max_val);
          int64_t param_idx = m * G + g;
          write_scale(param_idx, scale);
          if (!is_symmetric) {
            zp_out[param_idx] = zp;
          }
        }
      }
    }

    return status_t::success;
  }

  log_error("Unsupported granularity for dynamic quantization");
  return status_t::failure;
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl

