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
namespace reorder {

using namespace zendnnl::error_handling;
using namespace zendnnl::profile;
using zendnnl::lowoha::matmul::zendnnl_parallel_for;

/**
 * @brief Execute reorder kernel based on selected algorithm (per-tensor only)
 * 
 * This function is only called when granularity is per-tensor, so it uses
 * a single scale/zero_point value for all elements, enabling efficient
 * vectorized processing via AVX-512 kernels.
 */
static void reorder_dynamic_impl(const void *src, void *dst, size_t nelems,
                                   const reorder_params_t &params,
                                   reorder_algo_t algo) {
  // Extract single scale/zp values (per-tensor quantization only)
  const float scale = get_scale_value(params.quant_params.scale);
  const int zero_point = get_zero_point_value(params.quant_params.zero_point);

  // BF16 -> INT8 (Quantization)
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::s8) {
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
  if (params.src_dtype == data_type_t::s8 && params.dst_dtype == data_type_t::bf16) {
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
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::u8) {
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
  if (params.src_dtype == data_type_t::u8 && params.dst_dtype == data_type_t::bf16) {
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

  // FP32 -> INT8 (Quantization)
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::s8) {
    const float *src_f32 = static_cast<const float *>(src);
    int8_t *dst_int8 = static_cast<int8_t *>(dst);

    if (algo == reorder_algo_t::native) {
      quantize_f32_to_int8_avx512(src_f32, dst_int8, nelems, scale, zero_point);
    }
    else {
      quantize_f32_to_int8_ref(src_f32, dst_int8, nelems, scale, zero_point);
    }
    return;
  }

  // INT8 -> FP32 (Dequantization)
  if (params.src_dtype == data_type_t::s8 && params.dst_dtype == data_type_t::f32) {
    const int8_t *src_int8 = static_cast<const int8_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);

    if (algo == reorder_algo_t::native) {
      dequantize_int8_to_f32_avx512(src_int8, dst_f32, nelems, scale, zero_point);
    }
    else {
      dequantize_int8_to_f32_ref(src_int8, dst_f32, nelems, scale, zero_point);
    }
    return;
  }

  // FP32 -> UINT8 (Quantization)
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::u8) {
    const float *src_f32 = static_cast<const float *>(src);
    uint8_t *dst_uint8 = static_cast<uint8_t *>(dst);

    if (algo == reorder_algo_t::native) {
      quantize_f32_to_uint8_avx512(src_f32, dst_uint8, nelems, scale, zero_point);
    }
    else {
      quantize_f32_to_uint8_ref(src_f32, dst_uint8, nelems, scale, zero_point);
    }
    return;
  }

  // UINT8 -> FP32 (Dequantization)
  if (params.src_dtype == data_type_t::u8 && params.dst_dtype == data_type_t::f32) {
    const uint8_t *src_uint8 = static_cast<const uint8_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);

    if (algo == reorder_algo_t::native) {
      dequantize_uint8_to_f32_avx512(src_uint8, dst_f32, nelems, scale, zero_point);
    }
    else {
      dequantize_uint8_to_f32_ref(src_uint8, dst_f32, nelems, scale, zero_point);
    }
    return;
  }

  // FP32 -> BF16 (Type conversion with optional scaling)
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::bf16) {
    const float *src_f32 = static_cast<const float *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);

    if (algo == reorder_algo_t::native) {
      convert_f32_to_bf16_avx512(src_f32, dst_bf16, nelems, scale, zero_point);
    }
    else {
      convert_f32_to_bf16_ref(src_f32, dst_bf16, nelems, scale, zero_point);
    }
    return;
  }

  // BF16 -> FP32 (Type conversion with optional scaling)
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::f32) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);

    if (algo == reorder_algo_t::native) {
      convert_bf16_to_f32_avx512(src_bf16, dst_f32, nelems, scale, zero_point);
    }
    else {
      convert_bf16_to_f32_ref(src_bf16, dst_f32, nelems, scale, zero_point);
    }
    return;
  }
}

/**
 * @brief Execute element-wise quantization for 2D matrix (non-per-tensor granularity)
 * 
 * Handles per-channel, per-group, and mixed quantization where each element
 * may use a different scale/zero-point based on its position.
 */
static void reorder_granular_scaler_impl_2d(const void *src, void *dst,
                                         const reorder_params_t &params) {
  const int64_t M = params.M();
  const int64_t N = params.N();
  const int64_t total_work = M * N;
  
  // BF16 -> INT8
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::s8) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    int8_t *dst_int8 = static_cast<int8_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_int8[work_idx] = quantize_bf16_to_s8_scalar(src_bf16[work_idx], scale, zp);
      }
    });
    return;
  }
  
  // BF16 -> UINT8
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::u8) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    uint8_t *dst_uint8 = static_cast<uint8_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_uint8[work_idx] = quantize_bf16_to_u8_scalar(src_bf16[work_idx], scale, zp);
      }
    });
    return;
  }
  
  // INT8 -> BF16
  if (params.src_dtype == data_type_t::s8 && params.dst_dtype == data_type_t::bf16) {
    const int8_t *src_int8 = static_cast<const int8_t *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_bf16[work_idx] = dequantize_s8_to_bf16_scalar(src_int8[work_idx], scale, zp);
      }
    });
    return;
  }
  
  // UINT8 -> BF16
  if (params.src_dtype == data_type_t::u8 && params.dst_dtype == data_type_t::bf16) {
    const uint8_t *src_uint8 = static_cast<const uint8_t *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_bf16[work_idx] = dequantize_u8_to_bf16_scalar(src_uint8[work_idx], scale, zp);
      }
    });
    return;
  }

  // FP32 -> INT8
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::s8) {
    const float *src_f32 = static_cast<const float *>(src);
    int8_t *dst_int8 = static_cast<int8_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_int8[work_idx] = quantize_f32_to_s8_scalar(src_f32[work_idx], scale, zp);
      }
    });
    return;
  }
  
  // FP32 -> UINT8
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::u8) {
    const float *src_f32 = static_cast<const float *>(src);
    uint8_t *dst_uint8 = static_cast<uint8_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_uint8[work_idx] = quantize_f32_to_u8_scalar(src_f32[work_idx], scale, zp);
      }
    });
    return;
  }
  
  // INT8 -> FP32
  if (params.src_dtype == data_type_t::s8 && params.dst_dtype == data_type_t::f32) {
    const int8_t *src_int8 = static_cast<const int8_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_f32[work_idx] = dequantize_s8_to_f32_scalar(src_int8[work_idx], scale, zp);
      }
    });
    return;
  }
  
  // UINT8 -> FP32
  if (params.src_dtype == data_type_t::u8 && params.dst_dtype == data_type_t::f32) {
    const uint8_t *src_uint8 = static_cast<const uint8_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_f32[work_idx] = dequantize_u8_to_f32_scalar(src_uint8[work_idx], scale, zp);
      }
    });
    return;
  }

  // FP32 -> BF16 (with optional scaling)
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::bf16) {
    const float *src_f32 = static_cast<const float *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_bf16[work_idx] = convert_f32_to_bf16_scalar(src_f32[work_idx], scale, zp);
      }
    });
    return;
  }

  // BF16 -> FP32 (with optional scaling)
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::f32) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t i = work_idx / N;
        int64_t j = work_idx % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_f32[work_idx] = convert_bf16_to_f32_scalar(src_bf16[work_idx], scale, zp);
      }
    });
    return;
  }
}

/**
 * @brief Execute element-wise quantization for 3D batched matrix (non-per-tensor granularity)
 * 
 * Handles per-channel, per-group, and mixed quantization where each element
 * may use a different scale/zero-point based on its position.
 */
static void reorder_granular_scaler_impl_3d(const void *src, void *dst,
                                         const reorder_params_t &params) {
  const int64_t batch = params.batch();
  const int64_t M = params.M();
  const int64_t N = params.N();
  const int64_t matrix_size = M * N;
  const int64_t total_work = batch * matrix_size;
  
  // BF16 -> INT8
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::s8) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    int8_t *dst_int8 = static_cast<int8_t *>(dst);

    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_int8[work_idx] = quantize_bf16_to_s8_scalar(src_bf16[work_idx], scale, zp);
      }
    });
    return;
  }
  
  // BF16 -> UINT8
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::u8) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    uint8_t *dst_uint8 = static_cast<uint8_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_uint8[work_idx] = quantize_bf16_to_u8_scalar(src_bf16[work_idx], scale, zp);
      }
    });
    return;
  }
  
  // INT8 -> BF16
  if (params.src_dtype == data_type_t::s8 && params.dst_dtype == data_type_t::bf16) {
    const int8_t *src_int8 = static_cast<const int8_t *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_bf16[work_idx] = dequantize_s8_to_bf16_scalar(src_int8[work_idx], scale, zp);
      }
    });
    return;
  }
  
  // UINT8 -> BF16
  if (params.src_dtype == data_type_t::u8 && params.dst_dtype == data_type_t::bf16) {
    const uint8_t *src_uint8 = static_cast<const uint8_t *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_bf16[work_idx] = dequantize_u8_to_bf16_scalar(src_uint8[work_idx], scale, zp);
      }
    });
    return;
  }

  // FP32 -> INT8
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::s8) {
    const float *src_f32 = static_cast<const float *>(src);
    int8_t *dst_int8 = static_cast<int8_t *>(dst);

    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_int8[work_idx] = quantize_f32_to_s8_scalar(src_f32[work_idx], scale, zp);
      }
    });
    return;
  }
  
  // FP32 -> UINT8
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::u8) {
    const float *src_f32 = static_cast<const float *>(src);
    uint8_t *dst_uint8 = static_cast<uint8_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_uint8[work_idx] = quantize_f32_to_u8_scalar(src_f32[work_idx], scale, zp);
      }
    });
    return;
  }
  
  // INT8 -> FP32
  if (params.src_dtype == data_type_t::s8 && params.dst_dtype == data_type_t::f32) {
    const int8_t *src_int8 = static_cast<const int8_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_f32[work_idx] = dequantize_s8_to_f32_scalar(src_int8[work_idx], scale, zp);
      }
    });
    return;
  }
  
  // UINT8 -> FP32
  if (params.src_dtype == data_type_t::u8 && params.dst_dtype == data_type_t::f32) {
    const uint8_t *src_uint8 = static_cast<const uint8_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_f32[work_idx] = dequantize_u8_to_f32_scalar(src_uint8[work_idx], scale, zp);
      }
    });
    return;
  }

  // FP32 -> BF16 (with optional scaling)
  if (params.src_dtype == data_type_t::f32 && params.dst_dtype == data_type_t::bf16) {
    const float *src_f32 = static_cast<const float *>(src);
    uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_bf16[work_idx] = convert_f32_to_bf16_scalar(src_f32[work_idx], scale, zp);
      }
    });
    return;
  }

  // BF16 -> FP32 (with optional scaling)
  if (params.src_dtype == data_type_t::bf16 && params.dst_dtype == data_type_t::f32) {
    const uint16_t *src_bf16 = static_cast<const uint16_t *>(src);
    float *dst_f32 = static_cast<float *>(dst);
    
    zendnnl_parallel_for(0, total_work, 1, [&](int64_t start_idx, int64_t end_idx) {
      for (int64_t work_idx = start_idx; work_idx < end_idx; ++work_idx) {
        int64_t elem_in_matrix = work_idx % matrix_size;
        int64_t i = elem_in_matrix / N;
        int64_t j = elem_in_matrix % N;
        size_t scale_idx = get_quant_param_index(params.quant_params.scale.dims, params.src_shape, M, N, i, j);
        size_t zp_idx = get_quant_param_index(params.quant_params.zero_point.dims, params.src_shape, M, N, i, j);
        float scale = get_scale_value(params.quant_params.scale, scale_idx);
        int zp = get_zero_point_value(params.quant_params.zero_point, zp_idx);
        dst_f32[work_idx] = convert_bf16_to_f32_scalar(src_bf16[work_idx], scale, zp);
      }
    });
    return;
  }
}

/**
 * @brief Helper function to get element size for a given data type
 */
static inline size_t get_dtype_size(data_type_t dtype) {
  switch (dtype) {
    case data_type_t::f32:  return sizeof(float);
    case data_type_t::bf16: return sizeof(uint16_t);
    case data_type_t::s8:   return sizeof(int8_t);
    case data_type_t::u8:   return sizeof(uint8_t);
    default:                return 1;
  }
}

/**
 * @brief Parallel reorder execution for all memory layouts
 * 
 * This function handles:
 * - Contiguous memory (optimized vectorized path)
 * - 1D strided arrays
 * - 2D strided matrices
 * - 3D strided batched matrices
 */
static void reorder_wrapper(const void *src, void *dst, size_t nelems,
                                      const reorder_params_t &params,
                                      reorder_algo_t algo) {
  const size_t src_elem_size = get_dtype_size(params.src_dtype);
  const size_t dst_elem_size = get_dtype_size(params.dst_dtype);

  // Check granularity type
  granularity_type_t granularity = get_granularity_type(params);
  
  // Per-channel, per-group, or mixed granularity requires element-wise processing
  if (granularity != granularity_type_t::per_tensor) {
    if (params.is_3d()) {
      reorder_granular_scaler_impl_3d(src, dst, params);
    } else {
      reorder_granular_scaler_impl_2d(src, dst, params);
    }
    return;
  }

  // Fast path: Contiguous memory with per-tensor quantization
  if (!params.has_src_strides() || params.is_src_contiguous()) {
    constexpr int64_t grain_size = 1024;  // Minimum elements per thread
    zendnnl_parallel_for(0, static_cast<int64_t>(nelems), grain_size,
      [&](int64_t begin, int64_t end) {
        const uint8_t *src_ptr = static_cast<const uint8_t *>(src) + begin * src_elem_size;
        uint8_t *dst_ptr = static_cast<uint8_t *>(dst) + begin * dst_elem_size;
        size_t thread_nelems = static_cast<size_t>(end - begin);
        reorder_dynamic_impl(src_ptr, dst_ptr, thread_nelems, params, algo);
      });
    return;
  }

  // Strided access paths
  const auto &strides = params.src_strides;
  const size_t shape_size = params.src_shape.size();

  // 1D strided (stride != 1, otherwise would be contiguous)
  if (shape_size == 1) {
    const int64_t stride = strides[0];
    #pragma omp parallel for num_threads(params.num_threads)
    for (int64_t i = 0; i < static_cast<int64_t>(nelems); ++i) {
      const uint8_t *src_ptr = static_cast<const uint8_t *>(src) + i * stride * src_elem_size;
      uint8_t *dst_ptr = static_cast<uint8_t *>(dst) + i * dst_elem_size;
      reorder_dynamic_impl(src_ptr, dst_ptr, 1, params, algo);
    }
    return;
  }

  // 2D strided
  if (shape_size == 2) {
    const int64_t M = params.M();
    const int64_t N = params.N();
    const int64_t stride_M = strides[0];
    const int64_t stride_N = strides[1];

    // Optimized path: if stride_N == 1, rows are contiguous - use AVX512 per row
    if (stride_N == 1) {
      #pragma omp parallel for num_threads(params.num_threads)
      for (int64_t i = 0; i < M; ++i) {
        const uint8_t *src_ptr = static_cast<const uint8_t *>(src) + i * stride_M * src_elem_size;
        uint8_t *dst_ptr = static_cast<uint8_t *>(dst) + i * N * dst_elem_size;
        reorder_dynamic_impl(src_ptr, dst_ptr, static_cast<size_t>(N), params, algo);
      }
    } else {
      // Fallback: element-by-element for non-contiguous rows
      #pragma omp parallel for collapse(2) num_threads(params.num_threads)
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          const int64_t src_offset = i * stride_M + j * stride_N;
          const int64_t dst_offset = i * N + j;

          const uint8_t *src_ptr = static_cast<const uint8_t *>(src) + src_offset * src_elem_size;
          uint8_t *dst_ptr = static_cast<uint8_t *>(dst) + dst_offset * dst_elem_size;

          reorder_dynamic_impl(src_ptr, dst_ptr, 1, params, algo);
        }
      }
    }
    return;
  }

  // 3D strided (BMM)
  if (shape_size == 3) {
    const int64_t batch = params.batch();
    const int64_t M = params.M();
    const int64_t N = params.N();
    const int64_t matrix_size = M * N;
    const int64_t stride_batch = strides[0];
    const int64_t stride_M = strides[1];
    const int64_t stride_N = strides[2];

    // Optimized path: if stride_N == 1, rows are contiguous - use AVX512 per row
    if (stride_N == 1) {
      #pragma omp parallel for collapse(2) num_threads(params.num_threads)
      for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < M; ++i) {
          const int64_t src_offset = b * stride_batch + i * stride_M;
          const int64_t dst_offset = b * matrix_size + i * N;

          const uint8_t *src_ptr = static_cast<const uint8_t *>(src) + src_offset * src_elem_size;
          uint8_t *dst_ptr = static_cast<uint8_t *>(dst) + dst_offset * dst_elem_size;

          reorder_dynamic_impl(src_ptr, dst_ptr, static_cast<size_t>(N), params, algo);
        }
      }
    } else {
      // Fallback: element-by-element for non-contiguous rows
      #pragma omp parallel for collapse(3) num_threads(params.num_threads)
      for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < M; ++i) {
          for (int64_t j = 0; j < N; ++j) {
            const int64_t src_offset = b * stride_batch + i * stride_M + j * stride_N;
            const int64_t dst_offset = b * matrix_size + i * N + j;

            const uint8_t *src_ptr = static_cast<const uint8_t *>(src) + src_offset * src_elem_size;
            uint8_t *dst_ptr = static_cast<uint8_t *>(dst) + dst_offset * dst_elem_size;

            reorder_dynamic_impl(src_ptr, dst_ptr, 1, params, algo);
          }
        }
      }
    }
    return;
  }
}

status_t reorder_direct(const void *src, void *dst,
                         reorder_params_t params) {
  // Compute nelems from shape - shape is mandatory
  if (!params.is_shaped()) {
    log_error("Shape must be provided. "
              "For 1D arrays, use shape = {size}. "
              "For 2D matrices, use shape = {M, N}. "
              "For 3D batched, use shape = {batch, M, N}");
    return status_t::failure;
  }

  // Compute nelems from shape
  const size_t nelems = static_cast<size_t>(params.nelems());

  // Validate inputs and parameters
  if (validate_reorder_inputs(src, dst, nelems, params) != status_t::success) {
    return status_t::failure;
  }

  // Select algorithm
  reorder_algo_t algo = select_reorder_algo(params, nelems);

  // Create profiler instance for timing
  profiler_t profiler;
  bool is_profile = is_profile_enabled();

  // Build log string for API and profile logging
  [[maybe_unused]] std::ostringstream ss;
  if (apilog_info_enabled() || is_profile) {
    float scale_val = get_scale_value(params.quant_params.scale);
    int zp_val = get_zero_point_value(params.quant_params.zero_point);
    ss << "LOWOHA reorder_direct: nelems=" << nelems
       << ", src_dtype=" << reorder_data_type_to_string(params.src_dtype)
       << ", dst_dtype=" << reorder_data_type_to_string(params.dst_dtype)
       << ", scale=" << scale_val
       << ", zero_point=" << zp_val
       << ", algo=" << reorder_algo_to_string(algo)
       << ", granularity=" << granularity_to_string(get_granularity_type(params));

    // Add stride information to log
    if (params.has_src_strides()) {
      ss << ", strides=[";
      for (size_t i = 0; i < params.src_strides.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << params.src_strides[i];
      }
      ss << "]";
    }

    if (apilog_info_enabled()) {
      apilog_info(ss.str());
    }
  }

  // Start profiling timer
  if (is_profile) {
    profiler.tbp_start();
  }

  params.num_threads = params.num_threads > 0 ? params.num_threads :
                       omp_get_max_threads();

  reorder_threadlimit thread_guard(params.num_threads);
  // Execute reorder (handles all cases: contiguous, 1D/2D/3D strided)
  reorder_wrapper(src, dst, nelems, params, algo);

  // Stop profiling timer and log
  if (is_profile) {
    profiler.tbp_stop();
    profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(),
                       profiler.get_res_str());
  }

  return status_t::success;
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
