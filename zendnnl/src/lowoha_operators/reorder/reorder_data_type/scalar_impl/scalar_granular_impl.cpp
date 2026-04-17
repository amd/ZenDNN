/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "lowoha_operators/reorder/reorder_data_type/reorder_dtype_dispatch.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_utils.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace reorder {

using zendnnl::lowoha::matmul::zendnnl_parallel_for;

/**
 * @brief Execute element-wise quantization for 2D matrix (non-per-tensor granularity)
 * 
 * Handles per-channel, per-group, and mixed quantization where each element
 * may use a different scale/zero-point based on its position.
 */
void reorder_granular_scaler_impl_2d(const void *src, void *dst,
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
void reorder_granular_scaler_impl_3d(const void *src, void *dst,
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

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
