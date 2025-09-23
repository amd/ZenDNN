/********************************************************************************
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

#ifndef _LOWOHA_MATMUL_HPP
#define _LOWOHA_MATMUL_HPP

#include <omp.h>
#include "operators/matmul/matmul_context.hpp"
#include "memory/memory_utils.hpp"

#define ENABLE_ZENDNNL_PARALLEL_FOR 1

namespace zendnnl {
namespace lowoha {

using namespace zendnnl::common;
using namespace zendnnl::ops;

struct data_types {
  data_type_t src;
  data_type_t wei;
  data_type_t dst;
  data_type_t bias;
  data_type_t compute;

  /**
   * @brief Default constructor for `data_types`.
   *
   * Initializes all data types to `none` to ensure safe usage.
   */
  data_types() : src(data_type_t::none), wei(data_type_t::none), dst(data_type_t::none),
                 bias(data_type_t::none), compute(data_type_t::none) {}
};

struct postop {
  post_op_type_t po_type;
  void *buff;
  data_type_t dtype;
  std::vector<int> dims;

  /**
   * @brief Default constructor for `postop`.
   *
   * Initializes the post-op type to `none`, buffer to `nullptr`,
   * data type to `none`, and creates an empty dims vector.
   */
  postop() : po_type(post_op_type_t::none), buff(nullptr), dtype(data_type_t::none), dims() {}
};

/**
 * @struct lowoha_quantization_params_t
 * @brief A structure to encapsulate scale and zero-point information for quantized operations.
 *
 * This structure is used to store the scale and zero-point parameters for both the source
 * and weight tensors in quantized operations. It contains an inner structure `quant_t` to
 * represent individual scale or zero-point data, and the outer structure aggregates these
 * for the source and weight tensors.
 *
 * @details
 * The `lowoha_quantization_params_t` structure is designed to handle the following:
 * - Scale values for the source and weight tensors.
 * - Zero-point values for the source and weight tensors.
 * - Data type and size information for each scale and zero-point.
 *
 * The structure is initialized with default values to ensure safe usage.
 */
struct lowoha_quantization_params_t {
  /**
   * @struct quant_t
   * @brief A nested structure to represent individual scale or zero-point data.
   *
   * This inner structure contains a pointer to the data buffer, the data type,
   * and the size of the buffer. It is used to represent scale or zero-point
   * information for a single tensor.
   */
  struct quant_t {
    const void *buff;    /**< Pointer to the buffer holding scale or
                              zero-point data. */
    data_type_t dt;      /**< Data type of the buffer (e.g., float, int32_t). */
    size_t size;         /**< Size of the buffer in bytes. */
    /**
     * @brief Default constructor for `quant_t`.
     *
     * Initializes the buffer pointer to `nullptr`, the data type to `none`,
     * and the size to `0`.
     */
    quant_t() : buff(nullptr), dt(data_type_t::none), size(0) {}
  };
  quant_t src_scale;  /**< Scale information for the source tensor. */
  quant_t wei_scale;  /**< Scale information for the weight tensor. */
  quant_t dst_scale;  /**< Scale information for the destination tensor. */
  quant_t src_zp;     /**< Zero-point information for the source tensor. */
  quant_t wei_zp;     /**< Zero-point information for the weight tensor. */
  quant_t dst_zp;     /**< Zero-point information for the destination tensor. */
  /**
   * @brief Default constructor for `lowoha_quantization_params_t`.
   *
   * Initializes all members (`src_scale`, `wei_scale`, `dst_scale`, `src_zp`, `wei_zp`, `dst_zp`)
   * using the default constructor of `quant_t`.
   */
  lowoha_quantization_params_t() : src_scale(), wei_scale(), dst_scale(), src_zp(), wei_zp(), dst_zp() {}
};

struct lowoha_post_op {
  std::vector<postop> postop_;
};

void matmul_direct(const void *src, const void *weight, void *dst, void *bias,
                   float alpha, float beta, int M, int N, int K, bool transA, bool transB, int lda,
                   int ldb, int ldc, data_types &dtypes, lowoha_post_op post_op,
                   const lowoha_quantization_params_t &quant_params = lowoha_quantization_params_t(),
                   int Batch_A = 1, int Batch_B = 1);

} // namespace lowoha
} // namespace zendnnl

#endif

