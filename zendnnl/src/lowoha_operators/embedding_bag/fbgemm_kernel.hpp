/********************************************************************************
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

#ifndef _FBGEMM_KERNEL_HPP
#define _FBGEMM_KERNEL_HPP

#include <vector>
#include <cstdint>
#include "lowoha_embag_common.hpp"
#include "../matmul/lowoha_matmul_utils.hpp"
#if ZENDNNL_DEPENDS_FBGEMM
  #include "fbgemm/FbgemmEmbedding.h"
#endif

namespace zendnnl {
namespace lowoha {
namespace embag {

using namespace zendnnl::lowoha::matmul;

#if ZENDNNL_DEPENDS_FBGEMM
// Scale/bias size constant for quantized embeddings
constexpr int SCALE_BIAS_SIZE = 8;

/**
 * @brief Helper template to invoke FBGEMM embedding kernel
 *
 * This template function dispatches to the appropriate FBGEMM kernel based on
 * whether the input is quantized or not.
 *
 * @tparam IsQuantized Whether to use quantized (NBit) kernel
 * @tparam InType Input embedding table data type (ignored for quantized)
 * @tparam IndexType Index data type (int32_t or int64_t)
 * @tparam OffsetType Offset data type (int32_t or int64_t)
 * @tparam OutType Output data type
 *
 * @param table Pointer to embedding table data
 * @param indices Pointer to indices array
 * @param offsets Pointer to offsets array
 * @param weights Pointer to per-sample weights (can be nullptr)
 * @param dst Pointer to output destination
 * @param params Embedding bag parameters
 * @param bit_rate Bit rate for quantized types (4 for int4, 8 for int8)
 * @param is_bf16_in Whether input is bf16 (for non-quantized types)
 * @param is_bf16_out Whether output is bf16
 */
template <
  bool IsQuantized,
  typename InType,
  typename IndexType,
  typename OffsetType,
  typename OutType>
static void invoke_fbgemm_kernel(
  const void *table,
  const void *indices,
  const void *offsets,
  const float *weights,
  void *dst,
  const embag_params_t &params,
  int bit_rate,
  bool is_bf16_in,
  bool is_bf16_out) {

  const int64_t embedding_dim = static_cast<int64_t>(params.embedding_dim);
  const int64_t num_rows = static_cast<int64_t>(params.num_embeddings);
  const int64_t indices_size = static_cast<int64_t>(params.num_indices);
  const int64_t batch_size = static_cast<int64_t>(params.num_bags);
  const int64_t output_stride = static_cast<int64_t>(params.dst_stride);
  const bool use_weight = params.is_weights;
  const bool normalize_by_lengths = false;
  const bool scale_bias_last = true;
  constexpr bool prefetch = true;
  constexpr bool is_wt_positional = false;
  constexpr bool use_offsets = true;

  // Prepare offsets - FBGEMM requires last offset to be included
  OffsetType *fbgemm_offsets = nullptr;

  if (params.include_last_offset==0) {
    fbgemm_offsets = new OffsetType[batch_size+1];
    memcpy(fbgemm_offsets, static_cast<const OffsetType *>(offsets),
           batch_size * sizeof(OffsetType));
    fbgemm_offsets[batch_size]=indices_size;
  }
  else {
    fbgemm_offsets = const_cast<OffsetType *>(static_cast<const OffsetType *>
                     (offsets));
  }

  const IndexType *indices_ptr = static_cast<const IndexType *>(indices);
  OutType *dst_ptr = static_cast<OutType *>(dst);

  if constexpr(IsQuantized) {
    // Generate FBGEMM NBit kernel for quantized types (int4/int8)
    auto kernel =
      fbgemm::GenerateEmbeddingSpMDMNBitWithStrides<IndexType, OffsetType, OutType>(
        bit_rate,
        embedding_dim,
        use_weight,
        normalize_by_lengths,
        prefetch ? 16 : 0,
        is_wt_positional,
        use_offsets,
        output_stride,
        -1, /* input_stride - use default */
        scale_bias_last,
        is_bf16_out);

    const uint8_t *table_ptr = static_cast<const uint8_t *>(table);
    zendnnl_parallel_for(0, batch_size, 1, [&](int start_idx, int end_idx) {
      kernel(
        /*output_size=*/end_idx - start_idx,
        /*index_size=*/indices_size,
        /*data_size=*/num_rows,
        /*input=*/table_ptr,
        /*indices=*/&indices_ptr[fbgemm_offsets[start_idx]],
        /*offsets=*/&fbgemm_offsets[start_idx],
        /*weights=*/weights ? &weights[fbgemm_offsets[start_idx]] : nullptr,
        /*out=*/&dst_ptr[start_idx * output_stride]);
    });
  }
  else {
    // Generate FBGEMM kernel for non-quantized types (fp32/bf16)
    auto kernel =
      fbgemm::GenerateEmbeddingSpMDMWithStrides<InType, IndexType, OffsetType, OutType>
      (
        embedding_dim,
        use_weight,
        normalize_by_lengths,
        prefetch ? 16 : 0,
        is_wt_positional,
        use_offsets,
        output_stride,
        -1,    /* input_stride - use default */
        true,  /* scale_bias_last */
        false, /* no_bag */
        is_bf16_out,
        is_bf16_in);

    const InType *table_ptr = static_cast<const InType *>(table);
    zendnnl_parallel_for(0, batch_size, 1, [&](int start_idx, int end_idx) {
      kernel(
        /*output_size=*/end_idx - start_idx,
        /*index_size=*/indices_size,
        /*data_size=*/num_rows,
        /*input=*/table_ptr,
        /*indices=*/&indices_ptr[fbgemm_offsets[start_idx]],
        /*offsets=*/&fbgemm_offsets[start_idx],
        /*weights=*/weights ? &weights[fbgemm_offsets[start_idx]] : nullptr,
        /*out=*/&dst_ptr[start_idx * output_stride]);
    });
  }

  if (params.include_last_offset==0) {
    delete[] fbgemm_offsets;
  }
}

static inline bool can_use_fbgemm(const embag_params_t &params) {
  // TODO: Explore the feasibility of using FBGEMM for other algorithms and data types.
  return (params.algo == embag_algo_t::sum &&
          params.dtypes.table != data_type_t::s8 &&
          params.dtypes.table != data_type_t::s4 &&
          params.fp16_scale_bias == true);
}
#endif
} // namespace embag
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_EMBAG_FBGEMM_KERNELS_HPP

