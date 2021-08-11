/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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
*******************************************************************************/

#ifndef COMMON_GEMM_TYPES_HPP
#define COMMON_GEMM_TYPES_HPP

#include <assert.h>

#include "zendnn_types.h"

namespace zendnn {
namespace impl {

enum transpose_t { zendnn_notrans, zendnn_trans };

namespace transpose {
const transpose_t notrans = zendnn_notrans;
const transpose_t trans = zendnn_trans;
} // namespace transpose

enum offsetc_t { zendnn_fixed, zendnn_column, zendnn_row };

namespace offsetc {
const offsetc_t fixed = zendnn_fixed;
const offsetc_t column = zendnn_column;
const offsetc_t row = zendnn_row;
} // namespace offsetc

/** A descriptor for a matrix multiplication (gemm) operation */
struct zendnn_gemm_desc_t {
    /* To make the interface consistent, the descriptor represent the
     * GEMM operation in row major */

    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #zendnn_gemm. */
    zendnn_primitive_kind_t primitive_kind;
    zendnn_memory_desc_t a_desc;
    zendnn_memory_desc_t b_desc;
    zendnn_memory_desc_t c_desc;
    zendnn_memory_desc_t bias_desc;
    /** Type for accumulating A*B. */
    zendnn_data_type_t acc_type;

    // These accessors are to be used by the GEMM implementation
    // Because the GEMM implementation currently assumes column major
    // These accessors return data in column major fashion

    inline bool is_batched() const { return c_desc.ndims >= 3; }

    // Simplified accessors that comply to GEMM API
    transpose_t get_trans(zendnn_memory_desc_t md) const {
        return md.format_desc.blocking.strides[md.ndims - 1] != 1
                ? transpose::trans
                : transpose::notrans;
    }
    transpose_t transa() const { return get_trans(b_desc); };
    transpose_t transb() const { return get_trans(a_desc); };
    zendnn_dim_t batch() const {
        // if ndims < 3, it should return 1
        int64_t batch = 1;
        for (int i = 0; i < c_desc.ndims - 2; ++i)
            batch *= c_desc.dims[i];
        return batch;
    }

    /** Number of rows of C. */
    zendnn_dim_t m() const { return c_desc.dims[c_desc.ndims - 1]; }
    /** Number of columns of C. */
    zendnn_dim_t n() const { return c_desc.dims[c_desc.ndims - 2]; }
    /** Size of inner dimension shared between A and B. */
    zendnn_dim_t k() const { return a_desc.dims[a_desc.ndims - 1]; }

    /** Stride between 2 matrices A in a batch. */
    zendnn_dim_t stride_a() const {
        return b_desc.format_desc.blocking.strides[0];
    };
    /** Stride between 2 matrices B in a batch. */
    zendnn_dim_t stride_b() const {
        return a_desc.format_desc.blocking.strides[0];
    };
    /** Stride between 2 matrices C in a batch. */
    zendnn_dim_t stride_c() const {
        return c_desc.format_desc.blocking.strides[0];
    };

    // This assumes that one of the dimensions has strides 1
    zendnn_dim_t get_ld(zendnn_memory_desc_t md) const {
        auto strides = md.format_desc.blocking.strides;
        assert(strides[md.ndims - 1] == 1 || strides[md.ndims - 2] == 1);
        return strides[md.ndims - 1] != 1 ? strides[md.ndims - 1]
                                          : strides[md.ndims - 2];
    }
    /** Leading dimension of A. */
    zendnn_dim_t lda() const { return get_ld(b_desc); }
    /** Leading dimension of B. */
    zendnn_dim_t ldb() const { return get_ld(a_desc); }
    /** Leading dimension of C. */
    zendnn_dim_t ldc() const { return get_ld(c_desc); }

    /** Type of matrix A. */
    zendnn_data_type_t a_type() const { return b_desc.data_type; }
    /** Type of matrix B. */
    zendnn_data_type_t b_type() const { return a_desc.data_type; }
    /** Type of matrix C. */
    zendnn_data_type_t c_type() const { return c_desc.data_type; }
    /** Type of bias. */
    zendnn_data_type_t bias_type() const { return bias_desc.data_type; }
    /** Type of bias. */
    int bias_mask() const {
        assert(bias_desc.ndims <= 3);
        int mask = 0;
        // TODO: update the mask for batched dimension if we start
        // supporting more batch dimensions
        if (is_batched()) mask |= (bias_desc.dims[0] > 1) ? 1 << 0 : 0;

        // because the bias mask is in row major, we have to convert
        // to col major here by swapping two last dimensions
        int m_idx = is_batched();
        mask |= (bias_desc.dims[m_idx] > 1) ? 1 << (bias_desc.ndims - m_idx)
                                            : 0;
        mask |= (bias_desc.dims[m_idx + 1] > 1)
                ? 1 << (bias_desc.ndims - (m_idx + 1))
                : 0;
        return mask;
    }
};

} // namespace impl
} // namespace zendnn

#endif // COMMON_GEMM_TYPES_HPP
