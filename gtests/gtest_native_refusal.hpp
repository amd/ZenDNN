/********************************************************************************
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

#ifndef _GTEST_NATIVE_REFUSAL_HPP
#define _GTEST_NATIVE_REFUSAL_HPP

#include "lowoha_operators/matmul/lowoha_matmul.hpp"

/** @brief matmul_direct with its arguments.  When the native BRGEMM takes
 *         the call and does not serve it (native_brgemm_refusal), expects
 *         matmul_direct to return that refusal's status, to leave dst
 *         untouched and to log an error line naming the reason, and records
 *         the call for native_refusal_verified.
 *  @return matmul_direct's status
 */
zendnnl::error_handling::status_t run_matmul_direct(const char layout,
        const bool transA, const bool transB, const int M, const int N,
        const int K, const float alpha, const void *src, const int lda,
        const void *weight, const int ldb, const void *bias, const float beta,
        void *dst, const int ldc, const bool is_weights_const,
        const zendnnl::lowoha::matmul::matmul_batch_params_t &batch_params,
        const zendnnl::lowoha::matmul::matmul_params &params,
        const zendnnl::lowoha::matmul::matmul_gated_act_params *gated_act
        = nullptr);

/** @brief Whether the last run_matmul_direct of this thread since the last
 *         query (through matmul_kernel_test on a kernel, not the reference)
 *         was a call the native BRGEMM does not serve, checked there: the
 *         call has no result to compare.  For the suites that draw calls
 *         the native BRGEMM serves: a refusal fails the test unless the
 *         call's own arguments carry what its reason names, for the
 *         classes those suites draw on purpose (weight types it has no
 *         kernel for, a source of another type than the weight's, GGML
 *         packs, batched calls that name it, column-major calls, reordered
 *         sources, weight zero points, an f32 call into another dst, a
 *         softmax after a gated activation).  Clears the record.
 */
bool native_refusal_verified();

/** @brief native_refusal_verified without the check of the reason, for
 *         tests that set up the refusal and check its reason themselves.
 */
bool native_refusal_checked();

#endif
