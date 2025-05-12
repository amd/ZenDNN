/*******************************************************************************
* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
*
*******************************************************************************/

#include <immintrin.h>
#include "zendnn.hpp"

#include "zendnn_matmul_direct_utils.hpp"

namespace zendnn {
namespace zendnn_registerBlocking_kernel_fp32 {
// Helper: Load partial AVX-512 vector with masking
__attribute__((target("avx512f")))
__m512 loadu_masked(const float *ptr, int valid) {
    __mmask16 mask = (1 << valid) - 1;
    return _mm512_maskz_loadu_ps(mask, ptr);
}

// Helper: Store partial AVX-512 vector with masking
__attribute__((target("avx512f")))
void storeu_masked(float *ptr, __m512 vec, int valid) {
    __mmask16 mask = (1 << valid) - 1;
    _mm512_mask_storeu_ps(ptr, mask, vec);
}

/*
1. Accumulator Registers
    c: 1 register
    Live throughout the kernel

2. B Registers (Unrolled)
    b0 to b7: 8 registers
    Live during each k += 8 iteration

3. A Broadcast Registers
    a0 to a7: 8 registers
    Only 8 live at a time

4. Temporaries in Post-processing
    bias_vec: 1
    c_old: 1
    alpha, beta: up to 2
    Total: ~4

5. Total: 1 (accumulator) + 8 (B) + 8 (A) + 4 (temporaries) = 21 registers
*/
__attribute__((target("avx512f")))
void compute_tile_1x16(const float *A, const float *B, float *C,
                       const float *bias, float alpha, float beta,
                       int K, int M, int N, int i, int j, bool transB, ActivationPostOp post_op) {
    // Initialize accumulator for 1 row of output tile
    __m512 c = _mm512_setzero_ps();

    // Compute the number of valid columns in this tile (for masking)
    int len = std::min(16, N - j);

    // Main loop: process K dimension in chunks of 8 (loop unrolled for performance)
    int k = 0;
    for (; k <= K - 8; k += 8) {
        // Load 8 rows of B (each 16-wide) using masked loads to avoid out-of-bounds
        __m512 b0 = loadu_masked(&B[(k + 0) * N + j], len);
        __m512 b1 = loadu_masked(&B[(k + 1) * N + j], len);
        __m512 b2 = loadu_masked(&B[(k + 2) * N + j], len);
        __m512 b3 = loadu_masked(&B[(k + 3) * N + j], len);
        __m512 b4 = loadu_masked(&B[(k + 4) * N + j], len);
        __m512 b5 = loadu_masked(&B[(k + 5) * N + j], len);
        __m512 b6 = loadu_masked(&B[(k + 6) * N + j], len);
        __m512 b7 = loadu_masked(&B[(k + 7) * N + j], len);

        const float *a_ptr = &A[i * K + k];

        // Broadcast 8 A values for the row
        __m512 a0 = _mm512_set1_ps(a_ptr[0]);
        __m512 a1 = _mm512_set1_ps(a_ptr[1]);
        __m512 a2 = _mm512_set1_ps(a_ptr[2]);
        __m512 a3 = _mm512_set1_ps(a_ptr[3]);
        __m512 a4 = _mm512_set1_ps(a_ptr[4]);
        __m512 a5 = _mm512_set1_ps(a_ptr[5]);
        __m512 a6 = _mm512_set1_ps(a_ptr[6]);
        __m512 a7 = _mm512_set1_ps(a_ptr[7]);

        // Fused multiply-add: accumulate A * B into C
        c = _mm512_fmadd_ps(a0, b0, c);
        c = _mm512_fmadd_ps(a1, b1, c);
        c = _mm512_fmadd_ps(a2, b2, c);
        c = _mm512_fmadd_ps(a3, b3, c);
        c = _mm512_fmadd_ps(a4, b4, c);
        c = _mm512_fmadd_ps(a5, b5, c);
        c = _mm512_fmadd_ps(a6, b6, c);
        c = _mm512_fmadd_ps(a7, b7, c);

        // Zen 5 tuned prefetch: prefetch A data 3 iterations ahead (8 * 3 = 24 elements)
        _mm_prefetch((const char *)&A[i * K + k + 24], _MM_HINT_T0);

        // Zen 5 tuned prefetch: prefetch B data 4 iterations ahead (8 * 4 = 32 elements)
        _mm_prefetch((const char *)&B[(k + 32) * N + j], _MM_HINT_T0);
    }

    // Remainder loop: handle leftover K values when K is not divisible by 8
    for (; k < K; ++k) {
        __m512 b = loadu_masked(&B[k * N + j], len);
        __m512 a = _mm512_set1_ps(A[i * K + k]);
        c = _mm512_fmadd_ps(a, b, c);
    }

    // Apply bias if present
    if (bias) {
        __m512 bias_vec = loadu_masked(&bias[j], len);
        c = _mm512_add_ps(c, bias_vec);
    }

    if (beta != 0.0f) {
        __m512 c_old = loadu_masked(&C[i * N + j], len);
        c = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old,
                            _mm512_mul_ps(_mm512_set1_ps(alpha), c));
    }
    else {
        c = _mm512_mul_ps(_mm512_set1_ps(alpha), c);
    }

    storeu_masked(&C[i * N + j], apply_post_op(c, post_op), len);
}

/*
1. Accumulator Registers
    c[0] to c[3]: 4 registers
    Live throughout the kernel

2. B Registers (Unrolled)
    b0 to b7: 8 registers
    Live during each k += 8 iteration

3. A Broadcast Registers (Per Row)
    For each row r, we broadcast 8 values: a0 to a7
    These are reused per row, so only 8 A registers live at a time
    Since we loop over 4 rows and reuse the same names, we don’t accumulate 32 A registers — just 8 reused per row

4. Temporaries in Post-processing
    bias_vec: 1
    c_old: 1
    alpha, beta: up to 2
    Total: ~4

5. Total: 4 (accumulators) + 8 (B) + 8 (A reused) + 4 (temporaries) = 24 registers
*/
__attribute__((target("avx512f")))
void compute_tile_4x16(const float *A, const float *B, float *C,
                       const float *bias, float alpha, float beta,
                       int K, int M, int N, int i, int j, bool transB, ActivationPostOp post_op) {
    // Initialize 4 accumulators for 4 rows of output tile
    __m512 c[4] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps()
    };

    // Compute the number of valid columns in this tile (for masking)
    int len = std::min(16, N - j);

    // Main loop: process K dimension in chunks of 8 (loop unrolled for performance)
    int k = 0;
    for (; k <= K - 8; k += 8) {
        // Load 8 rows of B (each 16-wide) using masked loads to avoid out-of-bounds
        __m512 b0 = loadu_masked(&B[(k + 0) * N + j], len);
        __m512 b1 = loadu_masked(&B[(k + 1) * N + j], len);
        __m512 b2 = loadu_masked(&B[(k + 2) * N + j], len);
        __m512 b3 = loadu_masked(&B[(k + 3) * N + j], len);
        __m512 b4 = loadu_masked(&B[(k + 4) * N + j], len);
        __m512 b5 = loadu_masked(&B[(k + 5) * N + j], len);
        __m512 b6 = loadu_masked(&B[(k + 6) * N + j], len);
        __m512 b7 = loadu_masked(&B[(k + 7) * N + j], len);

        // Loop over up to 4 rows of A and accumulate into C
        for (int r = 0; r < 4; ++r) {
            if (i + r >= M) {
                break;    // Boundary check for M
            }

            const float *a_ptr = &A[(i + r) * K + k];

            // Broadcast 8 A values for row r
            __m512 a0 = _mm512_set1_ps(a_ptr[0]);
            __m512 a1 = _mm512_set1_ps(a_ptr[1]);
            __m512 a2 = _mm512_set1_ps(a_ptr[2]);
            __m512 a3 = _mm512_set1_ps(a_ptr[3]);
            __m512 a4 = _mm512_set1_ps(a_ptr[4]);
            __m512 a5 = _mm512_set1_ps(a_ptr[5]);
            __m512 a6 = _mm512_set1_ps(a_ptr[6]);
            __m512 a7 = _mm512_set1_ps(a_ptr[7]);

            // Fused multiply-add: accumulate A * B into C
            c[r] = _mm512_fmadd_ps(a0, b0, c[r]);
            c[r] = _mm512_fmadd_ps(a1, b1, c[r]);
            c[r] = _mm512_fmadd_ps(a2, b2, c[r]);
            c[r] = _mm512_fmadd_ps(a3, b3, c[r]);
            c[r] = _mm512_fmadd_ps(a4, b4, c[r]);
            c[r] = _mm512_fmadd_ps(a5, b5, c[r]);
            c[r] = _mm512_fmadd_ps(a6, b6, c[r]);
            c[r] = _mm512_fmadd_ps(a7, b7, c[r]);

            // Zen 5 tuned prefetch: prefetch A data 3 iterations ahead (8 * 3 = 24 elements)
            _mm_prefetch((const char *)&A[(i + r) * K + k + 24], _MM_HINT_T0);
        }

        // Zen 5 tuned prefetch: prefetch B data 4 iterations ahead (8 * 4 = 32 elements)
        _mm_prefetch((const char *)&B[(k + 32) * N + j], _MM_HINT_T0);
    }

    // Remainder loop: handle leftover K values when K is not divisible by 8
    for (; k < K; ++k) {
        __m512 b = loadu_masked(&B[k * N + j], len);
        for (int r = 0; r < 4; ++r) {
            if (i + r >= M) {
                break;
            }
            __m512 a = _mm512_set1_ps(A[(i + r) * K + k]);
            c[r] = _mm512_fmadd_ps(a, b, c[r]);
        }
    }

    // Apply bias, alpha/beta scaling, and post-op, then store result
    if (bias) {
        __m512 bias_vec = loadu_masked(&bias[j], len);
        if (beta != 0.0f) {
            for (int r = 0; r < 4 && i + r < M; ++r) {
                c[r] = _mm512_add_ps(c[r], bias_vec);
                __m512 c_old = loadu_masked(&C[(i + r) * N + j], len);
                c[r] = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old,
                                       _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]));
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
        else {
            for (int r = 0; r < 4 && i + r < M; ++r) {
                c[r] = _mm512_add_ps(c[r], bias_vec);
                c[r] = _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]);
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
    }
    else {
        if (beta != 0.0f) {
            for (int r = 0; r < 4 && i + r < M; ++r) {
                __m512 c_old = loadu_masked(&C[(i + r) * N + j], len);
                c[r] = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old,
                                       _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]));
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
        else {
            for (int r = 0; r < 4 && i + r < M; ++r) {
                c[r] = _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]);
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
    }
}



/*
1. Accumulator Registers
    c[0] to c[5]: 6 registers
    Live throughout the kernel

2. B Registers (Unrolled)
    b0 to b7: 8 registers
    Live during each k += 8 iteration

3. A Broadcast Registers (Per Row)
    For each row r, we broadcast 8 values: a0 to a7
    These are reused per row, so only 8 A registers live at a time
    But since we loop over 6 rows, and reuse the same names, we don’t accumulate 48 A registers — just 8 reused per row

4. Temporaries in Post-processing
    bias_vec: 1
    c_old: 1
    alpha, beta: up to 2
    Total: ~4

5. Total of 26
*/
__attribute__((target("avx512f")))
void compute_tile_6x16(const float *A, const float *B, float *C,
                       const float *bias, float alpha, float beta,
                       int K, int M, int N, int i, int j, bool transB, ActivationPostOp post_op) {
    // Initialize 6 accumulators for 6 rows of output tile
    __m512 c[6] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()
    };

    // Compute the number of valid columns in this tile (for masking)
    int len = std::min(16, N - j);

    // Main loop: process K dimension in chunks of 8 (loop unrolled for performance)
    int k = 0;
    for (; k <= K - 8; k += 8) {
        // Load 8 rows of B (each 16-wide) using masked loads to avoid out-of-bounds
        __m512 b0 = loadu_masked(&B[(k + 0) * N + j], len);
        __m512 b1 = loadu_masked(&B[(k + 1) * N + j], len);
        __m512 b2 = loadu_masked(&B[(k + 2) * N + j], len);
        __m512 b3 = loadu_masked(&B[(k + 3) * N + j], len);
        __m512 b4 = loadu_masked(&B[(k + 4) * N + j], len);
        __m512 b5 = loadu_masked(&B[(k + 5) * N + j], len);
        __m512 b6 = loadu_masked(&B[(k + 6) * N + j], len);
        __m512 b7 = loadu_masked(&B[(k + 7) * N + j], len);

        // Loop over up to 6 rows of A and accumulate into C
        for (int r = 0; r < 6; ++r) {
            if (i + r >= M) {
                break;    // Boundary check for M
            }

            const float *a_ptr = &A[(i + r) * K + k];

            // Broadcast 8 A values for row r
            __m512 a0 = _mm512_set1_ps(a_ptr[0]);
            __m512 a1 = _mm512_set1_ps(a_ptr[1]);
            __m512 a2 = _mm512_set1_ps(a_ptr[2]);
            __m512 a3 = _mm512_set1_ps(a_ptr[3]);
            __m512 a4 = _mm512_set1_ps(a_ptr[4]);
            __m512 a5 = _mm512_set1_ps(a_ptr[5]);
            __m512 a6 = _mm512_set1_ps(a_ptr[6]);
            __m512 a7 = _mm512_set1_ps(a_ptr[7]);

            // Fused multiply-add: accumulate A * B into C
            c[r] = _mm512_fmadd_ps(a0, b0, c[r]);
            c[r] = _mm512_fmadd_ps(a1, b1, c[r]);
            c[r] = _mm512_fmadd_ps(a2, b2, c[r]);
            c[r] = _mm512_fmadd_ps(a3, b3, c[r]);
            c[r] = _mm512_fmadd_ps(a4, b4, c[r]);
            c[r] = _mm512_fmadd_ps(a5, b5, c[r]);
            c[r] = _mm512_fmadd_ps(a6, b6, c[r]);
            c[r] = _mm512_fmadd_ps(a7, b7, c[r]);

            // Zen 5 tuned prefetch: prefetch A data 3 iterations ahead (8 * 3 = 24 elements)
            // Helps hide memory latency for future A loads
            _mm_prefetch((const char *)&A[(i + r) * K + k + 24], _MM_HINT_T0);
        }

        // Zen 5 tuned prefetch: prefetch B data 4 iterations ahead (8 * 4 = 32 elements)
        // B is accessed in a less cache-friendly pattern, so we prefetch more aggressively
        _mm_prefetch((const char *)&B[(k + 32) * N + j], _MM_HINT_T0);
    }

    // Remainder loop: handle leftover K values when K is not divisible by 8
    for (; k < K; ++k) {
        __m512 b = loadu_masked(&B[k * N + j], len);
        for (int r = 0; r < 6; ++r) {
            if (i + r >= M) {
                break;
            }
            __m512 a = _mm512_set1_ps(A[(i + r) * K + k]);
            c[r] = _mm512_fmadd_ps(a, b, c[r]);
        }
    }

    // Apply bias, alpha/beta scaling, and post-op, then store result
    if (bias) {
        __m512 bias_vec = loadu_masked(&bias[j], len);
        if (beta != 0.0f) {
            for (int r = 0; r < 6 && i + r < M; ++r) {
                c[r] = _mm512_add_ps(c[r], bias_vec);
                __m512 c_old = loadu_masked(&C[(i + r) * N + j], len);
                c[r] = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old,
                                       _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]));
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
        else {
            for (int r = 0; r < 6 && i + r < M; ++r) {
                c[r] = _mm512_add_ps(c[r], bias_vec);
                c[r] = _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]);
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
    }
    else {
        if (beta != 0.0f) {
            for (int r = 0; r < 6 && i + r < M; ++r) {
                __m512 c_old = loadu_masked(&C[(i + r) * N + j], len);
                c[r] = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old,
                                       _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]));
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
        else {
            for (int r = 0; r < 6 && i + r < M; ++r) {
                c[r] = _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]);
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
    }
}


/*
1. Accumulator Registers
    c[0] to c[7]: 8 registers
    Live throughout the kernel

2. B Registers (Unrolled)
    b0 to b7: 8 registers
    Live during each k += 8 iteration

3. A Broadcast Registers (Per Row)
    For each row r, we broadcast 8 values: a0 to a7
    These are reused per row, so only 8 A registers live at a time
    Since we loop over 8 rows and reuse the same names, we don’t accumulate 64 A registers — just 8 reused per row

4. Temporaries in Post-processing
    bias_vec: 1
    c_old: 1
    alpha, beta: up to 2
    Total: ~4

5. Total: 8 (accumulators) + 8 (B) + 8 (A reused) + 4 (temporaries) = 28 registers
*/
__attribute__((target("avx512f")))
void compute_tile_8x16(const float *A, const float *B, float *C,
                       const float *bias, float alpha, float beta,
                       int K, int M, int N, int i, int j, bool transB, ActivationPostOp post_op) {
    // Initialize 8 accumulators for 8 rows of output tile
    __m512 c[8] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()
    };

    // Compute the number of valid columns in this tile (for masking)
    int len = std::min(16, N - j);

    // Main loop: process K dimension in chunks of 8 (loop unrolled for performance)
    int k = 0;
    for (; k <= K - 8; k += 8) {
        // Load 8 rows of B (each 16-wide) using masked loads to avoid out-of-bounds
        __m512 b0 = loadu_masked(&B[(k + 0) * N + j], len);
        __m512 b1 = loadu_masked(&B[(k + 1) * N + j], len);
        __m512 b2 = loadu_masked(&B[(k + 2) * N + j], len);
        __m512 b3 = loadu_masked(&B[(k + 3) * N + j], len);
        __m512 b4 = loadu_masked(&B[(k + 4) * N + j], len);
        __m512 b5 = loadu_masked(&B[(k + 5) * N + j], len);
        __m512 b6 = loadu_masked(&B[(k + 6) * N + j], len);
        __m512 b7 = loadu_masked(&B[(k + 7) * N + j], len);

        // Loop over up to 8 rows of A and accumulate into C
        for (int r = 0; r < 8; ++r) {
            if (i + r >= M) {
                break;    // Boundary check for M
            }

            const float *a_ptr = &A[(i + r) * K + k];

            // Broadcast 8 A values for row r
            __m512 a0 = _mm512_set1_ps(a_ptr[0]);
            __m512 a1 = _mm512_set1_ps(a_ptr[1]);
            __m512 a2 = _mm512_set1_ps(a_ptr[2]);
            __m512 a3 = _mm512_set1_ps(a_ptr[3]);
            __m512 a4 = _mm512_set1_ps(a_ptr[4]);
            __m512 a5 = _mm512_set1_ps(a_ptr[5]);
            __m512 a6 = _mm512_set1_ps(a_ptr[6]);
            __m512 a7 = _mm512_set1_ps(a_ptr[7]);

            // Fused multiply-add: accumulate A * B into C
            c[r] = _mm512_fmadd_ps(a0, b0, c[r]);
            c[r] = _mm512_fmadd_ps(a1, b1, c[r]);
            c[r] = _mm512_fmadd_ps(a2, b2, c[r]);
            c[r] = _mm512_fmadd_ps(a3, b3, c[r]);
            c[r] = _mm512_fmadd_ps(a4, b4, c[r]);
            c[r] = _mm512_fmadd_ps(a5, b5, c[r]);
            c[r] = _mm512_fmadd_ps(a6, b6, c[r]);
            c[r] = _mm512_fmadd_ps(a7, b7, c[r]);

            // Zen 5 tuned prefetch: prefetch A data 3 iterations ahead (8 * 3 = 24 elements)
            _mm_prefetch((const char *)&A[(i + r) * K + k + 24], _MM_HINT_T0);
        }

        // Zen 5 tuned prefetch: prefetch B data 4 iterations ahead (8 * 4 = 32 elements)
        _mm_prefetch((const char *)&B[(k + 32) * N + j], _MM_HINT_T0);
    }

    // Remainder loop: handle leftover K values when K is not divisible by 8
    for (; k < K; ++k) {
        __m512 b = loadu_masked(&B[k * N + j], len);
        for (int r = 0; r < 8; ++r) {
            if (i + r >= M) {
                break;
            }
            __m512 a = _mm512_set1_ps(A[(i + r) * K + k]);
            c[r] = _mm512_fmadd_ps(a, b, c[r]);
        }
    }

    // Apply bias, alpha/beta scaling, and post-op, then store result
    if (bias) {
        __m512 bias_vec = loadu_masked(&bias[j], len);
        if (beta != 0.0f) {
            for (int r = 0; r < 8 && i + r < M; ++r) {
                c[r] = _mm512_add_ps(c[r], bias_vec);
                __m512 c_old = loadu_masked(&C[(i + r) * N + j], len);
                c[r] = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old,
                                       _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]));
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
        else {
            for (int r = 0; r < 8 && i + r < M; ++r) {
                c[r] = _mm512_add_ps(c[r], bias_vec);
                c[r] = _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]);
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
    }
    else {
        if (beta != 0.0f) {
            for (int r = 0; r < 8 && i + r < M; ++r) {
                __m512 c_old = loadu_masked(&C[(i + r) * N + j], len);
                c[r] = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old,
                                       _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]));
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
        else {
            for (int r = 0; r < 8 && i + r < M; ++r) {
                c[r] = _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]);
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
    }
}



/*
1. Accumulator Registers
    c[0] to c[11]: 12 registers
    Live throughout the kernel

2. B Registers (Unrolled by 4)
    b0 to b3: 4 registers
    Live during each kk iteration

3. A Broadcast Registers
    a0 to a3: 4 registers
    Reused per row per kk iteration

4. Temporaries in Post-processing
    bias_vec: 1
    c_old: 1
    alpha, beta: up to 2
    Total: ~4

5. Total: 12 (accumulators) + 4 (B) + 4 (A) + 4 (temporaries) = 24 registers
*/
__attribute__((target("avx512f")))
void compute_tile_12x16(const float *A, const float *B, float *C,
                        const float *bias, float alpha, float beta,
                        int K, int M, int N, int i, int j, bool transB, ActivationPostOp post_op) {
    // Initialize 12 accumulators for 12 rows of output tile
    __m512 c[12] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()
    };

    // Compute the number of valid columns in this tile (for masking)
    int len = std::min(16, N - j);

    // Main loop: process K dimension in chunks of 8, with inner loop unrolled in steps of 4
    int k = 0;
    for (; k <= K - 8; k += 8) {
        for (int kk = 0; kk < 8; kk += 4) {
            // Load 4 rows of B (each 16-wide) using masked loads
            __m512 b0 = loadu_masked(&B[(k + kk + 0) * N + j], len);
            __m512 b1 = loadu_masked(&B[(k + kk + 1) * N + j], len);
            __m512 b2 = loadu_masked(&B[(k + kk + 2) * N + j], len);
            __m512 b3 = loadu_masked(&B[(k + kk + 3) * N + j], len);

            // Loop over up to 12 rows of A and accumulate into C
            for (int r = 0; r < 12; ++r) {
                if (i + r >= M) {
                    break;
                }

                const float *a_ptr = &A[(i + r) * K + k + kk];

                // Broadcast 4 A values for row r
                __m512 a0 = _mm512_set1_ps(a_ptr[0]);
                __m512 a1 = _mm512_set1_ps(a_ptr[1]);
                __m512 a2 = _mm512_set1_ps(a_ptr[2]);
                __m512 a3 = _mm512_set1_ps(a_ptr[3]);

                // Fused multiply-add: accumulate A * B into C
                c[r] = _mm512_fmadd_ps(a0, b0, c[r]);
                c[r] = _mm512_fmadd_ps(a1, b1, c[r]);
                c[r] = _mm512_fmadd_ps(a2, b2, c[r]);
                c[r] = _mm512_fmadd_ps(a3, b3, c[r]);

                // Zen 5 tuned prefetch: prefetch A data 3 iterations ahead (8 * 3 = 24 elements)
                if (kk == 0) {
                    _mm_prefetch((const char *)&A[(i + r) * K + k + 24], _MM_HINT_T0);
                }
            }

            // Zen 5 tuned prefetch: prefetch B data 4 iterations ahead (8 * 4 = 32 elements)
            if (kk == 0) {
                _mm_prefetch((const char *)&B[(k + 32) * N + j], _MM_HINT_T0);
            }
        }
    }

    // Remainder loop: handle leftover K values when K is not divisible by 8
    for (; k < K; ++k) {
        __m512 b = loadu_masked(&B[k * N + j], len);
        for (int r = 0; r < 12; ++r) {
            if (i + r >= M) {
                break;
            }
            __m512 a = _mm512_set1_ps(A[(i + r) * K + k]);
            c[r] = _mm512_fmadd_ps(a, b, c[r]);
        }
    }

    // Apply bias, alpha/beta scaling, and post-op, then store result
    if (bias) {
        __m512 bias_vec = loadu_masked(&bias[j], len);
        if (beta != 0.0f) {
            for (int r = 0; r < 12 && i + r < M; ++r) {
                c[r] = _mm512_add_ps(c[r], bias_vec);
                __m512 c_old = loadu_masked(&C[(i + r) * N + j], len);
                c[r] = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old,
                                       _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]));
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
        else {
            for (int r = 0; r < 12 && i + r < M; ++r) {
                c[r] = _mm512_add_ps(c[r], bias_vec);
                c[r] = _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]);
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
    }
    else {
        if (beta != 0.0f) {
            for (int r = 0; r < 12 && i + r < M; ++r) {
                __m512 c_old = loadu_masked(&C[(i + r) * N + j], len);
                c[r] = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old,
                                       _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]));
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
        else {
            for (int r = 0; r < 12 && i + r < M; ++r) {
                c[r] = _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]);
                storeu_masked(&C[(i + r) * N + j], apply_post_op(c[r], post_op), len);
            }
        }
    }
}

// Main AVX-512 kernel with tunable register tiling
void matmul_avx512_fp32_registerBlocking(const float *A, const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, int MR, int NR, bool transB, ActivationPostOp post_op) {

    zendnnInfo(ZENDNN_CORELOG,
               "Runing matmul_avx512_fp32_registerBlocking with MR x NR = : ", MR, " x ", NR,
               " kernel");

    for (int i = 0; i < M; i += MR) {
        for (int j = 0; j < N; j += NR) {
            if (MR == 1 && NR == 16) {
                compute_tile_1x16(A, B, C, bias, alpha, beta, K, M, N, i, j, transB, post_op);
            }
            else if (MR == 4 && NR == 16) {
                compute_tile_4x16(A, B, C, bias, alpha, beta, K, M, N, i, j, transB, post_op);
            }
            else if (MR == 6 && NR == 16) {
                compute_tile_6x16(A, B, C, bias, alpha, beta, K, M, N, i, j, transB, post_op);
            }
            else  if (MR == 8 && NR == 16) {
                compute_tile_8x16(A, B, C, bias, alpha, beta, K, M, N, i, j, transB, post_op);
            }
            else if (MR == 12 && NR == 16) {
                compute_tile_12x16(A, B, C, bias, alpha, beta, K, M, N, i, j, transB, post_op);
            }

        }
    }
}

//greedy tiling strategy
void matmul_avx512_fp32_registerBlocking_auto(const float *A, const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, bool transB, ActivationPostOp post_op) {


    zendnnInfo(ZENDNN_CORELOG,
               "Running matmul_avx512_fp32_registerBlocking with dynamic MR x NR  kernel");

    for (int i = 0; i < M;) {
        int remaining = M - i;

        // Choose the largest tile that fits in the remaining rows
        if (remaining >= 12) {
            for (int j = 0; j < N; j += 16) {
                compute_tile_12x16(A, B, C, bias, alpha, beta, K, M, N, i, j, transB, post_op);
            }
            i += 12;
        }
        else if (remaining >= 8) {
            for (int j = 0; j < N; j += 16) {
                compute_tile_8x16(A, B, C, bias, alpha, beta, K, M, N, i, j, transB, post_op);
            }
            i += 8;
        }
        else if (remaining >= 6) {
            for (int j = 0; j < N; j += 16) {
                compute_tile_6x16(A, B, C, bias, alpha, beta, K, M, N, i, j, transB, post_op);
            }
            i += 6;
        }
        else if (remaining >= 4) {
            for (int j = 0; j < N; j += 16) {
                compute_tile_4x16(A, B, C, bias, alpha, beta, K, M, N, i, j, transB, post_op);
            }
            i += 4;
        }
        else {
            // Use 1x16 kernel for remaining rows
            for (int j = 0; j < N; j += 16) {
                compute_tile_1x16(A, B, C, bias, alpha, beta, K, M, N, i, j, transB, post_op);
            }
            i += 1;
        }
    }
}

void matmul_avx512_fp32_registerBlocking_batch(const float *A, const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, int MR, int NR, bool transB, ActivationPostOp post_op,
        int BATCH) {

    zendnnInfo(ZENDNN_CORELOG,
               "Running batched matmul_avx512_fp32_registerBlocking_batch with MR x NR = ", MR,
               " x ", NR,
               " kernel with Batch Size = ", BATCH);

    for (int b = 0; b < BATCH; ++b) {
        const float *A_batch = A + b * M * K;
        const float *B_batch = B + b * K * N;
        float *C_batch = C + b * M * N;

        for (int i = 0; i < M; i += MR) {
            for (int j = 0; j < N; j += NR) {
                if (MR == 1 && NR == 16) {
                    compute_tile_1x16(A_batch, B_batch, C_batch, bias, alpha, beta, K, M, N, i, j,
                                      transB, post_op);
                }
                else if (MR == 4 && NR == 16) {
                    compute_tile_4x16(A_batch, B_batch, C_batch, bias, alpha, beta, K, M, N, i, j,
                                      transB, post_op);
                }
                else if (MR == 6 && NR == 16) {
                    compute_tile_6x16(A_batch, B_batch, C_batch, bias, alpha, beta, K, M, N, i, j,
                                      transB, post_op);
                }
                else if (MR == 8 && NR == 16) {
                    compute_tile_8x16(A_batch, B_batch, C_batch, bias, alpha, beta, K, M, N, i, j,
                                      transB, post_op);
                }
                else if (MR == 12 && NR == 16) {
                    compute_tile_12x16(A_batch, B_batch, C_batch, bias, alpha, beta, K, M, N, i, j,
                                       transB, post_op);
                }

            }
        }
    }
}


void matmul_avx512_fp32_registerBlocking_auto_batch(const float *A,
        const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K, bool transB, ActivationPostOp post_op,
        int BATCH) {

    zendnnInfo(ZENDNN_CORELOG,
               "Running batched matmul_avx512_fp32_registerBlocking_auto with dynamic MR x NR kernel, BATCH = ",
               BATCH);

    for (int b = 0; b < BATCH; ++b) {
        const float *A_batch = A + b * M * K;
        const float *B_batch = B + b * K * N;
        float *C_batch = C + b * M * N;

        for (int i = 0; i < M;) {
            int remaining = M - i;

            // Choose the largest tile that fits in the remaining rows
            if (remaining >= 12) {
                for (int j = 0; j < N; j += 16) {
                    compute_tile_12x16(A_batch, B_batch, C_batch, bias, alpha, beta, K, M, N, i, j,
                                       transB, post_op);
                }
                i += 12;
            }
            else if (remaining >= 8) {
                for (int j = 0; j < N; j += 16) {
                    compute_tile_8x16(A_batch, B_batch, C_batch, bias, alpha, beta, K, M, N, i, j,
                                      transB, post_op);
                }
                i += 8;
            }
            else if (remaining >= 6) {
                for (int j = 0; j < N; j += 16) {
                    compute_tile_6x16(A_batch, B_batch, C_batch, bias, alpha, beta, K, M, N, i, j,
                                      transB, post_op);
                }
                i += 6;
            }
            else if (remaining >= 4) {
                for (int j = 0; j < N; j += 16) {
                    compute_tile_4x16(A_batch, B_batch, C_batch, bias, alpha, beta, K, M, N, i, j,
                                      transB, post_op);
                }
                i += 4;
            }
            else {
                for (int j = 0; j < N; j += 16) {
                    compute_tile_1x16(A_batch, B_batch, C_batch, bias, alpha, beta, K, M, N, i, j,
                                      transB, post_op);
                }
                i += 1;
            }
        }
    }
}
}
}

