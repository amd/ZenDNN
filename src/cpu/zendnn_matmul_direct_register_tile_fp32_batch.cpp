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
namespace zendnn_registerBlocking_kernel_fp32_batch {

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


__attribute__((target("avx512f")))
void compute_tile_1x16_batched(const float *A, const float *B, float *C,
                               const float *bias, float alpha, float beta,
                               int K, int M, int N, int i, int j, int batch,
                               bool transB, ActivationPostOp post_op) {
    for (int b = 0; b < batch; ++b) {
        const float *Ab = A + b * M * K;
        const float *Bb = B + b * K * N;
        float *Cb = C + b * M * N;

        __m512 c = _mm512_setzero_ps();
        int len = std::min(16, N - j);

        int k = 0;
        for (; k <= K - 8; k += 8) {
            __m512 b0 = loadu_masked(&Bb[(k + 0) * N + j], len);
            __m512 b1 = loadu_masked(&Bb[(k + 1) * N + j], len);
            __m512 b2 = loadu_masked(&Bb[(k + 2) * N + j], len);
            __m512 b3 = loadu_masked(&Bb[(k + 3) * N + j], len);
            __m512 b4 = loadu_masked(&Bb[(k + 4) * N + j], len);
            __m512 b5 = loadu_masked(&Bb[(k + 5) * N + j], len);
            __m512 b6 = loadu_masked(&Bb[(k + 6) * N + j], len);
            __m512 b7 = loadu_masked(&Bb[(k + 7) * N + j], len);

            const float *a_ptr = &Ab[i * K + k];

            __m512 a0 = _mm512_set1_ps(a_ptr[0]);
            __m512 a1 = _mm512_set1_ps(a_ptr[1]);
            __m512 a2 = _mm512_set1_ps(a_ptr[2]);
            __m512 a3 = _mm512_set1_ps(a_ptr[3]);
            __m512 a4 = _mm512_set1_ps(a_ptr[4]);
            __m512 a5 = _mm512_set1_ps(a_ptr[5]);
            __m512 a6 = _mm512_set1_ps(a_ptr[6]);
            __m512 a7 = _mm512_set1_ps(a_ptr[7]);

            c = _mm512_fmadd_ps(a0, b0, c);
            c = _mm512_fmadd_ps(a1, b1, c);
            c = _mm512_fmadd_ps(a2, b2, c);
            c = _mm512_fmadd_ps(a3, b3, c);
            c = _mm512_fmadd_ps(a4, b4, c);
            c = _mm512_fmadd_ps(a5, b5, c);
            c = _mm512_fmadd_ps(a6, b6, c);
            c = _mm512_fmadd_ps(a7, b7, c);

            _mm_prefetch((const char *)&Ab[i * K + k + 24], _MM_HINT_T0);
            _mm_prefetch((const char *)&Bb[(k + 32) * N + j], _MM_HINT_T0);
        }

        for (; k < K; ++k) {
            __m512 b = loadu_masked(&Bb[k * N + j], len);
            __m512 a = _mm512_set1_ps(Ab[i * K + k]);
            c = _mm512_fmadd_ps(a, b, c);
        }

        if (bias) {
            __m512 bias_vec = loadu_masked(&bias[j], len);
            c = _mm512_add_ps(c, bias_vec);
        }

        if (beta != 0.0f) {
            __m512 c_old = loadu_masked(&Cb[i * N + j], len);
            c = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old,
                                _mm512_mul_ps(_mm512_set1_ps(alpha), c));
        }
        else {
            c = _mm512_mul_ps(_mm512_set1_ps(alpha), c);
        }

        storeu_masked(&Cb[i * N + j], apply_post_op(c, post_op), len);
    }
}


__attribute__((target("avx512f")))
void compute_tile_4x16_batched(const float *A, const float *B, float *C,
                               const float *bias, float alpha, float beta,
                               int K, int M, int N, int i, int j, int batch,
                               bool transB, ActivationPostOp post_op) {
    int len = std::min(16, N - j);

    for (int b = 0; b < batch; ++b) {
        const float *Ab = A + b * M * K;
        const float *Bb = B + b * K * N;
        float *Cb = C + b * M * N;

        __m512 c[4] = {
            _mm512_setzero_ps(), _mm512_setzero_ps(),
            _mm512_setzero_ps(), _mm512_setzero_ps()
        };

        int k = 0;
        for (; k <= K - 8; k += 8) {
            __m512 b0 = loadu_masked(&Bb[(k + 0) * N + j], len);
            __m512 b1 = loadu_masked(&Bb[(k + 1) * N + j], len);
            __m512 b2 = loadu_masked(&Bb[(k + 2) * N + j], len);
            __m512 b3 = loadu_masked(&Bb[(k + 3) * N + j], len);
            __m512 b4 = loadu_masked(&Bb[(k + 4) * N + j], len);
            __m512 b5 = loadu_masked(&Bb[(k + 5) * N + j], len);
            __m512 b6 = loadu_masked(&Bb[(k + 6) * N + j], len);
            __m512 b7 = loadu_masked(&Bb[(k + 7) * N + j], len);

            for (int r = 0; r < 4; ++r) {
                if (i + r >= M) {
                    break;
                }

                const float *a_ptr = &Ab[(i + r) * K + k];

                __m512 a0 = _mm512_set1_ps(a_ptr[0]);
                __m512 a1 = _mm512_set1_ps(a_ptr[1]);
                __m512 a2 = _mm512_set1_ps(a_ptr[2]);
                __m512 a3 = _mm512_set1_ps(a_ptr[3]);
                __m512 a4 = _mm512_set1_ps(a_ptr[4]);
                __m512 a5 = _mm512_set1_ps(a_ptr[5]);
                __m512 a6 = _mm512_set1_ps(a_ptr[6]);
                __m512 a7 = _mm512_set1_ps(a_ptr[7]);

                c[r] = _mm512_fmadd_ps(a0, b0, c[r]);
                c[r] = _mm512_fmadd_ps(a1, b1, c[r]);
                c[r] = _mm512_fmadd_ps(a2, b2, c[r]);
                c[r] = _mm512_fmadd_ps(a3, b3, c[r]);
                c[r] = _mm512_fmadd_ps(a4, b4, c[r]);
                c[r] = _mm512_fmadd_ps(a5, b5, c[r]);
                c[r] = _mm512_fmadd_ps(a6, b6, c[r]);
                c[r] = _mm512_fmadd_ps(a7, b7, c[r]);

                _mm_prefetch((const char *)&Ab[(i + r) * K + k + 24], _MM_HINT_T0);
            }

            _mm_prefetch((const char *)&Bb[(k + 32) * N + j], _MM_HINT_T0);
        }

        for (; k < K; ++k) {
            __m512 b = loadu_masked(&Bb[k * N + j], len);
            for (int r = 0; r < 4; ++r) {
                if (i + r >= M) {
                    break;
                }
                __m512 a = _mm512_set1_ps(Ab[(i + r) * K + k]);
                c[r] = _mm512_fmadd_ps(a, b, c[r]);
            }
        }

        if (bias) {
            __m512 bias_vec = loadu_masked(&bias[j], len);
            for (int r = 0; r < 4 && i + r < M; ++r) {
                c[r] = _mm512_add_ps(c[r], bias_vec);
            }
        }

        for (int r = 0; r < 4 && i + r < M; ++r) {
            if (beta != 0.0f) {
                __m512 c_old = loadu_masked(&Cb[(i + r) * N + j], len);
                c[r] = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old,
                                       _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]));
            }
            else {
                c[r] = _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]);
            }

            storeu_masked(&Cb[(i + r) * N + j], apply_post_op(c[r], post_op), len);
        }
    }
}



__attribute__((target("avx512f")))
void compute_tile_6x16_batched(const float *A, const float *B, float *C,
                               const float *bias, float alpha, float beta,
                               int K, int M, int N, int i, int j, int batch,
                               bool transB, ActivationPostOp post_op) {
    int len = std::min(16, N - j);

    for (int b = 0; b < batch; ++b) {
        const float *Ab = A + b * M * K;
        const float *Bb = B + b * K * N;
        float *Cb = C + b * M * N;

        __m512 c[6] = {
            _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
            _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()
        };

        int k = 0;
        for (; k <= K - 8; k += 8) {
            __m512 b0 = loadu_masked(&Bb[(k + 0) * N + j], len);
            __m512 b1 = loadu_masked(&Bb[(k + 1) * N + j], len);
            __m512 b2 = loadu_masked(&Bb[(k + 2) * N + j], len);
            __m512 b3 = loadu_masked(&Bb[(k + 3) * N + j], len);
            __m512 b4 = loadu_masked(&Bb[(k + 4) * N + j], len);
            __m512 b5 = loadu_masked(&Bb[(k + 5) * N + j], len);
            __m512 b6 = loadu_masked(&Bb[(k + 6) * N + j], len);
            __m512 b7 = loadu_masked(&Bb[(k + 7) * N + j], len);

            for (int r = 0; r < 6; ++r) {
                if (i + r >= M) {
                    break;
                }

                const float *a_ptr = &Ab[(i + r) * K + k];

                __m512 a0 = _mm512_set1_ps(a_ptr[0]);
                __m512 a1 = _mm512_set1_ps(a_ptr[1]);
                __m512 a2 = _mm512_set1_ps(a_ptr[2]);
                __m512 a3 = _mm512_set1_ps(a_ptr[3]);
                __m512 a4 = _mm512_set1_ps(a_ptr[4]);
                __m512 a5 = _mm512_set1_ps(a_ptr[5]);
                __m512 a6 = _mm512_set1_ps(a_ptr[6]);
                __m512 a7 = _mm512_set1_ps(a_ptr[7]);

                c[r] = _mm512_fmadd_ps(a0, b0, c[r]);
                c[r] = _mm512_fmadd_ps(a1, b1, c[r]);
                c[r] = _mm512_fmadd_ps(a2, b2, c[r]);
                c[r] = _mm512_fmadd_ps(a3, b3, c[r]);
                c[r] = _mm512_fmadd_ps(a4, b4, c[r]);
                c[r] = _mm512_fmadd_ps(a5, b5, c[r]);
                c[r] = _mm512_fmadd_ps(a6, b6, c[r]);
                c[r] = _mm512_fmadd_ps(a7, b7, c[r]);

                _mm_prefetch((const char *)&Ab[(i + r) * K + k + 24], _MM_HINT_T0);
            }

            _mm_prefetch((const char *)&Bb[(k + 32) * N + j], _MM_HINT_T0);
        }

        for (; k < K; ++k) {
            __m512 b = loadu_masked(&Bb[k * N + j], len);
            for (int r = 0; r < 6; ++r) {
                if (i + r >= M) {
                    break;
                }
                __m512 a = _mm512_set1_ps(Ab[(i + r) * K + k]);
                c[r] = _mm512_fmadd_ps(a, b, c[r]);
            }
        }

        if (bias) {
            __m512 bias_vec = loadu_masked(&bias[j], len);
            for (int r = 0; r < 6 && i + r < M; ++r) {
                c[r] = _mm512_add_ps(c[r], bias_vec);
            }
        }

        for (int r = 0; r < 6 && i + r < M; ++r) {
            if (beta != 0.0f) {
                __m512 c_old = loadu_masked(&Cb[(i + r) * N + j], len);
                c[r] = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old,
                                       _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]));
            }
            else {
                c[r] = _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]);
            }

            storeu_masked(&Cb[(i + r) * N + j], apply_post_op(c[r], post_op), len);
        }
    }
}



__attribute__((target("avx512f")))
void compute_tile_8x16_batched(const float *A, const float *B, float *C,
                               const float *bias, float alpha, float beta,
                               int K, int M, int N, int i, int j, int batch,
                               bool transB, ActivationPostOp post_op) {
    int len = std::min(16, N - j);

    for (int b = 0; b < batch; ++b) {
        const float *Ab = A + b * M * K;
        const float *Bb = B + b * K * N;
        float *Cb = C + b * M * N;

        __m512 c[8] = {
            _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
            _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()
        };

        int k = 0;
        for (; k <= K - 8; k += 8) {
            __m512 b0 = loadu_masked(&Bb[(k + 0) * N + j], len);
            __m512 b1 = loadu_masked(&Bb[(k + 1) * N + j], len);
            __m512 b2 = loadu_masked(&Bb[(k + 2) * N + j], len);
            __m512 b3 = loadu_masked(&Bb[(k + 3) * N + j], len);
            __m512 b4 = loadu_masked(&Bb[(k + 4) * N + j], len);
            __m512 b5 = loadu_masked(&Bb[(k + 5) * N + j], len);
            __m512 b6 = loadu_masked(&Bb[(k + 6) * N + j], len);
            __m512 b7 = loadu_masked(&Bb[(k + 7) * N + j], len);

            for (int r = 0; r < 8; ++r) {
                if (i + r >= M) {
                    break;
                }

                const float *a_ptr = &Ab[(i + r) * K + k];

                __m512 a0 = _mm512_set1_ps(a_ptr[0]);
                __m512 a1 = _mm512_set1_ps(a_ptr[1]);
                __m512 a2 = _mm512_set1_ps(a_ptr[2]);
                __m512 a3 = _mm512_set1_ps(a_ptr[3]);
                __m512 a4 = _mm512_set1_ps(a_ptr[4]);
                __m512 a5 = _mm512_set1_ps(a_ptr[5]);
                __m512 a6 = _mm512_set1_ps(a_ptr[6]);
                __m512 a7 = _mm512_set1_ps(a_ptr[7]);

                c[r] = _mm512_fmadd_ps(a0, b0, c[r]);
                c[r] = _mm512_fmadd_ps(a1, b1, c[r]);
                c[r] = _mm512_fmadd_ps(a2, b2, c[r]);
                c[r] = _mm512_fmadd_ps(a3, b3, c[r]);
                c[r] = _mm512_fmadd_ps(a4, b4, c[r]);
                c[r] = _mm512_fmadd_ps(a5, b5, c[r]);
                c[r] = _mm512_fmadd_ps(a6, b6, c[r]);
                c[r] = _mm512_fmadd_ps(a7, b7, c[r]);

                _mm_prefetch((const char *)&Ab[(i + r) * K + k + 24], _MM_HINT_T0);
            }

            _mm_prefetch((const char *)&Bb[(k + 32) * N + j], _MM_HINT_T0);
        }

        for (; k < K; ++k) {
            __m512 b = loadu_masked(&Bb[k * N + j], len);
            for (int r = 0; r < 8; ++r) {
                if (i + r >= M) {
                    break;
                }
                __m512 a = _mm512_set1_ps(Ab[(i + r) * K + k]);
                c[r] = _mm512_fmadd_ps(a, b, c[r]);
            }
        }

        if (bias) {
            __m512 bias_vec = loadu_masked(&bias[j], len);
            for (int r = 0; r < 8 && i + r < M; ++r) {
                c[r] = _mm512_add_ps(c[r], bias_vec);
            }
        }

        for (int r = 0; r < 8 && i + r < M; ++r) {
            if (beta != 0.0f) {
                __m512 c_old = loadu_masked(&Cb[(i + r) * N + j], len);
                c[r] = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old,
                                       _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]));
            }
            else {
                c[r] = _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]);
            }

            storeu_masked(&Cb[(i + r) * N + j], apply_post_op(c[r], post_op), len);
        }
    }
}


__attribute__((target("avx512f")))
void compute_tile_12x16_batched(const float *A, const float *B, float *C,
                                const float *bias, float alpha, float beta,
                                int K, int M, int N, int i, int j, int batch,
                                bool transB, ActivationPostOp post_op) {
    int len = std::min(16, N - j);

    for (int b = 0; b < batch; ++b) {
        const float *Ab = A + b * M * K;
        const float *Bb = B + b * K * N;
        float *Cb = C + b * M * N;

        __m512 c[12] = {
            _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
            _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
            _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()
        };

        int k = 0;
        for (; k <= K - 8; k += 8) {
            for (int kk = 0; kk < 8; kk += 4) {
                __m512 b0 = loadu_masked(&Bb[(k + kk + 0) * N + j], len);
                __m512 b1 = loadu_masked(&Bb[(k + kk + 1) * N + j], len);
                __m512 b2 = loadu_masked(&Bb[(k + kk + 2) * N + j], len);
                __m512 b3 = loadu_masked(&Bb[(k + kk + 3) * N + j], len);

                for (int r = 0; r < 12; ++r) {
                    if (i + r >= M) {
                        break;
                    }

                    const float *a_ptr = &Ab[(i + r) * K + k + kk];

                    __m512 a0 = _mm512_set1_ps(a_ptr[0]);
                    __m512 a1 = _mm512_set1_ps(a_ptr[1]);
                    __m512 a2 = _mm512_set1_ps(a_ptr[2]);
                    __m512 a3 = _mm512_set1_ps(a_ptr[3]);

                    c[r] = _mm512_fmadd_ps(a0, b0, c[r]);
                    c[r] = _mm512_fmadd_ps(a1, b1, c[r]);
                    c[r] = _mm512_fmadd_ps(a2, b2, c[r]);
                    c[r] = _mm512_fmadd_ps(a3, b3, c[r]);

                    if (kk == 0) {
                        _mm_prefetch((const char *)&Ab[(i + r) * K + k + 24], _MM_HINT_T0);
                    }
                }

                if (kk == 0) {
                    _mm_prefetch((const char *)&Bb[(k + 32) * N + j], _MM_HINT_T0);
                }
            }
        }

        for (; k < K; ++k) {
            __m512 b = loadu_masked(&Bb[k * N + j], len);
            for (int r = 0; r < 12; ++r) {
                if (i + r >= M) {
                    break;
                }
                __m512 a = _mm512_set1_ps(Ab[(i + r) * K + k]);
                c[r] = _mm512_fmadd_ps(a, b, c[r]);
            }
        }

        if (bias) {
            __m512 bias_vec = loadu_masked(&bias[j], len);
            for (int r = 0; r < 12 && i + r < M; ++r) {
                c[r] = _mm512_add_ps(c[r], bias_vec);
            }
        }

        for (int r = 0; r < 12 && i + r < M; ++r) {
            if (beta != 0.0f) {
                __m512 c_old = loadu_masked(&Cb[(i + r) * N + j], len);
                c[r] = _mm512_fmadd_ps(_mm512_set1_ps(beta), c_old,
                                       _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]));
            }
            else {
                c[r] = _mm512_mul_ps(_mm512_set1_ps(alpha), c[r]);
            }

            storeu_masked(&Cb[(i + r) * N + j], apply_post_op(c[r], post_op), len);
        }
    }
}


void matmul_avx512_fp32_registerBlocking_batched(const float *A, const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K,
        int MR, int NR, bool transB, ActivationPostOp post_op, int batch) {

    zendnnInfo(ZENDNN_CORELOG,
               "Running Fused batched matmul_avx512_fp32_registerBlocking with MR x NR = : ",
               MR,
               " x ", NR,
               " kernel");

    for (int i = 0; i < M; i += MR) {
        for (int j = 0; j < N; j += NR) {
            if (MR == 1 && NR == 16) {
                compute_tile_1x16_batched(A, B, C, bias, alpha, beta, K, M, N, i, j, batch,
                                          transB, post_op);
            }
            else if (MR == 4 && NR == 16) {
                compute_tile_4x16_batched(A, B, C, bias, alpha, beta, K, M, N, i, j, batch,
                                          transB, post_op);
            }
            else if (MR == 6 && NR == 16) {
                compute_tile_6x16_batched(A, B, C, bias, alpha, beta, K, M, N, i, j, batch,
                                          transB, post_op);
            }
            else if (MR == 8 && NR == 16) {
                compute_tile_8x16_batched(A, B, C, bias, alpha, beta, K, M, N, i, j, batch,
                                          transB, post_op);
            }
            else if (MR == 12 && NR == 16) {
                compute_tile_12x16_batched(A, B, C, bias, alpha, beta, K, M, N, i, j, batch,
                                           transB, post_op);
            }
        }
    }
}


void matmul_avx512_fp32_registerBlocking_batched_auto(const float *A,
        const float *B,
        float *C,
        const float *bias, float alpha, float beta,
        int M, int N, int K,
        bool transB, ActivationPostOp post_op, int batch) {

    zendnnInfo(ZENDNN_CORELOG,
               "Running Fused batched matmul_avx512_fp32_registerBlocking with dynamic MR x NR kernel");

    for (int i = 0; i < M;) {
        int remaining = M - i;

        if (remaining >= 12) {
            for (int j = 0; j < N; j += 16) {
                compute_tile_12x16_batched(A, B, C, bias, alpha, beta, K, M, N, i, j, batch,
                                           transB, post_op);
            }
            i += 12;
        }
        else if (remaining >= 8) {
            for (int j = 0; j < N; j += 16) {
                compute_tile_8x16_batched(A, B, C, bias, alpha, beta, K, M, N, i, j, batch,
                                          transB, post_op);
            }
            i += 8;
        }
        else if (remaining >= 6) {
            for (int j = 0; j < N; j += 16) {
                compute_tile_6x16_batched(A, B, C, bias, alpha, beta, K, M, N, i, j, batch,
                                          transB, post_op);
            }
            i += 6;
        }
        else if (remaining >= 4) {
            for (int j = 0; j < N; j += 16) {
                compute_tile_4x16_batched(A, B, C, bias, alpha, beta, K, M, N, i, j, batch,
                                          transB, post_op);
            }
            i += 4;
        }
        else {
            for (int j = 0; j < N; j += 16) {
                compute_tile_1x16_batched(A, B, C, bias, alpha, beta, K, M, N, i, j, batch,
                                          transB, post_op);
            }
            i += 1;
        }
    }
}




}
}

