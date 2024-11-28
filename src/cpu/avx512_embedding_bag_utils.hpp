/*******************************************************************************
* Copyright (c) 2023-24 Advanced Micro Devices, Inc. All rights reserved.
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
*
*******************************************************************************/

#ifndef AVX512_EMBEDDING_BAG_UTILS_HPP
#define AVX512_EMBEDDING_BAG_UTILS_HPP

#include <immintrin.h>

#define ZEN_MM_STRIDE_FP32_512    (16)
#define ZEN_MM_STRIDE_BF16_256    (16)
#define ZEN_MM_STRIDE_BF16_512    (32)

template<typename input_type, typename dst_type, uint32_t DIM>
struct zenmmAVX512_ext_ps;

//fp32 type embedding bag
template<uint32_t DIM>
struct zenmmAVX512_ext_ps<float, float, DIM> {
    __attribute__((target("avx512f")))
    zenmmAVX512_ext_ps() {
        for (auto i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_setzero_ps();
        }
    }
    __attribute__((target("avx512f")))
    inline void setzero_ps() {
        for (auto i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_setzero_ps();
        }
    };
    __attribute__((target("avx512f")))
    inline void load_ps(float const *mem) {
        for (auto i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_loadu_ps(mem);
            mem += ZEN_MM_STRIDE_FP32_512;
        }
    };
    __attribute__((target("avx512f")))
    inline void fetch_add_ps(float const *mem) {
        for (auto i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_add_ps(_mm512_loadu_ps(mem), v[i]);
            mem += ZEN_MM_STRIDE_FP32_512;
        }
    };

    __attribute__((target("avx512f")))
    inline void fetch_fmadd_ps(float const *mem, const float mfactor) {
        __m512 mm = _mm512_set1_ps(mfactor);
        for (auto i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_fmadd_ps(_mm512_loadu_ps(mem), mm, v[i]);
            mem += ZEN_MM_STRIDE_FP32_512;
        }

    };
    __attribute__((target("avx512f")))
    inline void fetch_max_ps(float const *mem) {
        for (auto i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_max_ps(_mm512_loadu_ps(mem), v[i]);
            mem += ZEN_MM_STRIDE_FP32_512;
        }
    };
    __attribute__((target("avx512f")))
    inline void store_ps(float *mem) {
        for (auto i = 0; i< unroll_factor; ++i) {
            _mm512_storeu_ps(mem,v[i]);
            mem += ZEN_MM_STRIDE_FP32_512;
        }
    };
    __attribute__((target("avx512f")))
    inline void scale_store_ps(float *mem, const float mfactor) {
        __m512 mm = _mm512_set1_ps(mfactor);
        for (auto i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_mul_ps(v[i], mm);
            _mm512_storeu_ps(mem,v[i]);
            mem += ZEN_MM_STRIDE_FP32_512;
        }
    };

  private:
    __m512           v[DIM];
    const uint32_t   unroll_factor = DIM;
};
#if AVX512_BF16_EN
//bf16 type embedding bag
template<uint32_t DIM>
struct zenmmAVX512_ext_ps<zendnn::impl::bfloat16_t, float, DIM> {
    __attribute__((target("avx512f")))
    zenmmAVX512_ext_ps() {
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            v[i]     = _mm512_setzero_ps();
        }
    }
    __attribute__((target("avx512f")))
    inline void setzero_ps() {
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            v[i]     = _mm512_setzero_ps();
        }
    };
    __attribute__((target("avx512vl,avx512bf16")))
    inline void load_ps(zendnn::impl::bfloat16_t const *mem) {
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            __m256bh tbh = (__m256bh)(_mm256_loadu_epi32((void const *)mem));
            v[i]         = _mm512_cvtpbh_ps(tbh);
            mem         += ZEN_MM_STRIDE_BF16_256;
        }
    };
    __attribute__((target("avx512vl,avx512bf16")))
    inline void fetch_add_ps(zendnn::impl::bfloat16_t const *mem) {
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            __m256bh tbh = (__m256bh)(_mm256_loadu_epi32((void const *)mem));
            __m512   tps = _mm512_cvtpbh_ps(tbh);
            v[i]         = _mm512_add_ps(tps, v[i]);
            mem         += ZEN_MM_STRIDE_BF16_256;
        }
    };

    __attribute__((target("avx512vl,avx512bf16")))
    inline void fetch_fmadd_ps(zendnn::impl::bfloat16_t const *mem,
                               const float mfactor) {
        __m512 mm = _mm512_set1_ps(mfactor);
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            __m256bh tbh = (__m256bh)(_mm256_loadu_epi32((void const *)mem));
            __m512   tps = _mm512_cvtpbh_ps(tbh);
            v[i]         = _mm512_fmadd_ps(tps, mm, v[i]);
            mem         += ZEN_MM_STRIDE_BF16_256;
        }

    };
    __attribute__((target("avx512vl,avx512bf16")))
    inline void fetch_max_ps(zendnn::impl::bfloat16_t const *mem) {
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            __m256bh tbh = (__m256bh)(_mm256_loadu_epi32((void const *)mem));
            __m512   tps = _mm512_cvtpbh_ps(tbh);
            v[i]         = _mm512_max_ps(tps, v[i]);
            mem         += ZEN_MM_STRIDE_BF16_256;
        }
    };
    __attribute__((target("avx512f")))
    inline void store_ps(float *mem) {
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            _mm512_storeu_ps(mem, v[i]);
            mem += ZEN_MM_STRIDE_FP32_512;
        }
    };
    __attribute__((target("avx512f")))
    inline void scale_store_ps(float *mem, const float mfactor) {
        __m512 mm = _mm512_set1_ps(mfactor);
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_mul_ps(v[i], mm);
            _mm512_storeu_ps(mem,v[i]);
            mem += ZEN_MM_STRIDE_FP32_512;
        }
    };

  private:
    __m512             v[DIM];
    const uint32_t     unroll_factor = DIM;
};

template<uint32_t DIM>
struct zenmmAVX512_ext_ps<zendnn::impl::bfloat16_t, zendnn::impl::bfloat16_t, DIM> {
    __attribute__((target("avx512f")))
    zenmmAVX512_ext_ps() {
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            v[i]     = _mm512_setzero_ps();
        }
    }
    __attribute__((target("avx512f")))
    inline void setzero_ps() {
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            v[i]     = _mm512_setzero_ps();
        }
    };
    __attribute__((target("avx512vl,avx512bf16")))
    inline void load_ps(zendnn::impl::bfloat16_t const *mem) {
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            __m256bh tbh = (__m256bh)(_mm256_loadu_epi32((void const *)mem));
            v[i]         = _mm512_cvtpbh_ps(tbh);
            mem         += ZEN_MM_STRIDE_BF16_256;
        }
    };
    __attribute__((target("avx512vl,avx512bf16")))
    inline void fetch_add_ps(zendnn::impl::bfloat16_t const *mem) {
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            __m256bh tbh = (__m256bh)(_mm256_loadu_epi32((void const *)mem));
            __m512   tps = _mm512_cvtpbh_ps(tbh);
            v[i]         = _mm512_add_ps(tps, v[i]);
            mem         += ZEN_MM_STRIDE_BF16_256;
        }
    };

    __attribute__((target("avx512vl,avx512bf16")))
    inline void fetch_fmadd_ps(zendnn::impl::bfloat16_t const *mem,
                               const float mfactor) {
        __m512 mm = _mm512_set1_ps(mfactor);
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            __m256bh tbh = (__m256bh)(_mm256_loadu_epi32((void const *)mem));
            __m512   tps = _mm512_cvtpbh_ps(tbh);
            v[i]         = _mm512_fmadd_ps(tps, mm, v[i]);
            mem         += ZEN_MM_STRIDE_BF16_256;
        }

    };
    __attribute__((target("avx512vl,avx512bf16")))
    inline void fetch_max_ps(zendnn::impl::bfloat16_t const *mem) {
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            __m256bh tbh = (__m256bh)(_mm256_loadu_epi32((void const *)mem));
            __m512   tps = _mm512_cvtpbh_ps(tbh);
            v[i]         = _mm512_max_ps(tps, v[i]);
            mem         += ZEN_MM_STRIDE_BF16_256;
        }
    };
    __attribute__((target("avx512vl,avx512bf16")))
    inline void store_ps(zendnn::impl::bfloat16_t *mem) {
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            __m256bh res_bf16 = _mm512_cvtneps_pbh(v[i]);
            _mm256_storeu_epi32((void *)mem, (__m256i)res_bf16);
            mem += ZEN_MM_STRIDE_BF16_256;
        }
    };
    __attribute__((target("avx512vl,avx512bf16")))
    inline void scale_store_ps(zendnn::impl::bfloat16_t *mem, const float mfactor) {
        __m512 mm = _mm512_set1_ps(mfactor);
        for (uint32_t i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_mul_ps(v[i], mm);
            __m256bh res_bf16 = _mm512_cvtneps_pbh(v[i]);
            _mm256_storeu_epi32((void *)mem, (__m256i)res_bf16);
            mem += ZEN_MM_STRIDE_BF16_256;
        }
    };

  private:
    __m512             v[DIM];
    const uint32_t     unroll_factor = DIM;
};
#endif

// Templated embedding bag sum function
template <typename dst_type, typename input_type>
void emb_sum(dst_type *sum, const input_type *input, uint32_t width,
             uint32_t input_offset, float wt) {
    for (uint32_t j = 0; j < width; ++j) {
        //Convert input to float, multiply by wt, and add to sum
        sum[j] += dst_type(input[j + input_offset]) * wt;
    }
}

// Templated embedding bag sum function
template <typename dst_type, typename input_type>
void emb_max(dst_type *sum, const input_type *input, uint32_t width,
             uint32_t input_offset) {
    for (uint32_t j = 0; j < width; ++j) {
        // Convert input to float, multiply by wt, and add to sum
        if (sum[j] < dst_type(input[j + input_offset])) {
            sum[j] = dst_type(input[j + input_offset]);
        }
    }
}

using zenmmAVX512_ext_ps16   = zenmmAVX512_ext_ps<float, float, 1>;
using zenmmAVX512_ext_ps32   = zenmmAVX512_ext_ps<float, float, 2>;
using zenmmAVX512_ext_ps64   = zenmmAVX512_ext_ps<float, float, 4>;
using zenmmAVX512_ext_ps128  = zenmmAVX512_ext_ps<float, float, 8>;
using zenmmAVX512_ext_ps256  = zenmmAVX512_ext_ps<float, float, 16>;
using zenmmAVX512_ext_ps512  = zenmmAVX512_ext_ps<float, float, 32>;

#if AVX512_BF16_EN
using zenmmAVX512_ext_pbf16  =
    zenmmAVX512_ext_ps<zendnn::impl::bfloat16_t, zendnn::impl::bfloat16_t, 1>;
using zenmmAVX512_ext_pbf32  =
    zenmmAVX512_ext_ps<zendnn::impl::bfloat16_t, zendnn::impl::bfloat16_t, 2>;
using zenmmAVX512_ext_pbf64  =
    zenmmAVX512_ext_ps<zendnn::impl::bfloat16_t, zendnn::impl::bfloat16_t, 4>;
using zenmmAVX512_ext_pbf128 =
    zenmmAVX512_ext_ps<zendnn::impl::bfloat16_t, zendnn::impl::bfloat16_t, 8>;
using zenmmAVX512_ext_pbf256 =
    zenmmAVX512_ext_ps<zendnn::impl::bfloat16_t, zendnn::impl::bfloat16_t, 16>;
using zenmmAVX512_ext_pbf512 =
    zenmmAVX512_ext_ps<zendnn::impl::bfloat16_t, zendnn::impl::bfloat16_t, 32>;

using zenmmAVX512_ext_pbf_ps16  =
    zenmmAVX512_ext_ps<zendnn::impl::bfloat16_t, float, 1>;
using zenmmAVX512_ext_pbf_ps32  =
    zenmmAVX512_ext_ps<zendnn::impl::bfloat16_t, float, 2>;
using zenmmAVX512_ext_pbf_ps64  =
    zenmmAVX512_ext_ps<zendnn::impl::bfloat16_t, float, 4>;
using zenmmAVX512_ext_pbf_ps128 =
    zenmmAVX512_ext_ps<zendnn::impl::bfloat16_t, float, 8>;
using zenmmAVX512_ext_pbf_ps256 =
    zenmmAVX512_ext_ps<zendnn::impl::bfloat16_t, float, 16>;
using zenmmAVX512_ext_pbf_ps512 =
    zenmmAVX512_ext_ps<zendnn::impl::bfloat16_t, float, 32>;
#endif
#endif
