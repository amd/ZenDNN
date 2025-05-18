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

#ifndef _EMBAG_FP32_AVX512_UTILS_HPP_
#define _EMBAG_FP32_AVX512_UTILS_HPP_

#include <immintrin.h>

#define ZEN_MM_STRIDE_FP32_512    (16)
#define ZEN_MM_STRIDE_BF16_256    (16)
#define ZEN_MM_STRIDE_BF16_512    (32)

//fp32 type embedding bag
template<uint32_t DIM>
struct zenmmAVX512_ext_ps {
    zenmmAVX512_ext_ps() {
        for (auto i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_setzero_ps();
        }
    }

    inline void setzero_ps() {
        for (auto i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_setzero_ps();
        }
    };

    inline void load_ps(float const *mem) {
        for (auto i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_loadu_ps(mem);
            mem += ZEN_MM_STRIDE_FP32_512;
        }
    };

    inline void fetch_add_ps(float const *mem) {
        for (auto i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_add_ps(_mm512_loadu_ps(mem), v[i]);
            mem += ZEN_MM_STRIDE_FP32_512;
        }
    };


    inline void fetch_fmadd_ps(float const *mem, const float mfactor) {
        __m512 mm = _mm512_set1_ps(mfactor);
        for (auto i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_fmadd_ps(_mm512_loadu_ps(mem), mm, v[i]);
            mem += ZEN_MM_STRIDE_FP32_512;
        }

    };

    inline void fetch_max_ps(float const *mem) {
        for (auto i = 0; i< unroll_factor; ++i) {
            v[i] = _mm512_max_ps(_mm512_loadu_ps(mem), v[i]);
            mem += ZEN_MM_STRIDE_FP32_512;
        }
    };

    inline void store_ps(float *mem) {
        for (auto i = 0; i< unroll_factor; ++i) {
            _mm512_storeu_ps(mem,v[i]);
            mem += ZEN_MM_STRIDE_FP32_512;
        }
    };

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

using zenmmAVX512_ext_ps16   = zenmmAVX512_ext_ps<1>;
using zenmmAVX512_ext_ps32   = zenmmAVX512_ext_ps<2>;
using zenmmAVX512_ext_ps64   = zenmmAVX512_ext_ps<4>;
using zenmmAVX512_ext_ps128  = zenmmAVX512_ext_ps<8>;
using zenmmAVX512_ext_ps256  = zenmmAVX512_ext_ps<16>;
using zenmmAVX512_ext_ps512  = zenmmAVX512_ext_ps<32>;

#endif
