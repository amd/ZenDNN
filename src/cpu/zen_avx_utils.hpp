/*******************************************************************************
* Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ZEN_AVX_UTILS_HPP
#define ZEN_AVX_UTILS_HPP

#include <immintrin.h>

#define ZEN_MM_PS_STRIDE    (8)

//
// define a class for vectors longer than that can be accomodated by AVX2 registers
//

template<uint32_t DIM=8>
struct zenmm_ext_ps {

    static_assert(!(DIM & 0x07), "zenmm_ext_ps: DIM needs to be multiple of 8");

    zenmm_ext_ps() {
        for (auto i = 0; i< unroll_factor; ++i ) {
            const uint32_t offset = (i << 3);
            v[offset + 0] = _mm256_setzero_ps();
            v[offset + 1] = _mm256_setzero_ps();
            v[offset + 2] = _mm256_setzero_ps();
            v[offset + 3] = _mm256_setzero_ps();
            v[offset + 4] = _mm256_setzero_ps();
            v[offset + 5] = _mm256_setzero_ps();
            v[offset + 6] = _mm256_setzero_ps();
            v[offset + 7] = _mm256_setzero_ps();
        }
    }

    inline void setzero_ps(){
        for (auto i = 0; i< unroll_factor; ++i ) {
            const uint32_t offset = (i << 3);
            v[offset + 0] = _mm256_setzero_ps();
            v[offset + 1] = _mm256_setzero_ps();
            v[offset + 2] = _mm256_setzero_ps();
            v[offset + 3] = _mm256_setzero_ps();
            v[offset + 4] = _mm256_setzero_ps();
            v[offset + 5] = _mm256_setzero_ps();
            v[offset + 6] = _mm256_setzero_ps();
            v[offset + 7] = _mm256_setzero_ps();
        }
    };

    inline void load_ps(float const* mem){
        for (auto i = 0; i< unroll_factor; ++i ) {
            const uint32_t offset = (i << 3);
            v[offset + 0] = _mm256_load_ps(mem);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 1] = _mm256_load_ps(mem);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 2] = _mm256_load_ps(mem);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 3] = _mm256_load_ps(mem);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 4] = _mm256_load_ps(mem);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 5] = _mm256_load_ps(mem);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 6] = _mm256_load_ps(mem);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 7] = _mm256_load_ps(mem);
            mem += ZEN_MM_PS_STRIDE;
        }
    };

    inline void fetch_add_ps(float const* mem){
        for (auto i = 0; i< unroll_factor; ++i ) {
            const uint32_t offset = (i << 3);
            v[offset + 0] = _mm256_add_ps(_mm256_load_ps(mem), v[offset + 0]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 1] = _mm256_add_ps(_mm256_load_ps(mem), v[offset + 1]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 2] = _mm256_add_ps(_mm256_load_ps(mem), v[offset + 2]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 3] = _mm256_add_ps(_mm256_load_ps(mem), v[offset + 3]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 4] = _mm256_add_ps(_mm256_load_ps(mem), v[offset + 4]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 5] = _mm256_add_ps(_mm256_load_ps(mem), v[offset + 5]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 6] = _mm256_add_ps(_mm256_load_ps(mem), v[offset + 6]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 7] = _mm256_add_ps(_mm256_load_ps(mem), v[offset + 7]);
            mem += ZEN_MM_PS_STRIDE;
        }
    };


    inline void fetch_fmadd_ps(float const* mem, const float mfactor) {
        __m256 mm = _mm256_set1_ps(mfactor);
        for (auto i = 0; i< unroll_factor; ++i ) {
            const uint32_t offset = (i << 3);
            v[offset + 0] = _mm256_fmadd_ps(_mm256_load_ps(mem), mm, v[offset + 0]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 1] = _mm256_fmadd_ps(_mm256_load_ps(mem), mm, v[offset + 1]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 2] = _mm256_fmadd_ps(_mm256_load_ps(mem), mm, v[offset + 2]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 3] = _mm256_fmadd_ps(_mm256_load_ps(mem), mm, v[offset + 3]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 4] = _mm256_fmadd_ps(_mm256_load_ps(mem), mm, v[offset + 4]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 5] = _mm256_fmadd_ps(_mm256_load_ps(mem), mm, v[offset + 5]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 6] = _mm256_fmadd_ps(_mm256_load_ps(mem), mm, v[offset + 6]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 7] = _mm256_fmadd_ps(_mm256_load_ps(mem), mm, v[offset + 7]);
            mem += ZEN_MM_PS_STRIDE;
        }

    };

    inline void fetch_max_ps(float const* mem){
        for (auto i = 0; i< unroll_factor; ++i ) {
            const uint32_t offset = (i << 3);
            v[offset + 0] = _mm256_max_ps(_mm256_load_ps(mem), v[offset + 0]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 1] = _mm256_max_ps(_mm256_load_ps(mem), v[offset + 1]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 2] = _mm256_max_ps(_mm256_load_ps(mem), v[offset + 2]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 3] = _mm256_max_ps(_mm256_load_ps(mem), v[offset + 3]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 4] = _mm256_max_ps(_mm256_load_ps(mem), v[offset + 4]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 5] = _mm256_max_ps(_mm256_load_ps(mem), v[offset + 5]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 6] = _mm256_max_ps(_mm256_load_ps(mem), v[offset + 6]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset + 7] = _mm256_max_ps(_mm256_load_ps(mem), v[offset + 7]);
            mem += ZEN_MM_PS_STRIDE;
        }
    };

    inline void store_ps(float* mem) {
        for (auto i = 0; i< unroll_factor; ++i ) {
            const uint32_t offset = (i << 3);
            _mm256_store_ps(mem,v[offset +0]);
            mem += ZEN_MM_PS_STRIDE;
            _mm256_store_ps(mem,v[offset +1]);
            mem += ZEN_MM_PS_STRIDE;
            _mm256_store_ps(mem,v[offset +2]);
            mem += ZEN_MM_PS_STRIDE;
            _mm256_store_ps(mem,v[offset +3]);
            mem += ZEN_MM_PS_STRIDE;
            _mm256_store_ps(mem,v[offset +4]);
            mem += ZEN_MM_PS_STRIDE;
            _mm256_store_ps(mem,v[offset +5]);
            mem += ZEN_MM_PS_STRIDE;
            _mm256_store_ps(mem,v[offset +6]);
            mem += ZEN_MM_PS_STRIDE;
            _mm256_store_ps(mem,v[offset +7]);
            mem += ZEN_MM_PS_STRIDE;
        }
    };

    inline void scale_store_ps(float* mem, const float mfactor) {
        __m256 mm = _mm256_set1_ps(mfactor);
        for (auto i = 0; i< unroll_factor; ++i ) {
            const uint32_t offset = (i << 3);
            v[offset +0] = _mm256_mul_ps(v[offset +0], mm);
            _mm256_store_ps(mem,v[offset +0]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset +1] = _mm256_mul_ps(v[offset +1], mm);
            _mm256_store_ps(mem,v[offset +1]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset +2] = _mm256_mul_ps(v[offset +2], mm);
            _mm256_store_ps(mem,v[offset +2]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset +3] = _mm256_mul_ps(v[offset +3], mm);
            _mm256_store_ps(mem,v[offset +3]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset +4] = _mm256_mul_ps(v[offset +4], mm);
            _mm256_store_ps(mem,v[offset +4]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset +5] = _mm256_mul_ps(v[offset +5], mm);
            _mm256_store_ps(mem,v[offset +5]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset +6] = _mm256_mul_ps(v[offset +6], mm);
            _mm256_store_ps(mem,v[offset +6]);
            mem += ZEN_MM_PS_STRIDE;
            v[offset +7] = _mm256_mul_ps(v[offset +7], mm);
            _mm256_store_ps(mem,v[offset +7]);
            mem += ZEN_MM_PS_STRIDE;
        }
    };

private:
    __m256           v[DIM];
    const uint32_t   unroll_factor = (DIM >> 3);
};

using zenmm_ext_ps64=zenmm_ext_ps<8>;
using zenmm_ext_ps128=zenmm_ext_ps<16>;

#endif
