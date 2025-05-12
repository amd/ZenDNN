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

#ifndef ZENDNN_MATMUL_DIRECT_UTILS_HPP
#define ZENDNN_MATMUL_DIRECT_UTILS_HPP

#include <algorithm>
#include <cmath>
#include "zendnn.hpp"

#include <immintrin.h>

namespace zendnn {

void transpose_matrix(float *input, float *output, int N, int K);
float fast_exp_scalar(float x) ;

float fast_tanh_scalar(float x) ;

float apply_post_op_scalar(float val, ActivationPostOp post_op) ;

// Approximate exp(x) using a 5th-degree polynomial
__attribute__((target("avx512f")))
__m512 fast_exp_ps(__m512 x) ;

// Approximate tanh(x) using a rational function
__attribute__((target("avx512f")))
__m512 fast_tanh_ps(__m512 x) ;

__attribute__((target("avx512f")))
__m512 apply_post_op(__m512 vec, ActivationPostOp post_op) ;


}

#endif

