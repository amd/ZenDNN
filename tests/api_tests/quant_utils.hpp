/*******************************************************************************
* Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _QUANT_UTILS_
#define _QUANT_UTILS_

#include <cstdint>
#include "zendnn.hpp"

#define  QUANT_UTILS_SUCCESS   (0)
#define  QUANT_UTILS_FAILURE   (1)

using namespace zendnn;

union fp32bf16_t {
    fp32bf16_t(float ff):fp32(ff){}
    fp32bf16_t(uint16_t hf):bf16{0,hf}{}

    float       fp32;
    uint16_t    bf16[2];
};

inline uint16_t chfp32bf16(float ff) {
    return fp32bf16_t(ff).bf16[1];
}

inline float chbf16fp32(uint16_t hf) {
    return fp32bf16_t(hf).fp32;
}

inline int chfp32bf16mem(memory& output, memory& input) {
    // sanity check on input and output
    auto input_desc  = input.get_desc();
    auto output_desc = output.get_desc();

    if (input_desc.data_type() != memory::data_type::f32)
	return QUANT_UTILS_FAILURE;

    if ((output_desc.data_type() != memory::data_type::s16) &&
	(output_desc.data_type() != memory::data_type::bf16))
	return QUANT_UTILS_FAILURE;

    auto input_dims  = input_desc.dims();
    auto output_dims = output_desc.dims();

    if (input_dims != output_dims)
	return QUANT_UTILS_FAILURE;

    // number of elements in the tensor
    uint32_t nelem = 1;
    for (auto dim : input_dims)
	nelem *= dim;

    // conversion (little endian)
    float*    inptr    = reinterpret_cast<float *>(input.get_data_handle());
    uint16_t* outptr   = reinterpret_cast<uint16_t *>(output.get_data_handle());
    for (auto i = 0; i < nelem; ++i) {
	outptr[i] = chfp32bf16(inptr[i]);
    }

    return QUANT_UTILS_SUCCESS;
}

inline int chbf16fp32mem(memory& output, memory& input) {
    // sanity check on input and output
    auto input_desc  = input.get_desc();
    auto output_desc = output.get_desc();

    if (output_desc.data_type() != memory::data_type::f32)
	return QUANT_UTILS_FAILURE;

    if ((input_desc.data_type() != memory::data_type::s16) &&
	(input_desc.data_type() != memory::data_type::bf16))
	return QUANT_UTILS_FAILURE;

    auto input_dims  = input_desc.dims();
    auto output_dims = output_desc.dims();

    if (input_dims != output_dims)
	return QUANT_UTILS_FAILURE;

    // number of elements in the tensor
    uint32_t nelem = 1;
    for (auto dim : input_dims)
	nelem *= dim;

    // conversion (little endian)
    float*    outptr  = reinterpret_cast<float *>(output.get_data_handle());
    uint16_t* inptr   = reinterpret_cast<uint16_t *>(input.get_data_handle());
    for (auto i = 0; i < nelem; ++i) {
	outptr[i] = chbf16fp32(inptr[i]);
    }

    return QUANT_UTILS_SUCCESS;
}

#endif
