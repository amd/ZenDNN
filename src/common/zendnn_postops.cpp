/*******************************************************************************
* Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/zendnn_private.hpp"
#include <omp.h>

#ifndef ZENDNN_USE_AOCL_BLIS_API
    #include <cblas.h>
#else // ZENDNN_USE_AOCL_BLIS_API
    #include "cblas_with_blis_api.hpp"
#endif // ZENDNN_USE_AOCL_BLIS_API
#include <time.h>
#include "zendnn_helper.hpp"
#include "zendnn_logging.hpp"
#include <blis.h>


float gelu_const = sqrtf(2/M_PI);

#define GELU_VECTOR_ENABLE      1

#if GELU_VECTOR_ENABLE
    #define COMPUTE_GELU    COMPUTE_GELU_VEC16
    #define COMPUTE_GELU_TANH   COMPUTE_GELU_TANH_VEC16
    #define COMPUTE_GELU_ERF    COMPUTE_GELU_ERF_VEC16
#else
    #define COMPUTE_GELU    COMPUTE_GELU_VEC1
    #define COMPUTE_GELU_TANH   COMPUTE_GELU_TANH_VEC1
    #define COMPUTE_GELU_ERF    COMPUTE_GELU_ERF_VEC1
#endif

#define COMPUTE_BIAS_VEC16(out_layer, scale, bias, elementwise_input, offset, c) \
    { \
        for(int i=0;i<16;++i) \
        out_layer[offset+c+i] = out_layer[offset+c+i] + bias[c+i]; \
    }

#define COMPUTE_SCALE_BIAS_VEC16(out_layer, scale, bias, elementwise_input, offset, c) \
    { \
        for(int i=0;i<16;++i) \
        out_layer[offset+c+i] = out_layer[offset+c+i]*scale[c+i] + bias[c+i]; \
    }

#define COMPUTE_ADD_VEC16(out_layer, scale, bias, elementwise_input, offset, c) \
    { \
        for(int i=0;i<16;++i) \
        out_layer[offset+c+i] = out_layer[offset+c+i] + elementwise_input[offset + c + i]; \
    }

#define COMPUTE_BIAS_ADD_VEC16(out_layer, scale, bias, elementwise_input, offset, c) \
    { \
        for(int i=0;i<16;++i) \
        out_layer[offset+c+i] = out_layer[offset+c+i] + bias[c+i] + elementwise_input[offset + c + i]; \
}

#define COMPUTE_SCALE_BIAS_ADD_VEC16(out_layer, scale, bias, elementwise_input, offset, c) \
    { \
        for(int i=0;i<16;++i) \
        out_layer[offset+c+i] = out_layer[offset+c+i]*scale[c+i] + bias[c+i] + elementwise_input[offset + c + i]; \
    }

#define COMPUTE_GELU_TANH_VEC16() \
    { \
        aocl_gelu_tanh_f32(16, out_layer+offset+c, 1); \
    }

#define COMPUTE_GELU_ERF_VEC16() \
    { \
        aocl_gelu_erf_f32(16, out_layer+offset+c, 1); \
    }

#define COMPUTE_NONE_VEC16(out_layer, scale, bias, elementwise_input, offset, c) \
    { \
    }

#define COMPUTE_BIAS_VEC1(out_layer, scale, bias, elementwise_input, offset, c) \
    { \
        out_layer[offset+c] = out_layer[offset+c] + bias[c]; \
    }

#define COMPUTE_SCALE_BIAS_VEC1(out_layer, scale, bias, elementwise_input, offset, c) \
    { \
        out_layer[offset+c] = out_layer[offset+c]*scale[c] + bias[c]; \
    }

#define COMPUTE_ADD_VEC1(out_layer, scale, bias, elementwise_input, offset, c) \
    { \
        out_layer[offset+c] = out_layer[offset+c] + elementwise_input[offset + c]; \
    }

#define COMPUTE_BIAS_ADD_VEC1(out_layer, scale, bias, elementwise_input, offset, c) \
    { \
        out_layer[offset+c] = out_layer[offset+c] + bias[c] + elementwise_input[offset + c]; \
    }

#define COMPUTE_SCALE_BIAS_ADD_VEC1(out_layer, scale, bias, elementwise_input, offset, c) \
    { \
        out_layer[offset+c] = out_layer[offset+c]*scale[c] + bias[c] + elementwise_input[offset + c]; \
    }

#define COMPUTE_GELU_TANH_VEC1() \
    { \
        out_layer[ offset + c ] = 0.5 * out_layer[ offset + c ] * \
                                  (1 + tanhf(gelu_const * (out_layer[ offset + c ] \
                                  + 0.044715 * powf(out_layer[ offset + c ],3)))); \
    }

#define COMPUTE_GELU_ERF_VEC1() \
    { \
        out_layer[ offset + c ] = 0.5 * out_layer[ offset + c ] * \
                                  (1 + erff(out_layer[ offset + c ]/1.414213)); \
    }

#define COMPUTE_NONE_VEC1(out_layer, scale, bias, elementwise_input, offset, c) \
    { \
    }

#define COMPUTE_GELU_VEC16(out_layer, scale, bias, elementwise_input, biasOffset, i, no_of_filter, compute_postOp, compute_gelu_type) \
    { \
        unsigned int offset = biasOffset + i; \
        int c = 0; \
        for (c = 0; (c+16) <= no_of_filter; c+=16) { \
            compute_postOp##_VEC16(out_layer, scale, bias, elementwise_input, offset, c); \
                                    \
            compute_gelu_type##_VEC16();  \
                                    \
        } \
        for( ;c<no_of_filter; c++) \
        { \
            compute_postOp##_VEC1(out_layer, scale, bias, elementwise_input, offset, c); \
                                    \
            compute_gelu_type##_VEC1();  \
                                    \
        } \
    }

#define COMPUTE_GELU_VEC1(out_layer, scale, bias, elementwise_input, biasOffset, i, no_of_filter, compute_postOp, compute_gelu_type) \
    { \
        unsigned int offset = biasOffset + i; \
        for (int c = 0; c < no_of_filter; c++) { \
            compute_postOp##_VEC1(out_layer, scale, bias, elementwise_input, offset, c); \
                                    \
            compute_gelu_type##_VEC1();  \
                                    \
        } \
    }

using namespace zendnn;
//ZenClip clips the output values based on upperbound
void zenClipOp(zendnnEnv zenEnvObj,float *out_layer,float upper_bound,
               unsigned long size) {
    int remainder = size%8;
    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (unsigned long i=0; i < size-remainder; i+=8) {
        #pragma omp simd
        for (int j=0; j <=7; j++) {
            if (out_layer[i+j] > upper_bound) {
                out_layer[i+j] = upper_bound;
            }
        }
    }

    for (unsigned long k=size-remainder; k < size; k++) {
        if (out_layer[k] > upper_bound) {
            out_layer[k] = upper_bound;
        }
    }
}

void zenPostOps(
    zendnnEnv zenEnvObj,
    float *out_layer,
    const float *elementwise_input,
    const int out_height,
    const int out_width,
    const int no_of_filter,
    const int total_filters,
    unsigned long biasOffset,
    const float *bias,
    const bool relu,
    const int gelu,
    const float *scale,
    const int no_of_threads,
    const float *offset,
    const float  *mean,
    const int batch_size,
    const float alpha
) {

    if (zenEnvObj.zenConvAlgo!=zenConvAlgoType::DIRECT1) {  // NHWC Path

        unsigned long i;
        unsigned long total_size = (unsigned long)out_height*out_width*total_filters;
        if (!elementwise_input) {
            if (relu) {
                if (bias != NULL && scale != NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        #pragma omp simd
                        for (int c = 0; c < no_of_filter; c++) {
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] * scale[c] +
                                                              bias[c];
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c ]>0
                                                              ?out_layer[ biasOffset + i + c ]
                                                              :(alpha==0.0f)?alpha
                                                              :out_layer[ biasOffset + i + c ]*alpha;
                        }
                }
                else if (bias != NULL && scale == NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        #pragma omp simd
                        for (int c = 0; c < no_of_filter; c++) {
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] + bias[c];
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c ]>0
                                                              ?out_layer[ biasOffset + i + c ]
                                                              :(alpha==0.0f)?alpha
                                                              :out_layer[ biasOffset + i + c ]*alpha;
                        }
                }
                else if (bias == NULL && scale == NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        #pragma omp simd
                        for (int c = 0; c < no_of_filter; c++) {
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c ]>0
                                                              ?out_layer[ biasOffset + i + c ]
                                                              :(alpha==0.0f)?alpha
                                                              :out_layer[ biasOffset + i + c ]*alpha;
                        }
                }
            }
            else if (gelu) {

                //gelu=1 is tanh based gelu, else(i.e gelu=2) is
                // erf based
                if (gelu==1) {
                    if (bias != NULL && scale != NULL) {
                        #pragma omp parallel for num_threads(no_of_threads)
                        for (i = 0; i < total_size; i += total_filters) {
                            COMPUTE_GELU(out_layer, scale, bias, elementwise_input, biasOffset, i,
                                         no_of_filter, COMPUTE_SCALE_BIAS, COMPUTE_GELU_TANH);
                        }
                    }
                    else if (bias != NULL && scale == NULL) {
                        #pragma omp parallel for num_threads(no_of_threads)
                        for (i = 0; i < total_size; i += total_filters) {
                            COMPUTE_GELU(out_layer, scale, bias, elementwise_input, biasOffset, i,
                                         no_of_filter, COMPUTE_BIAS, COMPUTE_GELU_TANH);
                        }
                    }
                    else if (bias == NULL && scale == NULL) {
                        #pragma omp parallel for num_threads(no_of_threads)
                        for (i = 0; i < total_size; i += total_filters) {
                            COMPUTE_GELU(out_layer, scale, bias, elementwise_input, biasOffset, i,
                                         no_of_filter, COMPUTE_NONE, COMPUTE_GELU_TANH);
                        }
                    }
                }
                else { //erf based gelu
                    if (bias != NULL && scale != NULL) {
                        #pragma omp parallel for num_threads(no_of_threads)
                        for (i = 0; i < total_size; i += total_filters)
                            COMPUTE_GELU(out_layer, scale, bias, elementwise_input, biasOffset, i,
                                         no_of_filter, COMPUTE_SCALE_BIAS, COMPUTE_GELU_ERF);
                    }
                    else if (bias != NULL && scale == NULL) {
                        #pragma omp parallel for num_threads(no_of_threads)
                        for (i = 0; i < total_size; i += total_filters)
                            COMPUTE_GELU(out_layer, scale, bias, elementwise_input, biasOffset, i,
                                         no_of_filter, COMPUTE_BIAS, COMPUTE_GELU_ERF);
                    }
                    else if (bias == NULL && scale == NULL) {
                        #pragma omp parallel for num_threads(no_of_threads)
                        for (i = 0; i < total_size; i += total_filters)
                            COMPUTE_GELU(out_layer, scale, bias, elementwise_input, biasOffset, i,
                                         no_of_filter, COMPUTE_NONE, COMPUTE_GELU_ERF);
                    }
                }
            }
            else {
                if (bias != NULL && scale != NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        #pragma omp simd
                        for (int c = 0; c < no_of_filter; c++) {
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] * scale[c] +
                                                              bias[c];
                        }
                }
                else if (bias != NULL &&  scale == NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        #pragma omp simd
                        for (int c = 0; c < no_of_filter; c++) {
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] + bias[c];
                        }
                }
            }
        }
        else {
            if (relu) {
                if (bias != NULL && scale != NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        #pragma omp simd
                        for (int c = 0; c < no_of_filter; c++) {
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] * scale[c] +
                                                              bias[c] + elementwise_input[biasOffset + i + c];
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c ]>0
                                                              ?out_layer[ biasOffset + i + c ]
                                                              :(alpha==0.0f)?alpha
                                                              :out_layer[ biasOffset + i + c ]*alpha;

                        }
                }
                else if (bias != NULL && scale == NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        #pragma omp simd
                        for (int c = 0; c < no_of_filter; c++) {
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] + bias[c] +
                                                              elementwise_input[biasOffset + i + c];
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c ]>0
                                                              ?out_layer[ biasOffset + i + c ]
                                                              :(alpha==0.0f)?alpha
                                                              :out_layer[ biasOffset + i + c ]*alpha;
                        }
                }
                else if (bias == NULL && scale == NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        #pragma omp simd
                        for (int c = 0; c < no_of_filter; c++) {
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] +
                                                              elementwise_input[biasOffset + i + c];
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c ]>0
                                                              ?out_layer[ biasOffset + i + c ]
                                                              :(alpha==0.0f)?alpha
                                                              :out_layer[ biasOffset + i + c ]*alpha;
                        }
                }
            }
            else if (gelu) {

                //gelu=1 is tanh based gelu, else(i.e gelu=2) is
                // erf based
                if (gelu==1) {
                    if (bias != NULL && scale != NULL) {
                        #pragma omp parallel for num_threads(no_of_threads)
                        for (i = 0; i < total_size; i += total_filters)
                            #pragma omp simd
                            for (int c = 0; c < no_of_filter; c++) {
                                COMPUTE_GELU(out_layer, scale, bias, elementwise_input, biasOffset, i,
                                             no_of_filter, COMPUTE_SCALE_BIAS_ADD, COMPUTE_GELU_TANH);
                            }
                    }
                    else if (bias != NULL && scale == NULL) {
                        #pragma omp parallel for num_threads(no_of_threads)
                        for (i = 0; i < total_size; i += total_filters)
                            #pragma omp simd
                            for (int c = 0; c < no_of_filter; c++) {
                                COMPUTE_GELU(out_layer, scale, bias, elementwise_input, biasOffset, i,
                                             no_of_filter, COMPUTE_BIAS_ADD, COMPUTE_GELU_TANH);
                            }
                    }
                    else if (bias == NULL && scale == NULL) {
                        #pragma omp parallel for num_threads(no_of_threads)
                        for (i = 0; i < total_size; i += total_filters)
                            #pragma omp simd
                            for (int c = 0; c < no_of_filter; c++) {
                                COMPUTE_GELU(out_layer, scale, bias, elementwise_input, biasOffset, i,
                                             no_of_filter, COMPUTE_ADD, COMPUTE_GELU_TANH);
                            }
                    }
                }
                else { //erf based gelu
                    if (bias != NULL && scale != NULL) {
                        #pragma omp parallel for num_threads(no_of_threads)
                        for (i = 0; i < total_size; i += total_filters)
                            COMPUTE_GELU(out_layer, scale, bias, elementwise_input, biasOffset, i,
                                         no_of_filter, COMPUTE_SCALE_BIAS_ADD, COMPUTE_GELU_ERF);
                    }
                    else if (bias != NULL && scale == NULL) {
                        #pragma omp parallel for num_threads(no_of_threads)
                        for (i = 0; i < total_size; i += total_filters)
                            COMPUTE_GELU(out_layer, scale, bias, elementwise_input, biasOffset, i,
                                         no_of_filter, COMPUTE_BIAS_ADD, COMPUTE_GELU_ERF);
                    }
                    else if (bias == NULL && scale == NULL) {
                        #pragma omp parallel for num_threads(no_of_threads)
                        for (i = 0; i < total_size; i += total_filters)
                            COMPUTE_GELU(out_layer, scale, bias, elementwise_input, biasOffset, i,
                                         no_of_filter, COMPUTE_ADD, COMPUTE_GELU_ERF);
                    }
                }
            }
            else {
                if (bias != NULL && scale != NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        #pragma omp simd
                        for (int c = 0; c < no_of_filter; c++) {
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] * scale[c] +
                                                              bias[c] + elementwise_input[biasOffset + i + c];
                        }
                }
                else if (bias != NULL && scale == NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        #pragma omp simd
                        for (int c = 0; c < no_of_filter; c++) {
                            out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] + bias[c] +
                                                              elementwise_input[biasOffset + i + c];
                        }
                }
            }
        }
    }
    else  {
#ifdef _WIN32
        auto start = std::chrono::high_resolution_clock::now();
#else
        struct timeval start, end;
        gettimeofday(&start, 0);
#endif

        // This section of the code enables Batchorm , Elementwise & Relu support for Blocked Format
        int filter_block = no_of_filter/8;          // Assumes Filters are multiple of 8
        // If Filters are not multiple of 8 , source call should ensure padding
        unsigned long index = 0;
        unsigned long blocked_out_height_width = 8*out_height*out_width;
        if (scale) {

            if (relu) {
                if (elementwise_input) { // Batchnorm and element wise
                    #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                    for (int i=0; i< batch_size; i++)
                        for (int r=0; r< filter_block; r++) {
                            index = blocked_out_height_width*(i*filter_block + r);
                            unsigned long index_filter = 8*r;
                            #pragma omp simd
                            for (int m=0; m< blocked_out_height_width; m=m+8) {
                                for (int n=0; n < 8; n++) {
                                    out_layer[index + m + n]  = scale[index_filter + n]*(out_layer[index + m + n] -
                                                                mean[index_filter + n])
                                                                + offset[index_filter + n]  + elementwise_input[index + m + n];
                                    out_layer[index + m + n]=out_layer[index + m + n]>0
                                                             ? out_layer[index + m + n] :
                                                             (alpha==0.0f) ? alpha
                                                             : out_layer[index + m + n] * alpha;
                                }
                            }
                        }
                }
                else { // Batchnorm Only
                    #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                    for (int i=0; i< batch_size; i++)
                        for (int r=0; r< filter_block; r++) {
                            index = blocked_out_height_width*(i*filter_block + r);
                            unsigned long index_filter = 8*r;
                            #pragma omp simd
                            for (int m=0; m< blocked_out_height_width; m=m+8) {
                                for (int n=0; n < 8; n++) {
                                    out_layer[index + m +n]  = scale[index_filter + n]*(out_layer[index + m + n] -
                                                               mean[index_filter + n])
                                                               + offset[index_filter + n];
                                    out_layer[index + m + n]=out_layer[index + m + n]>0
                                                             ? out_layer[index + m + n] :
                                                             (alpha==0.0f) ? alpha
                                                             : out_layer[index + m + n]*alpha ;
                                }
                            }
                        }
                }
            }
            else if (gelu) {

                //gelu=1 is tanh based gelu, else(i.e gelu=2) is
                // erf based
                if (gelu==1) {
                    if (elementwise_input) { // Batchnorm and element wise
                        #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                        for (int i=0; i< batch_size; i++)
                            for (int r=0; r< filter_block; r++) {
                                index = blocked_out_height_width*(i*filter_block + r);
                                unsigned long index_filter = 8*r;
                                #pragma omp simd
                                for (int m=0; m< blocked_out_height_width; m=m+8) {
                                    for (int n=0; n < 8; n++) {
                                        out_layer[index+m + n]  = scale[index_filter + n]*(out_layer[index+m + n] -
                                                                  mean[index_filter + n])
                                                                  + offset[index_filter + n]  + elementwise_input[index+m + n];
                                        out_layer[index + m + n] = 0.5 * out_layer[index + m + n] * (1 + tanhf(
                                                                       gelu_const *
                                                                       (out_layer[index + m + n] + 0.044715 * powf(
                                                                            out_layer[index + m + n],3))));
                                    }
                                }
                            }
                    }
                    else { // Batchnorm Only
                        #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                        for (int i=0; i< batch_size; i++)
                            for (int r=0; r< filter_block; r++) {
                                index = blocked_out_height_width*(i*filter_block + r);
                                unsigned long index_filter = 8*r;
                                #pragma omp simd
                                for (int m=0; m< blocked_out_height_width; m=m+8) {
                                    for (int n=0; n < 8; n++) {
                                        out_layer[index+m +n]  = scale[index_filter + n]*(out_layer[index+m+n] -
                                                                 mean[index_filter + n])
                                                                 + offset[index_filter + n];
                                        out_layer[index + m + n] = 0.5 * out_layer[index + m + n] * (1 + tanhf(
                                                                       gelu_const *
                                                                       (out_layer[index + m + n] + 0.044715 * powf(
                                                                            out_layer[index + m + n],3))));
                                    }
                                }
                            }
                    }
                }
                else { //erf based gelu
                    if (elementwise_input) { // Batchnorm and element wise
                        #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                        for (int i=0; i< batch_size; i++)
                            for (int r=0; r< filter_block; r++) {
                                index = blocked_out_height_width*(i*filter_block + r);
                                unsigned long index_filter = 8*r;
                                #pragma omp simd
                                for (int m=0; m< blocked_out_height_width; m=m+8) {
                                    for (int n=0; n < 8; n++) {
                                        out_layer[index+m + n]  = scale[index_filter + n]*(out_layer[index+m + n] -
                                                                  mean[index_filter + n])
                                                                  + offset[index_filter + n]  + elementwise_input[index+m + n];
                                        out_layer[index + m + n] = 0.5 * out_layer[index + m + n] *
                                                                   (1 + erff(out_layer[index + m + n]/1.414213));
                                    }
                                }
                            }
                    }
                    else { // Batchnorm Only
                        #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                        for (int i=0; i< batch_size; i++)
                            for (int r=0; r< filter_block; r++) {
                                index = blocked_out_height_width*(i*filter_block + r);
                                unsigned long index_filter = 8*r;
                                #pragma omp simd
                                for (int m=0; m< blocked_out_height_width; m=m+8) {
                                    for (int n=0; n < 8; n++) {
                                        out_layer[index+m +n]  = scale[index_filter + n]*(out_layer[index+m+n] -
                                                                 mean[index_filter + n])
                                                                 + offset[index_filter + n];
                                        out_layer[index + m + n] = 0.5 * out_layer[index + m + n] *
                                                                   (1 + erff(out_layer[index + m + n]/1.414213));
                                    }
                                }
                            }
                    }
                }
            }
            else if (elementwise_input) { // Batchnorm and element wise
                #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                for (int i=0; i< batch_size; i++)
                    for (int r=0; r< filter_block; r++) {
                        index = blocked_out_height_width*(i*filter_block + r);
                        unsigned long index_filter = 8*r;
                        #pragma omp simd
                        for (int m=0; m< blocked_out_height_width; m=m+8) {
                            for (int n=0; n < 8; n++) {
                                out_layer[index+m + n]  = scale[index_filter + n]*(out_layer[index+m + n] -
                                                          mean[index_filter + n])
                                                          + offset[index_filter + n]  + elementwise_input[index+m + n];
                            }
                        }
                    }
            }
            else if (!elementwise_input) {
                #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                for (int i=0; i< batch_size; i++)
                    for (int r=0; r< filter_block; r++) {
                        index = blocked_out_height_width*(i*filter_block + r);
                        unsigned long index_filter = 8*r;
                        #pragma omp simd
                        for (int m=0; m< 8*out_height*out_width; m=m+8) {
                            for (int n=0; n < 8; n++) {
                                out_layer[index+m +n]  = scale[index_filter + n]*(out_layer[index+m+n] -
                                                         mean[index_filter + n])
                                                         + offset[index_filter + n];
                            }
                        }
                    }
            }
        }
        else {
            if (relu) {
                if (bias && !elementwise_input) { // bias
                    #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                    for (int i=0; i< batch_size; i++)
                        for (int r=0; r< filter_block; r++) {
                            index = blocked_out_height_width*(i*filter_block + r);
                            unsigned long index_filter = 8*r;
                            #pragma omp simd
                            for (int m=0; m< blocked_out_height_width; m=m+8) {
                                for (int n=0; n < 8; n++) {
                                    out_layer[index + m + n] = out_layer[index + m + n] + bias[index_filter + n];
                                    out_layer[index + m + n] = out_layer[index + m + n]>0
                                                               ? out_layer[index + m + n] :
                                                               (alpha==0.0f) ? alpha
                                                               :out_layer[index + m + n] * alpha ;
                                }
                            }
                        }
                }
                else if (bias && elementwise_input) { // bias and element wise
                    #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                    for (int i=0; i< batch_size; i++)
                        for (int r=0; r< filter_block; r++) {
                            index = blocked_out_height_width*(i*filter_block + r);
                            unsigned long index_filter = 8*r;
                            #pragma omp simd
                            for (int m=0; m< blocked_out_height_width; m=m+8) {
                                for (int n=0; n < 8; n++) {
                                    out_layer[index + m + n] = out_layer[index + m + n] + bias[index_filter + n] +
                                                               elementwise_input[index + m + n];
                                    out_layer[index + m + n]=out_layer[index + m + n]>0
                                                             ? out_layer[index + m + n] :
                                                             (alpha==0.0f) ? alpha :
                                                             out_layer[index + m + n] * alpha ;
                                }
                            }
                        }
                }
                else if (!bias && elementwise_input)  { // Elementwise
                    #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                    for (int i=0; i< batch_size; i++)
                        for (int r=0; r< filter_block; r++) {
                            index = blocked_out_height_width*(i*filter_block + r);
                            unsigned long index_filter = 8*r;
                            #pragma omp simd
                            for (int m=0; m< blocked_out_height_width; m++) {
                                out_layer[index + m] = out_layer[index + m ] + elementwise_input[index + m ];
                                out_layer[index + m] = out_layer[index + m]>0
                                                       ? out_layer[index + m] :
                                                       (alpha==0.0f) ? alpha :
                                                       out_layer[index + m]*alpha;
                            }
                        }
                }
            }
            else if (gelu) {

                //gelu=1 is tanh based gelu, else(i.e gelu=2) is
                // erf based
                if (gelu==1) {
                    if (bias && !elementwise_input) { // bias
                        #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                        for (int i=0; i< batch_size; i++)
                            for (int r=0; r< filter_block; r++) {
                                index = blocked_out_height_width*(i*filter_block + r);
                                unsigned long index_filter = 8*r;
                                #pragma omp simd
                                for (int m=0; m< blocked_out_height_width; m=m+8) {
                                    for (int n=0; n < 8; n++) {
                                        out_layer[index + m + n] = out_layer[index + m + n] + bias[index_filter + n];
                                        out_layer[index + m + n] = 0.5 * out_layer[index + m + n] * (1 + tanhf(
                                                                       gelu_const *
                                                                       (out_layer[index + m + n] + 0.044715 * powf(
                                                                            out_layer[index + m + n],3))));
                                    }
                                }
                            }
                    }
                    else if (bias && elementwise_input) { // bias and element wise
                        #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                        for (int i=0; i< batch_size; i++)
                            for (int r=0; r< filter_block; r++) {
                                index = blocked_out_height_width*(i*filter_block + r);
                                unsigned long index_filter = 8*r;
                                #pragma omp simd
                                for (int m=0; m< blocked_out_height_width; m=m+8) {
                                    for (int n=0; n < 8; n++) {
                                        out_layer[index + m + n] = out_layer[index + m + n] + bias[index_filter + n] +
                                                                   elementwise_input[index + m + n];
                                        out_layer[index + m + n] = 0.5 * out_layer[index + m + n] * (1 + tanhf(
                                                                       gelu_const *
                                                                       (out_layer[index + m + n] + 0.044715 * powf(
                                                                            out_layer[index + m + n],3))));
                                    }
                                }
                            }
                    }
                    else if (!bias && elementwise_input)  { // Elementwise
                        #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                        for (int i=0; i< batch_size; i++)
                            for (int r=0; r< filter_block; r++) {
                                index = blocked_out_height_width*(i*filter_block + r);
                                unsigned long index_filter = 8*r;
                                #pragma omp simd
                                for (int m=0; m< blocked_out_height_width; m++) {
                                    out_layer[index + m ] = out_layer[index + m ] + elementwise_input[index + m ];
                                    out_layer[index + m ] = 0.5 * out_layer[index + m ] * (1 + tanhf(gelu_const *
                                                            (out_layer[index + m ] + 0.044715 * powf(
                                                                 out_layer[index + m ],3))));
                                }
                            }
                    }
                }
                else { //erf based gelu
                    if (bias && !elementwise_input) { // bias
                        #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                        for (int i=0; i< batch_size; i++)
                            for (int r=0; r< filter_block; r++) {
                                index = blocked_out_height_width*(i*filter_block + r);
                                unsigned long index_filter = 8*r;
                                #pragma omp simd
                                for (int m=0; m< blocked_out_height_width; m=m+8) {
                                    for (int n=0; n < 8; n++) {
                                        out_layer[index + m + n] = out_layer[index + m + n] + bias[index_filter + n];
                                        out_layer[index + m + n] = 0.5 * out_layer[index + m + n] *
                                                                   (1 + erff(out_layer[index + m + n]/1.414213));
                                    }
                                }
                            }
                    }
                    else if (bias && elementwise_input) { // bias and element wise
                        #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                        for (int i=0; i< batch_size; i++)
                            for (int r=0; r< filter_block; r++) {
                                index = blocked_out_height_width*(i*filter_block + r);
                                unsigned long index_filter = 8*r;
                                #pragma omp simd
                                for (int m=0; m< blocked_out_height_width; m=m+8) {
                                    for (int n=0; n < 8; n++) {
                                        out_layer[index + m + n] = out_layer[index + m + n] + bias[index_filter + n] +
                                                                   elementwise_input[index + m + n];
                                        out_layer[index + m + n] = 0.5 * out_layer[index + m + n] *
                                                                   (1 + erff(out_layer[index + m + n]/1.414213));
                                    }
                                }
                            }
                    }
                    else if (!bias && elementwise_input)  { // Elementwise
                        #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                        for (int i=0; i< batch_size; i++)
                            for (int r=0; r< filter_block; r++) {
                                index = blocked_out_height_width*(i*filter_block + r);
                                unsigned long index_filter = 8*r;
                                #pragma omp simd
                                for (int m=0; m< blocked_out_height_width; m++) {
                                    out_layer[index + m ] = out_layer[index + m ] + elementwise_input[index + m ];
                                    out_layer[index + m ] = 0.5 * out_layer[index + m ] *
                                                            (1 + erff(out_layer[index + m ]/1.414213));
                                }
                            }
                    }
                }
            }
            else if (bias && !elementwise_input) { // bias
                #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                for (int i=0; i< batch_size; i++)
                    for (int r=0; r< filter_block; r++) {
                        index = blocked_out_height_width*(i*filter_block + r);
                        unsigned long index_filter = 8*r;
                        #pragma omp simd
                        for (int m=0; m< blocked_out_height_width; m=m+8) {
                            for (int n=0; n < 8; n++) {
                                out_layer[index + m + n] = out_layer[index + m + n] + bias[index_filter + n];
                            }
                        }
                    }
            }
            else if (bias && elementwise_input) { // bias and element wise
                #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                for (int i=0; i< batch_size; i++)
                    for (int r=0; r< filter_block; r++) {
                        index = blocked_out_height_width*(i*filter_block + r);
                        unsigned long index_filter = 8*r;
                        #pragma omp simd
                        for (int m=0; m< 8*out_height*out_width; m=m+8) {
                            for (int n=0; n < 8; n++) {
                                out_layer[index + m + n] = out_layer[index + m + n] + bias[index_filter + n] +
                                                           elementwise_input[index + m + n];
                            }
                        }
                    }
            }
            else if (!bias && elementwise_input)  { // Elementwise
                #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                for (int i=0; i< batch_size; i++)
                    for (int r=0; r< filter_block; r++) {
                        index = blocked_out_height_width*(i*filter_block + r);
                        unsigned long index_filter = 8*r;
                        #pragma omp simd
                        for (int m=0; m< blocked_out_height_width; m++) {
                            out_layer[index + m] = out_layer[index + m] + elementwise_input[index + m];
                        }
                    }
            }
        }
        bool batchNorm_enable = 0;
        bool elementWise_enable = 0;

        if (scale) {
            batchNorm_enable = 1;
        }
        if (elementwise_input) {
            elementWise_enable = 1;
        }

        float elapsed;
#ifdef _WIN32
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> difference = end - start;
        elapsed = difference.count();
#else
        gettimeofday(&end, 0);
        elapsed = timedifference_msec(start, end);
#endif
        if (!alpha)
            zendnnVerbose(ZENDNN_PROFLOG, "zenPostOps, no_of_images=", batch_size,
                          " height=", out_height, " width=", out_width,
                          " no_of_filter=", no_of_filter, " relu_enable=", relu, " gelu=", gelu,
                          " batchNorm_enable=", batchNorm_enable, " elementWise_enable=",
                          elementWise_enable, " Time=", elapsed, "ms");
        else
            zendnnVerbose(ZENDNN_PROFLOG, "zenPostOps, no_of_images=", batch_size,
                          " height=", out_height, " width=", out_width,
                          " no_of_filter=", no_of_filter, " leakyrelu_enable=", relu,
                          " with alpha= ", alpha," gelu=", gelu, " batchNorm_enable=", batchNorm_enable,
                          " elementWise_enable=", elementWise_enable, " Time=", elapsed, "ms");
    }
}
