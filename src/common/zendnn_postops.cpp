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

#define COMPUTE_BIAS_VEC16(out_layer, scale, bias, alpha, elementwise_input, offset, c) \
    { \
        for(int i=0;i<16;++i) \
        out_layer[offset+c+i] = out_layer[offset+c+i] + alpha*bias[c+i]; \
    }

#define COMPUTE_SCALE_BIAS_VEC16(out_layer, scale, bias, alpha, elementwise_input, offset, c) \
    { \
        for(int i=0;i<16;++i) \
        out_layer[offset+c+i] = out_layer[offset+c+i]*scale[c+i] + alpha*bias[c+i]; \
    }

#define COMPUTE_ADD_VEC16(out_layer, scale, bias, alpha, elementwise_input, offset, c) \
    { \
        for(int i=0;i<16;++i) \
        out_layer[offset+c+i] = out_layer[offset+c+i] + elementwise_input[offset + c + i]; \
    }

#define COMPUTE_BIAS_ADD_VEC16(out_layer, scale, bias, alpha, elementwise_input, offset, c) \
    { \
        for(int i=0;i<16;++i) \
        out_layer[offset+c+i] = out_layer[offset+c+i] + alpha*bias[c+i] + elementwise_input[offset + c + i]; \
}

#define COMPUTE_SCALE_BIAS_ADD_VEC16(out_layer, scale, bias, alpha, elementwise_input, offset, c) \
    { \
        for(int i=0;i<16;++i) \
        out_layer[offset+c+i] = out_layer[offset+c+i]*scale[c+i] + alpha*bias[c+i] + elementwise_input[offset + c + i]; \
    }

#define COMPUTE_GELU_TANH_VEC16() \
    { \
        aocl_gelu_tanh_f32(16, out_layer+offset+c, 1); \
    }

#define COMPUTE_GELU_ERF_VEC16() \
    { \
        aocl_gelu_erf_f32(16, out_layer+offset+c, 1); \
    }

#define COMPUTE_NONE_VEC16(out_layer, scale, bias, alpha, elementwise_input, offset, c) \
    { \
    }

#define COMPUTE_BIAS_VEC1(out_layer, scale, bias, alpha, elementwise_input, offset, c) \
    { \
        out_layer[offset+c] = out_layer[offset+c] + alpha*bias[c]; \
    }

#define COMPUTE_SCALE_BIAS_VEC1(out_layer, scale, bias, alpha, elementwise_input, offset, c) \
    { \
        out_layer[offset+c] = out_layer[offset+c]*scale[c] + alpha*bias[c]; \
    }

#define COMPUTE_ADD_VEC1(out_layer, scale, bias, alpha, elementwise_input, offset, c) \
    { \
        out_layer[offset+c] = out_layer[offset+c] + elementwise_input[offset + c]; \
    }

#define COMPUTE_BIAS_ADD_VEC1(out_layer, scale, bias, alpha, elementwise_input, offset, c) \
    { \
        out_layer[offset+c] = out_layer[offset+c] + alpha*bias[c] + elementwise_input[offset + c]; \
    }

#define COMPUTE_SCALE_BIAS_ADD_VEC1(out_layer, scale, bias, alpha, elementwise_input, offset, c) \
    { \
        out_layer[offset+c] = out_layer[offset+c]*scale[c] + alpha*bias[c] + elementwise_input[offset + c]; \
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

#define COMPUTE_NONE_VEC1(out_layer, scale, bias, alpha, elementwise_input, offset, c) \
    { \
    }

#define COMPUTE_GELU_VEC16(out_layer, scale, bias, alpha, elementwise_input, biasOffset, i, no_of_filter, compute_postOp, compute_gelu_type) \
    { \
        unsigned int offset = biasOffset + i; \
        int c = 0; \
        for (c = 0; (c+16) <= no_of_filter; c+=16) { \
            compute_postOp##_VEC16(out_layer, scale, bias, alpha, elementwise_input, offset, c); \
                                    \
            compute_gelu_type##_VEC16();  \
                                    \
        } \
        for( ;c<no_of_filter; c++) \
        { \
            compute_postOp##_VEC1(out_layer, scale, bias, alpha, elementwise_input, offset, c); \
                                    \
            compute_gelu_type##_VEC1();  \
                                    \
        } \
    }

#define COMPUTE_GELU_VEC1(out_layer, scale, bias, alpha, elementwise_input, biasOffset, i, no_of_filter, compute_postOp, compute_gelu_type) \
    { \
        unsigned int offset = biasOffset + i; \
        for (int c = 0; c < no_of_filter; c++) { \
            compute_postOp##_VEC1(out_layer, scale, bias, alpha, elementwise_input, offset, c); \
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
    const float alpha,
    const float *offset,
    const float  *mean,
    const int batch_size,
    const float leaky_alpha
) {
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
                                                          alpha*bias[c];
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c ]>0
                                                          ?out_layer[ biasOffset + i + c ]
                                                          :(leaky_alpha==0.0f)?leaky_alpha
                                                          :out_layer[ biasOffset + i + c ]*leaky_alpha;
                    }
            }
            else if (bias != NULL && scale == NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] +
                                                          alpha*bias[c];
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c ]>0
                                                          ?out_layer[ biasOffset + i + c ]
                                                          :(leaky_alpha==0.0f)?leaky_alpha
                                                          :out_layer[ biasOffset + i + c ]*leaky_alpha;
                    }
            }
            else if (bias == NULL && scale == NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c ]>0
                                                          ?out_layer[ biasOffset + i + c ]
                                                          :(leaky_alpha==0.0f)?leaky_alpha
                                                          :out_layer[ biasOffset + i + c ]*leaky_alpha;
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
                        COMPUTE_GELU(out_layer, scale, bias, alpha, elementwise_input, biasOffset, i,
                                     no_of_filter, COMPUTE_SCALE_BIAS, COMPUTE_GELU_TANH);
                    }
                }
                else if (bias != NULL && scale == NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters) {
                        COMPUTE_GELU(out_layer, scale, bias, alpha, elementwise_input, biasOffset, i,
                                     no_of_filter, COMPUTE_BIAS, COMPUTE_GELU_TANH);
                    }
                }
                else if (bias == NULL && scale == NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters) {
                        COMPUTE_GELU(out_layer, scale, bias, alpha, elementwise_input, biasOffset, i,
                                     no_of_filter, COMPUTE_NONE, COMPUTE_GELU_TANH);
                    }
                }
            }
            else { //erf based gelu
                if (bias != NULL && scale != NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        COMPUTE_GELU(out_layer, scale, bias, alpha, elementwise_input, biasOffset, i,
                                     no_of_filter, COMPUTE_SCALE_BIAS, COMPUTE_GELU_ERF);
                }
                else if (bias != NULL && scale == NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        COMPUTE_GELU(out_layer, scale, bias, alpha, elementwise_input, biasOffset, i,
                                     no_of_filter, COMPUTE_BIAS, COMPUTE_GELU_ERF);
                }
                else if (bias == NULL && scale == NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        COMPUTE_GELU(out_layer, scale, bias, alpha, elementwise_input, biasOffset, i,
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
                                                          alpha * bias[c];
                    }
            }
            else if (bias != NULL &&  scale == NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] + alpha *
                                                          bias[c];
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
                                                          alpha * bias[c] + elementwise_input[biasOffset + i + c];
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c ]>0
                                                          ?out_layer[ biasOffset + i + c ]
                                                          :(leaky_alpha==0.0f)?leaky_alpha
                                                          :out_layer[ biasOffset + i + c ]*leaky_alpha;

                    }
            }
            else if (bias != NULL && scale == NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] + alpha *
                                                          bias[c] +
                                                          elementwise_input[biasOffset + i + c];
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c ]>0
                                                          ?out_layer[ biasOffset + i + c ]
                                                          :(leaky_alpha==0.0f)?leaky_alpha
                                                          :out_layer[ biasOffset + i + c ]*leaky_alpha;
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
                                                          :(leaky_alpha==0.0f)?leaky_alpha
                                                          :out_layer[ biasOffset + i + c ]*leaky_alpha;
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
                            COMPUTE_GELU(out_layer, scale, bias, alpha, elementwise_input, biasOffset, i,
                                         no_of_filter, COMPUTE_SCALE_BIAS_ADD, COMPUTE_GELU_TANH);
                        }
                }
                else if (bias != NULL && scale == NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        #pragma omp simd
                        for (int c = 0; c < no_of_filter; c++) {
                            COMPUTE_GELU(out_layer, scale, bias, alpha, elementwise_input, biasOffset, i,
                                         no_of_filter, COMPUTE_BIAS_ADD, COMPUTE_GELU_TANH);
                        }
                }
                else if (bias == NULL && scale == NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        #pragma omp simd
                        for (int c = 0; c < no_of_filter; c++) {
                            COMPUTE_GELU(out_layer, scale, bias, alpha, elementwise_input, biasOffset, i,
                                         no_of_filter, COMPUTE_ADD, COMPUTE_GELU_TANH);
                        }
                }
            }
            else { //erf based gelu
                if (bias != NULL && scale != NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        COMPUTE_GELU(out_layer, scale, bias, alpha, elementwise_input, biasOffset, i,
                                     no_of_filter, COMPUTE_SCALE_BIAS_ADD, COMPUTE_GELU_ERF);
                }
                else if (bias != NULL && scale == NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        COMPUTE_GELU(out_layer, scale, bias, alpha, elementwise_input, biasOffset, i,
                                     no_of_filter, COMPUTE_BIAS_ADD, COMPUTE_GELU_ERF);
                }
                else if (bias == NULL && scale == NULL) {
                    #pragma omp parallel for num_threads(no_of_threads)
                    for (i = 0; i < total_size; i += total_filters)
                        COMPUTE_GELU(out_layer, scale, bias, alpha, elementwise_input, biasOffset, i,
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
                                                          alpha*bias[c] + elementwise_input[biasOffset + i + c];
                    }
            }
            else if (bias != NULL && scale == NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] + alpha*bias[c]
                                                          +
                                                          elementwise_input[biasOffset + i + c];
                    }
            }
        }
    }
}
