/*******************************************************************************
* Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <zendnn_private.hpp>
#include <omp.h>
#include <sys/sysinfo.h>
#include <cblas.h>
#include <time.h>
#include <sys/time.h>
#include "zendnn_logging.hpp"

#define ALIGNED_OFFSET          64

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

float activation_gelu(float input) {
    float out = 0.5 * input * (1 + tanhf(sqrtf(2/M_PI) * (input + 0.044715 * powf(
            input,3))));
    return out;
}

float apply_activation(float input, bool relu, bool gelu) {
    float out = 0;
    if (relu) {
        out = input>0?input:0;
    }
    if (gelu) {
        out = activation_gelu(input);
    }
    return out;
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
    const bool gelu,
    const float *scale,
    const int no_of_threads,
    const float *offset,
    const float  *mean,
    const int batch_size
) {

    bool activation = false;
    if (relu || gelu) {
        activation = true;
    }

    if (!zenEnvObj.zenBlockedFormat) {  // GEMM Path

        unsigned long i;
        unsigned long total_size = (unsigned long)out_height*out_width*total_filters;
        if (!elementwise_input) {
            if (bias != NULL && activation == true && scale != NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] * scale[c] +
                                                          bias[c];
                        out_layer[ biasOffset + i + c ] = apply_activation(out_layer[ biasOffset + i +
                                                          c], relu, gelu);
                    }
            }
            else if (bias != NULL && activation == false  && scale != NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] * scale[c] +
                                                          bias[c];
                    }
            }
            else if (bias != NULL && activation == true && scale == NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] + bias[c];
                        out_layer[ biasOffset + i + c ] = apply_activation(out_layer[ biasOffset + i +
                                                          c], relu, gelu);
                    }
            }
            else if (bias != NULL && activation == false &&  scale == NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] + bias[c];
                    }
            }
            else if (bias == NULL && activation == true && scale == NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = apply_activation(out_layer[ biasOffset + i +
                                                          c], relu, gelu);
                    }
            }
        }
        else {
            if (bias != NULL && activation == true && scale != NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] * scale[c] +
                                                          bias[c] + elementwise_input[biasOffset + i + c];
                        out_layer[ biasOffset + i + c ] = apply_activation(out_layer[ biasOffset + i +
                                                          c], relu, gelu);
                    }
            }
            else if (bias != NULL && activation == false && scale != NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] * scale[c] +
                                                          bias[c] + elementwise_input[biasOffset + i + c];
                    }
            }
            else if (bias != NULL && activation == true && scale == NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] + bias[c] +
                                                          elementwise_input[biasOffset + i + c];
                        out_layer[ biasOffset + i + c ] = apply_activation(out_layer[ biasOffset + i +
                                                          c], relu, gelu);
                    }
            }
            else if (bias != NULL && activation == false && scale == NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] + bias[c] +
                                                          elementwise_input[biasOffset + i + c];
                    }
            }
            else if (bias == NULL && activation == true && scale == NULL) {
                #pragma omp parallel for num_threads(no_of_threads)
                for (i = 0; i < total_size; i += total_filters)
                    #pragma omp simd
                    for (int c = 0; c < no_of_filter; c++) {
                        out_layer[ biasOffset + i + c ] = out_layer[ biasOffset + i + c] +
                                                          elementwise_input[biasOffset + i + c];
                        out_layer[ biasOffset + i + c ] = apply_activation(out_layer[ biasOffset + i +
                                                          c], relu, gelu);
                    }
            }
        }
    }
    else  {
        struct timeval start, end;
        gettimeofday(&start, 0);

        // This section of the code enables Batchorm , Elementwise & Relu support for Blocked Format
        int filter_block = no_of_filter/8;          // Assumes Filters are multiple of 8
        // If Filters are not multiple of 8 , source call should ensure padding
        unsigned long index = 0;
        unsigned long blocked_out_height_width = 8*out_height*out_width;
        if (scale) {
            if (activation && elementwise_input) { // Batchnorm and element wise

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
                                out_layer[ index+m +n ] = apply_activation(out_layer[index+m +n], relu, gelu);
                            }
                        }
                    }
            }
            else if (activation &&  !elementwise_input) { // Batchnorm Only
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
                                out_layer[ index+m +n ] = apply_activation(out_layer[index+m +n], relu, gelu);
                            }
                        }
                    }
            }
            else if (!activation &&  elementwise_input) { // Batchnorm and element wise
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
            else if (!activation && !elementwise_input) {
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
            if (activation && bias && !elementwise_input) { // bias
                #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                for (int i=0; i< batch_size; i++)
                    for (int r=0; r< filter_block; r++) {
                        index = blocked_out_height_width*(i*filter_block + r);
                        unsigned long index_filter = 8*r;
                        #pragma omp simd
                        for (int m=0; m< blocked_out_height_width; m=m+8) {
                            for (int n=0; n < 8; n++) {
                                out_layer[index+m+n] = out_layer[index+m+n] + bias[index_filter + n];
                                out_layer[ index+m +n ] = apply_activation(out_layer[index+m +n], relu, gelu);
                            }
                        }
                    }
            }
            else if (!activation && bias && !elementwise_input) { // bias
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
            else if (activation && bias && elementwise_input) { // bias and element wise
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
                                out_layer[ index+m +n ] = apply_activation(out_layer[index+m +n], relu, gelu);
                            }
                        }
                    }
            }
            else if (!activation && bias && elementwise_input) { // bias and element wise
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
            else if (activation && !bias && elementwise_input)  { // Elementwise
                #pragma omp parallel for num_threads(no_of_threads) collapse(2)
                for (int i=0; i< batch_size; i++)
                    for (int r=0; r< filter_block; r++) {
                        index = blocked_out_height_width*(i*filter_block + r);
                        unsigned long index_filter = 8*r;
                        #pragma omp simd
                        for (int m=0; m< blocked_out_height_width; m++) {
                            out_layer[index + m ] = out_layer[index + m ] + elementwise_input[index + m ];
                            out_layer[ index+m ] = apply_activation(out_layer[index+m], relu, gelu);
                        }
                    }
            }
            else if (!activation && !bias && elementwise_input)  { // Elementwise
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

        gettimeofday(&end, 0);
        float elapsed;
        elapsed = timedifference_msec(start, end);
        zendnnInfo(ZENDNN_PROFLOG, "zenPostOps, no_of_images=", batch_size,
                   " height=", out_height, " width=", out_width,
                   " no_of_filter=", no_of_filter, " relu_enable=", relu, " gelu_enable=", gelu,
                   " batchNorm_enable=", batchNorm_enable, " elementWise_enable=",
                   elementWise_enable, " Time=", elapsed, "ms");


    }
}
