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
#include <cblas.h>
#include <time.h>
#include "zendnn_logging.hpp"
#include "zendnn_helper.hpp"

using namespace zendnn;

#define ALIGNED_OFFSET          64


// This implementation uses NCHW (C/8) and parallelizes convolution by accumulation at sub channel level
void zenConvolution2D_Latency_blocked_layout(
    //const unsigned char* in_layer,
    zendnnEnv zenEnvObj,
    const float *in_layer,  // float or char input?
    const int no_of_images,
    const int channels,
    const int height,
    const int width,
    const float *filter,
    const int no_of_filter,
    //const int channels,           //no. of channels is same as no. of channels filters
    const int kernel_h,
    const int kernel_w,
    const float pad_h,
    const float pad_w,
    const int stride_h,
    const int stride_w,
    const float *bias,
    float *out_layer,       //o/p to this function
    //unsigned char *out_layer,     // float or char?
    //const int out_no_of_images,
    //const int out_channels,       // same as no. of filters
    const int out_height,           //o/p to this function
    const int out_width             //o/p to this function
) {
    zendnnVerbose(ZENDNN_ALGOLOG, "zenConvolution2D_Latency_blocked_layout [zendnn convolution blocked]");
    unsigned int thread_qty = zenEnvObj.omp_num_threads;
#ifdef _WIN32
    auto start = std::chrono::high_resolution_clock::now();
#else
    struct timeval start, end;
    gettimeofday(&start, 0);
#endif

    int channel_group =  8;    // Modified hardcoding of 8 based on thread quantity in future
    int remainder = channels % channel_group;
    int out_ch_per_group = (channels-remainder)/(channel_group);
    int w_offset = kernel_h * kernel_w * no_of_filter;
    int o_h_w    = out_height*out_width;

    unsigned long data_col_size = ((kernel_h*kernel_w*channels)*(out_height*out_width)*sizeof(float)*no_of_images);
    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size : (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    float *data_col = (float *)zendnn_aligned_alloc(ALIGNED_OFFSET, data_col_size);

    float *out_col;
    // out_col is an intermedidate output
    // out_col requires excess memory allocation depending on channel groups
    // The outputs of N channel groups are accumulated after sgemm calls
    out_col = (float *) malloc(o_h_w * no_of_filter * sizeof(float)*(channel_group+1));
    for (int i =0 ; i < o_h_w * no_of_filter *(channel_group+1); i++) {
        out_col[i] = 0.0;
    }
    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG, "zenConvolution2D_Latency_blocked_layout Memory Error while allocating patch matrix");
        return;
    }

    im2col_parNCHW(in_layer, channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, data_col);

    for (int itr=0; itr <= channel_group; itr++) {
        if (itr < channel_group) {
            int weight_offset = itr * out_ch_per_group *  kernel_h * kernel_w * no_of_filter;
            int data_offset = itr * out_ch_per_group *  out_height*out_width * kernel_h*kernel_w;
            if (out_ch_per_group)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            no_of_filter,  o_h_w,kernel_h*kernel_w*out_ch_per_group,  1.0F, filter + weight_offset,
                            kernel_h*kernel_w*out_ch_per_group, data_col + data_offset,o_h_w, 0.0F, out_col + itr*o_h_w* no_of_filter, o_h_w);

        }

        else {
            int weight_offset = channel_group * out_ch_per_group *  w_offset;
            int data_offset = channel_group * out_ch_per_group *  o_h_w * kernel_h*kernel_w;
            if (remainder)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            no_of_filter,  o_h_w,kernel_h*kernel_w*remainder,  1.0F, filter + weight_offset,
                            kernel_h*kernel_w*remainder, data_col + data_offset,o_h_w, 0.0F, out_col+channel_group*o_h_w* no_of_filter, o_h_w);
        }
    }

// initialize output array to zero
    #pragma omp parallel for
    for (int j = 0 ; j < o_h_w * no_of_filter ; j++) {
        out_layer[j] = 0.0;
    }
// Accumulates the output of channels to generate out_layer
    #pragma omp parallel for
    for (int j = 0 ; j < o_h_w * no_of_filter ; j++) {
        for (int ch = 0; ch <= channel_group; ch++) {
            out_layer[j]+= out_col[j+ (ch *o_h_w * no_of_filter)];
        }
    }
    // free the intermediate result
    free(out_col);
    out_col =NULL;



#if BIAS_ENABLED
    for (int r=0; r<no_of_filter; r++) {
        for (int m=0; m<out_height*out_width; m++) {
            out_layer[(r*(out_height*out_width)) + m]+= bias[r];
        }
    }
#endif
}





void zenConvolution2D_Filterwise_Latency(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int no_of_images,
    const int channels,
    const int height,
    const int width,
    const float *filter,
    const int no_of_filter,
    const int kernel_h,
    const int kernel_w,
    const float pad_t,
    const float pad_l,
    const float pad_b,
    const float pad_r,
    const int stride_h,
    const int stride_w,
    const float *bias,
    float *out_layer,
    const int out_height,
    const int out_width,
    const bool relu
) {
    zendnnVerbose(ZENDNN_ALGOLOG, "zenConvolution2D_Filterwise_Latency [zendnn convolution Filter parallelization]");

    unsigned int thread_qty = zenEnvObj.omp_num_threads;
    unsigned long data_col_size = ((kernel_h*kernel_w*channels)*(out_height*out_width)*sizeof(float)*no_of_images);
    unsigned long filter_col_size = ((kernel_h*kernel_w*channels*no_of_filter)*sizeof(float));
    unsigned long o_layer_size = ((no_of_filter)*(out_height*out_width)*sizeof(float)*no_of_images);

    data_col_size = (data_col_size%ALIGNED_OFFSET == 0) ?  data_col_size : (data_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    o_layer_size = (o_layer_size%ALIGNED_OFFSET == 0) ?  o_layer_size : (o_layer_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);
    filter_col_size = (filter_col_size%ALIGNED_OFFSET == 0) ?  filter_col_size : (filter_col_size/ALIGNED_OFFSET)*ALIGNED_OFFSET + (ALIGNED_OFFSET);

    // Allocate memory for reordering input , output and filters

    float *data_col = (float *)zendnn_aligned_alloc(ALIGNED_OFFSET, data_col_size);
    float *out_col = (float *)zendnn_aligned_alloc(ALIGNED_OFFSET, o_layer_size);
    float *filter_col = (float *)zendnn_aligned_alloc(ALIGNED_OFFSET, filter_col_size);

    if (data_col == NULL) {
        zendnnError(ZENDNN_ALGOLOG, "zenConvolution2D_Filterwise_Latency Memory Error while allocating patch matrix");
        return;
    }


    //Divide filters into channel groups based on the number of threads
    int channel_group =  thread_qty;

    if (no_of_filter < thread_qty) {
        channel_group = no_of_filter;
    }
    int remainder = no_of_filter % (channel_group);
    int out_ch_per_group = (no_of_filter-remainder)/(channel_group);
    int split_index,split_offset,index;


    // Reorder Filters to HWCN ( N/Split )
    // This routine should be implemented outside ZenDNN Library
    #pragma omp parallel for
    for (int j=0; j<kernel_h*kernel_w*channels; j++) {
        for (int i=0; i<no_of_filter; i++) {
            if (i < out_ch_per_group*channel_group) {
                int block_index = ((int)floor(i/out_ch_per_group));
                int sub_block_index =  j*out_ch_per_group;
                int filter_index =   i%out_ch_per_group;
                int index = kernel_h*kernel_w*channels*out_ch_per_group*block_index + sub_block_index + filter_index;
                filter_col[index]=filter[j*no_of_filter+i];
            }
            else {
                int index = kernel_h*kernel_w*channels*out_ch_per_group*channel_group + j*remainder + (i%no_of_filter)%remainder ;
                filter_col[index]=filter[j*no_of_filter+i];

            }
        }
    }

    // GEMM calls to implement parallel filter convolution
    for (int i=0; i<no_of_images; i++) {
        unsigned long bufferOffset = ((kernel_h*kernel_w*channels)*(out_height*out_width) * i);
        unsigned long inputOffset = channels*height*width*i;
        im2rowNHWC_par(in_layer + inputOffset, channels, height, width, kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, data_col + bufferOffset);
        int w_offset = kernel_h * kernel_w * channels;
        int o_h_w    = out_height*out_width;
        int weight_offset = 0,out_offset=0;
        int offset = i*no_of_filter*o_h_w;
        #pragma omp parallel for
        for (int itr=0; itr <= channel_group; itr++) {
            if (itr < channel_group) {
                weight_offset = ((itr * out_ch_per_group)) *  w_offset;
                out_offset = (itr*out_ch_per_group)* o_h_w + offset;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            o_h_w, out_ch_per_group, w_offset, 1.0F,
                            data_col + bufferOffset, w_offset, filter_col + weight_offset, out_ch_per_group,
                            0.0F, out_col+ out_offset, out_ch_per_group);
            }
            else {
                if (remainder) {
                    int weight_offset = out_ch_per_group * (channel_group) *  w_offset;
                    int out_offset = (itr * out_ch_per_group)* o_h_w + offset;
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                o_h_w, remainder, w_offset, 1.0F,
                                data_col + bufferOffset, w_offset, filter_col + weight_offset, remainder,
                                0.0F, out_col+out_offset, remainder);
                }
            }
        }

        // Reorder Output to NHWC from ( N/Split ) HWC
        #pragma omp parallel for
        for (int j=0; j<o_h_w; j++) {
            for (int l=0; l<no_of_filter; l++) {
                if (l < out_ch_per_group*channel_group) {
                    int block_index = ((int)floor(l/out_ch_per_group));
                    int sub_block_index =  j*out_ch_per_group;
                    int filter_index =   l%out_ch_per_group;
                    int index = o_h_w*out_ch_per_group*block_index + sub_block_index + filter_index;
                    out_layer[offset + j*no_of_filter + l]=out_col[offset + index];
                }
                else {
                    int index = o_h_w*out_ch_per_group*channel_group + j*remainder + (l%no_of_filter)%remainder ;
                    out_layer[offset + j*no_of_filter + l]=out_col[offset + index];

                }
            }
        }


        if (bias && !relu) {
            #pragma omp parallel for num_threads(thread_qty)
            for (int m=0; m<out_height*out_width; m++)
                for (int r=0; r<no_of_filter; r++) {
                    out_layer[i*out_height*out_width*no_of_filter + (m*(no_of_filter)) + r]+= bias[r];
                }
        }

        if (bias && relu) {
            #pragma omp parallel for num_threads(thread_qty)
            for (int m=0; m<out_height*out_width; m++)
                for (int r=0; r<no_of_filter; r++) {
                    out_layer[i*out_height*out_width*no_of_filter + (m*(no_of_filter)) + r]+=bias[r];
                    if (out_layer[i*out_height*out_width*no_of_filter + (m*(no_of_filter)) + r] < 0) {
                        out_layer[i*out_height*out_width*no_of_filter + (m*(no_of_filter)) + r] = 0;
                    }
                }
        }

    }
    free(data_col);
    free(filter_col);
    free(out_col);

}

