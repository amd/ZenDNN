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

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string>
#include "utils.hpp"
#include "zendnn_helper.hpp"
#include "zendnn_utils.hpp"

#ifndef ZENDNN_PRIVATE_HPP
#define ZENDNN_PRIVATE_HPP

//structure to make key
struct Key_conv {
    unsigned int m;
    unsigned int k;
    unsigned int n;
    unsigned int lda;
    unsigned int ldb;
    unsigned int ldc;
    unsigned int thread_count;
    const void *weights;

    bool operator==(const Key_conv &other) const {
        return (thread_count == other.thread_count
                && m == other.m
                && k == other.k
                && n == other.n
                && lda == other.lda
                && ldb == other.ldb
                && ldc == other.ldc
                && weights == other.weights
               );
    }
};

//structure to make key
struct Key_matmul {
    bool transpose_input;
    bool transpose_weights;
    unsigned int m;
    unsigned int k;
    unsigned int n;
    unsigned int lda;
    unsigned int ldb;
    unsigned int ldc;
    unsigned int thread_count;
    const void *weights;

    bool operator==(const Key_matmul &other) const {
        return (thread_count == other.thread_count
                && m == other.m
                && k == other.k
                && n == other.n
                && lda == other.lda
                && ldb == other.ldb
                && ldc == other.ldc
                && weights == other.weights
                && transpose_input == other.transpose_input
                && transpose_weights == other.transpose_weights
               );
    }
};

namespace std {

template <>
struct hash<Key_matmul> {
    std::size_t operator()(const Key_matmul &k) const {
        std::size_t seed = 0;
        seed = zendnn::impl::hash_combine(seed, (k.transpose_input));
        seed = zendnn::impl::hash_combine(seed, (k.transpose_weights));
        seed = zendnn::impl::hash_combine(seed, (k.m));
        seed = zendnn::impl::hash_combine(seed, (k.k));
        seed = zendnn::impl::hash_combine(seed, (k.n));
        seed = zendnn::impl::hash_combine(seed, (k.lda));
        seed = zendnn::impl::hash_combine(seed, (k.ldb));
        seed = zendnn::impl::hash_combine(seed, (k.ldc));
        seed = zendnn::impl::hash_combine(seed, (k.thread_count));
        seed = zendnn::impl::hash_combine(seed, (k.weights));
        return seed;
    }
};
}

//Updates the cpu information(BRAND String) in the given array.
//Makes use of inline assembly
inline int getCpuID_brandString(int *a) {

    __asm__ __volatile__("xor %eax , %eax\n\t");
    __asm__ __volatile__("xor %ebx , %ebx\n\t");
    __asm__ __volatile__("xor %ecx , %ecx\n\t");
    __asm__ __volatile__("xor %edx , %edx\n\t");

    __asm__ __volatile__("mov $0x80000002 , %eax\n\t");

    __asm__ __volatile__("cpuid\n\t");
    __asm__ __volatile__("mov %%eax, %0\n\t":"=r"(a[0]));
    __asm__ __volatile__("mov %%ebx, %0\n\t":"=r"(a[1]));
    __asm__ __volatile__("mov %%ecx, %0\n\t":"=r"(a[2]));
    __asm__ __volatile__("mov %%edx, %0\n\t":"=r"(a[3]));

    __asm__ __volatile__("mov $0x80000003 , %eax\n\t");
    __asm__ __volatile__("cpuid\n\t");
    __asm__ __volatile__("mov %%eax, %0\n\t":"=r"(a[4]));
    __asm__ __volatile__("mov %%ebx, %0\n\t":"=r"(a[5]));
    __asm__ __volatile__("mov %%ecx, %0\n\t":"=r"(a[6]));
    __asm__ __volatile__("mov %%edx, %0\n\t":"=r"(a[7]));

    __asm__ __volatile__("mov $0x80000004 , %eax\n\t");
    __asm__ __volatile__("cpuid\n\t");
    __asm__ __volatile__("mov %%eax, %0\n\t":"=r"(a[8]));
    __asm__ __volatile__("mov %%ebx, %0\n\t":"=r"(a[9]));
    __asm__ __volatile__("mov %%ecx, %0\n\t":"=r"(a[10]));
    __asm__ __volatile__("mov %%edx, %0\n\t":"=r"(a[11]));
    return 0;
}

namespace std {
template <>
struct hash<Key_conv> {
    std::size_t operator()(const Key_conv &k) const {
        std::size_t seed = 0;
        seed = zendnn::impl::hash_combine(seed, (k.m));
        seed = zendnn::impl::hash_combine(seed, (k.k));
        seed = zendnn::impl::hash_combine(seed, (k.n));
        seed = zendnn::impl::hash_combine(seed, (k.lda));
        seed = zendnn::impl::hash_combine(seed, (k.ldb));
        seed = zendnn::impl::hash_combine(seed, (k.ldc));
        seed = zendnn::impl::hash_combine(seed, (k.thread_count));
        seed = zendnn::impl::hash_combine(seed, (k.weights));
        return seed;
    }
};
}

extern "C"
{
    float timedifference_msec(struct timeval t0, struct timeval t1);

    bool padding_zone(int top_y, int top_x, int width_orig, int height_orig,
                      int padding_w, int padding_h);

    void maxPoolingRef(
        const float *input,
        const int number_of_images,
        const int number_of_channel,
        const int height,
        const int width,
        const int kernel_height,
        const int kernel_width,
        const int stride_height,
        const int stride_width,
        const int padding_height_top,
        const int padding_height_bottom,
        const int padding_width_left,
        const int padding_width_right,
        float *output,
        const bool data_format // 1 for NCHW and 0 for NHWC
    );

    void avgPoolingRef(
        const float *input,
        const int number_of_images,
        const int number_of_channel,
        const int height,
        const int width,
        const int kernel_height,
        const int kernel_width,
        const int stride_width,
        const int stride_height,
        const int padding_height_top,
        const int padding_height_bottom,
        const int padding_width_left,
        const int padding_width_right,
        float *output,
        const int data_format
    );

    void compute_padding(const int image_h, const int image_w,
                         const int filter_h, const int filter_w,
                         const int stride_h, const int stride_w,
                         const char *padding,
                         int *pad_t,int *pad_l,int *pad_b, int *pad_r);

//this will transform input having multiple images stored contiguously
    void im2col_multiple_batches(const float *data_im, const int batch_size,
                                 const int channels,
                                 const int height, const int width, const int kernel_h, const int kernel_w,
                                 const int pad_h, const int pad_w,
                                 const int stride_h, const int stride_w,
                                 float *data_col);



//Caffe version of im2col...modified for few cases
    void im2colNCHW(const float *data_im, const int channels,
                    const int height, const int width, const int kernel_h, const int kernel_w,
                    const int pad_h, const int pad_w,
                    const int stride_h, const int stride_w,
                    float *data_col);



//Parallel version of im2col using OpenMP
    void im2col_parNCHW(const float *data_im, const int channels,
                        const int height, const int width, const int kernel_h, const int kernel_w,
                        const int pad_h, const int pad_w,
                        const int stride_h, const int stride_w,
                        float *data_col);




//based on Low-memory GEMM-based convolution algorithms for deep neural networks
//https://arxiv.org/pdf/1709.03395.pdf
    void im2rowNHWC(const float *input_data, const int depth, const int height,
                    const int width, const int filter_h, const int filter_w,
                    const int pad_t, const int pad_l, const int pad_b, const int pad_r,
                    const int stride_h, const int stride_w, float *col_data);

    void im2rowNHWCsplit_lpgemm(const uint8_t *input_data, const int depth, const int height,
                         const int width, const int filter_h, const int filter_w,
                         const int pad_t, const int pad_l, const int pad_b, const int pad_r,
                         const int stride_h, const int stride_w, uint8_t *col_data, const int heightOffset,
                         const int heightStart, const int no_of_threads);

    void im2rowNHWCsplit(const float *input_data, const int depth, const int height,
                         const int width, const int filter_h, const int filter_w,
                         const int pad_t, const int pad_l, const int pad_b, const int pad_r,
                         const int stride_h, const int stride_w, float *col_data, const int heightOffset,
                         const int heightStart, const int no_of_threads);


    void im2rowNHWCsplit_par(const float *input_data, const int depth,
                             const int height,
                             const int width, const int filter_h, const int filter_w,
                             const int pad_t, const int pad_l, const int pad_b, const int pad_r,
                             const int stride_h, const int stride_w, float *col_data);


//based on Low-memory GEMM-based convolution algorithms for deep neural networks
//https://arxiv.org/pdf/1709.03395.pdf
    void im2rowNHWC_par(const float *input_data, const int depth, const int height,
                        const int width, const int filter_h, const int filter_w,
                        const int pad_t, const int pad_l, const int pad_b, const int pad_r,
                        const int stride_h, const int stride_w, float *col_data);






//Out of place transpose
    float *transpose(const float *matrix, int n, int m);

    void NCHW2NHWC(const float *nchw_data, int N, int C, int H, int W,
                   float *nhwc_data);
    void NHWC2NCHW(const float *nchw_data, int N, int C, int H, int W,
                   float *nhwc_data);

    void zenConvolution2DwithBiasSum(
        const float *in_layer,
        const int no_of_images,
        const int channels,
        const int height,
        const int width,
        const float *filter,
        const int no_of_filter,
        const int kernel_h,
        const int kernel_w,
        const int pad_t,
        const int pad_l,
        const int pad_b,
        const int pad_r,
        const int stride_h,
        const int stride_w,
        const float *bias,
        float *out_layer,
        const int out_height,
        const int out_width,
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0
    );

    void zenConvolution2DwithBiasSumRelu(
        const float *in_layer,
        const int no_of_images,
        const int channels,
        const int height,
        const int width,
        const float *filter,
        const int no_of_filter,
        const int kernel_h,
        const int kernel_w,
        const int pad_t,
        const int pad_l,
        const int pad_b,
        const int pad_r,
        const int stride_h,
        const int stride_w,
        const float *bias,
        float *out_layer,
        const int out_height,
        const int out_width,
        const bool concat = false,
        const int filter_offset = 0,
        const int total_filters = 0
    );

    void zenConvolution2D_Latency_blocked_layout(
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
        const float pad_h,
        const float pad_w,
        const int stride_h,
        const int stride_w,
        const float *bias,
        float *out_layer,
        const int out_height,
        const int out_width
    );

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
    );

    void zenBatchNorm(
        const int no_of_images,
        const int out_height,
        const int out_width,
        const int no_of_filter,
        const float *scale,
        const float *mean,
        const float *offset,
        float *out_layer,
        int data_format,
        const bool relu
    );


    void zenConvolution2DRef(
        const float *in_layer,
        const int no_of_images,
        const int channels,
        const int height,
        const int width,
        const float *filter,
        const int no_of_filter,
        const int kernel_h,
        const int kernel_w,
        const int pad_t,
        const int pad_l,
        const int pad_b,
        const int pad_r,
        const int stride_h,
        const int stride_w,
        float *out_layer,
        const int out_height,
        const int out_width
    );

    void zenConvolution2DwithBiasRef(
        const float *in_layer,
        const int no_of_images,
        const int channels,
        const int height,
        const int width,
        const float *filter,
        const int no_of_filter,
        const int kernel_h,
        const int kernel_w,
        const int pad_t,
        const int pad_l,
        const int pad_b,
        const int pad_r,
        const int stride_h,
        const int stride_w,
        const float *bias,
        float *out_layer,
        const int out_height,
        const int out_width
    );


    void zenConvolution2DwithBiasReluRef(
        const float *in_layer,
        const int no_of_images,
        const int channels,
        const int height,
        const int width,
        const float *filter,
        const int no_of_filter,
        const int kernel_h,
        const int kernel_w,
        const int pad_t,
        const int pad_l,
        const int pad_b,
        const int pad_r,
        const int stride_h,
        const int stride_w,
        const float *bias,
        float *out_layer,
        const int out_height,
        const int out_width
    );

    void zenConvolution2DwithBatchNormRef(
        const float *in_layer,
        const int no_of_images,
        const int channels,
        const int height,
        const int width,
        const float *filter,
        const int no_of_filter,
        const int kernel_h,
        const int kernel_w,
        const int pad_t,
        const int pad_l,
        const int pad_b,
        const int pad_r,
        const int stride_h,
        const int stride_w,
        const float *scale,
        const float *mean,
        const float *offset,
        float *out_layer,
        const int out_height,
        const int out_width
    );

    void zenConvolution2DwithBatchNormReluRef(
        const float *in_layer,
        const int no_of_images,
        const int channels,
        const int height,
        const int width,
        const float *filter,
        const int no_of_filter,
        const int kernel_h,
        const int kernel_w,
        const int pad_t,
        const int pad_l,
        const int pad_b,
        const int pad_r,
        const int stride_h,
        const int stride_w,
        const float *scale,
        const float *mean,
        const float *offset,
        float *out_layer,
        const int out_height,
        const int out_width
    );

    void zenConvolution2DgemmRef(
        const float *in_layer,
        const int no_of_images,
        const int channels,
        const int height,
        const int width,
        const float *filter,
        const int no_of_filter,
        const int kernel_h,
        const int kernel_w,
        const int pad_t,
        const int pad_l,
        const int pad_b,
        const int pad_r,
        const int stride_h,
        const int stride_w,
        const float *bias,
        float *out_layer,
        const int out_height,
        const int out_width,
        const bool relu
    );

    void zenBatchNormRef(
        const int no_of_images,
        const int out_height,
        const int out_width,
        const int no_of_filter,
        const float *scale,
        const float *mean,
        const float *offset,
        float *out_layer,
        int data_format,
        const bool relu
    );

    void zenMatMul_gemm_wrapper(
        const bool Layout,
        const bool transpose_input,
        const bool transpose_filter,
        const int m,
        const int k,
        const int n,
        const float alpha,
        const float *input,
        const int lda,
        const float *filter,
        const int ldb,
        const float *bias,
        const bool relu,
        const int gelu,
        const float beta,
        float *output,
        const int ldc
    );

    void zenMatMul(
        const bool Layout,
        const bool transpose_input,
        const bool transpose_filter,
        const int batch,
        const int *input_offsets,
        const int *weights_offsets,
        const int *dst_offsets,
        const int no_of_images,
        const int no_of_channels,
        const int no_of_filters,
        const float alpha,
        const float *input,
        const int lda,
        const float *filter,
        const int ldb,
        const float *bias,
        const bool relu,
        const int gelu,
        const float beta,
        float *output,
        const int ldc
    );

    void zenMatMulWithBias(
        const bool Layout,
        const bool transpose_input,
        const bool transpose_filter,
        const int batch,
        const int *input_offsets,
        const int *weights_offsets,
        const int *dst_offsets,
        const int no_of_images,
        const int no_of_channels,
        const int no_of_filters,
        const float alpha,
        const float *input,
        const int lda,
        const float *filter,
        const int ldb,
        const float *bias,
        const float beta,
        float *output,
        const int ldc
    );

    void zenMatMulWithBiasReLU(
        const bool Layout,
        const bool transpose_input,
        const bool transpose_filter,
        const int batch,
        const int *input_offsets,
        const int *weights_offsets,
        const int *dst_offsets,
        const int no_of_images,
        const int no_of_channels,
        const int no_of_filters,
        const float alpha,
        const float *input,
        const int lda,
        const float *filter,
        const int ldb,
        const float *bias,
        const float beta,
        float *output,
        const int ldc
    );

    void zenMatMulWithBiasGeLU(
        const bool Layout,
        const bool transpose_input,
        const bool transpose_filter,
        const int batch,
        const int *input_offsets,
        const int *weights_offsets,
        const int *dst_offsets,
        const int no_of_images,
        const int no_of_channels,
        const int no_of_filters,
        const float alpha,
        const float *input,
        const int lda,
        const float *filter,
        const int ldb,
        const float *bias,
        const float beta,
        float *output,
        const int ldc,
        const int geluType
    );

    void zenMatMul_refWrapper(
        const bool Layout,
        const bool transpose_input,
        const bool transpose_filter,
        const int MB,
        const int m,
        const int k,
        const int n,
        const float alpha,
        const float *input,
        const int lda,
        const float *filter,
        const int ldb,
        const float *bias,
        const bool relu,
        const float beta,
        float *output,
        const int ldc
    );

    void zenConvolution2Dbase_LPGEMM1x1_u8s8s32os32(
    zendnnEnv zenEnvObj,
    const uint8_t *in_layer,
    const int images,
    const int channels,
    const int height,
    const int width,
    const int8_t *filter,
    const int no_of_filter,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    int32_t *bias,
    int32_t *out_layer,
    const int out_height,
    const int out_width,
    const bool relu,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
    );

    void zenConvolution2Dbase_LPGEMM1x1_u8s8s32os8(
    zendnnEnv zenEnvObj,
    const uint8_t *in_layer,
    const int images,
    const int channels,
    const int height,
    const int width,
    const int8_t *filter,
    const int no_of_filter,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    int32_t *bias,
    int8_t *out_layer,
    const int out_height,
    const int out_width,
    const bool relu,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters,
    const int* zero_point_dst,
    const int scale_size
    );

    void zenConvolution2Dbase_LPGEMM1x1_s8s8s32os32(
    zendnnEnv zenEnvObj,
    const int8_t *in_layer,
    const int images,
    const int channels,
    const int height,
    const int width,
    const int8_t *filter,
    const int no_of_filter,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    int32_t *bias,
    int32_t *out_layer,
    const int out_height,
    const int out_width,
    const bool relu,
    int elementwiseType,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
    );

    void zenConvolution2Dbase_LPGEMM1x1_s8s8s32os8(
    zendnnEnv zenEnvObj,
    const int8_t *in_layer,
    const int images,
    const int channels,
    const int height,
    const int width,
    const int8_t *filter,
    const int no_of_filter,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    int32_t *bias,
    int8_t *out_layer,
    const int out_height,
    const int out_width,
    const bool relu,
    int elementwiseType,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters,
    const int* zero_point_dst,
    const int scale_size
    );

    void zenConvolution2Dbase_LPGEMM1x1_s8s8s16os16(
    zendnnEnv zenEnvObj,
    const int8_t *in_layer,
    const int images,
    const int channels,
    const int height,
    const int width,
    const int8_t *filter,
    const int no_of_filter,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    int16_t *bias,
    int16_t *out_layer,
    const int out_height,
    const int out_width,
    const bool relu,
    int elementwiseType,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
    );

    void zenConvolution2Dbase_LPGEMM1x1_s8s8s16os8(
    zendnnEnv zenEnvObj,
    const int8_t *in_layer,
    const int images,
    const int channels,
    const int height,
    const int width,
    const int8_t *filter,
    const int no_of_filter,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    int16_t *bias,
    int8_t *out_layer,
    const int out_height,
    const int out_width,
    const bool relu,
    int elementwiseType,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters,
    const int* zero_point_dst,
    const int scale_size
    );

    void zenConvolution2Dbase_LPGEMM1x1_u8s8s16(
    zendnnEnv zenEnvObj,
    const uint8_t *in_layer,
    const int images,
    const int channels,
    const int height,
    const int width,
    const int8_t *filter,
    const int no_of_filter,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    int16_t *bias,
    int16_t *out_layer,
    const int out_height,
    const int out_width,
    const bool relu,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
    );

    void zenConvolution2Dbase_LPGEMM1x1_u8s8s16os8(
    zendnnEnv zenEnvObj,
    const uint8_t *in_layer,
    const int images,
    const int channels,
    const int height,
    const int width,
    const int8_t *filter,
    const int no_of_filter,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    int16_t *bias,
    int8_t *out_layer,
    const int out_height,
    const int out_width,
    const bool relu,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters,
    const int* zero_point_dst,
    const int scale_size
    );

    void zenConvolution2Dbase_LPGEMM1x1_bf16bf16f32of32(
    zendnnEnv zenEnvObj,
    const int16_t *in_layer,
    const int images,
    const int channels,
    const int height,
    const int width,
    const int16_t *filter,
    const int no_of_filter,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    float *bias,
    float *out_layer,
    const int out_height,
    const int out_width,
    const bool relu,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters
    );

    void zenConvolution2Dbase_LPGEMM1x1_bf16bf16f32obf16(
    zendnnEnv zenEnvObj,
    const int16_t *in_layer,
    const int images,
    const int channels,
    const int height,
    const int width,
    const int16_t *filter,
    const int no_of_filter,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    float *bias,
    int16_t *out_layer,
    const int out_height,
    const int out_width,
    const bool relu,
    const float *scale,
    const float *elementwise_input,
    const bool concat,
    const int filter_offset,
    const int total_filters,
    const int* zero_point_dst,
    const int scale_size
    );

    void zenConvolution2D_direct(
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
        const int pad_t,
        const int pad_l,
        const int pad_b,
        const int pad_r,
        const int stride_h,
        const int stride_w,
        const float *bias,
        float *out_layer,
        const int out_height,
        const int out_width,
        const bool relu,
        const float *scale,
        const float *elementwise_input
    );

    void zenConvolution2D_directVer2(
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
        const int pad_t,
        const int pad_l,
        const int pad_b,
        const int pad_r,
        const int stride_h,
        const int stride_w,
        const float *bias,
        float *out_layer,
        const int out_height,
        const int out_width,
        const bool relu,
        const float *scale,
        const float *elementwise_input
    );

    void zenConvolution2D_directVer3(
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
        const int pad_t,
        const int pad_l,
        const int pad_b,
        const int pad_r,
        const int stride_h,
        const int stride_w,
        const float *bias,
        float *out_layer,
        const int out_height,
        const int out_width,
        const bool relu,
        const float *scale,
        const float *elementwise_input
    );

    void im2row_unrool_3x3(
        float *data_col_tmp,
        unsigned long data_col_offset,
        const float *in_layer,
        unsigned long offset
    );

    void im2row_unrool_7x3(
        float *data_col_tmp,
        unsigned long data_col_offset,
        const float *in_layer,
        unsigned long offset
    );
}

#endif
