/*******************************************************************************
* Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "memory.hpp"
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
    bool enable_src_dims;
    zendnn_blocking_desc_t blk_size;

    // Default constructor
    Key_matmul() : transpose_input(false), transpose_weights(false), m(1), k(1),
        n(1),
        lda(1), ldb(1), ldc(1), thread_count(1), weights(nullptr),
        enable_src_dims(false), blk_size() {}

    // Constructor to initialize all member variables
    Key_matmul(bool TransA, bool TransB, unsigned int M, unsigned int K,
               unsigned int N,
               unsigned int lda, unsigned int ldb, unsigned int ldc, const void *B_Array,
               unsigned int omp_num_threads, bool src_dims_)
        : transpose_input(false), transpose_weights(TransB), m(1), k(K), n(N),
          lda(1), ldb(ldb), ldc(1), thread_count(omp_num_threads),
          weights(B_Array), enable_src_dims(src_dims_), blk_size() {

        // Update specific variables if src_dims_ is true
        if (src_dims_) {
            transpose_input = false;
            m = M;
            lda = lda;
            ldc = ldc;
        }
    }

    // Constructor to initialize all member variables
    Key_matmul(bool TransA, bool TransB, unsigned int M, unsigned int K,
               unsigned int N,
               unsigned int lda, unsigned int ldb, unsigned int ldc, const void *B_Array,
               unsigned int omp_num_threads, bool src_dims_, zendnn_blocking_desc_t &blk_)
        : transpose_input(false), transpose_weights(TransB), m(1), k(K), n(N),
          lda(1), ldb(ldb), ldc(1), thread_count(omp_num_threads),
          weights(B_Array), enable_src_dims(src_dims_), blk_size(blk_) {

        // Update specific variables if src_dims_ is true
        if (src_dims_) {
            transpose_input = false;
            m = M;
            lda = lda;
            ldc = ldc;
        }
    }

    bool operator==(const Key_matmul &other) const {
        bool flag = true && other.blk_size.inner_nblks == blk_size.inner_nblks;
        for (int idx = 0; idx < blk_size.inner_nblks; idx++) {
            if (blk_size.inner_blks[idx] != other.blk_size.inner_blks[idx]) {
                return false;
            }
            if (blk_size.inner_idxs[idx] != other.blk_size.inner_idxs[idx]) {
                return false;
            }
        }
        // Since dealing with 2D MatMul Key
        for (int i = 0; i < 2; ++i) {
            if (other.blk_size.strides[i] != blk_size.strides[i]) {
                return false;
            }
        }
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
                && flag
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
        seed = zendnn::impl::hash_combine(seed, (k.blk_size.inner_nblks));
        for (int idx = 0; idx < k.blk_size.inner_nblks; idx++) {
            seed = zendnn::impl::hash_combine(seed, (k.blk_size.inner_idxs[idx]));
            seed = zendnn::impl::hash_combine(seed, (k.blk_size.inner_blks[idx]));
        }
        // Since dealing with 2D MatMul Key
        for (int i = 0; i < 2; ++i) {
            seed = zendnn::impl::hash_combine(seed, (k.blk_size.strides[i]));
        }
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

template<typename T>
aocl_post_op *create_aocl_post_ops(const impl::exec_ctx_t &ctx,
    const impl::post_ops_t &po,
    int n, const float alpha, const char *bias,
    int bias_type, const bool relu, const int gelu,
    T *sum_buff, int &postop_count,
    const float *scale, float *dummy_scale);

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

    void im2rowNHWCsplit_lpgemm(const uint8_t *input_data, const int depth,
                                const int height,
                                const int width, const int filter_h, const int filter_w,
                                const int pad_t, const int pad_l, const int pad_b, const int pad_r,
                                const int stride_h, const int stride_w, uint8_t *col_data,
                                const int heightOffset,
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
        const impl::exec_ctx_t &ctx,
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
        const impl::post_ops_t &po_ops,
        const bool relu,
        const int gelu,
        const float beta,
        float *output,
        const int ldc,
        bool is_weights_const = false,
        bool is_inplace = true
    );

    void zenMatMul(
        const impl::exec_ctx_t &ctx,
        zendnn::zendnnEnv zenEnvObj,
        const bool Layout,
        const bool transpose_input,
        const bool transpose_filter,
        const int batch,
        const unsigned long *input_offsets,
        const unsigned long *weights_offsets,
        const unsigned long *dst_offsets,
        const int no_of_images,
        const int no_of_channels,
        const int no_of_filters,
        const float alpha,
        const float *input,
        const int lda,
        const float *filter,
        const int ldb,
        const float *bias,
        const impl::post_ops_t &po_ops,
        const bool relu,
        const int gelu,
        const float beta,
        float *output,
        const int ldc,
        bool is_weights_const = false,
        bool is_inplace = true
    );

    void zenMatMulWithBias(
        const impl::exec_ctx_t &ctx,
        zendnn::zendnnEnv zenEnvObj,
        const bool Layout,
        const bool transpose_input,
        const bool transpose_filter,
        const int batch,
        const unsigned long *input_offsets,
        const unsigned long *weights_offsets,
        const unsigned long *dst_offsets,
        const int no_of_images,
        const int no_of_channels,
        const int no_of_filters,
        const float alpha,
        const float *input,
        const int lda,
        const float *filter,
        const int ldb,
        const float *bias,
        const impl::post_ops_t &po_ops,
        const float beta,
        float *output,
        const int ldc,
        bool is_weights_const = false,
        bool is_inplace = true
    );

    void zenMatMulWithBiasReLU(
        const impl::exec_ctx_t &ctx,
        zendnn::zendnnEnv zenEnvObj,
        const bool Layout,
        const bool transpose_input,
        const bool transpose_filter,
        const int batch,
        const unsigned long *input_offsets,
        const unsigned long *weights_offsets,
        const unsigned long *dst_offsets,
        const int no_of_images,
        const int no_of_channels,
        const int no_of_filters,
        const float alpha,
        const float *input,
        const int lda,
        const float *filter,
        const int ldb,
        const float *bias,
        const impl::post_ops_t &po_ops,
        const float beta,
        float *output,
        const int ldc,
        bool is_weights_const = false,
        bool is_inplace = true
    );

    void zenMatMul_gemm(
        const impl::exec_ctx_t &ctx,
        zendnn::zendnnEnv zenEnvObj,
        const bool auto_tuner,
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
        const impl::post_ops_t &po_ops,
        const bool relu,
        const int gelu,
        const float beta,
        float *output,
        const int ldc,
        bool is_weights_const,
        bool is_inplace
    );

    void zenMatmulSplit(
        const impl::exec_ctx_t &ctx,
        zendnn::zendnnEnv zenEnvObj,
        const bool auto_tuner,
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
        const impl::post_ops_t &po_ops,
        const bool relu,
        const int gelu,
        const float beta,
        float *output,
        const int ldc,
        bool is_weights_const,
        bool is_inplace
    );

    void zenMatMulWithBiasGeLU(
        const impl::exec_ctx_t &ctx,
        zendnn::zendnnEnv zenEnvObj,
        const bool Layout,
        const bool transpose_input,
        const bool transpose_filter,
        const int batch,
        const unsigned long *input_offsets,
        const unsigned long *weights_offsets,
        const unsigned long *dst_offsets,
        const int no_of_images,
        const int no_of_channels,
        const int no_of_filters,
        const float alpha,
        const float *input,
        const int lda,
        const float *filter,
        const int ldb,
        const float *bias,
        const impl::post_ops_t &po_ops,
        const float beta,
        float *output,
        const int ldc,
        const int geluType,
        bool is_weights_const = false,
        bool is_inplace = true
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
        const int *zero_point_dst,
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
        const int *zero_point_dst,
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
        const int *zero_point_dst,
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
        const int *zero_point_dst,
        const int scale_size
    );

    void zenConvolution2Dbase_LPGEMM1x1_u8s8s16ou8(
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
        uint8_t *out_layer,
        const int out_height,
        const int out_width,
        const bool relu,
        const float *scale,
        const float *elementwise_input,
        const bool concat,
        const int filter_offset,
        const int total_filters,
        const int *zero_point_dst,
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
        const int *zero_point_dst,
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

    int matmul_int8_wrapper(
        const zendnn::impl::exec_ctx_t &ctx,
        zendnn::zendnnEnv zenEnvObj,
        int src_type,
        int dst_type,
        int bias_type,
        const bool Layout,
        const bool transA,
        const bool transB,
        const int M,
        const int K,
        const int N,
        const float alpha,
        const char *src,
        const int lda,
        const int8_t *weights,
        const int ldb,
        const char *bias,
        const zendnn::impl::post_ops_t &po_ops,
        const float beta,
        char *dst,
        const int ldc,
        const int32_t zero_point_src,
        const int32_t zero_point_wei,
        const int32_t zero_point_dst,
        float do_sum,
        bool is_weights_const,
        bool is_inplace,
        float *src_scale,
        int src_scale_size,
        bool default_src_scales,
        float *wei_scale,
        int wei_scale_size,
        bool default_wei_scales,
        float *dst_scales,
        int dst_scale_size,
        bool default_dst_scales,
        int scale_type
    );

    int auto_compute_matmul(
        const impl::exec_ctx_t &ctx,
        zendnn::zendnnEnv zenEnvObj,
        const bool Layout,
        const bool transpose_input,
        const bool transpose_weights,
        const int m,
        const int k,
        const int n,
        const float alpha,
        const float *input,
        const int lda,
        const float *filter,
        const int ldb,
        const float *bias,
        const impl::post_ops_t &po_ops,
        const bool relu,
        const int gelu,
        const float beta,
        float *output,
        const int ldc,
        bool is_weights_const,
        bool is_inplace
    );

    int auto_compute_matmul_woq(
        const impl::exec_ctx_t &ctx,
        zendnn::zendnnEnv zenEnvObj,
        int src_type,
        int weights_type,
        int dst_type,
        int bias_type,
        const bool Layout,
        const bool transA,
        const bool transB,
        const int m,
        const int k,
        const int n,
        const float alpha,
        const char *input,
        const int lda,
        const char *weights,
        const int ldb,
        const char *bias,
        const impl::post_ops_t &po_ops,
        const bool has_eltwise_relu,
        const int geluType,
        const float beta,
        char *dst,
        const int ldc,
        bool is_weights_const,
        float *wei_scale,
        int scale_size,
        int group_size,
        zendnn_data_type_t scale_dt
    );

    int auto_compute_matmul_int8(
        const zendnn::impl::exec_ctx_t &ctx,
        zendnn::zendnnEnv zenEnvObj,
        int src_type,
        int dst_type,
        int bias_type,
        const bool Layout,
        const bool transpose_input,
        const bool transpose_weights,
        const int m,
        const int k,
        const int n,
        const float alpha,
        const char *input,
        const int lda,
        const int8_t *weights,
        const int ldb,
        const char *bias,
        const zendnn::impl::post_ops_t &po_ops,
        const float beta,
        char *output,
        const int ldc,
        const int32_t zero_point_src,
        const int32_t zero_point_wei,
        const int32_t zero_point_dst,
        float do_sum,
        bool is_weights_const,
        bool is_inplace,
        float *src_scale,
        int src_scale_size,
        bool default_src_scales,
        float *wei_scale,
        int wei_scale_size,
        bool default_wei_scales,
        float *dst_scales,
        int dst_scale_size,
        bool default_dst_scales,
        int scale_type
    );

    int matmul_woq_wrapper(
        const zendnn::impl::exec_ctx_t &ctx,
        zendnn::zendnnEnv zenEnvObj,
        int src_type,
        int weights_type,
        int dst_type,
        int bias_type,
        const bool Layout,
        const bool transA,
        const bool transB,
        const int M,
        const int K,
        const int N,
        const float alpha,
        const char *src,
        const int lda,
        const char *weights,
        const int ldb,
        const char *bias,
        const zendnn::impl::post_ops_t &po_ops,
        const bool has_eltwise_relu,
        const int geluType,
        const float beta,
        char *dst,
        const int ldc,
        float *wei_scale,
        const int32_t zero_point_weights,
        int scale_size,
        bool is_weights_const,
        int group_size,
        zendnn_data_type_t scale_dt
    );

    int matmul_bf16_wrapper(
        const impl::exec_ctx_t &ctx,
        zendnn::zendnnEnv zenEnvObj,
        int dst_type,
        int bias_type,
        const bool Layout,
        const bool transA,
        const bool transB,
        const int M,
        const int K,
        const int N,
        const float alpha,
        const zendnn::impl::bfloat16_t *src,
        const int lda,
        const zendnn::impl::bfloat16_t *weights,
        const int ldb,
        const char *bias,
        const bool has_eltwise_relu,
        const impl::post_ops_t &po_ops,
        int has_binary_index,
        const int geluType,
        const float beta,
        void *dst,
        const int ldc,
        const float *output_scales,
        const int scale_size,
        bool is_weights_const,
	bool is_inplace
    );

    int auto_compute_matmul_bf16(
        const impl::exec_ctx_t &ctx,
        zendnn::zendnnEnv zenEnvObj,
        int dst_type,
        int bias_type,
        const bool Layout,
        const bool transpose_input,
        const bool transpose_filter,
        const int M,
        const int K,
        const int N,
        const float alpha,
        const zendnn::impl::bfloat16_t *src,
        const int lda,
        const zendnn::impl::bfloat16_t *weights,
        const int ldb,
        const char *bias,
        const bool has_eltwise_relu,
        const impl::post_ops_t &po_ops,
        int has_binary_index,
        const int geluType,
        const float beta,
        void *dst,
        const int ldc,
        const float *output_scales,
        const int scale_size,
        bool is_weights_const,
	bool is_inplace
    );
}

#endif
