/*******************************************************************************
* Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <cmath>
#include <cassert>
#include <cblas.h>
#include <iostream>
#include <algorithm>
#include "common/zendnn_private.hpp"

#define AT(arr, nchannels, nwidth, h, w, ci) (arr[ h * nwidth * nchannels + w * nchannels + ci])
#define AT_HWCN(arr, nwidth, nchannels, nfilters, h, w, c, k) (arr[ h * nwidth * nchannels * nfilters + w * nchannels * nfilters + c * nfilters + k ])

//Function declarations
void filter_transform_2x2_3x3(zendnnEnv zenEnvObj, const float *filter,
                              const int num_channels, const int num_filters, float *out);

void input_transform_2x2_3x3(zendnnEnv zenEnvObj, const float *input,
                             const int batch_size,
                             const int height, const int width, const int num_channels,
                             const int pad_t, const int pad_l, const int pad_b, const int pad_r,
                             float *out, const int num_tiles, const int output_height,
                             const int output_width);

void batched_gemm_2x2_3x3(zendnnEnv zenEnvObj, float *transformed_image,
                          const int num_tiles, const int num_channels, const int num_images,
                          float *transformed_filter, const int num_filters,
                          float *out);

void out_transform_2x2_3x3(zendnnEnv zenEnvObj, float *tiled_input,
                           const int num_tiles, const int num_channels,
                           float *out, const int batch_size, const int output_height,
                           const int output_width);

void post_conv_transform(const int batch_size, const int output_height,
                         const int output_width, const int num_channels,
                         float *out,
                         const float *bias, const bool relu, const float *scale);

void winograd_2x2_3x3(
    zendnnEnv zenEnvObj,
    const float *in_layer,
    const int num_images,
    const int num_channels,
    const int height,
    const int width,
    const float *filter,
    const int num_filters,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const float *bias,
    float *out_layer,
    const int out_height,
    const int out_width,
    const bool relu,
    const bool sum_fused,
    const float *scale
);
