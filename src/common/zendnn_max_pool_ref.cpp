/*******************************************************************************
* Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <zendnn_private.hpp>
#include <time.h>
#include <sys/time.h>
#include "zendnn_logging.hpp"

using namespace zendnn;

void maxPoolingRefV1(
    zendnnEnv zenEnvObj,
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
) {
    zendnnInfo(ZENDNN_ALGOLOG, "zendnn maxpool [zendnn max_pool]");
    unsigned int thread_qty = zenEnvObj.omp_num_threads;

    // TensorFlow does not support NCHW data format
    // TODO: Validate this C++ API (NCHW) using MKLDNN and make changes accordingly
    if (data_format == DATA_FORMAT_NCHW) {
        zendnnInfo(ZENDNN_ALGOLOG, "zendnn maxpool DATA_FORMAT_NCHW [zendnn max_pool]");
        int out_index = 0;
        for (int n=0; n<number_of_images; n++) {
            for (int c=0; c<number_of_channel; c++) {
                for (int left_h=0;
                        left_h < height + (padding_height_top + padding_height_bottom) - kernel_height +
                        1; left_h += stride_height) {
                    for (int left_w = 0;
                            left_w < width + (padding_width_right + padding_width_left) - kernel_width + 1;
                            left_w += stride_width) {
                        float max = -FLT_MAX;

                        for (int kernel_i = 0; kernel_i < kernel_height; kernel_i++) {
                            for (int kernel_j = 0; kernel_j < kernel_width; kernel_j++) {
                                float data = 0;
                                if (! padding_zone(left_h+kernel_i, left_w+kernel_j, width, height,
                                                   padding_width_left, padding_height_top)) {
                                    int left_index = n * (number_of_channel*height*width) + c * (height*width) +
                                                     (left_h-padding_height_top) *(width) + (left_w-padding_width_left);
                                    int current_index = left_index + kernel_j + kernel_i * width;
                                    data = input[current_index];
                                    if (max < data) {
                                        max = data;
                                    }
                                }
                            }
                        }
                        output[out_index++] = max;
                    }
                }
            }
        }
    }
    else if (data_format == DATA_FORMAT_NHWC) { // NHWC
        zendnnInfo(ZENDNN_ALGOLOG, "zendnn maxpool DATA_FORMAT_NHWC [zendnn max_pool]");
        int n = 0, c = 0, left_h = 0, left_w = 0, kernel_i = 0, kernel_j = 0;
        int out_index = 0, left_index = 0, current_index = 0;
        float data = 0.0, max = 0.0;
        int new_batch_data_size = (int)(ceil((1.0*height + padding_height_top +
                                              padding_height_bottom - kernel_height + 1.0)/(stride_height)) * ceil((
                                                      1.0*width + padding_width_left + padding_width_right - kernel_width + 1.0)/
                                                      (stride_width)) * number_of_channel);
        int new_height = height + (padding_height_top + padding_height_bottom) -
                         kernel_height + 1;
        int new_width = width + (padding_width_left + padding_width_right) -
                        kernel_width + 1;
        int orig_batch_data_size = (height*width*number_of_channel);
        int orig_height_data_size = (width*number_of_channel);

        if (number_of_images != 1) {
            #pragma omp parallel num_threads(thread_qty) shared(input, output) private(n, c, left_h, left_w, kernel_i, kernel_j, left_index, data, current_index, max, out_index)
            {
                #pragma omp for
                for (n=0; n<number_of_images; n++) {
                    for (c=0; c<number_of_channel; c++) {
                        out_index = n * new_batch_data_size + c;
                        for (left_h=0; left_h < new_height; left_h += stride_height) {
                            for (left_w = 0; left_w < new_width; left_w += stride_width) {
                                max = -FLT_MAX;
                                left_index = n * orig_batch_data_size + (left_h-padding_height_top) *
                                orig_height_data_size + (left_w-padding_width_left) * (number_of_channel) + c;

                                for (kernel_i = 0; kernel_i < kernel_height; kernel_i++) {
                                    for (kernel_j = 0; kernel_j < kernel_width; kernel_j++) {
                                        data = 0;
                                        if (! padding_zone(left_h+kernel_i, left_w+kernel_j, width, height,
                                                           padding_width_left, padding_height_top)) {
                                            current_index = left_index + kernel_j * number_of_channel + kernel_i *
                                            orig_height_data_size;

                                            data = input[current_index];

                                            if (max < data) {
                                                max = data;
                                            }
                                        }
                                    }

                                }
                                output[out_index] = max;
                                out_index += number_of_channel;
                            }
                        }
                    }
                }
            }
        }
        else { // latency (OpenMP for channels)
            #pragma omp parallel num_threads(thread_qty) shared(input, output) private(n, c, left_h, left_w, kernel_i, kernel_j, left_index, data, current_index, max, out_index)
            {
                #pragma omp for
                for (c=0; c<number_of_channel; c++) {
                    out_index = c;
                    for (left_h=0; left_h < new_height; left_h += stride_height) {
                        for (left_w = 0; left_w < new_width; left_w += stride_width) {
                            max = -FLT_MAX;
                            left_index = (left_h-padding_height_top) * orig_height_data_size +
                                         (left_w-padding_width_left) * (number_of_channel) + c;

                            for (kernel_i = 0; kernel_i < kernel_height; kernel_i++) {
                                for (kernel_j = 0; kernel_j < kernel_width; kernel_j++) {
                                    data = 0;
                                    if (! padding_zone(left_h+kernel_i, left_w+kernel_j, width, height,
                                                       padding_width_left, padding_height_top)) {
                                        current_index = left_index + kernel_j * number_of_channel + kernel_i *
                                                        orig_height_data_size;

                                        data = input[current_index];

                                        if (max < data) {
                                            max = data;
                                        }
                                    }
                                }

                            }
                            output[out_index] = max;
                            out_index += number_of_channel;
                        }
                    }
                }
            }
        }
    }
}

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
) {
    zendnnEnv zenEnvObj = readEnv();

    struct timeval start, end;
    gettimeofday(&start, 0);

    maxPoolingRefV1(zenEnvObj, input, number_of_images, number_of_channel, height,
                    width, kernel_height,
                    kernel_width,
                    stride_height,
                    stride_width,
                    padding_height_top,
                    padding_height_bottom,
                    padding_width_left,
                    padding_width_right,
                    output,
                    data_format // 1 for NCHW and 0 for NHWC
                   );

    gettimeofday(&end, 0);
    float elapsed;
    elapsed = timedifference_msec(start, end);
    zendnnInfo(ZENDNN_PROFLOG, "ZENDNN MaxPool profile, no_of_images=",
               number_of_images,
               " channels=", number_of_channel, " height=", height, " width=", width,
               " kernel_h=", kernel_height, " kernel_w=", kernel_width,
               " pad_h_t=", padding_height_top, " pad_h_b=", padding_height_bottom,
               " pad_w_l=", padding_width_left, " pad_w_r=",padding_width_right,
               " stride_h=", stride_height, " stride_w=", stride_width,
               " Time=", elapsed, "ms");
}
