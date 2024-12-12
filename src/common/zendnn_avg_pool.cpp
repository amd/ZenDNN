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

#include "common/zendnn_private.hpp"
#include <time.h>
#include "zendnn_logging.hpp"
#include "zendnn_helper.hpp"
#include <omp.h>

using namespace zendnn;

void avg_pooling_v1(
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
    const int data_format // 1 for NCHW and 0 for NHWC
) {
    zendnnVerbose(ZENDNN_ALGOLOG, "zendnn avgpool [zendnn avg_pool]");
    unsigned int thread_qty = zenEnvObj.omp_num_threads;

    zendnnVerbose(ZENDNN_ALGOLOG, "ZENDNN AvgPool profile, no_of_images=",
                  number_of_images,
                  " channels=", number_of_channel, " height=", height, " width=", width,
                  " kernel_h=", kernel_height, " kernel_w=", kernel_width,
                  " pad_h_t=", padding_height_top, " pad_h_b=", padding_height_bottom,
                  " pad_w_l=", padding_width_left, " pad_w_r=",padding_width_right,
                  " stride_h=", stride_height, " stride_w=", stride_width);

    // TensorFlow does not support NCHW data format
    // TODO: Validate this C++ API (NCHW) using MKLDNN and make changes accordingly
    if (data_format == DATA_FORMAT_NCHW) {
        zendnnVerbose(ZENDNN_ALGOLOG,
                      "zendnn avgpool DATA_FORMAT_NCHW [zendnn avg_pool]");
        int out_index = 0;
        int kernel_HxW = kernel_height*kernel_width;
        for (int n=0; n<number_of_images; n++) {
            for (int c=0; c<number_of_channel; c++) {
                for (int left_h=0;
                        left_h < height + (padding_height_top + padding_height_bottom) - kernel_height +
                        1; left_h += stride_height) {
                    for (int left_w = 0;
                            left_w < width + (padding_width_right + padding_width_left) - kernel_width + 1;
                            left_w += stride_width) {
                        float avg = 0;

                        for (int kernel_i = 0; kernel_i < kernel_height; kernel_i++) {
                            for (int kernel_j = 0; kernel_j < kernel_width; kernel_j++) {
                                if (! padding_zone(left_h+kernel_i, left_w+kernel_j, width, height,
                                                   padding_width_left, padding_height_top)) {
                                    unsigned long left_index = (unsigned long)n * (number_of_channel*height*width) +
                                                               c * (height*width) +
                                                               (left_h-padding_height_top) *(width) + (left_w-padding_width_left);
                                    unsigned long current_index = left_index + kernel_j + kernel_i * width;
                                    avg += input[current_index];
                                }
                            }
                        }
                        output[out_index++] = avg/kernel_HxW;
                    }
                }
            }
        }
    }
    else if (data_format == DATA_FORMAT_NHWC) { // NHWC
        zendnnVerbose(ZENDNN_ALGOLOG,
                      "zendnn avgpool DATA_FORMAT_NHWC [zendnn avg_pool]");

        int height_col = (height + padding_height_top + padding_height_bottom -
                          kernel_height) / stride_height + 1;
        int width_col = (width + padding_width_left + padding_width_right -
                         kernel_width) / stride_width + 1;
        int kernel_HxW = kernel_height*kernel_width;

        if (number_of_images != 1) {

            int inner_thread_qty = 1;
            //Creating nested parallelism when thread_qty > number_of_images to
            //utilize all the cores. Outer work with BS and inner with height col
            //(inner_thread_qty*thread_qty) should be <= total_no_threads
            if (thread_qty > number_of_images) {
                inner_thread_qty = thread_qty/number_of_images;
                thread_qty = number_of_images;
                omp_set_max_active_levels(2);
            }
            int out_width = ((width + padding_width_left + padding_width_right -
                              kernel_width) / stride_width + 1)*number_of_channel;


            unsigned int loopCount = (number_of_images%thread_qty)==0 ?
                                     number_of_images/thread_qty :
                                     (number_of_images/thread_qty)+1;
            #pragma omp parallel num_threads(thread_qty)
            {
                for (int i=0; i<loopCount; i++) {

                    int threadOffset = omp_get_thread_num()+ (i*thread_qty);
                    if (threadOffset >= number_of_images) {
                        break;
                    }
                    unsigned long inputOffset = (unsigned long)
                                                height*width*number_of_channel*threadOffset;
                    unsigned long outputOffset =
                        (unsigned long)height_col*width_col*number_of_channel*threadOffset;

                    float *tmp_output;

                    int h_pad = -padding_height_top;
                    int w_pad = -padding_width_left;
                    int h_offset = -padding_height_top;

                    #pragma omp parallel for num_threads(inner_thread_qty) private(h_pad, w_pad, tmp_output)
                    for (int h = 0; h < height_col; ++h) {
                        w_pad = -padding_width_left;
                        h_pad = h_offset + (h * stride_height);
                        tmp_output = output + outputOffset + (h*out_width);

                        for (int w = 0; w < width_col; ++w) {
                            #pragma omp simd
                            for (int k=0; k<number_of_channel; k++) {
                                tmp_output[k] = 0;
                            }
                            int avg_count = 0;
                            for (int ih = h_pad; ih < h_pad + kernel_height; ++ih) {
                                for (int iw = w_pad; iw < w_pad + kernel_width; ++iw) {
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                        int offset = (ih * width + iw) * number_of_channel;
                                        //For NHWC, as data is stored contiguous along channel axis,
                                        //compute along same axis reduces cache misses and accessing
                                        //data multiple times doesnt degrade performance even for
                                        //write operation
                                        #pragma omp simd
                                        for (int k=0; k<number_of_channel; k++) {
                                            tmp_output[k] += input[inputOffset+offset + k];
                                        }
                                        avg_count++;
                                    }
                                }
                            }

                            avg_count = avg_count==0? 1 : avg_count;

                            #pragma omp simd
                            for (int k=0; k<number_of_channel; k++) {
                                tmp_output[k] = tmp_output[k]/avg_count;
                            }
                            tmp_output += number_of_channel;
                            w_pad += stride_width;
                        }
                    }
                }
            }
        }
        else { // latency (OpenMP for channels)

            int out_width = ((width + padding_width_left + padding_width_right -
                              kernel_width) / stride_width + 1)*number_of_channel;

            float *tmp_output = output;

            int h_pad = -padding_height_top;
            int w_pad = -padding_width_left;
            int h_offset = -padding_height_top;

            #pragma omp parallel for num_threads(thread_qty) private(h_pad, w_pad, tmp_output)
            for (int h = 0; h < height_col; ++h) {
                w_pad = -padding_width_left;
                h_pad = h_offset + (h * stride_height);
                tmp_output = output + (h*out_width);

                for (int w = 0; w < width_col; ++w) {
                    #pragma omp simd
                    for (int k=0; k<number_of_channel; k++) {
                        tmp_output[k] = 0;
                    }
                    int avg_count = 0;
                    for (int ih = h_pad; ih < h_pad + kernel_height; ++ih) {
                        for (int iw = w_pad; iw < w_pad + kernel_width; ++iw) {
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int offset = (ih * width + iw) * number_of_channel;
                                #pragma omp simd
                                for (int k=0; k<number_of_channel; k++) {
                                    tmp_output[k] += input[offset + k];
                                }
                                avg_count++;
                            }
                        }
                    }

                    avg_count = avg_count==0? 1 : avg_count;


                    #pragma omp simd
                    for (int k=0; k<number_of_channel; k++) {
                        tmp_output[k] = tmp_output[k]/avg_count;
                    }
                    tmp_output += number_of_channel;
                    w_pad += stride_width;
                }
                //h_pad += stride_height;
            }
        }
    }
}

void avg_pooling(
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
    const int data_format // 1 for NCHW and 0 for NHWC
) {
    zendnnEnv zenEnvObj = readEnv();

#ifdef _WIN32
    auto start = std::chrono::high_resolution_clock::now();
#else
    struct timeval start, end;
    gettimeofday(&start, 0);
#endif

    avg_pooling_v1(zenEnvObj, input, number_of_images, number_of_channel, height,
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

    float elapsed;
#ifdef _WIN32
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> difference = end - start;
    elapsed = difference.count();
#else
    gettimeofday(&end, 0);
    elapsed = timedifference_msec(start, end);
#endif
    zendnnVerbose(ZENDNN_PROFLOG, "ZENDNN AvgPool profile, no_of_images=",
                  number_of_images,
                  " channels=", number_of_channel, " height=", height, " width=", width,
                  " kernel_h=", kernel_height, " kernel_w=", kernel_width,
                  " pad_h_t=", padding_height_top, " pad_h_b=", padding_height_bottom,
                  " pad_w_l=", padding_width_left, " pad_w_r=",padding_width_right,
                  " stride_h=", stride_height, " stride_w=", stride_width,
                  " Time=", elapsed, "ms");
}
