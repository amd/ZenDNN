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

void avgPoolingRefV1(
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
    unsigned int thread_qty = 1;//zenEnvObj.omp_num_threads;

    zendnnVerbose(ZENDNN_ALGOLOG, "ZENDNN AvgPool profile, no_of_images=",
                  number_of_images,
                  " channels=", number_of_channel, " height=", height, " width=", width,
                  " kernel_h=", kernel_height, " kernel_w=", kernel_width,
                  " pad_h_t=", padding_height_top, " pad_h_b=", padding_height_bottom,
                  " pad_w_l=", padding_width_left, " pad_w_r=",padding_width_right,
                  " stride_h=", stride_height, " stride_w=", stride_width);

    // TensorFlow does not support NCHW data format
    if (data_format == DATA_FORMAT_NHWC) { // NHWC
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

                    for (int h = 0; h < height_col; ++h) {
                        w_pad = -padding_width_left;
                        h_pad = h_offset + (h * stride_height);
                        tmp_output = output + outputOffset + (h*out_width);

                        for (int w = 0; w < width_col; ++w) {
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
                                        for (int k=0; k<number_of_channel; k++) {
                                            tmp_output[k] += input[inputOffset+offset + k];
                                        }
                                        avg_count++;
                                    }
                                }
                            }

                            avg_count = avg_count==0? 1 : avg_count;

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

            for (int h = 0; h < height_col; ++h) {
                w_pad = -padding_width_left;
                h_pad = h_offset + (h * stride_height);
                tmp_output = output + (h*out_width);

                for (int w = 0; w < width_col; ++w) {
                    for (int k=0; k<number_of_channel; k++) {
                        tmp_output[k] = 0;
                    }
                    int avg_count = 0;
                    for (int ih = h_pad; ih < h_pad + kernel_height; ++ih) {
                        for (int iw = w_pad; iw < w_pad + kernel_width; ++iw) {
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                int offset = (ih * width + iw) * number_of_channel;
                                for (int k=0; k<number_of_channel; k++) {
                                    tmp_output[k] += input[offset + k];
                                }
                                avg_count++;
                            }
                        }
                    }

                    avg_count = avg_count==0? 1 : avg_count;

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

void avgPoolingRef(
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
    avgPoolingRefV1(zenEnvObj, input, number_of_images, number_of_channel, height,
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
